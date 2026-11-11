#!/usr/bin/env python3
"""
Lite Judgment Prompt 自动化优化 Pipeline

流程:
1. 对齐 GT (human xlsx) 与 ORM (predicted jsonl) 得到 FP / FN 错误案例
2. 为每条错误 case 收集 action 序列、ORM 推理原文、任务指令
3. 一次性将全部 FP / FN 案例送 LLM 分析，输出补充 / 新增规则
4. 规则写回 knowledge_base JSON

用法:
  python utils/auto_optimize_lite_prompt.py \\
    --table "data/seed_free/results (1).xlsx" \\
    --jsonl "data/seed_free/lite+action_list/2026032202055788_predicted.jsonl" \\
    --data-dir "data/seed_free/lite+action_list" \\
    --model seed_free
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eval_human_autojudge_vs_orm import (
    classify_human_row,
    iter_jsonl,
    load_label_rows,
    normalize_trajectory_key,
    parse_orm_is_task_success,
)
from utils.lite_prompt_knowledge import (
    apply_rules_to_knowledge_base,
    get_full_rules_text,
    load_completion_rules,
    load_trajectory_rules,
    format_boundary_as_text,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Step 1: GT vs ORM 对齐，分出 FP / FN
# ---------------------------------------------------------------------------

def _label_from_jsonl_obj(obj: Dict[str, Any]) -> Tuple[Optional[int], str]:
    """从 JSONL 样本中提取伪 GT 标签（只以 evaluation.status 为准）。

    对齐 `test_service.py` 的口径：
    - `evaluation.status == 0` -> gt=0（failure）
    - `evaluation.status != 0` -> gt=1（success）
    - 缺失/无法解析 -> None（跳过该样本）
    """
    ev = obj.get("evaluation")
    if not isinstance(ev, dict):
        return None, ""

    status = ev.get("status")
    if status is None:
        return None, ""

    status_int: Optional[int] = None
    if isinstance(status, int):
        status_int = status
    elif isinstance(status, str):
        s = status.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                status_int = int(s)
            except Exception:
                status_int = None

    if status_int is None:
        return None, ""

    gt = 0 if status_int == 0 else 1
    return gt, f"evaluation.status={status_int}"


def collect_error_cases(
    table_path: Optional[Path],
    jsonl_path: Path,
    *,
    human_column: str = "AutoJudge Result",
    id_column: str = "Trajectory ID",
    error_cause_column: str = "Error Cause",
    sheet: Any = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """对齐 GT 和 ORM，返回 (fp_cases, fn_cases)。

    FP: human=0, pred=1 (ORM 过宽)
    FN: human=1, pred=0 (ORM 过严)
    """
    csv_rows: List[Tuple[str, str, str, str]] = []
    if table_path is not None:
        _, csv_rows = load_label_rows(
            table_path,
            id_column=id_column,
            human_column=human_column,
            error_cause_column=error_cause_column,
            sheet=sheet,
        )

    orm_by_id: Dict[str, Tuple[int, Dict]] = {}
    pseudo_rows: List[Tuple[str, str, int, str]] = []
    for obj in iter_jsonl(jsonl_path):
        tid = (obj.get("task_id") or "").strip()
        if not tid:
            continue
        norm_tid = normalize_trajectory_key(tid)
        v = parse_orm_is_task_success(obj)
        if v is not None:
            orm_by_id[norm_tid] = (v, obj)
        if table_path is None:
            y, raw = _label_from_jsonl_obj(obj)
            if y is not None:
                pseudo_rows.append((norm_tid, tid, y, raw))

    fp_cases: List[Dict] = []
    fn_cases: List[Dict] = []

    if table_path is None:
        iter_rows: List[Tuple[str, str, int, str]] = pseudo_rows
    else:
        iter_rows = []
        for norm_id, raw_tid, human_cell, ec in csv_rows:
            kind, y = classify_human_row(human_cell, ec, also_skip_by_error_cause=True)
            if kind != "ok" or y is None:
                continue
            iter_rows.append((norm_id, raw_tid, y, str(human_cell)))

    for norm_id, raw_tid, y, human_raw in iter_rows:
        entry = orm_by_id.get(norm_id)
        if entry is None:
            continue
        pred, obj = entry
        if pred not in (0, 1):
            continue

        if y == 0 and pred == 1:
            fp_cases.append({
                "task_id": norm_id,
                "raw_tid": raw_tid,
                "human_label": y,
                "human_raw": human_raw,
                "orm_pred": pred,
                "task_obj": obj,
            })
        elif y == 1 and pred == 0:
            fn_cases.append({
                "task_id": norm_id,
                "raw_tid": raw_tid,
                "human_label": y,
                "human_raw": human_raw,
                "orm_pred": pred,
                "task_obj": obj,
            })

    return fp_cases, fn_cases


# ---------------------------------------------------------------------------
# Step 1b: 为每条 case 收集详细信息
# ---------------------------------------------------------------------------

def gather_case_details(
    cases: List[Dict],
    data_dir: str,
) -> List[Dict]:
    """为每条 case 补充 action_chain、orm_response 等字段。"""
    enriched: List[Dict] = []
    tmp2_dir = os.path.join(data_dir, "tmp2")

    for case in cases:
        task_id = case["task_id"]
        obj = case["task_obj"]

        instruction = obj.get("instruction", "")
        traj = obj.get("traj", [])
        actions = []
        for i, step in enumerate(traj, 1):
            action_text = (step.get("value") or {}).get("action", "")
            if action_text:
                actions.append(f"{i}. {action_text}")
        action_chain = "\n".join(actions) if actions else "(无操作链)"

        orm_response_text = ""
        orm_parsed: Dict = {}
        task_tmp2 = os.path.join(tmp2_dir, task_id)
        if os.path.isdir(task_tmp2):
            pattern = os.path.join(task_tmp2, "*_prompt_response.json")
            pr_files = glob.glob(pattern)
            if pr_files:
                try:
                    with open(pr_files[0], "r", encoding="utf-8") as f:
                        pr_data = json.load(f)
                    orm_response_text = pr_data.get("response_text", "")
                    orm_parsed = _parse_orm_response(orm_response_text)
                except Exception:
                    pass

        ev = obj.get("evaluation") or {}
        human_reasoning = (ev.get("reasoning") or "").strip()
        failure_reason = (ev.get("failure_reason") or "")
        if isinstance(failure_reason, list):
            failure_reason = ", ".join(str(x) for x in failure_reason if x)
        failure_reason = str(failure_reason).strip()

        enriched.append({
            "task_id": task_id,
            "instruction": instruction,
            "human_label": case["human_label"],
            "human_raw": case["human_raw"],
            "orm_pred": case["orm_pred"],
            "action_chain": action_chain,
            "orm_response_text": orm_response_text,
            "orm_parsed": orm_parsed,
            "human_reasoning": human_reasoning,
            "failure_reason": failure_reason,
        })

    return enriched


def _parse_orm_response(text: str) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# Step 2: Prompt 工程 — 构建 FP / FN 分析 prompt
# ---------------------------------------------------------------------------

def _format_case_for_prompt(i: int, case: Dict) -> str:
    parsed = case.get("orm_parsed", {})
    ta = parsed.get("trajectory_analysis", "(无)")
    ca = parsed.get("completion_analysis", "(无)")
    tp = parsed.get("trajectory_pass", "(无)")
    cp = parsed.get("completion_pass", "(无)")
    reason = parsed.get("reason", "(无)")

    human_reasoning = case.get("human_reasoning", "")
    failure_reason = case.get("failure_reason", "")
    human_note = ""
    if case.get("human_label") == 0 and (human_reasoning or failure_reason):
        parts = []
        if failure_reason:
            parts.append(f"失败类型: {failure_reason}")
        if human_reasoning:
            parts.append(f"标注说明: {human_reasoning}")
        human_note = "\n- 人工失败原因：" + "；".join(parts)

    return f"""### Case {i}: {case['task_id']}
- 任务指令：{case['instruction']}
- 人工标注：{'成功' if case['human_label'] == 1 else '失败'}（{case['human_raw']}）{human_note}
- ORM 预测：is_task_success = {case['orm_pred']}
- 操作链：
{case['action_chain']}
- ORM 推理原文：
  trajectory_analysis: {ta}
  completion_analysis: {ca}
  trajectory_pass: {tp}
  completion_pass: {cp}
  reason: {reason}
"""


def build_fp_analysis_prompt(cases: List[Dict]) -> str:
    """构建 FP 分析 prompt（ORM 判成功但人工判失败 → 需收紧）。"""
    traj_json = load_trajectory_rules()
    comp_json = load_completion_rules()

    traj_rules_text = get_full_rules_text(traj_json)
    comp_rules_text = get_full_rules_text(comp_json)
    boundary_text = format_boundary_as_text(
        comp_json.get("boundary_clarifications", [])
        + comp_json.get("extra_boundary_clarifications", [])
    )

    cases_text = "\n".join(_format_case_for_prompt(i, c) for i, c in enumerate(cases, 1))

    return f"""# 角色
你是移动端 GUI Agent 任务评估的 Prompt 工程专家。你的职责是分析评估模型（ORM）的错误判断案例，从中提炼出可泛化的判断规则，以改进评估 prompt。

# 背景
我们有一个评估 prompt，从「轨迹正确性」和「终态完成度」两个维度评判 GUI Agent 是否成功完成任务。
以下案例中，ORM **错误地判定为"成功"**（is_task_success=1），但人工标注为"失败"。
这说明当前规则**不够严格**，存在漏判，需要补充**收紧类规则**。

# 当前规则体系

## 维度一：轨迹正确性
{traj_rules_text}

## 维度二：终态完成度
{comp_rules_text}

### 边界辨析
{boundary_text}

# 错误案例（共 {len(cases)} 条，ORM 判成功但人工判失败）

{cases_text}

# 任务要求

请你仔细分析以上所有错误案例，完成以下工作：

1. **归因分析**：对每条案例，判断 ORM 为什么会误判为成功。是哪条现有规则被错误适用了？还是缺少某条关键规则？误判涉及的是「轨迹正确性」维度还是「终态完成度」维度？

2. **提炼通用规则**：从多条案例的共性中提炼出**可泛化的收紧规则**。要求：
   - 规则必须适用于一类场景，不能只针对单个 case
   - 规则的描述和场景示例风格必须与现有规则一致（使用「- **标题**：描述」的格式）
   - 至少覆盖 2 条以上案例才值得提炼为规则
   - 如果某些案例是孤例、无法泛化，归入 "ungeneralizable_cases" 说明原因

3. **规则归属判断**：对每条新规则，判断它应该放在哪里：
   - **supplement**（补充到某条现有规则下）：现有规则已覆盖这个方向但不够细致，需要增加子场景或补充说明。此时给出要补充的现有规则的完整 title。
   - **new**（新增一条独立规则）：现有规则完全未覆盖这种错误模式。此时给出新规则的 title。

4. **维度归属**：每条规则必须明确属于「轨迹正确性」还是「完成度」维度，以及属于该维度的 pass_rules 还是 fail_rules 或 boundary_clarifications。

# 输出格式（严格 JSON，仅输出此 JSON，无 markdown 围栏、无其他文字）

{{
  "case_analysis": [
    {{
      "task_id": "xxx",
      "error_dimension": "轨迹正确性 | 完成度 | both",
      "root_cause": "一句话说明 ORM 为什么误判",
      "misapplied_rule": "被错误适用的现有规则标题（如有，无则为空字符串）"
    }}
  ],
  "rules": [
    {{
      "action": "supplement",
      "affected_dimension": "轨迹正确性 | 完成度",
      "target_section": "pass_rules | fail_rules | boundary_clarifications",
      "existing_title": "要补充的现有规则的完整标题",
      "supplement_content": "补充的描述内容（一两句话）",
      "supplement_examples": ["场景：xxx → 应判失败（收紧）"]
    }},
    {{
      "action": "new",
      "affected_dimension": "轨迹正确性 | 完成度",
      "target_section": "pass_rules | fail_rules | boundary_clarifications",
      "title": "新规则标题",
      "description": "规则描述（风格与现有规则一致）",
      "examples": ["场景：xxx → 应判失败（收紧）"],
      "covered_cases": ["task_id_1", "task_id_2"]
    }}
  ],
  "ungeneralizable_cases": [
    {{
      "task_id": "xxx",
      "reason": "为什么无法泛化为规则"
    }}
  ]
}}"""


def build_fn_analysis_prompt(cases: List[Dict]) -> str:
    """构建 FN 分析 prompt（ORM 判失败但人工判成功 → 需放宽）。"""
    traj_json = load_trajectory_rules()
    comp_json = load_completion_rules()

    traj_rules_text = get_full_rules_text(traj_json)
    comp_rules_text = get_full_rules_text(comp_json)
    boundary_text = format_boundary_as_text(
        comp_json.get("boundary_clarifications", [])
        + comp_json.get("extra_boundary_clarifications", [])
    )

    cases_text = "\n".join(_format_case_for_prompt(i, c) for i, c in enumerate(cases, 1))

    return f"""# 角色
你是移动端 GUI Agent 任务评估的 Prompt 工程专家。你的职责是分析评估模型（ORM）的错误判断案例，从中提炼出可泛化的判断规则，以改进评估 prompt。

# 背景
我们有一个评估 prompt，从「轨迹正确性」和「终态完成度」两个维度评判 GUI Agent 是否成功完成任务。
以下案例中，ORM **错误地判定为"失败"**（is_task_success=0），但人工标注为"成功"。
这说明当前规则**过于严格**，存在误杀，需要补充**放宽类规则**（新的通过条件或对现有不通过规则的豁免/例外说明）。

# 当前规则体系

## 维度一：轨迹正确性
{traj_rules_text}

## 维度二：终态完成度
{comp_rules_text}

### 边界辨析
{boundary_text}

# 错误案例（共 {len(cases)} 条，ORM 判失败但人工判成功）

{cases_text}

# 任务要求

请你仔细分析以上所有错误案例，完成以下工作：

1. **归因分析**：对每条案例，判断 ORM 为什么会误判为失败。是哪条「不通过」规则被错误触发了？还是缺少某条「通过」规则或豁免条件？误判涉及的是「轨迹正确性」维度还是「终态完成度」维度？

2. **提炼通用规则**：从多条案例的共性中提炼出**可泛化的放宽规则**。要求：
   - 规则必须适用于一类场景，不能只针对单个 case
   - 规则的描述和场景示例风格必须与现有规则一致（使用「- **标题**：描述」的格式）
   - 至少覆盖 2 条以上案例才值得提炼为规则
   - 如果某些案例是孤例、无法泛化，归入 "ungeneralizable_cases" 说明原因
   - 放宽规则可以是：新增一条通过规则、对现有不通过规则增加豁免/例外说明、新增边界辨析条目

3. **规则归属判断**：对每条新规则，判断它应该放在哪里：
   - **supplement**（补充到某条现有规则下）：现有规则已覆盖这个方向但不够全面，需要增加豁免场景、例外说明或新的通过示例。此时给出要补充的现有规则的完整 title。
   - **new**（新增一条独立规则）：现有规则完全未覆盖这种应判成功的模式。此时给出新规则的 title。

4. **维度归属**：每条规则必须明确属于「轨迹正确性」还是「完成度」维度，以及属于该维度的 pass_rules 还是 fail_rules 或 boundary_clarifications。

# 输出格式（严格 JSON，仅输出此 JSON，无 markdown 围栏、无其他文字）

{{
  "case_analysis": [
    {{
      "task_id": "xxx",
      "error_dimension": "轨迹正确性 | 完成度 | both",
      "root_cause": "一句话说明 ORM 为什么误判为失败",
      "wrongly_triggered_rule": "被错误触发的现有不通过规则标题（如有，无则为空字符串）"
    }}
  ],
  "rules": [
    {{
      "action": "supplement",
      "affected_dimension": "轨迹正确性 | 完成度",
      "target_section": "pass_rules | fail_rules | boundary_clarifications",
      "existing_title": "要补充的现有规则的完整标题",
      "supplement_content": "补充的豁免/例外说明（一两句话）",
      "supplement_examples": ["场景：xxx → 应判成功（放宽）"]
    }},
    {{
      "action": "new",
      "affected_dimension": "轨迹正确性 | 完成度",
      "target_section": "pass_rules | fail_rules | boundary_clarifications",
      "title": "新规则标题",
      "description": "规则描述（风格与现有规则一致）",
      "examples": ["场景：xxx → 应判成功（放宽）"],
      "covered_cases": ["task_id_1", "task_id_2"]
    }}
  ],
  "ungeneralizable_cases": [
    {{
      "task_id": "xxx",
      "reason": "为什么无法泛化为规则"
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Step 2b: 调用 LLM
# ---------------------------------------------------------------------------

def call_llm_for_analysis(prompt: str, model_name: str = "seed_free") -> str:
    """调用 LLM API 进行分析，返回 response text。"""
    import config
    from utils.ask_qwen import call_ark_api, call_honor_cloud_api, call_mass_api

    if config.is_ark_api_model(model_name):
        resp_pair = call_ark_api(query=prompt, model_name=model_name)
        resp = resp_pair[0] if isinstance(resp_pair, tuple) else resp_pair
    elif config.is_mass_api_model(model_name):
        resp = call_mass_api(query=prompt, model_name=model_name, temperature=0.3, max_tokens=8000)
    else:
        resp = call_honor_cloud_api(query=prompt, model_name=model_name, temperature=0.3, max_tokens=8000)

    return resp or ""


def parse_llm_json_response(text: str) -> Dict:
    """从 LLM 返回的文本中提取 JSON。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {}


# ---------------------------------------------------------------------------
# Step 3: 规则写回 knowledge_base
# ---------------------------------------------------------------------------

def apply_analysis_result(result: Dict) -> Dict[str, int]:
    """从 LLM 分析结果中提取规则并写入 knowledge_base。"""
    rules = result.get("rules", [])
    if not rules:
        print("  LLM 未返回任何规则。")
        return {"supplements": 0, "new_rules": 0}

    stats = apply_rules_to_knowledge_base(rules)
    return stats


# ---------------------------------------------------------------------------
# Step 4: Main CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lite Judgment Prompt 自动化优化 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--table", type=Path, default=None, help="人工标注表 (.xlsx/.csv)，不传则回退到 jsonl 的 pseudo-GT")
    parser.add_argument("--jsonl", type=Path, required=True, help="predicted JSONL 路径")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="数据目录（含 tmp2/），默认为 jsonl 所在目录")
    parser.add_argument("--model", type=str, default="seed_free", help="LLM 模型名称（默认 seed_free）")
    parser.add_argument("--human-column", type=str, default="AutoJudge Result", help="人工结论列名")
    parser.add_argument("--id-column", type=str, default="Trajectory ID", help="轨迹 ID 列名")
    parser.add_argument("--error-cause-column", type=str, default="Error Cause", help="错误原因列名")
    parser.add_argument("--sheet", default="0", help="xlsx 工作表索引或名称")
    parser.add_argument("--dry-run", action="store_true", help="只分析不写入 knowledge_base")
    parser.add_argument("--skip-fp", action="store_true", help="跳过 FP 分析")
    parser.add_argument("--skip-fn", action="store_true", help="跳过 FN 分析")
    parser.add_argument("--save-prompts", type=str, default=None,
                        help="保存生成的分析 prompt 到指定目录（调试用）")
    args = parser.parse_args()

    sheet_arg: Any = args.sheet
    try:
        sheet_arg = int(args.sheet)
    except ValueError:
        sheet_arg = args.sheet

    data_dir = args.data_dir or str(args.jsonl.parent)

    if args.table is not None and not args.table.is_file():
        print(f"错误: 标注表不存在: {args.table}", file=sys.stderr)
        return 1
    if not args.jsonl.is_file():
        print(f"错误: JSONL 不存在: {args.jsonl}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Lite Judgment Prompt 自动化优化 Pipeline")
    print("=" * 70)
    print(f"标注来源: {args.table if args.table else 'jsonl.evaluation.status'}")
    print(f"JSONL:  {args.jsonl}")
    print(f"数据目录: {data_dir}")
    print(f"模型:   {args.model}")
    print(f"Dry-run: {args.dry_run}")
    print()

    # Step 1: 对齐
    print("[Step 1] 对齐 GT vs ORM ...")
    fp_cases, fn_cases = collect_error_cases(
        args.table,
        args.jsonl,
        human_column=args.human_column,
        id_column=args.id_column,
        error_cause_column=args.error_cause_column,
        sheet=sheet_arg,
    )
    print(f"  FP (ORM 过宽，应失败判成功): {len(fp_cases)} 条")
    print(f"  FN (ORM 过严，应成功判失败): {len(fn_cases)} 条")

    if not fp_cases and not fn_cases:
        print("  无错误案例，无需优化。")
        return 0

    # Step 1b: 收集详情
    print("\n[Step 1b] 收集错误案例详情 ...")
    fp_detailed = gather_case_details(fp_cases, data_dir) if fp_cases else []
    fn_detailed = gather_case_details(fn_cases, data_dir) if fn_cases else []
    print(f"  FP 详情收集完成: {len(fp_detailed)} 条 (有 ORM 推理: {sum(1 for c in fp_detailed if c['orm_response_text'])})")
    print(f"  FN 详情收集完成: {len(fn_detailed)} 条 (有 ORM 推理: {sum(1 for c in fn_detailed if c['orm_response_text'])})")

    total_new = 0
    total_supp = 0

    # Step 2 & 3: FP 分析
    if fp_detailed and not args.skip_fp:
        print(f"\n[Step 2a] 构建 FP 分析 prompt ({len(fp_detailed)} 条) ...")
        fp_prompt = build_fp_analysis_prompt(fp_detailed)
        print(f"  Prompt 长度: {len(fp_prompt)} 字符")

        if args.save_prompts:
            os.makedirs(args.save_prompts, exist_ok=True)
            fp_prompt_path = os.path.join(args.save_prompts, "fp_analysis_prompt.txt")
            with open(fp_prompt_path, "w", encoding="utf-8") as f:
                f.write(fp_prompt)
            print(f"  已保存 FP prompt 到: {fp_prompt_path}")

        print(f"\n[Step 2b] 调用 LLM 分析 FP 案例 (model={args.model}) ...")
        fp_response = call_llm_for_analysis(fp_prompt, args.model)
        if not fp_response:
            print("  错误: LLM 返回空响应", file=sys.stderr)
        else:
            print(f"  LLM 响应长度: {len(fp_response)} 字符")
            fp_result = parse_llm_json_response(fp_response)

            if args.save_prompts:
                resp_path = os.path.join(args.save_prompts, "fp_analysis_response.json")
                with open(resp_path, "w", encoding="utf-8") as f:
                    f.write(fp_response)
                print(f"  已保存 FP 响应到: {resp_path}")

            n_rules = len(fp_result.get("rules", []))
            n_ungeneralizable = len(fp_result.get("ungeneralizable_cases", []))
            print(f"  解析到 {n_rules} 条规则, {n_ungeneralizable} 条不可泛化案例")

            _print_analysis_summary(fp_result, "FP")

            if not args.dry_run and fp_result.get("rules"):
                print("\n[Step 3a] 写入 FP 规则到 knowledge_base ...")
                stats = apply_analysis_result(fp_result)
                total_supp += stats["supplements"]
                total_new += stats["new_rules"]
                print(f"  补充规则: {stats['supplements']}, 新增规则: {stats['new_rules']}")
            elif args.dry_run:
                print("  [Dry-run] 跳过写入")

    # Step 2 & 3: FN 分析
    if fn_detailed and not args.skip_fn:
        print(f"\n[Step 2c] 构建 FN 分析 prompt ({len(fn_detailed)} 条) ...")
        fn_prompt = build_fn_analysis_prompt(fn_detailed)
        print(f"  Prompt 长度: {len(fn_prompt)} 字符")

        if args.save_prompts:
            os.makedirs(args.save_prompts, exist_ok=True)
            fn_prompt_path = os.path.join(args.save_prompts, "fn_analysis_prompt.txt")
            with open(fn_prompt_path, "w", encoding="utf-8") as f:
                f.write(fn_prompt)
            print(f"  已保存 FN prompt 到: {fn_prompt_path}")

        print(f"\n[Step 2d] 调用 LLM 分析 FN 案例 (model={args.model}) ...")
        fn_response = call_llm_for_analysis(fn_prompt, args.model)
        if not fn_response:
            print("  错误: LLM 返回空响应", file=sys.stderr)
        else:
            print(f"  LLM 响应长度: {len(fn_response)} 字符")
            fn_result = parse_llm_json_response(fn_response)

            if args.save_prompts:
                resp_path = os.path.join(args.save_prompts, "fn_analysis_response.json")
                with open(resp_path, "w", encoding="utf-8") as f:
                    f.write(fn_response)
                print(f"  已保存 FN 响应到: {resp_path}")

            n_rules = len(fn_result.get("rules", []))
            n_ungeneralizable = len(fn_result.get("ungeneralizable_cases", []))
            print(f"  解析到 {n_rules} 条规则, {n_ungeneralizable} 条不可泛化案例")

            _print_analysis_summary(fn_result, "FN")

            if not args.dry_run and fn_result.get("rules"):
                print("\n[Step 3b] 写入 FN 规则到 knowledge_base ...")
                stats = apply_analysis_result(fn_result)
                total_supp += stats["supplements"]
                total_new += stats["new_rules"]
                print(f"  补充规则: {stats['supplements']}, 新增规则: {stats['new_rules']}")
            elif args.dry_run:
                print("  [Dry-run] 跳过写入")

    print("\n" + "=" * 70)
    if args.dry_run:
        print("完成（Dry-run 模式，未写入任何规则）")
    else:
        print(f"完成！共写入 {total_supp} 条补充规则 + {total_new} 条新增规则")
        print(f"  轨迹正确性: knowledge_base/corrected_strategies/common.json")
        print(f"  终态完成度:  knowledge_base/completed_strategies/common.json")
    print("=" * 70)
    return 0


def _print_analysis_summary(result: Dict, label: str) -> None:
    """打印分析结果的摘要。"""
    case_analysis = result.get("case_analysis", [])
    if case_analysis:
        print(f"\n  --- {label} 归因摘要 ---")
        for ca in case_analysis[:10]:
            tid = ca.get("task_id", "?")
            dim = ca.get("error_dimension", "?")
            cause = ca.get("root_cause", ca.get("root_cause", "?"))
            print(f"    [{tid}] 维度={dim} | {cause}")
        if len(case_analysis) > 10:
            print(f"    ... 还有 {len(case_analysis) - 10} 条")

    rules = result.get("rules", [])
    if rules:
        print(f"\n  --- {label} 生成规则 ---")
        for r in rules:
            action = r.get("action", "?")
            dim = r.get("affected_dimension", "?")
            section = r.get("target_section", "?")
            if action == "supplement":
                et = r.get("existing_title", "?")
                print(f"    [supplement] {dim}/{section} -> 「{et}」: {r.get('supplement_content', '')[:80]}")
            else:
                title = r.get("title", "?")
                print(f"    [new] {dim}/{section} -> 「{title}」: {r.get('description', '')[:80]}")


if __name__ == "__main__":
    raise SystemExit(main())
