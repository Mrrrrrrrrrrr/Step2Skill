#!/usr/bin/env python3
"""Generate case-driven ORM audit artifacts for one optimization iteration.

This tool is offline-only. It may read human labels to find badcases and build
reports, but the runtime ORM pipeline must not consume these labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openpyxl import load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.input_resolver import _map_status_class


HUMAN_HEADERS = ("GT Result", "人评结果", "human_result", "人工评估结果", "人工标注")
PRED_HEADERS = ("zj_status", "AutoJudge Result", "Autojudge Result", "算法status")
ID_HEADERS = ("Trajectory ID", "trajectory_id", "task_id", "Task ID")
TASK_HEADERS = ("Task", "task", "运行task", "instruction")

CLUSTER_TEMPLATES: Dict[str, Dict[str, str]] = {
    "R1": {
        "title": "只有 Agent 自报成功，截图无客观结果态，却判正例",
        "proposal": "要求 completion_evidence.evidence_source 不能是 agent_self_report_only/none；若判 success，必须有截图结果态或合法 fair_enough_exception。",
        "must_not_affect": "已有截图结果态的正例；合法 fair_enough 豁免样本。",
        "conflict": "terminate(success) 自报与截图证据原则冲突。",
    },
    "R2": {
        "title": "停在过程态/中间页/候选列表/规划页，却判正例",
        "proposal": "新增 terminal_state_type 枚举；intermediate_state/candidate_list 默认不能判 perfect/fair_enough，除非明确 fair_enough_exception。",
        "must_not_affect": "查看/查询类已给出可截图核验结论的正例。",
        "conflict": "只差最后一步 fair_enough 与动作结果态规则边界不清。",
    },
    "R3": {
        "title": "多约束/多子目标只完成部分，却判正例",
        "proposal": "要求 missing_requirements 结构字段承载未完成子目标；非空时不能 success，除非合法 fair_enough_exception。",
        "must_not_affect": "核心目标已完成且剩余动作非关键的 fair_enough。",
        "conflict": "复合任务完成度与 fair_enough 放宽规则冲突。",
    },
    "R4": {
        "title": "信息查询/统计类答案无法由截图覆盖验证，却判正例",
        "proposal": "信息类 success 必须用 screenshot_result_state 或 tool_or_system_result 证明答案来源和覆盖范围。",
        "must_not_affect": "截图可读数据与 answer 一致的查询类正例。",
        "conflict": "信息收集类不要求点击后续动作，但仍要求证据闭环。",
    },
    "R5": {
        "title": "普通购买/支付/订票类未到资金风险确认页，过早接管或过早成功",
        "proposal": "区分普通购买和会员权益开通；普通购买未到金额/支付方式/确认页不得 payment_or_manual_boundary 豁免。",
        "must_not_affect": "会员/权益开通已到正确付费入口的环境/接管边界样本。",
        "conflict": "会员权益接管 env_error 与普通电商过早接管 failed 冲突。",
    },
    "R6": {
        "title": "Agent 自报无响应/找不到，被误判 env_error，但截图显示可见入口或模型路径错误",
        "proposal": "新增 env_evidence_type/env_exclusion_hit；env_exclusion_hit 非 none 时禁止 env_error。",
        "must_not_affect": "客观错误页、重复中心点击无变化、登录验证、私有数据为空等真实环境样本。",
        "conflict": "控件无响应 env_error 与点错/路径错 failed 冲突。",
    },
    "R7": {
        "title": "真实系统阻断/登录/空数据，被误判 failed 或 success",
        "proposal": "env_error 必须声明 env_evidence_type；真实阻断用结构字段保护，避免被普通未完成规则覆盖。",
        "must_not_affect": "探索不足、可见目标漏识别、普通搜索无结果。",
        "conflict": "环境硬阻断与只完成部分任务 failed 规则冲突。",
    },
    "R8": {
        "title": "核心目标已可见完成，但因缺少非必要动作被误判 failed",
        "proposal": "用 fair_enough_exception 明确 last_step_only/executed_but_not_effective/verified_unavailable_with_honest_stop 的合法边界。",
        "must_not_affect": "候选列表、过程态、多约束缺失等未完成样本。",
        "conflict": "fair_enough 放宽与结果态证据规则冲突。",
    },
}


def _norm_key(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _norm_id(value: Any) -> str:
    text = _norm_key(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.endswith("_main"):
        text = text[:-5]
    return text


def _status_class(value: Any) -> Optional[int]:
    return _map_status_class(value)


def _first_header(headers: List[str], candidates: Iterable[str]) -> Optional[int]:
    for candidate in candidates:
        for idx, header in enumerate(headers):
            if header == candidate:
                return idx
    lowered = [h.lower().replace(" ", "").replace("_", "") for h in headers]
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in lowered:
            return lowered.index(key)
    return None


def _row_cell(row: Tuple[Any, ...], idx: Optional[int]) -> str:
    if idx is None or idx >= len(row):
        return ""
    return _norm_key(row[idx])


def _terminal_action_type(task: Dict[str, Any]) -> str:
    traj = task.get("traj")
    if not isinstance(traj, list) or not traj:
        return "no_terminate"
    has_takeover = any(isinstance(step, dict) and isinstance(step.get("take_over"), dict) for step in traj)
    last = traj[-1] if isinstance(traj[-1], dict) else {}
    value = last.get("value") if isinstance(last.get("value"), dict) else {}
    text = " ".join(str(value.get(k, "")) for k in ("action", "code", "tool_call", "thought"))
    lowered = text.lower()
    if "terminate" in lowered and "success" in lowered:
        return "terminate_success"
    if "terminate" in lowered and ("failure" in lowered or "environment_error" in lowered or "failed" in lowered):
        return "terminate_failure"
    if has_takeover:
        return "take_over"
    return "no_terminate"


def _load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            task_id = _norm_id(obj.get("task_id"))
            if task_id:
                out[task_id] = obj
    return out


def _completion_evidence(ev: Dict[str, Any]) -> Dict[str, Any]:
    rich = ev.get("unified_rich_judgment") if isinstance(ev.get("unified_rich_judgment"), dict) else {}
    ce = rich.get("completion_evidence") if isinstance(rich.get("completion_evidence"), dict) else {}
    return ce


def _pick_cluster(record: Dict[str, Any]) -> str:
    trans = record["transition"]
    ce = record.get("completion_evidence") or {}
    source = ce.get("evidence_source") or ""
    terminal_state = ce.get("terminal_state_type") or ""
    missing = record.get("missing_requirements") or []
    terminal_action = record.get("terminal_action_type", "")
    if trans == "0->1":
        if source in {"agent_self_report_only", "none"} or ce.get("evidence_derived_from_agent_self_report_only") == "yes":
            return "R1"
        if terminal_state in {"intermediate_state", "candidate_list"}:
            return "R2"
        if missing:
            return "R3"
        text = (record.get("instruction", "") + " " + record.get("zj_reasoning", "")).lower()
        if any(k in text for k in ("购买", "支付", "下单", "订票", "订单", "payment")):
            return "R5"
        if any(k in text for k in ("统计", "多少", "最低", "最高", "告诉", "查询", "查看")):
            return "R4"
        if terminal_action == "terminate_success":
            return "R1"
        return "R2"
    if trans in {"1->-1", "0->-1"}:
        return "R6"
    if trans in {"-1->0", "-1->1"}:
        return "R7"
    if trans == "1->0":
        return "R8"
    return "R2"


def build_records(xlsx_path: Path, jsonl_path: Path, sheet: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_id = _load_jsonl(jsonl_path)
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet not in wb.sheetnames:
        raise SystemExit(f"sheet not found: {sheet}; available={wb.sheetnames}")
    ws = wb[sheet]
    rows = ws.iter_rows(values_only=True)
    headers = [_norm_key(v) for v in next(rows)]
    id_idx = _first_header(headers, ID_HEADERS)
    human_idx = _first_header(headers, HUMAN_HEADERS)
    pred_idx = _first_header(headers, PRED_HEADERS)
    task_idx = _first_header(headers, TASK_HEADERS)
    if id_idx is None or human_idx is None or pred_idx is None:
        raise SystemExit(f"required columns missing: id={id_idx}, human={human_idx}, pred={pred_idx}, headers={headers}")

    records: List[Dict[str, Any]] = []
    matrix: Counter[Tuple[int, int]] = Counter()
    skipped = 0
    excluded = 0
    total = 0
    for excel_row, row in enumerate(rows, start=2):
        total += 1
        task_id = _norm_id(_row_cell(row, id_idx))
        human_raw = _row_cell(row, human_idx)
        pred_raw = _row_cell(row, pred_idx)
        human = _status_class(human_raw)
        pred = _status_class(pred_raw)
        if human is None:
            skipped += 1
            continue
        if pred is None:
            excluded += 1
            continue
        matrix[(human, pred)] += 1
        if human == pred:
            continue
        obj = by_id.get(task_id, {})
        ev = obj.get("evaluation") if isinstance(obj.get("evaluation"), dict) else {}
        orm = ev.get("orm_prediction") if isinstance(ev.get("orm_prediction"), dict) else {}
        prediction = ev.get("prediction") if isinstance(ev.get("prediction"), dict) else {}
        rich = ev.get("unified_rich_judgment") if isinstance(ev.get("unified_rich_judgment"), dict) else {}
        evidence = rich.get("evidence") if isinstance(rich.get("evidence"), dict) else {}
        ce = _completion_evidence(ev)
        record = {
            "task_id": task_id,
            "excel_row": excel_row,
            "instruction": obj.get("instruction") or _row_cell(row, task_idx),
            "APP": obj.get("APP") or obj.get("app") or "",
            "human_label": human,
            "human_raw": human_raw,
            "zj_status": pred,
            "zj_status_raw": pred_raw,
            "transition": f"{human}->{pred}",
            "zj_reasoning": _row_cell(row, headers.index("zj_reasoning") if "zj_reasoning" in headers else None) or prediction.get("reasoning", ""),
            "primary_error": _row_cell(row, headers.index("primary_error") if "primary_error" in headers else None) or prediction.get("primary_error_cause", ""),
            "secondary_error": _row_cell(row, headers.index("secondary_error") if "secondary_error" in headers else None) or prediction.get("secondary_error_cause", ""),
            "error_cause_skipped": orm.get("error_cause_skipped", 0),
            "task_status": rich.get("task_status") or prediction.get("task_status", ""),
            "final_judgment": rich.get("final_judgment"),
            "completion_judgment": rich.get("completion_judgment"),
            "trajectory_correctness_judgment": rich.get("trajectory_correctness_judgment") or orm.get("is_trajectory_correct"),
            "completion_evidence": ce,
            "missing_requirements": evidence.get("missing_requirements", []),
            "is_env_error": rich.get("is_env_error", pred == -1),
            "env_error_type": rich.get("env_error_type", prediction.get("env_error_type", "")),
            "env_error_reason": rich.get("env_error_reason", prediction.get("env_error_reason", "")),
            "first_deviation_step": ev.get("first_deviation_step") or rich.get("first_deviation_step"),
            "first_deviation_reason": ev.get("first_deviation_reason") or rich.get("first_deviation_reason", ""),
            "terminal_action_type": _terminal_action_type(obj),
        }
        record["rule_id"] = _pick_cluster(record)
        records.append(record)

    valid = sum(matrix.values())
    correct = sum(v for (h, p), v in matrix.items() if h == p)
    summary = {
        "xlsx": str(xlsx_path),
        "jsonl": str(jsonl_path),
        "sheet": sheet,
        "rows_total": total,
        "valid": valid,
        "skipped_unknown_human": skipped,
        "excluded_missing_pred": excluded,
        "correct": correct,
        "accuracy": correct / valid if valid else 0,
        "matrix": {f"{h}->{p}": matrix.get((h, p), 0) for h in (1, 0, -1) for p in (1, 0, -1)},
        "mismatch_counts": dict(Counter(r["transition"] for r in records)),
    }
    return records, summary


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sample_ids(records: List[Dict[str, Any]], rule_id: str, limit: int = 12) -> List[str]:
    out: List[str] = []
    for record in records:
        if record.get("rule_id") == rule_id and record["task_id"] not in out:
            out.append(record["task_id"])
        if len(out) >= limit:
            break
    return out


def write_reports(records: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "case_audit.json", {"summary": summary, "records": records})

    by_rule: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_rule[record["rule_id"]].append(record)

    cluster_lines = ["# iter01 Cluster Report", "", "## Metric Snapshot", "", f"- sheet: `{summary['sheet']}`", f"- valid: {summary['valid']}", f"- correct: {summary['correct']}", f"- accuracy: {summary['accuracy']:.2%}", f"- mismatches: {summary['mismatch_counts']}", ""]
    for rule_id in sorted(by_rule):
        tpl = CLUSTER_TEMPLATES[rule_id]
        cases = by_rule[rule_id]
        transitions = Counter(c["transition"] for c in cases)
        ids = [c["task_id"] for c in cases[:20]]
        cluster_lines += [
            f"## {rule_id}: {tpl['title']}",
            "",
            f"- case_count: {len(cases)}",
            f"- transitions: {dict(transitions)}",
            f"- shared_preconditions: {tpl['title']}",
            "- expected_correct_label: based on human_label bucket; runtime rule must not read it",
            f"- current_wrong_behavior: {dict(transitions)}",
            f"- general_rule_proposal: {tpl['proposal']}",
            f"- must_not_affect_buckets: {tpl['must_not_affect']}",
            f"- conflicting_existing_rules: {tpl['conflict']}",
            f"- target_task_ids: {ids}",
            "",
        ]
    (out_dir / "cluster_report.md").write_text("\n".join(cluster_lines).rstrip() + "\n", encoding="utf-8")

    conflict_md = """# iter01 Rule Conflict Report

## C1: Stage E0 vs terminate(success)
- Decision: E0 remains first, but env_error requires structured `env_evidence_type != none` and `env_exclusion_hit == none`; `terminate_success` is an exclusion.

## C2: fair_enough last-step exception vs result-state requirement
- Decision: success without result-state evidence is allowed only with `fair_enough_exception` in `last_step_only`, `executed_but_not_effective`, `verified_unavailable_with_honest_stop`, or `payment_or_manual_boundary`.

## C3: find-not-found honest stop vs insufficient exploration
- Decision: `verified_unavailable_with_honest_stop` requires correct container plus sufficient exploration; normal search-empty or visible target missed must not become success/env.

## C4: no-response env_error vs wrong path/click miss
- Decision: env requires objective enum evidence; wrong path, visible target missed, insufficient retry, or normal search-empty excludes env.

## C5: membership payment boundary vs ordinary purchase early takeover
- Decision: `payment_or_manual_boundary` is for membership/equity/subscription, true risk confirmation pages, or non-environment manual confirmation/input/authorization boundaries only; ordinary purchase before payment confirmation and login/verification blocks remain excluded.
"""
    (out_dir / "rule_conflict_report.md").write_text(conflict_md, encoding="utf-8")

    target_rules = ["R1", "R2", "R3", "R4", "R5"]
    target_ids: List[str] = []
    for rule_id in target_rules:
        for task_id in _sample_ids(records, rule_id, 12):
            if task_id not in target_ids:
                target_ids.append(task_id)
    anti_ids: List[str] = []
    for transition in ("1->0", "1->-1", "-1->0", "-1->1", "0->-1"):
        for record in records:
            if record["transition"] == transition and record["task_id"] not in anti_ids:
                anti_ids.append(record["task_id"])
            if len([x for x in anti_ids]) >= 80:
                break
    _write_json(out_dir / "target_task_ids.json", target_ids)
    _write_json(out_dir / "anti_regression_task_ids.json", anti_ids)

    change_md = """# iter01 Change Plan

## Hypothesis
- The largest repairable bucket is `0->1`: success is accepted with weak or missing objective terminal evidence.
- Adjacent env and fair_enough rules need explicit structured exclusions to prevent regressions.

## Planned Changes
| File | Type | Change |
|------|------|--------|
| `prompts/lite_trajectory_unified_prompt.py` | Prompt | Extend `completion_evidence` with `evidence_source`, `terminal_state_type`, `missing_requirements`, and `fair_enough_exception`; require these fields before success. |
| `prompts/lite_trajectory_unified_prompt.py` | Prompt | Add `env_evidence_type` and `env_exclusion_hit`; clarify E0 exclusions before env success. |
| `utils/pipeline_runner.py` | Framework | Enforce only structured enum consistency: downgrade success without objective evidence/valid fair_enough exception, and downgrade env when enum evidence is none or exclusion is hit. |

## Targeted Validation
- Use `target_task_ids.json` for R1-R5 false-positive patterns.
- Use `anti_regression_task_ids.json` as manual/secondary guard for `1->0`, `1->-1`, `0->-1`, and env recall buckets.

## Non-goals
- No prior/GT/history-based runtime rule.
- No free-text keyword postprocessing in runtime framework.
"""
    (out_dir / "change_plan.md").write_text(change_md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--sheet", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records, summary = build_records(args.xlsx, args.jsonl, args.sheet)
    write_reports(records, summary, args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
