#!/usr/bin/env python3
"""Use seed_free to diagnose human-failed / ORM-success false positives.

This script is intentionally outside the main evaluation path. It creates
analysis material for rule iteration; it does not change predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openpyxl import load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.input_resolver import STATUS_CLASS_MAP, _find_header_and_id_column
from utils.pipeline_runner import _build_raw_trajectory_steps_text, _is_usable_image_path
from utils.prompt_utils import (
    call_llm_with_prompt,
    get_image_path_from_step,
    parse_json_response,
    set_model_name,
)


TRI_SUCCESS = 1
TRI_FAILED = 0


def _cell(row: Tuple[Any, ...], headers: Dict[str, int], key: str) -> str:
    idx = headers.get(key)
    if idx is None:
        return ""
    zero_idx = idx - 1
    if zero_idx >= len(row):
        return ""
    value = row[zero_idx]
    return "" if value is None else str(value).strip()


def _norm_status(value: str) -> Optional[int]:
    return STATUS_CLASS_MAP.get((value or "").strip())


def _iter_fp_rows(xlsx: Path, limit: Optional[int]) -> Iterable[Dict[str, str]]:
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    header_row, _id_col, headers = _find_header_and_id_column(ws)
    count = 0
    wanted = [
        "task_id",
        "APP",
        "运行task",
        "人评结果",
        "zj_status",
        "zj_reasoning",
        "primary_error",
        "secondary_error",
        "zj_primary_error",
        "zj_secondary_error",
    ]
    for excel_row, row in enumerate(
        ws.iter_rows(min_row=header_row + 1, values_only=True), start=header_row + 1
    ):
        human = _norm_status(_cell(row, headers, "人评结果"))
        pred = _norm_status(_cell(row, headers, "zj_status"))
        if human != TRI_FAILED or pred != TRI_SUCCESS:
            continue
        item = {"excel_row": str(excel_row)}
        for key in wanted:
            item[key] = _cell(row, headers, key)
        yield item
        count += 1
        if limit is not None and count >= limit:
            break


def _load_jsonl_by_task_id(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            task_id = obj.get("task_id")
            if isinstance(task_id, str) and task_id:
                out[task_id] = obj
    return out


def _get_nested(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _existing_reason(row: Dict[str, str], current: Dict[str, Any], original: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ("zj_reasoning", "primary_error", "secondary_error", "zj_primary_error", "zj_secondary_error"):
        value = row.get(key, "").strip()
        if value:
            parts.append(f"{key}: {value}")
    for label, obj in (("current", current), ("original", original)):
        reasoning = _get_nested(obj, "evaluation", "prediction", "reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            parts.append(f"{label}.prediction.reasoning: {reasoning.strip()}")
    return "\n".join(parts).strip()


def _image_paths(trajectory: List[Dict[str, Any]], mode: str, last_images: int) -> List[str]:
    if mode == "none":
        return []
    steps = trajectory if mode == "all" else trajectory[-max(last_images, 1):]
    paths: List[str] = []
    for step in steps:
        try:
            path = get_image_path_from_step(step)
        except Exception:
            path = ""
        if _is_usable_image_path(path):
            paths.append(path)
    return paths


def _build_prompt(row: Dict[str, str], current: Dict[str, Any], original: Dict[str, Any]) -> str:
    instruction = current.get("instruction") or original.get("instruction") or row.get("运行task", "")
    trajectory = current.get("traj") or original.get("traj") or []
    steps_text = _build_raw_trajectory_steps_text(trajectory if isinstance(trajectory, list) else [])
    current_status = _get_nested(current, "evaluation", "prediction", "task_status")
    original_status = _get_nested(original, "evaluation", "prediction", "task_status")
    current_reason = _get_nested(current, "evaluation", "prediction", "reasoning")
    original_reason = _get_nested(original, "evaluation", "prediction", "reasoning")
    return f"""你是 GUI Agent 任务评估 badcase 诊断助手。请只分析为什么该样本的人评 GT=failed(0)，但 ORM/zj_status 误判为 success(1)。

要求：
- 不做错因分类体系优化，只诊断 ORM 判定应如何更保守。
- 以截图/轨迹客观证据为准，不相信 Agent terminate(success) 自报。
- 输出严格 JSON，不要 markdown。
- failure_category 从以下枚举中选最贴近的一项：
  missing_final_evidence, wrong_object_or_slot, unmet_hard_constraint, partial_subgoals, over_execution_changed_state, invalid_takeover_or_early_stop, environment_misclassified, insufficient_search_or_verification, app_or_entry_mismatch, other
- rule_suggestion 写成可沉淀到 ORM 判定规则的一句话。

任务指令：{instruction}
Excel 行号：{row.get('excel_row', '')}
task_id：{row.get('task_id', '')}
人评结果：{row.get('人评结果', '')}
当前 zj_status：{row.get('zj_status', '')}
当前输出 task_status：{current_status}
原始输入 task_status：{original_status}
当前 reasoning：{current_reason or ''}
原始 reasoning：{original_reason or ''}

轨迹文本：
{steps_text}

输出 JSON schema：
{{
  "task_id": "...",
  "should_be_failed": true,
  "failure_category": "missing_final_evidence",
  "most_likely_failure_reason": "1-3句，说明为什么人评为失败",
  "key_evidence_steps": [0, 1],
  "orm_misjudgment_pattern": "ORM为什么容易误判为成功",
  "rule_suggestion": "可加入规则/框架的一句话",
  "confidence": 0.0
}}
"""


def _diagnose_with_seed_free(
    *,
    row: Dict[str, str],
    current: Dict[str, Any],
    original: Dict[str, Any],
    model: str,
    image_mode: str,
    last_images: int,
    compress_quality: float,
) -> Dict[str, Any]:
    set_model_name(model)
    prompt = _build_prompt(row, current, original)
    trajectory = current.get("traj") or original.get("traj") or []
    images = _image_paths(trajectory if isinstance(trajectory, list) else [], image_mode, last_images)
    response = call_llm_with_prompt(
        prompt,
        images=images or None,
        temperature=0.01,
        max_tokens=1200,
        task_id=row.get("task_id") or None,
        step_name="seed_free_fp_diagnosis",
        compress_quality=compress_quality,
    )
    parsed = parse_json_response(response or "") or {}
    parsed.setdefault("task_id", row.get("task_id", ""))
    parsed["diagnosis_source"] = "seed_free"
    parsed["raw_response"] = response or ""
    parsed["image_count"] = len(images)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True, help="current output JSONL")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="original input raw JSONL")
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="seed_free")
    parser.add_argument("--image-mode", choices=["none", "last", "all"], default="none")
    parser.add_argument("--last-images", type=int, default=8)
    parser.add_argument("--compress-quality", type=float, default=0.6)
    parser.add_argument("--force-llm", action="store_true", help="call seed_free even when existing reason text exists")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    current_by_id = _load_jsonl_by_task_id(args.jsonl)
    original_by_id = _load_jsonl_by_task_id(args.input_jsonl)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for row in _iter_fp_rows(args.xlsx, args.limit):
            task_id = row.get("task_id", "")
            current = current_by_id.get(task_id, {})
            original = original_by_id.get(task_id, {})
            existing = _existing_reason(row, current, original)
            if args.dry_run:
                record = {
                    "task_id": task_id,
                    "excel_row": row.get("excel_row", ""),
                    "instruction": current.get("instruction") or original.get("instruction") or row.get("运行task", ""),
                    "existing_reason": existing,
                    "would_call_seed_free": args.force_llm or not existing,
                }
            elif existing and not args.force_llm:
                record = {
                    "task_id": task_id,
                    "excel_row": row.get("excel_row", ""),
                    "instruction": current.get("instruction") or original.get("instruction") or row.get("运行task", ""),
                    "diagnosis_source": "existing_reason",
                    "existing_reason": existing,
                }
            else:
                record = _diagnose_with_seed_free(
                    row=row,
                    current=current,
                    original=original,
                    model=args.model,
                    image_mode=args.image_mode,
                    last_images=args.last_images,
                    compress_quality=args.compress_quality,
                )
                record["excel_row"] = row.get("excel_row", "")
                record["instruction"] = current.get("instruction") or original.get("instruction") or row.get("运行task", "")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{written}] {task_id} source={record.get('diagnosis_source', 'dry_run')}")
    print(f"wrote {written} records to {args.out_jsonl}")


if __name__ == "__main__":
    main()
