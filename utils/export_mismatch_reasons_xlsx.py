#!/usr/bin/env python3
"""
导出 ali vs 自研 不一致样本（含双方判定原因）到 xlsx。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LABEL_SUCCESS = "success"
LABEL_FAILED = "failed"
LABEL_ENV_ERROR = "env_error"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path} 第 {line_no} 行不是 JSON 对象")
            yield obj


def _get(d: Dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def normalize_pred_tri(task_status: Any) -> Optional[str]:
    if task_status is None:
        return None
    s = str(task_status).strip().lower()
    if s in {"perfect", "fair_enough"}:
        return LABEL_SUCCESS
    if s == "failed":
        return LABEL_FAILED
    if s in {"env_error", "environment_error"}:
        return LABEL_ENV_ERROR
    return None


def normalize_orm_tri(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bool):
        return LABEL_SUCCESS if v else LABEL_FAILED
    if isinstance(v, (int, float)):
        iv = int(v)
        if iv == 1:
            return LABEL_SUCCESS
        if iv == 0:
            return LABEL_FAILED
        if iv == -1:
            return LABEL_ENV_ERROR
        return None
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"1", "true", "yes", "y", "success"}:
            return LABEL_SUCCESS
        if t in {"0", "false", "no", "n", "failed", "fail"}:
            return LABEL_FAILED
        if t in {"-1", "env_error", "environment_error"}:
            return LABEL_ENV_ERROR
    return None


def build_joined_records(in_path: Path, orm_path: Path) -> Iterable[Dict[str, Any]]:
    pred_records = list(iter_jsonl(in_path))

    orm_index: Dict[Any, Dict[str, Any]] = {}
    for obj in iter_jsonl(orm_path):
        tid = _get(obj, "task_id")
        if tid is None:
            continue
        orm_index[tid] = obj

    for obj in pred_records:
        tid = _get(obj, "task_id")
        orm_obj = orm_index.get(tid)
        yield {
            "task_id": tid,
            "pred_task_status": _get(obj, "evaluation", "prediction", "task_status"),
            "orm_is_task_success": _get(
                orm_obj or {}, "evaluation", "orm_prediction", "is_task_success"
            ),
            "_raw": obj,
            "_orm_raw": orm_obj,
        }


def _format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def _format_ali_reason(pred_obj: Dict[str, Any]) -> str:
    failure_reason = _get(pred_obj, "evaluation", "prediction", "failure_reason")
    reasoning = _get(pred_obj, "evaluation", "prediction", "reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = _get(pred_obj, "evaluation", "prediction", "vote", "error_analysis")

    parts: List[str] = []
    if failure_reason not in (None, "", [], {}):
        parts.append(f"failure_reason: {_format_value(failure_reason)}")
    if isinstance(reasoning, str) and reasoning.strip():
        parts.append(f"reasoning: {reasoning.strip()}")
    return "\n".join(parts)


def _format_orm_reason(orm_obj: Dict[str, Any]) -> str:
    first_deviation_reason = _get(orm_obj, "evaluation", "first_deviation_reason")
    reasoning = _get(orm_obj, "evaluation", "reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = _get(orm_obj, "evaluation", "unified_rich_judgment", "reason")

    parts: List[str] = []
    if isinstance(first_deviation_reason, str) and first_deviation_reason.strip():
        parts.append(f"first_deviation_reason: {first_deviation_reason.strip()}")
    if isinstance(reasoning, str) and reasoning.strip():
        parts.append(f"reasoning: {reasoning.strip()}")
    return "\n".join(parts)


def _display_ali_result(raw_status: Any, tri_status: Optional[str]) -> str:
    left = "" if raw_status is None else str(raw_status)
    right = "" if tri_status is None else tri_status
    return f"{left}({right})"


def _display_orm_result(raw_status: Any, tri_status: Optional[str]) -> str:
    left = "" if raw_status is None else str(raw_status)
    right = "" if tri_status is None else tri_status
    return f"{left}({right})"


def _write_xlsx(rows: List[Dict[str, str]], out_path: Path) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
    except ImportError as exc:
        raise RuntimeError("写入 .xlsx 需要安装 openpyxl：pip install openpyxl") from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "mismatch_reasons"

    headers = ["task_id", "指令简述", "ali结果", "自研结果", "ali判定原因", "自研判定原因"]
    ws.append(headers)
    for row in rows:
        ws.append(
            [
                row["task_id"],
                row["instruction"],
                row["ali_result"],
                row["orm_result"],
                row["ali_reason"],
                row["orm_reason"],
            ]
        )

    # 简单可读性设置
    widths = {"A": 30, "B": 44, "C": 22, "D": 22, "E": 70, "F": 70}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=6):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    wb.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 ali vs 自研 tri 不一致样本到 xlsx（含双方判定原因）")
    ap.add_argument("--input", required=True, help="ali 预测 jsonl")
    ap.add_argument("--orm-input", required=True, help="自研 ORM 预测 jsonl")
    ap.add_argument(
        "--output",
        default="",
        help="输出 xlsx 路径（默认：与 --input 同目录，文件名后缀 _mismatch_reasons.xlsx）",
    )
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    orm_path = Path(args.orm_input).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")
    if not orm_path.exists():
        raise FileNotFoundError(f"ORM 输入文件不存在: {orm_path}")

    out_path = (
        Path(args.output).expanduser()
        if args.output
        else (in_path.parent / f"{in_path.stem}_mismatch_reasons.xlsx")
    )

    total = 0
    both_present_tri = 0
    mismatch_rows: List[Dict[str, str]] = []

    for rec in build_joined_records(in_path, orm_path):
        total += 1
        pred_tri = normalize_pred_tri(rec["pred_task_status"])
        orm_tri = normalize_orm_tri(rec["orm_is_task_success"])
        if pred_tri is None or orm_tri is None:
            continue
        both_present_tri += 1
        if pred_tri == orm_tri:
            continue

        pred_obj = rec.get("_raw") or {}
        orm_obj = rec.get("_orm_raw") or {}
        mismatch_rows.append(
            {
                "task_id": "" if rec["task_id"] is None else str(rec["task_id"]),
                "instruction": _format_value(_get(pred_obj, "instruction")),
                "ali_result": _display_ali_result(rec["pred_task_status"], pred_tri),
                "orm_result": _display_orm_result(rec["orm_is_task_success"], orm_tri),
                "ali_reason": _format_ali_reason(pred_obj),
                "orm_reason": _format_orm_reason(orm_obj),
            }
        )

    _write_xlsx(mismatch_rows, out_path)

    print(f"input(pred):      {in_path}")
    print(f"input(orm):       {orm_path}")
    print(f"output(xlsx):     {out_path}")
    print(f"total_rows:       {total}")
    print(f"both_present_tri: {both_present_tri}")
    print(f"mismatch_count:   {len(mismatch_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
