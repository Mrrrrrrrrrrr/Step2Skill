#!/usr/bin/env python3
"""Summarize human-vs-zj_status metrics and mismatches for iteration logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openpyxl import load_workbook

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.input_resolver import (
    _find_header_and_id_column,
    _map_status_class,
    _normalize_header_key,
)


TRI_LABELS = [1, 0, -1]
TRI_NAMES = {
    1: "1(perfect/fair_enough)",
    0: "0(failed)",
    -1: "-1(env_error)",
}


def _cell(row: Tuple[Any, ...], headers: Dict[str, int], key: str) -> str:
    idx = headers.get(key)
    if idx is None:
        return ""
    zero_idx = idx - 1
    if zero_idx >= len(row):
        return ""
    value = row[zero_idx]
    return "" if value is None else str(value).strip()


def _find_header(headers: Dict[str, int], aliases: Iterable[str]) -> Optional[str]:
    alias_keys = {_normalize_header_key(alias) for alias in aliases}
    for header in headers:
        if _normalize_header_key(header) in alias_keys:
            return header
    return None


def _norm_status(value: str) -> Optional[int]:
    return _map_status_class(value)


def summarize_xlsx(path: Path, sample_per_group: int = 30, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    wb = load_workbook(path, read_only=True, data_only=True)
    if sheet_name:
        if sheet_name not in wb.sheetnames:
            raise ValueError(f"xlsx 未找到 sheet {sheet_name!r}，可选: {wb.sheetnames}")
        ws = wb[sheet_name]
    else:
        ws = wb.worksheets[0]
    header_row, _id_col, headers = _find_header_and_id_column(ws)
    human_key = _find_header(
        headers,
        (
            "人评结果",
            "人工标注",
            "人工结果",
            "human_status",
            "human_result",
            "human label",
            "GT Result",
            "gt_result",
            "ground_truth",
            "ground_truth_result",
        ),
    )
    pred_key = _find_header(headers, ("zj_status", "AutoJudge Result", "Autojudge Result"))
    if human_key is None or pred_key is None:
        raise ValueError(
            f"xlsx 缺少人评或 zj_status 列: human={human_key}, pred={pred_key}, headers={list(headers)}"
        )

    rows_total = 0
    valid = 0
    skipped = 0
    conf: Counter[Tuple[int, int]] = Counter()
    mismatches: Dict[str, List[Dict[str, str]]] = {}

    wanted_cols = [
        "task_id",
        "APP",
        "运行task",
        human_key,
        "zj_status",
        "primary_error",
        "secondary_error",
        "zj_reasoning",
    ]

    for excel_row, row in enumerate(
        ws.iter_rows(min_row=header_row + 1, values_only=True), start=header_row + 1
    ):
        rows_total += 1
        human = _norm_status(_cell(row, headers, human_key))
        pred = _norm_status(_cell(row, headers, pred_key))
        if human is None or pred is None:
            skipped += 1
            continue
        valid += 1
        conf[(human, pred)] += 1
        if human != pred:
            group = f"{human}->{pred}"
            bucket = mismatches.setdefault(group, [])
            if len(bucket) < sample_per_group:
                item = {"excel_row": str(excel_row)}
                for col in wanted_cols:
                    item[col] = _cell(row, headers, col)
                bucket.append(item)

    correct = sum(v for (human, pred), v in conf.items() if human == pred)
    accuracy = correct / valid if valid else 0.0
    matrix = {
        str(h): {str(p): conf.get((h, p), 0) for p in TRI_LABELS} for h in TRI_LABELS
    }
    mismatch_counts = {
        f"{h}->{p}": v for (h, p), v in sorted(conf.items()) if h != p
    }
    return {
        "xlsx": str(path),
        "sheet": ws.title,
        "human_column": human_key,
        "prediction_column": pred_key,
        "rows_total": rows_total,
        "valid": valid,
        "skipped": skipped,
        "correct": correct,
        "accuracy": accuracy,
        "matrix": matrix,
        "mismatch_counts": mismatch_counts,
        "mismatch_samples": mismatches,
    }


def format_summary(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"xlsx: {summary['xlsx']}")
    lines.append(f"sheet: {summary.get('sheet', '')}; human={summary.get('human_column', '')}; pred={summary.get('prediction_column', '')}")
    lines.append(
        "accuracy: {}/{} = {:.2%}; skipped={}".format(
            summary["correct"], summary["valid"], summary["accuracy"], summary["skipped"]
        )
    )
    lines.append("confusion matrix (human rows, pred columns):")
    lines.append("human\\pred\t" + "\t".join(TRI_NAMES[p] for p in TRI_LABELS))
    for h in TRI_LABELS:
        row = summary["matrix"].get(str(h), {})
        lines.append(TRI_NAMES[h] + "\t" + "\t".join(str(row.get(str(p), 0)) for p in TRI_LABELS))
    lines.append("mismatch counts:")
    for group, count in sorted(
        summary["mismatch_counts"].items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"  {group}: {count}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--sample-per-group", type=int, default=30)
    parser.add_argument("--sheet", default="")
    args = parser.parse_args()

    summary = summarize_xlsx(
        args.xlsx,
        sample_per_group=args.sample_per_group,
        sheet_name=args.sheet or None,
    )
    text = format_summary(summary)
    print(text)
    if args.out_json:
        args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        args.out_md.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
