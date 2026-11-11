#!/usr/bin/env python3
"""Count records judged as corrected in step-judge outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_INPUT_DIR = Path("judge_results_parallel_0829") / "judge_step_chunks" / "outputs"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_result_files(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.glob("*.json"))


def normalize_results(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        items = payload["results"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        return []
    return [item for item in items if isinstance(item, dict)]


def judgment_is_corrected(item: Dict[str, Any], field: str) -> bool:
    judgment = item.get(field)
    if not isinstance(judgment, dict):
        return False
    return judgment.get("is_corrected") is True


def matches_variant_filter(item: Dict[str, Any], keyword: Optional[str]) -> bool:
    if not keyword:
        return True
    variant = item.get("variant")
    return isinstance(variant, str) and keyword.lower() in variant.lower()


def row_for(path: Path, item: Dict[str, Any], judgment_field: str) -> Dict[str, Any]:
    judgment = item.get(judgment_field) if isinstance(item.get(judgment_field), dict) else {}
    repredicted_action = item.get("repredicted_action")
    original_action = item.get("original_action")
    return {
        "source_file": path.name,
        "task_id": item.get("task_id", ""),
        "category_id": item.get("category_id", ""),
        "category_name": item.get("category_name", ""),
        "wrong_action_step": item.get("wrong_action_step", ""),
        "variant": item.get("variant", ""),
        "step_action_correctness": judgment.get("step_action_correctness", ""),
        "comparison_to_original": judgment.get("comparison_to_original", ""),
        "confidence": judgment.get("confidence", ""),
        "remaining_error_type": judgment.get("remaining_error_type", ""),
        "original_action": json.dumps(original_action, ensure_ascii=False) if original_action is not None else "",
        "repredicted_action": json.dumps(repredicted_action, ensure_ascii=False) if repredicted_action is not None else "",
        "repredicted_image": item.get("repredicted_image", ""),
        "judge_reason": item.get("judge_reason", ""),
        "correction_reason": judgment.get("reason", ""),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "task_id",
        "category_id",
        "category_name",
        "wrong_action_step",
        "variant",
        "step_action_correctness",
        "comparison_to_original",
        "confidence",
        "remaining_error_type",
        "original_action",
        "repredicted_action",
        "repredicted_image",
        "judge_reason",
        "correction_reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 judge step outputs 中被 judge 判定为正确的纠错尝试记录。"
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"outputs 目录，默认: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--variant-keyword",
        default=None,
        help="可选：只统计 variant 中包含该关键字的记录。默认统计全部记录",
    )
    parser.add_argument(
        "--judgment-field",
        choices=["judgment", "llm_judgment"],
        default="judgment",
        help="使用最终判定 judgment 或原始模型判定 llm_judgment，默认: judgment",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="可选：导出判定为正确的 correction 明细 CSV",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=20,
        help="展示按 variant/category 汇总的前 N 项，默认: 20",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir not found: {input_dir}")

    files = list(iter_result_files(input_dir))
    total_records = 0
    attempt_records = 0
    correct_rows: List[Dict[str, Any]] = []
    correct_by_variant: Counter[str] = Counter()
    total_by_variant: Counter[str] = Counter()
    correct_by_category: Counter[str] = Counter()
    total_by_category: Counter[str] = Counter()
    correct_by_file: Counter[str] = Counter()
    total_by_file: Counter[str] = Counter()

    for path in files:
        payload = read_json(path)
        for item in normalize_results(payload):
            total_records += 1
            if not matches_variant_filter(item, args.variant_keyword):
                continue

            attempt_records += 1
            variant = str(item.get("variant", ""))
            category = f"{item.get('category_id', '')}:{item.get('category_name', '')}"
            total_by_variant[variant] += 1
            total_by_category[category] += 1
            total_by_file[path.name] += 1

            if judgment_is_corrected(item, args.judgment_field):
                correct_by_variant[variant] += 1
                correct_by_category[category] += 1
                correct_by_file[path.name] += 1
                correct_rows.append(row_for(path, item, args.judgment_field))

    print(f"input_dir: {input_dir}")
    print(f"json_files: {len(files)}")
    print(f"all_records: {total_records}")
    if args.variant_keyword:
        print(f"variant_keyword: {args.variant_keyword}")
    print(f"attempt_records: {attempt_records}")
    print(f"correct_records: {len(correct_rows)}")
    rate = (len(correct_rows) / attempt_records * 100) if attempt_records else 0.0
    print(f"correct_rate: {rate:.2f}%")

    def print_counter(title: str, correct_counter: Counter[str], total_counter: Counter[str]) -> None:
        print(f"\n{title}")
        for key, total in total_counter.most_common(args.show_top):
            correct = correct_counter.get(key, 0)
            item_rate = (correct / total * 100) if total else 0.0
            print(f"  {correct:5d}/{total:<5d} {item_rate:6.2f}%  {key}")

    print_counter("by_file", correct_by_file, total_by_file)
    print_counter("by_variant", correct_by_variant, total_by_variant)
    print_counter("by_category", correct_by_category, total_by_category)

    if args.csv:
        write_csv(args.csv.expanduser().resolve(), correct_rows)
        print(f"\ncsv_written: {args.csv.expanduser().resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
