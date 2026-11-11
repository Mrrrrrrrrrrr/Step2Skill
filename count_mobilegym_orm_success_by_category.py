#!/usr/bin/env python3
"""Count MobileGym judge success rates by trajectory category.

Success is defined exactly as in count_mobilegym_orm_success.py: judge_pass is true.
Categories are inferred from rollout_dir paths such as:
  .../trajs_0829/ad_popup/20260828_172458/trajectory/_ExecTask
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from count_mobilegym_orm_success import (
    build_summary,
    infer_manifest_path,
    load_manifest_groups,
    normalize_items,
    read_json,
)


TIMESTAMP_DIR_RE = re.compile(r"^\d{8}_\d+(?:_.+)?$")


def unwrap_judge_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Support both raw judge results and classified results containing a nested judge."""
    judge = item.get("judge")
    if isinstance(judge, dict):
        merged = dict(judge)
        for key in ("task_id", "instruction"):
            if key not in merged and key in item:
                merged[key] = item[key]
        return merged
    return item


def normalized_path_parts(value: Any) -> List[str]:
    if not isinstance(value, str):
        return []
    text = value.strip().replace("\\", "/").strip("/")
    if not text:
        return []
    return [part for part in text.split("/") if part]


def looks_like_run_dir(name: str) -> bool:
    return bool(TIMESTAMP_DIR_RE.fullmatch(name))


def category_from_rollout_dir(value: Any) -> str:
    parts = normalized_path_parts(value)
    if not parts:
        return "unknown"

    if len(parts) >= 3 and parts[-2:] == ["trajectory", "_ExecTask"]:
        before_trajectory = parts[-3]
        if len(parts) >= 4 and looks_like_run_dir(before_trajectory):
            return parts[-4]
        return before_trajectory

    if len(parts) >= 2 and parts[-1] == "trajectory":
        before_trajectory = parts[-2]
        if len(parts) >= 3 and looks_like_run_dir(before_trajectory):
            return parts[-3]
        return before_trajectory

    if len(parts) >= 2 and looks_like_run_dir(parts[-1]):
        return parts[-2]

    return parts[-1]


def discover_categories(root: Optional[Path]) -> Set[str]:
    if root is None:
        return set()
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--trajectory-root does not exist or is not a directory: {root}")
    return {path.name for path in root.iterdir() if path.is_dir()}


def build_category_summary(
    items: Iterable[Dict[str, Any]],
    *,
    run_dir_to_item_index: Dict[str, str],
    repeats: int,
    known_categories: Set[str],
) -> Dict[str, Any]:
    item_list = [unwrap_judge_item(item) for item in items]
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in item_list:
        category = category_from_rollout_dir(item.get("rollout_dir"))
        by_category[category].append(item)

    for category in sorted(known_categories):
        by_category.setdefault(category, [])

    category_summaries = {
        category: build_summary(category_items, run_dir_to_item_index=run_dir_to_item_index, repeats=repeats)
        for category, category_items in sorted(by_category.items())
    }

    return {
        "overall": build_summary(item_list, run_dir_to_item_index=run_dir_to_item_index, repeats=repeats),
        "by_category": category_summaries,
    }


def default_output_path(input_path: Path, output_dir: Optional[Path]) -> Path:
    base_dir = output_dir if output_dir is not None else input_path.parent
    return base_dir / f"{input_path.stem}_success_rate_by_category.json"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count MobileGym judge success rates by trajectory category.")
    parser.add_argument("--input", required=True, type=Path, help="path to MobileGym judge result JSON")
    parser.add_argument("--manifest", type=Path, default=None, help="optional path to batch_runs.jsonl for grouping repeats")
    parser.add_argument("--repeats", type=int, default=3, help="number of repeated runs per original task")
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=None,
        help="optional root whose direct subdirectories are included as categories, even if empty",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="optional directory for the summary JSON; defaults to the input JSON directory",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional explicit output JSON path")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent; use 0 for compact output")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    input_path = args.input.expanduser().resolve()
    payload = read_json(input_path)
    items = [unwrap_judge_item(item) for item in normalize_items(payload)]
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else infer_manifest_path(input_path, items)
    run_dir_to_item_index = load_manifest_groups(manifest_path)
    known_categories = discover_categories(args.trajectory_root)
    summary = build_category_summary(
        items,
        run_dir_to_item_index=run_dir_to_item_index,
        repeats=args.repeats,
        known_categories=known_categories,
    )

    output_path = args.output.expanduser().resolve() if args.output else default_output_path(input_path, args.output_dir).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=None if args.indent == 0 else args.indent)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
