#!/usr/bin/env python3
"""Count successful examples in a MobileGym judge result JSON file.

Success is defined by judge_pass in the MobileGym skill judge output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if isinstance(value, dict):
                records.append(value)
    return records


def normalize_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        items = payload["results"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise ValueError("input JSON must be an object, list, or object with results")
    return [item for item in items if isinstance(item, dict)]


def is_success(item: Dict[str, Any]) -> bool:
    judge_pass = item.get("judge_pass")
    if isinstance(judge_pass, bool):
        return judge_pass
    if isinstance(judge_pass, str):
        normalized = judge_pass.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return False


def rollout_run_dir(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    path = Path(value.strip()).expanduser()
    parts = path.parts
    if len(parts) >= 2 and parts[-2:] == ("trajectory", "_ExecTask"):
        path = path.parent.parent
    return str(path.resolve())


def infer_manifest_path(input_path: Path, items: List[Dict[str, Any]]) -> Optional[Path]:
    candidates: List[Path] = []

    for item in items:
        run_dir_text = rollout_run_dir(item.get("rollout_dir"))
        if not run_dir_text:
            continue
        run_dir = Path(run_dir_text)
        candidates.extend(
            [
                run_dir.parent.parent / "batch_runs.jsonl",
                run_dir.parent / "batch_runs.jsonl",
            ]
        )

    candidates.extend(
        [
            input_path.parent / "batch_runs.jsonl",
            Path.cwd() / "batch_runs.jsonl",
        ]
    )

    seen = set()
    for candidate in candidates:
        resolved = str(candidate.expanduser().resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        path = Path(resolved)
        if path.is_file():
            return path
    return None


def load_manifest_groups(manifest_path: Optional[Path]) -> Dict[str, str]:
    if manifest_path is None:
        return {}
    mapping: Dict[str, str] = {}
    for record in read_jsonl(manifest_path.expanduser().resolve()):
        item_index = record.get("item_index")
        run_dir = rollout_run_dir(record.get("run_dir"))
        if item_index is None or not run_dir:
            continue
        mapping[run_dir] = str(item_index)
    return mapping


def task_group_key(item: Dict[str, Any], run_dir_to_item_index: Dict[str, str]) -> str:
    run_dir = rollout_run_dir(item.get("rollout_dir"))
    if run_dir and run_dir in run_dir_to_item_index:
        return f"item:{run_dir_to_item_index[run_dir]}"
    instruction = item.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return f"instruction:{instruction.strip()}"
    task_id = item.get("task_id")
    return f"task_id:{task_id}"


def grouped_success_rates(
    items: List[Dict[str, Any]],
    run_dir_to_item_index: Dict[str, str],
    repeats: int,
) -> Tuple[float, float]:
    groups: Dict[str, List[bool]] = {}
    for item in items:
        groups.setdefault(task_group_key(item, run_dir_to_item_index), []).append(is_success(item))

    task_count = len(groups)
    if task_count == 0:
        return 0.0, 0.0

    highest_success_count = sum(1 for values in groups.values() if any(values))
    lowest_success_count = sum(1 for values in groups.values() if len(values) >= repeats and all(values))
    return highest_success_count / task_count, lowest_success_count / task_count


def build_summary(
    items: Iterable[Dict[str, Any]],
    *,
    run_dir_to_item_index: Dict[str, str],
    repeats: int,
) -> Dict[str, Any]:
    item_list = list(items)
    successes = [item for item in item_list if is_success(item)]
    failures = [item for item in item_list if not is_success(item)]

    total = len(item_list)
    success_count = len(successes)
    failure_count = len(failures)
    success_rate = success_count / total if total else 0.0
    highest_success_rate, lowest_success_rate = grouped_success_rates(
        item_list,
        run_dir_to_item_index,
        repeats,
    )

    return {
        "total": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": round(success_rate, 6),
        "highest_success_rate": round(highest_success_rate, 6),
        "lowest_success_rate": round(lowest_success_rate, 6),
    }


def default_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_success_rate.json"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count successful examples in MobileGym judge result JSON.")
    parser.add_argument("--input", required=True, type=Path, help="path to mobilegym_skill_result.json")
    parser.add_argument("--manifest", type=Path, default=None, help="optional path to batch_runs.jsonl for grouping repeats")
    parser.add_argument("--repeats", type=int, default=3, help="number of repeated runs per original task")
    parser.add_argument("--output-dir", type=Path, default=Path("sr"), help="directory for the summary JSON")
    parser.add_argument("--output", type=Path, default=None, help="optional explicit output JSON path")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent; use 0 for compact output")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    input_path = args.input.expanduser().resolve()
    payload = read_json(input_path)
    items = normalize_items(payload)
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else infer_manifest_path(input_path, items)
    run_dir_to_item_index = load_manifest_groups(manifest_path)
    summary = build_summary(items, run_dir_to_item_index=run_dir_to_item_index, repeats=args.repeats)

    output_path = args.output.expanduser().resolve() if args.output else default_output_path(input_path, args.output_dir).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=None if args.indent == 0 else args.indent)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
