#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set


def get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def normalize_pred_status(value: Any) -> Optional[int]:
    if not isinstance(value, str):
        return None
    if value in {"perfect", "fair_enough"}:
        return 1
    if value == "failed":
        return 0
    return None


def normalize_gt_status(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in {0, 1}:
            return value
        return None
    return None


def normalize_task_id(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    if value.endswith("_main"):
        return value[:-5]
    return value


def extract_task_id(obj: Dict[str, Any]) -> Optional[str]:
    task_id = normalize_task_id(obj.get("task_id"))
    if task_id is not None:
        return task_id
    return normalize_task_id(obj.get("trajectory_id"))


def build_empty_stats() -> Dict[str, Any]:
    return {
        "total_lines": 0,
        "valid_pairs": 0,
        "correct": 0,
        "skipped_missing": 0,
        "skipped_invalid": 0,
    }


def print_stats(title: str, file_path: Path, stats: Dict[str, Any]) -> None:
    print(title)
    print(f"文件: {file_path}")
    print(f"总有效JSON行数: {stats['total_lines']}")
    print(f"纳入统计样本数: {stats['valid_pairs']}")
    print(f"预测正确数: {stats['correct']}")
    if stats["valid_pairs"] == 0:
        print("有GT版本的准确率: N/A (没有可统计样本)")
    else:
        print(f"有GT版本的准确率: {stats['correct'] / stats['valid_pairs']:.6f}")
    print(f"跳过(缺字段)行数: {stats['skipped_missing']}")
    print(f"跳过(取值不在统计集合/坏JSON)行数: {stats['skipped_invalid']}")


def calc_base_stats(jsonl_path: Path) -> Dict[str, Any]:
    stats = build_empty_stats()
    included_task_ids: Set[str] = set()
    gt_by_task_id: Dict[str, int] = {}
    total_lines = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_invalid"] += 1
                continue

            gt_raw = get_nested(obj, "evaluation", "status")
            pred_raw = get_nested(obj, "evaluation", "prediction", "task_status")
            task_id = extract_task_id(obj)

            gt = normalize_gt_status(gt_raw)
            pred = normalize_pred_status(pred_raw)

            if task_id is None or gt is None or pred is None:
                if gt_raw is None or pred_raw is None:
                    stats["skipped_missing"] += 1
                else:
                    stats["skipped_invalid"] += 1
                continue

            stats["valid_pairs"] += 1
            included_task_ids.add(task_id)
            gt_by_task_id[task_id] = gt
            if gt == pred:
                stats["correct"] += 1

    stats["total_lines"] = total_lines
    stats["included_task_ids"] = included_task_ids
    stats["gt_by_task_id"] = gt_by_task_id
    return stats


def calc_replay_stats(
    replay_jsonl_path: Path,
    included_task_ids: Set[str],
    gt_by_task_id: Dict[str, int],
) -> Dict[str, Any]:
    stats = build_empty_stats()
    total_lines = 0
    found_subset_task_ids: Set[str] = set()

    with open(replay_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_invalid"] += 1
                continue

            task_id = extract_task_id(obj)
            if task_id is None:
                stats["skipped_missing"] += 1
                continue
            if task_id not in included_task_ids:
                continue
            found_subset_task_ids.add(task_id)

            pred_raw = get_nested(obj, "vote", "final_result")
            gt = gt_by_task_id.get(task_id)
            pred = normalize_pred_status(pred_raw)

            if gt is None or pred is None:
                if pred_raw is None:
                    stats["skipped_missing"] += 1
                else:
                    stats["skipped_invalid"] += 1
                continue

            stats["valid_pairs"] += 1
            if gt == pred:
                stats["correct"] += 1

    # 原版纳入子集中，但复现版文件内不存在的任务，记为缺字段/缺记录
    missing_in_replay = len(included_task_ids - found_subset_task_ids)
    stats["skipped_missing"] += missing_in_replay
    stats["total_lines"] = total_lines
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "统计 JSONL 的“有GT版本的准确率”。"
            "原版使用 evaluation.prediction.task_status；"
            "复现版使用 vote.final_result。"
            "perfect/fair_enough 视为 1，failed 视为 0，其他值不统计。"
        )
    )
    parser.add_argument("jsonl_path", help="原版 JSONL 路径")
    parser.add_argument("replay_jsonl_path", nargs="?", help="复现版 JSONL 路径（可选）")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"文件不存在: {jsonl_path}")

    base_stats = calc_base_stats(jsonl_path)
    included_task_ids = base_stats.pop("included_task_ids")
    gt_by_task_id = base_stats.pop("gt_by_task_id")
    print_stats("", jsonl_path, base_stats)

    if args.replay_jsonl_path:
        replay_jsonl_path = Path(args.replay_jsonl_path)
        if not replay_jsonl_path.exists():
            raise FileNotFoundError(f"文件不存在: {replay_jsonl_path}")
        replay_stats = calc_replay_stats(replay_jsonl_path, included_task_ids, gt_by_task_id)
        print()
        print_stats("复现版:", replay_jsonl_path, replay_stats)


if __name__ == "__main__":
    main()
