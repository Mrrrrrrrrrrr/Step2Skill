#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"解析失败: {path}, 第 {line_no} 行, 错误: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
    return records


def get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def normalize_task_id(task_id: str) -> str:
    task_id = task_id.strip()
    if task_id.endswith("_main"):
        return task_id[:-5]
    return task_id


def extract_vote_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    vote = obj.get("vote")
    if isinstance(vote, dict):
        return vote
    vote = get_nested(obj, "evaluation", "prediction", "vote")
    if isinstance(vote, dict):
        return vote
    return {}


def extract_task_id(obj: Dict[str, Any]) -> Optional[str]:
    candidates: List[str] = []
    for key in ("trajectory_id", "task_id"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    vote = extract_vote_obj(obj)
    vote_tid = vote.get("trajectory_id")
    if isinstance(vote_tid, str) and vote_tid.strip():
        candidates.append(vote_tid.strip())

    for candidate in candidates:
        normalized = normalize_task_id(candidate)
        if normalized:
            return normalized
    return None


def extract_final_result(obj: Dict[str, Any]) -> Optional[str]:
    vote = extract_vote_obj(obj)
    value = vote.get("final_result")
    if isinstance(value, str):
        return value.strip().lower()
    return None


def build_result_map(records: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}
    for obj in records:
        task_id = extract_task_id(obj)
        if not task_id:
            continue
        if task_id in result:
            continue
        result[task_id] = extract_final_result(obj)
    return result


def summarize_participated_samples(batch_dir: Path, predicted_ids: Set[str]) -> Tuple[List[str], List[str]]:
    existing_dirs: Set[str] = set()
    if batch_dir.exists():
        for child in batch_dir.iterdir():
            if child.is_dir() and child.name.endswith("_main"):
                existing_dirs.add(child.name)

    participated_dirs = sorted(f"{task_id}_main" for task_id in predicted_ids)
    existed_and_participated = [name for name in participated_dirs if name in existing_dirs]
    return participated_dirs, existed_and_participated


def print_id_samples(title: str, ids: List[str], limit: int) -> None:
    print(f"\n{title} (共 {len(ids)}):")
    if not ids:
        print("  无")
        return
    for item in ids[:limit]:
        print(f"  {item}")
    if len(ids) > limit:
        print(f"  ... 其余 {len(ids) - limit} 个未展示")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较 predicted/results 的 failed 差异，并输出参与预测样本。"
    )
    parser.add_argument(
        "--predicted",
        default="/var/code/zhaoj/orm/data/data_flywheel/2026031618351974/predicted.jsonl",
        help="predicted.jsonl 路径",
    )
    parser.add_argument(
        "--results",
        default="/var/code/zhaoj/orm/data/data_flywheel/2026031618351974/results.jsonl",
        help="results.jsonl 路径",
    )
    parser.add_argument(
        "--batch-dir",
        default="/var/code/zhaoj/orm/data/data_flywheel/2026031618351974",
        help="批次目录路径（用于筛选 *_main 样本目录）",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=50,
        help="打印任务 ID 的最大数量",
    )
    args = parser.parse_args()

    predicted_path = Path(args.predicted).resolve()
    results_path = Path(args.results).resolve()
    batch_dir = Path(args.batch_dir).resolve()

    predicted_records = load_jsonl(predicted_path)
    results_records = load_jsonl(results_path)
    predicted_map = build_result_map(predicted_records)
    results_map = build_result_map(results_records)

    predicted_ids = set(predicted_map.keys())
    results_ids = set(results_map.keys())
    intersection = sorted(predicted_ids & results_ids)

    predicted_failed = {tid for tid, fr in predicted_map.items() if fr == "failed"}
    results_failed = {tid for tid, fr in results_map.items() if fr == "failed"}

    both_failed = sorted(predicted_failed & results_failed)
    predicted_only_failed = sorted(predicted_failed - results_failed)
    results_only_failed = sorted(results_failed - predicted_failed)

    inter_pred_failed = {tid for tid in intersection if predicted_map.get(tid) == "failed"}
    inter_res_failed = {tid for tid in intersection if results_map.get(tid) == "failed"}

    confusion = {
        "pred_failed_res_failed": 0,
        "pred_failed_res_not_failed": 0,
        "pred_not_failed_res_failed": 0,
        "pred_not_failed_res_not_failed": 0,
    }
    for tid in intersection:
        p_failed = predicted_map.get(tid) == "failed"
        r_failed = results_map.get(tid) == "failed"
        if p_failed and r_failed:
            confusion["pred_failed_res_failed"] += 1
        elif p_failed and not r_failed:
            confusion["pred_failed_res_not_failed"] += 1
        elif (not p_failed) and r_failed:
            confusion["pred_not_failed_res_failed"] += 1
        else:
            confusion["pred_not_failed_res_not_failed"] += 1

    participated_dirs, existed_and_participated = summarize_participated_samples(
        batch_dir=batch_dir, predicted_ids=predicted_ids
    )

    print("=" * 90)
    print("基础信息")
    print("=" * 90)
    print(f"predicted: {predicted_path}")
    print(f"results:   {results_path}")
    print(f"batch_dir: {batch_dir}")
    print(f"predicted 条数: {len(predicted_records)}")
    print(f"results 条数:   {len(results_records)}")
    print(f"predicted 任务数: {len(predicted_ids)}")
    print(f"results 任务数:   {len(results_ids)}")
    print(f"任务交集数: {len(intersection)}")

    print("\n" + "=" * 90)
    print("参与预测样本过滤结果（基于 predicted task_id）")
    print("=" * 90)
    print(f"参与预测样本目录数（理论）: {len(participated_dirs)}")
    print(f"batch_dir 中实际存在且参与预测的目录数: {len(existed_and_participated)}")
    print_id_samples("参与预测目录示例", existed_and_participated, args.print_limit)

    print("\n" + "=" * 90)
    print("final_result='failed' 差异统计")
    print("=" * 90)
    print(f"predicted failed 数: {len(predicted_failed)}")
    print(f"results failed 数:   {len(results_failed)}")
    print(f"交集任务中 predicted failed 数: {len(inter_pred_failed)}")
    print(f"交集任务中 results failed 数:   {len(inter_res_failed)}")
    print(f"双方都 failed: {len(both_failed)}")
    print(f"仅 predicted failed: {len(predicted_only_failed)}")
    print(f"仅 results failed:   {len(results_only_failed)}")

    print("\n交集混淆统计（failed vs 非failed）:")
    for k, v in confusion.items():
        print(f"  {k}: {v}")

    print_id_samples("仅 predicted failed 的 task_id", predicted_only_failed, args.print_limit)
    print_id_samples("仅 results failed 的 task_id", results_only_failed, args.print_limit)


if __name__ == "__main__":
    main()
