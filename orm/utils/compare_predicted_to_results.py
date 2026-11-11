#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"解析 JSONL 失败: {path}, 第 {line_no} 行, 错误: {e}") from e
    return records


def normalize_task_id(task_id: str) -> str:
    if task_id.endswith("_main"):
        return task_id[:-5]
    return task_id


def get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def extract_vote_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    # 格式1: 顶层 vote
    vote = obj.get("vote")
    if isinstance(vote, dict):
        return vote

    # 格式2: evaluation.prediction.vote
    vote = get_nested(obj, "evaluation", "prediction", "vote")
    if isinstance(vote, dict):
        return vote

    return {}


def extract_task_id(obj: Dict[str, Any]) -> Optional[str]:
    candidates = []

    if isinstance(obj.get("trajectory_id"), str):
        candidates.append(obj["trajectory_id"])

    vote = extract_vote_obj(obj)
    if isinstance(vote.get("trajectory_id"), str):
        candidates.append(vote["trajectory_id"])

    if isinstance(obj.get("task_id"), str):
        candidates.append(obj["task_id"])
        candidates.append(obj["task_id"] + "_main")

    for c in candidates:
        if c:
            return normalize_task_id(c)
    return None


def extract_display_id(obj: Dict[str, Any]) -> Optional[str]:
    if isinstance(obj.get("trajectory_id"), str):
        return obj["trajectory_id"]

    vote = extract_vote_obj(obj)
    if isinstance(vote.get("trajectory_id"), str):
        return vote["trajectory_id"]

    if isinstance(obj.get("task_id"), str):
        return obj["task_id"]

    return None


def extract_final_result(obj: Dict[str, Any]) -> Optional[str]:
    vote = extract_vote_obj(obj)
    value = vote.get("final_result")
    return value if isinstance(value, str) else None


def extract_vote_details(obj: Dict[str, Any]) -> Dict[str, str]:
    vote = extract_vote_obj(obj)
    vote_details = vote.get("vote_details")
    if not isinstance(vote_details, dict):
        return {}

    result = {}
    for k, v in vote_details.items():
        if isinstance(k, str) and isinstance(v, str):
            result[k] = v
    return result


def build_task_map(records: List[Dict[str, Any]], source_name: str) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    task_map: Dict[str, Dict[str, Any]] = {}
    missing_ids = []
    duplicate_ids = []

    for idx, obj in enumerate(records, 1):
        task_id = extract_task_id(obj)
        if task_id is None:
            missing_ids.append(f"{source_name} 第 {idx} 条")
            continue

        if task_id in task_map:
            duplicate_ids.append(task_id)
            continue

        task_map[task_id] = {
            "task_id": task_id,
            "display_id": extract_display_id(obj) or task_id,
            "final_result": extract_final_result(obj),
            "vote_details": extract_vote_details(obj),
            "raw": obj,
        }

    return task_map, missing_ids, duplicate_ids


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def format_pair_counter(pair_counter: Counter) -> List[Dict[str, Any]]:
    return [
        {
            "predicted": p,
            "results": r,
            "count": cnt,
        }
        for (p, r), cnt in sorted(pair_counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ]


def normalize_result_label(value: Any) -> str:
    s = str(value)
    if s in {"perfect", "fair_enough"}:
        return "acceptable"
    return s


def build_merged_final_result_stats(
    predicted_map: Dict[str, Dict[str, Any]],
    results_map: Dict[str, Dict[str, Any]],
    intersection: List[str],
) -> Dict[str, Any]:
    pair_counter = Counter()
    same_count = 0
    diff_items = []

    for task_id in intersection:
        p_raw = predicted_map[task_id]["final_result"]
        r_raw = results_map[task_id]["final_result"]
        p = normalize_result_label(p_raw)
        r = normalize_result_label(r_raw)
        pair_counter[(p, r)] += 1
        if p == r:
            same_count += 1
        else:
            diff_items.append(
                {
                    "task_id": task_id,
                    "predicted_display_id": predicted_map[task_id]["display_id"],
                    "results_display_id": results_map[task_id]["display_id"],
                    "predicted_final_result_merged": p,
                    "results_final_result_merged": r,
                    "predicted_final_result_raw": p_raw,
                    "results_final_result_raw": r_raw,
                }
            )

    total = len(intersection)
    diff_count = total - same_count
    return {
        "merge_rule": {
            "perfect": "acceptable",
            "fair_enough": "acceptable",
        },
        "intersection_task_count": total,
        "same_count": same_count,
        "diff_count": diff_count,
        "same_ratio": (same_count / total) if total else None,
        "pair_stats": format_pair_counter(pair_counter),
        "diff_items_count": len(diff_items),
        "diff_items": diff_items,
    }


def build_merged_vote_details_stats(vote_details_report: Dict[str, Any]) -> Dict[str, Any]:
    merge_rule = {"perfect": "acceptable", "fair_enough": "acceptable"}
    per_model = vote_details_report.get("per_model", {})
    merged_per_model: Dict[str, Dict[str, Any]] = {}
    total_comparable = 0
    total_same = 0
    total_diff = 0

    for model, stats in per_model.items():
        pair_counter = Counter()
        diff_items = []

        for pair_item in stats.get("pair_stats", []):
            p = normalize_result_label(pair_item["predicted"])
            r = normalize_result_label(pair_item["results"])
            pair_counter[(p, r)] += pair_item["count"]

        comparable = int(stats.get("comparable", 0))
        for item in stats.get("diff_items", []):
            p_merged = normalize_result_label(item["predicted_judge"])
            r_merged = normalize_result_label(item["results_judge"])
            if p_merged != r_merged:
                diff_items.append(
                    {
                        "task_id": item["task_id"],
                        "predicted_display_id": item["predicted_display_id"],
                        "results_display_id": item["results_display_id"],
                        "predicted_judge_merged": p_merged,
                        "results_judge_merged": r_merged,
                        "predicted_judge_raw": item["predicted_judge"],
                        "results_judge_raw": item["results_judge"],
                    }
                )

        same = sum(cnt for (p, r), cnt in pair_counter.items() if p == r)
        diff = comparable - same
        ratio = same / comparable if comparable else 0.0

        total_comparable += comparable
        total_same += same
        total_diff += diff
        merged_per_model[model] = {
            "comparable": comparable,
            "same": same,
            "diff": diff,
            "same_ratio": ratio,
            "pair_stats": format_pair_counter(pair_counter),
            "diff_items_count": len(diff_items),
            "diff_items": diff_items,
        }

    return {
        "merge_rule": merge_rule,
        "total_comparable": total_comparable,
        "total_same": total_same,
        "total_diff": total_diff,
        "overall_same_ratio": (total_same / total_comparable) if total_comparable else None,
        "per_model": merged_per_model,
    }


def compare_final_result(
    predicted_map: Dict[str, Dict[str, Any]],
    results_map: Dict[str, Dict[str, Any]],
    intersection: List[str],
) -> Dict[str, Any]:
    print_header("1. 交集任务下 final_result 差异统计")

    pair_counter = Counter()
    same_count = 0
    diff_items = []

    for task_id in intersection:
        p = predicted_map[task_id]["final_result"]
        r = results_map[task_id]["final_result"]
        pair_counter[(str(p), str(r))] += 1

        if p == r:
            same_count += 1
        else:
            diff_items.append(
                {
                    "task_id": task_id,
                    "predicted_display_id": predicted_map[task_id]["display_id"],
                    "results_display_id": results_map[task_id]["display_id"],
                    "predicted_final_result": p,
                    "results_final_result": r,
                }
            )

    total = len(intersection)
    diff_count = total - same_count

    print(f"交集任务数: {total}")
    print(f"final_result 一致: {same_count}")
    print(f"final_result 不一致: {diff_count}")

    print("\nfinal_result 配对统计:")
    pair_stats = format_pair_counter(pair_counter)
    for item in pair_stats:
        p = item["predicted"]
        r = item["results"]
        cnt = item["count"]
        print(f"  predicted={p!r} | results={r!r} -> {cnt}")

    print("\nfinal_result 不一致任务明细:前20")
    if not diff_items:
        print("  无")
    else:
        for item in diff_items[:20]:
            print(
                f"  task_id={item['task_id']}\n"
                f"    predicted_display_id: {item['predicted_display_id']}\n"
                f"    results_display_id:   {item['results_display_id']}\n"
                f"    predicted_final_result: {item['predicted_final_result']}\n"
                f"    results_final_result:   {item['results_final_result']}"
            )

    return {
        "intersection_task_count": total,
        "same_count": same_count,
        "diff_count": diff_count,
        "same_ratio": (same_count / total) if total else None,
        "pair_stats": pair_stats,
        "diff_items_count": len(diff_items),
        "diff_items": diff_items,
    }


def compare_vote_details_consistency(
    predicted_map: Dict[str, Dict[str, Any]],
    results_map: Dict[str, Dict[str, Any]],
    intersection: List[str],
) -> Dict[str, Any]:
    print_header("2. 交集任务下 vote_details 同模型判断一致性")

    per_model_stats = defaultdict(
        lambda: {
            "comparable": 0,
            "same": 0,
            "diff": 0,
            "pair_counter": Counter(),
            "diff_items": [],
        }
    )

    total_comparable = 0
    total_same = 0
    total_diff = 0

    for task_id in intersection:
        p_vote_details = predicted_map[task_id]["vote_details"]
        r_vote_details = results_map[task_id]["vote_details"]

        common_models = set(p_vote_details.keys()) & set(r_vote_details.keys())
        if not common_models:
            continue

        for model in sorted(common_models):
            p_judge = p_vote_details[model]
            r_judge = r_vote_details[model]

            stats = per_model_stats[model]
            stats["comparable"] += 1
            stats["pair_counter"][(p_judge, r_judge)] += 1

            total_comparable += 1

            if p_judge == r_judge:
                stats["same"] += 1
                total_same += 1
            else:
                stats["diff"] += 1
                total_diff += 1
                stats["diff_items"].append(
                    {
                        "task_id": task_id,
                        "predicted_display_id": predicted_map[task_id]["display_id"],
                        "results_display_id": results_map[task_id]["display_id"],
                        "predicted_judge": p_judge,
                        "results_judge": r_judge,
                    }
                )

    print(f"可比的(task, model)总数: {total_comparable}")
    print(f"一致数: {total_same}")
    print(f"不一致数: {total_diff}")
    if total_comparable > 0:
        print(f"总体一致率: {total_same / total_comparable:.4f}")
    else:
        print("总体一致率: N/A")

    if not per_model_stats:
        print("\n没有任何可比的同任务同模型 vote_details。")
        return {
            "total_comparable": total_comparable,
            "total_same": total_same,
            "total_diff": total_diff,
            "overall_same_ratio": None,
            "per_model": {},
        }

    print("\n按模型统计:")
    per_model_report: Dict[str, Dict[str, Any]] = {}
    for model in sorted(per_model_stats.keys()):
        stats = per_model_stats[model]
        comparable = stats["comparable"]
        same = stats["same"]
        diff = stats["diff"]
        ratio = same / comparable if comparable else 0.0

        print(f"\n- 模型: {model}")
        print(f"  可比数: {comparable}")
        print(f"  一致数: {same}")
        print(f"  不一致数: {diff}")
        print(f"  一致率: {ratio:.4f}")
        print("  判断配对统计:")
        pair_stats = format_pair_counter(stats["pair_counter"])
        for item in pair_stats:
            p_judge = item["predicted"]
            r_judge = item["results"]
            cnt = item["count"]
            print(f"    predicted={p_judge!r} | results={r_judge!r} -> {cnt}")

        print("  不一致任务明细:")
        if not stats["diff_items"]:
            print("    无")
        else:
            for item in stats["diff_items"]:
                print(
                    f"    task_id={item['task_id']}\n"
                    f"      predicted_display_id: {item['predicted_display_id']}\n"
                    f"      results_display_id:   {item['results_display_id']}\n"
                    f"      predicted_judge: {item['predicted_judge']}\n"
                    f"      results_judge:   {item['results_judge']}"
                )

        per_model_report[model] = {
            "comparable": comparable,
            "same": same,
            "diff": diff,
            "same_ratio": ratio,
            "pair_stats": pair_stats,
            "diff_items_count": len(stats["diff_items"]),
            "diff_items": stats["diff_items"],
        }

    return {
        "total_comparable": total_comparable,
        "total_same": total_same,
        "total_diff": total_diff,
        "overall_same_ratio": (total_same / total_comparable) if total_comparable else None,
        "per_model": per_model_report,
    }


def main():
    parser = argparse.ArgumentParser(
        description="比较两个 JSONL 文件在交集任务上的 final_result 差异，以及 vote_details 的同任务同模型一致性。"
    )
    parser.add_argument(
        "predicted_path",
        nargs="?",
        default="data/data_flywheel/2026031618351974/predicted.jsonl",
        help="第一个 JSONL 路径",
    )
    parser.add_argument(
        "results_path",
        nargs="?",
        default="data/data_flywheel/2026031618351974/results_sjg.jsonl",
        help="第二个 JSONL 路径",
    )
    args = parser.parse_args()

    predicted_path = Path(args.predicted_path)
    results_path = Path(args.results_path)

    predicted_records = load_jsonl(str(predicted_path))
    results_records = load_jsonl(str(results_path))

    predicted_map, predicted_missing_ids, predicted_duplicate_ids = build_task_map(
        predicted_records, "predicted"
    )
    results_map, results_missing_ids, results_duplicate_ids = build_task_map(
        results_records, "results"
    )

    print_header("基础信息")
    print(f"predicted 文件: {predicted_path}")
    print(f"results 文件:   {results_path}")
    print(f"predicted 总条数: {len(predicted_records)}")
    print(f"results 总条数:   {len(results_records)}")
    print(f"predicted 可识别任务数: {len(predicted_map)}")
    print(f"results 可识别任务数:   {len(results_map)}")

    if predicted_missing_ids:
        print(f"predicted 缺少任务 ID 的条数: {len(predicted_missing_ids)}")
    if results_missing_ids:
        print(f"results 缺少任务 ID 的条数: {len(results_missing_ids)}")

    if predicted_duplicate_ids:
        print(f"predicted 重复任务 ID 数: {len(predicted_duplicate_ids)}")
        print("  重复任务 ID 示例:", predicted_duplicate_ids[:10])
    if results_duplicate_ids:
        print(f"results 重复任务 ID 数: {len(results_duplicate_ids)}")
        print("  重复任务 ID 示例:", results_duplicate_ids[:10])

    intersection = sorted(set(predicted_map.keys()) & set(results_map.keys()))
    predicted_only = sorted(set(predicted_map.keys()) - set(results_map.keys()))
    results_only = sorted(set(results_map.keys()) - set(predicted_map.keys()))

    print(f"交集任务数: {len(intersection)}")
    print(f"仅 predicted 有的任务数: {len(predicted_only)}")
    print(f"仅 results 有的任务数:   {len(results_only)}")

    final_result_report = compare_final_result(predicted_map, results_map, intersection)
    vote_details_report = compare_vote_details_consistency(predicted_map, results_map, intersection)

    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "predicted_path": str(predicted_path),
            "results_path": str(results_path),
        },
        "overview": {
            "predicted_total_records": len(predicted_records),
            "results_total_records": len(results_records),
            "predicted_identified_tasks": len(predicted_map),
            "results_identified_tasks": len(results_map),
            "intersection_task_count": len(intersection),
            "predicted_only_count": len(predicted_only),
            "results_only_count": len(results_only),
            "predicted_missing_id_count": len(predicted_missing_ids),
            "results_missing_id_count": len(results_missing_ids),
            "predicted_duplicate_id_count": len(predicted_duplicate_ids),
            "results_duplicate_id_count": len(results_duplicate_ids),
        },
        "final_result": final_result_report,
        "vote_details_consistency": vote_details_report,
        "details": {
            "predicted_only_task_ids": predicted_only,
            "results_only_task_ids": results_only,
            "predicted_missing_id_records": predicted_missing_ids,
            "results_missing_id_records": results_missing_ids,
            "predicted_duplicate_ids": predicted_duplicate_ids,
            "results_duplicate_ids": results_duplicate_ids,
        },
        "merged_stats": {
            "final_result": build_merged_final_result_stats(predicted_map, results_map, intersection),
            "vote_details_consistency": build_merged_vote_details_stats(vote_details_report),
        },
    }

    output_path = predicted_path.parent / f"{predicted_path.stem}_comparison_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 报告已输出: {output_path}")


if __name__ == "__main__":
    main()
