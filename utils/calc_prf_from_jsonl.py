#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def normalize_pred(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in {0, 1} else None
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in {"perfect", "fair_enough", "success", "passed", "pass", "1", "true"}:
        return 1
    if v in {"failed", "fail", "0", "false"}:
        return 0
    return None


def normalize_gt(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in {0, 1} else None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "success", "pass", "passed"}:
            return 1
        if v in {"0", "false", "failed", "fail"}:
            return 0
    return None


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 JSONL 中预测(task_status) vs 真值(status) 的准确率/精确率/召回率/F1。"
    )
    parser.add_argument("jsonl_path", help="输入 JSONL 文件路径")
    args = parser.parse_args()

    path = Path(args.jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    total_nonempty = 0
    valid = 0
    bad_json = 0
    missing = 0
    invalid_label = 0

    tp = fp = tn = fn = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_nonempty += 1
            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            pred_raw = get_nested(obj, "evaluation", "prediction", "task_status")
            gt_raw = get_nested(obj, "evaluation", "status")
            if pred_raw is None or gt_raw is None:
                missing += 1
                continue

            pred = normalize_pred(pred_raw)
            gt = normalize_gt(gt_raw)
            if pred is None or gt is None:
                invalid_label += 1
                continue

            valid += 1
            if pred == 1 and gt == 1:
                tp += 1
            elif pred == 1 and gt == 0:
                fp += 1
            elif pred == 0 and gt == 0:
                tn += 1
            else:
                fn += 1

    accuracy = safe_div(tp + tn, valid)
    precision = safe_div(tp, tp + fp)  # 正类=成功
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    print(f"文件: {path}")
    print(f"总非空行数: {total_nonempty}")
    print(f"纳入统计样本: {valid}")
    print(f"跳过(bad json): {bad_json}")
    print(f"跳过(缺字段): {missing}")
    print(f"跳过(标签非法): {invalid_label}")
    print()
    print("混淆矩阵(正类=成功):")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print()
    print(f"Accuracy : {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall   : {recall:.6f}")
    print(f"F1       : {f1:.6f}")


if __name__ == "__main__":
    main()

