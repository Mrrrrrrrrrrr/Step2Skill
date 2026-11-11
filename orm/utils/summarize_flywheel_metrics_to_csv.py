#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_INPUTS = [
    ("system_app", "2026021411011261"),
    ("third_party_app", "2026021411112512"),
    ("third_party_app", "2026021415524250"),
    ("hard_samples", "2026031017573591_rev2"),
    ("zhuhao", "2026032311274679_new"),
]


@dataclass
class Stats:
    scope: str
    group: str
    path: str
    total_nonempty: int = 0
    filtered_max_steps: int = 0
    labeled: int = 0
    valid: int = 0
    metric_valid: int = 0
    bad_json: int = 0
    missing: int = 0
    invalid_label: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def accuracy(self) -> float:
        return _safe_div(self.tp + self.tn, self.metric_valid)

    def success_precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    def success_recall(self) -> float:
        return _safe_div(self.tp, self.tp + self.fn)

    def failed_precision(self) -> float:
        return _safe_div(self.tn, self.tn + self.fn)

    def failed_recall(self) -> float:
        return _safe_div(self.tn, self.tn + self.fp)

    def f1(self) -> float:
        p = self.success_precision()
        r = self.success_recall()
        return _safe_div(2 * p * r, p + r)

    def to_row(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "group": self.group,
            "path": self.path,
            "total_nonempty": self.total_nonempty,
            "filtered_max_steps": self.filtered_max_steps,
            "labeled": self.labeled,
            "valid": self.valid,
            "metric_valid": self.metric_valid,
            "bad_json": self.bad_json,
            "missing": self.missing,
            "invalid_label": self.invalid_label,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": f"{self.accuracy():.6f}",
            "success_precision": f"{self.success_precision():.6f}",
            "success_recall": f"{self.success_recall():.6f}",
            "failed_precision": f"{self.failed_precision():.6f}",
            "failed_recall": f"{self.failed_recall():.6f}",
            "f1": f"{self.f1():.6f}",
        }


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _normalize_gt(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "success", "pass", "passed"):
            return 1
        if v in ("0", "false", "failed", "fail"):
            return 0
    return None


def _normalize_pred_task_status(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value in (0, 1) else None
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in ("perfect", "fair_enough", "success", "passed", "pass", "1", "true"):
        return 1
    if v in ("failed", "failure", "fail", "0", "false"):
        return 0
    return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            yield json.loads(raw)


def calc_stats_self(scope: str, group: str, path: Path) -> Stats:
    stats = Stats(scope=scope, group=group, path=str(path))
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            stats.total_nonempty += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                stats.bad_json += 1
                continue

            reason_raw = _get_nested(obj, "metadata", "reason")
            if isinstance(reason_raw, str) and reason_raw.strip() == "达到最大步数":
                stats.filtered_max_steps += 1
                continue

            # 过滤后剩余样本都计入有效样本（不要求有真值/预测）
            stats.valid += 1

            gt_raw = _get_nested(obj, "evaluation", "status")
            pred_raw = _get_nested(obj, "evaluation", "prediction", "task_status")
            gt = _normalize_gt(gt_raw)
            if gt is not None:
                stats.labeled += 1

            if gt_raw is None or pred_raw is None:
                stats.missing += 1
                continue

            pred = _normalize_pred_task_status(pred_raw)
            if gt is None or pred is None:
                stats.invalid_label += 1
                continue

            stats.metric_valid += 1
            if pred == 1 and gt == 1:
                stats.tp += 1
            elif pred == 1 and gt == 0:
                stats.fp += 1
            elif pred == 0 and gt == 0:
                stats.tn += 1
            else:
                stats.fn += 1
    return stats


def calc_stats_aliyun(scope: str, group: str, path: Path) -> Stats:
    """
    阿里 predicted.jsonl 口径：
    - 文件本身即有效样本，不再按 metadata.reason / metadata.status 过滤
    - valid = total_nonempty
    - 指标仅统计：
      * gt: evaluation.status in {0,1}
      * pred: evaluation.prediction.task_status in {perfect, fair_enough, failed}
      * env_error 等其他值不纳入 metric_valid
    """
    stats = Stats(scope=scope, group=group, path=str(path))
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            stats.total_nonempty += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                stats.bad_json += 1
                continue

            stats.valid += 1
            gt_raw = _get_nested(obj, "evaluation", "status")
            pred_raw = _get_nested(obj, "evaluation", "prediction", "task_status")

            gt = _normalize_gt(gt_raw)
            if gt is not None:
                stats.labeled += 1

            if gt_raw is None or pred_raw is None:
                stats.missing += 1
                continue

            pred = _normalize_pred_task_status(pred_raw)
            if gt is None or pred is None:
                stats.invalid_label += 1
                continue

            stats.metric_valid += 1
            if pred == 1 and gt == 1:
                stats.tp += 1
            elif pred == 1 and gt == 0:
                stats.fp += 1
            elif pred == 0 and gt == 0:
                stats.tn += 1
            else:
                stats.fn += 1
    return stats


def merge_stats(scope: str, group: str, items: List[Stats]) -> Stats:
    merged = Stats(scope=scope, group=group, path="MULTI")
    for it in items:
        merged.total_nonempty += it.total_nonempty
        merged.filtered_max_steps += it.filtered_max_steps
        merged.labeled += it.labeled
        merged.valid += it.valid
        merged.metric_valid += it.metric_valid
        merged.bad_json += it.bad_json
        merged.missing += it.missing
        merged.invalid_label += it.invalid_label
        merged.tp += it.tp
        merged.fp += it.fp
        merged.tn += it.tn
        merged.fn += it.fn
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总多个 tag 的 accuracy/precision 到 CSV"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/data_flywheel",
        help="tag jsonl 所在目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/data_flywheel/flywheel_metrics_summary.csv",
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="可选：自定义 tag 列表；不传则用内置 5 个 tag",
    )
    parser.add_argument(
        "--mode",
        choices=("self", "aliyun"),
        default="self",
        help="统计模式：self=统计 data/data_flywheel/<tag>.jsonl；aliyun=统计 data/data_flywheel/<tag>/predicted.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    output = Path(args.output).resolve()

    rows: List[Stats] = []
    if args.tags:
        plan = [("custom", t.strip()) for t in args.tags if t.strip()]
    else:
        plan = list(DEFAULT_INPUTS)

    for group, tag in plan:
        if args.mode == "aliyun":
            path = base_dir / tag / "predicted.jsonl"
            rows.append(calc_stats_aliyun(scope=tag, group=group, path=path))
        else:
            path = base_dir / f"{tag}.jsonl"
            rows.append(calc_stats_self(scope=tag, group=group, path=path))

    overall = merge_stats(scope="OVERALL", group="all", items=rows)
    all_rows = rows + [overall]

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].to_row().keys())
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row.to_row())

    print(f"wrote csv: {output}")
    for row in all_rows:
        print(
            f"{row.scope}: filtered_max_steps={row.filtered_max_steps}, "
            f"labeled={row.labeled}, valid={row.valid}, metric_valid={row.metric_valid}, "
            f"acc={row.accuracy():.4f}, "
            f"success_precision={row.success_precision():.4f}, "
            f"failed_precision={row.failed_precision():.4f}"
        )


if __name__ == "__main__":
    main()
