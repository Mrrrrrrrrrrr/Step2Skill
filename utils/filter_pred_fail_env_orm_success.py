#!/usr/bin/env python3
"""
从 jsonl 中筛出：prediction.task_status 为「失败」或「环境错误」类，
且 orm_prediction.is_task_success 归一化为 1 的样本，写入新 jsonl。

默认 task_status 取值为 failed、env_error（与当前数据集中取值一致，见数据盘点）。
ORM 归一化规则与 compare_prediction_vs_orm.py 一致。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_prediction_vs_orm import _get, iter_jsonl, normalize_is_task_success


def _parse_statuses(s: str) -> frozenset[str]:
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    return frozenset(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="输入 jsonl")
    ap.add_argument(
        "--output",
        default="",
        help="输出 jsonl（默认：<stem>_pred_fail_env_orm1.jsonl）",
    )
    ap.add_argument(
        "--pred-statuses",
        default="failed,env_error",
        help="视为「失败/环境错误」的 task_status 列表，逗号分隔、不区分大小写（默认 failed,env_error）",
    )
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"输入不存在: {in_path}")

    out_path = (
        Path(args.output).expanduser()
        if args.output
        else (in_path.parent / f"{in_path.stem}_pred_fail_env_orm1.jsonl")
    )

    want = _parse_statuses(args.pred_statuses)

    total = 0
    pred_hit = 0
    written = 0
    orm_missing_when_pred_hit = 0
    pred_status_missing = 0

    lines: list[str] = []

    for obj in iter_jsonl(in_path):
        total += 1
        ts = _get(obj, "evaluation", "prediction", "task_status")
        if ts is None:
            pred_status_missing += 1
            continue
        key = str(ts).strip().lower()
        if key not in want:
            continue
        pred_hit += 1

        orm_val = _get(obj, "evaluation", "orm_prediction", "is_task_success")
        orm_bin = normalize_is_task_success(orm_val)
        if orm_bin is None:
            orm_missing_when_pred_hit += 1
            continue
        if orm_bin != 1:
            continue

        lines.append(json.dumps(obj, ensure_ascii=False))
        written += 1

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"input: {in_path}")
    print(f"output: {out_path}")
    print(f"pred_statuses filter: {sorted(want)}")
    print(f"total_rows: {total}")
    print(f"prediction.task_status missing: {pred_status_missing}")
    print(f"rows matching pred_statuses: {pred_hit}")
    print(f"among pred_hit, orm missing/unparseable: {orm_missing_when_pred_hit}")
    print(f"written(pred in statuses AND orm==1): {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
