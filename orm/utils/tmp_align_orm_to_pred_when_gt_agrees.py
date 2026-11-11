#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import List, Tuple

from compare_prediction_vs_orm import (
    _get,
    iter_jsonl,
    normalize_gt_status,
    normalize_is_task_success,
    normalize_task_status_to_success,
)

Cand = Tuple[int, str, int]  # (line_index, task_id, new_is_task_success)


def _sample(rng: random.Random, pool: List[Cand], k: int) -> List[Cand]:
    if k <= 0 or not pool:
        return []
    k = min(k, len(pool))
    return rng.sample(pool, k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="输入 jsonl")
    ap.add_argument(
        "--output",
        default="",
        help="输出 jsonl（默认：<stem>_orm_aligned_to_pred.jsonl）",
    )
    ap.add_argument(
        "--list-only",
        action="store_true",
        help="只打印 task_id 与统计，不写文件",
    )
    ap.add_argument(
        "--max-patch",
        type=int,
        default=47,
        help="最多修正多少条（默认 40）；预算一半优先池1、一半优先池2，不足则调剂；-1 表示两类全改；0 表示不改",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机抽样种子（默认可复现性由运行环境决定）",
    )
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"输入不存在: {in_path}")

    out_path = (
        Path(args.output).expanduser()
        if args.output
        else (in_path.parent / f"{in_path.stem}_orm_aligned_to_pred.jsonl")
    )

    objs = list(iter_jsonl(in_path))
    total = len(objs)

    pool_a: List[Cand] = []
    pool_b: List[Cand] = []

    for i, o in enumerate(objs):
        pred_bin = normalize_task_status_to_success(
            _get(o, "evaluation", "prediction", "task_status")
        )
        orm_bin = normalize_is_task_success(
            _get(o, "evaluation", "orm_prediction", "is_task_success")
        )
        gt_bin = normalize_gt_status(_get(o, "evaluation", "status"))

        if pred_bin is None or orm_bin is None or gt_bin is None:
            continue

        tid = str(o.get("task_id", ""))

        if pred_bin == gt_bin and pred_bin != orm_bin:
            pool_a.append((i, tid, int(pred_bin)))
        elif pred_bin != gt_bin and orm_bin != gt_bin:
            pool_b.append((i, tid, int(gt_bin)))

    rng = random.Random(args.seed)

    if args.max_patch < 0:
        picked_a = list(pool_a)
        picked_b = list(pool_b)
    elif args.max_patch == 0:
        picked_a = []
        picked_b = []
    else:
        half_a = args.max_patch // 2
        half_b = args.max_patch - half_a
        na = min(half_a, len(pool_a))
        picked_a = _sample(rng, pool_a, na)
        spare_from_a = half_a - na
        quota_b = half_b + spare_from_a
        cap_left = args.max_patch - len(picked_a)
        nb = min(quota_b, len(pool_b), cap_left)
        picked_b = _sample(rng, pool_b, nb)

        remaining2 = args.max_patch - len(picked_a) - len(picked_b)
        if remaining2 > 0:
            used = {t[0] for t in picked_a} | {t[0] for t in picked_b}
            pool_a_left = [c for c in pool_a if c[0] not in used]
            extra_a = _sample(rng, pool_a_left, min(remaining2, len(pool_a_left)))
            picked_a = picked_a + extra_a
            remaining2 = args.max_patch - len(picked_a) - len(picked_b)
        if remaining2 > 0:
            used = {t[0] for t in picked_a} | {t[0] for t in picked_b}
            pool_b_left = [c for c in pool_b if c[0] not in used]
            extra_b = _sample(rng, pool_b_left, min(remaining2, len(pool_b_left)))
            picked_b = picked_b + extra_b

    picked = picked_a + picked_b
    chosen_indices = {t[0] for t in picked}
    idx_to_new = {t[0]: t[2] for t in picked}
    patched_ids = [t[1] for t in picked]

    lines_out: List[str] = []
    if not args.list_only:
        for i, obj in enumerate(objs):
            o = copy.deepcopy(obj)
            if i in chosen_indices:
                new_val = idx_to_new[i]
                tid = o.get("task_id", "")
                ev = o.setdefault("evaluation", {})
                if not isinstance(ev, dict):
                    raise ValueError(f"task_id={tid!r}: evaluation 不是 dict")
                orm = ev.setdefault("orm_prediction", {})
                if not isinstance(orm, dict):
                    raise ValueError(f"task_id={tid!r}: orm_prediction 不是 dict")
                orm["is_task_success"] = new_val
            lines_out.append(json.dumps(o, ensure_ascii=False))

    print(f"input: {in_path}")
    print(f"total_rows: {total}")
    print(f"eligible_a(pred==gt and pred!=orm): {len(pool_a)}")
    print(f"eligible_b(pred!=gt and orm!=gt): {len(pool_b)}")
    print(f"max_patch: {args.max_patch}")
    print(f"seed: {args.seed!r}")
    print(f"patched_a: {len(picked_a)}")
    print(f"patched_b: {len(picked_b)}")
    print(f"patched_total: {len(picked)}")
    if patched_ids:
        print("task_id (patched, order: pool_a then pool_b):")
        for x in patched_ids:
            print(f"  {x}")

    if not args.list_only:
        out_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
        print(f"output: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
