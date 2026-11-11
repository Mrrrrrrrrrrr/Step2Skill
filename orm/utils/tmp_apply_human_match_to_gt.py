#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _normalize_match(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        iv = int(v)
        return iv if iv in (0, 1) else None
    if isinstance(v, str):
        t = v.strip()
        if not t or t.lower() in {"nan", "none", "null"}:
            return None
        if t in {"0", "1"}:
            return int(t)
    return None


def _candidate_task_ids(traj_id: str) -> List[str]:
    cands = [traj_id]
    if traj_id.endswith("_main"):
        cands.append(traj_id[: -len("_main")])
    else:
        cands.append(f"{traj_id}_main")
    return cands


def main() -> int:
    parser = argparse.ArgumentParser(
        description="临时脚本：将 Excel 的 Human_vs_Auto_Match(0/1) 覆盖到 JSONL 的 evaluation.status"
    )
    parser.add_argument("--excel", required=True, help="Excel 路径")
    parser.add_argument("--jsonl", required=True, help="目标 JSONL 路径")
    parser.add_argument(
        "--output",
        default="",
        help="输出 JSONL 路径（默认：<jsonl_stem>_gt_overwritten.jsonl）",
    )
    parser.add_argument(
        "--sheet",
        default="",
        help="可选 sheet 名；不传则读取第一个 sheet",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="直接覆盖原 JSONL（慎用）",
    )
    args = parser.parse_args()

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("需要 pandas 才能读取 xlsx，请先安装 pandas/openpyxl") from exc

    excel_path = Path(args.excel).expanduser()
    jsonl_path = Path(args.jsonl).expanduser()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel 不存在: {excel_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL 不存在: {jsonl_path}")

    if args.inplace and args.output:
        raise ValueError("--inplace 与 --output 不能同时使用")

    if args.inplace:
        output_path = jsonl_path
    else:
        output_path = (
            Path(args.output).expanduser()
            if args.output
            else (jsonl_path.parent / f"{jsonl_path.stem}_gt_overwritten.jsonl")
        )

    df = pd.read_excel(excel_path, sheet_name=args.sheet or 0)
    required_cols = ["Trajectory ID", "Human_vs_Auto_Match"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Excel 缺少列: {c}，当前列: {list(df.columns)}")

    # trajectory_id -> 0/1
    mapping: Dict[str, int] = {}
    skipped_invalid = 0
    for _, row in df.iterrows():
        traj_raw = row.get("Trajectory ID")
        match_raw = row.get("Human_vs_Auto_Match")
        if traj_raw is None:
            continue
        traj_id = str(traj_raw).strip()
        if not traj_id:
            continue
        m = _normalize_match(match_raw)
        if m is None:
            skipped_invalid += 1
            continue
        mapping[traj_id] = m

    lines_out: List[str] = []
    total = 0
    updated = 0
    no_match = 0
    matched_by_main_strip = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            total += 1
            obj = json.loads(s)
            task_id = str(obj.get("task_id", "")).strip()
            if not task_id:
                no_match += 1
                lines_out.append(json.dumps(obj, ensure_ascii=False))
                continue

            gt_val: Optional[int] = None
            matched = False
            for traj_id, m in mapping.items():
                cands = _candidate_task_ids(traj_id)
                if task_id in cands:
                    gt_val = m
                    matched = True
                    if task_id != traj_id and task_id in cands[1:]:
                        matched_by_main_strip += 1
                    break

            if not matched:
                # 反向兜底：只做 O(1) 查找
                if task_id in mapping:
                    gt_val = mapping[task_id]
                    matched = True
                elif f"{task_id}_main" in mapping:
                    gt_val = mapping[f"{task_id}_main"]
                    matched = True
                    matched_by_main_strip += 1

            if matched and gt_val is not None:
                ev = obj.get("evaluation")
                if not isinstance(ev, dict):
                    ev = {}
                    obj["evaluation"] = ev
                ev["status"] = int(gt_val)
                updated += 1
            else:
                no_match += 1

            lines_out.append(json.dumps(obj, ensure_ascii=False))

    with output_path.open("w", encoding="utf-8") as fw:
        fw.write("\n".join(lines_out) + ("\n" if lines_out else ""))

    print(f"excel: {excel_path}")
    print(f"jsonl: {jsonl_path}")
    print(f"output: {output_path}")
    print(f"excel_valid_mapping_rows: {len(mapping)}")
    print(f"excel_invalid_match_rows_skipped: {skipped_invalid}")
    print(f"jsonl_total_rows: {total}")
    print(f"jsonl_updated_rows: {updated}")
    print(f"jsonl_no_match_rows: {no_match}")
    print(f"matched_by_taskid_vs_trajectoryid_suffix: {matched_by_main_strip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

