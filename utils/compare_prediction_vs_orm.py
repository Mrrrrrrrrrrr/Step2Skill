#!/usr/bin/env python3
"""
对比"阿里预测(evaluation.prediction.task_status)"与"自研ORM(evaluation.orm_prediction.is_task_success)"的一致性。

支持两种输入方式：
  1) 单文件模式：--input 一个文件，文件中每行同时包含 evaluation.prediction 与 evaluation.orm_prediction；
  2) 双文件模式：--input 阿里预测 + --orm-input 自研ORM 输出（按 task_id 对齐）。

输出：
  - 3 类标签（success / failed / env_error）下的 3x3 混淆矩阵 + 一致率；
  - 去除两侧任一为 env_error 的并集后，2x2 混淆矩阵（success / failed） + 一致率；
  - 兼容老版本的二分类(perfect/fair_enough->1, 其它含 failed/env_error->0) 一致率 / GT 对齐统计；
  - 可选导出不一致样本 jsonl。

标签归一化规则：
  - 阿里预测 evaluation.prediction.task_status：
      perfect / fair_enough -> success
      failed                -> failed
      env_error             -> env_error
      其它（含 unknown / llm_error / 缺失） -> None（即在 3 类口径下视为缺失）
  - 自研 ORM evaluation.orm_prediction.is_task_success：
       1 -> success
       0 -> failed
      -1 -> env_error
      其它/缺失 -> None
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

# ----------- IO -----------

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行 JSON 解析失败: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path} 第 {line_no} 行不是 JSON 对象")
            yield obj


def _get(d: Dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


# ----------- 归一化（二分类，保留老逻辑） -----------

def normalize_task_status_to_success(task_status: Any) -> Optional[int]:
    """老二分类口径：perfect / fair_enough => 1，其它（含 failed / env_error）=> 0；缺失为 None。"""
    if task_status is None:
        return None
    if not isinstance(task_status, str):
        task_status = str(task_status)
    s = task_status.strip().lower()
    return 1 if s in {"perfect", "fair_enough"} else 0


def normalize_is_task_success(v: Any) -> Optional[int]:
    """老二分类口径：1 -> 1，0 -> 0，其它（含 -1）-> 0；缺失为 None。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if int(v) == 1 else 0
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"1", "true", "yes", "y"}:
            return 1
        if t in {"0", "false", "no", "n", "-1"}:
            return 0
    return None


def normalize_gt_status(v: Any) -> Optional[int]:
    """人工 gt：evaluation.status；仅 0/1 视为有效。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv if iv in (0, 1) else None
    if isinstance(v, str):
        t = v.strip()
        if t in {"0", "1"}:
            return int(t)
    return None


# ----------- 归一化（三分类） -----------

LABEL_SUCCESS = "success"
LABEL_FAILED = "failed"
LABEL_ENV_ERROR = "env_error"
TRI_LABELS = (LABEL_SUCCESS, LABEL_FAILED, LABEL_ENV_ERROR)


def normalize_pred_tri(task_status: Any) -> Optional[str]:
    """阿里预测三分类。"""
    if task_status is None:
        return None
    s = str(task_status).strip().lower()
    if s in {"perfect", "fair_enough"}:
        return LABEL_SUCCESS
    if s == "failed":
        return LABEL_FAILED
    if s in {"env_error", "environment_error"}:
        return LABEL_ENV_ERROR
    return None  # llm_error / unknown / missing


def normalize_orm_tri(v: Any) -> Optional[str]:
    """自研 ORM 三分类。"""
    if v is None:
        return None
    if isinstance(v, bool):
        return LABEL_SUCCESS if v else LABEL_FAILED
    if isinstance(v, (int, float)):
        iv = int(v)
        if iv == 1:
            return LABEL_SUCCESS
        if iv == 0:
            return LABEL_FAILED
        if iv == -1:
            return LABEL_ENV_ERROR
        return None
    if isinstance(v, str):
        t = v.strip().lower()
        if t in {"1", "true", "yes", "y", "success"}:
            return LABEL_SUCCESS
        if t in {"0", "false", "no", "n", "failed", "fail"}:
            return LABEL_FAILED
        if t in {"-1", "env_error", "environment_error"}:
            return LABEL_ENV_ERROR
    return None


# ----------- 渲染矩阵 -----------

def _fmt_int(n: int, w: int) -> str:
    return f"{n:>{w}d}"


def render_matrix(mat: Dict[Tuple[str, str], int], row_labels, col_labels, *, title: str) -> str:
    rows = list(row_labels)
    cols = list(col_labels)
    col_w = max(10, max(len(c) for c in cols))
    row_hdr_w = max(len("pred\\orm"), max(len(r) for r in rows))

    total = sum(mat.values())
    diag = sum(mat.get((x, x), 0) for x in rows if x in cols)
    rate = (diag / total) if total else 0.0

    lines = []
    lines.append(f"=== {title} ===")
    lines.append(f"total={total}  diag(agree)={diag}  agreement_rate={rate:.6f}")
    # 表头
    header = (
        " " * row_hdr_w
        + " | "
        + " | ".join(f"{c:>{col_w}}" for c in cols)
        + " | "
        + f"{'row_sum':>{col_w}}"
    )
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)
    col_sums = {c: 0 for c in cols}
    for r in rows:
        row_vals = [mat.get((r, c), 0) for c in cols]
        row_sum = sum(row_vals)
        for c, v in zip(cols, row_vals):
            col_sums[c] += v
        lines.append(
            f"{r:<{row_hdr_w}} | "
            + " | ".join(_fmt_int(v, col_w) for v in row_vals)
            + " | "
            + _fmt_int(row_sum, col_w)
        )
    lines.append(sep)
    lines.append(
        f"{'col_sum':<{row_hdr_w}} | "
        + " | ".join(_fmt_int(col_sums[c], col_w) for c in cols)
        + " | "
        + _fmt_int(total, col_w)
    )
    return "\n".join(lines)


# ----------- 主逻辑 -----------

def build_joined_records(
    in_path: Path, orm_path: Optional[Path]
) -> Iterable[Dict[str, Any]]:
    """
    返回统一结构的记录迭代器：
        {"task_id": ..., "pred_task_status": ..., "orm_is_task_success": ..., "gt": ..., "_raw": <原始 obj>}
    单文件模式：直接从 in_path 抽取两个字段；
    双文件模式：以 task_id 作为 key 做 inner-join（缺一侧也保留，对应字段为 None）。
    """
    pred_records = list(iter_jsonl(in_path))

    if orm_path is None:
        for obj in pred_records:
            yield {
                "task_id": _get(obj, "task_id"),
                "pred_task_status": _get(obj, "evaluation", "prediction", "task_status"),
                "orm_is_task_success": _get(
                    obj, "evaluation", "orm_prediction", "is_task_success"
                ),
                "gt": _get(obj, "evaluation", "status"),
                "_raw": obj,
            }
        return

    orm_index: Dict[Any, Dict[str, Any]] = {}
    for obj in iter_jsonl(orm_path):
        tid = _get(obj, "task_id")
        if tid is None:
            continue
        orm_index[tid] = obj

    for obj in pred_records:
        tid = _get(obj, "task_id")
        orm_obj = orm_index.get(tid)
        yield {
            "task_id": tid,
            "pred_task_status": _get(obj, "evaluation", "prediction", "task_status"),
            "orm_is_task_success": _get(
                orm_obj or {}, "evaluation", "orm_prediction", "is_task_success"
            ),
            "gt": _get(obj, "evaluation", "status"),
            "_raw": obj,
            "_orm_raw": orm_obj,
        }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "统计阿里预测(evaluation.prediction.task_status) 与 "
            "自研ORM(evaluation.orm_prediction.is_task_success) 的一致性，"
            "支持 3x3（含 env_error） 与 2x2（去除 env_error 并集）两种口径。"
        )
    )
    ap.add_argument("--input", required=True, help="阿里预测 jsonl（或同时含两侧字段的合并文件）")
    ap.add_argument(
        "--orm-input",
        default="",
        help="自研 ORM 预测 jsonl。给出时启用双文件模式（按 task_id 对齐）；缺省走单文件模式",
    )
    ap.add_argument(
        "--output",
        default="",
        help="输出不一致样本 jsonl（默认：与 --input 同目录，文件名加 _mismatch.jsonl）",
    )
    ap.add_argument(
        "--mismatch-mode",
        choices=("tri", "binary", "none"),
        default="tri",
        help="按何种口径判定 mismatch 并导出：tri=三分类；binary=去 env_error 后的二分类；none=不导出",
    )
    ap.add_argument(
        "--include-missing",
        action="store_true",
        help="是否把任一侧缺失也视为'不一致'并导出（默认只导出两侧都有有效标签且不一致）",
    )
    args = ap.parse_args()

    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    orm_path: Optional[Path] = None
    if args.orm_input:
        orm_path = Path(args.orm_input).expanduser()
        if not orm_path.exists():
            raise FileNotFoundError(f"ORM 输入文件不存在: {orm_path}")

    out_path = (
        Path(args.output).expanduser()
        if args.output
        else (in_path.parent / f"{in_path.stem}_mismatch.jsonl")
    )

    # ===== 统计容器 =====
    total = 0
    pred_missing_bin = 0
    orm_missing_bin = 0
    both_present_bin = 0
    agree_bin = 0
    disagree_bin = 0
    cm_bin: Dict[Tuple[int, int], int] = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    # GT 相关
    gt_valid_total = 0
    pred_valid_for_gt = 0
    pred_correct_on_gt = 0
    orm_valid_for_gt = 0
    orm_correct_on_gt = 0
    mismatch_total_for_gt = 0
    mismatch_gt_valid = 0
    mismatch_gt_match_pred = 0
    mismatch_gt_match_orm = 0
    mismatch_gt_invalid_or_missing = 0

    # 三分类
    pred_missing_tri = 0
    orm_missing_tri = 0
    both_present_tri = 0
    cm_tri: Dict[Tuple[str, str], int] = {(r, c): 0 for r in TRI_LABELS for c in TRI_LABELS}

    # 去 env_error 后二分类（success/failed）
    cm_no_env: Dict[Tuple[str, str], int] = {
        (r, c): 0 for r in (LABEL_SUCCESS, LABEL_FAILED) for c in (LABEL_SUCCESS, LABEL_FAILED)
    }
    dropped_pred_env = 0
    dropped_orm_env = 0
    dropped_both_env = 0
    kept_after_drop_env = 0

    written = 0
    fw = out_path.open("w", encoding="utf-8") if args.mismatch_mode != "none" else None

    try:
        for rec in build_joined_records(in_path, orm_path):
            total += 1
            task_id = rec["task_id"]
            pred_status = rec["pred_task_status"]
            orm_val = rec["orm_is_task_success"]
            gt_raw = rec["gt"]

            # ---- 二分类（兼容老逻辑） ----
            pred_bin = normalize_task_status_to_success(pred_status)
            orm_bin = normalize_is_task_success(orm_val)
            gt_bin = normalize_gt_status(gt_raw)

            if gt_bin is not None:
                gt_valid_total += 1
                if pred_bin is not None:
                    pred_valid_for_gt += 1
                    if pred_bin == gt_bin:
                        pred_correct_on_gt += 1
                if orm_bin is not None:
                    orm_valid_for_gt += 1
                    if orm_bin == gt_bin:
                        orm_correct_on_gt += 1

            if pred_bin is None:
                pred_missing_bin += 1
            if orm_bin is None:
                orm_missing_bin += 1

            is_mismatch_bin = False
            if pred_bin is not None and orm_bin is not None:
                both_present_bin += 1
                cm_bin[(pred_bin, orm_bin)] += 1
                if pred_bin == orm_bin:
                    agree_bin += 1
                else:
                    disagree_bin += 1
                    is_mismatch_bin = True
                    mismatch_total_for_gt += 1
                    if gt_bin is None:
                        mismatch_gt_invalid_or_missing += 1
                    else:
                        mismatch_gt_valid += 1
                        if gt_bin == pred_bin:
                            mismatch_gt_match_pred += 1
                        elif gt_bin == orm_bin:
                            mismatch_gt_match_orm += 1

            # ---- 三分类 ----
            pred_tri = normalize_pred_tri(pred_status)
            orm_tri = normalize_orm_tri(orm_val)
            if pred_tri is None:
                pred_missing_tri += 1
            if orm_tri is None:
                orm_missing_tri += 1
            is_mismatch_tri = False
            if pred_tri is not None and orm_tri is not None:
                both_present_tri += 1
                cm_tri[(pred_tri, orm_tri)] += 1
                if pred_tri != orm_tri:
                    is_mismatch_tri = True

            # ---- 去 env_error 后的二分类 ----
            if pred_tri is not None and orm_tri is not None:
                pred_is_env = pred_tri == LABEL_ENV_ERROR
                orm_is_env = orm_tri == LABEL_ENV_ERROR
                if pred_is_env and orm_is_env:
                    dropped_both_env += 1
                elif pred_is_env:
                    dropped_pred_env += 1
                elif orm_is_env:
                    dropped_orm_env += 1
                else:
                    kept_after_drop_env += 1
                    cm_no_env[(pred_tri, orm_tri)] += 1

            # ---- 导出 mismatch ----
            if fw is not None:
                if args.mismatch_mode == "tri":
                    if pred_tri is None or orm_tri is None:
                        export_this = args.include_missing
                    else:
                        export_this = pred_tri != orm_tri
                else:  # binary
                    if pred_bin is None or orm_bin is None:
                        export_this = args.include_missing
                    else:
                        export_this = is_mismatch_bin

                if export_this:
                    obj_out = dict(rec["_raw"]) if rec.get("_raw") else {}
                    obj_out["_compare"] = {
                        "task_id": task_id,
                        "prediction.task_status": pred_status,
                        "prediction_tri": pred_tri,
                        "prediction_is_success_binary": pred_bin,
                        "orm_prediction.is_task_success": orm_val,
                        "orm_tri": orm_tri,
                        "orm_is_success_binary": orm_bin,
                    }
                    fw.write(json.dumps(obj_out, ensure_ascii=False) + "\n")
                    written += 1
    finally:
        if fw is not None:
            fw.close()

    # ===== 输出报表 =====
    print(f"input(pred):     {in_path}")
    if orm_path is not None:
        print(f"input(orm):      {orm_path}")
    else:
        print("input(orm):      <single-file mode>")
    if args.mismatch_mode != "none":
        print(f"output(mismatch): {out_path}  (mode={args.mismatch_mode}, written={written})")
    else:
        print("output(mismatch): <skipped>")
    print(f"total_rows: {total}")
    print()

    # ---- 三分类报表 ----
    print("[Tri-class label distribution]")
    print(f"  pred(task_status -> tri): missing={pred_missing_tri} / {total}")
    print(f"  orm (is_task_success -> tri): missing={orm_missing_tri} / {total}")
    print(f"  both_present(tri): {both_present_tri}")
    print()
    print(render_matrix(cm_tri, TRI_LABELS, TRI_LABELS, title="3x3 confusion (pred row × orm col)"))
    print()

    # ---- 2x2（去 env_error 并集）----
    print("[2x2 after dropping env_error union]")
    print(f"  dropped_both_env_error : {dropped_both_env}")
    print(f"  dropped_pred_env_only  : {dropped_pred_env}")
    print(f"  dropped_orm_env_only   : {dropped_orm_env}")
    print(
        f"  total_dropped(union)   : {dropped_both_env + dropped_pred_env + dropped_orm_env}"
    )
    print(f"  kept_after_drop_env    : {kept_after_drop_env}")
    print()
    print(
        render_matrix(
            cm_no_env,
            (LABEL_SUCCESS, LABEL_FAILED),
            (LABEL_SUCCESS, LABEL_FAILED),
            title="2x2 confusion (success/failed only)",
        )
    )
    print()

    # ---- 老二分类 + GT ----
    denom_bin = both_present_bin if both_present_bin else 0
    acc_bin = (agree_bin / denom_bin) if denom_bin else 0.0
    pred_acc_on_gt = (pred_correct_on_gt / pred_valid_for_gt) if pred_valid_for_gt else 0.0
    orm_acc_on_gt = (orm_correct_on_gt / orm_valid_for_gt) if orm_valid_for_gt else 0.0
    print("[Legacy binary (perfect/fair_enough -> 1, else -> 0)]")
    print(f"  prediction_missing: {pred_missing_bin}")
    print(f"  orm_missing:        {orm_missing_bin}")
    print(f"  both_present:       {both_present_bin}")
    print(f"  agree / disagree:   {agree_bin} / {disagree_bin}")
    print(f"  agreement_rate:     {acc_bin:.6f}")
    print(f"  confusion (pred,orm): (0,0)={cm_bin[(0,0)]}  (0,1)={cm_bin[(0,1)]}  "
          f"(1,0)={cm_bin[(1,0)]}  (1,1)={cm_bin[(1,1)]}")
    print(f"  gt_valid_total(0/1): {gt_valid_total}")
    print(
        f"  pred acc on gt: {pred_correct_on_gt}/{pred_valid_for_gt} = {pred_acc_on_gt:.6f}"
    )
    print(
        f"  orm  acc on gt: {orm_correct_on_gt}/{orm_valid_for_gt} = {orm_acc_on_gt:.6f}"
    )
    print("  gt_alignment_on_disagree_rows:")
    print(f"    disagree_rows_total: {mismatch_total_for_gt}")
    print(f"    gt_valid(0/1):       {mismatch_gt_valid}")
    print(f"    gt_invalid_or_missing: {mismatch_gt_invalid_or_missing}")
    print(f"    gt_match_prediction:   {mismatch_gt_match_pred}")
    print(f"    gt_match_orm_prediction: {mismatch_gt_match_orm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())







