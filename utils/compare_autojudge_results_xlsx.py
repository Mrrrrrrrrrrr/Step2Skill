#!/usr/bin/env python3
"""
比较两份导出表（.xlsx）中判定列的二值一致率（默认跑两组对比）。

读取方式与 utils/eval_human_autojudge_vs_orm copy.py 一致：
- Trajectory ID 列对齐轨迹；去掉后缀 _main 后作为键
- 标签：perfect / fair_enough → 1；failed → 0；env_error → 不参与
- 可选：Error Cause 含「环境错误」的行也排除（与 eval 脚本默认一致）

默认两次对比：
1. 两表均为 AutoJudge Result
2. A 表 doubao-seed Result vs B 表 doubao-seed-1-8 Result
   （表头若为换行，如 doubao-seed + 换行 + Result，会与带空格的写法自动对齐）
   豆包对比后额外输出四分类 4×4 联表（perfect / fair_enough / failed / env_error）及完全一致率
   另：默认 --a=复现 results.xlsx、--b=阿里 results (1).xlsx 时，在豆包段输出 5 张 4×4：
   3a 复现seed×阿里总(AutoJudge)；3b 人评×阿里seed；3c 人评×复现seed；
   3d 人评×阿里总；3e 人评×复现总；对比 4 人评×阿里 Claude Opus Result。
   每张 4×4 后附一行：仅用三分类 3×3（无 env_error）的二值准确率。可用 --no-gt-fourway 关闭。

典型用法：

  python utils/compare_autojudge_results_xlsx.py \\
    --a "data/seed_free/results.xlsx" \\
    --b "data/seed_free/results (1).xlsx"

  # 只比 AutoJudge 或只比 doubao
  python utils/compare_autojudge_results_xlsx.py --compare autojudge
  python utils/compare_autojudge_results_xlsx.py --compare doubao
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 细粒度四分类（行=A，列=B）；与表内取值对应，failed 兼容 fail
FOUR_WAY_LABELS: Tuple[str, ...] = ("perfect", "fair_enough", "failed", "env_error")
# 二值准确率子表：不含 env_error，对应 4×4 中左上 3×3（按行列顺序即前三类）
THREE_WAY_LABELS: Tuple[str, ...] = ("perfect", "fair_enough", "failed")

DEFAULT_AUTOJUDGE_COLUMN = "AutoJudge Result"
# 与表内「doubao-seed」+ 换行 +「Result」等价（见 resolve_header_column）
DEFAULT_DOUBAO_COLUMN_A = "doubao-seed Result"
DEFAULT_DOUBAO_COLUMN_B = "doubao-seed-1-8 Result"
DEFAULT_GT_COLUMN = "human_Judge"
# 两表「多模型汇总」列名（与当前导出一致：AutoJudge Result）
DEFAULT_OVERALL_COLUMN = "AutoJudge Result"
# 阿里表内 Claude Opus 判定列（表头多为 aws... + 换行 + Result）
DEFAULT_ALI_OPUS_RESULT_COLUMN = "aws.claude-opus-4-5-20251101 Result"


def normalize_trajectory_key(trajectory_id: str) -> str:
    t = (trajectory_id or "").strip()
    if t.endswith("_main"):
        t = t[: -len("_main")]
    return t


def classify_human_row(
    human_raw: str,
    error_cause: str,
    *,
    also_skip_by_error_cause: bool,
) -> Tuple[str, Optional[int]]:
    h = (human_raw or "").strip().lower()
    ec = (error_cause or "").strip()
    if h == "env_error":
        return "env", None
    if also_skip_by_error_cause and "环境错误" in ec:
        return "env", None
    if not h:
        return "bad", None
    if h in ("perfect", "fair_enough"):
        return "ok", 1
    if h in ("failed", "fail"):
        return "ok", 0
    return "bad", None


def parse_four_way_label(
    human_raw: str,
    error_cause: str,
    *,
    also_skip_by_error_cause: bool,
) -> Optional[str]:
    """
    映射到 FOUR_WAY_LABELS 之一；无法识别返回 None。
    环境规则与 classify_human_row 一致（env_error 或 Error Cause 含「环境错误」→ env_error）。
    """
    h = (human_raw or "").strip().lower()
    ec = (error_cause or "").strip()
    if h == "env_error":
        return "env_error"
    if also_skip_by_error_cause and "环境错误" in ec:
        return "env_error"
    if not h:
        return None
    if h == "perfect":
        return "perfect"
    if h == "fair_enough":
        return "fair_enough"
    if h in ("failed", "fail"):
        return "failed"
    return None


def three_way_to_binary(lbl: str) -> int:
    """perfect / fair_enough -> 1，failed -> 0。仅用于已落入 THREE_WAY 的标签。"""
    if lbl in ("perfect", "fair_enough"):
        return 1
    return 0


def binary_accuracy_from_3x3_submatrix(
    mat: Dict[Tuple[str, str], int],
) -> Tuple[int, int, float]:
    """
    仅统计 4×4 中 env_error 以外的 3×3 子块（行/列 ∈ perfect, fair_enough, failed）。
    行侧视为参考 gold，列侧视为预测 pred；二值一致条数 / 该子块总条数。
    返回 (correct, total, accuracy_percent)。
    """
    total = 0
    correct = 0
    for lr in THREE_WAY_LABELS:
        for lc in THREE_WAY_LABELS:
            n = mat.get((lr, lc), 0)
            total += n
            if three_way_to_binary(lr) == three_way_to_binary(lc):
                correct += n
    acc_pct = (correct / total * 100) if total else 0.0
    return correct, total, acc_pct


def _cell_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v).strip()


def _normalize_header_label(s: str) -> str:
    """与 Excel 中换行表头对齐：任意空白（含 \\n）折叠为单个空格。"""
    return re.sub(r"\s+", " ", (s or "").strip())


def resolve_header_column(headers: List[str], wanted: str, role: str) -> str:
    """
    将期望列名解析为工作表中的实际表头字符串。
    先精确匹配，再按归一化标签唯一匹配（兼容 doubao-seed\\nResult 与 doubao-seed Result）。
    """
    if wanted in headers:
        return wanted
    wn = _normalize_header_label(wanted)
    matches = [h for h in headers if _normalize_header_label(h) == wn]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"列 {wanted!r}（{role}）归一化后匹配到多列: {matches!r}，现有: {headers!r}"
        )
    raise KeyError(f"表缺少列 {wanted!r}（{role}），现有: {headers!r}")


def load_xlsx_rows(
    xlsx_path: Path,
    *,
    id_column: str,
    human_column: str,
    error_cause_column: str,
    sheet: Any,
) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("读取 .xlsx 需要安装 openpyxl：pip install openpyxl") from exc

    wb = load_workbook(xlsx_path, read_only=False, data_only=True)
    try:
        if isinstance(sheet, int):
            if sheet < 0 or sheet >= len(wb.sheetnames):
                raise ValueError(f"工作表索引 {sheet} 无效，共 {len(wb.sheetnames)} 个: {wb.sheetnames}")
            ws = wb.worksheets[sheet]
        else:
            name = str(sheet).strip()
            if name not in wb.sheetnames:
                raise ValueError(f"无工作表 {name!r}，可选: {wb.sheetnames}")
            ws = wb[name]

        max_col = int(ws.max_column or 0)
        max_row = int(ws.max_row or 0)
        if max_col < 1:
            raise ValueError("xlsx 无列")
        headers = [_cell_str(ws.cell(1, c).value) for c in range(1, max_col + 1)]
        id_resolved = resolve_header_column(headers, id_column, "轨迹 ID")
        human_resolved = resolve_header_column(headers, human_column, "判定列")
        idx_id = headers.index(id_resolved)
        idx_h = headers.index(human_resolved)
        if error_cause_column:
            ec_resolved = resolve_header_column(headers, error_cause_column, "Error Cause")
            idx_ec = headers.index(ec_resolved)
        else:
            idx_ec = None

        rows: List[Tuple[str, str, str, str]] = []

        def cell_at(row_1based: int, idx0: int) -> str:
            return _cell_str(ws.cell(row_1based, idx0 + 1).value)

        for r in range(2, max_row + 1):
            tid = cell_at(r, idx_id)
            if not tid:
                continue
            human = cell_at(r, idx_h)
            ec = cell_at(r, idx_ec) if idx_ec is not None else ""
            rows.append((normalize_trajectory_key(tid), tid, human, ec))
        return headers, rows
    finally:
        wb.close()


def rows_to_map(
    rows: List[Tuple[str, str, str, str]],
) -> Tuple[Dict[str, Tuple[str, str, str]], int]:
    """norm_id -> (raw_tid, human, error_cause)；重复键后者覆盖，返回重复次数。"""
    m: Dict[str, Tuple[str, str, str]] = {}
    dup = 0
    for norm_id, raw_tid, human, ec in rows:
        if norm_id in m:
            dup += 1
        m[norm_id] = (raw_tid, human, ec)
    return m, dup


def run(
    path_a: Path,
    path_b: Path,
    *,
    human_column_a: str,
    human_column_b: str,
    id_column: str,
    error_cause_column: str,
    skip_error_cause_env: bool,
    sheet: Any,
    section_title: str = "",
) -> None:
    _, rows_a = load_xlsx_rows(
        path_a,
        id_column=id_column,
        human_column=human_column_a,
        error_cause_column=error_cause_column,
        sheet=sheet,
    )
    _, rows_b = load_xlsx_rows(
        path_b,
        id_column=id_column,
        human_column=human_column_b,
        error_cause_column=error_cause_column,
        sheet=sheet,
    )

    map_a, dup_a = rows_to_map(rows_a)
    map_b, dup_b = rows_to_map(rows_b)
    keys_a = set(map_a.keys())
    keys_b = set(map_b.keys())
    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    compared = 0
    agree = 0
    skip_env = 0
    skip_bad = 0
    mismatches: List[Tuple[str, int, int, str, str]] = []

    for k in sorted(common):
        raw_a, h_a, ec_a = map_a[k]
        raw_b, h_b, ec_b = map_b[k]
        kind_a, y_a = classify_human_row(
            h_a, ec_a, also_skip_by_error_cause=skip_error_cause_env
        )
        kind_b, y_b = classify_human_row(
            h_b, ec_b, also_skip_by_error_cause=skip_error_cause_env
        )
        if kind_a == "env" or kind_b == "env":
            skip_env += 1
            continue
        if kind_a == "bad" or kind_b == "bad" or y_a is None or y_b is None:
            skip_bad += 1
            continue
        compared += 1
        if y_a == y_b:
            agree += 1
        else:
            mismatches.append((raw_a, y_a, y_b, h_a, h_b))

    rate = (agree / compared * 100) if compared else 0.0

    if section_title:
        print()
        print("=" * 72)
        print(section_title)
        print("=" * 72)
    print(f"A: {path_a}")
    print(f"B: {path_b}")
    print(f"列 A: {human_column_a!r}，列 B: {human_column_b!r}，轨迹: {id_column!r}")
    print(f"A 行数(有 Trajectory ID): {len(rows_a)}，重复 norm_id 覆盖次数: {dup_a}")
    print(f"B 行数(有 Trajectory ID): {len(rows_b)}，重复 norm_id 覆盖次数: {dup_b}")
    print(f"仅 A 有键: {len(only_a)}，仅 B 有键: {len(only_b)}，交集键: {len(common)}")
    print(f"交集内排除(任一侧 env_error / Error Cause 环境错误): {skip_env}")
    print(f"交集内排除(任一侧标签空或无法映射): {skip_bad}")
    print(f"双方均可比(0/1)条数: {compared}")
    print(f"一致率: {rate:.2f}% ({agree}/{compared})")

    if mismatches:
        show = min(30, len(mismatches))
        print(f"\n不一致样例（前 {show} 条，Trajectory ID_A, y_A, y_B, raw_A, raw_B）:")
        for item in mismatches[:show]:
            tid, ya, yb, ra, rb = item
            print(f"  {tid}\t y_A={ya}\t y_B={yb}\t {ra!r}\t {rb!r}")
        if len(mismatches) > show:
            print(f"  ... 共 {len(mismatches)} 条不一致")


def run_fourway_cross_matrix(
    path_row: Path,
    column_row: str,
    path_col: Path,
    column_col: str,
    *,
    id_column: str,
    error_cause_column: str,
    skip_error_cause_env: bool,
    sheet: Any,
    section_title: str = "",
    row_axis_name: str = "行",
    col_axis_name: str = "列",
    append_binary_accuracy_3x3: bool = False,
) -> None:
    """
    任意两列（可跨文件）：四分类 4×4 联表；行=path_row/column_row，列=path_col/column_col。
    各自使用该文件同行的 Error Cause 做环境错误判定。
    append_binary_accuracy_3x3：在 4×4 后追加一行，仅用 3×3（无 env_error）算二值准确率，
    行侧=参考 gold，列侧=预测；perfect/fair_enough→1，failed→0。
    """
    _, rows_r = load_xlsx_rows(
        path_row,
        id_column=id_column,
        human_column=column_row,
        error_cause_column=error_cause_column,
        sheet=sheet,
    )
    _, rows_c = load_xlsx_rows(
        path_col,
        id_column=id_column,
        human_column=column_col,
        error_cause_column=error_cause_column,
        sheet=sheet,
    )
    map_r, _ = rows_to_map(rows_r)
    map_c, _ = rows_to_map(rows_c)
    common = set(map_r.keys()) & set(map_c.keys())

    mat: Dict[Tuple[str, str], int] = {}
    skip_unmapped = 0
    for k in common:
        _raw_r, h_r, ec_r = map_r[k]
        _raw_c, h_c, ec_c = map_c[k]
        lr = parse_four_way_label(
            h_r, ec_r, also_skip_by_error_cause=skip_error_cause_env
        )
        lc = parse_four_way_label(
            h_c, ec_c, also_skip_by_error_cause=skip_error_cause_env
        )
        if lr is None or lc is None:
            skip_unmapped += 1
            continue
        key = (lr, lc)
        mat[key] = mat.get(key, 0) + 1

    total = sum(mat.values())
    diag = sum(mat.get((lbl, lbl), 0) for lbl in FOUR_WAY_LABELS)
    rate = (diag / total * 100) if total else 0.0

    if section_title:
        print()
        print("=" * 72)
        print(section_title)
        print("=" * 72)
    print(f"行侧文件: {path_row}，列: {column_row!r}")
    print(f"列侧文件: {path_col}，列: {column_col!r}")
    print(f"轨迹键: {id_column!r}")
    print(f"交集键(双方均有 Trajectory ID): {len(common)}")
    print(f"任一侧无法归入四分类(空或未知取值): {skip_unmapped}")
    print(f"计入 4×4 联表的样本数: {total}")
    print(f"四分类完全一致率: {rate:.2f}% ({diag}/{total})")
    print()
    print(f"4×4 联表（行={row_axis_name}，列={col_axis_name}，单元格为条数）:")
    cw = 12
    pad = max(16, len(row_axis_name) + 2)
    header = " " * pad + "".join(f"{lb:>{cw}}" for lb in FOUR_WAY_LABELS)
    print(header)
    for ra in FOUR_WAY_LABELS:
        row_cells = "".join(f"{mat.get((ra, rb), 0):>{cw}}" for rb in FOUR_WAY_LABELS)
        print(f"{row_axis_name}={ra:14}{row_cells}")

    if append_binary_accuracy_3x3:
        c_ok, n_tot, acc_pct = binary_accuracy_from_3x3_submatrix(mat)
        print()
        print(
            "二值准确率（仅 3×3：双方均为 perfect/fair_enough/failed，不含 env_error；"
            f"行={row_axis_name}=参考 gold，列={col_axis_name}=pred；"
            "1=perfect|fair_enough，0=failed）: "
            f"{acc_pct:.2f}% ({c_ok}/{n_tot})"
        )


def run_doubao_fourway_matrix(
    path_a: Path,
    path_b: Path,
    *,
    human_column_a: str,
    human_column_b: str,
    id_column: str,
    error_cause_column: str,
    skip_error_cause_env: bool,
    sheet: Any,
    section_title: str = "",
) -> None:
    """A/B 两表豆包列：四分类 4×4（行=A 列=B）。"""
    run_fourway_cross_matrix(
        path_a,
        human_column_a,
        path_b,
        human_column_b,
        id_column=id_column,
        error_cause_column=error_cause_column,
        skip_error_cause_env=skip_error_cause_env,
        sheet=sheet,
        section_title=section_title,
        row_axis_name="A",
        col_axis_name="B",
    )


def run_five_repro_ali_gt_fourways(
    path_repro: Path,
    path_ali: Path,
    *,
    repro_seed_col: str,
    ali_seed_col: str,
    repro_overall_col: str,
    ali_overall_col: str,
    ali_opus_column: str,
    gt_column: str,
    id_column: str,
    error_cause_column: str,
    skip_error_cause_env: bool,
    sheet: Any,
    compare_mode: str,
) -> None:
    """
    A=复现表、B=阿里表时，输出 3a–3e 五张 4×4，外加「对比 4」人评×阿里 Opus。

    3a 复现 seed × 阿里总体；3b 人评 × 阿里 seed；3c 人评 × 复现 seed；
    3d 人评 × 阿里总体；3e 人评 × 复现总体；对比 4 人评 × 阿里 Opus Result。
    """
    prefix = "对比 3" if compare_mode == "all" else "复现↔阿里/人评"

    def _one(
        num: str,
        title: str,
        path_row: Path,
        column_row: str,
        path_col: Path,
        column_col: str,
        row_axis_name: str,
        col_axis_name: str,
    ) -> None:
        st = f"{prefix}{num}：{title}" if compare_mode == "all" else f"{prefix} {num}：{title}"
        run_fourway_cross_matrix(
            path_row,
            column_row,
            path_col,
            column_col,
            id_column=id_column,
            error_cause_column=error_cause_column,
            skip_error_cause_env=skip_error_cause_env,
            sheet=sheet,
            section_title=st,
            row_axis_name=row_axis_name,
            col_axis_name=col_axis_name,
            append_binary_accuracy_3x3=True,
        )

    _one(
        "a",
        f"复现 seed({repro_seed_col!r}) × 阿里总体({ali_overall_col!r})",
        path_repro,
        repro_seed_col,
        path_ali,
        ali_overall_col,
        "复现seed",
        "阿里总",
    )
    _one(
        "b",
        f"人评({gt_column!r}) × 阿里 seed({ali_seed_col!r})",
        path_ali,
        gt_column,
        path_ali,
        ali_seed_col,
        "人评",
        "阿里seed",
    )
    _one(
        "c",
        f"人评({gt_column!r}) × 复现 seed({repro_seed_col!r})",
        path_ali,
        gt_column,
        path_repro,
        repro_seed_col,
        "人评",
        "复现seed",
    )
    _one(
        "d",
        f"人评({gt_column!r}) × 阿里总体({ali_overall_col!r})",
        path_ali,
        gt_column,
        path_ali,
        ali_overall_col,
        "人评",
        "阿里总",
    )
    _one(
        "e",
        f"人评({gt_column!r}) × 复现总体({repro_overall_col!r})",
        path_ali,
        gt_column,
        path_repro,
        repro_overall_col,
        "人评",
        "复现总",
    )

    title_opus = (
        f"对比 4：人评({gt_column!r}) × 阿里 Opus({ali_opus_column!r})"
        if compare_mode == "all"
        else f"复现↔阿里/人评 4：人评 × 阿里 Opus({ali_opus_column!r})"
    )
    run_fourway_cross_matrix(
        path_ali,
        gt_column,
        path_ali,
        ali_opus_column,
        id_column=id_column,
        error_cause_column=error_cause_column,
        skip_error_cause_env=skip_error_cause_env,
        sheet=sheet,
        section_title=title_opus,
        row_axis_name="人评",
        col_axis_name="阿里Opus",
        append_binary_accuracy_3x3=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="比较两份 xlsx 中判定列（AutoJudge / doubao 等）的二值一致率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--a",
        type=Path,
        default=root / "data" / "seed_free" / "results.xlsx",
        help="第一份 xlsx（默认 data/seed_free/results.xlsx）",
    )
    parser.add_argument(
        "--b",
        type=Path,
        default=root / "data" / "seed_free" / "results (1).xlsx",
        help="第二份 xlsx（默认 data/seed_free/results (1).xlsx）",
    )
    parser.add_argument(
        "--sheet",
        default="0",
        help="工作表：从 0 开始的索引，或名称（默认 0）",
    )
    parser.add_argument(
        "--compare",
        choices=("all", "autojudge", "doubao"),
        default="all",
        help="all：AutoJudge + doubao 两组；autojudge：仅 AutoJudge；doubao：仅 doubao-seed vs doubao-seed-1-8",
    )
    parser.add_argument(
        "--human-column",
        type=str,
        default=DEFAULT_AUTOJUDGE_COLUMN,
        help="AutoJudge 对比时 A/B 共用列名（默认 AutoJudge Result）",
    )
    parser.add_argument(
        "--doubao-column-a",
        type=str,
        default=DEFAULT_DOUBAO_COLUMN_A,
        help="doubao 对比时 A 表列名（默认 doubao-seed Result）",
    )
    parser.add_argument(
        "--doubao-column-b",
        type=str,
        default=DEFAULT_DOUBAO_COLUMN_B,
        help="doubao 对比时 B 表列名（默认 doubao-seed-1-8 Result）",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="Trajectory ID",
        help="轨迹 ID 列名",
    )
    parser.add_argument(
        "--error-cause-column",
        type=str,
        default="Error Cause",
        help="辅助排除环境错误；空字符串表示不读该列",
    )
    parser.add_argument(
        "--no-skip-error-cause-env",
        action="store_true",
        help="不因 Error Cause 含「环境错误」而排除",
    )
    parser.add_argument(
        "--no-doubao-fourway",
        action="store_true",
        help="豆包对比时不输出「A 豆包 vs B 豆包」四分类 4×4 联表",
    )
    parser.add_argument(
        "--gt-column",
        type=str,
        default=DEFAULT_GT_COLUMN,
        help="人评真值列名（默认 human_Judge，仅从 B/阿里表读取）",
    )
    parser.add_argument(
        "--overall-column-a",
        type=str,
        default=DEFAULT_OVERALL_COLUMN,
        help="A/复现表「多模型总体」列名（默认 AutoJudge Result）",
    )
    parser.add_argument(
        "--overall-column-b",
        type=str,
        default=DEFAULT_OVERALL_COLUMN,
        help="B/阿里表「多模型总体」列名（默认 AutoJudge Result）",
    )
    parser.add_argument(
        "--ali-opus-column",
        type=str,
        default=DEFAULT_ALI_OPUS_RESULT_COLUMN,
        help="B/阿里表 Claude Opus 判定列（默认 aws.claude-opus-4-5-20251101 Result，可与换行表头自动对齐）",
    )
    parser.add_argument(
        "--no-gt-fourway",
        action="store_true",
        help="不输出复现↔阿里/人评 3a–3e 及对比 4（Opus）共 6 张 4×4",
    )
    args = parser.parse_args()
    ec_col = args.error_cause_column.strip()

    sheet_arg: Any = args.sheet
    try:
        sheet_arg = int(args.sheet)
    except ValueError:
        sheet_arg = args.sheet

    for p, label in ((args.a, "A"), (args.b, "B")):
        if not p.is_file():
            print(f"找不到 {label} 表: {p}", file=sys.stderr)
            return 1

    path_a = args.a.resolve()
    path_b = args.b.resolve()
    skip_ec = not args.no_skip_error_cause_env

    try:
        if args.compare in ("all", "autojudge"):
            title_aj = (
                f"对比 1：{args.human_column!r}（A/B 同列）"
                if args.compare == "all"
                else f"AutoJudge：列 {args.human_column!r}（A/B 同列）"
            )
            run(
                path_a,
                path_b,
                human_column_a=args.human_column,
                human_column_b=args.human_column,
                id_column=args.id_column,
                error_cause_column=ec_col,
                skip_error_cause_env=skip_ec,
                sheet=sheet_arg,
                section_title=title_aj,
            )
        if args.compare in ("all", "doubao"):
            title_db = (
                f"对比 2：A={args.doubao_column_a!r} vs B={args.doubao_column_b!r}"
                if args.compare == "all"
                else f"doubao：A={args.doubao_column_a!r} vs B={args.doubao_column_b!r}"
            )
            run(
                path_a,
                path_b,
                human_column_a=args.doubao_column_a,
                human_column_b=args.doubao_column_b,
                id_column=args.id_column,
                error_cause_column=ec_col,
                skip_error_cause_env=skip_ec,
                sheet=sheet_arg,
                section_title=title_db,
            )
            if not args.no_doubao_fourway:
                title_fw = (
                    "对比 2b：豆包四分类 4×4（perfect / fair_enough / failed / env_error）"
                    if args.compare == "all"
                    else "豆包四分类 4×4（perfect / fair_enough / failed / env_error）"
                )
                run_doubao_fourway_matrix(
                    path_a,
                    path_b,
                    human_column_a=args.doubao_column_a,
                    human_column_b=args.doubao_column_b,
                    id_column=args.id_column,
                    error_cause_column=ec_col,
                    skip_error_cause_env=skip_ec,
                    sheet=sheet_arg,
                    section_title=title_fw,
                )
            if not args.no_gt_fourway:
                run_five_repro_ali_gt_fourways(
                    path_a,
                    path_b,
                    repro_seed_col=args.doubao_column_a,
                    ali_seed_col=args.doubao_column_b,
                    repro_overall_col=args.overall_column_a,
                    ali_overall_col=args.overall_column_b,
                    ali_opus_column=args.ali_opus_column,
                    gt_column=args.gt_column,
                    id_column=args.id_column,
                    error_cause_column=ec_col,
                    skip_error_cause_env=skip_ec,
                    sheet=sheet_arg,
                    compare_mode=args.compare,
                )
    except (KeyError, ValueError, RuntimeError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
