#!/usr/bin/env python3
"""
将 AutoJudge 导出表（.xlsx / .csv）中的人工结论与 predicted JSONL 中 ORM 的 is_task_success 对齐，统计准确率。

典型用法（推荐完整 xlsx；「人工结论」列名多为 AutoJudge Result）：

  python utils/eval_human_autojudge_vs_orm.py \\
    --table "data/seed_free/results (1).xlsx" \\
    --jsonl data/seed_free/lite+action_list/2026032202055788_predicted.jsonl

规则：
- 人工：perfect、fair_enough → 1；failed → 0；env_error → 不参与评估（环境错误）
- 可选：Error Cause 含「环境错误」的行也排除
- ORM：evaluation.orm_prediction.is_task_success；仅 0/1 参与准确率，-1 单独统计
- 轨迹键：Trajectory ID 去掉后缀 _main 后与 JSONL 的 task_id 对齐
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    """
    返回 (kind, y)：
    - ("env", None) 环境错误，不参与评估
    - ("bad", None) 人工标签缺失或无法映射
    - ("ok", 0|1) 可参与与 ORM 对比
    """
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
    if h == "failed":
        return "ok", 0
    return "bad", None


def parse_orm_is_task_success(obj: Dict[str, Any]) -> Optional[int]:
    ev = obj.get("evaluation")
    if not isinstance(ev, dict):
        return None
    orm = ev.get("orm_prediction")
    if not isinstance(orm, dict):
        return None
    v = orm.get("is_task_success")
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        iv = int(v)
        if iv in (-1, 0, 1):
            return iv
        return None
    if isinstance(v, str):
        t = v.strip()
        if t in ("1", "0", "-1"):
            return int(t)
    return None


def _cell_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v).strip()


def load_csv_rows(
    csv_path: Path,
    *,
    id_column: str,
    human_column: str,
    error_cause_column: str,
) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
    """返回 (fieldnames, list of (norm_id, raw_tid, human_cell, error_cause))."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV 无表头")
        names = list(reader.fieldnames)
        for req in (id_column, human_column):
            if req not in reader.fieldnames:
                raise KeyError(f"表缺少列 {req!r}，现有: {names}")
        rows: List[Tuple[str, str, str, str]] = []
        for row in reader:
            tid = (row.get(id_column) or "").strip()
            if not tid:
                continue
            human = (row.get(human_column) or "").strip()
            if error_cause_column:
                ec = (row.get(error_cause_column) or "").strip()
            else:
                ec = ""
            rows.append((normalize_trajectory_key(tid), tid, human, ec))
        return names, rows


def load_xlsx_rows(
    xlsx_path: Path,
    *,
    id_column: str,
    human_column: str,
    error_cause_column: str,
    sheet: Any,
) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
    """与 load_csv_rows 相同返回结构。需要 openpyxl。

    注意：read_only + iter_rows 在部分导出表上会把首行截成单列（维度/合并格问题），
    因此用普通模式并按 max_column 逐格读取第 1 行表头。
    """
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
        for req in (id_column, human_column):
            if req not in headers:
                raise KeyError(f"表缺少列 {req!r}，现有: {headers}")
        idx_id = headers.index(id_column)
        idx_h = headers.index(human_column)
        idx_ec = headers.index(error_cause_column) if error_cause_column else None

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


def load_label_rows(
    table_path: Path,
    *,
    id_column: str,
    human_column: str,
    error_cause_column: str,
    sheet: Any,
) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
    suf = table_path.suffix.lower()
    if suf in (".xlsx", ".xlsm"):
        return load_xlsx_rows(
            table_path,
            id_column=id_column,
            human_column=human_column,
            error_cause_column=error_cause_column,
            sheet=sheet,
        )
    if suf == ".csv":
        return load_csv_rows(
            table_path,
            id_column=id_column,
            human_column=human_column,
            error_cause_column=error_cause_column,
        )
    raise ValueError(f"不支持的标注表后缀 {suf!r}，请使用 .xlsx / .xlsm / .csv")


def run(
    table_path: Path,
    jsonl_path: Path,
    *,
    human_column: str,
    id_column: str,
    error_cause_column: str,
    skip_error_cause_env: bool,
    sheet: Any,
) -> int:
    _, csv_rows = load_label_rows(
        table_path,
        id_column=id_column,
        human_column=human_column,
        error_cause_column=error_cause_column,
        sheet=sheet,
    )
    orm_by_id: Dict[str, int] = {}
    for obj in iter_jsonl(jsonl_path):
        tid = (obj.get("task_id") or "").strip()
        if not tid:
            continue
        v = parse_orm_is_task_success(obj)
        if v is not None:
            orm_by_id[tid] = v

    human_values = [r[2].strip().lower() for r in csv_rows]
    if human_column == "human_Judge" and len(set(human_values)) == 1 and human_values[0] == "failed":
        print(
            "警告: human_Judge 列取值均为 failed，多为 rollout 状态而非人工 verdict。"
            " 若表内有 AutoJudge Result 列，请加参数: --human-column 'AutoJudge Result'",
            file=sys.stderr,
        )

    skipped_env = 0
    skipped_bad_human = 0
    skipped_no_jsonl = 0
    orm_uncertain = 0
    compared = 0
    correct = 0
    tp = fp = tn = fn = 0
    mismatches: List[Tuple[str, int, int]] = []

    seen_ids = {r[0] for r in csv_rows}
    for norm_id, raw_tid, human_cell, ec in csv_rows:
        kind, y = classify_human_row(
            human_cell,
            ec,
            also_skip_by_error_cause=skip_error_cause_env,
        )
        if kind == "env":
            skipped_env += 1
            continue
        if kind == "bad" or y is None:
            skipped_bad_human += 1
            continue
        pred = orm_by_id.get(norm_id)
        if pred is None:
            skipped_no_jsonl += 1
            continue
        if pred == -1:
            orm_uncertain += 1
            continue
        if pred not in (0, 1):
            skipped_no_jsonl += 1
            continue
        compared += 1
        if y == pred:
            correct += 1
        else:
            mismatches.append((raw_tid, y, pred))
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 0 and pred == 0:
            tn += 1
        elif y == 1 and pred == 0:
            fn += 1

    jsonl_ids = set(orm_by_id.keys())
    in_jsonl_not_csv = len(jsonl_ids - seen_ids)
    in_csv_not_jsonl = sum(1 for r in csv_rows if r[0] not in orm_by_id)

    acc = (correct / compared * 100) if compared else 0.0
    print(f"标注表: {table_path}")
    print(f"JSONL: {jsonl_path}")
    print(f"人工列: {human_column!r}，轨迹列: {id_column!r}")
    print(f"标注表行数(有 Trajectory ID): {len(csv_rows)}")
    print(f"排除环境错误(env_error 或 Error Cause 含环境错误): {skipped_env}")
    print(f"人工标签无法映射(空或未知): {skipped_bad_human}")
    print(f"JSONL 无对应 task_id: {in_csv_not_jsonl}（按轨迹键未命中 orm 预测）")
    print(f"ORM is_task_success == -1 跳过: {orm_uncertain}")
    print(f"参与对比条数: {compared}")
    print(f"准确率: {acc:.2f}% ({correct}/{compared})")
    print(f"混淆(人工=1 正类): TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"JSONL 中有而标注表未出现的 task_id 数: {in_jsonl_not_csv}")

    if mismatches and len(mismatches) <= 30:
        print("\n不一致样例 (Trajectory ID, human_bin, orm):")
        for tid, y, p in mismatches:
            print(f"  {tid}\t human={y}\t orm={p}")
    elif mismatches:
        print(f"\n不一致共 {len(mismatches)} 条（仅展示前 20）:")
        for tid, y, p in mismatches[:20]:
            print(f"  {tid}\t human={y}\t orm={p}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="人工 AutoJudge 表（xlsx/csv）与 ORM is_task_success 对齐评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help="人工标注表：.xlsx / .xlsm / .csv",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="兼容旧参数，等同于 --table（csv 或任意支持格式）",
    )
    parser.add_argument("--jsonl", type=Path, required=True, help="predicted JSONL 路径")
    parser.add_argument(
        "--sheet",
        default="0",
        help="xlsx 工作表：从 0 开始的索引，或工作表名称（默认 0）",
    )
    parser.add_argument(
        "--human-column",
        type=str,
        default="AutoJudge Result",
        help="人工结论列名（默认 AutoJudge Result；若表只有 human_Judge 且全为 failed 请改默认）",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="Trajectory ID",
        help="轨迹 ID 列名（默认 Trajectory ID）",
    )
    parser.add_argument(
        "--error-cause-column",
        type=str,
        default="Error Cause",
        help="用于辅助排除环境错误的列，空字符串表示不读该列",
    )
    parser.add_argument(
        "--no-skip-error-cause-env",
        action="store_true",
        help="不要因 Error Cause 含「环境错误」而排除（仅按 env_error 标签排除）",
    )
    args = parser.parse_args()
    ec_col = args.error_cause_column.strip()
    table_path = args.table or args.csv
    if table_path is None:
        print("请指定 --table 路径（或兼容 --csv）", file=sys.stderr)
        return 1

    sheet_arg: Any = args.sheet
    try:
        sheet_arg = int(args.sheet)
    except ValueError:
        sheet_arg = args.sheet

    if not table_path.is_file():
        print(f"找不到标注表: {table_path}", file=sys.stderr)
        return 1
    if not args.jsonl.is_file():
        print(f"找不到 JSONL: {args.jsonl}", file=sys.stderr)
        return 1

    try:
        return run(
            table_path,
            args.jsonl,
            human_column=args.human_column,
            id_column=args.id_column,
            error_cause_column=ec_col,
            skip_error_cause_env=not args.no_skip_error_cause_env,
            sheet=sheet_arg,
        )
    except (KeyError, ValueError, RuntimeError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
