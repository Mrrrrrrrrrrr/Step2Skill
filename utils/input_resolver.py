"""Input helpers for JSONL/XLSX task sources."""

from __future__ import annotations

import json
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ID_COLUMNS = ("trajectory_id", "task_id")
# 表头别名（大小写/空格/下划线不敏感），用于 xlsx 人工表等
_ID_HEADER_ALIASES = frozenset(
    {
        "trajectoryid",
        "taskid",
        "trajectory_id",
        "task_id",
    }
)
PREDICTION_BACKFILL_COLUMNS = (
    ("zj_status", "task_status"),
    ("primary_error", "primary_error_cause"),
    ("secondary_error", "secondary_error_cause"),
    ("zj_reasoning", "error_cause_reasoning"),
    ("takeover_missing", "takeover_missing"),
    ("takeover_overuse", "takeover_overuse"),
    ("takeover_type_correct", "takeover_type_correct"),
    ("expected_takeover_type", "expected_takeover_type"),
    ("takeover_diagnosis", "takeover_diagnosis"),
)
STATUS_CLASS_MAP = {
    "perfect": 1,
    "fair_enough": 1,
    "failed": 0,
    "env_error": -1,
}
STATUS_CLASS_LABELS = (1, 0, -1)
# Canonical status tokens after alias normalization (key = _normalize_header_key(raw)).
_STATUS_CANONICAL: Dict[str, Optional[str]] = {
    "perfect": "perfect",
    "fairenough": "fair_enough",
    "failed": "failed",
    "failure": "failed",
    "fail": "failed",
    "enverror": "env_error",
    "env": "env_error",
    "success": "perfect",
    "pass": "perfect",
    "成功": "perfect",
    "完美": "perfect",
    "基本完成": "fair_enough",
    "尚可": "fair_enough",
    "失败": "failed",
    "环境错误": "env_error",
    "环境问题": "env_error",
    "环境": "env_error",
    # Explicitly excluded from metrics
    "不确定": None,
    "uncertain": None,
    "unknown": None,
    "": None,
}
_ENV_SYSTEM_ERROR_KEYWORDS = (
    "环境",
    "系统",
    "env",
    "environment",
    "system",
)
_HUMAN_STATUS_HEADER_ALIASES = frozenset(
    {
        "人评结果",
        "人工标注",
        "人工结果",
        "humanstatus",
        "humanresult",
        "human_result",
        "humanlabel",
        "gt",
        "gtresult",
        "gt_result",
        "groundtruth",
        "groundtruthresult",
        "ground_truth_result",
    }
)


def normalize_task_for_missing_tags(task: Dict[str, Any]) -> Dict[str, Any]:
    """Make platform task dicts tolerant of missing top-level/step tags."""
    if not isinstance(task.get("tags"), list):
        tags = task.get("tags")
        task["tags"] = [tags] if tags else []

    traj = task.get("traj")
    if isinstance(traj, list):
        for step in traj:
            if not isinstance(step, dict):
                continue
            if not isinstance(step.get("tags"), list):
                step_tags = step.get("tags")
                step["tags"] = [step_tags] if step_tags else []
    return task


def _column_letters_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: List[str] = []
    for si in root.findall("x:si", ns):
        parts = [node.text or "" for node in si.findall(".//x:t", ns)]
        strings.append("".join(parts))
    return strings


def _sheet_path(zf: zipfile.ZipFile, sheet_name: Optional[str] = None) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    ns = {
        "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    sheets = workbook.findall("x:sheets/x:sheet", ns)
    if not sheets:
        raise ValueError("xlsx 中未找到 worksheet")
    selected_sheet = sheets[0]
    if sheet_name:
        wanted = sheet_name.strip()
        for sheet in sheets:
            if sheet.attrib.get("name") == wanted:
                selected_sheet = sheet
                break
        else:
            available = [sheet.attrib.get("name", "") for sheet in sheets]
            raise ValueError(f"xlsx 未找到 sheet {wanted!r}，可选: {available}")

    rel_id = selected_sheet.attrib.get(f"{{{ns['r']}}}id")
    if not rel_id:
        raise ValueError("xlsx worksheet 缺少 relationship id")

    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    for rel in rels.findall("rel:Relationship", rel_ns):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib.get("Target", "")
            if target.startswith("/"):
                return target.lstrip("/")
            if target.startswith("xl/"):
                return target
            return posixpath.normpath(posixpath.join("xl", target))
    raise ValueError(f"xlsx 未找到 worksheet relationship: {rel_id}")


def _first_sheet_path(zf: zipfile.ZipFile) -> str:
    return _sheet_path(zf)


def _cell_value(cell: ET.Element, shared_strings: Sequence[str], ns: dict) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//x:t", ns)).strip()

    value_node = cell.find("x:v", ns)
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(raw)].strip()
        except Exception:
            return raw
    return raw


def _normalize_header_key(name: str) -> str:
    """将表头归一化为小写字母数字，便于匹配 Trajectory ID / task_id 等变体。"""
    return "".join(ch for ch in name.strip().lower() if ch.isalnum())


def _map_status_class(label: Any) -> Optional[int]:
    """Map human/zj status cell text to evaluation class {1,0,-1}; None = skip."""
    if label is None:
        return None
    raw = str(label).strip()
    if not raw:
        return None
    if raw in STATUS_CLASS_MAP:
        return STATUS_CLASS_MAP[raw]
    canon = _STATUS_CANONICAL.get(_normalize_header_key(raw), raw)
    if canon is None:
        return None
    if isinstance(canon, str) and canon in STATUS_CLASS_MAP:
        return STATUS_CLASS_MAP[canon]
    return STATUS_CLASS_MAP.get(canon)


def _normalize_task_id_value(value: Any) -> str:
    """Normalize task_id values from xlsx/jsonl for stable matching/export."""
    task_id = "" if value is None else str(value).strip()
    if task_id.endswith(".0") and task_id[:-2].isdigit():
        task_id = task_id[:-2]
    if task_id.endswith("_main"):
        task_id = task_id[: -len("_main")]
    return task_id


def _match_id_column(headers: Dict[str, int]) -> Optional[Tuple[int, str]]:
    """在表头 dict（列名 -> 列号）中查找 trajectory_id / task_id 及其别名。"""
    for header_text, col_idx in headers.items():
        if _normalize_header_key(header_text) in _ID_HEADER_ALIASES:
            return col_idx, header_text
    return None


def _headers_from_row_values(row: Dict[int, str]) -> Dict[str, int]:
    return {value.strip(): idx for idx, value in row.items() if value.strip()}


def _read_sheet_rows(xlsx_path: str, sheet_name: Optional[str] = None) -> List[Dict[int, str]]:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_path = _sheet_path(zf, sheet_name=sheet_name)
        sheet = ET.fromstring(zf.read(sheet_path))

        rows: List[Dict[int, str]] = []
        for row in sheet.findall(".//x:sheetData/x:row", ns):
            values: Dict[int, str] = {}
            next_idx = 0
            for cell in row.findall("x:c", ns):
                ref = cell.attrib.get("r", "")
                idx = _column_letters_to_index(ref) if ref else next_idx
                values[idx] = _cell_value(cell, shared_strings, ns)
                next_idx = idx + 1
            if values:
                rows.append(values)
    return rows


def _read_first_sheet_rows(xlsx_path: str) -> List[Dict[int, str]]:
    return _read_sheet_rows(xlsx_path)


def _match_human_status_column(headers: Dict[str, int]) -> Optional[Tuple[int, str]]:
    for header_text, col_idx in headers.items():
        if _normalize_header_key(header_text) in _HUMAN_STATUS_HEADER_ALIASES:
            return col_idx, header_text
    return None


def read_human_status_from_xlsx(xlsx_path: str, sheet_name: Optional[str] = None) -> Dict[str, int]:
    """Read human four-class labels from xlsx and map them to evaluation.status."""
    rows = _read_sheet_rows(xlsx_path, sheet_name=sheet_name)
    if not rows:
        return {}

    header_row_idx: Optional[int] = None
    id_col_idx: Optional[int] = None
    status_col_idx: Optional[int] = None
    for row_idx, row in enumerate(rows[:20]):
        headers = _headers_from_row_values(row)
        id_matched = _match_id_column(headers)
        status_matched = _match_human_status_column(headers)
        if id_matched is not None and status_matched is not None:
            header_row_idx = row_idx
            id_col_idx = id_matched[0]
            status_col_idx = status_matched[0]
            break

    if header_row_idx is None or id_col_idx is None or status_col_idx is None:
        return {}

    status_by_task_id: Dict[str, int] = {}
    for row in rows[header_row_idx + 1 :]:
        task_id = _normalize_task_id_value(row.get(id_col_idx, ""))
        if not task_id:
            continue
        mapped = _map_status_class(row.get(status_col_idx, ""))
        if mapped is None:
            continue
        status_by_task_id[task_id] = mapped
    return status_by_task_id


def apply_human_status_to_jsonl(jsonl_path: str, status_by_task_id: Dict[str, int]) -> int:
    """Patch evaluation.status in a JSONL file by task_id; returns matched count."""
    if not status_by_task_id:
        return 0

    source = Path(jsonl_path)
    tmp_path = source.with_name(f"{source.name}.tmp")
    matched = 0
    with open(source, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                fout.write(line)
                continue
            obj = json.loads(line)
            task_id = _normalize_task_id_value(obj.get("task_id"))
            if task_id in status_by_task_id:
                evaluation = obj.get("evaluation")
                if not isinstance(evaluation, dict):
                    evaluation = {}
                    obj["evaluation"] = evaluation
                evaluation["status"] = status_by_task_id[task_id]
                matched += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp_path.replace(source)
    return matched


def read_task_ids_from_xlsx(xlsx_path: str, sheet_name: Optional[str] = None) -> List[str]:
    """Read trajectory_id/task_id values from the first worksheet of an .xlsx."""
    rows = _read_sheet_rows(xlsx_path, sheet_name=sheet_name)

    if not rows:
        raise ValueError(f"xlsx 无有效行: {xlsx_path}")

    header_row_idx: Optional[int] = None
    id_col_idx: Optional[int] = None
    for row_idx, row in enumerate(rows[:20]):
        headers = _headers_from_row_values(row)
        matched = _match_id_column(headers)
        if matched is not None:
            header_row_idx = row_idx
            id_col_idx = matched[0]
            break

    if header_row_idx is None or id_col_idx is None:
        raise ValueError(f"xlsx 未找到 trajectory_id 或 task_id 字段: {xlsx_path}")

    seen = set()
    task_ids: List[str] = []
    for row in rows[header_row_idx + 1 :]:
        value = _normalize_task_id_value(row.get(id_col_idx, ""))
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            task_ids.append(value)

    if not task_ids:
        raise ValueError(f"xlsx 的 ID 字段为空: {xlsx_path}")
    return task_ids


def _safe_path_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return slug.strip("._-") or "sheet"


def _default_jsonl_path_for_xlsx(xlsx_path: str, sheet_name: Optional[str] = None) -> str:
    path = Path(xlsx_path)
    suffix = f"_{_safe_path_slug(sheet_name)}" if sheet_name else ""
    return str(path.with_name(f"{path.stem}{suffix}_raw_data.jsonl"))


def export_xlsx_task_ids_to_jsonl(
    xlsx_path: str,
    output_jsonl: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> str:
    """Export task ids from xlsx to a sibling JSONL via data_flywheel/export_raw_data.py."""
    task_ids = read_task_ids_from_xlsx(xlsx_path, sheet_name=sheet_name)
    output_path = output_jsonl or _default_jsonl_path_for_xlsx(xlsx_path, sheet_name=sheet_name)

    export_dir = PROJECT_ROOT / "data" / "data_flywheel"
    if str(export_dir) not in sys.path:
        sys.path.insert(0, str(export_dir))

    from export_raw_data import main as export_raw_data_main  # type: ignore

    argv = [
        "--task-ids",
        *task_ids,
        "--out",
        output_path,
        "--mode",
        "eval",
    ]
    sheet_note = f" sheet={sheet_name}" if sheet_name else ""
    print(f"检测到 xlsx 输入{sheet_note}，读取 {len(task_ids)} 个 task_id，导出 JSONL: {output_path}")
    export_raw_data_main(argv)
    human_status = read_human_status_from_xlsx(xlsx_path, sheet_name=sheet_name)
    matched = apply_human_status_to_jsonl(output_path, human_status)
    if human_status:
        print(
            f"已从 xlsx 人评结果写入 evaluation.status: "
            f"匹配 {matched}/{len(human_status)} 条"
        )
    return output_path


def resolve_input_to_jsonl(input_path: str, xlsx_sheet: Optional[str] = None) -> str:
    """Return a JSONL path, exporting xlsx inputs when needed."""
    if not input_path:
        raise ValueError("input_path 不能为空")
    suffix = Path(input_path).suffix.lower()
    if suffix == ".jsonl":
        return input_path
    if suffix == ".xlsx":
        return export_xlsx_task_ids_to_jsonl(input_path, sheet_name=xlsx_sheet)
    raise ValueError(f"不支持的输入格式: {input_path}（仅支持 .jsonl/.xlsx）")


def is_xlsx_input(input_path: str) -> bool:
    """Return True when the original CLI input is an .xlsx file."""
    return Path(input_path or "").suffix.lower() == ".xlsx"


def _find_header_and_id_column(ws: Any) -> Tuple[int, int, Dict[str, int]]:
    """Find the header row and task id column in the first worksheet."""
    max_scan = min(ws.max_row, 20)
    for row_idx in range(1, max_scan + 1):
        headers: Dict[str, int] = {}
        for col_idx in range(1, ws.max_column + 1):
            value = ws.cell(row=row_idx, column=col_idx).value
            if value is None:
                continue
            text = str(value).strip()
            if text:
                headers[text] = col_idx
        matched = _match_id_column(headers)
        if matched is not None:
            return row_idx, matched[0], headers
    raise ValueError("xlsx 未找到 trajectory_id 或 task_id 字段")


def _jsonl_prediction_rows(predicted_jsonl_path: str) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with open(predicted_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            task_id = _normalize_task_id_value(obj.get("task_id"))
            if not task_id:
                continue
            ev = obj.get("evaluation") if isinstance(obj.get("evaluation"), dict) else {}
            pred = ev.get("prediction") if isinstance(ev.get("prediction"), dict) else {}
            orm_pred = ev.get("orm_prediction") if isinstance(ev.get("orm_prediction"), dict) else {}
            values: Dict[str, str] = {}
            for column_name, key in PREDICTION_BACKFILL_COLUMNS:
                value = pred.get(key, "") if isinstance(pred, dict) else ""
                values[column_name] = value if isinstance(value, str) else ("" if value is None else str(value))
            if orm_pred.get("prm_mode"):
                values["_prm_mode"] = "1"
                values["zj_status"] = ""
            rows[task_id] = values
    return rows


def backfill_prediction_fields_to_xlsx(
    source_xlsx_path: str,
    predicted_jsonl_path: str,
    output_xlsx_path: Optional[str] = None,
    include_status: bool = True,
    sheet_name: Optional[str] = None,
) -> str:
    """Write selected evaluation.prediction fields from a predicted JSONL into an XLSX copy."""
    try:
        from openpyxl import load_workbook
    except ImportError as e:
        raise ImportError("请安装: pip install openpyxl") from e

    source = Path(source_xlsx_path)
    output = Path(output_xlsx_path) if output_xlsx_path else Path(predicted_jsonl_path).with_suffix(".xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)

    prediction_by_task_id = _jsonl_prediction_rows(predicted_jsonl_path)
    wb = load_workbook(source)
    if sheet_name:
        if sheet_name not in wb.sheetnames:
            raise ValueError(f"xlsx 未找到 sheet {sheet_name!r}，可选: {wb.sheetnames}")
        ws = wb[sheet_name]
    else:
        ws = wb.worksheets[0]
    header_row, id_col, headers = _find_header_and_id_column(ws)

    backfill_columns = PREDICTION_BACKFILL_COLUMNS
    if not include_status:
        backfill_columns = tuple(
            item for item in PREDICTION_BACKFILL_COLUMNS if item[0] != "zj_status"
        )

    next_col = ws.max_column + 1
    target_cols: Dict[str, int] = {}
    for column_name, _ in backfill_columns:
        col = headers.get(column_name)
        if col is None:
            col = next_col
            next_col += 1
            ws.cell(row=header_row, column=col).value = column_name
        target_cols[column_name] = col

    matched = 0
    for row_idx in range(header_row + 1, ws.max_row + 1):
        task_id = _normalize_task_id_value(ws.cell(row=row_idx, column=id_col).value)
        values = prediction_by_task_id.get(task_id)
        if not values:
            continue
        matched += 1
        for field, col in target_cols.items():
            if field == "zj_status" and values.get("_prm_mode") == "1" and not values.get(field):
                continue
            ws.cell(row=row_idx, column=col).value = values.get(field, "")

    wb.save(output)
    status_note = "含状态" if include_status else "不回填状态，仅回填错因"
    print(f"表格回填完成: {output}（匹配 {matched}/{len(prediction_by_task_id)} 条，{status_note}）")
    return str(output)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _is_env_or_system_error(text: str) -> bool:
    if not text:
        return False
    lowered = text.strip().lower()
    if not lowered:
        return False
    return any(keyword in lowered for keyword in _ENV_SYSTEM_ERROR_KEYWORDS)


def load_task_ids_from_jsonl(jsonl_path: str) -> set[str]:
    """Load task_id set from a JSONL file; ignores invalid/empty lines."""
    task_ids: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            task_id = _normalize_task_id_value(obj.get("task_id"))
            if task_id:
                task_ids.add(task_id)
    return task_ids


def summarize_xlsx_human_vs_zj_status(
    xlsx_path: str,
    sheet_name: Optional[str] = None,
    excluded_task_ids: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Print confusion metrics when an output XLSX has human labels and zj_status."""
    try:
        from openpyxl import load_workbook
    except ImportError as e:
        raise ImportError("请安装: pip install openpyxl") from e

    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet_name:
        if sheet_name not in wb.sheetnames:
            raise ValueError(f"xlsx 未找到 sheet {sheet_name!r}，可选: {wb.sheetnames}")
        ws = wb[sheet_name]
    else:
        ws = wb.worksheets[0]

    rows = ws.iter_rows(values_only=True)
    try:
        header_values = next(rows)
    except StopIteration:
        return None

    headers: Dict[str, int] = {}
    for idx, value in enumerate(header_values):
        text = "" if value is None else str(value).strip()
        if text and text not in headers:
            headers[text] = idx

    human_matched = _match_human_status_column(headers)
    if human_matched is None or "zj_status" not in headers:
        return None

    id_matched = _match_id_column(headers)
    task_id_col = id_matched[0] if id_matched is not None else None
    excluded_ids_set = {
        _normalize_task_id_value(t)
        for t in (excluded_task_ids or [])
        if _normalize_task_id_value(t)
    }

    human_col = human_matched[0]
    human_header = human_matched[1]
    pred_col = headers["zj_status"]
    primary_error_col = headers.get("primary_error")
    secondary_error_col = headers.get("secondary_error")
    matrix = {
        human: {pred: 0 for pred in STATUS_CLASS_LABELS}
        for human in STATUS_CLASS_LABELS
    }
    corrected_matrix = {
        human: {pred: 0 for pred in STATUS_CLASS_LABELS}
        for human in STATUS_CLASS_LABELS
    }
    raw_human_counts: Dict[str, int] = {}
    raw_pred_counts: Dict[str, int] = {}
    total_rows = 0
    valid_count = 0
    correct_count = 0
    corrected_correct_count = 0
    corrected_failed_to_env_count = 0
    unmapped_count = 0
    excluded_count = 0

    for row in rows:
        total_rows += 1
        row_task_id = ""
        if task_id_col is not None and task_id_col < len(row) and row[task_id_col] is not None:
            row_task_id = _normalize_task_id_value(row[task_id_col])
        if row_task_id and row_task_id in excluded_ids_set:
            excluded_count += 1
            continue

        human_raw = ""
        pred_raw = ""
        if human_col < len(row) and row[human_col] is not None:
            human_raw = str(row[human_col]).strip()
        if pred_col < len(row) and row[pred_col] is not None:
            pred_raw = str(row[pred_col]).strip()

        raw_human_counts[human_raw] = raw_human_counts.get(human_raw, 0) + 1
        raw_pred_counts[pred_raw] = raw_pred_counts.get(pred_raw, 0) + 1

        human = _map_status_class(human_raw)
        pred = _map_status_class(pred_raw)
        if human is None or pred is None:
            unmapped_count += 1
            continue

        primary_error_raw = ""
        secondary_error_raw = ""
        if primary_error_col is not None and primary_error_col < len(row) and row[primary_error_col] is not None:
            primary_error_raw = str(row[primary_error_col]).strip()
        if (
            secondary_error_col is not None
            and secondary_error_col < len(row)
            and row[secondary_error_col] is not None
        ):
            secondary_error_raw = str(row[secondary_error_col]).strip()
        corrected_pred = pred
        if pred == 0 and (
            _is_env_or_system_error(primary_error_raw)
            or _is_env_or_system_error(secondary_error_raw)
        ):
            corrected_pred = -1
            corrected_failed_to_env_count += 1

        valid_count += 1
        if human == pred:
            correct_count += 1
        if human == corrected_pred:
            corrected_correct_count += 1
        matrix[human][pred] += 1
        corrected_matrix[human][corrected_pred] += 1

    def _calc_matrix_metrics(
        matrix_input: Dict[int, Dict[int, int]]
    ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, Dict[str, float]]]:
        row_totals_local = {
            label: sum(matrix_input[label].values())
            for label in STATUS_CLASS_LABELS
        }
        col_totals_local = {
            label: sum(matrix_input[human][label] for human in STATUS_CLASS_LABELS)
            for label in STATUS_CLASS_LABELS
        }
        class_metrics_local: Dict[int, Dict[str, float]] = {}
        for label in STATUS_CLASS_LABELS:
            tp = matrix_input[label][label]
            precision = _safe_div(tp, col_totals_local[label])
            recall = _safe_div(tp, row_totals_local[label])
            class_metrics_local[label] = {
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        return row_totals_local, col_totals_local, class_metrics_local

    row_totals, col_totals, class_metrics = _calc_matrix_metrics(matrix)
    corrected_row_totals, corrected_col_totals, corrected_class_metrics = _calc_matrix_metrics(corrected_matrix)

    accuracy = _safe_div(correct_count, valid_count)
    corrected_accuracy = _safe_div(corrected_correct_count, valid_count)
    summary: Dict[str, Any] = {
        "xlsx_path": xlsx_path,
        "human_status_column": human_header,
        "total_rows": total_rows,
        "valid_count": valid_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "unmapped_count": unmapped_count,
        "excluded_count": excluded_count,
        "matrix": matrix,
        "row_totals": row_totals,
        "col_totals": col_totals,
        "class_metrics": class_metrics,
        "corrected_matrix": corrected_matrix,
        "corrected_row_totals": corrected_row_totals,
        "corrected_col_totals": corrected_col_totals,
        "corrected_class_metrics": corrected_class_metrics,
        "corrected_correct_count": corrected_correct_count,
        "corrected_accuracy": corrected_accuracy,
        "corrected_failed_to_env_count": corrected_failed_to_env_count,
        "raw_human_counts": raw_human_counts,
        "raw_pred_counts": raw_pred_counts,
    }

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    label_text = {
        1: "1(perfect/fair_enough)",
        0: "0(failed)",
        -1: "-1(env_error)",
    }
    print()
    print(f"{human_header} vs zj_status 三分类统计:")
    print(f"  表格行数: {total_rows}")
    print(f"  有效对比数: {valid_count}")
    print(f"  跳过/未知标签数: {unmapped_count}")
    if excluded_ids_set:
        print(f"  排除过滤样本数: {excluded_count}")
    print(f"  正确数: {correct_count}")
    print(f"  三分类准确率: {pct(accuracy)}")
    print("  混淆矩阵（行=人评结果，列=zj_status）:")
    header = f"    {human_header}\\预测" + "".join(f"\t{label_text[label]}" for label in STATUS_CLASS_LABELS) + "\t行总数"
    print(header)
    for human in STATUS_CLASS_LABELS:
        values = "".join(f"\t{matrix[human][pred]}" for pred in STATUS_CLASS_LABELS)
        print(f"    {label_text[human]}{values}\t{row_totals[human]}")
    col_values = "".join(f"\t{col_totals[pred]}" for pred in STATUS_CLASS_LABELS)
    print(f"    列总数{col_values}\t{valid_count}")
    print("  三分类 Precision / Recall 明细:")

    metric_names = {
        0: "failed",
        1: "正例",
        -1: "环境",
    }
    for label in (0, 1, -1):
        tp = matrix[label][label]
        col_parts = [matrix[human][label] for human in STATUS_CLASS_LABELS]
        row_parts = [matrix[label][pred] for pred in STATUS_CLASS_LABELS]
        precision_denom = col_totals[label]
        recall_denom = row_totals[label]
        precision = class_metrics[label]["precision"]
        recall = class_metrics[label]["recall"]
        print(f"    {metric_names[label]}:")
        print(
            f"      Precision = {tp} / ({'+'.join(str(v) for v in col_parts)}) "
            f"= {tp}/{precision_denom} = {pct(precision)}"
        )
        print(
            f"      Recall = {tp} / ({'+'.join(str(v) for v in row_parts)}) "
            f"= {tp}/{recall_denom} = {pct(recall)}"
        )
    print("  二分类重点指标:")
    pos = class_metrics[1]
    neg = class_metrics[0]
    print(
        f"    正例1: precision={pct(pos['precision'])}, "
        f"recall={pct(pos['recall'])}, F1={pct(pos['f1'])}"
    )
    print(
        f"    负例0: precision={pct(neg['precision'])}, "
        f"recall={pct(neg['recall'])}, F1={pct(neg['f1'])}"
    )

    print()
    print(
        "  修正版口径（zj_status=failed 且 primary/secondary_error 属于环境或系统 -> 改为 env_error）:"
    )
    print(f"    修正条数: {corrected_failed_to_env_count}")
    print(f"    正确数: {corrected_correct_count}")
    print(f"    三分类准确率: {pct(corrected_accuracy)}")
    print("    混淆矩阵（行=人评结果，列=修正后 zj_status）:")
    corrected_header = (
        f"      {human_header}\\预测"
        + "".join(f"\t{label_text[label]}" for label in STATUS_CLASS_LABELS)
        + "\t行总数"
    )
    print(corrected_header)
    for human in STATUS_CLASS_LABELS:
        values = "".join(f"\t{corrected_matrix[human][pred]}" for pred in STATUS_CLASS_LABELS)
        print(f"      {label_text[human]}{values}\t{corrected_row_totals[human]}")
    corrected_col_values = "".join(f"\t{corrected_col_totals[pred]}" for pred in STATUS_CLASS_LABELS)
    print(f"      列总数{corrected_col_values}\t{valid_count}")
    print("    三分类 Precision / Recall 明细:")
    for label in (0, 1, -1):
        tp = corrected_matrix[label][label]
        col_parts = [corrected_matrix[human][label] for human in STATUS_CLASS_LABELS]
        row_parts = [corrected_matrix[label][pred] for pred in STATUS_CLASS_LABELS]
        precision_denom = corrected_col_totals[label]
        recall_denom = corrected_row_totals[label]
        precision = corrected_class_metrics[label]["precision"]
        recall = corrected_class_metrics[label]["recall"]
        print(f"      {metric_names[label]}:")
        print(
            f"        Precision = {tp} / ({'+'.join(str(v) for v in col_parts)}) "
            f"= {tp}/{precision_denom} = {pct(precision)}"
        )
        print(
            f"        Recall = {tp} / ({'+'.join(str(v) for v in row_parts)}) "
            f"= {tp}/{recall_denom} = {pct(recall)}"
        )
    corrected_pos = corrected_class_metrics[1]
    corrected_neg = corrected_class_metrics[0]
    print("    二分类重点指标:")
    print(
        f"      正例1: precision={pct(corrected_pos['precision'])}, "
        f"recall={pct(corrected_pos['recall'])}, F1={pct(corrected_pos['f1'])}"
    )
    print(
        f"      负例0: precision={pct(corrected_neg['precision'])}, "
        f"recall={pct(corrected_neg['recall'])}, F1={pct(corrected_neg['f1'])}"
    )
    return summary
