r"""Edit failed-step images and rerun one-step prediction for selected errors.

This script consumes mobilegym_error_classification.json. For category 1..5
failures it edits the failed-step screenshot, replaces the current-step image in
the prompt messages carried by the previous JSON, and calls OpenAI-compatible
model endpoints. For category 6..10 failures it builds contrastive correction
hints.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent

from api import ApiWrapper  # type: ignore  # noqa: E402
from tools.image_edit import (  # type: ignore  # noqa: E402
    random_black_corner_occlusion,
    random_small_crop,
)
from tools.som_action import make_marked_image  # type: ignore  # noqa: E402

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - handled at runtime by SOM fallback
    Image = None
    ImageDraw = None


DEFAULT_INPUT = PROJECT_ROOT / "mobilegym_error_classification.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "mobilegym_edited_repredictions.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "mobilegym_edit_repredict_results"
DEFAULT_MODEL_BASE_URL = "http://10.80.10.0:30772/v1"
DEFAULT_MODEL_NAME = "Qwen3.5-27B-vllm"
AGENT_CATEGORY_IDS = {1, 2, 3, 4, 5}
PITFALL_CATEGORY_IDS = {6, 7, 8, 9, 10}
TARGET_CATEGORY_IDS = AGENT_CATEGORY_IDS | PITFALL_CATEGORY_IDS
EDIT_VARIANTS = ("crop", "corner_occlusion")
WRONG_ACTION_MARKED_VARIANT = "wrong_action_marked"
ORIGINAL_API_VARIANTS = ("original_doubao", "original_qwen_plus")
PITFALL_LLM_VARIANTS = ("contrastive_correction_doubao", "contrastive_correction_qwen_plus")
PITFALL_REPREDICT_API_VARIANTS = ("default", "doubao", "qwen_plus")
PITFALL_HINT_REPREDICT_API_VARIANTS = ("doubao", "qwen_plus")
REPREDICT_FAMILIES = ("edit", "original_api", "wrong_action_marked", "contrastive_correction")
PITFALL_WRONG_ACTION_MARKED_VARIANTS = tuple(
    f"{WRONG_ACTION_MARKED_VARIANT}_{variant}" for variant in PITFALL_REPREDICT_API_VARIANTS
)
CONTRASTIVE_REPREDICT_VARIANTS = (
    "contrastive_correction_doubao_doubao",
    "contrastive_correction_qwen_plus_qwen_plus",
)
KNOWN_REPREDICT_VARIANTS = (
    *EDIT_VARIANTS,
    WRONG_ACTION_MARKED_VARIANT,
    *PITFALL_WRONG_ACTION_MARKED_VARIANTS,
    *ORIGINAL_API_VARIANTS,
    *CONTRASTIVE_REPREDICT_VARIANTS,
)
JUDGE_CONTEXT_KEYS = (
    "judge_profile",
    "judge_pass",
    "failed_state",
    "first_bad_state_step",
    "judge_reason",
    "wrong_action_reason",
    "step_alignment_reason",
    "failure_mode_for_skill",
    "can_extract_skill",
    "skill_signal_type",
    "first_deviation_reason",
)
SKILL_SYSTEM_MARKER = "# Skill execution mode"
SKILL_USER_MARKERS = (
    "[已召回 Skill]",
    "[Skill 类型执行规则]",
    "[Skill 文本]",
    "[按当前截图与样例原始图相似度选出的 Top 1 Skill Sample]",
    "[按当前截图与样例原始图相似度选出的 Top 3 Skill Samples]",
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def split_selector(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_selector(value: Optional[str], choices: Iterable[str], label: str) -> Tuple[str, ...]:
    valid = tuple(choices)
    if value is None or not value.strip() or value.strip().lower() == "all":
        return valid
    selected = split_selector(value)
    unknown = [item for item in selected if item not in valid]
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}; valid values: all,{','.join(valid)}")
    return tuple(dict.fromkeys(selected))


def family_enabled(args: argparse.Namespace, family: str) -> bool:
    return family in args.repredict_families


def variant_enabled(args: argparse.Namespace, variant: str) -> bool:
    return variant in args.repredict_variants


def enabled_variants(args: argparse.Namespace, variants: Iterable[str]) -> Tuple[str, ...]:
    return tuple(variant for variant in variants if variant_enabled(args, variant))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=None if indent == 0 else indent)
        f.write("\n")


def normalize_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        items = payload["results"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise ValueError("classification JSON must be an object, list, or object with results")
    return [item for item in items if isinstance(item, dict)]


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def judge_context_fields(judge: Dict[str, Any]) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for key in JUDGE_CONTEXT_KEYS:
        value = judge.get(key)
        if value is None or value == "":
            continue
        context[key] = value if isinstance(value, bool) else compact(value)
    return context


def category_id(item: Dict[str, Any]) -> Optional[int]:
    classification = item.get("classification")
    if not isinstance(classification, dict):
        return None
    value = classification.get("category_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def wrong_action_step_index(item: Dict[str, Any]) -> Optional[int]:
    judge = item.get("judge") if isinstance(item.get("judge"), dict) else item
    value = judge.get("wrong_action_step") if isinstance(judge, dict) else None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def wrong_action_step_info(item: Dict[str, Any]) -> Dict[str, Any]:
    judge = item.get("judge") if isinstance(item.get("judge"), dict) else item
    step = judge.get("wrong_action_step_record") if isinstance(judge, dict) else None
    return step if isinstance(step, dict) else {}


def judge_failure_reason(item: Dict[str, Any]) -> str:
    judge = item.get("judge") if isinstance(item.get("judge"), dict) else {}
    classification = item.get("classification") if isinstance(item.get("classification"), dict) else {}
    for key in (
        "wrong_action_reason",
        "step_alignment_reason",
        "failure_mode_for_skill",
        "first_deviation_reason",
    ):
        value = compact(judge.get(key))
        if value:
            return value
    return compact(classification.get("reason"))


def resolve_source_image(item: Dict[str, Any]) -> Path:
    step = wrong_action_step_info(item)
    raw = step.get("image_path")
    if isinstance(raw, str) and raw.strip():
        return Path(raw.strip())
    classification = item.get("classification")
    if isinstance(classification, dict):
        raw = classification.get("wrong_action_step_image")
        if isinstance(raw, str) and raw.strip():
            return Path(raw.strip())
    raise ValueError("previous classification JSON is missing wrong action step image path")


def prompt_messages_from_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    step = wrong_action_step_info(item)
    messages = step.get("prompt_messages")
    if not isinstance(messages, list):
        raise ValueError("previous classification JSON is missing wrong_action_step_record.prompt_messages")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("wrong_action_step_record.prompt_messages must be a list of message objects")
    return strip_skill_content_from_messages(messages)


def text_item_text(item: Any) -> str:
    if not isinstance(item, dict) or item.get("type") != "text":
        return ""
    value = item.get("text")
    return value if isinstance(value, str) else ""


def image_url_item(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict) or item.get("type") != "image_url":
        return None
    image_url = item.get("image_url")
    return image_url if isinstance(image_url, dict) else None


def contains_skill_marker(value: Any) -> bool:
    if isinstance(value, str):
        return SKILL_SYSTEM_MARKER in value or any(marker in value for marker in SKILL_USER_MARKERS)
    if isinstance(value, list):
        return any(contains_skill_marker(item.get("text")) for item in value if isinstance(item, dict))
    return False


def strip_skill_system_prompt(content: Any) -> Any:
    if not isinstance(content, str) or SKILL_SYSTEM_MARKER not in content:
        return content
    return content.split(SKILL_SYSTEM_MARKER, 1)[0].rstrip()


def strip_skill_user_text(text: str) -> str:
    cleaned = text
    marker_positions = [cleaned.find(marker) for marker in SKILL_USER_MARKERS if marker in cleaned]
    if marker_positions:
        cleaned = cleaned[: min(marker_positions)].rstrip()
    if "[当前截图]" not in cleaned:
        cleaned = cleaned.rstrip() + "\n\n[当前截图]\n"
    return cleaned


def strip_skill_user_content(content: Any) -> Any:
    if not isinstance(content, list) or not contains_skill_marker(content):
        return content

    first_text = ""
    first_image_url: Optional[Dict[str, Any]] = None
    for item in content:
        if not first_text:
            first_text = text_item_text(item)
        if first_image_url is None:
            first_image_url = image_url_item(item)

    if first_image_url is None:
        return [{"type": "text", "text": strip_skill_user_text(first_text)}]
    return [
        {"type": "text", "text": strip_skill_user_text(first_text)},
        {"type": "image_url", "image_url": copy.deepcopy(first_image_url)},
    ]


def strip_skill_content_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stripped_messages = copy.deepcopy(messages)
    for message in stripped_messages:
        if message.get("role") == "system":
            message["content"] = strip_skill_system_prompt(message.get("content"))
        elif message.get("role") == "user":
            message["content"] = strip_skill_user_content(message.get("content"))
    return stripped_messages


def create_edited_images(
    image_path: Path,
    output_dir: Path,
    variants: Iterable[str],
    seed: Optional[int],
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed) if seed is not None else None
    outputs: Dict[str, Path] = {}
    for variant in variants:
        output_path = output_dir / f"{image_path.stem}_{variant}.png"
        if variant == "crop":
            random_small_crop(image_path, output_path=output_path, rng=rng)
        elif variant == "corner_occlusion":
            random_black_corner_occlusion(image_path, output_path=output_path, rng=rng)
        else:
            raise ValueError(f"unknown edit variant: {variant}")
        outputs[variant] = output_path
    return outputs


def extract_first_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    in_str = False
    esc = False
    depth = 0
    for index in range(start, len(text)):
        ch = text[index]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_action_json(response: str) -> Optional[Dict[str, Any]]:
    raw = str(response or "").strip()
    if not raw:
        return None
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL | re.IGNORECASE)
    candidate = answer_match.group(1).strip() if answer_match else raw
    for text in (candidate, extract_first_json(candidate), extract_first_json(raw)):
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def image_size(path: Path) -> Tuple[int, int]:
    if Image is None:
        raise RuntimeError("PIL is required for SOM coordinate conversion")
    with Image.open(path) as img:
        return img.size


def as_point(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except Exception:
                return None
    return None


def point_to_pixel(point: Tuple[float, float], size: Tuple[int, int]) -> Tuple[int, int]:
    width, height = size
    x, y = point
    px = int(round(x / 1000.0 * width))
    py = int(round(y / 1000.0 * height))
    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    return px, py


def som_action_from_mobilegym_response(response: str, image_path: Path) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    action_json = extract_action_json(response)
    if not action_json:
        return None, None, "no_action_json"

    action_name = str(action_json.get("action") or action_json.get("action_type") or "").strip().upper()
    if not action_name:
        return None, action_json, "missing_action"

    size = image_size(image_path)
    if action_name in {"CLICK", "TAP", "DOUBLE_TAP", "DOUBLETAP"}:
        point = as_point(action_json.get("point") or action_json.get("coordinate") or action_json.get("element"))
        if not point:
            return None, action_json, "missing_point"
        x, y = point_to_pixel(point, size)
        return f"tap({x}, {y})", action_json, "ok"

    if action_name in {"LONGPRESS", "LONG_PRESS", "LONG PRESS"}:
        point = as_point(action_json.get("point") or action_json.get("coordinate") or action_json.get("element"))
        if not point:
            return None, action_json, "missing_point"
        x, y = point_to_pixel(point, size)
        return f"long_press({x}, {y})", action_json, "ok"

    if action_name in {"TYPE", "TEXT"}:
        point = as_point(action_json.get("point") or action_json.get("coordinate"))
        if not point:
            return None, action_json, "action_has_no_point"
        x, y = point_to_pixel(point, size)
        return f"text({x}, {y})", action_json, "ok"

    if action_name in {"SWIPE", "SLIDE", "DRAG"}:
        start = as_point(
            action_json.get("point1")
            or action_json.get("start")
            or action_json.get("start_point")
            or action_json.get("start_coordinate")
        )
        end = as_point(
            action_json.get("point2")
            or action_json.get("end")
            or action_json.get("end_point")
            or action_json.get("end_coordinate")
        )
        if start and end:
            x1, y1 = point_to_pixel(start, size)
            x2, y2 = point_to_pixel(end, size)
            return f"drag({x1}, {y1}, {x2}, {y2})", action_json, "ok"
        point = as_point(action_json.get("point") or action_json.get("coordinate"))
        direction = action_json.get("direction")
        if point and direction:
            x, y = point_to_pixel(point, size)
            return f"scroll({x}, {y}, {json.dumps(str(direction))})", action_json, "ok"
        return None, action_json, "missing_swipe_points"

    return None, action_json, "action_has_no_coordinate"


def make_som_image(response: str, image_path: Path, output_path: Path) -> Dict[str, Any]:
    try:
        som_action, action_json, status = som_action_from_mobilegym_response(response, image_path)
    except Exception as exc:
        return {
            "som_action": None,
            "som_status": f"som_prepare_failed: {exc}",
            "action_json": None,
        }
    result: Dict[str, Any] = {
        "som_action": som_action,
        "som_status": status,
        "action_json": action_json,
    }
    if not som_action:
        return result
    marked = make_marked_image(som_action, str(image_path), str(output_path))
    if marked:
        result["som_image"] = str(Path(marked).resolve())
        result["som_status"] = "ok"
    else:
        result["som_status"] = "som_render_failed"
    return result


def create_wrong_action_marked_image(step: Dict[str, Any], source_image: Path, output_path: Path) -> Dict[str, Any]:
    response = compact(step.get("response_text"))
    if not response and isinstance(step.get("action_json"), dict):
        response = json.dumps(step["action_json"], ensure_ascii=False)
    result = make_som_image(response, source_image, output_path)
    result["variant"] = WRONG_ACTION_MARKED_VARIANT
    return result


def wrong_action_json(step: Dict[str, Any], marked_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    original_action = step.get("action_json") if isinstance(step.get("action_json"), dict) else None
    if original_action is None:
        original_action = marked_info.get("action_json") if isinstance(marked_info.get("action_json"), dict) else None
    if original_action is None:
        original_action = extract_action_json(compact(step.get("response_text")))
    return original_action


def wrong_action_marked_prompt(item: Dict[str, Any], step_no: int, step: Dict[str, Any], marked_info: Dict[str, Any]) -> str:
    action_text = json.dumps(wrong_action_json(step, marked_info) or {}, ensure_ascii=False)
    reason = judge_failure_reason(item) or "(missing)"
    return "\n".join(
        [
            "任务 instruction:",
            compact(item.get("instruction")) or "(missing)",
            "",
            f"错误步骤: step {step_no}",
            "",
            "你会看到两张图:",
            "1. 原截图: 错误步骤发生时的真实手机截图，用它判断新的下一步动作。",
            "2. 错误动作 SOM 图: 仅用于参考原失败轨迹做了什么；图上的标注表示错误动作的位置或轨迹。这个标注不是推荐动作。",
            "",
            "错误动作 JSON:",
            action_text,
            "错误动作 SOM 信息:",
            compact(marked_info.get("som_action")) or "(missing)",
            "错误原因:",
            reason,
            "",
            "请基于原截图重新输出该步骤应该执行的动作决策。输出动作必须与错误动作 JSON 以及错误动作 SOM 图中标注的动作不一致。",
            "如果错误动作是点击某个位置，新的 CLICK 不能落在同一控件、同一目标区域或该标注点附近。",
            "如果错误动作是 SWIPE/DRAG，新的动作不能使用相同的起止区域或方向。",
            "如果错误动作是 TYPE，新的动作不能继续使用同样的直接输入方式。",
            "",
            "坐标系：左上角为原点，x 向右，y 向下，取值范围均为 0-1000（归一化坐标）。",
            "",
            "可用动作（JSON 格式）：",
            "",
            "1. 点击：{\"action\": \"CLICK\", \"point\": [x, y]}",
            "2. 双击：{\"action\": \"DOUBLE_TAP\", \"point\": [x, y]}",
            "3. 长按：{\"action\": \"LONGPRESS\", \"point\": [x, y]}",
            "4. 输入：{\"action\": \"TYPE\", \"value\": \"文本内容\"}  // 可选 \"point\": [x, y] 指定输入位置；可选 \"clear\": true 先清空输入框再输入（默认追加到已有文本后面）",
            "5. 滑动：{\"action\": \"SWIPE\", \"point1\": [x1, y1], \"point2\": [x2, y2]}",
            "6. 拖拽：{\"action\": \"DRAG\", \"point1\": [x1, y1], \"point2\": [x2, y2]}  // 按住起点拖动到终点",
            "7. 返回：{\"action\": \"BACK\"}",
            "8. 输入回车：{\"action\": \"ENTER\"}",
            "9. 提交答案：{\"action\": \"ANSWER\", \"value\": \"纯答案文本\"}",
            "",
            "输出格式:",
            "<THINK>",
            "在这里描述你对当前屏幕的理解、分析和决策过程。",
            "包括：",
            "1. 当前屏幕显示的内容是什么",
            "2. 为了完成任务，下一步应该做什么",
            "3. 具体要点击/操作哪个元素",
            "4. 为什么该动作不同于错误动作",
            "</THINK>",
            "<ANSWER>",
            "{",
            "  \"action\": \"动作类型\"",
            "  // 根据动作类型填写相应参数",
            "}",
            "</ANSWER>",
            "",
            "要求:",
            "- 坐标必须为数字，范围 0-1000",
            "- JSON 必须是有效格式",
            "- 仔细观察原截图，根据视觉信息做出判断",
            "- 需要回答问题时，必须使用 ANSWER 提交答案",
        ]
    )


def wrong_action_marked_messages(
    source_image: Path,
    marked_image: Path,
    item: Dict[str, Any],
    step_no: int,
    step: Dict[str, Any],
    marked_info: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    prompt = wrong_action_marked_prompt(item, step_no, step, marked_info)
    messages = [
        {
            "role": "system",
            "content": "你是一个手机 GUI-Agent。根据给定截图和错误动作信息，输出一个新的单步动作决策。",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "原截图:"},
                {"type": "image_url", "image_url": {"url": str(source_image.resolve())}},
                {"type": "text", "text": "错误动作 SOM 图:"},
                {"type": "image_url", "image_url": {"url": str(marked_image.resolve())}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    return messages, prompt


def duplicate_unsom_as_som(image_path: Path, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        with Image.open(image_path) as img:
            img.convert("RGB").save(output_path, format="JPEG", quality=88, optimize=True)
    else:
        shutil.copyfile(image_path, output_path)
    return str(output_path.resolve())


def pixel_to_normalized(point: Tuple[int, int], size: Tuple[int, int]) -> Tuple[int, int]:
    width, height = size
    x, y = point
    norm_x = int(round(x / max(1, width) * 1000.0))
    norm_y = int(round(y / max(1, height) * 1000.0))
    norm_x = max(0, min(1000, norm_x))
    norm_y = max(0, min(1000, norm_y))
    return norm_x, norm_y


def bbox_to_normalized(bbox: List[int], size: Tuple[int, int]) -> List[int]:
    x1, y1, x2, y2 = bbox
    nx1, ny1 = pixel_to_normalized((x1, y1), size)
    nx2, ny2 = pixel_to_normalized((x2, y2), size)
    return [nx1, ny1, nx2, ny2]


def clamp_bbox(bbox: Any, size: Tuple[int, int]) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"bbox must be [x1, y1, x2, y2], got: {bbox}")
    width, height = size
    values = [int(round(float(value))) for value in bbox]
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def validate_hint_bbox(bbox: List[int], size: Tuple[int, int]) -> None:
    width, height = size
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    image_area = max(1, width * height)
    box_area = box_width * box_height
    if box_width < 5 or box_height < 5:
        raise ValueError(f"hint bbox is too small: {bbox}")
    if box_area / image_area > 0.70:
        raise ValueError(f"hint bbox is too large: {bbox}")


def bbox_area(bbox: List[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def normalize_evidence_list(value: Any, size: Tuple[int, int]) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = [item for item in value if isinstance(item, dict)]
    else:
        raw_items = []

    evidences: List[Dict[str, Any]] = []
    for item in raw_items:
        raw_bbox = item.get("bbox")
        if raw_bbox is None and isinstance(item.get("top_left"), (list, tuple)) and isinstance(item.get("bottom_right"), (list, tuple)):
            top_left = item["top_left"]
            bottom_right = item["bottom_right"]
            raw_bbox = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        if raw_bbox is None:
            continue
        try:
            bbox = clamp_bbox(raw_bbox, size)
            validate_hint_bbox(bbox, size)
        except Exception:
            continue
        normalized = dict(item)
        normalized["bbox"] = bbox
        normalized["normalized_bbox"] = bbox_to_normalized(bbox, size)
        normalized["label"] = compact(normalized.get("label")) or "evidence"
        normalized["reason"] = compact(normalized.get("reason") or normalized.get("description"))
        evidences.append(normalized)
    return evidences


def primary_evidence_bbox(evidences: List[Dict[str, Any]]) -> List[int]:
    if not evidences:
        raise ValueError("contrastive hint is missing wrong_evidence bbox")
    return max((evidence["bbox"] for evidence in evidences), key=bbox_area)


def render_red_rectangle_image(source_image: Path, output_path: Path, bbox: List[int]) -> str:
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL is required to render redbox image")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image) as img:
        rendered = img.convert("RGB")
        draw = ImageDraw.Draw(rendered)
        width, height = rendered.size
        stroke_width = max(3, min(width, height) // 160)
        x1, y1, x2, y2 = bbox
        for offset in range(stroke_width):
            draw.rectangle(
                [x1 + offset, y1 + offset, max(x1, x2 - 1 - offset), max(y1, y2 - 1 - offset)],
                outline=(255, 0, 0),
            )
        rendered.save(output_path, format="JPEG", quality=88, optimize=True)
    return str(output_path.resolve())


def render_contrastive_correction_image(
    source_image: Path,
    output_path: Path,
    wrong_evidence: List[Dict[str, Any]],
    correct_evidence: List[Dict[str, Any]],
) -> str:
    if Image is None or ImageDraw is None:
        raise RuntimeError("PIL is required to render contrastive correction image")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image) as img:
        rendered = img.convert("RGB")
        draw = ImageDraw.Draw(rendered)
        width, height = rendered.size
        stroke_width = max(3, min(width, height) // 160)

        def draw_boxes(evidences: List[Dict[str, Any]], color: Tuple[int, int, int]) -> None:
            for evidence in evidences:
                x1, y1, x2, y2 = evidence["bbox"]
                for offset in range(stroke_width):
                    draw.rectangle(
                        [x1 + offset, y1 + offset, max(x1, x2 - 1 - offset), max(y1, y2 - 1 - offset)],
                        outline=color,
                    )

        draw_boxes(wrong_evidence, (255, 0, 0))
        draw_boxes(correct_evidence, (0, 180, 0))
        rendered.save(output_path, format="JPEG", quality=88, optimize=True)
    return str(output_path.resolve())


def build_pitfall_context_lines(item: Dict[str, Any], step_no: int) -> List[str]:
    classification = item.get("classification") if isinstance(item.get("classification"), dict) else {}
    step = wrong_action_step_info(item)
    original_action = step.get("action_json") if isinstance(step.get("action_json"), dict) else None
    if original_action is None:
        original_action = extract_action_json(compact(step.get("response_text")))
    return [
        "Task instruction:",
        compact(item.get("instruction")) or "(missing)",
        "",
        f"Failed step index: {step_no}",
        "",
        "Original failed action JSON:",
        json.dumps(original_action or {}, ensure_ascii=False),
        "",
        "Failure category:",
        f"{compact(classification.get('category_id'))} {compact(classification.get('category_name'))}".strip() or "(missing)",
        "",
        "Optional failure reason:",
        judge_failure_reason(item) or "(missing)",
    ]


def build_pitfall_llm_prompt(item: Dict[str, Any], step_no: int, image_size_value: Tuple[int, int]) -> str:
    width, height = image_size_value
    return "\n".join(
        [
            "You are a strict mobile GUI failure analyst.",
            "",
            *build_pitfall_context_lines(item, step_no),
            "",
            "The input image is the original screenshot at the failed step.",
            "Your job is to build a contrastive correction hint for category 6-10 failures.",
            "Identify what visible evidence caused the original mistake, and what visible evidence or operation target should correct it.",
            "",
            "Category-specific guidance:",
            "- Category 6: red/wrong_evidence should cover the wrongly selected image/video item or misleading media-type cue; green/correct_evidence should cover the correct media item or the visual cue that distinguishes image from video.",
            "- Category 7: red/wrong_evidence should cover the live-stream entry or live cue; green/correct_evidence should cover the product entry or product cue.",
            "- Category 8: red/wrong_evidence should cover the ad/promoted target or ad button; green/correct_evidence should cover the real target or the ad-label cue that proves it is an ad.",
            "- Category 9: red/wrong_evidence should cover the ineffective type/input context if visible; green/correct_evidence should cover the custom keyboard, keypad, candidate panel, or clickable input controls that must be used instead.",
            "- Category 10: red/wrong_evidence should cover the repeated scroll area or stale content region; green/correct_evidence should cover the boundary cue such as top/bottom/end-of-list/no-more-content/unchanged terminal item.",
            "",
            "Return exactly one JSON object. Do not output Markdown. Do not include extra text.",
            "All bboxes must use pixel coordinates in the original image coordinate system.",
            f"Image size: width={width}, height={height}.",
            "Use [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.",
            "Each bbox should tightly cover one visible evidence region, not the whole screen.",
            "wrong_evidence must contain at least one bbox. correct_evidence should contain at least one bbox when a visible correction cue exists; otherwise use an empty list and explain in action_hint.reason.",
            "",
            "Schema:",
            "{\"hint_type\": \"contrastive_correction\", \"wrong_evidence\": [{\"bbox\": [x1, y1, x2, y2], \"label\": \"wrong_target|wrong_action_area|wrong_state_cue\", \"reason\": \"short Chinese reason\"}], \"correct_evidence\": [{\"bbox\": [x1, y1, x2, y2], \"label\": \"correct_target_or_cue|keyboard|boundary_cue\", \"reason\": \"short Chinese reason\"}], \"action_hint\": {\"preferred_action_type\": \"click|click_key_by_key|stop_scrolling_or_change_strategy|inspect_cue\", \"wrong_action_type\": \"type|repeat_same_scroll|wrong_target_click|null\", \"reason\": \"short Chinese reason\"}, \"reason\": \"short Chinese summary\"}",
        ]
    )


def extract_contrastive_hint_json(response: str, size: Tuple[int, int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], str]:
    parsed = extract_action_json(response)
    if not isinstance(parsed, dict):
        raise ValueError("contrastive correction LLM response does not contain a JSON object")

    wrong_evidence = normalize_evidence_list(parsed.get("wrong_evidence") or parsed.get("wrong_regions"), size)
    correct_evidence = normalize_evidence_list(parsed.get("correct_evidence") or parsed.get("correct_regions"), size)

    if not wrong_evidence:
        legacy_bbox = parsed.get("bbox")
        if legacy_bbox is not None:
            wrong_evidence = normalize_evidence_list(
                [{"bbox": legacy_bbox, "label": "wrong_target", "reason": parsed.get("reason")}],
                size,
            )
    if not wrong_evidence:
        raise ValueError(f"contrastive correction response missing wrong_evidence bbox: {parsed}")

    action_hint = parsed.get("action_hint") if isinstance(parsed.get("action_hint"), dict) else {}
    action_hint = dict(action_hint)
    action_hint["preferred_action_type"] = compact(action_hint.get("preferred_action_type")) or "inspect_cue"
    wrong_action_type = action_hint.get("wrong_action_type")
    action_hint["wrong_action_type"] = None if wrong_action_type is None else compact(wrong_action_type)
    action_hint["reason"] = compact(action_hint.get("reason"))
    reason = compact(parsed.get("reason") or parsed.get("description"))
    return wrong_evidence, correct_evidence, action_hint, reason


def build_single_image_messages(query: str, image_path: Path, system_prompt: str) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": str(image_path.resolve())}},
                {"type": "text", "text": query},
            ],
        },
    ]


def run_pitfall_llm_variant_or_error(
    task_id: str,
    step_no: int,
    variant: str,
    source_image: Path,
    item: Dict[str, Any],
    item_output_dir: Path,
    wrapper: ApiWrapper,
) -> Dict[str, Any]:
    try:
        item_output_dir.mkdir(parents=True, exist_ok=True)
        size = image_size(source_image)
        prompt = build_pitfall_llm_prompt(item, step_no, size)
        prompt_path = item_output_dir / f"prompt_{variant}.txt"
        response_path = item_output_dir / f"response_{variant}.txt"
        hint_image_path = item_output_dir / f"{source_image.stem}_{variant}_hint.jpg"
        prompt_path.write_text(prompt, encoding="utf-8")
        messages = build_single_image_messages(
            query=prompt,
            image_path=source_image,
            system_prompt="Return only valid JSON for the requested contrastive correction hint.",
        )
        response = wrapper.chat_messages(messages)
        response_path.write_text(response, encoding="utf-8")
        wrong_evidence, correct_evidence, action_hint, reason = extract_contrastive_hint_json(response, size)
        hint_image = render_contrastive_correction_image(source_image, hint_image_path, wrong_evidence, correct_evidence)
        return {
            "hint_image": hint_image,
            "hint_type": "contrastive_correction",
            "source": variant,
            "image_size": [size[0], size[1]],
            "wrong_evidence": wrong_evidence,
            "correct_evidence": correct_evidence,
            "action_hint": action_hint,
            "reason": reason,
            "prompt_path": str(prompt_path.resolve()),
        }
    except Exception as exc:
        log(f"[variant_error] task_id={task_id} step={step_no} variant={variant} {exc}")
        return {
            "hint_type": "contrastive_correction",
            "source": variant,
            "error": str(exc),
        }


def replace_current_step_image(messages: List[Dict[str, Any]], replacement_image_path: Path) -> List[Dict[str, Any]]:
    replaced_messages = copy.deepcopy(messages)
    image_items: List[Dict[str, Any]] = []
    for message in replaced_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_items.append(image_url)

    if not image_items:
        raise ValueError("no image_url item found in step prompt")
    image_items[-1]["url"] = str(replacement_image_path.resolve())
    return replaced_messages


def build_wrapper(args: argparse.Namespace) -> ApiWrapper:
    return ApiWrapper(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        model_name=args.model_name,
        base_url=args.model_base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        debug=args.debug,
        max_tokens=args.max_tokens,
        verify=False,
        trust_env=False,
    )


def build_original_api_wrappers(args: argparse.Namespace) -> Dict[str, ApiWrapper]:
    return {
        "original_doubao": ApiWrapper(
            model_name="DouBao-Seed-2.0-Pro",
            timeout=args.timeout,
            max_retries=args.max_retries,
            debug=args.debug,
        ),
        "original_qwen_plus": ApiWrapper(
            model_name="Qwen3.6-Plus",
            api_key="",
            timeout=args.timeout,
            max_retries=args.max_retries,
            debug=args.debug,
            max_tokens=args.max_tokens,
        ),
    }


def build_pitfall_reprediction_wrappers(
    default_wrapper: ApiWrapper,
    original_api_wrappers: Dict[str, ApiWrapper],
) -> Dict[str, ApiWrapper]:
    return {
        "default": default_wrapper,
        "doubao": original_api_wrappers["original_doubao"],
        "qwen_plus": original_api_wrappers["original_qwen_plus"],
    }


def same_model_reprediction_wrappers(
    hint_variant: str,
    repredict_wrappers: Dict[str, ApiWrapper],
) -> Dict[str, ApiWrapper]:
    if hint_variant.endswith("_doubao"):
        return {"doubao": repredict_wrappers["doubao"]}
    if hint_variant.endswith("_qwen_plus"):
        return {"qwen_plus": repredict_wrappers["qwen_plus"]}
    raise ValueError(f"cannot match hint variant to reprediction wrapper: {hint_variant}")


def build_pitfall_api_wrappers(args: argparse.Namespace) -> Dict[str, ApiWrapper]:
    return {
        "contrastive_correction_doubao": ApiWrapper(
            model_name="DouBao-Seed-2.0-Pro",
            timeout=args.timeout,
            max_retries=args.max_retries,
            debug=args.debug,
        ),
        "contrastive_correction_qwen_plus": ApiWrapper(
            model_name="Qwen3.6-Plus",
            api_key="",
            timeout=args.timeout,
            max_retries=args.max_retries,
            debug=args.debug,
            max_tokens=args.max_tokens,
        ),
    }


def run_reprediction_variant(
    task_id: str,
    step_no: int,
    variant: str,
    image_path: Path,
    messages: List[Dict[str, Any]],
    item_output_dir: Path,
    args: argparse.Namespace,
    wrapper: ApiWrapper,
    image_replaced: bool,
    som_base_image_path: Optional[Path] = None,
) -> Dict[str, Any]:
    log(f"[predict] task_id={task_id} step={step_no} variant={variant}")
    item_output_dir.mkdir(parents=True, exist_ok=True)
    response = wrapper.chat_messages(messages)
    response_path = item_output_dir / f"response_{variant}.txt"
    response_path.write_text(response, encoding="utf-8")
    som_base_image = som_base_image_path or image_path
    som_image_path = item_output_dir / f"{som_base_image.stem}_{variant}_som.jpg"
    som_info = make_som_image(response, som_base_image, som_image_path)
    if not som_info.get("som_image"):
        som_info["som_image"] = duplicate_unsom_as_som(som_base_image, som_image_path)
    result = {
        "unsom_image": str(image_path.resolve()),
        "som_base_image": str(som_base_image.resolve()),
        "som_image": som_info.get("som_image"),
        "som_status": som_info.get("som_status"),
        "som_action": som_info.get("som_action"),
        "action_json": som_info.get("action_json"),
        "response": response,
        "response_path": str(response_path.resolve()),
        "api_wrapper": wrapper.__class__.__name__,
        "image_replaced": image_replaced,
    }
    return result


def run_reprediction_variant_or_error(
    task_id: str,
    step_no: int,
    variant: str,
    image_path: Path,
    messages: List[Dict[str, Any]],
    item_output_dir: Path,
    args: argparse.Namespace,
    wrapper: ApiWrapper,
    image_replaced: bool,
    som_base_image_path: Optional[Path] = None,
) -> Dict[str, Any]:
    try:
        return run_reprediction_variant(
            task_id=task_id,
            step_no=step_no,
            variant=variant,
            image_path=image_path,
            messages=messages,
            item_output_dir=item_output_dir,
            args=args,
            wrapper=wrapper,
            image_replaced=image_replaced,
            som_base_image_path=som_base_image_path,
        )
    except Exception as exc:
        log(f"[variant_error] task_id={task_id} step={step_no} variant={variant} {exc}")
        som_base_image = som_base_image_path or image_path
        result = {
            "unsom_image": str(image_path.resolve()),
            "som_base_image": str(som_base_image.resolve()),
            "som_image": "",
            "som_status": "skipped_after_prediction_error",
            "som_action": None,
            "action_json": None,
            "response": "",
            "response_path": "",
            "api_wrapper": wrapper.__class__.__name__,
            "image_replaced": image_replaced,
            "error": str(exc),
        }
        return result


def run_pitfall_hint_reprediction_variants(
    task_id: str,
    step_no: int,
    hint_variant: str,
    source_image: Path,
    item: Dict[str, Any],
    prompt_messages: List[Dict[str, Any]],
    item_output_dir: Path,
    hint_wrapper: ApiWrapper,
    repredict_wrappers: Dict[str, ApiWrapper],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    hint_info = run_pitfall_llm_variant_or_error(
        task_id=task_id,
        step_no=step_no,
        variant=hint_variant,
        source_image=source_image,
        item=item,
        item_output_dir=item_output_dir,
        wrapper=hint_wrapper,
    )
    if hint_info.get("error"):
        return {
            f"{hint_variant}_hint_failed": {
                "unsom_image": "",
                "som_image": "",
                "som_status": "skipped_after_hint_generation_error",
                "som_action": None,
                "action_json": None,
                "response": "",
                "response_path": "",
                "api_wrapper": hint_wrapper.__class__.__name__,
                "image_replaced": True,
                "hint_source": hint_variant,
                "hint_type": hint_info.get("hint_type"),
                "error": hint_info.get("error"),
            }
        }

    hint_image = Path(str(hint_info["hint_image"]))
    outputs: Dict[str, Dict[str, Any]] = {}
    for action_variant, action_wrapper in repredict_wrappers.items():
        variant = f"{hint_variant}_{action_variant}"
        messages = replace_current_step_image(prompt_messages, hint_image)
        result = run_reprediction_variant_or_error(
            task_id=task_id,
            step_no=step_no,
            variant=variant,
            image_path=hint_image,
            messages=messages,
            item_output_dir=item_output_dir,
            args=args,
            wrapper=action_wrapper,
            image_replaced=True,
            som_base_image_path=source_image,
        )
        result.update(
            {
                "hint_source": hint_variant,
                "hint_image": str(hint_image.resolve()),
                "hint_type": hint_info.get("hint_type"),
                "hint_image_size": hint_info.get("image_size"),
                "wrong_evidence": hint_info.get("wrong_evidence"),
                "correct_evidence": hint_info.get("correct_evidence"),
                "hint_action_hint": hint_info.get("action_hint"),
                "hint_reason": hint_info.get("reason"),
                "hint_prompt_path": hint_info.get("prompt_path"),
            }
        )
        outputs[variant] = result
    return outputs


def process_one(
    item: Dict[str, Any],
    index: int,
    args: argparse.Namespace,
    wrapper: ApiWrapper,
    original_api_wrappers: Dict[str, ApiWrapper],
    pitfall_api_wrappers: Dict[str, ApiWrapper],
    pitfall_reprediction_wrappers: Dict[str, ApiWrapper],
) -> Dict[str, Any]:
    task_id = compact(item.get("task_id")) or f"item_{index}"
    cat_id = category_id(item)
    judge = item.get("judge") if isinstance(item.get("judge"), dict) else {}
    step = wrong_action_step_info(item)
    wrong_action_step = wrong_action_step_index(item)
    if wrong_action_step is None:
        return {
            "task_id": task_id,
            "category_id": cat_id,
            "error": "cannot resolve wrong_action_step from previous classification JSON",
        }
    step_no = wrong_action_step

    source_image = resolve_source_image(item)
    item_output_dir = args.results_dir / task_id / f"step_{step_no:03d}"

    variants: Dict[str, Dict[str, Any]] = {}
    if cat_id in AGENT_CATEGORY_IDS:
        prompt_messages = prompt_messages_from_item(item)
        edit_variants = enabled_variants(args, EDIT_VARIANTS) if family_enabled(args, "edit") else ()
        if edit_variants:
            edited_images = create_edited_images(source_image, item_output_dir, edit_variants, args.seed)
            for variant, edited_image in edited_images.items():
                messages = replace_current_step_image(prompt_messages, edited_image)
                variants[variant] = run_reprediction_variant_or_error(
                    task_id=task_id,
                    step_no=step_no,
                    variant=variant,
                    image_path=edited_image,
                    messages=messages,
                    item_output_dir=item_output_dir,
                    args=args,
                    wrapper=wrapper,
                    image_replaced=True,
                )

        if family_enabled(args, "wrong_action_marked") and variant_enabled(args, WRONG_ACTION_MARKED_VARIANT):
            marked_image = item_output_dir / f"{source_image.stem}_{WRONG_ACTION_MARKED_VARIANT}.jpg"
            marked_info = create_wrong_action_marked_image(step, source_image, marked_image)
            if marked_info.get("som_image"):
                messages, marked_instruction = wrong_action_marked_messages(
                    source_image, marked_image, item, step_no, step, marked_info
                )
                variants[WRONG_ACTION_MARKED_VARIANT] = run_reprediction_variant_or_error(
                    task_id=task_id,
                    step_no=step_no,
                    variant=WRONG_ACTION_MARKED_VARIANT,
                    image_path=marked_image,
                    messages=messages,
                    item_output_dir=item_output_dir,
                    args=args,
                    wrapper=wrapper,
                    image_replaced=True,
                    som_base_image_path=source_image,
                )
                variants[WRONG_ACTION_MARKED_VARIANT]["source_action_mark"] = marked_info
                variants[WRONG_ACTION_MARKED_VARIANT]["wrong_action_marked_instruction"] = marked_instruction
            else:
                variants[WRONG_ACTION_MARKED_VARIANT] = {
                    "unsom_image": str(marked_image.resolve()),
                    "som_image": "",
                    "som_status": marked_info.get("som_status") or "source_action_mark_failed",
                    "som_action": marked_info.get("som_action"),
                    "action_json": marked_info.get("action_json"),
                    "response": "",
                    "response_path": "",
                    "api_wrapper": wrapper.__class__.__name__,
                    "image_replaced": True,
                    "source_action_mark": marked_info,
                    "error": marked_info.get("som_status") or "source_action_mark_failed",
                }

        if family_enabled(args, "original_api"):
            for variant, api_wrapper in original_api_wrappers.items():
                if not variant_enabled(args, variant):
                    continue
                messages = replace_current_step_image(prompt_messages, source_image)
                variants[variant] = run_reprediction_variant_or_error(
                    task_id=task_id,
                    step_no=step_no,
                    variant=variant,
                    image_path=source_image,
                    messages=messages,
                    item_output_dir=item_output_dir,
                    args=args,
                    wrapper=api_wrapper,
                    image_replaced=False,
                )

    if cat_id in PITFALL_CATEGORY_IDS:
        prompt_messages = prompt_messages_from_item(item)
        pitfall_mark_variants = (
            enabled_variants(args, PITFALL_WRONG_ACTION_MARKED_VARIANTS)
            if family_enabled(args, "wrong_action_marked")
            else ()
        )
        if pitfall_mark_variants:
            marked_image = item_output_dir / f"{source_image.stem}_{WRONG_ACTION_MARKED_VARIANT}.jpg"
            marked_info = create_wrong_action_marked_image(step, source_image, marked_image)
            if marked_info.get("som_image"):
                messages, marked_instruction = wrong_action_marked_messages(
                    source_image, marked_image, item, step_no, step, marked_info
                )
                for action_variant, action_wrapper in pitfall_reprediction_wrappers.items():
                    variant = f"{WRONG_ACTION_MARKED_VARIANT}_{action_variant}"
                    if variant not in pitfall_mark_variants:
                        continue
                    variants[variant] = run_reprediction_variant_or_error(
                        task_id=task_id,
                        step_no=step_no,
                        variant=variant,
                        image_path=marked_image,
                        messages=messages,
                        item_output_dir=item_output_dir,
                        args=args,
                        wrapper=action_wrapper,
                        image_replaced=True,
                        som_base_image_path=source_image,
                    )
                    variants[variant]["source_action_mark"] = marked_info
                    variants[variant]["wrong_action_marked_instruction"] = marked_instruction
            else:
                for variant in pitfall_mark_variants:
                    variants[variant] = {
                        "unsom_image": str(marked_image.resolve()),
                        "som_image": "",
                        "som_status": marked_info.get("som_status") or "source_action_mark_failed",
                        "som_action": marked_info.get("som_action"),
                        "action_json": marked_info.get("action_json"),
                        "response": "",
                        "response_path": "",
                        "api_wrapper": "",
                        "image_replaced": True,
                        "source_action_mark": marked_info,
                        "error": marked_info.get("som_status") or "source_action_mark_failed",
                    }
        if family_enabled(args, "contrastive_correction"):
            for variant, api_wrapper in pitfall_api_wrappers.items():
                selected_repredict_wrappers = {
                    action_variant: action_wrapper
                    for action_variant, action_wrapper in same_model_reprediction_wrappers(
                        variant, pitfall_reprediction_wrappers
                    ).items()
                    if variant_enabled(args, f"{variant}_{action_variant}")
                }
                if not selected_repredict_wrappers:
                    continue
                variants.update(run_pitfall_hint_reprediction_variants(
                    task_id=task_id,
                    step_no=step_no,
                    hint_variant=variant,
                    source_image=source_image,
                    item=item,
                    prompt_messages=prompt_messages,
                    item_output_dir=item_output_dir,
                    hint_wrapper=api_wrapper,
                    repredict_wrappers=selected_repredict_wrappers,
                    args=args,
                ))

    result = {
        "task_id": task_id,
        "instruction": item.get("instruction"),
        "category_id": cat_id,
        "category_name": (item.get("classification") or {}).get("category_name") if isinstance(item.get("classification"), dict) else "",
        "rollout_dir": compact(judge.get("rollout_dir")) if isinstance(judge, dict) else "",
        **judge_context_fields(judge),
        "wrong_action_step": wrong_action_step,
        "original_response_path": compact(step.get("response_path")),
        "original_response": compact(step.get("response_text")),
        "original_think": compact(step.get("think")),
        "original_answer": compact(step.get("answer")),
        "original_action": step.get("action_json") if isinstance(step.get("action_json"), dict) else None,
        "prompt_path": compact(step.get("prompt_path")),
        "source_image": str(source_image.resolve()),
        "variants": variants,
    }
    return result


def process(args: argparse.Namespace) -> Dict[str, Any]:
    payload = read_json(args.input)
    items = [item for item in normalize_items(payload) if category_id(item) in args.category_ids]
    if args.max_items is not None:
        items = items[: max(0, args.max_items)]
    log(f"[start] selected {len(items)} item(s) from {args.input}")
    wrapper = build_wrapper(args)
    original_api_wrappers = build_original_api_wrappers(args)
    pitfall_api_wrappers = build_pitfall_api_wrappers(args)
    pitfall_reprediction_wrappers = build_pitfall_reprediction_wrappers(wrapper, original_api_wrappers)
    results = []
    for index, item in enumerate(items, start=1):
        log(f"[progress] {index}/{len(items)} task_id={item.get('task_id')} category={category_id(item)}")
        try:
            results.append(process_one(item, index, args, wrapper, original_api_wrappers, pitfall_api_wrappers, pitfall_reprediction_wrappers))
        except Exception as exc:
            log(f"[error] task_id={item.get('task_id')} {exc}")
            results.append({"task_id": item.get("task_id"), "category_id": category_id(item), "error": str(exc)})
    return {
        "input": str(args.input),
        "model_base_url": args.model_base_url,
        "model_name": args.model_name,
        "repredict_families": list(args.repredict_families),
        "repredict_variants": list(args.repredict_variants),
        "edit_variants": list(EDIT_VARIANTS),
        "wrong_action_marked_variant": WRONG_ACTION_MARKED_VARIANT,
        "original_api_variants": list(original_api_wrappers.keys()),
        "pitfall_llm_variants": list(pitfall_api_wrappers.keys()),
        "pitfall_reprediction_api_variants": list(pitfall_reprediction_wrappers.keys()),
        "pitfall_hint_reprediction_api_variants": list(PITFALL_HINT_REPREDICT_API_VARIANTS),
        "count": len(results),
        "results": results,
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repredict MobileGym failed steps with edited or hint-annotated screenshots.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-base-url", default=DEFAULT_MODEL_BASE_URL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument(
        "--repredict-families",
        default="all",
        help="Comma-separated reprediction families to run, or all. Valid: edit,original_api,wrong_action_marked,contrastive_correction",
    )
    parser.add_argument(
        "--repredict-variants",
        default="all",
        help="Comma-separated exact reprediction variants to run, or all.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    args.category_ids = TARGET_CATEGORY_IDS
    try:
        args.repredict_families = normalize_selector(
            args.repredict_families,
            REPREDICT_FAMILIES,
            "repredict family",
        )
        args.repredict_variants = normalize_selector(
            args.repredict_variants,
            KNOWN_REPREDICT_VARIANTS,
            "repredict variant",
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    payload = process(args)
    write_json(args.output, payload, args.indent)
    log(f"[finish] output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
