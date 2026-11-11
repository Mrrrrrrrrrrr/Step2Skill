"""Convert a rollout folder to one ORM task and run predict_one.

Input folder format matches honor-verl rollout/valid dumps:
``messages.json`` plus numbered screenshots such as ``0.jpg``, ``1.jpg``.

Example:
    conda run --no-capture-output -n zj_base310 python -u \
      utils/predict_rollout_folder_orm.py \
      /path/to/20260703000329867 \
      --dump-input-jsonl /tmp/one_orm_input.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (  # noqa: E402
    _error_cause_failure_reason,
    _error_cause_prediction_reasoning,
    _normalize_error_cause_trajectory_fields,
    PREDICTION_PROVIDER,
    _process_single_task_wrapper,
    _task_status_from_lite_final,
    configure_knowledge_base,
)


DEFAULT_PIPELINE_JSON = PROJECT_ROOT / "pipeline" / "pipeline_error_cause_analysis.json"
DEFAULT_KNOWLEDGE = "knowledge_base"
DEFAULT_MODEL = "seed_free"

IMAGE_NAME_RE = re.compile(r"^(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)
IMAGE_ONLY_RE = re.compile(r"^\s*<image>\s*$", re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _normalize_messages(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("messages"), list):
        return [m for m in raw["messages"] if isinstance(m, dict)]
    return []


def list_screenshot_paths(folder: Path) -> List[str]:
    numbered: List[tuple[int, Path]] = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = IMAGE_NAME_RE.match(path.name)
        if match:
            numbered.append((int(match.group(1)), path.resolve()))
    numbered.sort(key=lambda item: item[0])
    return [str(path) for _, path in numbered]


def _extract_assistant_tags(text: str) -> Dict[str, str]:
    def pick(regex: re.Pattern[str]) -> str:
        match = regex.search(text)
        return (match.group(1) or "").strip() if match else ""

    thought = pick(THINK_RE)
    if not thought and "</think>" in text.lower():
        thought = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return {
        "thought": thought,
        "action": pick(ACTION_RE),
        "tool_call": pick(TOOL_CALL_RE),
    }


def build_traj(messages: List[Dict[str, Any]], image_paths: List[str]) -> List[Dict[str, Any]]:
    traj: List[Dict[str, Any]] = []
    img_idx = 0
    n_messages = len(messages)

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = _message_text(msg.get("content"))

        if role == "user" and IMAGE_ONLY_RE.match(content):
            img_idx += 1
            continue
        if role != "assistant" or img_idx <= 0:
            continue

        step_img_idx = img_idx - 1
        image_path = image_paths[step_img_idx] if step_img_idx < len(image_paths) else ""
        extracted = _extract_assistant_tags(content)
        value: Dict[str, Any] = {
            "thought": extracted["thought"],
            "action": extracted["action"],
            "tool_call": extracted["tool_call"],
        }
        step: Dict[str, Any] = {
            "index": step_img_idx,
            "image_local_path": image_path,
            "image": image_path,
            "value": value,
        }

        tool_call = extracted["tool_call"]
        if tool_call:
            try:
                parsed = json.loads(tool_call)
                args = parsed.get("arguments", {}) if isinstance(parsed, dict) else {}
                if isinstance(args, dict) and args.get("action_type") == "take_over":
                    take_over = {
                        "summarize": str(args.get("text") or ""),
                        "category": str(args.get("tag") or ""),
                        "continue_goal": "外部接续",
                        "input": "",
                    }
                    for nxt in messages[i + 1:n_messages]:
                        if nxt.get("role") != "user":
                            continue
                        nxt_content = _message_text(nxt.get("content"))
                        if IMAGE_ONLY_RE.match(nxt_content):
                            continue
                        if nxt_content.strip():
                            take_over["input"] = nxt_content.strip()
                            take_over["continue_goal"] = "Auto任务继续"
                            break
                    value["take_over"] = take_over
                    step["take_over"] = {
                        "summarize": take_over["summarize"],
                        "category": take_over["category"],
                        "continue_goal": take_over["continue_goal"],
                    }
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        traj.append(step)

    return traj


def rollout_folder_to_task(folder: str | Path, task_id: Optional[str] = None) -> Dict[str, Any]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"rollout folder not found: {root}")
    messages_path = root / "messages.json"
    if not messages_path.is_file():
        raise FileNotFoundError(f"messages.json not found: {messages_path}")

    with messages_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("messages.json root must be an object")

    instruction = str(data.get("task") or "").strip()
    if not instruction:
        raise ValueError("messages.json missing non-empty field: task")
    messages = _normalize_messages(data.get("messages"))
    if not messages:
        raise ValueError("messages.json missing non-empty messages list")
    image_paths = list_screenshot_paths(root)
    if not image_paths:
        raise FileNotFoundError(f"no numbered screenshots found in: {root}")

    metadata = copy.deepcopy(data.get("metadata")) if isinstance(data.get("metadata"), dict) else {}
    ext_info = copy.deepcopy(data.get("ext_info")) if isinstance(data.get("ext_info"), dict) else None
    if ext_info is not None and "ext_info" not in metadata:
        metadata["ext_info"] = ext_info
    level = data.get("level")
    if isinstance(level, str):
        metadata.setdefault("ext_info", {})
        if isinstance(metadata["ext_info"], dict):
            metadata["ext_info"].setdefault("level", level)

    metadata.update({
        "rollout_dir": str(root),
        "state": data.get("state", ""),
        "finish_reason_type": data.get("finish_reason_type", ""),
        "image_count": len(image_paths),
        "message_count": len(messages),
    })

    return {
        "task_id": task_id or root.name,
        "instruction": instruction,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "honor_verl_rollout_folder",
        "metadata": metadata,
        "tags": [],
        "traj": build_traj(messages, image_paths),
    }


def load_pipeline_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"pipeline json root must be an object: {path}")
    return data


def _clean_empty(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            cleaned = _clean_empty(value)
            if cleaned is not None and cleaned != "":
                out[key] = cleaned
        return out
    if isinstance(obj, list):
        return [_clean_empty(v) for v in obj]
    return obj


def _lite_gate_payload(judgment: Dict[str, Any], cache_used: Dict[str, Any]) -> Dict[str, Any]:
    gate = {
        "final_judgment": judgment.get("lite_final_judgment"),
        "task_status": judgment.get("task_status"),
        "reasoning": judgment.get("lite_reasoning"),
        "cache_used": cache_used.get("judgment_lite_gate"),
        "trajectory_correctness_judgment": judgment.get("trajectory_correctness_judgment"),
        "trajectory_reasoning": judgment.get("trajectory_reasoning"),
        "first_deviation_step": judgment.get("first_deviation_step"),
        "first_deviation_reason": judgment.get("first_deviation_reason"),
        "max_correct_prefix_end_step": judgment.get("max_correct_prefix_end_step"),
        "max_correct_prefix_steps": judgment.get("max_correct_prefix_steps"),
        "task_knowledge_hit": judgment.get("task_knowledge_hit"),
        "task_knowledge_confidence": judgment.get("task_knowledge_confidence"),
        "task_knowledge_reasoning": judgment.get("task_knowledge_reasoning"),
    }
    return _clean_empty(gate)


def _minimal_error_cause_response(result: Dict[str, Any]) -> Dict[str, Any]:
    task_id = result.get("task_id", "")
    judgment = result.get("judgment") if isinstance(result.get("judgment"), dict) else {}
    cache_used = result.get("cache_used") if isinstance(result.get("cache_used"), dict) else {}

    gate_status = judgment.get("task_status")
    if not isinstance(gate_status, str):
        gate_status = ""
    success_status = gate_status if gate_status in {"perfect", "fair_enough"} else "fair_enough"
    task_status = _task_status_from_lite_final(
        judgment.get("lite_final_judgment"),
        success_status=success_status,
        uncertain_status="env_error",
    )
    if task_status is None:
        task_status = judgment.get("task_status") if isinstance(judgment.get("task_status"), str) else ""

    trajectory_extra = _normalize_error_cause_trajectory_fields(judgment, task_status)
    orm_prediction: Dict[str, Any] = {}
    if judgment.get("error_cause_skipped"):
        orm_prediction.update({"error_cause_analysis": 0, "error_cause_skipped": 1})
    else:
        orm_prediction["error_cause_analysis"] = 1
    if "is_trajectory_correct" in trajectory_extra:
        orm_prediction["is_trajectory_correct"] = trajectory_extra["is_trajectory_correct"]
    if result.get("error"):
        orm_prediction["error"] = result["error"]

    prediction: Dict[str, Any] = {"provider": PREDICTION_PROVIDER}
    if task_status:
        prediction["task_status"] = task_status
    prediction["failure_reason"] = _error_cause_failure_reason(judgment)
    prediction["reasoning"] = _error_cause_prediction_reasoning(judgment)
    if not judgment.get("error_cause_skipped"):
        prediction.update({
            "primary_error_cause": judgment.get("primary_error_cause", ""),
            "secondary_error_cause": judgment.get("secondary_error_cause", ""),
            "error_cause_reasoning": judgment.get("error_cause_reasoning", ""),
        })

    evaluation: Dict[str, Any] = {
        "orm_prediction": _clean_empty(orm_prediction),
        "lite_gate": _lite_gate_payload(judgment, cache_used),
        "prediction": _clean_empty(prediction),
    }
    if cache_used:
        evaluation["cache_used"] = cache_used
    if "trajectory_reasoning" in trajectory_extra:
        evaluation["reasoning"] = trajectory_extra["trajectory_reasoning"]
    if "first_deviation_step" in trajectory_extra:
        evaluation["first_deviation_step"] = trajectory_extra["first_deviation_step"]
    if "first_deviation_reason" in trajectory_extra:
        evaluation["first_deviation_reason"] = trajectory_extra["first_deviation_reason"]

    return {"task_id": task_id, "evaluation": _clean_empty(evaluation)}


def run_orm(task: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    pipeline_data = load_pipeline_json(args.pipeline_json)
    configure_knowledge_base(args.knowledge, verbose=False)
    preset = pipeline_data.get("preset")
    result = _process_single_task_wrapper(
        task,
        index=0,
        total=1,
        use_cache=not args.no_cache,
        lite_mode=False,
        lite_action_list=False,
        verbose=args.verbose,
        model_name=args.model,
        compress_quality=pipeline_data.get("compress_quality"),
        use_keyframe_images=bool(pipeline_data.get("use_keyframe_images", False)),
        use_qa_images=bool(pipeline_data.get("use_qa_images", False)),
        no_qa=bool(pipeline_data.get("no_qa", False)),
        skip_takeover=bool(pipeline_data.get("skip_takeover", True)),
        pipeline_json=pipeline_data,
        pipeline_preset=preset.strip() if isinstance(preset, str) and preset.strip() else None,
        error_cause_lite_gate_all=args.lite_gate_all,
        error_cause_enable_trajectory=False,
        use_task_knowledge=not args.no_task_knowledge,
    )
    final_mode = result.get("pipeline_spec", {}).get("final_judgment_mode")
    if final_mode == "error_cause_analysis":
        return _minimal_error_cause_response(result)
    raise ValueError(f"unsupported pipeline final_judgment_mode for this helper: {final_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one honor-verl rollout folder and run the ORM pipeline."
    )
    parser.add_argument("folder", help="folder containing messages.json and numbered screenshots")
    parser.add_argument("--pipeline-json", default=str(DEFAULT_PIPELINE_JSON))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--knowledge", default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--task-id", default=None, help="override task_id, default is folder name")
    parser.add_argument("--no-cache", action="store_true", help="disable ORM cache")
    parser.add_argument(
        "--no-task-knowledge",
        action="store_true",
        help="skip external task-knowledge retrieval/injection in lite gate",
    )
    parser.add_argument(
        "--lite-gate-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run error-cause lite gate first; default matches full status filter CLI",
    )
    parser.add_argument("--verbose", action="store_true", help="print internal task progress details")
    parser.add_argument("--dump-input-jsonl", default=None, help="write converted ORM input as one jsonl line")
    parser.add_argument("--output", default=None, help="write minimal ORM response json to this file")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent for stdout/output; use 0 for one line")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = rollout_folder_to_task(args.folder, task_id=args.task_id)

    if args.dump_input_jsonl:
        out_path = Path(args.dump_input_jsonl).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    result = run_orm(task, args)
    indent = None if args.indent == 0 else args.indent
    text = json.dumps(result, ensure_ascii=False, indent=indent)
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
