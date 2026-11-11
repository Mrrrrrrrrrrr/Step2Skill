r"""Judge rollout trajectories with the existing lite unified mode.

Input can be either:
  - a task directory such as D:\Traj\2026052122421251, containing rollout subdirectories
  - one rollout directory containing step_XXXX.json/png files

The script outputs where the trajectory first fails and the lite reasoning for
failed/env-error states. It does not inspect image pixels while parsing; image
file names are only resolved into paths for the lite model call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from utils.pipeline_runner import run_lite_trajectory_unified_all_images_binary_judgment  # noqa: E402
from utils.prompt_utils import bind_pipeline_preset, set_model_name, unbind_pipeline_preset  # noqa: E402


DEFAULT_MODEL = "seed_free"
DEFAULT_KNOWLEDGE = "knowledge_base"
DEFAULT_PROFILE_ID = "traj_failure_lite"
DEFAULT_ACTION_CHAIN_SOURCE = "raw"
PREDICTION_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
PREDICTION_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
PREDICTION_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
STEP_JSON_RE = re.compile(r"^step_(\d+)\.json$", re.IGNORECASE)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def compact_jsonish(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)


def extract_tag(regex: re.Pattern[str], text: str) -> str:
    match = regex.search(text or "")
    return (match.group(1) or "").strip() if match else ""


def step_sort_key(path: Path) -> Tuple[int, str]:
    match = STEP_JSON_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def list_step_jsons(rollout_dir: Path) -> List[Path]:
    files = [p for p in rollout_dir.iterdir() if p.is_file() and STEP_JSON_RE.match(p.name)]
    files.sort(key=step_sort_key)
    return files


def resolve_image_path(rollout_dir: Path, step_json_path: Path, data: Dict[str, Any]) -> str:
    image_file = data.get("image_file")
    candidates: List[Path] = []
    if isinstance(image_file, str) and image_file.strip():
        candidates.append(rollout_dir / image_file.strip())
    candidates.extend([
        rollout_dir / f"{step_json_path.stem}.png",
        rollout_dir / f"{step_json_path.stem}.jpg",
        rollout_dir / f"{step_json_path.stem}.jpeg",
        rollout_dir / f"{step_json_path.stem}.webp",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return ""


def parse_takeover(tool_call_text: str, data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    takeover_input = compact_jsonish(data.get("takeover_input"))
    if isinstance(data.get("take_over"), dict):
        raw = data["take_over"]
        return {
            "summarize": compact_jsonish(raw.get("summarize") or raw.get("summary") or raw.get("text")),
            "category": compact_jsonish(raw.get("category") or raw.get("tag")),
            "continue_goal": compact_jsonish(raw.get("continue_goal")),
            "input": takeover_input or compact_jsonish(raw.get("input")),
        }

    if not tool_call_text:
        return None
    try:
        parsed = json.loads(tool_call_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    args = parsed.get("arguments")
    if not isinstance(args, dict) or args.get("action_type") != "take_over":
        return None
    return {
        "summarize": compact_jsonish(args.get("text")),
        "category": compact_jsonish(args.get("tag")),
        "continue_goal": "Auto task continuation" if takeover_input else "external continuation",
        "input": takeover_input,
    }


def rollout_dir_to_task(rollout_dir: Path, task_id: str) -> Dict[str, Any]:
    step_files = list_step_jsons(rollout_dir)
    if not step_files:
        raise FileNotFoundError(f"no step_XXXX.json files found in: {rollout_dir}")

    trajectory: List[Dict[str, Any]] = []
    instruction = ""
    for fallback_index, step_path in enumerate(step_files):
        data = read_json(step_path)
        if not instruction and isinstance(data.get("instruction"), str):
            instruction = data["instruction"].strip()

        prediction = data.get("prediction") if isinstance(data.get("prediction"), str) else ""
        action_text = extract_tag(PREDICTION_ACTION_RE, prediction)
        thought_text = compact_jsonish(data.get("thought")) or extract_tag(PREDICTION_THINK_RE, prediction)
        tool_call_text = extract_tag(PREDICTION_TOOL_CALL_RE, prediction)
        if not action_text:
            action_text = compact_jsonish(data.get("action"))

        step_index = data.get("step_index")
        if not isinstance(step_index, int):
            step_index = fallback_index
        image_path = resolve_image_path(rollout_dir, step_path, data)
        value: Dict[str, Any] = {
            "thought": thought_text,
            "action": action_text,
            "tool_call": tool_call_text,
        }
        if isinstance(data.get("action"), dict):
            value["normalized_tool_call"] = data["action"]

        step: Dict[str, Any] = {
            "index": step_index,
            "image_local_path": image_path,
            "image": image_path,
            "value": value,
        }
        takeover = parse_takeover(tool_call_text, data)
        if takeover:
            value["take_over"] = takeover
            step["take_over"] = takeover
        trajectory.append(step)

    if not instruction:
        raise ValueError(f"missing instruction in step JSON files: {rollout_dir}")
    return {
        "task_id": task_id,
        "instruction": instruction,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "traj": trajectory,
        "metadata": {"rollout_dir": str(rollout_dir.resolve())},
    }


def discover_rollout_dirs(path: Path) -> List[Path]:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory not found: {root}")
    if list_step_jsons(root):
        return [root]
    dirs = [p for p in root.iterdir() if p.is_dir() and list_step_jsons(p)]
    dirs.sort(key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"no rollout directories with step_XXXX.json found under: {root}")
    return dirs


def strip_lite_tag(text: Any) -> str:
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    return s.replace("<lite判定>", "").strip()


def infer_failure_step(judgment: Dict[str, Any], step_count: int) -> Tuple[Optional[int], str]:
    first_deviation = judgment.get("first_deviation_step")
    if isinstance(first_deviation, int) and 0 <= first_deviation < step_count:
        return first_deviation, "first_deviation_step"

    max_prefix = judgment.get("max_correct_prefix_end_step")
    if isinstance(max_prefix, int):
        candidate = max_prefix + 1
        if 0 <= candidate < step_count:
            return candidate, "max_correct_prefix_end_step + 1"

    failure_steps = judgment.get("evidence", {}).get("failure_steps")
    if isinstance(failure_steps, list):
        for item in failure_steps:
            if isinstance(item, int) and 0 <= item < step_count:
                return item, "evidence.failure_steps[0]"

    if judgment.get("final_judgment") in (0, -1) and step_count > 0:
        return step_count - 1, "fallback_last_step"
    return None, ""


def summarize_judgment(task: Dict[str, Any], rollout_dir: Path, judgment: Dict[str, Any], cache_used: bool) -> Dict[str, Any]:
    step_count = len(task["traj"])
    failure_step_0_based, failure_step_source = infer_failure_step(judgment, step_count)
    lite_reasoning = strip_lite_tag(judgment.get("lite_reasoning") or judgment.get("reasoning") or judgment.get("reason"))
    final_judgment = judgment.get("final_judgment")
    task_status = judgment.get("task_status")
    failed_state = final_judgment in (0, -1) or task_status in {"failed", "env_error"}
    return {
        "task_id": task["task_id"],
        "rollout_dir": str(rollout_dir.resolve()),
        "step_count": step_count,
        "final_judgment": final_judgment,
        "task_status": task_status,
        "trajectory_correctness_judgment": judgment.get("trajectory_correctness_judgment"),
        "completion_judgment": judgment.get("completion_judgment"),
        "failed_state": failed_state,
        "failure_step_0_based": failure_step_0_based if failed_state else None,
        "failure_step_1_based": (failure_step_0_based + 1) if failed_state and failure_step_0_based is not None else None,
        "failure_step_source": failure_step_source if failed_state else "",
        "first_deviation_reason": judgment.get("first_deviation_reason", "") if failed_state else "",
        "lite_reasoning": lite_reasoning if failed_state else "",
        "cache_used": cache_used,
    }


def run_one(rollout_dir: Path, root_task_id: str, args: argparse.Namespace, *, multi_rollout: bool) -> Dict[str, Any]:
    task_id = args.task_id or root_task_id
    if multi_rollout:
        task_id = f"{task_id}_{rollout_dir.name}"
    task = rollout_dir_to_task(rollout_dir, task_id=task_id)
    cache_task_id = task["task_id"] if args.enable_data_cache else ""
    judgment, cache_used = run_lite_trajectory_unified_all_images_binary_judgment(
        instruction=task["instruction"],
        trajectory=task["traj"],
        task_id=cache_task_id,
        use_cache=args.enable_data_cache and not args.no_cache,
        cache_cleared=False,
        verbose=args.verbose,
        profile_id=args.profile_id,
        lite_include_action_list=not args.no_action_list,
        lite_action_chain_source=args.action_chain_source,
        compress_quality=args.compress_quality,
        is_negative_task=args.negative_task,
        created_at=task.get("created_at", ""),
        use_task_knowledge=not args.no_task_knowledge,
        judge_takeover=args.judge_takeover,
    )
    return summarize_judgment(task, rollout_dir, judgment, cache_used)


def write_output(payload: Any, output_path: Optional[str], indent: int) -> None:
    json_indent = None if indent == 0 else indent
    text = json.dumps(payload, ensure_ascii=False, indent=json_indent)
    if output_path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    print(text)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use lite mode to locate rollout trajectory failure steps.")
    parser.add_argument("--input", type=Path, help="task directory or one rollout directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--knowledge", default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--task-id", default=None, help="override task_id prefix used for cache")
    parser.add_argument("--profile-id", default=DEFAULT_PROFILE_ID)
    parser.add_argument("--action-chain-source", default=DEFAULT_ACTION_CHAIN_SOURCE, choices=["raw", "validated", "validated_thought"])
    parser.add_argument("--compress-quality", type=float, default=0.6)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-action-list", action="store_true")
    parser.add_argument("--no-task-knowledge", action="store_true")
    parser.add_argument("--negative-task", action="store_true")
    parser.add_argument("--judge-takeover", default="end-to-end", choices=["end-to-end", "only-lmd", "all", "none"])
    parser.add_argument("--max-rollouts", type=int, default=None, help="only judge the first N rollout directories")
    parser.add_argument(
        "--enable-data-cache",
        action="store_true",
        help="allow lite runner cache/prompt_response JSON files under data/; disabled by default",
    )
    parser.add_argument("--output", default=None, help="write JSON output to this file")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent; use 0 for jsonl-style compact output")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    set_model_name(args.model)
    config.set_knowledge_base(args.knowledge)
    bind_pipeline_preset("lite_trajectory_failure")
    try:
        rollout_dirs = discover_rollout_dirs(args.input)
        if args.max_rollouts is not None:
            rollout_dirs = rollout_dirs[: max(0, args.max_rollouts)]
        root_task_id = args.input.expanduser().resolve().name
        multi_rollout = len(rollout_dirs) > 1
        results = [run_one(rollout_dir, root_task_id, args, multi_rollout=multi_rollout) for rollout_dir in rollout_dirs]
    finally:
        unbind_pipeline_preset()

    payload: Any = results[0] if len(results) == 1 else {"input": str(args.input), "count": len(results), "results": results}
    write_output(payload, args.output, args.indent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
