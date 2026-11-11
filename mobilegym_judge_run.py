r"""Judge MobileGym trajectories for skill-pipeline failure extraction.

Input can be either:
  - a MobileGym runs directory such as D:\Code\mobilegym-main\runs
  - one MobileGym run directory containing trajectory/<rollout>
  - one rollout directory containing meta.json, step_XXX_response.txt/images

The script converts MobileGym saved prompts/responses into the trajectory shape
used by the ORM runner utilities, then reports completion and the actionable
wrong action step for MobileGym reprediction/skill extraction.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
ORM_ROOT = PROJECT_ROOT / "orm"
if str(ORM_ROOT) not in sys.path:
    sys.path.insert(0, str(ORM_ROOT))

import config  # type: ignore  # noqa: E402
from utils.pipeline_runner import run_mobilegym_skill_trajectory_judgment  # type: ignore  # noqa: E402
from utils.prompt_utils import bind_pipeline_preset, set_model_name, unbind_pipeline_preset  # type: ignore  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "mobilegym" / "runs"
DEFAULT_MODEL = "seed20"
DEFAULT_KNOWLEDGE = "knowledge_base"
DEFAULT_PROFILE_ID = "mobilegym_skill_pipeline"
DEFAULT_ACTION_CHAIN_SOURCE = "raw"
DEFAULT_COMPRESS_QUALITY = 0.6
STEP_RESPONSE_RE = re.compile(r"^step_(\d+)_response\.txt$", re.IGNORECASE)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as f:
        return f.read()


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
    match = STEP_RESPONSE_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def list_step_responses(rollout_dir: Path) -> List[Path]:
    if not rollout_dir.is_dir():
        return []
    files = [p for p in rollout_dir.iterdir() if p.is_file() and STEP_RESPONSE_RE.match(p.name)]
    files.sort(key=step_sort_key)
    return files


def parse_action(answer_text: str) -> Any:
    if not answer_text:
        return None
    try:
        return json.loads(answer_text)
    except json.JSONDecodeError:
        return None


def action_text(action: Any, fallback: str) -> str:
    if not isinstance(action, dict):
        return fallback.strip()
    name = compact_jsonish(action.get("action"))
    if not name:
        return compact_jsonish(action)
    if "point" in action:
        return f"{name} {compact_jsonish(action.get('point'))}"
    if "point1" in action or "point2" in action:
        return f"{name} {compact_jsonish(action.get('point1'))}->{compact_jsonish(action.get('point2'))}"
    if "value" in action:
        return f"{name} {compact_jsonish(action.get('value'))}"
    if "return" in action:
        return f"{name} {compact_jsonish(action.get('return'))}"
    return name


def resolve_image_path(rollout_dir: Path, step_no: int) -> str:
    stem = f"step_{step_no:03d}"
    for suffix in (".jpg", ".png", ".jpeg", ".webp"):
        candidate = rollout_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return str(candidate.resolve())
    return ""


def resolve_annot_image_path(rollout_dir: Path, step_no: int) -> str:
    stem = f"step_{step_no:03d}"
    for suffix in ("_annot.jpg", "_annot.png", "_annot.jpeg", "_annot.webp"):
        candidate = rollout_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return str(candidate.resolve())
    return ""


def resolve_prompt_path(rollout_dir: Path, step_no: int) -> str:
    candidate = rollout_dir / f"step_{step_no:03d}_prompt.json"
    return str(candidate.resolve()) if candidate.is_file() else ""


def load_rollout_meta(rollout_dir: Path) -> Dict[str, Any]:
    meta_path = rollout_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    data = read_json(meta_path)
    return data if isinstance(data, dict) else {}


def rollout_dir_to_task(rollout_dir: Path, task_id: str) -> Dict[str, Any]:
    step_files = list_step_responses(rollout_dir)
    if not step_files:
        raise FileNotFoundError(f"no step_XXX_response.txt files found in: {rollout_dir}")

    trajectory: List[Dict[str, Any]] = []
    mobilegym_steps: List[Dict[str, Any]] = []
    meta = load_rollout_meta(rollout_dir)
    instruction = compact_jsonish(meta.get("task_name"))
    for fallback_index, response_path in enumerate(step_files):
        match = STEP_RESPONSE_RE.match(response_path.name)
        step_no = int(match.group(1)) if match else fallback_index + 1
        response = read_text(response_path)
        answer = extract_tag(ANSWER_RE, response)
        think = extract_tag(THINK_RE, response)
        action = parse_action(answer)
        image_path = resolve_image_path(rollout_dir, step_no)
        annot_image_path = resolve_annot_image_path(rollout_dir, step_no)
        prompt_path = resolve_prompt_path(rollout_dir, step_no)
        prompt_messages: Any = None
        if prompt_path:
            try:
                prompt_messages = read_json(Path(prompt_path))
            except Exception as exc:
                prompt_messages = {"error": f"failed to read prompt JSON: {exc}"}
        value: Dict[str, Any] = {
            "thought": think,
            "action": action_text(action, answer),
            "tool_call": answer,
        }
        if isinstance(action, dict):
            value["normalized_tool_call"] = action

        trajectory.append(
            {
                "index": fallback_index,
                "image_local_path": image_path,
                "image": image_path,
                "value": value,
            }
        )
        mobilegym_steps.append(
            {
                "step_no": step_no,
                "step_index": fallback_index,
                "image_path": image_path,
                "annot_image_path": annot_image_path,
                "response_path": str(response_path.resolve()),
                "response_text": response,
                "think": think,
                "answer": answer,
                "action_json": action if isinstance(action, dict) else None,
                "prompt_path": prompt_path,
                "prompt_messages": prompt_messages,
            }
        )

    if not instruction:
        raise ValueError(f"missing instruction in MobileGym rollout: {rollout_dir}")
    return {
        "task_id": task_id,
        "instruction": instruction,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "traj": trajectory,
        "mobilegym_steps": mobilegym_steps,
        "metadata": {"rollout_dir": str(rollout_dir.resolve())},
    }


def discover_rollout_dirs(path: Path) -> List[Path]:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory not found: {root}")
    if list_step_responses(root):
        return [root]
    if (root / "trajectory").is_dir():
        exec_task = root / "trajectory" / "_ExecTask"
        if list_step_responses(exec_task):
            return [exec_task]
        root = root / "trajectory"
    dirs = [p for p in root.iterdir() if p.is_dir() and list_step_responses(p)]
    if not dirs:
        for run_dir in root.iterdir():
            exec_task = run_dir / "trajectory" / "_ExecTask"
            if list_step_responses(exec_task):
                dirs.append(exec_task)
    dirs.sort(key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"no MobileGym rollout directories found under: {root}")
    return dirs


def infer_wrong_action_step(judgment: Dict[str, Any], step_count: int) -> Optional[int]:
    wrong_action = judgment.get("wrong_action_step")
    if isinstance(wrong_action, int) and 0 <= wrong_action < step_count:
        return wrong_action
    return None


def first_nonempty_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = value if isinstance(value, str) else str(value)
        text = text.strip()
        if text:
            return text
    return ""


def summarize_judgment(task: Dict[str, Any], rollout_dir: Path, judgment: Dict[str, Any], cache_used: bool) -> Dict[str, Any]:
    step_count = len(task["traj"])
    wrong_action_step = infer_wrong_action_step(judgment, step_count)
    final_judgment = judgment.get("final_judgment")
    task_status = judgment.get("task_status")
    judge_pass = final_judgment == 1 or task_status in {"perfect", "fair_enough"}
    failed_state = not judge_pass
    wrong_action_step_record = None
    if failed_state and wrong_action_step is not None:
        mobilegym_steps = task.get("mobilegym_steps")
        if isinstance(mobilegym_steps, list) and 0 <= wrong_action_step < len(mobilegym_steps):
            step = mobilegym_steps[wrong_action_step]
            if isinstance(step, dict):
                wrong_action_step_record = step
    judge_reason = first_nonempty_text(
        judgment.get("wrong_action_reason"),
        judgment.get("step_alignment_reason"),
        judgment.get("failure_mode_for_skill"),
        judgment.get("first_deviation_reason"),
        judgment.get("reason"),
    )
    return {
        "task_id": task["task_id"],
        "instruction": task["instruction"],
        "rollout_dir": str(rollout_dir.resolve()),
        "judge_profile": DEFAULT_PROFILE_ID,
        "step_count": step_count,
        "judge_pass": judge_pass,
        "failed_state": failed_state,
        "first_bad_state_step": judgment.get("first_bad_state_step") if failed_state else None,
        "wrong_action_step": wrong_action_step if failed_state else None,
        "wrong_action_step_record": wrong_action_step_record,
        "judge_reason": judge_reason if failed_state else "",
        "wrong_action_reason": judgment.get("wrong_action_reason", "") if failed_state else "",
        "step_alignment_reason": judgment.get("step_alignment_reason", "") if failed_state else "",
        "failure_mode_for_skill": judgment.get("failure_mode_for_skill", "") if failed_state else "",
        "can_extract_skill": bool(judgment.get("can_extract_skill")) if failed_state else False,
        "skill_signal_type": judgment.get("skill_signal_type", "") if failed_state else "",
        "first_deviation_reason": judgment.get("first_deviation_reason", "") if failed_state else "",
        "cache_used": cache_used,
    }


def run_one(rollout_dir: Path, root_task_id: str, args: argparse.Namespace, *, multi_rollout: bool) -> Dict[str, Any]:
    run_id = rollout_dir.parent.parent.name if rollout_dir.parent.name == "trajectory" else root_task_id
    task_id = args.task_id or run_id
    if multi_rollout and rollout_dir.name != "_ExecTask":
        task_id = f"{task_id}_{rollout_dir.name}"
    log(f"[parse] task_id={task_id} rollout={rollout_dir}")
    task = rollout_dir_to_task(rollout_dir, task_id=task_id)
    log(f"[judge] task_id={task_id} instruction={task['instruction']} steps={len(task['traj'])}")
    judgment, cache_used = run_mobilegym_skill_trajectory_judgment(
        instruction=task["instruction"],
        trajectory=task["traj"],
        task_id="",
        use_cache=False,
        cache_cleared=False,
        verbose=args.verbose,
        profile_id=DEFAULT_PROFILE_ID,
        lite_include_action_list=not args.no_action_list,
        lite_action_chain_source=DEFAULT_ACTION_CHAIN_SOURCE,
        compress_quality=DEFAULT_COMPRESS_QUALITY,
        is_negative_task=False,
        created_at=task.get("created_at", ""),
        use_task_knowledge=True,
        judge_takeover="end-to-end",
    )
    summary = summarize_judgment(task, rollout_dir, judgment, cache_used)
    log(
        f"[done] task_id={task_id} judge_pass={summary.get('judge_pass')} "
        f"failed_state={summary.get('failed_state')} "
        f"wrong_action_step={summary.get('wrong_action_step')} can_extract_skill={summary.get('can_extract_skill')}"
    )
    return summary


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
    parser = argparse.ArgumentParser(description="Use MobileGym skill-pipeline judge on saved runs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="runs directory, one run directory, or one rollout directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--knowledge", default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--task-id", default=None, help="override task_id for a single rollout")
    parser.add_argument("--no-action-list", action="store_true")
    parser.add_argument("--max-rollouts", type=int, default=None, help="only judge the first N rollout directories")
    parser.add_argument("--output", default="mobilegym_skill_result.json", help="write JSON output to this file")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent; use 0 for compact output")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    log(f"[start] input={args.input} model={args.model} knowledge={args.knowledge}")
    set_model_name(args.model)
    config.set_knowledge_base(args.knowledge)
    bind_pipeline_preset("mobilegym_skill_pipeline")
    try:
        rollout_dirs = discover_rollout_dirs(args.input)
        if args.max_rollouts is not None:
            rollout_dirs = rollout_dirs[: max(0, args.max_rollouts)]
        log(f"[discover] found {len(rollout_dirs)} rollout(s)")
        root_task_id = args.input.expanduser().resolve().name
        multi_rollout = len(rollout_dirs) > 1
        results = []
        for index, rollout_dir in enumerate(rollout_dirs, start=1):
            log(f"[progress] {index}/{len(rollout_dirs)}")
            results.append(run_one(rollout_dir, root_task_id, args, multi_rollout=multi_rollout))
    finally:
        unbind_pipeline_preset()

    payload: Any = results[0] if len(results) == 1 else {"input": str(args.input), "count": len(results), "results": results}
    write_output(payload, args.output, args.indent)
    log("[finish] output written" if args.output else "[finish] output printed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
