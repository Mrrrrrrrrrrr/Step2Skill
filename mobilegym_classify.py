r"""Map MobileGym trajectory judge results to the ten error classes in classify.md.

Input is the JSON produced by mobilegym_judge_run.py. Successful trajectories
are assigned to 正确类. Failed trajectories are classified by an LLM using the
judge reason, the wrong-action step response text, and the wrong-action step
annotated screenshot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent

from api import ApiWrapper  # type: ignore  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "mobilegym_skill_result.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "mobilegym_error_classification.json"
DEFAULT_CLASSIFY_MD = PROJECT_ROOT / "classify.md"
VALID_ERROR_IDS = set(range(1, 11))


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig") as f:
        return f.read()


def write_json(path: Path, payload: Any, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=None if indent == 0 else indent)
        f.write("\n")


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_judge_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        items = payload["results"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise ValueError("MobileGym skill result JSON must be an object, a list, or an object with results")
    return [item for item in items if isinstance(item, dict)]


def is_success(item: Dict[str, Any]) -> bool:
    if item.get("judge_pass") is True:
        return True
    if item.get("failed_state") is False:
        return True
    return False


def wrong_action_step_info(item: Dict[str, Any]) -> Dict[str, Any]:
    step = item.get("wrong_action_step_record")
    return step if isinstance(step, dict) else {}


def step_files_from_item(item: Dict[str, Any]) -> Dict[str, str]:
    step = wrong_action_step_info(item)
    annot_image = compact(step.get("annot_image_path"))
    image = annot_image or compact(step.get("image_path"))
    return {
        "annot_image_path": image,
        "response_path": compact(step.get("response_path")),
        "response_text": compact(step.get("response_text")),
    }


def build_system_prompt(classify_text: str) -> str:
    return (
        "你是GUI Agent失败轨迹的错误类型分类器。你必须严格根据给定的十类错误定义进行分类。\n"
        "只允许把失败轨迹分类为1到10中的一个类别；成功轨迹不会发送给你。\n"
        "优先依据wrong_action_step对应截图、模型在该步的思考/回答、trajectory judge给出的失败原因。\n"
        "若后续步骤出现连锁反应，不要用后续现象覆盖首次直接失败原因。\n"
        "必须输出JSON，不要输出Markdown代码块或额外解释。\n\n"
        f"十类错误定义如下：\n{classify_text}"
    )


def build_user_prompt(item: Dict[str, Any], step_files: Dict[str, str]) -> str:
    evidence = {
        "task_id": item.get("task_id"),
        "instruction": item.get("instruction"),
        "judge_pass": item.get("judge_pass"),
        "failed_state": item.get("failed_state"),
        "wrong_action_step": item.get("wrong_action_step"),
        "first_bad_state_step": item.get("first_bad_state_step"),
        "judge_reason": item.get("judge_reason"),
        "wrong_action_reason": item.get("wrong_action_reason"),
        "step_alignment_reason": item.get("step_alignment_reason"),
        "failure_mode_for_skill": item.get("failure_mode_for_skill"),
        "can_extract_skill": item.get("can_extract_skill"),
        "skill_signal_type": item.get("skill_signal_type"),
        "first_deviation_reason": item.get("first_deviation_reason"),
    }
    return (
        "请根据截图、wrong_action_step模型输出和trajectory judge信息，把该失败轨迹归入十类错误之一。\n\n"
        "trajectory judge信息：\n"
        f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
        "上一步 JSON 中携带的wrong_action_step模型思考与回答：\n"
        f"{step_files.get('response_text', '').strip()}\n\n"
        "输出JSON格式必须为：\n"
        "{\n"
        "  \"category_id\": 1到10的整数,\n"
        "  \"category_name\": \"对应的类别名称\",\n"
        "  \"confidence\": 0到1之间的小数,\n"
        "  \"reason\": \"简要说明为什么属于该类，并引用关键证据\"\n"
        "}"
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM output JSON root must be an object")
    category_id = data.get("category_id")
    if isinstance(category_id, str) and category_id.strip().isdigit():
        data["category_id"] = int(category_id.strip())
    return data


def build_llm(api_provider: str, timeout: int) -> Any:
    if api_provider == "doubao":
        return ApiWrapper(model_name="DouBao-Seed-2.0-Pro", timeout=timeout)
    if api_provider == "qwen_plus":
        return ApiWrapper(model_name="Qwen3.6-Plus", api_key="", timeout=timeout)
    raise ValueError(f"unsupported api provider: {api_provider}")


def classify_failure(
    item: Dict[str, Any],
    llm: Any,
    system_prompt: str,
) -> Dict[str, Any]:
    step_files = step_files_from_item(item)
    if not step_files.get("response_text"):
        return {
            "category_id": None,
            "category_name": "分类失败",
            "confidence": 0.0,
            "reason": "无法定位失败轨迹目录或wrong_action_step",
        }

    image_path = step_files.get("annot_image_path") or None
    query = build_user_prompt(item, step_files)
    raw = llm.chat(query=query, image_path=image_path, system_prompt=system_prompt)
    try:
        parsed = extract_json_object(raw)
    except Exception as exc:
        return {
            "category_id": None,
            "category_name": "分类失败",
            "confidence": 0.0,
            "reason": f"LLM输出不是合法JSON: {exc}",
            "wrong_action_step_image": step_files.get("annot_image_path", ""),
            "wrong_action_step_response": step_files.get("response_path", ""),
        }

    if parsed.get("category_id") not in VALID_ERROR_IDS:
        parsed["category_name"] = parsed.get("category_name") or "分类失败"
        parsed["reason"] = parsed.get("reason") or "LLM未返回1到10之间的category_id"

    parsed["wrong_action_step_image"] = step_files.get("annot_image_path", "")
    parsed["wrong_action_step_response"] = step_files.get("response_path", "")
    return parsed


def build_success_result(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "category_id": 0,
        "category_name": "正确类",
        "confidence": 1.0,
        "reason": "trajectory judge评判该轨迹成功",
    }


def compact_judge_item(item: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "task_id",
        "instruction",
        "rollout_dir",
        "judge_profile",
        "judge_pass",
        "failed_state",
        "wrong_action_step",
        "wrong_action_step_record",
        "first_bad_state_step",
        "judge_reason",
        "wrong_action_reason",
        "step_alignment_reason",
        "failure_mode_for_skill",
        "can_extract_skill",
        "skill_signal_type",
        "first_deviation_reason",
    )
    return {key: item.get(key) for key in keys if item.get(key) not in (None, "")}


def process(args: argparse.Namespace) -> Dict[str, Any]:
    judge_payload = read_json(args.input)
    classify_text = read_text(args.classify_md)
    items = normalize_judge_items(judge_payload)
    system_prompt = build_system_prompt(classify_text)
    llm = None
    results = []

    log(f"[start] input={args.input} tasks={len(items)}")
    for index, item in enumerate(items, start=1):
        task_id = compact(item.get("task_id")) or f"item_{index}"
        log(f"[progress] {index}/{len(items)} task_id={task_id}")
        if is_success(item):
            classification = build_success_result(item)
            log(f"[done] task_id={task_id} category=正确类")
        else:
            if llm is None:
                llm = build_llm(args.api_provider, args.api_timeout)
            classification = classify_failure(item, llm, system_prompt)
            log(f"[done] task_id={task_id} category={classification.get('category_id')} {classification.get('category_name')}")
        results.append(
            {
                "task_id": item.get("task_id"),
                "instruction": item.get("instruction"),
                "judge": compact_judge_item(item),
                "classification": classification,
            }
        )

    return {"input": str(args.input), "count": len(results), "results": results}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify MobileGym trajectory judge results into classify.md error classes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="mobilegym_skill_result.json path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output JSON path")
    parser.add_argument("--classify-md", type=Path, default=DEFAULT_CLASSIFY_MD)
    parser.add_argument("--indent", type=int, default=2, help="JSON indent; 0 writes compact JSON")
    parser.add_argument("--api-provider", choices=("qwen_plus", "doubao"), default="qwen_plus")
    parser.add_argument("--api-timeout", type=int, default=180, help="API request timeout in seconds")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    payload = process(args)
    write_json(args.output, payload, args.indent)
    log(f"[finish] output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
