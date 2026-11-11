"""
将 process_task 拆分为可复用的 stage，便于：
1) 用 PipelineSpec 控制启用顺序（lite / 全量 / 统一终判）
2) 支持消融时更清晰地替换 stage 逻辑
3) 统一 profile_id 的缓存/中间 prompt 隔离策略
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from prompts.task_classification import classify_task
from prompts.slot_extraction import extract_slots
from prompts.action_thought_caption import (
    generate_action_thought_caption_with_takeover,
    validate_actions_for_trajectory,
    validate_actions_and_thoughts_for_trajectory,
)
from prompts.keyframe_extraction import extract_keyframes
from prompts.task_completion_judgment import (
    judge_task_completion,
    load_trajectory_correctness_strategy,
    build_trajectory_correctness_judgment_prompt,
)
from prompts.lite_judgment_prompt import get_lite_judgment_prompt
from prompts.lite_trajectory_unified_prompt import (
    get_lite_trajectory_unified_binary_prompt,
)
from prompts.mobilegym_skill_trajectory_prompt import (
    MOBILEGYM_SKILL_TRAJECTORY_PROMPT_VERSION,
    get_mobilegym_skill_trajectory_prompt,
)
from prompts.error_cause_analysis_prompt import get_error_cause_analysis_prompt
from prompts.judgment_config import STATUS_MAPPING

from utils.prompt_utils import (
    load_from_cache,
    save_to_cache,
    set_task_context,
    get_image_path_from_step,
    call_llm_with_prompt,
    parse_json_response,
    get_prompt_response_path,
    PayloadTooLargeError,
)
from utils.lite_prompt_knowledge import (
    render_trajectory_section,
    render_completion_section,
    render_negative_task_section,
    render_takeover_knowledge_section,
    takeover_knowledge_hash,
    normalize_takeover_judge_mode,
)
from utils.task_knowledge import (
    render_task_knowledge_section,
    search_task_knowledge,
)
from utils.mobilegym_skill_knowledge import get_mobilegym_skill_common_knowledge_and_version


LITE_LAST_N_IMAGES = 10
LITE_GATE_PROMPT_VERSION = "devfix_takeover_current_page_boundary_v1"
ERROR_CAUSE_PROMPT_VERSION = "devfix_takeover_current_page_boundary_v1"


def _render_validation_diff_section(action_thought_results: List[Dict[str, Any]]) -> str:
    """Render changed action-validation records for lite gate prompts."""
    lines: List[str] = []
    for i, step in enumerate(action_thought_results):
        if not isinstance(step, dict) or not step.get("action_changed"):
            continue
        original_action = (step.get("original_action") or "").strip()
        validated_action = (step.get("validated_action") or step.get("action") or "").strip()
        if not original_action or not validated_action or original_action == validated_action:
            continue
        line = f"{i + 1}. 原始操作: {original_action} -> 校验后操作: {validated_action}"
        first_pass = (step.get("first_pass_corrected_action") or "").strip()
        if first_pass and first_pass != validated_action:
            line += f"；一轮修正: {first_pass}"
        validation_reason = (step.get("validation_reason") or "").strip()
        if validation_reason:
            line += f"；一轮理由: {validation_reason}"
        double_check_choice = step.get("double_check_choice")
        double_check_reason = (step.get("double_check_reason") or "").strip()
        if double_check_choice is not None or double_check_reason:
            line += f"；二次选择: {double_check_choice if double_check_choice is not None else '未知'}"
            if double_check_reason:
                line += f"；二次理由: {double_check_reason}"
        lines.append(line)
    return "\n".join(lines)


def _is_usable_image_path(path: str) -> bool:
    """图片输入可用性判断：本地存在文件，或为 URL。"""
    if not isinstance(path, str):
        return False
    p = path.strip()
    if not p:
        return False
    if p.lower().startswith(("http://", "https://")):
        return True
    return os.path.exists(p)


_PRIMARY_ERROR_CAUSES = {
    "环境或系统问题",
    "感知/Grounding能力不足",
    "背景知识不足",
    "任务理解/推理规划错误",
    "其他错误",
}

_SECONDARY_ERROR_CAUSES = {
    "环境或系统问题": {
        "登录问题",
        "人机验证",
        "前置条件状态不满足",
        "系统级异常",
        "目标应用未安装",
        "接续模块问题",
        "接管问题",
        "润色模块问题",
    },
    "感知/Grounding能力不足": {
        "元素未识别",
        "元素识别错误",
        "界面状态理解错误",
        "元素Grounding错误",
        "界面元素幻觉",
        "接管问题",
    },
    "背景知识不足": {
        "功能入口或页面结构理解错误",
        "功能用途理解错误",
        "操作空间不支持",
    },
    "任务理解/推理规划错误": {
        "欠执行",
        "过度执行",
        "推理规划错误",
        "query模糊问题",
        "接续指令跟随问题",
        "负例识别问题",
        "terminate总结问题",
    },
    "其他错误": {"其他错误"},
}

_DEFAULT_SECONDARY_ERROR_CAUSE = {
    "环境或系统问题": "系统级异常",
    "感知/Grounding能力不足": "元素未识别",
    "背景知识不足": "功能入口或页面结构理解错误",
    "任务理解/推理规划错误": "推理规划错误",
    "其他错误": "其他错误",
}


def _stringify_jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _normalize_error_cause_output(parsed: Any) -> Dict[str, str]:
    if not isinstance(parsed, dict):
        return {
            "primary_error_cause": "其他错误",
            "secondary_error_cause": "其他错误",
            "error_cause_reasoning": "模型输出无法解析，按其他错误兜底。",
            "takeover_missing": False,
            "takeover_overuse": False,
            "takeover_type_correct": None,
            "expected_takeover_type": "",
            "takeover_diagnosis": "",
        }

    primary = _stringify_jsonish(
        parsed.get("primary_error_cause")
        or parsed.get("primary_failure")
        or parsed.get("一级错因")
    )
    secondary = _stringify_jsonish(
        parsed.get("secondary_error_cause")
        or parsed.get("failure_subtype")
        or parsed.get("二级错因")
    )
    reasoning = _stringify_jsonish(
        parsed.get("error_cause_reasoning")
        or parsed.get("reasoning")
        or parsed.get("错因分类理由")
        or parsed.get("evidence")
    )
    evidence = _stringify_jsonish(parsed.get("evidence"))

    if primary not in _PRIMARY_ERROR_CAUSES:
        primary = "其他错误"
    allowed_secondary = _SECONDARY_ERROR_CAUSES.get(primary, {"其他错误"})
    if secondary not in allowed_secondary:
        secondary = _DEFAULT_SECONDARY_ERROR_CAUSE.get(primary, "其他错误")
    if not reasoning:
        reasoning = evidence or "模型未给出有效错因分类理由。"
    elif evidence and evidence not in reasoning:
        reasoning = f"{reasoning}\n证据：{evidence}"

    type_correct = parsed.get("takeover_type_correct")
    if type_correct not in (True, False, None):
        type_correct = None

    return {
        "primary_error_cause": primary,
        "secondary_error_cause": secondary,
        "error_cause_reasoning": reasoning,
        "takeover_missing": bool(parsed.get("takeover_missing", False)),
        "takeover_overuse": bool(parsed.get("takeover_overuse", False)),
        "takeover_type_correct": type_correct,
        "expected_takeover_type": _stringify_jsonish(parsed.get("expected_takeover_type")),
        "takeover_diagnosis": _stringify_jsonish(parsed.get("takeover_diagnosis")),
    }


def _build_prior_reasoning_section(prior_task_status: str, prior_reasoning: str) -> str:
    parts: List[str] = []
    if prior_task_status:
        parts.append(f"先验 task_status: {prior_task_status}")
    if prior_reasoning:
        parts.append(f"先验 reasoning: {prior_reasoning}")
    return "\n".join(parts)


def _build_prm_context_section(
    *,
    prm_mode: bool,
    key_step_indices: Optional[List[int]],
    invalid_key_step_indices: Optional[List[int]] = None,
) -> str:
    if not prm_mode:
        return ""
    valid = key_step_indices or []
    invalid = invalid_key_step_indices or []
    parts = [
        "当前样本 metadata.list_key_index 非空，启用 PRM关键步骤模式。",
        "本次只关心指定关键/残缺步骤内首次出现的问题；完整轨迹仅作上下文。",
        "不要基于部分轨迹输出完整任务完成度、完整轨迹正确性或 task_status 结论。",
        "lite gate 的结果只作为辅助线索，无论其成功或失败都必须继续做错因分类。",
    ]
    parts.append("关键步骤（1-based）: " + (", ".join(str(i) for i in valid) if valid else "无有效索引"))
    if invalid:
        parts.append("忽略的无效关键步骤索引: " + ", ".join(str(i) for i in invalid))
    return "\n".join(f"- {p}" for p in parts)


def _build_raw_trajectory_steps_text(trajectory: List[Dict]) -> str:
    lines: List[str] = []
    for i, step in enumerate(trajectory):
        value = step.get("value") if isinstance(step.get("value"), dict) else {}
        fields = [
            f"Step {i}:",
            f"- activity: {_stringify_jsonish(step.get('activity')) or '(空)'}",
            f"- thought: {_stringify_jsonish(value.get('thought')) or '(空)'}",
            f"- action: {_stringify_jsonish(value.get('action')) or '(空)'}",
        ]
        for key in ("code", "tool_call", "normalized_tool_call", "extras"):
            text = _stringify_jsonish(value.get(key))
            if text:
                fields.append(f"- {key}: {text}")
        take_over = step.get("take_over")
        if take_over:
            fields.append(f"- take_over: {_stringify_jsonish(take_over)}")
        user_interrupt = _stringify_jsonish(step.get("user_interrupt"))
        if user_interrupt:
            fields.append(f"- user_interrupt: {user_interrupt}")
        takeover_source = step.get("takeover_source")
        if takeover_source:
            fields.append(f"- takeover_source: {_stringify_jsonish(takeover_source)}")
        lines.append("\n".join(fields))
    return "\n\n".join(lines)


def _build_takeover_continuation_context(trajectory: List[Dict]) -> str:
    """Render real user continuation/interrupt inputs for lite gate prompts."""
    lines: List[str] = []
    for i, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        take_over = step.get("take_over")
        if isinstance(take_over, dict):
            user_input = _stringify_jsonish(take_over.get("input"))
            if user_input:
                category = _stringify_jsonish(take_over.get("category")) or "未知"
                continue_goal = _stringify_jsonish(take_over.get("continue_goal")) or "未知"
                summarize = _stringify_jsonish(take_over.get("summarize"))
                takeover_source = _stringify_jsonish(step.get("takeover_source")) or "未知"
                line = (
                    f"- Step {i}: source='take_over.input', takeover_source={takeover_source!r}, "
                    f"category={category!r}, continue_goal={continue_goal!r}, input={user_input!r}"
                )
                if summarize:
                    line += f", summarize={summarize!r}"
                lines.append(line)
        user_interrupt = _stringify_jsonish(step.get("user_interrupt"))
        if user_interrupt:
            lines.append(f"- Step {i}: source='user_interrupt', input={user_interrupt!r}")
    if not lines:
        return ""
    return "\n".join([
        "轨迹中存在真实用户输入。take_over.input 是用户在接管后给出的输入；user_interrupt 是用户随时打断说的话。二者都不是 Agent 自报。",
        "判定时必须先判断该输入与当时上下文的关系：可能是在回答/补充上一步接管或追问，也可能是突然打断并开启新任务，或要求并行处理多个任务。",
        "若输入是在补充上一步接管/追问，则从该 step 之后的任务目标必须合并原始 instruction 与该输入中的新增约束，不能只看原始 instruction。",
        "若输入是新任务或并行任务，默认 Agent 应按用户最新要求执行新任务或协调多个任务；完成其中一个后，除非用户明确取消，仍应继续处理此前未完成的任务。",
        "真实用户输入只改变后续应执行的目标和约束，不直接构成完成证据；最终仍必须按截图结果态核验。",
        *lines,
    ])


def _build_unified_knowledge_rules_and_version() -> Tuple[str, str]:
    """
    构建 unified 二分类 prompt 使用的 knowledge 规则文本，并生成短哈希版本号。
    版本号用于缓存隔离：knowledge 变化或 prompt 模板变化时都应触发新 cache key。
    """
    try:
        traj_rules = render_trajectory_section()
        comp_rules = render_completion_section()
        rules_text = (
            f"{traj_rules}\n\n---\n\n{comp_rules}".strip()
            if traj_rules or comp_rules
            else ""
        )
        # 把 prompt 模板源文件也纳入 version 计算，模板里 Hard Constraints / 输出 schema
        # 改动时同步触发 cache 失效，避免用旧 prompt 跑出来的缓存覆盖新规则的真实判定。
        prompt_template_signature = ""
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            tmpl_path = os.path.normpath(
                os.path.join(here, "..", "prompts", "lite_trajectory_unified_prompt.py")
            )
            with open(tmpl_path, "rb") as f:
                prompt_template_signature = hashlib.md5(f.read()).hexdigest()[:10]
        except Exception:
            prompt_template_signature = "tmplerr"

        if not rules_text and not prompt_template_signature:
            return "", "kbnone"
        version_seed = (rules_text + "||" + prompt_template_signature).encode("utf-8")
        version = hashlib.md5(version_seed).hexdigest()[:10]
        return rules_text, version
    except Exception:
        return "", "kberror"


def _trajectory_reason_from_parsed_or_text(text: Optional[str]) -> str:
    """从大模型返回的 JSON 文本中解析 trajectory prompt 的 reason 字段。"""
    if not text or not str(text).strip():
        return ""
    parsed = parse_json_response(str(text))
    if not isinstance(parsed, dict):
        return ""
    r = parsed.get("reason")
    if isinstance(r, str):
        return r.strip()
    if r is None:
        return ""
    return str(r).strip()


def _load_trajectory_reasoning_from_tmp2(task_id: str, cache_name: str) -> str:
    """缓存 hit 但 tmp 无 reasoning 时，从 tmp2 的 prompt_response 补全。"""
    p = get_prompt_response_path(task_id, cache_name)
    if not os.path.exists(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            pr = json.load(f)
        return _trajectory_reason_from_parsed_or_text(pr.get("response_text"))
    except Exception:
        return ""


def _lite_reason_from_parsed_or_text(text: Optional[str]) -> str:
    """从 lite 综合判断 prompt 返回的 JSON 文本中解析 reason 字段。"""
    if not text or not str(text).strip():
        return ""
    parsed = parse_json_response(str(text))
    if not isinstance(parsed, dict):
        return ""
    r = parsed.get("reason")
    if isinstance(r, str):
        return r.strip()
    if r is None:
        return ""
    return str(r).strip()


def _load_lite_reasoning_from_tmp2(task_id: str, cache_name: str) -> str:
    """缓存 hit 但 tmp 无 lite reasoning 时，从 tmp2 的 prompt_response 补全。"""
    p = get_prompt_response_path(task_id, cache_name)
    if not os.path.exists(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            pr = json.load(f)
        return _lite_reason_from_parsed_or_text(pr.get("response_text"))
    except Exception:
        return ""


def _to_binary_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        i = int(v)
        return i if i in (0, 1) else None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "correct", "completed", "pass"):
            return 1
        if s in ("0", "false", "no", "incorrect", "not_completed", "fail", "failed", "uncertain", "-1"):
            return 0
    return None


def _is_negative_one_value(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return int(v) == -1
    if isinstance(v, str):
        return v.strip() == "-1"
    return False


def _to_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    b = _to_binary_int(v)
    if b is None:
        return None
    return bool(b)


def _normalize_task_knowledge_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
    hit = parsed.get("task_knowledge_hit")
    if not isinstance(hit, bool):
        hit = str(hit).strip().lower() in {"true", "1", "yes"}
    conf = parsed.get("task_knowledge_confidence", 0.0)
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = max(0.0, min(1.0, conf_f))
    used = parsed.get("task_knowledge_used", "")
    reasoning = parsed.get("task_knowledge_reasoning", "")
    return {
        "task_knowledge_hit": bool(hit),
        "task_knowledge_confidence": conf_f,
        "task_knowledge_used": used.strip() if isinstance(used, str) else ("" if used is None else str(used).strip()),
        "task_knowledge_reasoning": reasoning.strip() if isinstance(reasoning, str) else ("" if reasoning is None else str(reasoning).strip()),
    }


def _attach_task_knowledge_match(judgment: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(judgment, dict):
        return judgment
    if isinstance(match, dict):
        judgment["task_knowledge_match"] = match
        if match.get("hit") and not judgment.get("task_knowledge_hit"):
            judgment["task_knowledge_hit"] = True
        if match.get("hit") and not judgment.get("task_knowledge_confidence"):
            judgment["task_knowledge_confidence"] = match.get("confidence", 0.0)
        if match.get("hit") and not judgment.get("task_knowledge_used"):
            judgment["task_knowledge_used"] = match.get("matched_query", "")
    return judgment


VALID_UNIFIED_TASK_STATUSES = frozenset({"perfect", "fair_enough", "failed"})
VALID_COMPLETION_EVIDENCE_SOURCES = frozenset({
    "screenshot_result_state",
    "screenshot_intermediate_state",
    "tool_or_system_result",
    "agent_self_report_only",
    "none",
})
VALID_TERMINAL_STATE_TYPES = frozenset({
    "result_state",
    "intermediate_state",
    "candidate_list",
    "blocked_state",
    "unknown",
})
VALID_FAIR_ENOUGH_EXCEPTIONS = frozenset({
    "none",
    "last_step_only",
    "executed_but_not_effective",
    "verified_unavailable_with_honest_stop",
})
FAIR_ENOUGH_EXCEPTION_ALIASES = {}
VALID_ENV_EVIDENCE_TYPES = frozenset({
    "none",
    "objective_error_page",
    "repeated_center_click_no_change",
    "login_or_verification_block",
    "private_data_empty",
    "app_missing",
    "payment_membership_boundary",
})
VALID_ENV_EXCLUSION_HITS = frozenset({
    "none",
    "terminate_success",
    "wrong_path",
    "visible_target_missed",
    "insufficient_retry",
    "normal_search_empty",
})
VALID_TAKEOVER_EVIDENCE_SOURCES = frozenset({
    "none",
    "explicit_take_over",
    "display_fields",
    "compressed_summary",
})
VALID_MANUAL_BOUNDARY_EVIDENCE_TYPES = frozenset({
    "none",
    "required_input_form_visible",
    "login_or_verification_visible",
    "payment_or_irreversible_confirm_visible",
    "authorization_or_agreement_visible",
    "ambiguous_choice_confirm_visible",
    "objective_blocked_state_visible",
    "process_entry_or_intermediate",
})
VALID_MANUAL_BOUNDARY_ENTRY_STATES = frozenset({
    "not_applicable",
    "process_entry_not_clicked",
    "process_entry_clicked_but_next_page_unknown",
    "process_entry_clicked_and_manual_boundary_visible",
    "manual_boundary_visible_without_entry_click",
    "objective_blocked_without_entry",
})


def _normalize_int_list(value: Any, *, step_count: int) -> List[int]:
    if not isinstance(value, list):
        return []
    out: List[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            idx = item
        elif isinstance(item, str) and re.fullmatch(r"\d+", item.strip()):
            idx = int(item.strip())
        else:
            continue
        if 0 <= idx < step_count and idx not in out:
            out.append(idx)
    return out


def _normalize_takeover_assessment(parsed: Dict[str, Any], *, step_count: int) -> Dict[str, Any]:
    raw = parsed.get("takeover_assessment")
    data: Dict[str, Any] = raw if isinstance(raw, dict) else {}
    has_takeover = _to_bool(data.get("has_takeover"))
    takeover_evidence_source = _enum_value(
        data.get("takeover_evidence_source"),
        VALID_TAKEOVER_EVIDENCE_SOURCES,
        "none",
    )
    has_over_takeover = _to_bool(data.get("has_over_takeover"))
    over_steps = _normalize_int_list(data.get("over_takeover_steps"), step_count=step_count)
    type_error_steps = _normalize_int_list(data.get("takeover_type_error_steps"), step_count=step_count)
    missing_steps = _normalize_int_list(data.get("takeover_missing_steps"), step_count=step_count)
    manual_intervention_clues = data.get("manual_intervention_clues")
    if not isinstance(manual_intervention_clues, list):
        manual_intervention_clues = []
    manual_intervention_clues = [
        str(item).strip()
        for item in manual_intervention_clues
        if str(item).strip()
    ]
    reason = data.get("reason", "")
    if not isinstance(reason, str):
        reason = "" if reason is None else str(reason)
    if has_takeover is None:
        has_takeover = bool(over_steps or type_error_steps or missing_steps or manual_intervention_clues)
    if manual_intervention_clues:
        has_takeover = True
    if manual_intervention_clues and takeover_evidence_source == "none":
        takeover_evidence_source = "compressed_summary"
    if has_over_takeover is None:
        has_over_takeover = bool(over_steps)
    return {
        "has_takeover": bool(has_takeover),
        "takeover_evidence_source": takeover_evidence_source,
        "manual_intervention_clues": manual_intervention_clues,
        "has_over_takeover": bool(has_over_takeover),
        "over_takeover_steps": over_steps,
        "takeover_type_error_steps": type_error_steps,
        "takeover_missing_steps": missing_steps,
        "reason": reason.strip(),
    }


def _enum_value(value: Any, allowed: frozenset[str], default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        normalized = FAIR_ENOUGH_EXCEPTION_ALIASES.get(normalized, normalized)
        if normalized in allowed:
            return normalized
    return default


def _resolve_unified_task_status(
    *,
    success: int,
    is_env_error: bool,
    raw_task_status: Any = None,
) -> str:
    """将 unified 判定归一化为 success/failed task_status。"""
    if is_env_error or success == -1:
        return "failed"
    if success == 1:
        ts = raw_task_status.strip().lower() if isinstance(raw_task_status, str) else ""
        if ts == "fair_enough":
            return "fair_enough"
        return "perfect"
    return "failed"


def _coerce_env_error_to_failed(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """兼容旧缓存/旧模型输出：把 env_error/-1 归一为 failed。"""
    if not isinstance(judgment, dict):
        return judgment
    task_status = judgment.get("task_status")
    has_env_error_status = isinstance(task_status, str) and task_status.strip().lower() == "env_error"
    if not (
        judgment.get("is_env_error")
        or has_env_error_status
        or _is_negative_one_value(judgment.get("final_judgment"))
        or _is_negative_one_value(judgment.get("is_task_success"))
    ):
        return judgment

    judgment = dict(judgment)
    judgment["is_env_error"] = False
    judgment["env_error_type"] = ""
    judgment["env_error_reason"] = ""
    judgment["task_status"] = "failed"
    if "final_judgment" in judgment:
        judgment["final_judgment"] = 0
    if "is_task_success" in judgment:
        judgment["is_task_success"] = 0
    existing_note = judgment.get("normalization_note")
    note = "旧 env_error/-1 缓存按当前二分类评估归一为 failed"
    judgment["normalization_note"] = f"{existing_note}；{note}" if isinstance(existing_note, str) and existing_note.strip() else note
    return judgment


def _normalize_unified_binary_output(parsed: Dict[str, Any], step_count: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    normalization_notes: List[str] = []

    # 兼容旧 prompt/cache 中的 env_error/-1；当前评估只输出成功或失败。
    env_evidence_type = _enum_value(
        parsed.get("env_evidence_type"),
        VALID_ENV_EVIDENCE_TYPES,
        "none",
    )
    env_exclusion_hit = _enum_value(
        parsed.get("env_exclusion_hit"),
        VALID_ENV_EXCLUSION_HITS,
        "none",
    )
    is_env_error_raw = parsed.get("is_env_error")
    is_env_error = bool(is_env_error_raw) if isinstance(is_env_error_raw, bool) else (
        str(is_env_error_raw).strip().lower() in ("true", "1", "yes") if is_env_error_raw is not None else False
    )
    raw_task_status = parsed.get("task_status")
    raw_task_status_norm = raw_task_status.strip().lower() if isinstance(raw_task_status, str) else ""
    if (
        raw_task_status_norm == "env_error"
        or _is_negative_one_value(parsed.get("is_task_success"))
        or _is_negative_one_value(parsed.get("final_judgment"))
    ):
        is_env_error = True
    if is_env_error:
        parsed = dict(parsed)
        parsed["is_env_error"] = False
        parsed["env_error_type"] = ""
        parsed["env_error_reason"] = ""
        parsed["is_task_success"] = 0
        parsed["final_judgment"] = 0
        parsed["task_status"] = "failed"
        is_env_error = False
        normalization_notes.append("模型输出 env_error/-1，当前二分类评估按 failed 处理")

    out["is_env_error"] = False
    out["env_error_type"] = ""
    out["env_error_reason"] = ""
    out["env_evidence_type"] = env_evidence_type
    out["env_exclusion_hit"] = env_exclusion_hit

    reason = parsed.get("reason")
    out["reason"] = reason.strip() if isinstance(reason, str) else ("" if reason is None else str(reason).strip())
    for k in ("trajectory_analysis", "completion_analysis", "task_decomposition", "key_step_analysis"):
        v = parsed.get(k, "")
        out[k] = v.strip() if isinstance(v, str) else ("" if v is None else str(v).strip())

    constraints = parsed.get("constraints")
    if isinstance(constraints, dict):
        hard_constraints = constraints.get("hard_constraints", [])
        subgoals = constraints.get("subgoals", [])
        out["constraints"] = {
            "hard_constraints": hard_constraints if isinstance(hard_constraints, list) else [],
            "subgoals": subgoals if isinstance(subgoals, list) else [],
        }
    else:
        out["constraints"] = {"hard_constraints": [], "subgoals": []}

    evidence = parsed.get("evidence")
    if isinstance(evidence, dict):
        success_steps = evidence.get("success_steps", [])
        failure_steps = evidence.get("failure_steps", [])
        missing_requirements = evidence.get("missing_requirements", [])
        out["evidence"] = {
            "success_steps": success_steps if isinstance(success_steps, list) else [],
            "failure_steps": failure_steps if isinstance(failure_steps, list) else [],
            "missing_requirements": missing_requirements if isinstance(missing_requirements, list) else [],
        }
    else:
        out["evidence"] = {"success_steps": [], "failure_steps": [], "missing_requirements": []}

    trajectory_pass = _to_bool(parsed.get("trajectory_pass"))
    completion_pass = _to_bool(parsed.get("completion_pass"))
    traj_j = _to_binary_int(parsed.get("trajectory_correctness_judgment"))
    comp_j = _to_binary_int(parsed.get("completion_judgment"))
    success = _to_binary_int(parsed.get("is_task_success"))
    final_j = _to_binary_int(parsed.get("final_judgment"))

    if traj_j is None and trajectory_pass is not None:
        traj_j = 1 if trajectory_pass else 0
    if comp_j is None and completion_pass is not None:
        comp_j = 1 if completion_pass else 0
    if trajectory_pass is None and traj_j is not None:
        trajectory_pass = bool(traj_j)
    if completion_pass is None and comp_j is not None:
        completion_pass = bool(comp_j)

    if success is None and final_j is not None:
        success = final_j
    if final_j is None and success is not None:
        final_j = success
    if success is None:
        if traj_j is not None and comp_j is not None:
            success = 1 if (traj_j == 1 and comp_j == 1) else 0
        else:
            # 二分类模式下兜底保守失败
            success = 0
    if final_j is None:
        final_j = success

    # ------------------------------------------------------------------
    # 结构一致性校验（仅读取模型自己填的结构化枚举字段，不做任何自由文本/关键词启发式后处理）：
    # 模型若自报「completion_pass=true 主要依据 Agent 自报文本」，与「截图为第一证据」直接矛盾，
    # 按它自己填的字段把 completion_pass 降为 false。
    # 具体的完成度经验规则（安装进度、动作结果态、序位/范围、信息来源、子目标等）全部下沉到 prompt，
    # 由模型自行判断并决定 completion_pass，框架不再逐项硬否决。
    # ------------------------------------------------------------------
    ce_raw = parsed.get("completion_evidence")
    completion_evidence: Dict[str, Any] = ce_raw if isinstance(ce_raw, dict) else {}
    evidence_source = _enum_value(
        completion_evidence.get("evidence_source"),
        VALID_COMPLETION_EVIDENCE_SOURCES,
        "none",
    )
    terminal_state_type = _enum_value(
        completion_evidence.get("terminal_state_type"),
        VALID_TERMINAL_STATE_TYPES,
        "unknown",
    )
    fair_enough_exception = _enum_value(
        completion_evidence.get("fair_enough_exception"),
        VALID_FAIR_ENOUGH_EXCEPTIONS,
        "none",
    )
    ce_missing_requirements = completion_evidence.get("missing_requirements")
    if not isinstance(ce_missing_requirements, list):
        ce_missing_requirements = []
    evidence_missing_requirements = out.get("evidence", {}).get("missing_requirements", [])
    if not isinstance(evidence_missing_requirements, list):
        evidence_missing_requirements = []
    completion_evidence = dict(completion_evidence)
    completion_evidence["evidence_source"] = evidence_source
    completion_evidence["terminal_state_type"] = terminal_state_type
    completion_evidence["missing_requirements"] = ce_missing_requirements
    completion_evidence["fair_enough_exception"] = fair_enough_exception
    ce_self_report_only = completion_evidence.get("evidence_derived_from_agent_self_report_only")
    ce_self_report_only_yes = (
        isinstance(ce_self_report_only, str)
        and ce_self_report_only.strip().lower() == "yes"
    )

    if (completion_pass is True or comp_j == 1) and ce_self_report_only_yes:
        completion_pass = False
        if comp_j == 1:
            comp_j = 0
        normalization_notes.append(
            "completion_pass 与自填字段冲突（evidence_derived_from_agent_self_report_only=yes），按模型自报降为 false"
        )

    success_like = completion_pass is True or comp_j == 1 or success == 1 or final_j == 1
    weak_or_missing_evidence = (
        evidence_source in {"agent_self_report_only", "none"}
        or terminal_state_type in {"intermediate_state", "candidate_list"}
        or bool(ce_missing_requirements)
        or bool(evidence_missing_requirements)
    )
    if success_like and weak_or_missing_evidence and fair_enough_exception == "none":
        completion_pass = False
        comp_j = 0
        normalization_notes.append(
            "success 与结构化完成证据冲突（弱证据/过程态/候选态/缺失项且无 fair_enough_exception），降级 completion_pass=false"
        )
    # 强一致约束：成功样本的轨迹与完成度子判定必须同时通过。
    if success == 1 and (traj_j == 0 or comp_j == 0):
        success = 0
        final_j = 0
        normalization_notes.append("is_task_success=1 与子判定冲突，强制降为失败")
    if success == 1:
        traj_j = 1
        comp_j = 1
        trajectory_pass = True
        completion_pass = True
    else:
        if traj_j is None and comp_j is None:
            traj_j = 0
            comp_j = 0
            normalization_notes.append("缺少子判定，二分类兜底置为失败")
        elif traj_j is None:
            traj_j = 0 if comp_j == 1 else comp_j
        elif comp_j is None:
            comp_j = 0 if traj_j == 1 else traj_j
        if traj_j == 1 and comp_j == 1:
            traj_j = 0
            normalization_notes.append("失败样本且子判定全为1，轨迹判定强制置0")
        trajectory_pass = bool(traj_j)
        completion_pass = bool(comp_j)
        final_j = 0
        success = 0

    out["trajectory_pass"] = bool(trajectory_pass)
    out["completion_pass"] = bool(completion_pass)
    out["trajectory_correctness_judgment"] = int(traj_j)
    out["completion_judgment"] = int(comp_j)
    out["is_task_success"] = int(success)
    out["final_judgment"] = int(final_j)
    out["task_status"] = _resolve_unified_task_status(
        success=int(success),
        is_env_error=False,
        raw_task_status=parsed.get("task_status"),
    )
    out["completion_evidence"] = completion_evidence if completion_evidence else {}

    first_deviation_step = parsed.get("first_deviation_step")
    if isinstance(first_deviation_step, str):
        s = first_deviation_step.strip()
        if re.fullmatch(r"-?\d+", s):
            try:
                first_deviation_step = int(s)
            except Exception:
                first_deviation_step = None
        else:
            first_deviation_step = None
    if not isinstance(first_deviation_step, int):
        first_deviation_step = None
    if isinstance(first_deviation_step, int):
        if first_deviation_step < 0 or first_deviation_step >= step_count:
            first_deviation_step = None

    first_deviation_reason = parsed.get("first_deviation_reason")
    if isinstance(first_deviation_reason, str):
        first_deviation_reason = first_deviation_reason.strip()
    elif first_deviation_reason is None:
        first_deviation_reason = ""
    else:
        first_deviation_reason = str(first_deviation_reason).strip()

    max_prefix = parsed.get("max_correct_prefix_end_step")
    if isinstance(max_prefix, str):
        s = max_prefix.strip()
        if re.fullmatch(r"-?\d+", s):
            try:
                max_prefix = int(s)
            except Exception:
                max_prefix = None
        else:
            max_prefix = None
    if not isinstance(max_prefix, int):
        max_prefix = None

    if out["trajectory_correctness_judgment"] == 0:
        if first_deviation_step is None:
            failure_steps = out["evidence"].get("failure_steps", [])
            if failure_steps and isinstance(failure_steps[0], int):
                fd = failure_steps[0]
                if 0 <= fd < step_count:
                    first_deviation_step = fd
            if first_deviation_step is None and step_count > 0:
                first_deviation_step = 0
            normalization_notes.append("失败样本缺首次偏航，已自动补全")
        if not first_deviation_reason:
            first_deviation_reason = "路径与任务意图不一致，存在未纠正偏航"
        if max_prefix is None:
            max_prefix = first_deviation_step - 1 if first_deviation_step is not None else -1
    else:
        first_deviation_step = None
        first_deviation_reason = ""
        if max_prefix is None:
            max_prefix = step_count - 1

    if max_prefix < -1:
        max_prefix = -1
    if step_count > 0 and max_prefix >= step_count:
        max_prefix = step_count - 1

    out["max_correct_prefix_end_step"] = int(max_prefix)
    if max_prefix >= 0:
        out["max_correct_prefix_steps"] = list(range(max_prefix + 1))
    else:
        out["max_correct_prefix_steps"] = []

    if first_deviation_step is not None:
        out["first_deviation_step"] = int(first_deviation_step)
        out["first_deviation_reason"] = first_deviation_reason

    lite_reason = out["reason"] or out["completion_analysis"]
    traj_reason = out.get("first_deviation_reason", "") or out["trajectory_analysis"] or out["reason"]
    out["lite_reasoning"] = f"<lite判定>{lite_reason}".strip() if lite_reason else ""
    out["trajectory_reasoning"] = f"<轨迹正确性判断原因>{traj_reason}".strip() if traj_reason else ""
    out["reasoning"] = out["trajectory_reasoning"] or out["lite_reasoning"]
    out.update(_normalize_task_knowledge_fields(parsed))
    if normalization_notes:
        out["normalization_note"] = "；".join(normalization_notes)
    return out


def _normalize_optional_step(value: Any, *, step_count: int) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        step = value
    elif isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        try:
            step = int(value.strip())
        except Exception:
            return None
    else:
        return None
    if step < 0 or step >= step_count:
        return None
    return step


def _normalize_mobilegym_skill_output(parsed: Dict[str, Any], step_count: int) -> Dict[str, Any]:
    out = _normalize_unified_binary_output(parsed, step_count)
    notes: List[str] = []
    allowed_failure_modes = {
        "wrong_action",
        "target_distractor_confusion",
        "special_control_error",
        "scroll_range_error",
        "popup_overlay_error",
        "completion_evidence_missing",
        "unclear",
        "env_error",
    }

    existing_note = out.get("normalization_note")
    if isinstance(existing_note, str) and existing_note.strip():
        notes.append(existing_note.strip())

    first_bad_state_step = _normalize_optional_step(
        parsed.get("first_bad_state_step"),
        step_count=step_count,
    )
    wrong_action_step = _normalize_optional_step(
        parsed.get("wrong_action_step"),
        step_count=step_count,
    )

    def text_field(name: str) -> str:
        value = parsed.get(name, "")
        return value.strip() if isinstance(value, str) else ("" if value is None else str(value).strip())

    if out.get("final_judgment") == 1:
        first_bad_state_step = None
        wrong_action_step = None
        can_extract_skill = False
        failure_mode_for_skill = ""
        skill_signal_type = ""
    else:
        can_extract_skill = bool(_to_bool(parsed.get("can_extract_skill")))
        failure_mode_for_skill = text_field("failure_mode_for_skill") or "unclear"
        skill_signal_type = text_field("skill_signal_type")
        if failure_mode_for_skill not in allowed_failure_modes:
            notes.append(
                f"failure_mode_for_skill={failure_mode_for_skill!r} 不在允许枚举内，已置为 unclear"
            )
            failure_mode_for_skill = "unclear"
        if not skill_signal_type:
            skill_signal_type = "unclear"
        if wrong_action_step is None:
            can_extract_skill = False
            failure_mode_for_skill = "unclear"
            notes.append("失败样本缺少有效 wrong_action_step，can_extract_skill 置为 false")
        elif not can_extract_skill:
            can_extract_skill = True
            notes.append("失败样本存在有效 wrong_action_step，can_extract_skill 置为 true")
        if first_bad_state_step is None:
            first_bad_state_step = wrong_action_step
            if first_bad_state_step is not None:
                notes.append("失败样本缺少有效 first_bad_state_step，已用 wrong_action_step 补全")

    out["first_bad_state_step"] = first_bad_state_step
    out["wrong_action_step"] = wrong_action_step
    out["wrong_action_reason"] = text_field("wrong_action_reason")
    out["step_alignment_reason"] = text_field("step_alignment_reason")
    out["failure_mode_for_skill"] = failure_mode_for_skill
    out["can_extract_skill"] = can_extract_skill
    out["skill_signal_type"] = skill_signal_type
    if notes:
        out["normalization_note"] = "；".join(notes)
    return out


def _apply_negative_task_overrides(
    judgment: Dict[str, Any], *, is_negative_task: bool
) -> Dict[str, Any]:
    """
    负例任务专用后处理：

    - 当 is_negative_task=True 时，把 trajectory_correctness_judgment / completion_judgment
      / trajectory_pass / completion_pass 全部置 -2 / None，表示「不适用」；
      下游输出层会因 -2 自动剔除该字段。
    - 正例任务（is_negative_task=False）原样返回。

    is_task_success / final_judgment 始终保留模型输出（1/0）。
    """
    if not is_negative_task or not isinstance(judgment, dict):
        return judgment

    judgment["trajectory_correctness_judgment"] = -2
    judgment["completion_judgment"] = -2
    judgment["trajectory_pass"] = None
    judgment["completion_pass"] = None
    judgment["task_kind"] = "negative"

    note = "负例：轨迹/完成度不适用，已置 -2"
    existing_note = judgment.get("normalization_note")
    if isinstance(existing_note, str) and existing_note.strip():
        judgment["normalization_note"] = f"{existing_note}；{note}"
    else:
        judgment["normalization_note"] = note
    return judgment


def run_error_cause_analysis_judgment(
    *,
    instruction: str,
    trajectory: List[Dict],
    task_id: str,
    use_cache: bool,
    cache_cleared: bool,
    verbose: bool,
    profile_id: str,
    compress_quality: Optional[float],
    prior_reasoning: str = "",
    prior_task_status: str = "",
    prm_mode: bool = False,
    key_step_indices: Optional[List[int]] = None,
    invalid_key_step_indices: Optional[List[int]] = None,
    judge_takeover: str = "end-to-end",
) -> Tuple[Dict[str, Any], bool]:
    """
    错因分析快路径：全图 + 原始 thought/action + 接管信息，不做轨迹正确性判定。
    """
    judge_takeover = normalize_takeover_judge_mode(judge_takeover)
    takeover_version = takeover_knowledge_hash(judge_takeover)
    prm_version = ""
    if prm_mode:
        prm_seed = json.dumps(
            {
                "key_step_indices": key_step_indices or [],
                "invalid_key_step_indices": invalid_key_step_indices or [],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        prm_version = "_prm" + hashlib.md5(prm_seed.encode("utf-8")).hexdigest()[:8]
    continuation_context_section = _build_takeover_continuation_context(trajectory)
    continuation_version = ""
    if continuation_context_section:
        continuation_version = "_ctx" + hashlib.md5(
            continuation_context_section.encode("utf-8")
        ).hexdigest()[:8]
    cache_name = f"judgment_error_cause_{profile_id}_{ERROR_CAUSE_PROMPT_VERSION}_tko{takeover_version}{continuation_version}{prm_version}"
    set_task_context(task_id, cache_name)

    cached = None
    if use_cache and task_id and not cache_cleared:
        cached = load_from_cache(task_id, cache_name)
    if (
        cached is not None
        and isinstance(cached, dict)
        and "primary_error_cause" in cached
        and "secondary_error_cause" in cached
        and "error_cause_reasoning" in cached
    ):
        tmp2_path = get_prompt_response_path(task_id, cache_name)
        if os.path.exists(tmp2_path):
            if verbose:
                print("  [使用缓存]")
            return _normalize_error_cause_output(cached), True
        if verbose:
            print("  [tmp缓存存在但tmp2缺失，重新执行LLM调用]")

    image_paths: List[str] = []
    for step in trajectory:
        try:
            p = get_image_path_from_step(step)
            if _is_usable_image_path(p):
                image_paths.append(p)
        except Exception:
            pass

    prompt = get_error_cause_analysis_prompt(
        instruction=instruction,
        steps_text=_build_raw_trajectory_steps_text(trajectory),
        prior_reasoning_section=_build_prior_reasoning_section(
            prior_task_status=prior_task_status,
            prior_reasoning=prior_reasoning,
        ),
        image_notice_section="**注意**：本任务将同时提供每一步截图（按时间顺序），请结合截图与原始轨迹信息做错因分类。",
        takeover_knowledge_section=render_takeover_knowledge_section(mode=judge_takeover),
        prm_context_section=_build_prm_context_section(
            prm_mode=prm_mode,
            key_step_indices=key_step_indices,
            invalid_key_step_indices=invalid_key_step_indices,
        ),
    )

    current_quality = compress_quality if compress_quality is not None else 0.6
    min_quality = 0.4
    quality_step = 0.05
    response = None
    while True:
        try:
            response = call_llm_with_prompt(
                prompt,
                images=image_paths if image_paths else None,
                temperature=0.01,
                max_tokens=1200,
                task_id=task_id,
                step_name=cache_name,
                compress_quality=current_quality,
            )
            break
        except PayloadTooLargeError:
            next_quality = round(current_quality - quality_step, 2)
            if next_quality < min_quality:
                if verbose:
                    print(f"  [413] 压缩系数已降至最低({current_quality})仍失败，按其他错误兜底")
                fallback = {
                    "primary_error_cause": "其他错误",
                    "secondary_error_cause": "其他错误",
                    "error_cause_reasoning": "请求体过大，无法完成可靠错因分析。",
                }
                if use_cache and task_id:
                    save_to_cache(task_id, cache_name, fallback)
                return fallback, False
            if verbose:
                print(f"  [413] 请求体过大，降低压缩系数 {current_quality} -> {next_quality} 重试...")
            current_quality = next_quality

    parsed = parse_json_response(response) if response else None
    judgment = _normalize_error_cause_output(parsed)
    if use_cache and task_id and response is not None:
        save_to_cache(task_id, cache_name, judgment)
    return judgment, False


def run_lite_trajectory_unified_all_images_binary_judgment(
    *,
    instruction: str,
    trajectory: List[Dict],
    task_id: str,
    use_cache: bool,
    cache_cleared: bool,
    verbose: bool,
    profile_id: str,
    lite_include_action_list: bool,
    lite_action_chain_source: str,
    compress_quality: Optional[float],
    is_negative_task: bool = False,
    client_report: Optional[Dict[str, Any]] = None,
    created_at: str = "",
    prm_mode: bool = False,
    key_step_indices: Optional[List[int]] = None,
    invalid_key_step_indices: Optional[List[int]] = None,
    use_task_knowledge: bool = True,
    judge_takeover: str = "end-to-end",
) -> Tuple[Dict[str, Any], bool]:
    knowledge_rules_section, knowledge_version = _build_unified_knowledge_rules_and_version()
    task_knowledge_match, task_knowledge_version = search_task_knowledge(
        instruction,
        task_id=task_id,
        step_name=f"task_knowledge_search_{profile_id}",
        enabled=use_task_knowledge,
    )
    task_knowledge_section = render_task_knowledge_section(task_knowledge_match)
    runtime_context_parts: List[str] = []
    created_at = created_at.strip() if isinstance(created_at, str) else ""
    if created_at:
        runtime_context_parts.append(
            f"- created_at: {created_at}（模型执行 GUI 任务的当前时间；遇到明天/今天/昨天/本周等相对时间指令时，以此为基准换算。）"
        )
    if isinstance(client_report, dict) and client_report:
        data_source = client_report.get("data_source", "")
        terminal_step_id = client_report.get("terminal_step_id", "")
        error_type = client_report.get("error_type", "")
        error_response = client_report.get("error_response", "")
        runtime_context_parts.append(
            "- client_report: "
            f"data_source={data_source!r}, terminal_step_id={terminal_step_id!r}, "
            f"error_type={error_type!r}, error_response={error_response!r}。"
        )
        runtime_context_parts.append(
            "  说明：client_report 是客户端/用户侧终止信息。若 error_type 为“用户主动停止/用户主动取消”或 error_response 为“用户退出”，"
            "它本身不等于环境错误；仍需根据截图和轨迹判断是否已达成任务、路径是否合理、是否只是完成度不足。"
        )
    runtime_context_section = "\n".join(runtime_context_parts)
    runtime_context_version = ""
    if runtime_context_section:
        runtime_context_version = "_ctx" + hashlib.md5(
            runtime_context_section.encode("utf-8")
        ).hexdigest()[:8]
    prm_context_section = _build_prm_context_section(
        prm_mode=prm_mode,
        key_step_indices=key_step_indices,
        invalid_key_step_indices=invalid_key_step_indices,
    )
    prm_version = ""
    if prm_context_section:
        prm_version = "_prm" + hashlib.md5(prm_context_section.encode("utf-8")).hexdigest()[:8]
    # task_kind 缓存隔离：正例保持原有 cache_name 以最大化历史缓存命中；负例追加 _neg 后缀
    kind_suffix = "_neg" if is_negative_task else ""
    cache_name = (
        f"judgment_unified_binary_allimg_{lite_action_chain_source}_{profile_id}_{LITE_GATE_PROMPT_VERSION}_{knowledge_version}_tk{task_knowledge_version}{kind_suffix}{runtime_context_version}{prm_version}"
    )
    set_task_context(task_id, cache_name)

    cached = None
    if use_cache and task_id and not cache_cleared:
        cached = load_from_cache(task_id, cache_name)
    if (
        cached is not None
        and isinstance(cached, dict)
        and "final_judgment" in cached
        and ("trajectory_correctness_judgment" in cached or cached.get("is_env_error"))
    ):
        tmp2_path = get_prompt_response_path(task_id, cache_name)
        if os.path.exists(tmp2_path):
            if verbose:
                print("  [使用缓存]")
            cached = _coerce_env_error_to_failed(cached)
            cached = _apply_negative_task_overrides(cached, is_negative_task=is_negative_task)
            cached = _attach_task_knowledge_match(cached, task_knowledge_match)
            return cached, True
        if verbose:
            print("  [tmp缓存存在但tmp2缺失，重新执行LLM调用]")

    image_paths: List[str] = []
    step_lines: List[str] = []
    for i, step in enumerate(trajectory):
        try:
            p = get_image_path_from_step(step)
            if _is_usable_image_path(p):
                image_paths.append(p)
        except Exception:
            pass
        action = step.get("value", {}).get("action", "")
        step_lines.append(f"Step {i}: action={action or '(空)'}")
    steps_text = "\n".join(step_lines)

    action_chain_text = ""
    thought_action_chain_text = ""
    validation_diff_section = ""
    if lite_include_action_list:
        if lite_action_chain_source == "raw":
            action_chain_text = "\n".join(
                f"{i+1}. {step.get('value', {}).get('action', '')}"
                for i, step in enumerate(trajectory)
                if step.get("value", {}).get("action", "")
            )
        elif lite_action_chain_source == "validated_thought":
            at_results = validate_actions_and_thoughts_for_trajectory(
                trajectory,
                task_id=task_id,
                use_cache=use_cache and (not cache_cleared),
                profile_id=profile_id,
            )
            thought_action_chain_text = "\n".join(
                f"{i+1}. 意图: {thought} -> 操作: {action}"
                if thought
                else f"{i+1}. 操作: {action}"
                for i, step in enumerate(at_results)
                for thought, action in [((step.get("thought", "") or "").strip(), (step.get("action", "") or "").strip())]
                if action
            )
            validation_diff_section = _render_validation_diff_section(at_results)
        else:
            validated_actions = validate_actions_for_trajectory(
                trajectory,
                task_id=task_id,
                use_cache=use_cache and (not cache_cleared),
                profile_id=profile_id,
            )
            action_chain_text = "\n".join(
                f"{i+1}. {act}" for i, act in enumerate(validated_actions) if act
            )

    task_kind = "negative" if is_negative_task else "positive"
    task_kind_section = render_negative_task_section() if is_negative_task else ""

    prompt = get_lite_trajectory_unified_binary_prompt(
        instruction=instruction,
        steps_text=steps_text,
        action_chain=action_chain_text,
        thought_action_chain=thought_action_chain_text,
        image_notice_section="**注意**：本任务将同时提供每一步截图（按时间顺序），请结合截图与轨迹信息做二分类判断。\n\n",
        knowledge_rules_section=knowledge_rules_section,
        action_chain_source=lite_action_chain_source,
        task_kind=task_kind,
        task_kind_section=task_kind_section,
        runtime_context_section=runtime_context_section,
        task_knowledge_section=task_knowledge_section,
        validation_diff_section=validation_diff_section,
        prm_context_section=prm_context_section,
    )

    current_quality = compress_quality if compress_quality is not None else 0.6
    min_quality = 0.4
    quality_step = 0.05
    response = None
    while True:
        try:
            response = call_llm_with_prompt(
                prompt,
                images=image_paths if image_paths else None,
                temperature=0.01,
                max_tokens=1800,
                task_id=task_id,
                step_name=cache_name,
                compress_quality=current_quality,
            )
            break
        except PayloadTooLargeError:
            next_quality = round(current_quality - quality_step, 2)
            if next_quality < min_quality:
                if verbose:
                    print(f"  [413] 压缩系数已降至最低({current_quality})仍失败，返回无效 lite gate")
                invalid = {
                    "error": "lite gate request payload too large after compression retries",
                    "normalization_note": "lite_gate_invalid_no_fallback",
                }
                invalid = _attach_task_knowledge_match(invalid, task_knowledge_match)
                return invalid, False
            if verbose:
                print(f"  [413] 请求体过大，降低压缩系数 {current_quality} -> {next_quality} 重试...")
            current_quality = next_quality

    parsed = parse_json_response(response) if response else None
    if not isinstance(parsed, dict):
        invalid = {
            "error": "lite gate produced no parseable JSON response",
            "normalization_note": "lite_gate_invalid_no_fallback",
        }
        invalid = _attach_task_knowledge_match(invalid, task_knowledge_match)
        return invalid, False

    judgment = _normalize_unified_binary_output(parsed, len(trajectory))
    judgment = _attach_task_knowledge_match(judgment, task_knowledge_match)
    judgment = _apply_negative_task_overrides(judgment, is_negative_task=is_negative_task)
    if use_cache and task_id and response is not None:
        save_to_cache(task_id, cache_name, judgment)
    return judgment, False


def run_mobilegym_skill_trajectory_judgment(
    *,
    instruction: str,
    trajectory: List[Dict],
    task_id: str,
    use_cache: bool,
    cache_cleared: bool,
    verbose: bool,
    profile_id: str,
    lite_include_action_list: bool,
    lite_action_chain_source: str,
    compress_quality: Optional[float],
    is_negative_task: bool = False,
    client_report: Optional[Dict[str, Any]] = None,
    created_at: str = "",
    prm_mode: bool = False,
    key_step_indices: Optional[List[int]] = None,
    invalid_key_step_indices: Optional[List[int]] = None,
    use_task_knowledge: bool = True,
    judge_takeover: str = "end-to-end",
) -> Tuple[Dict[str, Any], bool]:
    del judge_takeover
    knowledge_rules_section, knowledge_version = get_mobilegym_skill_common_knowledge_and_version()
    task_knowledge_match, task_knowledge_version = search_task_knowledge(
        instruction,
        task_id=task_id,
        step_name=f"task_knowledge_search_{profile_id}",
        enabled=use_task_knowledge,
    )
    task_knowledge_section = render_task_knowledge_section(task_knowledge_match)
    runtime_context_parts: List[str] = []
    created_at = created_at.strip() if isinstance(created_at, str) else ""
    if created_at:
        runtime_context_parts.append(
            f"- created_at: {created_at}（模型执行 GUI 任务的当前时间；遇到明天/今天/昨天/本周等相对时间指令时，以此为基准换算。）"
        )
    if isinstance(client_report, dict) and client_report:
        data_source = client_report.get("data_source", "")
        terminal_step_id = client_report.get("terminal_step_id", "")
        error_type = client_report.get("error_type", "")
        error_response = client_report.get("error_response", "")
        runtime_context_parts.append(
            "- client_report: "
            f"data_source={data_source!r}, terminal_step_id={terminal_step_id!r}, "
            f"error_type={error_type!r}, error_response={error_response!r}。"
        )
        runtime_context_parts.append(
            "  说明：client_report 是客户端/用户侧终止信息。若 error_type 为“用户主动停止/用户主动取消”或 error_response 为“用户退出”，"
            "它本身不等于任务完成或失败动作；仍需根据截图和轨迹判断是否已达成任务、路径是否合理、以及可行动错误步骤。"
        )
    runtime_context_section = "\n".join(runtime_context_parts)
    runtime_context_version = ""
    if runtime_context_section:
        runtime_context_version = "_ctx" + hashlib.md5(
            runtime_context_section.encode("utf-8")
        ).hexdigest()[:8]
    prm_context_section = _build_prm_context_section(
        prm_mode=prm_mode,
        key_step_indices=key_step_indices,
        invalid_key_step_indices=invalid_key_step_indices,
    )
    prm_version = ""
    if prm_context_section:
        prm_version = "_prm" + hashlib.md5(prm_context_section.encode("utf-8")).hexdigest()[:8]
    kind_suffix = "_neg" if is_negative_task else ""
    cache_name = (
        f"judgment_mobilegym_skill_allimg_{lite_action_chain_source}_{profile_id}_"
        f"{MOBILEGYM_SKILL_TRAJECTORY_PROMPT_VERSION}_{knowledge_version}_"
        f"tk{task_knowledge_version}{kind_suffix}{runtime_context_version}{prm_version}"
    )
    set_task_context(task_id, cache_name)

    cached = None
    if use_cache and task_id and not cache_cleared:
        cached = load_from_cache(task_id, cache_name)
    if (
        cached is not None
        and isinstance(cached, dict)
        and "final_judgment" in cached
        and ("wrong_action_step" in cached or cached.get("final_judgment") == 1)
    ):
        tmp2_path = get_prompt_response_path(task_id, cache_name)
        if os.path.exists(tmp2_path):
            if verbose:
                print("  [使用缓存]")
            cached = _coerce_env_error_to_failed(cached)
            cached = _apply_negative_task_overrides(cached, is_negative_task=is_negative_task)
            cached = _attach_task_knowledge_match(cached, task_knowledge_match)
            return cached, True
        if verbose:
            print("  [tmp缓存存在但tmp2缺失，重新执行LLM调用]")

    image_paths: List[str] = []
    step_lines: List[str] = []
    for i, step in enumerate(trajectory):
        try:
            p = get_image_path_from_step(step)
            if _is_usable_image_path(p):
                image_paths.append(p)
        except Exception:
            pass
        action = step.get("value", {}).get("action", "")
        step_lines.append(f"Step {i}: action={action or '(空)'}")
    steps_text = "\n".join(step_lines)

    action_chain_text = ""
    thought_action_chain_text = ""
    validation_diff_section = ""
    if lite_include_action_list:
        if lite_action_chain_source == "raw":
            action_chain_text = "\n".join(
                f"{i+1}. {step.get('value', {}).get('action', '')}"
                for i, step in enumerate(trajectory)
                if step.get("value", {}).get("action", "")
            )
        elif lite_action_chain_source == "validated_thought":
            at_results = validate_actions_and_thoughts_for_trajectory(
                trajectory,
                task_id=task_id,
                use_cache=use_cache and (not cache_cleared),
                profile_id=profile_id,
            )
            thought_action_chain_text = "\n".join(
                f"{i+1}. 意图: {thought} -> 操作: {action}"
                if thought
                else f"{i+1}. 操作: {action}"
                for i, step in enumerate(at_results)
                for thought, action in [((step.get("thought", "") or "").strip(), (step.get("action", "") or "").strip())]
                if action
            )
            validation_diff_section = _render_validation_diff_section(at_results)
        else:
            validated_actions = validate_actions_for_trajectory(
                trajectory,
                task_id=task_id,
                use_cache=use_cache and (not cache_cleared),
                profile_id=profile_id,
            )
            action_chain_text = "\n".join(
                f"{i+1}. {act}" for i, act in enumerate(validated_actions) if act
            )

    task_kind = "negative" if is_negative_task else "positive"
    task_kind_section = render_negative_task_section() if is_negative_task else ""

    prompt = get_mobilegym_skill_trajectory_prompt(
        instruction=instruction,
        steps_text=steps_text,
        action_chain=action_chain_text,
        thought_action_chain=thought_action_chain_text,
        image_notice_section="**注意**：本任务将同时提供每一步截图（按时间顺序），请结合截图与轨迹信息判断任务结果，并定位可行动错误步骤。\n\n",
        knowledge_rules_section=knowledge_rules_section,
        action_chain_source=lite_action_chain_source,
        task_kind=task_kind,
        task_kind_section=task_kind_section,
        runtime_context_section=runtime_context_section,
        task_knowledge_section=task_knowledge_section,
        validation_diff_section=validation_diff_section,
        prm_context_section=prm_context_section,
    )

    current_quality = compress_quality if compress_quality is not None else 0.6
    min_quality = 0.4
    quality_step = 0.05
    response = None
    while True:
        try:
            response = call_llm_with_prompt(
                prompt,
                images=image_paths if image_paths else None,
                temperature=0.01,
                max_tokens=2200,
                task_id=task_id,
                step_name=cache_name,
                compress_quality=current_quality,
            )
            break
        except PayloadTooLargeError:
            next_quality = round(current_quality - quality_step, 2)
            if next_quality < min_quality:
                if verbose:
                    print(f"  [413] 压缩系数已降至最低({current_quality})仍失败，返回无效 MobileGym judge")
                invalid = {
                    "error": "mobilegym skill trajectory request payload too large after compression retries",
                    "normalization_note": "mobilegym_skill_judge_invalid_no_fallback",
                }
                invalid = _attach_task_knowledge_match(invalid, task_knowledge_match)
                return invalid, False
            if verbose:
                print(f"  [413] 请求体过大，降低压缩系数 {current_quality} -> {next_quality} 重试...")
            current_quality = next_quality

    parsed = parse_json_response(response) if response else None
    if not isinstance(parsed, dict):
        invalid = {
            "error": "mobilegym skill trajectory judge produced no parseable JSON response",
            "normalization_note": "mobilegym_skill_judge_invalid_no_fallback",
        }
        invalid = _attach_task_knowledge_match(invalid, task_knowledge_match)
        return invalid, False

    judgment = _normalize_mobilegym_skill_output(parsed, len(trajectory))
    judgment = _attach_task_knowledge_match(judgment, task_knowledge_match)
    judgment = _apply_negative_task_overrides(judgment, is_negative_task=is_negative_task)
    if use_cache and task_id and response is not None:
        save_to_cache(task_id, cache_name, judgment)
    return judgment, False


def run_trajectory_correctness_all_images_judgment(
    *,
    instruction: str,
    trajectory: List[Dict],
    task_id: str,
    use_cache: bool,
    cache_cleared: bool,
    verbose: bool,
    profile_id: str,
    action_chain_source: str,
    compress_quality: Optional[float],
) -> Tuple[Dict[str, Any], bool]:
    """
    快路径轨迹正确性评估：全图 + action 链（raw/validated）+ 轨迹正确性 prompt。
    Returns: (judgment_dict, cache_used)；judgment 含 trajectory_correctness_judgment、final_judgment、reasoning（模型 JSON 的 reason）。
    """
    cache_name = f"judgment_traj_allimg_{action_chain_source}_{profile_id}"
    set_task_context(task_id, cache_name)

    cached = None
    if use_cache and task_id and not cache_cleared:
        cached = load_from_cache(task_id, cache_name)
    if (
        cached is not None
        and isinstance(cached, dict)
        and "trajectory_correctness_judgment" in cached
    ):
        tmp2_path = get_prompt_response_path(task_id, cache_name)
        if os.path.exists(tmp2_path):
            if verbose:
                print("  [使用缓存]")
            reasoning = cached.get("reasoning")
            if not (isinstance(reasoning, str) and reasoning.strip()):
                reasoning = _load_trajectory_reasoning_from_tmp2(task_id, cache_name)
            out = {
                "trajectory_correctness_judgment": cached["trajectory_correctness_judgment"],
                "final_judgment": cached.get(
                    "final_judgment", cached["trajectory_correctness_judgment"]
                ),
                "reasoning": reasoning if isinstance(reasoning, str) else "",
            }
            if isinstance(cached.get("first_deviation_step"), int):
                out["first_deviation_step"] = cached["first_deviation_step"]
            fdr = cached.get("first_deviation_reason")
            if isinstance(fdr, str) and fdr.strip():
                out["first_deviation_reason"] = fdr.strip()
            return out, True
        else:
            if verbose:
                print("  [tmp缓存存在但tmp2缺失，重新执行LLM调用]")
            cached = None

    image_paths: List[str] = []
    for step in trajectory:
        try:
            p = get_image_path_from_step(step)
            if _is_usable_image_path(p):
                image_paths.append(p)
        except Exception:
            pass

    if action_chain_source == "validated":
        actions = validate_actions_for_trajectory(
            trajectory,
            task_id=task_id,
            use_cache=use_cache and (not cache_cleared),
            profile_id=profile_id,
        )
    else:
        actions = [step.get("value", {}).get("action", "") for step in trajectory]

    action_thought_caption = [{"action": act} for act in actions]
    keyframe_indices = list(range(len(action_thought_caption)))
    keyframe_captions = ["" for _ in keyframe_indices]
    keyframe_actions = [step.get("action", "") for step in action_thought_caption]

    # 仅在“只判断轨迹正确性”的全图快路径使用旧版策略文件，避免影响其他评估分支。
    judge_config = load_trajectory_correctness_strategy("common_old.json")
    if not judge_config:
        return {
            "trajectory_correctness_judgment": -1,
            "final_judgment": -1,
            "reasoning": "",
        }, False

    prompt = build_trajectory_correctness_judgment_prompt(
        instruction=instruction,
        keyframe_captions=keyframe_captions,
        keyframe_actions=keyframe_actions,
        action_thought_caption=action_thought_caption,
        judge_config=judge_config,
        use_keyframe_images=bool(image_paths),
        keyframe_indices=keyframe_indices,
        all_steps_are_keyframes=True,
    )

    # HTTP 413 专用重试：逐步降低压缩系数，优先级高于通用重试
    current_quality = compress_quality if compress_quality is not None else 0.6
    min_quality = 0.4
    quality_step = 0.05
    response = None

    while True:
        try:
            response = call_llm_with_prompt(
                prompt,
                images=image_paths if image_paths else None,
                temperature=0.01,
                max_tokens=800,
                task_id=task_id,
                step_name=cache_name,
                compress_quality=current_quality,
            )
            break
        except PayloadTooLargeError:
            next_quality = round(current_quality - quality_step, 2)
            if next_quality < min_quality:
                if verbose:
                    print(f"  [413] 压缩系数已降至最低({current_quality})仍失败，跳过该任务")
                judgment = {
                    "trajectory_correctness_judgment": -1,
                    "final_judgment": -1,
                    "reasoning": "",
                }
                return judgment, False
            if verbose:
                print(f"  [413] 请求体过大，降低压缩系数 {current_quality} -> {next_quality} 重试...")
            current_quality = next_quality

    pred = -1
    reason_str = ""
    first_deviation_step = None
    first_deviation_reason = ""
    if response:
        parsed = parse_json_response(response)
        if isinstance(parsed, dict):
            status = parsed.get("status", -1)
            if isinstance(status, str):
                status = STATUS_MAPPING.get(status, -1)
            if status in (1, 0, -1):
                pred = status
            r = parsed.get("reason")
            if isinstance(r, str):
                reason_str = r.strip()
            elif r is not None:
                reason_str = str(r).strip()
            fds = parsed.get("first_deviation_step")
            if isinstance(fds, int):
                first_deviation_step = fds
            elif isinstance(fds, str):
                fds_strip = fds.strip()
                if re.fullmatch(r"-?\d+", fds_strip):
                    try:
                        first_deviation_step = int(fds_strip)
                    except Exception:
                        first_deviation_step = None
            fdr = parsed.get("first_deviation_reason")
            if isinstance(fdr, str):
                first_deviation_reason = fdr.strip()
            elif fdr is not None:
                first_deviation_reason = str(fdr).strip()

    reasoning = rf"<轨迹正确性判断原因>{reason_str}"
    if pred == 0 and isinstance(first_deviation_step, int):
        reasoning = rf"<轨迹正确性判断原因>[{first_deviation_step}] {reason_str}"

    judgment = {
        "trajectory_correctness_judgment": pred,
        "final_judgment": pred,
        "reasoning": reasoning,
    }
    if isinstance(first_deviation_step, int):
        judgment["first_deviation_step"] = first_deviation_step
    if first_deviation_reason:
        judgment["first_deviation_reason"] = first_deviation_reason
    if use_cache and task_id and response is not None:
        save_to_cache(task_id, cache_name, judgment)
    return judgment, False


def _normalize_lite_endframe_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """解析 lite 末帧判定 JSON，仅输出成功/失败二分类。"""
    is_env_error_raw = parsed.get("is_env_error")
    is_env_error = bool(is_env_error_raw) if isinstance(is_env_error_raw, bool) else (
        str(is_env_error_raw).strip().lower() in ("true", "1", "yes")
        if is_env_error_raw is not None
        else False
    )
    raw_task_status = parsed.get("task_status")
    raw_task_status_norm = raw_task_status.strip().lower() if isinstance(raw_task_status, str) else ""
    if (
        raw_task_status_norm == "env_error"
        or _is_negative_one_value(parsed.get("is_task_success"))
        or _is_negative_one_value(parsed.get("final_judgment"))
    ):
        is_env_error = True

    if is_env_error:
        parsed = dict(parsed)
        parsed["is_task_success"] = 0
        parsed["final_judgment"] = 0
        parsed["task_status"] = "failed"
        parsed["is_env_error"] = False
        parsed["env_error_type"] = ""
        parsed["env_error_reason"] = ""

    reason = parsed.get("reason")
    reason_str = reason.strip() if isinstance(reason, str) else ("" if reason is None else str(reason).strip())

    is_task_success = 0
    if "is_task_success" in parsed:
        v = parsed["is_task_success"]
        if isinstance(v, (int, float)):
            v = int(v)
        elif isinstance(v, str):
            try:
                v = int(v.strip())
            except ValueError:
                v = None
        else:
            v = None
        if v == -1:
            # 模型误输出 -1 但未标 is_env_error：保守按失败处理
            is_task_success = 0
        elif v in (0, 1):
            is_task_success = v

    raw_task_status = parsed.get("task_status", "")
    task_status = raw_task_status.strip() if isinstance(raw_task_status, str) else ""
    if is_task_success == 1:
        if task_status not in {"perfect", "fair_enough"}:
            task_status = "perfect"
    else:
        task_status = "failed"

    reasoning = f"<lite判定>{reason_str}".strip() if reason_str else ""
    out = {
        "final_judgment": is_task_success,
        "task_status": task_status,
        "is_env_error": False,
        "env_error_type": "",
        "env_error_reason": "",
        "reasoning": reasoning,
    }
    out.update(_normalize_task_knowledge_fields(parsed))
    return out


def run_lite_endframe_judgment(
    *,
    instruction: str,
    trajectory: List[Dict],
    task_id: str,
    use_cache: bool,
    cache_cleared: bool,
    verbose: bool,
    profile_id: str,
    lite_include_action_list: bool,
    lite_action_chain_source: str,
    compress_quality: Optional[float],
    use_task_knowledge: bool = True,
) -> Tuple[Dict[str, Any], bool]:
    """
    Lite eval：只做末尾综合判断。
    Returns: (judgment_dict, cache_used)
    """
    task_knowledge_match, task_knowledge_version = search_task_knowledge(
        instruction,
        task_id=task_id,
        step_name=f"task_knowledge_search_{profile_id}",
        enabled=use_task_knowledge,
    )
    task_knowledge_section = render_task_knowledge_section(task_knowledge_match)
    takeover_context_section = _build_takeover_continuation_context(trajectory)
    runtime_context_section = takeover_context_section
    runtime_context_version = ""
    if runtime_context_section:
        runtime_context_version = "_ctx" + hashlib.md5(
            runtime_context_section.encode("utf-8")
        ).hexdigest()[:8]
    last_n = min(LITE_LAST_N_IMAGES, len(trajectory))
    last_steps = trajectory[-last_n:] if last_n else []
    last_image_paths: List[str] = []
    for step in last_steps:
        try:
            p = get_image_path_from_step(step)
            if _is_usable_image_path(p):
                last_image_paths.append(p)
        except Exception:
            pass

    judgment_lite_cache_name = f"judgment_lite_{profile_id}_{LITE_GATE_PROMPT_VERSION}_tk{task_knowledge_version}{runtime_context_version}"
    set_task_context(task_id, judgment_lite_cache_name)

    judgment_lite = None
    if use_cache and task_id and not cache_cleared:
        judgment_lite = load_from_cache(task_id, judgment_lite_cache_name)
    if (
        judgment_lite is not None
        and isinstance(judgment_lite, dict)
        and "final_judgment" in judgment_lite
        and "task_status" in judgment_lite
    ):
        tmp2_path = get_prompt_response_path(task_id, judgment_lite_cache_name)
        if os.path.exists(tmp2_path):
            if verbose:
                print("  [使用缓存]")
            reasoning = judgment_lite.get("reasoning")
            if not (isinstance(reasoning, str) and reasoning.strip()):
                reasoning = _load_lite_reasoning_from_tmp2(task_id, judgment_lite_cache_name)
            reas = reasoning if isinstance(reasoning, str) else ""
            reas = reas.strip()
            if reas and not reas.startswith("<lite判定>"):
                reas = f"<lite判定>{reas}"
            out: Dict[str, Any] = {
                "final_judgment": judgment_lite["final_judgment"],
                "task_status": judgment_lite.get("task_status", ""),
                "reasoning": reas,
            }
            for _k in ("is_env_error", "env_error_type", "env_error_reason"):
                if _k in judgment_lite:
                    out[_k] = judgment_lite[_k]
            for _k in (
                "task_knowledge_hit",
                "task_knowledge_confidence",
                "task_knowledge_used",
                "task_knowledge_reasoning",
                "task_knowledge_match",
            ):
                if _k in judgment_lite:
                    out[_k] = judgment_lite[_k]
            out = _coerce_env_error_to_failed(out)
            out = _attach_task_knowledge_match(out, task_knowledge_match)
            return out, True
        else:
            if verbose:
                print("  [tmp缓存存在但tmp2缺失，重新执行LLM调用]")
            judgment_lite = None

    last_steps_text = "以下为轨迹末尾若干步的截图（按时间顺序提供），请重点结合最后几张图判断终态是否达成与路径是否正确。"
    image_notice = "**注意**：本任务将同时提供轨迹末尾的若干张截图（按时间顺序），请结合图片与任务指令判断。\n\n"

    action_chain_text = ""
    thought_action_chain_text = ""
    validation_diff_section = ""
    if lite_include_action_list:
        if lite_action_chain_source == "raw":
            action_chain_text = "\n".join(
                f"{i+1}. {step.get('value', {}).get('action', '')}"
                for i, step in enumerate(trajectory)
                if step.get("value", {}).get("action", "")
            )
        elif lite_action_chain_source == "validated_thought":
            at_results = validate_actions_and_thoughts_for_trajectory(
                trajectory,
                task_id=task_id,
                use_cache=use_cache and (not cache_cleared),
                profile_id=profile_id,
            )
            thought_action_chain_text = "\n".join(
                f"{i+1}. 意图: {thought} -> 操作: {action}"
                if thought
                else f"{i+1}. 操作: {action}"
                for i, step in enumerate(at_results)
                for thought, action in [((step.get("thought", "") or "").strip(), (step.get("action", "") or "").strip())]
                if action
            )
            validation_diff_section = _render_validation_diff_section(at_results)
        else:
            validated_actions = validate_actions_for_trajectory(
                trajectory,
                task_id=task_id,
                use_cache=use_cache and (not cache_cleared),
                profile_id=profile_id,
            )
            action_chain_text = "\n".join(
                f"{i+1}. {act}" for i, act in enumerate(validated_actions) if act
            )

    prompt = get_lite_judgment_prompt(
        instruction,
        last_steps_text=last_steps_text,
        action_chain=action_chain_text,
        thought_action_chain=thought_action_chain_text,
        image_notice_section=image_notice,
        runtime_context_section=runtime_context_section,
        inspection_section="",
        task_knowledge_section=task_knowledge_section,
        validation_diff_section=validation_diff_section,
    )

    response = call_llm_with_prompt(
        prompt,
        images=last_image_paths,
        temperature=0.01,
        max_tokens=1400,
        task_id=task_id,
        step_name=judgment_lite_cache_name,
        compress_quality=compress_quality,
    )

    judgment_out: Dict[str, Any] = {
        "final_judgment": 0,
        "task_status": "failed",
        "is_env_error": False,
        "env_error_type": "",
        "env_error_reason": "",
        "reasoning": "",
    }
    if response:
        parsed = parse_json_response(response)
        if isinstance(parsed, dict):
            judgment_out = _normalize_lite_endframe_output(parsed)
    judgment_out = _attach_task_knowledge_match(judgment_out, task_knowledge_match)

    if use_cache and task_id and response is not None:
        save_to_cache(
            task_id,
            judgment_lite_cache_name,
            judgment_out,
        )

    return judgment_out, False


def run_category_stage(
    *,
    instruction: str,
    task_id: str,
    use_cache: bool,
    cache_cleared: bool,
    verbose: bool,
) -> Tuple[Optional[str], bool]:
    set_task_context(task_id, "category")
    category = None
    if use_cache and task_id:
        category = load_from_cache(task_id, "category")
    if category is None:
        category = classify_task(instruction)
        if use_cache and task_id:
            save_to_cache(task_id, "category", category)
        cache_used = False
    else:
        cache_used = True
    if verbose:
        print(f"  分类：{category}")
    return category, cache_used


def run_slots_qa_stage(
    *,
    instruction: str,
    task_id: str,
    use_cache: bool,
    no_qa: bool,
    cache_cleared: bool,
    verbose: bool,
) -> Tuple[Dict, List[Dict], bool]:
    if no_qa:
        if verbose:
            print("  [no_qa] 跳过槽位/QA提取")
        return {}, [], False

    set_task_context(task_id, "slots_qa")
    slots = None
    qa_pairs = None
    if use_cache and task_id:
        cached_data = load_from_cache(task_id, "slots_qa")
        if cached_data:
            slots = cached_data.get("slots")
            qa_pairs = cached_data.get("qa_pairs")
    if slots is None or qa_pairs is None:
        slots, qa_pairs = extract_slots(instruction)
        if use_cache and task_id:
            save_to_cache(task_id, "slots_qa", {"slots": slots, "qa_pairs": qa_pairs})
        cache_used = False
    else:
        cache_used = True
    if verbose:
        print(f"  提取槽位数：{len(slots)}")
        print(f"  生成QA对数：{len(qa_pairs)}")
    return slots, qa_pairs, cache_used


def run_atc_stage(
    *,
    instruction: str,
    trajectory: List[Dict],
    task_id: str,
    use_cache: bool,
    lite_mode: bool,
    takeover_ground_truth_list: List[int],
    soft_cache_mode: bool,
    skip_takeover: bool,
    profile_id: str,
    verbose: bool,
    parallel_caption: bool = True,
    judge_takeover: str = "end-to-end",
) -> Tuple[List[Dict], List[Dict], bool, bool]:
    results, takeover_results, cache_used_atc, cache_used_takeover = (
        generate_action_thought_caption_with_takeover(
            instruction=instruction,
            trajectory=trajectory,
            lite_mode=lite_mode,
            parallel_caption=parallel_caption,
            task_id=task_id,
            use_cache=use_cache,
            takeover_ground_truth_list=takeover_ground_truth_list,
            soft_cache_mode=soft_cache_mode,
            skip_takeover=skip_takeover,
            profile_id=profile_id,
            judge_takeover=normalize_takeover_judge_mode(judge_takeover),
        )
    )
    if verbose:
        if cache_used_atc:
            print("  [使用缓存]")
        print(f"  生成步骤数：{len(results)}")
    return results, takeover_results, cache_used_atc, cache_used_takeover


def run_keyframes_stage(
    *,
    instruction: str,
    task_id: str,
    use_cache: bool,
    profile_id: str,
    action_thought_caption_results: List[Dict],
    category: str,
    verbose: bool,
) -> Tuple[Dict, bool]:
    cache_name = f"keyframes_full_{profile_id}"
    set_task_context(task_id, cache_name)
    keyframes = None
    if use_cache and task_id:
        keyframes = load_from_cache(task_id, cache_name)
    if keyframes is None:
        keyframes = extract_keyframes(instruction, action_thought_caption_results, category)
        if use_cache and task_id:
            save_to_cache(task_id, cache_name, keyframes)
        cache_used = False
    else:
        cache_used = True
    if verbose:
        print(f"  关键帧数：{len(keyframes.get('keyframe_indices', []))}")
    return keyframes, cache_used


def run_final_judgment_stage(
    *,
    instruction: str,
    task_id: str,
    use_cache: bool,
    cache_cleared: bool,
    soft_cache_mode: bool,
    verbose: bool,
    profile_id: str,
    final_judgment_mode: str,
    qa_pairs: Optional[List[Dict[str, str]]],
    keyframes: Dict,
    action_thought_caption_results: List[Dict],
    ground_truth: Optional[int],
    category: Optional[str],
    use_keyframe_images: bool,
    use_qa_images: bool,
    no_qa: bool,
) -> Tuple[Dict[str, int], bool]:
    judgment_cache_name = f"judgment_{profile_id}"
    set_task_context(task_id, judgment_cache_name)
    judgment = None
    cache_used = False

    if use_cache and task_id and not cache_cleared and not soft_cache_mode:
        judgment = load_from_cache(task_id, judgment_cache_name)
        if judgment is not None:
            cache_used = True

    if judgment is None:
        judgment = judge_task_completion(
            instruction,
            qa_pairs=qa_pairs,
            keyframe_captions=keyframes["keyframe_captions"],
            keyframe_actions=keyframes["keyframe_actions"],
            action_thought_caption=action_thought_caption_results,
            ground_truth=ground_truth,
            category=category,
            keyframe_indices=keyframes["keyframe_indices"],
            use_keyframe_images=use_keyframe_images,
            use_qa_images=use_qa_images,
            no_qa=no_qa,
            final_judgment_mode=final_judgment_mode,
            profile_id=profile_id,
        )
        if use_cache and task_id:
            save_to_cache(task_id, judgment_cache_name, judgment)
            cache_used = False

    if verbose:
        print(f"  综合判断：{judgment.get('final_judgment', -1)}")

    return judgment, cache_used
