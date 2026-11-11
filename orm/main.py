"""
主程序入口
整合所有模块，实现完整的GUI任务完成判断流程
"""
import copy
import json
import sys
import os
import csv
import glob
import shutil
import threading
import time
import logging
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts.slot_extraction import extract_slots
from prompts.task_classification import classify_task, get_category_name
from prompts.action_thought_caption import (
    generate_action_thought_caption,
    generate_action_thought_caption_with_takeover,
    validate_actions_for_trajectory,
)
from prompts.keyframe_extraction import extract_keyframes
from prompts.task_completion_judgment import judge_task_completion
from utils.prompt_utils import (
    load_from_cache,
    save_to_cache,
    generate_cache_key,
    set_task_context,
    set_model_name,
    get_image_path_from_step,
    call_llm_with_prompt,
    parse_json_response,
    bind_pipeline_preset,
    unbind_pipeline_preset,
    get_pipeline_preset,
)
from prompts.lite_judgment_prompt import get_lite_judgment_prompt
from utils.eval_pipeline import (
    build_pipeline_spec_from_preset,
    preset_from_legacy_flags,
    load_pipeline_spec_from_json,
)
from utils.lite_postprocess import apply_lite_task_success_postprocess
from utils.pipeline_runner import (
    run_error_cause_analysis_judgment,
    run_lite_endframe_judgment,
    run_lite_trajectory_unified_all_images_binary_judgment,
    run_trajectory_correctness_all_images_judgment,
    run_category_stage,
    run_slots_qa_stage,
    run_atc_stage,
    run_keyframes_stage,
    run_final_judgment_stage,
)
from pathlib import Path
from utils.task_knowledge import summarize_match_for_reasoning
from utils.lite_prompt_knowledge import normalize_takeover_judge_mode
from utils.input_resolver import (
    backfill_prediction_fields_to_xlsx,
    is_xlsx_input,
    load_task_ids_from_jsonl,
    normalize_task_for_missing_tags,
    resolve_input_to_jsonl,
    summarize_xlsx_human_vs_zj_status,
)

# Lite 模式：轨迹末尾取图张数
LITE_LAST_N_IMAGES = 10

# 模块日志（由上层应用决定 handler/level；这里只负责产生日志）
logger = logging.getLogger(__name__)

# 模块级缓存：用于缓存文件列表和检查结果，避免重复 glob 和文件系统操作
_file_list_cache = {}  # 缓存文件列表：{(task_id, dir_path, mtime): file_list}
_error_check_cache = {}  # 缓存错误检查结果：{(task_id, strategy): result}
SUPPORTED_KNOWLEDGE_BASES = ("knowledge_base", "knowledge_base_ali", "knowledge_base_ali_original", "knowledge_base_ali_hybrid", "knowledge_base_train")
FULL_STATUS_FILTER = {"failed", "fair_enough", "perfect"}
PREDICTION_PROVIDER = "zj"
TAKEOVER_SOURCE_END_TO_END = "end-to-end"
TAKEOVER_SOURCE_ONLY_LMD = "only-lmd"


def _step_has_reply_prediction_display_field(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    display_fields = step.get("display_fields")
    if not isinstance(display_fields, list):
        return False
    for field in display_fields:
        if isinstance(field, dict) and field.get("key") == "replyPrediction":
            return True
    return False


def _step_has_takeover_signal(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    if isinstance(step.get("take_over"), dict):
        return True
    value = step.get("value")
    if not isinstance(value, dict):
        return False
    normalized = value.get("normalized_tool_call")
    if isinstance(normalized, dict) and normalized.get("action") == "take_over":
        return True
    tool_call = value.get("tool_call")
    if isinstance(tool_call, str) and "take_over" in tool_call:
        return True
    code = value.get("code")
    if isinstance(code, str) and "take_over" in code:
        return True
    return False


def annotate_takeover_sources(trajectory: Any) -> Any:
    """Mark each takeover step as end-to-end output or LMD output for prompts."""
    if not isinstance(trajectory, list):
        return trajectory
    for step in trajectory:
        if not isinstance(step, dict) or not _step_has_takeover_signal(step):
            continue
        step["takeover_source"] = (
            TAKEOVER_SOURCE_END_TO_END
            if _step_has_reply_prediction_display_field(step)
            else TAKEOVER_SOURCE_ONLY_LMD
        )
    return trajectory


def _task_status_from_lite_final(
    lite_final: Any,
    *,
    is_env_error: Optional[bool] = None,
    success_status: str = "perfect",
    uncertain_status: str = "failed",
) -> Optional[str]:
    if lite_final == 1:
        return success_status
    if lite_final == 0:
        return "failed"
    if lite_final == -1:
        return uncertain_status
    return None


def _apply_lite_env_fields_to_prediction(
    prediction: Dict[str, Any],
    judgment: Dict[str, Any],
) -> None:
    if not judgment.get("is_env_error"):
        return
    for field in ("env_error_type", "env_error_reason"):
        value = judgment.get(field, "")
        if isinstance(value, str) and value.strip():
            prediction[field] = value.strip()


def _iter_step_strings(step: Dict[str, Any]) -> Iterable[str]:
    value = step.get("value")
    if isinstance(value, dict):
        for key in ("action", "code", "thought", "tool_call"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                yield item
        normalized = value.get("normalized_tool_call")
        if isinstance(normalized, dict):
            for key in ("action", "text"):
                item = normalized.get(key)
                if isinstance(item, str) and item.strip():
                    yield item
    takeover = step.get("take_over")
    if isinstance(takeover, dict):
        for key in ("category", "input", "summarize", "continue_goal"):
            item = takeover.get(key)
            if isinstance(item, str) and item.strip():
                yield item

def _is_cast_to_tv_instruction(instruction: Any) -> bool:
    if not isinstance(instruction, str):
        return False
    return "投屏" in instruction or ("电视" in instruction and any(k in instruction for k in ("播放", "投放", "连接")))


def _has_douyin_together_watch_cast_attempt(trajectory: Any) -> bool:
    if not isinstance(trajectory, list):
        return False
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        step_text = "\n".join(_iter_step_strings(step))
        if "一起看视频" in step_text:
            return True
    return False


def _apply_wrong_cast_entry_override(
    judgment: Dict[str, Any],
    instruction: Any,
    trajectory: Any,
) -> None:
    if not (_is_cast_to_tv_instruction(instruction) and _has_douyin_together_watch_cast_attempt(trajectory)):
        return
    reason = (
        "首先排除环境或系统问题：轨迹中的失败提示发生在模型选择了错误功能入口之后，"
        "不能把错误入口触发的业务失败反向归因成系统异常；界面元素可见且模型能点击，"
        "排除感知/Grounding能力不足。模型将抖音分享栏里的“一起看视频”误认为投屏入口，"
        "属于对抖音投屏功能入口或页面结构的理解错误，符合背景知识不足下的功能入口或页面结构理解错误。"
        "证据：Step 6、Step 9 轨迹中模型点击“一起看视频”尝试完成投屏，该入口不是抖音投屏到电视的正确入口，"
        "后续出现“发起失败”只是错误入口导致的结果。"
    )
    lite_reason = (
        "模型把抖音分享栏里的“一起看视频”当成投屏到电视入口，首次未恢复偏航发生在功能入口选择错误；"
        "后续发起失败不能覆盖此前的模型能力错误，应判 failed。"
    )
    judgment["lite_final_judgment"] = 0
    judgment["lite_reasoning"] = lite_reason
    judgment["task_status"] = "failed"
    judgment["primary_error_cause"] = "背景知识不足"
    judgment["secondary_error_cause"] = "功能入口或页面结构理解错误"
    judgment["error_cause_reasoning"] = reason
    judgment["wrong_cast_entry_override"] = True


def _should_force_error_cause_after_lite_success(instruction: Any, trajectory: Any) -> bool:
    """Route lite-success takeover endings through rich structured analysis."""
    if not isinstance(trajectory, list):
        return False
    return any(_step_has_takeover_signal(step) for step in trajectory)


def _lite_gate_reasoning_text(judgment: Dict[str, Any]) -> str:
    reasoning = judgment.get("lite_reasoning") or judgment.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = "" if reasoning is None else str(reasoning)
    return reasoning.strip()


def _is_valid_lite_gate_judgment(judgment: Dict[str, Any]) -> bool:
    if not isinstance(judgment, dict):
        return False
    if judgment.get("final_judgment") not in {-1, 0, 1}:
        return False
    task_status = judgment.get("task_status")
    if not isinstance(task_status, str) or task_status not in FULL_STATUS_FILTER:
        return False
    return bool(_lite_gate_reasoning_text(judgment))


def _describe_lite_gate_invalid(judgment: Any) -> str:
    if not isinstance(judgment, dict):
        return f"non-dict output: {type(judgment).__name__}"
    return (
        f"final_judgment={judgment.get('final_judgment')!r}, "
        f"task_status={judgment.get('task_status')!r}, "
        f"reasoning_empty={not bool(_lite_gate_reasoning_text(judgment))}"
    )


def _string_prediction_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _error_cause_failure_reason(judgment: Dict[str, Any]) -> List[str]:
    primary = _string_prediction_value(judgment.get("primary_error_cause", "")).strip()
    secondary = _string_prediction_value(judgment.get("secondary_error_cause", "")).strip()
    if primary and secondary:
        return [f"{primary} | {secondary}"]
    reason = primary or secondary
    return [reason] if reason else []


def _strip_reasoning_prefix(reasoning: str, prefix: str) -> str:
    reasoning = reasoning.strip()
    if reasoning.startswith(prefix):
        return reasoning[len(prefix):].strip()
    return reasoning


def _nonempty_reasoning_or_empty(value: Any, *, prefix: str = "") -> str:
    reasoning = _string_prediction_value(value)
    if prefix:
        reasoning = _strip_reasoning_prefix(reasoning, prefix)
    reasoning = reasoning.strip()
    return reasoning or "空"


def _error_cause_prediction_reasoning(judgment: Dict[str, Any]) -> str:
    lite_reasoning = _nonempty_reasoning_or_empty(
        judgment.get("lite_reasoning", ""),
        prefix="<lite判定>",
    )
    if judgment.get("error_cause_skipped"):
        error_cause_reasoning = "空"
    else:
        error_cause_reasoning = _nonempty_reasoning_or_empty(
            judgment.get("error_cause_reasoning") or judgment.get("reasoning")
        )
        task_status = _lite_task_status_from_judgment(judgment)
        if task_status in {"failed", "env_error"}:
            failure_summary = "；".join(_error_cause_failure_reason(judgment))
            lite_reasoning = f"最终判定为{task_status}：{failure_summary or error_cause_reasoning}"
    knowledge_reasoning = summarize_match_for_reasoning(judgment)
    return _format_orm_prediction_reasoning(
        lite_reasoning=lite_reasoning,
        knowledge_reasoning=knowledge_reasoning,
        error_cause_reasoning=error_cause_reasoning,
    )


def _format_orm_prediction_reasoning(
    *,
    lite_reasoning: Any = "",
    knowledge_reasoning: Any = "",
    error_cause_reasoning: Any = "",
) -> str:
    lite_text = _nonempty_reasoning_or_empty(lite_reasoning, prefix="<lite判定>")
    knowledge_text = _string_prediction_value(knowledge_reasoning).strip() or "空"
    error_text = _string_prediction_value(error_cause_reasoning).strip() or "空"
    return (
        f"orm判定原因：{lite_text} | "
        f"命中知识：{knowledge_text} | "
        f"错因归类原因：{error_text}"
    )


def _merge_lite_gate_trajectory_fields(
    judgment: Dict[str, Any],
    lite_judgment: Dict[str, Any],
) -> None:
    """Carry trajectory fields from the unified gate into error-cause output."""
    for field in (
        "trajectory_correctness_judgment",
        "trajectory_reasoning",
        "first_deviation_step",
        "first_deviation_reason",
        "max_correct_prefix_end_step",
        "max_correct_prefix_steps",
        "task_knowledge_hit",
        "task_knowledge_confidence",
        "task_knowledge_used",
        "task_knowledge_reasoning",
        "task_knowledge_match",
    ):
        if field in lite_judgment:
            judgment[field] = lite_judgment[field]


def _build_error_cause_lite_gate_spec(
    *,
    action_chain_source: str,
    skip_takeover: bool,
    keep_last_action: bool,
    compress_quality: Optional[float],
):
    """Build the full-image lite gate used by error-cause analysis."""
    source = (action_chain_source or "raw").strip()
    if source == "raw":
        eval_profile = "lite_trajectory_allimg_unified_binary"
    elif source == "validated":
        eval_profile = "lite_trajectory_allimg_unified_binary_action_validated"
    else:
        return build_pipeline_spec_from_preset(
            preset=f"lite_trajectory_allimg_unified_binary_{source}",
            eval_lite_mode=True,
            output_lite_mode=False,
            final_judgment_mode="lite_plus_trajectory_all_images_unified_binary",
            lite_include_action_list=True,
            lite_action_chain_source=source,
            unified_scope="lite_end_only",
            use_keyframe_images=True,
            use_qa_images=False,
            no_qa=True,
            skip_takeover=skip_takeover,
            keep_last_action=keep_last_action,
            compress_quality=compress_quality,
        )

    return preset_from_legacy_flags(
        lite_mode=True,
        lite_action_list=True,
        eval_profile=eval_profile,
        use_keyframe_images=True,
        use_qa_images=False,
        no_qa=True,
        skip_takeover=skip_takeover,
        keep_last_action=keep_last_action,
        compress_quality=compress_quality,
    )


def _is_success_task_status(status: Any) -> bool:
    return isinstance(status, str) and status in {"perfect", "fair_enough"}


def _lite_task_status_from_judgment(judgment: Dict[str, Any]) -> Optional[str]:
    status = judgment.get("task_status")
    if isinstance(status, str) and status in FULL_STATUS_FILTER:
        return status
    status = judgment.get("lite_task_status")
    if isinstance(status, str) and status in FULL_STATUS_FILTER:
        return status
    return _task_status_from_lite_final(
        judgment.get("lite_final_judgment"),
        is_env_error=judgment.get("is_env_error"),
    )


def _error_cause_structured_task_status(
    judgment: Dict[str, Any],
    current_status: Optional[str] = None,
) -> Optional[str]:
    """Resolve internal lite/error-cause status conflicts from structured fields only."""
    if not isinstance(judgment, dict) or judgment.get("error_cause_skipped"):
        return current_status if current_status in FULL_STATUS_FILTER else None

    primary = _string_prediction_value(judgment.get("primary_error_cause", "")).strip()
    secondary = _string_prediction_value(judgment.get("secondary_error_cause", "")).strip()
    takeover_bad = (
        bool(judgment.get("takeover_overuse"))
        or bool(judgment.get("takeover_missing"))
        or judgment.get("takeover_type_correct") is False
        or secondary == "接管问题"
    )

    if takeover_bad:
        return "failed"
    if primary == "环境或系统问题":
        return "failed"
    return current_status if current_status in FULL_STATUS_FILTER else None


def _is_error_cause_success_judgment(judgment: Dict[str, Any]) -> bool:
    if _is_success_task_status(_lite_task_status_from_judgment(judgment)):
        return True
    return judgment.get("lite_final_judgment") == 1


def _normalize_error_cause_trajectory_fields(
    judgment: Dict[str, Any],
    task_status: Optional[str],
) -> Dict[str, Any]:
    """Build trajectory fields for error-cause mode with success consistency."""
    is_success = _is_success_task_status(task_status) or _is_error_cause_success_judgment(judgment)
    if is_success:
        reasoning = _string_prediction_value(judgment.get("trajectory_reasoning", ""))
        return {
            "is_trajectory_correct": 1,
            "trajectory_reasoning": reasoning,
            "first_deviation_step": None,
            "first_deviation_reason": "",
        }

    pred = judgment.get("trajectory_correctness_judgment")
    if pred not in (0, 1, -1):
        return {}
    reasoning = _string_prediction_value(judgment.get("trajectory_reasoning", ""))
    out: Dict[str, Any] = {
        "is_trajectory_correct": pred,
        "trajectory_reasoning": reasoning,
    }
    if pred == 1:
        return out
    fds = judgment.get("first_deviation_step")
    if isinstance(fds, int):
        out["first_deviation_step"] = fds
    fdr = judgment.get("first_deviation_reason")
    if isinstance(fdr, str) and fdr.strip():
        out["first_deviation_reason"] = fdr.strip()
    return out


def _clear_first_deviation_steps(task: Dict[str, Any]) -> None:
    traj = task.get("traj")
    if not isinstance(traj, list):
        return
    for step in traj:
        if not isinstance(step, dict):
            continue
        value = step.get("value")
        if isinstance(value, dict):
            value.pop("first_deviation_step", None)


def _mark_first_deviation_step(task: Dict[str, Any], step_index: int) -> None:
    _clear_first_deviation_steps(task)
    traj = task.get("traj")
    if not isinstance(traj, list):
        return
    for step in traj:
        if not isinstance(step, dict):
            continue
        if step.get("index") == step_index:
            value = step.get("value")
            if not isinstance(value, dict):
                value = {}
                step["value"] = value
            value["first_deviation_step"] = True
            break


def _unified_prediction_from_judgment(judgment: Dict[str, Any]) -> Dict[str, Any]:
    """从 unified_binary judgment 构建 evaluation.prediction 子字段。"""
    pred_success = judgment.get("final_judgment", 0)
    task_status = judgment.get("task_status")
    if not isinstance(task_status, str) or task_status not in FULL_STATUS_FILTER:
        task_status = _task_status_from_lite_final(
            pred_success,
            is_env_error=judgment.get("is_env_error"),
        )
    lite_reasoning = judgment.get("lite_reasoning", "")
    if not isinstance(lite_reasoning, str):
        lite_reasoning = "" if lite_reasoning is None else str(lite_reasoning)
    prediction: Dict[str, Any] = {
        "provider": PREDICTION_PROVIDER,
        "reasoning": _format_orm_prediction_reasoning(
            lite_reasoning=lite_reasoning,
            knowledge_reasoning=summarize_match_for_reasoning(judgment),
            error_cause_reasoning="",
        )
    }
    if task_status:
        prediction["task_status"] = task_status
    _apply_lite_env_fields_to_prediction(prediction, judgment)
    return prediction


def configure_knowledge_base(knowledge_base_name: Optional[str], verbose: bool = False) -> str:
    """切换当前进程使用的知识库目录，并清空评估规则缓存。"""
    import config

    kb_name = (knowledge_base_name or "").strip() or config.DEFAULT_KNOWLEDGE_BASE
    applied = config.set_knowledge_base(kb_name)

    try:
        import prompts.task_completion_judgment as _tcj

        _tcj._completion_strategy_cache = None
        _tcj._trajectory_strategy_cache = None
    except Exception:
        # 缓存清理失败不阻塞主流程
        pass

    if verbose:
        print(f"知识库: {applied} ({config.KNOWLEDGE_BASE_DIR})")
    return applied


def load_final_improvements(
    improvement_type: str,
    model_name: Optional[str] = None
) -> Optional[Dict]:
    """
    从缓存目录读取最终改进知识
    
    Args:
        improvement_type: 改进知识类型 ("keyframe_judgment" 或 "qa_judgment")
        model_name: 模型名称（用于确定缓存目录）
    
    Returns:
        Dict: 包含改进知识的字典，格式为：
            {
                "type": "keyframe_judgment" 或 "qa_judgment",
                "final_level": int,
                "source_groups": int,
                "improvements": str,
                "timestamp": str,
                "model_name": str
            }
            如果不存在则返回 None
    """
    try:
        import config
        
        # 确定缓存目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        if improvement_type == "keyframe_judgment":
            cache_dir = os.path.join(project_root, "data", model_name or config.DEFAULT_MODEL, "prompt_improvements_cache")
        elif improvement_type == "qa_judgment":
            cache_dir = os.path.join(project_root, "data", model_name or config.DEFAULT_MODEL, "qa_prompt_improvements_cache")
        else:
            print(f"警告: 未知的改进知识类型: {improvement_type}")
            return None
        
        # 读取最终改进知识文件
        final_cache_file = os.path.join(cache_dir, "final_improvements.json")
        
        if not os.path.exists(final_cache_file):
            return None
        
        with open(final_cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # 验证类型是否匹配
        if cache_data.get("type") != improvement_type:
            print(f"警告: 缓存文件类型不匹配，期望 {improvement_type}，实际 {cache_data.get('type')}")
            return None
        
        return cache_data
        
    except Exception as e:
        print(f"读取最终改进知识失败: {e}")
        return None


def get_final_improvements_text(
    improvement_type: str,
    model_name: Optional[str] = None
) -> Optional[str]:
    """
    获取最终改进知识的文本内容（便捷函数）
    
    Args:
        improvement_type: 改进知识类型 ("keyframe_judgment" 或 "qa_judgment")
        model_name: 模型名称
    
    Returns:
        str: 改进知识文本，如果不存在则返回 None
    """
    data = load_final_improvements(improvement_type, model_name)
    if data:
        return data.get("improvements", "")
    return None


def extract_task_ids_from_bdcs(model_name: str) -> List[str]:
    """
    从 bdcs 目录中提取所有 task_id
    
    Args:
        model_name: 模型名称
    
    Returns:
        List[str]: task_id 列表（去重）
    """
    import config
    
    bdcs_dir = os.path.join(config.get_model_data_dir(model_name), "bdcs")
    if not os.path.exists(bdcs_dir):
        return []
    
    task_ids = set()
    
    # 查找所有 JSON 文件
    json_files = glob.glob(os.path.join(bdcs_dir, "*.json"))
    
    for json_file in json_files:
        basename = os.path.basename(json_file)
        # 去掉 .json 后缀
        name_without_ext = basename[:-5] if basename.endswith('.json') else basename
        
        # 处理新格式：{task_id}_completion_judgment_{c_gt}_to_{c_pred}.json
        # 或 {task_id}_trajectory_correctness_judgment_{t_gt}_to_{t_pred}.json
        if "_completion_judgment_" in name_without_ext:
            task_id = name_without_ext.split("_completion_judgment_")[0]
            task_ids.add(task_id)
        elif "_trajectory_correctness_judgment_" in name_without_ext:
            task_id = name_without_ext.split("_trajectory_correctness_judgment_")[0]
            task_ids.add(task_id)
        else:
            # 可选：{task_id}.json
            task_ids.add(name_without_ext)
    
    return sorted(list(task_ids))


def load_judgment_from_save_dir(
    task_id: str,
    ground_truth: Optional[int],
    model_name: Optional[str] = None
) -> Optional[Dict]:
    """
    从 save 目录的多个结果文件中选择并加载 judgment 结果
    
    选择策略：
    1. 优先选择正确的结果（0_to_0.json 或 1_to_1.json）
    2. 如果没有正确的结果，选择最新的文件（按修改时间）
    
    Args:
        task_id: 任务ID
        ground_truth: 任务真实完成状态（1=完成, 0=未完成），用于优先选择正确结果
        model_name: 模型名称
    
    Returns:
        Dict: judgment 结果，格式为 {
            "qa_judgment": int,
            "keyframe_judgment": int,
            "final_judgment": int
        }，如果找不到则返回 None
    """
    try:
        import config
        
        save_dir = os.path.join(config.get_save_dir(model_name), task_id)
        if not os.path.exists(save_dir):
            return None
        
        # 优化：使用缓存避免重复 glob
        dir_mtime = os.path.getmtime(save_dir) if os.path.exists(save_dir) else 0
        cache_key = (task_id, save_dir, dir_mtime)
        if cache_key in _file_list_cache:
            all_judgment_files = _file_list_cache[cache_key]
        else:
            # 一次性查找所有判断结果文件（优化：合并4次glob调用为1次）
            all_judgment_files = glob.glob(os.path.join(save_dir, "*_judgment_*.json"))
            _file_list_cache[cache_key] = all_judgment_files
            # 限制缓存大小，避免内存泄漏
            if len(_file_list_cache) > 1000:
                # 删除最旧的缓存项（简单策略：删除第一个）
                _file_list_cache.pop(next(iter(_file_list_cache)), None)
        
        if not all_judgment_files:
            return None
        
        # 按文件名前缀分类
        qa_files = [f for f in all_judgment_files if os.path.basename(f).startswith("qa_judgment_")]
        keyframe_files = [f for f in all_judgment_files if os.path.basename(f).startswith("keyframe_judgment_")]
        completion_files = [f for f in all_judgment_files if os.path.basename(f).startswith("completion_judgment_")]
        trajectory_correctness_files = [f for f in all_judgment_files if os.path.basename(f).startswith("trajectory_correctness_judgment_")]
        
        # 解析文件名，提取 ground_truth 和 predicted 值
        def parse_judgment_file(filename: str) -> Optional[Dict]:
            """解析文件名，返回 {ground_truth, predicted, file_path, is_correct, mtime}"""
            basename = os.path.basename(filename)
            # 格式：qa_judgment_1_to_0.json 或 completion_judgment_1_to_1.json
            parts = basename.replace('.json', '').split('_to_')
            if len(parts) != 2:
                return None
            
            try:
                # 从 "qa_judgment_1" 或 "completion_judgment_1" 中提取 "1"
                prefix_parts = parts[0].split('_')
                gt_str = prefix_parts[-1]
                pred_str = parts[1]
                
                if gt_str == "None":
                    gt = None
                else:
                    gt = int(gt_str)
                
                if pred_str == "-1":
                    pred = -1
                else:
                    pred = int(pred_str)
                
                is_correct = (gt is not None and pred != -1 and gt == pred)
                mtime = os.path.getmtime(filename)
                
                return {
                    "ground_truth": gt,
                    "predicted": pred,
                    "file_path": filename,
                    "is_correct": is_correct,
                    "mtime": mtime
                }
            except (ValueError, IndexError):
                return None
        
        # 解析所有文件
        qa_results = []
        for qa_file in qa_files:
            parsed = parse_judgment_file(qa_file)
            if parsed:
                qa_results.append(parsed)
        
        keyframe_results = []
        for keyframe_file in keyframe_files:
            parsed = parse_judgment_file(keyframe_file)
            if parsed:
                keyframe_results.append(parsed)
        
        completion_results = []
        for completion_file in completion_files:
            parsed = parse_judgment_file(completion_file)
            if parsed:
                completion_results.append(parsed)
        
        trajectory_correctness_results = []
        for trajectory_correctness_file in trajectory_correctness_files:
            parsed = parse_judgment_file(trajectory_correctness_file)
            if parsed:
                trajectory_correctness_results.append(parsed)
        
        # 选择策略：优先选择正确的结果，然后选择最新的
        def select_best_result(results: List[Dict], gt: Optional[int]) -> Optional[int]:
            """从结果列表中选择最佳结果，返回 predicted 值"""
            if not results:
                return None
            
            # 如果有 ground_truth，优先选择正确的结果
            if gt is not None:
                correct_results = [r for r in results if r["is_correct"]]
                if correct_results:
                    # 如果有多个正确结果，选择最新的
                    best = max(correct_results, key=lambda x: x["mtime"])
                    return best["predicted"]
            
            # 如果没有正确的结果，选择最新的
            best = max(results, key=lambda x: x["mtime"])
            return best["predicted"]
        
        # 选择各个判断结果
        qa_judgment = select_best_result(qa_results, ground_truth)
        keyframe_judgment = select_best_result(keyframe_results, ground_truth)
        completion_judgment = select_best_result(completion_results, ground_truth)
        trajectory_correctness_judgment = select_best_result(trajectory_correctness_results, ground_truth)
        
        # 构建 judgment 字典
        judgment = {
            "qa_judgment": qa_judgment if qa_judgment is not None else -1,
            "keyframe_judgment": keyframe_judgment if keyframe_judgment is not None else -1,
            "completion_judgment": completion_judgment if completion_judgment is not None else -1,
            "trajectory_correctness_judgment": trajectory_correctness_judgment if trajectory_correctness_judgment is not None else -1,
            "final_judgment": -1
        }
        
        # 计算 final_judgment：完成度=1 且 轨迹正确性=1 时判对
        if completion_judgment != -1 and trajectory_correctness_judgment != -1:
            # 两个都对时判对
            if completion_judgment == 1 and trajectory_correctness_judgment == 1:
                judgment["final_judgment"] = 1
            else:
                judgment["final_judgment"] = 0
        elif completion_judgment != -1:
            # 只有完成度判断，使用完成度结果
            judgment["final_judgment"] = completion_judgment
        elif trajectory_correctness_judgment != -1:
            # 只有轨迹正确性判断，使用轨迹正确性结果
            judgment["final_judgment"] = trajectory_correctness_judgment
        elif qa_judgment != -1 and keyframe_judgment != -1:
            # 若两个都有值，优先使用 keyframe_judgment
            judgment["final_judgment"] = keyframe_judgment
        elif qa_judgment != -1:
            judgment["final_judgment"] = qa_judgment
        elif keyframe_judgment != -1:
            judgment["final_judgment"] = keyframe_judgment
        
        return judgment
        
    except Exception as e:
        print(f"  警告: 从 save 目录加载 judgment 失败 {task_id}: {e}")
        return None


def check_and_clear_error_prediction_cache(
    task_id: str,
    model_name: Optional[str] = None,
    save_data: bool = False,
    lite_mode: bool = False,
    verbose: bool = True,
    no_cache_strategy: Optional[str] = None,  # None: 不检查, "qa": 检查QA错误, "key": 检查关键帧错误, "completion": 检查完成度错误, "trajectory": 检查轨迹正确性错误
    pipeline_profile_id: Optional[str] = None,
    pipeline_preset: Optional[str] = None,
) -> bool:
    """
    检查是否存在错误预测文件，如果存在则清除相关缓存
    
    Args:
        task_id: 任务ID
        model_name: 模型名称
        save_data: 是否启用保存
        lite_mode: 是否使用简略模式
        verbose: 是否输出详细信息
        no_cache_strategy: 缓存策略
            - None: 不检查（正常使用缓存）
            - "qa": 检查QA错误，清除错误任务的所有tmp和tmp2缓存以及save对应的json文件
            - "key": 检查关键帧错误，清除错误任务的所有tmp和tmp2缓存以及save对应的json文件
            - "completion": 检查完成度错误，清除错误任务的所有tmp和tmp2缓存以及save对应的json文件
            - "trajectory": 检查轨迹正确性错误，清除错误任务的所有tmp和tmp2缓存以及save对应的json文件
    
    Returns:
        bool: 如果清除了缓存返回True，否则返回False
    """
    if not save_data or not task_id or no_cache_strategy is None:
        return False
    
    try:
        import config
        
        # 检查save目录
        save_dir = os.path.join(config.get_save_dir(model_name), task_id)
        if pipeline_profile_id:
            save_dir = os.path.join(save_dir, pipeline_profile_id)
        if not os.path.exists(save_dir):
            return False
        
        # 优化：使用缓存避免重复检查
        cache_key = (task_id, no_cache_strategy)
        if cache_key in _error_check_cache:
            # 如果之前检查过且没有错误，直接返回
            cached_result = _error_check_cache[cache_key]
            if not cached_result:
                return False
        
        found_error_file = False
        error_files_to_delete = []
        
        # 优化：一次性查找所有 judgment 文件，然后按类型分类（使用缓存）
        dir_mtime = os.path.getmtime(save_dir) if os.path.exists(save_dir) else 0
        file_cache_key = (task_id, save_dir, dir_mtime)
        if file_cache_key in _file_list_cache:
            all_judgment_files = _file_list_cache[file_cache_key]
        else:
            all_judgment_files = glob.glob(os.path.join(save_dir, "*_judgment_*.json"))
            _file_list_cache[file_cache_key] = all_judgment_files
            # 限制缓存大小
            if len(_file_list_cache) > 1000:
                _file_list_cache.pop(next(iter(_file_list_cache)), None)
        judgment_files_by_type = {
            "qa": [f for f in all_judgment_files if os.path.basename(f).startswith("qa_judgment_")],
            "keyframe": [f for f in all_judgment_files if os.path.basename(f).startswith("keyframe_judgment_")],
            "completion": [f for f in all_judgment_files if os.path.basename(f).startswith("completion_judgment_")],
            "trajectory": [f for f in all_judgment_files if os.path.basename(f).startswith("trajectory_correctness_judgment_")]
        }
        
        # 根据策略检查不同的错误文件
        if no_cache_strategy == "qa":
            # 检查QA错误文件
            qa_error_files = [
                "qa_judgment_0_to_1.json",
                "qa_judgment_1_to_0.json"
            ]
            
            for error_file in qa_error_files:
                error_file_path = os.path.join(save_dir, error_file)
                if os.path.exists(error_file_path):
                    found_error_file = True
                    error_files_to_delete.append(error_file_path)
                    if verbose:
                        print(f"  发现QA错误预测文件: {error_file}")
        
        elif no_cache_strategy == "key":
            # 检查关键帧错误文件
            keyframe_error_files = [
                "keyframe_judgment_0_to_1.json",
                "keyframe_judgment_1_to_0.json"
            ]
            
            for error_file in keyframe_error_files:
                error_file_path = os.path.join(save_dir, error_file)
                if os.path.exists(error_file_path):
                    found_error_file = True
                    error_files_to_delete.append(error_file_path)
                    if verbose:
                        print(f"  发现关键帧错误预测文件: {error_file}")
            
            # 如果发现错误文件，标记需要删除所有 keyframe_judgment 文件（包括 0_0, 0_1, 1_0, 1_1）
            if found_error_file:
                if verbose:
                    print(f"  将删除所有 keyframe_judgment 文件（包括 0_0, 0_1, 1_0, 1_1）")
            else:
                # 如果没有错误文件，检查是否存在任何 keyframe_judgment 文件（使用已缓存的列表）
                existing_files = judgment_files_by_type["keyframe"]
                if not existing_files:
                    # 如果不存在任何 keyframe_judgment 文件，也需要清除缓存并重新生成
                    found_error_file = True
                    if verbose:
                        print(f"  未找到 keyframe_judgment 文件，将清除缓存并重新生成")
        
        elif no_cache_strategy == "completion":
            # 检查完成度错误文件
            completion_error_files = [
                "completion_judgment_0_to_1.json",
                "completion_judgment_1_to_0.json"
            ]
            
            for error_file in completion_error_files:
                error_file_path = os.path.join(save_dir, error_file)
                if os.path.exists(error_file_path):
                    found_error_file = True
                    error_files_to_delete.append(error_file_path)
                    if verbose:
                        print(f"  发现完成度错误预测文件: {error_file}")
            
            # 如果发现错误文件，标记需要删除所有 completion_judgment 文件
            if found_error_file:
                if verbose:
                    print(f"  将删除所有 completion_judgment 文件")
            else:
                # 如果没有错误文件，检查是否存在任何 completion_judgment 文件（使用已缓存的列表）
                existing_files = judgment_files_by_type["completion"]
                if not existing_files:
                    # 如果不存在任何 completion_judgment 文件，也需要清除缓存并重新生成
                    found_error_file = True
                    if verbose:
                        print(f"  未找到 completion_judgment 文件，将清除缓存并重新生成")
        
        elif no_cache_strategy == "trajectory":
            # 检查轨迹正确性错误文件
            trajectory_error_files = [
                "trajectory_correctness_judgment_0_to_1.json",
                "trajectory_correctness_judgment_1_to_0.json"
            ]
            
            for error_file in trajectory_error_files:
                error_file_path = os.path.join(save_dir, error_file)
                if os.path.exists(error_file_path):
                    found_error_file = True
                    error_files_to_delete.append(error_file_path)
                    if verbose:
                        print(f"  发现轨迹正确性错误预测文件: {error_file}")
            
            # 如果发现错误文件，标记需要删除所有 trajectory_correctness_judgment 文件
            if found_error_file:
                if verbose:
                    print(f"  将删除所有 trajectory_correctness_judgment 文件")
            else:
                # 如果没有错误文件，检查是否存在任何 trajectory_correctness_judgment 文件（使用已缓存的列表）
                existing_files = judgment_files_by_type["trajectory"]
                if not existing_files:
                    # 如果不存在任何 trajectory_correctness_judgment 文件，也需要清除缓存并重新生成
                    found_error_file = True
                    if verbose:
                        print(f"  未找到 trajectory_correctness_judgment 文件，将清除缓存并重新生成")
        
        if not found_error_file:
            # 更新错误检查缓存：没有发现错误
            _error_check_cache[cache_key] = False
            # 限制缓存大小
            if len(_error_check_cache) > 1000:
                _error_check_cache.pop(next(iter(_error_check_cache)), None)
            return False
        
        # 删除错误预测文件
        for error_file_path in error_files_to_delete:
            try:
                os.remove(error_file_path)
                if verbose:
                    print(f"  已删除错误预测文件: {os.path.basename(error_file_path)}")
            except Exception as e:
                print(f"  警告: 删除错误预测文件失败 {error_file_path}: {e}")
        
        # 清除所有相关缓存和文件
        if verbose:
            print("  清除所有相关缓存文件...")
        
        cache_root = config.get_cache_dir(model_name)
        tmp2_root = config.get_tmp2_dir(model_name)
        if pipeline_preset:
            cache_dir = os.path.join(config.get_model_data_dir(model_name), pipeline_preset, "tmp")
            tmp2_dir = os.path.join(config.get_model_data_dir(model_name), pipeline_preset, "tmp2")
        else:
            cache_dir = cache_root
            tmp2_dir = tmp2_root
        safe_task_id = task_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        cleared_count = 0
        
        # 清除tmp目录中的所有缓存文件（按task_id分文件夹）
        task_cache_dir = os.path.join(cache_dir, task_id)
        if os.path.exists(task_cache_dir):
            pattern = os.path.join(task_cache_dir, f"{safe_task_id}_*.json")
            cache_files = glob.glob(pattern)
            for cache_file in cache_files:
                try:
                    os.remove(cache_file)
                    cleared_count += 1
                    if verbose:
                        print(f"    已删除tmp缓存: {os.path.basename(cache_file)}")
                except Exception as e:
                    print(f"  警告: 删除tmp缓存文件失败 {cache_file}: {e}")
            # 如果目录为空，删除目录
            try:
                if not os.listdir(task_cache_dir):
                    os.rmdir(task_cache_dir)
            except:
                pass
        
        # 清除tmp2目录中的所有相关文件（按task_id分文件夹）
        task_tmp2_dir = os.path.join(tmp2_dir, task_id)
        if os.path.exists(task_tmp2_dir):
            # 删除所有prompt_response.json文件
            pattern = os.path.join(task_tmp2_dir, f"{safe_task_id}_*_prompt_response.json")
            tmp2_files = glob.glob(pattern)
            for tmp2_file in tmp2_files:
                try:
                    os.remove(tmp2_file)
                    cleared_count += 1
                    if verbose:
                        print(f"    已删除tmp2文件: {os.path.basename(tmp2_file)}")
                except Exception as e:
                    print(f"  警告: 删除tmp2文件失败 {tmp2_file}: {e}")
            
            # 删除QA详情CSV文件
            csv_file = os.path.join(task_tmp2_dir, f"{safe_task_id}_judgment_qa_details.csv")
            if os.path.exists(csv_file):
                try:
                    os.remove(csv_file)
                    cleared_count += 1
                    if verbose:
                        print(f"    已删除tmp2 CSV: {os.path.basename(csv_file)}")
                except Exception as e:
                    print(f"  警告: 删除tmp2 CSV文件失败 {csv_file}: {e}")
            
            # 如果目录为空，删除目录
            try:
                if not os.listdir(task_tmp2_dir):
                    os.rmdir(task_tmp2_dir)
            except:
                pass
        
        # 清除save目录中对应的json文件（优化：使用已缓存的文件列表）
        if no_cache_strategy == "qa":
            # 删除所有qa_judgment文件
            save_files = judgment_files_by_type["qa"]
            for save_file in save_files:
                try:
                    os.remove(save_file)
                    cleared_count += 1
                    if verbose:
                        print(f"    已删除save文件: {os.path.basename(save_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  警告: 删除save文件失败 {save_file}: {e}")
        
        elif no_cache_strategy == "key":
            # 删除所有keyframe_judgment文件（包括 0_0, 0_1, 1_0, 1_1）
            save_files = judgment_files_by_type["keyframe"]
            for save_file in save_files:
                try:
                    os.remove(save_file)
                    cleared_count += 1
                    if verbose:
                        print(f"    已删除save文件: {os.path.basename(save_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  警告: 删除save文件失败 {save_file}: {e}")
        
        elif no_cache_strategy == "completion":
            # 删除所有completion_judgment文件
            save_files = judgment_files_by_type["completion"]
            for save_file in save_files:
                try:
                    os.remove(save_file)
                    cleared_count += 1
                    if verbose:
                        print(f"    已删除save文件: {os.path.basename(save_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  警告: 删除save文件失败 {save_file}: {e}")
        
        elif no_cache_strategy == "trajectory":
            # 删除所有trajectory_correctness_judgment文件
            save_files = judgment_files_by_type["trajectory"]
            for save_file in save_files:
                try:
                    os.remove(save_file)
                    cleared_count += 1
                    if verbose:
                        print(f"    已删除save文件: {os.path.basename(save_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  警告: 删除save文件失败 {save_file}: {e}")
        
        # 按 task_id 清除时同时删除 lite 的 save 文件（tmp/tmp2 已通过上面的 pattern 覆盖）
        save_dir = os.path.join(config.get_save_dir(model_name), task_id)
        if pipeline_profile_id:
            save_dir = os.path.join(save_dir, pipeline_profile_id)
        if os.path.exists(save_dir):
            lite_save_pattern = os.path.join(save_dir, "lite_judgment_*_to_*.json")
            for lite_file in glob.glob(lite_save_pattern):
                try:
                    os.remove(lite_file)
                    cleared_count += 1
                    if verbose:
                        print(f"    已删除save lite文件: {os.path.basename(lite_file)}")
                except Exception as e:
                    if verbose:
                        print(f"  警告: 删除save lite文件失败 {lite_file}: {e}")
        
        if verbose:
            print(f"  共清除 {cleared_count} 个缓存/文件")
        
        # 更新错误检查缓存
        _error_check_cache[cache_key] = True
        # 限制缓存大小
        if len(_error_check_cache) > 1000:
            _error_check_cache.pop(next(iter(_error_check_cache)), None)
        
        return True
        
    except Exception as e:
        print(f"  警告: 检查错误预测缓存失败: {e}")
        return False


def process_task(
    instruction: str, 
    trajectory: List[Dict], 
    task_id: str = "",
    use_cache: bool = True,
    lite_mode: bool = False,
    lite_action_list: bool = False,
    ground_truth: Optional[int] = None,
    verbose: bool = True,
    save_data: bool = False,
    source_jsonl_path: Optional[str] = None,
    model_name: str = None,
    no_cache_strategy: Optional[str] = None,  # None: 不检查, "qa": 检查QA错误, "key": 检查关键帧错误
    compress_quality: Optional[float] = None,  # 压缩质量（0-1），如果为None则不压缩
    use_keyframe_images: bool = False,  # 是否传入关键帧图片
    use_qa_images: bool = False,  # 是否传入关键帧图片（用于QA判断）
    no_qa: bool = False,  # 是否跳过QA相关的prompt判定
    completion_ground_truth: Optional[int] = None,  # 新增：用于 completion_judgment 的 ground_truth
    trajectory_ground_truth: Optional[int] = None,  # 新增：用于 trajectory_correctness_judgment 的 ground_truth
    soft_cache_mode: bool = False,  # 新增：软缓存模式，只跳过 judgment 缓存
    enable_timing: bool = False,  # 新增：是否启用时延统计
    takeover_ground_truth_list: Optional[List[int]] = None,  # 新增：每个步骤的接管真值列表
    skip_takeover: bool = True,  # 是否跳过步骤级接管判定，默认跳过
    eval_profile: Optional[str] = None,  # 用于消融/配置：如 lite, lite+action_list, full_unified
    pipeline_json: Optional[Dict[str, Any]] = None,  # 自定义 spec JSON（直接覆盖内置字段）
    pipeline_preset: Optional[str] = None,  # pipeline.json 的 preset：用于 tmp/tmp2/prompt 缓存隔离
    is_negative_task: bool = False,  # 任务类型：True 表示「负例」，在 unified_binary 管道中启用负例后处理
    prior_reasoning: str = "",  # 错因分析管道：仅允许传入本轮 lite gate 产生的 reasoning
    prior_task_status: str = "",  # 错因分析管道：仅允许传入本轮 lite gate 产生的 task_status
    error_cause_lite_gate_all: bool = False,  # 错因分析管道：True 时所有任务都先过 lite 门控（无 filter 全量模式）
    error_cause_enable_trajectory: bool = False,  # 错因分析管道：部分 filter 时是否额外跑轨迹正确性
    source_task_status: str = "",  # 输入 JSONL 原始 evaluation.prediction.task_status，仅用于路由
    client_report: Optional[Dict[str, Any]] = None,  # 顶层 client_report，用于 lite gate 判定用户主动停止等场景
    created_at: str = "",  # 顶层 created_at，用于相对时间任务换算
    prm_mode: bool = False,  # metadata.list_key_index 非空时启用关键步骤/残缺步骤模式
    key_step_indices: Optional[List[int]] = None,  # PRM 1-based 关键步骤索引
    invalid_key_step_indices: Optional[List[int]] = None,
    use_task_knowledge: bool = True,  # False 时不检索/不注入外挂任务知识（--no-task-knowledge）
    judge_takeover: str = "end-to-end",
) -> Dict:
    """
    处理单个任务，执行完整的判断流程
    
    Args:
        instruction: 任务指令
        trajectory: 操作序列
        task_id: 任务ID（用于缓存）
        use_cache: 是否使用缓存
        lite_mode: 是否使用简略模式（提速第四步）
        ground_truth: 任务真实完成状态（1=完成, 0=未完成）
        verbose: 是否输出详细步骤信息
        save_data: 是否保存任务数据到data/save目录
        source_jsonl_path: 原始JSONL文件路径
        model_name: 模型名称（用于目录隔离）gpt-4o glm-4.5v
        no_cache_strategy: 缓存策略 (None: 不检查, "qa": 检查QA错误, "key": 检查关键帧错误)
        compress_quality: 压缩质量（0-1），如果为None则不压缩
        use_keyframe_images: 是否传入关键帧图片
        use_qa_images: 是否传入关键帧图片（用于QA判断）
        no_qa: 是否跳过QA相关的prompt判定
        completion_ground_truth: 完成度判断的 ground_truth（如果存在则优先使用，用于保存文件名）
        trajectory_ground_truth: 轨迹正确性判断的 ground_truth（如果存在则优先使用，用于保存文件名）
        soft_cache_mode: 软缓存模式，只跳过 judgment 缓存，其他步骤正常使用缓存
        enable_timing: 是否启用时延统计
        
    Returns:
        Dict: 处理结果
    """
    result = {
        "instruction": instruction,
        "trajectory_length": len(trajectory),
        "category": None,
        "slots": {},
        "qa_pairs": [],
        "action_thought_caption": [],
        "takeover_judgment": [],
        "keyframes": {},
        "judgment": {},
        "error": None,
        "cache_used": {}
    }

    # 构造评估流水线 spec（用于消融实验和 profile_id 隔离）
    if pipeline_json:
        pipeline_spec = load_pipeline_spec_from_json(pipeline_json)
    else:
        pipeline_spec = preset_from_legacy_flags(
            lite_mode=lite_mode,
            lite_action_list=lite_action_list,
            eval_profile=eval_profile,
            use_keyframe_images=use_keyframe_images,
            use_qa_images=use_qa_images,
            no_qa=no_qa,
            skip_takeover=skip_takeover,
            compress_quality=compress_quality,
        )

    eval_lite_mode = pipeline_spec.eval_lite_mode
    output_lite_mode = pipeline_spec.output_lite_mode
    profile_id = pipeline_spec.profile_id

    result["lite_mode"] = output_lite_mode
    result["judge_takeover"] = normalize_takeover_judge_mode(judge_takeover)
    result["is_negative_task"] = bool(is_negative_task)
    result["prm_mode"] = bool(prm_mode)
    if prm_mode:
        result["key_step_indices"] = key_step_indices or []
        result["invalid_key_step_indices"] = invalid_key_step_indices or []
    result["pipeline_spec"] = {
        "preset": pipeline_spec.preset,
        "profile_id": profile_id,
        "eval_lite_mode": eval_lite_mode,
        "output_lite_mode": output_lite_mode,
        "final_judgment_mode": pipeline_spec.final_judgment_mode,
        "lite_include_action_list": pipeline_spec.lite_include_action_list,
        "lite_action_chain_source": pipeline_spec.lite_action_chain_source,
        "unified_scope": pipeline_spec.unified_scope,
        "keep_last_action": pipeline_spec.keep_last_action,
    }

    # pipeline 缓存隔离：线程局部 preset（勿写 os.environ，否则多线程服务会互相覆盖 tmp 路径）
    pipeline_preset_final = pipeline_preset or pipeline_spec.preset

    # 初始化时延统计
    if enable_timing:
        result["timing"] = {}
        task_start_time = time.perf_counter()
    
    # 设置模型名称（全局）
    if model_name:
        set_model_name(model_name)
    
    # 检查并清除错误预测的缓存
    cache_cleared = False
    if save_data and task_id:
        cache_cleared = check_and_clear_error_prediction_cache(
            task_id,
            model_name,
            save_data,
            output_lite_mode,
            verbose,
            no_cache_strategy,
            pipeline_profile_id=profile_id,
            pipeline_preset=pipeline_preset_final,
        )
        if cache_cleared:
            if verbose:
                print("  由于发现错误预测，将重新生成相关结果（不使用缓存）")
    
    try:
        if pipeline_preset_final:
            bind_pipeline_preset(pipeline_preset_final)

        # Lite 快路径评估：跳过步骤 1～5（末尾综合 / 轨迹全图 二选一）
        if eval_lite_mode:
            lite_start = time.perf_counter() if enable_timing else None
            if pipeline_spec.final_judgment_mode == "trajectory_correctness_all_images":
                if verbose:
                    print("Lite 快路径：轨迹正确性（全图）判断...")
                judgment, cache_used = run_trajectory_correctness_all_images_judgment(
                    instruction=instruction,
                    trajectory=trajectory,
                    task_id=task_id,
                    use_cache=use_cache,
                    cache_cleared=cache_cleared,
                    verbose=False,
                    profile_id=profile_id,
                    action_chain_source=pipeline_spec.lite_action_chain_source,
                    compress_quality=compress_quality,
                )
                result["cache_used"]["judgment_trajectory_all_images"] = cache_used
                if verbose and cache_used:
                    print("  [使用缓存]")
                if verbose:
                    print(
                        f"  轨迹正确性判断：{judgment.get('trajectory_correctness_judgment', -1)}"
                    )
            elif pipeline_spec.final_judgment_mode == "error_cause_analysis":
                # 全量错因分析先做 gate 判定：成功类跳过错因分析；失败进入错因分析。
                # 使用全图 unified gate，避免末尾 10 图漏看前序关键失败；知识库沿用 CLI/服务传入值。
                should_gate = error_cause_lite_gate_all or bool(prm_mode)
                eff_prior_reasoning = ""
                eff_prior_task_status = ""
                gate_skip = False
                lite_judgment: Dict[str, Any] = {}
                lite_cache_used = False
                if should_gate:
                    if verbose:
                        print("Lite 门控：全图 unified 二分类预判...")
                    gate_spec = _build_error_cause_lite_gate_spec(
                        action_chain_source=pipeline_spec.lite_action_chain_source,
                        skip_takeover=skip_takeover,
                        keep_last_action=pipeline_spec.keep_last_action,
                        compress_quality=compress_quality,
                    )
                    lite_judgment, lite_cache_used = run_lite_trajectory_unified_all_images_binary_judgment(
                        instruction=instruction,
                        trajectory=trajectory,
                        task_id=task_id,
                        use_cache=use_cache,
                        cache_cleared=cache_cleared,
                        verbose=False,
                        profile_id=gate_spec.profile_id,
                        lite_include_action_list=gate_spec.lite_include_action_list,
                        lite_action_chain_source=gate_spec.lite_action_chain_source,
                        compress_quality=compress_quality,
                        is_negative_task=is_negative_task,
                        client_report=client_report,
                        created_at=created_at,
                        prm_mode=prm_mode,
                        key_step_indices=key_step_indices,
                        invalid_key_step_indices=invalid_key_step_indices,
                        use_task_knowledge=use_task_knowledge,
                        judge_takeover=judge_takeover,
                    )
                    if not _is_valid_lite_gate_judgment(lite_judgment):
                        if verbose:
                            print(
                                "  lite 门控无有效结果，绕过缓存重试一次："
                                f"{_describe_lite_gate_invalid(lite_judgment)}"
                            )
                        lite_judgment, lite_cache_used = run_lite_trajectory_unified_all_images_binary_judgment(
                            instruction=instruction,
                            trajectory=trajectory,
                            task_id=task_id,
                            use_cache=False,
                            cache_cleared=True,
                            verbose=False,
                            profile_id=gate_spec.profile_id,
                            lite_include_action_list=gate_spec.lite_include_action_list,
                            lite_action_chain_source=gate_spec.lite_action_chain_source,
                            compress_quality=compress_quality,
                            is_negative_task=is_negative_task,
                            client_report=client_report,
                            created_at=created_at,
                            prm_mode=prm_mode,
                            key_step_indices=key_step_indices,
                            invalid_key_step_indices=invalid_key_step_indices,
                            use_task_knowledge=use_task_knowledge,
                            judge_takeover=judge_takeover,
                        )
                    if not _is_valid_lite_gate_judgment(lite_judgment):
                        raise RuntimeError(
                            "lite gate produced no valid judgment after retry: "
                            f"{_describe_lite_gate_invalid(lite_judgment)}"
                        )
                    result["cache_used"]["judgment_lite_gate"] = lite_cache_used
                    lite_final = lite_judgment.get("final_judgment", -1)
                    lite_reasoning = _lite_gate_reasoning_text(lite_judgment)
                    if verbose:
                        print(f"  lite 门控判定：{lite_final}")
                    force_error_cause_after_success = _should_force_error_cause_after_lite_success(
                        instruction,
                        trajectory,
                    )
                    if (
                        lite_final == 1
                        and not prm_mode
                        and not force_error_cause_after_success
                    ):
                        # lite 判成功：不参与错因分析
                        gate_skip = True
                        judgment = {
                            "error_cause_skipped": True,
                            "lite_final_judgment": lite_final,
                            "task_status": lite_judgment.get("task_status", ""),
                            "lite_reasoning": lite_reasoning,
                        }
                        _merge_lite_gate_trajectory_fields(judgment, lite_judgment)
                    else:
                        # lite 判失败：用 lite reasoning 作为先验进入错因分析
                        if not force_error_cause_after_success:
                            stripped = lite_reasoning.strip()
                            if stripped.startswith("<lite判定>"):
                                stripped = stripped[len("<lite判定>"):].strip()
                            if stripped:
                                eff_prior_reasoning = stripped
                            eff_prior_task_status = "failed"
                if not gate_skip:
                    if verbose:
                        print("Lite 快路径：错因分析（全图原始轨迹）...")
                    judgment, cache_used = run_error_cause_analysis_judgment(
                        instruction=instruction,
                        trajectory=trajectory,
                        task_id=task_id,
                        use_cache=use_cache,
                        cache_cleared=cache_cleared,
                        verbose=False,
                        profile_id=profile_id,
                        compress_quality=compress_quality,
                        prior_reasoning=eff_prior_reasoning,
                        prior_task_status=eff_prior_task_status,
                        prm_mode=prm_mode,
                        key_step_indices=key_step_indices,
                        invalid_key_step_indices=invalid_key_step_indices,
                        judge_takeover=judge_takeover,
                    )
                    result["cache_used"]["judgment_error_cause"] = cache_used
                    if prm_mode:
                        judgment["prm_mode"] = True
                        judgment["key_step_indices"] = key_step_indices or []
                        judgment["invalid_key_step_indices"] = invalid_key_step_indices or []
                    if should_gate:
                        judgment["lite_final_judgment"] = lite_final
                        judgment["lite_task_status"] = lite_judgment.get("task_status", "")
                        judgment["lite_reasoning"] = lite_reasoning
                        _merge_lite_gate_trajectory_fields(judgment, lite_judgment)
                        if not prm_mode:
                            judgment["task_status"] = _error_cause_structured_task_status(
                                judgment,
                                lite_judgment.get("task_status", ""),
                            ) or lite_judgment.get("task_status", judgment.get("task_status", ""))
                    if not prm_mode:
                        _apply_wrong_cast_entry_override(
                            judgment,
                            instruction,
                            trajectory,
                        )
                    if verbose and cache_used:
                        print("  [使用缓存]")
                    if verbose:
                        print(
                            f"  错因分析：{judgment.get('primary_error_cause', '')} / "
                            f"{judgment.get('secondary_error_cause', '')}"
                        )
                elif verbose:
                    print("  lite 门控判成功，跳过错因分析")
                if (
                    (not should_gate)
                    and error_cause_enable_trajectory
                    and source_task_status in {"failed", "env_error"}
                ):
                    if verbose:
                        print("Lite 快路径：补充轨迹正确性判断（全图 unified）...")
                    traj_gate_spec = _build_error_cause_lite_gate_spec(
                        action_chain_source=pipeline_spec.lite_action_chain_source,
                        skip_takeover=skip_takeover,
                        keep_last_action=pipeline_spec.keep_last_action,
                        compress_quality=compress_quality,
                    )
                    traj_judgment, traj_cache_used = run_lite_trajectory_unified_all_images_binary_judgment(
                        instruction=instruction,
                        trajectory=trajectory,
                        task_id=task_id,
                        use_cache=use_cache,
                        cache_cleared=cache_cleared,
                        verbose=False,
                        profile_id=traj_gate_spec.profile_id,
                        lite_include_action_list=traj_gate_spec.lite_include_action_list,
                        lite_action_chain_source=traj_gate_spec.lite_action_chain_source,
                        compress_quality=compress_quality,
                        is_negative_task=is_negative_task,
                        client_report=client_report,
                        created_at=created_at,
                        prm_mode=prm_mode,
                        key_step_indices=key_step_indices,
                        invalid_key_step_indices=invalid_key_step_indices,
                        use_task_knowledge=use_task_knowledge,
                        judge_takeover=judge_takeover,
                    )
                    result["cache_used"]["judgment_trajectory_gate"] = traj_cache_used
                    _merge_lite_gate_trajectory_fields(judgment, traj_judgment)
                    if verbose and traj_cache_used:
                        print("  [使用缓存]")
            elif pipeline_spec.final_judgment_mode == "lite_plus_trajectory_all_images_unified_binary":
                if verbose:
                    print("Lite 快路径：融合二分类（单次调用，全图）评估...")
                judgment, cache_used_unified = run_lite_trajectory_unified_all_images_binary_judgment(
                    instruction=instruction,
                    trajectory=trajectory,
                    task_id=task_id,
                    use_cache=use_cache,
                    cache_cleared=cache_cleared,
                    verbose=False,
                    profile_id=profile_id,
                    lite_include_action_list=pipeline_spec.lite_include_action_list,
                    lite_action_chain_source=pipeline_spec.lite_action_chain_source,
                    compress_quality=compress_quality,
                    is_negative_task=is_negative_task,
                    client_report=client_report,
                    created_at=created_at,
                    use_task_knowledge=use_task_knowledge,
                    judge_takeover=judge_takeover,
                )
                result["cache_used"]["judgment_unified_binary_all_images"] = cache_used_unified
                if verbose and cache_used_unified:
                    print("  [使用缓存]")
                if verbose:
                    print(
                        f"  综合判断：{judgment.get('final_judgment', -1)}; "
                        f"轨迹正确性判断：{judgment.get('trajectory_correctness_judgment', -1)}; "
                        f"完成度判断：{judgment.get('completion_judgment', -1)}"
                    )
            elif pipeline_spec.final_judgment_mode == "lite_plus_trajectory_all_images":
                if verbose:
                    print("Lite 快路径：综合判断 + 轨迹正确性（全图）联合评估...")
                judgment_lite, cache_used_lite = run_lite_endframe_judgment(
                    instruction=instruction,
                    trajectory=trajectory,
                    task_id=task_id,
                    use_cache=use_cache,
                    cache_cleared=cache_cleared,
                    verbose=False,
                    profile_id=profile_id,
                    lite_include_action_list=pipeline_spec.lite_include_action_list,
                    lite_action_chain_source=pipeline_spec.lite_action_chain_source,
                    compress_quality=compress_quality,
                    use_task_knowledge=use_task_knowledge,
                )
                judgment_traj, cache_used_traj = run_trajectory_correctness_all_images_judgment(
                    instruction=instruction,
                    trajectory=trajectory,
                    task_id=task_id,
                    use_cache=use_cache,
                    cache_cleared=cache_cleared,
                    verbose=False,
                    profile_id=profile_id,
                    action_chain_source=pipeline_spec.lite_action_chain_source,
                    compress_quality=compress_quality,
                )
                result["cache_used"]["judgment_lite"] = cache_used_lite
                result["cache_used"]["judgment_trajectory_all_images"] = cache_used_traj
                judgment = {
                    "final_judgment": judgment_lite.get("final_judgment", -1),
                    "trajectory_correctness_judgment": judgment_traj.get("trajectory_correctness_judgment", -1),
                    "lite_reasoning": judgment_lite.get("reasoning", ""),
                    "trajectory_reasoning": judgment_traj.get("reasoning", ""),
                }
                for _ek in ("is_env_error", "env_error_type", "env_error_reason"):
                    if _ek in judgment_lite:
                        judgment[_ek] = judgment_lite[_ek]
                if isinstance(judgment_traj.get("first_deviation_step"), int):
                    judgment["first_deviation_step"] = judgment_traj["first_deviation_step"]
                fdr = judgment_traj.get("first_deviation_reason")
                if isinstance(fdr, str) and fdr.strip():
                    judgment["first_deviation_reason"] = fdr.strip()
                if verbose:
                    print(
                        f"  综合判断：{judgment.get('final_judgment', -1)}; "
                        f"轨迹正确性判断：{judgment.get('trajectory_correctness_judgment', -1)}"
                    )
            else:
                if verbose:
                    print("Lite 模式：轨迹末尾综合判断...")
                judgment, cache_used = run_lite_endframe_judgment(
                    instruction=instruction,
                    trajectory=trajectory,
                    task_id=task_id,
                    use_cache=use_cache,
                    cache_cleared=cache_cleared,
                    verbose=False,
                    profile_id=profile_id,
                    lite_include_action_list=pipeline_spec.lite_include_action_list,
                    lite_action_chain_source=pipeline_spec.lite_action_chain_source,
                    compress_quality=compress_quality,
                    use_task_knowledge=use_task_knowledge,
                )
                result["cache_used"]["judgment_lite"] = cache_used
                if verbose and cache_used:
                    print("  [使用缓存]")
                if verbose:
                    print(f"  综合判断：{result['judgment'].get('final_judgment', -1)}")
            result["judgment"] = judgment
            if enable_timing:
                result["timing"]["step_lite_judgment"] = time.perf_counter() - lite_start
            if save_data and task_id:
                save_start = time.perf_counter() if enable_timing else None
                save_task_data(
                    task_id, trajectory, [], result["judgment"], ground_truth,
                    source_jsonl_path, model_name, compress_quality,
                    completion_ground_truth=completion_ground_truth,
                    trajectory_ground_truth=trajectory_ground_truth,
                    takeover_judgment=[],
                    lite_mode=output_lite_mode,
                    pipeline_profile_id=profile_id,
                )
                if enable_timing:
                    result["timing"]["save_data"] = time.perf_counter() - save_start
            elif enable_timing:
                result["timing"]["save_data"] = 0.0
            if enable_timing:
                result["timing"]["total"] = time.perf_counter() - task_start_time
            return result

        # 步骤1：任务分类
        if verbose:
            print("步骤1：任务分类...")
        step1_start = time.perf_counter() if enable_timing else None
        category, cache_used_category = run_category_stage(
            instruction=instruction,
            task_id=task_id,
            use_cache=use_cache,
            cache_cleared=cache_cleared,
            verbose=False,
        )
        result["category"] = category
        result["cache_used"]["category"] = cache_used_category
        if verbose:
            if cache_used_category:
                print("  [使用缓存]")
            print(f"  分类：{category} ({get_category_name(category)})")
        if enable_timing:
            result["timing"]["step1_category"] = time.perf_counter() - step1_start
        
        # 步骤2：槽位提取（策略生成步骤已移除，现在直接从策略库加载新格式策略）
        step2_start = time.perf_counter() if enable_timing else None
        if verbose:
            if no_qa:
                print("步骤2：槽位提取...（已跳过，--no-qa 模式）")
            else:
                print("步骤2：槽位提取...")
        slots, qa_pairs, cache_used_slots_qa = run_slots_qa_stage(
            instruction=instruction,
            task_id=task_id,
            use_cache=use_cache,
            no_qa=no_qa,
            cache_cleared=cache_cleared,
            verbose=False,
        )
        result["slots"] = slots
        result["qa_pairs"] = qa_pairs
        result["cache_used"]["slots_qa"] = cache_used_slots_qa
        if verbose and not no_qa:
            if cache_used_slots_qa:
                print("  [使用缓存]")
            print(f"  提取槽位数：{len(slots)}")
            print(f"  生成QA对数：{len(qa_pairs)}")
        if enable_timing:
            if no_qa:
                result["timing"]["step2_slots_qa"] = 0.0
            else:
                result["timing"]["step2_slots_qa"] = time.perf_counter() - step2_start
        
        # 步骤3：Action/Thought/Caption + 接管判断（统一入口）
        if verbose:
            print("步骤3：生成Action/Thought/Caption...")
        step3_start = time.perf_counter() if enable_timing else None
        action_thought_caption_results, takeover_judgment_results, cache_used_atc, cache_used_takeover = run_atc_stage(
            instruction=instruction,
            trajectory=trajectory,
            task_id=task_id,
            use_cache=use_cache,
            lite_mode=eval_lite_mode,
            takeover_ground_truth_list=takeover_ground_truth_list,
            soft_cache_mode=soft_cache_mode,
            skip_takeover=skip_takeover,
            profile_id=profile_id,
            verbose=False,
            parallel_caption=True,
            judge_takeover=judge_takeover,
        )
        result["cache_used"]["action_thought_caption"] = cache_used_atc
        result["cache_used"]["takeover_judgment"] = cache_used_takeover
        if verbose:
            if cache_used_atc:
                print("  [使用缓存]")
            print(f"  生成步骤数：{len(action_thought_caption_results)}")
        if enable_timing:
            result["timing"]["step3_action_thought_caption"] = time.perf_counter() - step3_start
        result["action_thought_caption"] = action_thought_caption_results

        if not skip_takeover:
            if verbose:
                print("步骤3.5：接管判断...")
                if cache_used_takeover:
                    print("  [使用缓存]")
                print(f"  判断步骤数：{len(takeover_judgment_results)}")
            if enable_timing:
                result["timing"]["step3_5_takeover_judgment"] = 0.0
        else:
            if verbose:
                print("步骤3.5：接管判断...（已跳过）")
        result["takeover_judgment"] = takeover_judgment_results

        # 步骤4：关键帧提取
        if verbose:
            print("步骤4：关键帧提取...")
        step4_start = time.perf_counter() if enable_timing else None
        keyframes, cache_used_keyframes = run_keyframes_stage(
            instruction=instruction,
            task_id=task_id,
            use_cache=use_cache,
            profile_id=profile_id,
            action_thought_caption_results=action_thought_caption_results,
            category=category,
            verbose=False,
        )
        result["keyframes"] = keyframes
        result["cache_used"]["keyframes"] = cache_used_keyframes
        if verbose:
            if cache_used_keyframes:
                print("  [使用缓存]")
            print(f"  关键帧数：{len(keyframes['keyframe_indices'])}")
        if enable_timing:
            result["timing"]["step4_keyframes"] = time.perf_counter() - step4_start
        
        # 步骤5：任务完成判断
        if verbose:
            print("步骤5：任务完成判断...")
        step5_start = time.perf_counter() if enable_timing else None
        judgment, cache_used_judgment = run_final_judgment_stage(
            instruction=instruction,
            task_id=task_id,
            use_cache=use_cache,
            cache_cleared=cache_cleared,
            soft_cache_mode=soft_cache_mode,
            verbose=False,
            profile_id=profile_id,
            final_judgment_mode=pipeline_spec.final_judgment_mode,
            qa_pairs=qa_pairs,
            keyframes=keyframes,
            action_thought_caption_results=action_thought_caption_results,
            ground_truth=ground_truth,
            category=category,
            use_keyframe_images=use_keyframe_images,
            use_qa_images=use_qa_images,
            no_qa=no_qa,
        )
        result["judgment"] = judgment
        result["cache_used"]["judgment"] = cache_used_judgment
        if verbose:
            if "completion_judgment" in judgment or "trajectory_correctness_judgment" in judgment:
                print(f"  QA对判断：{judgment.get('qa_judgment', -1)}")
                print(f"  关键帧判断：{judgment.get('keyframe_judgment', -1)}")
                print(f"  完成度判断：{judgment.get('completion_judgment', -1)}")
                print(f"  轨迹正确性判断：{judgment.get('trajectory_correctness_judgment', -1)}")
            print(f"  综合判断：{judgment.get('final_judgment', -1)}")
        if enable_timing:
            result["timing"]["step5_judgment"] = time.perf_counter() - step5_start
        
        # 如果启用保存，保存任务数据
        if save_data and task_id:
            save_start = time.perf_counter() if enable_timing else None
            save_task_data(
                task_id, trajectory, qa_pairs, judgment, ground_truth, 
                source_jsonl_path, model_name, compress_quality,
                completion_ground_truth=completion_ground_truth,
                trajectory_ground_truth=trajectory_ground_truth,
                takeover_judgment=result.get("takeover_judgment", []),
                lite_mode=output_lite_mode,
                pipeline_profile_id=profile_id,
            )
            if enable_timing:
                result["timing"]["save_data"] = time.perf_counter() - save_start
        elif enable_timing:
            result["timing"]["save_data"] = 0.0  # 未启用保存
        
        # 计算总耗时
        if enable_timing:
            result["timing"]["total"] = time.perf_counter() - task_start_time
        
    except Exception as e:
        result["error"] = str(e)
        logger.error("process_task failed task_id=%s error=%s", task_id, e)
        logger.error(traceback.format_exc())
        if verbose:
            print(f"  错误：{e}")
            traceback.print_exc()
    finally:
        if pipeline_preset_final:
            unbind_pipeline_preset()

    return result


def _compute_judgment_stats_from_results(results: List[Dict]) -> Dict:
    """
    基于本次运行结果（内存中的 judgment）统计准确率，不依赖 save 目录。
    返回与打印逻辑兼容的 statistics 字典。
    """
    total = len(results)
    valid_by_type = {"completion": 0, "trajectory_correctness": 0, "final": 0}
    correct_by_type = {"completion": 0, "trajectory_correctness": 0, "final": 0}

    # 针对最终判定（is_task_success / final_judgment）的混淆矩阵统计，
    # 口径与 test_service.py 一致：ground_truth 从 evaluation.status 归一化而来，
    # pred_success 为 0/1（-1 视为无效，不参与统计）。
    gt1_total = 0      # 真值为 1 的条数 (TP+FN)
    gt1_correct = 0    # 真值为 1 且预测为 1 (TP)
    gt0_total = 0      # 真值为 0 的条数 (TN+FP)
    gt0_correct = 0    # 真值为 0 且预测为 0 (TN)
    pred1_total = 0    # 预测为 1 且 gt 已知 (TP+FP)
    pred0_total = 0    # 预测为 0 且 gt 已知 (TN+FN)

    for result in results:
        judgment = result.get("judgment", {})
        ground_truth = result.get("ground_truth", None)
        gt_completion = result.get("is_task_completed", None)
        gt_trajectory = result.get("is_trajectory_correct", None)

        # Lite 模式：judgment 仅含 final_judgment，无 completion_judgment / trajectory_correctness_judgment
        if "completion_judgment" not in judgment:
            pred_success = judgment.get("final_judgment", -1)
            if pred_success != -1:
                valid_by_type["final"] += 1
                if ground_truth is not None:
                    if ground_truth == 1:
                        gt1_total += 1
                        if pred_success == 1:
                            gt1_correct += 1
                    elif ground_truth == 0:
                        gt0_total += 1
                        if pred_success == 0:
                            gt0_correct += 1
                    if pred_success == 1:
                        pred1_total += 1
                    elif pred_success == 0:
                        pred0_total += 1
                    if pred_success == ground_truth:
                        correct_by_type["final"] += 1
            continue

        pred_completion = judgment.get("completion_judgment", -1)
        pred_trajectory = judgment.get("trajectory_correctness_judgment", -1)
        if pred_completion == 1 and pred_trajectory == 1:
            pred_success = 1
        elif pred_completion == -1 or pred_trajectory == -1:
            pred_success = -1
        else:
            pred_success = 0

        if pred_completion != -1:
            valid_by_type["completion"] += 1
            if gt_completion is not None and pred_completion == gt_completion:
                correct_by_type["completion"] += 1
        if pred_trajectory != -1:
            valid_by_type["trajectory_correctness"] += 1
            if gt_trajectory is not None and pred_trajectory == gt_trajectory:
                correct_by_type["trajectory_correctness"] += 1
        if pred_success != -1:
            valid_by_type["final"] += 1
            if ground_truth is not None:
                if ground_truth == 1:
                    gt1_total += 1
                    if pred_success == 1:
                        gt1_correct += 1
                elif ground_truth == 0:
                    gt0_total += 1
                    if pred_success == 0:
                        gt0_correct += 1
                if pred_success == 1:
                    pred1_total += 1
                elif pred_success == 0:
                    pred0_total += 1
                if pred_success == ground_truth:
                    correct_by_type["final"] += 1

    valid_count = valid_by_type["final"]
    correct_count = correct_by_type["final"]
    accuracy = (correct_count / valid_count * 100) if valid_count > 0 else 0.0
    acc_completion = (correct_by_type["completion"] / valid_by_type["completion"] * 100) if valid_by_type["completion"] > 0 else 0.0
    acc_trajectory = (correct_by_type["trajectory_correctness"] / valid_by_type["trajectory_correctness"] * 100) if valid_by_type["trajectory_correctness"] > 0 else 0.0
    acc_final = (correct_by_type["final"] / valid_by_type["final"] * 100) if valid_by_type["final"] > 0 else 0.0

    # 统一计算最终判定上的各类召回/精度。
    overall_valid_for_confusion = gt1_total + gt0_total
    overall_correct_for_confusion = gt1_correct + gt0_correct
    overall_accuracy = (overall_correct_for_confusion / overall_valid_for_confusion * 100) if overall_valid_for_confusion > 0 else 0.0
    success_recall = (gt1_correct / gt1_total * 100) if gt1_total > 0 else 0.0
    success_precision = (gt1_correct / pred1_total * 100) if pred1_total > 0 else 0.0
    failed_recall = (gt0_correct / gt0_total * 100) if gt0_total > 0 else 0.0
    failed_precision = (gt0_correct / pred0_total * 100) if pred0_total > 0 else 0.0

    return {
        "total": total,
        "valid_count": valid_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "by_type": {
            "completion": {"valid": valid_by_type["completion"], "correct": correct_by_type["completion"], "accuracy": acc_completion},
            "trajectory_correctness": {"valid": valid_by_type["trajectory_correctness"], "correct": correct_by_type["trajectory_correctness"], "accuracy": acc_trajectory},
            "final": {"valid": valid_by_type["final"], "correct": correct_by_type["final"], "accuracy": acc_final},
        },
        "confusion_final": {
            "gt1_total": gt1_total,
            "gt1_correct": gt1_correct,
            "gt0_total": gt0_total,
            "gt0_correct": gt0_correct,
            "pred1_total": pred1_total,
            "pred0_total": pred0_total,
            "overall_accuracy": overall_accuracy,
            "success_recall": success_recall,
            "success_precision": success_precision,
            "failed_recall": failed_recall,
            "failed_precision": failed_precision,
        },
    }


def _task_knowledge_match_from_judgment(judgment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(judgment, dict):
        return None
    match = judgment.get("task_knowledge_match")
    if isinstance(match, dict):
        return match
    if "task_knowledge_hit" in judgment:
        return {
            "hit": bool(judgment.get("task_knowledge_hit")),
            "matched_id": "",
            "matched_query": judgment.get("task_knowledge_used", ""),
            "confidence": judgment.get("task_knowledge_confidence", 0.0),
            "match_reasoning": judgment.get("task_knowledge_reasoning", ""),
        }
    return None


def _compute_task_knowledge_stats_from_results(results: List[Dict]) -> Dict[str, Any]:
    total = 0
    hit_count = 0
    no_hit_count = 0
    missing_count = 0
    id_counts: Dict[str, int] = {}
    query_counts: Dict[str, int] = {}

    for result in results:
        if not isinstance(result, dict) or result.get("_original_task_data") is None:
            continue
        total += 1
        match = _task_knowledge_match_from_judgment(result.get("judgment", {}))
        if not isinstance(match, dict):
            missing_count += 1
            continue
        if not match.get("hit"):
            no_hit_count += 1
            continue
        hit_count += 1
        matched_id = str(match.get("matched_id") or "").strip() or "<unknown>"
        matched_query = str(match.get("matched_query") or "").strip() or "<empty>"
        id_counts[matched_id] = id_counts.get(matched_id, 0) + 1
        query_counts[matched_query] = query_counts.get(matched_query, 0) + 1

    return {
        "total": total,
        "hit_count": hit_count,
        "no_hit_count": no_hit_count,
        "missing_count": missing_count,
        "hit_rate": (hit_count / total * 100.0) if total else 0.0,
        "id_counts": id_counts,
        "query_counts": query_counts,
    }


def _print_task_knowledge_stats(stats: Dict[str, Any], *, top_n: int = 10) -> None:
    total = int(stats.get("total") or 0)
    if total <= 0:
        return
    hit_count = int(stats.get("hit_count") or 0)
    no_hit_count = int(stats.get("no_hit_count") or 0)
    missing_count = int(stats.get("missing_count") or 0)
    hit_rate = float(stats.get("hit_rate") or 0.0)
    print("\n任务知识命中统计:")
    print(f"  总任务数: {total}")
    print(f"  命中知识数: {hit_count}/{total} = {hit_rate:.2f}%")
    print(f"  未命中数: {no_hit_count}")
    if missing_count:
        print(f"  无知识检索字段数: {missing_count}")

    id_counts = stats.get("id_counts") if isinstance(stats.get("id_counts"), dict) else {}
    query_counts = stats.get("query_counts") if isinstance(stats.get("query_counts"), dict) else {}
    if id_counts:
        print(f"  命中知识ID Top{top_n}:")
        for matched_id, count in sorted(id_counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]:
            print(f"    {matched_id}: {count}")
    if query_counts:
        print(f"  命中任务 Top{top_n}:")
        for query, count in sorted(query_counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]:
            display = query if len(query) <= 80 else query[:77] + "..."
            print(f"    {count} - {display}")


def _attach_task_knowledge_fields_to_orm_prediction(
    orm_prediction: Dict[str, Any],
    judgment: Dict[str, Any],
) -> None:
    match = _task_knowledge_match_from_judgment(judgment)
    if not isinstance(match, dict):
        return
    orm_prediction["task_knowledge_hit"] = bool(match.get("hit"))
    orm_prediction["task_knowledge_matched_id"] = str(match.get("matched_id") or "")
    orm_prediction["task_knowledge_matched_query"] = str(match.get("matched_query") or "")
    orm_prediction["task_knowledge_confidence"] = match.get("confidence", 0.0)
    orm_prediction["task_knowledge_reasoning"] = str(match.get("match_reasoning") or "")


def generate_judgment_summary(results: List[Dict], output_path: str) -> Dict:
    """
    [已废弃] 生成判断结果汇总和准确率统计（依赖 save 目录）。
    请使用 _compute_judgment_stats_from_results(results) 基于本次运行结果统计。
    """
    import config
    from utils.prompt_utils import get_model_name
    
    summary_data = {
        "summary": [],
        "statistics": {}
    }
    
    total = len(results)
    valid_count = 0  # 有效判断数（排除-1）
    correct_count = 0  # 正确数（找到0_to_0.json或1_to_1.json）
    correct_by_type = {"qa": 0, "keyframe": 0, "completion": 0, "trajectory_correctness": 0, "final": 0}  # 各类型判断的正确数
    valid_by_type = {"qa": 0, "keyframe": 0, "completion": 0, "trajectory_correctness": 0, "final": 0}  # 各类型判断的有效数
    
    # 获取模型名称用于检查文件
    model_name = get_model_name()
    if not model_name:
        import config
        model_name = config.DEFAULT_MODEL
    
    for result in results:
        task_id = result.get("task_id", "")
        ground_truth = result.get("ground_truth", None)
        
        if not task_id:
            continue
        
        # 从save目录加载judgment
        judgment = load_judgment_from_save_dir(task_id, ground_truth, model_name)
        
        if judgment is None:
            # 未找到judgment文件
            summary_item = {
                "task_id": task_id,
                "ground_truth": ground_truth,
                "found": False,
                "qa_judgment": -1,
                "keyframe_judgment": -1,
                "completion_judgment": -1,
                "trajectory_correctness_judgment": -1,
                "final_judgment": -1,
                "is_correct": False
            }
            summary_data["summary"].append(summary_item)
            continue
        
        qa_judgment = judgment.get("qa_judgment", -1)
        keyframe_judgment = judgment.get("keyframe_judgment", -1)
        completion_judgment = judgment.get("completion_judgment", -1)
        trajectory_correctness_judgment = judgment.get("trajectory_correctness_judgment", -1)
        final_judgment = judgment.get("final_judgment", -1)
        
        # 判断是否正确：final_judgment 与 ground_truth 一致
        is_correct = (ground_truth is not None and final_judgment != -1 and ground_truth == final_judgment)
        
        # 统计（不依赖ground_truth，只要找到judgment文件就统计）
        # QA判断统计
        if qa_judgment != -1:
            valid_by_type["qa"] += 1
            if ground_truth is not None and qa_judgment == ground_truth:
                correct_by_type["qa"] += 1
        
        # 关键帧判断统计
        if keyframe_judgment != -1:
            valid_by_type["keyframe"] += 1
            if ground_truth is not None and keyframe_judgment == ground_truth:
                correct_by_type["keyframe"] += 1
        
        # 完成度判断统计
        if completion_judgment != -1:
            valid_by_type["completion"] += 1
            if ground_truth is not None and completion_judgment == ground_truth:
                correct_by_type["completion"] += 1
        
        # 轨迹正确性判断统计
        if trajectory_correctness_judgment != -1:
            valid_by_type["trajectory_correctness"] += 1
            if ground_truth is not None and trajectory_correctness_judgment == ground_truth:
                correct_by_type["trajectory_correctness"] += 1
        
        # 最终判断统计（完成度=1 且 轨迹正确性=1 时判对）
        if final_judgment != -1:
            valid_count += 1
            valid_by_type["final"] += 1
            if is_correct:
                correct_count += 1
                correct_by_type["final"] += 1
        
        # 添加到汇总
        summary_item = {
            "task_id": task_id,
            "ground_truth": ground_truth,
            "found": True,
            "qa_judgment": qa_judgment,
            "keyframe_judgment": keyframe_judgment,
            "completion_judgment": completion_judgment,
            "trajectory_correctness_judgment": trajectory_correctness_judgment,
            "final_judgment": final_judgment,
            "is_correct": is_correct
        }
        summary_data["summary"].append(summary_item)
    
    # 计算准确率
    accuracy = (correct_count / valid_count * 100) if valid_count > 0 else 0.0
    qa_accuracy = (correct_by_type["qa"] / valid_by_type["qa"] * 100) if valid_by_type["qa"] > 0 else 0.0
    keyframe_accuracy = (correct_by_type["keyframe"] / valid_by_type["keyframe"] * 100) if valid_by_type["keyframe"] > 0 else 0.0
    completion_accuracy = (correct_by_type["completion"] / valid_by_type["completion"] * 100) if valid_by_type["completion"] > 0 else 0.0
    trajectory_correctness_accuracy = (correct_by_type["trajectory_correctness"] / valid_by_type["trajectory_correctness"] * 100) if valid_by_type["trajectory_correctness"] > 0 else 0.0
    final_accuracy = (correct_by_type["final"] / valid_by_type["final"] * 100) if valid_by_type["final"] > 0 else 0.0
    
    # 统计信息
    summary_data["statistics"] = {
        "total": total,
        "valid_count": valid_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "by_type": {
            "qa": {
                "valid": valid_by_type["qa"],
                "correct": correct_by_type["qa"],
                "accuracy": qa_accuracy
            },
            "keyframe": {
                "valid": valid_by_type["keyframe"],
                "correct": correct_by_type["keyframe"],
                "accuracy": keyframe_accuracy
            },
            "completion": {
                "valid": valid_by_type["completion"],
                "correct": correct_by_type["completion"],
                "accuracy": completion_accuracy
            },
            "trajectory_correctness": {
                "valid": valid_by_type["trajectory_correctness"],
                "correct": correct_by_type["trajectory_correctness"],
                "accuracy": trajectory_correctness_accuracy
            },
            "final": {
                "valid": valid_by_type["final"],
                "correct": correct_by_type["final"],
                "accuracy": final_accuracy
            }
        }
    }
    
    # 保存汇总文件
    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    summary_data["statistics"]["summary_path"] = summary_path
    
    return summary_data["statistics"]


def save_task_data(
    task_id: str,
    trajectory: List[Dict],
    qa_pairs: List[Dict[str, str]],
    judgment: Dict,
    ground_truth: Optional[int] = None,
    source_jsonl_path: Optional[str] = None,
    model_name: Optional[str] = None,
    compress_quality: Optional[float] = None,
    completion_ground_truth: Optional[int] = None,  # 用于 completion_judgment 的 ground_truth
    trajectory_ground_truth: Optional[int] = None,  # 用于 trajectory_correctness_judgment 的 ground_truth
    takeover_judgment: Optional[List[Dict]] = None,
    lite_mode: bool = False,  # Lite 模式：只保存 lite_judgment_{gt}_to_{pre}.json
    pipeline_profile_id: Optional[str] = None,
) -> None:
    """
    保存任务数据到data/{model_name}/save/{task_id}目录
    
    Args:
        lite_mode: True 时只保存 lite_judgment_{gt}_to_{pre}.json，不保存 completion/trajectory 等 judgment 文件。
    """
    try:
        import config
        from utils.ask_qwen import compress_image
        
        save_dir = os.path.join(config.get_save_dir(model_name), task_id)
        if pipeline_profile_id:
            save_dir = os.path.join(save_dir, pipeline_profile_id)
        os.makedirs(save_dir, exist_ok=True)

        # Lite 模式：只保存 lite_judgment_{gt}_to_{pre}.json
        if lite_mode:
            final_judgment = judgment.get("final_judgment", -1)
            if completion_ground_truth is not None and trajectory_ground_truth is not None:
                gt = 1 if (completion_ground_truth == 1 and trajectory_ground_truth == 1) else 0
            else:
                gt = ground_truth if ground_truth is not None else 0
            pre_str = str(final_judgment) if final_judgment != -1 else "-1"
            gt_str = str(gt)
            lite_path = os.path.join(save_dir, f"lite_judgment_{gt_str}_to_{pre_str}.json")
            lite_data = {"is_task_success": final_judgment}
            with open(lite_path, 'w', encoding='utf-8') as f:
                json.dump(lite_data, f, ensure_ascii=False, indent=2)
            return
        
        # 快速路径检查：如果所有必需文件都已存在，跳过保存（优化）
        if os.path.exists(save_dir):
            # 计算预期的文件名（不再包含 qa_judgment / keyframe_judgment 文件）
            completion_judgment = judgment.get("completion_judgment", -1)
            trajectory_correctness_judgment = judgment.get("trajectory_correctness_judgment", -1)
            
            ground_truth_str = str(ground_truth) if ground_truth is not None else "None"
            completion_gt = completion_ground_truth if completion_ground_truth is not None else ground_truth
            completion_gt_str = str(completion_gt) if completion_gt is not None else "None"
            completion_judgment_str = str(completion_judgment) if completion_judgment != -1 else "-1"
            trajectory_gt = trajectory_ground_truth if trajectory_ground_truth is not None else ground_truth
            trajectory_gt_str = str(trajectory_gt) if trajectory_gt is not None else "None"
            trajectory_correctness_judgment_str = str(trajectory_correctness_judgment) if trajectory_correctness_judgment != -1 else "-1"
            
            # 检查所有必需的 judgment 文件是否存在
            required_files = []
            # 不再保存 qa_judgment / keyframe_judgment 文件，此处不加入 required_files
            if completion_judgment != -1:
                required_files.append(os.path.join(save_dir, f"completion_judgment_{completion_gt_str}_to_{completion_judgment_str}.json"))
            if trajectory_correctness_judgment != -1:
                required_files.append(os.path.join(save_dir, f"trajectory_correctness_judgment_{trajectory_gt_str}_to_{trajectory_correctness_judgment_str}.json"))
            
            # 检查接管判断文件（如果存在）
            if takeover_judgment:
                for step_judgment in takeover_judgment:
                    step_index = step_judgment.get("step_index", -1)
                    if step_index >= 0:
                        gt = step_judgment.get("ground_truth", 0)
                        pr = step_judgment.get("prediction", -1)
                        gt_str = str(gt) if gt is not None else "None"
                        pr_str = str(pr) if pr != -1 else "-1"
                        required_files.append(os.path.join(save_dir, f"takeover_judgment_step_{step_index}_{gt_str}_to_{pr_str}.json"))
            
            # 检查所有必需文件是否都存在
            all_files_exist = all(os.path.exists(f) for f in required_files) if required_files else False
            
            # 检查图片文件（如果 compress_quality 不为 None，检查压缩后的图片）
            if all_files_exist and compress_quality is not None:
                all_images_exist = True
                for i, step in enumerate(trajectory):
                    image_path = get_image_path_from_step(step)
                    if image_path:
                        image_ext = os.path.splitext(image_path)[1] or ".png"
                        dest_path = os.path.join(save_dir, f"image_{i}{image_ext}")
                        if not os.path.exists(dest_path):
                            all_images_exist = False
                            break
                if all_images_exist:
                    return  # 快速返回，跳过所有处理
            elif all_files_exist:
                # 不压缩图片时，检查原始图片是否存在
                all_images_exist = True
                for i, step in enumerate(trajectory):
                    image_path = get_image_path_from_step(step)
                    if image_path:
                        image_ext = os.path.splitext(image_path)[1] or ".png"
                        dest_path = os.path.join(save_dir, f"image_{i}{image_ext}")
                        if not os.path.exists(dest_path):
                            all_images_exist = False
                            break
                if all_images_exist:
                    return  # 快速返回，跳过所有处理
        
        for i, step in enumerate(trajectory):
            image_path = get_image_path_from_step(step)
            if image_path and os.path.exists(image_path):
                try:
                    image_ext = os.path.splitext(image_path)[1] or ".png"
                    dest_path = os.path.join(save_dir, f"image_{i}{image_ext}")
                    
                    # 如果需要压缩，先压缩再保存
                    if compress_quality is not None:
                        compressed_path = compress_image(image_path, compress_quality, dest_path)
                        if not compressed_path:
                            # 压缩失败，使用原图
                            shutil.copy2(image_path, dest_path)
                    else:
                        # 不压缩，直接复制
                        shutil.copy2(image_path, dest_path)
                except Exception as e:
                    print(f"  警告: 复制图片失败 {image_path}: {e}")
        
        preset = get_pipeline_preset()
        if preset:
            tmp2_dir = os.path.join(config.get_model_data_dir(model_name), preset, "tmp2")
        else:
            tmp2_dir = config.get_tmp2_dir(model_name)
        safe_task_id = task_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # 按task_id分文件夹
        task_tmp2_dir = os.path.join(tmp2_dir, task_id)
        
        # 优化：一次性查找所有 prompt_response.json 文件，然后按类型分类
        all_prompt_response_files = glob.glob(os.path.join(task_tmp2_dir, f"{safe_task_id}_*_prompt_response.json")) if os.path.exists(task_tmp2_dir) else []
        
        # 按文件类型分类（不再保存 qa_judgment / keyframe_judgment json，仅保留完成度、轨迹等）
        completion_file = next((f for f in all_prompt_response_files if '_judgment_completion_' in os.path.basename(f)), None)
        if completion_file is None:
            completion_file = os.path.join(task_tmp2_dir, f"{safe_task_id}_judgment_completion_prompt_response.json")
        
        trajectory_correctness_file = next((f for f in all_prompt_response_files if '_judgment_trajectory_correctness_' in os.path.basename(f)), None)
        if trajectory_correctness_file is None:
            trajectory_correctness_file = os.path.join(task_tmp2_dir, f"{safe_task_id}_judgment_trajectory_correctness_prompt_response.json")
        
        takeover_files = {f: os.path.basename(f) for f in all_prompt_response_files if '_takeover_judgment_step_' in os.path.basename(f)}
        
        # 不再保存 qa_judgment_*.json 与 keyframe_judgment_*.json（关键帧选择逻辑保留，供后续使用）
        
        # 保存完成度判断结果
        completion_judgment = judgment.get("completion_judgment", -1)
        completion_judgment_data = {
            "prompt_lines": [],
            "response_lines": [],
            "completion_status": completion_judgment
        }
        
        if os.path.exists(completion_file):
            try:
                with open(completion_file, 'r', encoding='utf-8') as f:
                    completion_data = json.load(f)
                
                prompt = completion_data.get("prompt", "")
                response_text = completion_data.get("response_text", "")
                
                completion_judgment_data["prompt_lines"] = prompt.split('\n') if prompt else []
                completion_judgment_data["response_lines"] = response_text.split('\n') if response_text else []
                
                try:
                    parsed_response = json.loads(response_text)
                    if isinstance(parsed_response, dict):
                        completion_judgment_data["parsed_response"] = parsed_response
                except:
                    pass
            except Exception as e:
                print(f"  警告: 读取完成度判断文件失败 {completion_file}: {e}")
        
        completion_judgment_str = str(completion_judgment) if completion_judgment != -1 else "-1"
        # 使用 completion_ground_truth（如果存在），否则使用 ground_truth
        completion_gt = completion_ground_truth if completion_ground_truth is not None else ground_truth
        completion_gt_str = str(completion_gt) if completion_gt is not None else "None"
        completion_judgment_filename = f"completion_judgment_{completion_gt_str}_to_{completion_judgment_str}.json"
        completion_judgment_path = os.path.join(save_dir, completion_judgment_filename)
        with open(completion_judgment_path, 'w', encoding='utf-8') as f:
            json.dump(completion_judgment_data, f, ensure_ascii=False, indent=2)
        
        # 保存轨迹正确性判断结果
        trajectory_correctness_judgment = judgment.get("trajectory_correctness_judgment", -1)
        trajectory_correctness_judgment_data = {
            "prompt_lines": [],
            "response_lines": [],
            "correctness_status": trajectory_correctness_judgment
        }
        
        if os.path.exists(trajectory_correctness_file):
            try:
                with open(trajectory_correctness_file, 'r', encoding='utf-8') as f:
                    trajectory_correctness_data = json.load(f)
                
                prompt = trajectory_correctness_data.get("prompt", "")
                response_text = trajectory_correctness_data.get("response_text", "")
                
                trajectory_correctness_judgment_data["prompt_lines"] = prompt.split('\n') if prompt else []
                trajectory_correctness_judgment_data["response_lines"] = response_text.split('\n') if response_text else []
                
                try:
                    parsed_response = json.loads(response_text)
                    if isinstance(parsed_response, dict):
                        trajectory_correctness_judgment_data["parsed_response"] = parsed_response
                except:
                    pass
            except Exception as e:
                print(f"  警告: 读取轨迹正确性判断文件失败 {trajectory_correctness_file}: {e}")
        
        trajectory_correctness_judgment_str = str(trajectory_correctness_judgment) if trajectory_correctness_judgment != -1 else "-1"
        # 使用 trajectory_ground_truth（如果存在），否则使用 ground_truth
        trajectory_gt = trajectory_ground_truth if trajectory_ground_truth is not None else ground_truth
        trajectory_gt_str = str(trajectory_gt) if trajectory_gt is not None else "None"
        trajectory_correctness_judgment_filename = f"trajectory_correctness_judgment_{trajectory_gt_str}_to_{trajectory_correctness_judgment_str}.json"
        trajectory_correctness_judgment_path = os.path.join(save_dir, trajectory_correctness_judgment_filename)
        with open(trajectory_correctness_judgment_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_correctness_judgment_data, f, ensure_ascii=False, indent=2)
        
        # 保存接管判断结果（每个步骤一个文件）
        if takeover_judgment:
            for step_judgment in takeover_judgment:
                step_index = step_judgment.get("step_index", -1)
                if step_index < 0:
                    continue
                
                # 从 tmp2 读取 prompt 和 response（优化：使用已缓存的文件列表）
                takeover_file = next((f for f, basename in takeover_files.items() if f"_step_{step_index}_" in basename), None)
                if takeover_file is None:
                    takeover_file = os.path.join(task_tmp2_dir, f"{safe_task_id}_takeover_judgment_step_{step_index}_prompt_response.json")
                takeover_judgment_data = {
                    "step_index": step_index,
                    "ground_truth": step_judgment.get("ground_truth", 0),
                    "prediction": step_judgment.get("prediction", -1),
                    "handover": step_judgment.get("handover", "uncertain"),
                    "handover_categories": step_judgment.get("handover_categories", []),
                    "evidence": step_judgment.get("evidence", []),
                    "recommended_user_action": step_judgment.get("recommended_user_action", ""),
                    "next_step_if_no_handover": step_judgment.get("next_step_if_no_handover", ""),
                    "prompt_lines": [],
                    "response_lines": [],
                    "parsed_response": step_judgment.get("full_response", {})
                }
                
                if os.path.exists(takeover_file):
                    try:
                        with open(takeover_file, 'r', encoding='utf-8') as f:
                            takeover_data = json.load(f)
                        
                        prompt = takeover_data.get("prompt", "")
                        response_text = takeover_data.get("response_text", "")
                        
                        takeover_judgment_data["prompt_lines"] = prompt.split('\n') if prompt else []
                        takeover_judgment_data["response_lines"] = response_text.split('\n') if response_text else []
                    except Exception as e:
                        print(f"  警告: 读取接管判断文件失败 {takeover_file}: {e}")
                
                # 保存到 save 目录
                gt = takeover_judgment_data["ground_truth"]
                pr = takeover_judgment_data["prediction"]
                gt_str = str(gt) if gt is not None else "None"
                pr_str = str(pr) if pr != -1 else "-1"
                takeover_judgment_filename = f"takeover_judgment_step_{step_index}_{gt_str}_to_{pr_str}.json"
                takeover_judgment_path = os.path.join(save_dir, takeover_judgment_filename)
                with open(takeover_judgment_path, 'w', encoding='utf-8') as f:
                    json.dump(takeover_judgment_data, f, ensure_ascii=False, indent=2)
        
        # 保存原始JSONL文件中的任务数据
        if source_jsonl_path and os.path.exists(source_jsonl_path):
            try:
                original_task_data = load_task_from_jsonl(source_jsonl_path, task_id=task_id)
                if original_task_data:
                    original_task_filename = "original_task_data.json"
                    original_task_path = os.path.join(save_dir, original_task_filename)
                    with open(original_task_path, 'w', encoding='utf-8') as f:
                        json.dump(original_task_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  警告: 保存原始任务数据失败 {task_id}: {e}")
        
    except Exception as e:
        print(f"  警告: 保存任务数据失败 {task_id}: {e}")


def filter_last_action_if_needed(trajectory: List[Dict], keep_last_action: bool = False) -> List[Dict]:
    """
    默认情况下：如果最后一个step的action不含"接管"且不含"等待"，将其置为空。
    keep_last_action=True 时，不对最后一步 action 做任何清空处理。
    
    Args:
        trajectory: 操作序列
        keep_last_action: 是否保留最后一步 action（不做置空）
    
    Returns:
        List[Dict]: 处理后的操作序列
    """
    if not trajectory:
        return trajectory
    if keep_last_action:
        return trajectory
    
    # 获取最后一个step
    last_step = trajectory[-1]
    action = last_step.get("value", {}).get("action", "")
    
    # 如果不含"接管"且不含"等待"，将action置为空
    if action and "接管" not in action and "等待" not in action:
        if "value" not in last_step:
            last_step["value"] = {}
        last_step["value"]["action"] = ""
    
    return trajectory


def _parse_negative_task_flag(task_data: Dict) -> bool:
    """从 task_data.metadata.ext_info.level 解析是否为「负例」任务。

    判定路径：task_data["metadata"]["ext_info"]["level"] 若等于「负例」则返回 True；
    其它取值（如 "L1"/"L2"/"L3" 等正例 level）或缺失字段一律返回 False。
    """
    if not isinstance(task_data, dict):
        return False
    metadata = task_data.get("metadata")
    if not isinstance(metadata, dict):
        return False
    ext_info = metadata.get("ext_info")
    if not isinstance(ext_info, dict):
        return False
    level = ext_info.get("level")
    if not isinstance(level, str):
        return False
    return level.strip() == "负例"


def _parse_prm_key_step_info(task_data: Dict) -> Tuple[bool, List[int], List[int]]:
    """Parse metadata.list_key_index as 1-based PRM key-step indices."""
    metadata = task_data.get("metadata") if isinstance(task_data, dict) else None
    if not isinstance(metadata, dict):
        return False, [], []
    raw = metadata.get("list_key_index")
    if not isinstance(raw, list) or not raw:
        return False, [], []
    traj = task_data.get("traj")
    traj_len = len(traj) if isinstance(traj, list) else 0
    valid: List[int] = []
    invalid: List[int] = []
    for item in raw:
        try:
            idx = int(item)
        except Exception:
            invalid.append(item)
            continue
        if idx < 1 or (traj_len and idx > traj_len):
            invalid.append(idx)
            continue
        if idx not in valid:
            valid.append(idx)
    return bool(valid), valid, invalid


def _parse_ground_truth_from_task(task_data: Dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    从单条 task 的 evaluation 解析 ground_truth、is_task_completed、is_trajectory_correct。

    ground_truth 与 test_service 一致：优先读 evaluation.status，
    0=失败(gt=0)，非0=成功(gt=1)；无该字段时再回退到
    is_task_completed + is_trajectory_correct（二者均为 1 时 gt=1，否则 0）。
    is_task_completed / is_trajectory_correct 仍从 evaluation 独立字段读取，供非 lite 流程与保存逻辑使用。
    返回 (ground_truth, is_task_completed, is_trajectory_correct)。
    """
    evaluation = task_data.get("evaluation")
    if evaluation is not None and isinstance(evaluation, dict):
        is_task_completed = evaluation.get("is_task_completed", None)
        is_trajectory_correct = evaluation.get("is_trajectory_correct", None)
        s = evaluation.get("status", None)
        if s is not None:
            ground_truth = 1 if s != 0 else 0
        elif is_task_completed is not None and is_trajectory_correct is not None:
            ground_truth = 1 if (is_task_completed == 1 and is_trajectory_correct == 1) else 0
        else:
            ground_truth = None
        return (ground_truth, is_task_completed, is_trajectory_correct)
    return (None, None, None)


def _process_single_task_wrapper(
    task_data: Dict,
    index: int,
    total: int,
    use_cache: bool,
    lite_mode: bool,
    lite_action_list: bool = False,
    verbose: bool = True,
    print_lock: Optional[threading.Lock] = None,
    save_data: bool = False,
    source_jsonl_path: Optional[str] = None,
    model_name: Optional[str] = None,
    no_cache_strategy: Optional[str] = None,  # 新增参数
    compress_quality: Optional[float] = None,  # 压缩质量（0-1），如果为None则不压缩
    use_keyframe_images: bool = False,  # 是否传入关键帧图片
    use_qa_images: bool = False,  # 是否传入关键帧图片（用于QA判断）
    no_qa: bool = False,  # 是否跳过QA相关的prompt判定
    soft_cache_mode: bool = False,  # 新增：软缓存模式
    enable_timing: bool = False,  # 新增：是否启用时延统计
    skip_takeover: bool = True,  # 是否跳过步骤级接管判定
    eval_profile: Optional[str] = None,
    pipeline_json: Optional[Dict[str, Any]] = None,
    pipeline_preset: Optional[str] = None,
    error_cause_lite_gate_all: bool = False,  # 错因分析管道：True 时所有任务都先过 lite 门控
    error_cause_enable_trajectory: bool = False,  # 错因分析管道：部分 filter 时是否补充轨迹正确性
    use_task_knowledge: bool = True,
    judge_takeover: str = "end-to-end",
) -> Dict:
    """包装process_task调用，用于并行处理"""
    task_id = task_data.get("task_id", "")
    instruction = task_data.get("instruction", "")
    trajectory = task_data.get("traj", [])
    client_report = task_data.get("client_report") if isinstance(task_data.get("client_report"), dict) else None
    created_at = task_data.get("created_at") if isinstance(task_data.get("created_at"), str) else ""
    
    keep_last_action = bool((pipeline_json or {}).get("keep_last_action", False))
    # 过滤最后一个action（可由 pipeline_json.keep_last_action 控制是否跳过）
    trajectory = filter_last_action_if_needed(trajectory, keep_last_action=keep_last_action)
    annotate_takeover_sources(trajectory)
    task_data["traj"] = trajectory
    
    # 提取每个步骤的 need_to_takeover_gt 真值：从 traj[].take_over 读取，非空则为 1
    takeover_ground_truth_list = []
    for step in trajectory:
        take_over = step.get("take_over")
        need_to_takeover_gt = 1 if take_over else 0
        takeover_ground_truth_list.append(need_to_takeover_gt)
    
    # 从 task 的 evaluation 解析 ground_truth / is_task_completed / is_trajectory_correct
    ground_truth, is_task_completed, is_trajectory_correct = _parse_ground_truth_from_task(task_data)

    # 解析任务类型：metadata.ext_info.level == "负例" 时按负例分支评估
    is_negative_task = _parse_negative_task_flag(task_data)
    prm_mode, key_step_indices, invalid_key_step_indices = _parse_prm_key_step_info(task_data)

    # Naked evaluation: upstream evaluation.prediction.* is historical output and
    # must not affect this run's judgment. Downstream prior fields are populated
    # only from the current run's lite gate when the gate executes.
    prior_reasoning = ""
    prior_task_status = ""
    source_task_status = ""
    evaluation = task_data.get("evaluation") if isinstance(task_data.get("evaluation"), dict) else {}
    prediction = evaluation.get("prediction") if isinstance(evaluation.get("prediction"), dict) else {}
    raw_source_task_status = prediction.get("task_status")
    if isinstance(raw_source_task_status, str):
        source_task_status = raw_source_task_status.strip()

    if print_lock:
        with print_lock:
            print(f"处理任务 {index + 1}/{total}: {task_id}")
            if verbose:
                print(f"  指令：{instruction[:50]}...")
                if is_negative_task:
                    print("  [task_kind=negative] 负例任务：启用负例 prompt 分支，is_trajectory_correct/is_task_completed 将置 -2")
                if prm_mode:
                    print(f"  [prm_mode] 关键步骤: {key_step_indices}")
    else:
        print(f"处理任务 {index + 1}/{total}: {task_id}")
        if verbose:
            print(f"  指令：{instruction[:50]}...")
            if is_negative_task:
                print("  [task_kind=negative] 负例任务：启用负例 prompt 分支，is_trajectory_correct/is_task_completed 将置 -2")
            if prm_mode:
                print(f"  [prm_mode] 关键步骤: {key_step_indices}")
    
    try:
        result = process_task(
            instruction,
            trajectory,
            task_id=task_id,
            use_cache=use_cache,
            lite_mode=lite_mode,
            lite_action_list=lite_action_list,
            ground_truth=ground_truth,
            verbose=verbose,
            save_data=save_data,
            source_jsonl_path=source_jsonl_path,
            model_name=model_name,
            no_cache_strategy=no_cache_strategy,
            compress_quality=compress_quality,
            use_keyframe_images=use_keyframe_images,
            use_qa_images=use_qa_images,
            no_qa=no_qa,
            completion_ground_truth=is_task_completed,
            trajectory_ground_truth=is_trajectory_correct,
            soft_cache_mode=soft_cache_mode,
            enable_timing=enable_timing,
            takeover_ground_truth_list=takeover_ground_truth_list,
            skip_takeover=skip_takeover,
            eval_profile=eval_profile,
            pipeline_json=pipeline_json,
            pipeline_preset=pipeline_preset,
            is_negative_task=is_negative_task,
            prior_reasoning=prior_reasoning,
            prior_task_status=prior_task_status,
            error_cause_lite_gate_all=error_cause_lite_gate_all,
            error_cause_enable_trajectory=error_cause_enable_trajectory,
            source_task_status=source_task_status,
            client_report=client_report,
            created_at=created_at,
            prm_mode=prm_mode,
            key_step_indices=key_step_indices,
            invalid_key_step_indices=invalid_key_step_indices,
            use_task_knowledge=use_task_knowledge,
            judge_takeover=judge_takeover,
        )
        result["task_id"] = task_id
        result["ground_truth"] = ground_truth
        result["is_negative_task"] = is_negative_task
        result["prm_mode"] = prm_mode
        if prm_mode:
            result["key_step_indices"] = key_step_indices
            result["invalid_key_step_indices"] = invalid_key_step_indices
        result["_original_task_data"] = task_data
        result["lite_mode"] = result.get("lite_mode", lite_mode)
        # 如果存在新格式的字段，也保存到 result 中
        if is_task_completed is not None:
            result["is_task_completed"] = is_task_completed
        if is_trajectory_correct is not None:
            result["is_trajectory_correct"] = is_trajectory_correct
        
        if print_lock:
            with print_lock:
                print(f"任务 {index + 1}/{total} 完成: {task_id}")
        else:
            print(f"任务 {index + 1}/{total} 完成: {task_id}")
        
        return result
    except Exception as e:
        logger.error("_process_single_task_wrapper failed task_id=%s error=%s", task_id, e)
        logger.error(traceback.format_exc())
        error_result = {
            "task_id": task_id,
            "instruction": instruction,
            "trajectory_length": len(trajectory),
            "error": str(e),
            "ground_truth": ground_truth,
            "is_negative_task": is_negative_task,
            "_original_task_data": task_data
        }
        # 如果存在新格式的字段，也保存到 error_result 中
        if is_task_completed is not None:
            error_result["is_task_completed"] = is_task_completed
        if is_trajectory_correct is not None:
            error_result["is_trajectory_correct"] = is_trajectory_correct
        if print_lock:
            with print_lock:
                print(f"任务 {index + 1}/{total} 处理失败: {task_id}, 错误: {e}")
        else:
            print(f"任务 {index + 1}/{total} 处理失败: {task_id}, 错误: {e}")
        return error_result


def load_task_from_jsonl(file_path: str, task_id: Optional[str] = None, index: Optional[int] = None) -> Optional[Dict]:
    """
    从JSONL文件加载任务
    
    Args:
        file_path: JSONL文件路径
        task_id: 任务ID（可选）
        index: 任务索引（可选，从0开始）
        
    Returns:
        Dict: 任务数据，包含instruction和traj
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if index is not None and i != index:
                    continue
                
                task = json.loads(line.strip())
                
                if task_id is not None and task.get("task_id") != task_id:
                    continue
                
                # 过滤最后一个action（如果不含"接管"且不含"等待"，置为空）
                if "traj" in task:
                    task["traj"] = filter_last_action_if_needed(task["traj"])
                
                return task
        
        return None
    except Exception as e:
        print(f"加载任务失败：{e}")
        return None


def predict_one(
    task_data: Dict,
    use_cache: bool = True,
    lite_mode: bool = False,
    lite_action_list: bool = False,
    verbose: bool = False,
    save_data: bool = False,
    model_name: Optional[str] = None,
    no_cache_strategy: Optional[str] = None,
    compress_quality: Optional[float] = None,
    use_keyframe_images: bool = False,
    use_qa_images: bool = False,
    no_qa: bool = False,
    soft_cache_mode: bool = False,
    enable_timing: bool = False,
    skip_takeover: bool = True,
    pipeline_preset: Optional[str] = None,
    eval_profile: Optional[str] = None,
    pipeline_json: Optional[Dict[str, Any]] = None,
    knowledge: str = "knowledge_base",
    error_cause_lite_gate_all: bool = False,
    use_task_knowledge: bool = True,
    judge_takeover: str = "end-to-end",
) -> Dict:
    """
    单条预测：输入一个 task JSON，返回 task_id + orm_prediction（最小响应，便于客户端合并）。
    复用 _process_single_task_wrapper。
    """
    task_data = normalize_task_for_missing_tags(task_data)
    task_id = task_data.get("task_id", "")

    def _minimal_response(
        orm_prediction: dict,
        evaluation_extra: Optional[Dict[str, Any]] = None,
    ):
        """只返回 task_id + evaluation（orm_prediction + 可选字段如 reasoning），减小响应体。"""
        def _drop_minus2(obj: Any) -> Any:
            """
            语义与旧递归实现保持一致：
            - int(-2) -> None
            - dict: 值清洗后为 None 的键被删除
            - list: 元素清洗后即使为 None 也保留占位
            采用显式栈避免深层嵌套触发 Python 递归深度限制。
            """
            def _clean_scalar(v: Any) -> Any:
                if isinstance(v, int) and v == -2:
                    return None
                return v

            if not isinstance(obj, (dict, list)):
                return _clean_scalar(obj)

            def _new_frame(
                node: Any,
                parent: Optional[Dict[str, Any]],
                slot: Any,
                parent_is_dict: bool,
            ) -> Dict[str, Any]:
                if isinstance(node, dict):
                    return {
                        "node": node,
                        "kind": "dict",
                        "items": list(node.items()),
                        "idx": 0,
                        "out": {},
                        "parent": parent,
                        "slot": slot,
                        "parent_is_dict": parent_is_dict,
                    }
                return {
                    "node": node,
                    "kind": "list",
                    "items": list(node),
                    "idx": 0,
                    "out": [None] * len(node),
                    "parent": parent,
                    "slot": slot,
                    "parent_is_dict": parent_is_dict,
                }

            root_result: Any = None
            stack: List[Dict[str, Any]] = [_new_frame(obj, None, None, False)]
            active_container_ids = {id(obj)}

            while stack:
                frame = stack[-1]
                items = frame["items"]
                idx = frame["idx"]
                kind = frame["kind"]

                # 当前容器已处理完成：弹栈并回填到父容器
                if idx >= len(items):
                    stack.pop()
                    active_container_ids.discard(id(frame["node"]))
                    cleaned_container = frame["out"]
                    parent = frame["parent"]
                    if parent is None:
                        root_result = cleaned_container
                        continue
                    if frame["parent_is_dict"]:
                        if cleaned_container is not None:
                            parent["out"][frame["slot"]] = cleaned_container
                    else:
                        parent["out"][frame["slot"]] = cleaned_container
                    continue

                frame["idx"] = idx + 1
                if kind == "dict":
                    key, raw_value = items[idx]
                    slot = key
                    parent_is_dict = True
                else:
                    raw_value = items[idx]
                    slot = idx
                    parent_is_dict = False

                if isinstance(raw_value, (dict, list)):
                    child_id = id(raw_value)
                    if child_id in active_container_ids:
                        # 旧实现遇到循环引用会最终栈溢出；这里显式抛出同类错误。
                        raise RecursionError("maximum recursion depth exceeded")
                    child = _new_frame(raw_value, frame, slot, parent_is_dict)
                    stack.append(child)
                    active_container_ids.add(child_id)
                    continue

                cleaned_value = _clean_scalar(raw_value)
                if parent_is_dict:
                    if cleaned_value is not None:
                        frame["out"][slot] = cleaned_value
                else:
                    frame["out"][slot] = cleaned_value

            return root_result

        cleaned_orm = _drop_minus2(orm_prediction) if isinstance(orm_prediction, dict) else orm_prediction
        if not isinstance(cleaned_orm, dict):
            cleaned_orm = {}
        ev: Dict[str, Any] = {"orm_prediction": cleaned_orm}
        if evaluation_extra:
            cleaned_extra = _drop_minus2(evaluation_extra)
            if isinstance(cleaned_extra, dict):
                for k, v in cleaned_extra.items():
                    if v is not None:
                        ev[k] = v
        return {"task_id": task_id, "evaluation": ev}

    try:
        configure_knowledge_base(knowledge, verbose=False)
    except Exception as e:
        logger.error("predict_one invalid knowledge=%s error=%s", knowledge, e)
        return _minimal_response({
            "is_task_completed": -2,
            "is_trajectory_correct": -2,
            "is_task_success": -2,
            "error": str(e),
        })

    try:
        result = _process_single_task_wrapper(
            task_data,
            index=0,
            total=1,
            use_cache=use_cache,
            lite_mode=lite_mode,
            lite_action_list=lite_action_list,
            verbose=verbose,
            print_lock=None,
            save_data=save_data,
            source_jsonl_path=None,
            model_name=model_name,
            no_cache_strategy=no_cache_strategy,
            compress_quality=compress_quality,
            use_keyframe_images=use_keyframe_images,
            use_qa_images=use_qa_images,
            no_qa=no_qa,
            soft_cache_mode=soft_cache_mode,
            enable_timing=enable_timing,
            skip_takeover=skip_takeover,
            eval_profile=eval_profile,
            pipeline_json=pipeline_json,
            pipeline_preset=pipeline_preset,
            error_cause_lite_gate_all=error_cause_lite_gate_all,
            error_cause_enable_trajectory=True,
            use_task_knowledge=use_task_knowledge,
            judge_takeover=judge_takeover,
        )
    except Exception as e:
        logger.error("predict_one failed task_id=%s error=%s", task_id, e)
        logger.error(traceback.format_exc())
        return _minimal_response({
            "is_task_completed": -2,
            "is_trajectory_correct": -2,
            "is_task_success": -2,
            "error": str(e),
        })

    original = result.get("_original_task_data")
    if original is None:
        return _minimal_response({
            "is_task_completed": -2,
            "is_trajectory_correct": -2,
            "is_task_success": -2,
            "error": result.get("error", "missing _original_task_data"),
        })

    _cache_used = result.get("cache_used")
    if not isinstance(_cache_used, dict):
        _cache_used = None

    def _with_cache_used(extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if _cache_used is None and not extra:
            return None
        merged: Dict[str, Any] = {}
        if _cache_used is not None:
            merged["cache_used"] = _cache_used
        if isinstance(extra, dict):
            merged.update(extra)
        return merged

    judgment = result.get("judgment", {})
    pipeline_spec = result.get("pipeline_spec", {})
    final_mode = pipeline_spec.get("final_judgment_mode")
    if final_mode == "error_cause_analysis":
        prm_mode = bool(judgment.get("prm_mode") or result.get("prm_mode"))
        if result.get("error") and not judgment:
            return _minimal_response({"error": result["error"]})
        task_status = judgment.get("task_status")
        if prm_mode:
            task_status = _lite_task_status_from_judgment(judgment)
        elif not isinstance(task_status, str) or task_status not in FULL_STATUS_FILTER:
            task_status = _lite_task_status_from_judgment(judgment)
        prediction_extra: Dict[str, Any] = {"provider": PREDICTION_PROVIDER}
        if task_status:
            prediction_extra["task_status"] = task_status
        _apply_lite_env_fields_to_prediction(prediction_extra, judgment)
        prediction_extra["failure_reason"] = _error_cause_failure_reason(judgment)
        prediction_extra["reasoning"] = _error_cause_prediction_reasoning(judgment)
        trajectory_extra = _normalize_error_cause_trajectory_fields(judgment, task_status)

        # lite 门控判成功的任务：不参与错因分析，不回填错因字段
        if judgment.get("error_cause_skipped"):
            orm_prediction = {"error_cause_analysis": 0, "error_cause_skipped": 1}
            if "is_trajectory_correct" in trajectory_extra:
                orm_prediction["is_trajectory_correct"] = trajectory_extra["is_trajectory_correct"]
            if result.get("error"):
                orm_prediction["error"] = result["error"]
            eval_extra: Dict[str, Any] = {"prediction": prediction_extra}
            if "trajectory_reasoning" in trajectory_extra:
                eval_extra["reasoning"] = trajectory_extra["trajectory_reasoning"]
            if "first_deviation_step" in trajectory_extra:
                eval_extra["first_deviation_step"] = trajectory_extra["first_deviation_step"]
            if "first_deviation_reason" in trajectory_extra:
                eval_extra["first_deviation_reason"] = trajectory_extra["first_deviation_reason"]
            return _minimal_response(orm_prediction, evaluation_extra=_with_cache_used(eval_extra))
        orm_prediction = {"error_cause_analysis": 1}
        if prm_mode:
            orm_prediction["prm_mode"] = 1
            orm_prediction["key_step_indices"] = judgment.get("key_step_indices") or result.get("key_step_indices") or []
            invalid_indices = judgment.get("invalid_key_step_indices") or result.get("invalid_key_step_indices") or []
            if invalid_indices:
                orm_prediction["invalid_key_step_indices"] = invalid_indices
        if "is_trajectory_correct" in trajectory_extra:
            orm_prediction["is_trajectory_correct"] = trajectory_extra["is_trajectory_correct"]
        if result.get("error"):
            orm_prediction["error"] = result["error"]
        prediction_extra.update({
            "primary_error_cause": judgment.get("primary_error_cause", ""),
            "secondary_error_cause": judgment.get("secondary_error_cause", ""),
            "error_cause_reasoning": judgment.get("error_cause_reasoning", ""),
            "takeover_missing": judgment.get("takeover_missing", False),
            "takeover_overuse": judgment.get("takeover_overuse", False),
            "takeover_type_correct": judgment.get("takeover_type_correct", None),
            "expected_takeover_type": judgment.get("expected_takeover_type", ""),
            "takeover_diagnosis": judgment.get("takeover_diagnosis", ""),
        })
        eval_extra = {"prediction": prediction_extra}
        if "trajectory_reasoning" in trajectory_extra:
            eval_extra["reasoning"] = trajectory_extra["trajectory_reasoning"]
        if "first_deviation_step" in trajectory_extra:
            eval_extra["first_deviation_step"] = trajectory_extra["first_deviation_step"]
        if "first_deviation_reason" in trajectory_extra:
            eval_extra["first_deviation_reason"] = trajectory_extra["first_deviation_reason"]
        return _minimal_response(orm_prediction, evaluation_extra=_with_cache_used(eval_extra))
    if final_mode == "trajectory_correctness_all_images":
        pred_trajectory = judgment.get("trajectory_correctness_judgment", -1)
        # 仅轨迹正确性、无完成度，无法定义最终 is_task_success，统一填 -2
        orm_prediction = {
            "is_trajectory_correct": pred_trajectory,
            "is_task_success": -2,
        }
        if result.get("error"):
            orm_prediction["error"] = result["error"]
        _reas = judgment.get("reasoning", "")
        if not isinstance(_reas, str):
            _reas = "" if _reas is None else str(_reas)
        eval_extra = {"reasoning": _reas}
        _fds = judgment.get("first_deviation_step")
        if isinstance(_fds, int):
            eval_extra["first_deviation_step"] = _fds
        _fdr = judgment.get("first_deviation_reason")
        if isinstance(_fdr, str) and _fdr.strip():
            eval_extra["first_deviation_reason"] = _fdr.strip()
        return _minimal_response(orm_prediction, evaluation_extra=_with_cache_used(eval_extra))
    if final_mode == "lite_plus_trajectory_all_images_unified_binary":
        pred_success = judgment.get("final_judgment", 0)
        pred_trajectory = judgment.get("trajectory_correctness_judgment", 0)
        pred_completion = judgment.get("completion_judgment", 0)
        orm_prediction = {
            "is_task_success": pred_success,
            "is_trajectory_correct": pred_trajectory,
            "is_task_completed": pred_completion,
        }
        if result.get("error"):
            orm_prediction["error"] = result["error"]
        traj_reasoning = judgment.get("trajectory_reasoning", "")
        if not isinstance(traj_reasoning, str):
            traj_reasoning = "" if traj_reasoning is None else str(traj_reasoning)
        prediction_extra = _unified_prediction_from_judgment(judgment)
        eval_extra = {
            "prediction": prediction_extra,
            "reasoning": traj_reasoning,
            "unified_rich_judgment": judgment,
        }
        if result.get("is_negative_task"):
            eval_extra["task_kind"] = "negative"
            eval_extra["prediction"]["task_kind"] = "negative"
        _fds = judgment.get("first_deviation_step")
        if isinstance(_fds, int):
            eval_extra["first_deviation_step"] = _fds
        _fdr = judgment.get("first_deviation_reason")
        if isinstance(_fdr, str) and _fdr.strip():
            eval_extra["first_deviation_reason"] = _fdr.strip()
        return _minimal_response(orm_prediction, evaluation_extra=_with_cache_used(eval_extra))
    if final_mode == "lite_plus_trajectory_all_images":
        pred_success = judgment.get("final_judgment", -1)
        pred_trajectory = judgment.get("trajectory_correctness_judgment", -1)
        orm_prediction = {
            "is_task_success": pred_success,
            "is_trajectory_correct": pred_trajectory,
        }
        if result.get("error"):
            orm_prediction["error"] = result["error"]
        lite_reasoning = judgment.get("lite_reasoning", "")
        if not isinstance(lite_reasoning, str):
            lite_reasoning = "" if lite_reasoning is None else str(lite_reasoning)
        traj_reasoning = judgment.get("trajectory_reasoning", "")
        if not isinstance(traj_reasoning, str):
            traj_reasoning = "" if traj_reasoning is None else str(traj_reasoning)
        pred_task_status = _task_status_from_lite_final(
            pred_success,
            is_env_error=judgment.get("is_env_error"),
        )
        prediction_extra: Dict[str, Any] = {
            "provider": PREDICTION_PROVIDER,
            "reasoning": _format_orm_prediction_reasoning(
                lite_reasoning=lite_reasoning,
                knowledge_reasoning=summarize_match_for_reasoning(judgment),
                error_cause_reasoning="",
            ),
        }
        if pred_task_status is not None:
            prediction_extra["task_status"] = pred_task_status
        _apply_lite_env_fields_to_prediction(prediction_extra, judgment)
        eval_extra = {
            "prediction": prediction_extra,
            "reasoning": traj_reasoning,
        }
        _fds = judgment.get("first_deviation_step")
        if isinstance(_fds, int):
            eval_extra["first_deviation_step"] = _fds
        _fdr = judgment.get("first_deviation_reason")
        if isinstance(_fdr, str) and _fdr.strip():
            eval_extra["first_deviation_reason"] = _fdr.strip()
        return _minimal_response(orm_prediction, evaluation_extra=_with_cache_used(eval_extra))

    is_lite = result.get("lite_mode", False)
    if is_lite:
        pred_success = judgment.get("final_judgment", -1)
        # Lite 无轨迹正确性判断，is_trajectory_correct 填 -2（与 1/0/-1 不确定区分）
        orm_prediction = {
            "is_task_success": pred_success,
            "is_trajectory_correct": -2,
        }
        if result.get("error"):
            orm_prediction["error"] = result["error"]
        _reas = judgment.get("reasoning", "")
        if not isinstance(_reas, str):
            _reas = "" if _reas is None else str(_reas)
        pred_task_status = _task_status_from_lite_final(
            pred_success,
            is_env_error=judgment.get("is_env_error"),
        )
        lite_prediction: Dict[str, Any] = {
            "provider": PREDICTION_PROVIDER,
            "reasoning": _format_orm_prediction_reasoning(
                lite_reasoning=_reas,
                knowledge_reasoning=summarize_match_for_reasoning(judgment),
                error_cause_reasoning="",
            )
        }
        if pred_task_status is not None:
            lite_prediction["task_status"] = pred_task_status
        _apply_lite_env_fields_to_prediction(lite_prediction, judgment)
        return _minimal_response(orm_prediction, evaluation_extra=_with_cache_used({
            "prediction": lite_prediction,
        }))
    else:
        pred_completion = judgment.get("completion_judgment", -1)
        pred_trajectory = judgment.get("trajectory_correctness_judgment", -1)
        if pred_completion == 1 and pred_trajectory == 1:
            pred_success = 1
        elif pred_completion == -1 or pred_trajectory == -1:
            pred_success = -1
        else:
            pred_success = 0
        orm_prediction = {
            "is_task_completed": pred_completion,
            "is_trajectory_correct": pred_trajectory,
            "is_task_success": pred_success,
        }
        if result.get("error"):
            orm_prediction["error"] = result["error"]
        return _minimal_response(orm_prediction, evaluation_extra=_with_cache_used())


def process_jsonl_file(
    file_path: str, 
    output_path: str, 
    max_tasks: Optional[int] = None,
    use_cache: bool = True,
    lite_mode: bool = False,
    lite_action_list: bool = False,
    batch_size: int = 10,
    save_data: bool = False,
    model_name: Optional[str] = None,
    no_cache_strategy: Optional[str] = None,  # 新增参数
    compress_quality: Optional[float] = None,  # 压缩质量（0-1），如果为None则不压缩
    use_keyframe_images: bool = False,  # 是否传入关键帧图片
    use_qa_images: bool = False,  # 是否传入关键帧图片（用于QA判断）
    no_qa: bool = False,  # 是否跳过QA相关的prompt判定
    task_list: Optional[List[str]] = None,  # 任务ID列表，如果为None则处理所有任务
    soft_cache_mode: bool = False,  # 新增：软缓存模式
    enable_timing: bool = False,  # 新增：是否启用时延统计
    skip_takeover: bool = True,  # 是否跳过步骤级接管判定
    eval_profile: Optional[str] = None,
    pipeline_json: Optional[Dict[str, Any]] = None,
    status_filter: Optional[List[str]] = None,
    source_xlsx_path: Optional[str] = None,
    xlsx_sheet: Optional[str] = None,
    use_task_knowledge: bool = True,
    judge_takeover: str = "end-to-end",
):
    """
    批量处理JSONL文件中的任务（支持并行处理）
    
    Args:
        file_path: 输入JSONL文件路径
        output_path: 输出JSONL文件路径
        max_tasks: 最大处理任务数（None表示处理所有）
        use_cache: 是否使用缓存
        lite_mode: 是否使用简略模式
        batch_size: 并行处理的批次大小（默认10，1表示串行处理）
        save_data: 是否保存任务数据到data/{model_name}/save目录
        model_name: 模型名称（用于目录隔离）
        no_cache_strategy: 缓存策略 (None: 不检查, "qa": 检查QA错误, "key": 检查关键帧错误)
        task_list: 任务ID列表，如果为None则处理所有任务，否则只处理task_id在task_list中的任务
    """
    tasks = []
    judge_takeover = normalize_takeover_judge_mode(judge_takeover)
    
    # 如果提供了task_list，转换为set以便快速查找
    task_set = None
    if task_list is not None:
        task_set = set(task_list)
        print(f"任务过滤：只处理 {len(task_list)} 个指定任务")
    status_filter_set = set(status_filter or [])
    if status_filter_set:
        print(f"状态过滤：只处理 task_status in {sorted(status_filter_set)}")
    # 错因分析 lite 门控：无 filter 或显式全量 filter 时，所有任务都先过 lite ORM 判断。
    # 非全量 filter（如 failed,fair_enough）只对过滤后的样本做错因，不覆盖旧 status。
    is_full_status_filter = (not status_filter_set) or (status_filter_set == FULL_STATUS_FILTER)
    error_cause_lite_gate_all = is_full_status_filter
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_tasks is not None and len(tasks) >= max_tasks:
                    break
                
                task = normalize_task_for_missing_tags(json.loads(line.strip()))
                task_id = task.get("task_id", "")
                
                # 如果提供了task_list，只处理task_id在列表中的任务
                if task_set is not None:
                    if task_id not in task_set:
                        continue
                # 全量 filter（三种状态全选）等价于“全量评估”：包含没有先验
                # evaluation/task_status 的样本也要评估，因此不按先验状态预过滤。
                # 仅当是“部分子集”过滤时，才按先验 task_status 筛掉不匹配的样本。
                if status_filter_set and not is_full_status_filter:
                    ev = task.get("evaluation") if isinstance(task.get("evaluation"), dict) else {}
                    pred = ev.get("prediction") if isinstance(ev.get("prediction"), dict) else {}
                    if pred.get("task_status") not in status_filter_set:
                        continue
                
                tasks.append(task)
        
        total_tasks = len(tasks)
        if total_tasks == 0:
            if task_set is not None:
                print(f"没有找到需要处理的任务（在 {len(task_list)} 个指定任务中）")
            else:
                print("没有找到需要处理的任务")
            return
        
        if task_set is not None:
            print(f"共找到 {total_tasks} 个匹配的任务（从 {len(task_list)} 个指定任务中）")
        else:
            print(f"共 {total_tasks} 个任务需要处理")
        if batch_size > 1:
            print(f"使用并行处理，批次大小: {batch_size}")
        else:
            print("使用串行处理")
        
        # 如果启用时延统计，记录开始时间
        overall_start_time = None
        if enable_timing:
            overall_start_time = time.perf_counter()
            print("时延统计已启用")
        
        results = []
        verbose = (batch_size == 1)
        print_lock = threading.Lock() if batch_size > 1 else None
        
        if batch_size > 1:
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = []
                for i, task in enumerate(tasks):
                    future = executor.submit(
                        _process_single_task_wrapper,
                        task, i, total_tasks, use_cache, lite_mode, lite_action_list, verbose, print_lock, save_data, file_path, model_name, no_cache_strategy, compress_quality, use_keyframe_images, use_qa_images, no_qa, soft_cache_mode, enable_timing, skip_takeover, eval_profile, pipeline_json,
                        error_cause_lite_gate_all=error_cause_lite_gate_all,
                        error_cause_enable_trajectory=not is_full_status_filter,
                        use_task_knowledge=use_task_knowledge,
                        judge_takeover=judge_takeover,
                    )
                    futures.append((i, future))
                
                results_dict = {}
                for i, future in futures:
                    try:
                        result = future.result()
                        results_dict[i] = result
                    except Exception as e:
                        print(f"任务 {i + 1} 执行异常: {e}")
                        logger.error("process_jsonl_file future failed index=%s error=%s", i, e)
                        logger.error(traceback.format_exc())
                        gt, itc, itr = _parse_ground_truth_from_task(tasks[i])
                        error_result = {
                            "task_id": tasks[i].get("task_id", ""),
                            "error": str(e),
                            "ground_truth": gt
                        }
                        if itc is not None:
                            error_result["is_task_completed"] = itc
                        if itr is not None:
                            error_result["is_trajectory_correct"] = itr
                        results_dict[i] = error_result
                
                results = [results_dict[i] for i in sorted(results_dict.keys())]
        else:
            for i, task in enumerate(tasks):
                result = _process_single_task_wrapper(
                    task, i, total_tasks, use_cache, lite_mode, lite_action_list, verbose, print_lock, save_data, file_path, model_name, no_cache_strategy, compress_quality, use_keyframe_images, use_qa_images, no_qa, soft_cache_mode, enable_timing, skip_takeover, eval_profile, pipeline_json,
                    error_cause_lite_gate_all=error_cause_lite_gate_all,
                    error_cause_enable_trajectory=not is_full_status_filter,
                    use_task_knowledge=use_task_knowledge,
                    judge_takeover=judge_takeover,
                )
                results.append(result)
        
        # 写出：原始 task 数据 + 覆盖 evaluation.orm_prediction（不依赖 save 目录）
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                original = result.get("_original_task_data")
                if original is None:
                    continue
                output_task = copy.deepcopy(original)
                _clear_first_deviation_steps(output_task)
                judgment = result.get("judgment", {})
                pipeline_spec = result.get("pipeline_spec", {})
                final_mode = pipeline_spec.get("final_judgment_mode")
                is_lite = result.get("lite_mode", False)
                if "evaluation" not in output_task or not isinstance(output_task.get("evaluation"), dict):
                    output_task["evaluation"] = {}
                ev = output_task["evaluation"]
                merged_orm: Dict[str, Any] = {}
                base_orm = ev.get("orm_prediction")
                if isinstance(base_orm, dict):
                    merged_orm = copy.deepcopy(base_orm)
                if final_mode == "error_cause_analysis":
                    prm_mode = bool(judgment.get("prm_mode") or result.get("prm_mode"))
                    if result.get("error") and not judgment:
                        merged_orm = {"error": result["error"]}
                        ev["orm_prediction"] = merged_orm
                        f.write(json.dumps(output_task, ensure_ascii=False) + '\n')
                        continue
                    if "prediction" not in ev or not isinstance(ev["prediction"], dict):
                        ev["prediction"] = {}
                    pred = ev["prediction"]
                    pred["provider"] = PREDICTION_PROVIDER
                    for field in (
                        "primary_error_cause",
                        "secondary_error_cause",
                        "error_cause_reasoning",
                    ):
                        pred[field] = ""
                    merged_orm = {}
                    task_status = None
                    # lite 门控判成功的任务：非全量 filter 下保持原 status；全量 filter 下覆盖为本轮 lite ORM 状态。
                    if is_full_status_filter:
                        pred.pop("task_status", None)
                        gate_status = _lite_task_status_from_judgment(judgment) or ""
                        success_status = gate_status if gate_status in {"perfect", "fair_enough"} else "fair_enough"
                        task_status = _task_status_from_lite_final(
                            judgment.get("lite_final_judgment"),
                            success_status=success_status,
                            uncertain_status="failed",
                        )
                        if task_status is not None:
                            pred["task_status"] = task_status
                    effective_task_status = pred.get("task_status") if isinstance(pred.get("task_status"), str) else _lite_task_status_from_judgment(judgment)
                    trajectory_fields = _normalize_error_cause_trajectory_fields(judgment, effective_task_status)
                    if "is_trajectory_correct" in trajectory_fields:
                        merged_orm["is_trajectory_correct"] = trajectory_fields["is_trajectory_correct"]
                    if "trajectory_reasoning" in trajectory_fields:
                        ev["reasoning"] = trajectory_fields["trajectory_reasoning"]
                    if "first_deviation_step" in trajectory_fields:
                        ev["first_deviation_step"] = trajectory_fields["first_deviation_step"]
                    else:
                        ev.pop("first_deviation_step", None)
                    if "first_deviation_reason" in trajectory_fields:
                        ev["first_deviation_reason"] = trajectory_fields["first_deviation_reason"]
                    else:
                        ev.pop("first_deviation_reason", None)
                    _fds = trajectory_fields.get("first_deviation_step")
                    if isinstance(_fds, int):
                        _mark_first_deviation_step(output_task, _fds)
                    if judgment.get("error_cause_skipped"):
                        if is_full_status_filter:
                            pred["failure_reason"] = _error_cause_failure_reason(judgment)
                            pred["reasoning"] = _error_cause_prediction_reasoning(judgment)
                        merged_orm.update({"error_cause_analysis": 0, "error_cause_skipped": 1})
                        ev["orm_prediction"] = merged_orm
                        if result.get("error"):
                            merged_orm["error"] = result["error"]
                            ev["orm_prediction"] = merged_orm
                    else:
                        merged_orm["error_cause_analysis"] = 1
                        if prm_mode:
                            merged_orm["prm_mode"] = 1
                            merged_orm["key_step_indices"] = judgment.get("key_step_indices") or result.get("key_step_indices") or []
                            invalid_indices = judgment.get("invalid_key_step_indices") or result.get("invalid_key_step_indices") or []
                            if invalid_indices:
                                merged_orm["invalid_key_step_indices"] = invalid_indices
                        ev["orm_prediction"] = merged_orm
                        for field in (
                            "primary_error_cause",
                            "secondary_error_cause",
                            "error_cause_reasoning",
                            "expected_takeover_type",
                            "takeover_diagnosis",
                        ):
                            value = judgment.get(field, "")
                            pred[field] = (
                                value if isinstance(value, str) else ("" if value is None else str(value))
                            )
                        pred["takeover_missing"] = bool(judgment.get("takeover_missing", False))
                        pred["takeover_overuse"] = bool(judgment.get("takeover_overuse", False))
                        pred["takeover_type_correct"] = judgment.get("takeover_type_correct", None)
                        if is_full_status_filter:
                            if task_status is not None:
                                pred["task_status"] = task_status
                            pred["failure_reason"] = _error_cause_failure_reason(judgment)
                            pred["reasoning"] = _error_cause_prediction_reasoning(judgment)
                        if result.get("error"):
                            merged_orm["error"] = result["error"]
                            ev["orm_prediction"] = merged_orm
                elif final_mode == "trajectory_correctness_all_images":
                    pred_trajectory = judgment.get("trajectory_correctness_judgment", -1)
                    orm_update = {
                        "is_trajectory_correct": pred_trajectory,
                        "is_task_success": -2,
                    }
                    for k, v in orm_update.items():
                        if v == -2:
                            continue
                        merged_orm[k] = v
                    if result.get("error"):
                        merged_orm["error"] = result["error"]
                    ev["orm_prediction"] = merged_orm
                    _jr = judgment.get("reasoning", "")
                    ev["reasoning"] = (
                        _jr if isinstance(_jr, str) else ("" if _jr is None else str(_jr))
                    )
                    _fds = judgment.get("first_deviation_step")
                    if isinstance(_fds, int):
                        ev["first_deviation_step"] = _fds
                    _fdr = judgment.get("first_deviation_reason")
                    if isinstance(_fdr, str) and _fdr.strip():
                        ev["first_deviation_reason"] = _fdr.strip()
                    if isinstance(_fds, int):
                        traj = output_task.get("traj")
                        if isinstance(traj, list):
                            for step in traj:
                                if not isinstance(step, dict):
                                    continue
                                if step.get("index") == _fds:
                                    value = step.get("value")
                                    if not isinstance(value, dict):
                                        value = {}
                                        step["value"] = value
                                    value["first_deviation_step"] = True
                                    break
                elif final_mode == "lite_plus_trajectory_all_images_unified_binary":
                    pred_success = judgment.get("final_judgment", 0)
                    pred_trajectory = judgment.get("trajectory_correctness_judgment", 0)
                    pred_completion = judgment.get("completion_judgment", 0)
                    orm_update = {
                        "is_task_success": pred_success,
                        "is_trajectory_correct": pred_trajectory,
                        "is_task_completed": pred_completion,
                    }
                    for k, v in orm_update.items():
                        if v == -2:
                            continue
                        merged_orm[k] = v
                    if result.get("error"):
                        merged_orm["error"] = result["error"]
                    ev["orm_prediction"] = merged_orm

                    if "prediction" not in ev or not isinstance(ev["prediction"], dict):
                        ev["prediction"] = {}
                    ev["prediction"].update(_unified_prediction_from_judgment(judgment))

                    if result.get("is_negative_task"):
                        ev["task_kind"] = "negative"
                        ev["prediction"]["task_kind"] = "negative"

                    traj_reasoning = judgment.get("trajectory_reasoning", "")
                    ev["reasoning"] = (
                        traj_reasoning if isinstance(traj_reasoning, str) else ("" if traj_reasoning is None else str(traj_reasoning))
                    )
                    ev["unified_rich_judgment"] = judgment

                    _fds = judgment.get("first_deviation_step")
                    if isinstance(_fds, int):
                        ev["first_deviation_step"] = _fds
                    _fdr = judgment.get("first_deviation_reason")
                    if isinstance(_fdr, str) and _fdr.strip():
                        ev["first_deviation_reason"] = _fdr.strip()
                    if isinstance(_fds, int):
                        traj = output_task.get("traj")
                        if isinstance(traj, list):
                            for step in traj:
                                if not isinstance(step, dict):
                                    continue
                                if step.get("index") == _fds:
                                    value = step.get("value")
                                    if not isinstance(value, dict):
                                        value = {}
                                        step["value"] = value
                                    value["first_deviation_step"] = True
                                    break
                elif final_mode == "lite_plus_trajectory_all_images":
                    pred_success = judgment.get("final_judgment", -1)
                    pred_trajectory = judgment.get("trajectory_correctness_judgment", -1)
                    orm_update = {
                        "is_task_success": pred_success,
                        "is_trajectory_correct": pred_trajectory,
                    }
                    for k, v in orm_update.items():
                        if v == -2:
                            continue
                        merged_orm[k] = v
                    if result.get("error"):
                        merged_orm["error"] = result["error"]
                    ev["orm_prediction"] = merged_orm
                    lite_reasoning = judgment.get("lite_reasoning", "")
                    if not isinstance(lite_reasoning, str):
                        lite_reasoning = "" if lite_reasoning is None else str(lite_reasoning)
                    if "prediction" not in ev or not isinstance(ev["prediction"], dict):
                        ev["prediction"] = {}
                    ev["prediction"]["reasoning"] = _format_orm_prediction_reasoning(
                        lite_reasoning=lite_reasoning,
                        knowledge_reasoning=summarize_match_for_reasoning(judgment),
                        error_cause_reasoning="",
                    )
                    task_status = _task_status_from_lite_final(
                        pred_success,
                        is_env_error=judgment.get("is_env_error"),
                    )
                    if task_status is not None:
                        ev["prediction"]["task_status"] = task_status
                    _apply_lite_env_fields_to_prediction(ev["prediction"], judgment)

                    traj_reasoning = judgment.get("trajectory_reasoning", "")
                    ev["reasoning"] = (
                        traj_reasoning if isinstance(traj_reasoning, str) else ("" if traj_reasoning is None else str(traj_reasoning))
                    )
                    _fds = judgment.get("first_deviation_step")
                    if isinstance(_fds, int):
                        ev["first_deviation_step"] = _fds
                    _fdr = judgment.get("first_deviation_reason")
                    if isinstance(_fdr, str) and _fdr.strip():
                        ev["first_deviation_reason"] = _fdr.strip()
                    if isinstance(_fds, int):
                        traj = output_task.get("traj")
                        if isinstance(traj, list):
                            for step in traj:
                                if not isinstance(step, dict):
                                    continue
                                if step.get("index") == _fds:
                                    value = step.get("value")
                                    if not isinstance(value, dict):
                                        value = {}
                                        step["value"] = value
                                    value["first_deviation_step"] = True
                                    break
                elif is_lite:
                    pred_success = judgment.get("final_judgment", -1)
                    orm_update = {
                        "is_task_success": pred_success,
                        "is_trajectory_correct": -2,
                    }
                    for k, v in orm_update.items():
                        if v == -2:
                            continue
                        merged_orm[k] = v
                    if result.get("error"):
                        merged_orm["error"] = result["error"]
                    ev["orm_prediction"] = merged_orm
                    # lite 管道仅回填 prediction 子字段，避免与轨迹管道的 evaluation.reasoning 相互覆盖
                    pred_reasoning = judgment.get("reasoning", "")
                    if not isinstance(pred_reasoning, str):
                        pred_reasoning = "" if pred_reasoning is None else str(pred_reasoning)
                    if "prediction" not in ev or not isinstance(ev["prediction"], dict):
                        ev["prediction"] = {}
                    ev["prediction"]["reasoning"] = _format_orm_prediction_reasoning(
                        lite_reasoning=pred_reasoning,
                        knowledge_reasoning=summarize_match_for_reasoning(judgment),
                        error_cause_reasoning="",
                    )
                    task_status = _task_status_from_lite_final(
                        pred_success,
                        is_env_error=judgment.get("is_env_error"),
                    )
                    if task_status is not None:
                        ev["prediction"]["task_status"] = task_status
                    _apply_lite_env_fields_to_prediction(ev["prediction"], judgment)
                else:
                    pred_completion = judgment.get("completion_judgment", -1)
                    pred_trajectory = judgment.get("trajectory_correctness_judgment", -1)
                    if pred_completion == 1 and pred_trajectory == 1:
                        pred_success = 1
                    elif pred_completion == -1 or pred_trajectory == -1:
                        pred_success = -1
                    else:
                        pred_success = 0
                    orm_update = {
                        "is_task_completed": pred_completion,
                        "is_trajectory_correct": pred_trajectory,
                        "is_task_success": pred_success,
                    }
                    for k, v in orm_update.items():
                        if v == -2:
                            continue
                        merged_orm[k] = v
                    if result.get("error"):
                        merged_orm["error"] = result["error"]
                    ev["orm_prediction"] = merged_orm

                is_lite_pipeline = is_lite or final_mode in {
                    "trajectory_correctness_all_images",
                    "lite_plus_trajectory_all_images_unified_binary",
                    "lite_plus_trajectory_all_images",
                }
                if is_lite_pipeline and isinstance(ev.get("orm_prediction"), dict):
                    post_processed = apply_lite_task_success_postprocess(
                        task_data=output_task,
                        orm_prediction=ev["orm_prediction"],
                    )
                    if post_processed and isinstance(ev.get("prediction"), dict):
                        post_success = ev["orm_prediction"].get("is_task_success")
                        rich_status = None
                        rich = ev.get("unified_rich_judgment")
                        if isinstance(rich, dict):
                            rs = rich.get("task_status")
                            if isinstance(rs, str) and rs in FULL_STATUS_FILTER:
                                rich_status = rs
                        if final_mode == "lite_plus_trajectory_all_images_unified_binary":
                            if rich_status:
                                ev["prediction"]["task_status"] = rich_status
                            elif post_success == 1:
                                ev["prediction"]["task_status"] = "perfect"
                            elif post_success == -1:
                                ev["prediction"]["task_status"] = "failed"
                            else:
                                ev["prediction"]["task_status"] = "failed"
                        elif final_mode == "lite_plus_trajectory_all_images" or is_lite:
                            if post_success == 1:
                                ev["prediction"]["task_status"] = "perfect"
                            elif post_success == 0:
                                ev["prediction"]["task_status"] = "failed"
                            elif post_success == -1:
                                ev["prediction"]["task_status"] = "uncertain"
                if isinstance(ev.get("orm_prediction"), dict):
                    _attach_task_knowledge_fields_to_orm_prediction(ev["orm_prediction"], judgment)
                if isinstance(ev.get("prediction"), dict):
                    ev["prediction"]["provider"] = PREDICTION_PROVIDER
                f.write(json.dumps(output_task, ensure_ascii=False) + '\n')
        if source_xlsx_path:
            try:
                output_xlsx_path = backfill_prediction_fields_to_xlsx(
                    source_xlsx_path,
                    output_path,
                    include_status=is_full_status_filter,
                    sheet_name=xlsx_sheet,
                )
                if is_full_status_filter:
                    excluded_task_ids: List[str] = []
                    filter_eval_path = Path(file_path).with_name(f"{Path(file_path).stem}_filter_eval.jsonl")
                    if filter_eval_path.exists():
                        try:
                            excluded_task_ids = sorted(load_task_ids_from_jsonl(str(filter_eval_path)))
                            print(
                                f"统计时排除 filter_eval 样本: {len(excluded_task_ids)} "
                                f"（{filter_eval_path}）"
                            )
                        except Exception as load_err:
                            print(f"警告: 读取 filter_eval 失败，统计将不排除过滤样本: {load_err}")
                    summarize_xlsx_human_vs_zj_status(
                        output_xlsx_path,
                        sheet_name=xlsx_sheet,
                        excluded_task_ids=excluded_task_ids,
                    )
            except Exception as e:
                print(f"警告: 表格回填失败 {source_xlsx_path}: {e}")
        
        # 基于内存结果统计准确率（不读 save 目录）
        summary = _compute_judgment_stats_from_results(results)
        task_knowledge_stats = _compute_task_knowledge_stats_from_results(results)
        
        # 如果启用时延统计，输出统计信息
        if enable_timing and results:
            overall_end_time = time.perf_counter()
            overall_total_time = overall_end_time - overall_start_time
            
            # 从结果中提取时延信息（取第一个结果，因为max_tasks=1时只有一个结果）
            timing_data = results[0].get("timing", {})
            
            print(f"\n处理完成！共处理 {total_tasks} 个任务")
            print(f"结果已保存到：\n{output_path}")
            print("\n" + "="*60)
            print("时延统计（max-tasks=1）:")
            print("="*60)
            print(f"总耗时: {overall_total_time:.2f} 秒")
            print()
            print("各步骤耗时:")
            
            step_names = {
                "step1_category": "步骤1 - 任务分类",
                "step2_slots_qa": "步骤2 - 槽位提取",
                "step3_action_thought_caption": "步骤3 - Action/Thought/Caption生成",
                "step4_keyframes": "步骤4 - 关键帧提取",
                "step5_judgment": "步骤5 - 任务完成判断",
                "save_data": "保存数据"
            }
            
            total_steps_time = 0.0
            for key, name in step_names.items():
                step_time = timing_data.get(key, 0.0)
                total_steps_time += step_time
                if key == "step2_slots_qa" and step_time == 0.0:
                    print(f"  {name}: {step_time:.2f} 秒 [已跳过]")
                elif key == "save_data" and step_time == 0.0:
                    print(f"  {name}: {step_time:.2f} 秒 [未启用]")
                else:
                    print(f"  {name}: {step_time:.2f} 秒")
            
            print()
            print("各步骤占比:")
            for key, name in step_names.items():
                step_time = timing_data.get(key, 0.0)
                if total_steps_time > 0:
                    percentage = (step_time / total_steps_time) * 100
                    print(f"  {name}: {percentage:.2f}%")
                else:
                    print(f"  {name}: 0.00%")
            
            print("="*60)
        else:
            print(f"\n处理完成！共处理 {total_tasks} 个任务")
            print(f"结果已保存到：{output_path}")

        _print_task_knowledge_stats(task_knowledge_stats)

        print(f"\n准确率统计（基于本次运行结果）：")
        print(f"  总任务数: {summary['total']}")
        print(f"  有效判断数: {summary['valid_count']}")
        print(f"  正确数: {summary['correct_count']}")
        print(f"  综合准确率: {summary['accuracy']:.2f}%")
        print()
        if lite_mode:
            # lite 模式只统计综合任务成功 (is_task_success)，不统计完成度与轨迹正确性
            print("按类型统计（lite 模式仅综合任务成功）:")
            print(f"  综合任务成功 (is_task_success): 有效 {summary['by_type']['final']['valid']}, 正确 {summary['by_type']['final']['correct']}, 准确率 {summary['by_type']['final']['accuracy']:.2f}%")
        else:
            print("按类型统计:")
            print(f"  完成度 (is_task_completed): 有效 {summary['by_type']['completion']['valid']}, 正确 {summary['by_type']['completion']['correct']}, 准确率 {summary['by_type']['completion']['accuracy']:.2f}%")
            print(f"  轨迹正确性 (is_trajectory_correct): 有效 {summary['by_type']['trajectory_correctness']['valid']}, 正确 {summary['by_type']['trajectory_correctness']['correct']}, 准确率 {summary['by_type']['trajectory_correctness']['accuracy']:.2f}%")
            print(f"  综合任务成功 (is_task_success): 有效 {summary['by_type']['final']['valid']}, 正确 {summary['by_type']['final']['correct']}, 准确率 {summary['by_type']['final']['accuracy']:.2f}%")

        # 输出与 test_service.py 保持一致的最终判定混淆指标（基于 evaluation.status vs is_task_success）
        cf = summary.get("confusion_final", {})
        if cf:
            print("\n最终判定混淆指标（与 test_service.py 口径一致）:")
            gt1_total = cf.get("gt1_total", 0)
            gt0_total = cf.get("gt0_total", 0)
            gt1_correct = cf.get("gt1_correct", 0)
            gt0_correct = cf.get("gt0_correct", 0)
            pred1_total = cf.get("pred1_total", 0)
            pred0_total = cf.get("pred0_total", 0)
            print(f"  overall_accuracy:   {gt1_correct + gt0_correct}/{gt1_total + gt0_total} = {cf.get('overall_accuracy', 0.0):.2f}%")
            if gt1_total > 0:
                print(f"  success_recall:     {gt1_correct}/{gt1_total} = {cf.get('success_recall', 0.0):.2f}%")
            else:
                print("  success_recall:     无正样本，未统计")
            if pred1_total > 0:
                print(f"  success_precision:  {gt1_correct}/{pred1_total} = {cf.get('success_precision', 0.0):.2f}%")
            else:
                print("  success_precision:  无预测为success，未统计")
            if gt0_total > 0:
                print(f"  failed_recall:      {gt0_correct}/{gt0_total} = {cf.get('failed_recall', 0.0):.2f}%")
            else:
                print("  failed_recall:      无负样本，未统计")
            if pred0_total > 0:
                print(f"  failed_precision:   {gt0_correct}/{pred0_total} = {cf.get('failed_precision', 0.0):.2f}%")
            else:
                print("  failed_precision:   无预测为failure，未统计")
        
    except Exception as e:
        print(f"批量处理失败：{e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 调试模式：如果没有提供参数，使用默认值
    if len(sys.argv) < 2:
        default_input = "data/query_20260212_orm_testset.jsonl"
        default_input = "data/batch_export_2026031618351974.jsonl"
        default_ouput = default_input.replace("query","output")
        sys.argv = [sys.argv[0], default_input,"--batch-size", "5", "--save", "--model", "gpt-4o", "--no-cache", "qa"]
        sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "gpt-4o"]
        sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "gpt-4o",  "--use-keyframe-images"]
        sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "doubao-1.8", "--no-qa"]        
        sys.argv = [sys.argv[0], default_input,"--batch-size", "1", "--save", "--model", "gpt-4o", "--no-qa", "--pro"]
        sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "doubao-seed-2.0-pro", "--no-qa", "--max-tasks", "200", "--compress-quality", "0.8"]
        sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "doubao-seed-1.8", "--no-qa", "--max-tasks", "200", "--compress-quality", "0.8"]
        # sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "doubao-1.8", "--no-qa", "--max-tasks", "200", "--compress-quality", "0.8"]
        # sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "doubao-seed-1.6-vision", "--no-qa", "--max-tasks", "200", "--compress-quality", "0.8"]
        sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "gemini-3-pro", "--no-qa", "--max-tasks", "200", "--compress-quality", "0.8"]
        sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "gpt-5.2", "--no-qa", "--max-tasks", "200", "--compress-quality", "0.8"]
        sys.argv = [sys.argv[0], default_input,"--batch-size", "1", "--model", "doubao-seed-1.8", "--no-qa", "--lite", "--action-list"]
        # sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "qwen3.5-plus", "--no-qa", "--max-tasks", "200", "--compress-quality", "0.8"]
        # sys.argv = [sys.argv[0], default_input,"--batch-size", "10", "--save", "--model", "qwen3-vl-plus", "--no-qa", "--max-tasks", "200", "--compress-quality", "0.8"]

        

    # 解析并切换知识库（默认 knowledge_base）
    knowledge_base = "knowledge_base"
    if "--knowledge" in sys.argv:
        idx = sys.argv.index("--knowledge")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            knowledge_base = sys.argv[idx + 1].strip()
        else:
            print(
                "错误：--knowledge 需要传入目录名，可选: "
                + ", ".join(SUPPORTED_KNOWLEDGE_BASES)
            )
            return
    try:
        configure_knowledge_base(knowledge_base, verbose=True)
    except Exception as e:
        print(f"错误：知识库配置失败: {e}")
        return

    # 检查是否是测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 测试模式：使用真实数据文件中的第一行数据
        print("运行测试模式...")
        default_test_file = "data/query_2025122916413136_Auto_mid_data.jsonl"
        
        # 从文件中加载第一行真实数据
        test_task = load_task_from_jsonl(default_test_file, index=0)
        if not test_task:
            print(f"错误：无法从 {default_test_file} 加载测试数据")
            return
        
        instruction = test_task.get("instruction", "")
        trajectory = test_task.get("traj", [])
        
        # 过滤最后一个action（如果不含"接管"且不含"等待"，置为空）
        trajectory = filter_last_action_if_needed(trajectory)
        
        task_id = test_task.get("task_id", "")
        
        print(f"测试任务ID: {task_id}")
        print(f"测试指令: {instruction[:50]}...")
        print(f"轨迹长度: {len(trajectory)}")
        print()
        
        result = process_task(instruction, trajectory, task_id=task_id, use_cache=True, lite_mode=False)
        print("\n" + "="*60)
        print("测试结果：")
        print("="*60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    
    input_arg = None
    if "--input" in sys.argv:
        idx = sys.argv.index("--input")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            input_arg = sys.argv[idx + 1]
        else:
            print("错误：--input 需要传入 JSONL 或 XLSX 文件路径")
            return
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        input_arg = sys.argv[1]

    if not input_arg:
        print("错误：缺少输入文件，请使用 --input <jsonl/xlsx>（兼容旧位置参数）")
        return

    # 检查输入文件是否存在
    if not os.path.exists(input_arg):
        print(f"错误：输入文件不存在: {input_arg}")
        print(f"请检查文件路径是否正确")
        return

    xlsx_sheet = None
    if "--xlsx-sheet" in sys.argv:
        idx = sys.argv.index("--xlsx-sheet")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            xlsx_sheet = sys.argv[idx + 1].strip()
        else:
            print("错误：--xlsx-sheet 需要传入工作表名")
            return

    source_xlsx_path = input_arg if is_xlsx_input(input_arg) else None
    try:
        input_file = resolve_input_to_jsonl(input_arg, xlsx_sheet=xlsx_sheet)
    except Exception as e:
        print(f"错误：解析输入失败: {e}")
        return
    

    # 解析模型名称（需要在生成输出路径之前）
    model_name = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_name = sys.argv[idx + 1]
    
    if model_name is None:
        import config
        model_name = config.DEFAULT_MODEL
    
    # 基于输入文件路径，将文件名中的"query"替换为"output"
    # 输出文件路径改为使用模型相关的目录
    import config
    model_data_dir = config.get_model_data_dir(model_name)
    base_name = os.path.basename(input_file)
    # 替换文件名中的query为output
    new_base_name = base_name.replace("query", "output", 1)  # 只替换第一个出现的query
    if new_base_name == base_name:  # 如果没有query，则在文件名前加output_
        name, ext = os.path.splitext(base_name)
        new_base_name = f"output_{name}{ext}"
    output_file = os.path.join(model_data_dir, new_base_name)
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print()
    
    # 解析参数
    # 处理 --no-cache 参数
    no_cache_strategy = None
    use_cache = True
    soft_cache_mode = False  # 新增：软缓存模式，只跳过 judgment 缓存
    
    if "--no-cache" in sys.argv:
        idx = sys.argv.index("--no-cache")
        if idx + 1 < len(sys.argv):
            cache_value = sys.argv[idx + 1].strip()
            # 检查下一个参数是否是另一个选项（以--开头），如果是则视为空值
            if cache_value.startswith("--"):
                # 下一个参数是另一个选项，视为 --no-cache 没有值
                use_cache = False
                no_cache_strategy = None
            elif cache_value == "qa":
                no_cache_strategy = "qa"
                use_cache = True  # 仍然使用缓存，只是检查QA错误
            elif cache_value == "key":
                no_cache_strategy = "key"
                use_cache = True  # 仍然使用缓存，只是检查关键帧错误
            elif cache_value == "soft":
                # 新增：软缓存模式，只跳过 judgment 缓存
                soft_cache_mode = True
                use_cache = True  # 其他步骤仍然使用缓存
                no_cache_strategy = None
            else:
                # 值为空或其他值，不使用任何缓存，全重新生成
                use_cache = False
                no_cache_strategy = None
        else:
            # --no-cache 后面没有值，相当于不使用任何缓存
            use_cache = False
            no_cache_strategy = None
    
    lite_mode = "--lite" in sys.argv
    lite_action_list = "--action-list" in sys.argv if lite_mode else False

    # 解析消融/流水线配置参数
    eval_profile = None
    if "--eval-profile" in sys.argv:
        idx = sys.argv.index("--eval-profile")
        if idx + 1 < len(sys.argv):
            eval_profile = sys.argv[idx + 1].strip()

    pipeline_json = None
    if "--pipeline-json" in sys.argv:
        idx = sys.argv.index("--pipeline-json")
        if idx + 1 < len(sys.argv):
            pipeline_json_path = sys.argv[idx + 1]
            try:
                with open(pipeline_json_path, "r", encoding="utf-8") as f:
                    pipeline_json = json.load(f)
            except Exception as e:
                print(f"警告: 读取 --pipeline-json 失败 {pipeline_json_path}: {e}")
                pipeline_json = None

    status_filter = ["failed", "fair_enough"]
    if "--filter" in sys.argv:
        idx = sys.argv.index("--filter")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            raw_filter = sys.argv[idx + 1].strip()
            status_filter = [s.strip() for s in raw_filter.split(",") if s.strip()]
        else:
            status_filter = []

    max_tasks = None
    if "--max-tasks" in sys.argv:
        idx = sys.argv.index("--max-tasks")
        if idx + 1 < len(sys.argv):
            max_tasks = int(sys.argv[idx + 1])
    
    batch_size = 10
    if "--batch-size" in sys.argv:
        idx = sys.argv.index("--batch-size")
        if idx + 1 < len(sys.argv):
            batch_size = int(sys.argv[idx + 1])
            if batch_size < 1:
                batch_size = 1
    
    save_data = "--save" in sys.argv
    
    # 解析 --compress-quality 参数
    compress_quality = None
    if "--compress-quality" in sys.argv:
        idx = sys.argv.index("--compress-quality")
        if idx + 1 < len(sys.argv):
            try:
                compress_quality = float(sys.argv[idx + 1])
                if compress_quality < 0 or compress_quality > 1:
                    print("警告: 压缩质量应在 0-1 之间，使用默认值 0.6")
                    compress_quality = 0.6
            except ValueError:
                print("警告: 压缩质量参数无效，使用默认值 0.6")
                compress_quality = 0.6
    
    # 解析 --use-keyframe-images 参数
    use_keyframe_images = "--use-keyframe-images" in sys.argv
    
    # 解析 --use-qa-images 参数
    use_qa_images = "--use-qa-images" in sys.argv
    
    # 解析 --no-qa 参数（先解析，因为 --pro 可能会覆盖它）
    no_qa = "--no-qa" in sys.argv
    
    # 解析 --pro 参数（需要在解析 --task-list 之前）
    use_pro = "--pro" in sys.argv
    bdcs_task_ids = []
    
    if use_pro:
        bdcs_task_ids = extract_task_ids_from_bdcs(model_name)
        if bdcs_task_ids:
            task_list = bdcs_task_ids
            # 覆盖 --no-cache 设置，强制使用 soft 模式
            soft_cache_mode = True  # 相当于 --no-cache soft
            use_cache = True  # 确保其他步骤可以使用缓存
            no_cache_strategy = None  # 清除之前的策略
            no_qa = True  # 相当于 --no-qa
            print(f"使用 --pro 模式：从 bdcs 目录提取了 {len(bdcs_task_ids)} 个 task_id")
            print(f"  已自动设置：--no-cache soft, --no-qa")
        else:
            print(f"警告: --pro 模式未找到任何 task_id，继续使用原有参数")
    
    # 解析 --task-list 参数（如果 --pro 没有覆盖它）
    if not use_pro or not bdcs_task_ids:
        task_list = None
        if "--task-list" in sys.argv:
            idx = sys.argv.index("--task-list")
            if idx + 1 < len(sys.argv):
                task_list_value = sys.argv[idx + 1]
                # 如果是一个JSON文件路径，读取文件
                if task_list_value.endswith('.json'):
                    try:
                        with open(task_list_value, 'r', encoding='utf-8') as f:
                            task_list = json.load(f)
                        if not isinstance(task_list, list):
                            print(f"警告: task_list文件应包含一个JSON数组，使用默认值（处理所有任务）")
                            task_list = None
                    except Exception as e:
                        print(f"警告: 读取task_list文件失败 {task_list_value}: {e}，使用默认值（处理所有任务）")
                        task_list = None
                else:
                    # 尝试解析为JSON数组字符串
                    # 先去除可能的引号
                    task_list_value_clean = task_list_value.strip()
                    if (task_list_value_clean.startswith('"') and task_list_value_clean.endswith('"')) or \
                       (task_list_value_clean.startswith("'") and task_list_value_clean.endswith("'")):
                        task_list_value_clean = task_list_value_clean[1:-1]
                    
                    try:
                        task_list = json.loads(task_list_value_clean)
                        if not isinstance(task_list, list):
                            print(f"警告: task_list应是一个JSON数组，使用默认值（处理所有任务）")
                            task_list = None
                    except:
                        # 如果不是JSON格式，尝试按逗号分割
                        task_list = [task_id.strip() for task_id in task_list_value_clean.split(',') if task_id.strip()]
                        if not task_list:
                            print(f"警告: 无法解析task_list参数，使用默认值（处理所有任务）")
                            task_list = None
    
    # 解析 --takeover 参数（默认跳过步骤级接管判定，加 --takeover 则启用）
    skip_takeover = "--takeover" not in sys.argv

    judge_takeover = "end-to-end"
    if "--judge-takeover" in sys.argv:
        idx = sys.argv.index("--judge-takeover")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):
            judge_takeover = normalize_takeover_judge_mode(sys.argv[idx + 1])
        else:
            print("警告: --judge-takeover 需要 end-to-end 或 only-lmd，使用默认 end-to-end")
    else:
        judge_takeover = normalize_takeover_judge_mode(judge_takeover)

    # 解析 --no-task-knowledge：不检索/不注入外挂任务知识
    use_task_knowledge = "--no-task-knowledge" not in sys.argv
    
    # model_name 已经在上面解析过了
    print(f"使用模型: {model_name}")
    if soft_cache_mode:
        print("缓存策略: --no-cache soft (只重新生成完成度和轨迹正确性判断)")
    elif no_cache_strategy is not None:
        print(f"缓存策略: --no-cache {no_cache_strategy}")
    elif not use_cache:
        print("缓存策略: --no-cache (全重新生成)")
    if compress_quality is not None:
        print(f"图像压缩质量: {compress_quality}")
    if use_keyframe_images:
        print("关键帧判断: 使用图片模式（传入关键帧图片，不传Caption文本）")
    if use_qa_images:
        print("QA判断: 使用图片模式（传入关键帧图片，不传Caption文本）")
    if no_qa:
        print("QA判断: 已跳过（--no-qa 模式）")
    if status_filter:
        print(f"状态过滤: {status_filter}")
    else:
        print("状态过滤: 关闭")
    if skip_takeover:
        print("接管判定: 已跳过（默认，使用 --takeover 启用）")
    else:
        print("接管判定: 已启用（--takeover）")
    print(f"接管知识判定模式: {judge_takeover}（--judge-takeover，可选 end-to-end/only-lmd）")
    if use_task_knowledge:
        print("任务知识: 已启用（默认）")
    else:
        print("任务知识: 已跳过（--no-task-knowledge）")
    if task_list is not None:
        print(f"任务过滤: 只处理 {len(task_list)} 个指定任务")
    
    # 当时延统计启用时，设置 enable_timing
    enable_timing = (max_tasks == 1)
    
    process_jsonl_file(
        input_file,
        output_file,
        max_tasks,
        use_cache=use_cache,
        lite_mode=lite_mode,
        lite_action_list=lite_action_list,
        batch_size=batch_size,
        save_data=save_data,
        model_name=model_name,
        no_cache_strategy=no_cache_strategy,
        compress_quality=compress_quality,
        use_keyframe_images=use_keyframe_images,
        use_qa_images=use_qa_images,
        no_qa=no_qa,
        task_list=task_list,
        soft_cache_mode=soft_cache_mode,
        enable_timing=enable_timing,
        skip_takeover=skip_takeover,
        eval_profile=eval_profile,
        pipeline_json=pipeline_json,
        status_filter=status_filter,
        source_xlsx_path=source_xlsx_path,
        xlsx_sheet=xlsx_sheet,
        use_task_knowledge=use_task_knowledge,
        judge_takeover=judge_takeover,
    )


if __name__ == "__main__":
    main()
