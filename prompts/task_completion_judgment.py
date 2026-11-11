"""
任务完成判断模块
判断任务是否完成
包含两个Prompt：基于QA对判断和基于关键帧信息判断
"""
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import csv
import glob
import re
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_utils import call_llm_with_prompt, parse_json_response, get_task_context, set_task_context, get_model_name, load_json_file
from prompts.judgment_config import STATUS_MAPPING, DEFAULT_STATUS_OPTIONS
from prompts.lite_judgment_prompt import get_lite_judgment_prompt
import config

_completion_strategy_cache = None
_trajectory_strategy_cache = None
_improvements_cache = {}


def _parse_status_to_int(raw_status) -> int:
    """将 LLM 返回的 status 规范化为 1/0/-1。"""
    if isinstance(raw_status, str):
        raw_status = STATUS_MAPPING.get(raw_status, -1)
    if raw_status not in [1, 0, -1]:
        return -1
    return raw_status


def load_completion_strategy() -> Dict:
    """
    加载完成度判断公共策略
    
    Returns:
        Dict: 完成度判断策略配置
    """
    global _completion_strategy_cache
    if _completion_strategy_cache is not None:
        return _completion_strategy_cache
    strategy_path = os.path.join(config.KNOWLEDGE_BASE_DIR, "completed_strategies", "common.json")
    strategy_data = load_json_file(strategy_path, {})
    _completion_strategy_cache = strategy_data.get("judge_completion_by_keyframe", {})
    return _completion_strategy_cache


def load_trajectory_correctness_strategy(strategy_filename: str = "common.json") -> Dict:
    """
    加载轨迹正确性判断公共策略
    
    Returns:
        Dict: 轨迹正确性判断策略配置
    """
    global _trajectory_strategy_cache
    if strategy_filename == "common.json" and _trajectory_strategy_cache is not None:
        return _trajectory_strategy_cache
    strategy_path = os.path.join(config.KNOWLEDGE_BASE_DIR, "corrected_strategies", strategy_filename)
    strategy_data = load_json_file(strategy_path, {})
    strategy = strategy_data.get("judge_trajectory_correctness_by_keyframe", {})
    if strategy_filename == "common.json":
        _trajectory_strategy_cache = strategy
    return strategy

def format_enhanced_caption(step: Dict, use_detailed: bool = True) -> str:
    """
    从步骤信息中提取并格式化 caption，利用新 JSON 格式的额外信息
    
    Args:
        step: 步骤信息字典，可能包含新格式的 JSON 字段
        use_detailed: 是否使用详细格式（包含状态变化信息）
    
    Returns:
        str: 格式化后的 caption 描述
    """
    # 优先使用 sparse_caption（向后兼容）
    caption = step.get("sparse_caption", "")
    
    # 如果存在新格式的 JSON 字段，构建更详细的描述
    if use_detailed:
        has_new_format = any(key in step for key in [
            "prev_image_state", "current_image_state", "action_result",
            "control_changes", "page_changes", "current_state"
        ])
        
        if has_new_format:
            parts = []
            
            # 如果有操作结果，优先显示
            action_result = step.get("action_result", "")
            if action_result and action_result != "初始状态，无前序操作":
                parts.append(f"操作结果: {action_result}")
            
            # 如果有控件变化，显示变化
            control_changes = step.get("control_changes", "")
            if control_changes and control_changes != "无":
                parts.append(f"控件变化: {control_changes}")
            
            # 如果有页面变化，显示变化
            page_changes = step.get("page_changes", "")
            if page_changes and page_changes != "无":
                parts.append(f"页面变化: {page_changes}")
            
            # 如果有当前状态，显示状态
            current_state = step.get("current_state", "")
            if current_state:
                parts.append(f"当前状态: {current_state}")
            
            # 如果构建了详细描述，使用它；否则回退到 sparse_caption
            if parts:
                detailed_caption = " | ".join(parts)
                # 如果 sparse_caption 存在且较短，可以结合使用
                if caption and len(caption) < 100:
                    return f"{caption} | {detailed_caption}"
                else:
                    return detailed_caption
    
    # 如果没有新格式字段或 use_detailed=False，直接返回 sparse_caption
    return caption if caption else f"步骤{step.get('index', '?')}的界面"


def judge_by_qa_pairs(
    qa_pairs: List[Dict[str, str]], 
    action_thought_caption: List[Dict],
    ground_truth: Optional[int] = None,
    instruction: Optional[str] = None,
    use_qa_images: bool = False,
    keyframe_indices: Optional[List[int]] = None,
    action_thought_caption_results: Optional[List[Dict]] = None,
    profile_id: Optional[str] = None,
) -> int:
    """
    Prompt A：基于QA对预测答案并判断是否一致
    
    Args:
        qa_pairs: QA对列表 [{"question": "...", "answer": "..."}]
        action_thought_caption: 操作序列信息
        ground_truth: 任务真实完成状态（1=完成, 0=未完成）
        instruction: 任务指令（用于任务完成状态判断）
        use_qa_images: 是否传入关键帧图片
        keyframe_indices: 关键帧索引列表（用于获取图片URL）
        action_thought_caption_results: 所有步骤的结果（用于获取图片URL）
        
    Returns:
        int: 判断结果（1=预测答案与真实答案一致, 0=不一致, -1=无法判断）
    """
    task_id, _ = get_task_context()
    step_name_base = "judgment_qa"
    if profile_id:
        step_name_base = f"{step_name_base}_{profile_id}"
    set_task_context(task_id, step_name_base)
    
    # 如果传图，获取关键帧图片URL列表
    images = None
    if use_qa_images and keyframe_indices and action_thought_caption_results:
        images = []
        for idx in sorted(keyframe_indices):
            if 0 <= idx < len(action_thought_caption_results):
                result = action_thought_caption_results[idx]
                image_url = result.get("image_url", "")
                if image_url and image_url.strip():
                    images.append(image_url)
        # 如果没有获取到有效图片，不传图
        if not images:
            images = None
    
    all_correct = True
    has_valid_result = False
    qa_details = []
    
    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        true_answer = qa.get("answer", "")
        
        if not question or not true_answer:
            continue
        
        prompt = build_qa_prediction_prompt(question, action_thought_caption, use_qa_images)
        
        response = call_llm_with_prompt(prompt, images=images, temperature=0.1, max_tokens=200, 
                                       task_id=task_id, step_name=f"judgment_qa_q{i}_{profile_id}" if profile_id else f"judgment_qa_q{i}")
        
        if not response:
            continue
        
        predicted_answer = response.strip()
        predicted_answer = predicted_answer.strip('"').strip("'")
        
        is_correct = (predicted_answer.lower() == true_answer.lower())
        
        qa_details.append({
            "q": question,
            "a_pred": predicted_answer,
            "a_gt": true_answer,
            "tf": is_correct
        })
        
        if not is_correct:
            all_correct = False
        
        has_valid_result = True
    
    completion_judgment = -1
    if ground_truth is not None:
        completion_question = "这些涉及到的问题的任务模块，是否最终实现完成？"
        completion_true_answer = "是" if ground_truth == 1 else "否"
        
        if instruction:
            prompt = build_task_completion_prompt(instruction, action_thought_caption, use_qa_images)
        else:
            prompt = build_qa_prediction_prompt(completion_question, action_thought_caption, use_qa_images)
        
        response = call_llm_with_prompt(prompt, images=images, temperature=0.1, max_tokens=200, 
                                       task_id=task_id, step_name=f"judgment_qa_q{len(qa_pairs)}_{profile_id}" if profile_id else f"judgment_qa_q{len(qa_pairs)}")
        
        if response:
            predicted_answer = response.strip()
            predicted_answer = predicted_answer.strip('"').strip("'")
            
            predicted_lower = predicted_answer.lower()
            completion_positive_keywords = ["是", "yes", "完成", "true", "1", "已", "成功", "实现"]
            completion_negative_keywords = ["否", "no", "未完成", "false", "0", "未", "失败", "未实现"]
            
            if completion_true_answer == "是":
                is_correct = any(keyword in predicted_lower for keyword in completion_positive_keywords)
                completion_judgment = 1 if any(keyword in predicted_lower for keyword in completion_positive_keywords) else 0
            else:
                is_correct = any(keyword in predicted_lower for keyword in completion_negative_keywords)
                completion_judgment = 0 if any(keyword in predicted_lower for keyword in completion_negative_keywords) else 1
            
            qa_details.append({
                "q": completion_question,
                "a_pred": predicted_answer,
                "a_gt": completion_true_answer,
                "tf": is_correct
            })
            
            if not is_correct:
                all_correct = False
            
            has_valid_result = True
    
    if qa_details:
        save_qa_details_to_csv(task_id, qa_details, ground_truth)
    
    if not has_valid_result:
        return -1
    
    if ground_truth is not None and completion_judgment != -1:
        return completion_judgment
    
    return 1 if all_correct else 0


def judge_completion_by_keyframes(
    instruction: str, 
    keyframe_captions: List[str], 
    keyframe_actions: List[str], 
    keyframe_indices: Optional[List[int]] = None, 
    action_thought_caption_results: Optional[List[Dict]] = None, 
    use_keyframe_images: bool = False,
    profile_id: Optional[str] = None,
) -> int:
    """
    完成度判断：判断关键帧是否有任务完成时的最终页面
    
    Args:
        instruction: 任务指令
        keyframe_captions: 关键帧的sparse caption列表
        keyframe_actions: 关键帧的action desc列表
        keyframe_indices: 关键帧索引列表（用于获取图片URL）
        action_thought_caption_results: 所有步骤的结果（用于获取图片URL）
        use_keyframe_images: 是否传入关键帧图片
        
    Returns:
        int: 完成状态（1=完成, 0=未完成, -1=无法判断）
    """
    task_id, _ = get_task_context()
    step_name_base = "judgment_completion"
    if profile_id:
        step_name_base = f"{step_name_base}_{profile_id}"
    set_task_context(task_id, step_name_base)
    
    # 加载完成度判断公共策略
    judge_config = load_completion_strategy()
    if not judge_config:
        return -1
    
    # 如果传图，获取图片URL列表
    images = None
    if use_keyframe_images and keyframe_indices and action_thought_caption_results:
        images = []
        for idx in sorted(keyframe_indices):
            if 0 <= idx < len(action_thought_caption_results):
                result = action_thought_caption_results[idx]
                image_url = result.get("image_url", "")
                if image_url and image_url.strip():
                    images.append(image_url)
        # 如果没有获取到有效图片，不传图
        if not images:
            images = None
    
    prompt = build_completion_judgment_prompt(instruction, keyframe_captions, keyframe_actions, judge_config, use_keyframe_images)
    
    response = call_llm_with_prompt(prompt, images=images, temperature=0.01, max_tokens=500,
                                   task_id=task_id, step_name=step_name_base)
    
    if not response:
        return -1
    
    result = parse_json_response(response)
    if not result:
        return -1
    
    return _parse_status_to_int(result.get("status", -1))


def judge_trajectory_correctness_by_keyframes(
    instruction: str, 
    keyframe_captions: List[str], 
    keyframe_actions: List[str], 
    action_thought_caption: List[Dict],
    keyframe_indices: Optional[List[int]] = None, 
    action_thought_caption_results: Optional[List[Dict]] = None, 
    use_keyframe_images: bool = False,
    profile_id: Optional[str] = None,
) -> int:
    """
    轨迹正确性判断：判断当前GUI轨迹的正确性
    
    Args:
        instruction: 任务指令
        keyframe_captions: 关键帧的sparse caption列表
        keyframe_actions: 关键帧的action desc列表
        action_thought_caption: 完整的操作序列信息（用于轨迹分析）
        keyframe_indices: 关键帧索引列表（用于获取图片URL）
        action_thought_caption_results: 所有步骤的结果（用于获取图片URL）
        use_keyframe_images: 是否传入关键帧图片
        
    Returns:
        int: 正确性状态（1=正确, 0=错误, -1=无法判断）
    """
    task_id, _ = get_task_context()
    step_name_base = "judgment_trajectory_correctness"
    if profile_id:
        step_name_base = f"{step_name_base}_{profile_id}"
    set_task_context(task_id, step_name_base)
    
    # 加载轨迹正确性判断公共策略
    judge_config = load_trajectory_correctness_strategy()
    if not judge_config:
        return -1
    
    # 如果传图，获取图片URL列表
    images = None
    if use_keyframe_images and keyframe_indices and action_thought_caption_results:
        images = []
        for idx in sorted(keyframe_indices):
            if 0 <= idx < len(action_thought_caption_results):
                result = action_thought_caption_results[idx]
                image_url = result.get("image_url", "")
                if image_url and image_url.strip():
                    images.append(image_url)
        # 如果没有获取到有效图片，不传图
        if not images:
            images = None
    
    prompt = build_trajectory_correctness_judgment_prompt(
        instruction, keyframe_captions, keyframe_actions, action_thought_caption, judge_config, 
        use_keyframe_images, keyframe_indices=keyframe_indices
    )
    
    # 增大 max_tokens 以容纳结构化输出（task_decomposition + key_step_analysis + status + reason）
    response = call_llm_with_prompt(prompt, images=images, temperature=0.01, max_tokens=800,
                                   task_id=task_id, step_name=step_name_base)

    if not response:
        return -1

    result = parse_json_response(response)
    if not result:
        return -1

    return _parse_status_to_int(result.get("status", -1))


def _format_keyframes_for_unified_inspection(
    keyframe_indices: List[int],
    keyframe_captions: List[str],
    keyframe_actions: List[str],
) -> str:
    lines = ["关键帧证据（用于轨迹正确性与终态完成度）：" ]
    for i, idx in enumerate(keyframe_indices):
        cap = keyframe_captions[i] if i < len(keyframe_captions) else ""
        act = keyframe_actions[i] if i < len(keyframe_actions) else ""
        cap = cap.strip().replace("\n", " ")
        act = act.strip().replace("\n", " ")
        lines.append(f"- 关键帧{i+1}（步骤{idx}）：{cap}；Action：{act}")
    return "\n".join(lines)


def _format_action_chain_from_keyframes(
    keyframe_indices: List[int],
    keyframe_actions: List[str],
) -> str:
    lines = []
    for i, idx in enumerate(keyframe_indices):
        act = keyframe_actions[i] if i < len(keyframe_actions) else ""
        act = act.strip().replace("\n", " ")
        lines.append(f"{i+1}. 关键帧步骤{idx}：{act}")
    return "\n".join(lines)


def judge_unified_task_success(
    instruction: str,
    keyframe_indices: List[int],
    keyframe_captions: List[str],
    keyframe_actions: List[str],
    action_thought_caption_results: List[Dict],
    profile_id: Optional[str] = None,
) -> int:
    """
    unified_single：用一个 prompt 同时判断「轨迹正确性」与「终态完成度」，输出 is_task_success。

    该函数复用 lite_judgment_prompt 的判断规则，只是把关键帧证据拼入 inspection_section，并传入末尾若干步图片。
    """

    # 与 main.py 保持一致
    last_n = 10
    last_n = min(last_n, len(action_thought_caption_results))
    last_steps = action_thought_caption_results[-last_n:] if last_n else []
    last_image_paths: List[str] = []
    for step in last_steps:
        p = step.get("image_url", "")
        if p:
            last_image_paths.append(p)

    # lite prompt 里的 last_steps_text
    last_steps_text = "以下为轨迹末尾若干步的截图（按时间顺序提供），请重点结合最后几张图判断终态是否达成与路径是否正确。"
    image_notice = "**注意**：本任务将同时提供轨迹末尾的若干张截图（按时间顺序），请结合图片与任务指令判断。\n\n"

    inspection_section = _format_keyframes_for_unified_inspection(
        keyframe_indices=keyframe_indices,
        keyframe_captions=keyframe_captions,
        keyframe_actions=keyframe_actions,
    )
    action_chain_text = _format_action_chain_from_keyframes(
        keyframe_indices=keyframe_indices,
        keyframe_actions=keyframe_actions,
    )

    prompt = get_lite_judgment_prompt(
        instruction,
        last_steps_text=last_steps_text,
        action_chain=action_chain_text,
        thought_action_chain="",
        image_notice_section=image_notice,
        inspection_section=inspection_section,
    )

    task_id, _ = get_task_context()
    step_name = "judgment_unified"
    if profile_id:
        step_name = f"{step_name}_{profile_id}"
    response = call_llm_with_prompt(
        prompt,
        images=last_image_paths if last_image_paths else None,
        temperature=0.01,
        max_tokens=1200,
        task_id=task_id,
        step_name=step_name,
    )

    if not response:
        return -1

    parsed = parse_json_response(response)
    if parsed is None or "is_task_success" not in parsed:
        return -1

    v = parsed.get("is_task_success")
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return -1
    return -1
    
    if not response:
        return -1
    
    result = parse_json_response(response)
    if not result:
        return -1
    
    # 新格式返回status字段
    correctness_status = result.get("status", -1)
    
    # 如果status是字符串，转换为数字
    # correct -> 1, incorrect -> 0, uncertain_need_better_keyframe -> -1
    if isinstance(correctness_status, str):
        if correctness_status == "correct":
            correctness_status = 1
        elif correctness_status == "incorrect":
            correctness_status = 0
        else:
            correctness_status = -1
    
    # 验证状态值
    if correctness_status not in [1, 0, -1]:
        return -1
    
    return correctness_status


def build_qa_prediction_prompt(question: str, action_thought_caption: List[Dict], use_qa_images: bool = False) -> str:
    """构建基于操作信息预测答案的prompt"""
    
    # 根据use_qa_images决定是否显示caption
    operations_text = ""
    for i, step in enumerate(action_thought_caption):
        action = step.get("action", "")
        thought = step.get("thought", "")
        operations_text += f"步骤{i}:\n"
        operations_text += f"  操作: {action}\n"
        operations_text += f"  意图: {thought}\n"
        if not use_qa_images:
            # 使用增强的 caption 格式化函数
            caption = format_enhanced_caption(step, use_detailed=True)
            operations_text += f"  界面描述: {caption}\n"
        operations_text += "\n"
    
    # 根据use_qa_images添加图片说明
    image_notice = ""
    image_suffix = ""
    if use_qa_images:
        image_notice = "**注意**：已提供关键帧图片序列，请结合图片和文本信息回答问题。\n\n"
        image_suffix = "（关键帧图片已按步骤顺序提供）"
    
    prompt = f"""{image_notice}你是一个任务分析专家。请根据操作序列信息回答问题。

问题：
{question}

操作序列信息：
{operations_text}{image_suffix}

请仔细分析操作序列，从操作步骤、界面描述等信息中提取答案。

只输出答案，不要添加其他说明文字。如果无法从操作信息中确定答案，输出"无法确定"。"""
    
    return prompt


def load_latest_improvements(
    improvement_type: str = "qa_judgment",
    model_name: Optional[str] = None
) -> str:
    """
    动态加载最新的改进知识（从 final_improvements.json 读取）
    
    Args:
        improvement_type: 改进知识类型 ("qa_judgment" 或 "keyframe_judgment")
        model_name: 模型名称，如果为None则从prompt_utils获取
    
    Returns:
        str: 改进知识文本，如果找不到则返回空字符串
    """
    try:
        if model_name is None:
            model_name = get_model_name()
        cache_key = (improvement_type, model_name or "")
        if cache_key in _improvements_cache:
            return _improvements_cache[cache_key]
        if not model_name:
            return ""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if improvement_type == "keyframe_judgment":
            cache_dir = os.path.join(project_root, "data", model_name, "prompt_improvements_cache")
        elif improvement_type == "qa_judgment":
            cache_dir = os.path.join(project_root, "data", model_name, "qa_prompt_improvements_cache")
        else:
            return ""
        final_cache_file = os.path.join(cache_dir, "final_improvements.json")
        if not os.path.exists(final_cache_file):
            _improvements_cache[cache_key] = ""
            return ""
        with open(final_cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        if cache_data.get("type") != improvement_type:
            _improvements_cache[cache_key] = ""
            return ""
        improvements = cache_data.get("improvements", "")
        _improvements_cache[cache_key] = improvements
        return improvements
    except Exception:
        return ""


def build_task_completion_prompt(instruction: str, action_thought_caption: List[Dict], use_qa_images: bool = False) -> str:
    """构建任务完成状态判断的专门prompt"""
    
    # 根据use_qa_images决定是否显示caption
    operations_text = ""
    for i, step in enumerate(action_thought_caption):
        action = step.get("action", "")
        thought = step.get("thought", "")
        operations_text += f"步骤{i}:\n"
        operations_text += f"  操作: {action}\n"
        operations_text += f"  意图: {thought}\n"
        if not use_qa_images:
            # 使用增强的 caption 格式化函数
            caption = format_enhanced_caption(step, use_detailed=True)
            operations_text += f"  界面描述: {caption}\n"
        operations_text += "\n"
    
    last_steps_text = ""
    last_n = min(3, len(action_thought_caption))
    for i in range(max(0, len(action_thought_caption) - last_n), len(action_thought_caption)):
        step = action_thought_caption[i]
        action = step.get("action", "")
        thought = step.get("thought", "")
        last_steps_text += f"步骤{i}:\n"
        last_steps_text += f"  操作: {action}\n"
        last_steps_text += f"  意图: {thought}\n"
        if not use_qa_images:
            # 使用增强的 caption 格式化函数
            caption = format_enhanced_caption(step, use_detailed=True)
            last_steps_text += f"  界面描述: {caption}\n"
        last_steps_text += "\n"
    
    # 动态加载最新改进知识（QA判断的改进知识）
    improvements = load_latest_improvements("qa_judgment")
    improvements_section = ""
    if improvements:
        improvements_section = f"""
- **基于错误案例分析的改进建议**：
{improvements}

"""
    
    # 根据use_qa_images添加图片说明
    image_notice = ""
    image_suffix = ""
    if use_qa_images:
        image_notice = "**注意**：已提供关键帧图片序列，请结合图片和文本信息判断任务完成状态。\n\n"
        image_suffix = "（关键帧图片已按步骤顺序提供，请重点关注最后几个步骤的图片）"
    
    prompt = f"""{image_notice}你是一个任务完成度评估专家。请根据任务指令和操作序列，判断任务是否真正完成。

任务指令：
{instruction}

完整操作序列：
{operations_text}{image_suffix}

最后几个步骤（重点关注）：
{last_steps_text}

判断标准：
1. **任务完成**（回答"是"）：必须满足以下条件之一：
   
   **情况A：完全完成**
   - 操作序列包含了任务指令要求的所有关键步骤
   - 最后的状态显示了任务已成功完成（如：支付完成、订单确认、提交成功、任务结束等）
   - 有明确的完成性操作（如：点击"确认"、"提交"、"支付"、"完成"等）
   - 最终界面显示任务已达成目标
   
   **情况B：只差最后一步提交（可接受）**
   - 所有必要的信息都已填写完成（如：商品已选择、地址已填写、支付方式已选择等）
   - 已经到达了提交/确认页面，所有表单字段都已填写
   - 只差最后一步点击"提交"或"确认"或"发送"按钮
   - 这种情况下，虽然技术上没有提交，但用户已经完成了所有准备工作，可以算作完成
   
   **情况C：信息缺失场景接管（可接受）**
   - 如果最后一个操作是"信息缺失场景接管"或类似的人工介入提示
   - 但在此之前已经到达了具体的信息填写/待提交/待确认的页面，所有必要信息已填写
   - 界面显示已准备好提交，只是需要人工确认或补充少量信息
   - 这种情况下，如果用户可以直接提交，应该算作完成

2. **任务未完成**（回答"否"）：如果出现以下情况：
   - **订购类任务**：只停留在搜索/筛选/浏览页面，没有进行实际的选择、确认或提交操作
     - 只搜索了商品/酒店/民宿 → 未完成
     - 只筛选了条件，停留在列表浏览页面 → 未完成
     - 必须看到：已选择具体商品/服务、已填写必要信息、已到达提交/确认页面（或已提交）
   - 查询类任务，查询的列表内容与当前任务要求的槽位值没有关联
   - 只执行了部分步骤（如：订购商品任务，只搜索了商品，但没有选择商品或填写信息）
   - 只填写了部分信息，还缺少关键信息（如：只填写了地址，但没有选择商品）
   - 还在中间步骤（如：输入中、选择中、浏览中、未到达提交页面等）
   - 缺少多个关键步骤（如：没有选择商品、没有填写地址、没有选择支付方式等）
   - 最终界面显示任务未达成目标
   - **任务无法执行**：如果任务指令要求的操作无法执行（如：删除地址但原本就没有地址、查看资源但资源不存在等），应该判断为未完成

特别注意：
- **不要仅根据槽位是否填写来判断**，必须看是否到达了提交阶段
- **重点关注最后几个步骤**，看是否已经准备好提交
- **区分"只差提交"和"缺少信息"**：
  - 如果所有信息都已填写，只差提交按钮 → 算完成（情况B）
  - 如果还缺少关键信息，还在填写过程中 → 算未完成
- **对于订购类任务**：
  - 只搜索/筛选/浏览 → 未完成
  - 必须看到已选择具体商品/服务、已填写信息、已到达提交页面（或已提交）
  - 不能只停留在列表浏览页面，必须进行实际的选择和确认操作
- **对于查询类任务**：必须看到查询结果页面才算完成
- **对于设置类任务**：必须看到已填写所有设置、已到达保存页面（或已保存）
- **对于创建类任务**：如果出现"信息缺失场景接管"，需要判断是否已经到达可以提交的状态
{improvements_section}请仔细分析操作序列，特别是最后几个步骤，判断任务是否真正完成或已准备好提交。

只输出答案（"是"或"否"），不要添加其他说明文字。"""
    
    return prompt


def save_qa_details_to_csv(task_id: str, qa_details: List[Dict], ground_truth: Optional[int] = None) -> None:
    """保存QA预测详情到CSV文件（按task_id分文件夹）"""
    try:
        import config
        from utils.prompt_utils import get_model_name
        model_name = get_model_name()
        base_dir = config.get_tmp2_dir(model_name)
        
        # 按task_id分文件夹
        task_dir = os.path.join(base_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        safe_task_id = task_id.replace('/', '_').replace('\\', '_').replace(':', '_')
        csv_path = os.path.join(task_dir, f"{safe_task_id}_judgment_qa_details.csv")
        
        response_value = True if ground_truth == 1 else False
        
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["task_id", "q", "a_pred", "a_gt", "tf", "response"])
            writer.writeheader()
            for detail in qa_details:
                writer.writerow({
                    "task_id": task_id,
                    "q": detail["q"],
                    "a_pred": detail["a_pred"],
                    "a_gt": detail["a_gt"],
                    "tf": detail["tf"],
                    "response": response_value
                })
    except Exception as e:
        print(f"保存QA详情CSV失败: {e}")


def build_completion_judgment_prompt(
    instruction: str, 
    keyframe_captions: List[str], 
    keyframe_actions: List[str], 
    judge_config: Dict,
    use_keyframe_images: bool = False
) -> str:
    """构建完成度判断prompt"""
    
    # 根据use_keyframe_images设置变量
    if use_keyframe_images:
        image_notice_section = "**注意**：本任务将同时提供关键帧图片序列，请结合图片内容和Action描述进行判断。\n\n"
        keyframe_info_suffix = "（将提供关键帧图片序列，请结合图片和Action信息进行判断）"
        # keyframes_text 不包含Caption行
        keyframes_text = ""
        for i, action in enumerate(keyframe_actions, 1):
            keyframes_text += f"关键帧{i}:\n"
            keyframes_text += f"  Action: {action}\n\n"
    else:
        image_notice_section = ""
        keyframe_info_suffix = ""
        # keyframes_text 包含Caption行
        keyframes_text = ""
        for i, (caption, action) in enumerate(zip(keyframe_captions, keyframe_actions), 1):
            keyframes_text += f"关键帧{i}:\n"
            keyframes_text += f"  Caption: {caption}\n"
            keyframes_text += f"  Action: {action}\n\n"
    
    # 提取判断配置
    judgment_criteria = judge_config.get("judgment_criteria", {})
    output_config = judge_config.get("output", {})
    status_options = output_config.get("status", ["completed", "not_completed", "uncertain_need_better_keyframe"])
    required_reason = output_config.get("required_reason", "判断理由")
    
    # 构建各判断标准的文本（支持 {definition, conditions} 结构和纯字符串）
    def format_criteria(criteria_data):
        if isinstance(criteria_data, dict):
            definition = criteria_data.get("definition", "")
            conditions = criteria_data.get("conditions", [])
            text = f"{definition}\n" if definition else ""
            for cond in conditions:
                text += f"- {cond}\n"
            return text.strip()
        return str(criteria_data)
    
    completed_text = format_criteria(judgment_criteria.get("completed", ""))
    not_completed_text = format_criteria(judgment_criteria.get("not_completed", ""))
    uncertain_text = format_criteria(judgment_criteria.get("uncertain", ""))
    
    prompt = f"""你是一个任务完成度评估专家（Outcome-based Evaluation）。

核心问题：结合关键帧截图，任务最终要求的结果态是否可被直接观察到？

总则（优先级最高）：
- 截图是唯一可信证据；Action 仅作辅助，不可替代截图证据。
- “看不到结果”不等于“已经完成”：若截图无法看到关键结果，不得判 completed。
- 到达页面不等于完成操作：必须看到核心操作结果，而不是仅停在入口/设置/搜索页。
- 判定前先提取显式约束并逐项核查（入口/模式/场景/工具/筛选排序/操作动词/关键参数）。
- 多子目标任务必须全部达成（如“并/同时/先...再...”引出的可独立执行动作）。
- 差一步提交可判完成：所有必填信息已齐、仅差最终确认点击。
- 判定优先级：当 completed 与 not_completed 均可能命中时，优先检查 not_completed。
- 主观属性任务在无客观筛选维度时，合理选择可接受。

输入：
{image_notice_section}任务指令：
{instruction}

关键帧信息{keyframe_info_suffix}：
{keyframes_text}

判断标准：

**任务完成（返回"completed"）**：
{completed_text}

**任务未完成（返回"not_completed"）**：
{not_completed_text}

**无法确定（返回"uncertain_need_better_keyframe"）**：
{uncertain_text}

重点核查：
- 是否有清晰终态证据（成功提示、结果页、确认态、目标信息已呈现）。
- 是否仅停留在中间状态（搜索/筛选/浏览/填写中）。
- 是否存在关键约束不满足但被错误放行。
- 是否仅完成主目标而遗漏独立子目标。

输出格式（必须是有效的JSON）：
{{
    "status": "{status_options[0]}" 或 "{status_options[1]}" 或 "{status_options[2] if len(status_options) > 2 else 'uncertain_need_better_keyframe'}",
    "reason": "{required_reason}"
}}

请直接输出JSON，不要添加其他说明文字。"""
    
    return prompt


def build_trajectory_correctness_judgment_prompt(
    instruction: str, 
    keyframe_captions: List[str], 
    keyframe_actions: List[str], 
    action_thought_caption: List[Dict],
    judge_config: Dict,
    use_keyframe_images: bool = False,
    keyframe_indices: Optional[List[int]] = None,
    all_steps_are_keyframes: bool = False,
) -> str:
    """构建轨迹正确性判断prompt
    
    优化设计：
    - 合并为单一操作链，用 [关键帧] 标记区分关键决策步骤和普通步骤
    - 关键帧展示详细状态信息（current_state, control_changes, page_changes）
    - 非关键帧仅展示 Action + sparse_caption（<=80字摘要）
    - 引入结构化评估流程引导 LLM 分步推理
    """
    
    # 构建关键帧索引集合
    kf_set = set(keyframe_indices) if keyframe_indices else set()
    
    # 图片提示
    image_notice_section = ""
    if use_keyframe_images:
        if all_steps_are_keyframes:
            image_notice_section = "**注意**：本任务将同时提供每一步对应的截图（按时间顺序），请逐步结合图片内容和操作描述进行判断。\n\n"
        else:
            image_notice_section = "**注意**：本任务将同时提供关键帧图片序列（标有 [关键帧] 的步骤），请结合图片内容和操作描述进行判断。\n\n"
    
    # 构建单一操作链（严格按时间因果排序：页面 → 动作 → 变化结果 → 下一页面）
    # 字段时序说明：step[i] 的 action_result/control_changes/page_changes 描述的是
    # step[i-1] 的操作导致的变化（即从页面i-1到页面i的转变），不是 step[i] 自身操作的结果。
    # 因此，要展示动作i的结果，需要取 step[i+1] 的这些字段。
    action_chain_lines = []
    
    def _is_valid_field(val):
        """判断字段值是否有效（非空、非占位值）"""
        return val and val not in ("无", "无变化", "初始状态，无前序操作")
    
    for i, step in enumerate(action_thought_caption):
        is_keyframe = i in kf_set
        action = step.get("action", "")
        is_last_step = (i == len(action_thought_caption) - 1)
        
        # 获取结构化字段
        operation_intent = step.get("thought_operation_intent", "")
        current_state = step.get("current_state", "") or step.get("current_image_state", "") or step.get("sparse_caption", "")
        
        if is_keyframe:
            lines = []
            
            # ① 页面i（step i 执行前看到的画面）
            if not use_keyframe_images:
                lines.append(f"[关键帧] 页面{i}: {current_state}" if current_state else f"[关键帧] 页面{i}")
            else:
                lines.append(f"[关键帧] 页面{i}: (见图片)")
            
            # ② 动作（最后一步不显示动作）
            if not is_last_step:
                lines.append(f"    第{i}步: [{action}]")
                
                # ③ 操作意图（为什么做这个操作）
                if _is_valid_field(operation_intent):
                    lines.append(f"    操作意图: {operation_intent}")
                
                # ④ 动作i的执行结果（取自 step[i+1] 的字段）
                if i + 1 < len(action_thought_caption):
                    next_step = action_thought_caption[i + 1]
                    next_action_result = next_step.get("action_result", "")
                    next_control_changes = next_step.get("control_changes", "")
                    next_page_changes = next_step.get("page_changes", "")
                    
                    if _is_valid_field(next_action_result):
                        lines.append(f"    → 操作结果: {next_action_result}")
                    if _is_valid_field(next_control_changes):
                        lines.append(f"    → 控件变化: {next_control_changes}")
                    if _is_valid_field(next_page_changes):
                        lines.append(f"    → 页面变化: {next_page_changes}")
            
            action_chain_lines.append("\n".join(lines))
            
        else:
            # 非关键帧：紧凑格式（只有页面 + 动作）
            if not use_keyframe_images:
                line = f"         页面{i}: {current_state}"
                if not is_last_step:
                    line += f"\n         第{i}步: [{action}]"
            else:
                if not is_last_step:
                    line = f"         第{i}步: [{action}]"
                else:
                    line = ""
            if line:
                action_chain_lines.append(line)
    
    action_chain = "\n".join(action_chain_lines)
    
    # 提取判断配置
    judgment_criteria = judge_config.get("judgment_criteria", {})
    output_config = judge_config.get("output", {})
    status_options = output_config.get("status", ["correct", "incorrect", "uncertain_need_better_keyframe"])
    
    # 提取关键检查点
    key_inspection_points = judge_config.get("key_inspection_points", [])
    inspection_text = ""
    if key_inspection_points:
        for point in key_inspection_points:
            inspection_text += f"- {point}\n"
    
    # 构建各判断标准的文本
    def format_criteria(criteria_data):
        if isinstance(criteria_data, dict):
            definition = criteria_data.get("definition", "")
            conditions = criteria_data.get("conditions", [])
            text = f"{definition}\n" if definition else ""
            for cond in conditions:
                text += f"- {cond}\n"
            return text.strip()
        return str(criteria_data)
    
    correct_text = format_criteria(judgment_criteria.get("correct", ""))
    incorrect_text = format_criteria(judgment_criteria.get("incorrect", ""))
    uncertain_text = format_criteria(judgment_criteria.get("uncertain", ""))
    
    # 构建关键检查点段落
    inspection_section = ""
    if inspection_text:
        inspection_section = f"""重点检查项：
{inspection_text}
"""
    
    principles = [
        "- **只评过程，不评完成度**：路径没走完但方向正确 = correct；走完了但中途有未纠正关键错误 = incorrect",
        "- **截图证据优先**：截图是第一证据，动作链仅作辅助，不可替代可见证据",
        "- **显式约束逐项核查**：先提取入口/模式/场景/工具/筛选排序/操作动词/关键参数，逐项核实",
        "- **判定优先级**：当 correct 与 incorrect 同时可能命中时，优先检查 incorrect；命中则不得以 correct 豁免",
        "- **指定入口/模式不可替代**：任务明确指定入口/模式/场景/工具时，其他路径不等价",
        "- **未接管的替代选择 = incorrect**：关键选项不存在时，未接管而选不匹配替代项，判 incorrect",
        "- **自我纠正不扣分**：中途探索、回退、修正可接受，但最终过程不得保留未纠正偏航"
    ]
    if all_steps_are_keyframes:
        principles.append("- **逐步审查**：每一步均有截图，请结合每步动作与截图连续性判断路径是否偏离任务意图")
        step2_non_keyframe_text = ""
    else:
        principles.append("- **聚焦关键决策点**：标有 [关键帧] 的步骤是导航决策点，权重最高")
        step2_non_keyframe_text = """对非关键步骤，只检查：
- 是否存在明显的方向性错误（如突然离开了正确页面没纠正）
"""

    prompt = f"""# Role
你是一个 GUI Agent 轨迹正确性评估器（Process-based Evaluator）。

# Task
判断 Agent 已执行的操作路径是否与任务意图一致。

# 评估原则
{chr(10).join(principles)}

{image_notice_section}# Input

## 任务指令
{instruction}

## Agent 操作链（按时间顺序，[关键帧] 标记表示关键决策步骤）
{action_chain}

# 评估流程

## Step 1: 解析任务意图与显式约束
从任务指令中提取：
- **目标动作**：最终要做什么（浏览/购买/设置/查询...）
- **关键约束**：必须满足的槽位值（App名/页面/商品名/规格/地址/日期等）
- **显式约束**：指定入口/模式/场景/工具、筛选排序条件、操作动词、多子目标要求
- **必经路径**：要到达目标，逻辑上必须经过哪些页面或功能模块

## Step 2: 逐步审查操作链
对每个 [关键帧] 步骤，回答：
- 这一步选择的目标（按钮/选项/输入值）是否与任务意图一致？
- 操作后到达的状态是否在正确的路径上？
- 如果出现偏差，后续是否有纠正操作？

{step2_non_keyframe_text}

{inspection_section}## Step 3: 路径一致性判断
将操作链串联起来，判断整条链路的意图是否与任务指令一致。
注意防止"表面通顺但目标偏移"：即使每一步都是合法操作，但串起来做的是另一件事，则为 incorrect。

# 判断标准

**correct**：
{correct_text}

**incorrect**：
{incorrect_text}

**uncertain_need_better_keyframe**：
{uncertain_text}

# Output
输出格式（必须是有效的JSON）：
{{
    "task_decomposition": "任务意图分解：目标动作 + 关键约束 + 必经路径",
    "key_step_analysis": "按步骤回顾每步做了什么，再总结整条轨迹为何正确/错误（过程化说明）",
    "status": "{status_options[0]}" 或 "{status_options[1]}" 或 "{status_options[2] if len(status_options) > 2 else 'uncertain_need_better_keyframe'}",
    "reason": "先给整条轨迹错误/正确的总原因，再给最终判定依据",
    "first_deviation_step": "首次未纠正偏航对应的真实步骤 index（0-based，与输入轨迹步骤一致；仅在存在未纠正偏航时输出）",
    "first_deviation_reason": "该首次未纠正偏航步骤为何偏航（简要原因）"
}}

输出顺序要求：
1) 先在 key_step_analysis 中按步骤说明每步做了什么（可简短过程化推理）；
2) 再在 reason 中总结整条轨迹为什么错/对；
3) 若 status=incorrect，最后明确 first_deviation_step（必须是首次且未被后续纠正）。
4) 若不存在未纠正偏航（status!=incorrect 或虽有偏差但已纠正），不要输出 first_deviation_step / first_deviation_reason 字段。

请直接输出JSON，不要添加其他说明文字。"""
    
    return prompt


def judge_task_completion(
    instruction: str,
    qa_pairs: Optional[List[Dict[str, str]]] = None,
    keyframe_captions: Optional[List[str]] = None,
    keyframe_actions: Optional[List[str]] = None,
    action_thought_caption: Optional[List[Dict]] = None,
    ground_truth: Optional[int] = None,
    category: Optional[str] = None,
    keyframe_indices: Optional[List[int]] = None,
    use_keyframe_images: bool = False,
    use_qa_images: bool = False,
    no_qa: bool = False,
    final_judgment_mode: str = "and_split",
    profile_id: Optional[str] = None,
) -> Dict[str, int]:
    """
    综合判断任务完成状态（使用多个Prompt）
    
    Args:
        instruction: 任务指令
        qa_pairs: QA对列表（可选）
        keyframe_captions: 关键帧caption列表（可选）
        keyframe_actions: 关键帧action列表（可选）
        action_thought_caption: 操作序列信息（可选，用于QA预测和轨迹分析）
        ground_truth: 任务真实完成状态（1=完成, 0=未完成）
        category: 任务分类
        keyframe_indices: 关键帧索引列表（用于获取图片URL）
        use_keyframe_images: 是否传入关键帧图片（用于关键帧判断）
        use_qa_images: 是否传入关键帧图片（用于QA判断）
        no_qa: 是否跳过QA相关的prompt判定
        
    Returns:
        Dict: {
            "qa_judgment": 基于QA对的判断结果,
            "completion_judgment": 完成度判断结果,
            "trajectory_correctness_judgment": 轨迹正确性判断结果,
            "keyframe_judgment": 基于关键帧的判断结果（兼容性，保留原字段）,
            "final_judgment": 综合判断结果（完成度=1 且 轨迹正确性=1 时判对）
        }
    """
    # unified：只产出 final_judgment（让主流程按 lite 分支统计）
    if final_judgment_mode == "unified_single":
        if not (keyframe_indices and keyframe_captions and keyframe_actions and action_thought_caption):
            return {"final_judgment": -1}
        task_id, _ = get_task_context()
        set_task_context(task_id, "judgment_unified")
        is_task_success = judge_unified_task_success(
            instruction=instruction,
            keyframe_indices=keyframe_indices,
            keyframe_captions=keyframe_captions,
            keyframe_actions=keyframe_actions,
            action_thought_caption_results=action_thought_caption,
            profile_id=profile_id,
        )
        return {"final_judgment": is_task_success}

    results = {
        "qa_judgment": -1,
        "completion_judgment": -1,
        "trajectory_correctness_judgment": -1,
        "keyframe_judgment": -1,  # 兼容性字段
        "final_judgment": -1,
    }
    
    futures = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        if not no_qa and qa_pairs and action_thought_caption:
            futures["qa_judgment"] = executor.submit(
                judge_by_qa_pairs,
                qa_pairs, action_thought_caption, ground_truth, instruction,
                use_qa_images=use_qa_images,
                keyframe_indices=keyframe_indices,
                action_thought_caption_results=action_thought_caption,
                profile_id=profile_id,
            )
        if keyframe_captions and keyframe_actions:
            futures["completion_judgment"] = executor.submit(
                judge_completion_by_keyframes,
                instruction, keyframe_captions, keyframe_actions,
                keyframe_indices=keyframe_indices,
                action_thought_caption_results=action_thought_caption,
                use_keyframe_images=use_keyframe_images,
                profile_id=profile_id,
            )
        if keyframe_captions and keyframe_actions and action_thought_caption:
            futures["trajectory_correctness_judgment"] = executor.submit(
                judge_trajectory_correctness_by_keyframes,
                instruction, keyframe_captions, keyframe_actions, action_thought_caption,
                keyframe_indices=keyframe_indices,
                action_thought_caption_results=action_thought_caption,
                use_keyframe_images=use_keyframe_images,
                profile_id=profile_id,
            )
        for key, future in futures.items():
            results[key] = future.result()
    
    # 兼容性：保留原有的keyframe_judgment字段（使用完成度判断结果）
    results["keyframe_judgment"] = results["completion_judgment"]
    
    # 综合判断：完成度=1 且 轨迹正确性=1 时判对
    results["final_judgment"] = combine_completion_and_correctness(
        results["completion_judgment"], results["trajectory_correctness_judgment"]
    )
    
    return results


def combine_completion_and_correctness(completion_judgment: int, trajectory_correctness_judgment: int) -> int:
    """
    综合完成度和轨迹正确性判断结果
    
    规则：
    - 完成度=1 且 轨迹正确性=1 时返回1（判对）
    - 否则返回0（判错）
    - 如果任一为-1，返回-1
    """
    if completion_judgment == -1 or trajectory_correctness_judgment == -1:
        return -1
    
    # 两个都对时判对
    if completion_judgment == 1 and trajectory_correctness_judgment == 1:
        return 1
    
    # 有一个错时判错
    return 0


def combine_judgments(qa_judgment: int, keyframe_judgment: int) -> int:
    """
    综合两个判断结果（兼容性函数，保留原有逻辑）
    
    规则：
    - 如果两个判断一致，返回该结果
    - 如果两个判断不一致，优先使用keyframe_judgment（因为关键帧包含最后状态）
    - 如果有一个是-1，返回另一个
    - 如果两个都是-1，返回-1
    """
    if qa_judgment == -1 and keyframe_judgment == -1:
        return -1
    
    if qa_judgment == -1:
        return keyframe_judgment
    
    if keyframe_judgment == -1:
        return qa_judgment
    
    # 两个都有结果
    if qa_judgment == keyframe_judgment:
        return qa_judgment
    
    # 不一致时，优先使用keyframe_judgment（因为包含最后状态）
    return keyframe_judgment


if __name__ == "__main__":
    pass
