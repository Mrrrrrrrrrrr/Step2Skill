"""
关键帧提取模块
根据策略从操作序列中提取关键帧
必须包含最后N张图的caption和最后一个action
"""
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_utils import call_llm_with_prompt, parse_json_response
from prompts.strategy_generation import load_strategy
from prompts.keyframe_config import REQUIRED_LAST_FRAMES_COUNT


def extract_keyframes(instruction: str, action_thought_caption_results: List[Dict], category: str) -> Dict:
    """
    根据策略从操作序列中提取关键帧
    
    Args:
        instruction: 任务指令
        action_thought_caption_results: 2.1模块的输出（所有步骤的action、thought、caption）
        category: 任务分类
        
    Returns:
        Dict: {
            "keyframe_indices": [关键帧索引列表],
            "keyframe_captions": [关键帧的sparse caption列表],
            "keyframe_actions": [关键帧的action desc列表]
        }
    """
    # 加载策略（新格式）
    strategy_lib = load_strategy(category)
    
    # 检查是否是新格式
    if "select_keyframe" not in strategy_lib:
        raise ValueError(f"策略库 {category} 不是新格式，缺少 select_keyframe 字段")
    
    # 使用策略库
    strategy = strategy_lib
    
    # 应用策略选择关键帧
    keyframe_indices = apply_strategy_to_select_keyframes(
        instruction, action_thought_caption_results, strategy, category
    )
    
    # 确保包含最后N张图和最后一个action（公共策略）
    ensure_last_frames_included(keyframe_indices, len(action_thought_caption_results))
    
    # 提取关键帧的caption和action
    keyframe_captions = []
    keyframe_actions = []
    
    for idx in sorted(keyframe_indices):
        if 0 <= idx < len(action_thought_caption_results):
            result = action_thought_caption_results[idx]
            # 使用 current_state（如果有，更适合关键帧描述），否则回退到 sparse_caption
            caption = result.get("current_state", "") or result.get("sparse_caption", "")
            if not caption:
                caption = f"步骤{idx}的界面"
            
            keyframe_captions.append(caption)
            keyframe_actions.append(result.get("action", ""))
    
    return {
        "keyframe_indices": sorted(keyframe_indices),
        "keyframe_captions": keyframe_captions,
        "keyframe_actions": keyframe_actions
    }


def apply_strategy_to_select_keyframes(
    instruction: str, 
    action_thought_caption_results: List[Dict], 
    strategy: Dict,
    category: str
) -> List[int]:
    """
    应用策略选择关键帧
    
    Args:
        instruction: 任务指令
        action_thought_caption_results: 所有步骤的结果
        strategy: 策略（新格式或旧格式）
        category: 任务分类
        
    Returns:
        List[int]: 关键帧索引列表
    """
    # 使用新格式：select_keyframe字段
    select_keyframe_config = strategy.get("select_keyframe", {})
    goal = strategy.get("goal", "")
    
    # 构建prompt
    prompt = build_keyframe_selection_prompt(
        instruction, action_thought_caption_results, goal, select_keyframe_config, category
    )
    
    # 调用LLM
    response = call_llm_with_prompt(prompt, temperature=0.1, max_tokens=1000)
    
    if not response:
        # Fallback：只包含最后N张图
        return fallback_keyframe_selection(action_thought_caption_results)
    
    result = parse_json_response(response)
    if not result:
        # Fallback：只包含最后N张图
        return fallback_keyframe_selection(action_thought_caption_results)
    
    indices = result.get("keyframe_indices", [])
    
    # 验证索引有效性
    valid_indices = [idx for idx in indices if 0 <= idx < len(action_thought_caption_results)]
    
    if valid_indices:
        return valid_indices
    else:
        # Fallback：只包含最后N张图
        return fallback_keyframe_selection(action_thought_caption_results)


def build_keyframe_selection_prompt(
    instruction: str,
    action_thought_caption_results: List[Dict],
    goal: str,
    select_keyframe_config: Dict,
    category: str
) -> str:
    """构建关键帧选择prompt"""
    
    # 构建步骤摘要（使用 sparse_caption，关键帧选择只需要摘要级信息）
    steps_summary = []
    for i, result in enumerate(action_thought_caption_results):
        caption = result.get('sparse_caption', f"步骤{i}的界面")
        steps_summary.append(
            f"步骤{i}: Action={result.get('action', '')}, "
            f"Thought={result.get('thought', '')}, "
            f"Caption={caption}"
        )
    
    steps_text = "\n".join(steps_summary)
    
    # 提取策略配置
    must_include = select_keyframe_config.get("must_include", [])
    prefer = select_keyframe_config.get("prefer", [])
    avoid = select_keyframe_config.get("avoid", [])
    
    must_include_text = "\n".join([f"- {item}" for item in must_include])
    prefer_text = "\n".join([f"- {item}" for item in prefer]) if prefer else "无"
    avoid_text = "\n".join([f"- {item}" for item in avoid]) if avoid else "无"
    
    total_steps = len(action_thought_caption_results)
    # 使用配置的最后N张图数量
    last_n_indices = []
    if total_steps >= REQUIRED_LAST_FRAMES_COUNT:
        last_n_indices = list(range(total_steps - REQUIRED_LAST_FRAMES_COUNT, total_steps))
    elif total_steps >= 2:
        last_n_indices = list(range(total_steps))
    elif total_steps == 1:
        last_n_indices = [0]
    
    prompt = f"""你是一个关键帧选择专家。请根据策略从操作序列中选择关键帧。

任务指令：{instruction}
任务分类：{category}

策略目标：{goal}

关键帧选择规则：

**必须包含（must_include）**：
{must_include_text}

**优先选择（prefer）**：
{prefer_text}

**避免选择（avoid）**：
{avoid_text}

**公共策略（所有类别都必须遵守）**：
- 必须包含最后{REQUIRED_LAST_FRAMES_COUNT}张图的索引：{last_n_indices}
- 必须包含最后一个action对应的索引：{total_steps - 1 if total_steps > 0 else 0}

所有步骤信息：
{steps_text}

请根据上述规则选择关键帧，确保：
1. 满足所有"必须包含"的要求
2. 优先选择满足"优先选择"条件的帧
3. 避免选择"避免选择"的帧
4. 必须包含最后{REQUIRED_LAST_FRAMES_COUNT}张图（公共策略）

请输出JSON格式：
{{
    "keyframe_indices": [关键帧索引列表],
    "reason": "选择这些关键帧的理由"
}}

请直接输出JSON，不要添加其他说明文字。"""
    
    return prompt


def fallback_keyframe_selection(action_thought_caption_results: List[Dict]) -> List[int]:
    """Fallback关键帧选择：只包含最后N张图（公共策略）"""
    indices = []
    total = len(action_thought_caption_results)
    
    # 必须包含最后N张图（公共策略）
    if total >= REQUIRED_LAST_FRAMES_COUNT:
        indices = list(range(total - REQUIRED_LAST_FRAMES_COUNT, total))
    elif total >= 2:
        indices = list(range(total))
    elif total == 1:
        indices = [0]
    
    return indices


def ensure_last_frames_included(keyframe_indices: List[int], total_steps: int):
    """确保最后N张图和最后一个action被包含（公共策略）"""
    if total_steps == 0:
        return
    
    # 确保包含最后N张图（公共策略）
    if total_steps >= REQUIRED_LAST_FRAMES_COUNT:
        for idx in range(total_steps - REQUIRED_LAST_FRAMES_COUNT, total_steps):
            if idx not in keyframe_indices:
                keyframe_indices.append(idx)
    elif total_steps >= 2:
        for idx in range(total_steps):
            if idx not in keyframe_indices:
                keyframe_indices.append(idx)
    elif total_steps == 1:
        if 0 not in keyframe_indices:
            keyframe_indices.append(0)


if __name__ == "__main__":
    # 测试
    test_instruction = "帮我在酒店预订 App 上订成都顺安村居酒店的精致小单间房型"
    test_results = [
        {
            "index": 0,
            "action": "我点击了酒店",
            "thought": "选择酒店预订功能",
            "sparse_caption": "酒店预订首页"
        },
        {
            "index": 1,
            "action": "我选择了日期",
            "thought": "选择入住日期",
            "sparse_caption": "日期选择界面"
        },
        {
            "index": 2,
            "action": "我选择了房型",
            "thought": "选择房间类型",
            "sparse_caption": "房型选择界面"
        },
        {
            "index": 3,
            "action": "我点击了支付",
            "thought": "完成支付",
            "sparse_caption": "支付确认界面"
        }
    ]
    
    result = extract_keyframes(test_instruction, test_results, "booking_purchase")
    print("关键帧索引：", result["keyframe_indices"])
    print("关键帧Captions：", result["keyframe_captions"])
    print("关键帧Actions：", result["keyframe_actions"])
