"""
接管判断模块
为每个步骤判断是否需要用户接管
"""
from typing import Dict, List, Optional
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_utils import call_llm_with_prompt, parse_json_response, get_task_context, set_task_context
from utils.lite_prompt_knowledge import render_takeover_knowledge_section


def build_takeover_judgment_prompt(
    instruction: str,
    history_context: str,
    current_caption: str,
    takeover_knowledge_section: str = "",
) -> str:
    """
    构建接管判断的 prompt
    
    Args:
        instruction: 任务指令
        history_context: 历史上下文（cap+action序列）
        current_caption: 当前步骤的 caption
        
    Returns:
        str: 完整的 prompt 文本
    """
    knowledge_block = (
        f"\n# 接管知识（唯一规则来源）\n{takeover_knowledge_section.strip()}\n"
        if takeover_knowledge_section and takeover_knowledge_section.strip()
        else "\n# 接管知识（唯一规则来源）\n未注入接管知识时，保守输出 uncertain，并说明缺少规则依据。\n"
    )
    prompt = f"""# Role
你是一个手机 GUI 自动化 Agent 的接管判断器（Handover Gatekeeper）。
你的任务：根据任务指令与历史轨迹（截图+动作）以及当前截图，判断此刻是否必须让用户接管。

# Capabilities Assumption
Agent 能通过 ADB/自动化完成：点击、双击、长按、滑动、返回、输入文本、选择列表项、打开/关闭开关、等待加载、截图识别与常规表单填写。
如果仅靠这些操作就能继续推进任务，则不需要接管。
只有需要真实用户授权/确认/人工交互/责任承担时才接管。

# Input
任务指令：{instruction}

历史轨迹（按时间顺序）：
{history_context}

当前：Screenshot_current（最新屏幕）
{current_caption}

# Decision
你只判断当前是否需要用户接管（不是判断任务是否完成）。
当且仅当当前截图/任务进度命中下方接管知识的必须接管范围，并且确实已经推进到合理人工边界时，判定需要接管（并给出类别与证据）。
接管范围、可选范围、过度接管和正确接管类型只能依据下方接管知识判断；不要引入知识外固定场景。
若当前页面仍可由 Agent 自动推进，或虽命中知识但任务关键条件尚未就位，应判不需要接管或 uncertain，并说明还缺什么。
接管必须有当前页面证据：当前截图应已经显示必须由用户处理的输入框、登录/验证码/实名、支付/删除/授权/协议确认、多个候选且无唯一依据的选择确认、明确错误/不可继续提示等人工边界。
不能因为预判点击某入口后未来可能需要手机号/地址/证件/支付/授权/确认，就在普通店铺页、详情页、列表页、入口页、设置页或功能首页提前接管。
若当前只看到“订座/排队/购买/去填写/下一步/确认信息”等可继续推进入口，且尚未进入实际填写、确认、授权、支付、验证或错误阻断页面，应先由 Agent 点击入口继续探索；此时以信息缺失、隐私输入、联系方式、用户确认等理由接管，应输出 handover=no 或 uncertain。
对看似收尾或确认性质的动作，必须判断它是否真的已经是最终人工边界：
- 如果历史轨迹或当前上下文显示点击后还会进入下一层选择、配置、填写或信息确认流程，则当前只是流程推进，应输出 handover=no 或 uncertain。
- 只有当前截图已展示完整待处理内容、后果/风险和最终确认性质，且下一步会真实产生状态变化、风险确认或最终结果时，才可输出 handover=yes。
{knowledge_block}

# Non-Handover Guidance（不要误判接管）
不在接管知识必须/允许范围内，或仍可自动推进的普通页面跳转、搜索与浏览、滚动查找、填写非敏感字段、选择一般筛选项、打开某个详情页、关闭弹窗、返回上一步、等待加载、重复尝试点击无响应按钮等，通常不需要接管。

# Output (必须是有效 JSON)
仅输出 JSON，不要输出多余文字：
{{
"handover": "yes" | "no" | "uncertain",
"handover_categories": ["security"|"missing_info"|"irreversible"|"human_verification"|"content_responsibility"|"legal_compliance"],
"evidence": [
"从当前截图/按钮文案/弹窗内容提取的关键证据（尽量引用可见文字）"
],
"recommended_user_action": "若 handover=yes，告诉用户需要做什么（如确认支付/输入验证码/选择账号/同意条款等）；否则填空字符串",
"next_step_if_no_handover": "若 handover=no，给出下一步自动化可执行动作（点击哪个、输入什么、滑到哪里）；否则填空字符串"
}}

# Final Check
你判定接管的依据必须能从"任务指令+轨迹+当前截图"中找到；不要凭空猜测。
如果看不清关键文字/被遮挡/无法确认是否付款或是否不可逆 → 输出 "uncertain" 并说明缺什么信息。

请直接输出JSON，不要添加其他说明文字。"""
    
    return prompt


def judge_takeover_for_step(
    instruction: str,
    history_context: str,
    current_caption: str,
    current_image_url: str,
    step_index: int,
    judge_takeover: str = "end-to-end",
    profile_id: Optional[str] = None,
) -> Dict:
    """
    为单个步骤判断是否需要接管
    
    Args:
        instruction: 任务指令
        history_context: 历史上下文（cap+action序列）
        current_caption: 当前步骤的 caption
        current_image_url: 当前步骤的图片 URL
        step_index: 步骤索引
        
    Returns:
        Dict: 判断结果，包含 handover、handover_categories、evidence 等字段，如果解析失败返回 {"handover": "uncertain", "prediction": -1}
    """
    task_id, _ = get_task_context()
    step_name = f"takeover_judgment_step_{step_index}_{judge_takeover}"
    if profile_id:
        step_name = f"{step_name}_{profile_id}"
    set_task_context(task_id, step_name)
    
    # 构建 prompt
    prompt = build_takeover_judgment_prompt(
        instruction,
        history_context,
        current_caption,
        takeover_knowledge_section=render_takeover_knowledge_section(mode=judge_takeover),
    )
    
    # 调用 LLM（传入当前图片，自动保存到 tmp2）
    images = [current_image_url] if current_image_url and current_image_url.strip() else None
    response = call_llm_with_prompt(
        prompt,
        images=images,
        temperature=0.1,
        max_tokens=800,
        task_id=task_id,
        step_name=step_name
    )
    
    if not response:
        return {"handover": "uncertain", "prediction": -1, "handover_categories": [], "evidence": []}
    
    # 解析 JSON 响应
    result = parse_json_response(response)
    if not result:
        return {"handover": "uncertain", "prediction": -1, "handover_categories": [], "evidence": []}
    
    # 提取字段
    handover = result.get("handover", "uncertain")
    handover_categories = result.get("handover_categories", [])
    evidence = result.get("evidence", [])
    recommended_user_action = result.get("recommended_user_action", "")
    next_step_if_no_handover = result.get("next_step_if_no_handover", "")
    
    # 转换预测值
    if handover == "yes":
        prediction = 1
    elif handover == "no":
        prediction = 0
    else:  # uncertain
        prediction = -1
    
    return {
        "handover": handover,
        "prediction": prediction,
        "handover_categories": handover_categories,
        "evidence": evidence,
        "recommended_user_action": recommended_user_action,
        "next_step_if_no_handover": next_step_if_no_handover,
        "full_response": result
    }
