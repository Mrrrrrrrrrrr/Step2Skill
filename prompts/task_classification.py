"""
任务分类模块
对任务进行初分类，用于选择对应的策略库
"""
from typing import Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_utils import call_llm_with_prompt, parse_json_response
from prompts.category_config import TASK_CATEGORIES, CATEGORY_DESCRIPTIONS, CATEGORY_DIFFERENCES
import config


def classify_task(instruction: str) -> str:
    """
    对任务进行分类
    
    Args:
        instruction: 任务指令
        
    Returns:
        str: 任务分类（分类键名，如 "hotel_booking"）
    """
    prompt = build_classification_prompt(instruction)
    
    response = call_llm_with_prompt(prompt, temperature=0.1, max_tokens=500)
    
    if not response:
        return "other"
    
    result = parse_json_response(response)
    if not result:
        # 尝试简单关键词匹配作为fallback
        return fallback_classification(instruction)
    
    category = result.get("category", "other")
    
    # 验证分类是否有效
    if category not in TASK_CATEGORIES:
        return "other"
    
    return category


def build_classification_prompt(instruction: str) -> str:
    """构建分类prompt"""
    
    categories_text = "\n".join([f"- {key}: {value}" for key, value in TASK_CATEGORIES.items()])
    
    # 从配置文件加载分类标准描述
    category_descriptions_text = "\n".join([f"- {key}：{desc}" for key, desc in CATEGORY_DESCRIPTIONS.items()])
    
    # 从配置文件加载分类区别说明
    differences_text = "\n".join([f"- {diff}" for diff in CATEGORY_DIFFERENCES])
    
    prompt = f"""你是一个任务分类专家。请根据任务指令的动作意图（动词）将其分类到以下类别之一。

任务指令：
{instruction}

可选分类：
{categories_text}

分类标准（基于动作意图，而非对象）：
{category_descriptions_text}

重要区别：
{differences_text}

请输出JSON格式：
{{
    "category": "分类键名",
    "reason": "分类理由（说明动作意图）"
}}

请直接输出JSON，不要添加其他说明文字。"""
    
    return prompt


def fallback_classification(instruction: str) -> str:
    """基于动作意图关键词的简单分类（fallback）"""
    # 简化fallback，直接返回other，让LLM进行分类
    return "other"


def get_category_name(category_key: str) -> str:
    """获取分类的中文名称"""
    return TASK_CATEGORIES.get(category_key, "其他类")


if __name__ == "__main__":
    # 测试
    test_instruction = "帮我在酒店预订 App 上订成都顺安村居酒店的精致小单间房型，入住人是郑宝财和王子，联系电话是15827364950，用支付宝支付"
    category = classify_task(test_instruction)
    print(f"分类：{category} ({get_category_name(category)})")
