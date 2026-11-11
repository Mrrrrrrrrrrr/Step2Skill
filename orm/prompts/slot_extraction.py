"""
槽位提取模块
从任务指令中提取槽位值，并生成QA对
"""
import json
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_utils import load_json_file, save_json_file, call_llm_with_prompt, parse_json_response
import config


def extract_slots(instruction: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    从任务指令中提取槽位值，并生成QA对
    
    Args:
        instruction: 任务指令
        
    Returns:
        tuple: (槽位字典, QA对列表)
    """
    # 加载历史槽位知识库
    knowledge_base = load_json_file(config.SLOT_KNOWLEDGE_BASE_PATH, {"slots": {}, "common_questions": []})
    
    # 构建prompt
    prompt = build_slot_extraction_prompt(instruction, knowledge_base)
    
    # 调用LLM
    response = call_llm_with_prompt(prompt, temperature=0.1, max_tokens=2000)
    
    if not response:
        return {}, []
    
    # 解析响应
    result = parse_json_response(response)
    if not result:
        print("无法解析槽位提取结果")
        return {}, []
    
    slots = result.get("slots", {})
    qa_pairs = result.get("qa_pairs", [])
    
    # 更新知识库
    update_knowledge_base(slots, qa_pairs, knowledge_base)
    
    return slots, qa_pairs


def build_slot_extraction_prompt(instruction: str, knowledge_base: Dict) -> str:
    """构建槽位提取prompt"""
    
    # 构建历史槽位信息
    historical_slots_text = ""
    if knowledge_base.get("slots"):
        historical_slots_text = "\n历史槽位示例：\n"
        for slot_name, slot_values in knowledge_base["slots"].items():
            if slot_values:
                # 确保所有值都是字符串，处理嵌套列表的情况
                str_values = []
                for val in slot_values[:5]:
                    if isinstance(val, list):
                        # 如果值是列表，展平并转换为字符串
                        str_values.append(', '.join(str(v) for v in val))
                    else:
                        str_values.append(str(val))
                historical_slots_text += f"- {slot_name}: {', '.join(str_values)}\n"
    
    common_questions_text = ""
    if knowledge_base.get("common_questions"):
        common_questions_text = "\n常见问题示例：\n"
        for qa in knowledge_base["common_questions"][:5]:
            common_questions_text += f"- Q: {qa.get('question', '')}\n  A: {qa.get('answer', '')}\n"
    
    prompt = f"""你是一个专业的任务分析专家。请从给定的任务指令中提取所有重要的槽位信息，并生成相应的QA对。

任务指令：
{instruction}
{historical_slots_text}
{common_questions_text}

请分析任务指令，提取以下信息：
1. 所有重要的槽位（如：酒店名称、日期、入住人、联系电话、支付方式、出发地、目的地、车次、座位类型等）
2. 为每个槽位生成一个QA对，问题应该针对该槽位，答案应该是从指令中提取的值

输出格式（必须是有效的JSON）：
{{
    "slots": {{
        "槽位名1": "槽位值1",
        "槽位名2": "槽位值2"
    }},
    "qa_pairs": [
        {{
            "question": "针对槽位1的问题",
            "answer": "槽位值1"
        }},
        {{
            "question": "针对槽位2的问题",
            "answer": "槽位值2"
        }}
    ]
}}

请直接输出JSON，不要添加其他说明文字。"""
    
    return prompt


def update_knowledge_base(slots: Dict[str, str], qa_pairs: List[Dict[str, str]], knowledge_base: Dict):
    """更新槽位知识库"""
    
    # 更新槽位值
    if "slots" not in knowledge_base:
        knowledge_base["slots"] = {}
    
    for slot_name, slot_value in slots.items():
        if slot_name not in knowledge_base["slots"]:
            knowledge_base["slots"][slot_name] = []
        
        # 如果值不在历史记录中，添加进去
        if slot_value not in knowledge_base["slots"][slot_name]:
            knowledge_base["slots"][slot_name].append(slot_value)
            # 保持最多20个历史值
            if len(knowledge_base["slots"][slot_name]) > 20:
                knowledge_base["slots"][slot_name] = knowledge_base["slots"][slot_name][-20:]
    
    # 更新常见问题
    if "common_questions" not in knowledge_base:
        knowledge_base["common_questions"] = []
    
    for qa in qa_pairs:
        # 检查是否已存在
        exists = False
        for existing_qa in knowledge_base["common_questions"]:
            if existing_qa.get("question") == qa.get("question"):
                exists = True
                break
        
        if not exists:
            knowledge_base["common_questions"].append(qa)
            # 保持最多50个常见问题
            if len(knowledge_base["common_questions"]) > 50:
                knowledge_base["common_questions"] = knowledge_base["common_questions"][-50:]
    
    # 保存知识库
    save_json_file(config.SLOT_KNOWLEDGE_BASE_PATH, knowledge_base)


if __name__ == "__main__":
    # 测试
    test_instruction = "帮我在酒店预订 App 上订成都顺安村居酒店的精致小单间房型，入住人是郑宝财和王子，联系电话是15827364950，用支付宝支付"
    slots, qa_pairs = extract_slots(test_instruction)
    print("槽位：", slots)
    print("QA对：", qa_pairs)
