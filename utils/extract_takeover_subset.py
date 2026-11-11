"""
从 JSONL 文件中提取包含接管类型的数据子集
每个接管类型只保留一个样本
"""
import json
import re
import sys
import os
from typing import Optional, Dict, List
from collections import defaultdict


def categorize_takeover_type(takeover_action: str) -> str:
    """
    将细分的接管类型归类到大的类别
    
    Args:
        takeover_action: 原始的接管 action 描述
        
    Returns:
        大类别的接管类型
    """
    if "信息缺失" in takeover_action:
        return "信息缺失"
    elif "支付" in takeover_action:
        return "支付场景"
    elif "下订单" in takeover_action:
        return "下订单"
    elif "删除" in takeover_action:
        return "删除"
    elif "权限管理" in takeover_action:
        return "权限管理"
    elif "消息发送" in takeover_action:
        return "消息发送"
    elif "同意协议" in takeover_action:
        return "同意协议"
    elif "人机验证" in takeover_action:
        return "人机验证"
    else:
        return "其他场景"


def extract_takeover_type(task_data: dict) -> Optional[str]:
    """
    从任务数据中提取接管类型
    优先从 query 格式（traj 中的 value.action）提取，如果没有则从 batchID 格式（history_image_pairs 中的 action_describe）提取
    
    Args:
        task_data: 任务数据字典
        
    Returns:
        标准化的接管类型字符串，如果没有则返回 None
    """
    # 优先处理 query 格式（traj），从 value.action 提取
    traj = task_data.get("traj", [])
    if traj:
        for step in traj:
            value = step.get("value", {})
            action = value.get("action", "")
            
            # 从 value.action 中提取接管类型
            if "接管" in action:
                # 直接返回 action 作为接管类型
                return action
    
    # 处理 batchID 格式（history_image_pairs），从 action_describe 提取（作为后备）
    history_image_pairs = task_data.get("history_image_pairs", [])
    if history_image_pairs:
        for pair in history_image_pairs:
            action_describe = pair.get("action_describe", "")
            if "接管" in action_describe:
                # 直接返回 action_describe 作为接管类型
                return action_describe
    
    return None


def filter_takeover_subset(input_file: str, output_file: str):
    """
    过滤并提取接管类型数据的子集
    
    Args:
        input_file: 输入 JSONL 文件路径
        output_file: 输出 JSONL 文件路径
    """
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return
    
    total_count = 0
    takeover_count = 0
    takeover_type_count = defaultdict(int)  # 细分类别统计
    takeover_category_count = defaultdict(int)  # 大类别统计
    takeover_category_seen = {}  # 记录每个大类别是否已经保留了一个样本
    takeover_category_detail = defaultdict(list)  # 记录每个大类别下的细分类别
    output_data = []
    
    print(f"正在处理文件: {input_file}")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            try:
                task_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：跳过无效的 JSON 行 {total_count}: {e}")
                continue
            
            # 提取接管类型（细分类别）
            takeover_type = extract_takeover_type(task_data)
            
            if takeover_type:
                takeover_count += 1
                takeover_type_count[takeover_type] += 1
                
                # 归类到大类别
                category = categorize_takeover_type(takeover_type)
                takeover_category_count[category] += 1
                takeover_category_detail[category].append(takeover_type)
                
                # 如果该大类别还没有保留样本，则保留
                if category not in takeover_category_seen:
                    takeover_category_seen[category] = True
                    output_data.append(task_data)
    
    # 写入输出文件
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for task_data in output_data:
            f.write(json.dumps(task_data, ensure_ascii=False) + '\n')
    
    # 输出统计信息
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"总数据量: {total_count}")
    print(f"包含接管的数据量: {takeover_count}")
    print()
    print("大类别统计（按大类去重）:")
    for category, count in sorted(takeover_category_count.items()):
        print(f"  {category}: {count} (输出样本数: 1)")
    print()
    print("细分类别统计（仅供参考）:")
    for takeover_type, count in sorted(takeover_type_count.items()):
        print(f"  {takeover_type}: {count}")
    print()
    print(f"最终输出数据量: {len(output_data)} (每个大类别保留1个样本)")
    print(f"已保存到: {output_file}")
    print("="*60)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        # 默认使用指定的文件
        input_file = "data/query_20260212_orm_testset.jsonl"
        output_file = "data/query_20260212_orm_testset_takeover_subset.jsonl"
        print(f"使用默认文件:")
        print(f"  输入: {input_file}")
        print(f"  输出: {output_file}")
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        # 自动生成输出文件名
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_takeover_subset.jsonl"
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
    else:
        print("用法: python extract_takeover_subset.py [输入文件] [输出文件]")
        print("  如果不提供参数，使用默认文件")
        print("  如果只提供输入文件，自动生成输出文件名")
        return
    
    filter_takeover_subset(input_file, output_file)


if __name__ == "__main__":
    main()
