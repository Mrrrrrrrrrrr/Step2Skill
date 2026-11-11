"""
为 JSONL 文件中的每个任务的每一步添加接管标签字段
为每个步骤添加 need_to_takeover_gt 字段（当前全部设为 0）
"""
import json
import sys
import os
from typing import Dict, List


def add_takeover_labels(input_file: str, output_file: str):
    """
    为 JSONL 文件中的每个任务的每一步添加 need_to_takeover_gt 字段
    
    Args:
        input_file: 输入 JSONL 文件路径
        output_file: 输出 JSONL 文件路径
    """
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return
    
    total_tasks = 0
    total_steps = 0
    labeled_steps = 0
    output_data = []
    
    print(f"正在处理文件: {input_file}")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total_tasks += 1
            try:
                task_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：跳过无效的 JSON 行 {total_tasks}: {e}")
                continue
            
            # 处理 traj 数组中的每个步骤
            traj = task_data.get("traj", [])
            if not traj:
                print(f"警告：任务 {task_data.get('task_id', 'unknown')} 没有 traj 字段或 traj 为空")
                output_data.append(task_data)
                continue
            
            # 为每个步骤添加 need_to_takeover_gt 字段
            for step in traj:
                total_steps += 1
                # 如果字段已存在，跳过；否则添加
                if "need_to_takeover_gt" not in step:
                    step["need_to_takeover_gt"] = 0
                    labeled_steps += 1
                else:
                    # 如果已存在，确保值为 0（根据用户要求）
                    step["need_to_takeover_gt"] = 0
            
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
    print(f"总任务数: {total_tasks}")
    print(f"总步骤数: {total_steps}")
    print(f"已添加标签的步骤数: {labeled_steps}")
    print(f"已保存到: {output_file}")
    print("="*60)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        # 默认使用指定的文件
        input_file = "data/query_20260212_orm_testset_takeover_subset.jsonl"
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_labeled.jsonl"
        print(f"使用默认文件:")
        print(f"  输入: {input_file}")
        print(f"  输出: {output_file}")
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        # 自动生成输出文件名
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_labeled.jsonl"
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
    else:
        print("用法: python label_take_over.py [输入文件] [输出文件]")
        print("  如果不提供参数，使用默认文件")
        print("  如果只提供输入文件，自动生成输出文件名")
        return
    
    add_takeover_labels(input_file, output_file)


if __name__ == "__main__":
    main()
