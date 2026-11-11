"""
统计脚本
从 main.py 的输出文件读取数据，生成包含评估结果的 CSV 文件
"""
import os
import sys
import json
import csv
import glob
import re
import argparse
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_output_file_path(input_file: str, model_name: str) -> str:
    """
    根据输入文件和模型名称确定输出文件路径（复用 main.py 的逻辑）
    
    Args:
        input_file: 输入文件路径
        model_name: 模型名称
    
    Returns:
        str: 输出文件路径
    """
    model_data_dir = config.get_model_data_dir(model_name)
    base_name = os.path.basename(input_file)
    # 替换文件名中的query为output
    new_base_name = base_name.replace("query", "output", 1)  # 只替换第一个出现的query
    if new_base_name == base_name:  # 如果没有query，则在文件名前加output_
        name, ext = os.path.splitext(base_name)
        new_base_name = f"output_{name}{ext}"
    output_file = os.path.join(model_data_dir, new_base_name)
    return output_file


def load_validate_actions(task_id: str, model_name: str) -> str:
    """
     从 original_task_data.json 读取原始轨迹数据中的 action
    
    Args:
        task_id: 任务ID
        model_name: 模型名称
    
    Returns:
        str: 合并后的 action（换行符连接），如果没有则返回空字符串
    """
    try:
        save_dir = config.get_save_dir(model_name)
        task_dir = os.path.join(save_dir, task_id)
        original_task_file = os.path.join(task_dir, "original_task_data.json")
        
        if not os.path.exists(original_task_file):
            return ""
        
        with open(original_task_file, 'r', encoding='utf-8') as f:
            original_task = json.load(f)
            traj = original_task.get("traj", [])
            
            if not traj:
                return ""
            
            # 按 index 排序
            traj_sorted = sorted(traj, key=lambda x: x.get("index", -1))
            
            # 提取所有 action
            actions = []
            for step in traj_sorted:
                action = step.get("value", {}).get("action", "")
                if action and action.strip():
                    actions.append(action.strip())
            
            # 用换行符连接
            return "\n".join(actions) if actions else ""
            
    except Exception as e:
        print(f"  警告: 读取原始 action 失败 {task_id}: {e}")
        return ""


def get_keyframe_label(task_id: str, gt: int, model_name: str) -> str:
    """
    从 save 目录读取 keyframe_judgment 文件，确定 label
    
    Args:
        task_id: 任务ID
        gt: ground truth（0 或 1）
        model_name: 模型名称
    
    Returns:
        str: label（0 或 1），如果找不到则返回 "N/A"
    """
    try:
        save_dir = config.get_save_dir(model_name)
        task_dir = os.path.join(save_dir, task_id)
        
        if not os.path.exists(task_dir):
            return "N/A"
        
        # 查找所有 keyframe_judgment 文件
        pattern = os.path.join(task_dir, "keyframe_judgment_*_to_*.json")
        judgment_files = glob.glob(pattern)
        
        if not judgment_files:
            return "N/A"
        
        # 优先查找 0_0 和 1_1 的文件
        priority_files = []
        other_files = []
        
        for judgment_file in judgment_files:
            filename = os.path.basename(judgment_file)
            match = re.match(r'keyframe_judgment_(\d+)_to_(\d+)\.json', filename)
            if match:
                file_gt = int(match.group(1))
                file_pred = int(match.group(2))
                # 如果 gt == pred，这是优先文件
                if file_gt == file_pred:
                    priority_files.append((file_gt, file_pred, judgment_file))
                else:
                    other_files.append((file_gt, file_pred, judgment_file))
        
        # 如果找到优先文件，使用第一个（应该只有一个）
        if priority_files:
            file_gt, file_pred, _ = priority_files[0]
            return str(file_gt)
        
        # 如果没有优先文件，但有其他文件，使用第一个
        if other_files:
            file_gt, file_pred, _ = other_files[0]
            return str(file_pred)
        
        return "N/A"
        
    except Exception as e:
        print(f"  警告: 读取 keyframe_label 失败 {task_id}: {e}")
        return "N/A"


def get_num_tra(task_id: str, trajectory_length: Optional[int], model_name: str) -> int:
    """
    获取轨迹长度（图片个数）
    
    Args:
        task_id: 任务ID
        trajectory_length: 从输出文件获取的轨迹长度（如果可用）
        model_name: 模型名称
    
    Returns:
        int: 轨迹长度
    """
    # 如果输出文件中有 trajectory_length，直接使用
    if trajectory_length is not None:
        return trajectory_length
    
    # 否则从 original_task_data.json 读取
    try:
        save_dir = config.get_save_dir(model_name)
        task_dir = os.path.join(save_dir, task_id)
        original_task_file = os.path.join(task_dir, "original_task_data.json")
        
        if os.path.exists(original_task_file):
            with open(original_task_file, 'r', encoding='utf-8') as f:
                original_task = json.load(f)
                traj = original_task.get("traj", [])
                return len(traj)
    except Exception as e:
        print(f"  警告: 读取轨迹长度失败 {task_id}: {e}")
    
    return 0


def process_statistics(input_file: str, model_name: str, output_csv: str):
    """
    主处理函数：读取输出文件，收集数据，生成 CSV
    
    Args:
        input_file: 输入文件路径
        model_name: 模型名称
        output_csv: 输出 CSV 文件路径
    """
    # 确定输出文件路径
    output_file = get_output_file_path(input_file, model_name)
    
    if not os.path.exists(output_file):
        print(f"错误: 输出文件不存在: {output_file}")
        print(f"请确保已经运行过 main.py 处理输入文件: {input_file}")
        return
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"模型: {model_name}")
    print(f"CSV 输出: {output_csv}")
    print("=" * 60)
    
    # 读取输出文件
    results = []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line.strip())
                    results.append(result)
    except Exception as e:
        print(f"错误: 读取输出文件失败: {e}")
        return
    
    print(f"找到 {len(results)} 个任务结果")
    
    # 收集统计数据
    csv_rows = []
    for i, result in enumerate(results, 1):
        task_id = result.get("task_id", "")
        instruction = result.get("instruction", "")
        gt = result.get("ground_truth")
        trajectory_length = result.get("trajectory_length")
        
        if not task_id:
            print(f"  警告: 第 {i} 个结果缺少 task_id，跳过")
            continue
        
        print(f"处理任务 {i}/{len(results)}: {task_id}")
        
        # 收集各项数据
        action_prime = load_validate_actions(task_id, model_name)
        num_tra = get_num_tra(task_id, trajectory_length, model_name)
        label = get_keyframe_label(task_id, gt, model_name)
        
        # 构建 CSV 行
        csv_row = {
            "instruction": instruction,
            "task_id": task_id,
            "action_prime": action_prime,
            "num_tra": num_tra,
            "gt": gt if gt is not None else "N/A",
            "label": label
        }
        
        csv_rows.append(csv_row)
    
    # 写入 CSV 文件
    if csv_rows:
        fieldnames = ["instruction", "task_id", "action_prime", "num_tra", "gt", "label"]
        
        try:
            with open(output_csv, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            print(f"\n统计完成！共处理 {len(csv_rows)} 个任务")
            print(f"CSV 文件已保存到: {output_csv}")
        except Exception as e:
            print(f"错误: 保存 CSV 文件失败: {e}")
    else:
        print("警告: 没有数据可写入 CSV")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="统计脚本：从 main.py 输出文件生成评估结果 CSV")
    parser.add_argument("--input", type=str, default="data/query_20260127_testset.jsonl", 
                       help="输入文件路径（默认: data/query_20260127_testset.jsonl）")
    parser.add_argument("--model", type=str, default=None, 
                       help="模型名称（用于确定目录），如果不指定则使用 config.DEFAULT_MODEL")
    parser.add_argument("--output", type=str, default=None,
                       help="输出 CSV 文件路径，如果不指定则自动生成")
    
    args = parser.parse_args()
    
    # 确定模型名称
    if args.model is None:
        args.model = config.DEFAULT_MODEL
    
    # 确定输出 CSV 文件路径
    if args.output is None:
        # 基于输入文件路径生成
        base_name = os.path.basename(args.input)
        name, ext = os.path.splitext(base_name)
        csv_name = f"{name}_statistics.csv"
        model_data_dir = config.get_model_data_dir(args.model)
        args.output = os.path.join(model_data_dir, csv_name)
    
    # 执行统计
    process_statistics(args.input, args.model, args.output)


if __name__ == "__main__":
    main()
