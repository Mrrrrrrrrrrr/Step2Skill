"""
从summary文件中提取判断错误的case，生成badcase JSONL文件
"""
import json
import os
import sys
import argparse
from typing import List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_summary(summary_path: str) -> List[dict]:
    """加载summary JSON文件"""
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("summary", [])


def find_badcase_task_ids(summary_data: List[dict]) -> Set[str]:
    """找出判断错误的task_id（qa_correct、keyframe_correct、final_correct任意一个为false）"""
    badcase_ids = set()
    
    for item in summary_data:
        task_id = item.get("task_id", "")
        qa_correct = item.get("qa_correct")
        keyframe_correct = item.get("keyframe_correct")
        final_correct = item.get("final_correct")
        
        if qa_correct is False or keyframe_correct is False or final_correct is False:
            badcase_ids.add(task_id)
    
    return badcase_ids


def filter_jsonl_by_task_ids(jsonl_path: str, task_ids: Set[str], output_path: str) -> int:
    """从JSONL文件中过滤出指定task_id的记录"""
    count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                task = json.loads(line)
                task_id = task.get("task_id", "")
                
                if task_id in task_ids:
                    f_out.write(line + '\n')
                    count += 1
            except json.JSONDecodeError as e:
                print(f"警告: 解析JSON失败: {e}")
                continue
    
    return count


def main():
    parser = argparse.ArgumentParser(description="从summary中提取判断错误的case")
    parser.add_argument("--summary", type=str, required=True,
                        help="summary JSON文件路径")
    parser.add_argument("--query", type=str, required=True,
                        help="原始query JSONL文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出badcase JSONL文件路径（默认：query文件同目录下的query_XXX_badcase.jsonl）")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("提取Badcase工具")
    print("=" * 60)
    
    print(f"\n步骤1: 加载summary文件...")
    print(f"  文件: {args.summary}")
    summary_data = load_summary(args.summary)
    print(f"  加载了 {len(summary_data)} 个任务记录")
    
    print(f"\n步骤2: 查找判断错误的task_id...")
    badcase_ids = find_badcase_task_ids(summary_data)
    print(f"  找到 {len(badcase_ids)} 个badcase任务")
    
    if not badcase_ids:
        print("\n没有找到badcase，退出")
        return
    
    print(f"\n步骤3: 从query文件中过滤badcase...")
    print(f"  输入文件: {args.query}")
    
    if args.output:
        output_path = args.output
    else:
        base_dir = os.path.dirname(args.query)
        base_name = os.path.basename(args.query)
        if base_name.startswith("query_") and base_name.endswith(".jsonl"):
            prefix = base_name[6:-6]
            output_path = os.path.join(base_dir, f"query_{prefix}_badcase.jsonl")
        else:
            output_path = os.path.join(base_dir, f"{base_name}_badcase.jsonl")
    
    print(f"  输出文件: {output_path}")
    
    count = filter_jsonl_by_task_ids(args.query, badcase_ids, output_path)
    
    print(f"\n步骤4: 完成")
    print(f"  成功提取 {count} 个badcase记录")
    print(f"  输出文件: {output_path}")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
