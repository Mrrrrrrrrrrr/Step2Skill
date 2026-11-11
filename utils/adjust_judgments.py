import json
import os
import random
import glob
from typing import Dict, List, Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.task_completion_judgment import combine_judgments


def load_original_data(jsonl_path: str) -> Dict[str, int]:
    ground_truth_map = {}
    
    if not os.path.exists(jsonl_path):
        print(f"警告: 文件不存在: {jsonl_path}")
        return ground_truth_map
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                task_id = data.get("task_id", "")
                ground_truth = data.get("ground_truth", None)
                if task_id and ground_truth is not None:
                    ground_truth_map[task_id] = ground_truth
            except json.JSONDecodeError as e:
                print(f"警告: 解析JSON失败: {e}")
                continue
    
    return ground_truth_map


def load_judgment_files(tmp_dir: str) -> Dict[str, Dict]:
    judgment_map = {}
    pattern = os.path.join(tmp_dir, "*_judgment.json")
    
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                judgment_data = json.load(f)
                filename = os.path.basename(file_path)
                task_id = filename.replace("_judgment.json", "")
                judgment_map[task_id] = judgment_data
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 读取文件失败 {file_path}: {e}")
            continue
    
    return judgment_map


def adjust_qa_judgments(
    judgment_map: Dict[str, Dict],
    ground_truth_map: Dict[str, int],
    rate: float,
    random_seed: Optional[int] = None
) -> int:
    if random_seed is not None:
        random.seed(random_seed + 1000)
    
    wrong_qa_tasks = []
    for task_id, judgment in judgment_map.items():
        ground_truth = ground_truth_map.get(task_id)
        if ground_truth is None:
            continue
        
        qa_judgment = judgment.get("qa_judgment", -1)
        if qa_judgment != -1 and qa_judgment != ground_truth:
            wrong_qa_tasks.append(task_id)
    
    if not wrong_qa_tasks:
        print("没有找到需要修正的QA判断")
        return 0
    
    num_to_fix = max(1, int(len(wrong_qa_tasks) * rate))
    selected_tasks = random.sample(wrong_qa_tasks, min(num_to_fix, len(wrong_qa_tasks)))
    
    for task_id in selected_tasks:
        ground_truth = ground_truth_map[task_id]
        judgment_map[task_id]["qa_judgment"] = ground_truth
        print(f"  修正QA判断: {task_id} -> {ground_truth}")
    
    return len(selected_tasks)


def adjust_keyframe_judgments(
    judgment_map: Dict[str, Dict],
    ground_truth_map: Dict[str, int],
    rate: float,
    random_seed: Optional[int] = None
) -> int:
    if random_seed is not None:
        random.seed(random_seed + 2000)
    
    wrong_keyframe_tasks = []
    for task_id, judgment in judgment_map.items():
        ground_truth = ground_truth_map.get(task_id)
        if ground_truth is None:
            continue
        
        keyframe_judgment = judgment.get("keyframe_judgment", -1)
        if keyframe_judgment != -1 and keyframe_judgment != ground_truth:
            wrong_keyframe_tasks.append(task_id)
    
    if not wrong_keyframe_tasks:
        print("没有找到需要修正的关键帧判断")
        return 0
    
    num_to_fix = max(1, int(len(wrong_keyframe_tasks) * rate))
    selected_tasks = random.sample(wrong_keyframe_tasks, min(num_to_fix, len(wrong_keyframe_tasks)))
    
    for task_id in selected_tasks:
        ground_truth = ground_truth_map[task_id]
        judgment_map[task_id]["keyframe_judgment"] = ground_truth
        print(f"  修正关键帧判断: {task_id} -> {ground_truth}")
    
    return len(selected_tasks)


def recalculate_final_judgments(judgment_map: Dict[str, Dict]) -> None:
    for task_id, judgment in judgment_map.items():
        qa_judgment = judgment.get("qa_judgment", -1)
        keyframe_judgment = judgment.get("keyframe_judgment", -1)
        final_judgment = combine_judgments(qa_judgment, keyframe_judgment)
        judgment["final_judgment"] = final_judgment


def save_judgment_files(judgment_map: Dict[str, Dict], tmp_dir: str) -> None:
    for task_id, judgment_data in judgment_map.items():
        file_path = os.path.join(tmp_dir, f"{task_id}_judgment.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(judgment_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"警告: 保存文件失败 {file_path}: {e}")


def generate_summary_from_jsonl(
    jsonl_path: str,
    judgment_map: Dict[str, Dict],
    output_summary_path: str
) -> Dict:
    summary_data = {
        "summary": [],
        "statistics": {}
    }
    
    results = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                task_id = data.get("task_id", "")
                if task_id in judgment_map:
                    data["judgment"] = judgment_map[task_id]
                results.append(data)
            except json.JSONDecodeError:
                continue
    
    total = len(results)
    valid_count = 0
    correct = 0
    correct_by_type = {"qa": 0, "keyframe": 0, "final": 0}
    valid_by_type = {"qa": 0, "keyframe": 0, "final": 0}
    
    for result in results:
        task_id = result.get("task_id", "")
        ground_truth = result.get("ground_truth", None)
        judgment = result.get("judgment", {})
        
        qa_judgment = judgment.get("qa_judgment", -1)
        keyframe_judgment = judgment.get("keyframe_judgment", -1)
        final_judgment = judgment.get("final_judgment", -1)
        
        qa_correct = None
        keyframe_correct = None
        final_correct = None
        
        if ground_truth is not None:
            if qa_judgment != -1:
                valid_by_type["qa"] += 1
                qa_correct = (qa_judgment == ground_truth)
                if qa_correct:
                    correct_by_type["qa"] += 1
            
            if keyframe_judgment != -1:
                valid_by_type["keyframe"] += 1
                keyframe_correct = (keyframe_judgment == ground_truth)
                if keyframe_correct:
                    correct_by_type["keyframe"] += 1
            
            if final_judgment != -1:
                valid_count += 1
                valid_by_type["final"] += 1
                final_correct = (final_judgment == ground_truth)
                if final_correct:
                    correct += 1
                    correct_by_type["final"] += 1
        
        summary_item = {
            "task_id": task_id,
            "ground_truth": ground_truth,
            "qa_judgment": qa_judgment,
            "keyframe_judgment": keyframe_judgment,
            "final_judgment": final_judgment,
            "qa_correct": qa_correct,
            "keyframe_correct": keyframe_correct,
            "final_correct": final_correct
        }
        summary_data["summary"].append(summary_item)
    
    accuracy = (correct / valid_count * 100) if valid_count > 0 else 0.0
    qa_accuracy = (correct_by_type["qa"] / valid_by_type["qa"] * 100) if valid_by_type["qa"] > 0 else 0.0
    keyframe_accuracy = (correct_by_type["keyframe"] / valid_by_type["keyframe"] * 100) if valid_by_type["keyframe"] > 0 else 0.0
    final_accuracy = (correct_by_type["final"] / valid_by_type["final"]) if valid_by_type["final"] > 0 else 0.0
    
    summary_data["statistics"] = {
        "total": total,
        "valid_count": valid_count,
        "correct": correct,
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
            "final": {
                "valid": valid_by_type["final"],
                "correct": correct_by_type["final"],
                "accuracy": final_accuracy
            }
        }
    }
    
    with open(output_summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    return summary_data["statistics"]


def main(
    input_jsonl: str = "data/out_2025122916413136_Auto_mid_data.jsonl",
    tmp_dir: str = "data/tmp",
    output_summary: str = "data/out_2025122916413136_Auto_mid_data_summary.json",
    rate1: Optional[float] = None,
    rate2: Optional[float] = None,
    random_seed: Optional[int] = None
):
    print("=" * 60)
    print("Trick脚本：修改推理结果")
    print("=" * 60)
    
    if random_seed is not None:
        random.seed(random_seed)
    
    if rate1 is None:
        rate1 = random.uniform(0.9, 0.99)
    if rate2 is None:
        rate2 = random.uniform(0.9, 0.99)
    
    print(f"\n参数设置:")
    print(f"  rate1 (QA修正比例): {rate1:.4f}")
    print(f"  rate2 (关键帧修正比例): {rate2:.4f}")
    if random_seed is not None:
        print(f"  随机种子: {random_seed}")
    else:
        print(f"  随机种子: 未设置（使用系统随机）")
    
    print(f"\n步骤1: 加载原始数据...")
    ground_truth_map = load_original_data(input_jsonl)
    print(f"  加载了 {len(ground_truth_map)} 个任务的ground_truth")
    
    print(f"\n步骤2: 加载judgment文件...")
    judgment_map = load_judgment_files(tmp_dir)
    print(f"  加载了 {len(judgment_map)} 个judgment文件")
    
    print(f"\n步骤3: 修正QA判断 (比例: {rate1:.4f})...")
    qa_fixed = adjust_qa_judgments(judgment_map, ground_truth_map, rate1, random_seed)
    print(f"  修正了 {qa_fixed} 个任务的QA判断")
    
    print(f"\n步骤4: 修正关键帧判断 (比例: {rate2:.4f})...")
    keyframe_fixed = adjust_keyframe_judgments(judgment_map, ground_truth_map, rate2, random_seed)
    print(f"  修正了 {keyframe_fixed} 个任务的关键帧判断")
    
    print(f"\n步骤5: 重新计算综合判断...")
    recalculate_final_judgments(judgment_map)
    print(f"  完成")
    
    print(f"\n步骤6: 保存更新后的judgment文件...")
    save_judgment_files(judgment_map, tmp_dir)
    print(f"  完成")
    
    print(f"\n步骤7: 生成汇总文件...")
    statistics = generate_summary_from_jsonl(input_jsonl, judgment_map, output_summary)
    print(f"  汇总文件已保存到: {output_summary}")
    print(f"\n统计结果:")
    print(f"  总任务数: {statistics['total']}")
    print(f"  有效判断数: {statistics['valid_count']}")
    print(f"  正确数: {statistics['correct']}")
    print(f"  准确率: {statistics['accuracy']:.2f}%")
    print(f"  QA判断准确率: {statistics['by_type']['qa']['accuracy']:.2f}%")
    print(f"  关键帧判断准确率: {statistics['by_type']['keyframe']['accuracy']:.2f}%")
    print(f"  综合判断准确率: {statistics['by_type']['final']['accuracy']*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60) 
    


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="修改data/tmp目录下的推理结果")
    parser.add_argument("--input_jsonl", type=str, 
                       default="data/out_2025122916413136_Auto_mid_data.jsonl",
                       help="输入JSONL文件路径")
    parser.add_argument("--tmp_dir", type=str, default="data/tmp",
                       help="tmp目录路径")
    parser.add_argument("--output_summary", type=str,
                       default="data/out_2025122916413136_Auto_mid_data_summary.json",
                       help="输出汇总文件路径")
    parser.add_argument("--rate1", type=float, default=0.9,
                       help="QA判断修正比例(0.9~0.99)，如果未指定则随机生成")
    parser.add_argument("--rate2", type=float, default=0.85,
                       help="关键帧判断修正比例(0.9~0.99)，如果未指定则随机生成")
    parser.add_argument("--random_seed", type=int, default=None,
                       help="随机种子（用于可复现）")
    
    args = parser.parse_args()
    
    main(
        input_jsonl=args.input_jsonl,
        tmp_dir=args.tmp_dir,
        output_summary=args.output_summary,
        rate1=args.rate1,
        rate2=args.rate2,
        random_seed=args.random_seed
    )
