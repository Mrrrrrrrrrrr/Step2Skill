"""
从 tmp、tmp2 和 query jsonl 文件重新生成 save 文件夹

用法：
    python utils/regenerate_save_from_cache.py <query_jsonl_file> --model <model_name>
    
示例：
    python utils/regenerate_save_from_cache.py data/query_20260212_orm_testset.jsonl --model gpt-4o
"""

import json
import os
import sys
import glob
import shutil
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_task_from_jsonl(file_path: str, task_id: Optional[str] = None) -> Optional[Dict]:
    """
    从JSONL文件加载任务
    
    Args:
        file_path: JSONL文件路径
        task_id: 任务ID（可选）
        
    Returns:
        Dict: 任务数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line.strip())
                if task_id is None or task.get("task_id") == task_id:
                    return task
        return None
    except Exception as e:
        print(f"加载任务失败：{e}")
        return None


def load_judgment_from_cache(task_id: str, model_name: str) -> Optional[Dict]:
    """
    从 tmp 目录加载 judgment 缓存
    
    Args:
        task_id: 任务ID
        model_name: 模型名称
        
    Returns:
        Dict: judgment 数据，如果不存在则返回 None
    """
    cache_dir = config.get_cache_dir(model_name)
    task_cache_dir = os.path.join(cache_dir, task_id)
    safe_task_id = task_id.replace('/', '_').replace('\\', '_').replace(':', '_')
    cache_file = os.path.join(task_cache_dir, f"{safe_task_id}_judgment.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  警告: 读取judgment缓存失败 {task_id}: {e}")
    return None


def get_prompt_response_files(task_id: str, model_name: str) -> Dict[str, str]:
    """
    从 tmp2 目录获取所有 prompt_response 文件路径
    
    Args:
        task_id: 任务ID
        model_name: 模型名称
        
    Returns:
        Dict: {step_name: file_path} 映射
    """
    tmp2_dir = config.get_tmp2_dir(model_name)
    task_tmp2_dir = os.path.join(tmp2_dir, task_id)
    safe_task_id = task_id.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    if not os.path.exists(task_tmp2_dir):
        return {}
    
    # 查找所有 prompt_response.json 文件
    pattern = os.path.join(task_tmp2_dir, f"{safe_task_id}_*_prompt_response.json")
    files = glob.glob(pattern)
    
    result = {}
    for file_path in files:
        basename = os.path.basename(file_path)
        # 提取 step_name: {safe_task_id}_{step_name}_prompt_response.json
        if f"{safe_task_id}_" in basename and "_prompt_response.json" in basename:
            step_name = basename.replace(f"{safe_task_id}_", "").replace("_prompt_response.json", "")
            result[step_name] = file_path
    
    return result


def regenerate_save_for_task(
    task_data: Dict,
    model_name: str,
    query_jsonl_path: str
) -> bool:
    """
    为单个任务重新生成 save 文件
    
    Args:
        task_data: 任务数据（从 query jsonl 读取）
        model_name: 模型名称
        query_jsonl_path: query jsonl 文件路径（用于保存原始任务数据）
        
    Returns:
        bool: 是否成功
    """
    task_id = task_data.get("task_id", "")
    if not task_id:
        return False
    
    try:
        # 从缓存加载 judgment
        judgment = load_judgment_from_cache(task_id, model_name)
        if judgment is None:
            print(f"  跳过 {task_id}: 未找到 judgment 缓存")
            return False
        
        # 获取 ground_truth
        response = task_data.get("response", None)
        if isinstance(response, dict):
            ground_truth = response.get("label", None)
            completion_ground_truth = response.get("is_task_completed", None)
            trajectory_ground_truth = response.get("is_trajactory_correct", None)
        else:
            ground_truth = response
            if response == 1:
                completion_ground_truth = 1
                trajectory_ground_truth = 1
            elif response == 0:
                completion_ground_truth = 0
                trajectory_ground_truth = 0
            else:
                completion_ground_truth = None
                trajectory_ground_truth = None
        
        # 获取轨迹数据
        trajectory = task_data.get("traj", [])
        qa_pairs = []  # 从 task_data 中获取，如果没有则使用空列表
        
        # 获取 prompt_response 文件
        prompt_response_files = get_prompt_response_files(task_id, model_name)
        
        # 创建 save 目录
        save_dir = os.path.join(config.get_save_dir(model_name), task_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # 复制图片文件
        for i, step in enumerate(trajectory):
            image_path = step.get("image", "")
            if image_path and os.path.exists(image_path):
                try:
                    image_ext = os.path.splitext(image_path)[1] or ".png"
                    dest_path = os.path.join(save_dir, f"image_{i}{image_ext}")
                    if not os.path.exists(dest_path):
                        shutil.copy2(image_path, dest_path)
                except Exception as e:
                    print(f"  警告: 复制图片失败 {image_path}: {e}")
        
        # 保存 QA judgment（如果有）
        qa_judgment = judgment.get("qa_judgment", -1)
        if qa_judgment != -1 and ground_truth is not None:
            qa_judgment_data = {
                "qa_pairs": [],
                "completion_question": None
            }
            
            # 从 prompt_response 文件读取 QA 数据
            qa_files = [f for name, f in prompt_response_files.items() if '_judgment_qa_q' in name]
            qa_files = sorted(qa_files, key=lambda x: int(os.path.basename(x).split('_judgment_qa_q')[1].split('_')[0]) if '_judgment_qa_q' in os.path.basename(x) else 0)
            
            for qa_file in qa_files:
                try:
                    with open(qa_file, 'r', encoding='utf-8') as f:
                        qa_data = json.load(f)
                    
                    prompt = qa_data.get("prompt", "")
                    response_text = qa_data.get("response_text", "")
                    prompt_lines = prompt.split('\n') if prompt else []
                    
                    qa_index_str = os.path.basename(qa_file).split('_judgment_qa_q')[1].split('_')[0]
                    qa_index = int(qa_index_str) if qa_index_str.isdigit() else -1
                    
                    if qa_index >= 0 and qa_index < len(qa_pairs):
                        qa_item = qa_pairs[qa_index]
                        qa_judgment_data["qa_pairs"].append({
                            "question": qa_item.get("question", ""),
                            "true_answer": qa_item.get("answer", ""),
                            "prompt_lines": prompt_lines,
                            "predicted_answer": response_text.strip()
                        })
                except Exception as e:
                    print(f"  警告: 读取QA文件失败 {qa_file}: {e}")
            
            ground_truth_str = str(ground_truth) if ground_truth is not None else "None"
            qa_judgment_str = str(qa_judgment) if qa_judgment != -1 else "-1"
            qa_judgment_filename = f"qa_judgment_{ground_truth_str}_to_{qa_judgment_str}.json"
            qa_judgment_path = os.path.join(save_dir, qa_judgment_filename)
            with open(qa_judgment_path, 'w', encoding='utf-8') as f:
                json.dump(qa_judgment_data, f, ensure_ascii=False, indent=2)
        
        # 保存 keyframe judgment（如果有）
        keyframe_judgment = judgment.get("keyframe_judgment", -1)
        if keyframe_judgment != -1 and ground_truth is not None:
            keyframe_judgment_data = {
                "prompt_lines": [],
                "response_lines": [],
                "completion_status": keyframe_judgment
            }
            
            keyframe_file = prompt_response_files.get("judgment_keyframe", None)
            if keyframe_file and os.path.exists(keyframe_file):
                try:
                    with open(keyframe_file, 'r', encoding='utf-8') as f:
                        keyframe_data = json.load(f)
                    
                    prompt = keyframe_data.get("prompt", "")
                    response_text = keyframe_data.get("response_text", "")
                    
                    keyframe_judgment_data["prompt_lines"] = prompt.split('\n') if prompt else []
                    keyframe_judgment_data["response_lines"] = response_text.split('\n') if response_text else []
                    
                    try:
                        parsed_response = json.loads(response_text)
                        if isinstance(parsed_response, dict):
                            keyframe_judgment_data["parsed_response"] = parsed_response
                    except:
                        pass
                except Exception as e:
                    print(f"  警告: 读取关键帧文件失败 {keyframe_file}: {e}")
            
            ground_truth_str = str(ground_truth) if ground_truth is not None else "None"
            keyframe_judgment_str = str(keyframe_judgment) if keyframe_judgment != -1 else "-1"
            keyframe_judgment_filename = f"keyframe_judgment_{ground_truth_str}_to_{keyframe_judgment_str}.json"
            keyframe_judgment_path = os.path.join(save_dir, keyframe_judgment_filename)
            with open(keyframe_judgment_path, 'w', encoding='utf-8') as f:
                json.dump(keyframe_judgment_data, f, ensure_ascii=False, indent=2)
        
        # 保存 completion judgment（如果有）
        completion_judgment = judgment.get("completion_judgment", -1)
        if completion_judgment != -1:
            completion_judgment_data = {
                "prompt_lines": [],
                "response_lines": [],
                "completion_status": completion_judgment
            }
            
            completion_file = prompt_response_files.get("judgment_completion", None)
            if completion_file and os.path.exists(completion_file):
                try:
                    with open(completion_file, 'r', encoding='utf-8') as f:
                        completion_data = json.load(f)
                    
                    prompt = completion_data.get("prompt", "")
                    response_text = completion_data.get("response_text", "")
                    
                    completion_judgment_data["prompt_lines"] = prompt.split('\n') if prompt else []
                    completion_judgment_data["response_lines"] = response_text.split('\n') if response_text else []
                    
                    try:
                        parsed_response = json.loads(response_text)
                        if isinstance(parsed_response, dict):
                            completion_judgment_data["parsed_response"] = parsed_response
                    except:
                        pass
                except Exception as e:
                    print(f"  警告: 读取完成度判断文件失败 {completion_file}: {e}")
            
            completion_judgment_str = str(completion_judgment) if completion_judgment != -1 else "-1"
            completion_gt = completion_ground_truth if completion_ground_truth is not None else ground_truth
            completion_gt_str = str(completion_gt) if completion_gt is not None else "None"
            
            # 只在 ground_truth 不为 None 时保存
            if completion_gt is not None:
                completion_judgment_filename = f"completion_judgment_{completion_gt_str}_to_{completion_judgment_str}.json"
                completion_judgment_path = os.path.join(save_dir, completion_judgment_filename)
                with open(completion_judgment_path, 'w', encoding='utf-8') as f:
                    json.dump(completion_judgment_data, f, ensure_ascii=False, indent=2)
        
        # 保存 trajectory_correctness judgment（如果有）
        trajectory_correctness_judgment = judgment.get("trajectory_correctness_judgment", -1)
        if trajectory_correctness_judgment != -1:
            trajectory_correctness_judgment_data = {
                "prompt_lines": [],
                "response_lines": [],
                "correctness_status": trajectory_correctness_judgment
            }
            
            trajectory_correctness_file = prompt_response_files.get("judgment_trajectory_correctness", None)
            if trajectory_correctness_file and os.path.exists(trajectory_correctness_file):
                try:
                    with open(trajectory_correctness_file, 'r', encoding='utf-8') as f:
                        trajectory_correctness_data = json.load(f)
                    
                    prompt = trajectory_correctness_data.get("prompt", "")
                    response_text = trajectory_correctness_data.get("response_text", "")
                    
                    trajectory_correctness_judgment_data["prompt_lines"] = prompt.split('\n') if prompt else []
                    trajectory_correctness_judgment_data["response_lines"] = response_text.split('\n') if response_text else []
                    
                    try:
                        parsed_response = json.loads(response_text)
                        if isinstance(parsed_response, dict):
                            trajectory_correctness_judgment_data["parsed_response"] = parsed_response
                    except:
                        pass
                except Exception as e:
                    print(f"  警告: 读取轨迹正确性判断文件失败 {trajectory_correctness_file}: {e}")
            
            trajectory_correctness_judgment_str = str(trajectory_correctness_judgment) if trajectory_correctness_judgment != -1 else "-1"
            trajectory_gt = trajectory_ground_truth if trajectory_ground_truth is not None else ground_truth
            trajectory_gt_str = str(trajectory_gt) if trajectory_gt is not None else "None"
            
            # 只在 ground_truth 不为 None 时保存
            if trajectory_gt is not None:
                trajectory_correctness_judgment_filename = f"trajectory_correctness_judgment_{trajectory_gt_str}_to_{trajectory_correctness_judgment_str}.json"
                trajectory_correctness_judgment_path = os.path.join(save_dir, trajectory_correctness_judgment_filename)
                with open(trajectory_correctness_judgment_path, 'w', encoding='utf-8') as f:
                    json.dump(trajectory_correctness_judgment_data, f, ensure_ascii=False, indent=2)
        
        # 保存原始任务数据
        original_task_filename = "original_task_data.json"
        original_task_path = os.path.join(save_dir, original_task_filename)
        with open(original_task_path, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"  错误: 重新生成 save 文件失败 {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法：")
        print("  python utils/regenerate_save_from_cache.py <query_jsonl_file> --model <model_name>")
        print()
        print("示例：")
        print("  python utils/regenerate_save_from_cache.py data/query_20260212_orm_testset.jsonl --model gpt-4o")
        return
    
    query_jsonl_path = sys.argv[1]
    
    # 解析模型名称
    model_name = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_name = sys.argv[idx + 1]
    
    if model_name is None:
        model_name = config.DEFAULT_MODEL
    
    if not os.path.exists(query_jsonl_path):
        print(f"错误：查询文件不存在: {query_jsonl_path}")
        return
    
    print(f"查询文件: {query_jsonl_path}")
    print(f"模型名称: {model_name}")
    print()
    
    # 读取所有任务
    tasks = []
    with open(query_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                tasks.append(task)
            except Exception as e:
                print(f"警告: 解析任务失败: {e}")
                continue
    
    print(f"共找到 {len(tasks)} 个任务")
    print()
    
    # 重新生成 save 文件
    success_count = 0
    skip_count = 0
    
    for i, task in enumerate(tasks, 1):
        task_id = task.get("task_id", "")
        if not task_id:
            continue
        
        print(f"处理任务 {i}/{len(tasks)}: {task_id}")
        
        if regenerate_save_for_task(task, model_name, query_jsonl_path):
            success_count += 1
        else:
            skip_count += 1
    
    print()
    print("=" * 60)
    print(f"处理完成！")
    print(f"成功: {success_count} 个")
    print(f"跳过: {skip_count} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()
