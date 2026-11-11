"""
自动化 Prompt 工程改进脚本
分析 keyframe judgment 的错误案例，总结改进知识，并更新 prompt
"""
import os
import sys
import json
import glob
import re
import argparse
import shutil
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.ask_qwen import call_honor_cloud_api, call_vllm_api
from prompts.prompt_improvement import build_error_analysis_prompt, build_merge_improvements_prompt


def collect_error_cases(data_dir: str) -> List[Dict]:
    """
    收集所有错误判断案例
    
    Args:
        data_dir: 数据目录路径（如 data/gpt-4o/save）
    
    Returns:
        List[Dict]: 错误案例列表
    """
    error_cases = []
    
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")
        return error_cases
    
    # 遍历所有任务子目录
    task_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for task_id in task_dirs:
        task_dir = os.path.join(data_dir, task_id)
        
        # 查找 keyframe_judgment_X_to_Y.json 文件
        pattern = os.path.join(task_dir, "keyframe_judgment_*_to_*.json")
        judgment_files = glob.glob(pattern)
        
        for judgment_file in judgment_files:
            # 从文件名提取 ground_truth 和 predicted
            filename = os.path.basename(judgment_file)
            match = re.match(r'keyframe_judgment_(\d+)_to_(\d+)\.json', filename)
            if not match:
                continue
            
            ground_truth = int(match.group(1))
            predicted = int(match.group(2))
            
            # 只收集错误案例（X != Y）
            if ground_truth == predicted:
                continue
            
            try:
                # 读取 JSON 文件
                with open(judgment_file, 'r', encoding='utf-8') as f:
                    judgment_data = json.load(f)
                
                # 提取信息
                parsed_response = judgment_data.get("parsed_response", {})
                prompt_lines = judgment_data.get("prompt_lines", [])
                
                # 提取任务指令（从 prompt_lines 中）
                instruction = ""
                for i, line in enumerate(prompt_lines):
                    if "任务指令：" in line and i + 1 < len(prompt_lines):
                        instruction = prompt_lines[i + 1].strip()
                        break
                
                # 尝试从 original_task_data.json 读取完整任务信息
                original_task_file = os.path.join(task_dir, "original_task_data.json")
                if os.path.exists(original_task_file):
                    try:
                        with open(original_task_file, 'r', encoding='utf-8') as f:
                            original_task = json.load(f)
                            if not instruction:
                                instruction = original_task.get("instruction", "")
                    except:
                        pass
                
                error_case = {
                    "task_id": task_id,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "instruction": instruction,
                    "reason": parsed_response.get("reason", ""),
                    "key_evidence": parsed_response.get("key_evidence", []),
                    "missing_steps": parsed_response.get("missing_steps", []),
                    "completion_status": parsed_response.get("completion_status", -1),
                    "prompt_lines": prompt_lines,
                    "judgment_file": judgment_file
                }
                
                error_cases.append(error_case)
                
            except Exception as e:
                print(f"警告: 读取文件失败 {judgment_file}: {e}")
                continue
    
    return error_cases


def analyze_error_patterns(error_cases: List[Dict]) -> Dict:
    """
    分析错误模式
    
    Args:
        error_cases: 错误案例列表
    
    Returns:
        Dict: 错误模式分析结果
    """
    if not error_cases:
        return {
            "total_errors": 0,
            "error_types": {},
            "common_reasons": [],
            "summary": "没有找到错误案例"
        }
    
    # 按错误类型分组
    error_types = {
        "0_to_1": [],  # 应该是0但预测为1（假阳性）
        "1_to_0": []   # 应该是1但预测为0（假阴性）
    }
    
    for case in error_cases:
        gt = case["ground_truth"]
        pred = case["predicted"]
        if gt == 0 and pred == 1:
            error_types["0_to_1"].append(case)
        elif gt == 1 and pred == 0:
            error_types["1_to_0"].append(case)
    
    # 统计常见错误原因
    all_reasons = []
    for case in error_cases:
        reason = case.get("reason", "")
        if reason:
            all_reasons.append(reason)
    
    # 提取关键证据和缺失步骤
    all_key_evidence = []
    all_missing_steps = []
    for case in error_cases:
        all_key_evidence.extend(case.get("key_evidence", []))
        all_missing_steps.extend(case.get("missing_steps", []))
    
    return {
        "total_errors": len(error_cases),
        "error_types": {
            "0_to_1": len(error_types["0_to_1"]),
            "1_to_0": len(error_types["1_to_0"])
        },
        "error_cases_by_type": error_types,
        "all_reasons": all_reasons,
        "all_key_evidence": all_key_evidence,
        "all_missing_steps": all_missing_steps,
        "sample_cases": error_cases[:10]  # 前10个案例作为样本
    }


def summarize_improvements_in_batches(
    error_cases: List[Dict], 
    model_name: str = "gpt-4o",
    cache_dir: str = None
) -> str:
    """
    分批生成改进知识并缓存（每5个案例一组）
    
    Args:
        error_cases: 所有错误案例列表
        model_name: 模型名称（用于选择 API）
        cache_dir: 缓存目录路径
    
    Returns:
        str: 缓存目录路径
    """
    if not error_cases:
        return cache_dir if cache_dir else ""
    
    # 确保缓存目录存在
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    else:
        return ""
    
    # 按5个一组分组
    batch_size = 5
    total_batches = (len(error_cases) + batch_size - 1) // batch_size
    
    print(f"  将 {len(error_cases)} 个错误案例分为 {total_batches} 组（每组 {batch_size} 个）")
    
    for group_id in range(total_batches):
        start_idx = group_id * batch_size
        end_idx = min(start_idx + batch_size, len(error_cases))
        group_cases = error_cases[start_idx:end_idx]
        
        # 检查缓存文件是否存在且有效
        cache_file = os.path.join(cache_dir, f"group_{group_id}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # 检查缓存是否有效（有改进知识且没有错误标记）
                    if cache_data.get("improvements") and not cache_data.get("error", False):
                        print(f"  跳过第 {group_id + 1}/{total_batches} 组（案例 {start_idx}-{end_idx-1}），使用已有缓存...")
                        continue
            except:
                # 如果读取失败，继续生成新的
                pass
        
        print(f"  处理第 {group_id + 1}/{total_batches} 组（案例 {start_idx}-{end_idx-1}）...")
        
        # 构建总结 prompt
        error_summary = f"""
## 错误案例统计（本组）
- 本组案例数: {len(group_cases)}
- 总错误数: {len(error_cases)}

## 错误案例样本（本组）

"""
        
        # 添加本组错误案例
        for i, case in enumerate(group_cases, 1):
            error_summary += f"""
### 错误案例 {i}
- 任务ID: {case['task_id']}
- 任务指令: {case['instruction']}
- 真实值: {case['ground_truth']}, 预测值: {case['predicted']}
- 错误原因: {case['reason']}
- 关键证据: {', '.join(case['key_evidence']) if case['key_evidence'] else '无'}
- 缺失步骤: {', '.join(case['missing_steps']) if case['missing_steps'] else '无'}

"""
        
        prompt = build_error_analysis_prompt(error_summary, "keyframe")
        
        # 调用 LLM API
        try:
            # 判断使用哪个 API
            mapped_name = config.get_model_name(model_name)
            use_vllm = (mapped_name == '')
            
            if use_vllm:
                response = call_vllm_api(prompt, temperature=0.3, max_tokens=2000)
                if response and 'choices' in response and len(response['choices']) > 0:
                    improvements = response['choices'][0]['message']['content']
                else:
                    improvements = "LLM 调用失败，无法生成改进知识。"
            else:
                improvements = call_honor_cloud_api(
                    query=prompt,
                    imgs=None,
                    model_name=model_name,
                    temperature=0.3,
                    max_tokens=2000
                )
                if not improvements:
                    improvements = "LLM 调用失败，无法生成改进知识。"
            
            # 保存到缓存文件
            cache_data = {
                "group_id": group_id,
                "case_indices": list(range(start_idx, end_idx)),
                "improvements": improvements,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            cache_file = os.path.join(cache_dir, f"group_{group_id}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            print(f"    已保存到: {cache_file}")
            
        except Exception as e:
            print(f"    错误: 第 {group_id + 1} 组 LLM 调用失败: {e}")
            # 保存错误信息
            cache_data = {
                "group_id": group_id,
                "case_indices": list(range(start_idx, end_idx)),
                "improvements": f"LLM 调用失败: {e}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": True
            }
            cache_file = os.path.join(cache_dir, f"group_{group_id}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    return cache_dir


def merge_improvements_batch(
    improvements_list: List[str],
    level: int,
    batch_id: int,
    cache_dir: str,
    model_name: str
) -> str:
    """
    合并一批改进知识（最多5个）
    
    Args:
        improvements_list: 要合并的改进知识列表
        level: 当前层级（0=原始, 1=第1次合并, ...）
        batch_id: 批次ID
        cache_dir: 缓存目录
        model_name: 模型名称
    
    Returns:
        str: 合并后的改进知识
    """
    # 构建合并prompt
    aggregated_content = ""
    for i, improvements in enumerate(improvements_list, 1):
        aggregated_content += f"""
## 改进知识 {i}

{improvements}

"""
    
    prompt = build_merge_improvements_prompt(aggregated_content)
    
    # 调用 LLM API
    try:
        mapped_name = config.get_model_name(model_name) 
        use_vllm = (mapped_name == '')
        
        if use_vllm:
            response = call_vllm_api(prompt, temperature=0.3, max_tokens=2000)
            if response and 'choices' in response and len(response['choices']) > 0:
                merged_improvements = response['choices'][0]['message']['content']
            else:
                merged_improvements = "LLM 合并调用失败，无法生成合并知识。"
        else:
            merged_improvements = call_honor_cloud_api(
                query=prompt,
                imgs=None,
                model_name=model_name,
                temperature=0.3,
                max_tokens=2000
            )
            if not merged_improvements:
                merged_improvements = "LLM 合并调用失败，无法生成合并知识。"
        
        # 保存到缓存
        cache_file = os.path.join(cache_dir, f"level_{level}_{batch_id}.json")
        cache_data = {
            "level": level,
            "batch_id": batch_id,
            "source_count": len(improvements_list),
            "improvements": merged_improvements,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"      已保存到: {cache_file}")
        return merged_improvements
        
    except Exception as e:
        print(f"      错误: LLM 合并调用失败: {e}")
        # 保存错误信息
        cache_file = os.path.join(cache_dir, f"level_{level}_{batch_id}.json")
        cache_data = {
            "level": level,
            "batch_id": batch_id,
            "source_count": len(improvements_list),
            "improvements": f"LLM 合并调用失败: {e}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": True
        }
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        return f"LLM 合并调用失败: {e}"


def aggregate_improvements(
    cache_dir: str,
    model_name: str = "gpt-4o",
    batch_size: int = 5
) -> str:
    """
    递归汇总改进知识：每5个知识合并一次，直到只剩1个最终知识
    
    流程：
    - Level 0: group_0.json, group_1.json, ... (原始改进知识)
    - Level 1: level_1_0.json, level_1_1.json, ... (每5个Level 0合并)
    - Level 2: level_2_0.json, level_2_1.json, ... (每5个Level 1合并)
    - ... 递归直到只剩1个
    
    Args:
        cache_dir: 缓存目录路径
        model_name: 模型名称
        batch_size: 每批合并的知识数量（默认5个）
    
    Returns:
        str: 最终汇总的改进知识
    """
    if not cache_dir or not os.path.exists(cache_dir):
        return "缓存目录不存在或为空。"
    
    # 读取Level 0（原始改进知识）
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "group_*.json")))
    if not cache_files:
        return "没有找到缓存文件。"
    
    print(f"  找到 {len(cache_files)} 个原始改进知识，开始递归合并...")
    
    # 收集Level 0的改进知识
    level_0_improvements = []
    for cache_file in cache_files:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                improvements = cache_data.get("improvements", "")
                if improvements and not cache_data.get("error", False):
                    level_0_improvements.append(improvements)
        except Exception as e:
            print(f"  警告: 读取缓存文件失败 {cache_file}: {e}")
            continue
    
    if not level_0_improvements:
        return "所有缓存文件都无效或为空。"
    
    # 递归合并
    current_level = 0
    current_improvements = level_0_improvements
    
    while len(current_improvements) > 1:
        print(f"\n  Level {current_level} → Level {current_level + 1}: 将 {len(current_improvements)} 个知识合并...")
        
        # 按batch_size分组
        next_level_improvements = []
        total_batches = (len(current_improvements) + batch_size - 1) // batch_size
        
        for batch_id in range(total_batches):
            start_idx = batch_id * batch_size
            end_idx = min(start_idx + batch_size, len(current_improvements))
            batch_improvements = current_improvements[start_idx:end_idx]
            
            # 检查缓存
            cache_file = os.path.join(cache_dir, f"level_{current_level + 1}_{batch_id}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                        merged = cache_data.get("improvements", "")
                        if merged and not cache_data.get("error", False):
                            print(f"    跳过 Level {current_level + 1} 批次 {batch_id + 1}/{total_batches}，使用已有缓存...")
                            next_level_improvements.append(merged)
                            continue
                except:
                    pass
            
            # 合并这批知识
            print(f"    合并 Level {current_level + 1} 批次 {batch_id + 1}/{total_batches}（{len(batch_improvements)} 个知识）...")
            merged = merge_improvements_batch(
                batch_improvements, 
                current_level + 1, 
                batch_id, 
                cache_dir, 
                model_name
            )
            
            if merged and not merged.startswith("LLM"):
                next_level_improvements.append(merged)
        
        current_improvements = next_level_improvements
        current_level += 1
    
    # 返回最终结果
    if len(current_improvements) == 1:
        print(f"\n  递归合并完成！最终知识来自 Level {current_level}")
        final_improvements_text = current_improvements[0]
        
        # 保存最终知识为标准格式文件，便于 main.py 读取
        final_cache_file = os.path.join(cache_dir, "final_improvements.json")
        final_cache_data = {
            "type": "keyframe_judgment",  # 标识这是关键帧判断的改进知识
            "final_level": current_level,
            "source_groups": len(level_0_improvements),  # 原始组数
            "improvements": final_improvements_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name
        }
        
        with open(final_cache_file, 'w', encoding='utf-8') as f:
            json.dump(final_cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"  最终知识已保存到: {final_cache_file}")
        
        return final_improvements_text
    else:
        return "递归合并失败：最终结果数量异常。"


def update_prompt(improvements: str, backup: bool = True, dry_run: bool = False) -> Tuple[bool, str]:
    """
    将总结的知识替换到现有的 prompt 中（替换而非增量添加）
    
    Args:
        improvements: 总结的改进知识
        backup: 是否备份原文件
        dry_run: 是否只预览不实际更新
    
    Returns:
        Tuple[bool, str]: (是否成功, 消息)
    """
    prompt_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "prompts", "task_completion_judgment.py")
    
    if not os.path.exists(prompt_file):
        return False, f"Prompt 文件不存在: {prompt_file}"
    
    try:
        # 读取原文件
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 备份原文件
        if backup and not dry_run:
            backup_file = prompt_file + ".bak"
            shutil.copy2(prompt_file, backup_file)
            print(f"已备份原文件到: {backup_file}")
        
        # 构建新的改进知识部分（保持缩进格式）
        improvements_section = f"""
- **基于错误案例分析的改进建议**：
{improvements}
"""
        
        # 查找是否已存在"基于错误案例分析的改进建议"部分
        # 匹配模式：从 "- **基于错误案例分析的改进建议**：" 开始，到下一个 "- **" 或 prompt 结束（"""）为止
        pattern = r'(- \*\*基于错误案例分析的改进建议\*\*：.*?)(?=\n- \*\*|\n"""|$)'
        
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # 如果存在，替换整个部分
            print("  找到已有的改进建议部分，将替换为新的改进建议")
            new_content = content[:match.start()] + improvements_section + content[match.end():]
        else:
            # 如果不存在，在"特别注意："部分末尾添加
            print("  未找到已有的改进建议部分，将在'特别注意：'部分末尾添加")
            # 查找"特别注意："部分的结束位置
            pattern = r'(特别注意：.*?- 如果最后的状态是输入中、选择中、筛选中、浏览中等，通常表示任务未完成\n)'
            
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                # 尝试更宽松的模式
                pattern = r'(特别注意：.*?)(- 必须包含最后两个图的caption和最后一个action.*?\n- 如果最后的状态是.*?\n)'
                match = re.search(pattern, content, re.DOTALL)
                if not match:
                    return False, "无法找到 '特别注意：' 部分的结束位置，请检查 prompt 文件格式"
            
            # 在"特别注意："部分末尾添加
            insert_pos = match.end()
            new_content = content[:insert_pos] + improvements_section + content[insert_pos:]
        
        if dry_run:
            # 只预览，不实际更新
            print("=" * 60)
            print("预览更新内容（改进知识部分）:")
            print("=" * 60)
            print(improvements_section)
            print("=" * 60)
            return True, "预览模式：未实际更新文件"
        else:
            # 实际更新文件
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True, f"已更新 prompt 文件: {prompt_file}"
            
    except Exception as e:
        return False, f"更新 prompt 失败: {e}"


def generate_documentation(analysis: Dict, improvements: str, output_dir: str, model_name: str) -> str:
    """
    生成改进知识文档
    
    Args:
        analysis: 错误模式分析结果
        improvements: 总结的改进知识
        output_dir: 输出目录
        model_name: 模型名称
    
    Returns:
        str: 生成的文档文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_file = os.path.join(output_dir, f"prompt_improvements_{timestamp}.md")
    
    doc_content = f"""# Prompt 改进知识 - {timestamp}

## 模型
{model_name}

## 错误案例统计
- 总错误数: {analysis['total_errors']}
- 0_to_1 错误（假阳性）: {analysis['error_types']['0_to_1']}
- 1_to_0 错误（假阴性）: {analysis['error_types']['1_to_0']}

## 错误案例详情

### 0_to_1 错误案例（应该是0但预测为1）
"""
    
    for i, case in enumerate(analysis['error_cases_by_type']['0_to_1'][:10], 1):
        doc_content += f"""
#### 案例 {i}
- 任务ID: {case['task_id']}
- 任务指令: {case['instruction']}
- 错误原因: {case['reason']}
- 关键证据: {', '.join(case['key_evidence']) if case['key_evidence'] else '无'}
- 缺失步骤: {', '.join(case['missing_steps']) if case['missing_steps'] else '无'}

"""
    
    doc_content += """
### 1_to_0 错误案例（应该是1但预测为0）
"""
    
    for i, case in enumerate(analysis['error_cases_by_type']['1_to_0'][:10], 1):
        doc_content += f"""
#### 案例 {i}
- 任务ID: {case['task_id']}
- 任务指令: {case['instruction']}
- 错误原因: {case['reason']}
- 关键证据: {', '.join(case['key_evidence']) if case['key_evidence'] else '无'}
- 缺失步骤: {', '.join(case['missing_steps']) if case['missing_steps'] else '无'}

"""
    
    doc_content += f"""
## 总结的改进知识

{improvements}

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    return doc_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自动化 Prompt 工程改进脚本")
    parser.add_argument("--model", type=str, default="gpt-4o", help="模型名称（默认: gpt-4o）")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只分析，不更新 prompt")
    parser.add_argument("--no-backup", action="store_true", help="更新前不备份原文件")
    parser.add_argument("--cache", action="store_const", const=True, default=True, dest="cache", help="使用缓存模式：跳过分组生成，直接汇总已有缓存（默认: True）")
    parser.add_argument("--no-cache", action="store_const", const=False, dest="cache", help="不使用缓存模式：重新生成所有分组")
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.data_dir is None:
        # 默认路径：/opt/nas/p/mm/ie_env/zhaoj/orm/data/{model}/save
        default_base = "/opt/nas/p/mm/ie_env/zhaoj/orm/data"
        args.data_dir = os.path.join(default_base, args.model, "save")
    
    if args.output_dir is None:
        # 默认输出目录：data/{model}/prompt_improvements/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.output_dir = os.path.join(project_root, "data", args.model, "prompt_improvements")
    
    # 设置缓存目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, "data", args.model, "prompt_improvements_cache")
    
    print("=" * 60)
    print("自动化 Prompt 工程改进脚本")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"缓存目录: {cache_dir}")
    print(f"预览模式: {args.dry_run}")
    print(f"缓存模式: {args.cache} (跳过分组生成，直接汇总缓存)")
    print("=" * 60)
    print()
    
    if args.cache:
        # 缓存模式：直接读取缓存，跳过错误案例收集和分析
        print("[缓存模式] 跳过错误案例收集和分析，直接使用已有缓存...")
        error_cases = []
        analysis = None
        cache_dir_result = cache_dir
        
        # 直接汇总所有缓存的改进知识
        print("\n[1/2] 汇总所有缓存的改进知识...")
        final_improvements = aggregate_improvements(cache_dir, args.model)
    else:
        # 非缓存模式：正常流程
        # 步骤1: 收集错误案例
        print("[1/6] 收集错误案例...")
        error_cases = collect_error_cases(args.data_dir)
        print(f"找到 {len(error_cases)} 个错误案例")
        
        if len(error_cases) == 0:
            print("没有找到错误案例，退出。")
            return
        
        if len(error_cases) < 5:
            print(f"警告: 错误案例数量较少（{len(error_cases)}），建议至少 5 个案例才进行总结")
            if args.dry_run:
                print("预览模式：继续执行...")
            else:
                print("警告: 错误案例数量较少，但将继续执行")
        
        # 步骤2: 分析错误模式
        print("\n[2/6] 分析错误模式...")
        analysis = analyze_error_patterns(error_cases)
        print(f"错误类型分布: 0_to_1={analysis['error_types']['0_to_1']}, 1_to_0={analysis['error_types']['1_to_0']}")
        
        # 步骤3: 分批生成改进知识并缓存
        print("\n[3/6] 分批生成改进知识并缓存（每组5个案例）...")
        cache_dir_result = summarize_improvements_in_batches(error_cases, args.model, cache_dir)
        if not cache_dir_result:
            print("错误: 缓存目录设置失败")
            return
        print(f"所有改进知识已缓存到: {cache_dir_result}")
        
        # 步骤4: 汇总所有缓存的改进知识
        print("\n[4/6] 汇总所有缓存的改进知识...")
        final_improvements = aggregate_improvements(cache_dir_result, args.model)
    print("最终改进知识已生成")
    print("\n最终改进知识预览（前300字符）:")
    print("-" * 60)
    print(final_improvements[:300] + "..." if len(final_improvements) > 300 else final_improvements)
    print("-" * 60)
    
    if args.cache:
        # 缓存模式：最终知识已保存到 final_improvements.json，将在运行时动态加载
        print("\n[2/2] 最终知识已保存到 final_improvements.json，将在运行时动态加载（不再直接修改 prompt 文件）")
    else:
        # 非缓存模式：正常流程
        # 步骤5: 最终知识已保存，将在运行时动态加载（不再直接修改 prompt 文件）
        print("\n[5/6] 最终知识已保存到 final_improvements.json，将在运行时动态加载（不再直接修改 prompt 文件）")
        
        # 步骤6: 生成文档
        print("\n[6/6] 生成文档...")
        doc_file = generate_documentation(analysis, final_improvements, args.output_dir, args.model)
        print(f"文档已保存到: {doc_file}")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
