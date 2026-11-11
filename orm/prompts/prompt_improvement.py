"""
Prompt 改进相关模板
用于自动化 Prompt 工程改进脚本
"""


def build_error_analysis_prompt(error_summary: str, judgment_type: str) -> str:
    """
    构建错误案例分析的 prompt
    
    Args:
        error_summary: 错误案例摘要（已格式化的字符串）
        judgment_type: 判断类型，"keyframe" 或 "qa"
    
    Returns:
        str: 完整的 prompt 字符串
    """
    # 根据判断类型设置描述
    if judgment_type == "keyframe":
        task_description = "keyframe judgment"
    elif judgment_type == "qa":
        task_description = "QA 完成问题判断"
    else:
        task_description = f"{judgment_type} judgment"
    
    prompt = f"""你是一个 Prompt 工程专家。请分析以下 {task_description} 的错误案例，总结出可以直接添加到 prompt 中的判断标准或注意事项。

{error_summary}

**重要**：你需要生成的是可以直接添加到 prompt 的"判断标准"或"特别注意"章节中的具体判断规则，而不是系统层面的改进建议。

请根据这些错误案例，总结出：

1. **具体的判断标准**（格式与现有 prompt 中的"判断标准"部分一致）：
   - 基于错误案例，生成类似"如果出现XX情况，应该判断为完成/未完成"的具体规则
   - 每个规则应该包含：触发条件 + 判断结果
   - 格式示例：
     - **对于XX类任务**：如果只执行了XX操作，没有XX操作 → 未完成
     - **如果任务要求XX，但操作序列中显示XX** → 未完成

2. **具体的注意事项**（格式与现有 prompt 中的"特别注意"部分一致）：
   - 基于错误案例，生成具体的注意事项
   - 格式示例：
     - **对于XX场景**：必须检查XX，如果XX则判断为未完成
     - **注意XX情况**：如果XX，应该判断为XX

要求：
- **必须基于具体的错误案例**，不要生成通用的系统改进建议
- **格式必须与现有 prompt 保持一致**（使用 `- **XX**：` 的格式）
- **每个规则必须具体、可操作**，可以直接用于判断
- **不要重复现有 prompt 中已有的内容**
- **重点关注可以避免这些具体错误的判断规则**

输出格式示例：
```
- **对于搜索类任务**：
  - 如果搜索后显示"暂无搜索结果"或"未找到相关内容" → 未完成
  - 必须成功打开或显示目标内容才算完成

- **对于打开/查看类任务**：
  - 如果操作序列中只执行了搜索，但没有实际打开目标页面/内容 → 未完成
  - 必须看到目标页面/内容已显示才算完成
```

请直接输出改进知识，使用与现有 prompt 一致的格式，不要添加其他说明文字。"""
    
    return prompt


def build_merge_improvements_prompt(aggregated_content: str) -> str:
    """
    构建合并改进知识的 prompt
    
    Args:
        aggregated_content: 聚合的改进知识内容（已格式化的字符串）
    
    Returns:
        str: 完整的 prompt 字符串
    """
    prompt = f"""你是一个 Prompt 工程专家。以下是多个改进知识。请将这些改进知识汇总、去重、合并，生成一个更精炼的、可以直接添加到 prompt 中的判断标准或注意事项。

{aggregated_content}

**要求**：
- 汇总所有改进知识，去除重复内容
- 合并相似的规则
- 保持格式与现有 prompt 一致（使用 `- **XX**：` 的格式）
- 生成精炼的、可以直接添加到 prompt 的改进建议
- 如果多个知识提出了相同的规则，只保留一个
- 如果多个知识提出了相似的规则，合并成一个更通用的规则

**输出格式**（与现有 prompt 一致）：
```
- **对于XX类任务**：
  - 如果XX情况 → 未完成
  - 必须XX才算完成

- **对于XX场景**：
  - 必须检查XX，如果XX则判断为未完成
```

请直接输出合并后的改进知识，使用与现有 prompt 一致的格式，不要添加其他说明文字。"""
    
    return prompt
