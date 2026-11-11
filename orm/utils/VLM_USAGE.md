# VLM API 使用说明

## 概述

`utils/ask_qwen.py` 现在支持两种模式：
1. **LLM模式**：纯文本调用
2. **VLM模式**：多模态调用（文本 + 图像）

## API 函数

### 1. `call_vllm_api()` - 通用API调用函数

这是主要的API调用函数，自动根据输入判断使用LLM还是VLM模式。

```python
from utils.ask_qwen import call_vllm_api

# LLM模式（纯文本）
response = call_vllm_api("给我推荐几首好听的音乐")

# VLM模式（文本 + 图像）
response = call_vllm_api(
    message="描述这张图片",
    images=["http://example.com/image.png"]
)

# 支持多张图片
response = call_vllm_api(
    message="比较这两张图片的差异",
    images=["http://example.com/image1.png", "http://example.com/image2.png"]
)
```

### 2. `call_llm_api()` - LLM专用函数

专门用于纯文本调用，语义更清晰。

```python
from utils.ask_qwen import call_llm_api

response = call_llm_api(
    "给我推荐几首好听的音乐",
    temperature=0.8,
    max_tokens=200
)
```

### 3. `call_vlm_api()` - VLM专用函数

专门用于多模态调用，语义更清晰。

```python
from utils.ask_qwen import call_vlm_api

response = call_vlm_api(
    message="描述这张图片",
    images=["http://example.com/image.png"],
    temperature=0.1,
    max_tokens=500
)
```

## 使用场景

### 场景1：纯文本任务（LLM）

适用于不需要图像的任务，如文本生成、问答等。

```python
from utils.ask_qwen import call_llm_api

# 槽位提取
prompt = "从以下任务指令中提取槽位：帮我在酒店预订 App 上订成都顺安村居酒店..."
response = call_llm_api(prompt)
```

### 场景2：图像理解任务（VLM）

适用于需要理解图像的任务，如GUI截图分析、图像描述等。

```python
from utils.ask_qwen import call_vlm_api

# GUI截图分析
image_url = "http://10.162.200.22:9000/digitalworld/task_id/000.png"
prompt = "请描述这张截图中的界面状态"
response = call_vlm_api(message=prompt, images=[image_url])
```

### 场景3：在 Action/Thought/Caption 模块中使用

`prompts/action_thought_caption.py` 模块已经自动支持VLM模式。当提供图像URL时，会自动使用VLM模式生成action、thought和caption。

```python
from prompts.action_thought_caption import generate_action_thought_caption

trajectory = [
    {
        "image": "http://example.com/000.png",  # 有图像URL
        "value": {"action": "点击酒店", "code": "tap(329, 121)"}
    }
]

# 自动使用VLM模式，因为trajectory中包含image字段
results = generate_action_thought_caption(instruction, trajectory)
```

## 参数说明

### `call_vllm_api()` 参数

- `message` (str or List[Dict]): 
  - 字符串：文本消息
  - 列表：已构建的多模态内容（高级用法）
  
- `images` (List[str], optional): 图像URL列表
  
- `stream` (bool): 是否使用流式输出，默认False
  
- `temperature` (float): 温度参数，控制随机性，默认0.1
  
- `max_tokens` (int, optional): 最大token数量

### 返回值

返回字典格式的API响应：

```python
{
    'choices': [{
        'message': {
            'content': '回复内容'
        }
    }]
}
```

## 消息格式

### LLM模式（纯文本）

```json
{
    "messages": [{
        "role": "user",
        "content": "文本消息"
    }]
}
```

### VLM模式（多模态）

```json
{
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "文本消息"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "http://example.com/image.png"
                }
            }
        ]
    }]
}
```

## 向后兼容性

所有原有的代码无需修改即可继续使用：

```python
# 旧代码仍然有效（自动使用LLM模式）
response = call_vllm_api("hello")
```

## 注意事项

1. **图像URL**：确保图像URL可以被API服务器访问
2. **性能**：VLM模式比LLM模式慢，因为需要处理图像
3. **token限制**：多模态调用会消耗更多token，注意max_tokens设置
4. **错误处理**：如果图像URL无效或无法访问，API可能返回错误

## 示例

### 完整示例：处理GUI任务轨迹

```python
from utils.ask_qwen import call_vlm_api
from prompts.action_thought_caption import generate_action_thought_caption

# 任务数据
instruction = "帮我在酒店预订 App 上订成都顺安村居酒店的精致小单间房型"
trajectory = [
    {
        "image": "http://10.162.200.22:9000/digitalworld/task_id/000.png",
        "value": {"action": "点击酒店", "code": "tap(329, 121)"}
    },
    {
        "image": "http://10.162.200.22:9000/digitalworld/task_id/001.png",
        "value": {"action": "选择日期", "code": "tap(153, 333)"}
    }
]

# 自动使用VLM模式处理
results = generate_action_thought_caption(instruction, trajectory)

for result in results:
    print(f"步骤{result['index']}:")
    print(f"  Action: {result['action']}")
    print(f"  Thought: {result['thought']}")
    print(f"  Caption: {result['sparse_caption']}")
```

## 测试

运行测试：

```bash
python utils/ask_qwen.py
```

这将测试：
1. 纯文本LLM调用
2. 多模态VLM调用（需要提供图像URL）
