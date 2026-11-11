"""
Action/Thought/Caption模块
为操作序列中的每个步骤生成action、thought和sparse caption
生成顺序：Action -> Thought -> Sparse Caption
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_utils import (
    bind_pipeline_preset,
    call_llm_with_prompt,
    get_pipeline_preset,
    get_image_path_from_step,
    get_task_context,
    load_from_cache,
    parse_json_response,
    save_to_cache,
    set_task_context,
    unbind_pipeline_preset,
)
from utils.lite_prompt_knowledge import normalize_takeover_judge_mode

TAKEOVER_THOUGHT_CONTINUE_GOALS = {"Auto任务继续", "外部模型接管"}
TAKEOVER_THOUGHT_FALLBACK = "当前模型给出接管指令，接管后用户进行了action操作"

ACTION_ALIASES = {
    "set_text": "type",
}

CLICK_LIKE_ACTIONS = {"click", "double_click", "long_press"}
TEXT_LIKE_ACTIONS = {"type", "set_text"}
SKIP_VALIDATION_ACTIONS = {
    "wait",
    "scroll",
    "take_over",
    "terminate",
    "open",
    "open_app",
    "openapp",
    "launch_app",
    "start_app",
    "system_button",
    "unknown_action",
    "call_api",
    "api",
    "input_event",
    "ack",
}
SKIP_VALIDATION_PREFIXES = tuple(f"{action}(" for action in sorted(SKIP_VALIDATION_ACTIONS))


def _normalize_action_name(action: Any) -> str:
    name = str(action or "").strip().lower()
    return ACTION_ALIASES.get(name, name)


def _get_normalized_action_name(step: Dict) -> str:
    value = step.get("value", {}) if isinstance(step, dict) else {}
    if not isinstance(value, dict):
        return ""
    normalized = value.get("normalized_tool_call")
    if isinstance(normalized, dict):
        return _normalize_action_name(normalized.get("action", ""))
    return ""


def _is_ack_tool_call(step: Dict) -> bool:
    value = step.get("value", {}) if isinstance(step, dict) else {}
    if not isinstance(value, dict):
        return False
    raw_tool_call = value.get("tool_call")
    if isinstance(raw_tool_call, dict):
        parsed = raw_tool_call
    elif isinstance(raw_tool_call, str) and raw_tool_call.strip():
        try:
            parsed = json.loads(raw_tool_call)
        except Exception:
            return False
    else:
        return False
    if not isinstance(parsed, dict):
        return False
    args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
    return parsed.get("name") == "input_event" and args.get("contextOperation") == "ACK"


def _get_takeover_thought_override(step: Dict) -> Optional[str]:
    """
    接管接续步骤 thought 覆盖逻辑：
    - continue_goal 命中白名单时，优先用原始 value.thought
    - 原始 thought 为空时，使用固定兜底文案
    - 未命中返回 None（继续原有 thought 生成逻辑）
    """
    take_over = step.get("take_over")
    if not isinstance(take_over, dict):
        return None
    continue_goal = (take_over.get("continue_goal", "") or "").strip()
    if continue_goal not in TAKEOVER_THOUGHT_CONTINUE_GOALS:
        return None
    raw_thought = step.get("value", {}).get("thought", "")
    if isinstance(raw_thought, str) and raw_thought.strip():
        return raw_thought.strip()
    return TAKEOVER_THOUGHT_FALLBACK


def _should_skip_action_validation(step: Dict, code: str) -> bool:
    """Return True for actions that should not be validated from before/after screenshots."""
    normalized_action = _get_normalized_action_name(step)
    if normalized_action in SKIP_VALIDATION_ACTIONS or _is_ack_tool_call(step):
        return True

    code_lower = (code or "").strip().lower()
    return code_lower.startswith(SKIP_VALIDATION_PREFIXES)


def _action_validation_result(
    original_action: str,
    validated_action: str,
    first_pass_corrected_action: str,
    validation_reason: str = "",
    double_check_choice: Optional[int] = None,
    double_check_reason: str = "",
    return_meta: bool = False,
) -> Union[str, Dict[str, Any]]:
    """Return either the legacy action string or a metadata-rich validation record."""
    if not return_meta:
        return validated_action
    return {
        "action": validated_action,
        "original_action": original_action,
        "validated_action": validated_action,
        "action_changed": bool((original_action or "").strip() != (validated_action or "").strip()),
        "first_pass_corrected_action": first_pass_corrected_action,
        "validation_reason": validation_reason or "",
        "double_check_choice": double_check_choice,
        "double_check_reason": double_check_reason or "",
    }


def _invoke_validate_and_correct_action(
    index: int,
    current_image_url: str,
    next_image_url: Optional[str],
    preset_action: str,
    trajectory: List[Dict],
    *,
    action_code: Optional[str] = None,
    return_meta: bool = False,
) -> Union[str, Dict[str, Any]]:
    """Call validate_and_correct_action; tolerate stale workers missing return_meta."""
    try:
        return validate_and_correct_action(
            index,
            current_image_url,
            next_image_url,
            preset_action,
            trajectory,
            action_code=action_code,
            return_meta=return_meta,
        )
    except TypeError as exc:
        if return_meta and "return_meta" in str(exc):
            legacy_result = validate_and_correct_action(
                index,
                current_image_url,
                next_image_url,
                preset_action,
                trajectory,
                action_code=action_code,
            )
            if isinstance(legacy_result, dict):
                return legacy_result
            return _action_validation_result(
                preset_action,
                legacy_result,
                legacy_result,
                return_meta=True,
            )
        raise


def _double_check_result(
    action: str,
    choice: Optional[int] = None,
    reason: str = "",
    return_details: bool = False,
) -> Union[str, Dict[str, Any]]:
    if not return_details:
        return action
    return {"action": action, "choice": choice, "reason": reason or ""}


def find_next_non_wait_state(trajectory: List[Dict], start_index: int) -> Optional[int]:
    """
    从start_index开始，查找下一个非等待状态的索引
    
    Args:
        trajectory: 操作序列
        start_index: 起始索引
    
    Returns:
        int: 下一个非等待状态的索引，如果不存在则返回None
    """
    for i in range(start_index + 1, len(trajectory)):
        action = trajectory[i].get("value", {}).get("action", "")
        if "等待" not in action:
            return i
    return None


def find_prev_non_wait_state(trajectory: List[Dict], start_index: int) -> Optional[int]:
    """
    从start_index向前查找上一个非等待状态的索引
    
    Args:
        trajectory: 操作序列
        start_index: 起始索引
    
    Returns:
        int: 上一个非等待状态的索引，如果不存在则返回None
    """
    for i in range(start_index - 1, -1, -1):
        action = trajectory[i].get("value", {}).get("action", "")
        if "等待" not in action:
            return i
    return None


def _get_next_image_url_for_validation(index: int, image_url: str, trajectory: List[Dict]) -> Optional[str]:
    """获取用于 action 校验的下一张图 URL（与 generate_action 中逻辑一致）。"""
    if index >= len(trajectory) - 1:
        return None
    next_action = trajectory[index + 1].get("value", {}).get("action", "")
    if "等待" in next_action:
        next_non_wait_index = find_next_non_wait_state(trajectory, index)
        if next_non_wait_index is None:
            return None
        next_image = get_image_path_from_step(trajectory[next_non_wait_index])
    else:
        next_image = get_image_path_from_step(trajectory[index + 1])
    if not next_image or not next_image.strip() or next_image == image_url:
        return None
    return next_image


def _get_prev_image_url_for_caption(index: int, image_url: str, trajectory: List[Dict]) -> Optional[str]:
    """获取用于 caption 的「前一张图」URL（处理等待逻辑，只读 trajectory）。"""
    if index <= 0:
        return None
    prev_action_check = trajectory[index - 1].get("value", {}).get("action", "")
    if "等待" in prev_action_check:
        prev_non_wait_index = find_prev_non_wait_state(trajectory, index)
        if prev_non_wait_index is not None:
            prev_image = get_image_path_from_step(trajectory[prev_non_wait_index])
            if prev_image and prev_image.strip() and prev_image != image_url:
                return prev_image
    else:
        prev_image = get_image_path_from_step(trajectory[index - 1])
        if prev_image and prev_image.strip() and prev_image != image_url:
            return prev_image
    return None


def _generate_sparse_caption_parallel(
    index: int,
    action: str,
    thought: str,
    image_url: str,
    trajectory: List[Dict],
    task_id: Optional[str] = None,
    step_name: Optional[str] = None,
    lite_mode: bool = False,
):
    """并行版 caption：不依赖前一步的 action/thought 生成结果，也不依赖 context。"""
    current_step_name = f"{step_name}_caption" if step_name else None

    if "等待" in action:
        return "等待中的页面"

    if lite_mode:
        prompt = f"""生成简短的界面描述。

操作：{action}
步骤：{index}

只输出简短的界面描述。"""
        response = call_llm_with_prompt(prompt, images=None, temperature=0.1, max_tokens=100,
                                        task_id=task_id, step_name=current_step_name)
    else:
        if index == 0:
            prompt = f"""你正在执行一个GUI任务。请仔细观察当前截图，生成一个精简的截图描述。

当前步骤：
- 步骤索引：{index}
- 操作描述：{action}
- 操作意图：{thought}

请仔细观察当前截图，生成一个精简的截图描述，重点描述：
1. 当前屏幕的主要内容和状态
2. 与操作相关的关键元素
3. 操作后的预期结果或实际变化

描述要简洁，只包含关键信息，不要过于详细。

只输出截图描述，不要添加其他说明。"""
            images = [image_url] if image_url else None
            response = call_llm_with_prompt(prompt, images=images, temperature=0.1, max_tokens=200,
                                            task_id=task_id, step_name=current_step_name)
        else:
            prev_image_url = _get_prev_image_url_for_caption(index, image_url, trajectory)
            if not prev_image_url:
                prompt = f"""你正在执行一个GUI任务。请仔细观察当前截图，生成一个精简的截图描述。

当前步骤：
- 步骤索引：{index}
- 操作描述：{action}
- 操作意图：{thought}

请仔细观察当前截图，生成一个精简的截图描述。

只输出截图描述，不要添加其他说明。"""
                images = [image_url] if image_url else None
                response = call_llm_with_prompt(prompt, images=images, temperature=0.1, max_tokens=200,
                                                task_id=task_id, step_name=current_step_name)
            else:
                prompt = f"""# Role
你是一个专业的 GUI 状态分析器。你将接收前后两张截图（image_1=操作前，image_2=操作后），需要精确对比差异并输出结构化描述。

# Input Context
- 当前步骤索引：{index}
- 当前操作：{action}
- 当前意图：{thought}

# Analysis Strategy

## Step 1: 锚定 image_1 状态
逐区域扫描 image_1（顶部导航栏 → 主内容区 → 底部/悬浮元素），记录：
- 页面类型与标题（如"商品列表页"、"设置页"）
- 关键控件及其当前值（输入框内容、开关状态、按钮文案与可用性、列表项数量、选中项等）

## Step 2: 锚定 image_2 状态
同样逐区域扫描 image_2，重点关注：
- 页面类型是否改变（是否发生跳转）
- 是否出现/消失覆盖层（弹窗、Toast、下拉菜单、底部弹出面板等）
- 所有可见控件的当前值

## Step 3: 差异对比（核心）
将 image_1 与 image_2 逐项对比，分三个维度归纳变化：
1. **操作直接结果**：当前操作触发的直接效果（如点击"提交"→ 出现成功弹窗）
2. **控件值变化**：任何控件的值/状态从 A→B 的变化，必须写明变化前后的具体值（如 输入框从空变为"张三"、数量从 2→3、开关从关→开）
3. **页面级变化**：页面跳转、布局重排、内容刷新、元素出现/消失

# Output Schema
严格输出以下 JSON，不要添加任何 JSON 之外的内容：

```json
{{{{
    "prev_image_state": "image_1 的状态快照。格式：[页面类型] + 关键元素及其值的枚举描述。要求具体到控件级别",

    "current_image_state": "image_2 的状态快照。格式同上，独立完整描述，不要用「变为」等对比性措辞",

    "action_result": "操作的直接执行结果。格式：[操作类型]+[目标元素]+[直接效果]",

    "control_changes": "所有控件值的前后对比变化列表。每项变化格式为'[控件名称]: [旧值] → [新值]'，用分号分隔多项变化。若无变化则写'无'",

    "page_changes": "页面级别的变化描述。涵盖：页面跳转、覆盖层出现/消失、内容区域刷新、布局变化。若无变化则写'无'",

    "current_state": "当前页面的可操作性摘要。格式：当前所在页面 + 页面核心状态 + 与当前操作（{action}）相关的目标元素位置及状态",

    "sparse_caption": "≤80字的变化摘要。格式：[操作做了什么] → [产生了什么变化] → [当前页面是什么状态]。用于关键帧筛选，需信息密度最大化"
}}}}
```

# Constraints
- 所有描述必须且只能基于两张截图中的可见信息，严禁推测不可见内容
- 控件值变化必须写明前后具体值，禁止模糊描述（如"内容发生了变化"）
- prev_image_state 和 current_image_state 必须各自独立完整，可脱离上下文单独理解
- sparse_caption 严格控制在 80 字以内"""

                images = [prev_image_url, image_url] if prev_image_url else [image_url]
                response = call_llm_with_prompt(prompt, images=images, temperature=0.1, max_tokens=1200,
                                                task_id=task_id, step_name=current_step_name)

    if response:
        result = parse_json_response(response)
        if result:
            caption = result.get("sparse_caption", "")
            if not caption:
                parts = []
                if result.get("action_result"):
                    parts.append(result["action_result"])
                if result.get("control_changes") and result.get("control_changes") != "无":
                    parts.append(result["control_changes"])
                if result.get("page_changes") and result.get("page_changes") != "无":
                    parts.append(result["page_changes"])
                if result.get("current_state"):
                    parts.append(result["current_state"])
                caption = "。".join(parts) if parts else f"步骤{index}的截图"
                result["sparse_caption"] = caption
            return result

        caption = response.strip()
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]
        return caption

    return f"步骤{index}的截图"


def _parallel_validate_and_thought(
    trajectory: List[Dict],
    indices_to_process: List[int],
    task_id: Optional[str],
    base_step_name: Optional[str],
    lite_mode: bool,
    parallel_caption: bool = False,
    max_workers: int = 10,
    pipeline_preset: Optional[str] = None,
) -> Dict[int, Tuple[Dict[str, Any], Any, Any]]:
    """
    并行：每步先校验 action（含二次校验），再生成 thought，可选再生成 caption。同一 worker 内顺序执行。
    返回 step_index -> (validation_info, thought_result, caption_result)。
    caption_result 在 parallel_caption=False 时为 None。

    pipeline_preset：主线程在进线程池前解析的 preset，绑定到各 worker，使 tmp/tmp2 与 judgment 一致
    （否则 threading.local 在子线程无 preset，prompt_response 会落到 data/<model>/tmp2）。
    """
    results: Dict[int, Tuple[Dict[str, Any], Any, Any]] = {}
    tls_preset = pipeline_preset if pipeline_preset is not None else get_pipeline_preset()

    def do_step(idx: int) -> Tuple[int, Dict[str, Any], Any, Any]:
        if tls_preset:
            bind_pipeline_preset(tls_preset)
        try:
            step = trajectory[idx]
            image_url = get_image_path_from_step(step)
            action_code = step.get("value", {}).get("action", "")
            code = step.get("value", {}).get("code", "")
            action_preset = (action_code or "").strip()

            # 1. action 校验
            if not action_preset:
                validation_info = _action_validation_result(
                    action_preset, "", "", return_meta=True
                )
            elif _should_skip_action_validation(step, code):
                validation_info = _action_validation_result(
                    action_preset, action_preset, action_preset, return_meta=True
                )
            else:
                next_image_url = _get_next_image_url_for_validation(idx, image_url, trajectory)
                if not next_image_url or idx >= len(trajectory) - 1 or image_url == next_image_url:
                    validation_info = _action_validation_result(
                        action_preset, action_preset, action_preset, return_meta=True
                    )
                else:
                    if task_id is not None and base_step_name is not None:
                        set_task_context(task_id, f"{base_step_name}_step{idx}_validate_action")
                    validation_result = _invoke_validate_and_correct_action(
                        idx, image_url, next_image_url, action_preset, trajectory,
                        action_code=code, return_meta=True,
                    )
                    if isinstance(validation_result, dict):
                        validation_info = validation_result
                    else:
                        validation_info = _action_validation_result(
                            action_preset, validation_result, validation_result, return_meta=True
                        )
            validated_action = validation_info.get("action", "")

            # 2. thought
            takeover_thought_override = _get_takeover_thought_override(step)
            if lite_mode:
                thought_result = None
            elif takeover_thought_override is not None:
                # 接管接续场景直接使用原始/兜底 thought，跳过生成
                thought_result = takeover_thought_override
            else:
                if task_id is not None and base_step_name is not None:
                    set_task_context(task_id, f"{base_step_name}_step{idx}_thought")
                thought_result = generate_thought(idx, validated_action, image_url, trajectory)

            # 3. caption（仅 parallel_caption 时）
            caption_result = None
            if parallel_caption:
                if thought_result is None:
                    thought_str = f"执行{action_code}以完成任务"
                elif isinstance(thought_result, dict):
                    thought_str = thought_result.get("operation_intent", "")
                else:
                    thought_str = thought_result if thought_result else ""

                sn = f"{base_step_name}_step{idx}" if base_step_name else None
                if task_id is not None and sn is not None:
                    set_task_context(task_id, sn)
                caption_result = _generate_sparse_caption_parallel(
                    idx, validated_action, thought_str, image_url, trajectory,
                    task_id=task_id, step_name=sn, lite_mode=lite_mode,
                )

            return idx, validation_info, thought_result, caption_result
        finally:
            if tls_preset:
                unbind_pipeline_preset()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(do_step, i): i for i in indices_to_process}
        for future in as_completed(futures):
            idx, validation_info, thought_result, caption_result = future.result()
            results[idx] = (validation_info, thought_result, caption_result)

    return results


def generate_action_thought_caption(
    trajectory: List[Dict],
    lite_mode: bool = False,
    parallel_caption: bool = True,
) -> List[Dict]:
    """
    为操作序列中的每个步骤生成action、thought和sparse caption

    Args:
        trajectory: 操作序列，每个元素包含image和value字段
        lite_mode: 简略模式，只生成关键步骤
        parallel_caption: True 时 caption 也在线程池内并行生成（不依赖前步生成结果），
                          False 时 caption 按步骤顺序串行生成（依赖前步 action/thought）。
    """
    results = []

    # 简略模式：只生成关键步骤和最后两步
    if lite_mode:
        total_steps = len(trajectory)
        key_indices = set()
        if total_steps > 0:
            key_indices.add(0)
        if total_steps > 1:
            key_indices.add(total_steps - 2)
        if total_steps > 0:
            key_indices.add(total_steps - 1)
        for i in range(1, total_steps - 2, 5):
            key_indices.add(i)
        for i, step in enumerate(trajectory):
            action = step.get("value", {}).get("action", "").lower()
            if any(keyword in action for keyword in ["支付", "确认", "提交", "完成", "选择", "填写"]):
                key_indices.add(i)
        indices_to_process = sorted(key_indices)
        print(f"  简略模式：只处理 {len(indices_to_process)}/{total_steps} 个关键步骤")
    else:
        indices_to_process = list(range(len(trajectory)))

    task_id, base_step_name = get_task_context()
    # 并行：action 校验 → thought → 可选 caption（同一 worker 内顺序执行）
    pool_results = _parallel_validate_and_thought(
        trajectory, indices_to_process, task_id, base_step_name, lite_mode,
        parallel_caption=parallel_caption,
    )
    for i in indices_to_process:
        step = trajectory[i]
        image_url = get_image_path_from_step(step)
        action_code = step.get("value", {}).get("action", "")
        validation_info, thought_result, caption_result = pool_results[i]
        action = validation_info.get("action", "")

        if thought_result is None:
            thought = f"执行{action_code}以完成任务"
        elif isinstance(thought_result, dict):
            thought = thought_result.get("operation_intent", "")
        else:
            thought = thought_result if thought_result else ""

        if task_id and base_step_name:
            set_task_context(task_id, f"{base_step_name}_step{i}")

        if parallel_caption:
            sparse_caption = caption_result
        else:
            # 串行 caption：依赖前步生成结果
            prev_action = None
            prev_thought = None
            prev_image_url = None
            if i > 0 and results:
                prev_result = None
                for prev_r in results:
                    if prev_r["index"] == i - 1:
                        prev_result = prev_r
                        break
                if prev_result is not None:
                    prev_action = prev_result.get("action", "")
                    prev_thought = prev_result.get("thought", "")
                    prev_image_url = prev_result.get("image_url", "")
                    if prev_action and "等待" in prev_action:
                        prev_non_wait_index = find_prev_non_wait_state(trajectory, i)
                        if prev_non_wait_index is not None:
                            prev_image_url = get_image_path_from_step(trajectory[prev_non_wait_index])
                            for r in results:
                                if r.get("index") == prev_non_wait_index:
                                    prev_action = r.get("action", "")
                                    prev_thought = r.get("thought", "")
                                    break

            sparse_caption = generate_sparse_caption(
                i, action, thought, image_url, trajectory,
                prev_action=prev_action,
                prev_thought=prev_thought,
                prev_image_url=prev_image_url,
                lite_mode=lite_mode,
            )

        step_result = {
            "index": i,
            "action": action,
            "thought": thought,
            "image_url": image_url,
            "lite_mode": lite_mode,
        }
        step_result.update(validation_info)
        if thought_result is not None and isinstance(thought_result, dict):
            step_result["thought_state_change_analysis"] = thought_result.get("state_change_analysis", "")
            step_result["thought_operation_intent"] = thought_result.get("operation_intent", "")
        if isinstance(sparse_caption, dict):
            step_result["sparse_caption"] = sparse_caption.get("sparse_caption", f"步骤{i}的界面")
            for field in ["prev_image_state", "current_image_state", "action_result",
                          "control_changes", "page_changes", "current_state"]:
                if field in sparse_caption:
                    step_result[field] = sparse_caption[field]
        else:
            step_result["sparse_caption"] = sparse_caption
        results.append(step_result)
    
    # 简略模式下，为未处理的步骤填充简化数据
    if lite_mode:
        filled_results = []
        result_dict = {r["index"]: r for r in results}
        for i in range(len(trajectory)):
            if i in result_dict:
                filled_results.append(result_dict[i])
            else:
                # 填充简化数据
                step = trajectory[i]
                action_code = step.get("value", {}).get("action", "")
                filled_results.append({
                    "index": i,
                    "action": action_code,  # 使用原始action
                    "thought": "执行操作步骤",  # 简化thought
                    "sparse_caption": f"步骤{i}的界面",  # 简化caption
                    "image_url": get_image_path_from_step(step),
                    "lite_mode": True  # 标记为简略模式生成
                })
        results = filled_results
    
    return results


def _run_takeover_judgment_parallel(
    instruction: str,
    trajectory: List[Dict],
    action_thought_caption_results: List[Dict],
    task_id: Optional[str],
    takeover_ground_truth_list: List[int],
    profile_id: Optional[str] = None,
    judge_takeover: str = "end-to-end",
) -> List[Dict]:
    """
    在 action_thought_caption 全部完成后，并行执行接管判断。
    返回按 step_index 排序的 takeover_judgment 列表。
    """
    from prompts.takeover_judgment import judge_takeover_for_step
    judge_takeover = normalize_takeover_judge_mode(judge_takeover)

    step_inputs = []
    for step_idx in range(len(action_thought_caption_results)):
        step_result = action_thought_caption_results[step_idx]
        current_caption = step_result.get("sparse_caption", "") or step_result.get("current_state", "")
        current_image_url = get_image_path_from_step(trajectory[step_idx])
        history_lines = []
        for prev_idx in range(step_idx):
            prev_result = action_thought_caption_results[prev_idx]
            prev_caption = prev_result.get("sparse_caption", "") or prev_result.get("current_state", "")
            prev_action = prev_result.get("action", "")
            if prev_caption or prev_action:
                history_lines.append(f"Step {prev_idx}: [{prev_caption}] + {prev_action}")
        history_context = "\n".join(history_lines) if history_lines else "（无历史步骤）"
        step_inputs.append((step_idx, history_context, current_caption, current_image_url))

    def _judge_one_step(args):
        step_idx, history_context, current_caption, current_image_url = args
        step_name = f"takeover_judgment_step_{step_idx}"
        if profile_id:
            step_name = f"{step_name}_{profile_id}"
        set_task_context(task_id, step_name)
        judgment_result = judge_takeover_for_step(
            instruction=instruction,
            history_context=history_context,
            current_caption=current_caption,
            current_image_url=current_image_url,
            step_index=step_idx,
            judge_takeover=judge_takeover,
            profile_id=profile_id,
        )
        ground_truth = takeover_ground_truth_list[step_idx] if step_idx < len(takeover_ground_truth_list) else 0
        judgment_result["step_index"] = step_idx
        judgment_result["ground_truth"] = ground_truth
        return step_idx, judgment_result

    if not step_inputs:
        return []
    with ThreadPoolExecutor(max_workers=min(len(step_inputs), 10)) as executor:
        futures = {executor.submit(_judge_one_step, inp): inp[0] for inp in step_inputs}
        collected = {}
        for future in as_completed(futures):
            step_idx, judgment_result = future.result()
            collected[step_idx] = judgment_result
    return [collected[i] for i in range(len(step_inputs))]


def generate_action_thought_caption_with_takeover(
    instruction: str,
    trajectory: List[Dict],
    lite_mode: bool = False,
    parallel_caption: bool = True,
    task_id: Optional[str] = None,
    use_cache: bool = True,
    takeover_ground_truth_list: Optional[List[int]] = None,
    soft_cache_mode: bool = False,
    skip_takeover: bool = False,
    profile_id: Optional[str] = None,
    judge_takeover: str = "end-to-end",
) -> Tuple[List[Dict], List[Dict], bool, bool]:
    """
    生成 action/thought/caption 后紧接着执行接管判断，统一入口。
    skip_takeover=True 时跳过接管判断，返回空列表。
    返回 (action_thought_caption_results, takeover_judgment_results, cache_used_atc, cache_used_takeover)。
    """
    # 1) Action/Thought/Caption
    judge_takeover = normalize_takeover_judge_mode(judge_takeover)
    cache_name_atc = "action_thought_caption_lite" if lite_mode else "action_thought_caption"
    if profile_id:
        cache_name_atc = f"{cache_name_atc}_{profile_id}"
    set_task_context(task_id, cache_name_atc)
    action_thought_caption_results = None
    if use_cache and task_id:
        action_thought_caption_results = load_from_cache(task_id, cache_name_atc)
    if action_thought_caption_results is None:
        action_thought_caption_results = generate_action_thought_caption(
            trajectory, lite_mode=lite_mode, parallel_caption=parallel_caption
        )
        if use_cache and task_id:
            save_to_cache(task_id, cache_name_atc, action_thought_caption_results)
        cache_used_atc = False
    else:
        cache_used_atc = True

    # 删除最后一步的 thought
    if action_thought_caption_results and len(action_thought_caption_results) > 0:
        last_step = action_thought_caption_results[-1]
        if "thought" in last_step and last_step["thought"]:
            last_step["thought"] = ""

    # 2) 接管判断（skip_takeover 时跳过）
    if skip_takeover:
        return action_thought_caption_results, [], cache_used_atc, True

    if takeover_ground_truth_list is None:
        takeover_ground_truth_list = [0] * len(trajectory)
    while len(takeover_ground_truth_list) < len(trajectory):
        takeover_ground_truth_list.append(0)

    cache_name_takeover = f"takeover_judgment_{judge_takeover}"
    if profile_id:
        cache_name_takeover = f"{cache_name_takeover}_{profile_id}"
    set_task_context(task_id, cache_name_takeover)
    takeover_judgment_results = None
    if use_cache and task_id and not soft_cache_mode:
        takeover_judgment_results = load_from_cache(task_id, cache_name_takeover)
    if takeover_judgment_results is None:
        takeover_judgment_results = _run_takeover_judgment_parallel(
            instruction=instruction,
            trajectory=trajectory,
            action_thought_caption_results=action_thought_caption_results,
            task_id=task_id,
            takeover_ground_truth_list=takeover_ground_truth_list,
            profile_id=profile_id,
            judge_takeover=judge_takeover,
        )
        if use_cache and task_id:
            save_to_cache(task_id, cache_name_takeover, takeover_judgment_results)
        cache_used_takeover = False
    else:
        cache_used_takeover = True

    return action_thought_caption_results, takeover_judgment_results, cache_used_atc, cache_used_takeover


def validate_actions_and_thoughts_for_trajectory(
    trajectory: List[Dict],
    task_id: Optional[str] = None,
    use_cache: bool = True,
    profile_id: Optional[str] = None,
    parallel_workers: int = 10,
) -> List[Dict]:
    """
    Action 校验 + thought 生成（含接管接续覆盖），不生成 caption。
    返回 [{"action": str, "thought": str}, ...]
    """
    cache_name = "validated_action_thought_list"
    if profile_id:
        cache_name = f"{cache_name}_{profile_id}"

    if task_id:
        set_task_context(task_id, cache_name)

    cached = None
    if use_cache and task_id:
        cached = load_from_cache(task_id, cache_name)
    if cached is not None and isinstance(cached, list):
        if len(cached) == len(trajectory):
            return cached

    task_ctx_id, base_step_name = (task_id, cache_name) if task_id else get_task_context()
    indices_to_process = list(range(len(trajectory)))

    pool_results = _parallel_validate_and_thought(
        trajectory,
        indices_to_process,
        task_ctx_id,
        base_step_name,
        lite_mode=False,
        parallel_caption=False,
        max_workers=parallel_workers,
    )

    result_list: List[Dict] = []
    for idx in range(len(trajectory)):
        validation_info, thought_result, _ = pool_results[idx]
        validated_action = validation_info.get("action", "")
        action_code = trajectory[idx].get("value", {}).get("action", "")
        if thought_result is None:
            thought = f"执行{action_code}以完成任务"
        elif isinstance(thought_result, dict):
            thought = thought_result.get("operation_intent", "")
        else:
            thought = thought_result if thought_result else ""
        result_item = {"action": validated_action, "thought": thought}
        result_item.update(validation_info)
        result_list.append(result_item)

    if use_cache and task_id:
        save_to_cache(task_id, cache_name, result_list)
    return result_list


def validate_actions_for_trajectory(
    trajectory: List[Dict],
    task_id: Optional[str] = None,
    use_cache: bool = True,
    profile_id: Optional[str] = None,
    parallel_workers: int = 10,
) -> List[str]:
    """
    仅做 action 校验（含二次校验），不生成 thought/caption。

    用于：
    - lite + action_list_validated：让 action_chain 使用 corrected action。
    """

    # 缓存 key：不同 profile_id 下 action 可能不同（prompt/图片/开关不同），要隔离
    cache_name = "validated_action_list"
    if profile_id:
        cache_name = f"{cache_name}_{profile_id}"

    if task_id:
        set_task_context(task_id, cache_name)

    cached = None
    if use_cache and task_id:
        cached = load_from_cache(task_id, cache_name)
    if cached is not None and isinstance(cached, list):
        # 基本校验：长度一致
        if len(cached) == len(trajectory):
            return cached

    # 用现有并行验证实现：lite_mode=True -> 不生成 thought/caption；parallel_caption=False -> 不生成 caption。
    task_ctx_id, base_step_name = (task_id, cache_name) if task_id else get_task_context()
    indices_to_process = list(range(len(trajectory)))

    pool_results = _parallel_validate_and_thought(
        trajectory,
        indices_to_process,
        task_ctx_id,
        base_step_name,
        lite_mode=True,
        parallel_caption=False,
        max_workers=parallel_workers,
    )

    validated_actions = [""] * len(trajectory)
    for idx in indices_to_process:
        validation_info = pool_results[idx][0]
        validated_actions[idx] = validation_info.get("action", "") if isinstance(validation_info, dict) else validation_info

    if use_cache and task_id:
        save_to_cache(task_id, cache_name, validated_actions)
    return validated_actions


def generate_for_step(
    index: int, 
    image_url: str, 
    action_code: str, 
    code: str, 
    trajectory: List[Dict],
    prev_result: Optional[Dict] = None,  # 前一步的结果
    all_results: Optional[List[Dict]] = None,  # 所有已生成的结果（用于处理等待状态）
    lite_mode: bool = False,
    validated_action: Optional[str] = None,  # 若传入则跳过 generate_action，直接使用
) -> Dict:
    """
    为单个步骤生成action、thought和caption
    
    Args:
        prev_result: 前一步的生成结果，包含action和thought（用于生成caption）
        lite_mode: 简略模式，使用更简单的prompt
    
    生成顺序：
    1. 先生成Action（包含校验）
    2. 基于Action生成Thought
    3. 基于Action和Thought生成Sparse Caption
       - 第一张图：使用单图
       - 后续图：使用前后两张图 + 前一步的action和thought
    """
    
    # 若已传入并行校验得到的 action，直接使用；否则调用 generate_action
    if validated_action is not None:
        action = validated_action
    elif lite_mode:
        action = generate_action(index, image_url, action_code, code, trajectory, lite_mode=True)
    else:
        action = generate_action(index, image_url, action_code, code, trajectory)

    thought_result = None  # 初始化，lite_mode下不使用
    if lite_mode:
        # 只生成 caption，thought 使用简化版本
        thought = f"执行{action_code}以完成任务"
        sparse_caption = generate_sparse_caption(
            index, action, thought, image_url, trajectory, 
            prev_action=None, prev_thought=None, prev_image_url=None,
            lite_mode=True
        )
    else:
        # 步骤2：基于Action生成Thought（action 已在上方确定）
        takeover_thought_override = _get_takeover_thought_override(trajectory[index])
        if takeover_thought_override is not None:
            thought_result = takeover_thought_override
        else:
            thought_result = generate_thought(index, action, image_url, trajectory)
        
        # 处理 thought 返回值：可能是 dict 或 str
        if isinstance(thought_result, dict):
            # JSON格式：提取两个字段，组合为 thought 字符串用于传递给 caption
            thought = thought_result.get("operation_intent", "")
        else:
            thought = thought_result  # 字符串（向后兼容或空字符串）
        
        # 步骤3：基于Action和Thought生成Sparse Caption
        # 获取前一步的信息
        prev_action = None
        prev_thought = None
        prev_image_url = None
        
        if prev_result:
            prev_action = prev_result.get("action", "")
            prev_thought = prev_result.get("thought", "")
            prev_image_url = prev_result.get("image_url", "")
            
            # 如果前一个action是"等待"，需要找到更前面的非等待状态
            if prev_action and "等待" in prev_action:
                prev_non_wait_index = find_prev_non_wait_state(trajectory, index)
                if prev_non_wait_index is not None:
                    # 使用更前面的非等待状态
                    prev_image_url = get_image_path_from_step(trajectory[prev_non_wait_index])
                    # 从all_results中获取对应的action和thought
                    if all_results:
                        for result in all_results:
                            if result.get("index") == prev_non_wait_index:
                                prev_action = result.get("action", "")
                                prev_thought = result.get("thought", "")
                                break
        
        sparse_caption = generate_sparse_caption(
            index, action, thought, image_url, trajectory,
            prev_action=prev_action,
            prev_thought=prev_thought,
            prev_image_url=prev_image_url,
            lite_mode=False
        )
    
    # 构建返回结果
    step_result = {
        "index": index,
        "action": action,
        "thought": thought,  # 字符串（operation_intent 或原始文本）
        "image_url": image_url,
        "lite_mode": lite_mode
    }
    
    # 如果 thought 返回了结构化 dict，展开字段存储
    if thought_result is not None and isinstance(thought_result, dict):
        step_result["thought_state_change_analysis"] = thought_result.get("state_change_analysis", "")
        step_result["thought_operation_intent"] = thought_result.get("operation_intent", "")
    
    # 如果 sparse_caption 是 dict（结构化JSON响应），展开所有字段到 step_result
    if isinstance(sparse_caption, dict):
        step_result["sparse_caption"] = sparse_caption.get("sparse_caption", f"步骤{index}的界面")
        # 展开结构化字段
        for field in ["prev_image_state", "current_image_state", "action_result",
                       "control_changes", "page_changes", "current_state"]:
            if field in sparse_caption:
                step_result[field] = sparse_caption[field]
    else:
        # 向后兼容：sparse_caption 是字符串
        step_result["sparse_caption"] = sparse_caption
    
    return step_result


def format_action_code_for_prompt(code: str) -> Optional[str]:
    """
    格式化 action code 用于 prompt 显示
    
    规则：
    - tap(x, y): 保持原状
    - text(x, y, "内容"): 显示坐标和输入文本
    - scroll(x, y, "方向"): 显示坐标和方向
    - scroll(..., "方向", ...): 提取方向，坐标用 ... 表示
    - 其他类型: 直接返回
    
    Args:
        code: action code，如 "tap(675, 78)"、"text(216, 962, \"好啊\")"、"scroll(560, 348, \"left\")"
    
    Returns:
        格式化后的 code 字符串，如果无法解析或为空则返回 None
    """
    if not code or not code.strip():
        return None
    
    import re
    
    # tap(x, y) - 保持原状
    tap_match = re.match(r'^tap\((\d+),\s*(\d+)\)$', code)
    if tap_match:
        return code  # 保持原状
    
    # text(x, y, "内容") - 显示坐标和输入文本
    text_match = re.match(r'^text\((\d+),\s*(\d+),\s*"([^"]+)"\)$', code)
    if text_match:
        x, y, content = text_match.groups()
        return f'text({x}, {y}, "{content}")'
    
    # scroll(x, y, "方向") - 显示坐标和方向
    scroll_match = re.match(r'^scroll\((\d+),\s*(\d+),\s*"([^"]+)"\)$', code)
    if scroll_match:
        x, y, direction = scroll_match.groups()
        return f'scroll({x}, {y}, "{direction}")'
    
    # scroll(..., "方向", ...) - 提取方向，坐标用 ... 表示
    scroll_partial_match = re.match(r'^scroll\([^,]*,\s*"([^"]+)"', code)
    if scroll_partial_match:
        direction = scroll_partial_match.group(1)
        return f'scroll(..., "{direction}")'
    
    # 其他类型（call_api, take_over等）- 直接返回
    return code


def generate_action(
    index: int, 
    image_url: str, 
    action_code: str, 
    code: str, 
    trajectory: List[Dict],
    lite_mode: bool = False
) -> str:
    """生成Action（第一人称描述）- 不包含任务指令，包含校验"""
    
    # 如果action为空，直接返回空字符串，不进行校验或生成
    if not action_code or not action_code.strip():
        return ""

    current_step = trajectory[index] if 0 <= index < len(trajectory) else {}
    if _should_skip_action_validation(current_step, code or ""):
        return action_code.strip()
    
    # 如果原始数据中已经有action描述，先校验再返回
    # 获取下一张图，确保不是空字符串，并且与当前图不同
    # 如果下一个action是"等待"，需要跳过等待状态，找到下一个非等待状态的图
    next_image_url = None
    if index < len(trajectory) - 1:
        # 检查下一个action是否是"等待"
        next_action = trajectory[index + 1].get("value", {}).get("action", "")
        if "等待" in next_action:
            # 如果下一个是等待，找到下一个非等待的state
            next_non_wait_index = find_next_non_wait_state(trajectory, index)
            if next_non_wait_index:
                next_image = get_image_path_from_step(trajectory[next_non_wait_index])
                if next_image and next_image.strip() and next_image != image_url:
                    next_image_url = next_image
        else:
            # 正常情况：使用下一个state
            next_image = get_image_path_from_step(trajectory[index + 1])
            # 确保不是空字符串，并且与当前图不同
            if next_image and next_image.strip() and next_image != image_url:
                next_image_url = next_image
    
    # 校验并修正action（传入完整的 code）；旧链路只消费 action 字符串
    validated_action = _invoke_validate_and_correct_action(
        index, image_url, next_image_url, action_code.strip(), trajectory,
        action_code=code,
        return_meta=False,
    )
    if isinstance(validated_action, dict):
        return validated_action.get("action", action_code.strip())
    return validated_action
    

def generate_thought(index: int, action: str, image_url: str, trajectory: List[Dict]):
    """基于Action和当前图、下一张图生成Thought（推测操作意图）- 不包含任务指令
    
    Returns:
        Dict 或 str: 成功时返回包含 state_change_analysis 和 operation_intent 的 dict，
                      失败或跳过时返回空字符串 ""
    """
    
    # 如果action是"无操作"、"等待"、"接管"，不生成thought
    if "无操作" in action or "等待" in action or "接管" in action:
        return ""
    
    # 如果是最后一步，不生成thought
    if index == len(trajectory) - 1:
        return ""
    
    # 获取下一张图，确保不是空字符串，并且与当前图不同
    next_image_url = None
    if index < len(trajectory) - 1:
        # 检查下一个action是否是"等待"
        next_action = trajectory[index + 1].get("value", {}).get("action", "")
        if "等待" in next_action:
            # 如果下一个是等待，找到下一个非等待的state
            next_non_wait_index = find_next_non_wait_state(trajectory, index)
            if next_non_wait_index:
                next_image = get_image_path_from_step(trajectory[next_non_wait_index])
                if next_image and next_image.strip() and next_image != image_url:
                    next_image_url = next_image
        else:
            # 正常情况：使用下一个state
            next_image = get_image_path_from_step(trajectory[index + 1])
            # 确保不是空字符串，并且与当前图不同
            if next_image and next_image.strip() and next_image != image_url:
                next_image_url = next_image
    
    # 如果没有下一张图，无法基于状态变化生成thought，返回空字符串
    if not next_image_url:
        return ""
    
    context = get_context_info(index, trajectory)
    
    prompt = f"""你正在执行一个GUI任务。请仔细观察前后两张截图，根据操作描述和页面状态变化，完成两项分析。

**重要：图片说明**：
- 第一张图（current_image）：步骤 {index} 的截图，表示**执行操作前的页面状态**
- 第二张图（next_image）：步骤 {index + 1} 的截图，表示**执行操作后的页面状态**
- 当前操作描述：{action}（这个操作应该能解释从第一张图到第二张图的变化）

当前操作：
- 步骤索引：{index}
- 操作描述：{action}

上下文信息：
{context}

**重要**：你必须基于前后两张截图的实际状态变化来分析，不能臆测。

请完成以下两项分析：

1. **对比分析状态变化**：
   - 明确说明从第一张图（操作前）到第二张图（操作后）的具体变化
   - 包括：页面跳转、弹窗出现/消失、内容更新、列表滚动、输入框内容变化、按钮/开关状态变化、控件值变化等
   - 分析这些变化与操作描述"{action}"的对应关系

2. **推测操作意图**：
   - 基于实际观察到的状态变化，说明为什么要执行这个操作
   - 说明这个操作在完成任务中的作用
   - 必须与观察到的状态变化一致

输出格式（必须是有效的JSON）：
{{
    "state_change_analysis": "≤100字，对比两张图的具体变化描述。格式：[变化类型]+[变化内容]。例如'页面从商品列表跳转到商品详情页，显示iPhone 16 Pro的价格和规格信息；底部出现「加入购物车」按钮'",
    "operation_intent": "≤60字，推测操作意图。格式：[目的]+[作用]。例如'点击目标商品进入详情页，以便查看规格和价格并进行购买操作'"
}}

请直接输出JSON，不要添加其他说明文字。"""
    
    # 使用当前图和下一张图
    images = [image_url, next_image_url]
    task_id, step_name = get_task_context()
    current_step_name = f"{step_name}_thought" if step_name else None
    response = call_llm_with_prompt(prompt, images=images, temperature=0.2, max_tokens=300,
                                    task_id=task_id, step_name=current_step_name)
    
    if response:
        result = parse_json_response(response)
        if result:
            # 确保两个字段都存在
            if "state_change_analysis" not in result:
                result["state_change_analysis"] = ""
            if "operation_intent" not in result:
                result["operation_intent"] = ""
            return result
        
        # JSON解析失败，尝试直接使用响应文本（向后兼容）
        thought = response.strip()
        if thought.startswith('"') and thought.endswith('"'):
            thought = thought[1:-1]
        return thought
    
    return ""


def generate_sparse_caption(
    index: int, 
    action: str, 
    thought: str, 
    image_url: str, 
    trajectory: List[Dict],
    prev_action: Optional[str] = None,
    prev_thought: Optional[str] = None,
    prev_image_url: Optional[str] = None,
    lite_mode: bool = False
):
    """基于Action和Thought生成Sparse Caption（精简的截图描述）
    
    返回值：
    - 当VLM返回结构化JSON时，返回包含所有字段的dict（保证有sparse_caption键）
    - 当VLM返回纯文本或解析失败时，返回str
    
    - 第一张图（index == 0）：使用单图生成
    - 后续图（index > 0）：使用前后两张图 + 前一步的action和thought生成，强调变化和详细描述
    """
    
    task_id, step_name = get_task_context()
    current_step_name = f"{step_name}_caption" if step_name else None
    
    # 如果当前action是"等待"，硬编码caption
    if "等待" in action:
        return "等待中的页面"
    
    if lite_mode:
        # 简略模式：使用简化的prompt，不传图像
        prompt = f"""生成简短的界面描述。

操作：{action}
步骤：{index}

只输出简短的界面描述。"""
        response = call_llm_with_prompt(prompt, images=None, temperature=0.1, max_tokens=100,
                                        task_id=task_id, step_name=current_step_name)
    else:
        # 第一张图：使用单图生成
        if index == 0:
            context = get_context_info(index, trajectory)
            
            prompt = f"""你正在执行一个GUI任务。请仔细观察当前截图，生成一个精简的截图描述。

当前步骤：
- 步骤索引：{index}
- 操作描述：{action}
- 操作意图：{thought}

上下文信息：
{context}

请仔细观察当前截图，生成一个精简的截图描述，重点描述：
1. 当前屏幕的主要内容和状态
2. 与操作相关的关键元素
3. 操作后的预期结果或实际变化

描述要简洁，只包含关键信息，不要过于详细。

只输出截图描述，不要添加其他说明。"""
            
            # 使用单图（当前图）
            images = [image_url] if image_url else None
                
            response = call_llm_with_prompt(prompt, images=images, temperature=0.1, max_tokens=200,
                                            task_id=task_id, step_name=current_step_name)
        
        # 后续图：使用前后两张图 + 前一步的action和thought生成
        else:
            # 获取前一张图（复用 _get_prev_image_url_for_caption，与并行版逻辑一致）
            if not prev_image_url:
                prev_image_url = _get_prev_image_url_for_caption(index, image_url, trajectory)
            
            # 如果没有前一张图，fallback到单图模式
            if not prev_image_url:
                context = get_context_info(index, trajectory)
                prompt = f"""你正在执行一个GUI任务。请仔细观察当前截图，生成一个精简的截图描述。

当前步骤：
- 步骤索引：{index}
- 操作描述：{action}
- 操作意图：{thought}

上下文信息：
{context}

请仔细观察当前截图，生成一个精简的截图描述。

只输出截图描述，不要添加其他说明。"""
                images = [image_url] if image_url else None
                response = call_llm_with_prompt(prompt, images=images, temperature=0.1, max_tokens=200,
                                                task_id=task_id, step_name=current_step_name)
            else:
                # 使用前后两张图生成，强调变化
                context = get_context_info(index, trajectory)
                
                prompt = f"""# Role
你是一个专业的 GUI 状态分析器。你将接收前后两张截图（image_1=操作前，image_2=操作后），需要精确对比差异并输出结构化描述。

# Input Context
- 前一步操作：{prev_action if prev_action else "无（初始状态）"}
- 前一步意图：{prev_thought if prev_thought else "无"}
- 当前步骤索引：{index}
- 当前操作：{action}
- 当前意图：{thought}
- 任务上下文：{context}

# Analysis Strategy

## Step 1: 锚定 image_1 状态
逐区域扫描 image_1（顶部导航栏 → 主内容区 → 底部/悬浮元素），记录：
- 页面类型与标题（如"商品列表页"、"设置页"）
- 关键控件及其当前值（输入框内容、开关状态、按钮文案与可用性、列表项数量、选中项等）
- 与「前一步操作」直接相关的目标元素的精确状态

## Step 2: 锚定 image_2 状态
同样逐区域扫描 image_2，重点关注：
- 页面类型是否改变（是否发生跳转）
- 是否出现/消失覆盖层（弹窗、Toast、下拉菜单、底部弹出面板等）
- 所有可见控件的当前值

## Step 3: 差异对比（核心）
将 image_1 与 image_2 逐项对比，分三个维度归纳变化：
1. **操作直接结果**：前一步操作触发的直接效果（如点击"提交"→ 出现成功弹窗）
2. **控件值变化**：任何控件的值/状态从 A→B 的变化，必须写明变化前后的具体值（如 输入框从空变为"张三"、数量从 2→3、开关从关→开）
3. **页面级变化**：页面跳转、布局重排、内容刷新、元素出现/消失

# Output Schema
严格输出以下 JSON，不要添加任何 JSON 之外的内容：

```json
{{
    "prev_image_state": "image_1 的状态快照。格式：[页面类型] + 关键元素及其值的枚举描述。要求具体到控件级别，例如'商品详情页，顶部显示商品名「XX手机」，价格¥2999，数量选择器当前值为1，「加入购物车」按钮为橙色可点击状态'",

    "current_image_state": "image_2 的状态快照。格式同上，独立完整描述，不要用「变为」等对比性措辞",

    "action_result": "操作的直接执行结果。格式：[操作类型]+[目标元素]+[直接效果]。例如'点击了「加入购物车」按钮后，页面底部弹出一个确认弹窗，显示「已成功加入购物车」'。若当前为初始状态则写'初始状态，无前序操作'",

    "control_changes": "所有控件值的前后对比变化列表。每项变化格式为'[控件名称]: [旧值] → [新值]'，用分号分隔多项变化。若无变化则写'无'。例如'数量选择器: 1 → 2; 小计金额: ¥2999 → ¥5998; 购物车角标: 无 → 1'",

    "page_changes": "页面级别的变化描述。涵盖：页面跳转（从X页→Y页）、覆盖层出现/消失（弹窗/Toast/菜单等，含内容摘要）、内容区域刷新、布局变化。若无变化则写'无'",

    "current_state": "当前页面的可操作性摘要。格式：当前所在页面 + 页面核心状态 + 与当前操作（{action}）相关的目标元素位置及状态。此字段服务于下一步操作的执行",

    "sparse_caption": "≤80字的变化摘要。格式：[前一步操作做了什么] → [产生了什么变化] → [当前页面是什么状态]。用于关键帧筛选，需信息密度最大化"
}}
```

# Constraints
- 所有描述必须且只能基于两张截图中的可见信息，严禁推测不可见内容
- 控件值变化必须写明前后具体值，禁止模糊描述（如"内容发生了变化"）
- prev_image_state 和 current_image_state 必须各自独立完整，可脱离上下文单独理解
- sparse_caption 严格控制在 80 字以内"""
                
                # 使用前后两张图
                images = [prev_image_url, image_url] if prev_image_url else [image_url]
                    
                response = call_llm_with_prompt(prompt, images=images, temperature=0.1, max_tokens=1200,
                                                task_id=task_id, step_name=current_step_name)
    
    if response:
        # 解析JSON响应
        result = parse_json_response(response)
        if result:
            # 确保 sparse_caption 字段存在
            caption = result.get("sparse_caption", "")
            if not caption:
                # 如果没有 sparse_caption，尝试组合其他字段生成
                parts = []
                if result.get("action_result"):
                    parts.append(result["action_result"])
                if result.get("control_changes") and result.get("control_changes") != "无":
                    parts.append(result["control_changes"])
                if result.get("page_changes") and result.get("page_changes") != "无":
                    parts.append(result["page_changes"])
                if result.get("current_state"):
                    parts.append(result["current_state"])
                caption = "。".join(parts) if parts else f"步骤{index}的截图"
                result["sparse_caption"] = caption
            
            # 返回完整的结构化dict（包含所有JSON字段）
            return result
        
        # 如果JSON解析失败，尝试直接使用响应文本（向后兼容）
        caption = response.strip()
        if caption.startswith('"') and caption.endswith('"'):
            caption = caption[1:-1]
        return caption
    
    return f"步骤{index}的截图"


def validate_and_correct_action(
    index: int,
    current_image_url: str,
    next_image_url: Optional[str],
    preset_action: str,
    trajectory: List[Dict],
    action_code: Optional[str] = None,  # 新增：完整的 action code
    return_meta: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    根据当前图和下一张图校验action
    
    校验逻辑：
    1. 首先判断原始action是否合理（能否用原始action连接当前图和下一张图）
       - 如果合理：确认action确实改变了页面，返回原action
       - 如果不合理：进入下一步判断
    2. 如果原始action不合理，判断页面是否发生变化
       - 如果页面未发生变化：返回"无操作"（可能点击错位、点击到空白处等）
       - 如果页面发生变化：根据实际页面变化生成新的action描述
    
    Args:
        index: 当前步骤索引
        current_image_url: 当前图片URL（步骤index的图片，操作前的状态）
        next_image_url: 下一张图片URL（步骤index+1的图片，操作后的状态）
        preset_action: 预设的action描述（步骤index的操作）
        trajectory: 操作序列
        action_code: 完整的 action code（如 "tap(675, 78)"、"text(216, 962, \"好啊\")"）
        
    Returns:
        str: 校验后的action（从JSON响应中提取的corrected_action字段）
             - 如果原始action合理：返回原action
             - 如果原始action不合理且页面未变化：返回"无操作"
             - 如果原始action不合理且页面有变化：返回新生成的action
             - 如果解析失败：返回预设action（fallback）
    """
    task_id, step_name = get_task_context()
    current_step_name = f"{step_name}_validate_action" if step_name else None
    
    # 只跳过无法稳定用前后截图校验的函数类操作；点击类即使文案包含“打开”也继续校验。
    current_step = trajectory[index] if 0 <= index < len(trajectory) else {}
    if _should_skip_action_validation(current_step, action_code or ""):
        return _action_validation_result(
            preset_action, preset_action, preset_action, return_meta=return_meta
        )
    
    # 如果没有下一张图，无法校验，直接返回预设action
    if not next_image_url or index >= len(trajectory) - 1:
        return _action_validation_result(
            preset_action, preset_action, preset_action, return_meta=return_meta
        )
    
    # 确保当前图和下一张图不同
    if current_image_url == next_image_url:
        # 如果当前图和下一张图相同，无法校验，直接返回预设action
        return _action_validation_result(
            preset_action, preset_action, preset_action, return_meta=return_meta
        )
    
    # 获取上下文信息
    context = get_context_info(index, trajectory)
    
    # 格式化 code 用于 prompt 显示
    code_info = ""
    if action_code:
        formatted_code = format_action_code_for_prompt(action_code)
        if formatted_code:
            code_info = f"""**操作代码**：
- {formatted_code}

"""
    
    prompt = f"""你是一个GUI操作校验专家。请根据前后两张截图和预设的操作描述，判断操作是否正确。

**重要：图片说明**：
- 第一张图（current_image）：步骤 {index} 的截图，表示**执行操作前的页面状态**
- 第二张图（next_image）：步骤 {index + 1} 的截图，表示**执行操作后的页面状态**
- 预设操作描述：{preset_action}（这个操作应该能解释从第一张图到第二张图的变化）

当前步骤信息：
- 步骤索引：{index}
- 预设操作描述：{preset_action}
{code_info}上下文信息：
{context}

    **校验前的Action动作空间（标准格式）**：
    1. **点击类**：
    - 点击XX（通用点击）
    - 勾选XXX（勾选复选框/选项）
    - 取消勾选XXX（取消勾选复选框/选项）
    - 选中XXX（选中选项/项目）
    - 取消选中XXX（取消选中选项/项目）
    - 开启XXX（开启开关/选项）
    - 关闭XXX（关闭开关/选项）
    - 长按XXX（长按操作）
    - 双击XXX（双击操作）

    2. **输入类**：
    - 输入文本XXX（在输入框中输入文本）

    3. **滑动类**：
    - 向上滑动（向上滑动屏幕/列表）
    - 向下滑动（向下滑动屏幕/列表）
    - 向左滑动（向左滑动屏幕/列表）
    - 向右滑动（向右滑动屏幕/列表）

    ---

    ## 🔴 核心校验流程（必须严格按顺序执行）

    ### 第一步：细粒度UI状态变化检测（最关键）

    **你必须逐一检查以下所有细微变化类型，任何一项发生变化都说明有操作执行：**

    | 变化类型 | 具体检查项 | 示例 |
    |---------|-----------|------|
    | **选中/高亮状态** | 下划线、背景色、边框、加粗、颜色变化 | Tab选项卡下方出现/消失下划线 |
    | **勾选状态** | 复选框打勾/取消、开关ON/OFF、单选按钮填充/空心 | ☑️ ↔ ☐ 或 ◉ ↔ ○ |
    | **展开/折叠** | 箭头方向、内容区域展开收起、accordion状态 | ▶ 变成 ▼，子菜单出现/消失 |
    | **焦点状态** | 输入框光标、聚焦边框、placeholder消失 | 输入框出现蓝色边框和闪烁光标 |
    | **按钮状态** | 按下效果、禁用/启用、颜色深浅变化 | 按钮从灰色变为可点击的蓝色 |
    | **列表/内容位置** | 滚动位置、可见项目变化、新内容露出 | 列表顶部项目不同、滚动条位置变化 |
    | **弹出层** | Toast提示、下拉菜单、弹窗、气泡提示 | 出现"操作成功"提示 |
    | **文本内容** | 数字变化、文案更新、计数器变化 | 购物车数量从"2"变成"3" |

    ### 第二步：状态方向判断（防止搞反）

    **对于涉及状态切换的操作，必须明确判断状态变化方向：**

    【前后状态对比模板】

    元素名称：XXX
    操作前状态：[具体描述，如：复选框未勾选/Tab未选中无下划线]
    操作后状态：[具体描述，如：复选框已勾选/Tab选中有下划线]
    状态变化方向：[未勾选→已勾选] 或 [已勾选→未勾选]

    **常见状态对判断：**
    - 勾选 vs 取消勾选：看操作**后**是否有勾 ☑️=勾选了，☐=取消勾选了
    - 选中 vs 取消选中：看操作**后**是否高亮/有下划线
    - 展开 vs 折叠：看操作**后**内容是否可见

    ### 第三步：校验原始action是否合理

    判断原始action描述的操作是否能解释从第一张图（步骤{index}，操作前）到第二张图（步骤{index + 1}，操作后）的变化：
    - **合理**：原始action与检测到的变化一致（包括状态方向）
    - **不合理**：原始action与实际变化不符，或状态方向搞反

    ### 第四步："无操作"的严格判定（高门槛）

    **⚠️ 只有同时满足以下所有条件，才能判定为"无操作"：**

    1. ✅ 第一步检查的所有UI状态变化类型均无变化
    2. ✅ 两张截图的像素级内容完全一致（允许时间戳等动态元素差异，允许视频播放的差异）
    3. ✅ 没有任何新元素出现或旧元素消失
    4. ✅ 没有任何位置、颜色、大小的变化

    **如果你对"是否有变化"存在任何不确定，请倾向于判定为"有变化"，而非"无操作"。**

    ---

    ## 输出格式

    请先进行内部分析（不输出），然后只输出以下JSON：

    ```json  
    {{  
        "original_action": "{preset_action}",  
        "detected_changes": "检测到的具体UI变化描述（如：'XXX选项卡下方出现蓝色下划线，表示被选中'；如果确实无变化则为'页面完全无变化'）",  
        "state_direction": "状态变化方向（如：'未选中→已选中'；如不涉及状态切换则为'N/A'）",  
        "corrected_action": "修正后的操作描述",  
        "reason": "修改理由（如果未修改则必须为空字符串''）"  
    }}  
    输出规则：

    ⚠️ corrected_action 必须严格遵循上方「Action动作空间」的标准格式：
    - 点击类：「点击XX」「勾选XX」「选中XX」「取消勾选XX」「取消选中XX」「开启XX」「关闭XX」「长按XX」「双击XX」，XX只填目标元素名称，禁止附加描述性语句
    - 滑动类：「向上滑动」「向下滑动」「向左滑动」「向右滑动」，禁止附加修饰语（如"屏幕上方导航栏"）
    - 输入类：「输入文本XXX」

    如果原始action合理且状态方向正确：corrected_action = original_action（原样复制，不得改动），reason = ""
    如果原始action状态方向搞反：corrected_action = 正确的状态描述（遵循动作空间格式），reason = 说明状态搞反
    如果原始action不合理但页面有变化：corrected_action = 根据实际变化生成新action（必须遵循动作空间格式）
    只有确认页面完全无变化：corrected_action = "无操作"，detected_changes = "页面完全无变化"
    请直接输出JSON，不要添加其他说明文字。"""
    
    # 使用当前图和下一张图（确保两张图都存在且不同）
    images = [current_image_url, next_image_url]
    response = call_llm_with_prompt(
        prompt, 
        images=images, 
        temperature=0.1, 
        max_tokens=500,  # 增加token数量以支持JSON格式
        task_id=task_id, 
        step_name=current_step_name
    )
    
    if response:
        # 解析JSON响应
        result = parse_json_response(response)
        if result:
            corrected_action = result.get("corrected_action", preset_action)
            reason = result.get("reason", "")
            
            # 如果corrected_action为"无操作"，返回"无操作"
            if corrected_action == "无操作":
                return _action_validation_result(
                    preset_action,
                    "无操作",
                    corrected_action,
                    validation_reason=reason,
                    return_meta=return_meta,
                )
            
            # 如果reason为空字符串，说明没有修改，直接返回corrected_action（无需二次校验）
            if not reason or reason.strip() == "":
                return _action_validation_result(
                    preset_action,
                    corrected_action,
                    corrected_action,
                    validation_reason=reason,
                    return_meta=return_meta,
                )
            
            # 如果有修改理由，记录日志并进行二次校验
            print(f"  Action修正：{preset_action} -> {corrected_action}, 理由：{reason}")
            
            # 进行二次校验：对比修正前后的action，判断哪个更合理
            # 使用已传入的next_image_url（已经在函数开始时检查过）
            if next_image_url:
                final_action = double_check_action(
                    index, current_image_url, next_image_url,
                    preset_action, corrected_action, trajectory,
                    return_details=return_meta,
                )
                if isinstance(final_action, dict):
                    return _action_validation_result(
                        preset_action,
                        final_action.get("action", corrected_action),
                        corrected_action,
                        validation_reason=reason,
                        double_check_choice=final_action.get("choice"),
                        double_check_reason=final_action.get("reason", ""),
                        return_meta=return_meta,
                    )
            else:
                # 如果没有下一张图，无法二次校验，返回修正后的action
                final_action = corrected_action

            return _action_validation_result(
                preset_action,
                final_action,
                corrected_action,
                validation_reason=reason,
                return_meta=return_meta,
            )
        else:
            # JSON解析失败，尝试直接提取文本（兼容旧格式）
            corrected_action = response.strip()
            # 移除可能的引号
            if corrected_action.startswith('"') and corrected_action.endswith('"'):
                corrected_action = corrected_action[1:-1]
            # 如果包含JSON格式的提示，尝试提取
            if "corrected_action" in response:
                try:
                    import json
                    import re
                    # 尝试提取JSON部分
                    json_match = re.search(r'\{[^{}]*"corrected_action"[^{}]*\}', response)
                    if json_match:
                        result = json.loads(json_match.group())
                        extracted_action = result.get("corrected_action", preset_action)
                        return _action_validation_result(
                            preset_action,
                            extracted_action,
                            extracted_action,
                            validation_reason="JSON解析失败，使用兼容提取",
                            return_meta=return_meta,
                        )
                except:
                    pass
            return _action_validation_result(
                preset_action,
                corrected_action,
                corrected_action,
                validation_reason="JSON解析失败，使用原始响应文本",
                return_meta=return_meta,
            )
    
    # Fallback：返回预设action
    return _action_validation_result(
        preset_action, preset_action, preset_action, return_meta=return_meta
    )


def double_check_action(
    index: int,
    current_image_url: str,
    next_image_url: str,
    original_action: str,
    corrected_action: str,
    trajectory: List[Dict],
    return_details: bool = False,
) -> Union[str, Dict[str, Any]]:
    """
    二次校验action：对比修正前后的action，判断哪个更合理
    
    只在第一次校验修改了action时调用（reason不为空）
    
    Args:
        index: 当前步骤索引
        current_image_url: 当前图片URL（步骤index的图片，操作前的状态）
        next_image_url: 下一张图片URL（步骤index+1的图片，操作后的状态）
        original_action: 修正前的action
        corrected_action: 修正后的action
        trajectory: 操作序列
        
    Returns:
        str: 更合理的action（original_action或corrected_action）
    """
    task_id, step_name = get_task_context()
    current_step_name = f"{step_name}_double_check_action" if step_name else None
    
    # 如果没有下一张图，无法校验，返回修正后的action
    if not next_image_url or index >= len(trajectory) - 1:
        return _double_check_result(corrected_action, return_details=return_details)
    
    # 确保当前图和下一张图不同
    if current_image_url == next_image_url:
        return _double_check_result(corrected_action, return_details=return_details)
    
    # 如果修正后的action和原始action相同，无需二次校验
    if original_action == corrected_action:
        return _double_check_result(corrected_action, return_details=return_details)
    
    # 如果修正后的action是"无操作"，直接返回
    if corrected_action == "无操作":
        return _double_check_result(corrected_action, return_details=return_details)
    
    # 获取上下文信息
    # context = get_context_info(index, trajectory)
    
    prompt = f"""你是一个GUI操作校验专家。请仔细对比前后两张截图，判断下面两个action描述哪个更合理。

**重要：图片说明**：
- 第一张图（current_image）：表示**执行操作前的页面状态**
- 第二张图（next_image）：表示**执行操作后的页面状态**
- 需要判断：哪个action能更好地解释从第一张图到第二张图的变化
- 原则：优先选择 Action 1，除非 Action 2 **非常明显地更合理**，才能选择 Action 2

**需要对比的两个action**：
- Action 1：{original_action}
- Action 2：{corrected_action}

**判断标准**：   
- 先尝试解释 Action 1 导致页面变化的原因，这个原因一定程度解释得通，直接选择 1
- 否则，才考虑 Action 2。如果 Action 2 **非常明显地更合理**，才能选择 2
- 如果不确定，即使 Action 2 解释得更好，还是优先选择 1
- 特例1：如果是弹窗消失，直接选择 1，跳过判定   
- 特例2：如果 Action 1 是播放或暂停类的点击，直接选择 1，跳过判定

**输出要求**：只输出编号 1 或 2，不要输出action的具体内容。

输出格式（必须是有效的JSON）：
{{
    "choice": 1,
    "reason": "选择理由"
}}

请直接输出JSON，不要添加其他说明文字。"""
    
    # 使用当前图和下一张图
    images = [current_image_url, next_image_url]
    response = call_llm_with_prompt(
        prompt, 
        images=images, 
        temperature=0.1, 
        max_tokens=300,
        task_id=task_id, 
        step_name=current_step_name
    )
    
    if response:
        # 解析JSON响应
        result = parse_json_response(response)
        if result:
            choice = result.get("choice", 1)
            reason = result.get("reason", "")
            
            # 确保 choice 是整数
            try:
                choice = int(choice)
            except (ValueError, TypeError):
                choice = 1
            
            if choice == 2:
                if reason:
                    print(f"  二次校验：确认修正后的action，理由：{reason}")
                return _double_check_result(
                    corrected_action, choice=choice, reason=reason, return_details=return_details
                )
            else:
                if reason:
                    print(f"  二次校验：选择修正前的action，理由：{reason}")
                return _double_check_result(
                    original_action, choice=choice, reason=reason, return_details=return_details
                )
        else:
            # JSON解析失败，优先返回原始action
            print(f"  警告：二次校验JSON解析失败，使用原始action")
            return _double_check_result(
                original_action,
                reason="二次校验JSON解析失败，使用原始action",
                return_details=return_details,
            )
    
    # Fallback：返回原始action（优先保守）
    return _double_check_result(original_action, return_details=return_details)


def get_context_info(index: int, trajectory: List[Dict]) -> str:
    """获取上下文信息"""
    context_parts = []
    
    # 前一步操作
    if index > 0:
        prev_action = trajectory[index - 1].get("value", {}).get("action", "")
        if prev_action:
            context_parts.append(f"前一步操作：{prev_action}")
    
    # 后一步操作
    if index < len(trajectory) - 1:
        next_action = trajectory[index + 1].get("value", {}).get("action", "")
        if next_action:
            context_parts.append(f"后一步操作：{next_action}")
    
    # 总步骤数
    context_parts.append(f"总步骤数：{len(trajectory)}")
    context_parts.append(f"当前步骤：{index + 1}/{len(trajectory)}")
    
    return "\n".join(context_parts) if context_parts else "无上下文信息"


def batch_generate_action_thought_caption(trajectory: List[Dict], batch_size: int = 5) -> List[Dict]:
    """
    批量生成action、thought和caption（可选优化）
    如果轨迹很长，可以批量处理以提高效率
    """
    if len(trajectory) <= batch_size:
        return generate_action_thought_caption(trajectory)
    
    # 分批处理
    results = []
    for i in range(0, len(trajectory), batch_size):
        batch = trajectory[i:i + batch_size]
        batch_results = generate_action_thought_caption(batch)
        results.extend(batch_results)
    
    return results


if __name__ == "__main__":
    # 测试
    test_trajectory = [
        {
            "image": "http://example.com/000.png",
            "value": {"action": "点击酒店", "code": "tap(329, 121)"}
        },
        {
            "image": "http://example.com/001.png",
            "value": {"action": "选择日期", "code": "tap(153, 333)"}
        }
    ]
    
    results = generate_action_thought_caption(test_trajectory)
    for result in results:
        print(f"步骤{result['index']}:")
        print(f"  Action: {result['action']}")
        print(f"  Thought: {result['thought']}")
        print(f"  Caption: {result['sparse_caption']}")
        print()
