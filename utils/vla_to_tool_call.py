"""
VLA字符串到Qwen Tool Call转换模块

将VLA格式的动作字符串转换为Qwen Tool Call格式的字典。
参考 convert_vla_str_to_label_action 的输入格式和 function_call_mobile_answer.py 的动作空间定义。

# 完整的动作映射关系

## 基础动作映射
tap(x, y) -> {"action": "click", "coordinate": [x, y]}
long_press(x, y) -> {"action": "long_press", "coordinate": [x, y], "time": 0.8}
scroll(x, y, "direction") / scroll(x, y, "direction", distance) -> {"action": "swipe", "coordinate": [x, y], "coordinate2": [end_x, end_y]}
drag(x1, y1, x2, y2) -> {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2]}

## 复合动作映射(需要特殊处理)
tap(x, y, "text") -> 返回列表, 包含两个 tool_call:
    {"action": "click", "coordinate": [x, y]}
    {"action": "type", "text": "text"}
text(x, y, "text") -> 返回列表, 包含两个 tool_call:
    {"action": "click", "coordinate": [x, y]}
    {"action": "type", "text": "text"}

## 系统动作映射
navigate_back() -> {"action": "system_button", "button": "Back"}
navigate_home() -> {"action": "system_button", "button": "Home"}
enter() -> {"action": "system_button", "button": "Enter"}

## 其他动作映射
wait() / wait("message") / wait(3) -> {"action": "wait", "time": 1.0} (默认1秒, 如果message中有数字则解析为时间)
terminate() / terminate("message") -> {"action": "terminate", "status": "failure"} (默认failure, 可通过参数指定success)
finish() -> {"action": "terminate", "status": "success"} 或返回 None(根据需求决定)
call_api("启动", "微信") -> {"action": "launch_app", "app_name": "微信"} (取第二个参数作为app名称)
launch_app("app_name") -> {"action": "launch_app", "app_name": "app_name"}
take_over("message") -> {"action": "answer", "text": "message"} 或需要其他处理
"""

import re
import logging
from typing import Dict, List, Any, Union, Tuple

from ..parsers.vla_parser import VLAParser
from ..core.constants import API_LAUNCH_KEYWORD
from ..coords.coordinate_utils import normalize_label_action_coordinates

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_VLA_PARSER = VLAParser()


def _label_action_to_tool_call(action_dict: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """Convert a label action dictionary to tool_call format.

    This function handles the conversion from label action format to tool_call format,
    eliminating the need for duplicate regex parsing.

    Args:
        action_dict: Label action dictionary with event_type, info, etc.

    Returns:
        Tool call dictionary or list, or None if conversion fails
    """
    # Handle empty or missing action_dict, or empty event_type (e.g., from unknown actions)
    # Empty string "" is falsy, so this check handles both None and empty strings
    if not action_dict or not action_dict.get("event_type"):
        return None

    event_type = action_dict.get("event_type")
    info = action_dict.get("info", {})
    text = action_dict.get("text", "")
    message = action_dict.get("message", "")

    # Handle compound actions: tap/text with text input
    if event_type == "tap" and text:
        if "action_touch_xy" in info:
            x, y = info["action_touch_xy"]
            return [
                {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [x, y]}},
                {"name": "mobile_use", "arguments": {"action": "type", "text": text}}
            ]

    # Handle text action (event_type == "text")
    if event_type == "text":
        if "action_touch_xy" in info:
            x, y = info["action_touch_xy"]
            if text:
                # Return list with click and type actions
                return [
                    {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [x, y]}},
                    {"name": "mobile_use", "arguments": {"action": "type", "text": text}}
                ]
            else:
                # Just click if no text provided
                return {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [x, y]}}

    # Handle input action without coordinates
    if event_type == "input":
        input_text = text or info.get("text", "")
        if "action_touch_xy" in info:
            x, y = info["action_touch_xy"]
            return [
                {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [x, y]}},
                {"name": "mobile_use", "arguments": {"action": "type", "text": input_text}}
            ]
        else:
            return {"name": "mobile_use", "arguments": {"action": "type", "text": input_text}}

    # Handle tap and long_press
    if event_type == "tap":
        if "action_touch_xy" in info:
            x, y = info["action_touch_xy"]
            return {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [x, y]}}

    if event_type == "long_press":
        if "action_touch_xy" in info:
            x, y = info["action_touch_xy"]
            return {"name": "mobile_use", "arguments": {"action": "long_press", "coordinate": [x, y], "time": 0.8}}

    # Handle scroll/drag actions
    if event_type in ("scroll", "drag"):
        if "action_touch_xy" in info and "action_lift_xy" in info:
            start_x, start_y = info["action_touch_xy"]
            end_x, end_y = info["action_lift_xy"]
            return {"name": "mobile_use", "arguments": {"action": "swipe", "coordinate": [start_x, start_y], "coordinate2": [end_x, end_y]}}

    # Handle system actions
    if event_type == "navigate_back":
        return {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Back"}}

    if event_type == "navigate_home":
        return {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Home"}}

    if event_type == "enter":
        return {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Enter"}}

    if event_type == "back":
        return {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Back"}}

    if event_type == "menu":
        return {"name": "mobile_use", "arguments": {"action": "system_button", "button": "Menu"}}

    # Handle wait
    if event_type == "wait":
        wait_message = message
        arguments: Dict[str, Any] = {"action": "wait"}

        # 如果有显式数值（参数或消息中包含数字），则解析为 time，并保留原始文本
        if isinstance(wait_message, (int, float)):
            wait_seconds = float(wait_message)
            arguments["time"] = wait_seconds
            arguments["text"] = str(wait_message)
        elif isinstance(wait_message, str) and wait_message.strip():
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", wait_message)
            if match:
                try:
                    wait_seconds = float(match.group(1))
                    arguments["time"] = wait_seconds
                except ValueError:
                    pass
            # 无论是否解析到 time，都保留 message 文本
            arguments["text"] = wait_message

        return {"name": "mobile_use", "arguments": arguments}

    # Handle terminate (including Ali format with status and answer)
    if event_type == "terminate":
        # Check if status/answer are provided at top level (Ali format)
        status_value = action_dict.get("status", "failure")
        answer_value = action_dict.get("answer", "")

        # If no explicit status, check message for implicit status
        if "status" not in action_dict and message:
            status_raw = str(message).strip().lower()
            if status_raw in ["success", "ok", "done"]:
                status_value = "success"
            else:
                status_value = "failure"

        # Normalize status: "fail" -> "failure"
        if status_value == "fail":
            status_value = "failure"

        arguments = {"action": "terminate", "status": status_value}
        if answer_value:
            arguments["answer"] = answer_value
        return {"name": "mobile_use", "arguments": arguments}

    # Handle finish (maps to terminate with success)
    if event_type == "finish":
        return {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}

    # Handle call_api
    # call_api is a generic tool call interface: first param is tool name/intent, second is tool params
    if event_type == "call_api":
        if not isinstance(message, list):
            logger.warning(
                f"Invalid call_api format: expected message to be a list, "
                f"got {type(message).__name__} with value: {message}"
            )
            return None

        # Filter out empty strings from the list
        message = [arg for arg in message if arg and str(arg).strip()]

        if len(message) < 1:
            logger.warning(
                f"Invalid call_api format: expected message to be a list with at least 1 element "
                f"(tool name/intent), got empty list"
            )
            return None

        tool_name = str(message[0]).strip()
        if not tool_name:
            logger.warning(
                f"Invalid call_api: tool name is empty. Full message: {message}"
            )
            return None

        # Handle Ali format: call_api(app_name, "open/close") -> launch_app
        # Check if second param is "open", "close", or "open/close" and treat app_name as tool_name
        if len(message) >= 2:
            param2 = str(message[1]).strip().strip('"').strip("'")
            # Match "open", "close", or "open/close" (after stripping quotes)
            if param2 in ["open", "close", "open/close"]:
                # Ali format: call_api(app_name, "open") -> launch app
                app_name = tool_name
                if not app_name:
                    logger.warning(f"Invalid call_api: app name is empty. Full message: {message}")
                    return None

                return {"name": "mobile_use", "arguments": {"action": "launch_app", "app_name": app_name}}

        # Handle launch app case (most common): call_api("启动", "app_name")
        if tool_name == API_LAUNCH_KEYWORD:
            if len(message) < 2:
                logger.warning(
                    f"Invalid call_api for launch: expected at least 2 elements "
                    f"(e.g., ['{API_LAUNCH_KEYWORD}', 'app_name']), got {len(message)} element(s): {message}"
                )
                return None
            app_name = str(message[1]).strip()
            if not app_name:
                logger.warning(
                    f"Invalid call_api: app name is empty. Full message: {message}"
                )
                return None
            return {"name": "mobile_use", "arguments": {"action": "launch_app", "app_name": app_name}}

        # For other tool names, map to unknown_action to preserve data
        # This prevents data loss and allows downstream processing
        logger.warning(
            f"[VLAToToolCall] Converting to unknown_action - "
            f"source=conversion_stage, tool_name={tool_name}, "
            f"message={message}, full_action_dict={action_dict}"
        )
        # Map to unknown_action with tool_name and params
        params = message[1:] if len(message) > 1 else []
        unknown_msg = f"tool_name={tool_name}"
        if params:
            unknown_msg += f", params={params}"
        return {"name": "mobile_use", "arguments": {"action": "unknown_action", "message": unknown_msg}}

    # Handle take_over
    if event_type == "take_over":
        if message:
            return {"name": "mobile_use", "arguments": {"action": "answer", "text": str(message)}}

    return None


def convert_vla_str_to_tool_call(
    action_code_or_action: Union[str, Dict[str, Any]],
    *,
    image_size: Tuple[int, int] | None = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """
    Convert an action code string or action dictionary into a tool_call dictionary.

    This is a unified entry function to handle multiple input formats:
    - action_code string (e.g. "tap(393, 2603)")
    - action dictionary (Jarvis format)
    - already in tool_call dictionary format

    For text(x, y, "text") and tap(x, y, "text"), returns a list containing two tool_calls:
    - The first: click at the coordinate
    - The second: input the text

    Args:
        action_code_or_action: Action code string or action dictionary

    Returns:
        A tool_call dictionary or list, in the format {"name": "mobile_use", "arguments": {...}}.
        For compound actions (text/tap with text), returns a list; if unable to parse, returns None.
    """
    if not action_code_or_action:
        return None

    # 如果已经是dict格式，检查是否是tool_call格式
    if isinstance(action_code_or_action, dict):
        if "name" in action_code_or_action and "arguments" in action_code_or_action:
            return action_code_or_action
        # If it's a label action dict, convert it
        return _label_action_to_tool_call(action_code_or_action)

    # 处理action_code字符串 - 使用统一的解析器
    if not isinstance(action_code_or_action, str):
        return None

    # Use the unified parser to convert string to label action
    label_action = _VLA_PARSER.parse(action_code_or_action)

    # Convert label action to tool_call
    # parser.parse() may return None, so check before converting
    if label_action is None:
        return None

    if isinstance(label_action, dict) and image_size:
        try:
            width, height = int(image_size[0]), int(image_size[1])
            if width > 0 and height > 0:
                label_action = normalize_label_action_coordinates(label_action, width, height)
        except Exception:
            pass
    return _label_action_to_tool_call(label_action)


if __name__ == "__main__":
    """
    python -m task.action_space.converters.vla_to_tool_call

    注意：测试代码已移动到 tests/test_task/test_action/test_vla_to_tool_call.py
    运行测试请使用：
        pytest tests/test_task/test_action/test_vla_to_tool_call.py
    或：
        python -m pytest tests/test_task/test_action/test_vla_to_tool_call.py
    """
    pass
