"""
将batchID格式的jsonl文件转换为query格式的jsonl文件

========================================
字段映射关系（batchID格式 -> query格式）
========================================

【直接映射】
- task_id -> task_id
- task -> instruction

【转换映射】
- history_image_pairs[] -> traj[]
  * action_describe -> value.action (直接映射)
  * image_path -> image (直接映射)
  * index -> 从0开始顺序自动生成（不依赖step_num）
  * value.code -> 优先从meta_action推断，如果没有或推断失败，则从action_describe智能推断
  * meta_action -> 用于推断value.code（新格式字段，如果存在则优先使用）

【缺失字段（置空）】
- created_at -> 使用当前时间戳（格式：YYYY-MM-DD HH:MM:SS）
- tags -> [] (空列表)
- activity.activity_name -> "" (空字符串)
- activity.package_name -> "" (空字符串)

【保留的字段（batchID格式独有，直接继承）】
- app: 应用名称
- historys: 操作历史列表（如果history_image_pairs存在则不用）
- images: 图像路径列表（如果history_image_pairs存在则不用）
- think: 思考内容
- response: 仅支持最新格式 dict，必须包含 is_task_completed、is_trajectory_correct（正确拼写），用于生成 evaluation，输出中不保留 response
- max_step_idx: 最大步骤索引
- max_step_content: 最大步骤内容
- dag: DAG信息
- takeover_step_idx: 接管步骤索引数组（新格式字段，如 [5] 或 [5, 7]）
- 其他所有字段：全部按照原字段格式直接继承到新数据中

【value.code推断规则】
优先使用meta_action推断（如果存在）：
- touch(x, y) -> tap(x, y)
- set_text(x, y),内容 -> text(x, y, "内容")
- open_app应用名 -> call_api("启动", "应用名")
- take_over信息缺失 -> take_over("信息缺失")
- 其他meta_action格式 -> 尝试解析或回退到action_describe推断

如果meta_action不存在或无法解析，使用action_describe推断：
- 包含"点击" -> tap(...) 或 tap(x, y)（如果描述中有坐标）
- 包含"输入" -> text("内容") 或 text(...)
- 包含"滑动" -> scroll(..., "方向", ...)
- 包含"启动"/"打开" -> call_api("启动", "应用名")
- 包含"结束"/"任务结束" -> finish()
- 其他 -> "" (空字符串)

【示例转换】

batchID格式（新格式支持meta_action字段）：
{
  "task_id": "1769067579239TDBYQnlX",
  "task": "在高德地图打开牛人严选热度榜",
  "history_image_pairs": [
    {
      "step_num": 1,
      "meta_action": "open_app高德地图",
      "action_describe": "打开高德地图",
      "image_path": "/path/to/image1.png"
    },
    {
      "step_num": 2,
      "meta_action": "touch(675,78)",
      "action_describe": "点击更多工具",
      "image_path": "/path/to/image2.png"
    }
  ],
  "response": {"is_task_completed": 0, "is_trajectory_correct": 1},
  "takeover_step_idx": [5]
}

转换为query格式：
{
  "created_at": "2026-01-21 15:30:00",
  "instruction": "在高德地图打开牛人严选热度榜",
  "tags": [],
  "task_id": "1769067579239TDBYQnlX",
  "traj": [
    {
      "activity": {"activity_name": "", "package_name": ""},
      "image": "/path/to/image1.png",
      "index": 0,
      "value": {
        "action": "打开高德地图",
        "code": "call_api(\"启动\", \"高德地图\")"
      }
    },
    {
      "activity": {"activity_name": "", "package_name": ""},
      "image": "/path/to/image2.png",
      "index": 1,
      "value": {
        "action": "点击更多工具",
        "code": "tap(675, 78)"
      }
    }
  ],
  "evaluation": {
    "is_task_completed": 0,
    "is_trajectory_correct": 1
  }
}
"""

import json
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

# #region agent log
_DEBUG_LOG_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "debug-0160e6.log"))
_DEBUG_GID_CALLS = 0
_DEBUG_GID_TIME = 0.0
_DEBUG_PIL_USED = 0
_DEBUG_CV2_USED = 0


def _debug_write(payload: dict) -> None:
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({**payload, "sessionId": "0160e6", "timestamp": int(time.time() * 1000)}, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion


def get_image_dimensions(image_path: str) -> Tuple[Optional[int], Optional[int]]:
    """
    读取图片的宽高尺寸。优先用 PIL 只读文件头取尺寸（快），失败时回退到 OpenCV 全图解码。
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        (width, height) 元组，如果读取失败返回 (None, None)
    """
    # #region agent log
    t0 = time.perf_counter()
    try:
    # #endregion
        if not image_path or not os.path.exists(image_path):
            return None, None
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
            # #region agent log
            global _DEBUG_PIL_USED
            _DEBUG_PIL_USED += 1
            # #endregion
            return width, height
        except Exception:
            try:
                image_data = np.fromfile(image_path, dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if image is None:
                    return None, None
                height, width = image.shape[:2]
                # #region agent log
                global _DEBUG_CV2_USED
                _DEBUG_CV2_USED += 1
                # #endregion
                return width, height
            except Exception:
                return None, None
    # #region agent log
    finally:
        global _DEBUG_GID_CALLS, _DEBUG_GID_TIME
        _DEBUG_GID_CALLS += 1
        _DEBUG_GID_TIME += time.perf_counter() - t0
    # #endregion


def convert_coordinate(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    """
    将千分坐标转换为原始坐标
    
    Args:
        x: 千分坐标 x (0-1000)
        y: 千分坐标 y (0-1000)
        width: 图片宽度
        height: 图片高度
        
    Returns:
        (x_new, y_new) 转换后的原始坐标
    """
    x_new = int(round(x * width / 1000))
    y_new = int(round(y * height / 1000))
    return x_new, y_new


def infer_action_code(action_describe: str, image_width: Optional[int] = None, image_height: Optional[int] = None) -> str:
    """
    从action描述推断action code
    这是一个简单的推断，可以根据实际情况扩展
    
    Args:
        action_describe: 操作描述文本
        image_width: 图片宽度（用于坐标转换）
        image_height: 图片高度（用于坐标转换）
        
    Returns:
        action code字符串，如果无法推断则返回空字符串
    """
    if not action_describe:
        return ''
    
    action_describe = action_describe.strip()
    action_lower = action_describe.lower()
    
    # 如果包含"点击"，尝试提取坐标（如果有）
    if "点击" in action_describe:
        # 尝试从描述中提取坐标（如果有类似"点击(100, 200)"的格式）
        import re
        coord_match = re.search(r'\((\d+),\s*(\d+)\)', action_describe)
        if coord_match:
            x, y = int(coord_match.group(1)), int(coord_match.group(2))
            # 如果提供了图片尺寸，转换坐标
            if image_width is not None and image_height is not None:
                x, y = convert_coordinate(x, y, image_width, image_height)
            return f'tap({x}, {y})'
        return 'tap(...)'  # 无法推断具体坐标，使用占位符
    
    elif "输入" in action_describe:
        # 尝试提取输入内容
        if '"' in action_describe:
            # 提取引号中的内容
            import re
            text_match = re.search(r'["\']([^"\']+)["\']', action_describe)
            if text_match:
                content = text_match.group(1)
                return f'text("{content}")'
        # 尝试提取"输入"后面的内容
        if "输入" in action_describe:
            parts = action_describe.split("输入")
            if len(parts) > 1:
                content = parts[-1].strip().strip('"').strip("'")
                if content:
                    return f'text("{content}")'
        return 'text(...)'
    
    elif "滑动" in action_describe:
        # 尝试推断滑动方向和参数
        direction = ""
        if "向上" in action_describe or "up" in action_lower:
            direction = "up"
        elif "向下" in action_describe or "down" in action_lower:
            direction = "down"
        elif "向左" in action_describe or "left" in action_lower:
            direction = "left"
        elif "向右" in action_describe or "right" in action_lower:
            direction = "right"
        
        if direction:
            return f'scroll(..., "{direction}", ...)'
        return 'scroll(...)'
    
    elif "启动" in action_describe or "打开" in action_describe:
        # 尝试提取应用名
        if "打开" in action_describe:
            parts = action_describe.split("打开")
            if len(parts) > 1:
                app_name = parts[-1].strip()
                return f'call_api("启动", "{app_name}")'
        elif "启动" in action_describe:
            parts = action_describe.split("启动")
            if len(parts) > 1:
                app_name = parts[-1].strip()
                return f'call_api("启动", "{app_name}")'
        return 'call_api("启动", "...")'
    
    elif "任务结束" in action_describe or "结束" in action_describe or "finish" in action_lower:
        return 'finish()'
    
    # 接管场景检测（如果 meta_action 没有提供，从 action_describe 推断）
    elif any(keyword in action_describe for keyword in ["接管", "信息缺失", "验证码", "支付", "确认", "人机验证", "请接管"]):
        # 提取接管原因
        reason = action_describe.strip()
        # 如果包含括号，尝试提取括号内容
        import re
        paren_match = re.search(r'[（(]([^）)]+)[）)]', action_describe)
        if paren_match:
            reason = paren_match.group(1).strip()
        return f'take_over("{reason}")'
    
    else:
        return ''  # 无法推断，置空


def infer_action_code_from_meta(meta_action: str, image_width: Optional[int] = None, image_height: Optional[int] = None) -> Optional[str]:
    """
    从meta_action格式推断action code
    meta_action格式示例：
    - touch(675,78) -> tap(675, 78)
    - set_text(340,127),一加ace5 -> text(340, 127, "一加ace5")
    - open_app抖音 -> call_api("启动", "抖音")
    - take_over信息缺失 -> take_over("信息缺失")
    
    Args:
        meta_action: meta_action格式的字符串
        image_width: 图片宽度（用于坐标转换）
        image_height: 图片高度（用于坐标转换）
        
    Returns:
        action code字符串，如果无法解析则返回None（让回退逻辑处理）
    """
    if not meta_action:
        return None
    
    meta_action = meta_action.strip()
    import re
    
    # touch(x, y) -> tap(x, y)
    touch_match = re.match(r'^touch\((\d+),\s*(\d+)\)$', meta_action)
    if touch_match:
        x, y = int(touch_match.group(1)), int(touch_match.group(2))
        # 如果提供了图片尺寸，转换坐标
        if image_width is not None and image_height is not None:
            x, y = convert_coordinate(x, y, image_width, image_height)
        return f'tap({x}, {y})'
    
    # tap(x, y) -> tap(x, y) (需要转换坐标)
    tap_match = re.match(r'^tap\((\d+),\s*(\d+)\)$', meta_action)
    if tap_match:
        x, y = int(tap_match.group(1)), int(tap_match.group(2))
        # 如果提供了图片尺寸，转换坐标
        if image_width is not None and image_height is not None:
            x, y = convert_coordinate(x, y, image_width, image_height)
        return f'tap({x}, {y})'
    
    # set_text(x, y, 内容) 或 set_text(x, y),内容 -> text(x, y, "内容")
    set_text_match = re.match(r'^set_text\((\d+),\s*(\d+)\)(?:,\s*)?(.+)?$', meta_action)
    if set_text_match:
        x, y = int(set_text_match.group(1)), int(set_text_match.group(2))
        # 如果提供了图片尺寸，转换坐标
        if image_width is not None and image_height is not None:
            x, y = convert_coordinate(x, y, image_width, image_height)
        content = set_text_match.group(3)
        if content:
            content = content.strip().strip('"').strip("'")
            return f'text({x}, {y}, "{content}")'
        else:
            return f'text({x}, {y}, ...)'
    
    # text(x, y, 内容) -> text(x, y, "内容")
    text_match = re.match(r'^text\((\d+),\s*(\d+),\s*(.+)\)$', meta_action)
    if text_match:
        x, y = int(text_match.group(1)), int(text_match.group(2))
        # 如果提供了图片尺寸，转换坐标
        if image_width is not None and image_height is not None:
            x, y = convert_coordinate(x, y, image_width, image_height)
        content = text_match.group(3)
        content = content.strip().strip('"').strip("'")
        return f'text({x}, {y}, "{content}")'
    
    # open_app应用名 -> call_api("启动", "应用名")
    open_app_match = re.match(r'^open_app(.+)$', meta_action)
    if open_app_match:
        app_name = open_app_match.group(1).strip()
        return f'call_api("启动", "{app_name}")'
    
    # call_api(应用名, open) -> call_api("启动", "应用名")
    call_api_match = re.match(r'^call_api\((.+),\s*open\)$', meta_action)
    if call_api_match:
        app_name = call_api_match.group(1).strip().strip('"').strip("'")
        return f'call_api("启动", "{app_name}")'
    
    # scroll(x, y, 方向) 或 scroll(x, y),方向 -> scroll(x, y, "方向")
    # 支持格式：
    #   - scroll(560,348,left) -> scroll(560, 348, "left")
    #   - scroll(400, 800, "up") -> scroll(400, 800, "up")
    #   - scroll(865,694),up -> scroll(865, 694, "up")
    #   - scroll(500, 200, "left") -> scroll(500, 200, "left")
    
    # 先尝试匹配标准格式：scroll(x, y, direction) 或 scroll(x, y, "direction")
    scroll_match = re.match(r'^scroll\((\d+),\s*(\d+)(?:,\s*)?([^)]+)?\)$', meta_action)
    if scroll_match:
        x, y = int(scroll_match.group(1)), int(scroll_match.group(2))
        # 如果提供了图片尺寸，转换坐标
        if image_width is not None and image_height is not None:
            x, y = convert_coordinate(x, y, image_width, image_height)
        direction = scroll_match.group(3)
        if direction:
            direction = direction.strip().strip('"').strip("'").lower()
            if direction in ['up', 'down', 'left', 'right', '向上', '向下', '向左', '向右']:
                direction_map = {
                    'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
                    '向上': 'up', '向下': 'down', '向左': 'left', '向右': 'right'
                }
                dir_en = direction_map.get(direction, direction)
                return f'scroll({x}, {y}, "{dir_en}")'
        return f'scroll({x}, {y}, ...)'
    
    # scroll(x, y),方向（方向在括号外，用逗号分隔）
    scroll_match_old = re.match(r'^scroll\((\d+),\s*(\d+)\)(?:,\s*)?(.+)?$', meta_action)
    if scroll_match_old:
        x, y = int(scroll_match_old.group(1)), int(scroll_match_old.group(2))
        # 如果提供了图片尺寸，转换坐标
        if image_width is not None and image_height is not None:
            x, y = convert_coordinate(x, y, image_width, image_height)
        direction = scroll_match_old.group(3)
        if direction:
            direction = direction.strip().strip('"').strip("'").lower()
            if direction in ['up', 'down', 'left', 'right', '向上', '向下', '向左', '向右']:
                direction_map = {
                    'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
                    '向上': 'up', '向下': 'down', '向左': 'left', '向右': 'right'
                }
                dir_en = direction_map.get(direction, direction)
                return f'scroll({x}, {y}, "{dir_en}")'
        return f'scroll({x}, {y}, ...)'
    
    # take_over信息缺失 -> take_over("信息缺失")
    # take_over(请确认是否下订单) -> take_over("请确认是否下订单")
    # take_over(请确认付款) -> take_over("请确认付款")
    take_over_match = re.match(r'^take_over(?:\((.+)\))?(.+)?$', meta_action)
    if take_over_match:
        param1 = take_over_match.group(1)  # 括号内的参数
        param2 = take_over_match.group(2)  # 括号外的参数（如"信息缺失"）
        
        # 优先使用括号内的参数
        if param1:
            param1 = param1.strip().strip('"').strip("'")
            return f'take_over("{param1}")'
        elif param2:
            param2 = param2.strip()
            return f'take_over("{param2}")'
        else:
            return 'take_over(...)'
    
    # 如果无法解析，返回None让回退逻辑处理
    return None


def is_takeover_step(meta_action: str, action_describe: str = "") -> bool:
    """
    判断是否为接管步骤
    
    Args:
        meta_action: meta_action 字符串
        action_describe: action_describe 字符串（当 meta_action 为空时用于兜底判断）
        
    Returns:
        bool: 是否为接管步骤
    """
    if meta_action and meta_action.strip().startswith("take_over"):
        return True
    # 兜底：meta_action 为空但 action_describe 以"接管"开头，或包含"场景接管"
    if action_describe:
        desc = action_describe.strip()
        if desc.startswith("接管") or "场景接管" in desc:
            return True
    return False


def infer_takeover_category(meta_action: str, action_describe: str) -> str:
    """
    推断接管步骤的 category
    
    Args:
        meta_action: meta_action 字符串
        action_describe: action_describe 字符串
        
    Returns:
        str: category 字符串
    """
    # 合并 meta_action 和 action_describe 用于关键词匹配
    text = (meta_action or "") + " " + (action_describe or "")
    
    # ── 接管类型列表（优先级从高到低）──
    # 来源：平台标准接管类别
    #   页面限制读取 / 涉及钱 / 涉及账号管理 / 涉及数据 / 涉及法律和授权
    #   涉及安全 / 确认弹窗 / 登录 / 密码输入 / 密码填写 / 生物验证
    #   验证码填写 / 涉及订单 / 涉及社交 / 涉及内容确认 / 人机验证
    #   求助 / 删除确认
    
    # 人机验证（滑块 / 拼图 / 验证码弹窗）
    if any(kw in text for kw in ["人机验证", "人机", "滑块", "拼图"]):
        return "人机验证"
    
    # 验证码填写（短信验证码、图片验证码输入）
    if any(kw in text for kw in ["验证码", "验证码填写", "输入验证码"]):
        return "验证码填写"
    
    # 生物验证（指纹、人脸）
    if any(kw in text for kw in ["指纹", "人脸", "生物验证", "面容"]):
        return "生物验证"
    
    # 登录
    if any(kw in text for kw in ["登录", "登陆", "注册"]):
        return "登录"
    
    # 密码输入 / 密码填写
    if any(kw in text for kw in ["密码输入", "输入密码", "密码填写", "填写密码"]):
        return "密码输入"
    
    # 删除确认
    if "删除" in text:
        return "删除确认"
    
    # 涉及钱（支付、付款、充值、购买、下单、抢购、退款、退货）
    if any(kw in text for kw in ["支付", "付款", "充值", "购买", "下单", "抢购", "退款", "退货", "涉及钱"]):
        return "涉及钱"
    
    # 涉及订单（订单、预订、预约、结算、提交订单）
    if any(kw in text for kw in ["订单", "预订", "预约", "结算", "提交", "涉及订单"]):
        return "涉及订单"
    
    # 涉及法律和授权（协议、授权、同意、隐私）
    if any(kw in text for kw in ["协议", "授权", "同意", "隐私", "法律"]):
        return "涉及法律和授权"
    
    # 涉及安全（权限、安全、允许）
    if any(kw in text for kw in ["权限", "安全", "允许", "仅使用期间"]):
        return "涉及安全"
    
    # 涉及账号管理（注销、注册、退出登录、账号、账号封禁）
    if any(kw in text for kw in ["注销", "退出登录", "账号管理", "绑定", "账号封禁", "封禁"]):
        return "涉及账号管理"
    
    # 涉及社交（发送消息、评论、发表、分享、关注、点赞）
    if any(kw in text for kw in ["发送", "评论", "发表", "分享", "关注", "点赞", "消息"]):
        return "涉及社交"
    
    # 涉及内容确认（确认、发布、上传、提交建议）
    if any(kw in text for kw in ["发布", "上传", "内容确认"]):
        return "涉及内容确认"
    
    # 页面限制读取（无法加载、未找到、未安装、网络异常、加载失败、暂无内容、距离较远）
    if any(kw in text for kw in ["页面限制", "无法加载", "网络异常", "加载失败", "未找到", 
                                  "未安装", "没有安装", "暂无内容", "距离较远", "打卡失败", 
                                  "无法继续", "查看详情场景接管"]):
        return "页面限制读取"
    
    # 涉及订单（订单、预订、预约、结算、提交订单、选择商品、规格信息缺失、收货地址、车次、勾选）
    if any(kw in text for kw in ["订单", "预订", "预约", "结算", "提交", "涉及订单", 
                                  "选择商品", "规格信息缺失", "选择目的地", "选择", "商品"]):
        return "涉及订单"
    
    # 确认弹窗（确认、弹窗、下订单、同意协议）
    if any(kw in text for kw in ["确认弹窗", "弹窗", "下订单", "同意协议"]):
        return "确认弹窗"
    
    # 涉及数据（数据、收藏、历史、记录）
    if any(kw in text for kw in ["数据", "收藏", "历史", "记录"]):
        return "涉及数据"
    
    # 信息缺失（但未匹配到其他类别）
    if "信息缺失" in text:
        return "信息缺失"
    
    # 求助（无法完成、请求接管、接管处理、请您接管）
    if any(kw in text for kw in ["无法完成", "请求接管", "接管处理", "请您接管", "求助", "场景接管"]):
        return "求助"
    
    # 默认
    return "求助"


def _extract_takeover_reason_from_describe(action_describe: str) -> str:
    """
    从 action_describe 中提取接管原因/类型
    
    支持格式：
    - "接管（删除）：点击删除账号" -> "删除"
    - "接管（信息缺失--请选择商品）：点击..." -> "信息缺失"
    - "接管（同意协议--请阅读协议并确认）：点击..." -> "同意协议"
    - "信息缺失场景接管" -> "信息缺失"
    - "支付场景接管" -> "支付"
    - "人机验证场景接管" -> "人机验证"
    
    Returns:
        str: 接管原因
    """
    import re
    
    # 格式1: "接管（类型）" 或 "接管（类型--详情）"
    paren_match = re.search(r'接管[（(]([^）)]+)[）)]', action_describe)
    if paren_match:
        content = paren_match.group(1).strip()
        # 如果有 "--" 分隔，取第一部分作为类型
        if "--" in content:
            return content.split("--")[0].strip()
        return content
    
    # 格式2: "XXX场景接管"
    scene_match = re.match(r'^(.+?)场景接管', action_describe)
    if scene_match:
        return scene_match.group(1).strip()
    
    # 格式3: 直接包含关键词
    if "接管" in action_describe:
        return action_describe.strip()
    
    return ""


def convert_tool_call_coords(tool_call_str: str, image_width: Optional[int], image_height: Optional[int]) -> str:
    """
    转换 tool_call 字符串中的坐标（从千分坐标转换为原始坐标）
    
    Args:
        tool_call_str: tool_call 字符串（可能是 JSON 格式）
        image_width: 图片宽度
        image_height: 图片高度
        
    Returns:
        转换后的 tool_call 字符串
    """
    if not tool_call_str or image_width is None or image_height is None:
        return tool_call_str
    
    try:
        import json as _json
        tool_call_dict = _json.loads(tool_call_str)
        
        # 检查是否有 coordinate 字段需要转换
        if isinstance(tool_call_dict, dict):
            arguments = tool_call_dict.get("arguments", {})
            if isinstance(arguments, dict) and "coordinate" in arguments:
                coord = arguments["coordinate"]
                if isinstance(coord, list) and len(coord) >= 2:
                    x, y = int(coord[0]), int(coord[1])
                    x_new, y_new = convert_coordinate(x, y, image_width, image_height)
                    arguments["coordinate"] = [x_new, y_new]
                    return _json.dumps(tool_call_dict, ensure_ascii=False)
    except Exception:
        # 如果解析失败，返回原字符串
        pass
    
    return tool_call_str


def infer_tool_call(meta_action: str, is_takeover: bool = False, action_describe: str = "", image_width: Optional[int] = None, image_height: Optional[int] = None) -> str:
    """
    将 meta_action 转换为 tool_call 格式字符串
    
    Args:
        meta_action: meta_action 字符串
        is_takeover: 是否为接管步骤
        action_describe: action_describe 字符串（兜底用）
        image_width: 图片宽度（用于坐标转换）
        image_height: 图片高度（用于坐标转换）
        
    Returns:
        str: tool_call 字符串
    """
    import json as _json
    
    if is_takeover:
        # 优先从 meta_action 提取，其次从 action_describe 提取
        category = infer_takeover_category(meta_action, action_describe)
        
        return _json.dumps({
            "name": "mobile_use",
            "arguments": {
                "action": "take_over",
                "tag": category
            }
        }, ensure_ascii=False)
    else:
        # 普通步骤，返回 meta_action 原值，并转换其中的坐标
        result = meta_action
        if result:
            result = convert_tool_call_coords(result, image_width, image_height)
        return result


def convert_batchid_to_query(batchid_data: Dict, tag: str = "zj_20260303_test") -> Dict:
    """
    将batchID格式的数据转换为batch_export格式（与batch_export_2026021016390014.jsonl格式一致）
    
    Args:
        batchid_data: batchID格式的数据字典
        tag: 要添加到tags列表的标签，默认为"zj_20260303_test"
        
    Returns:
        batch_export格式的数据字典
    """
    # 构建traj数组
    traj = []
    app_name = batchid_data.get("app", "")
    
    # 优先使用history_image_pairs，如果没有则使用historys和images组合
    if "history_image_pairs" in batchid_data and batchid_data["history_image_pairs"]:
        for pair in batchid_data["history_image_pairs"]:
            action_describe = pair.get("action_describe", "")
            image_path = pair.get("image_path", "")
            meta_action = pair.get("meta_action", "")
            step_num = pair.get("step_num", len(traj) + 1)
            
            # 读取图片尺寸
            image_width, image_height = get_image_dimensions(image_path)
            
            # 判断是否为接管步骤（同时检查 meta_action 和 action_describe）
            takeover = is_takeover_step(meta_action, action_describe)
            
            if takeover:
                # 接管步骤：按 batch_export 格式，value 中不含 code，包含 tool_call
                category = infer_takeover_category(meta_action, action_describe)
                
                # 提取接管原因作为 summarize
                if meta_action:
                    reason = _extract_takeover_reason_from_describe(action_describe) or action_describe
                else:
                    reason = _extract_takeover_reason_from_describe(action_describe) or action_describe
                
                traj_item = {
                    "activity": {
                        "app_name": app_name,
                        "package_name": ""
                    },
                    "image": image_path,
                    "image_local_path": image_path,
                    "image_width": image_width,
                    "image_height": image_height,
                    "index": step_num - 1,
                    "tags": [],
                    "take_over": {
                        "category": category,
                        "continue_goal": "外部模型接管",
                        "summarize": action_describe
                    },
                    "value": {
                        "action": action_describe,
                        "thought": "",
                        "tool_call": infer_tool_call(meta_action, True, action_describe, image_width, image_height)
                    }
                }
            else:
                # 普通步骤：value 中包含 code 和 tool_call
                if meta_action:
                    action_code = infer_action_code_from_meta(meta_action, image_width, image_height) or infer_action_code(action_describe, image_width, image_height)
                else:
                    action_code = infer_action_code(action_describe, image_width, image_height)
                
                # 转换 tool_call 中的坐标
                tool_call_result = meta_action if meta_action else ""
                if tool_call_result:
                    tool_call_result = convert_tool_call_coords(tool_call_result, image_width, image_height)
                
                traj_item = {
                    "activity": {
                        "app_name": app_name,
                        "package_name": ""
                    },
                    "image": image_path,
                    "image_local_path": image_path,
                    "image_width": image_width,
                    "image_height": image_height,
                    "index": step_num - 1,
                    "tags": [],
                    "value": {
                        "action": action_describe,
                        "code": action_code,
                        "thought": "",
                        "tool_call": tool_call_result
                    }
                }
            
            traj.append(traj_item)
    
    elif "historys" in batchid_data and "images" in batchid_data:
        # 如果没有history_image_pairs，使用historys和images组合
        historys = batchid_data.get("historys", [])
        images = batchid_data.get("images", [])
        
        for i, (history, image) in enumerate(zip(historys, images)):
            # 读取图片尺寸
            image_width, image_height = get_image_dimensions(image)
            
            action_code = infer_action_code(history, image_width, image_height)
            
            # 转换 tool_call 中的坐标
            tool_call_result = history
            if tool_call_result:
                tool_call_result = convert_tool_call_coords(tool_call_result, image_width, image_height)
            
            traj_item = {
                "activity": {
                    "app_name": app_name,
                    "package_name": ""
                },
                "image": image,
                "image_local_path": image,
                "image_width": image_width,
                "image_height": image_height,
                "index": i,
                "tags": [],
                "value": {
                    "action": history,
                    "code": action_code,
                    "thought": "",
                    "tool_call": tool_call_result
                }
            }
            traj.append(traj_item)
    
    # 处理tags列表：合并原有tags并添加新tag，去重
    existing_tags = batchid_data.get("tags", [])
    if not isinstance(existing_tags, list):
        existing_tags = []
    
    # 合并tags并去重
    tags_list = list(existing_tags)
    if tag and tag not in tags_list:
        tags_list.append(tag)
    
    # 仅支持最新格式：response 必须为 dict，且包含 is_task_completed、is_trajectory_correct（无拼写错误）
    response = batchid_data.get("response")
    if not isinstance(response, dict):
        raise ValueError("response 必须为 dict，仅支持最新格式")
    if "is_task_completed" not in response:
        raise ValueError("response 缺少字段 is_task_completed，仅支持最新格式")
    if "is_trajectory_correct" not in response:
        raise ValueError("response 缺少字段 is_trajectory_correct（正确拼写），仅支持最新格式，不支持 is_trajactory_correct")
    is_task_completed = response["is_task_completed"]
    is_trajectory_correct = response["is_trajectory_correct"]

    evaluation = {
        "is_task_completed": is_task_completed,      # 人工校验终态完成度
        "is_trajectory_correct": is_trajectory_correct,  # 人工校验轨迹正确性
    }

    # 构建 batch_export 格式数据（完整字段）
    batch_export_data = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation": evaluation,
        "instruction": batchid_data.get("task", ""),
        "isChecked": True,  # 默认值
        "metadata": {
            "app": batchid_data.get("app", ""),
            "max_step": batchid_data.get("max_step_idx"),
            "status": "finished"  # 默认值
        },
        "source": "Backflow",  # 固定值
        "tags": tags_list,
        "task_id": batchid_data.get("task_id", ""),
        "traj": traj,
    }
    
    return batch_export_data


def convert_file(input_file: str, output_file: str, tag: str = "zj_20260303_test"):
    """
    转换整个jsonl文件
    
    Args:
        input_file: 输入的batchID格式jsonl文件路径
        output_file: 输出的query格式jsonl文件路径
        tag: 要添加到tags列表的标签，默认为"zj_20260303_test"
    """
    # #region agent log
    global _DEBUG_GID_CALLS, _DEBUG_GID_TIME, _DEBUG_PIL_USED, _DEBUG_CV2_USED
    _DEBUG_GID_CALLS = 0
    _DEBUG_GID_TIME = 0.0
    _DEBUG_PIL_USED = 0
    _DEBUG_CV2_USED = 0
    t_start = time.perf_counter()
    _debug_write({"location": "convert_file:start", "message": "convert_file_start", "data": {"input_file": input_file}, "hypothesisId": "H4"})
    # #endregion
    converted_count = 0
    error_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:

            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue

                # #region agent log
                t_line = time.perf_counter()
                # #endregion
                try:
                    batchid_data = json.loads(line)
                    query_data = convert_batchid_to_query(batchid_data, tag=tag)
                    f_out.write(json.dumps(query_data, ensure_ascii=False) + '\n')
                    converted_count += 1
                except json.JSONDecodeError as e:
                    print(f"第{line_num}行JSON解析错误: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"第{line_num}行转换错误: {e}")
                    error_count += 1
                # #region agent log
                elapsed_line = time.perf_counter() - t_line
                _debug_write({"location": "convert_file:line", "message": "line_done", "data": {"line_num": line_num, "elapsed_line_sec": round(elapsed_line, 4)}, "hypothesisId": "H3"})
                if line_num in (1, 2, 3):
                    _debug_write({"location": "convert_file:first_lines", "message": "first_line_timing", "data": {"line_num": line_num, "elapsed_line_sec": round(elapsed_line, 4)}, "hypothesisId": "H4"})
                # #endregion

        # #region agent log
        total_sec = time.perf_counter() - t_start
        gid_ratio = (_DEBUG_GID_TIME / total_sec * 100) if total_sec > 0 else 0
        _debug_write({"location": "convert_file:end", "message": "convert_file_summary", "data": {"total_sec": round(total_sec, 4), "gid_calls": _DEBUG_GID_CALLS, "gid_time_sec": round(_DEBUG_GID_TIME, 4), "gid_ratio_pct": round(gid_ratio, 2), "converted_count": converted_count, "pil_used": _DEBUG_PIL_USED, "cv2_used": _DEBUG_CV2_USED}, "hypothesisId": "H1"})
        # #endregion
        print(f"\n转换完成！")
        print(f"成功转换: {converted_count} 条")
        print(f"错误: {error_count} 条")
        print(f"输出文件: {output_file}")
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except Exception as e:
        print(f"转换失败: {e}")


def print_field_mapping():
    """打印字段映射关系"""
    print("=" * 80)
    print("字段映射关系")
    print("=" * 80)
    print()
    print("batchID格式 -> query格式")
    print()
    print("直接映射：")
    print("  task_id -> task_id")
    print("  task -> instruction")
    print()
    print("转换映射：")
    print("  history_image_pairs -> traj[]")
    print("    - step_num -> index (注意：step_num从1开始，index从0开始)")
    print("    - action_describe -> value.action")
    print("    - image_path -> image")
    print("    - meta_action -> 优先用于推断value.code（新格式字段）")
    print("    - value.code -> 优先从meta_action推断，如果没有或推断失败，则从action_describe推断")
    print()
    print("缺失字段（置空）：")
    print("  created_at -> 使用当前时间")
    print("  tags -> []")
    print("  activity.activity_name -> \"\"")
    print("  activity.package_name -> \"\"")
    print()
    print("batchID格式中的其他字段（不转换）：")
    print("  - app")
    print("  - historys (如果history_image_pairs存在则不用)")
    print("  - images (如果history_image_pairs存在则不用)")
    print("  - think")
    print("  - response (仅支持最新格式 dict，必须含 is_task_completed、is_trajectory_correct，正确拼写)")
    print("  - max_step_idx")
    print("  - max_step_content")
    print("  - dag")
    print("  - takeover_step_idx (新格式字段：接管步骤索引数组)")
    print()
    print("=" * 80)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法：")
        print("  python utils/convert_batchid_to_query.py <input_file> [output_file] [--tag TAG]")
        print("  python utils/convert_batchid_to_query.py --mapping  # 查看字段映射关系")
        print()
        print("示例：")
        print("  python utils/convert_batchid_to_query.py batchID_xxx.jsonl query_xxx.jsonl")
        print("  python utils/convert_batchid_to_query.py batchID_xxx.jsonl  # 输出到query_xxx.jsonl")
        print("  python utils/convert_batchid_to_query.py batchID_xxx.jsonl query_xxx.jsonl --tag custom_tag")
        return

    
    input_file = sys.argv[1]
    
    # 解析tag参数（如果通过命令行传入）
    tag = "zj_20260303_test"  # 默认值
    if len(sys.argv) > 1 and "--tag" in sys.argv:
        tag_index = sys.argv.index("--tag")
        if tag_index + 1 < len(sys.argv):
            tag = sys.argv[tag_index + 1]
    
    # 确定output_file（自动生成）
    output_file = input_file.replace("batchID", "query").replace("_mid_data", "")
    
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在: {input_file}")
        return
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"标签: {tag}")
    print()
    
    convert_file(input_file, output_file, tag=tag)


if __name__ == "__main__":
    main()
