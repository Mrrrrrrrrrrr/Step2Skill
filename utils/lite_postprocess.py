"""
Lite 评估结果后处理逻辑。
"""

from typing import Any, Dict, Optional


def _extract_last_step_take_over(task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 traj 最后一步提取 take_over 字段。"""
    traj = task_data.get("traj")
    if not isinstance(traj, list) or not traj:
        return None

    last_step = traj[-1]
    if not isinstance(last_step, dict):
        return None

    take_over = last_step.get("take_over")
    if isinstance(take_over, dict):
        return take_over

    value = last_step.get("value")
    if isinstance(value, dict):
        nested_take_over = value.get("take_over")
        if isinstance(nested_take_over, dict):
            return nested_take_over

    return None


def _is_finish_take_over(task_data: Dict[str, Any]) -> bool:
    """判断最后一步 take_over.continue_goal 是否为 finish。"""
    take_over = _extract_last_step_take_over(task_data)
    if not isinstance(take_over, dict):
        return False

    continue_goal = take_over.get("continue_goal")
    if not isinstance(continue_goal, str):
        return False
    return continue_goal.strip() == "finish"


def _is_user_active_cancel(task_data: Dict[str, Any]) -> bool:
    """判断 client_report.error_type 是否为用户主动取消。"""
    client_report = task_data.get("client_report")
    if not isinstance(client_report, dict):
        return False
    error_type = client_report.get("error_type")
    if not isinstance(error_type, str):
        return False
    return error_type.strip() == "用户主动取消"


def apply_lite_task_success_postprocess(
    task_data: Dict[str, Any],
    orm_prediction: Dict[str, Any],
) -> bool:
    """
    根据 lite 后处理规则修正 is_task_success。

    生效前提：
    - is_task_success == 0
    - is_trajectory_correct == 1

    命中任一规则后，将 is_task_success 修正为 1：
    1) 最后一步 take_over.continue_goal == "finish"
    2) client_report.error_type == "用户主动取消"

    返回：
        bool: 是否发生修正
    """
    if not isinstance(task_data, dict) or not isinstance(orm_prediction, dict):
        return False

    if orm_prediction.get("is_task_success") != 0:
        return False
    if orm_prediction.get("is_trajectory_correct") != 1:
        return False

    if _is_finish_take_over(task_data) or _is_user_active_cancel(task_data):
        orm_prediction["is_task_success"] = 1
        return True

    return False
