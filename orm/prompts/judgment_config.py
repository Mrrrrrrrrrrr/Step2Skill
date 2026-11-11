"""
任务完成判断相关配置
"""

# 状态字符串到数字的映射
STATUS_MAPPING = {
    "completed": 1,
    "correct": 1,
    "not_completed": 0,
    "incorrect": 0,
    "uncertain_need_better_keyframe": -1,
}

# 默认状态选项
DEFAULT_STATUS_OPTIONS = ["completed", "not_completed", "uncertain_need_better_keyframe"]
