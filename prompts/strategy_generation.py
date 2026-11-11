"""
策略加载模块
从策略库文件加载新格式的策略配置
"""
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.prompt_utils import load_json_file, save_json_file
import config


_strategy_cache = {}


def get_strategy_file_path(category: str) -> str:
    """获取策略文件路径"""
    return os.path.join(config.STRATEGIES_DIR, f"{category}.json")


def load_strategy(category: str) -> Dict:
    """
    加载指定分类的策略库（新格式）
    
    Args:
        category: 任务分类
        
    Returns:
        Dict: 策略库配置（新格式，包含select_keyframe和judge_completion_by_keyframe）
    """
    if category in _strategy_cache:
        return _strategy_cache[category]
    strategy_path = get_strategy_file_path(category)
    data = load_json_file(strategy_path, {})
    _strategy_cache[category] = data
    return data


def save_strategy(category: str, strategy_data: Dict):
    """保存策略库"""
    strategy_path = get_strategy_file_path(category)
    save_json_file(strategy_path, strategy_data)
