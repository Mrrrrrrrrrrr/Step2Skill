"""MobileGym skill trajectory knowledge rendering utilities."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Tuple


def _mobilegym_skill_common_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "knowledge_base_mobilegym", "skill_trajectory", "common.json")
    )


def load_mobilegym_skill_common_knowledge() -> Dict[str, Any]:
    with open(_mobilegym_skill_common_path(), "r", encoding="utf-8") as f:
        return json.load(f)


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def render_mobilegym_skill_common_knowledge_section() -> str:
    data = load_mobilegym_skill_common_knowledge()
    lines: List[str] = []
    title = str(data.get("title") or "MobileGym skill trajectory common knowledge").strip()
    version = str(data.get("version") or "").strip()
    header = f"# {title}"
    if version:
        header += f" ({version})"
    lines.append(header)

    general_rules = [str(x).strip() for x in _as_list(data.get("general_rules")) if str(x).strip()]
    if general_rules:
        lines.append("\n## 通用判定与步骤对齐规则")
        for rule in general_rules:
            lines.append(f"- {rule}")

    families = _as_list(data.get("recurring_failure_families"))
    if families:
        lines.append("\n## 常见可抽取技能失败族")
        for item in families:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("id") or "未命名失败族").strip()
            signal = str(item.get("skill_signal_type") or "").strip()
            mode = str(item.get("failure_mode_for_skill") or "").strip()
            suffix_parts = []
            if signal:
                suffix_parts.append(f"skill_signal_type={signal}")
            if mode:
                suffix_parts.append(f"failure_mode_for_skill={mode}")
            suffix = f"（{'；'.join(suffix_parts)}）" if suffix_parts else ""
            lines.append(f"### {label}{suffix}")
            for hint in [str(x).strip() for x in _as_list(item.get("judge_hints")) if str(x).strip()]:
                lines.append(f"- {hint}")
            step_rules = [str(x).strip() for x in _as_list(item.get("step_alignment_rules")) if str(x).strip()]
            for rule in step_rules:
                lines.append(f"- 步骤对齐：{rule}")
            lines.append("- 抽取策略：若截图和动作能定位 wrong_action_step，通常 can_extract_skill=true。")

    return "\n".join(lines).strip()


def mobilegym_skill_common_knowledge_hash() -> str:
    try:
        data = load_mobilegym_skill_common_knowledge()
        text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10] if text else "mgkbnone"


def get_mobilegym_skill_common_knowledge_and_version() -> Tuple[str, str]:
    try:
        return render_mobilegym_skill_common_knowledge_section(), mobilegym_skill_common_knowledge_hash()
    except Exception:
        return "", "mgkberror"
