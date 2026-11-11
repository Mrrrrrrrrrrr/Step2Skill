"""
Lite Judgment Prompt 知识库工具：
- 从 knowledge_base JSON 加载规则
- 将 JSON 规则格式化为 prompt 风格文本（用于分析 prompt 和 improvements_section）
- 管理 extra_rules（自动生成的补充/新增规则）
"""
from __future__ import annotations

import json
import os
import hashlib
from typing import Any, Dict, List, Optional

import config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _corrected_json_path() -> str:
    return os.path.join(config.KNOWLEDGE_BASE_DIR, "corrected_strategies", "common.json")


def _completed_json_path() -> str:
    return os.path.join(config.KNOWLEDGE_BASE_DIR, "completed_strategies", "common.json")


def _env_error_json_path() -> str:
    return os.path.join(config.KNOWLEDGE_BASE_DIR, "env_error_strategies", "common.json")


def _negative_task_json_path() -> str:
    return os.path.join(config.KNOWLEDGE_BASE_DIR, "negative_task_strategies", "common.json")


def _takeover_knowledge_json_path() -> str:
    return os.path.join(config.KNOWLEDGE_BASE_DIR, "takeover_knowledge", "takeover_knowledge.json")


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_dimension_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dimension_json(json_path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_trajectory_rules() -> Dict[str, Any]:
    return load_dimension_json(_corrected_json_path())


def load_completion_rules() -> Dict[str, Any]:
    return load_dimension_json(_completed_json_path())


def load_takeover_knowledge() -> Dict[str, Any]:
    try:
        return load_dimension_json(_takeover_knowledge_json_path())
    except Exception:
        return {"version": 1, "items": []}


TAKEOVER_JUDGE_MODES = {"end-to-end", "only-lmd"}


def normalize_takeover_judge_mode(mode: Optional[str] = None) -> str:
    value = (mode or "end-to-end").strip().lower()
    if value in TAKEOVER_JUDGE_MODES:
        return value
    return "end-to-end"


def takeover_knowledge_hash(mode: Optional[str] = None) -> str:
    data = load_takeover_knowledge()
    try:
        payload = {"mode": normalize_takeover_judge_mode(mode), "data": data}
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10] if text else "tkvnone"


def render_takeover_knowledge_section(
    max_examples_per_type: int = 8,
    mode: Optional[str] = None,
) -> str:
    """Render takeover knowledge shared by lite gate and error-cause prompts."""
    mode = normalize_takeover_judge_mode(mode)
    data = load_takeover_knowledge()
    items = data.get("items") if isinstance(data, dict) else []
    if not isinstance(items, list) or not items:
        return ""

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        primary = str(item.get("primary_category") or "").strip() or "未分类"
        secondary = str(item.get("secondary_category") or "").strip() or "未分类"
        grouped.setdefault(primary, {}).setdefault(secondary, []).append(item)

    mode_meta = data.get("modes", {}).get(mode, {}) if isinstance(data, dict) else {}
    required_field = "end_to_end_required" if mode == "end-to-end" else "only_lmd_required"
    optional_field = None if mode == "end-to-end" else "only_lmd_optional"
    mode_title = "端到端输出接管" if mode == "end-to-end" else "大模型输出接管"

    lines: List[str] = [
        f"# 接管知识（judge_takeover={mode}，{mode_title}；用于判断漏接管、过度接管、接管类型错误）",
        "- 判断时以截图、当前意图、take_over/category/tool_call tag 为准；不要仅凭 Agent 自述判定。",
        "- 当前页面证据门槛：即使命中下方某类接管知识，也必须由接管发生时的当前截图证明已经到达需要用户人工介入的页面状态，例如当前页已显示必须填写的输入框、登录/验证码/实名验证、支付/删除/授权确认、隐私/协议确认、多个候选且无唯一依据的选择确认、或明确错误/不可继续提示。",
        "- 禁止预判式接管：不能因为 Agent 推测“点击某入口后未来可能需要手机号/地址/身份证/支付/授权/确认”等信息，就在普通店铺页、商品详情页、列表页、入口页、设置页、功能首页或其它仍可点击入口继续推进的中间页提前接管。当前只看到“订座/排队/购买/下一步/去填写/确认信息”等入口时，应先由 Agent 点击入口继续探索；若后续页面实际要求用户输入或确认，再按接管知识判断。",
        "- 信息缺失类只在当前步骤确实缺少继续执行所必需且 Agent 无法自行获得/填写的信息时成立；若当前页面尚未要求该信息，或还有明确入口/流程可继续推进以到达填写页/确认页，则提前以信息缺失、隐私输入、联系方式等理由接管属于过度接管。",
        f"- 漏接管：{mode_meta.get('required_rule', '下方标记为必须接管的场景必须接管且类型正确。')}",
    ]
    optional_rule = mode_meta.get("optional_rule", "")
    if optional_rule and optional_rule != "无。":
        lines.append(f"- 可选接管：{optional_rule}")
    lines.extend([
        f"- 过度接管/错接管：{mode_meta.get('overuse_rule', '不在本接管知识范围内请求接管，属于过度接管/错接管。')}",
        "- 接管类型错误：已接管但类型与对应二级标签/expected_takeover_type 有明显偏差；同义表述可视为正确。",
        "- 任务成功、失败或环境错误仍按主评估规则判断；接管知识只定义哪些场景应接管、可接管、不可过度接管以及正确类型。",
    ])
    for primary, secondaries in grouped.items():
        lines.append(f"## {primary}")
        for secondary, rows in secondaries.items():
            first = rows[0]
            required = any(bool(r.get(required_field)) for r in rows if isinstance(r, dict))
            optional = bool(optional_field) and any(bool(r.get(optional_field)) for r in rows if isinstance(r, dict))
            if required:
                status = "必须接管且类型正确"
            elif optional:
                status = "不强制接管；如已接管则类型必须正确"
            else:
                status = "不属于本模式允许/必须接管范围"
            expected_type = str(first.get("expected_takeover_type") or secondary).strip()
            suffix = f"（{status}；正确类型={expected_type}）"
            lines.append(f"- {secondary}{suffix}")
            examples = [str(r.get("example") or "").strip() for r in rows]
            examples = [e for e in examples if e]
            if examples:
                lines.append("  - 常见例子：" + "；".join(examples[:max_examples_per_type]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON -> Prompt Text  (用于 LLM 分析 prompt 中展示现有规则)
# ---------------------------------------------------------------------------

def _format_single_rule_text(rule: Dict[str, Any], bullet: str = "-") -> str:
    """将单条规则格式化为 prompt 风格文本。

    支持的字段: title, description, examples, sub_notes,
    sub_rules, not_applicable, counter_examples, judgment_key
    """
    lines: List[str] = []
    title = rule.get("title", "")
    desc = rule.get("description", "")

    if desc:
        lines.append(f"{bullet} **{title}**：{desc}")
    else:
        lines.append(f"{bullet} **{title}**：")

    for note in rule.get("sub_notes", []):
        lines.append(f"  - {note}")

    jk = rule.get("judgment_key", "")
    if jk:
        lines.append(f"  - 判定要点：{jk}")

    for ex in rule.get("examples", []):
        lines.append(f"  - {ex}")

    na = rule.get("not_applicable", "")
    if na:
        lines.append(f"  - **不适用**：{na}")

    for ce in rule.get("counter_examples", []):
        lines.append(f"  - {ce}")

    for sr in rule.get("sub_rules", []):
        lines.append(_format_single_rule_text(sr, bullet="  - "))

    return "\n".join(lines)


def format_rules_list_as_text(rules: List[Dict[str, Any]], bullet: str = "-") -> str:
    """将规则列表格式化为 prompt 风格文本块"""
    parts = [_format_single_rule_text(r, bullet=bullet) for r in rules]
    return "\n".join(parts)


def format_boundary_as_text(boundaries: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for b in boundaries:
        situation = b.get("situation", "")
        verdict = b.get("verdict", "")
        detail = b.get("detail", "")
        lines.append(f"- 「{situation}」：{detail} → **{verdict}**")
    return "\n".join(lines)


def get_full_rules_text(dim_json: Dict[str, Any]) -> str:
    """将一个维度的全部规则（原始 + extra）格式化为完整文本，用于分析 prompt。"""
    parts: List[str] = []

    pass_title = dim_json.get("pass_section_title", "通过规则")
    parts.append(f"### {pass_title}")
    parts.append(format_rules_list_as_text(dim_json.get("pass_rules", [])))

    extra_pass = dim_json.get("extra_pass_rules", [])
    if extra_pass:
        parts.append("")
        parts.append("（以下为自动生成的补充规则）")
        parts.append(_format_extra_rules_text(extra_pass))

    parts.append("")
    fail_title = dim_json.get("fail_section_title", "不通过规则")
    parts.append(f"### {fail_title}")
    parts.append(format_rules_list_as_text(dim_json.get("fail_rules", [])))

    extra_fail = dim_json.get("extra_fail_rules", [])
    if extra_fail:
        parts.append("")
        parts.append("（以下为自动生成的补充规则）")
        parts.append(_format_extra_rules_text(extra_fail))

    boundaries = dim_json.get("boundary_clarifications", [])
    extra_boundaries = dim_json.get("extra_boundary_clarifications", [])
    if boundaries or extra_boundaries:
        parts.append("")
        parts.append("### 必须辨析的边界")
        if boundaries:
            parts.append(format_boundary_as_text(boundaries))
        if extra_boundaries:
            parts.append("")
            parts.append("（以下为自动生成的补充边界）")
            parts.append(format_boundary_as_text(extra_boundaries))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Extra Rules -> improvements_section text  (注入到 eval prompt)
# ---------------------------------------------------------------------------

def _format_extra_rules_text(extra_rules: List[Dict[str, Any]]) -> str:
    """将 extra_rules 列表格式化为 prompt 文本。

    extra_rules 中每条可以是:
    - supplement: 有 existing_title, supplement_content, supplement_examples
    - new: 有 title, description, examples
    """
    lines: List[str] = []
    supplements = [r for r in extra_rules if "existing_title" in r]
    new_rules = [r for r in extra_rules if "existing_title" not in r and "title" in r]

    if supplements:
        for s in supplements:
            et = s.get("existing_title", "")
            sc = s.get("supplement_content", "")
            lines.append(f"- 对现有规则「{et}」的补充：{sc}")
            for ex in s.get("supplement_examples", []):
                lines.append(f"  - {ex}")

    if new_rules:
        for nr in new_rules:
            title = nr.get("title", "")
            desc = nr.get("description", "")
            lines.append(f"- **{title}**：{desc}")
            for ex in nr.get("examples", []):
                lines.append(f"  - {ex}")

    return "\n".join(lines)


def assemble_improvements_section() -> str:
    """从两个维度的 knowledge_base JSON 读取 extra_* 字段，
    组装为可直接传入 get_lite_judgment_prompt() 的 improvements_section 文本。

    如果没有任何 extra 规则，返回空字符串。
    """
    traj = load_trajectory_rules()
    comp = load_completion_rules()

    sections: List[str] = []

    traj_extra = _collect_extra_text(traj, "轨迹正确性")
    if traj_extra:
        sections.append(traj_extra)

    comp_extra = _collect_extra_text(comp, "终态完成度")
    if comp_extra:
        sections.append(comp_extra)

    if not sections:
        return ""

    header = "# 补充规则（基于错误案例分析自动生成）\n"
    return header + "\n\n".join(sections)


def _collect_extra_text(dim_json: Dict[str, Any], dim_name: str) -> str:
    """收集一个维度下所有 extra 规则，格式化为文本块。"""
    parts: List[str] = []

    extra_pass = dim_json.get("extra_pass_rules", [])
    extra_fail = dim_json.get("extra_fail_rules", [])
    extra_bd = dim_json.get("extra_boundary_clarifications", [])

    if not extra_pass and not extra_fail and not extra_bd:
        return ""

    parts.append(f"## {dim_name} - 补充")

    if extra_pass:
        parts.append("")
        parts.append("**通过规则补充：**")
        parts.append(_format_extra_rules_text(extra_pass))

    if extra_fail:
        parts.append("")
        parts.append("**不通过规则补充：**")
        parts.append(_format_extra_rules_text(extra_fail))

    if extra_bd:
        parts.append("")
        parts.append("**边界辨析补充：**")
        parts.append(format_boundary_as_text(extra_bd))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Apply LLM-generated rules to knowledge base
# ---------------------------------------------------------------------------

def apply_rules_to_knowledge_base(llm_rules: List[Dict[str, Any]]) -> Dict[str, int]:
    """将 LLM 输出的规则列表写入对应维度的 knowledge_base JSON。

    每条 rule 需包含:
      - action: "supplement" | "new"
      - affected_dimension: "轨迹正确性" | "完成度"
      - target_section: "pass_rules" | "fail_rules" | "boundary_clarifications"
      - (supplement) existing_title, supplement_content, supplement_examples
      - (new) title, description, examples

    返回 {"supplements": N, "new_rules": N} 统计。
    """
    traj = load_trajectory_rules()
    comp = load_completion_rules()

    stats = {"supplements": 0, "new_rules": 0}

    for rule in llm_rules:
        dim = rule.get("affected_dimension", "")
        dim_json = traj if "轨迹" in dim else comp
        section = rule.get("target_section", "fail_rules")
        action = rule.get("action", "new")

        extra_key = f"extra_{section}"
        if extra_key not in dim_json:
            extra_key = "extra_fail_rules"
        extra_list: List[Dict] = dim_json.setdefault(extra_key, [])

        if action == "supplement":
            _apply_supplement(extra_list, rule)
            stats["supplements"] += 1
        else:
            _apply_new_rule(extra_list, rule)
            stats["new_rules"] += 1

    save_dimension_json(_corrected_json_path(), traj)
    save_dimension_json(_completed_json_path(), comp)
    return stats


def _apply_supplement(extra_list: List[Dict], rule: Dict) -> None:
    existing_title = rule.get("existing_title", "")
    content = rule.get("supplement_content", "")
    examples = rule.get("supplement_examples", [])

    for item in extra_list:
        if item.get("existing_title") == existing_title:
            if content and content not in item.get("supplement_content", ""):
                old = item.get("supplement_content", "")
                item["supplement_content"] = f"{old} {content}".strip() if old else content
            existing_exs = item.setdefault("supplement_examples", [])
            for ex in examples:
                if ex not in existing_exs:
                    existing_exs.append(ex)
            return

    extra_list.append({
        "existing_title": existing_title,
        "supplement_content": content,
        "supplement_examples": examples,
    })


def _apply_new_rule(extra_list: List[Dict], rule: Dict) -> None:
    title = rule.get("title", "")
    desc = rule.get("description", "")
    examples = rule.get("examples", [])

    for item in extra_list:
        if item.get("title") == title and "existing_title" not in item:
            existing_exs = item.setdefault("examples", [])
            for ex in examples:
                if ex not in existing_exs:
                    existing_exs.append(ex)
            if desc and not item.get("description"):
                item["description"] = desc
            return

    extra_list.append({
        "title": title,
        "description": desc,
        "examples": examples,
    })


# ---------------------------------------------------------------------------
# JSON -> Prompt Section  (完整维度渲染，base + extra 合并)
# ---------------------------------------------------------------------------

def _match_title(base_title: str, extra_title: str) -> bool:
    """模糊匹配：任一方是另一方的子串即视为同一规则。"""
    a, b = base_title.strip(), extra_title.strip()
    return a in b or b in a


def _find_supplements(extra_rules: List[Dict], base_title: str) -> List[Dict]:
    return [r for r in extra_rules if "existing_title" in r
            and _match_title(base_title, r["existing_title"])]


def _get_new_rules(extra_rules: List[Dict]) -> List[Dict]:
    return [r for r in extra_rules if "existing_title" not in r and "title" in r]


def _render_supplement_lines(supplements: List[Dict], indent: str = "  ") -> List[str]:
    """将 supplement 规则渲染为缩进的补充行。"""
    lines: List[str] = []
    for s in supplements:
        content = s.get("supplement_content", "")
        if content:
            lines.append(f"{indent}- 【补充】{content}")
        for ex in s.get("supplement_examples", []):
            lines.append(f"{indent}  - {ex}")
    return lines


# ---- 轨迹正确性 (维度一) ----

def _render_traj_rule(rule: Dict, supplements: List[Dict]) -> List[str]:
    """渲染一条轨迹规则 (dash 格式)，并内联补充规则。"""
    title = rule.get("title", "")
    desc = rule.get("description", "")
    lines: List[str] = []
    lines.append(f"- **{title}**：{desc}" if desc else f"- **{title}**：")
    for ex in rule.get("examples", []):
        lines.append(f"  - {ex}")
    lines.extend(_render_supplement_lines(supplements))
    return lines


def render_trajectory_section() -> str:
    """从 knowledge_base JSON 反序列化维度一的完整 prompt 文本。"""
    t = load_trajectory_rules()
    extra_pass = t.get("extra_pass_rules", [])
    extra_fail = t.get("extra_fail_rules", [])
    lines: List[str] = []

    lines.append(f"# {t['section_title']}")
    lines.append("")
    lines.append(f"**核心问题**：{t['core_question']}")
    lines.append(f"注意：{t['note']}")
    lines.append("")
    lines.append(f"## {t['pass_section_title']}")
    lines.append("")
    for rule in t.get("pass_rules", []):
        sups = _find_supplements(extra_pass, rule["title"])
        lines.extend(_render_traj_rule(rule, sups))
    new_pass = _get_new_rules(extra_pass)
    for nr in new_pass:
        lines.extend(_render_traj_rule(nr, []))

    lines.append("")
    lines.append(f"## {t['fail_section_title']}")
    lines.append("")
    for rule in t.get("fail_rules", []):
        sups = _find_supplements(extra_fail, rule["title"])
        lines.extend(_render_traj_rule(rule, sups))
    new_fail = _get_new_rules(extra_fail)
    for nr in new_fail:
        lines.extend(_render_traj_rule(nr, []))

    return "\n".join(lines)


# ---- 终态完成度 (维度二) ----

def _render_comp_pass_rule(rule: Dict, supplements: List[Dict]) -> List[str]:
    """渲染完成度 pass 规则 (无 dash, 加粗编号标题)。"""
    title = rule.get("title", "")
    desc = rule.get("description", "")
    lines: List[str] = []

    if desc:
        lines.append(f"**{title}**：{desc}")
    else:
        lines.append(f"**{title}**：")

    for note in rule.get("sub_notes", []):
        lines.append(f"  - {note}")

    for sr in rule.get("sub_rules", []):
        sr_title = sr.get("title", "")
        sr_desc = sr.get("description", "")
        lines.append(f"  - **{sr_title}**：{sr_desc}")
        for ex in sr.get("examples", []):
            lines.append(f"    - {ex}")
        na = sr.get("not_applicable", "")
        if na:
            lines.append(f"    - **不适用**：{na}")
        for ce in sr.get("counter_examples", []):
            lines.append(f"    - {ce}")
        sr_sups = _find_supplements(supplements, sr_title)
        for sl in _render_supplement_lines(sr_sups, indent="    "):
            lines.append(sl)

    jk = rule.get("judgment_key", "")
    if jk:
        lines.append(f"  - 判定要点：{jk}")

    for ex in rule.get("examples", []):
        lines.append(f"  - {ex}")

    na = rule.get("not_applicable", "")
    if na:
        lines.append(f"  - **不适用**：{na}")

    for ce in rule.get("counter_examples", []):
        lines.append(f"  - {ce}")

    top_sups = _find_supplements(supplements, title)
    lines.extend(_render_supplement_lines(top_sups))

    return lines


def _render_comp_fail_rule(rule: Dict, supplements: List[Dict]) -> List[str]:
    """渲染完成度 fail 规则 (dash 格式)。"""
    title = rule.get("title", "")
    desc = rule.get("description", "")
    lines: List[str] = []
    lines.append(f"- **{title}**：{desc}" if desc else f"- **{title}**：")
    for note in rule.get("sub_notes", []):
        lines.append(f"  - {note}")
    for ex in rule.get("examples", []):
        lines.append(f"  - {ex}")
    lines.extend(_render_supplement_lines(supplements))
    return lines


def render_completion_section() -> str:
    """从 knowledge_base JSON 反序列化维度二的完整 prompt 文本。"""
    c = load_completion_rules()
    extra_pass = c.get("extra_pass_rules", [])
    extra_fail = c.get("extra_fail_rules", [])
    extra_bd = c.get("extra_boundary_clarifications", [])
    lines: List[str] = []

    lines.append(f"# {c['section_title']}")
    lines.append("")
    lines.append(f"**核心问题**：{c['core_question']}")
    lines.append(f"「完美执行」与「基本完成」都算**本维度通过**。")
    lines.append("")
    lines.append(f"## {c['pass_section_title']}")
    lines.append("")
    for rule in c.get("pass_rules", []):
        sups = _find_supplements(extra_pass, rule["title"])
        lines.extend(_render_comp_pass_rule(rule, sups))
        lines.append("")
    new_pass = _get_new_rules(extra_pass)
    for nr in new_pass:
        title = nr.get("title", "")
        desc = nr.get("description", "")
        lines.append(f"**{title}**：{desc}" if desc else f"**{title}**：")
        for ex in nr.get("examples", []):
            lines.append(f"  - {ex}")
        lines.append("")

    lines.append(f"## {c['fail_section_title']}")
    lines.append("")
    for rule in c.get("fail_rules", []):
        sups = _find_supplements(extra_fail, rule["title"])
        lines.extend(_render_comp_fail_rule(rule, sups))
    new_fail = _get_new_rules(extra_fail)
    for nr in new_fail:
        lines.extend(_render_comp_fail_rule(nr, []))

    all_bd = c.get("boundary_clarifications", []) + extra_bd
    if all_bd:
        lines.append("")
        lines.append("**必须辨析的边界**")
        lines.append(format_boundary_as_text(all_bd))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Env Error rules (E1~E6) — 外挂到 knowledge_base/env_error_strategies
# ---------------------------------------------------------------------------

def load_env_error_rules() -> Dict[str, Any]:
    """读取环境错误规则 JSON。文件缺失或解析失败时返回空 dict（调用方需做兜底）。"""
    try:
        return load_dimension_json(_env_error_json_path())
    except Exception:
        return {}


def render_env_error_section() -> str:
    """渲染 E1~E6 环境错误规则为 markdown 文本。

    输出格式与原 hard-coded 段落保持等价：
        # <section_title>
        <preamble>

        **E1. <title>**
        <description>
        > 注意区分：...
        > 例：...
        > 反例：...

        ...
    """
    data = load_env_error_rules()
    if not data:
        return ""

    section_title = data.get("section_title", "环境错误判定规则（不计入成功率）")
    preamble = data.get("preamble", "").strip()
    rules = data.get("rules", []) or []

    lines: List[str] = []
    lines.append(f"# {section_title}")
    if preamble:
        lines.append(preamble)
    lines.append("")

    for rule in rules:
        code = (rule.get("code") or "").strip()
        title = (rule.get("title") or "").strip()
        desc = (rule.get("description") or "").strip()
        notes = rule.get("notes", []) or []
        examples = rule.get("examples", []) or []
        counter_examples = rule.get("counter_examples", []) or []

        header = f"**{code}. {title}**" if code else f"**{title}**"
        lines.append(header)
        if desc:
            lines.append(desc)
        for n in notes:
            n_text = str(n).strip()
            if n_text:
                lines.append(f"> 注意区分：{n_text}" if not n_text.startswith("注意") else f"> {n_text}")
        for ex in examples:
            ex_text = str(ex).strip()
            if ex_text:
                lines.append(f"> 例：{ex_text}")
        for ce in counter_examples:
            ce_text = str(ce).strip()
            if ce_text:
                lines.append(f"> 反例：{ce_text}")
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Negative-task rules — 外挂到 knowledge_base/negative_task_strategies
# 仅在最终判定阶段且 task_kind=="negative" 时注入
# ---------------------------------------------------------------------------

def load_negative_task_rules() -> Dict[str, Any]:
    try:
        return load_dimension_json(_negative_task_json_path())
    except Exception:
        return {}


def _render_simple_rule_lines(rule: Dict[str, Any], bullet: str = "-") -> List[str]:
    """简单规则渲染（title + description + examples）。"""
    title = (rule.get("title") or "").strip()
    desc = (rule.get("description") or "").strip()
    examples = rule.get("examples", []) or []
    lines: List[str] = []
    if title and desc:
        lines.append(f"{bullet} **{title}**：{desc}")
    elif title:
        lines.append(f"{bullet} **{title}**")
    elif desc:
        lines.append(f"{bullet} {desc}")
    for ex in examples:
        ex_text = str(ex).strip()
        if ex_text:
            lines.append(f"  - {ex_text}")
    return lines


def render_negative_task_section() -> str:
    """渲染负例任务（task_kind=negative）规则块。

    返回完整的 markdown 段落，可直接作为 task_kind_section 注入到 prompt 中。
    若 JSON 缺失返回空串。
    """
    data = load_negative_task_rules()
    if not data:
        return ""

    section_title = data.get("section_title", "本题类型：负例（任务本身不可完成）")
    scope_note = (data.get("scope_note") or "").strip()
    intent = (data.get("intent") or "").strip()
    pass_rules = data.get("pass_rules", []) or []
    fail_rules = data.get("fail_rules", []) or []
    preferred_failure_reason = (data.get("preferred_failure_reason") or "").strip()
    judgment_hints = data.get("judgment_hints", []) or []

    lines: List[str] = []
    lines.append(f"### {section_title}")
    if scope_note:
        lines.append(scope_note)
    if intent:
        lines.append(f"- **意图**：{intent}")

    if pass_rules:
        lines.append("")
        lines.append("**通过情形**：")
        for r in pass_rules:
            lines.extend(_render_simple_rule_lines(r, bullet="-"))

    if fail_rules:
        lines.append("")
        lines.append("**失败情形**：")
        for r in fail_rules:
            lines.extend(_render_simple_rule_lines(r, bullet="-"))

    if preferred_failure_reason:
        lines.append("")
        lines.append(
            f"- 若判为**失败**且符合上述「负例场景下未合理求助/终止」，请优先把 first_deviation_reason 填为 **{preferred_failure_reason}**。"
        )

    if judgment_hints:
        lines.append("")
        lines.append("**判定提示**：")
        for h in judgment_hints:
            h_text = str(h).strip()
            if h_text:
                lines.append(f"- {h_text}")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)
