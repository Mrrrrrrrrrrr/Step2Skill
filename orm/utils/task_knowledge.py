"""Task-level knowledge search and rendering for lite gate prompts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import config
from utils.prompt_utils import (
    parse_json_response,
    save_prompt_response,
)
from utils.ask_qwen import call_ark_api


TASK_KNOWLEDGE_REL_PATH = os.path.join("task_knowledge", "task_knowledge.json")
TASK_KNOWLEDGE_HIT_THRESHOLD = 0.82
DEFAULT_MAX_CANDIDATES = 80


def task_knowledge_path() -> str:
    return os.path.join(config.KNOWLEDGE_BASE_DIR, TASK_KNOWLEDGE_REL_PATH)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def empty_task_knowledge() -> Dict[str, Any]:
    return {"version": 1, "items": []}


def load_task_knowledge() -> Dict[str, Any]:
    path = task_knowledge_path()
    if not os.path.exists(path):
        return empty_task_knowledge()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return empty_task_knowledge()
        items = data.get("items")
        if not isinstance(items, list):
            data["items"] = []
        data.setdefault("version", 1)
        return data
    except Exception:
        return empty_task_knowledge()


def save_task_knowledge(data: Dict[str, Any]) -> None:
    path = task_knowledge_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = data if isinstance(data, dict) else empty_task_knowledge()
    data.setdefault("version", 1)
    data.setdefault("items", [])
    fd, tmp_path = tempfile.mkstemp(
        prefix="task_knowledge_", suffix=".json", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def task_knowledge_hash(data: Optional[Dict[str, Any]] = None) -> str:
    if data is None:
        data = load_task_knowledge()
    try:
        text = json.dumps(data, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10] if text else "tknone"


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item) if isinstance(item, dict) else {}
    out["id"] = str(out.get("id") or "").strip()
    out["enabled"] = bool(out.get("enabled", True))
    out["canonical_query"] = str(out.get("canonical_query") or "").strip()
    out["generalized_queries"] = _string_list(out.get("generalized_queries"))[:8]
    out["common_knowledge"] = _string_list(out.get("common_knowledge"))[:12]
    out["success_summary"] = _string_list(out.get("success_summary"))[:12]
    out["failure_summary"] = _string_list(out.get("failure_summary"))[:12]
    source = out.get("source")
    out["source"] = source if isinstance(source, dict) else {}
    return out


def _tokens(text: str) -> set:
    text = (text or "").lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    chars = set(re.findall(r"[\u4e00-\u9fff]", text))
    return words | chars


def _candidate_score(instruction: str, item: Dict[str, Any]) -> float:
    q_tokens = _tokens(instruction)
    if not q_tokens:
        return 0.0
    texts = [item.get("canonical_query", "")] + _string_list(item.get("generalized_queries"))
    kb_tokens = _tokens("\n".join(texts))
    if not kb_tokens:
        return 0.0
    return len(q_tokens & kb_tokens) / max(1, min(len(q_tokens), len(kb_tokens)))


def _select_candidates(instruction: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    max_candidates = int(os.environ.get("ORM_TASK_KNOWLEDGE_MAX_CANDIDATES", DEFAULT_MAX_CANDIDATES))
    enabled = [normalize_item(i) for i in items if isinstance(i, dict)]
    enabled = [i for i in enabled if i.get("enabled") and i.get("id") and i.get("canonical_query")]
    scored = [(_candidate_score(instruction, item), idx, item) for idx, item in enumerate(enabled)]
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return [item for _, _, item in scored[:max(1, max_candidates)]]


def _compact_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id", ""),
        "canonical_query": item.get("canonical_query", "")[:180],
        "generalized_queries": _string_list(item.get("generalized_queries"))[:3],
    }


def _knowledge_text(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    cq = item.get("canonical_query", "")
    if cq:
        parts.append(f"真实任务指令：{cq}")
    gq = _string_list(item.get("generalized_queries"))
    if gq:
        parts.append("泛化类似任务：" + "；".join(gq[:3]))
    ck = _string_list(item.get("common_knowledge"))
    if ck:
        parts.append("通用知识：" + "；".join(ck))
    ss = _string_list(item.get("success_summary"))
    if ss:
        parts.append("成功总结：" + "；".join(ss))
    fs = _string_list(item.get("failure_summary"))
    if fs:
        parts.append("失败总结：" + "；".join(fs))
    return "\n".join(parts).strip()


def _build_search_prompt(instruction: str, candidates: List[Dict[str, Any]]) -> str:
    candidate_json = json.dumps(
        [_compact_candidate(c) for c in candidates], ensure_ascii=False, indent=2
    )
    return f"""你是 GUI Agent 评估前的任务知识检索器。请判断当前 query 是否命中外挂任务知识库中的“同一任务”。

相似度/同一任务定义：
- 必须是同一 App/业务域、同一核心动作、同一核心对象类型、同一最终目标。
- 日期、时间、星期、人数、姓名、金额、地点、商品规格等普通槽位值可以不同。
- 数量、序位和覆盖范围不是普通槽位："第一个"、"前 N 个"、"N 个不同对象"、"全部/所有"、"最新/最近"等会改变完成标准，必须保持同类约束；单个对象收藏知识不能命中多个不同对象收藏任务，多对象知识也不能命中单对象任务。
- 核心对象粒度必须一致："第一个视频"与"5篇不同教程/笔记"不是同一任务，即使平台和收藏动作相同也必须判不命中。
- 订明天的火车票 与 订后天的火车票：同一任务。
- 订火车票 与 订飞机票：不同任务。
- 美团签到 与 支付宝签到：不同任务。
- 设置每周一闹钟 与 设置每周二闹钟：同一任务。
- 若只是泛泛同类但核心对象或平台不同，必须判不命中。

当前 query：
{instruction}

候选知识项：
{candidate_json}

请只输出严格 JSON，不要 markdown：
{{
  "hit": true,
  "matched_id": "候选 id 或空",
  "matched_query": "命中的 canonical_query 或空",
  "confidence": 0.0,
  "match_reasoning": "一句话说明为什么是/不是同一任务"
}}
"""


def _empty_match(reason: str = "") -> Dict[str, Any]:
    return {
        "hit": False,
        "matched_id": "",
        "matched_query": "",
        "confidence": 0.0,
        "matched_knowledge_text": "",
        "match_reasoning": reason,
    }


def search_task_knowledge(
    instruction: str,
    *,
    task_id: str = "",
    step_name: str = "task_knowledge_search",
    threshold: float = TASK_KNOWLEDGE_HIT_THRESHOLD,
    enabled: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """Return (match, knowledge_hash). Failures degrade to no-hit.

    When enabled=False (CLI --no-task-knowledge), skip retrieval entirely and
    return a stable cache-isolating version token ``off``.
    """
    if not enabled:
        return _empty_match("已禁用任务知识（--no-task-knowledge）"), "off"
    data = load_task_knowledge()
    version = task_knowledge_hash(data)
    items = data.get("items") if isinstance(data, dict) else []
    if not isinstance(items, list) or not items:
        return _empty_match("知识库为空"), version
    candidates = _select_candidates(instruction, items)
    if not candidates:
        return _empty_match("无可用知识项"), version

    prompt = _build_search_prompt(instruction, candidates)
    try:
        response, usage = call_ark_api(query=prompt, imgs=None, model_name="seed_free")
        if task_id and response:
            full_response = {"choices": [{"message": {"content": response}}]}
            if usage:
                full_response["usage"] = usage
            save_prompt_response(
                task_id,
                f"{step_name}_{version}",
                prompt,
                images=None,
                response_text=response,
                full_response=full_response,
            )
    except Exception as exc:
        return _empty_match(f"知识检索失败：{exc}"), version

    parsed = parse_json_response(response or "")
    if not isinstance(parsed, dict):
        return _empty_match("知识检索输出无法解析"), version

    matched_id = str(parsed.get("matched_id") or "").strip()
    by_id = {str(c.get("id")): c for c in candidates}
    item = by_id.get(matched_id)
    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    hit = bool(parsed.get("hit")) and item is not None and confidence >= threshold
    if not hit:
        return {
            "hit": False,
            "matched_id": matched_id if item is not None else "",
            "matched_query": str(parsed.get("matched_query") or "").strip(),
            "confidence": confidence,
            "matched_knowledge_text": "",
            "match_reasoning": str(parsed.get("match_reasoning") or "").strip(),
        }, version

    return {
        "hit": True,
        "matched_id": item.get("id", ""),
        "matched_query": item.get("canonical_query", ""),
        "confidence": confidence,
        "matched_knowledge_text": _knowledge_text(item),
        "match_reasoning": str(parsed.get("match_reasoning") or "").strip(),
    }, version


def render_task_knowledge_section(match: Dict[str, Any]) -> str:
    if not isinstance(match, dict) or not match.get("hit"):
        return ""
    text = str(match.get("matched_knowledge_text") or "").strip()
    if not text:
        return ""
    confidence = match.get("confidence", 0.0)
    try:
        confidence_text = f"{float(confidence):.2f}"
    except Exception:
        confidence_text = "0.00"
    reason = str(match.get("match_reasoning") or "").strip()
    return f"""# 外挂任务知识（高置信参考，必须结合截图证据使用）
- 命中 query：{match.get('matched_query', '')}
- 命中置信度：{confidence_text}
- 命中原因：{reason or '空'}

{text}

使用要求：
- 这份知识只适用于同一任务的小范围经验，不可扩展到不同 App、不同核心动作或不同最终目标。
- 它只能帮助你核查关键约束、终态要求、常见成功/失败模式；最终判定仍必须以截图和轨迹证据为准。
- 输出 JSON 时必须填写 task_knowledge_hit / task_knowledge_confidence / task_knowledge_used / task_knowledge_reasoning。
""".strip()


def summarize_match_for_reasoning(judgment: Dict[str, Any]) -> str:
    if not isinstance(judgment, dict):
        return "空"
    match = judgment.get("task_knowledge_match")
    used = judgment.get("task_knowledge_used")
    reasoning = judgment.get("task_knowledge_reasoning")
    if isinstance(used, str) and used.strip():
        return used.strip()
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    if isinstance(match, dict) and match.get("hit"):
        query = str(match.get("matched_query") or "").strip()
        conf = match.get("confidence", 0.0)
        try:
            conf_text = f"{float(conf):.2f}"
        except Exception:
            conf_text = "0.00"
        return f"{query}（置信度 {conf_text}）" if query else f"命中任务知识（置信度 {conf_text}）"
    return "空"
