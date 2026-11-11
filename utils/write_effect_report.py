#!/usr/bin/env python3
"""Auto-generate iterations/iterNN/effect_report.md from cluster/change_plan + eval artifacts."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpyxl import load_workbook

from utils.codex_thread_helper import metrics_from_eval_summary
from utils.input_resolver import STATUS_CLASS_MAP, _find_header_and_id_column, _jsonl_prediction_rows

STATUS_NAME = {1: "success", 0: "failed", -1: "env_error"}


def _resolve_input_xlsx(workdir: Path, override: Optional[Path] = None) -> Path:
    if override is not None:
        return override
    manifest = workdir / "manifest.json"
    if manifest.exists():
        art = json.loads(manifest.read_text(encoding="utf-8")).get("artifacts") or {}
        rel = art.get("input_xlsx")
        if rel:
            p = Path(rel)
            return p if p.is_absolute() else (REPO_ROOT / p)
    return REPO_ROOT / "data/tmp/回流数据评估情况.xlsx"


def _cell_value(row: Tuple[Any, ...], headers: Dict[str, int], key: str) -> str:
    idx = headers.get(key)
    if idx is None:
        return ""
    zero = idx - 1
    if zero >= len(row):
        return ""
    v = row[zero]
    return "" if v is None else str(v).strip()


def _iter_xlsx_rows(xlsx: Path):
    """Load xlsx without read_only (read_only breaks dimension on some files)."""
    wb = load_workbook(xlsx, data_only=True)
    ws = wb.worksheets[0]
    header_row, _id_col, headers = _find_header_and_id_column(ws)
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        yield row, headers
    wb.close()


def _human_status_map(xlsx: Path) -> Dict[str, int]:
    """task_id -> human label (1/0/-1)."""
    if not xlsx.exists():
        return {}
    out: Dict[str, int] = {}
    for row, headers in _iter_xlsx_rows(xlsx):
        tid = _cell_value(row, headers, "task_id")
        if not tid:
            continue
        human = STATUS_CLASS_MAP.get(_cell_value(row, headers, "人评结果"))
        if human is not None:
            out[tid] = human
    return out


def _zj_status_map_from_xlsx(xlsx: Path) -> Dict[str, int]:
    """task_id -> zj_status from xlsx column (if backfilled)."""
    if not xlsx.exists():
        return {}
    out: Dict[str, int] = {}
    for row, headers in _iter_xlsx_rows(xlsx):
        if "zj_status" not in headers:
            return out
        tid = _cell_value(row, headers, "task_id")
        if not tid:
            continue
        pred = STATUS_CLASS_MAP.get(_cell_value(row, headers, "zj_status"))
        if pred is not None:
            out[tid] = pred
    return out


def _read_text(path: Path, limit: int = 8000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:limit].strip()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    return f"{v * 100:.2f}%"


def _delta(cur: Optional[float], base: Optional[float]) -> str:
    if cur is None or base is None:
        return "NA"
    d = (cur - base) * 100
    return f"{d:+.2f}%"


def _delta_int(cur: int, base: int) -> str:
    d = cur - base
    return f"{d:+d}" if d else "0"


def _task_status_map(xlsx: Path) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
    """Return task_id -> (human, zj_status)."""
    humans = _human_status_map(xlsx)
    preds = _zj_status_map_from_xlsx(xlsx)
    keys = set(humans) | set(preds)
    return {tid: (humans.get(tid), preds.get(tid)) for tid in keys}


def _baseline_zj_from_json(baseline: Dict[str, Any], task_id: str) -> Optional[int]:
    samples = baseline.get("mismatch_samples") or {}
    for bucket in samples.values():
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if isinstance(item, dict) and item.get("task_id") == task_id:
                return STATUS_CLASS_MAP.get(str(item.get("zj_status", "")))
    return None


CLUSTER_TITLE_ZH: Dict[str, str] = {
    "Action Result State Missing": "动作结果态缺失",
    "Object Identity / Type Mismatch": "对象身份/类型不匹配",
    "Sequence / Time / Coverage Not Closed": "序位/时间/覆盖范围未闭合",
    "Information Answer Source Not Verifiable": "信息答案来源不可验证",
    "Structured Checklist Evasion via `n/a`": "用 n/a 规避结构化检查清单",
    "Superficial Success Evidence": "成功证据过于表面",
    "Multiple Evidence Dimensions in One Task": "单任务多维度证据要求叠加",
}

CHANGE_TYPE_ZH = {
    "Prompt": "提示词",
    "Knowledge": "知识库",
    "Framework": "框架结构化字段",
    "Framework structured normalization": "框架结构化字段",
}

_PLAN_PHRASE_ZH = [
    (r"Add a compact hard-veto checklist", "新增硬性否决检查清单"),
    (r"object identity/type", "对象身份/类型"),
    (r"result-state verification", "动作结果态核验"),
    (r"sequence/time/range coverage", "序位/时间/范围覆盖"),
    (r"information answer source closure", "信息答案来源闭环"),
    (r"completion_evidence", "completion_evidence"),
    (r"rather than `n/a` when triggered", "触发时禁止填 n/a"),
    (r"Require at least one substantive", "要求至少一项实质检查为 yes"),
    (r"force `completion_pass=false`", "强制 completion_pass=false"),
]

_CLUSTER_BODY_ZH: Dict[str, str] = {
    "task requires a state-changing action, but final screenshots do not show the post-action result state.":
        "任务要求状态变更动作，但终态截图未展示动作后的结果态。",
    "message send lacks sent bubble; video call lacks calling/waiting page; pause/fullscreen/fast-forward lacks final playback state; copy/save/delete/collect lacks toast, button state, object disappearance, or list update.":
        "发消息无气泡、通话无等待页、暂停/全屏/快进无最终播放态；复制/保存/删除/收藏无 toast、按钮态变化、对象消失或列表更新。",
    "the agent performs a plausible operation on the wrong object type or wrong target object, then terminates success.":
        "Agent 对错误对象类型或错误目标执行了看似合理的操作，随后自报 success。",
    "Weibo main app vs Weibo clone; photo vs screenshot; text note vs handwriting note; video vs image post; product not matching requested model/category.":
        "微博主 App vs 分身、照片 vs 截图、文字笔记 vs 手写笔记、视频 vs 图文帖、商品型号/品类不符。",
    "task includes latest/recent/first/top N/all/today/yesterday/this Wednesday, but screenshots only show a default list, partial list, or one candidate without proving ordering, range, or coverage.":
        "任务含最新/最近/第一条/前 N/全部/今天/昨天/本周三等约束，但截图只有默认列表、局部列表或单个候选，未证明排序/范围/覆盖。",
    "latest favorite article deletion, yesterday order lookup, recent screenshot/photo checks, first comment like, all recent moments liked.":
        "删最新收藏文章、查昨天订单、核对最近截图/照片、给第一条评论点赞、给近期朋友圈全部点赞。",
    "query/statistics/recommendation tasks return success with an answer, but screenshots do not support the final value, comparison, or coverage.":
        "查询/统计/推荐类任务输出了答案，但截图无法支撑最终数值、比较结论或覆盖范围。",
    "total marathon count, running mileage today, all empty folders, lowest unit price, cold-spot travel note organization.":
        "马拉松总场次、今日跑步里程、全部空文件夹、单价最低、冷门景点攻略整理到笔记。",
}


def _translate_snippet(text: str) -> str:
    s = text.strip()
    if s in _CLUSTER_BODY_ZH:
        return _CLUSTER_BODY_ZH[s]
    out = s
    m = re.match(
        r"Add a compact hard-veto checklist for (.+)\. Require these checks before success and require `completion_evidence` gates to be `no` rather than `n/a` when triggered\.",
        s,
        flags=re.I,
    )
    if m:
        items = m.group(1)
        return (
            f"新增硬性否决检查清单（{items}）；成功前必须完成对应检查，"
            "触发项 completion_evidence 禁止填 n/a、证据不足须填 no。"
        )
    for pat, zh in _PLAN_PHRASE_ZH:
        if pat.startswith("("):
            continue
        out = re.sub(re.escape(pat), zh, out, flags=re.I)
    return out


def _plan_summary_zh(file_path: str, ctype: str, plan: str) -> str:
    if "lite_trajectory_unified_prompt" in file_path and _cjk_ratio(plan) < 0.2:
        return (
            "在统一轨迹判定提示词中新增「四项否决检查」与 completion_evidence 触发绑定表；"
            "明确对象身份、动作结果态、序位/时间/范围、信息来源四类硬性规则；"
            "适用检查项禁止填 n/a，截图证据不足须填 no 并判 failed。"
        )
    if file_path.endswith(".json") and "completed" in file_path:
        return "在 completed 知识库中补充场景化判例与检查门槛说明。"
    if "pipeline_runner" in file_path:
        return "在框架结构化归一化层收紧 completion_pass 与 checklist 字段约束。"
    direct = _translate_snippet(plan)
    if _cjk_ratio(direct) > 0.15:
        return direct
    return direct if _cjk_ratio(direct) > 0.05 else plan


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(len(text), 1)


def _format_problem_section_zh(cluster_md: str) -> str:
    if not cluster_md:
        return "- （未找到 cluster_report.md）"
    if _cjk_ratio(cluster_md) > 0.25:
        lines = [ln for ln in cluster_md.splitlines() if ln.strip() and not ln.startswith("# Scope")]
        return "\n".join(lines[:20])

    out: List[str] = [
        "本轮针对 **0→1 假阳性**（人评 failed、ORM 判 success）做聚类，主要问题模式如下：",
        "",
    ]
    cluster_re = re.compile(r"^##\s+Cluster\s+([A-Z]):\s*(.+)$", re.M)
    matches = list(cluster_re.finditer(cluster_md))
    if not matches:
        return cluster_md[:1200]

    for idx, m in enumerate(matches):
        cid, title_en = m.group(1), m.group(2).strip()
        title_zh = CLUSTER_TITLE_ZH.get(title_en, title_en)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cluster_md)
        block = cluster_md[start:end]
        out.append(f"### 聚类 {cid}：{title_zh}")
        for ln in block.splitlines():
            s = ln.strip()
            if s.startswith("- Pattern:"):
                out.append(f"- **模式**：{_translate_snippet(s[len('- Pattern:'):].strip())}")
            elif s.startswith("- Examples:"):
                out.append(f"- **示例**：{_translate_snippet(s[len('- Examples:'):].strip())}")
            elif s.startswith("- Impact:"):
                out.append(f"- **影响**：{_translate_snippet(s[len('- Impact:'):].strip())}")
            elif s.startswith("- Risk:"):
                out.append(f"- **风险**：{_translate_snippet(s[len('- Risk:'):].strip())}")
            elif s.startswith("- Representative task_ids:"):
                ids = s.split(":", 1)[-1].strip().strip("`")
                out.append(f"- **代表样本**：{s.split(':', 1)[-1].strip()}")
        out.append("")

    return "\n".join(out).strip()


def _find_snapshot_file(snap_dir: Path, rel_path: str) -> Optional[Path]:
    if not snap_dir.exists():
        return None
    direct = snap_dir / rel_path
    if direct.is_file():
        return direct
    by_name = snap_dir / Path(rel_path).name
    if by_name.is_file():
        return by_name
    name = Path(rel_path).name
    for hit in snap_dir.rglob(name):
        if hit.is_file():
            return hit
    return None


def _after_snapshot_dir(workdir: Path, iteration: int) -> Tuple[Path, str]:
    iter_id = f"iter{iteration:02d}"
    after = workdir / "snapshots" / f"{iter_id}_after"
    if after.is_dir() and any(after.iterdir()):
        return after, f"snapshots/{iter_id}_after"
    nxt = workdir / "snapshots" / f"iter{iteration + 1:02d}_before"
    if nxt.is_dir() and any(nxt.iterdir()):
        return nxt, f"snapshots/iter{iteration + 1:02d}_before（本轮末态）"
    return REPO_ROOT, "当前代码库（无 after snapshot，仅供参考）"


def _grep_style_diff(
    rel_path: str,
    before_lines: List[str],
    after_lines: List[str],
    max_lines: int = 55,
    context: int = 1,
) -> List[str]:
    sm = difflib.SequenceMatcher(None, before_lines, after_lines)
    rows: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("delete", "replace") and i1 > 0 and context:
            for i in range(max(i1 - context, 0), i1):
                if len(rows) < max_lines:
                    rows.append(f"{rel_path}-{i + 1}-- {before_lines[i]}")
        if tag in ("delete", "replace"):
            for i in range(i1, i2):
                if len(rows) >= max_lines:
                    break
                rows.append(f"{rel_path}-{i + 1}-- {before_lines[i]}")
        if tag in ("insert", "replace"):
            for j in range(j1, j2):
                if len(rows) >= max_lines:
                    break
                line = after_lines[j]
                if not line.strip():
                    continue
                marker = "+" if tag == "insert" else ":"
                rows.append(f"{rel_path}:{j + 1}:{marker} {line}")
    if len(rows) > max_lines:
        extra = len(rows) - max_lines
        rows = rows[:max_lines] + [f"...（另有 {extra} 行变更未展示）"]
    return rows


def _files_from_change_plan(change_md: str) -> List[str]:
    seen: List[str] = []
    for m in re.finditer(r"`([^`]+\.(?:py|json))`", change_md):
        rel = m.group(1)
        if rel not in seen:
            seen.append(rel)
    return seen


def _parse_change_plan_bullets_zh(change_md: str) -> List[str]:
    bullets: List[str] = []
    for ln in change_md.splitlines():
        s = ln.strip()
        if not s.startswith("|") or "---" in s:
            continue
        low = s.lower()
        if low.startswith("| area") or low.startswith("| 文件") or low.startswith("| file"):
            continue
        parts = [p.strip() for p in s.split("|")[1:-1]]
        if len(parts) < 4:
            continue
        _area, file_cell, ctype, plan = parts[0], parts[1], parts[2], parts[3]
        file_path = file_cell.strip("`").strip()
        ctype_zh = CHANGE_TYPE_ZH.get(ctype.strip(), ctype.strip())
        plan_zh = _plan_summary_zh(file_path, ctype.strip(), plan)
        bullets.append(f"- **{ctype_zh}** `{file_path}`：{plan_zh}")
    return bullets


def _format_changes_section(workdir: Path, iteration: int, change_md: str) -> str:
    iter_id = f"iter{iteration:02d}"
    before_dir = workdir / "snapshots" / f"{iter_id}_before"
    after_dir, after_label = _after_snapshot_dir(workdir, iteration)
    bullets = _parse_change_plan_bullets_zh(change_md)
    files = _files_from_change_plan(change_md)

    sections: List[str] = ["### 改动摘要", ""]
    if bullets:
        sections.extend(bullets)
    else:
        sections.append("- （见 change_plan.md）")
    sections.extend(["", "### 代码变更摘录（snapshot diff）", ""])
    sections.append(f"> 对比 `snapshots/{iter_id}_before/` → `{after_label}`")
    sections.append("")

    diff_blocks: List[str] = []
    for rel in files:
        before_path = _find_snapshot_file(before_dir, rel)
        after_path = _find_snapshot_file(after_dir, rel)
        if not before_path or not after_path:
            diff_blocks.append(f"# {rel}：缺少 snapshot，无法生成 diff")
            continue
        before_text = before_path.read_text(encoding="utf-8", errors="replace")
        after_text = after_path.read_text(encoding="utf-8", errors="replace")
        if before_text == after_text:
            continue
        rows = _grep_style_diff(rel, before_text.splitlines(), after_text.splitlines())
        if rows:
            diff_blocks.append(f"# {rel}")
            diff_blocks.extend(rows)
            diff_blocks.append("")

    if diff_blocks:
        sections.append("```bash")
        sections.extend(diff_blocks)
        if sections[-1] != "```":
            sections.append("```")
    else:
        sections.append("（未找到可对比的 snapshot diff；请确认 `snapshots/{0}_before/` 存在）".format(iter_id))

    return "\n".join(sections)


def _decide(
    cur: Dict[str, Any],
    base: Dict[str, Any],
    prev: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    cur_acc = float(cur.get("accuracy") or 0)
    base_acc = float(base.get("accuracy") or 0)
    ref = prev or base
    ref_acc = float(ref.get("accuracy") or 0)
    cur_mm = cur.get("mismatch_counts") or {}
    ref_mm = ref.get("mismatch_counts") or {}
    fp = int(cur_mm.get("0->1", 0) or 0)
    ref_fp = int(ref_mm.get("0->1", 0) or 0)
    fn = int(cur_mm.get("1->0", 0) or 0)
    ref_fn = int(ref_mm.get("1->0", 0) or 0)

    if cur_acc < ref_acc - 0.005:
        return "Revert", f"准确率较上轮下降（{ref_acc:.2%} → {cur_acc:.2%}），优先回退本轮改动。"
    if fp > ref_fp + 3:
        return "Revert", f"0→1 假阳性从 {ref_fp} 增至 {fp}，改动方向可能错误。"
    if cur_acc >= ref_acc and fp <= ref_fp:
        return "Keep", f"准确率 {ref_acc:.2%} → {cur_acc:.2%}，0→1 {ref_fp} → {fp}，可保留。"
    return "Continue", (
        f"准确率 {ref_acc:.2%} → {cur_acc:.2%}，0→1 {ref_fp} → {fp}，"
        f"1→0 {ref_fn} → {fn}；未达标但可继续迭代。"
    )


def _metrics_row(label: str, cur: Dict[str, Any], base: Dict[str, Any]) -> List[str]:
    cm = metrics_from_eval_summary(cur)
    bm = metrics_from_eval_summary(base)
    mm = cur.get("mismatch_counts") or {}
    bmm = base.get("mismatch_counts") or {}
    return [
        label,
        _pct(float(cur.get("accuracy") or 0)),
        _pct(float(base.get("accuracy") or 0)),
        _delta(float(cur.get("accuracy") or 0), float(base.get("accuracy") or 0)),
        _pct(cm["positive_f1"]),
        _pct(bm["positive_f1"]),
        _delta(cm["positive_f1"], bm["positive_f1"]),
        _pct(cm["negative_f1"]),
        _pct(bm["negative_f1"]),
        _delta(cm["negative_f1"], bm["negative_f1"]),
        str(mm.get("0->1", "")),
        str(bmm.get("0->1", "")),
        _delta_int(int(mm.get("0->1", 0) or 0), int(bmm.get("0->1", 0) or 0)),
        str(mm.get("1->0", "")),
        str(bmm.get("1->0", 0) or 0),
        _delta_int(int(mm.get("1->0", 0) or 0), int(bmm.get("1->0", 0) or 0)),
    ]


def _task_status_map_from_jsonl(jsonl_path: Path) -> Dict[str, int]:
    """task_id -> zj_status from evaluation.prediction.task_status."""
    if not jsonl_path.exists():
        return {}
    rows = _jsonl_prediction_rows(str(jsonl_path))
    out: Dict[str, int] = {}
    for tid, fields in rows.items():
        pred = STATUS_CLASS_MAP.get(str(fields.get("zj_status", "")).strip())
        if pred is not None:
            out[tid] = pred
    return out


def _resolve_after_map(workdir: Path, iter_id: str) -> Dict[str, int]:
    """Prefer jsonl predictions; output xlsx often lacks zj_status backfill."""
    targeted_dir = workdir / "eval" / f"{iter_id}_targeted"
    merged: Dict[str, int] = {}
    for jsonl in sorted(targeted_dir.glob("output_*.jsonl")):
        merged.update(_task_status_map_from_jsonl(jsonl))
    if merged:
        return merged
    for xlsx in sorted(targeted_dir.glob("output_*.xlsx")):
        preds = _zj_status_map_from_xlsx(xlsx)
        if preds:
            return preds
    return {}


def build_targeted_table(
    workdir: Path,
    iter_id: str,
    iteration: int,
    baseline: Dict[str, Any],
    input_xlsx: Path,
) -> Tuple[str, List[str]]:
    ids_path = workdir / "iterations" / iter_id / "target_task_ids.json"
    if not ids_path.exists():
        return "- 本轮无 targeted 集（无 target_task_ids.json）\n", []

    task_ids: List[str] = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(task_ids, list):
        task_ids = []

    after_xlsx = workdir / "eval" / f"{iter_id}_targeted" / "output_回流数据评估情况_raw_data.xlsx"
    prev_id = f"iter{iteration - 1:02d}"
    before_xlsx = workdir / "eval" / prev_id / "output_回流数据评估情况_raw_data.xlsx"
    after_map = _resolve_after_map(workdir, iter_id)
    before_map = _zj_status_map_from_xlsx(before_xlsx) if before_xlsx.exists() else {}
    human_map = _human_status_map(input_xlsx)
    # targeted 输出 xlsx 常含人评但缺 zj_status；补读 targeted xlsx 的人评列
    if after_xlsx.exists():
        for tid, human in _human_status_map(after_xlsx).items():
            human_map.setdefault(tid, human)

    lines = [
        f"- 目标 task_id：`iterations/{iter_id}/target_task_ids.json`（{len(task_ids)} 个）",
        f"- targeted 输出：`eval/{iter_id}_targeted/`",
        f"- targeted log：`iterations/{iter_id}/targeted_validation.log`",
        "",
        "| task_id | 人评 | 修改前 zj | 修改后 zj | 符合预期 | 备注 |",
        "|---------|------|-----------|-----------|----------|------|",
    ]
    regressions: List[str] = []

    for tid in task_ids:
        human = human_map.get(tid)
        after_pred = after_map.get(tid)
        if before_map:
            before_pred = before_map.get(tid)
        else:
            before_pred = _baseline_zj_from_json(baseline, tid)

        human_s = STATUS_NAME.get(human, "NA") if human is not None else "NA"
        before_s = STATUS_NAME.get(before_pred, "NA") if before_pred is not None else "NA"
        after_s = STATUS_NAME.get(after_pred, "NA") if after_pred is not None else "NA"

        if human is None:
            ok = "未知"
            note = "输入表无人评标注"
        elif after_pred is None:
            ok = "未知"
            note = "targeted 未跑出（常见原因：默认 status 过滤排除了 perfect，需加 --filter failed,fair_enough,env_error,perfect）"
        elif human == after_pred:
            ok = "是"
            note = "未回归" if before_pred == after_pred else "已修正"
        else:
            ok = "否"
            note = "仍不一致"
            if human == 1 and after_pred == 0:
                regressions.append(tid)
                note = "回归：人评成功但本轮判失败"

        lines.append(f"| `{tid}` | {human_s} | {before_s} | {after_s} | {ok} | {note} |")

    lines.append("")
    if regressions:
        lines.append("非预期变化：" + "；".join(f"`{t}`" for t in regressions))
    else:
        lines.append("非预期变化：无")
    return "\n".join(lines) + "\n", regressions


def write_effect_report(workdir: Path, iteration: int, input_xlsx: Optional[Path] = None) -> Path:
    iter_id = f"iter{iteration:02d}"
    iter_dir = workdir / "iterations" / iter_id
    iter_dir.mkdir(parents=True, exist_ok=True)
    out_path = iter_dir / "effect_report.md"

    cluster_md = _read_text(iter_dir / "cluster_report.md")
    change_md = _read_text(iter_dir / "change_plan.md")
    baseline = _load_json(workdir / "baseline" / "current_orm_iter_summary.json")
    cur = _load_json(workdir / "eval" / f"{iter_id}_metrics.json")
    prev = _load_json(workdir / "eval" / f"iter{iteration - 1:02d}_metrics.json") if iteration > 1 else None

    if not cur:
        raise FileNotFoundError(f"missing eval/{iter_id}_metrics.json")

    input_xlsx = _resolve_input_xlsx(workdir, input_xlsx)

    decision, reason = _decide(cur, baseline, prev)
    targeted_sec, _ = build_targeted_table(workdir, iter_id, iteration, baseline, input_xlsx)

    cur_acc = float(cur.get("accuracy") or 0)
    base_acc = float(baseline.get("accuracy") or 0)
    prev_acc = float((prev or baseline).get("accuracy") or 0)
    cur_mm = cur.get("mismatch_counts") or {}
    base_mm = baseline.get("mismatch_counts") or {}
    prev_mm = (prev or baseline).get("mismatch_counts") or {}

    cm = metrics_from_eval_summary(cur)
    bm = metrics_from_eval_summary(baseline)

    change_section = _format_changes_section(workdir, iteration, change_md)

    body = f"""# {iter_id.upper()} 效果报告

> 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（`utils/write_effect_report.py`）

## 1. 问题描述

{_format_problem_section_zh(cluster_md)}

- **影响桶**：主要关注 `0→1`（人评 failed → ORM success）
- **baseline 准确率**：{_pct(base_acc)}（{baseline.get('correct')}/{baseline.get('valid')}）
- **本轮准确率**：{_pct(cur_acc)}（{cur.get('correct')}/{cur.get('valid')}）

## 2. 修改内容

{change_section}

详见：`iterations/{iter_id}/change_plan.md`

## 3. Targeted Validation

{targeted_sec}

## 4. Full Evaluation

- 命令：`scripts/run_iter_eval.sh --workdir ... --iter {iteration}`
- 全量 log：`eval/full_{iter_id}/full_eval_no_cache.log`
- 归档：`eval/{iter_id}/output_*.xlsx`
- metrics：`eval/{iter_id}_metrics.json`

| 指标 | 本轮 | baseline | 较 baseline | 较上轮 |
|------|------|----------|-------------|--------|
| 三分类准确率 | {_pct(cur_acc)} | {_pct(base_acc)} | {_delta(cur_acc, base_acc)} | {_delta(cur_acc, prev_acc)} |
| 正例 F1 | {_pct(cm['positive_f1'])} | {_pct(bm['positive_f1'])} | {_delta(cm['positive_f1'], bm['positive_f1'])} | — |
| 负例 F1 | {_pct(cm['negative_f1'])} | {_pct(bm['negative_f1'])} | {_delta(cm['negative_f1'], bm['negative_f1'])} | — |
| 0→1 FP 数 | {cur_mm.get('0->1', '')} | {base_mm.get('0->1', '')} | {_delta_int(int(cur_mm.get('0->1', 0) or 0), int(base_mm.get('0->1', 0) or 0))} | {_delta_int(int(cur_mm.get('0->1', 0) or 0), int(prev_mm.get('0->1', 0) or 0))} |
| 1→0 FN 数 | {cur_mm.get('1->0', '')} | {base_mm.get('1->0', '')} | {_delta_int(int(cur_mm.get('1->0', 0) or 0), int(base_mm.get('1->0', 0) or 0))} | {_delta_int(int(cur_mm.get('1->0', 0) or 0), int(prev_mm.get('1->0', 0) or 0))} |

指标变化摘要：**准确率 {prev_acc:.2%} → {cur_acc:.2%}**；**0→1 {prev_mm.get('0->1', '')} → {cur_mm.get('0->1', '')}**

## 5. 决策

- [{'x' if decision == 'Keep' else ' '}] Keep
- [{'x' if decision == 'Continue' else ' '}] Continue
- [{'x' if decision == 'Revert' else ' '}] Revert

决策：**{decision}**

理由：{reason}
"""
    out_path.write_text(body, encoding="utf-8")
    _update_timeline_decision(workdir, iter_id, decision, cur)
    return out_path


def _update_timeline_decision(workdir: Path, iter_id: str, decision: str, cur: Dict[str, Any]) -> None:
    timeline = workdir / "metrics_timeline.md"
    if not timeline.exists():
        return
    acc = float(cur.get("accuracy") or 0)
    acc_s = f"{acc * 100:.2f}%"
    mm = cur.get("mismatch_counts") or {}
    text = timeline.read_text(encoding="utf-8")
    new_lines: List[str] = []
    for ln in text.splitlines():
        if ln.startswith(f"| {iter_id} |"):
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 10:
                parts[2] = f"见 effect_report"
                parts[3] = acc_s
                parts[4] = f"{cur.get('correct')}/{cur.get('valid')}"
                parts[5] = str(mm.get("0->1", ""))
                parts[6] = str(mm.get("1->0", ""))
                parts[7] = str(mm.get("-1->0", ""))
                parts[8] = str(mm.get("0->-1", ""))
                parts[9] = decision.lower()
                new_lines.append("| " + " | ".join(parts[1:-1]) + " |")
                continue
        new_lines.append(ln)
    timeline.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate iterations/iterNN/effect_report.md")
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--iteration", type=int, required=True)
    args = p.parse_args()
    out = write_effect_report(args.workdir.resolve(), args.iteration)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
