#!/usr/bin/env python3
"""Codex thread helper: context usage check + new-thread handoff brief."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_THRESHOLD = 200_000
DEFAULT_LIMIT = 258_400
CODEX_SESSIONS = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEY_DIR_CANDIDATES = [
    "README.md",
    "main.py",
    "src/",
    "app/",
    "scripts/",
    "utils/",
    "tests/",
    ".cursor/",
]


def _latest_rollout(cwd: Optional[Path] = None) -> Optional[Path]:
    if not CODEX_SESSIONS.exists():
        return None
    candidates = list(CODEX_SESSIONS.glob("**/*.jsonl"))
    if not candidates:
        return None
    cwd_s = str(cwd or Path.cwd()).rstrip("/")

    def score(p: Path) -> Tuple[float, float]:
        mtime = p.stat().st_mtime
        match = 0.0
        if cwd is not None:
            try:
                for line in p.open(encoding="utf-8"):
                    if '"cwd"' not in line:
                        continue
                    if cwd_s in line:
                        match = 1.0
                        break
            except OSError:
                pass
        return (match, mtime)

    return max(candidates, key=score)


def read_context_usage(rollout: Optional[Path] = None) -> Dict[str, Any]:
    path = rollout or _latest_rollout()
    if path is None or not path.exists():
        return {
            "rollout": None,
            "tokens": 0,
            "context_window": DEFAULT_LIMIT,
            "thread_id": None,
            "updated_at": None,
        }

    tokens = 0
    context_window = DEFAULT_LIMIT
    updated_at = None
    thread_id = None
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "event_msg" and (obj.get("payload") or {}).get("type") == "token_count":
            info = (obj.get("payload") or {}).get("info") or {}
            last = info.get("last_token_usage") or {}
            tokens = int(last.get("total_tokens") or 0)
            context_window = int(info.get("model_context_window") or DEFAULT_LIMIT)
            updated_at = obj.get("timestamp")
            break
    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "turn_context":
            thread_id = (obj.get("payload") or {}).get("thread_id")
            break

    return {
        "rollout": str(path),
        "tokens": tokens,
        "context_window": context_window,
        "thread_id": thread_id,
        "updated_at": updated_at,
    }


def _read_text(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...(truncated)"


def _latest_iter_notes(workdir: Path) -> Optional[Path]:
    iters = sorted((workdir / "iterations").glob("iter*/notes.md"))
    return iters[-1] if iters else None


def next_iteration(workdir: Path) -> int:
    iters_dir = workdir / "iterations"
    if not iters_dir.exists():
        return 1
    nums: List[int] = []
    for p in iters_dir.iterdir():
        name = p.name
        if name.startswith("iter") and name[4:].isdigit():
            nums.append(int(name[4:]))
    if not nums:
        return 1
    for n in sorted(nums):
        metrics = workdir / "eval" / f"iter{n:02d}_metrics.json"
        full_done = workdir / "eval" / f"full_iter{n:02d}" / "full_eval.done"
        if not metrics.exists() or not full_done.exists():
            return n
    return max(nums) + 1


def build_iter_start_prompt(
    workdir: Path,
    iteration: int,
    hypothesis: str = "",
    extra: str = "",
    loop_index: Optional[int] = None,
    loop_total: Optional[int] = None,
) -> str:
    iter_id = f"iter{iteration:02d}"
    handoff = build_handoff(workdir, extra=extra)
    hypothesis = hypothesis.strip()
    hypothesis_block = hypothesis or _read_text(workdir / "iterations" / iter_id / "notes.md", 800)
    text = "\n".join(
        [
            handoff,
            "",
            f"## 本轮任务 ({iter_id})",
            f"- iteration: `{iter_id}`",
            f"- workspace: `{workdir}`",
            f"- skill: `.cursor/skills/orm-judgment-optimizer/SKILL.md`",
            "",
            "### 硬性约束（违反则本轮无效）",
            "- **裸评估**：不得用 `evaluation.prediction.task_status/reasoning`、`prior_task_status`、`prior_reasoning` 影响判定。",
            "- 只允许改 `knowledge_base` / `prompts` / 不读先验的 framework 逻辑。",
            "- 禁止复用 iter01 式 `_has_substantive_prior_failure` 先验补丁。",
            "",
            "### 假设",
            hypothesis_block or "（见 iterations 下 notes.md）",
            "",
            "### 执行顺序（不要 narration loop，直接 tool call）",
            "1. 读 skill 核心约束、baseline、`badcases/`、`diagnosis/`、上一轮 `iterations/*/effect_report.md`。",
            f"2. 写 `iterations/{iter_id}/cluster_report.md` 与 `change_plan.md`（明确 knowledge vs framework，不得含 prior 条件）。",
            f"3. 改代码 + 写 `iterations/{iter_id}/target_task_ids.json`（来自 cluster 代表 task_id）。",
            f"4. snapshot：`snapshots/{iter_id}_before/` → 改代码 → `snapshots/{iter_id}_after/`。",
            "5. **不要自己跑 targeted/全量**；改完代码即结束回合。外层 `codex-iter.sh` 会调用 `scripts/run_iter_eval.sh` 强制跑评估并自动生成 `effect_report.md`。",
            "6. conda：`source \"$(conda info --base)/etc/profile.d/conda.sh\" && conda activate zj_base310`。",
            "7. 长任务启动优先用仓库内包装脚本；若无脚本，使用 `nohup setsid ... </dev/null &` 并写 `.done` 标记。",
            "",
            "### 完成标准（未达标不得结束本轮）",
            f"- 必须有 `cluster_report.md`、`change_plan.md`、`target_task_ids.json`、snapshot。",
            f"- 评估与 `effect_report.md` 由外层 `run_iter_eval.sh` 生成，不要求 agent 手写。",
            "- 不要停在「准备执行」；不要 narration loop；不要用 `rg`（环境无 rg，用内置 Grep/Read）。",
            "",
            "### 全局停止条件（外层自动循环判定）",
            "- 三分类准确率 > 90%",
            "- 正例 F1 > 85%",
            "- 负例 F1 > 90%",
            "",
            "现在开始执行本轮优化。",
        ]
    )
    if loop_index is not None and loop_total is not None:
        loop_banner = "\n".join(
            [
                "",
                "## 自动迭代模式",
                f"- 当前：第 {loop_index}/{loop_total} 轮自动 iter",
                f"- 本轮编号：`{iter_id}`",
                "- 外层 `validate-iter` 会检查交付物；无效轮次不算迭代成功。",
            ]
        )
        text = text.replace(f"## 本轮任务 ({iter_id})", loop_banner + f"\n\n## 本轮任务 ({iter_id})", 1)
    return text


def build_handoff(workdir: Path, extra: str = "") -> str:
    manifest = {}
    manifest_path = workdir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sections: List[str] = [
        "# Codex New Thread Handoff",
        "",
        "请接续以下工作。优先读 workspace 文件，少叙述，多调用工具执行。",
        "",
        "## 工作目标",
        _read_text(workdir / "README.md", 1500) or f"Project run workspace: `{workdir.name}`",
        "",
        "## 目标指标",
        json.dumps(manifest.get("target_metrics") or {}, ensure_ascii=False, indent=2),
        "",
        "## 关键目录与文件",
        f"- 项目根目录: `{PROJECT_ROOT}`",
        f"- 本次 workspace: `{workdir}`",
    ]
    for rel in KEY_DIR_CANDIDATES:
        p = PROJECT_ROOT / rel
        if p.exists():
            sections.append(f"- `{rel}`")
    artifacts = manifest.get("artifacts") or {}
    if artifacts:
        sections.append("")
        sections.append("### 数据产物")
        for k, v in artifacts.items():
            sections.append(f"- {k}: `{v}`")

    sections.extend(["", "## 当前进度"])
    timeline = _read_text(workdir / "metrics_timeline.md", 2000)
    if timeline:
        sections.append(timeline)
    else:
        sections.append("- 见 `metrics_timeline.md`（若不存在，先根据 baseline/eval 补写）")

    notes_path = _latest_iter_notes(workdir)
    if notes_path:
        sections.extend(["", f"## 最近迭代 ({notes_path.parent.name})", _read_text(notes_path, 2500)])

    manifest_iters = manifest.get("iterations") or []
    if manifest_iters:
        sections.extend(["", "## manifest 迭代记录"])
        for item in manifest_iters[-5:]:
            sections.append(
                f"- iter{item.get('iteration')}: {item.get('status')} — {item.get('summary', '')}"
            )

    full_cmd = manifest.get("full_eval_command")
    if full_cmd:
        sections.extend(["", "## 全量评估命令", "```bash", full_cmd, "```"])

    skill = (manifest.get("skill") or {}).get("path")
    if skill:
        sections.extend(["", "## Skill", f"- `{skill}`"])

    usage = read_context_usage()
    sections.extend(
        [
            "",
            "## Context 状态（上一 thread）",
            f"- last_tokens: {usage['tokens']} / {usage['context_window']}",
            f"- rollout: `{usage.get('rollout')}`",
            "",
            "## 执行要求",
            "- 不要 narration loop；每步直接 tool call。",
            "- **长任务禁止裸 `nohup ... &`**：必须用 `setsid`（或 `data/run_orm_flywheel.sh` 内置 daemon）。",
            "- 启动长任务后**本 turn 内持续 poll**（`--status` / `tail` / `Await`），直到 `.done` 出现；禁止「等几秒再查」后结束回合。",
            "- 产物写入 workspace，不要散落到 /tmp。",
        ]
    )
    if extra.strip():
        sections.extend(["", "## 用户补充", extra.strip()])
    sections.append("")
    return "\n".join(sections)


def cmd_status(args: argparse.Namespace) -> int:
    usage = read_context_usage()
    tokens = usage["tokens"]
    limit = usage["context_window"] or DEFAULT_LIMIT
    pct = (100.0 * tokens / limit) if limit else 0.0
    level = "ok"
    if tokens >= args.threshold:
        level = "warn"
    if tokens >= int(limit * 0.9):
        level = "critical"

    print(f"tokens={tokens} limit={limit} pct={pct:.1f}% level={level}")
    if usage.get("rollout"):
        print(f"rollout={usage['rollout']}")
    if usage.get("thread_id"):
        print(f"thread_id={usage['thread_id']}")
    if usage.get("updated_at"):
        print(f"updated_at={usage['updated_at']}")

    if level == "warn":
        print(
            f"\n⚠️  已超过 {args.threshold} tokens，建议新开 Codex thread。",
            file=sys.stderr,
        )
        print(
            "生成交接文档: python utils/codex_thread_helper.py handoff --workdir <workspace>",
            file=sys.stderr,
        )
        return 2
    if level == "critical":
        print("\n🛑 接近 context 上限，请立即 New Thread + handoff。", file=sys.stderr)
        return 3
    return 0


def cmd_handoff(args: argparse.Namespace) -> int:
    text = build_handoff(args.workdir, args.extra or "")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(str(args.output))
    else:
        print(text)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    print(f"watching context threshold={args.threshold}, interval={args.interval}s")
    while True:
        usage = read_context_usage()
        tokens = usage["tokens"]
        limit = usage["context_window"] or DEFAULT_LIMIT
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] tokens={tokens}/{limit}"
        if tokens >= args.threshold:
            line += "  ⚠️  NEW THREAD RECOMMENDED"
            print(line, file=sys.stderr)
            if args.workdir:
                out = args.workdir / "handoff_latest.md"
                out.write_text(build_handoff(args.workdir), encoding="utf-8")
                print(f"handoff written: {out}", file=sys.stderr)
        else:
            print(line)
        time.sleep(args.interval)


def parse_threshold(value: str) -> float:
    text = str(value).strip()
    if text.startswith(">"):
        text = text[1:].strip()
    if text.endswith("%"):
        return float(text[:-1].strip()) / 100.0
    num = float(text)
    return num / 100.0 if num > 1.0 else num


def metrics_from_eval_summary(summary: Dict[str, Any]) -> Dict[str, float]:
    labels = ["1", "0", "-1"]
    matrix = summary.get("matrix") or {}

    def f1_for(label: str) -> float:
        row = matrix.get(label, {}) or {}
        tp = float(row.get(label, 0) or 0)
        row_total = sum(float(row.get(p, 0) or 0) for p in labels)
        col_total = sum(float((matrix.get(h, {}) or {}).get(label, 0) or 0) for h in labels)
        precision = tp / col_total if col_total else 0.0
        recall = tp / row_total if row_total else 0.0
        denom = precision + recall
        return (2 * precision * recall / denom) if denom else 0.0

    return {
        "three_class_accuracy": float(summary.get("accuracy") or 0.0),
        "positive_f1": f1_for("1"),
        "negative_f1": f1_for("0"),
    }


def load_manifest_targets(workdir: Path) -> Dict[str, str]:
    manifest_path = workdir / "manifest.json"
    if not manifest_path.exists():
        return {
            "three_class_accuracy": ">90%",
            "positive_f1": ">85%",
            "negative_f1": ">90%",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("target_metrics") or {}


def _resolve_iter_xlsx(workdir: Path, iteration: int) -> Optional[Path]:
    """Resolve iter-specific eval xlsx only. No fallback to manifest default output."""
    iter_id = f"iter{iteration:02d}"
    candidates: List[Path] = [
        workdir / "eval" / iter_id / "output_回流数据评估情况_raw_data.xlsx",
        workdir / "eval" / f"full_{iter_id}" / "output_回流数据评估情况_raw_data.xlsx",
    ]
    artifacts = workdir / "eval" / f"{iter_id}_artifacts.json"
    if artifacts.exists():
        data = json.loads(artifacts.read_text(encoding="utf-8"))
        for key in ("archived_xlsx", "output_xlsx", "xlsx"):
            p = data.get(key)
            if p:
                candidates.append(Path(p))
    candidates.append(
        workdir / "eval" / f"{iter_id}_targeted" / "output_回流数据评估情况_raw_data.xlsx"
    )
    for c in candidates:
        p = c if c.is_absolute() else PROJECT_ROOT / c
        if p.exists():
            return p
    return None


def _log_looks_valid(log_path: Path, *, min_bytes: int = 200) -> tuple[bool, str]:
    """Return (ok, reason). Reject empty/tiny logs and interrupted runs."""
    if not log_path.exists():
        return False, "missing"
    try:
        data = log_path.read_bytes()
    except OSError as exc:
        return False, str(exc)
    if len(data) < min_bytes:
        return False, f"too small ({len(data)} bytes)"
    text = data.decode("utf-8", errors="replace").lower()
    if "keyboardinterrupt" in text or "exited 130" in text or "traceback (most recent call last)" in text:
        return False, "interrupted/error (KeyboardInterrupt / exit 130 / Traceback)"
    return True, ""


def _done_marker_ok(done_path: Path) -> tuple[bool, str]:
    """Check a `.done` marker written as `EXIT=<code>` by the nohup wrapper."""
    if not done_path.exists():
        return False, "missing (process killed mid-run or not finished)"
    text = done_path.read_text(encoding="utf-8", errors="replace").strip()
    if "exit=0" not in text.lower():
        return False, f"non-zero/invalid marker: {text!r}"
    return True, ""


def validate_iter_completion(workdir: Path, iteration: int) -> Dict[str, Any]:
    """Return whether an iteration has the required deliverables for a real optimization round."""
    iter_id = f"iter{iteration:02d}"
    errors: List[str] = []
    warnings: List[str] = []

    artifacts_json = workdir / "eval" / f"{iter_id}_artifacts.json"
    iter_xlsx_dir = workdir / "eval" / iter_id
    full_eval_dir = workdir / "eval" / f"full_{iter_id}"
    effect_report = workdir / "iterations" / iter_id / "effect_report.md"
    metrics_json = workdir / "eval" / f"{iter_id}_metrics.json"
    snapshot_after = workdir / "snapshots" / f"{iter_id}_after"
    timeline = workdir / "metrics_timeline.md"

    has_iter_xlsx = any(iter_xlsx_dir.glob("output_*.xlsx")) if iter_xlsx_dir.exists() else False
    full_logs = list(full_eval_dir.glob("*.log")) if full_eval_dir.exists() else []
    has_full_log = bool(full_logs)
    targeted_log = workdir / "iterations" / iter_id / "targeted_validation.log"
    full_done = full_eval_dir / "full_eval.done"
    targeted_done = workdir / "iterations" / iter_id / "targeted_validation.done"

    if not artifacts_json.exists() and not has_iter_xlsx:
        errors.append(f"missing eval/{iter_id}_artifacts.json or eval/{iter_id}/output_*.xlsx")
    if not has_full_log:
        errors.append(f"missing eval/full_{iter_id}/*.log (full --no-cache evaluation)")
    else:
        for lp in full_logs:
            ok, reason = _log_looks_valid(lp)
            if not ok:
                errors.append(f"eval/full_{iter_id}/{lp.name} invalid: {reason}")
    # full-eval done marker (nohup wrapper writes EXIT=<code>); proves the run finished cleanly
    ok, reason = _done_marker_ok(full_done)
    if not ok:
        errors.append(f"eval/full_{iter_id}/full_eval.done {reason}")
    if targeted_log.exists():
        ok, reason = _log_looks_valid(targeted_log, min_bytes=80)
        if not ok:
            errors.append(f"iterations/{iter_id}/targeted_validation.log invalid: {reason}")
    elif (workdir / "iterations" / iter_id / "target_task_ids.json").exists():
        errors.append(f"missing iterations/{iter_id}/targeted_validation.log (targeted run required)")
    if (workdir / "iterations" / iter_id / "target_task_ids.json").exists():
        ok, reason = _done_marker_ok(targeted_done)
        if not ok:
            errors.append(f"iterations/{iter_id}/targeted_validation.done {reason}")
    if not effect_report.exists():
        errors.append(f"missing iterations/{iter_id}/effect_report.md")
    else:
        body = effect_report.read_text(encoding="utf-8").strip()
        if len(body) < 80 or "pending" in body.lower():
            errors.append(f"iterations/{iter_id}/effect_report.md is empty or still pending")
        elif "缺标注或本轮无输出" in body:
            errors.append(
                f"iterations/{iter_id}/effect_report.md is stale (targeted table bug); "
                "re-run: python utils/write_effect_report.py --workdir ... --iteration N"
            )
    if not metrics_json.exists():
        errors.append(f"missing eval/{iter_id}_metrics.json")
    if not snapshot_after.exists():
        warnings.append(f"missing snapshots/{iter_id}_after/ (required if code/knowledge changed)")
    if timeline.exists():
        if iter_id not in timeline.read_text(encoding="utf-8"):
            errors.append(f"metrics_timeline.md has no row for {iter_id}")
    else:
        errors.append("missing metrics_timeline.md")

    resolved_xlsx = _resolve_iter_xlsx(workdir, iteration)
    if metrics_json.exists() and resolved_xlsx is None:
        errors.append(
            f"eval/{iter_id}_metrics.json exists but no iter-specific xlsx/artifacts found "
            "(may be stale fallback from global output)"
        )

    if iteration > 1 and metrics_json.exists():
        prev_metrics = workdir / "eval" / f"iter{iteration - 1:02d}_metrics.json"
        if prev_metrics.exists():
            cur = json.loads(metrics_json.read_text(encoding="utf-8"))
            prev = json.loads(prev_metrics.read_text(encoding="utf-8"))
            same_core = (
                cur.get("accuracy") == prev.get("accuracy")
                and cur.get("correct") == prev.get("correct")
                and cur.get("mismatch_counts") == prev.get("mismatch_counts")
            )
            if same_core and not effect_report.exists():
                warnings.append("metrics identical to previous iter without effect_report")
            elif (
                cur.get("accuracy") == prev.get("accuracy")
                and cur.get("correct") == prev.get("correct")
                and cur.get("mismatch_counts") != prev.get("mismatch_counts")
            ):
                warnings.append(
                    f"accuracy tied with iter{iteration - 1:02d} "
                    f"({float(cur.get('accuracy') or 0):.2%}) but mismatch_counts differ — not a stale copy"
                )

    return {
        "iteration": iteration,
        "iter_id": iter_id,
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "resolved_xlsx": str(resolved_xlsx) if resolved_xlsx else None,
    }


def ensure_iter_metrics(workdir: Path, iteration: int, *, force: bool = False) -> Path:
    iter_id = f"iter{iteration:02d}"
    out_json = workdir / "eval" / f"{iter_id}_metrics.json"
    xlsx = _resolve_iter_xlsx(workdir, iteration)
    if xlsx is None:
        raise FileNotFoundError(
            f"no iter-specific eval xlsx for {iter_id}; "
            f"archive outputs to eval/{iter_id}/ and write eval/{iter_id}_artifacts.json"
        )
    if out_json.exists() and not force:
        return out_json
    out_md = workdir / "eval" / f"{iter_id}_metrics.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    import subprocess

    cmd = (
        f'source "$(conda info --base)/etc/profile.d/conda.sh" && '
        f'conda activate zj_base310 && '
        f'python "{PROJECT_ROOT / "utils" / "iter_eval_summary.py"}" '
        f'"{xlsx}" --out-json "{out_json}" --out-md "{out_md}"'
    )
    subprocess.run(["bash", "-lc", cmd], check=True, cwd=str(PROJECT_ROOT))
    return out_json


def evaluate_targets(workdir: Path, iteration: Optional[int] = None) -> Dict[str, Any]:
    targets_raw = load_manifest_targets(workdir)
    thresholds = {k: parse_threshold(v) for k, v in targets_raw.items()}
    if iteration is None:
        iters = sorted((workdir / "eval").glob("iter*_metrics.json"))
        if not iters:
            raise FileNotFoundError("no eval/iterNN_metrics.json found")
        metrics_path = iters[-1]
    else:
        metrics_path = workdir / "eval" / f"iter{iteration:02d}_metrics.json"
        if not metrics_path.exists():
            ensure_iter_metrics(workdir, iteration)
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = metrics_from_eval_summary(summary)
    checks = {}
    all_met = True
    for key, threshold in thresholds.items():
        actual = metrics.get(key)
        if actual is None:
            checks[key] = {"actual": None, "threshold": threshold, "met": False}
            all_met = False
            continue
        met = actual > threshold
        checks[key] = {"actual": actual, "threshold": threshold, "met": met}
        if not met:
            all_met = False
    return {
        "iteration": iteration,
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "targets": thresholds,
        "checks": checks,
        "all_met": all_met,
    }


def cmd_check_targets(args: argparse.Namespace) -> int:
    try:
        result = evaluate_targets(args.workdir, args.iteration)
    except FileNotFoundError as exc:
        print(f"metrics_missing: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for key, chk in result["checks"].items():
        status = "OK" if chk["met"] else "FAIL"
        actual = chk["actual"]
        threshold = chk["threshold"]
        print(
            f"{status} {key}: {actual:.2%} (need > {threshold:.2%})"
            if actual is not None
            else f"FAIL {key}: missing"
        )
    if result["all_met"]:
        print("ALL_TARGETS_MET")
        return 0
    return 1


def cmd_ensure_metrics(args: argparse.Namespace) -> int:
    path = ensure_iter_metrics(
        args.workdir,
        args.iteration,
        force=bool(getattr(args, "force", False)),
    )
    print(str(path))
    return 0


def cmd_iter_prompt(args: argparse.Namespace) -> int:
    text = build_iter_start_prompt(
        args.workdir,
        args.iteration,
        args.hypothesis or "",
        args.extra or "",
        loop_index=getattr(args, "loop_index", None),
        loop_total=getattr(args, "loop_total", None),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(str(args.output))
    else:
        print(text)
    return 0


def cmd_validate_iter(args: argparse.Namespace) -> int:
    result = validate_iter_completion(args.workdir, args.iteration)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for w in result.get("warnings") or []:
        print(f"WARN {w}", file=sys.stderr)
    for e in result.get("errors") or []:
        print(f"FAIL {e}", file=sys.stderr)
    return 0 if result["valid"] else 1


def cmd_next_iter(args: argparse.Namespace) -> int:
    print(next_iteration(args.workdir))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Codex context monitor and thread handoff helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status", help="Check latest Codex session token usage")
    p_status.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    p_status.set_defaults(func=cmd_status)

    p_handoff = sub.add_parser("handoff", help="Generate new-thread handoff brief")
    p_handoff.add_argument("--workdir", type=Path, required=True)
    p_handoff.add_argument("--output", type=Path, default=None)
    p_handoff.add_argument("--extra", default="")
    p_handoff.set_defaults(func=cmd_handoff)

    p_watch = sub.add_parser("watch", help="Poll token usage and warn at threshold")
    p_watch.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    p_watch.add_argument("--interval", type=int, default=60)
    p_watch.add_argument("--workdir", type=Path, default=None)
    p_watch.set_defaults(func=cmd_watch)

    p_iter = sub.add_parser("iter-prompt", help="Build iter-start prompt with handoff + task plan")
    p_iter.add_argument("--workdir", type=Path, required=True)
    p_iter.add_argument("--iteration", type=int, required=True)
    p_iter.add_argument("--hypothesis", default="")
    p_iter.add_argument("--extra", default="")
    p_iter.add_argument("--output", type=Path, default=None)
    p_iter.add_argument("--loop-index", type=int, default=None)
    p_iter.add_argument("--loop-total", type=int, default=None)
    p_iter.set_defaults(func=cmd_iter_prompt)

    p_check = sub.add_parser("check-targets", help="Check iter metrics against manifest targets")
    p_check.add_argument("--workdir", type=Path, required=True)
    p_check.add_argument("--iteration", type=int, default=None)
    p_check.set_defaults(func=cmd_check_targets)

    p_ensure = sub.add_parser("ensure-metrics", help="Create eval/iterNN_metrics.json from iter-specific xlsx")
    p_ensure.add_argument("--workdir", type=Path, required=True)
    p_ensure.add_argument("--iteration", type=int, required=True)
    p_ensure.add_argument("--force", action="store_true")
    p_ensure.set_defaults(func=cmd_ensure_metrics)

    p_validate = sub.add_parser("validate-iter", help="Check iter deliverables before accepting a round")
    p_validate.add_argument("--workdir", type=Path, required=True)
    p_validate.add_argument("--iteration", type=int, required=True)
    p_validate.set_defaults(func=cmd_validate_iter)

    p_next = sub.add_parser("next-iter", help="Print next iteration number for a workspace")
    p_next.add_argument("--workdir", type=Path, required=True)
    p_next.set_defaults(func=cmd_next_iter)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
