#!/usr/bin/env python3
"""Workspace helper for ORM judgment optimization runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_TARGETS = {
    "three_class_accuracy": ">90%",
    "positive_f1": ">85%",
    "negative_f1": ">90%",
}

DEFAULT_FULL_EVAL_COMMAND = (
    "conda activate zj_base310 && python main.py "
    "--input data/tmp/回流数据评估情况.xlsx "
    "--pipeline-json pipeline/pipeline_error_cause_analysis.json "
    "--model seed_free --batch-size 100 --knowledge knowledge_base "
    "--max-tasks 10000 --filter failed,fair_enough,env_error,perfect"
)

KEY_FILES = [
    "main.py",
    "prompts/lite_trajectory_unified_prompt.py",
    "prompts/lite_judgment_prompt.py",
    "utils/pipeline_runner.py",
    "utils/lite_prompt_knowledge.py",
    "pipeline/pipeline_error_cause_analysis.json",
    "knowledge_base/completed_strategies/common.json",
    "knowledge_base/corrected_strategies/common.json",
    "knowledge_base/env_error_strategies/common.json",
]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _run_id(name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name.strip())
    return f"{stamp}_{clean or 'orm_judgment'}"


def _git_state() -> Dict[str, Any]:
    def run(cmd: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "git_root": run(["git", "rev-parse", "--show-toplevel"]),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_status_short": run(["git", "status", "--short"]),
    }


def _mkdirs(workdir: Path) -> None:
    for rel in (
        "baseline",
        "badcases",
        "diagnosis",
        "iterations",
        "eval",
        "snapshots",
    ):
        (workdir / rel).mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def init(args: argparse.Namespace) -> None:
    run_id = args.run_id or _run_id(args.name)
    workdir = args.root / run_id
    _mkdirs(workdir)

    manifest = {
        "run_id": run_id,
        "created_at": _now(),
        "workspace": str(workdir),
        "target_metrics": DEFAULT_TARGETS,
        "artifacts": {
            "input_xlsx": str(args.input_xlsx),
            "input_jsonl": str(args.input_jsonl),
            "output_xlsx": str(args.output_xlsx),
            "output_jsonl": str(args.output_jsonl),
        },
        "full_eval_command": args.full_eval_command or DEFAULT_FULL_EVAL_COMMAND,
        "skill": {
            "path": ".cursor/skills/orm-judgment-optimizer/SKILL.md",
            "version_note": "Chinese workflow with workspace, seed_free diagnosis, and ORM-only optimization constraints.",
        },
        "git": _git_state(),
        "iterations": [],
    }
    (workdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (workdir / "README.md").write_text(
        f"# ORM Judgment Optimization Run\n\nrun_id: `{run_id}`\ncreated_at: `{manifest['created_at']}`\n\n",
        encoding="utf-8",
    )
    print(str(workdir))


def snapshot(args: argparse.Namespace) -> None:
    workdir = args.workdir
    label = args.label
    snap_dir = workdir / "snapshots" / label
    snap_dir.mkdir(parents=True, exist_ok=True)
    files = args.files or KEY_FILES
    for file_name in files:
        src = Path(file_name)
        dst = snap_dir / file_name
        _copy_if_exists(src, dst)
    print(str(snap_dir))


def start_iter(args: argparse.Namespace) -> None:
    iter_id = f"iter{args.iteration:02d}"
    iter_dir = args.workdir / "iterations" / iter_id
    iter_dir.mkdir(parents=True, exist_ok=True)
    notes = iter_dir / "notes.md"
    if not notes.exists():
        notes.write_text(
            "\n".join(
                [
                    f"# {iter_id}",
                    "",
                    "## Hypothesis",
                    args.hypothesis or "",
                    "",
                    "## Badcase Clusters",
                    "",
                    "## Planned Changes",
                    "",
                    "## Targeted Validation",
                    "",
                    "## Full Evaluation",
                    "",
                    "## Decision",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    print(str(iter_dir))


def update_manifest(args: argparse.Namespace) -> None:
    manifest_path = args.workdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event = {
        "at": _now(),
        "iteration": args.iteration,
        "status": args.status,
        "summary": args.summary,
    }
    manifest.setdefault("iterations", []).append(event)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(manifest_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--root", type=Path, default=Path("logs/orm_judgment_optimization"))
    p_init.add_argument("--name", default="orm_judgment")
    p_init.add_argument("--run-id", default="")
    p_init.add_argument("--input-xlsx", type=Path, required=True)
    p_init.add_argument("--input-jsonl", type=Path, required=True)
    p_init.add_argument("--output-xlsx", type=Path, required=True)
    p_init.add_argument("--output-jsonl", type=Path, required=True)
    p_init.add_argument("--full-eval-command", default="")
    p_init.set_defaults(func=init)

    p_snap = sub.add_parser("snapshot")
    p_snap.add_argument("--workdir", type=Path, required=True)
    p_snap.add_argument("--label", required=True)
    p_snap.add_argument("--files", nargs="*")
    p_snap.set_defaults(func=snapshot)

    p_iter = sub.add_parser("start-iter")
    p_iter.add_argument("--workdir", type=Path, required=True)
    p_iter.add_argument("--iteration", type=int, required=True)
    p_iter.add_argument("--hypothesis", default="")
    p_iter.set_defaults(func=start_iter)

    p_update = sub.add_parser("update-manifest")
    p_update.add_argument("--workdir", type=Path, required=True)
    p_update.add_argument("--iteration", type=int, required=True)
    p_update.add_argument("--status", required=True)
    p_update.add_argument("--summary", default="")
    p_update.set_defaults(func=update_manifest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
