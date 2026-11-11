#!/usr/bin/env python3
"""
将 2026031610572649 格式的轨迹目录转换为 sessions_examples 格式。

源结构（中间目录名可变，层级固定）:
  {input_root}/
    {session_id}/
      {任意名}/
        {任意名}/
          meta.json
          1/instruction.json
          2/instruction.json
          ...

目标结构:
  {output_root}/
    {batch_id}_{session_id}_main/
      step_0000.json
      step_0000.png
      step_0001.json
      step_0001.png
      ...

用法:
  python convert_to_sessions.py --input 2026031610572649 --output sessions_examples
  python convert_to_sessions.py -i ./2026031610572649 -o ./sessions_converted
  python convert_to_sessions.py -i ./2026031610572649 --zip   # 转换后压缩为 zip
"""
import argparse
import base64
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

# 最小合法 PNG（1x1 透明像素），用于无截图时的占位
_MINIMAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _extract_goal_from_meta(meta: Dict) -> str:
    """从 meta.json 的 goal 中提取主任务描述"""
    goal = meta.get("goal", "") or meta.get("ext_info", {}).get("goal", "")
    if isinstance(goal, str):
        m = re.search(r"<main>(.+?)</main>", goal, re.DOTALL)
        return m.group(1).strip() if m else goal.strip()
    return str(goal)


def _map_action(src: Dict) -> Dict[str, Any]:
    """将源 instruction.json 的 action 映射为 sessions_examples 的 action"""
    atype = (src.get("action") or {}).get("type") or ""
    detail = (src.get("action") or {}).get("detail") or {}

    if atype == "open_app":
        return {
            "action_type": "open_app",
            "app_name": detail.get("app_name", ""),
        }
    if atype == "type":
        pt = detail.get("point", [0, 0])
        x, y = int(pt[0]) if len(pt) > 0 else 0, int(pt[1]) if len(pt) > 1 else 0
        out: Dict[str, Any] = {"action_type": "type", "text": detail.get("text", "")}
        if x or y:
            out["x"], out["y"] = x, y
        return out
    if atype == "touch":
        pt = detail.get("point", [0, 0])
        x = int(pt[0]) if len(pt) > 0 else 0
        y = int(pt[1]) if len(pt) > 1 else 0
        return {"action_type": "click", "x": x, "y": y}
    if atype == "long_press":
        pt = detail.get("point", [0, 0])
        x = int(pt[0]) if len(pt) > 0 else 0
        y = int(pt[1]) if len(pt) > 1 else 0
        return {"action_type": "click", "x": x, "y": y}
    if atype == "double_tap":
        pt = detail.get("point", [0, 0])
        x = int(pt[0]) if len(pt) > 0 else 0
        y = int(pt[1]) if len(pt) > 1 else 0
        return {"action_type": "click", "x": x, "y": y}
    if atype == "drag":
        start_pt = detail.get("start", [0, 0])
        end_pt = detail.get("end", [0, 0])
        x = int(start_pt[0]) if len(start_pt) > 0 else 0
        y = int(start_pt[1]) if len(start_pt) > 1 else 0
        x2 = int(end_pt[0]) if len(end_pt) > 0 else 0
        y2 = int(end_pt[1]) if len(end_pt) > 1 else 0
        return {"action_type": "drag", "start_x": x, "start_y": y, "end_x": x2, "end_y": y2}
    if atype == "wait":
        return {"action_type": "wait", "duration": detail.get("duration", "2s")}
    if atype == "terminate":
        return {"action_type": "finished", "summary": detail.get("summary", ""), "status": detail.get("status", "success"), "answer": detail.get("answer", "")}
    if atype == "finished":
        return {"action_type": "finished", "summary": detail.get("summary", ""), "status": detail.get("status", "success"), "answer": detail.get("answer", "")}
    if atype == "takeover":
        return {"action_type": "takeover"}
    if atype == "navigate_home":
        return {"action_type": "navigate_home"}
    if atype == "navigate_back":
        return {"action_type": "navigate_home"}
    # 未知类型保留基本信息
    return {"action_type": atype or "unknown", "detail": detail}


def _is_trajectory_dir(d: Path) -> bool:
    """判断目录是否为轨迹根：含 meta.json 且至少有一个编号子目录含 instruction.json"""
    if not (d / "meta.json").exists():
        return False
    for child in d.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            if (child / "instruction.json").exists():
                int(child.name)
                return True
        except ValueError:
            continue
    return False


def _find_trajectory_dir(session_root: Path) -> Optional[Path]:
    """在 session 下递归查找轨迹根目录（含 meta.json 和编号 step 子目录，目录名不固定）"""
    # 性能优化：不要遍历所有目录做判定，而是先只找 meta.json。
    # 根据源结构说明，meta.json 通常位于 session_root 下两层/三层目录之下：
    #   {session_id}/{任意名}/{任意名}/meta.json
    # 这里给出少量深度兜底，避免漏掉数据。
    meta_patterns = [
        "*/*/meta.json",
        "*/*/*/meta.json",
        "*/*/*/*/meta.json",
        "*/*/*/*/*/meta.json",
    ]

    candidates: List[Path] = []
    for pat in meta_patterns:
        for meta_path in session_root.glob(pat):
            traj_dir = meta_path.parent
            # 跳过包含隐藏目录的路径
            if any(p.startswith(".") for p in traj_dir.relative_to(session_root).parts):
                continue
            if _is_trajectory_dir(traj_dir):
                candidates.append(traj_dir)

    if not candidates:
        return None
    return max(candidates, key=lambda p: len(p.parts))


def _collect_steps(phone_dir: Path) -> List[Tuple[int, Path]]:
    """收集步数目录，返回 [(step_num, instruction.json 所在目录), ...]"""
    steps: List[Tuple[int, Path]] = []
    for child in phone_dir.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            n = int(child.name)
            inst = child / "instruction.json"
            if inst.exists():
                steps.append((n, child))
        except ValueError:
            continue
    return sorted(steps, key=lambda x: x[0])


def convert_trajectory(
    phone_dir: Path,
    meta_path: Path,
    out_dir: Path,
    batch_id: str,
    session_id: str,
    instruction_override: Optional[str] = None,
) -> int:
    """
    转换单条轨迹，写入 step_*.json 和 step_*.png。
    返回写入的 step 数量。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    goal = instruction_override or _extract_goal_from_meta(meta)

    steps = _collect_steps(phone_dir)
    if not steps:
        return 0

    def _write_step_json_and_png(
        *,
        out_step_index: int,
        step_dir: Path,
        step_data: Dict[str, Any],
    ) -> None:
        """写入 step json + png（优先复制 origin_pre_pic，否则写最小占位图）。"""
        step_name = f"step_{out_step_index:04d}"
        json_path = out_dir / f"{step_name}.json"
        png_path = out_dir / f"{step_name}.png"

        # 更新文件名相关字段，保证 step_index 与文件一一对应
        step_data["step_index"] = out_step_index
        step_data["image_file"] = f"{step_name}.png"
        step_data["image_path"] = ""

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(step_data, f, ensure_ascii=False, indent=2)

        found_png = False
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = step_dir / f"origin_pre_pic{ext}"
            if candidate.exists() and candidate.is_file():
                shutil.copy2(candidate, png_path)
                found_png = True
                break
        if not found_png:
            png_path.write_bytes(_MINIMAL_PNG)

    prev_original_action_type: Optional[str] = None
    out_step_idx = 0

    for _, (step_num, step_dir) in enumerate(steps):
        inst_path = step_dir / "instruction.json"
        with open(inst_path, "r", encoding="utf-8") as f:
            src = json.load(f)

        action = _map_action(src)
        step_desc = src.get("step_desc", "")
        thought = src.get("thought", "")
        prediction = src.get("prediction", "")
        current_action_type = (src.get("action") or {}).get("type") or ""
        takeover_input = src.get("takeover_input", "") or ""
        take_over_summarize = src.get("take_over_summarize", "") or ""
        take_over_category = src.get("take_over_category", "") or ""

        # 如果原始数据当前步 takeover_input 非空，且“上一步动作”不是 takeover，
        # 则把当前步拆成两步：先 takeover，再执行当前步动作（保留当前步 takeover_input）。
        #
        # 注意：这里判断“上一步是否为 takeover”基于原始数据的上一条 instruction.json 的 action.type，
        # 不是基于我们插入后的人工步骤。
        if (
            prev_original_action_type is not None
            and str(takeover_input).strip() != ""
            and prev_original_action_type != "takeover"
        ):
            takeover_step_data: Dict[str, Any] = {
                "instruction": goal,
                "takeover_input": "",
                "prediction": "",
                "action": {"action_type": "takeover", "summarize": take_over_summarize, "tag": take_over_category},
                "thought": "",
                "reasoning_content": "",
                "timestamp": "",
                "image_file": "",
                "image_path": "",
            }
            _write_step_json_and_png(
                out_step_index=out_step_idx,
                step_dir=step_dir,
                step_data=takeover_step_data,
            )
            out_step_idx += 1

        step_data = {
            "instruction": goal,
            "takeover_input": takeover_input,
            "prediction": prediction,
            "action": action,
            "thought": thought,
            "reasoning_content": "",
            "timestamp": "",
            "image_file": "",
            "image_path": "",
        }
        _write_step_json_and_png(
            out_step_index=out_step_idx,
            step_dir=step_dir,
            step_data=step_data,
        )
        out_step_idx += 1

        prev_original_action_type = current_action_type

    return out_step_idx


def convert_all(
    input_root: Path,
    output_root: Path,
    batch_id: Optional[str] = None,
) -> int:
    """
    转换 input_root 下所有轨迹到 output_root。
    batch_id 默认取 input_root 目录名。
    返回转换的轨迹数量。
    """
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    batch_id = batch_id or input_root.name

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_root}")

    count = 0
    sessions_to_convert: List[Tuple[Path, Path, Path, str, str]] = []
    for session_dir in sorted(input_root.iterdir()):
        if not session_dir.is_dir() or session_dir.name.startswith("."):
            continue
        session_id = session_dir.name
        traj_dir = _find_trajectory_dir(session_dir)
        if not traj_dir:
            print(f"  跳过 {session_id}: 未找到含 meta.json 及编号 step 子目录的轨迹目录")
            continue
        meta_path = traj_dir / "meta.json"
        if not meta_path.exists():
            print(f"  跳过 {session_id}: 缺少 meta.json")
            continue

        out_name = f"{batch_id}_{session_id}_main"
        out_dir = output_root / out_name
        sessions_to_convert.append((traj_dir, meta_path, out_dir, batch_id, session_id))

    total = len(sessions_to_convert)
    if total == 0:
        return 0

    start_ts = time.time()
    completed = 0

    def _format_eta(sec: float) -> str:
        sec = max(0.0, sec)
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}m{s:02d}s"

    bar_len = 30
    last_out_name = ""
    last_steps = 0

    for traj_dir, meta_path, out_dir, batch_id2, session_id in sessions_to_convert:
        n = convert_trajectory(traj_dir, meta_path, out_dir, batch_id2, session_id)
        completed += 1
        if n > 0:
            count += 1
            last_out_name = f"{batch_id2}_{session_id}_main"
            last_steps = n
        else:
            last_out_name = f"{batch_id2}_{session_id}_main"
            last_steps = 0

        ratio = completed / total
        filled = int(bar_len * ratio)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - start_ts
        eta = elapsed / completed * (total - completed) if completed > 0 else 0.0
        msg = f"\r[{bar}] {completed}/{total} ({ratio * 100:.1f}%) ETA { _format_eta(eta) }"
        if last_steps > 0:
            msg += f"  {last_out_name} ✓ {last_steps} steps"
        print(msg, end="", flush=True)

    print()  # 换行
    return count


def make_zip_archive(output_root: Path) -> Path:
    """
    将 output_root 目录压缩为 zip 文件。
    输出: {output_root}.zip，解压后得到 {output_root.name}/ 及其内容。
    返回生成的 zip 文件路径。
    """
    output_root = output_root.resolve()
    if not output_root.exists() or not output_root.is_dir():
        raise FileNotFoundError(f"输出目录不存在: {output_root}")
    zip_path = Path(shutil.make_archive(
        base_name=str(output_root),
        format="zip",
        root_dir=output_root.parent,
        base_dir=output_root.name,
    ))
    return zip_path


# 固定输出目录：脚本所在目录下的 sessions_output 子目录
# 可通过 --output 覆盖
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "sessions_output"


def main():
    parser = argparse.ArgumentParser(
        description="将 2026031610572649 格式轨迹转换为 sessions_examples 格式"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="源轨迹根目录（如 2026031610572649）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="输出根目录（默认: 脚本同目录下的 sessions_output）",
    )
    parser.add_argument(
        "--batch-id",
        help="批次 ID，默认使用输入目录名",
    )
    parser.add_argument(
        "--zip",
        "-z",
        action="store_true",
        help="转换完成后将输出目录压缩为 zip 文件",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR

    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print()

    n = convert_all(input_path, output_path, batch_id=args.batch_id or None)
    print(f"\n转换完成，共 {n} 条轨迹。")

    if args.zip and n > 0:
        zip_path = make_zip_archive(output_path)
        print(f"已压缩: {zip_path}")


if __name__ == "__main__":
    main()
