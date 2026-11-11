#!/usr/bin/env python3
"""
采集平台原始 dump  ->  本地 Agent sessions 格式

把「荣耀采集平台跑完后落盘的原始目录」转换成「ah_agent_server 在线推理时
SessionManager.save_step_to_disk 落盘的 sessions 格式」。

────────────────────────────────────────────────────────────────────────
输入（平台原始 dump，嵌套结构）：

    <rollout_id>/                       # 例如 2026052514380246
      <task_id>/                        # 例如 6604438743650331
        <model_name>/                   # 例如 ali_10090_takeover_server
          <DEVICE>/                     # 例如 PHONE
            meta.json                   # 任务级元信息（goal / max_step / level ...）
            1/                          # 每个数字目录 = 一个 agent 推理 step（从 1 开始）
              instruction.json          #   prediction / action / thought / metaAction / extInfo.step_count
              origin_pre_pic.png        #   ★ 动作前的原始截图 = 模型看到的 state（落盘用这张）
              pre_pic.png               #   另一张（带标注/缩放，不用）
              mid_pic.png  post_pic.png reward_score.json
            2/ ...
            5/
              instruction.json          #   带 take_over_* 字段 = 这一步发生过“接管”
              takeover/
                takeover_result.json    #   takeover_input / continue_status / describe
            ...

输出（本地 sessions 格式）：

    <output_root>/<session_id>/
      step_0000.json  step_0000.png
      step_0001.json  step_0001.png
      ...

────────────────────────────────────────────────────────────────────────
拆步规则（核心，已用真实 ground-truth sessions 逐字段比对验证）：

  • 普通 step 目录（无 takeover/）：产出 1 个 sessions step。
        prediction     = instruction.json["prediction"]
        action         = 由 meta_output.action 解析（见 _meta_action）
        thought        = instruction.json["thought"]
        takeover_input = ""
        图              = origin_pre_pic.png

  • 接管 step 目录（含 takeover/ 或带 take_over_summarize 字段）：拆成 2 个 sessions step，
    正好填补 step_count 序列里被“接管步”占用的那个缺号：

      ① 接管步（在前）
            prediction     = "<think>\n</think>\n<action>\n{summarize}\n</action>\n"
                             "<tool_call>\n{json}\n</tool_call>"
                             json = {"name":"mobile_use","arguments":
                                       {"action_type":"take_over","tag":category,"text":summarize}}
            action         = {"action_type":"takeover","summarize":summarize,"category":category}
            thought        = instruction.json["thought"]   # 沿用（在线 patch 只改 prediction/action，不改 thought）
            takeover_input = ""
            图              = 本目录 origin_pre_pic.png
              summarize = instruction.json["take_over_summarize"]
              category  = instruction.json["take_over_category"]

      ② 续作步（在后）
            prediction     = instruction.json["prediction"]
            action         = _meta_action(instruction)
            thought        = instruction.json["thought"]
            takeover_input = instruction.json["takeover_input"]   # 接管时用户补充的输入
            图              = 本目录 origin_pre_pic.png（与接管步同一张）

  • step_index 在每个 session 内从 0 连续递增；按数字目录顺序遍历，遇接管目录先写接管步、
    再写续作步，自然得到与在线一致的连续索引。

字段 schema 对齐 app/session_manager.py::SessionManager.save_step_to_disk。
────────────────────────────────────────────────────────────────────────

用法：
    # 转换一个 rollout 下的全部 task（input 指向 <rollout_id> 目录）
    python convert_platform_to_sessions.py \
        --input  /opt/nas/p/zj/tmp/2026052514380246 \
        --output /opt/nas/p/zj/tmp/sessions_out

    # input 也可以指向更上层目录，脚本会自动递归找出所有 task（含 meta.json 的设备目录）
"""

import argparse
import json
import os
import re
import shutil
from collections import OrderedDict
from datetime import datetime

# 模型看到的 state 截图优先级（经 md5 与真实 sessions 落盘图逐一校验）：
#   - 首步/主屏：home_screen_pic.png  （仅 home 步存在，== 在线 step_0000.png）
#   - 其余步：origin_pre_pic.png      （== 在线该步落盘图；pre_pic.png 是另一张带处理的图，不用）
STATE_IMAGE_CANDIDATES = ["home_screen_pic.png", "origin_pre_pic.png", "pre_pic.png", "mid_pic.png"]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_task_dirs(root):
    """递归找出所有“设备级任务目录”：含 meta.json，且有形如 1/instruction.json 的数字步目录。"""
    task_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "meta.json" not in filenames:
            continue
        has_step = any(
            d.isdigit() and os.path.isfile(os.path.join(dirpath, d, "instruction.json"))
            for d in dirnames
        )
        if has_step:
            task_dirs.append(dirpath)
    return sorted(task_dirs)


def _state_image(step_dir):
    for name in STATE_IMAGE_CANDIDATES:
        p = os.path.join(step_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _platform_action_raw(instruction):
    """取平台真实动作源：meta_output.action；与 gui_action 类型不一致时用 gui_action。"""
    mo = instruction.get("meta_output") or {}
    raw = mo.get("action") or instruction.get("action")
    gui = mo.get("gui_action")
    if raw and gui and gui.get("type") != raw.get("type"):
        return gui
    return raw or gui


def _meta_action(instruction):
    """meta_output.action -> sessions action（不再信任 metaAction 字符串）。"""
    raw = _platform_action_raw(instruction)
    if not raw or not isinstance(raw, dict):
        return {}
    t = raw.get("type")
    d = raw.get("detail") or {}

    if t == "touch":
        pt = d.get("point") or [None, None]
        return {"action_type": "click", "x": pt[0], "y": pt[1]}
    if t == "drag":
        s, e = d.get("start") or [None, None], d.get("end") or [None, None]
        return {"action_type": "swipe", "start_x": s[0], "start_y": s[1], "end_x": e[0], "end_y": e[1]}
    if t == "back":
        return {"action_type": "navigate_back"}
    if t == "open_app":
        return {"action_type": "open_app", "app_name": d.get("app_name")}
    if t == "type":
        return {"action_type": "type", "text": d.get("text")}
    if t == "wait":
        return {"action_type": "wait", "duration": d.get("duration")}
    if t == "long_touch":
        pt = d.get("point") or [None, None]
        return {"action_type": "long_press", "x": pt[0], "y": pt[1]}
    if t == "terminate":
        return {
            "action_type": "finished",
            "text": d.get("answer"),
            "summary": d.get("summary"),
            "status": d.get("status"),
        }
    if t == "takeover":
        return {
            "action_type": "takeover",
            "summarize": d.get("summarize") or d.get("summary") or instruction.get("take_over_summarize"),
            "category": d.get("category") or instruction.get("take_over_category"),
        }
    if t == "navigate_home":
        return {"action_type": "navigate_home"}
    if t == "enter":
        return {"action_type": "keyboard_enter"}
    if t == "double_touch":
        pt = d.get("point") or [None, None]
        return {"action_type": "double_tap", "x": pt[0], "y": pt[1]}
    if t == "error_env":
        return {"action_type": "error_env"}
    return {"action_type": t}


def _is_takeover_step(step_dir, instruction):
    if os.path.isdir(os.path.join(step_dir, "takeover")):
        return True
    return bool(instruction.get("take_over_summarize") or instruction.get("take_over_category"))


def _build_takeover_prediction(summarize, category):
    tool_call = {
        "name": "mobile_use",
        "arguments": {"action_type": "take_over", "tag": category, "text": summarize},
    }
    tool_call_str = json.dumps(tool_call, ensure_ascii=False)
    return (
        "<think>\n</think>\n"
        f"<action>\n{summarize}\n</action>\n"
        f"<tool_call>\n{tool_call_str}\n</tool_call>"
    )


def _step_count(instruction):
    """读取平台 instruction 的 extInfo.step_count，失败返回 None。"""
    ext = instruction.get("extInfo") or instruction.get("ext_info") or {}
    v = ext.get("step_count")
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return None


def _session_id(instruction):
    """读取该 step 归属的 session_id（区分 _pre_1 / _main 等阶段）。"""
    ext = instruction.get("extInfo") or instruction.get("ext_info") or {}
    return ext.get("session_id")


def _instruction_for_stage(goal, stage):
    """从 `<pre>..</pre><main>..</main>` 形式的 goal 中，按阶段取出对应的任务描述。

    - 无标签：原样返回 goal（绝大多数 main-only 任务）。
    - stage 以 pre 开头：取 <pre>..</pre>。
    - 其余（main）：取 <main>..</main>。
    - 兜底：去掉标签返回剩余文本。
    """
    if not goal:
        return ""
    if "<pre>" not in goal and "<main>" not in goal:
        return goal
    if stage and str(stage).startswith("pre"):
        m = re.search(r"<pre>(.*?)</pre>", goal, re.S)
        if m:
            return m.group(1)
    m = re.search(r"<main>(.*?)</main>", goal, re.S)
    if m:
        return m.group(1)
    return re.sub(r"</?(?:pre|main)>", "", goal)


def _write_step(out_session_dir, idx, *, instruction_goal, prediction, action,
                thought, takeover_input, src_image, session_id):
    img_name = f"step_{idx:04d}.png"
    json_name = f"step_{idx:04d}.json"
    img_dst = os.path.join(out_session_dir, img_name)
    if src_image and os.path.isfile(src_image):
        shutil.copyfile(src_image, img_dst)
    else:
        img_dst = ""  # 缺图时不写 image_path

    payload = {
        "step_index": idx,
        "instruction": instruction_goal,
        "takeover_input": takeover_input or "",
        "prediction": prediction or "",
        "action": action or {},
        "thought": thought or "",
        "reasoning_content": "",
        "timestamp": datetime.now().isoformat(),
        "image_file": img_name if src_image else "",
        "image_path": os.path.abspath(img_dst) if img_dst else "",
    }
    with open(os.path.join(out_session_dir, json_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _emit_session(session_id, items, goal, output_root, verbose):
    """把同一个 session_id 下的 step 目录序列落成一个 sessions 目录。

    items: [(name, step_dir, instruction), ...]，已按数字目录顺序排好。
    每个 session 内 step_index / step_count gap 追踪都独立从头计数。
    """
    stage = items[0][2].get("task_stage")
    instruction_text = _instruction_for_stage(goal, stage)

    out_session_dir = os.path.join(output_root, session_id)
    os.makedirs(out_session_dir, exist_ok=True)

    idx = 0
    prev_step_count = None
    for name, step_dir, instruction in items:
        src_image = _state_image(step_dir)
        thought = instruction.get("thought", "")
        main_prediction = instruction.get("prediction", "")
        main_action = _meta_action(instruction)
        curr_step_count = _step_count(instruction)

        is_take = _is_takeover_step(step_dir, instruction)
        meta_type = main_action.get("action_type") if isinstance(main_action, dict) else None

        # step_count 相对上一目录的缺号数（gap）：
        #   gap>=1 → 在线轨迹里这一步前面有独立的“接管决策步”被占号，需要补：
        #            「合成接管步」+「续作步（本目录真实动作）」共 2 步。
        #   gap==0 → 本目录自身就是接管步，不存在被占号的前置决策步。
        if isinstance(prev_step_count, int) and isinstance(curr_step_count, int):
            gap = curr_step_count - prev_step_count - 1
        elif prev_step_count is None and isinstance(curr_step_count, int):
            gap = curr_step_count - 1  # 首个目录前的缺号
        else:
            gap = 0

        continue_goal = instruction.get("take_over_continue_goal")

        should_split_takeover = is_take and gap >= 1
        # gap==0、真实动作不是 takeover，且 take_over_continue_goal == "外部模型接管"：
        # 这是“外部模型直接接管”，在线侧把该动作替换成了一个接管步（真实动作被丢弃），
        # 只落 1 个合成接管步，而不是拆成两步。
        #   反例（应保留真实动作，不替换）：
        #     - continue_goal == "finish"  → 终态标记步（如 click/takeover），保留原动作
        #     - continue_goal == "Auto任务继续" 且 gap==0 → 接管后的续作，保留原动作
        #     - continue_goal 为空（None）→ 普通终止步（如 finished），保留原动作
        should_replace_takeover = (
            is_take
            and gap == 0
            and meta_type not in ("takeover", "take_over")
            and continue_goal == "外部模型接管"
        )

        if should_split_takeover or should_replace_takeover:
            summarize = instruction.get("take_over_summarize", "")
            category = instruction.get("take_over_category", "")
            takeover_input = instruction.get("takeover_input")
            if not takeover_input:
                tr = os.path.join(step_dir, "takeover", "takeover_result.json")
                if os.path.isfile(tr):
                    takeover_input = _read_json(tr).get("takeover_input", "")

            # 合成接管步
            # 注意：接管步对应在线 step_count 序列里被“接管”占用的缺号，它的原始 thought
            # （接管前 agent 的推理）平台 dump 并未导出（dir 里只保留了接管后的续作 instruction），
            # 故无法恢复。这里置空，与合成 prediction 的 <think></think> 保持一致。
            _write_step(
                out_session_dir, idx,
                instruction_goal=instruction_text,
                prediction=_build_takeover_prediction(summarize, category),
                action={"action_type": "takeover", "summarize": summarize, "category": category},
                thought="",
                takeover_input="",
                src_image=src_image,
                session_id=session_id,
            )
            idx += 1

            # 续作步：仅 gap>=1（接管后 agent 继续执行了真实动作）才补
            if should_split_takeover:
                _write_step(
                    out_session_dir, idx,
                    instruction_goal=instruction_text,
                    prediction=main_prediction,
                    action=main_action,
                    thought=thought,
                    takeover_input=takeover_input or "",
                    src_image=src_image,
                    session_id=session_id,
                )
                idx += 1
        else:
            _write_step(
                out_session_dir, idx,
                instruction_goal=instruction_text,
                prediction=main_prediction,
                action=main_action,
                thought=thought,
                takeover_input=instruction.get("takeover_input", ""),
                src_image=src_image,
                session_id=session_id,
            )
            idx += 1

        if isinstance(curr_step_count, int):
            prev_step_count = curr_step_count

    if verbose:
        print(f"[ok] {session_id}: {len(items)} 个平台目录 -> {idx} 个 sessions step  ({out_session_dir})")
    return session_id, idx


def convert_task(task_dir, output_root, verbose=True):
    """转换一个设备级 task 目录。

    注意：同一个 PHONE 目录里可能同时包含多个阶段（如 pre_1 + main），靠每个
    step 的 extInfo.session_id / task_stage 区分。这里按 session_id 分组，
    每组各自落成一个独立的 sessions 目录（step_index 独立从 0 开始）。
    """
    meta = _read_json(os.path.join(task_dir, "meta.json"))
    goal = meta.get("goal") or (meta.get("ext_info") or {}).get("goal") or ""

    step_names = sorted(
        [d for d in os.listdir(task_dir)
         if d.isdigit() and os.path.isfile(os.path.join(task_dir, d, "instruction.json"))],
        key=int,
    )
    if not step_names:
        return None

    # 按 session_id 分组（保持目录数字顺序）。session_id 缺失时用目录名+阶段兜底。
    fallback_base = os.path.basename(os.path.dirname(os.path.dirname(task_dir)))
    groups = OrderedDict()
    for name in step_names:
        step_dir = os.path.join(task_dir, name)
        instruction = _read_json(os.path.join(step_dir, "instruction.json"))
        sid = _session_id(instruction)
        if not sid:
            stage = instruction.get("task_stage") or "main"
            sid = f"{fallback_base}_{stage}"
        groups.setdefault(sid, []).append((name, step_dir, instruction))

    results = []
    for sid, items in groups.items():
        results.append(_emit_session(sid, items, goal, output_root, verbose))
    return results


def main():
    ap = argparse.ArgumentParser(description="采集平台原始 dump -> 本地 sessions 格式")
    ap.add_argument("--input", required=True, help="平台 dump 根目录（rollout_id 目录或更上层）")
    ap.add_argument("--output", required=True, help="输出 sessions 根目录")
    args = ap.parse_args()

    task_dirs = find_task_dirs(args.input)
    if not task_dirs:
        raise SystemExit(f"未在 {args.input} 下找到任何 task（含 meta.json + 数字步目录）")

    os.makedirs(args.output, exist_ok=True)
    print(f"发现 {len(task_dirs)} 个 task，开始转换 -> {args.output}")
    total_steps = 0
    total_sessions = 0
    for td in task_dirs:
        res = convert_task(td, args.output)
        if not res:
            continue
        for _sid, n_steps in res:
            total_sessions += 1
            total_steps += n_steps
    print(f"完成：{len(task_dirs)} 个 task -> {total_sessions} 个 session，共 {total_steps} 个 step。")


if __name__ == "__main__":
    main()
