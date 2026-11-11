"""
MobileGym skill-pipeline trajectory judge prompt.

This prompt keeps a chronological trajectory-evaluation structure while focusing
on the MobileGym skill pipeline's actionable failure decision point.
"""


MOBILEGYM_SKILL_TRAJECTORY_PROMPT_VERSION = "v1"


def get_mobilegym_skill_trajectory_prompt(
    instruction: str,
    steps_text: str,
    action_chain: str = "",
    thought_action_chain: str = "",
    image_notice_section: str = "",
    knowledge_rules_section: str = "",
    task_kind: str = "positive",
    task_kind_section: str = "",
    action_chain_source: str = "",
    runtime_context_section: str = "",
    task_knowledge_section: str = "",
    validation_diff_section: str = "",
    prm_context_section: str = "",
) -> str:
    del action_chain_source

    action_chain_section = (
        f"# 操作链（仅辅助理解，截图证据优先）\n{action_chain}\n\n" if action_chain else ""
    )
    thought_action_chain_section = (
        f"# 意图-操作链（仅辅助理解，截图证据优先）\n{thought_action_chain}\n\n"
        if thought_action_chain
        else ""
    )
    validation_diff_block = (
        "# 动作校验差异（原始动作 vs 实际图像校验动作）\n"
        "以下步骤的原始操作描述与前后截图校验出的实际动作不一致。判定时必须优先核查这些差异，"
        "尤其要判断错误后果是由哪个前序动作造成的。\n"
        f"{validation_diff_section.strip()}\n\n"
        if validation_diff_section and validation_diff_section.strip()
        else ""
    )
    rules_block = (
        f"# MobileGym 通用评估规则\n{knowledge_rules_section.strip()}\n\n"
        if knowledge_rules_section and knowledge_rules_section.strip()
        else ""
    )
    task_kind_block = ""
    task_kind_norm = (task_kind or "positive").strip().lower()
    if task_kind_norm == "negative" and task_kind_section.strip():
        task_kind_block = f"# 本题任务类型（task_kind=negative）\n{task_kind_section}\n\n"
    task_kind_stage_hint = (
        "若 task_kind=negative，按负例任务规则判断合理终止、充分探索和是否误报成功；否则按通用正例任务规则判断。"
        if task_kind_norm == "negative"
        else "task_kind=positive，按通用正例任务规则进行判定。"
    )
    runtime_context_block = (
        f"# 运行上下文（来自任务元数据，仅作判定辅助）\n{runtime_context_section.strip()}\n\n"
        if runtime_context_section and runtime_context_section.strip()
        else ""
    )
    task_knowledge_block = (
        f"{task_knowledge_section.strip()}\n\n"
        if task_knowledge_section and task_knowledge_section.strip()
        else ""
    )
    prm_context_block = (
        f"# PRM关键步骤模式\n{prm_context_section.strip()}\n\n"
        if prm_context_section and prm_context_section.strip()
        else ""
    )

    return f"""# Role
你是 MobileGym 技能抽取流水线的轨迹级评估专家。你需要一次性判断：
1) 任务是否成功完成：`perfect` / `fair_enough` / `failed`
2) 轨迹是否沿正确任务主线推进、终态是否有截图证据
3) 如果失败，定位可用于重采样和技能抽取的错误动作步骤 `wrong_action_step`

# Hard Constraints
- 只输出严格 JSON，不输出 Markdown、解释文本或代码块。
- 截图序列是第一证据。Agent 的 thought/action/terminate 文本只能辅助理解，不能替代截图里的客观状态。
- `task_status` 只能是 `perfect` / `fair_enough` / `failed`。
- `task_status` 为 `perfect` / `fair_enough` 时：`is_task_success=1`、`final_judgment=1`、`trajectory_correctness_judgment=1`、`completion_judgment=1`。
- `task_status` 为 `failed` 时：`is_task_success=0`、`final_judgment=0`，且 `trajectory_correctness_judgment` 与 `completion_judgment` 至少一项为 0。
- 任务知识是高置信参考。命中任务知识时仍必须回到截图核查。

# MobileGym 技能流水线目标
本评估的失败定位目标是找技能流水线可采取行动的错误决策点。

必须区分两个步骤：
- `first_bad_state_step`：第一次在截图/状态中能看到错误后果的步骤，0-based。
- `wrong_action_step`：导致该错误后果的动作步骤，0-based。它是重采样、对比纠错、技能抽取应该使用的步骤。

例：step 2 点错入口，step 3 首次显示错误页面，则 `first_bad_state_step=3`，`wrong_action_step=2`。

按因果关系确定 `wrong_action_step`：
- 如果当前步骤本身包含错误动作，`wrong_action_step` 就是当前步骤。
- 如果坏状态来自更早的输入、选择、切换、滑动或终止判断，要回溯到真正造成错误的动作。
- 如果只是完成度失败、缺少后续动作，`wrong_action_step` 通常是最早不该停止、遗漏关键动作或错误终止的步骤。

# 成功/失败证据原则
- 判成功必须看到任务核心对象、核心动作、关键约束和结果态的截图证据；Agent 自报成功不算完成证据。
- 发送、收藏、点赞、保存、购买、设置、删除、下载、提交等改变状态的动作，必须看到结果态或可靠系统反馈。
- 查询、查看、统计、比较、告诉我类任务，必须看到可核验的数据来源和结论依据，不能只用 Agent answer 中的数字或总结。
- 最新、最近、第一条、前 N 个、全部、时间、日期、范围、排序、数量等约束必须闭环。
- 多子目标任务必须逐项完成；只完成部分子目标不算成功。
- 目标客观不存在且 Agent 已进入正确容器充分核验并如实告知，可判 `fair_enough`；探索不足、漏看可见目标或谎报成功则判 `failed`。

# 首次未恢复偏航原则
- 先按时间顺序找第一次偏离任务目标、目标容器、目标对象、关键约束或执行主线的步骤。
- 如果偏航后真正恢复到正确主线、继续执行原任务、没有遗漏关键约束，也没有留下不可逆错误，则不把这次偏航作为失败。
- 如果偏航未恢复，记录 `first_deviation_step`、`first_deviation_reason` 和 `max_correct_prefix_end_step`。
- `first_deviation_step` 用于解释轨迹何时开始错；`wrong_action_step` 用于 MobileGym 后续动作重采样，两者可以相同，也可以不同。

# 常见 MobileGym 失败类型
- `wrong_action`：点错、滑错、输入错、选择错、错误终止或遗漏必要动作。
- `target_distractor_confusion`：把相似控件、相似商品、广告、推荐项、相邻按钮、错误对象当成目标。
- `special_control_error`：滑块、分段控件、开关、日期时间选择器、评分控件等特殊控件操作错误。
- `scroll_range_error`：滚动方向、边界判断、覆盖范围或列表探索不足。
- `popup_overlay_error`：前景弹窗、广告浮层、活动遮罩层处理错误或误关闭。
- `completion_evidence_missing`：路径可能合理但终态缺少截图结果证据。
- `unclear`：无法可靠判断可行动错误点。

{task_kind_block}{runtime_context_block}{prm_context_block}{task_knowledge_block}{image_notice_section}# 任务指令
{instruction}

{rules_block}{action_chain_section}{validation_diff_block}{thought_action_chain_section}# 输入轨迹（全图，按时间顺序）
{steps_text}

---

# 推理流程（按顺序执行）

## Stage A: 约束解析
- 提取 hard_constraints 与 subgoals。
- 标记任务是否含数量、序位、时间、范围、覆盖、特殊控件、目标/干扰项等高风险约束。

## Stage B: 联合成功判定
- 判断 trajectory_pass：路径是否对齐任务意图。
- 判断 completion_pass：终态是否由截图证据证明完成。
- 在给出 success 前，必须填写 `completion_evidence.seen_at_step` 和 `completion_evidence.what_seen`；如果主要依赖 Agent 自报，必须判 failed。
- 得出 `task_status`、`is_task_success`、`final_judgment`。
- {task_kind_stage_hint}

## Stage C: MobileGym 动作对齐失败定位
- 若失败，先找 `first_bad_state_step`：第一次截图可见错误后果、错误页面、错误对象、错误状态、缺失终态或错误终止的位置。
- 再找 `wrong_action_step`：导致该后果的动作步骤，即后续应该重采样或抽取技能的步骤。
- 写明 `step_alignment_reason`：解释为什么坏状态和错误动作相同或不同。
- 若无法确定可行动步骤，`wrong_action_step=null`、`failure_mode_for_skill="unclear"`、`can_extract_skill=false`。

## Stage D: 一致性自检
- 校验 `task_status`、`is_task_success`、`final_judgment`、`trajectory_correctness_judgment`、`completion_judgment` 一致。
- 成功样本不得输出非空 `wrong_action_step` / `first_bad_state_step`。
- 失败样本若能定位动作，`wrong_action_step` 必须在轨迹步骤范围内。

---

# 输出格式（严格 JSON，仅输出 JSON）
{{
  "constraints": {{
    "hard_constraints": ["..."],
    "subgoals": ["..."]
  }},
  "task_decomposition": "任务拆解摘要",
  "trajectory_analysis": "2到4句，说明路径是否对齐任务意图，引用步骤证据",
  "completion_analysis": "2到4句，说明终态是否达成，引用截图证据",
  "key_step_analysis": "按 step 顺序概述关键步骤，并指出是否偏航",
  "evidence": {{
    "success_steps": [0, 1],
    "failure_steps": [2],
    "missing_requirements": ["未看到目标结果态"]
  }},
  "completion_evidence": {{
    "seen_at_step": null,
    "what_seen": "",
    "evidence_source": "agent_self_report_only",
    "terminal_state_type": "unknown",
    "missing_requirements": ["未看到目标结果态"],
    "fair_enough_exception": "none",
    "evidence_derived_from_agent_self_report_only": "yes"
  }},
  "trajectory_pass": false,
  "completion_pass": false,
  "trajectory_correctness_judgment": 0,
  "completion_judgment": 0,
  "is_task_success": 0,
  "final_judgment": 0,
  "task_status": "failed",
  "max_correct_prefix_end_step": 1,
  "max_correct_prefix_steps": [0, 1],
  "first_deviation_step": 2,
  "first_deviation_reason": "第2步开始偏离任务主线，后续未恢复",
  "first_bad_state_step": 3,
  "wrong_action_step": 2,
  "wrong_action_reason": "第2步动作选择了干扰项，导致第3步进入错误页面",
  "step_alignment_reason": "错误后果在第3步截图首次可见，但由第2步点击造成，因此可行动失败点是第2步",
  "failure_mode_for_skill": "wrong_action",
  "can_extract_skill": true,
  "skill_signal_type": "target_distractor_confusion",
  "task_knowledge_hit": false,
  "task_knowledge_confidence": 0.0,
  "task_knowledge_used": "",
  "task_knowledge_reasoning": "",
  "reason": "一句话总结最终判定依据"
}}

字段规则：
- `task_status` 只能是 `perfect` / `fair_enough` / `failed`。
- `is_task_success` 与 `final_judgment` 必须相同，成功为 1，失败为 0。
- `task_knowledge_hit`、`task_knowledge_confidence`、`task_knowledge_used`、`task_knowledge_reasoning` 必须填写；未命中则填 false、0.0、""、""。
- `first_deviation_step`、`first_bad_state_step`、`wrong_action_step` 均为 0-based；无法可靠定位时填 null，并在 reason 字段说明。
- `failure_mode_for_skill` 只能填 `wrong_action` / `target_distractor_confusion` / `special_control_error` / `scroll_range_error` / `popup_overlay_error` / `completion_evidence_missing` / `unclear` / `env_error`。
- `skill_signal_type` 可以更细，如 `target_distractor_confusion`、`previous_action_caused_bad_state`、`wrong_dismissal`、`range_boundary_miss`、`special_control_misuse`、`premature_stop`；无明确技能信号则填 `unclear`。
- `can_extract_skill` 只表示是否能定位清晰可行动错误步骤和可复用纠错信号。命中常见 MobileGym 失败族且能定位 `wrong_action_step` 时，通常填 true。
- 成功样本必须令 `first_deviation_step=null`、`first_bad_state_step=null`、`wrong_action_step=null`、`failure_mode_for_skill=""`、`can_extract_skill=false`、`skill_signal_type=""`。
- `completion_evidence.evidence_source` 只能填 `screenshot_result_state` / `screenshot_intermediate_state` / `tool_or_system_result` / `agent_self_report_only` / `none`。
- `completion_evidence.terminal_state_type` 只能填 `result_state` / `intermediate_state` / `candidate_list` / `blocked_state` / `unknown`。
- `completion_evidence.evidence_derived_from_agent_self_report_only="yes"` 时，必须 `completion_pass=false`、`task_status="failed"`。
"""


def get_mobilegym_skill_trajectory_binary_prompt(*args, **kwargs) -> str:
    """Compatibility alias for runner code that names prompts by judgment type."""
    return get_mobilegym_skill_trajectory_prompt(*args, **kwargs)
