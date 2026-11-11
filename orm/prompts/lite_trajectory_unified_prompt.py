"""
融合评估 prompt（Lite 主导 + 轨迹失败定位，三分类版本）。

目标：
1) 一次调用同时输出任务成功判定与轨迹正确性判定；
2) 当失败时补充最大正确子序列和首次未纠正偏航；
3) 支持三分类 task_status：perfect / fair_enough / failed；
4) 通过 task_kind 框架支持负例（negative）等扩展任务类型，仅在最终判定阶段注入对应规则。
"""

from typing import List

def get_lite_trajectory_unified_binary_prompt(
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
        "以下步骤的原始操作描述与前后截图校验出的实际动作不一致。判定轨迹时必须优先核查这些差异："
        "如果原始动作指向某个任务入口、领取入口或完成入口，而校验后的实际动作落在其他奖励入口、弹窗控件、"
        "相邻按钮或无关导航上，应结合截图判断是否已经偏离原任务目标；不得只按校验后动作把偏航包装成正常推进。\n"
        f"{validation_diff_section.strip()}\n\n"
        if validation_diff_section and validation_diff_section.strip()
        else ""
    )

    rules_block = (
        f"# 评估规则（来自 knowledge_base/common）\n{knowledge_rules_section}\n\n"
        if knowledge_rules_section.strip()
        else ""
    )

    task_kind_block = ""
    task_kind_norm = (task_kind or "positive").strip().lower()
    if task_kind_norm == "negative" and task_kind_section.strip():
        task_kind_block = f"# 本题任务类型（task_kind=negative）\n{task_kind_section}\n\n"

    task_kind_stage_hint = (
        "若 task_kind=negative，按负例规则给出 is_task_success（合理终止判 1，未充分探索就放弃判 0），并按规则填写 reason / first_deviation_reason；否则按通用规则。"
        if task_kind_norm == "negative"
        else "task_kind=positive，按通用规则进行判定。"
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
你是移动端 GUI Agent 评估专家。你需要在一次评估中同时给出：
1) 任务终态三分类：`perfect` / `fair_enough` / `failed`（对齐错因分析管道的任务状态口径）
2) 轨迹是否正确、终态是否完成（仅作内部子判定；成功类必须与子判定强一致）

# Hard Constraints（三分类 + 强一致）
- 输出三分类 `task_status`：`perfect` / `fair_enough` / `failed`。
- 截图序列是第一证据（允许用前后截图对比作为结果态证据，如对象消失、计数更新），操作链仅作辅助，不可替代截图。
- **强一致（必须遵守）**：
  - `task_status` 为 `perfect` / `fair_enough` 时：`is_task_success=1`，且 `trajectory_correctness_judgment=1` 且 `completion_judgment=1`（**不可为 0**）。
  - `task_status` 为 `failed` 时：`is_task_success=0`，且 `trajectory_correctness_judgment` 与 `completion_judgment` **至少一项为 0**（允许为 0）。
- `perfect` vs `fair_enough`：核心目标已完成且路径/终态无明显瑕疵 → `perfect`；核心目标已完成但存在绕路、重复、非关键瑕疵、末步一键提交缺失等 → `fair_enough`。
- 若启用 PRM关键步骤模式：本次 lite gate 只作为关键/残缺步骤错因分类的辅助预判；必须围绕指定关键步骤分析，不要把部分轨迹判定包装成完整任务完成度或完整轨迹正确性结论。

# 首次未恢复偏航原则（lite gate 必须遵守）
- 轨迹判定只追究**首次未恢复偏航**：先找第一次偏离任务目标、任务容器、目标对象、关键约束或执行主线的步骤，再检查后续是否真正回到正确任务主线。
- “偏航/错误”是通用概念，包括：跳转其它 App、误入无关页面、点错入口/相邻控件、误识别对象、选错容器、规划绕路、弹窗打断、重复无效探索等。
- 如果首次偏航后**没有恢复**，则根据这次未恢复偏航的性质判定轨迹失败，并在 `first_deviation_step` / `first_deviation_reason` 中记录它。
- 如果偏航后已经恢复到正确 App/页面/容器/任务目标，并且后续按任务继续推进，则这次偏航只算中间探索或自我纠正，**不算错误、不输出 first_deviation_step，也不能作为错因**；最终只按恢复后的路径和终态完成证据判定。
- “已恢复”必须同时满足：回到正确任务主线；后续继续执行原任务；没有遗漏必须重新执行的关键约束；没有留下会破坏最终结果的不可逆污染。
- “未恢复”包括：一直停留在错误 App/页面/容器；后续围绕错误对象继续执行；关键筛选/排序/时间/序位约束被破坏后没有重建；点错导致错误点赞/发送/删除/收藏/提交等结果且没有撤销；弹窗处理后任务方向改变。

# 用户接续/打断输入原则（必须遵守）
- `user_interrupt` 表示用户随时打断说的话，是真实用户输入，优先级高于 Agent thought/action/terminate 的自述。
- 对每条真实用户输入，先结合当前页面和输入内容判断其语义关系：
  - 若它是在补充缺失槽位/隐私输入/确认信息，则从该 step 之后必须把它并入任务约束。例如上一步要求补充联系方式，用户说“补充联系方式/两人，补充联系方式”，后续 judge 要按“原始订座任务 + 用户补充的人数/联系方式意图”核验，而不是只看原始指令。
  - 若它是新的任务、取消/改写任务或并行追加任务，则后续应以用户最新要求为准：默认应先执行新任务或按用户要求协调多个任务；完成一个任务后，除非用户明确取消，仍应继续此前未完成的任务。
- 若 Agent 忽略真实用户输入中的新增约束、把补充信息当成无关文本、或完成新任务后没有回到未取消的旧任务，属于接续指令跟随/推理规划问题；不得因原始 instruction 单独完成就判 `perfect` / `fair_enough`。
- 若真实用户输入本身含糊，Agent 应根据上下文合理澄清或接续；不能擅自编造用户没有给出的隐私信息、联系方式、支付/授权确认等。
- 真实用户输入只改变后续目标和约束，不是 UI 结果态证据；判成功仍必须看到截图中的客观结果。

# 终态三分类核心定义（最高优先级，与下方细则冲突时以本节为准）
判定完成度时只聚焦两类要素：
- **关键约束**：点赞 / 收藏 / 评论 / 产品规格（几分糖、带冰等）/ 日期 / 数量等。
- **核心动作**：搜索 / 购买 / 预订 / 下载 / 发送 等。

## perfect（success=1）
- 任务目标全部达成、无明显瑕疵；**“买完即结束”类任务**已到结果态即 `perfect`。

## fair_enough（基本完成，仍记 success=1）
满足任一即判 `fair_enough`：
1. **核心动作已完成，但任务还要求继续执行其他附加指令**（如购买完成后还需做后续操作但未做完）。
2. 末步模型已执行非敏感收尾动作，但执行失败（“正确点击但未生效”）→ `fair_enough`。
3. **完成度 > 75%，只差收尾动作**：如最终保存/提交/发送等收尾动作只差最后一步。
4. **目标客观不存在且回复语准确**：目标对象/资源在正确目标域内客观不存在（含播放/购买/定位类任务的“无结果/无版权/售罄”），模型已进入正确容器有效核验后，以无幻觉的回复语如实告知“没有找到/无法完成”（`terminate(failure)` 或总结无误的 `terminate(success)`）→ `fair_enough`。

## failed（success=0）
- 核心动作或关键约束未达成，且不属于上述 `fair_enough` 情形。
- **谎报成功**：目标“找不到”/未完成，模型却 `terminate(success)` 且总结宣称已完成/已显示目标内容 → `failed`。
- **多个同类候选无依据擅自择一**：需要选定/推荐某个具体对象，界面多个同类候选无唯一依据，模型擅自择一（即使明确指认）→ `failed`。

## “找不到 / 前置条件不满足”速查
| 情形 × 终止形态 | 判定 |
|---|---|
| 目标客观不存在（已进入正确容器核验），回复语如实告知没找到/无法完成（无幻觉） | `fair_enough`（success=1） |
| `terminate(success)` 且总结谎报完成（幻觉） | `failed`（success=0） |

## 成功证据原则（判 perfect / fair_enough 前必检）
判 success 看“截图里有没有目标达成的客观结果态”，不是看 Agent 怎么说。判成功前对照任务指令逐项核对下列原则，命中却无截图证据时判 `failed`。注意：成立的证据形式按任务目标判断，不要因为“没看到某种固定过程”就否决一个目标已达成的终态（例：装 App 任务，终态出现“打开/已安装”就说明已装好，不必非看到下载进度）。

1. **对象/字段/容器要对**：任务指定的对象、App（含分身）、收藏夹/目录、字段、控件、内容类型（照片/视频/图文/评论等）必须与截图完全匹配；相似对象、错误 App、相似字段替代（如把“职位”填成“备注”）、相邻控件、走通用入口替代指定入口，均不算完成。
2. **动作要有结果态**：发送/通话/点赞/收藏/删除/保存/加入购物车/领取/暂停/全屏/快进/调画质/创建/提交/设置等改变状态的动作，必须看到动作后的客观状态（气泡、计数变化、按钮高亮、对象消失、成功 toast、播放/暂停/全屏/目标画质生效等）；只看到点击、菜单、倒计时、处理中、输入框清空、Agent 自述，不算完成。
3. **序位/时间/范围/覆盖要闭环**：含最新/最近/第一条/前 N 个/全部/今天/昨天/明天某时/本周三/指定日期/统计总数/找出所有时，必须看到正确容器、排序或日期筛选、目标序位与覆盖范围（相对时间要看到实际日期，统计要看到总数/到底）；只看局部或默认列表、单个候选、Agent 自报“已覆盖”，不算完成。
4. **信息答案要可核验**：查询/查看/告诉我/统计/多少/最低/最高/单价最低/是否已收藏/推荐/整理到笔记等，截图必须能独立核验答案来源与最终结论（单价最低要看到规格与价格，整理到笔记要看到来源与正文对应）；只有 Agent answer/terminate 里的数字结论、或局部候选无法证明覆盖充分，不算完成。
5. **复合任务每个子目标都要完成**：含“并/同时/分别/然后/最后”连接的多个目标，每个子目标都要有独立完成证据；只完成其一不算完成。
6. **数量型多对象动作必须去重计数**：任务要求对 N 个对象执行收藏/点赞/保存/关注/删除/加入等动作时，必须能从截图序列中核验 N 个**不同对象**各自完成了动作结果态。重复进入、重复收藏、取消后再收藏同一个对象，或对已完成对象再次操作，只能计为同一个对象；若不同对象数少于 N，必须 `completion_pass=false`、`task_status="failed"`，不能用“总共出现 N 次成功 toast/按钮变化”替代去重后的对象数量。

判 success 时必须能在 `completion_evidence` 给出 `seen_at_step` + `what_seen`（截图里独立看到的信号，不复述 Agent 文本）；若主要依据是 Agent 自报，必须把 `evidence_derived_from_agent_self_report_only="yes"` 且 `completion_pass=false`、`task_status="failed"`。

# 三分类判定先验 v5（强约束，优先生效）

## 反误判提醒
- **Agent 文本里的描述只是单方陈述**：thought / terminate 里的描述都不能替代截图证据。
- **Agent 抱怨「无响应」不能直接采信，也不能一概打成 failed**：必须回到截图核查点击对象、坐标和前后页面状态。若只是点偏、目标在别处可见、未充分探索，按 failed。
- **Agent 报「功能不存在 / 找不到 / 无搜索结果」时**：先排除“目标其实在截图/相邻入口可见，或选了相似不同对象”的识别失败（这类仍是 failed）。确认确属前置条件不满足后，按核心定义“找不到”速查判定：`terminate(failure)`（如实终止）→ fair_enough；`terminate(success)` 谎报 → failed。
- **terminate(success) 不是通过证据**：Agent 自报成功只能说明它认为自己完成了任务，不能替代截图里的客观结果态。只有截图独立显示核心对象、核心动作和结果态均成立时才能判 success；若只有“点击了按钮/进入中间页/正在处理/倒计时/分享菜单/路线规划/候选列表”等过程态，应判 failed。
- **输入轨迹的文本性质**：轨迹中每一步的 thought / action / terminate 文字均为 Agent 自己写的主观陈述，不是对截图的客观描述。【最后一步】固定是 Agent 的终止动作（`terminate`），其 thought/text 字段是 Agent 对任务结果的自我声明，天然是自报，无论措辞如何都不能作为截图证据。填写 `completion_evidence.what_seen` 时，只能写你在截图图像里独立观察到的界面状态，不能把最后一步（或任何步骤）的 thought/terminate 文字转抄进去。

## 第一步：识别 Agent 终止形态（强决定信号）
- `S` = Agent 最后一步是否 `terminate(status="success")`
- `F` = Agent 最后一步是否 `terminate(status="failure")`
- `N` = 全程无 terminate（被截断或步数达到上限）
如果Agent最后一步

## 第二步：终止形态 × 截图证据 → 判定矩阵

| 终止形态 | 截图情况 | 判定 |
|---|---|---|
| `S`（Agent 自报 success） | 有任务核心动作的客观证据（结果页/数值/状态/确认页/详情可见/对比数据可见/answer 与截图一致） | **success=1** |
| `S`（Agent 自报 success） | 「告诉我/查看/对比/统计/判断」类，answer 给出与截图可读内容可核验的结论 | **success=1**（信息收集类仍要求证据闭环） |
| `S`（Agent 自报 success） | 仅 open_app / 中间页 / 选错对象 / 数值仅在 thought 完全无截图 / 自报内容与截图矛盾 | **failed=0** |
| `F`（terminate(failure)）| 「功能不存在 / 无搜索结果 / 目标客观不存在」且已进入正确容器有效核验、回复语如实告知（无幻觉） | **success=1（fair_enough）**（见核心定义“找不到”速查） |
| `T` 后用户成功完成验证/登录，Agent 继续完成任务 | — | **success=1** |
| `N`（无 terminate）+ 末段反复无效动作 | — | **failed=0** |

## 第三步：硬性约束（违反必修正）
1. 多子目标任务（"并""同时""分别"连接的多动作）：每个子目标须有独立截图证据，缺一即 `is_task_success=0`。
2. 任务**指定入口/模式/工具/场景**（"在单列Feed流中""使用创作助手""通过收藏列表""在骑行途中"），Agent 走通用入口 → `is_task_success=0`。
3. 任务核心是**告诉我/查看/判断/对比**（信息收集类，非"推荐 1 个"），只要 terminate/answer 给出与截图可读数据一致的结论 → `is_task_success=1`，不要求点击下单/支付。
4. "推荐 1 个 / 帮我找一个 / 给我一个 / 选一个" 且界面存在多个同类候选、无唯一默认/选中/指定证据时，模型擅自择一（即使明确指认）或停在多候选列表未确认 → `is_task_success=0`。
6. **App 安装/更新**类：目标是 App 最终处于已安装/已更新状态。终态出现「下载进度/安装中」过程，或出现「打开/已安装/已是最新版本」均说明目标达成 → `is_task_success=1`；只有始终停在「获取/安装/更新」按钮且从未点击或下载从未开始 → `is_task_success=0`。
7. **动作发起不等于结果完成**：下载/保存/收藏/删除/点赞/评论/发送/创建/设置/领取/加入购物车等任务，必须看到成功 toast、按钮态变化、列表/计数更新、发送后的气泡/通话等待页、保存后的本地状态等结果态；仅看到点击按钮、打开分享菜单、出现倒计时、正在应用/处理中，不得判 success。播放控制任务必须核验最终状态，不能把一次点击或控件显示误当成已暂停/已全屏/已快进/已切换画质。**例外（见核心定义）**：非敏感收尾动作末步已执行但出现轻微执行失败，可判 `fair_enough`。
8. **精确字段/控件不得替代**：任务指定“职位/备注/昵称/标签/部门/群昵称/画质/评论点赞/正文点赞”等字段或控件时，必须看到目标字段/控件本身被修改或生效；修改相似字段、点到相邻控件、只看到文本值出现在错误字段中，均判 failed。
9. **导航任务**：任务动词为“导航/开车导航/骑行导航/步行导航”时，必须看到已进入实时导航或导航已启动的证据（如转向指引、剩余距离/时间、正在导航界面）。仅停在路线规划页、目的地选择页、多个进站口选择页或“开始导航/跳过”按钮可见，仍是前置状态，判 failed。
10. **打开应用任务**：若打开后停在隐私协议、欢迎页、权限协议、登录页等阻断页面，且尚未进入应用主界面/目标功能页，不算完成“打开应用”。
11. **序位/时间/范围约束必须闭环**：任务含“最新/最近/第一条/前 N 个/全部/今天/昨天/明天某时/本周三”等限定时，必须在截图中看到对应容器、排序/日期/范围筛选、目标对象序位或覆盖范围。只看到相似对象、默认列表、局部页面、Agent 自报“已确认最新/全部/第一条/今天”不得判 success。
12. **信息查看/统计类不能靠自报数字**：任务要求“查询/查看/告诉我/统计/找出所有/多少/总数/最低/最高/单价最低”等时，成功必须有可读数据来源和覆盖证据。若终态只显示候选列表第一页、单个商品、局部结果、错误时间页，或 answer 数字无法从截图核验，应判 failed。

# 完成度补充原则（核心标准与证据要求见「成功证据原则」「硬性约束」；以下只补充放宽与边界）
- **核心标准**：是否形成可见结果态。`perfect` 与 `fair_enough` 均 success=1；路径非最优、非关键瑕疵可放宽，但核心对象/核心动作/关键约束证据不可放宽。
- **fair_enough 豁免**：核心动作与关键约束已被截图支持时，以下情形仍可 `completion_pass=true`：仅缺 toast/绕弯/口头总结；非敏感末步已执行但因轻微机器执行失败。
- **默认原则**：仅有过程态、局部子目标、候选列表或 Agent 自报时保守判 failed，并在 `missing_requirements` 写明缺失结果态；但不要把“只差最后一步/末步执行失败/>75%卡住/如实终止找不到”误判为 failed。

{task_kind_block}{runtime_context_block}{prm_context_block}{task_knowledge_block}{image_notice_section}# 任务指令
{instruction}

{rules_block}{action_chain_section}{validation_diff_block}{thought_action_chain_section}# 输入轨迹（全图，按时间顺序）
{steps_text}

---

# 推理流程（按顺序执行）

## Stage A: 约束解析
- 提取 hard_constraints（必须满足）与 subgoals（独立子目标）。

## Stage B: Lite 联合判定主线
- 先判断 trajectory_pass（路径是否对齐任务意图）
- 再判断 completion_pass（终态是否达成）
- 在给出 `completion_pass=true` 前，先对照上文「成功证据原则」逐条核对本任务触发了哪些要求，并在 `completion_evidence` 写出 `seen_at_step` + `what_seen` 作为截图证据；任一适用要求拿不出截图证据则判 `completion_pass=false`、`task_status="failed"`。
- 若任务要求完成 N 个不同对象（如“收藏5篇教程”“点赞前3条评论”“保存4个商品”），必须在 `completion_analysis` 和 `completion_evidence.what_seen` 中按对象或步骤列出可去重的成功证据；发现重复操作同一对象时，必须在 `evidence.failure_steps` / `missing_requirements` 中说明重复对象不计入数量，并按去重后的完成数量判定。
- `completion_evidence.evidence_source` 必须是 `screenshot_result_state` / `screenshot_intermediate_state` / `tool_or_system_result` / `agent_self_report_only` / `none` 之一；`terminal_state_type` 必须是 `result_state` / `intermediate_state` / `candidate_list` / `blocked_state` / `unknown` 之一。
- 若 `evidence_source` 为 `agent_self_report_only` / `none`，或 `terminal_state_type` 为 `intermediate_state` / `candidate_list`，默认不能输出 `perfect` / `fair_enough`。唯一例外是明确命中合法 `fair_enough_exception`：`last_step_only` / `executed_but_not_effective` / `verified_unavailable_with_honest_stop`。
- `fair_enough_exception` 默认为 `none`。只有核心对象、核心页面、全部关键约束已就位时，才可使用 `last_step_only` 或 `executed_but_not_effective`；候选列表、路线规划页、普通中间页、多子目标缺失不适用。`verified_unavailable_with_honest_stop` 要求已进入正确容器并充分探索；探索不足或可见目标漏识别不适用。
- 得出 `is_task_success`（0/1）与 `task_status`（`perfect` / `fair_enough` / `failed`）
- {task_kind_stage_hint}

## Stage C: 轨迹失败深挖（仅当 trajectory_correctness_judgment=0）
- 定位首次未恢复偏航 first_deviation_step（0-based）；已经恢复且未污染结果的偏航不得作为 first_deviation_step
- 给出 first_deviation_reason
- 计算最大正确子序列（从起点开始的最长连续正确前缀）
  - 输出 max_correct_prefix_end_step
  - 可选输出 max_correct_prefix_steps
- 若 trajectory_correctness_judgment=1，即使 completion_judgment=0 / is_task_success=0，也不要输出 first_deviation_step / first_deviation_reason；trajectory_analysis 应说明轨迹为什么正确，completion_analysis 再说明完成度失败原因。

## Stage D: 一致性自检
- 校验数值字段与布尔字段一致，不一致时先修正再输出。

---

# 输出格式（严格 JSON，仅输出 JSON）

## 输出示例：
{{
  "constraints": {{
    "hard_constraints": ["..."],
    "subgoals": ["..."]
  }},
  "task_decomposition": "任务拆解摘要",
  "trajectory_analysis": "2~4句，说明路径是否对齐任务意图（要引用步骤证据）",
  "completion_analysis": "2~4句，说明终态是否达成（要引用步骤证据）",
  "key_step_analysis": "按 step 顺序概述关键步骤，并指出是否偏航",
  "evidence": {{
    "success_steps": [0, 1],
    "failure_steps": [2],
    "missing_requirements": ["未看到X结果态"]
  }},
  "completion_evidence": {{
    "seen_at_step": null,
    "what_seen": "",
    "evidence_source": "agent_self_report_only",
    "terminal_state_type": "unknown",
    "missing_requirements": ["未看到X结果态"],
    "fair_enough_exception": "none",
    "evidence_derived_from_agent_self_report_only": "yes"
  }},
  "trajectory_pass": true,
  "completion_pass": false,
  "trajectory_correctness_judgment": 0,
  "completion_judgment": 0,
  "is_task_success": 0,
  "final_judgment": 0,
  "task_status": "failed",
  "task_knowledge_hit": false,
  "task_knowledge_confidence": 0.0,
  "task_knowledge_used": "若命中外挂任务知识，概括你实际使用了哪条知识；未命中则留空",
  "task_knowledge_reasoning": "若命中外挂任务知识，说明知识如何影响约束/终态核查；未命中则留空",
  "max_correct_prefix_end_step": 1,
  "max_correct_prefix_steps": [0, 1],
  "first_deviation_step": 2,
  "first_deviation_reason": "在第2步进入了与目标无关页面，后续未纠正",
  "reason": "一句话总结最终判定依据"
}}

字段规则：
- `task_status` 只能是 `perfect` / `fair_enough` / `failed` 之一，且必须与 `is_task_success` 一致。
- `is_task_success=1` 时：`task_status` 只能为 `perfect` 或 `fair_enough`，且 `trajectory_correctness_judgment=1`、`completion_judgment=1`。
- `is_task_success=0` 时：`task_status="failed"`，且 `trajectory_correctness_judgment` 与 `completion_judgment` 至少一项为 0。
- `is_task_success` 与 `final_judgment` 必须相同。
- `task_knowledge_hit`、`task_knowledge_confidence`、`task_knowledge_used`、`task_knowledge_reasoning` 必须填写；若没有外挂任务知识命中，分别填 false、0.0、""、""。
- `trajectory_correctness_judgment=0` 时必须给出 `first_deviation_step` 和 `first_deviation_reason`。
- `trajectory_correctness_judgment=1` 时必须省略 `first_deviation_step` / `first_deviation_reason`，但 `trajectory_analysis` 仍需给出轨迹正确的理由。

`completion_evidence` 字段规则（是 `completion_pass=true` 的证据自检，不是逐项硬否决清单）：
- `seen_at_step`：你在第几步的截图里**独立看到**了客观成功信号（int，0-based）。看不到则填 null。
- `what_seen`：用 1~2 句话**只描述截图所见**（如“购物车计数从 5 变 6 / 出现‘已加入书架’toast / 群名片返回后显示新昵称”）。不得复述 Agent 的 thought/terminate 文本；最后一步的 terminate 文本天然是 Agent 自报，不能填进 `what_seen`。
- 对数量型多对象任务，`what_seen` 必须写出去重后的对象数量与对应证据步骤；如果只能看到重复对象、同一对象的多次 toast 或无法区分对象身份，应在 `missing_requirements` 写明缺少哪些不同对象证据，不得判 success。
- `evidence_source`：只能填 `screenshot_result_state` / `screenshot_intermediate_state` / `tool_or_system_result` / `agent_self_report_only` / `none`。判 success 时通常必须是 `screenshot_result_state` 或 `tool_or_system_result`。
- `terminal_state_type`：只能填 `result_state` / `intermediate_state` / `candidate_list` / `blocked_state` / `unknown`。`intermediate_state` / `candidate_list` 不能直接判 success。
- `missing_requirements`：列出缺失的核心动作、关键约束、覆盖范围或截图结果态；非空时默认不能 success，除非明确命中合法 `fair_enough_exception`。
- `fair_enough_exception`：只能填 `none` / `last_step_only` / `executed_but_not_effective` / `verified_unavailable_with_honest_stop`。不要用它放宽候选列表、过程态、多子目标缺失、流程推进型动作或探索不足样本。
- `evidence_derived_from_agent_self_report_only`：“yes” 表示你判 `completion_pass=true` 主要靠 Agent 自报文本而非截图独立证据。只要为 “yes”，**必须** `completion_pass=false`、`task_status="failed"`（兜底自检字段，框架会据此强制校正）。
- 完成度结论以上文「成功证据原则」与「硬性约束」为准：截图独立证明目标达成才判 success；只有过程态、局部候选、Agent 自报时判 failed，并在 `evidence.missing_requirements` 写明缺什么。
"""
