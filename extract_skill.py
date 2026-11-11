"""从 better 重预测实例生成或迭代 MobileGym skill。

脚本只负责流程编排，不手写具体 skill 内容：读取
mobilegym_repredicted_step_judgments.json，按顺序遍历重预测实例，筛选
judgment.comparison_to_original 为 "better" 的实例，然后把已有 skill、
skillsample.md、单个完整实例 JSON 和可用截图交给指定 LLM 生成或更新
skill.md。
"""

from __future__ import annotations

import argparse
import shutil
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "mobilegym_repredicted_step_judgments.json"
DEFAULT_SAMPLE = PROJECT_ROOT / "skillsample.md"
DEFAULT_TEXT_SAMPLE = PROJECT_ROOT / "skillsample_text.md"
DEFAULT_TASK_CASE = PROJECT_ROOT / "case_task.md"
DEFAULT_TASK_SAMPLE = PROJECT_ROOT / "mobilegym" / "sample.jsonl"
DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"
DEFAULT_MESSAGES_DIR = PROJECT_ROOT / "skill_generation_messages"
DEFAULT_MODEL_BASE_URL = "http://10.80.10.0:30772/v1"
DEFAULT_MODEL_NAME = "Qwen3.5-27B-vllm"
SAMPLE_INDEX_FILENAME = "sample_index.json"
DEFAULT_IMAGE_SSIM_THRESHOLD = 0.96
DEFAULT_SSIM_IMAGE_SIZE = 256
DEFAULT_SSIM_TILE_SIZE = 16
DEFAULT_ACTION_COORDINATE_BUCKET_SIZE = 100
AGENT_TEMPLATE_IMAGE_FIELDS = [
    ("original_image", "orig", "原错误轨迹中错误步骤的屏幕截图"),
    ("som_image", "som", "重预测后的 SOM 辅助截图"),
]
AGENT_RESOLVED_IMAGE_FIELDS = [
    ("original_image", "原错误轨迹中错误步骤的屏幕截图"),
    ("repredicted_image", "重预测后的屏幕截图"),
    ("som_image", "重预测后的 SOM 辅助截图"),
]
CATEGORY_SKILL_NAMES = {
    1: "dismiss_popups",
    2: "scroll_region_targeting",
    3: "control_semantics",
    4: "prerequisite_actions",
    5: "captcha_handling",
    6: "media_type_disambiguation",
    7: "live_product_disambiguation",
    8: "ad_target_disambiguation",
    9: "custom_keyboard_input",
    10: "scroll_boundary_awareness",
}
CATEGORY_NAME_TO_SKILL_NAME = {
    "广告弹窗等未能关闭导致的失败": "dismiss_popups",
    "滑动起止点选择超出区域限制导致的失败": "scroll_region_targeting",
    "特殊形态控件理解有误导致的失败": "control_semantics",
    "按钮点击存在前置导致的失败": "prerequisite_actions",
    "人机验证导致的失败": "captcha_handling",
    "图片与视频区分有误导致的失败": "media_type_disambiguation",
    "直播与商品区分有误导致的失败": "live_product_disambiguation",
    "目标与广告区分有误导致的失败": "ad_target_disambiguation",
    "需要使用自定义键盘输入导致的失败": "custom_keyboard_input",
    "滑动达到上下限导致的失败": "scroll_boundary_awareness",
}
TASK_CATEGORY_NAME_TO_SKILL_NAME = {
    "通用搜索与信息查询": "task_general_search_info_query",
    "电商商品检索、浏览与比价": "task_ecommerce_product_browsing",
    "电商加购与购物车操作": "task_ecommerce_cart_actions",
    "账号登录、个人中心与权益/订单管理": "task_account_center_management",
    "金融、支付与车牌缴费": "task_finance_payment_plate_fee",
    "本地笔记创建与文本编辑": "task_note_text_editing",
    "电话记录查看、统计与回拨": "task_call_log_operations",
    "AI助手对话与回答分享": "task_ai_assistant_dialog_sharing",
    "视频搜索、剧集选择与推荐流互动": "task_video_search_episode_feed_interaction",
    "图片、短视频与相册操作": "task_image_video_album_operations",
    "电商直播间进入与直播内容查看": "task_ecommerce_live_room",
    "小说搜索、书籍详情与阅读": "task_novel_search_reading",
    "历史账单与时间范围查询": "task_history_bill_date_range_query",
    "设置项与媒体播放参数调节": "task_settings_media_parameter_adjustment",
}
SKILL_CONTRACTS = {
    "dismiss_popups": {
        "scope": (
            "处理遮挡当前任务流程的前景弹窗、浮层、modal、红包/优惠/活动面板、开屏广告等。"
            "核心纠错是先关闭、跳过、返回、取消、等待消失或绕过遮挡层，再继续原任务。"
        ),
        "out_of_scope": (
            "不要覆盖搜索结果、信息流、商品列表、推荐列表、下载列表等页面内容区中的广告/推广内容与真实目标混淆。"
        ),
        "neighbor_boundary": "内容区广告/推广卡片被误当成真实目标时归 ad_target_disambiguation。",
    },
    "scroll_region_targeting": {
        "scope": (
            "处理需要滚动但滑动起点、终点或手势路径落在错误容器、固定导航栏、不可滚动区域、"
            "遮挡区域之外，导致目标列表/面板没有按预期滚动的动作纠错。"
        ),
        "out_of_scope": (
            "不要覆盖已经滚到顶部/底部/左端/右端后的重复滑动无效，也不要覆盖不需要滚动、"
            "或仅仅滑动方向选择错误但区域本身正确的情况。"
        ),
        "neighbor_boundary": "滚动区域仍可滚但手势位置错归本技能；已经到达边界、继续同向滑动不会出现新内容归 scroll_boundary_awareness。",
    },
    "control_semantics": {
        "scope": (
            "处理对特殊形态控件、图标按钮、标签、状态 chip、开关、播放/暂停、展开/收起、"
            "筛选项等控件语义理解错误，导致点击了错误功能或漏点正确控件的动作纠错。"
        ),
        "out_of_scope": (
            "不要覆盖广告/推广内容与真实目标混淆、前置条件未满足导致按钮不可用、"
            "弹窗关闭问题或自定义键盘输入问题。"
        ),
        "neighbor_boundary": (
            "广告内容误判归 ad_target_disambiguation；按钮因缺少必填项/授权/勾选等前置而不可用归 prerequisite_actions；"
            "前景弹窗关闭归 dismiss_popups。"
        ),
    },
    "prerequisite_actions": {
        "scope": (
            "处理目标按钮或下一步入口本身存在，但因为未完成必填输入、勾选协议、选择规格/地址/身份、"
            "授权确认等前置动作，导致直接点击按钮无效或按钮处于不可用状态的纠错。"
        ),
        "out_of_scope": (
            "不要覆盖控件图形语义误解、自定义键盘输入方式错误、人机验证挑战、弹窗关闭或广告目标混淆。"
        ),
        "neighbor_boundary": (
            "控件含义看错归 control_semantics；必须用应用内键盘输入归 custom_keyboard_input；"
            "安全验证/验证码归 captcha_handling。"
        ),
    },
    "captcha_handling": {
        "scope": (
            "处理验证码、人机验证、安全验证、滑块验证、点选/拼图/图形验证等 verification 流程导致的失败，"
            "核心纠错是识别验证形态并按验证要求操作或等待/重试。"
        ),
        "out_of_scope": (
            "不要覆盖普通登录/权限/业务确认弹窗、营销广告弹窗、协议勾选前置条件，"
            "或非验证用途的应用内自定义键盘。"
        ),
        "neighbor_boundary": (
            "营销或遮挡弹窗归 dismiss_popups；应用内键盘输入归 custom_keyboard_input；"
            "缺少勾选/填写等前置动作归 prerequisite_actions。"
        ),
    },
    "media_type_disambiguation": {
        "scope": (
            "处理任务要求选择图片、视频、短视频、相册图片、播放内容等媒体类型时，"
            "把图片和视频入口/卡片混淆导致选错目标的语义纠错。"
        ),
        "out_of_scope": (
            "不要覆盖直播与商品卡片混淆、广告/推广内容与真实目标混淆，"
            "或任务本身没有媒体类型约束的普通点击选择。"
        ),
        "neighbor_boundary": "直播入口与商品卡片混淆归 live_product_disambiguation；广告内容误判归 ad_target_disambiguation。",
    },
    "live_product_disambiguation": {
        "scope": (
            "处理电商/内容平台中直播间入口、直播中商品、商品卡片、店铺入口等相邻目标被混淆，"
            "导致进入直播而非商品详情，或选择商品而非直播入口的语义纠错。"
        ),
        "out_of_scope": (
            "不要覆盖普通广告商品混淆、图片/视频媒体类型混淆，或任务明确要求进入直播且已正确选择直播入口的情况。"
        ),
        "neighbor_boundary": "广告/推广卡片误当目标归 ad_target_disambiguation；图片/视频入口混淆归 media_type_disambiguation。",
    },
    "ad_target_disambiguation": {
        "scope": (
            "处理页面内容区中的广告、推广、赞助、推荐、下载入口或推广卡片被误当成任务真实目标的语义混淆。"
            "核心纠错是识别真实目标、避开广告/推广内容，或返回后重新选择正确结果。"
        ),
        "out_of_scope": (
            "任何前景遮挡型广告弹窗、红包浮层、优惠弹层、开屏广告、modal 遮罩、关闭/跳过/返回入口定位失败都不属于本技能。"
            "纯弹窗关闭问题必须交给 dismiss_popups。"
        ),
        "neighbor_boundary": "只要失败直接原因是没有关闭/跳过/退出遮挡弹层，即使弹窗内有广告、领取、下载、立即查看等内容，也归 dismiss_popups。",
    },
    "custom_keyboard_input": {
        "scope": (
            "处理必须通过应用内自定义键盘、数字键盘、候选词面板、九宫格/按钮网格等屏幕控件输入，"
            "而不能仅依赖系统 TYPE 文本动作的输入失败。"
        ),
        "out_of_scope": (
            "不要覆盖验证码/安全验证中的点选或滑块挑战，也不要覆盖字段格式错误、缺少前置选择、"
            "协议未勾选等非键盘输入方式问题。"
        ),
        "neighbor_boundary": "验证型输入挑战归 captcha_handling；缺少前置选择或按钮不可用归 prerequisite_actions。",
    },
    "scroll_boundary_awareness": {
        "scope": (
            "处理列表、页面、横向容器或面板已经到达顶部/底部/左端/右端/内容末尾后，"
            "继续同方向滑动不会出现新内容，需要停止滚动、反向、返回或改用其他入口的策略纠错。"
        ),
        "out_of_scope": (
            "不要覆盖内容仍可滚动但滑动起止点落在错误区域，也不要覆盖仅仅手势方向或距离不理想但尚未到边界的情况。"
        ),
        "neighbor_boundary": "手势没有命中正确滚动容器归 scroll_region_targeting；已经命中容器但到达边界归本技能。",
    },
}

CATEGORY_DIAGNOSIS_PRIORS = {
    "dismiss_popups": {
        "wrong_actions": (
            "前景红包、优惠券、活动、百亿补贴等弹窗覆盖主任务页面时，错误动作常点击弹窗无效区域、"
            "弹窗旁边区域、错误角落，或点击“立即去抢/领取/查看”等营销入口。"
        ),
        "observable_pattern": (
            "主任务内容仍在弹窗背后，前景遮挡层阻止继续操作；这类弹窗通常存在自身关闭按钮或关闭文案，"
            "关闭入口可能很小、低对比、贴近边缘或不在常规右上角。"
        ),
        "correction_method": (
            "优先在弹窗自身范围内找关闭入口并点准关闭按钮，例如角落、底部、边缘、小图标、“关闭/跳过/暂不/取消”附近；"
            "确认弹窗消失后再继续原任务。"
        ),
        "writing_boundary": "核心经验是识别遮挡主任务的前景弹窗并关闭它；优先寻找弹窗自身关闭入口，系统返回只适合作为少数兜底。",
    },
    "scroll_region_targeting": {
        "wrong_actions": (
            "需要在列表、区域选择器或滑杆内细调时，错误动作常一次滑动过大、起终点落在错误区域，"
            "或拖动没有命中滑杆旋钮/轨道。"
        ),
        "observable_pattern": (
            "目标项通常已在当前列表附近或只需小幅移动；错误动作后目标被跳过、列表没有按预期移动，"
            "或滑杆数值/位置没有变化。"
        ),
        "correction_method": (
            "先判断真正可滚动容器，再在容器内部用短距离、低幅度手势逼近目标；遇到滑杆时从可见旋钮或轨道中心开始拖动，"
            "用结果变化校验是否命中。"
        ),
        "writing_boundary": "重点写区域命中和幅度控制，不把启动器、系统桌面或无关空白滑动作为核心规则。",
    },
    "control_semantics": {
        "wrong_actions": (
            "错误动作常把加号、排序、下拉、收藏、播放/进入详情、开关、滑杆等控件语义看错，"
            "点击后没有触发期望功能或进入了错误页面。"
        ),
        "observable_pattern": (
            "同屏存在多个形态相近但语义不同的图标/按钮；错误动作后没有新增、没有展开、没有选中、没有排序变化，"
            "或进入的页面与任务目标不匹配。"
        ),
        "correction_method": (
            "根据任务动词和控件附近文案判断控件功能，优先选择能直接产生目标反馈的控件；"
            "动作后检查页面状态，若没有对应反馈，应回到语义更明确的按钮、标签或目标项。"
        ),
        "writing_boundary": "强调控件语义和结果校验；前置条件、弹窗、广告目标、自定义键盘分别由对应技能处理。",
    },
    "prerequisite_actions": {
        "wrong_actions": (
            "错误动作常在登录、字段激活、规格/地址/设置选择、协议勾选、确认提交等前置步骤缺失时，"
            "直接点击最终按钮或下一步入口。"
        ),
        "observable_pattern": (
            "目标按钮存在但灰置、点击无反馈、弹出补全提示，或页面仍停留在需要填写/选择/确认的状态。"
        ),
        "correction_method": (
            "先补齐当前页面明确要求的前置动作：激活输入框、填写必要字段、选择规格或选项、完成登录/授权/确认，"
            "再点击提交、确认、购买、下一步等最终动作。"
        ),
        "writing_boundary": "把重点放在按钮可用前必须完成的界面条件；验证码和应用内键盘输入交给更专门的技能。",
    },
    "captcha_handling": {
        "wrong_actions": (
            "错误动作常把滑块拖到固定终点、重复相同拖动、点错验证码图标顺序，或完成输入后没有点击验证/登录/提交。"
        ),
        "observable_pattern": (
            "界面出现滑块缺口、拼图块、点选提示、验证码输入框、安全验证文案；错误动作后验证仍停留或提示失败。"
        ),
        "correction_method": (
            "按验证码类型选择动作：滑块对齐可见缺口，点选按提示顺序点击目标，输入类验证码完成后点击验证/登录/提交；"
            "根据失败提示重新定位目标而不是机械重复上一次动作。"
        ),
        "writing_boundary": "核心是安全验证流程的识别、精确操作和提交确认；普通业务前置步骤另行处理。",
    },
    "media_type_disambiguation": {
        "wrong_actions": (
            "任务要求图片、相册图、视频或短视频时，错误动作常把带播放入口/时长的视频当图片，"
            "或把静态图片/封面当作需要进入的视频内容。"
        ),
        "observable_pattern": (
            "卡片上有播放三角、时长、视频标签、相册数量、图片网格、封面叠层等媒体类型线索；"
            "进入详情后内容形态与任务要求不一致。"
        ),
        "correction_method": (
            "先用图标、时长、标签和页面标题判断媒体类型，只点击符合任务要求的卡片；"
            "进入错误详情页时返回列表，重新选择类型匹配的目标。"
        ),
        "writing_boundary": "规则应围绕图片/视频类型判别和重新选择，不扩展到直播、商品或广告混淆。",
    },
    "live_product_disambiguation": {
        "wrong_actions": (
            "错误动作常把“直播中/LIVE/主播/观看人数”等直播入口当作普通商品详情，"
            "或在需要直播入口时点到普通商品卡片。"
        ),
        "observable_pattern": (
            "直播卡片通常含直播标识、主播信息、观看人数、直播间封面；商品卡片通常含标题、价格、销量、规格、购物车或购买入口。"
        ),
        "correction_method": (
            "根据任务要求区分直播入口和商品入口：商品任务优先商品标题/价格/购买相关区域，直播任务优先直播标识和直播间入口；"
            "进入错误类型页面时返回并选择另一类入口。"
        ),
        "writing_boundary": "专注直播与商品两类入口的语义区分；广告推广和图片/视频类型由相邻技能处理。",
    },
    "ad_target_disambiguation": {
        "wrong_actions": (
            "错误动作常把会员购、商品推广、续时长、下载/开通/赞助等商业入口当成目标视频、目标功能或真实结果。"
        ),
        "observable_pattern": (
            "相关区域带有会员购、推广、广告、商品、开通、续费、下载、赞助等商业语义；"
            "进入后页面偏向交易、商品或会员服务，与原任务目标不一致。"
        ),
        "correction_method": (
            "先把任务目标拆成真实内容/功能需求，避开商业推广入口；误入推广或交易页时返回原列表/搜索结果，"
            "选择标题、内容类型和上下文更匹配的真实目标。"
        ),
        "writing_boundary": "本类关注内容区内的推广目标误选；前景遮挡弹窗的关闭问题由弹窗技能处理。",
    },
    "custom_keyboard_input": {
        "wrong_actions": (
            "界面使用数字键盘、车牌键盘、银行安全键盘、分段验证码或应用内候选键盘时，错误动作常直接使用 TYPE，"
            "导致输入没有进入控件或字符缺失。"
        ),
        "observable_pattern": (
            "屏幕上出现应用自绘键盘、随机安全键盘、车牌省份/字母/数字面板、分格验证码输入框；"
            "错误动作后输入框内容没有变化或焦点仍等待逐键输入。"
        ),
        "correction_method": (
            "把要输入的字符串拆成字符序列，在屏幕键盘上逐个点击对应按键；"
            "输入后检查每一格或输入框内容，再点击完成、确定、下一步或提交。"
        ),
        "writing_boundary": "核心是应用内键盘必须用屏幕点击完成；普通可编辑文本框仍可使用系统文本输入。",
    },
    "scroll_boundary_awareness": {
        "wrong_actions": (
            "错误动作常在列表、详情页或横向剧集栏已经到达顶部、底部、左端或右端后继续同向滑动，"
            "或保存一项后仍在详情页继续滑动而没有回到列表选择下一项。"
        ),
        "observable_pattern": (
            "连续同向滑动后页面内容几乎不变、边缘项目重复出现、进度已到末端，或当前页面已从列表进入单个详情。"
        ),
        "correction_method": (
            "先判断是否已到滚动边界或处在错误层级；边界处改为反向、返回上一级、点击当前可见目标或切换入口。"
            "完成单个详情动作后需要返回列表再处理下一个目标。"
        ),
        "writing_boundary": "重点是识别边界和页面层级变化；如果只是手势没有命中可滚动区域，应写入滚动区域定位技能。",
    },
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {path}")
            items.append(payload)
    return items


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any, indent: int = 2) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=indent) + "\n")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_image_as_jpeg(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("PIL is required to write template images as JPEG") from exc
    with Image.open(src) as image:
        image.convert("RGB").save(dst, format="JPEG", quality=88, optimize=True)


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(value)


def normalize_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        items = payload["results"]
    elif isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        raise ValueError("input JSON must be an object, list, or object with results")
    return [item for item in items if isinstance(item, dict)]


def parse_task_sample_lines(line_text: str) -> List[int]:
    lines: List[int] = []
    match = re.search(r"覆盖样本：(.+?)行", line_text)
    if not match:
        return lines
    for raw_number in re.findall(r"\d+", match.group(1)):
        lines.append(int(raw_number))
    return lines


def parse_task_case_sections(case_text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    current_task_category = ""
    current_lines: List[str] = []
    for raw_line in case_text.splitlines():
        heading = re.match(r"^#\s+第.+?类任务的场景特点：(.+)$", raw_line.strip())
        if heading:
            if current_task_category:
                sections[current_task_category] = "\n".join(current_lines).strip()
            current_task_category = compact(heading.group(1))
            current_lines = [raw_line]
            continue
        if current_task_category:
            current_lines.append(raw_line)
    if current_task_category:
        sections[current_task_category] = "\n".join(current_lines).strip()
    return sections


def load_task_instruction_mapping(case_task_path: Path, sample_jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    case_text = read_text(case_task_path)
    case_sections = parse_task_case_sections(case_text)
    sample_items = read_jsonl(sample_jsonl_path)
    mapping: Dict[str, Dict[str, Any]] = {}
    current_task_category = ""

    for raw_line in case_text.splitlines():
        line = raw_line.strip()
        heading = re.match(r"^#\s+第.+?类任务的场景特点：(.+)$", line)
        if heading:
            current_task_category = compact(heading.group(1))
            continue
        if not current_task_category or not line.startswith("覆盖样本："):
            continue
        skill_name = TASK_CATEGORY_NAME_TO_SKILL_NAME.get(current_task_category) or safe_folder_name(current_task_category)
        for sample_line in parse_task_sample_lines(line):
            if sample_line < 1 or sample_line > len(sample_items):
                raise ValueError(
                    f"case_task sample line {sample_line} is out of range for {sample_jsonl_path}"
                )
            instruction = compact(sample_items[sample_line - 1].get("instruction"))
            if not instruction:
                raise ValueError(f"sample line {sample_line} has empty instruction in {sample_jsonl_path}")
            existing = mapping.get(instruction)
            if existing and existing.get("task_category_name") != current_task_category:
                raise ValueError(
                    "duplicate task instruction mapped to different task categories: "
                    f"instruction={instruction!r} first={existing.get('task_category_name')} "
                    f"second={current_task_category}"
                )
            mapping[instruction] = {
                "task_category_name": current_task_category,
                "task_skill_name": skill_name,
                "task_sample_line": sample_line,
                "task_category_guidance": case_sections.get(current_task_category, ""),
            }

    if not mapping:
        raise ValueError(f"no task instruction mapping loaded from {case_task_path}")
    return mapping


def apply_task_mapping_to_item(
    item: Dict[str, Any],
    task_instruction_mapping: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    instruction = compact(item.get("instruction"))
    task_info = task_instruction_mapping.get(instruction)
    if not task_info:
        return None
    updated = dict(item)
    updated["error_category_id"] = item.get("category_id")
    updated["error_category_name"] = category_name(item)
    updated["task_category_name"] = task_info["task_category_name"]
    updated["task_skill_name"] = task_info["task_skill_name"]
    updated["task_sample_line"] = task_info["task_sample_line"]
    updated["task_category_guidance"] = task_info.get("task_category_guidance", "")
    return updated


def judgment_of(item: Dict[str, Any]) -> Dict[str, Any]:
    judgment = item.get("judgment")
    return judgment if isinstance(judgment, dict) else {}


def is_better_item(item: Dict[str, Any], require_correct: bool) -> bool:
    judgment = judgment_of(item)
    if judgment.get("comparison_to_original") != "better":
        return False
    if require_correct:
        return judgment.get("is_corrected") is True and judgment.get("step_action_correctness") == "correct"
    return True


def category_name(item: Dict[str, Any]) -> str:
    task_name = compact(item.get("task_category_name"))
    if task_name:
        return task_name
    name = compact(item.get("category_name"))
    if name:
        return name
    category_id = compact(item.get("category_id"))
    return f"category_{category_id}" if category_id else "uncategorized"


def category_id(item: Dict[str, Any]) -> Optional[int]:
    value = item.get("category_id")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def template_image_fields_for_item(item: Dict[str, Any]) -> List[tuple[str, str, str]]:
    return AGENT_TEMPLATE_IMAGE_FIELDS


def resolved_image_fields_for_item(item: Dict[str, Any]) -> List[tuple[str, str]]:
    return AGENT_RESOLVED_IMAGE_FIELDS


def support_image_key_for_item(item: Dict[str, Any]) -> str:
    return "som_image"


def skill_name_for_item(item: Dict[str, Any]) -> str:
    task_skill_name = compact(item.get("task_skill_name"))
    if task_skill_name:
        return task_skill_name
    cat_id = category_id(item)
    if cat_id in CATEGORY_SKILL_NAMES:
        return CATEGORY_SKILL_NAMES[cat_id]
    cat_name = category_name(item)
    if cat_name in CATEGORY_NAME_TO_SKILL_NAME:
        return CATEGORY_NAME_TO_SKILL_NAME[cat_name]
    return safe_folder_name(cat_name)


def image_ssim_fingerprint(raw_path: Any, image_size: int = DEFAULT_SSIM_IMAGE_SIZE) -> List[float]:
    path = resolve_source_path(raw_path)
    if path is None or not path.is_file():
        return []
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return []
    try:
        with Image.open(path) as image:
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
            image = image.convert("L").resize((image_size, image_size), resampling)
            pixel_source = image.get_flattened_data() if hasattr(image, "get_flattened_data") else image.getdata()
            pixels = [float(pixel) for pixel in pixel_source]
    except Exception:
        return []
    return pixels


def ssim_score(
    left: List[float],
    right: List[float],
    image_size: int = DEFAULT_SSIM_IMAGE_SIZE,
    tile_size: int = DEFAULT_SSIM_TILE_SIZE,
) -> Optional[float]:
    if not left or not right or len(left) != len(right) or len(left) != image_size * image_size:
        return None
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    scores: List[float] = []
    for top in range(0, image_size, tile_size):
        for left_pos in range(0, image_size, tile_size):
            values_left: List[float] = []
            values_right: List[float] = []
            for row in range(top, min(top + tile_size, image_size)):
                start = row * image_size + left_pos
                end = start + min(tile_size, image_size - left_pos)
                values_left.extend(left[start:end])
                values_right.extend(right[start:end])
            if not values_left:
                continue
            count = len(values_left)
            mean_left = sum(values_left) / count
            mean_right = sum(values_right) / count
            variance_left = sum((value - mean_left) ** 2 for value in values_left) / count
            variance_right = sum((value - mean_right) ** 2 for value in values_right) / count
            covariance = sum(
                (value_left - mean_left) * (value_right - mean_right)
                for value_left, value_right in zip(values_left, values_right)
            ) / count
            denominator = (mean_left**2 + mean_right**2 + c1) * (variance_left + variance_right + c2)
            if denominator == 0:
                scores.append(1.0)
                continue
            score = ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / denominator
            scores.append(max(-1.0, min(1.0, score)))
    if not scores:
        return None
    return sum(scores) / len(scores)


def image_ssim(left_path: Any, right_path: Any) -> Optional[float]:
    return ssim_score(image_ssim_fingerprint(left_path), image_ssim_fingerprint(right_path))


def numeric_pair(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < 2:
        return None
    first, second = value[0], value[1]
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        return None
    return float(first), float(second)


def bucket_coordinate(value: float, bucket_size: int = DEFAULT_ACTION_COORDINATE_BUCKET_SIZE) -> int:
    return int(round(value / bucket_size) * bucket_size)


def coordinate_bucket(point: Optional[tuple[float, float]], bucket_size: int = DEFAULT_ACTION_COORDINATE_BUCKET_SIZE) -> str:
    if point is None:
        return "unknown"
    return f"x{bucket_coordinate(point[0], bucket_size)}_y{bucket_coordinate(point[1], bucket_size)}"


def action_payload_for_vote(item: Dict[str, Any]) -> Dict[str, Any]:
    repredicted_action = item.get("repredicted_action")
    return repredicted_action if isinstance(repredicted_action, dict) else {}


def canonical_action_for_vote(item: Dict[str, Any]) -> str:
    action = action_payload_for_vote(item)
    if not action:
        return "missing_action"

    action_type = compact(action.get("action") or action.get("action_type")).upper() or "UNKNOWN"
    if action_type in {"CLICK", "TAP"}:
        point = numeric_pair(action.get("point")) or numeric_pair(action.get("coordinate")) or numeric_pair(action.get("pixel"))
        return f"{action_type}:{coordinate_bucket(point)}"
    if action_type == "SWIPE":
        start = numeric_pair(action.get("point1") or action.get("start"))
        end = numeric_pair(action.get("point2") or action.get("end"))
        if start is None or end is None:
            return "SWIPE:unknown"
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        return f"SWIPE:{direction}:{coordinate_bucket(start)}:{coordinate_bucket(end)}"
    if action_type == "TYPE":
        text = compact(action.get("text") or action.get("content") or action.get("value"))
        return f"TYPE:{text[:80]}" if text else "TYPE:unknown"
    return action_type


def sample_quality_score(item: Dict[str, Any]) -> int:
    judgment = judgment_of(item)
    score = 1
    if judgment.get("is_corrected") is True:
        score += 2
    elif judgment.get("is_corrected") is False:
        score -= 1
    correctness = compact(judgment.get("step_action_correctness")).lower()
    if correctness == "correct":
        score += 2
    elif correctness in {"incorrect", "wrong"}:
        score -= 2
    confidence = compact(judgment.get("confidence")).lower()
    if confidence == "high":
        score += 1
    elif confidence == "low":
        score -= 1
    if len(compact(judgment.get("reason"))) >= 20:
        score += 1
    if compact(item.get(support_image_key_for_item(item))):
        score += 1
    return max(0, score)


def sample_source_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(item, ensure_ascii=False))


def vote_action_summary(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    votes: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        canonical_action = compact(sample.get("canonical_action")) or "missing_action"
        quality_score = sample.get("quality_score")
        weight = quality_score if isinstance(quality_score, (int, float)) else 1
        vote = votes.setdefault(
            canonical_action,
            {
                "canonical_action": canonical_action,
                "weight": 0,
                "count": 0,
                "item_indices": [],
            },
        )
        vote["weight"] += max(0, weight)
        vote["count"] += 1
        vote["item_indices"].append(sample.get("item_index"))

    ranked = sorted(votes.values(), key=lambda vote: (-vote["count"], -vote["weight"], vote["canonical_action"]))
    return {
        "winning_action": ranked[0]["canonical_action"] if ranked else "",
        "winning_count": ranked[0]["count"] if ranked else 0,
        "winning_weight": ranked[0]["weight"] if ranked else 0,
        "total_count": sum(vote["count"] for vote in ranked),
        "total_weight": sum(vote["weight"] for vote in ranked),
        "votes": ranked,
    }


def build_sample_signature(item: Dict[str, Any], item_index: int, sample_number: Optional[int] = None) -> Dict[str, Any]:
    support_image_key = support_image_key_for_item(item)
    signature = {
        "item_index": item_index,
        "task_id": compact(item.get("task_id")),
        "wrong_action_step": item.get("wrong_action_step"),
        "variant": compact(item.get("variant")),
        "category_id": item.get("category_id"),
        "category_name": category_name(item),
        "task_category_name": compact(item.get("task_category_name")),
        "task_skill_name": compact(item.get("task_skill_name")),
        "task_sample_line": item.get("task_sample_line"),
        "task_category_guidance": compact(item.get("task_category_guidance")),
        "error_category_id": item.get("error_category_id"),
        "error_category_name": compact(item.get("error_category_name")),
        "instruction": compact(item.get("instruction")),
        "judgment_reason": compact(judgment_of(item).get("reason")),
        "original_image": compact(item.get("original_image")),
        "repredicted_image": compact(item.get("repredicted_image")),
        "som_image": compact(item.get("som_image")),
        "hint_image": compact(item.get("hint_image")),
        "original_action": item.get("original_action") if isinstance(item.get("original_action"), dict) else {},
        "repredicted_action": item.get("repredicted_action") if isinstance(item.get("repredicted_action"), dict) else {},
        "support_image_key": support_image_key,
        "support_image": compact(item.get(support_image_key)),
        "canonical_action": canonical_action_for_vote(item),
        "quality_score": sample_quality_score(item),
        "source_item": sample_source_item(item),
    }
    if sample_number is not None:
        signature["sample_number"] = sample_number
    return signature


def sample_index_path(folder: Path) -> Path:
    return folder / SAMPLE_INDEX_FILENAME


def load_sample_index(folder: Path) -> List[Dict[str, Any]]:
    path = sample_index_path(folder)
    if not path.exists():
        return []
    try:
        payload = read_json(path)
    except Exception as exc:
        raise ValueError(f"failed to read sample index: {path}: {exc}") from exc
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"sample index must be a JSON list: {path}")


def save_sample_index(folder: Path, samples: List[Dict[str, Any]]) -> None:
    write_json(sample_index_path(folder), samples, indent=2)


def sample_skill_image_paths(sample: Dict[str, Any]) -> List[str]:
    image_paths = sample.get("skill_md_image_paths")
    if not isinstance(image_paths, list):
        return []
    return [compact(path) for path in image_paths if compact(path)]


def template_markdown_path_exists(folder: Path, markdown_path: str) -> bool:
    path = compact(markdown_path)
    if not path.startswith("template/"):
        return False
    return (folder / path).is_file()


def prune_deleted_sample_index_entries(
    *,
    folder: Path,
    sample_index: List[Dict[str, Any]],
    existing_skill: Optional[str],
    skill_name: str,
) -> List[Dict[str, Any]]:
    if not sample_index:
        return sample_index

    existing_paths = set(skill_template_image_paths(existing_skill))
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for sample in sample_index:
        image_paths = sample_skill_image_paths(sample)
        if not image_paths:
            kept.append(sample)
            continue
        missing_from_skill = [path for path in image_paths if path not in existing_paths]
        missing_files = [path for path in image_paths if not template_markdown_path_exists(folder, path)]
        if missing_from_skill or missing_files:
            sample["pruned_reason"] = "deleted from skill.md or template directory"
            sample["pruned_missing_from_skill_md"] = missing_from_skill
            sample["pruned_missing_template_files"] = missing_files
            removed.append(sample)
            continue
        kept.append(sample)

    if removed:
        save_sample_index(folder, kept)
        removed_items = ", ".join(
            str(sample.get("item_index")) for sample in removed if sample.get("item_index") is not None
        )
        log(
            f"[prune-sample-index] skill={skill_name} removed={len(removed)} "
            f"remaining={len(kept)} item_indices={removed_items}"
        )
    return kept


def renumber_template_path(markdown_path: str, new_number: int) -> str:
    path = compact(markdown_path)
    match = re.match(
        r"^(template/sample)(?P<number>\d+)(?P<suffix>_[^/\\]+?\.(?:png|jpg|jpeg|webp))$",
        path,
        flags=re.IGNORECASE,
    )
    if not match:
        return path
    return f"{match.group(1)}{new_number}{match.group('suffix')}"


def sample_order_in_skill(existing_skill: str, image_paths: List[str]) -> int:
    positions = [existing_skill.find(path) for path in image_paths if path and path in existing_skill]
    positions = [pos for pos in positions if pos >= 0]
    return min(positions) if positions else len(existing_skill)


def sample_visible_in_skill(existing_skill: str, image_paths: List[str]) -> bool:
    return any(path and path in existing_skill for path in image_paths)


def replace_skill_sample_image_paths_once(existing_skill: str, path_mapping: Dict[str, str]) -> str:
    if not path_mapping:
        return existing_skill
    pattern = re.compile(
        r"template/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+?\.(?:png|jpg|jpeg|webp)",
        flags=re.IGNORECASE,
    )
    return pattern.sub(lambda match: path_mapping.get(match.group(0), match.group(0)), existing_skill)


def replace_skill_sample_headings_once(existing_skill: str, number_mapping: Dict[int, int]) -> str:
    if not number_mapping:
        return existing_skill

    def replace_heading(match: re.Match[str]) -> str:
        old_number = int(match.group("number"))
        new_number = number_mapping.get(old_number, old_number)
        return f"{match.group('prefix')}{new_number}{match.group('suffix')}"

    return re.sub(
        r"^(?P<prefix>###\s+Sample\s+)(?P<number>\d+)(?P<suffix>\b.*)$",
        replace_heading,
        existing_skill,
        flags=re.MULTILINE,
    )


def replace_sample_image_paths(sample: Dict[str, Any], path_mapping: Dict[str, str]) -> None:
    image_paths = sample.get("skill_md_image_paths")
    if isinstance(image_paths, list):
        sample["skill_md_image_paths"] = [path_mapping.get(compact(path), compact(path)) for path in image_paths]

    template_images = sample.get("template_images")
    if not isinstance(template_images, dict):
        return
    for image_info in template_images.values():
        if not isinstance(image_info, dict):
            continue
        old_md_path = compact(image_info.get("skill_md_path"))
        new_md_path = path_mapping.get(old_md_path)
        if not new_md_path:
            continue
        image_info["skill_md_path"] = new_md_path
        template_path = compact(image_info.get("template_path"))
        if template_path:
            image_info["template_path"] = str(Path(template_path).with_name(Path(new_md_path).name))


def rename_template_files(folder: Path, path_mapping: Dict[str, str]) -> None:
    renames = []
    for old_path, new_path in path_mapping.items():
        if old_path == new_path:
            continue
        src = folder / old_path
        dst = folder / new_path
        if not src.is_file():
            continue
        renames.append((src, dst))
    if not renames:
        return

    sources = {src.resolve() for src, _dst in renames}
    for _src, dst in renames:
        if dst.exists() and dst.resolve() not in sources:
            raise RuntimeError(f"cannot renumber sample image; target already exists: {dst}")

    temp_pairs = []
    for index, (src, dst) in enumerate(renames, start=1):
        temp = src.with_name(f".renumber_tmp_{index}_{src.name}")
        src.replace(temp)
        temp_pairs.append((temp, dst))
    for temp, dst in temp_pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        temp.replace(dst)


def renumber_sample_index_entries(
    *,
    folder: Path,
    sample_index: List[Dict[str, Any]],
    existing_skill: Optional[str],
    skill_name: str,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    if not sample_index or existing_skill is None:
        return sample_index, existing_skill

    visible_samples = [
        sample
        for sample in sample_index
        if sample_visible_in_skill(existing_skill, sample_skill_image_paths(sample))
    ]
    if not visible_samples:
        return sample_index, existing_skill

    ordered_samples = sorted(
        visible_samples,
        key=lambda sample: (
            sample_order_in_skill(existing_skill, sample_skill_image_paths(sample)),
            sample.get("sample_number") if isinstance(sample.get("sample_number"), int) else 10**9,
        ),
    )

    path_mapping: Dict[str, str] = {}
    number_mapping: Dict[int, int] = {}
    for new_number, sample in enumerate(ordered_samples, start=1):
        old_number = sample.get("sample_number")
        if isinstance(old_number, int):
            number_mapping[old_number] = new_number
        for old_path in sample_skill_image_paths(sample):
            path_mapping[old_path] = renumber_template_path(old_path, new_number)

    changed_paths = {old: new for old, new in path_mapping.items() if old != new}
    changed_numbers = {old: new for old, new in number_mapping.items() if old != new}
    if not changed_paths and not changed_numbers:
        return sample_index, existing_skill

    rename_template_files(folder, changed_paths)

    updated_skill = replace_skill_sample_image_paths_once(existing_skill, changed_paths)
    updated_skill = replace_skill_sample_headings_once(updated_skill, changed_numbers)

    for sample in ordered_samples:
        old_number = sample.get("sample_number")
        if isinstance(old_number, int) and old_number in number_mapping:
            sample["sample_number"] = number_mapping[old_number]
        replace_sample_image_paths(sample, path_mapping)

    write_text(folder / "skill.md", updated_skill)
    save_sample_index(folder, sample_index)
    log(
        f"[renumber-samples] skill={skill_name} samples={len(ordered_samples)} "
        f"renamed_paths={len(changed_paths)}"
    )
    return sample_index, updated_skill


def preprocess_existing_skills(skills_dir: Path) -> None:
    if not skills_dir.exists():
        return

    for folder in sorted(path for path in skills_dir.iterdir() if path.is_dir()):
        skill_path = folder / "skill.md"
        existing_skill = read_text(skill_path) if skill_path.exists() else None
        sample_index = prune_deleted_sample_index_entries(
            folder=folder,
            sample_index=load_sample_index(folder),
            existing_skill=existing_skill,
            skill_name=folder.name,
        )
        renumber_sample_index_entries(
            folder=folder,
            sample_index=sample_index,
            existing_skill=existing_skill,
            skill_name=folder.name,
        )


def next_visible_sample_number(sample_index: List[Dict[str, Any]]) -> int:
    numbers: List[int] = []
    for sample in sample_index:
        if not sample_skill_image_paths(sample):
            continue
        sample_number = sample.get("sample_number")
        if isinstance(sample_number, int):
            numbers.append(sample_number)
    return (max(numbers) + 1) if numbers else 1


def image_similarity(candidate: Dict[str, Any], existing: Dict[str, Any]) -> Dict[str, Any]:
    original_ssim = image_ssim(candidate.get("original_image"), existing.get("original_image"))
    return {
        "existing_item_index": existing.get("item_index"),
        "existing_task_id": existing.get("task_id"),
        "existing_variant": existing.get("variant"),
        "original_image_ssim": original_ssim,
    }


def duplicate_reason(similarity: Dict[str, Any], image_ssim_threshold: float) -> str:
    original_ssim = similarity.get("original_image_ssim")
    if original_ssim is not None and original_ssim >= image_ssim_threshold:
        return "similar original image ssim"
    return ""


def find_similar_sample(
    candidate: Dict[str, Any],
    existing_samples: List[Dict[str, Any]],
    image_ssim_threshold: float,
) -> Optional[Dict[str, Any]]:
    best_match: Optional[Dict[str, Any]] = None
    for existing in existing_samples:
        similarity = image_similarity(candidate, existing)
        reason = duplicate_reason(similarity, image_ssim_threshold)
        if not reason:
            continue
        match = {
            "reason": reason,
            "similarity": similarity,
        }
        if best_match is None:
            best_match = match
            continue
        current_ssim = similarity.get("original_image_ssim")
        best_ssim = best_match["similarity"].get("original_image_ssim")
        if current_ssim is not None and (best_ssim is None or current_ssim > best_ssim):
            best_match = match
    return best_match


def find_similar_samples(
    candidate: Dict[str, Any],
    existing_samples: List[Dict[str, Any]],
    image_ssim_threshold: float,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for existing in existing_samples:
        similarity = image_similarity(candidate, existing)
        reason = duplicate_reason(similarity, image_ssim_threshold)
        if not reason:
            continue
        matches.append(
            {
                "sample": existing,
                "reason": reason,
                "similarity": similarity,
            }
        )
    return sorted(
        matches,
        key=lambda match: match["similarity"].get("original_image_ssim") or 0,
        reverse=True,
    )


def attach_cluster_vote(
    sample: Dict[str, Any],
    similar_matches: List[Dict[str, Any]],
) -> None:
    cluster_samples = [match["sample"] for match in similar_matches] + [sample]
    vote_summary = vote_action_summary(cluster_samples)
    sample["similar_cluster"] = {
        "size": len(cluster_samples),
        "matched_item_indices": [match["sample"].get("item_index") for match in similar_matches],
        "best_original_image_ssim": (
            similar_matches[0]["similarity"].get("original_image_ssim") if similar_matches else None
        ),
        "vote": vote_summary,
    }


def winning_samples(cluster_samples: List[Dict[str, Any]], vote_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    winning_action = compact(vote_summary.get("winning_action"))
    if not winning_action:
        return []
    return [sample for sample in cluster_samples if compact(sample.get("canonical_action")) == winning_action]


def best_vote_representative(cluster_samples: List[Dict[str, Any]], vote_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    winners = winning_samples(cluster_samples, vote_summary)
    if not winners:
        return None
    return sorted(
        winners,
        key=lambda sample: (
            sample.get("quality_score") if isinstance(sample.get("quality_score"), (int, float)) else 0,
            1 if sample_skill_image_paths(sample) else 0,
            -(sample.get("item_index") if isinstance(sample.get("item_index"), int) else 0),
        ),
        reverse=True,
    )[0]


def numeric_quality_score(sample: Dict[str, Any]) -> float:
    quality_score = sample.get("quality_score")
    return float(quality_score) if isinstance(quality_score, (int, float)) else 0.0


def item_from_sample_signature(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source_item = sample.get("source_item")
    if isinstance(source_item, dict):
        return dict(source_item)

    reconstructed: Dict[str, Any] = {}
    for key in (
        "task_id",
        "wrong_action_step",
        "variant",
        "category_id",
        "category_name",
        "task_category_name",
        "task_skill_name",
        "task_sample_line",
        "task_category_guidance",
        "error_category_id",
        "error_category_name",
        "instruction",
        "original_image",
        "repredicted_image",
        "som_image",
        "hint_image",
        "original_action",
        "repredicted_action",
    ):
        value = sample.get(key)
        if value not in (None, ""):
            reconstructed[key] = value
    judgment_reason = compact(sample.get("judgment_reason"))
    if judgment_reason:
        reconstructed["judgment"] = {
            "comparison_to_original": "better",
            "reason": judgment_reason,
        }
    canonical_action = compact(sample.get("canonical_action"))
    if canonical_action:
        reconstructed["vote_canonical_action"] = canonical_action
    required_image_keys = [key for key, _kind, _label in template_image_fields_for_item(reconstructed)]
    if any(not compact(reconstructed.get(key)) for key in required_image_keys):
        return None
    if not isinstance(reconstructed.get("repredicted_action"), dict):
        return None
    return reconstructed if reconstructed else None


def cluster_vote_note(sample: Dict[str, Any]) -> Dict[str, Any]:
    cluster = sample.get("similar_cluster")
    return cluster if isinstance(cluster, dict) else {}


def cluster_sample_image_paths(similar_matches: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for match in similar_matches:
        paths.extend(sample_skill_image_paths(match["sample"]))
    return sorted(set(paths))


def reusable_template_path_by_kind(paths: List[str]) -> Dict[str, str]:
    reusable: Dict[str, str] = {}
    for path in paths:
        match = re.search(r"_(orig|som|hint)\.(?:png|jpg|jpeg|webp)\Z", path, flags=re.IGNORECASE)
        if match:
            reusable.setdefault(match.group(1).lower(), path)
    return reusable


def retained_sample_index_image_paths(
    sample_index: List[Dict[str, Any]],
    excluded_paths: List[str],
) -> List[str]:
    excluded = set(excluded_paths)
    return sorted(path for path in sample_index_skill_image_paths(sample_index) if path not in excluded)


def clear_replaced_sample_images(
    similar_matches: List[Dict[str, Any]],
    replaced_paths: List[str],
    representative_signature: Dict[str, Any],
) -> None:
    if not replaced_paths:
        return
    replaced = set(replaced_paths)
    for match in similar_matches:
        sample = match["sample"]
        if sample is representative_signature:
            continue
        current_paths = sample_skill_image_paths(sample)
        if not current_paths or not any(path in replaced for path in current_paths):
            continue
        sample["replaced_by_vote_representative_item_index"] = representative_signature.get("item_index")
        sample["replaced_skill_md_image_paths"] = current_paths
        sample.pop("skill_md_image_paths", None)
        sample.pop("template_images", None)
        sample.pop("sample_number", None)


def safe_folder_name(raw: str) -> str:
    name = compact(raw)
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    name = name.rstrip(" .")
    return name or "uncategorized"


def safe_file_stem(raw: str) -> str:
    stem = safe_folder_name(raw)
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"_+", "_", stem)
    return stem.strip("_") or "sample"


def sample_image_stem(item: Dict[str, Any], sample_number: int) -> str:
    return safe_file_stem(f"sample{sample_number}")


def expected_template_image_paths(item: Dict[str, Any], sample_number: int, folder: Path) -> List[str]:
    stem = sample_image_stem(item, sample_number)
    paths: List[str] = []
    for key, image_kind, _label in template_image_fields_for_item(item):
        source_path = resolve_source_path(item.get(key))
        if not source_path:
            continue
        target = folder / "template" / f"{stem}_{image_kind}.jpg"
        paths.append(path_for_markdown(target, folder))
    return paths


def existing_skill_contains_candidate_images(existing_skill: Optional[str], expected_paths: List[str]) -> bool:
    if not existing_skill or not expected_paths:
        return False
    return all(path in existing_skill for path in expected_paths)


def skill_template_image_paths(skill_md: Optional[str]) -> List[str]:
    if not skill_md:
        return []
    pattern = r"template/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+?\.(?:png|jpg|jpeg|webp)"
    return sorted(set(re.findall(pattern, skill_md, flags=re.IGNORECASE)))


def sample_index_skill_image_paths(sample_index: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for sample in sample_index:
        image_paths = sample.get("skill_md_image_paths")
        if not isinstance(image_paths, list):
            continue
        paths.extend(compact(path) for path in image_paths if compact(path))
    return sorted(set(paths))


def copied_skill_image_paths(copied_images: Dict[str, Dict[str, str]]) -> List[str]:
    return sorted(
        set(
            image_info.get("skill_md_path") or ""
            for image_info in copied_images.values()
            if image_info.get("skill_md_path")
        )
    )


def missing_required_skill_image_paths(skill_md: str, required_paths: List[str]) -> List[str]:
    return [path for path in sorted(set(required_paths)) if path and path not in skill_md]


def path_for_markdown(path: Path, base_dir: Path) -> str:
    try:
        relative = path.resolve().relative_to(base_dir.resolve())
        return relative.as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_source_path(raw_path: Any) -> Optional[Path]:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    return Path(raw_path)


def copy_template_images_for_item(
    *,
    item: Dict[str, Any],
    sample_number: int,
    folder: Path,
    template_dir: Path,
) -> Dict[str, Dict[str, str]]:
    copied: Dict[str, Dict[str, str]] = {}
    stem = sample_image_stem(item, sample_number)

    for key, image_kind, _label in template_image_fields_for_item(item):
        raw_path = item.get(key)
        source_path = resolve_source_path(raw_path)
        entry = {
            "raw_path": compact(raw_path),
            "source_path": str(source_path) if source_path else "",
            "template_path": "",
            "skill_md_path": "",
            "source_path_provided": "true" if source_path else "false",
        }
        if source_path:
            target = template_dir / f"{stem}_{image_kind}.jpg"
            copy_image_as_jpeg(source_path, target)
            entry["template_path"] = str(target)
            entry["skill_md_path"] = path_for_markdown(target, folder)
        copied[key] = entry

    return copied


def image_entries_for_item(item: Dict[str, Any]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for key, label in resolved_image_fields_for_item(item):
        raw_path = item.get(key)
        source_path = resolve_source_path(raw_path)
        entries.append(
            {
                "key": key,
                "label": label,
                "raw_path": compact(raw_path),
                "skill_md_path": compact(item.get(f"{key}_skill_md_path")),
                "template_path": compact(item.get(f"{key}_template_path")),
                "source_path": str(source_path) if source_path else "",
                "source_path_provided": "true" if source_path else "false",
            }
        )
    return entries


def serializable_instance(item: Dict[str, Any]) -> Dict[str, Any]:
    copied = dict(item)
    copied["resolved_images"] = image_entries_for_item(item)
    return copied


def instance_with_template_images(
    item: Dict[str, Any],
    copied_images: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    enriched = dict(item)
    enriched["template_images"] = copied_images
    for key, image_info in copied_images.items():
        skill_md_path = image_info.get("skill_md_path") or ""
        template_path = image_info.get("template_path") or ""
        if skill_md_path:
            enriched[f"{key}_skill_md_path"] = skill_md_path
            enriched[f"{key}_template_path"] = template_path
    return enriched


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def build_create_system_prompt() -> str:
    return (
        "你负责从 MobileGym 多模态 better 重预测实例中新建一个 skill.md。"
        "只能使用用户提供的 better 实例以及 skillsample.md 的结构。"
        "你的目标不是复述样例，而是从原错误动作与 better 改正动作中抽象出可迁移的 GUI 操作技能。"
        "必须区分稳定可迁移规则与当前截图特有细节；不要编造实例不支持的规则。"
        "只输出完整的 skill.md Markdown，不要输出额外解释。"
    )


def build_update_system_prompt() -> str:
    return (
        "你负责根据 MobileGym 多模态 better 重预测实例迭代更新已有 skill.md。"
        "只能使用用户提供的 better 实例、已有 skill.md 以及 skillsample.md 的结构。"
        "保留已有 skill 中仍然正确且被实例支持的内容，修订或补充当前实例证明的新经验。"
        "你的目标不是堆叠样例描述，而是维护一份更准确、更可迁移、更少误召回的技能文档。"
        "必须收窄过宽规则、删除无证据规则、合并重复规则；不要编造实例不支持的规则。"
        "只输出更新后的完整 skill.md Markdown，不要输出额外解释。"
    )


def build_text_only_system_prompt() -> str:
    return (
        "你负责根据用户提供的 MobileGym better 重预测信息生成纯文本类型的 skill.md。"
        "你的目标是从原错误动作与 better 改正动作中抽象出可迁移的 GUI 操作技能。"
        "必须区分稳定可迁移规则与单个实例特有细节；不要编造提供信息中没有支撑的规则。"
        "最终回答直接给出完整的 skill.md Markdown 正文。"
    )


def sample_requirements_for_prompt() -> str:
    retention_requirement = (
        "Multimodal Samples 是完整证据索引，不参与去重压缩。"
        "更新时必须保留 required_sample_image_paths 中列出的所有 template 图片路径。"
        "如果当前实例 JSON 中没有 replaced_sample_image_paths，则追加当前 better 实例；不得删除、改名、合并已有 sample。"
        "但如果当前实例 JSON 中提供了 replaced_sample_image_paths，表示投票赢家正在替换相似簇代表样本，"
        "这些旧路径必须从 Multimodal Samples 中移除，并改用投票赢家实例的 template 图片路径。"
        "如果 prompt 中提供了 required_sample_image_paths，输出的 skill.md 必须逐字包含其中每一个路径。"
        "“合并重复规则”和“压缩表达”只适用于 Overview、Iron Law、When to Use、Method，不适用于 Multimodal Samples。\n"
    )
    return (
        retention_requirement
        +
        "Multimodal Samples 必须吸收当前 better 实例。每个 sample 至少包含："
        "原错误轨迹中错误步骤的屏幕截图路径、原错误步骤的具体 action、"
        "重预测改正后的 action、改正后截图路径、改正后 SOM 截图路径、judgment.reason。\n"
        "写入 skill.md 的截图路径必须使用当前实例 JSON 中的 original_image_skill_md_path "
        "和 som_image_skill_md_path；这些路径指向当前 skill 文件夹下的 template 图片。"
        "不要把 /opt/nas/... 原始路径写入 skill.md。"
    )


def skill_contract_for_prompt(skill_name: str) -> str:
    contract = SKILL_CONTRACTS.get(skill_name)
    if not contract:
        return "无额外 skill 边界合同；仍必须严格遵守实例证据和相邻场景的保守边界。"
    return "\n".join(
        [
            f"适用范围：{contract['scope']}",
            f"明确排除：{contract['out_of_scope']}",
            f"相邻技能边界：{contract['neighbor_boundary']}",
        ]
    )


def skill_diagnosis_prior_for_prompt(skill_name: str) -> str:
    prior = CATEGORY_DIAGNOSIS_PRIORS.get(skill_name)
    if not prior:
        return "无额外类别经验；仍应从当前 better 实例中提炼可执行的动作纠错方法。"
    return "\n".join(
        [
            f"常见错误动作：{prior['wrong_actions']}",
            f"运行时可观察的界面和历史现象：{prior['observable_pattern']}",
            f"skill.md 应提炼的纠错方法：{prior['correction_method']}",
            f"写作边界：{prior['writing_boundary']}",
        ]
    )


def is_task_skill_name(skill_name: str) -> bool:
    return skill_name.startswith("task_")


def category_prompt_label(skill_name: str) -> str:
    return "task_category_name，用于说明 GUI 任务类型语义" if is_task_skill_name(skill_name) else "category_name，用于说明错误类别语义"


def generation_task_title(skill_name: str, mode: str) -> str:
    target = "task category" if is_task_skill_name(skill_name) else "category"
    action = "新建生成" if mode == "create" else "更新迭代已有"
    return f"请为下面这个 {target} {action}一个 MobileGym skill.md。"


def skill_generation_guidelines_for_prompt() -> str:
    shared = """
写作目标与抽象粒度：
- skill.md 面向运行时 GUI agent 使用，应帮助 agent 在新界面上识别同类错误并选择正确动作，而不是记住当前样例。
- 所有规则都必须能从 better 实例、judgment.reason、原错误 action、改正 action 和截图/辅助图中得到支撑。
- 抽象时保留可迁移的界面现象、任务目标关系、操作约束和纠错策略；不要把样例编号、任务 id、绝对路径、具体坐标、具体商品/视频/账号/文案当作通用规则。
- 如果只有一个实例，只能写保守规则：说明它支持的最小充分条件，不要扩展到未被证明的场景。
- 类别诊断经验用于帮助正文更像运行时动作纠错手册；应落到截图可观察证据、动作选择、结果校验和边界处理。

章节写作要求：
- frontmatter: 必须包含 Name、Description。Name 使用 skill_name；Description 用一句话写清适用条件和核心用途，不要写成分类名复述。
- Overview: 写该 skill 解决的失败模式、原错误为什么错、better 动作为什么更可靠，控制在 1 到 3 段。
- Iron Law: 写 3 到 6 条硬约束，每条必须是执行时可检查的规则。优先写禁止误点、禁止越界、禁止跳转到错误目标、必须先确认目标语义等约束。
- When to Use: 必须分成“应该使用”和“不应该使用”两组。每组写界面信号、任务条件、历史动作/失败迹象。不要只写类别名，也不要只写“出现类似情况”。
- Method: 按“观察界面 -> 定位目标/风险 -> 选择动作 -> 校验动作结果 -> 边界处理”的顺序写步骤。每一步说明判断依据和动作选择原则。
- Multimodal Samples: 只记录样例事实和从该样例得到的可迁移 reason；样例段落不能替代 Overview/When to Use/Method 的抽象总结。

禁止事项：
- 不要输出代码块围栏、额外解释、分析前言或“以下是”。
- 不要编造截图中没有的 UI 元素、OCR 文本、按钮位置或任务要求。
- 不要把 better 实例里的单个动作直接提升为无条件规则；必须写清触发条件。
- 不要写“总是点击某坐标”“固定向上滑动”“固定关闭某按钮”这类依赖单个截图布局的规则。
- 不要因为 skill_name 或 category_name 看起来相关就扩大适用范围；适用条件必须能被运行时截图/历史动作观察到。
""".strip()
    specific = """
动作纠正技能额外要求：
- 重点抽象原错误 action 为什么失败，以及 better action 如何修正这个失败。
- Method 中必须说明如何从当前截图重新定位真实可操作目标，不能复制样例坐标。
- When to Use 应同时利用界面现象和历史动作效果，例如点击后无变化、滑动区域错误、控件语义误解、前置步骤缺失等。
- When Not to Use 应写清该技能可能误伤的相邻场景，例如目标已可直接操作、弹层/控件不是阻塞因素、当前任务目标与样例不同等。
""".strip()
    return shared + "\n\n" + specific


def build_create_user_text(
    *,
    skill_name: str,
    category: str,
    sample_md: str,
    instances: List[Dict[str, Any]],
) -> str:
    sample_requirements = sample_requirements_for_prompt()
    generation_guidelines = skill_generation_guidelines_for_prompt()
    skill_contract = skill_contract_for_prompt(skill_name)
    skill_diagnosis_prior = skill_diagnosis_prior_for_prompt(skill_name)
    category_label = category_prompt_label(skill_name)
    task_title = generation_task_title(skill_name, "create")
    payload = {
        "skill_name": skill_name,
        "category_name": category,
        "category_mode": "task" if is_task_skill_name(skill_name) else "error",
        "skill_contract": skill_contract,
        "mode": "create",
        "better_instance_count": len(instances),
        "better_instances": [serializable_instance(item) for item in instances],
    }

    return f"""
任务：{task_title}

{category_label}：
{category}

skill_name，将作为 skill 文件夹名和建议技能名：
{skill_name}

skill 边界合同，优先级高于单个实例中的偶然现象；Overview、Iron Law、When to Use、Method 和 Multimodal Samples 都不得违反：
{skill_contract}

类别诊断经验，用于提升 skill.md 在运行时的动作纠错质量；Overview、When to Use、Method 和 Iron Law 应吸收其中与当前实例一致的经验：
{skill_diagnosis_prior}

硬性要求：
1. 只能使用 judgment.comparison_to_original == "better" 的实例总结技能。
2. 当前 category 就是上面这个类名；该类尚不存在 skill.md，本次只做新建，不要写成更新说明。
3. 输出必须是一个完整的 skill.md，结构参考 skillsample.md。
4. skill.md 必须包含 frontmatter、Overview、Iron Law、When to Use、Method、Multimodal Samples。
5. 必须严格遵守下方“skill.md 生成细则”，尤其是抽象粒度、章节写作要求和禁止事项。
6. frontmatter 的 Name 和正文标题优先使用 skill_name，不要使用冗长的中文 category_name 作为技能名。
7. 必须把 skill 边界合同转写为可执行的 When to Use / When Not to Use / Iron Law 规则；不能只在 sample 中提到。
8. 必须把类别诊断经验转写为运行时 agent 能根据当前截图、任务和历史动作使用的判断规则与动作策略；不能只在 sample 中提到。
9. {sample_requirements}
10. 不要输出代码块围栏，不要输出额外解释，只输出 skill.md 正文。

skill.md 生成细则：
{generation_guidelines}

skillsample.md:
{sample_md}

当前 better 实例完整 JSON：
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def build_update_user_text(
    *,
    skill_name: str,
    category: str,
    sample_md: str,
    existing_skill: str,
    instances: List[Dict[str, Any]],
    required_sample_image_paths: Optional[List[str]] = None,
) -> str:
    sample_requirements = sample_requirements_for_prompt()
    generation_guidelines = skill_generation_guidelines_for_prompt()
    skill_contract = skill_contract_for_prompt(skill_name)
    skill_diagnosis_prior = skill_diagnosis_prior_for_prompt(skill_name)
    category_label = category_prompt_label(skill_name)
    task_title = generation_task_title(skill_name, "update")
    payload = {
        "skill_name": skill_name,
        "category_name": category,
        "category_mode": "task" if is_task_skill_name(skill_name) else "error",
        "skill_contract": skill_contract,
        "mode": "update",
        "better_instance_count": len(instances),
        "better_instances": [serializable_instance(item) for item in instances],
        "required_sample_image_paths": sorted(set(required_sample_image_paths or [])),
    }

    return f"""
任务：{task_title}

{category_label}：
{category}

skill_name，将作为 skill 文件夹名和建议技能名：
{skill_name}

skill 边界合同，优先级高于已有 skill.md 和单个实例中的偶然现象；Overview、Iron Law、When to Use、Method 和 Multimodal Samples 都不得违反：
{skill_contract}

类别诊断经验，用于提升 skill.md 在运行时的动作纠错质量；Overview、When to Use、Method 和 Iron Law 应吸收其中与当前实例一致的经验：
{skill_diagnosis_prior}

硬性要求：
1. 只能使用 judgment.comparison_to_original == "better" 的实例总结或修订技能。
2. 当前 category 就是上面这个类名；本次只做已有 skill.md 的更新迭代，不要重新从零生成一个无视旧内容的 skill。
3. 输出必须是更新后的完整 skill.md，结构参考 skillsample.md。
4. skill.md 必须包含 frontmatter、Overview、Iron Law、When to Use、Method、Multimodal Samples。
5. 保留已有 skill 中仍然正确、仍被已有内容或当前 better 实例支持的规则；删除、收窄或改写缺少实例支撑、与当前实例冲突或过度泛化的规则。
6. frontmatter 的 Name 和正文标题优先使用 skill_name，不要使用冗长的中文 category_name 作为技能名。
7. 必须严格遵守下方“skill.md 更新细则”，尤其是抽象粒度、章节写作要求和禁止事项。
8. 必须把 skill 边界合同转写为可执行的 When to Use / When Not to Use / Iron Law 规则；已有 skill.md 中违反合同的内容必须删除或改写。
9. 必须把类别诊断经验转写为运行时 agent 能根据当前截图、任务和历史动作使用的判断规则与动作策略；不能只在 sample 中提到。
10. {sample_requirements}
11. 输出前自检：当前实例的 template 图片路径、已有 skill.md 中的 template/sample* 图片引用、以及当前 better 实例完整 JSON 里的 required_sample_image_paths 必须全部保留在 Multimodal Samples 中。
12. 不要输出代码块围栏，不要输出额外解释，只输出 skill.md 正文。

skill.md 更新细则：
{generation_guidelines}

更新策略：
- 如果已有规则仍被已有 skill 内容或当前 better 实例支持，保留并可压缩表达。
- 如果已有规则与当前 better 实例冲突，必须改写为更窄、更准确的条件。
- 如果当前 better 实例只是已有规则的重复证据，不要在 Overview、Iron Law、When to Use、Method 中重复堆叠；只把它加入 Multimodal Samples。
- 如果当前 better 实例提供了新的边界条件或反例，优先补充到 When to Use 的“不应该使用”部分或 Method 的边界处理部分。
- Multimodal Samples 不执行普通去重压缩；没有 replaced_sample_image_paths 时，即使样例机制重复，也必须保留所有历史 sample 并追加当前 sample。
- 如果当前 better 实例 JSON 提供 replaced_sample_image_paths，表示当前样本属于相似截图投票簇；必须删除这些旧路径对应的相似簇代表 sample，并用投票赢家实例及其 template 图片路径替换，避免同一相似簇保留错误代表。

skillsample.md:
{sample_md}

已有 skill.md：
{existing_skill}

当前 better 实例完整 JSON：
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def build_text_only_user_text(
    *,
    skill_name: str,
    category: str,
    sample_md: str,
    existing_skill: Optional[str],
    instances: List[Dict[str, Any]],
) -> str:
    category_label = category_prompt_label(skill_name)
    payload = {
        "skill_name": skill_name,
        "category_name": category,
        "category_mode": "task" if is_task_skill_name(skill_name) else "error",
        "mode": "text_only_update" if existing_skill is not None else "text_only_create",
        "better_instance_count": len(instances),
        "better_instances": [serializable_instance(item) for item in instances],
    }
    existing_block = f"\n已有 skill.md：\n{existing_skill}\n" if existing_skill is not None else ""

    return f"""
任务：请根据下面提供的信息生成一个纯文本类型的 MobileGym skill.md。

{category_label}：
{category}

skill_name，将作为 skill 文件夹名和建议技能名：
{skill_name}
{existing_block}
硬性要求：
1. 只能使用 judgment.comparison_to_original == "better" 的实例总结或修订技能。
2. 这是纯文本 skill 生成任务，只根据本 prompt 中提供的信息生成。
3. 输出必须是一个完整的 skill.md Markdown，内容以自然语言规则、判断条件和执行方法为主。
4. skill.md 面向运行时 GUI agent 使用，应帮助 agent 在新界面上识别同类错误并选择正确动作。
5. skill.md 必须包含 frontmatter、Overview、Iron Law、When to Use、Method。
6. frontmatter 必须包含 Name、Description。Name 使用 skill_name；Description 用一句话写清适用条件和核心用途。
7. Overview 写失败模式、原错误为什么错、better 动作为什么更可靠，控制在 1 到 3 段。
8. Iron Law 写 3 到 6 条执行时可检查的硬约束。
9. When to Use 必须分成“应该使用”和“不应该使用”两组，写清界面信号、任务条件、历史动作/失败迹象。
10. Method 按“观察界面 -> 定位目标/风险 -> 选择动作 -> 校验动作结果 -> 边界处理”的顺序写步骤。
11. 如果已有 skill.md 存在，保留仍被已有内容或当前 better 实例支持的规则，删除、收窄或改写缺少证据、与当前实例冲突或过度泛化的规则。
12. 最终回答直接输出 skill.md 正文。

skillsample_text.md:
{sample_md}

当前 better 实例完整 JSON：
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def build_messages(
    *,
    skill_name: str,
    category: str,
    sample_md: str,
    existing_skill: Optional[str],
    instances: List[Dict[str, Any]],
    required_sample_image_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if existing_skill is None:
        system_prompt = build_create_system_prompt()
        user_text = build_create_user_text(
            skill_name=skill_name,
            category=category,
            sample_md=sample_md,
            instances=instances,
        )
    else:
        system_prompt = build_update_system_prompt()
        user_text = build_update_user_text(
            skill_name=skill_name,
            category=category,
            sample_md=sample_md,
            existing_skill=existing_skill,
            instances=instances,
            required_sample_image_paths=required_sample_image_paths,
        )

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": user_text,
        }
    ]

    for index, item in enumerate(instances, start=1):
        for image in image_entries_for_item(item):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"实例 {index} 图片：{image['label']}\n"
                        f"JSON 字段: {image['key']}\n"
                        f"原始路径: {image['raw_path']}\n"
                        f"skill.md 应写入路径: {image['skill_md_path'] or '无 template 路径'}\n"
                        f"源图路径: {image['source_path'] or '未提供源图路径'}"
                    ),
                }
            )
            if image["source_path"]:
                content.append({"type": "image_url", "image_url": {"url": image["source_path"]}})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def build_text_only_messages(
    *,
    skill_name: str,
    category: str,
    sample_md: str,
    existing_skill: Optional[str],
    instances: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": build_text_only_system_prompt()},
        {
            "role": "user",
            "content": build_text_only_user_text(
                skill_name=skill_name,
                category=category,
                sample_md=sample_md,
                existing_skill=existing_skill,
                instances=instances,
            ),
        },
    ]


def build_wrapper(args: argparse.Namespace) -> Any:
    from api import ApiWrapper  # type: ignore

    if args.skill_llm == "doubao":
        return ApiWrapper(
            model_name="DouBao-Seed-2.0-Pro",
            timeout=args.timeout,
            max_retries=args.max_retries,
            debug=args.debug,
        )
    if args.skill_llm == "qwen":
        return ApiWrapper(
            model_name="Qwen3.6-Plus",
            api_key="",
            timeout=args.timeout,
            max_retries=args.max_retries,
            debug=args.debug,
            max_tokens=args.max_tokens,
        )
    return ApiWrapper(
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        model_name=args.model_name,
        base_url=args.model_base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        debug=args.debug,
        max_tokens=args.max_tokens,
        verify=not args.no_verify_false,
        trust_env=False,
    )


def debug_message_stem(skill_name: str, item: Dict[str, Any], item_index: int) -> str:
    parts = [
        f"{item_index:06d}",
        safe_folder_name(skill_name),
        compact(item.get("task_id")),
        f"step_{compact(item.get('wrong_action_step'))}",
        compact(item.get("variant")),
    ]
    return "__".join(safe_folder_name(part) for part in parts if part)


def save_debug_messages(
    messages_dir: Path,
    skill_name: str,
    item: Dict[str, Any],
    item_index: int,
    messages: List[Dict[str, Any]],
) -> None:
    debug_path = messages_dir / f"{debug_message_stem(skill_name, item, item_index)}.json"
    write_text(debug_path, json.dumps(messages, ensure_ascii=False, indent=2))


def update_skill_for_instance(
    *,
    item: Dict[str, Any],
    item_index: int,
    skills_dir: Path,
    messages_dir: Path,
    sample_md: str,
    save_messages: bool,
    debug: bool,
    wrapper: Any,
    image_ssim_threshold: float,
) -> bool:
    category = category_name(item)
    skill_name = skill_name_for_item(item)
    folder = skills_dir / safe_folder_name(skill_name)
    template_dir = folder / "template"
    skill_path = folder / "skill.md"
    existing_skill = read_text(skill_path) if skill_path.exists() else None
    sample_index = load_sample_index(folder)
    sample_number = next_visible_sample_number(sample_index)
    sample_signature = build_sample_signature(item, item_index, sample_number=sample_number)
    replaced_cluster_image_paths: List[str] = []
    similar_matches: List[Dict[str, Any]] = []
    representative_item = item
    representative_signature = sample_signature
    expected_paths = expected_template_image_paths(item, sample_number, folder)
    if existing_skill_contains_candidate_images(existing_skill, expected_paths):
        sample_signature["dedupe_bootstrap"] = "existing skill.md already contains expected template images"
        sample_signature["skill_md_image_paths"] = expected_paths
        sample_index.append(sample_signature)
        save_sample_index(folder, sample_index)
        log(f"[skip-similar] item_index={item_index} skill={skill_name} reason=already present in skill.md images")
        return False

    similar_matches = find_similar_samples(
        sample_signature,
        sample_index,
        image_ssim_threshold=image_ssim_threshold,
    )
    if similar_matches:
        attach_cluster_vote(sample_signature, similar_matches)
        similar = similar_matches[0]
        similarity = similar["similarity"]
        sample_signature["dedupe_reason"] = similar["reason"]
        sample_signature["dedupe_matched_item_index"] = similarity.get("existing_item_index")
        sample_signature["dedupe_original_image_ssim"] = similarity.get("original_image_ssim")
        cluster_samples = [match["sample"] for match in similar_matches] + [sample_signature]
        vote = sample_signature["similar_cluster"]["vote"]
        representative = best_vote_representative(cluster_samples, vote)
        if representative is None:
            sample_signature["dedupe_outcome"] = "skipped_no_vote_representative"
            sample_signature.pop("sample_number", None)
            sample_index.append(sample_signature)
            save_sample_index(folder, sample_index)
            log(
                "[skip-similar] "
                f"item_index={item_index} skill={skill_name} reason={similar['reason']} "
                f"existing_item_index={similarity.get('existing_item_index')} "
                f"original_image_ssim={similarity.get('original_image_ssim')} "
                f"winning_action={vote['winning_action']} "
                f"cluster_size={sample_signature['similar_cluster']['size']}"
            )
            return False

        representative_signature = representative
        representative_item = item if representative is sample_signature else (item_from_sample_signature(representative) or {})
        if not representative_item:
            sample_signature["dedupe_outcome"] = "skipped_unrecoverable_vote_representative"
            sample_signature["vote_representative_item_index"] = representative.get("item_index")
            sample_signature.pop("sample_number", None)
            sample_index.append(sample_signature)
            save_sample_index(folder, sample_index)
            log(
                "[skip-similar] "
                f"item_index={item_index} skill={skill_name} reason=unrecoverable vote representative "
                f"representative_item_index={representative.get('item_index')} "
                f"winning_action={vote['winning_action']} "
                f"cluster_size={sample_signature['similar_cluster']['size']}"
            )
            return False

        previous_representatives = [match["sample"] for match in similar_matches if sample_skill_image_paths(match["sample"])]
        if previous_representatives:
            sample_number = previous_representatives[0].get("sample_number") or sample_number
        representative_signature["sample_number"] = sample_number
        replaced_cluster_image_paths = cluster_sample_image_paths(similar_matches)
        sample_signature["dedupe_outcome"] = (
            "update_from_current_vote_representative"
            if representative is sample_signature
            else "recorded_historical_vote_representative"
        )
        sample_signature["vote_representative_item_index"] = representative.get("item_index")
        representative_signature["dedupe_outcome"] = "update_from_vote_representative"
        representative_signature["vote_representative_item_index"] = representative.get("item_index")
        log(
            "[vote-update] "
            f"item_index={item_index} skill={skill_name} reason={similar['reason']} "
            f"existing_item_index={similarity.get('existing_item_index')} "
            f"original_image_ssim={similarity.get('original_image_ssim')} "
            f"winning_action={vote['winning_action']} "
            f"representative_item_index={representative.get('item_index')} "
            f"cluster_size={sample_signature['similar_cluster']['size']}"
        )

    template_dir.mkdir(parents=True, exist_ok=True)
    copied_images = copy_template_images_for_item(
        item=representative_item,
        sample_number=sample_number,
        folder=folder,
        template_dir=template_dir,
    )
    prompt_item = instance_with_template_images(representative_item, copied_images)
    if cluster_vote_note(sample_signature):
        prompt_item["similar_cluster_vote"] = cluster_vote_note(sample_signature)
        prompt_item["vote_representative_item_index"] = representative_signature.get("item_index")
    if replaced_cluster_image_paths:
        prompt_item["replaced_sample_image_paths"] = replaced_cluster_image_paths
    required_sample_image_paths = sorted(
        set(
            [path for path in skill_template_image_paths(existing_skill) if path not in set(replaced_cluster_image_paths)]
            + retained_sample_index_image_paths(sample_index, replaced_cluster_image_paths)
            + copied_skill_image_paths(copied_images)
        )
    )

    mode = "更新迭代" if existing_skill is not None else "新建生成"
    messages = build_messages(
        skill_name=skill_name,
        category=category,
        sample_md=sample_md,
        existing_skill=existing_skill,
        instances=[prompt_item],
        required_sample_image_paths=required_sample_image_paths,
    )

    if save_messages:
        save_debug_messages(messages_dir, skill_name, prompt_item, item_index, messages)

    log(f"[{mode}] item_index={item_index} skill={skill_name} category={category} path={skill_path}")

    response = wrapper.chat_messages(messages)
    skill_md = strip_markdown_fence(response)
    if not skill_md:
        raise RuntimeError(f"skill LLM returned empty skill for category: {category}")
    missing_sample_paths = missing_required_skill_image_paths(skill_md, required_sample_image_paths)
    if missing_sample_paths:
        raise RuntimeError(
            "skill LLM output is missing required Multimodal Samples image paths for "
            f"skill={skill_name} item_index={item_index}: "
            + ", ".join(missing_sample_paths)
        )
    write_text(skill_path, skill_md + "\n")
    clear_replaced_sample_images(similar_matches, replaced_cluster_image_paths, representative_signature)
    representative_signature["template_images"] = copied_images
    representative_signature["skill_md_image_paths"] = copied_skill_image_paths(copied_images)
    if representative_signature is not sample_signature:
        sample_signature.pop("sample_number", None)
        sample_index.append(sample_signature)
    else:
        sample_index.append(representative_signature)
    save_sample_index(folder, sample_index)
    return True


def update_text_only_skill_for_instance(
    *,
    item: Dict[str, Any],
    item_index: int,
    skills_dir: Path,
    messages_dir: Path,
    sample_md: str,
    save_messages: bool,
    wrapper: Any,
) -> bool:
    category = category_name(item)
    skill_name = skill_name_for_item(item)
    folder = skills_dir / safe_folder_name(skill_name)
    skill_path = folder / "skill.md"
    existing_skill = read_text(skill_path) if skill_path.exists() else None
    mode = "纯文本更新迭代" if existing_skill is not None else "纯文本新建生成"

    messages = build_text_only_messages(
        skill_name=skill_name,
        category=category,
        sample_md=sample_md,
        existing_skill=existing_skill,
        instances=[item],
    )

    if save_messages:
        save_debug_messages(messages_dir, skill_name, item, item_index, messages)

    log(f"[{mode}] item_index={item_index} skill={skill_name} category={category} path={skill_path}")

    response = wrapper.chat_messages(messages)
    skill_md = strip_markdown_fence(response)
    if not skill_md:
        raise RuntimeError(f"skill LLM returned empty text-only skill for category: {category}")
    write_text(skill_path, skill_md + "\n")
    return True


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or update MobileGym skills by iterating better reprediction examples one by one.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--messages-dir", type=Path, default=DEFAULT_MESSAGES_DIR)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--task-create",
        action="store_true",
        help="Create/update skills by mapping each instance instruction to the 14 task categories in case_task.md",
    )
    parser.add_argument(
        "--task-case",
        type=Path,
        default=DEFAULT_TASK_CASE,
        help="Task category document used by --task-create",
    )
    parser.add_argument(
        "--task-sample",
        type=Path,
        default=DEFAULT_TASK_SAMPLE,
        help="JSONL task sample file whose line numbers are referenced by --task-case",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start processing from this 1-based source item_index shown in logs",
    )
    parser.add_argument("--max-items", type=int, default=0, help="Maximum number of better examples to process; 0 means unlimited")
    parser.add_argument("--require-correct", action="store_true", help="Also require is_corrected=true and step_action_correctness=correct")
    parser.add_argument(
        "--no-verifier",
        action="store_true",
        help="Process every input item without filtering by verifier/judgment results",
    )
    parser.add_argument("--save-messages", action="store_true", help="Save skill generation messages for inspection")
    parser.add_argument(
        "--text-only-skill",
        action="store_true",
        help="Use an independent text-only skill generation flow that writes only skill.md and does not create template files",
    )
    parser.add_argument(
        "--image-ssim-threshold",
        type=float,
        default=DEFAULT_IMAGE_SSIM_THRESHOLD,
        help="Treat samples as similar when original image SSIM is at least this value",
    )
    parser.add_argument("--skill-llm", choices=["27b", "doubao", "qwen"], default="doubao", help="skill generation backend to use")
    parser.add_argument("--model-base-url", default=DEFAULT_MODEL_BASE_URL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--no-verify-false", action="store_true", help="keep TLS verification enabled")
    parser.add_argument("--debug", action="store_true", help="Print the raw API wrapper response")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    payload = read_json(args.input)
    sample_md = read_text(DEFAULT_TEXT_SAMPLE) if args.text_only_skill else read_text(args.sample)
    items = normalize_items(payload)
    task_instruction_mapping: Dict[str, Dict[str, Any]] = {}
    if args.task_create:
        task_instruction_mapping = load_task_instruction_mapping(args.task_case, args.task_sample)
        log(
            f"[task-create] loaded_task_instruction_mappings={len(task_instruction_mapping)} "
            f"case_task={args.task_case} task_sample={args.task_sample}"
        )
    if not args.text_only_skill:
        preprocess_existing_skills(args.skills_dir)
    wrapper = build_wrapper(args)

    processed = 0
    matched_better = 0
    skipped_unmapped_task = 0
    skipped_similar = 0
    for item_index, item in enumerate(items, start=1):
        if item_index < args.start_index:
            continue
        if not args.no_verifier and not is_better_item(item, require_correct=args.require_correct):
            continue

        if args.task_create:
            mapped_item = apply_task_mapping_to_item(item, task_instruction_mapping)
            if mapped_item is None:
                skipped_unmapped_task += 1
                log(
                    "[skip-task-unmapped] "
                    f"item_index={item_index} instruction={compact(item.get('instruction'))!r}"
                )
                continue
            item = mapped_item

        matched_better += 1
        if args.text_only_skill:
            updated = update_text_only_skill_for_instance(
                item=item,
                item_index=item_index,
                skills_dir=args.skills_dir,
                messages_dir=args.messages_dir,
                sample_md=sample_md,
                save_messages=args.save_messages,
                wrapper=wrapper,
            )
        else:
            updated = update_skill_for_instance(
                item=item,
                item_index=item_index,
                skills_dir=args.skills_dir,
                messages_dir=args.messages_dir,
                sample_md=sample_md,
                save_messages=args.save_messages,
                debug=args.debug,
                wrapper=wrapper,
                image_ssim_threshold=args.image_ssim_threshold,
            )
        if updated:
            processed += 1
        else:
            skipped_similar += 1
        if args.max_items > 0 and processed >= args.max_items:
            break

    if matched_better == 0:
        raise SystemExit("没有找到 judgment.comparison_to_original == 'better' 的实例")

    log(
        f"[finish] processed_better_examples={processed} "
        f"skipped_similar_examples={skipped_similar} "
        f"skipped_unmapped_task_examples={skipped_unmapped_task} "
        f"matched_better_examples={matched_better} skills_dir={args.skills_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
