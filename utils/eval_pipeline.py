"""
评估流水线：将 lite / 全量 / 消融组合抽象为 PipelineSpec。

设计目标：
1) 用显式 profile_id 区分不同消融配置，避免缓存/中间 prompt 互相污染。
2) 尽量复用现有 judge/ATC/keyframe 逻辑，只在「终判策略」与「lite action_chain 来源」上可切换。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any, Dict, Optional


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_profile_id(spec_dict: Dict[str, Any], length: int = 10) -> str:
    raw = _stable_json_dumps(spec_dict).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:length]


# JSON 内仅作文档用，不参与 PipelineSpec 与 profile_id
_PIPELINE_SPEC_JSON_DOC_KEYS = frozenset({"desc", "examples"})


@dataclass(frozen=True)
class PipelineSpec:
    """
    约束：
    - 该 spec 主要用于离线评估（main.py process_task）。
    - spec 里尽量只放「会影响 prompt/LLM输入与输出」的开关。
    """

    # 预设名（仅用于可读性）
    preset: str

    # eval 阶段：是否执行完整 1~5（lite endframe 只做末尾综合判断）
    eval_lite_mode: bool

    # output 阶段：是否只输出 is_task_success（lite 风格），以兼容预测服务
    output_lite_mode: bool

    # 终判策略
    # - and_split：完成度+轨迹正确性各自判断，final=AND
    # - unified_single：一次 prompt 直接输出 is_task_success（像 lite 一样）
    # - lite_plus_trajectory_all_images：lite 综合 + 轨迹全图同时输出
    final_judgment_mode: str

    # lite 下：是否在综合 prompt 中附带 action_chain
    lite_include_action_list: bool

    # lite 下：action_chain 来源
    # - raw：使用 trajectory[].value.action
    # - validated：使用 action 校验后的 corrected action（含二次校验）
    lite_action_chain_source: str

    # unified_full_context：统一 prompt 是否加入关键帧检查信息（需要 keyframes 结果）
    unified_scope: str

    # prompt/LLM 输入影响项（放入 profile_id，避免缓存/中间 prompt 混用）
    use_keyframe_images: bool
    use_qa_images: bool
    no_qa: bool
    skip_takeover: bool
    # 是否保留最后一步 action（True 时不做“接管/等待”之外清空）
    keep_last_action: bool
    compress_quality: Optional[float]

    # profile_id：由构造过程填充
    profile_id: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_pipeline_spec_from_preset(
    *,
    preset: str,
    eval_lite_mode: bool,
    output_lite_mode: bool,
    final_judgment_mode: str,
    lite_include_action_list: bool,
    lite_action_chain_source: str,
    unified_scope: str,
    use_keyframe_images: bool,
    use_qa_images: bool,
    no_qa: bool,
    skip_takeover: bool,
    keep_last_action: bool,
    compress_quality: Optional[float],
) -> PipelineSpec:
    spec_dict = {
        "preset": preset,
        "eval_lite_mode": eval_lite_mode,
        "output_lite_mode": output_lite_mode,
        "final_judgment_mode": final_judgment_mode,
        "lite_include_action_list": lite_include_action_list,
        "lite_action_chain_source": lite_action_chain_source,
        "unified_scope": unified_scope,
        "use_keyframe_images": use_keyframe_images,
        "use_qa_images": use_qa_images,
        "no_qa": no_qa,
        "skip_takeover": skip_takeover,
        "keep_last_action": keep_last_action,
        "compress_quality": compress_quality,
    }
    profile_id = compute_profile_id(spec_dict)
    return PipelineSpec(profile_id=profile_id, **spec_dict)


def load_pipeline_spec_from_json(data: Dict[str, Any]) -> PipelineSpec:
    """
    允许用户用 JSON 覆盖 spec。
    JSON 中可以不包含 profile_id；会自动计算。
    可选字段 desc（说明）、examples（示例命令字符串列表）仅供阅读，加载时会剔除且不计入 profile_id。
    """

    spec_dict = dict(data)
    spec_dict.pop("profile_id", None)
    for k in _PIPELINE_SPEC_JSON_DOC_KEYS:
        spec_dict.pop(k, None)
    required_fields = {
        "preset",
        "eval_lite_mode",
        "output_lite_mode",
        "final_judgment_mode",
        "lite_include_action_list",
        "lite_action_chain_source",
        "unified_scope",
        "use_keyframe_images",
        "use_qa_images",
        "no_qa",
        "skip_takeover",
        "compress_quality",
    }
    missing = required_fields - set(spec_dict.keys())
    if missing:
        raise ValueError(f"pipeline spec json 缺少字段: {sorted(missing)}")
    optional_fields = {"keep_last_action"}
    extra = set(spec_dict.keys()) - required_fields - optional_fields
    if extra:
        raise ValueError(f"pipeline spec json 含未知字段: {sorted(extra)}")
    spec_dict.setdefault("keep_last_action", False)
    profile_id = compute_profile_id(spec_dict)
    return PipelineSpec(profile_id=profile_id, **spec_dict)


def build_pipeline_spec(
    *,
    preset: Optional[str] = None,
    eval_lite_mode: bool,
    output_lite_mode: bool,
    use_keyframe_images: bool,
    use_qa_images: bool,
    no_qa: bool,
    skip_takeover: bool,
    keep_last_action: bool,
    compress_quality: Optional[float],
    lite_include_action_list: bool,
    lite_action_chain_source: str,
    final_judgment_mode: str,
    unified_scope: str,
) -> PipelineSpec:
    return build_pipeline_spec_from_preset(
        preset=preset or "custom",
        eval_lite_mode=eval_lite_mode,
        output_lite_mode=output_lite_mode,
        final_judgment_mode=final_judgment_mode,
        lite_include_action_list=lite_include_action_list,
        lite_action_chain_source=lite_action_chain_source,
        unified_scope=unified_scope,
        use_keyframe_images=use_keyframe_images,
        use_qa_images=use_qa_images,
        no_qa=no_qa,
        skip_takeover=skip_takeover,
        keep_last_action=keep_last_action,
        compress_quality=compress_quality,
    )


def preset_from_legacy_flags(
    *,
    lite_mode: bool,
    lite_action_list: bool,
    eval_profile: Optional[str] = None,
    use_keyframe_images: bool = False,
    use_qa_images: bool = False,
    no_qa: bool = False,
    skip_takeover: bool = True,
    keep_last_action: bool = False,
    compress_quality: Optional[float] = None,
) -> PipelineSpec:
    """
    兼容旧参数：
    - lite_mode=True -> eval_lite_mode=True, output_lite_mode=True
    - lite_mode=False -> eval_lite_mode=False, output_lite_mode=False
    同时支持用户通过 eval_profile 覆盖。
    """

    # eval_profile 作为最高优先级：允许指定一个 preset 名称（见下内置映射）
    if eval_profile:
        profile = eval_profile.strip()
        # 内置 preset 映射：只支持我们计划的实验组合
        if profile == "lite":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=True,
                final_judgment_mode="unified_single",
                lite_include_action_list=False,
                lite_action_chain_source="raw",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "lite+action_list":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=True,
                final_judgment_mode="unified_single",
                lite_include_action_list=True,
                lite_action_chain_source="raw",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "lite+action_list_validated":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=True,
                final_judgment_mode="unified_single",
                lite_include_action_list=True,
                lite_action_chain_source="validated",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "lite_action_validated_thought":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=True,
                final_judgment_mode="unified_single",
                lite_include_action_list=True,
                lite_action_chain_source="validated_thought",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "full":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=False,
                output_lite_mode=False,
                final_judgment_mode="and_split",
                lite_include_action_list=False,
                lite_action_chain_source="raw",
                unified_scope="full_context",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "full_unified":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=False,
                output_lite_mode=True,
                final_judgment_mode="unified_single",
                lite_include_action_list=False,
                lite_action_chain_source="raw",
                unified_scope="full_context",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "trajectory_allimg_raw":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=False,
                final_judgment_mode="trajectory_correctness_all_images",
                lite_include_action_list=True,
                lite_action_chain_source="raw",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "trajectory_allimg_validated":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=False,
                final_judgment_mode="trajectory_correctness_all_images",
                lite_include_action_list=True,
                lite_action_chain_source="validated",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "lite_action+trajectory_allimg":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=False,
                final_judgment_mode="lite_plus_trajectory_all_images",
                lite_include_action_list=True,
                lite_action_chain_source="raw",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "lite_trajectory_allimg_unified_binary":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=False,
                final_judgment_mode="lite_plus_trajectory_all_images_unified_binary",
                lite_include_action_list=True,
                lite_action_chain_source="raw",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "lite_trajectory_allimg_unified_binary_action_validated":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=False,
                final_judgment_mode="lite_plus_trajectory_all_images_unified_binary",
                lite_include_action_list=True,
                lite_action_chain_source="validated",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )
        if profile == "lite_action_validated+trajectory_allimg_validated":
            return build_pipeline_spec_from_preset(
                preset=profile,
                eval_lite_mode=True,
                output_lite_mode=False,
                final_judgment_mode="lite_plus_trajectory_all_images",
                lite_include_action_list=True,
                lite_action_chain_source="validated",
                unified_scope="lite_end_only",
                use_keyframe_images=use_keyframe_images,
                use_qa_images=use_qa_images,
                no_qa=no_qa,
                skip_takeover=skip_takeover,
                keep_last_action=keep_last_action,
                compress_quality=compress_quality,
            )

        raise ValueError(f"未知 eval_profile: {profile}")

    # 旧逻辑映射
    if lite_mode:
        return build_pipeline_spec_from_preset(
            preset="legacy_lite" if not lite_action_list else "legacy_lite+action_list",
            eval_lite_mode=True,
            output_lite_mode=True,
            final_judgment_mode="unified_single",
            lite_include_action_list=bool(lite_action_list),
            lite_action_chain_source="raw",
            unified_scope="lite_end_only",
            use_keyframe_images=use_keyframe_images,
            use_qa_images=use_qa_images,
            no_qa=no_qa,
            skip_takeover=skip_takeover,
            keep_last_action=keep_last_action,
            compress_quality=compress_quality,
        )

    return build_pipeline_spec_from_preset(
        preset="legacy_full",
        eval_lite_mode=False,
        output_lite_mode=False,
        final_judgment_mode="and_split",
        lite_include_action_list=False,
        lite_action_chain_source="raw",
        unified_scope="full_context",
        use_keyframe_images=use_keyframe_images,
        use_qa_images=use_qa_images,
        no_qa=no_qa,
        skip_takeover=skip_takeover,
        keep_last_action=keep_last_action,
        compress_quality=compress_quality,
    )
