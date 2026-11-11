# Pipeline Spec 枚举值说明

`--pipeline-json` 对应的 JSON 由 `utils/eval_pipeline.py` 的 `load_pipeline_spec_from_json()` 校验字段存在性。

除 `compress_quality` 允许为 `0~1` 浮点值外，其它字段基本都是布尔或枚举字符串。

## preset（可读性用）
当前内置预设：
- `lite`
- `lite+action_list`
- `lite+action_list_validated`
- `trajectory_allimg_raw`
- `trajectory_allimg_validated`
- `lite_action+trajectory_allimg`
- `lite_action_validated+trajectory_allimg_validated`
- `full`
- `full_unified`

你也可以写 `custom`，但最终行为主要由下面的关键枚举字段决定。

## final_judgment_mode（终判方式）
- `and_split`
  - 走 completion 分判 + trajectory_correctness 分判，然后 `final_judgment = completion==1 AND trajectory==1`
- `unified_single`
  - 走一次综合 prompt 输出 `final_judgment`（即 `is_task_success`）

## lite_action_chain_source（lite 模式 action_list 来源）
- `raw`：直接使用 `trajectory[].value.action`
- `validated`：先跑 action 校验（含二次校验），使用校验后的 corrected action 链

## unified_scope（unified prompt 注入范围）
- `lite_end_only`
  - 仅末尾若干步证据（与 lite 的末尾综合判断一致）
- `full_context`
  - 需要关键帧证据拼入 unified prompt（与 full_unified 的行为一致）

## 其它字段（布尔/数值）
- `eval_lite_mode`：是否在 eval 阶段走 lite 跳过 1~5 的流程
- `output_lite_mode`：是否只输出 `is_task_success`（lite 风格）；否则输出 completion/trajectory 的拆分字段
- `lite_include_action_list`：lite unified prompt 中是否包含 action list
- `use_keyframe_images`：关键帧判断是否传图片
- `use_qa_images`：QA 判断是否传图片
- `no_qa`：是否跳过 QA 相关 prompt/判断
- `skip_takeover`：是否跳过步骤级接管判定
- `compress_quality`：图像压缩质量（`0~1`）；数值越小压缩越强

## 用法回忆
例如：
```bash
python /var/code/zhaoj/orm/test_service.py --model doubao-seed-1.8 --pipeline-json /abs/path/pipeline_full_unified.json
```

