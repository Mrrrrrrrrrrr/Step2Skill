#!/usr/bin/env bash
set -euo pipefail

INPUT="D:/Traj/ali320k/train_data_sessions_general_123apps_90tasks_11k_20260420_with_judge_and_filtering_20260514_using_reasoning_content.jsonl"
SCREENSHOT_ROOT="D:/Traj/ali320k/sessions_general_123apps_90tasks_11k_20260420"

conda run --no-capture-output -n zj_base310 python -u main.py \
  --input "${INPUT}" \
  --screenshot-root "${SCREENSHOT_ROOT}" \
  --task-num 5 \
  --batch-size 1 \
  --model seed_free \
  --eval-profile lite_trajectory_allimg_unified_binary \
  --no-qa \
  --compress-quality 0.8
