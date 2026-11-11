#!/usr/bin/env bash

INPUT=""
OUTPUT=""

python -u judge_lite_trajectory_failure.py \
    --input "${INPUT}" \
    --max-rollouts 1 \
    --output "${OUTPUT}"
