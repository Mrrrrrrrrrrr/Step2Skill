#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAJS_DIR="${1:-${TRAJS_DIR:-${ROOT_DIR}/trajs_skill_3rep}}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/judge_results_parallel_skill_3rep}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-8}"
MODEL="${MODEL:-seed20}"
KNOWLEDGE="${KNOWLEDGE:-knowledge_base}"
INDENT="${INDENT:-2}"
MERGED_OUTPUT="${MERGED_OUTPUT:-${OUT_DIR}/mobilegym_skill_result.json}"

if [[ ! "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 1
fi

if [[ ! -d "${TRAJS_DIR}" ]]; then
  echo "[error] trajs directory not found: ${TRAJS_DIR}" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/mobilegym_judge_run.py" ]]; then
  echo "[error] mobilegym_judge_run.py not found under: ${ROOT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/rollouts" "${OUT_DIR}/logs"

extra_args=()
if [[ "${NO_ACTION_LIST:-0}" == "1" ]]; then
  extra_args+=(--no-action-list)
fi
if [[ "${VERBOSE:-0}" == "1" ]]; then
  extra_args+=(--verbose)
fi

sanitize_name() {
  local name="$1"
  name="${name//[^A-Za-z0-9._-]/_}"
  printf '%s' "${name}"
}

rollout_inputs=()
rollout_task_ids=()
rollout_outputs=()
rollout_logs=()

shopt -s nullglob
category_dirs=("${TRAJS_DIR}"/*/)

if (( ${#category_dirs[@]} == 0 )); then
  echo "[error] no category directories found under: ${TRAJS_DIR}" >&2
  exit 1
fi

for category_dir in "${category_dirs[@]}"; do
  category_dir="${category_dir%/}"
  category_name="$(basename "${category_dir}")"
  instance_dirs=("${category_dir}"/*/)

  if (( ${#instance_dirs[@]} == 0 )); then
    instance_dirs=("${category_dir}/")
  fi

  for instance_dir in "${instance_dirs[@]}"; do
    instance_dir="${instance_dir%/}"
    instance_name="$(basename "${instance_dir}")"
    task_id="$(sanitize_name "${category_name}_${instance_name}")"

    rollout_inputs+=("${instance_dir}")
    rollout_task_ids+=("${task_id}")
    rollout_outputs+=("${OUT_DIR}/rollouts/${task_id}.judge.json")
    rollout_logs+=("${OUT_DIR}/logs/${task_id}.judge.log")

    if [[ -n "${MAX_ROLLOUTS:-}" ]] && (( ${#rollout_inputs[@]} >= MAX_ROLLOUTS )); then
      break 2
    fi
  done
done

if (( ${#rollout_inputs[@]} == 0 )); then
  echo "[error] no rollout instance directories found under: ${TRAJS_DIR}" >&2
  exit 1
fi

echo "[info] input: ${TRAJS_DIR}"
echo "[info] output: ${OUT_DIR}"
echo "[info] merged: ${MERGED_OUTPUT}"
echo "[info] workers: ${WORKERS}"
echo "[info] rollout instances: ${#rollout_inputs[@]}"

run_one_index() {
  local index="$1"
  local input_path="${rollout_inputs[$index]}"
  local task_id="${rollout_task_ids[$index]}"
  local output_path="${rollout_outputs[$index]}"
  local log_path="${rollout_logs[$index]}"
  local total="${#rollout_inputs[@]}"
  local ordinal=$((index + 1))

  echo "[judge] ${ordinal}/${total} ${task_id}"
  if "${PYTHON_BIN}" -u "${ROOT_DIR}/mobilegym_judge_run.py" \
      --input "${input_path}" \
      --task-id "${task_id}" \
      --model "${MODEL}" \
      --knowledge "${KNOWLEDGE}" \
      --output "${output_path}" \
      --indent "${INDENT}" \
      "${extra_args[@]}" \
      >"${log_path}" 2>&1; then
    echo "[ok] ${task_id} -> ${output_path}"
    return 0
  fi

  echo "[fail] ${task_id}; see ${log_path}" >&2
  return 1
}

active=0
fail_count=0

for index in "${!rollout_inputs[@]}"; do
  run_one_index "${index}" &
  active=$((active + 1))

  if (( active >= WORKERS )); then
    if ! wait -n; then
      fail_count=$((fail_count + 1))
    fi
    active=$((active - 1))
  fi
done

while (( active > 0 )); do
  if ! wait -n; then
    fail_count=$((fail_count + 1))
  fi
  active=$((active - 1))
done

"${PYTHON_BIN}" - "${OUT_DIR}/rollouts" "${TRAJS_DIR}" "${MERGED_OUTPUT}" "${INDENT}" <<'PY'
import json
import sys
from pathlib import Path

rollouts_dir = Path(sys.argv[1]).expanduser().resolve()
input_dir = Path(sys.argv[2]).expanduser().resolve()
output_path = Path(sys.argv[3]).expanduser().resolve()
indent_arg = int(sys.argv[4])
json_indent = None if indent_arg == 0 else indent_arg

results = []
for path in sorted(rollouts_dir.glob("*.judge.json")):
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        results.extend(payload["results"])
    else:
        results.append(payload)

merged = {
    "input": str(input_dir),
    "count": len(results),
    "results": results,
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=json_indent)
    f.write("\n")

print(f"[merge] {len(results)} result(s) -> {output_path}")
PY

ok_count=$(find "${OUT_DIR}/rollouts" -maxdepth 1 -type f -name '*.judge.json' | wc -l | tr -d ' ')
echo "[finish] ok=${ok_count} fail=${fail_count}"

if (( fail_count > 0 )); then
  exit 1
fi
