#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_INPUT="${DEFAULT_INPUT:-/opt/nas/p/mm/ie_env/qmr/mmskill/judge_results_parallel/mobilegym_skill_result.json}"
DEFAULT_MANIFEST="${DEFAULT_MANIFEST:-/opt/nas/p/mm/ie_env/qmr/mmskill/trajs/batch_runs.jsonl}"

has_input_arg=0
has_manifest_arg=0
for arg in "$@"; do
  if [[ "${arg}" == "--input" || "${arg}" == --input=* ]]; then
    has_input_arg=1
  fi
  if [[ "${arg}" == "--manifest" || "${arg}" == --manifest=* ]]; then
    has_manifest_arg=1
  fi
done

extra_args=()
if [[ "${has_input_arg}" == "0" ]]; then
  extra_args+=(--input "${DEFAULT_INPUT}")
fi
if [[ "${has_manifest_arg}" == "0" && -f "${DEFAULT_MANIFEST}" ]]; then
  extra_args+=(--manifest "${DEFAULT_MANIFEST}")
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/count_mobilegym_orm_success.py" "${extra_args[@]}" "$@"
