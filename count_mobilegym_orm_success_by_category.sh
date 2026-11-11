#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Set this to the judge_results_parallel* directory you want to count.
INPUT_DIR="${INPUT_DIR:-${SCRIPT_DIR}/judge_results_parallel_0829}"
DEFAULT_RESULT_NAME="${DEFAULT_RESULT_NAME:-mobilegym_skill_result.json}"

show_usage() {
  cat <<'EOF'
Usage:
  bash count_mobilegym_orm_success_by_category.sh [optional_judge_results_dir|optional_result_json] [options]

Examples:
  bash count_mobilegym_orm_success_by_category.sh judge_results_parallel_0829
  bash count_mobilegym_orm_success_by_category.sh judge_results_parallel_0829/mobilegym_skill_result.json
  bash ../count_mobilegym_orm_success_by_category.sh

Environment:
  PYTHON_BIN            Python executable, default: python
  INPUT_DIR             Default judge results directory
  DEFAULT_RESULT_NAME   Result JSON name inside a judge directory, default: mobilegym_skill_result.json
  MANIFEST              Optional batch_runs.jsonl path
  REPEATS               Optional repeats value, default: script default
  TRAJECTORY_ROOT       Optional category root passed to --trajectory-root
EOF
}

target="${INPUT_DIR}"
if (($# > 0)); then
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --*)
      ;;
    *)
      target="$1"
      shift
      ;;
  esac
fi

if [[ -d "${target}" ]]; then
  input_json="${target%/}/${DEFAULT_RESULT_NAME}"
else
  input_json="${target}"
fi

if [[ ! -f "${input_json}" ]]; then
  echo "[error] result JSON not found: ${input_json}" >&2
  exit 1
fi

extra_args=(--input "${input_json}")

if [[ -n "${MANIFEST:-}" ]]; then
  extra_args+=(--manifest "${MANIFEST}")
fi
if [[ -n "${REPEATS:-}" ]]; then
  extra_args+=(--repeats "${REPEATS}")
fi
if [[ -n "${TRAJECTORY_ROOT:-}" ]]; then
  extra_args+=(--trajectory-root "${TRAJECTORY_ROOT}")
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/count_mobilegym_orm_success_by_category.py" "${extra_args[@]}" "$@"
