#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/judge_results_parallel}"

INPUT_JSON="${INPUT_JSON:-${RESULTS_DIR}/mobilegym_repredicted_step_judgments.json}"
NO_VERIFIER_CLI=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-verifier)
      NO_VERIFIER_CLI=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "[error] unknown argument: $1" >&2
      exit 1
      ;;
    *)
      INPUT_JSON="$1"
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  INPUT_JSON="$1"
fi
SKILLS_DIR="${SKILLS_DIR:-${ROOT_DIR}/skills}"
MESSAGES_DIR="${MESSAGES_DIR:-${ROOT_DIR}/skill_generation_messages}"
SAMPLE_MD="${SAMPLE_MD:-${ROOT_DIR}/skillsample.md}"
TASK_CASE="${TASK_CASE:-${ROOT_DIR}/case_task.md}"
TASK_SAMPLE="${TASK_SAMPLE:-${ROOT_DIR}/mobilegym/sample.jsonl}"
LOG_FILE="${LOG_FILE:-${RESULTS_DIR}/extract_skill.log}"

SKILL_LLM="${SKILL_LLM:-doubao}"
TIMEOUT="${TIMEOUT:-300}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MODEL_BASE_URL="${MODEL_BASE_URL:-*}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-vllm}"
API_KEY="${API_KEY:-EMPTY}"
MAX_ITEMS="${MAX_ITEMS:-0}"
IMAGE_SSIM_THRESHOLD="${IMAGE_SSIM_THRESHOLD:-0.995}"

if [[ "${SKILL_LLM}" != "27b" && "${SKILL_LLM}" != "doubao" && "${SKILL_LLM}" != "qwen" ]]; then
  echo "[error] SKILL_LLM must be one of: 27b, doubao, qwen; got: ${SKILL_LLM}" >&2
  exit 1
fi

if [[ ! "${TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] TIMEOUT must be a positive integer, got: ${TIMEOUT}" >&2
  exit 1
fi

if [[ ! "${MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "[error] MAX_RETRIES must be a non-negative integer, got: ${MAX_RETRIES}" >&2
  exit 1
fi

if [[ ! "${MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] MAX_TOKENS must be a positive integer, got: ${MAX_TOKENS}" >&2
  exit 1
fi

if [[ ! "${MAX_ITEMS}" =~ ^[0-9]+$ ]]; then
  echo "[error] MAX_ITEMS must be a non-negative integer, got: ${MAX_ITEMS}" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/extract_skill.py" ]]; then
  echo "[error] extract_skill.py not found under: ${ROOT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_JSON}" ]]; then
  echo "[error] step judgment JSON not found: ${INPUT_JSON}" >&2
  echo "[hint] run judge_repredict_steps_parallel.sh first, or pass INPUT_JSON/path as the first argument" >&2
  exit 1
fi

if [[ ! -f "${SAMPLE_MD}" ]]; then
  echo "[error] skillsample.md not found: ${SAMPLE_MD}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}" "${SKILLS_DIR}" "${MESSAGES_DIR}" "$(dirname "${LOG_FILE}")"

extra_args=()
if [[ "${REQUIRE_CORRECT:-0}" == "1" ]]; then
  extra_args+=(--require-correct)
fi
if [[ "${NO_VERIFIER:-0}" == "1" || "${NO_VERIFIER_CLI}" == "1" ]]; then
  extra_args+=(--no-verifier)
fi
if [[ "${SAVE_MESSAGES:-0}" == "1" ]]; then
  extra_args+=(--save-messages)
fi
if [[ "${TEXT_ONLY_SKILL:-0}" == "1" ]]; then
  extra_args+=(--text-only-skill)
fi
if [[ "${TASK_CREATE:-0}" == "1" ]]; then
  extra_args+=(--task-create --task-case "${TASK_CASE}" --task-sample "${TASK_SAMPLE}")
fi
if [[ "${NO_DEDUPE_SAMPLES:-0}" == "1" ]]; then
  extra_args+=(--no-dedupe-samples)
fi
if [[ "${NO_VERIFY_FALSE:-0}" == "1" ]]; then
  extra_args+=(--no-verify-false)
fi
if [[ "${DEBUG:-0}" == "1" ]]; then
  extra_args+=(--debug)
fi

echo "[info] input: ${INPUT_JSON}"
echo "[info] skills_dir: ${SKILLS_DIR}"
echo "[info] messages_dir: ${MESSAGES_DIR}"
echo "[info] sample: ${SAMPLE_MD}"
echo "[info] log: ${LOG_FILE}"
echo "[info] skill_llm: ${SKILL_LLM}"
echo "[info] timeout: ${TIMEOUT}"
echo "[info] max_items: ${MAX_ITEMS}"
echo "[info] no_verifier: $([[ "${NO_VERIFIER:-0}" == "1" || "${NO_VERIFIER_CLI}" == "1" ]] && echo 1 || echo 0)"
echo "[info] text_only_skill: ${TEXT_ONLY_SKILL:-0}"
echo "[info] task_create: ${TASK_CREATE:-0}"
echo "[info] model_base_url: ${MODEL_BASE_URL}"
echo "[info] model_name: ${MODEL_NAME}"

"${PYTHON_BIN}" -u "${ROOT_DIR}/extract_skill.py" \
  --input "${INPUT_JSON}" \
  --skills-dir "${SKILLS_DIR}" \
  --messages-dir "${MESSAGES_DIR}" \
  --sample "${SAMPLE_MD}" \
  --max-items "${MAX_ITEMS}" \
  --image-ssim-threshold "${IMAGE_SSIM_THRESHOLD}" \
  --skill-llm "${SKILL_LLM}" \
  --model-base-url "${MODEL_BASE_URL}" \
  --model-name "${MODEL_NAME}" \
  --api-key "${API_KEY}" \
  --timeout "${TIMEOUT}" \
  --max-retries "${MAX_RETRIES}" \
  --max-tokens "${MAX_TOKENS}" \
  "${extra_args[@]}" \
  2>&1 | tee "${LOG_FILE}"

echo "[finish] skills updated under: ${SKILLS_DIR}"
