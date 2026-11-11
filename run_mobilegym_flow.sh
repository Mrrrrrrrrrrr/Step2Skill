#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAJ_DIR="${TRAJ_DIR:-${SCRIPT_DIR}/traj_0826}"
JUDGE_DIR="${JUDGE_DIR:-${SCRIPT_DIR}/judge_results_parallel_0826}"
SKILLS_DIR="${SKILLS_DIR:-${SCRIPT_DIR}/skills_0826}"
SKILL_TRAJ_DIR="${SKILL_TRAJ_DIR:-${SCRIPT_DIR}/traj_0826_skill}"
SKILL_JUDGE_DIR="${SKILL_JUDGE_DIR:-${SCRIPT_DIR}/judge_results_parallel_0826_skill}"

FIRST_REPEATS="${FIRST_REPEATS:-${REPEATS:-3}}"
SKILL_REPEATS="${SKILL_REPEATS:-3}"
CLASS_WORKERS="${CLASS_WORKERS:-10}"
WORKERS="${WORKERS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://10.80.10.0:30772/v1}"
MOBILEGYM_MODEL_BASE_URL="${MOBILEGYM_MODEL_BASE_URL:-${MODEL_BASE_URL}}"

JUDGE_JSON="${JUDGE_DIR}/mobilegym_skill_result.json"
CLASSIFY_JSON="${JUDGE_DIR}/mobilegym_error_classification.json"
REPREDICT_JSON="${JUDGE_DIR}/mobilegym_edited_repredictions.json"
STEP_JUDGMENT_JSON="${JUDGE_DIR}/mobilegym_repredicted_step_judgments.json"
SKILL_JUDGE_JSON="${SKILL_JUDGE_DIR}/mobilegym_skill_result.json"

run_step() {
  local name="$1"
  shift
  echo
  echo "========== ${name} =========="
  "$@"
}

mkdir -p \
  "${TRAJ_DIR}" \
  "${JUDGE_DIR}" \
  "${SKILLS_DIR}" \
  "${SKILL_TRAJ_DIR}" \
  "${SKILL_JUDGE_DIR}"

run_step "1. Run trajectories without skills" \
  env \
    MOBILEGYM_CLASS_WORKERS="${CLASS_WORKERS}" \
    MOBILEGYM_MODEL_BASE_URL="${MOBILEGYM_MODEL_BASE_URL}" \
    bash "${SCRIPT_DIR}/mobilegym/run_sample_batch.sh" \
    --output-dir "${TRAJ_DIR}" \
    --repeats "${FIRST_REPEATS}"

run_step "2. Judge trajectories" \
  env \
    OUT_DIR="${JUDGE_DIR}" \
    MERGED_OUTPUT="${JUDGE_JSON}" \
    WORKERS="${WORKERS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/judge_trajs_parallel.sh" "${TRAJ_DIR}"

run_step "3. Count first judge success rate" \
  bash "${SCRIPT_DIR}/count_mobilegym_orm_success.sh" \
    --input "${JUDGE_JSON}" \
    --manifest "${TRAJ_DIR}/batch_runs.jsonl" \
    --repeats "${FIRST_REPEATS}" \
    --output-dir "${JUDGE_DIR}"

run_step "4. Classify judge results" \
  env \
    RESULTS_DIR="${JUDGE_DIR}" \
    OUTPUT_JSON="${CLASSIFY_JSON}" \
    LOG_FILE="${JUDGE_DIR}/mobilegym_classify.log" \
    WORKERS="${WORKERS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/classify_judge_results.sh" "${JUDGE_JSON}"

run_step "5. Repredict errors" \
  env \
    RESULTS_DIR="${JUDGE_DIR}" \
    OUTPUT_JSON="${REPREDICT_JSON}" \
    LOG_FILE="${JUDGE_DIR}/mobilegym_repredict.log" \
    WORKERS="${WORKERS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL_BASE_URL="${MODEL_BASE_URL}" \
    bash "${SCRIPT_DIR}/repredict_errors_parallel.sh" "${CLASSIFY_JSON}"

run_step "6. Judge repredicted steps" \
  env \
    RESULTS_DIR="${JUDGE_DIR}" \
    OUTPUT_JSON="${STEP_JUDGMENT_JSON}" \
    LOG_FILE="${JUDGE_DIR}/mobilegym_judge_step.log" \
    WORKERS="${WORKERS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL_BASE_URL="${MODEL_BASE_URL}" \
    bash "${SCRIPT_DIR}/judge_repredict_steps_parallel.sh" "${REPREDICT_JSON}"

run_step "7. Extract skills" \
  env \
    RESULTS_DIR="${JUDGE_DIR}" \
    SKILLS_DIR="${SKILLS_DIR}" \
    MESSAGES_DIR="${JUDGE_DIR}/skill_generation_messages" \
    LOG_FILE="${JUDGE_DIR}/extract_skill.log" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MODEL_BASE_URL="${MODEL_BASE_URL}" \
    bash "${SCRIPT_DIR}/extract_skills_from_judgments.sh" "${STEP_JUDGMENT_JSON}"

run_step "8. Run trajectories with extracted skills" \
  env \
    MOBILEGYM_SKILL_OUTPUT_DIR="${SKILL_TRAJ_DIR}" \
    MOBILEGYM_SKILLS_DIR="${SKILLS_DIR}" \
    MOBILEGYM_REPEATS="${SKILL_REPEATS}" \
    MOBILEGYM_CLASS_WORKERS="${CLASS_WORKERS}" \
    MOBILEGYM_MODEL_BASE_URL="${MOBILEGYM_MODEL_BASE_URL}" \
    bash "${SCRIPT_DIR}/mobilegym/run_sample_batch_skill.sh"

run_step "9. Judge skill trajectories" \
  env \
    OUT_DIR="${SKILL_JUDGE_DIR}" \
    MERGED_OUTPUT="${SKILL_JUDGE_JSON}" \
    WORKERS="${WORKERS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${SCRIPT_DIR}/judge_trajs_parallel.sh" "${SKILL_TRAJ_DIR}"

run_step "10. Count skill judge success rate" \
  bash "${SCRIPT_DIR}/count_mobilegym_orm_success.sh" \
    --input "${SKILL_JUDGE_JSON}" \
    --manifest "${SKILL_TRAJ_DIR}/batch_runs.jsonl" \
    --repeats "${SKILL_REPEATS}" \
    --output-dir "${SKILL_JUDGE_DIR}"

echo
echo "[finish] flow completed"
echo "[output] first trajectories: ${TRAJ_DIR}"
echo "[output] first judge results: ${JUDGE_DIR}"
echo "[output] skills: ${SKILLS_DIR}"
echo "[output] skill trajectories: ${SKILL_TRAJ_DIR}"
echo "[output] skill judge results: ${SKILL_JUDGE_DIR}"
