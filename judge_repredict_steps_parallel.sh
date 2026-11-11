#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/judge_results_parallel}"
INPUT_JSON="${1:-${INPUT_JSON:-${RESULTS_DIR}/mobilegym_edited_repredictions.json}}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULTS_DIR}/mobilegym_repredicted_step_judgments.json}"
LOG_FILE="${LOG_FILE:-${RESULTS_DIR}/mobilegym_judge_step.log}"
WORKERS="${WORKERS:-8}"
INDENT="${INDENT:-2}"
RESUME="${RESUME:-0}"
JUDGE_LLM="${JUDGE_LLM:-doubao}"
TIMEOUT="${TIMEOUT:-300}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MODEL_BASE_URL="${MODEL_BASE_URL:-*}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-27B-vllm}"
API_KEY="${API_KEY:-EMPTY}"

CHUNKS_DIR="${CHUNKS_DIR:-${RESULTS_DIR}/judge_step_chunks}"
CHUNK_INPUT_DIR="${CHUNKS_DIR}/inputs"
CHUNK_OUTPUT_DIR="${CHUNKS_DIR}/outputs"
CHUNK_LOG_DIR="${CHUNKS_DIR}/logs"

if [[ ! "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] WORKERS must be a positive integer, got: ${WORKERS}" >&2
  exit 1
fi

if [[ ! "${INDENT}" =~ ^[0-9]+$ ]]; then
  echo "[error] INDENT must be a non-negative integer, got: ${INDENT}" >&2
  exit 1
fi

if [[ "${RESUME}" != "0" && "${RESUME}" != "1" ]]; then
  echo "[error] RESUME must be 0 or 1, got: ${RESUME}" >&2
  exit 1
fi

if [[ "${JUDGE_LLM}" != "27b" && "${JUDGE_LLM}" != "doubao" && "${JUDGE_LLM}" != "qwen" ]]; then
  echo "[error] JUDGE_LLM must be one of: 27b, doubao, qwen; got: ${JUDGE_LLM}" >&2
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

if [[ ! -f "${ROOT_DIR}/mobilegym_judge_step.py" ]]; then
  echo "[error] mobilegym_judge_step.py not found under: ${ROOT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_JSON}" ]]; then
  echo "[error] reprediction JSON not found: ${INPUT_JSON}" >&2
  echo "[hint] run repredict_errors_parallel.sh first, or pass INPUT_JSON/path as the first argument" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}" "$(dirname "${OUTPUT_JSON}")" "$(dirname "${LOG_FILE}")"

CHUNKS_DIR_ABS="$(mkdir -p "${CHUNKS_DIR}" && cd "${CHUNKS_DIR}" && pwd)"
RESULTS_DIR_ABS="$(mkdir -p "${RESULTS_DIR}" && cd "${RESULTS_DIR}" && pwd)"
case "${CHUNKS_DIR_ABS}" in
  "${RESULTS_DIR_ABS}"/judge_step_chunks|"${RESULTS_DIR_ABS}"/judge_step_chunks/*) ;;
  *)
    echo "[error] refusing to clean unexpected CHUNKS_DIR: ${CHUNKS_DIR_ABS}" >&2
    exit 1
    ;;
esac

if [[ "${RESUME}" == "1" ]]; then
  echo "[info] resume: keeping existing chunk outputs under ${CHUNKS_DIR_ABS}"
else
  rm -rf "${CHUNKS_DIR_ABS:?}"/*
fi
mkdir -p "${CHUNK_INPUT_DIR}" "${CHUNK_OUTPUT_DIR}" "${CHUNK_LOG_DIR}"

extra_args=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  extra_args+=(--debug)
fi

echo "[info] input: ${INPUT_JSON}"
echo "[info] output: ${OUTPUT_JSON}"
echo "[info] log: ${LOG_FILE}"
echo "[info] workers: ${WORKERS}"
echo "[info] resume: ${RESUME}"
echo "[info] judge_llm: ${JUDGE_LLM}"
echo "[info] timeout: ${TIMEOUT}"
echo "[info] model_base_url: ${MODEL_BASE_URL}"
echo "[info] model_name: ${MODEL_NAME}"

"${PYTHON_BIN}" - "${INPUT_JSON}" "${CHUNK_INPUT_DIR}" "${WORKERS}" "${INDENT}" <<'PY'
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1]).expanduser().resolve()
chunk_dir = Path(sys.argv[2]).expanduser().resolve()
workers = int(sys.argv[3])
indent_arg = int(sys.argv[4])
json_indent = None if indent_arg == 0 else indent_arg

with input_path.open("r", encoding="utf-8-sig") as f:
    payload = json.load(f)

if isinstance(payload, dict) and isinstance(payload.get("results"), list):
    source_items = payload["results"]
elif isinstance(payload, list):
    source_items = payload
elif isinstance(payload, dict):
    source_items = [payload]
else:
    raise SystemExit("[error] reprediction JSON must be an object, a list, or an object with results")

items = [item for item in source_items if isinstance(item, dict) and isinstance(item.get("variants"), dict)]
if not items:
    raise SystemExit("[error] no reprediction item with variants found")

chunk_count = min(workers, len(items))
chunk_size = (len(items) + chunk_count - 1) // chunk_count
for index in range(chunk_count):
    start = index * chunk_size
    end = min(start + chunk_size, len(items))
    chunk_items = items[start:end]
    if not chunk_items:
        continue
    chunk_payload = {
        "input": str(input_path),
        "chunk_index": index,
        "chunk_count": chunk_count,
        "count": len(chunk_items),
        "results": chunk_items,
    }
    chunk_path = chunk_dir / f"chunk_{index:03d}.json"
    with chunk_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_payload, f, ensure_ascii=False, indent=json_indent)
        f.write("\n")

print(f"[split] {len(items)} reprediction item(s) -> {chunk_count} chunk(s)")
PY

chunk_inputs=("${CHUNK_INPUT_DIR}"/chunk_*.json)
if (( ${#chunk_inputs[@]} == 0 )); then
  echo "[error] no judge chunks generated under: ${CHUNK_INPUT_DIR}" >&2
  exit 1
fi

run_one_chunk() {
  local chunk_input="$1"
  local chunk_name
  local chunk_output
  local chunk_log

  chunk_name="$(basename "${chunk_input}" .json)"
  chunk_output="${CHUNK_OUTPUT_DIR}/${chunk_name}.step_judged.json"
  chunk_log="${CHUNK_LOG_DIR}/${chunk_name}.log"

  if [[ "${RESUME}" == "1" && -s "${chunk_output}" ]]; then
    if "${PYTHON_BIN}" - "${chunk_output}" >/dev/null 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8-sig") as f:
    payload = json.load(f)
if not (isinstance(payload, dict) and isinstance(payload.get("results"), list)):
    raise SystemExit(1)
PY
    then
      echo "[skip] ${chunk_name} already judged"
      return 0
    fi
    echo "[rerun] ${chunk_name} existing output is invalid"
  fi

  echo "[judge-step] ${chunk_name}"
  if "${PYTHON_BIN}" -u "${ROOT_DIR}/mobilegym_judge_step.py" \
      --input "${chunk_input}" \
      --output "${chunk_output}" \
      --judge-llm "${JUDGE_LLM}" \
      --model-base-url "${MODEL_BASE_URL}" \
      --model-name "${MODEL_NAME}" \
      --api-key "${API_KEY}" \
      --timeout "${TIMEOUT}" \
      --max-retries "${MAX_RETRIES}" \
      --max-tokens "${MAX_TOKENS}" \
      --indent "${INDENT}" \
      "${extra_args[@]}" \
      >"${chunk_log}" 2>&1; then
    echo "[ok] ${chunk_name} -> ${chunk_output}"
    return 0
  fi

  echo "[fail] ${chunk_name}; see ${chunk_log}" >&2
  return 1
}

active=0
fail_count=0

for chunk_input in "${chunk_inputs[@]}"; do
  run_one_chunk "${chunk_input}" &
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

if (( fail_count > 0 )); then
  echo "[error] ${fail_count} judge chunk(s) failed" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${INPUT_JSON}" "${CHUNK_OUTPUT_DIR}" "${OUTPUT_JSON}" "${INDENT}" <<'PY' 2>&1 | tee "${LOG_FILE}"
import json
import sys
from pathlib import Path

input_path = Path(sys.argv[1]).expanduser().resolve()
chunk_output_dir = Path(sys.argv[2]).expanduser().resolve()
output_path = Path(sys.argv[3]).expanduser().resolve()
indent_arg = int(sys.argv[4])
json_indent = None if indent_arg == 0 else indent_arg

results = []
metadata = {}
for path in sorted(chunk_output_dir.glob("chunk_*.step_judged.json")):
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not (isinstance(payload, dict) and isinstance(payload.get("results"), list)):
        raise SystemExit(f"[error] invalid judge chunk output: {path}")
    if not metadata:
        metadata = {key: value for key, value in payload.items() if key not in {"results", "count", "input"}}
    results.extend(payload["results"])

merged = {
    "input": str(input_path),
    **metadata,
    "count": len(results),
    "results": results,
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=json_indent)
    f.write("\n")

print(f"[merge] {len(results)} result(s) -> {output_path}")
PY

echo "[finish] step judgments -> ${OUTPUT_JSON}"
