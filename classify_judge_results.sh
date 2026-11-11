#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/judge_results_parallel}"
INPUT_JSON="${1:-${INPUT_JSON:-${RESULTS_DIR}/mobilegym_skill_result.json}}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULTS_DIR}/mobilegym_error_classification.json}"
CLASSIFY_MD="${CLASSIFY_MD:-${ROOT_DIR}/classify.md}"
LOG_FILE="${LOG_FILE:-${RESULTS_DIR}/mobilegym_classify.log}"
WORKERS="${WORKERS:-8}"
INDENT="${INDENT:-2}"
RESUME="${RESUME:-0}"
API_PROVIDER="${API_PROVIDER:-doubao}"
API_TIMEOUT="${API_TIMEOUT:-600}"

CHUNKS_DIR="${CHUNKS_DIR:-${RESULTS_DIR}/classify_chunks}"
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

if [[ "${API_PROVIDER}" != "doubao" && "${API_PROVIDER}" != "qwen_plus" ]]; then
  echo "[error] API_PROVIDER must be doubao or qwen_plus, got: ${API_PROVIDER}" >&2
  exit 1
fi

if [[ ! "${API_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] API_TIMEOUT must be a positive integer, got: ${API_TIMEOUT}" >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/mobilegym_classify.py" ]]; then
  echo "[error] mobilegym_classify.py not found under: ${ROOT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_JSON}" ]]; then
  echo "[error] MobileGym skill result JSON not found: ${INPUT_JSON}" >&2
  echo "[hint] run judge_trajs_parallel.sh first, or pass INPUT_JSON/path as the first argument" >&2
  exit 1
fi

if [[ ! -f "${CLASSIFY_MD}" ]]; then
  echo "[error] classify.md not found: ${CLASSIFY_MD}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}" "$(dirname "${OUTPUT_JSON}")" "$(dirname "${LOG_FILE}")"

CHUNKS_DIR_ABS="$(mkdir -p "${CHUNKS_DIR}" && cd "${CHUNKS_DIR}" && pwd)"
RESULTS_DIR_ABS="$(mkdir -p "${RESULTS_DIR}" && cd "${RESULTS_DIR}" && pwd)"
case "${CHUNKS_DIR_ABS}" in
  "${RESULTS_DIR_ABS}"/classify_chunks|"${RESULTS_DIR_ABS}"/classify_chunks/*) ;;
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

echo "[info] input: ${INPUT_JSON}"
echo "[info] output: ${OUTPUT_JSON}"
echo "[info] classify_md: ${CLASSIFY_MD}"
echo "[info] log: ${LOG_FILE}"
echo "[info] workers: ${WORKERS}"
echo "[info] resume: ${RESUME}"
echo "[info] api_provider: ${API_PROVIDER}"
echo "[info] api_timeout: ${API_TIMEOUT}"

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
    items = [item for item in payload["results"] if isinstance(item, dict)]
elif isinstance(payload, list):
    items = [item for item in payload if isinstance(item, dict)]
elif isinstance(payload, dict):
    items = [payload]
else:
    raise SystemExit("[error] MobileGym skill result JSON must be an object, a list, or an object with results")

if not items:
    raise SystemExit("[error] MobileGym skill result JSON contains no result objects")

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

print(f"[split] {len(items)} item(s) -> {chunk_count} chunk(s)")
PY

chunk_inputs=("${CHUNK_INPUT_DIR}"/chunk_*.json)
if (( ${#chunk_inputs[@]} == 0 )); then
  echo "[error] no classify chunks generated under: ${CHUNK_INPUT_DIR}" >&2
  exit 1
fi

run_one_chunk() {
  local chunk_input="$1"
  local chunk_name
  local chunk_output
  local chunk_log

  chunk_name="$(basename "${chunk_input}" .json)"
  chunk_output="${CHUNK_OUTPUT_DIR}/${chunk_name}.classified.json"
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
      echo "[skip] ${chunk_name} already classified"
      return 0
    fi
    echo "[rerun] ${chunk_name} existing output is invalid"
  fi

  echo "[classify] ${chunk_name}"
  if "${PYTHON_BIN}" -u "${ROOT_DIR}/mobilegym_classify.py" \
      --input "${chunk_input}" \
      --output "${chunk_output}" \
      --classify-md "${CLASSIFY_MD}" \
      --indent "${INDENT}" \
      --api-provider "${API_PROVIDER}" \
      --api-timeout "${API_TIMEOUT}" \
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
  echo "[error] ${fail_count} classify chunk(s) failed" >&2
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
for path in sorted(chunk_output_dir.glob("chunk_*.classified.json")):
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        results.extend(payload["results"])
    else:
        raise SystemExit(f"[error] invalid classify chunk output: {path}")

merged = {
    "input": str(input_path),
    "count": len(results),
    "results": results,
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=json_indent)
    f.write("\n")

print(f"[merge] {len(results)} result(s) -> {output_path}")
PY

echo "[finish] classification -> ${OUTPUT_JSON}"
