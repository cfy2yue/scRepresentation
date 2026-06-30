#!/usr/bin/env bash
# Local/dev script; not part of public API
set -euo pipefail
SCFM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DELIVERY_ROOT="$(cd "${SCFM_ROOT}/.." && pwd)"
export SCFM_OUTPUT_ROOT="${SCFM_OUTPUT_ROOT:-$DELIVERY_ROOT/scFM_output}"
export LATENT_BENCH_OUTPUT_ROOT="${LATENT_BENCH_OUTPUT_ROOT:-$SCFM_OUTPUT_ROOT}"
LOGS="${LATENT_BENCH_OUTPUT_ROOT}/logs"
RUNS="${LATENT_BENCH_OUTPUT_ROOT}/embedding_runs"
mkdir -p "${LOGS}"
SCRIPT="${SCFM_ROOT}/scripts/run_embedding_full_nohup.sh"
NOHUP_OUT="${LOGS}/nohup_embedding_main.out"
QUEUE_LOG="${LOGS}/embedding_queue.log"
PID_FILE="${LOGS}/embedding_queue.pid"
STATUS_JSONL="${RUNS}/run_status.jsonl"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Already running PID $(cat "${PID_FILE}")" >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
for f in "${NOHUP_OUT}" "${QUEUE_LOG}" "${STATUS_JSONL}"; do
  if [[ -s "${f}" ]]; then
    mv "${f}" "${f}.${stamp}.bak"
  fi
done
rm -f "${PID_FILE}"

chmod +x "${SCRIPT}" 2>/dev/null || true
nohup bash "${SCRIPT}" >> "${NOHUP_OUT}" 2>&1 &
echo $! > "${PID_FILE}"
echo "Started PID $(cat "${PID_FILE}"). Log: ${NOHUP_OUT} queue: ${LOGS}/embedding_queue.log tail -f both"
