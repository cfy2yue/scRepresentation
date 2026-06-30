#!/usr/bin/env bash
# Local/dev script; not part of public API
# Formal multi-GPU embedding queue (nohup-friendly).
# Uses only h5ad with materialized X (see manifest_with_X.jsonl).

set -euo pipefail

export PYTHONUNBUFFERED=1
SCFM_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DELIVERY_ROOT="$(cd "${SCFM_ROOT}/.." && pwd)"
export SCFM_PRETRAINED_ROOT="${SCFM_PRETRAINED_ROOT:-$DELIVERY_ROOT/scFM_pretrained}"
export SCFM_OUTPUT_ROOT="${SCFM_OUTPUT_ROOT:-$DELIVERY_ROOT/scFM_output}"
export SCFM_DATA_ROOT="${SCFM_DATA_ROOT:-$DELIVERY_ROOT/scFM_data}"
export SCFM_THIRD_PARTY_ROOT="${SCFM_THIRD_PARTY_ROOT:-$DELIVERY_ROOT/scFM_third_party}"
export COUPLEDFM_PRETRAINED_ROOT="${COUPLEDFM_PRETRAINED_ROOT:-$SCFM_PRETRAINED_ROOT}"
export LATENT_BENCH_OUTPUT_ROOT="${LATENT_BENCH_OUTPUT_ROOT:-$SCFM_OUTPUT_ROOT}"
PYTHON="${PYTHON:-python}"

TOOLS="${SCFM_ROOT}/fm/tools"
RUNS="${LATENT_BENCH_OUTPUT_ROOT}/embedding_runs"
LOGS="${LATENT_BENCH_OUTPUT_ROOT}/logs"
EMB="${LATENT_BENCH_OUTPUT_ROOT}/embeddings"

MANIFEST="${RUNS}/manifest_with_X.jsonl"
PREFLIGHT="${RUNS}/preflight.json"
QUEUE_LOG="${LOGS}/embedding_queue.log"

mkdir -p "${RUNS}" "${LOGS}" "${EMB}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: missing ${MANIFEST}" >&2
  echo "Run: cd ${TOOLS} && python3 preflight_embedding.py" >&2
  exit 1
fi
if [[ ! -f "${PREFLIGHT}" ]]; then
  echo "ERROR: missing ${PREFLIGHT}" >&2
  exit 1
fi

cd "${TOOLS}"

"${PYTHON}" validate_resources.py --skip-import-test

# Optional: refresh preflight + full manifest (does not overwrite manifest_with_X unless you regenerate it)
# python3 preflight_embedding.py

"${PYTHON}" submit_embedding_queue.py \
  --manifest "${MANIFEST}" \
  --preflight "${PREFLIGHT}" \
  --export-root "${EMB}" \
  --status-jsonl "${RUNS}/run_status.jsonl" \
  --log-file "${QUEUE_LOG}" \
  --gpus 0 1 2 3 \
  --device cuda \
  --batch-size "${BATCH_SIZE:-4}" \
  --skip-existing \
  --abort-after-consecutive-fails "${ABORT_CONSEC_FAILS:-3}" \
  "$@"
