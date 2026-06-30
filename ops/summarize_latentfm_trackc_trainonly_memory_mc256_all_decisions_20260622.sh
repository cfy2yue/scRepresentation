#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

"${PY}" "${ROOT}/ops/summarize_latentfm_trackc_manifest_decisions_20260622.py" \
  --manifest "${ROOT}/reports/latentfm_trackc_trainonly_memory_parallel_mc256_manifest_20260622.jsonl" \
  --manifest "${ROOT}/reports/latentfm_trackc_trainonly_memory_parallel_mc256_ext_manifest_20260622.jsonl" \
  --out-md "${ROOT}/reports/LATENTFM_TRACKC_TRAINONLY_MEMORY_MC256_ALL_DECISION_SUMMARY_20260622.md" \
  --out-csv "${ROOT}/reports/latentfm_trackc_trainonly_memory_mc256_all_decision_summary_20260622.csv"
