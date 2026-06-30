#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

"${PYTHON}" "${ROOT}/ops/summarize_latentfm_trackc_routefocus_uncapped_noharm_20260622.py" \
  --index-json "${ROOT}/reports/latentfm_trackc_trainonly_memory_mc256_uncapped_noharm_20260622/uncapped_posthoc_index.json" \
  --out-json "${ROOT}/reports/latentfm_trackc_trainonly_memory_mc256_uncapped_noharm_decision_20260622.json" \
  --out-md "${ROOT}/reports/LATENTFM_TRACKC_TRAINONLY_MEMORY_MC256_UNCAPPED_NOHARM_DECISION_20260622.md" \
  --boot-dir "${ROOT}/reports/latentfm_trackc_trainonly_memory_mc256_uncapped_noharm_bootstrap_20260622" \
  --n-boot 2000 \
  --seed 42
