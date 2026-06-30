#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

"${PYTHON}" "${ROOT}/ops/summarize_latentfm_trackc_routefocus_uncapped_noharm_20260622.py" \
  --index-json "${ROOT}/reports/latentfm_trackc_support_film_uncapped_noharm_20260623/uncapped_posthoc_index.json" \
  --out-json "${ROOT}/reports/latentfm_trackc_support_film_uncapped_noharm_decision_20260623.json" \
  --out-md "${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_FILM_UNCAPPED_NOHARM_DECISION_20260623.md" \
  --boot-dir "${ROOT}/reports/latentfm_trackc_support_film_uncapped_noharm_bootstrap_20260623" \
  --report-title "LatentFM Track C Support-FiLM Uncapped Canonical No-Harm Decision" \
  --n-boot "${LATENTFM_BOOTSTRAP_N:-2000}" \
  --seed "${LATENTFM_BOOTSTRAP_SEED:-42}" \
  --python "${PYTHON}"
