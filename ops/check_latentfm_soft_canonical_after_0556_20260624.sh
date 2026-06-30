#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

NOT_BEFORE_EPOCH=$(date -d '2026-06-24 05:56:00' +%s)
NOW_EPOCH=$(date +%s)
if (( NOW_EPOCH < NOT_BEFORE_EPOCH )); then
  echo "Refusing to check soft canonical no-harm before 2026-06-24 05:56:00 CST; now=$(date '+%F %T %Z')" >&2
  exit 3
fi

RUN_DIR=${ROOT}/runs/latentfm_xverse_soft_exposure_canonical_noharm_20260624/xverse_softvisit_p085_no_cap_3k_seed42
exit_code=$(cat "${RUN_DIR}/POSTHOC_EXIT_CODE" 2>/dev/null || true)
echo "[$(date '+%F %T %Z')] soft_canonical_posthoc_exit=${exit_code:-missing}"

if [[ "${exit_code}" == "0" ]]; then
  echo "Summarizing soft canonical no-harm decision"
  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_soft_exposure_canonical_noharm_20260624.py"
else
  echo "Soft canonical no-harm is not complete with exit 0; not summarizing yet."
fi
