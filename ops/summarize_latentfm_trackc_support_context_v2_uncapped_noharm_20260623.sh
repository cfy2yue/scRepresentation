#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

LABEL=${LATENTFM_TRACKC_V2_UNCAPPED_LABEL:-latentfm_trackc_support_context_v2_uncapped_noharm_20260623}
INDEX_JSON=${LATENTFM_TRACKC_V2_UNCAPPED_INDEX_JSON:-${ROOT}/reports/${LABEL}/uncapped_posthoc_index.json}
OUT_JSON=${LATENTFM_TRACKC_V2_UNCAPPED_OUT_JSON:-${ROOT}/reports/latentfm_trackc_support_context_v2_uncapped_noharm_decision_20260623.json}
OUT_MD=${LATENTFM_TRACKC_V2_UNCAPPED_OUT_MD:-${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_UNCAPPED_NOHARM_DECISION_20260623.md}
BOOT_DIR=${LATENTFM_TRACKC_V2_UNCAPPED_BOOT_DIR:-${ROOT}/reports/latentfm_trackc_support_context_v2_uncapped_noharm_bootstrap_20260623}
REPORT_TITLE=${LATENTFM_TRACKC_V2_UNCAPPED_REPORT_TITLE:-LatentFM Track C Support-Context V2 Uncapped Canonical No-Harm Decision}

"${PYTHON}" "${ROOT}/ops/summarize_latentfm_trackc_routefocus_uncapped_noharm_20260622.py" \
  --index-json "${INDEX_JSON}" \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}" \
  --boot-dir "${BOOT_DIR}" \
  --report-title "${REPORT_TITLE}" \
  --n-boot "${LATENTFM_BOOTSTRAP_N:-2000}" \
  --seed "${LATENTFM_BOOTSTRAP_SEED:-42}" \
  --python "${PYTHON}" \
  ${LATENTFM_TRACKC_V2_UNCAPPED_SPLIT_GROUPS:+--split-groups ${LATENTFM_TRACKC_V2_UNCAPPED_SPLIT_GROUPS}} \
  ${LATENTFM_TRACKC_V2_UNCAPPED_FAMILY_GROUPS:+--family-groups ${LATENTFM_TRACKC_V2_UNCAPPED_FAMILY_GROUPS}}
