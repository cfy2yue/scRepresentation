#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

NOT_BEFORE="2026-06-24 04:31:00"

"${PYTHON}" - "${NOT_BEFORE}" <<'PY'
from __future__ import annotations

import sys
from datetime import datetime

not_before = datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M:%S")
now = datetime.now()
if now < not_before:
    print(
        f"Refusing to check active long jobs before {not_before:%F %T} CST; "
        f"now={now:%F %T} CST",
        file=sys.stderr,
    )
    raise SystemExit(3)
PY

GENERAL_RUN=${ROOT}/runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_general_exposure_cap_v2_3k_seed42
HUNGARIAN_RUN=${ROOT}/runs/latentfm_xverse_ot_pairmode_hungarian_20260624/xverse_otpair_hungarian_2k_seed42

GENERAL_TRAIN_EXIT=${GENERAL_RUN}/xverse_scaling_general_exposure_cap_v2_3k_seed42.EXIT_CODE
GENERAL_POSTHOC_EXIT=${GENERAL_RUN}/POSTHOC_EXIT_CODE
HUNGARIAN_TRAIN_EXIT=${HUNGARIAN_RUN}/xverse_otpair_hungarian_2k_seed42.EXIT_CODE
HUNGARIAN_POSTHOC_EXIT=${HUNGARIAN_RUN}/POSTHOC_EXIT_CODE

echo "[$(date '+%F %T %Z')] active long-job marker check"

if [[ -e "${GENERAL_TRAIN_EXIT}" ]]; then
  echo "general_train_exit=$(cat "${GENERAL_TRAIN_EXIT}")"
else
  echo "general_train_exit=still_running"
fi
if [[ -e "${GENERAL_POSTHOC_EXIT}" ]]; then
  echo "general_posthoc_exit=$(cat "${GENERAL_POSTHOC_EXIT}")"
else
  echo "general_posthoc_exit=posthoc_not_complete"
fi

if [[ -e "${HUNGARIAN_TRAIN_EXIT}" ]]; then
  echo "hungarian_train_exit=$(cat "${HUNGARIAN_TRAIN_EXIT}")"
else
  echo "hungarian_train_exit=still_running"
fi
if [[ -e "${HUNGARIAN_POSTHOC_EXIT}" ]]; then
  echo "hungarian_posthoc_exit=$(cat "${HUNGARIAN_POSTHOC_EXIT}")"
else
  echo "hungarian_posthoc_exit=posthoc_not_complete"
fi

if [[ -e "${GENERAL_POSTHOC_EXIT}" && "$(cat "${GENERAL_POSTHOC_EXIT}")" == "0" ]]; then
  echo "Summarizing general exposure-cap v2 internal decision"
  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_scaling_count_smokes_20260624.py"
else
  echo "General exposure-cap v2 summary skipped because posthoc is not complete with exit 0"
fi

if [[ -e "${HUNGARIAN_POSTHOC_EXIT}" && "$(cat "${HUNGARIAN_POSTHOC_EXIT}")" == "0" ]]; then
  echo "Summarizing Hungarian OT decision"
  LATENTFM_XVERSE_OTPAIR_RUN_ROOT="${ROOT}/runs/latentfm_xverse_ot_pairmode_hungarian_20260624" \
  LATENTFM_XVERSE_OTPAIR_RUNS=xverse_otpair_hungarian_2k_seed42 \
  LATENTFM_XVERSE_OTPAIR_DECISION_JSON="${ROOT}/reports/latentfm_xverse_ot_pairmode_hungarian_decision_20260624.json" \
  LATENTFM_XVERSE_OTPAIR_DECISION_MD="${ROOT}/reports/LATENTFM_XVERSE_OT_PAIRMODE_HUNGARIAN_DECISION_20260624.md" \
  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_ot_pairmode_smokes_20260624.py"
else
  echo "Hungarian OT summary skipped because posthoc is not complete with exit 0"
fi
