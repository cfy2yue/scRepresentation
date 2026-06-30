#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

NOT_BEFORE="2026-06-24 03:55:00"

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

JIANG_RUN=${ROOT}/runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_jiang_exposure_capped_3k_seed42
OT_RANDOM_RUN=${ROOT}/runs/latentfm_xverse_ot_pairmode_random_control_20260624/xverse_otpair_random_2k_seed42

JIANG_TRAIN_EXIT=${JIANG_RUN}/xverse_scaling_jiang_exposure_capped_3k_seed42.EXIT_CODE
JIANG_POSTHOC_EXIT=${JIANG_RUN}/POSTHOC_EXIT_CODE
OT_RANDOM_TRAIN_EXIT=${OT_RANDOM_RUN}/xverse_otpair_random_2k_seed42.EXIT_CODE
OT_RANDOM_POSTHOC_EXIT=${OT_RANDOM_RUN}/POSTHOC_EXIT_CODE

echo "[$(date '+%F %T %Z')] active long-job marker check"

if [[ -e "${JIANG_TRAIN_EXIT}" ]]; then
  echo "jiang_train_exit=$(cat "${JIANG_TRAIN_EXIT}")"
else
  echo "jiang_train_exit=still_running"
fi
if [[ -e "${JIANG_POSTHOC_EXIT}" ]]; then
  echo "jiang_posthoc_exit=$(cat "${JIANG_POSTHOC_EXIT}")"
else
  echo "jiang_posthoc_exit=posthoc_not_complete"
fi

if [[ -e "${OT_RANDOM_TRAIN_EXIT}" ]]; then
  echo "ot_random_train_exit=$(cat "${OT_RANDOM_TRAIN_EXIT}")"
else
  echo "ot_random_train_exit=still_running"
fi
if [[ -e "${OT_RANDOM_POSTHOC_EXIT}" ]]; then
  echo "ot_random_posthoc_exit=$(cat "${OT_RANDOM_POSTHOC_EXIT}")"
else
  echo "ot_random_posthoc_exit=posthoc_not_complete"
fi

if [[ -e "${JIANG_POSTHOC_EXIT}" && "$(cat "${JIANG_POSTHOC_EXIT}")" == "0" ]]; then
  echo "Summarizing Jiang exposure-capped internal decision"
  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_scaling_count_smokes_20260624.py"
else
  echo "Jiang exposure-capped summary skipped because posthoc is not complete with exit 0"
fi

if [[ -e "${OT_RANDOM_POSTHOC_EXIT}" && "$(cat "${OT_RANDOM_POSTHOC_EXIT}")" == "0" ]]; then
  echo "Summarizing OT random-control decision"
  LATENTFM_XVERSE_OTPAIR_RUN_ROOT="${ROOT}/runs/latentfm_xverse_ot_pairmode_random_control_20260624" \
  LATENTFM_XVERSE_OTPAIR_RUNS=xverse_otpair_random_2k_seed42 \
  LATENTFM_XVERSE_OTPAIR_DECISION_JSON="${ROOT}/reports/latentfm_xverse_ot_pairmode_random_control_decision_20260624.json" \
  LATENTFM_XVERSE_OTPAIR_DECISION_MD="${ROOT}/reports/LATENTFM_XVERSE_OT_PAIRMODE_RANDOM_CONTROL_DECISION_20260624.md" \
  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_ot_pairmode_smokes_20260624.py"
else
  echo "OT random-control summary skipped because posthoc is not complete with exit 0"
fi
