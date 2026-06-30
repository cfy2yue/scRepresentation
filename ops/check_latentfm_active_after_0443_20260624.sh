#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

NOT_BEFORE="2026-06-24 04:43:00"

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

echo "[$(date '+%F %T %Z')] active long-job marker check"

check_one() {
  local label=$1
  local run_dir=$2
  local exit_file=$3
  local posthoc_file=${run_dir}/POSTHOC_EXIT_CODE
  if [[ -e "${exit_file}" ]]; then
    echo "${label}_train_exit=$(cat "${exit_file}")"
  else
    echo "${label}_train_exit=still_running"
  fi
  if [[ -e "${posthoc_file}" ]]; then
    echo "${label}_posthoc_exit=$(cat "${posthoc_file}")"
  else
    echo "${label}_posthoc_exit=posthoc_not_complete"
  fi
}

HUNGARIAN=${ROOT}/runs/latentfm_xverse_ot_pairmode_hungarian_20260624/xverse_otpair_hungarian_2k_seed42
W05=${ROOT}/runs/latentfm_xverse_cap120_anchor_replay_smokes_20260624/xverse_cap120_anchor_replay_w05_2k_seed42
W10=${ROOT}/runs/latentfm_xverse_cap120_anchor_replay_smokes_20260624/xverse_cap120_anchor_replay_w10_2k_seed42

check_one hungarian "${HUNGARIAN}" "${HUNGARIAN}/xverse_otpair_hungarian_2k_seed42.EXIT_CODE"
check_one replay_w05 "${W05}" "${W05}/xverse_cap120_anchor_replay_w05_2k_seed42.EXIT_CODE"
check_one replay_w10 "${W10}" "${W10}/xverse_cap120_anchor_replay_w10_2k_seed42.EXIT_CODE"

if [[ -e "${HUNGARIAN}/POSTHOC_EXIT_CODE" && "$(cat "${HUNGARIAN}/POSTHOC_EXIT_CODE")" == "0" ]]; then
  echo "Summarizing Hungarian OT decision"
  LATENTFM_XVERSE_OTPAIR_RUN_ROOT="${ROOT}/runs/latentfm_xverse_ot_pairmode_hungarian_20260624" \
  LATENTFM_XVERSE_OTPAIR_RUNS=xverse_otpair_hungarian_2k_seed42 \
  LATENTFM_XVERSE_OTPAIR_DECISION_JSON="${ROOT}/reports/latentfm_xverse_ot_pairmode_hungarian_decision_20260624.json" \
  LATENTFM_XVERSE_OTPAIR_DECISION_MD="${ROOT}/reports/LATENTFM_XVERSE_OT_PAIRMODE_HUNGARIAN_DECISION_20260624.md" \
  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_ot_pairmode_smokes_20260624.py"
else
  echo "Hungarian OT summary skipped because posthoc is not complete with exit 0"
fi

if [[ -e "${W05}/POSTHOC_EXIT_CODE" && "$(cat "${W05}/POSTHOC_EXIT_CODE")" == "0" && -e "${W10}/POSTHOC_EXIT_CODE" && "$(cat "${W10}/POSTHOC_EXIT_CODE")" == "0" ]]; then
  echo "Summarizing cap120 anchor-replay decisions"
  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_cap120_anchor_replay_smokes_20260624.py"
else
  echo "Cap120 anchor-replay summary skipped because both posthocs are not complete with exit 0"
fi
