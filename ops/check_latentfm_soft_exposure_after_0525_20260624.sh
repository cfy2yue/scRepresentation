#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

NOT_BEFORE_EPOCH=$(date -d '2026-06-24 05:25:00' +%s)
NOW_EPOCH=$(date +%s)
if (( NOW_EPOCH < NOT_BEFORE_EPOCH )); then
  echo "Refusing to check soft-exposure long jobs before 2026-06-24 05:25:00 CST; now=$(date '+%F %T %Z')" >&2
  exit 3
fi

RUN_ROOT=${ROOT}/runs/latentfm_xverse_soft_exposure_smokes_20260624
P090=${RUN_ROOT}/xverse_softvisit_p090_no_cap_3k_seed42
P085=${RUN_ROOT}/xverse_softvisit_p085_no_cap_3k_seed42

echo "[$(date '+%F %T %Z')] soft-exposure marker check"

check_one() {
  local label=$1
  local run_dir=$2
  local train_marker=$3
  local posthoc_marker=${run_dir}/POSTHOC_EXIT_CODE
  local train_exit
  local posthoc_exit
  train_exit=$(cat "${train_marker}" 2>/dev/null || true)
  posthoc_exit=$(cat "${posthoc_marker}" 2>/dev/null || true)
  echo "${label}_train_exit=${train_exit:-missing}"
  echo "${label}_posthoc_exit=${posthoc_exit:-missing}"
}

check_one p090 "${P090}" "${P090}/xverse_softvisit_p090_no_cap_3k_seed42.EXIT_CODE"
check_one p085 "${P085}" "${P085}/xverse_softvisit_p085_no_cap_3k_seed42.EXIT_CODE"

p090_train=$(cat "${P090}/xverse_softvisit_p090_no_cap_3k_seed42.EXIT_CODE" 2>/dev/null || true)
p085_train=$(cat "${P085}/xverse_softvisit_p085_no_cap_3k_seed42.EXIT_CODE" 2>/dev/null || true)
p090_post=$(cat "${P090}/POSTHOC_EXIT_CODE" 2>/dev/null || true)
p085_post=$(cat "${P085}/POSTHOC_EXIT_CODE" 2>/dev/null || true)

if [[ "${p090_train}" == "0" && "${p085_train}" == "0" && "${p090_post}" == "0" && "${p085_post}" == "0" ]]; then
  echo "Summarizing soft-exposure decisions"
  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_soft_exposure_smokes_20260624.py"
else
  echo "Not all soft-exposure jobs have train/posthoc exit 0; not summarizing yet."
fi
