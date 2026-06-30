#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

OT_RUN_ROOT=${ROOT}/runs/latentfm_xverse_ot_pairmode_random_rerun_20260624
SCALING_RUN_ROOT=${ROOT}/runs/latentfm_scaling_highthroughput_smokes_refill_20260624

echo "[$(date '+%F %T %Z')] active refill marker-only checkpoint"
echo
echo "## tmux"
tmux ls 2>/dev/null || true
echo
echo "## gpu"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
echo
echo "## memory"
free -h
echo
echo "## exit markers"
find "${OT_RUN_ROOT}" "${SCALING_RUN_ROOT}" -maxdepth 2 \
  \( -name '*EXIT_CODE' -o -name 'POSTHOC_EXIT_CODE' \) \
  -print -exec sh -c 'printf "%s: " "$1"; cat "$1"' _ {} \; 2>/dev/null || true
echo
echo "## decision summaries"
LATENTFM_XVERSE_OTPAIR_RUN_ROOT="${OT_RUN_ROOT}" \
LATENTFM_XVERSE_OTPAIR_RUNS=xverse_otpair_random_2k_seed42 \
LATENTFM_XVERSE_OTPAIR_DECISION_JSON=${ROOT}/reports/latentfm_xverse_ot_pairmode_random_rerun_decision_20260624.json \
LATENTFM_XVERSE_OTPAIR_DECISION_MD=${ROOT}/reports/LATENTFM_XVERSE_OT_PAIRMODE_RANDOM_RERUN_DECISION_20260624.md \
"${PYTHON}" "${ROOT}/ops/summarize_latentfm_xverse_ot_pairmode_smokes_20260624.py"

LATENTFM_SCALING_HT_RUN_ROOT="${SCALING_RUN_ROOT}" \
LATENTFM_SCALING_HT_DECISION_JSON=${ROOT}/reports/latentfm_scaling_highthroughput_smokes_refill_decision_20260624.json \
LATENTFM_SCALING_HT_DECISION_MD=${ROOT}/reports/LATENTFM_SCALING_HIGH_THROUGHPUT_SMOKES_REFILL_DECISION_20260624.md \
"${PYTHON}" "${ROOT}/ops/summarize_latentfm_scaling_highthroughput_smokes_20260624.py"

echo
echo "Do not tail long logs from this helper. If a decision is still pending, wait for the normal long-task cadence or work on independent branches."
