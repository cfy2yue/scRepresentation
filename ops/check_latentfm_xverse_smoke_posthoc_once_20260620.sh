#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_xverse_smoke_posthoc_20260620
SESSION=latentfm_xverse_smoke_posthoc_20260620
RUN_DIR=${ROOT}/CoupledFM/output/latentfm_runs/xverse_smoke_20260620/xverse_comp006_endpoint5_2k_smoke
REPORT=${REPORT:-"${ROOT}/reports/LATENTFM_XVERSE_SMOKE_POSTHOC_ONE_SHOT_STATUS_$(date +%Y%m%d_%H%M%S).txt"}
MIN_INTERVAL_SECONDS=${MIN_INTERVAL_SECONDS:-1800}
STAMP=${STAMP:-"${RUN_ROOT}/LAST_ONE_SHOT_CHECK_EPOCH"}

mkdir -p "$(dirname "${REPORT}")"

now_epoch="$(date +%s)"
if [[ -f "${STAMP}" && "${FORCE_ONE_SHOT_CHECK:-0}" != "1" ]]; then
  last_epoch="$(cat "${STAMP}")"
  if [[ "${last_epoch}" =~ ^[0-9]+$ ]]; then
    next_epoch=$((last_epoch + MIN_INTERVAL_SECONDS))
    if (( now_epoch < next_epoch )); then
      {
        echo "Too early for another 30-minute cadence check."
        echo "now=$(date '+%F %T %Z')"
        echo "next_allowed=$(date -d "@${next_epoch}" '+%F %T %Z')"
      } | tee "${REPORT}"
      exit 11
    fi
  fi
fi

{
  echo "[$(date '+%F %T %Z')] one-shot LatentFM xverse smoke posthoc status check"
  echo "report=${REPORT}"
  echo
  tmux ls 2>/dev/null | grep "${SESSION}" || true
  echo
  if [[ -f "${RUN_ROOT}/EXIT_CODE" ]]; then
    echo "EXIT_CODE=$(cat "${RUN_ROOT}/EXIT_CODE")"
    [[ -f "${RUN_ROOT}/FINISHED" ]] && echo "FINISHED=$(cat "${RUN_ROOT}/FINISHED")"
  else
    echo "still-running-or-not-started"
    [[ -f "${RUN_ROOT}/STARTED" ]] && echo "STARTED=$(cat "${RUN_ROOT}/STARTED")"
  fi
  echo
  for p in \
    "${RUN_DIR}/posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
    "${RUN_DIR}/posthoc_eval/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
    "${ROOT}/reports/LATENTFM_STABLECAPS_SELECTION_AUDIT_xverse_comp006_endpoint5_2k_smoke_20260620.md"; do
    if [[ -s "${p}" ]]; then
      echo "present ${p}"
    else
      echo "missing ${p}"
    fi
  done
  echo
  if [[ -f "${RUN_ROOT}/logs/run.log" ]]; then
    echo "tail ${RUN_ROOT}/logs/run.log"
    tail -n 120 "${RUN_ROOT}/logs/run.log"
  else
    echo "missing ${RUN_ROOT}/logs/run.log"
  fi
} | tee "${REPORT}"

echo "${now_epoch}" > "${STAMP}"
