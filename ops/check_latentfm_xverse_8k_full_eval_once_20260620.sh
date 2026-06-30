#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_xverse_8k_full_eval_20260620
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620
LOG_ROOT=${ROOT}/logs/latentfm_xverse_8k_full_eval_20260620
RUN_NAME=${RUN_NAME:-xverse_comp006_endpoint5_8k_seed42_fulleval}
SESSION=${SESSION:-lfm_${RUN_NAME}}
REPORT=${REPORT:-"${ROOT}/reports/LATENTFM_XVERSE_8K_FULL_EVAL_ONE_SHOT_STATUS_$(date +%Y%m%d_%H%M%S).txt"}
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
  echo "[$(date '+%F %T %Z')] one-shot LatentFM xverse 8k full-eval status check"
  echo "report=${REPORT}"
  echo
  tmux ls 2>/dev/null | grep "${SESSION}" || true
  echo
  if [[ -f "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE" ]]; then
    echo "EXIT_CODE=$(cat "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE")"
    [[ -f "${RUN_ROOT}/${RUN_NAME}.FINISHED" ]] && echo "FINISHED=$(cat "${RUN_ROOT}/${RUN_NAME}.FINISHED")"
  else
    echo "still-running-or-not-started"
    [[ -f "${RUN_ROOT}/${RUN_NAME}.STARTED" ]] && echo "STARTED=$(cat "${RUN_ROOT}/${RUN_NAME}.STARTED")"
  fi
  echo
  for p in \
    "${OUT_ROOT}/${RUN_NAME}/best.pt" \
    "${OUT_ROOT}/${RUN_NAME}/latest.pt" \
    "${OUT_ROOT}/${RUN_NAME}/config.json" \
    "${OUT_ROOT}/${RUN_NAME}/iid_eval_results.json"; do
    if [[ -s "${p}" ]]; then
      echo "present ${p}"
    else
      echo "missing ${p}"
    fi
  done
  echo
  if [[ -f "${LOG_ROOT}/${RUN_NAME}.log" ]]; then
    echo "tail ${LOG_ROOT}/${RUN_NAME}.log"
    tail -n 120 "${LOG_ROOT}/${RUN_NAME}.log"
  else
    echo "missing ${LOG_ROOT}/${RUN_NAME}.log"
  fi
} | tee "${REPORT}"

echo "${now_epoch}" > "${STAMP}"
