#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_xverse_sampling_sensitivity_20260620
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_sampling_sensitivity_20260620
LOG_ROOT=${ROOT}/logs/latentfm_xverse_sampling_sensitivity_20260620
RUN_A=xverse_comp006_endpoint5_visitcap8_power05_floor32_4k_seed42
RUN_B=xverse_comp006_endpoint5_visitcap8_power05_floor32_dsloss05_4k_seed42
REPORT=${REPORT:-"${ROOT}/reports/LATENTFM_XVERSE_SAMPLING_SENSITIVITY_ONE_SHOT_STATUS_$(date +%Y%m%d_%H%M%S).txt"}
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
  echo "[$(date '+%F %T %Z')] one-shot xverse sampling sensitivity status check"
  echo "report=${REPORT}"
  echo
  tmux ls 2>/dev/null | grep -E "lfm_${RUN_A}|lfm_${RUN_B}" || true
  echo
  for run in "${RUN_A}" "${RUN_B}"; do
    echo "## ${run}"
    if [[ -f "${RUN_ROOT}/${run}.EXIT_CODE" ]]; then
      echo "EXIT_CODE=$(cat "${RUN_ROOT}/${run}.EXIT_CODE")"
      [[ -f "${RUN_ROOT}/${run}.FINISHED" ]] && echo "FINISHED=$(cat "${RUN_ROOT}/${run}.FINISHED")"
    else
      echo "still-running-or-not-started"
      [[ -f "${RUN_ROOT}/${run}.STARTED" ]] && echo "STARTED=$(cat "${RUN_ROOT}/${run}.STARTED")"
    fi
    for p in \
      "${OUT_ROOT}/${run}/best.pt" \
      "${OUT_ROOT}/${run}/latest.pt" \
      "${OUT_ROOT}/${run}/config.json" \
      "${OUT_ROOT}/${run}/iid_eval_results.json"; do
      if [[ -s "${p}" ]]; then
        echo "present ${p}"
      else
        echo "missing ${p}"
      fi
    done
    if [[ -f "${LOG_ROOT}/${run}.log" ]]; then
      echo "tail ${LOG_ROOT}/${run}.log"
      tail -n 80 "${LOG_ROOT}/${run}.log"
    else
      echo "missing ${LOG_ROOT}/${run}.log"
    fi
    echo
  done
} | tee "${REPORT}"

echo "${now_epoch}" > "${STAMP}"
