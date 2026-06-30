#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/xverse_full_de5000_pack_validate_recovery_20260620
SESSION=xverse_full_de5000_pack_validate_recovery_20260620
REPORT=${REPORT:-"${ROOT}/reports/XVERSE_FULL_DE5000_PACK_VALIDATE_RECOVERY_ONE_SHOT_STATUS_$(date +%Y%m%d_%H%M%S).txt"}
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
  echo "[$(date '+%F %T %Z')] one-shot xverse pack/validate recovery status check"
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
    "${ROOT}/reports/LATENTFM_XVERSE_FULL_DE5000_EXPORT_PACK_20260620.md" \
    "${ROOT}/reports/xverse_full_de5000_bundle_validation_20260620.json" \
    "${ROOT}/dataset/latentfm_full/xverse/manifest.json"; do
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
