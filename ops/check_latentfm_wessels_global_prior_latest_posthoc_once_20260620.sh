#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_wessels_global_prior_latest_posthoc_20260620
REPORT=${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_LATEST_POSTHOC_ONE_SHOT_STATUS_$(date +%Y%m%d_%H%M%S).txt

{
  echo "[$(date '+%F %T %Z')] one-shot LatentFM Wessels global-prior latest-posthoc status check"
  echo "report=${REPORT}"
  echo
  tmux ls 2>/dev/null | grep 'latentfm_wessels_global_prior_latest_posthoc_20260620' || true
  echo
  if [[ -f "${RUN_ROOT}/EXIT_CODE" ]]; then
    echo "EXIT_CODE=$(cat "${RUN_ROOT}/EXIT_CODE")"
    echo "FINISHED=$(cat "${RUN_ROOT}/FINISHED" 2>/dev/null || true)"
  else
    echo "still-running-or-not-started"
  fi
  echo
  if [[ -f "${RUN_ROOT}/logs/run.log" ]]; then
    tail -n 120 "${RUN_ROOT}/logs/run.log"
  else
    echo "missing ${RUN_ROOT}/logs/run.log"
  fi
  echo
  for path in \
    "${ROOT}/reports/latentfm_wessels_global_prior_latest_gate_audit_20260620.json" \
    "${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_LATEST_GATE_AUDIT_20260620.md" \
    "${ROOT}/reports/latentfm_wessels_global_prior_latest_summary_20260620.json" \
    "${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_LATEST_SUMMARY_20260620.md"; do
    if [[ -s "${path}" ]]; then
      echo "present ${path}"
    else
      echo "missing ${path}"
    fi
  done
} | tee "${REPORT}"
