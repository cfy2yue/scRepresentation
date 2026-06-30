#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_wessels_global_prior_sweep_20260620
POSTHOC_ROOT=${ROOT}/runs/latentfm_wessels_global_prior_sweep_posthoc_20260620
REPORT=${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_SWEEP_ONE_SHOT_STATUS_$(date +%Y%m%d_%H%M%S).txt
RUNS=(
  scf_globalprior020_add010_wessels_4k
  scf_globalprior000_add010_wessels_4k
)

mkdir -p "${ROOT}/reports"
{
  echo "[$(date '+%F %T %Z')] one-shot LatentFM Wessels global-prior sweep status check"
  echo "report=${REPORT}"
  echo
  tmux ls 2>/dev/null | grep -E 'lfm_scf_globalprior020_add010_wessels_4k|lfm_scf_globalprior000_add010_wessels_4k|latentfm_wessels_global_prior_sweep_posthoc_20260620' || true
  echo
  for run in "${RUNS[@]}"; do
    echo "### ${run}"
    if [[ -f "${RUN_ROOT}/${run}.EXIT_CODE" ]]; then
      echo "EXIT_CODE=$(cat "${RUN_ROOT}/${run}.EXIT_CODE")"
      echo "FINISHED=$(cat "${RUN_ROOT}/${run}.FINISHED" 2>/dev/null || true)"
    else
      echo "still-running-or-not-started"
    fi
    log="${ROOT}/logs/latentfm_wessels_global_prior_sweep_20260620/${run}.log"
    if [[ -f "${log}" ]]; then
      sed -n '1,12p' "${log}"
      tail -n 30 "${log}"
    fi
    echo
  done
  echo "### posthoc watcher"
  if [[ -f "${POSTHOC_ROOT}/EXIT_CODE" ]]; then
    echo "EXIT_CODE=$(cat "${POSTHOC_ROOT}/EXIT_CODE")"
    echo "FINISHED=$(cat "${POSTHOC_ROOT}/FINISHED" 2>/dev/null || true)"
  else
    echo "still-running-or-not-started"
  fi
  if [[ -f "${POSTHOC_ROOT}/logs/run.log" ]]; then
    sed -n '1,5p' "${POSTHOC_ROOT}/logs/run.log"
    tail -n 30 "${POSTHOC_ROOT}/logs/run.log"
  fi
  echo
  for p in \
    "${ROOT}/reports/latentfm_wessels_global_prior_sweep_summary_20260620.csv" \
    "${ROOT}/reports/latentfm_wessels_global_prior_sweep_summary_20260620.json" \
    "${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_SWEEP_SUMMARY_20260620.md"; do
    if [[ -f "${p}" ]]; then
      echo "present ${p}"
    else
      echo "missing ${p}"
    fi
  done
} | tee "${REPORT}"
