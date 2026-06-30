#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi
RUN_ROOT=${ROOT}/runs/latentfm_scaling_protocol_matrix_20260624
SUMMARIZER=${ROOT}/ops/summarize_latentfm_scaling_protocol_matrix_20260624.py
NOT_BEFORE="2026-06-24 08:06:00"

now_epoch=$(date +%s)
not_before_epoch=$(date -d "${NOT_BEFORE}" +%s)
if (( now_epoch < not_before_epoch )); then
  echo "Refusing early check. Not before ${NOT_BEFORE} CST." >&2
  exit 3
fi

missing=0
failed=0
for run_dir in "${RUN_ROOT}"/xverse_scaling_protocol_*_3k_seed42; do
  [[ -d "${run_dir}" ]] || continue
  run_name=$(basename "${run_dir}")
  train_exit_file="${run_dir}/${run_name}.EXIT_CODE"
  posthoc_exit_file="${run_dir}/POSTHOC_EXIT_CODE"
  if [[ ! -e "${train_exit_file}" ]]; then
    echo "${run_name}: training still running"
    missing=1
    continue
  fi
  train_exit=$(cat "${train_exit_file}")
  echo "${run_name}: train_exit=${train_exit}"
  if [[ "${train_exit}" != "0" ]]; then
    failed=1
    continue
  fi
  if [[ ! -e "${posthoc_exit_file}" ]]; then
    echo "${run_name}: posthoc pending"
    missing=1
    continue
  fi
  posthoc_exit=$(cat "${posthoc_exit_file}")
  echo "${run_name}: posthoc_exit=${posthoc_exit}"
  if [[ "${posthoc_exit}" != "0" ]]; then
    failed=1
  fi
done

if (( failed )); then
  "${PYTHON}" "${SUMMARIZER}" || true
  exit 2
fi
if (( missing )); then
  echo "At least one run is still pending; do not poll again before the next 30-minute window."
  exit 1
fi

"${PYTHON}" "${SUMMARIZER}"
