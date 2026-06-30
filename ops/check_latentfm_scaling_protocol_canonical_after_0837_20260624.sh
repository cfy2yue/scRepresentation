#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi
RUN_ROOT=${ROOT}/runs/latentfm_scaling_protocol_canonical_noharm_20260624
SUMMARIZER=${ROOT}/ops/summarize_latentfm_scaling_protocol_canonical_noharm_20260624.py
NOT_BEFORE="2026-06-24 08:37:00"

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
  exit_file="${run_dir}/POSTHOC_EXIT_CODE"
  if [[ ! -e "${exit_file}" ]]; then
    echo "${run_name}: canonical posthoc still running"
    missing=1
    continue
  fi
  exit_code=$(cat "${exit_file}")
  echo "${run_name}: posthoc_exit=${exit_code}"
  if [[ "${exit_code}" != "0" ]]; then
    failed=1
  fi
done

if (( failed )); then
  "${PYTHON}" "${SUMMARIZER}" || true
  exit 2
fi
if (( missing )); then
  echo "At least one canonical posthoc is still pending; do not poll again before the next 30-minute window."
  exit 1
fi

"${PYTHON}" "${SUMMARIZER}"
