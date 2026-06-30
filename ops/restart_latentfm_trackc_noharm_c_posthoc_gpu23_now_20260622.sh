#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_noharm_adapter_parallel_c_20260622
MANIFEST=${ROOT}/reports/latentfm_trackc_noharm_adapter_parallel_c_manifest_20260622.jsonl
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 2
fi

"${PYTHON}" - "${MANIFEST}" <<'PY' | while IFS=$'\t' read -r run gpu; do
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    gpu = int(row.get("forced_gpu", -1))
    if gpu in {2, 3}:
        print(f"{row['run_name']}\t{gpu}")
PY
  run_dir=${RUN_ROOT}/${run}
  posthoc_script=${run_dir}/scripts/posthoc_${run}.sh
  posthoc_log=${run_dir}/logs/${run}.posthoc.log
  posthoc_session=trackc_route_posthoc_${run}
  exit_code_file=${run_dir}/${run}.EXIT_CODE
  posthoc_code_file=${run_dir}/${run}.POSTHOC_EXIT_CODE

  if [[ ! -f "${exit_code_file}" ]]; then
    echo "Skip ${run}: training exit missing" >&2
    continue
  fi
  if [[ "$(cat "${exit_code_file}")" != "0" ]]; then
    echo "Skip ${run}: training exit $(cat "${exit_code_file}")" >&2
    continue
  fi
  if [[ -f "${posthoc_code_file}" ]]; then
    echo "Skip ${run}: posthoc already has exit $(cat "${posthoc_code_file}")" >&2
    continue
  fi
  if [[ ! -x "${posthoc_script}" ]]; then
    echo "Missing posthoc script for ${run}: ${posthoc_script}" >&2
    exit 2
  fi

  tmux kill-session -t "${posthoc_session}" 2>/dev/null || true
  tmux new -d -s "${posthoc_session}" \
    "bash ${posthoc_script} > ${posthoc_log} 2>&1; echo \$? > ${posthoc_code_file}"
  echo "restarted_posthoc run=${run} gpu=${gpu} session=${posthoc_session}"
done
