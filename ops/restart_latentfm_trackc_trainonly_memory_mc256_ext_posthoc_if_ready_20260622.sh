#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_trainonly_memory_parallel_mc256_ext_20260622
MANIFEST=${ROOT}/reports/latentfm_trackc_trainonly_memory_parallel_mc256_ext_manifest_20260622.jsonl
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

now_epoch=$(date +%s)
boundary_epoch=$(date -d '2026-06-22 21:10:00' +%s)
if (( now_epoch < boundary_epoch )); then
  echo "Refusing to inspect/restart posthoc before 2026-06-22 21:10:00 CST" >&2
  exit 3
fi

restarted=0
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  run="$("${PY}" -c 'import json,sys; print(json.loads(sys.stdin.read())["run_name"])' <<<"${line}")"
  run_dir="${RUN_ROOT}/${run}"
  train_exit="${run_dir}/${run}.EXIT_CODE"
  posthoc_exit="${run_dir}/${run}.POSTHOC_EXIT_CODE"
  decision="${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${run}.json"
  posthoc_script="${run_dir}/scripts/posthoc_${run}.sh"
  posthoc_log="${run_dir}/logs/${run}.posthoc.log"
  session="trackc_route_posthoc_${run}"
  if [[ ! -s "${train_exit}" ]]; then
    echo "${run}: training not finished"
    continue
  fi
  if [[ "$(cat "${train_exit}")" != "0" ]]; then
    echo "${run}: training exit $(cat "${train_exit}"), skip posthoc restart"
    continue
  fi
  if [[ -s "${posthoc_exit}" || -s "${decision}" ]]; then
    echo "${run}: posthoc already finished or decision present"
    continue
  fi
  if [[ ! -x "${posthoc_script}" ]]; then
    echo "${run}: missing executable posthoc script ${posthoc_script}" >&2
    continue
  fi
  tmux kill-session -t "${session}" 2>/dev/null || true
  tmux new -d -s "${session}" "bash ${posthoc_script} > ${posthoc_log} 2>&1; echo \$? > ${posthoc_exit}"
  echo "${run}: restarted ${session}"
  restarted=$((restarted + 1))
done < "${MANIFEST}"

echo "restarted=${restarted}"
