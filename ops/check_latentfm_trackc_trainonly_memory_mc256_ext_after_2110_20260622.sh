#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_trainonly_memory_parallel_mc256_ext_20260622
MANIFEST=${ROOT}/reports/latentfm_trackc_trainonly_memory_parallel_mc256_ext_manifest_20260622.jsonl
SUMMARY_MD=${ROOT}/reports/LATENTFM_TRACKC_TRAINONLY_MEMORY_MC256_EXT_DECISION_SUMMARY_20260622.md
SUMMARY_CSV=${ROOT}/reports/latentfm_trackc_trainonly_memory_mc256_ext_decision_summary_20260622.csv
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

now_epoch=$(date +%s)
boundary_epoch=$(date -d '2026-06-22 21:10:00' +%s)
if (( now_epoch < boundary_epoch )); then
  echo "Refusing to check before 2026-06-22 21:10:00 CST" >&2
  exit 3
fi

echo "# Track C train-only memory mc256 extension one-shot status"
echo "checked_at=$(date '+%F %T %Z')"
echo
echo "## tmux"
tmux ls 2>/dev/null | grep -E 'trackc_route_(train|posthoc)_xverse_trackc_mem256ext' || echo "no mc256 extension memory tmux sessions"
echo
echo "## per-run artifacts"

missing=0
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  run="$("${PY}" -c 'import json,sys; print(json.loads(sys.stdin.read())["run_name"])' <<<"${line}")"
  exit_path="${RUN_ROOT}/${run}/${run}.EXIT_CODE"
  posthoc_exit_path="${RUN_ROOT}/${run}/${run}.POSTHOC_EXIT_CODE"
  decision_path="${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${run}.json"
  exit_status="$(cat "${exit_path}" 2>/dev/null || echo running)"
  posthoc_status="$(cat "${posthoc_exit_path}" 2>/dev/null || echo running)"
  decision_status="missing"
  if [[ -s "${decision_path}" ]]; then
    decision_status="present"
  else
    missing=$((missing + 1))
  fi
  echo "${run} train_exit=${exit_status} posthoc_exit=${posthoc_status} decision=${decision_status}"
done < "${MANIFEST}"

if (( missing == 0 )); then
  "${PY}" "${ROOT}/ops/summarize_latentfm_trackc_manifest_decisions_20260622.py" \
    --manifest "${MANIFEST}" \
    --out-md "${SUMMARY_MD}" \
    --out-csv "${SUMMARY_CSV}"
  echo
  echo "summary=${SUMMARY_MD}"
else
  echo
  echo "decisions_missing=${missing}"
fi
