#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_trackc_pairwise_latest_posthoc_20260622
REPORT_DIR=${ROOT}/reports
PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/1030/software/miniconda3/envs/scdfm/bin/python
fi
DEADLINE_EPOCH=$(date -d '2026-06-22 20:10:00' +%s)
NOW_EPOCH=$(date +%s)
if (( NOW_EPOCH < DEADLINE_EPOCH )); then
  echo "Refusing to check before 2026-06-22 20:10:00 CST." >&2
  exit 3
fi

RUNS=(
  xverse_trackc_noharm_pc_ep050_replay2_all_2k_seed42
  xverse_trackc_noharm_pc_ep050_replay4_nongm_2k_seed42
  xverse_trackc_noharm_pc_ep050del_replay2_all_2k_seed42
  xverse_trackc_noharm_pc_ep100_replay2_all_2k_seed42
  xverse_trackc_noharm_pc_ep100del_replay4_all_2k_seed42
  xverse_trackc_noharm_pc_ep100del_replay4_nongm_2k_seed42
)

echo "# Track C pairwise latest one-shot status"
date '+checked_at=%Y-%m-%d %H:%M:%S %Z'
echo
echo "## tmux"
tmux ls 2>/dev/null | rg 'trackc_latest_' || echo "no trackc_latest tmux sessions"
echo
echo "## per-run artifacts"

all_exit=1
all_decision=1
any_failed=0
for run in "${RUNS[@]}"; do
  code_path="${RUN_ROOT}/${run}/${run}.LATEST_POSTHOC_EXIT_CODE"
  decision_json="${REPORT_DIR}/latentfm_trackc_pairwise_latest_decision_${run}.json"
  decision_md="${REPORT_DIR}/LATENTFM_TRACKC_PAIRWISE_LATEST_DECISION_${run}.md"
  code="missing"
  if [[ -f "${code_path}" ]]; then
    code="$(cat "${code_path}")"
  else
    all_exit=0
  fi
  if [[ "${code}" != "0" && "${code}" != "missing" ]]; then
    any_failed=1
  fi
  decision="missing"
  if [[ -s "${decision_json}" && -s "${decision_md}" ]]; then
    decision="present"
  else
    all_decision=0
  fi
  printf '%s exit=%s decision=%s\n' "${run}" "${code}" "${decision}"
done

if (( any_failed )); then
  echo
  echo "At least one latest posthoc job failed; inspect the corresponding log before relaunch."
  exit 2
fi

if (( all_exit && all_decision )); then
  "${PY}" "${ROOT}/ops/summarize_latentfm_trackc_pairwise_latest_decisions_20260622.py" \
    --out-md "${REPORT_DIR}/LATENTFM_TRACKC_PAIRWISE_LATEST_DECISION_SUMMARY_20260622.md" \
    --out-csv "${REPORT_DIR}/latentfm_trackc_pairwise_latest_decision_summary_20260622.csv"
  echo
  echo "summary=${REPORT_DIR}/LATENTFM_TRACKC_PAIRWISE_LATEST_DECISION_SUMMARY_20260622.md"
  exit 0
fi

echo
echo "Latest posthoc block is not complete yet. Do not poll again before the next 30-minute window unless there is crash evidence."
exit 1
