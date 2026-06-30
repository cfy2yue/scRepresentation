#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
RUN_ROOT="${ROOT}/runs/latentfm_strategy_all_decision_20260619"
LOG_ROOT="${ROOT}/logs/latentfm_strategy_all_decision_20260619"
mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}"

FOUR_CSV="${ROOT}/reports/latentfm_strategy_probe_20260619.csv"
EXPANDED_CSV="${ROOT}/reports/latentfm_strategy_probe_expanded_20260619.csv"
FOUR_STATUS="${ROOT}/runs/latentfm_strategy_probe_posthoc_20260619/RUN_STATUS.md"
EXPANDED_STATUS="${ROOT}/runs/latentfm_strategy_probe_expanded_20260619/RUN_STATUS.md"
SUMMARY="${ROOT}/ops/summarize_latentfm_strategy_all_20260619.py"
PLOT="${ROOT}/ops/plot_latentfm_strategy_all_decision_20260619.py"
PYTHON="${ROOT}/software/miniconda3/envs/scdfm/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
fi

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy All-Decision Watcher 2026-06-19

Started: $(date '+%F %T %Z')
Status: waiting_for_strategy_csvs
Runtime classification: Long task.
Polling policy: checks only file existence every 30 minutes; does not inspect training logs or GPU state.
Inputs:
- ${FOUR_CSV}
- ${EXPANDED_CSV}
Output:
- ${ROOT}/reports/LATENTFM_STRATEGY_ALL_DECISION_20260619.md
EOF

while true; do
  have_four=0
  have_expanded=0
  four_done=0
  expanded_done=0
  [[ -s "${FOUR_CSV}" ]] && have_four=1
  [[ -s "${EXPANDED_CSV}" ]] && have_expanded=1
  grep -qi '^Status: finished' "${FOUR_STATUS}" 2>/dev/null && four_done=1
  grep -qi '^Status: finished' "${EXPANDED_STATUS}" 2>/dev/null && expanded_done=1
  if [[ "${have_four}" == "1" && "${have_expanded}" == "1" ]]; then
    break
  fi
  if [[ "${four_done}" == "1" && "${expanded_done}" == "1" ]]; then
    break
  fi
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy All-Decision Watcher 2026-06-19

Started: see logs
Status: waiting_for_strategy_csvs
Last checked: $(date '+%F %T %Z')
Next internal check: about 30 minutes
Four-run CSV present: ${have_four}
Expanded CSV present: ${have_expanded}
Four-run upstream finished: ${four_done}
Expanded upstream finished: ${expanded_done}
EOF
  sleep 1800
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy All-Decision Watcher 2026-06-19

Started: see logs
Status: running_summary
Started summary: $(date '+%F %T %Z')
Note: summary runs when both CSVs exist, or when both upstream RUN_STATUS files are finished.
EOF

{
  echo "[$(date +%F_%T)] running combined strategy decision"
  "${PYTHON}" "${SUMMARY}"
  code=$?
  echo "[$(date +%F_%T)] finished combined strategy decision exit=${code}"
  if [[ "${code}" == "0" ]]; then
    echo "[$(date +%F_%T)] running combined strategy plot"
    "${PYTHON}" "${PLOT}"
    plot_code=$?
    echo "[$(date +%F_%T)] finished combined strategy plot exit=${plot_code}"
    echo "${plot_code}" > "${RUN_ROOT}/PLOT_EXIT_CODE"
  fi
  echo "${code}" > "${RUN_ROOT}/EXIT_CODE"
  date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
  exit "${code}"
} > "${RUN_ROOT}/logs/summary.log" 2>&1
code=$?

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy All-Decision Watcher 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${code}
Report: ${ROOT}/reports/LATENTFM_STRATEGY_ALL_DECISION_20260619.md
CSV: ${ROOT}/reports/latentfm_strategy_all_decision_20260619.csv
JSON: ${ROOT}/reports/latentfm_strategy_all_decision_20260619.json
Figure base: ${ROOT}/reports/latentfm_strategy_all_decision_20260619.{pdf,png,svg}
Plot exit code: $(cat "${RUN_ROOT}/PLOT_EXIT_CODE" 2>/dev/null || echo "NA")
EOF

exit "${code}"
