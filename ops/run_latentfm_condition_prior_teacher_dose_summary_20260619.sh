#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_condition_prior_teacher_dose_summary_20260619
SUMMARY=${ROOT}/ops/summarize_latentfm_condition_prior_teacher_dose_20260619.py
PLOT=${ROOT}/ops/plot_latentfm_condition_prior_teacher_dose_20260619.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

WATCHERS=(
  "${ROOT}/runs/latentfm_condition_prior_teacher_posthoc_20260619"
  "${ROOT}/runs/latentfm_condition_prior_teacher_sister_posthoc_20260619"
)

mkdir -p "${RUN_ROOT}/logs"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Dose Summary 2026-06-19

Started: $(date '+%F %T %Z')
Status: waiting_for_posthoc
Runtime classification: Long task.
Polling policy: checks only posthoc EXIT_CODE files every 30 minutes; does not inspect training logs.
Expected report: ${ROOT}/reports/LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md
EOF

all_done() {
  for root in "${WATCHERS[@]}"; do
    [[ -f "${root}/EXIT_CODE" ]] || return 1
  done
  return 0
}

while ! all_done; do
  done_count=0
  for root in "${WATCHERS[@]}"; do
    [[ -f "${root}/EXIT_CODE" ]] && done_count=$((done_count + 1))
  done
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Dose Summary 2026-06-19

Started: see logs
Status: waiting_for_posthoc
Last checked: $(date '+%F %T %Z')
Next internal check: about 30 minutes
Finished posthoc markers: ${done_count} / ${#WATCHERS[@]}
EOF
  sleep 1800
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Dose Summary 2026-06-19

Started: see logs
Status: running_summary
Started summary: $(date '+%F %T %Z')
EOF

"${PYTHON}" "${SUMMARY}" > "${RUN_ROOT}/logs/summary.log" 2>&1
code=$?
plot_code=NA
if [[ "${code}" == "0" ]]; then
  "${PYTHON}" "${PLOT}" > "${RUN_ROOT}/logs/plot.log" 2>&1
  plot_code=$?
fi

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Dose Summary 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${code}
Plot exit code: ${plot_code}
Report: ${ROOT}/reports/LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md
CSV: ${ROOT}/reports/latentfm_condition_prior_teacher_dose_20260619.csv
JSON: ${ROOT}/reports/latentfm_condition_prior_teacher_dose_20260619.json
Figure base: ${ROOT}/reports/latentfm_condition_prior_teacher_dose_20260619.{pdf,svg,png}
EOF

echo "${code}" > "${RUN_ROOT}/EXIT_CODE"
date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
exit "${code}"
