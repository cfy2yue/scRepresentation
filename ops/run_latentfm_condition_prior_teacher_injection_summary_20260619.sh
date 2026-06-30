#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_condition_prior_teacher_injection_summary_20260619
POSTHOC_RUN_ROOT=${ROOT}/runs/latentfm_condition_prior_teacher_injection_posthoc_20260619
SUMMARY=${ROOT}/ops/summarize_latentfm_condition_prior_injection_20260619.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${RUN_ROOT}/logs"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Injection Summary 2026-06-19

Started: $(date '+%F %T %Z')
Status: waiting_for_posthoc
Runtime classification: Long task.
Polling policy: checks only injection posthoc EXIT_CODE every 30 minutes; does not inspect training or posthoc logs.
Posthoc run: ${POSTHOC_RUN_ROOT}
Expected report: ${ROOT}/reports/LATENTFM_CONDITION_PRIOR_INJECTION_COMPARISON_20260619.md
EOF

while [[ ! -f "${POSTHOC_RUN_ROOT}/EXIT_CODE" ]]; do
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Injection Summary 2026-06-19

Started: see logs
Status: waiting_for_posthoc
Last checked: $(date '+%F %T %Z')
Next internal check: about 30 minutes
Posthoc EXIT_CODE present: no
EOF
  sleep 1800
done

posthoc_code="$(cat "${POSTHOC_RUN_ROOT}/EXIT_CODE" 2>/dev/null || echo 99)"
if [[ "${posthoc_code}" != "0" ]]; then
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Injection Summary 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: skipped_posthoc_failed
Posthoc exit code: ${posthoc_code}
Exit code: ${posthoc_code}
EOF
  echo "${posthoc_code}" > "${RUN_ROOT}/EXIT_CODE"
  date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
  exit "${posthoc_code}"
fi

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Injection Summary 2026-06-19

Started: see logs
Status: running_summary
Started summary: $(date '+%F %T %Z')
EOF

"${PYTHON}" "${SUMMARY}" > "${RUN_ROOT}/logs/summary.log" 2>&1
code=$?

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Injection Summary 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${code}
Report: ${ROOT}/reports/LATENTFM_CONDITION_PRIOR_INJECTION_COMPARISON_20260619.md
CSV: ${ROOT}/reports/latentfm_condition_prior_injection_comparison_20260619.csv
JSON: ${ROOT}/reports/latentfm_condition_prior_injection_comparison_20260619.json
EOF

echo "${code}" > "${RUN_ROOT}/EXIT_CODE"
date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
exit "${code}"
