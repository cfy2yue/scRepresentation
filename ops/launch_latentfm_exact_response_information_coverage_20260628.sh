#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="latentfm_exact_response_information_coverage_20260628_${TS}"
RUN_DIR="${ROOT}/runs/latentfm_exact_response_information_coverage_20260628/${RUN_NAME}"
SESSION_NAME="${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
OUT_DIR="${RUN_DIR}/outputs"
LOG_PATH="${LOG_DIR}/run.log"
RUN_STATUS="${RUN_DIR}/RUN_STATUS.md"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

CMD="cd ${ROOT} && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 /data/cyx/software/miniconda3/envs/scdfm/bin/python ops/materialize_latentfm_exact_response_information_coverage_20260628.py --out-dir ${OUT_DIR} --max-file-mb 800 --max-conditions-per-dataset 160 --min-pert-cells 25"

cat > "${RUN_STATUS}" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
${CMD}
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%Y-%m-%d %H:%M:%S %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION_NAME}\`

## Log path

\`${LOG_PATH}\`

## Expected outputs

* \`${OUT_DIR}/LATENTFM_EXACT_RESPONSE_INFORMATION_COVERAGE_20260628.md\`
* \`${OUT_DIR}/exact_response_information_condition_rows.csv\`
* \`${OUT_DIR}/exact_response_information_budget_rows.csv\`
* \`${OUT_DIR}/exact_response_information_budget_summary.csv\`
* \`${OUT_DIR}/exact_response_information_condition_summary.csv\`
* \`${OUT_DIR}/exact_response_information_dataset_meta.csv\`
* \`${OUT_DIR}/latentfm_exact_response_information_coverage_20260628.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_PATH}
cat ${RUN_DIR}/EXIT_CODE 2>/dev/null || echo "still running"
free -h
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head
\`\`\`

## Current status

Started.

## Notes

Hypothesis: exact per-dataset/per-condition expressed-gene response-information
coverage can replace group-level priors in the downstream scaling-law design
matrix.

Resource plan: CPU-only, 8 BLAS threads, one raw-expression dataset loaded at a
time, files capped at 800 MB, max 160 conditions per dataset. No GPU, no
training, no inference, no canonical multi, no Track C query, no checkpoint
selection.

Promotion gate: at least 1000 condition rows across at least 10 usable
datasets, with top-1000 abundance/HVG coverage and k80/k90 summaries written.

Fail-close: if controls are missing, matrix loading fails, or row count is too
low, keep this as partial CPU evidence and do not launch GPU.
EOF

echo "${SESSION_NAME}" > "${RUN_DIR}/SESSION_NAME"
date > "${RUN_DIR}/STARTED"

tmux new -d -s "${SESSION_NAME}" "bash -lc '${CMD} > ${LOG_PATH} 2>&1; code=\$?; echo \${code} > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED; if [ \${code} -eq 0 ]; then sed -i \"s/^Started\\./Finished./\" ${RUN_STATUS}; else sed -i \"s/^Started\\./Failed./\" ${RUN_STATUS}; fi'"

echo "${RUN_DIR}"
