#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-zscape_metadata_coverage_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="/data/cyx/1030/scLatent/runs/zscape_metadata_coverage_20260628/${RUN_NAME}"
DATA_DIR="/data/cyx/1030/dataset/external/zscape_20260628"
SESSION_NAME="${RUN_NAME}"
LOG_PATH="${RUN_ROOT}/logs/run.log"
COMMAND_PATH="${RUN_ROOT}/command.sh"

mkdir -p "${RUN_ROOT}/logs" "${DATA_DIR}"

cat > "${COMMAND_PATH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
bash /data/cyx/1030/scLatent/ops/run_zscape_metadata_coverage_audit_20260628.sh '${RUN_ROOT}' '${DATA_DIR}'
EOF
chmod +x "${COMMAND_PATH}"

date > "${RUN_ROOT}/STARTED"
echo "${SESSION_NAME}" > "${RUN_ROOT}/SESSION_NAME"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
tmux new -d -s ${SESSION_NAME} "bash ${COMMAND_PATH} > ${LOG_PATH} 2>&1; echo \\\$? > ${RUN_ROOT}/EXIT_CODE; date > ${RUN_ROOT}/FINISHED"
\`\`\`

## Runtime classification

Long task. Network metadata download plus CPU metadata coverage audit; no GPU.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${SESSION_NAME}\`

## Log path

\`${LOG_PATH}\`

## Expected outputs

* \`${DATA_DIR}/GSE202639_zperturb_full_cell_metadata.csv.gz\`
* \`${DATA_DIR}/GSE202639_reference_cell_metadata.csv.gz\`
* \`${RUN_ROOT}/outputs/LATENTFM_ZSCAPE_METADATA_COVERAGE_AUDIT_20260628.md\`
* \`${RUN_ROOT}/outputs/zscape_metadata_coverage_audit.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_PATH}
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Hypothesis: ZSCAPE metadata contains enough biological structure to support
\`perturbation x timepoint x embryo/sample x cell_type\` coverage gates.

Resource plan: CPU/network only, single audit process, no GPU, no expression
matrix/CDS/raw-count download.

Promotion gate: at least two broad cell types pass coverage in both ZPERTURB
and reference metadata with multiple perturbation targets, timepoints, embryos,
and control/perturbed cells.

Fail-close: if metadata coverage is sparse or lacks shared lineage/time/target
structure, close ZSCAPE as the immediate biological-scaling source and look for
another annotated perturbation atlas.
EOF

tmux new -d -s "${SESSION_NAME}" "bash '${COMMAND_PATH}' > '${LOG_PATH}' 2>&1; echo \$? > '${RUN_ROOT}/EXIT_CODE'; date > '${RUN_ROOT}/FINISHED'"

echo "${RUN_ROOT}"
echo "${LOG_PATH}"
echo "${SESSION_NAME}"
