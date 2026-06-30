#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
RUN_ROOT="${ROOT}/runs/zscape_expression_ot_continuity_gate_20260628"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="zscape_expression_ot_continuity_gate_${TS}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
SESSION_NAME="${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
LOG_PATH="${LOG_DIR}/run.log"
PYTHON="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SCRIPT="${ROOT}/ops/audit_zscape_expression_ot_continuity_20260628.py"
EXTRACT_RUN_DIR="${ZSCAPE_EXTRACT_RUN_DIR:-}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}/outputs"
date > "${RUN_DIR}/STARTED"
echo "${SESSION_NAME}" > "${RUN_DIR}/SESSION_NAME"

if [[ -z "${EXTRACT_RUN_DIR}" ]]; then
  echo "Set ZSCAPE_EXTRACT_RUN_DIR to a completed extraction run directory." >&2
  echo "${RUN_DIR}"
  exit 0
fi

COUNTS="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_counts_csc.npz"
CELL_INDEX="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_expression_cell_index.csv"
MATCHED_MANIFEST="${EXTRACT_RUN_DIR}/outputs/zscape_expression_selected_cell_ids_matched.csv"
OUT_DIR="${RUN_DIR}/outputs"
CMD="${PYTHON} ${SCRIPT} --counts-npz ${COUNTS} --cell-index ${CELL_INDEX} --matched-manifest ${MATCHED_MANIFEST} --out-dir ${OUT_DIR}"

cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED
\`\`\`

## Runtime classification

Long task.

## Start time

$(date)

## PID / tmux / scheduler ID

tmux session: \`${SESSION_NAME}\`

## Log path

\`${LOG_PATH}\`

## Expected outputs

* \`${OUT_DIR}/LATENTFM_ZSCAPE_EXPRESSION_OT_CONTINUITY_GATE_20260628.md\`
* \`${OUT_DIR}/zscape_expression_ot_row_results.csv\`
* \`${OUT_DIR}/zscape_expression_temporal_control_results.csv\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_PATH}
cat ${RUN_DIR}/EXIT_CODE 2>/dev/null || echo "still running"
free -h
nvidia-smi
\`\`\`

## Current status

Prepared.

## Notes

- Hypothesis: primary ZSCAPE biological rows show expression-space perturbation
  OT distances that beat control-control and label-shuffle nulls.
- Resource plan: CPU/RAM only, no GPU.
- Boundary: no training, no scFM embedding, no canonical multi, no Track C query.
- Fail-close: if primary rows do not beat nulls, close this as
  metadata-coordinate-only evidence or redesign the biological subset.
EOF

if [[ "${ZSCAPE_OT_ACK:-}" != "extraction_complete" ]]; then
  echo "Set ZSCAPE_OT_ACK=extraction_complete after extraction gate passes." >&2
  echo "${RUN_DIR}"
  exit 0
fi

tmux new -d -s "${SESSION_NAME}" \
  "${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED"

echo "${RUN_DIR}"
