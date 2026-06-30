#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
RUN_ROOT="${ROOT}/runs/zscape_expression_ot_strict_controls_gate_20260628"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="zscape_expression_ot_strict_controls_gate_${TS}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
SESSION_NAME="${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
LOG_PATH="${LOG_DIR}/run.log"
PYTHON="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SCRIPT="${ROOT}/ops/audit_zscape_expression_ot_strict_controls_20260628.py"
EXTRACT_RUN_DIR="${ZSCAPE_EXTRACT_RUN_DIR:-}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}/outputs"
date > "${RUN_DIR}/STARTED"
echo "${SESSION_NAME}" > "${RUN_DIR}/SESSION_NAME"

if [[ -z "${EXTRACT_RUN_DIR}" ]]; then
  echo "Set ZSCAPE_EXTRACT_RUN_DIR to the completed extraction run directory." >&2
  echo "${RUN_DIR}"
  exit 0
fi

COUNTS="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_counts_csc.npz"
CELL_INDEX="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_expression_cell_index.csv"
MATCHED_MANIFEST="${EXTRACT_RUN_DIR}/outputs/zscape_expression_selected_cell_ids_matched.csv"
OUT_DIR="${RUN_DIR}/outputs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

CMD="${PYTHON} ${SCRIPT} --counts-npz ${COUNTS} --cell-index ${CELL_INDEX} --matched-manifest ${MATCHED_MANIFEST} --out-dir ${OUT_DIR} --null-repeats ${ZSCAPE_STRICT_NULL_REPEATS:-500}"

cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} ${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED
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

* \`${OUT_DIR}/LATENTFM_ZSCAPE_EXPRESSION_OT_STRICT_CONTROLS_GATE_20260628.md\`
* \`${OUT_DIR}/zscape_expression_ot_strict_primary_rows.csv\`
* \`${OUT_DIR}/zscape_expression_ot_strict_diagnostics.csv\`

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

- Hypothesis: the primary mature fast muscle/periderm expression OT signal
  survives control-only HVG/SVD, subtype/library-matched controls, and matched
  nulls.
- Resource plan: CPU/RAM only, no GPU; thread caps default to 8 each.
- Boundary: no model training, no scFM embedding, no canonical multi, no Track C
  query.
- Promotion gate: pass only authorizes bounded latent/trajectory design review,
  not GPU model training or promotion.
- Fail-close: if strict controls fail, treat the previous OT pass as
  exploratory expression separation and redesign controls before any GPU.
EOF

if [[ "${ZSCAPE_STRICT_OT_ACK:-}" != "expression_ot_pass_external_audit_integrated" ]]; then
  echo "Set ZSCAPE_STRICT_OT_ACK=expression_ot_pass_external_audit_integrated after integrating external audit." >&2
  echo "${RUN_DIR}"
  exit 0
fi

tmux new -d -s "${SESSION_NAME}" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} ${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED"

sed -i 's/^Prepared\\.$/Running./' "${RUN_DIR}/RUN_STATUS.md"

echo "${RUN_DIR}"
