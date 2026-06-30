#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PYTHON="${PYTHON:-/data/cyx/software/miniconda3/envs/scdfm/bin/python}"
SCRIPT="${ROOT}/ops/audit_zscape_periderm_placebo_control_20260628.py"
EXTRACT_RUN_DIR="${ZSCAPE_EXTRACT_RUN_DIR:-}"
FIXEDCELL_RUN_DIR="${ZSCAPE_FIXEDCELL_RUN_DIR:-}"

if [[ -z "${EXTRACT_RUN_DIR}" ]]; then
  echo "Set ZSCAPE_EXTRACT_RUN_DIR=<completed extraction run directory>" >&2
  exit 2
fi
if [[ -z "${FIXEDCELL_RUN_DIR}" ]]; then
  echo "Set ZSCAPE_FIXEDCELL_RUN_DIR=<completed fixed-cell robustness run directory>" >&2
  exit 2
fi
if [[ "${ZSCAPE_PLACEBO_ACK:-}" != "fixedcell_pass_integrated" ]]; then
  echo "Set ZSCAPE_PLACEBO_ACK=fixedcell_pass_integrated only after fixed-cell periderm robustness passes and is integrated." >&2
  exit 2
fi

RUN_NAME="zscape_periderm_placebo_control_20260628_$(date +%H%M%S)"
RUN_DIR="${ROOT}/runs/zscape_periderm_placebo_control_20260628/${RUN_NAME}"
SESSION_NAME="${RUN_NAME}"
LOG_PATH="${RUN_DIR}/logs/run.log"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/outputs"

COUNTS="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_counts_csc.npz"
CELL_INDEX="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_expression_cell_index.csv"
MATCHED_MANIFEST="${EXTRACT_RUN_DIR}/outputs/zscape_expression_selected_cell_ids_matched.csv"
FIXED_ROWS="${FIXEDCELL_RUN_DIR}/outputs/zscape_bioinformation_fixedcell_row_results.csv"
OUT_DIR="${RUN_DIR}/outputs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

CMD="${PYTHON} ${SCRIPT} --counts-npz ${COUNTS} --cell-index ${CELL_INDEX} --matched-manifest ${MATCHED_MANIFEST} --fixedcell-row-results ${FIXED_ROWS} --out-dir ${OUT_DIR} --repeats ${ZSCAPE_PLACEBO_REPEATS:-200} --ot-cells ${ZSCAPE_PLACEBO_OT_CELLS:-96}"

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

* \`${OUT_DIR}/LATENTFM_ZSCAPE_PERIDERM_PLACEBO_CONTROL_20260628.md\`
* \`${OUT_DIR}/zscape_periderm_placebo_rows.csv\`

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

- Hypothesis: real fixed-cell periderm robustness beats periderm-internal
  wrong-target/wrong-time placebo controls.
- Resource plan: CPU/RAM only, no GPU; thread caps default to 8 each.
- Boundary: no model training, no scFM embedding, no canonical multi, no Track C
  query.
- Launch guard: requires fixed-cell periderm robustness pass to be integrated.
- Promotion gate: pass supports only a bounded design review, not GPU.
EOF

tmux new -d -s "${SESSION_NAME}" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} ${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED"

sed -i 's/^Prepared\.$/Running./' "${RUN_DIR}/RUN_STATUS.md"

echo "${RUN_DIR}"
