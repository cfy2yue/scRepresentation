#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PYTHON="${PYTHON:-/data/cyx/software/miniconda3/envs/scdfm/bin/python}"
SCRIPT="${ROOT}/ops/audit_zscape_bioinformation_fixedcell_robustness_20260628.py"
EXTRACT_RUN_DIR="${ZSCAPE_EXTRACT_RUN_DIR:-}"
SUBSET_SPEC="${ZSCAPE_BIOINFO_SUBSET_SPEC:-${ROOT}/reports/zscape_bioinformation_subset_specs_20260628/zscape_bioinformation_subset_specs_20260628.json}"

if [[ -z "${EXTRACT_RUN_DIR}" ]]; then
  echo "Set ZSCAPE_EXTRACT_RUN_DIR=<completed extraction run directory>" >&2
  exit 2
fi

if [[ "${ZSCAPE_FIXEDCELL_ROBUSTNESS_ACK:-}" != "strict_support_integrated" ]]; then
  echo "Set ZSCAPE_FIXEDCELL_ROBUSTNESS_ACK=strict_support_integrated only after strict-controls pass or interpretable lineage-specific partial support is integrated." >&2
  exit 2
fi

RUN_NAME="zscape_bioinformation_fixedcell_robustness_gate_20260628_$(date +%H%M%S)"
RUN_DIR="${ROOT}/runs/zscape_bioinformation_fixedcell_robustness_gate_20260628/${RUN_NAME}"
SESSION_NAME="${RUN_NAME}"
LOG_PATH="${RUN_DIR}/logs/run.log"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/outputs"

COUNTS="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_counts_csc.npz"
CELL_INDEX="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_expression_cell_index.csv"
MATCHED_MANIFEST="${EXTRACT_RUN_DIR}/outputs/zscape_expression_selected_cell_ids_matched.csv"
OUT_DIR="${RUN_DIR}/outputs"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

CMD="${PYTHON} ${SCRIPT} --counts-npz ${COUNTS} --cell-index ${CELL_INDEX} --matched-manifest ${MATCHED_MANIFEST} --subset-spec-json ${SUBSET_SPEC} --out-dir ${OUT_DIR} --null-repeats ${ZSCAPE_FIXEDCELL_NULL_REPEATS:-500} --ot-cells ${ZSCAPE_FIXEDCELL_OT_CELLS:-96} --decision-mode ${ZSCAPE_FIXEDCELL_DECISION_MODE:-all_primary}"
if [[ -n "${ZSCAPE_FIXEDCELL_SUBSET_NAMES:-}" ]]; then
  CMD="${CMD} --subset-names ${ZSCAPE_FIXEDCELL_SUBSET_NAMES}"
fi

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

* \`${OUT_DIR}/LATENTFM_ZSCAPE_BIOINFORMATION_FIXEDCELL_ROBUSTNESS_20260628.md\`
* \`${OUT_DIR}/zscape_bioinformation_fixedcell_subset_summary.csv\`
* \`${OUT_DIR}/zscape_bioinformation_fixedcell_row_results.csv\`

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

- Hypothesis: frozen high-information primary subsets retain stronger strict
  fixed-cell expression OT robustness than low-signal/response-control subsets.
- Resource plan: CPU/RAM only, no GPU; thread caps default to 8 each.
- Boundary: no model training, no scFM embedding, no canonical multi, no Track C
  query.
- Launch guard: requires strict-controls pass or interpretable lineage-specific
  partial support to be integrated first.
- Promotion gate: pass supports only a later bounded LatentFM design review.
- Decision mode: \`${ZSCAPE_FIXEDCELL_DECISION_MODE:-all_primary}\`.
- Subset names: \`${ZSCAPE_FIXEDCELL_SUBSET_NAMES:-all}\`.
EOF

tmux new -d -s "${SESSION_NAME}" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} ${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED"

sed -i 's/^Prepared\.$/Running./' "${RUN_DIR}/RUN_STATUS.md"

echo "${RUN_DIR}"
