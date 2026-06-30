#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PYTHON="${PYTHON:-/data/cyx/software/miniconda3/envs/scdfm/bin/python}"
SCRIPT="${ROOT}/ops/audit_zscape_expression_latent_biology_preflight_20260628.py"
EXTRACT_RUN_DIR="${ZSCAPE_EXTRACT_RUN_DIR:-${ROOT}/runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523}"

if [[ "${ZSCAPE_QC_SENSITIVITY_ACK:-}" != "qc_log1p_policy_recorded" ]]; then
  echo "Set ZSCAPE_QC_SENSITIVITY_ACK=qc_log1p_policy_recorded after confirming this independent QC sensitivity run is intended." >&2
  exit 2
fi

RUN_NAME="zscape_expression_latent_biology_qc_sensitivity_20260628_$(date +%H%M%S)"
RUN_DIR="${ROOT}/runs/zscape_expression_latent_biology_qc_sensitivity_20260628/${RUN_NAME}"
SESSION_NAME="${RUN_NAME}"
LOG_PATH="${RUN_DIR}/logs/run.log"
OUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${RUN_DIR}/logs" "${OUT_DIR}"

COUNTS="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_counts_csc.npz"
CELL_INDEX="${EXTRACT_RUN_DIR}/outputs/zscape_manifest_selected_expression_cell_index.csv"
MATCHED_MANIFEST="${EXTRACT_RUN_DIR}/outputs/zscape_expression_selected_cell_ids_matched.csv"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

CMD="${PYTHON} ${SCRIPT} --counts-npz ${COUNTS} --cell-index ${CELL_INDEX} --matched-manifest ${MATCHED_MANIFEST} --out-dir ${OUT_DIR} --apply-qc --min-umi ${ZSCAPE_QC_MIN_UMI:-100} --min-genes ${ZSCAPE_QC_MIN_GENES:-100}"

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

* \`${OUT_DIR}/LATENTFM_ZSCAPE_EXPRESSION_LATENT_BIOLOGY_PREFLIGHT_20260628.md\`
* \`${OUT_DIR}/zscape_expression_de_row_summary.csv\`
* \`${OUT_DIR}/zscape_expression_de_top_genes.csv\`
* \`${OUT_DIR}/zscape_latent_alignment_rows.csv\`
* \`${OUT_DIR}/zscape_qc_row_summary.csv\`

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

- Hypothesis: periderm pathway and trajectory signals that survive explicit
  QC filtering are more credible biological mechanisms than signals that
  collapse after QC.
- Resource plan: CPU/RAM only, no GPU; thread caps default to 4 each.
- Boundary: no model training, no true scFM embedding, no canonical multi, no
  Track C query, and no checkpoint selection.
- Preprocessing policy: raw counts are size-factor normalized to the median
  selected-cell library and log1p transformed exactly once inside the audited
  script; this run only changes the QC filtering flag.
- Stop rule: if top biological terms or latent-proxy alignment are unstable
  after QC, demote expression-space enrichment to QC-sensitive interpretation.
EOF

tmux new -d -s "${SESSION_NAME}" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} ${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED"

sed -i 's/^Prepared\.$/Running./' "${RUN_DIR}/RUN_STATUS.md"

echo "${RUN_DIR}"
