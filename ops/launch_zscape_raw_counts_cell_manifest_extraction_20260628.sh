#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
RUN_ROOT="${ROOT}/runs/zscape_raw_counts_cell_manifest_extraction_20260628"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="zscape_raw_counts_cell_manifest_extraction_${TS}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
SESSION_NAME="${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
LOG_PATH="${LOG_DIR}/run.log"
PYTHON="/data/cyx/software/miniconda3/envs/scdfm/bin/python"
SCRIPT="${ROOT}/ops/extract_zscape_raw_counts_for_cell_manifest_20260628.py"
RAW_COUNTS="${ROOT}/dataset/external/zscape_20260628/GSE202639_zperturb_full_raw_counts.RDS.gz"
CELL_MANIFEST="${ROOT}/reports/zscape_expression_cell_manifest_20260628/zscape_expression_selected_cell_ids.csv"
OUT_DIR="${RUN_DIR}/outputs"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
date > "${RUN_DIR}/STARTED"
echo "${SESSION_NAME}" > "${RUN_DIR}/SESSION_NAME"

CMD="PYTHONPATH=${ROOT}/software/python_deps/rdata_20260628_nodeps ${PYTHON} ${SCRIPT} --raw-counts ${RAW_COUNTS} --cell-manifest ${CELL_MANIFEST} --out-dir ${OUT_DIR}"

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

* \`${OUT_DIR}/zscape_manifest_selected_counts_csc.npz\`
* \`${OUT_DIR}/zscape_manifest_selected_gene_names.txt\`
* \`${OUT_DIR}/zscape_manifest_selected_expression_cell_index.csv\`
* \`${OUT_DIR}/LATENTFM_ZSCAPE_RAW_COUNTS_CELL_MANIFEST_EXTRACTION_20260628.md\`

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

- Hypothesis: if raw-count cells join cleanly to the audited manifest, the
  ZSCAPE branch can proceed to CPU expression continuity/OT validation.
- Resource plan: CPU/RAM only, no GPU; do not run until the raw-count download
  is complete and gzip/SHA256 have passed.
- Boundary: no model training, no scFM embedding, no canonical multi, no Track C
  query.
- Promotion gate after extraction: CPU expression continuity/OT must beat
  control-control, label-shuffle, time-shuffle, and embryo-bootstrap nulls.
EOF

if [[ "${ZSCAPE_EXTRACT_ACK:-}" != "raw_counts_download_complete" ]]; then
  echo "Set ZSCAPE_EXTRACT_ACK=raw_counts_download_complete after raw-count download finishes." >&2
  echo "${RUN_DIR}"
  exit 0
fi

tmux new -d -s "${SESSION_NAME}" \
  "${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED"

echo "${RUN_DIR}"
