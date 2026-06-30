#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PYTHON="${PYTHON:-/data/cyx/software/miniconda3/envs/scdfm/bin/python}"
SCRIPT="${ROOT}/ops/audit_zscape_latent_preprocessing_sensitivity_20260628.py"

RUN_NAME="zscape_latent_preprocessing_sensitivity_20260628_$(date +%H%M%S)"
RUN_DIR="${ROOT}/runs/zscape_latent_preprocessing_sensitivity_20260628/${RUN_NAME}"
SESSION_NAME="${RUN_NAME}"
LOG_PATH="${RUN_DIR}/logs/run.log"
OUT_DIR="${RUN_DIR}/outputs"
mkdir -p "${RUN_DIR}/logs" "${OUT_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

CMD="${PYTHON} ${SCRIPT} --out-dir ${OUT_DIR}"

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

* \`${OUT_DIR}/LATENTFM_ZSCAPE_LATENT_PREPROCESSING_SENSITIVITY_20260628.md\`
* \`${OUT_DIR}/zscape_latent_preprocessing_sensitivity_summary.csv\`
* \`${OUT_DIR}/zscape_latent_preprocessing_sensitivity_alignment_rows.csv\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_PATH}
cat ${RUN_DIR}/EXIT_CODE 2>/dev/null || echo "still running"
free -h
\`\`\`

## Current status

Prepared.

## Notes

- Hypothesis: QC filtering is stable, HVG-budget latent geometry is stable
  enough for information-content analysis, and no-log1p changes the geometry
  enough that expression-space claims must keep exactly one log1p.
- Resource plan: CPU/RAM only, no GPU, four BLAS threads.
- Boundary: no model training, no inference, no true scFM embedding extraction,
  no canonical multi, no Track C query.
- Promotion gate: pass supports ZSCAPE preprocessing policy and scaling
  x-variable design only; it does not authorize LatentFM training.
EOF

tmux new -d -s "${SESSION_NAME}" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} ${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED"

sed -i 's/^Prepared\.$/Running./' "${RUN_DIR}/RUN_STATUS.md"

echo "${RUN_DIR}"
