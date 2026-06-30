#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
PYTHON="${PYTHON:-/data/cyx/software/miniconda3/envs/scdfm/bin/python}"
SCRIPT="${ROOT}/ops/audit_zscape_expression_trajectory_time_gate_20260628.py"
EXTRACT_RUN_DIR="${ZSCAPE_EXTRACT_RUN_DIR:-}"

if [[ -z "${EXTRACT_RUN_DIR}" ]]; then
  echo "Set ZSCAPE_EXTRACT_RUN_DIR=<completed extraction run directory>" >&2
  exit 2
fi

RUN_NAME="zscape_expression_trajectory_time_gate_20260628_$(date +%H%M%S)"
RUN_DIR="${ROOT}/runs/zscape_expression_trajectory_time_gate_20260628/${RUN_NAME}"
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

CMD="${PYTHON} ${SCRIPT} --counts-npz ${COUNTS} --cell-index ${CELL_INDEX} --matched-manifest ${MATCHED_MANIFEST} --out-dir ${OUT_DIR} --null-repeats ${ZSCAPE_TRAJECTORY_NULL_REPEATS:-200}"

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

* \`${OUT_DIR}/LATENTFM_ZSCAPE_EXPRESSION_TRAJECTORY_TIME_GATE_20260628.md\`
* \`${OUT_DIR}/zscape_expression_trajectory_time_temporal_controls.csv\`
* \`${OUT_DIR}/zscape_expression_trajectory_time_perturb_alignment.csv\`

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

- Hypothesis: primary perturbation responses are time-aware, not only
  condition-ID separated. Mature fast muscle and periderm controls should show
  adjacent-time structure, and primary perturbation displacement should align
  more with true lineage-time vectors than wrong-lineage vectors.
- Resource plan: CPU/RAM only, no GPU; thread caps default to 8 each.
- Boundary: no model training, no scFM embedding, no canonical multi, no Track C
  query.
- Promotion gate: pass only supports bounded dynamic/trajectory design review
  after strict-controls integration.
- Fail-close: partial/fail is dynamic negative evidence and does not authorize
  GPU.
EOF

if [[ "${ZSCAPE_TRAJECTORY_ACK:-}" != "strict_parallel_cpu_ok" ]]; then
  echo "Set ZSCAPE_TRAJECTORY_ACK=strict_parallel_cpu_ok after confirming this independent CPU gate is intended." >&2
  echo "${RUN_DIR}"
  exit 0
fi

tmux new -d -s "${SESSION_NAME}" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} ${CMD} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED"

sed -i 's/^Prepared\.$/Running./' "${RUN_DIR}/RUN_STATUS.md"

echo "${RUN_DIR}"
