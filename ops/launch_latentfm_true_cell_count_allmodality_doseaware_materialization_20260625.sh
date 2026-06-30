#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_ALLMOD_DOSEAWARE_MATERIALIZE_ACK:-}" != "materialize_cpu_artifacts" ]]; then
  cat >&2 <<'EOF'
Refusing to launch dose-aware all-modality materialization.

Set:
  LATENTFM_ALLMOD_DOSEAWARE_MATERIALIZE_ACK=materialize_cpu_artifacts

Boundary:
  - CPU artifact generation only
  - no GPU, no training, no inference
  - no canonical metrics, canonical multi, or held-out Track C query
  - canonical reference drugs excluded from the dose-aware internal split
EOF
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_allmodality_doseaware_materialization_20260625
LOG_ROOT=${ROOT}/logs/latentfm_true_cell_count_allmodality_doseaware_materialization_20260625
SESSION=lfm_allmod_doseaware_mat_20260625
SCRIPT=${ROOT}/ops/materialize_latentfm_true_cell_count_allmodality_doseaware_capped_h5_20260625.py
STATUS=${RUN_ROOT}/RUN_STATUS.md

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi
if [[ -e "${RUN_ROOT}/EXIT_CODE" && "${FORCE_LATENTFM_ALLMOD_DOSEAWARE_MATERIALIZE:-0}" != "1" ]]; then
  echo "Existing EXIT_CODE found; set FORCE_LATENTFM_ALLMOD_DOSEAWARE_MATERIALIZE=1 to relaunch" >&2
  exit 3
fi

cat > "${STATUS}" <<EOF
# Run Status: latentfm_true_cell_count_allmodality_doseaware_materialization_20260625

## Command

\`\`\`bash
LATENTFM_ALLMOD_DOSEAWARE_MATERIALIZE_ACK=materialize_cpu_artifacts bash ${ROOT}/ops/launch_latentfm_true_cell_count_allmodality_doseaware_materialization_20260625.sh
\`\`\`

## Runtime classification

Long/unknown CPU artifact generation task. Detached to tmux because materializing
9 capped H5 artifact rows may exceed 10 minutes.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

## Log path

\`${LOG_ROOT}/materialize.log\`

## Expected outputs

* \`${ROOT}/runs/latentfm_true_cell_count_allmodality_doseaware_capped_h5_20260625/artifacts/*/manifest.json\`
* \`${ROOT}/runs/latentfm_true_cell_count_allmodality_doseaware_capped_h5_20260625/artifacts/*/sampled_indices.npz\`
* \`${ROOT}/runs/latentfm_true_cell_count_allmodality_doseaware_capped_h5_20260625/artifacts/*/pert_means.npz\`
* \`${ROOT}/dataset/biFlow_data/xverse_true_cell_count_allmodality_doseaware_splits_20260625/split_*.json\`
* \`${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_MATERIALIZER_GATE_20260625.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_ROOT}/materialize.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
find ${ROOT}/runs/latentfm_true_cell_count_allmodality_doseaware_capped_h5_20260625/artifacts -maxdepth 2 -name manifest.json | wc -l
\`\`\`

## Current status

Started.

## Notes

- CPU-only artifact generation; no GPU use expected.
- Uses xverse per-cell SciPlex embeddings grouped by dose-level
  \`cov_drug_dose_name\`.
- GPU training remains blocked until schema, dryload, design, and tail gates
  pass.
EOF

rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"

tmux new -d -s "${SESSION}" "set -euo pipefail; source ${ROOT}/init-scdfm.sh >/dev/null 2>&1 || true; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4; cd ${ROOT}; ${PYTHON} ${SCRIPT} --materialize > ${LOG_ROOT}/materialize.log 2>&1; code=\$?; echo \${code} > ${RUN_ROOT}/EXIT_CODE; date '+%F %T %Z' > ${RUN_ROOT}/FINISHED; exit \${code}"

echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

tmux ls
tail -n 20 "${LOG_ROOT}/materialize.log" 2>/dev/null || true
