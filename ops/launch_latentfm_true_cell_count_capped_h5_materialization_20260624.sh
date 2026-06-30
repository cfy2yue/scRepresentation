#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_TRUE_CELL_COUNT_CAPH5_ACK:-}" != "materialize_cpu_artifacts" ]]; then
  cat >&2 <<'EOF'
Refusing to launch true cell-count capped-H5 materialization.

Set:
  LATENTFM_TRUE_CELL_COUNT_CAPH5_ACK=materialize_cpu_artifacts

Boundary:
  - CPU artifact generation only
  - no canonical metrics, canonical multi, Track C query, training, inference, or GPU
  - train conditions capped by budget; internal validation/test conditions kept at source row counts
EOF
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_capped_h5_materialization_20260624
LOG_ROOT=${ROOT}/logs/latentfm_true_cell_count_capped_h5_materialization_20260624
SESSION=lfm_true_cell_count_caph5_20260624
SCRIPT=${ROOT}/ops/materialize_latentfm_true_cell_count_capped_h5_20260624.py
STATUS=${RUN_ROOT}/RUN_STATUS.md

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi
if [[ -e "${RUN_ROOT}/EXIT_CODE" && "${FORCE_LATENTFM_TRUE_CELL_COUNT_CAPH5:-0}" != "1" ]]; then
  echo "Existing EXIT_CODE found; set FORCE_LATENTFM_TRUE_CELL_COUNT_CAPH5=1 to relaunch" >&2
  exit 3
fi

cat > "${STATUS}" <<EOF
# Run Status: latentfm_true_cell_count_capped_h5_materialization_20260624

## Command

\`\`\`bash
LATENTFM_TRUE_CELL_COUNT_CAPH5_ACK=materialize_cpu_artifacts bash ${ROOT}/ops/launch_latentfm_true_cell_count_capped_h5_materialization_20260624.sh
\`\`\`

## Runtime classification

Long CPU artifact generation task. Detached to tmux because runtime is unknown and may exceed 10 minutes.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

## Log path

\`${LOG_ROOT}/materialize.log\`

## Expected outputs

* \`${ROOT}/runs/latentfm_true_cell_count_scaling_capped_h5_20260624/artifacts/*/manifest.json\`
* \`${ROOT}/runs/latentfm_true_cell_count_scaling_capped_h5_20260624/artifacts/*/pert_means.npz\`
* \`${ROOT}/dataset/biFlow_data/xverse_true_cell_count_scaling_splits_20260624/split_*.json\`
* \`${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_CAPPED_H5_MATERIALIZER_GATE_20260624.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_ROOT}/materialize.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
find ${ROOT}/runs/latentfm_true_cell_count_scaling_capped_h5_20260624/artifacts -maxdepth 2 -name manifest.json | wc -l
\`\`\`

## Current status

Started.

## Notes

- This job materializes only capped latent H5 data dirs that pass launcher-readiness checks.
- Current all-modality rows are excluded until dose-level SciPlex labels are made compatible with the xverse split/H5 artifacts.
- It does not launch GPU training and does not authorize GPU by itself.
- GPU promotion still requires schema/provenance pass plus a separate bounded smoke launcher using the capped DATA_DIR.
EOF

rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"

tmux new -d -s "${SESSION}" "set -euo pipefail; source ${ROOT}/init-scdfm.sh >/dev/null; ${PYTHON} ${SCRIPT} --materialize --only-launcher-ready > ${LOG_ROOT}/materialize.log 2>&1; code=\$?; echo \${code} > ${RUN_ROOT}/EXIT_CODE; date '+%F %T %Z' > ${RUN_ROOT}/FINISHED; exit \${code}"

echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

tmux ls
tail -n 20 "${LOG_ROOT}/materialize.log" 2>/dev/null || true
