#!/usr/bin/env bash
set -euo pipefail

if [[ "${LATENTFM_XVERSE_COND_PATH_ACK:-}" != "after_p3_closed" ]]; then
  echo "Refusing to launch: set LATENTFM_XVERSE_COND_PATH_ACK=after_p3_closed" >&2
  exit 2
fi

GPU="${LATENTFM_COND_PATH_GPU:-1}"
RUN_NAME="xverse_condition_path_sensitivity_v2_true_shuffle_zero_cap8_cell128_ode10"
RUN_ROOT="/data/cyx/1030/scLatent/runs/latentfm_xverse_condition_path_sensitivity_20260622/${RUN_NAME}"
LOG_ROOT="/data/cyx/1030/scLatent/logs/latentfm_xverse_condition_path_sensitivity_20260622/${RUN_NAME}"
SESSION="latentfm_xverse_condpath_sens_20260622"

CHECKPOINT="/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
SPLIT_FILE="/data/cyx/1030/dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
PERT_MEANS_FILE="/data/cyx/1030/scLatent/runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
SCRIPT="/data/cyx/1030/scLatent/ops/audit_latentfm_xverse_condition_path_sensitivity_20260622.py"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
LATENTFM_XVERSE_COND_PATH_ACK=after_p3_closed LATENTFM_COND_PATH_GPU=${GPU} bash /data/cyx/1030/scLatent/ops/launch_latentfm_xverse_condition_path_sensitivity_gate_20260622.sh
\`\`\`

## Runtime classification

Long/unknown GPU inference audit. Check at most every 30 minutes unless marker
files appear naturally.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

## GPU assignment

Physical GPU: \`${GPU}\`

GPU selection audit:

\`/data/cyx/1030/scLatent/reports/gpu_selection_condition_path_sensitivity_20260622_0330.json\`

## Log path

\`${LOG_ROOT}/audit.log\`

## Expected outputs

* \`${RUN_ROOT}/condition_path_sensitivity_gate.json\`
* \`${RUN_ROOT}/CONDITION_PATH_SENSITIVITY_GATE.md\`

## Current status

Started.

## Notes

P9 CPU/GPU-light gate after P3 conservative sampling failed. Uses train-only v2
internal proxy groups only:
\`internal_val_cross_background_seen_gene_proxy\` and
\`internal_val_family_gene_proxy\`. Compares true condition vs shuffled condition
vs zero/no condition. Does not read canonical test for gating and does not train.
EOF

tmux new -d -s "${SESSION}" \
"set -euo pipefail; \
export CUDA_VISIBLE_DEVICES='${GPU}'; \
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 BLIS_NUM_THREADS=2; \
date > '${RUN_ROOT}/STARTED'; \
/data/cyx/software/miniconda3/envs/scdfm/bin/python '${SCRIPT}' \
  --checkpoint '${CHECKPOINT}' \
  --split-file '${SPLIT_FILE}' \
  --pert-means-file '${PERT_MEANS_FILE}' \
  --groups internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy \
  --device cuda:0 \
  --gpu 0 \
  --ode-steps 10 \
  --max-cells 128 \
  --max-chunk 128 \
  --max-conditions-per-dataset 8 \
  --out-json '${RUN_ROOT}/condition_path_sensitivity_gate.json' \
  --out-md '${RUN_ROOT}/CONDITION_PATH_SENSITIVITY_GATE.md' \
  > '${LOG_ROOT}/audit.log' 2>&1; \
code=\$?; echo \${code} > '${RUN_ROOT}/EXIT_CODE'; date '+%F %T %Z' > '${RUN_ROOT}/FINISHED'; exit \${code}"

echo "Launched ${SESSION} on physical GPU ${GPU}"
echo "RUN_STATUS: ${RUN_ROOT}/RUN_STATUS.md"
