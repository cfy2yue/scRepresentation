#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_wessels_global_prior_20260620
OUT_ROOT=${COUPLED}/output/latentfm_runs/wessels_global_prior_20260620
LOG_ROOT=${ROOT}/logs/latentfm_wessels_global_prior_20260620
SPLIT_FILE=${ROOT}/runs/latentfm_dataset_upper_bound_20260620/latentfm_upperbound_wessels_split_seed42_20260620.json
PRIOR_SPLIT_FILE=${ROOT}/dataset/biFlow_data/split_seed42.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
POSTHOC_SCRIPT=${ROOT}/ops/run_latentfm_wessels_global_prior_posthoc_20260620.sh
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
BASE_INIT=${COUPLED}/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
BASELINE_DIR=${COUPLED}/output/latentfm_runs/dataset_upper_bound_20260620/scf_prior010_upperbound_wessels_4k
RUN_NAME=scf_globalprior010_add005_wessels_4k
POSTHOC_SESSION=latentfm_wessels_global_prior_posthoc_20260620

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"
for required in \
  "${DATA_DIR}" \
  "${SPLIT_FILE}" \
  "${PRIOR_SPLIT_FILE}" \
  "${BASE_INIT}" \
  "${GENE_CACHE}/manifest.json" \
  "${BASELINE_DIR}/posthoc_eval_upperbound/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${OUT_ROOT}/${RUN_NAME}" && "${FORCE_WESSELS_GLOBAL_PRIOR_RERUN:-0}" != "1" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_WESSELS_GLOBAL_PRIOR_RERUN=1 to override" >&2
  exit 6
fi

echo "[$(date '+%F %T %Z')] exact GPU status before Wessels global-prior launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --need 1 \
  --max-jobs-per-gpu 3 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

gpu="$("${PYTHON}" - "${gpu_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = payload.get("suggested_job_gpus") or []
print(chosen[0] if chosen else "")
PY
)"
if [[ -z "${gpu}" ]]; then
  echo "No GPU selected by helper; see ${gpu_json}" >&2
  exit 4
fi

resource_audit="${RUN_ROOT}/logs/resource_audit_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${resource_audit}" <<'PY'
import json
import os
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
system = payload.get("system") or {}
min_mem = float(os.environ.get("MIN_LAUNCH_MEM_AVAILABLE_GIB", "64"))
max_load = float(os.environ.get("MAX_LAUNCH_LOAD1_PER_CPU", "2.0"))
mem = float(system.get("mem_available_gib") or 0.0)
load = float(system.get("load1_per_cpu") or 0.0)
audit = {
    "status": "pass",
    "min_mem_available_gib": min_mem,
    "max_load1_per_cpu": max_load,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if mem < min_mem:
    reasons.append(f"MemAvailable {mem:.1f} GiB < {min_mem:.1f} GiB")
if load > max_load:
    reasons.append(f"load1_per_cpu {load:.3f} > {max_load:.3f}")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 5)
PY

run_script="${RUN_ROOT}/run_${RUN_NAME}.sh"
cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
cd ${COUPLED}
source ${ROOT}/init-scdfm.sh >/dev/null
${PYTHON} -m model.latent.train \\
  --data-dir ${DATA_DIR} \\
  --biflow-dir ${BIFLOW_DIR} \\
  --split-file ${SPLIT_FILE} \\
  --latent-backbone scfoundation \\
  --emb-dim 3072 \\
  --save-dir ${OUT_ROOT}/${RUN_NAME} \\
  --log-file train.log \\
  --model-type control_mlp \\
  --init-checkpoint ${BASE_INIT} \\
  --use-pert-condition \\
  --pert-gene-emb-cache-dir ${GENE_CACHE} \\
  --pert-condition-embedding-source scgpt_embed_gene \\
  --pert-pool-aggregations mean max min \\
  --pert-pool-scale-init 1.0 1.0 1.0 \\
  --pert-pool-fusion-mode sum \\
  --pert-type-adapter-mode scalar \\
  --pert-chem-projector-hidden 1024 \\
  --pert-gene-projector-hidden 1024 \\
  --pert-to-c-init-mode xavier_small \\
  --use-pert-in-fusion \\
  --condition-delta-head-use-in-model \\
  --batch-size 64 \\
  --grad-accum-steps 1 \\
  --min-cells 32 \\
  --ds-alpha 0.7 \\
  --ds-loss-alpha 0.0 \\
  --ds-loss-warmup-start 0 \\
  --min-selected-conditions-per-dataset 0 \\
  --condition-visit-power 1.0 \\
  --condition-visit-cap 0 \\
  --total-steps 4000 \\
  --lr 0.0001 \\
  --warmup-steps 300 \\
  --lr-decay-steps 4000 \\
  --eval-max-conditions 256 \\
  --eval-max-conditions-per-dataset 12 \\
  --eval-max-mmd-cells 512 \\
  --eval-max-chunk 128 \\
  --selection-metric pearson_pert_minus_mmd \\
  --selection-mmd-lambda 0.5 \\
  --endpoint-delta-loss-weight 2.0 \\
  --endpoint-delta-loss-warmup-start 0 \\
  --endpoint-delta-loss-warmup-end 1000 \\
  --composition-delta-loss-weight 0.0 \\
  --condition-prior-bank-scope global \\
  --condition-prior-bank-split-file ${PRIOR_SPLIT_FILE} \\
  --condition-prior-bank-aggregation gene_mean \\
  --condition-prior-delta-loss-weight 0.10 \\
  --condition-prior-delta-loss-warmup-start 0 \\
  --condition-prior-delta-loss-warmup-end 1000 \\
  --condition-prior-delta-loss-every 1 \\
  --condition-prior-additive-delta-loss-weight 0.05 \\
  --condition-prior-additive-delta-loss-warmup-start 0 \\
  --condition-prior-additive-delta-loss-warmup-end 1000 \\
  --condition-prior-bank-max-cells 512 \\
  --condition-prior-num-genes 2 \\
  --ot-method torch_sinkhorn \\
  --ot-sinkhorn-iter 50 \\
  --use-amp \\
  --amp-dtype bf16
EOF
chmod +x "${run_script}"

session="lfm_${RUN_NAME}"
tmux new -d -s "${session}" \
  "bash -lc 'bash ${run_script} > ${LOG_ROOT}/${RUN_NAME}.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/${RUN_NAME}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/${RUN_NAME}.FINISHED; exit \$rc'"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_wessels_global_prior_20260620

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_wessels_global_prior_20260620.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

* \`${session}\`
* posthoc watcher: \`${POSTHOC_SESSION}\`

## Log path

* \`${LOG_ROOT}/${RUN_NAME}.log\`
* \`${RUN_ROOT}/logs/posthoc_launcher.log\`

## Expected outputs

* \`${OUT_ROOT}/${RUN_NAME}/best.pt\`
* \`${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_SUMMARY_20260620.md\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/${RUN_NAME}.EXIT_CODE 2>/dev/null || echo ${RUN_NAME}: still-running
cat ${ROOT}/runs/latentfm_wessels_global_prior_posthoc_20260620/EXIT_CODE 2>/dev/null || echo posthoc: still-running
tmux ls
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Uses Wessels-only training split, but builds condition-prior teacher records
from the full canonical train split only:
\`${PRIOR_SPLIT_FILE}\`.

Prior mode:
`condition_prior_bank_scope=global`,
`condition_prior_bank_aggregation=gene_mean`,
`condition_prior_delta_loss_weight=0.10`,
`condition_prior_additive_delta_loss_weight=0.05`.
EOF

tmux new -d -s "${POSTHOC_SESSION}" \
  "bash -lc 'bash ${POSTHOC_SCRIPT} > ${RUN_ROOT}/logs/posthoc_launcher.log 2>&1'"

echo "Launched ${RUN_NAME} on physical GPU${gpu}"
echo "Training log: ${LOG_ROOT}/${RUN_NAME}.log"
echo "Posthoc watcher: ${POSTHOC_SESSION}"
