#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_CAP60_RESPONSE_SEED_ROBUST_ACK:-}" != "seed43_internal_only" ]]; then
  echo "Set LATENTFM_CAP60_RESPONSE_SEED_ROBUST_ACK=seed43_internal_only" >&2
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_scaling_cap60_response_seed_robustness_20260624
OUT_ROOT=${COUPLED}/output/latentfm_runs/scaling_cap60_response_seed_robustness_20260624
LOG_ROOT=${ROOT}/logs/latentfm_scaling_cap60_response_seed_robustness_20260624
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
SPLIT_FILE=${BIFLOW_DIR}/xverse_scaling_protocol_splits_20260624/split_seed42_xverse_scaling_protocol_cap60_primary19.json
PERT_MEANS=${ROOT}/runs/latentfm_scaling_protocol_splits_20260624/artifacts/cap60_primary19_trainonly_pert_means.npz
NORMALIZER=${ROOT}/runs/latentfm_scaling_highthroughput_smokes_20260624/artifacts/xverse_cap60_primary19_trainonly_dataset_scale_pca32.npz
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
SUMMARIZER=${ROOT}/ops/summarize_latentfm_scaling_cap60_response_seed_robustness_20260624.py

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in "${DATA_DIR}/manifest.json" "${SPLIT_FILE}" "${PERT_MEANS}" "${NORMALIZER}" "${ANCHOR_CKPT}" "${GENE_CACHE}/manifest.json" "${GPU_HELPER}" "${TRAIN_LAUNCHER}" "${SUMMARIZER}"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

declare -a RUN_NAMES=("xverse_scaling_cap60_resp025_replay05_4k_seed43")
declare -a RESPONSE_WEIGHTS=(0.25)
declare -a SEEDS=(43)
need=${#RUN_NAMES[@]}

for run_name in "${RUN_NAMES[@]}"; do
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  session=lfm_${run_name}
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_CAP60_RESPONSE_SEED_ROBUST_RERUN:-0}" != "1" ]]; then
    echo "Output exists for ${run_name}; set FORCE_LATENTFM_CAP60_RESPONSE_SEED_ROBUST_RERUN=1" >&2
    exit 3
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

echo "[$(date '+%F %T %Z')] exact GPU status before cap60 response seed robustness launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" --samples 3 --interval-seconds 10 --util-threshold-pct 10 --memory-threshold-mib 4096 --max-user-gpus 4 --max-jobs-per-gpu 4 --need "${need}" --json-only > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"
assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" "${need}" <<'PY'
import json, sys
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
need=int(sys.argv[3])
suggested=[int(x) for x in payload.get("suggested_job_gpus", [])]
system=payload.get("system") or {}
audit={"status":"pass","need":need,"assigned_gpus":suggested[:need],"active_user_gpus":payload.get("active_user_gpus"),"allowed_physical_user_gpus":payload.get("allowed_physical_user_gpus"),"system":system,"gpu_selection_json":str(sys.argv[1])}
reasons=[]
if len(suggested)<need:
    reasons.append(f"only {len(suggested)} GPU slots suggested for need={need}")
if float(system.get("mem_available_gib") or 0)<128:
    reasons.append("low_mem")
if float(system.get("load1_per_cpu") or 0)>2:
    reasons.append("high_cpu_load")
if reasons:
    audit["status"]="fail"
    audit["reasons"]=reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
raise SystemExit(0 if audit["status"]=="pass" else 4)
PY
mapfile -t ASSIGNED_GPUS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json, sys
from pathlib import Path
for gpu in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["assigned_gpus"]:
    print(int(gpu))
PY
)

for i in "${!RUN_NAMES[@]}"; do
  run_name=${RUN_NAMES[$i]}
  weight=${RESPONSE_WEIGHTS[$i]}
  seed=${SEEDS[$i]}
  gpu=${ASSIGNED_GPUS[$i]}
  run_dir=${RUN_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  session=lfm_${run_name}
  train_script=${run_dir}/scripts/run_${run_name}.sh
  posthoc_script=${run_dir}/scripts/posthoc_${run_name}.sh
  cat > "${train_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
export LATENT_BACKBONE=xverse
export DATA_DIR=${DATA_DIR}
export BIFLOW_DIR=${BIFLOW_DIR}
export SPLIT_FILE=${SPLIT_FILE}
export PERT_MEANS_FILE=${PERT_MEANS}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${log_dir}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${gpu}
export RUN_TAG=${run_name}
export SEED=${seed}
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=1
export FINETUNE_TRAINABLE_SCOPE=all
export TOTAL_STEPS=4000
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export LR=1e-4
export GAMMA=0.03
export GAMMA_WARMUP_START=500
export GAMMA_WARMUP_END=1500
export MMD_EVERY=4
export SELECTION_METRIC=pearson_pert_minus_mmd
export SELECTION_MMD_LAMBDA=0.5
export COMPOSITION_DELTA_LOSS_WEIGHT=0.06
export COMPOSITION_DELTA_LOSS_WARMUP_START=500
export COMPOSITION_DELTA_LOSS_WARMUP_END=1500
export ENDPOINT_DELTA_LOSS_WEIGHT=5.0
export ENDPOINT_DELTA_LOSS_WARMUP_START=500
export ENDPOINT_DELTA_LOSS_WARMUP_END=1500
export RESPONSE_GEOMETRY_LOSS_WEIGHT=${weight}
export RESPONSE_GEOMETRY_LOSS_WARMUP_START=500
export RESPONSE_GEOMETRY_LOSS_WARMUP_END=1500
export RESPONSE_NORMALIZATION_MODE=dataset_scale_pca
export RESPONSE_NORMALIZATION_ARTIFACT=${NORMALIZER}
export RESPONSE_GEOMETRY_CONDITION_FILTER=all
export ANCHOR_REPLAY_LOSS_WEIGHT=0.5
export ANCHOR_REPLAY_LOSS_WARMUP_START=500
export ANCHOR_REPLAY_LOSS_WARMUP_END=1500
export ANCHOR_REPLAY_CHECKPOINT=${ANCHOR_CKPT}
export ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
export EVAL_MAX_CONDITIONS=256
export EVAL_MAX_CONDITIONS_PER_DATASET=12
export EVAL_MAX_MSE_CELLS=1024
export EVAL_MAX_MMD_CELLS=1024
export EVAL_MAX_CHUNK=256
export PERT_POOL_AGGREGATIONS="sum mean max min"
export PERT_POOL_SCALE_INIT="0.5 1.0 1.0 1.0"
export PERT_POOL_FUSION_MODE=sum
export PERT_GENE_PROJECTOR_HIDDEN=1024
export PERT_CHEM_PROJECTOR_HIDDEN=1024
export PERT_TO_C_INIT_MODE=xavier_small
export USE_PERT_IN_FUSION=1
bash ${TRAIN_LAUNCHER}
EOF
  chmod +x "${train_script}"

  cat > "${posthoc_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
eval_dir=${run_dir}/posthoc_eval_internal
mkdir -p "\${eval_dir}"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${SPLIT_FILE} --pert-means-file ${PERT_MEANS} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out "\${eval_dir}/split_group_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out "\${eval_dir}/split_group_eval_candidate_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_candidate_internal_ode20.json" "\${common[@]}"
${PYTHON} ${SUMMARIZER}
EOF
  chmod +x "${posthoc_script}"

  rm -f "${run_dir}/${run_name}.EXIT_CODE" "${run_dir}/POSTHOC_EXIT_CODE"
  date '+%F %T %Z' > "${run_dir}/${run_name}.STARTED"
  tmux new -d -s "${session}" "bash -lc 'bash ${train_script} > ${log_dir}/launcher.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/${run_name}.EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/${run_name}.FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${log_dir}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"
  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: ${run_name}

## Hypothesis

The strongest seed42 response-normalized cap60 arm should retain a positive train-only internal cross/family signal under seed43 if the repair is not just stochastic seed42 luck.

## Command

\`\`\`bash
LATENTFM_CAP60_RESPONSE_SEED_ROBUST_ACK=seed43_internal_only bash ${ROOT}/ops/launch_latentfm_scaling_cap60_response_seed_robustness_20260624.sh
\`\`\`

## Runtime classification

Long GPU training plus posthoc task. Use 30-minute cadence for checks.

## Start time

$(cat "${run_dir}/${run_name}.STARTED")

## PID / tmux / scheduler ID

tmux session: \`${session}\`; physical GPU: ${gpu}

## Log path

\`${log_dir}/launcher.log\`

## Expected outputs

* \`${out_dir}/best.pt\`
* \`${run_dir}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`
* \`${ROOT}/reports/LATENTFM_SCALING_CAP60_RESPONSE_SEED_ROBUSTNESS_DECISION_20260624.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${log_dir}/launcher.log
cat ${run_dir}/${run_name}.EXIT_CODE 2>/dev/null || echo "still running"
cat ${run_dir}/POSTHOC_EXIT_CODE 2>/dev/null || echo "posthoc not complete"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Seed: ${seed}; response weight: ${weight}; anchor replay weight: 0.5.
- Response normalizer: \`${NORMALIZER}\`.
- Train split: \`${SPLIT_FILE}\`.
- Train-only internal gate only; canonical multi, canonical metrics, and held-out Track C query are not used.
EOF
  echo "Launched ${run_name} on GPU ${gpu} in tmux ${session}"
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_scaling_cap60_response_seed_robustness_20260624

## Command

\`\`\`bash
LATENTFM_CAP60_RESPONSE_SEED_ROBUST_ACK=seed43_internal_only bash ${ROOT}/ops/launch_latentfm_scaling_cap60_response_seed_robustness_20260624.sh
\`\`\`

## Runtime classification

Long GPU training batch. Each child run has its own RUN_STATUS.md.

## Start time

$(date '+%F %T %Z')

## Log path

\`${LOG_ROOT}/<run_name>/launcher.log\`

## Current status

Started ${need} cap60 response seed-robustness smoke.

## Notes

- Uses the cap60-specific train-only normalizer:
  \`${NORMALIZER}\`
- Internal train-only gate only; no canonical multi, canonical metrics, or Track C query.
EOF
