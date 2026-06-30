#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

ACK=${LATENTFM_OBSINFO_WEIGHT_ACK:-}
if [[ "${ACK}" != "gate_pass_bounded_smoke" ]]; then
  cat >&2 <<'EOF'
Refusing to launch observable-information condition-weight smoke.

Set:
  LATENTFM_OBSINFO_WEIGHT_ACK=gate_pass_bounded_smoke

Required boundary:
  - CPU weight gate status must be observable_information_condition_loss_weights_gate_pass_bounded_smoke_ready
  - run only observable-vs-same-marginal-random internal smoke
  - no canonical multi, Track C query, or promotion/checkpoint claim
EOF
  exit 4
fi

GATE_JSON=${ROOT}/reports/observable_information_condition_loss_weights_20260629/latentfm_observable_information_condition_loss_weights_20260629.json
OBS_WEIGHT_CSV=${ROOT}/reports/observable_information_condition_loss_weights_20260629/observable_information_condition_loss_weights.csv
RAND_WEIGHT_CSV=${ROOT}/reports/observable_information_condition_loss_weights_20260629/observable_information_condition_loss_weights_random_seed43.csv
TRAIN_SPLIT=${ROOT}/dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
GPU_HELPER=${ROOT}/ops/select_available_gpus.py

RUN_ROOT=${ROOT}/runs/latentfm_observable_information_condition_weight_smoke_20260629
OUT_ROOT=${COUPLED}/output/latentfm_runs/observable_information_condition_weight_smoke_20260629
LOG_ROOT=${ROOT}/logs/latentfm_observable_information_condition_weight_smoke_20260629
REPORT_DIR=${ROOT}/reports/observable_information_condition_weight_smoke_20260629
TOTAL_STEPS=${LATENTFM_OBSINFO_WEIGHT_STEPS:-2000}
SEED=${LATENTFM_OBSINFO_WEIGHT_SEED:-42}

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${REPORT_DIR}"

for required in \
  "${GATE_JSON}" \
  "${OBS_WEIGHT_CSV}" \
  "${RAND_WEIGHT_CSV}" \
  "${TRAIN_SPLIT}" \
  "${DATA_DIR}/manifest.json" \
  "${BIFLOW_DIR}/split_seed42.json" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${TRAIN_LAUNCHER}" \
  "${GPU_HELPER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - "${GATE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status != "observable_information_condition_loss_weights_gate_pass_bounded_smoke_ready":
    raise SystemExit(f"weight gate does not authorize bounded smoke: {status!r}")
if payload.get("gpu_authorized") is not True:
    raise SystemExit("weight gate gpu_authorized is not true")
PY

declare -a ARMS=("observable" "random")
declare -a WEIGHTS=("${OBS_WEIGHT_CSV}" "${RAND_WEIGHT_CSV}")
declare -a HYPOTHESES=(
  "Observable-information condition loss weighting should reduce harmful low-information updates beyond a same-marginal random-weight control."
  "Same-marginal stratified random condition weights control for weight variance and optimizer noise."
)
need=${#ARMS[@]}

for i in "${!ARMS[@]}"; do
  arm=${ARMS[$i]}
  run_name="xverse_obsinfo_condition_weight_${arm}_${TOTAL_STEPS}step_seed${SEED}"
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  session="lfm_obsinfo_w_${arm}_${TOTAL_STEPS}_s${SEED}"
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_OBSINFO_WEIGHT_RERUN:-0}" != "1" ]]; then
    echo "Output exists for ${run_name}; set FORCE_LATENTFM_OBSINFO_WEIGHT_RERUN=1 to relaunch" >&2
    exit 3
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

echo "[$(date '+%F %T %Z')] resource audit before observable-info condition-weight smoke" | tee "${RUN_ROOT}/logs/resource_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/resource_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/resource_launch_audit.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/resource_launch_audit.log"
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 25 | tee -a "${RUN_ROOT}/logs/resource_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 4 \
  --need "${need}" \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" "${need}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
need = int(sys.argv[3])
suggested = [int(x) for x in payload.get("suggested_job_gpus", [])]
system = payload.get("system") or {}
reasons = []
if len(suggested) < need:
    reasons.append(f"only {len(suggested)} GPU job slots suggested for need={need}")
if int(payload.get("max_user_gpus") or 0) > 4:
    reasons.append("max_user_gpus exceeds active cap 4")
if int(payload.get("max_jobs_per_gpu") or 0) > 4:
    reasons.append("max_jobs_per_gpu exceeds active cap 4")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
audit = {
    "status": "fail" if reasons else "pass",
    "assigned_gpus": suggested[:need],
    "reasons": reasons,
    "gpu_selection_json": str(sys.argv[1]),
    "system": system,
    "policy": "max 4 physical GPUs, max 4 LatentFM training jobs/GPU, CPU <=48 cores",
}
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if not reasons else 4)
PY

mapfile -t ASSIGNED_GPUS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for gpu in payload["assigned_gpus"]:
    print(int(gpu))
PY
)

for i in "${!ARMS[@]}"; do
  arm=${ARMS[$i]}
  weight_file=${WEIGHTS[$i]}
  gpu=${ASSIGNED_GPUS[$i]}
  run_name="xverse_obsinfo_condition_weight_${arm}_${TOTAL_STEPS}step_seed${SEED}"
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  session="lfm_obsinfo_w_${arm}_${TOTAL_STEPS}_s${SEED}"
  train_script=${run_dir}/scripts/train_${run_name}.sh
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
export SPLIT_FILE=${TRAIN_SPLIT}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${log_dir}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${gpu}
export RUN_TAG=${run_name}
export SEED=${SEED}
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=1
export FINETUNE_TRAINABLE_SCOPE=all
export TOTAL_STEPS=${TOTAL_STEPS}
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export LR=1e-4
export GAMMA=0.03
export GAMMA_WARMUP_START=500
export GAMMA_WARMUP_END=1500
export MMD_EVERY=4
export SELECTION_METRIC=test_mmd
export SELECTION_MMD_LAMBDA=1.0
export COMPOSITION_DELTA_LOSS_WEIGHT=0.06
export COMPOSITION_DELTA_LOSS_WARMUP_START=500
export COMPOSITION_DELTA_LOSS_WARMUP_END=1500
export ENDPOINT_DELTA_LOSS_WEIGHT=5.0
export ENDPOINT_DELTA_LOSS_WARMUP_START=500
export ENDPOINT_DELTA_LOSS_WARMUP_END=1500
export DS_ALPHA=1.0
export DS_LOSS_ALPHA=0.0
export MIN_SELECTED_CONDITIONS_PER_DATASET=0
export CONDITION_VISIT_POWER=1.0
export CONDITION_VISIT_CAP=0
export CONDITION_LOSS_WEIGHT_FILE=${weight_file}
export CONDITION_LOSS_WEIGHT_COLUMN=weight
export CONDITION_LOSS_WEIGHT_NORMALIZE_MEAN=1
export OT_THREADS=2
export PREFETCH=4
export N_OT_WORKERS=2
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
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${TRAIN_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out "\${eval_dir}/split_group_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out "\${eval_dir}/split_group_eval_candidate_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_candidate_internal_ode20.json" "\${common[@]}"
EOF
  chmod +x "${posthoc_script}"

  rm -f "${run_dir}/EXIT_CODE" "${run_dir}/FINISHED" "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  date '+%F %T %Z' > "${run_dir}/STARTED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${train_script} > ${log_dir}/launcher.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${log_dir}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"

  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: ${run_name}

## Hypothesis

${HYPOTHESES[$i]}

## Command

\`\`\`bash
LATENTFM_OBSINFO_WEIGHT_ACK=gate_pass_bounded_smoke bash ${ROOT}/ops/launch_latentfm_observable_information_condition_weight_smoke_20260629.sh
\`\`\`

## Runtime classification

Long GPU training plus internal posthoc task. Use 30-minute cadence for checks.

## Start time

$(cat "${run_dir}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${session}\`

Physical GPU: ${gpu}

## Log path

\`${log_dir}/launcher.log\`

Posthoc log:

\`${log_dir}/posthoc.log\`

## Expected outputs

* \`${out_dir}/best.pt\`
* \`${run_dir}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`
* \`${run_dir}/posthoc_eval_internal/condition_family_eval_candidate_internal_ode20.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${log_dir}/launcher.log
cat ${run_dir}/EXIT_CODE 2>/dev/null || echo "still running"
cat ${run_dir}/POSTHOC_EXIT_CODE 2>/dev/null || echo "posthoc not complete"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Split file: \`${TRAIN_SPLIT}\`
- Condition weight file: \`${weight_file}\`
- Gate: \`${GATE_JSON}\`
- Selection/eval is internal to this train-only scaling split; canonical multi
  and Track C query are not used.
- Promotion remains blocked until the observable arm beats the same-marginal
  random arm and subsequent controls/no-harm pass.
EOF
  echo "Launched ${run_name} on physical GPU ${gpu}"
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_observable_information_condition_weight_smoke_20260629

## Command

\`\`\`bash
LATENTFM_OBSINFO_WEIGHT_ACK=gate_pass_bounded_smoke bash ${ROOT}/ops/launch_latentfm_observable_information_condition_weight_smoke_20260629.sh
\`\`\`

## Runtime classification

Long GPU two-arm mechanism smoke. Child runs have their own RUN_STATUS.md.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

* \`lfm_obsinfo_w_observable_${TOTAL_STEPS}_s${SEED}\`
* \`lfm_obsinfo_w_random_${TOTAL_STEPS}_s${SEED}\`

## Log path

\`${LOG_ROOT}/<run_name>/launcher.log\`

## Expected outputs

* \`${RUN_ROOT}/xverse_obsinfo_condition_weight_observable_${TOTAL_STEPS}step_seed${SEED}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`
* \`${RUN_ROOT}/xverse_obsinfo_condition_weight_random_${TOTAL_STEPS}step_seed${SEED}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`

## How to check manually

\`\`\`bash
tmux ls
cat ${RUN_ROOT}/xverse_obsinfo_condition_weight_observable_${TOTAL_STEPS}step_seed${SEED}/EXIT_CODE 2>/dev/null || echo "observable still running"
cat ${RUN_ROOT}/xverse_obsinfo_condition_weight_random_${TOTAL_STEPS}step_seed${SEED}/EXIT_CODE 2>/dev/null || echo "random still running"
nvidia-smi
\`\`\`

## Current status

Started observable and random arms.

## Notes

- CPU gate/report: \`${GATE_JSON}\`
- This is mechanism-only; no canonical multi or Track C query selection.
EOF
