#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

PACKET_JSON=${LATENTFM_SCALING_V2_INFO_PACKET_JSON:-${ROOT}/reports/scaling_v2_condition_information_packet_audit_20260628/latentfm_scaling_v2_condition_information_packet_audit_20260628.json}
HIGH_SPLIT=${LATENTFM_SCALING_V2_INFO_HIGH_SPLIT:-${ROOT}/reports/scaling_v2_condition_information_draft_splits_20260628/draft_split_seed42_xverse_info_composite_high_from_cap120_all_v2.json}
LOW_SPLIT=${LATENTFM_SCALING_V2_INFO_LOW_SPLIT:-${ROOT}/reports/scaling_v2_condition_information_draft_splits_20260628/draft_split_seed42_xverse_info_composite_low_from_cap120_all_v2.json}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
GPU_HELPER=${ROOT}/ops/select_available_gpus.py

RUN_ROOT=${LATENTFM_SCALING_V2_INFO_RUN_ROOT:-${ROOT}/runs/latentfm_scaling_v2_condition_information_highlow_smoke_20260628}
OUT_ROOT=${LATENTFM_SCALING_V2_INFO_OUT_ROOT:-${COUPLED}/output/latentfm_runs/scaling_v2_condition_information_highlow_smoke_20260628}
LOG_ROOT=${LATENTFM_SCALING_V2_INFO_LOG_ROOT:-${ROOT}/logs/latentfm_scaling_v2_condition_information_highlow_smoke_20260628}
REPORT_DIR=${LATENTFM_SCALING_V2_INFO_REPORT_DIR:-${ROOT}/reports/scaling_v2_condition_information_highlow_smoke_20260628}
RUN_PREFIX=${LATENTFM_SCALING_V2_INFO_RUN_PREFIX:-xverse_scaling_v2_info}
SESSION_PREFIX=${LATENTFM_SCALING_V2_INFO_SESSION_PREFIX:-lfm_scv2}
RUN_STATUS_TITLE=${LATENTFM_SCALING_V2_INFO_RUN_STATUS_TITLE:-latentfm_scaling_v2_condition_information_highlow_smoke_20260628}
ALLOWED_PACKET_STATUS=${LATENTFM_SCALING_V2_INFO_ALLOWED_PACKET_STATUS:-scaling_v2_condition_information_packet_audit_pass_prepare_gpu_smoke}
LAUNCH_COMMAND=${LATENTFM_SCALING_V2_INFO_LAUNCH_COMMAND:-bash ${ROOT}/ops/launch_latentfm_scaling_v2_condition_information_highlow_smoke_20260628.sh}

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${REPORT_DIR}"

for required in \
  "${PACKET_JSON}" \
  "${HIGH_SPLIT}" \
  "${LOW_SPLIT}" \
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

"${PYTHON}" - "${PACKET_JSON}" "${ALLOWED_PACKET_STATUS}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
allowed = {item.strip() for item in sys.argv[2].split(",") if item.strip()}
status = payload.get("status")
if status not in allowed:
    raise SystemExit(f"packet/manifest status {status!r} not in allowed {sorted(allowed)!r}")
if payload.get("gpu_authorized") not in (False, None):
    raise SystemExit("packet audit should remain GPU-authorized False; launcher owns the new bounded smoke protocol")
PY

declare -A SPLITS
SPLITS[high]="${HIGH_SPLIT}"
SPLITS[low]="${LOW_SPLIT}"
ARMS=(high low)
TOTAL_STEPS=${LATENTFM_SCALING_V2_INFO_STEPS:-2000}
SEED=${LATENTFM_SCALING_V2_INFO_SEED:-42}
LR=${LATENTFM_SCALING_V2_INFO_LR:-1e-4}
DS_ALPHA=${LATENTFM_SCALING_V2_INFO_DS_ALPHA:-1.0}
DS_LOSS_ALPHA=${LATENTFM_SCALING_V2_INFO_DS_LOSS_ALPHA:-0.0}
CONDITION_VISIT_POWER=${LATENTFM_SCALING_V2_INFO_CONDITION_VISIT_POWER:-1.0}
CONDITION_VISIT_CAP=${LATENTFM_SCALING_V2_INFO_CONDITION_VISIT_CAP:-0}
COMPOSITION_DELTA_LOSS_WEIGHT=${LATENTFM_SCALING_V2_INFO_COMPOSITION_DELTA_LOSS_WEIGHT:-0.06}
COMPOSITION_DELTA_LOSS_WARMUP_START=${LATENTFM_SCALING_V2_INFO_COMPOSITION_DELTA_LOSS_WARMUP_START:-500}
COMPOSITION_DELTA_LOSS_WARMUP_END=${LATENTFM_SCALING_V2_INFO_COMPOSITION_DELTA_LOSS_WARMUP_END:-1500}
ENDPOINT_DELTA_LOSS_WEIGHT=${LATENTFM_SCALING_V2_INFO_ENDPOINT_DELTA_LOSS_WEIGHT:-5.0}
ENDPOINT_DELTA_LOSS_WARMUP_START=${LATENTFM_SCALING_V2_INFO_ENDPOINT_DELTA_LOSS_WARMUP_START:-500}
ENDPOINT_DELTA_LOSS_WARMUP_END=${LATENTFM_SCALING_V2_INFO_ENDPOINT_DELTA_LOSS_WARMUP_END:-1500}
ANCHOR_REPLAY_LOSS_WEIGHT=${LATENTFM_SCALING_V2_INFO_ANCHOR_REPLAY_LOSS_WEIGHT:-0.0}
ANCHOR_REPLAY_LOSS_WARMUP_START=${LATENTFM_SCALING_V2_INFO_ANCHOR_REPLAY_LOSS_WARMUP_START:-500}
ANCHOR_REPLAY_LOSS_WARMUP_END=${LATENTFM_SCALING_V2_INFO_ANCHOR_REPLAY_LOSS_WARMUP_END:-1500}
OT_THREADS=${LATENTFM_SCALING_V2_INFO_OT_THREADS:-2}
PREFETCH=${LATENTFM_SCALING_V2_INFO_PREFETCH:-4}
N_OT_WORKERS=${LATENTFM_SCALING_V2_INFO_N_OT_WORKERS:-2}

for arm in "${ARMS[@]}"; do
  run_name="${RUN_PREFIX}_${arm}_${TOTAL_STEPS}step_seed${SEED}"
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  session="${SESSION_PREFIX}_${arm}_${TOTAL_STEPS}_s${SEED}"
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_SCALING_V2_INFO_RERUN:-0}" != "1" ]]; then
    echo "Output exists for ${run_name}; set FORCE_LATENTFM_SCALING_V2_INFO_RERUN=1 to relaunch" >&2
    exit 3
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

echo "[$(date '+%F %T %Z')] resource audit before scaling-v2 launch (${RUN_PREFIX})" | tee "${RUN_ROOT}/logs/resource_launch_audit.log"
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
  --max-user-gpus 2 \
  --max-jobs-per-gpu 2 \
  --need 2 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
suggested = [int(x) for x in payload.get("suggested_job_gpus", [])]
system = payload.get("system") or {}
reasons = []
if len(suggested) < 2:
    reasons.append(f"only {len(suggested)} GPU job slots suggested for need=2")
if int(payload.get("max_user_gpus") or 0) > 2:
    reasons.append("max_user_gpus exceeds current active goal cap 2")
if int(payload.get("max_jobs_per_gpu") or 0) > 2:
    reasons.append("max_jobs_per_gpu exceeds current active goal cap 2")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
audit = {
    "status": "fail" if reasons else "pass",
    "assigned_gpus": suggested[:2],
    "reasons": reasons,
    "gpu_selection_json": str(sys.argv[1]),
    "system": system,
    "policy": "current active cap: max 2 physical GPUs, max 2 training jobs/GPU, CPU <=24 cores",
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

idx=0
for arm in "${ARMS[@]}"; do
  split_file=${SPLITS[$arm]}
  gpu=${ASSIGNED_GPUS[$idx]}
  idx=$((idx + 1))
  run_name="${RUN_PREFIX}_${arm}_${TOTAL_STEPS}step_seed${SEED}"
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  session="${SESSION_PREFIX}_${arm}_${TOTAL_STEPS}_s${SEED}"
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
export SPLIT_FILE=${split_file}
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
export LR=${LR}
export GAMMA=0.03
export GAMMA_WARMUP_START=500
export GAMMA_WARMUP_END=1500
export MMD_EVERY=4
export SELECTION_METRIC=test_mmd
export SELECTION_MMD_LAMBDA=1.0
export COMPOSITION_DELTA_LOSS_WEIGHT=${COMPOSITION_DELTA_LOSS_WEIGHT}
export COMPOSITION_DELTA_LOSS_WARMUP_START=${COMPOSITION_DELTA_LOSS_WARMUP_START}
export COMPOSITION_DELTA_LOSS_WARMUP_END=${COMPOSITION_DELTA_LOSS_WARMUP_END}
export ENDPOINT_DELTA_LOSS_WEIGHT=${ENDPOINT_DELTA_LOSS_WEIGHT}
export ENDPOINT_DELTA_LOSS_WARMUP_START=${ENDPOINT_DELTA_LOSS_WARMUP_START}
export ENDPOINT_DELTA_LOSS_WARMUP_END=${ENDPOINT_DELTA_LOSS_WARMUP_END}
export DS_ALPHA=${DS_ALPHA}
export DS_LOSS_ALPHA=${DS_LOSS_ALPHA}
export MIN_SELECTED_CONDITIONS_PER_DATASET=0
export CONDITION_VISIT_POWER=${CONDITION_VISIT_POWER}
export CONDITION_VISIT_CAP=${CONDITION_VISIT_CAP}
export ANCHOR_REPLAY_LOSS_WEIGHT=${ANCHOR_REPLAY_LOSS_WEIGHT}
export ANCHOR_REPLAY_LOSS_WARMUP_START=${ANCHOR_REPLAY_LOSS_WARMUP_START}
export ANCHOR_REPLAY_LOSS_WARMUP_END=${ANCHOR_REPLAY_LOSS_WARMUP_END}
export ANCHOR_REPLAY_CHECKPOINT=${ANCHOR_CKPT}
export ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
export ANCHOR_REPLAY_CONDITION_FILTER=all
export OT_THREADS=${OT_THREADS}
export PREFETCH=${PREFETCH}
export N_OT_WORKERS=${N_OT_WORKERS}
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
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${split_file} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
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

Condition-information split arm \`${arm}\` tests whether train-condition
information/observability changes LatentFM internal generalization under a
matched high-vs-low design. This is a bounded mechanism smoke, not promotion.

## Command

\`\`\`bash
${LAUNCH_COMMAND}
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

- Split file: \`${split_file}\`
- Uses packet audit: \`${PACKET_JSON}\`
- Selection is internal to this train-only scaling split; canonical multi and
  Track C query are not used.
- Promotion remains blocked until high beats low, placebo/random controls are
  run, and a separate dual-baseline no-harm gate passes.
EOF
  echo "Launched ${run_name} on physical GPU ${gpu}"
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_STATUS_TITLE}

## Command

\`\`\`bash
${LAUNCH_COMMAND}
\`\`\`

## Runtime classification

Long GPU high-vs-low mechanism smoke. Child runs have their own RUN_STATUS.md.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

* \`${SESSION_PREFIX}_high_${TOTAL_STEPS}_s${SEED}\`
* \`${SESSION_PREFIX}_low_${TOTAL_STEPS}_s${SEED}\`

## Log path

\`${LOG_ROOT}/<run_name>/launcher.log\`

## Expected outputs

* \`${RUN_ROOT}/${RUN_PREFIX}_high_${TOTAL_STEPS}step_seed${SEED}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`
* \`${RUN_ROOT}/${RUN_PREFIX}_low_${TOTAL_STEPS}step_seed${SEED}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`

## How to check manually

\`\`\`bash
tmux ls
cat ${RUN_ROOT}/${RUN_PREFIX}_high_${TOTAL_STEPS}step_seed${SEED}/EXIT_CODE 2>/dev/null || echo "high still running"
cat ${RUN_ROOT}/${RUN_PREFIX}_low_${TOTAL_STEPS}step_seed${SEED}/EXIT_CODE 2>/dev/null || echo "low still running"
nvidia-smi
\`\`\`

## Current status

Started high and low arms.

## Notes

- Current active cap applied: max 2 physical GPUs, max 2 training jobs/GPU,
  CPU <=24 cores.
- GPU authorized only for this bounded high-vs-low smoke after packet audit
  pass; no promotion/no canonical claims.
EOF
