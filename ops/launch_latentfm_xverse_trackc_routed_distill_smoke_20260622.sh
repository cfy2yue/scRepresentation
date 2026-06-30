#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_NAME=${LATENTFM_TRACKC_RUN_NAME:-xverse_trackc_route_condprior_w05_replay1_2k_seed42}
RUN_ROOT=${LATENTFM_TRACKC_RUN_ROOT:-${ROOT}/runs/latentfm_xverse_trackc_routed_distill_20260622}
OUT_ROOT=${LATENTFM_TRACKC_OUT_ROOT:-${COUPLED}/output/latentfm_runs/xverse_trackc_routed_distill_20260622}
LOG_ROOT=${LATENTFM_TRACKC_LOG_ROOT:-${ROOT}/logs/latentfm_xverse_trackc_routed_distill_20260622}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
TRAINSELECT_SPLIT=${BIFLOW_DIR}/split_seed42_multi_support_v2_trainselect.json
TRAINSELECT_SPLIT=${LATENTFM_TRACKC_TRAINSELECT_SPLIT:-${TRAINSELECT_SPLIT}}
TRACKC_BANK_SPLIT_FILE=${LATENTFM_TRACKC_BANK_SPLIT_FILE:-}
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
ROUTE_FILE=${LATENTFM_TRACKC_ROUTE_FILE:-${ROOT}/reports/latentfm_trackc_support_route_teacher_20260622.json}
ANCHOR_CKPT=${LATENTFM_TRACKC_ANCHOR_CKPT:-${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt}
INIT_CHECKPOINT_USE_EMA=${LATENTFM_TRACKC_INIT_CHECKPOINT_USE_EMA:-0}
ANCHOR_REPLAY_CHECKPOINT_USE_EMA=${LATENTFM_TRACKC_ANCHOR_REPLAY_CHECKPOINT_USE_EMA:-0}
FINETUNE_TRAINABLE_SCOPE=${LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE:-condition_prior_adapter}
TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=${LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT:-0.5}
TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START=${LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START:-0}
TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END=${LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END:-500}
TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT=${LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT:-0.0}
TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START=${LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START:-0}
TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END=${LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END:-500}
TRACKC_ROUTED_DISTILL_MEMORY_MODE=${LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MODE:-off}
TRACKC_ROUTED_DISTILL_MEMORY_K=${LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_K:-3}
TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE=${LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE:-0.25}
TRACKC_ROUTED_DISTILL_MEMORY_SCOPE=${LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_SCOPE:-same_dataset}
TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL=${LATENTFM_TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL:-0}
TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL=${LATENTFM_TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL:-0}
TRACKC_SUPPORT_FILM_USE_IN_MODEL=${LATENTFM_TRACKC_SUPPORT_FILM_USE_IN_MODEL:-0}
TRACKC_SUPPORT_CONTEXT_DIM=${LATENTFM_TRACKC_SUPPORT_CONTEXT_DIM:-0}
TRACKC_SUPPORT_CONTEXT_SOURCE=${LATENTFM_TRACKC_SUPPORT_CONTEXT_SOURCE:-off}
if [[ -z "${TRACKC_BANK_SPLIT_FILE}" ]] && {
  [[ "${TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL}" == "1" || "${TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL}" == "true" ]] ||
  [[ "${TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL}" == "1" || "${TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL}" == "true" ]] ||
  [[ "${TRACKC_SUPPORT_FILM_USE_IN_MODEL}" == "1" || "${TRACKC_SUPPORT_FILM_USE_IN_MODEL}" == "true" ]];
}; then
  TRACKC_BANK_SPLIT_FILE=${TRAINSELECT_SPLIT}
fi
CONDITION_PRIOR_BANK_MAX_CELLS=${LATENTFM_TRACKC_CONDITION_PRIOR_BANK_MAX_CELLS:-512}
ANCHOR_REPLAY_LOSS_WEIGHT=${LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WEIGHT:-1.0}
ANCHOR_REPLAY_LOSS_WARMUP_START=${LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_START:-0}
ANCHOR_REPLAY_LOSS_WARMUP_END=${LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_END:-500}
ANCHOR_REPLAY_CONDITION_FILTER=${LATENTFM_TRACKC_ANCHOR_REPLAY_CONDITION_FILTER:-non_gene_multi}
CONDITION_DELTA_HEAD_USE_IN_MODEL=${LATENTFM_TRACKC_CONDITION_DELTA_HEAD_USE_IN_MODEL:-1}
PERT_PAIRWISE_MODE=${LATENTFM_TRACKC_PERT_PAIRWISE_MODE:-off}
TOTAL_STEPS=${LATENTFM_TRACKC_TOTAL_STEPS:-2000}
TRACKC_SMOKE_HYPOTHESIS=${LATENTFM_TRACKC_SMOKE_HYPOTHESIS:-route-dataset-focused training split plus full trainselect routed teacher bank}
FORCE_GPU=${LATENTFM_TRACKC_FORCE_GPU:-}
RELAXED_GPU_SELECTION=${LATENTFM_TRACKC_RELAXED_GPU_SELECTION:-0}
RELAXED_GPU_MIN_FREE_MIB=${LATENTFM_TRACKC_RELAXED_GPU_MIN_FREE_MIB:-8192}
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

run_root=${RUN_ROOT}/${RUN_NAME}
log_root=${LOG_ROOT}/${RUN_NAME}
out_dir=${OUT_ROOT}/${RUN_NAME}
mkdir -p "${run_root}/logs" "${run_root}/scripts" "${log_root}" "${OUT_ROOT}" "${ROOT}/reports"

for required in \
  "${DATA_DIR}/manifest.json" \
  "${TRAINSELECT_SPLIT}" \
  "${CANONICAL_SPLIT}" \
  "${ROUTE_FILE}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done
if [[ -n "${TRACKC_BANK_SPLIT_FILE}" && ! -e "${TRACKC_BANK_SPLIT_FILE}" ]]; then
  echo "Missing Track C routed bank split artifact: ${TRACKC_BANK_SPLIT_FILE}" >&2
  exit 2
fi

if [[ -e "${out_dir}" && "${FORCE_TRACKC_RERUN:-0}" != "1" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_TRACKC_RERUN=1 to relaunch" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] exact GPU status before Track C launch" | tee "${run_root}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${run_root}/logs/gpu_launch_audit.log"

gpu_json="${run_root}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 5 \
  --max-jobs-per-gpu 4 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${run_root}/logs/gpu_selection.stderr"

assignment_json="${run_root}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" "${FORCE_GPU}" "${RELAXED_GPU_SELECTION}" "${RELAXED_GPU_MIN_FREE_MIB}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
force_gpu = str(sys.argv[3]).strip()
relaxed = str(sys.argv[4]).strip().lower() in {"1", "true", "yes", "on"}
relaxed_min_free = int(float(str(sys.argv[5]).strip() or "8192"))
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
stable = [g for g in gpus.values() if g.get("stable_light")]
stable_count = len(stable)
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
physical_budget = min(5, stable_count) if stable_count >= 5 else max(0, min(5, stable_count - 1))
if relaxed:
    physical_budget = min(5, len(gpus))
chosen = None
candidate_order = [int(x) for x in payload.get("candidate_order", [])]
if force_gpu:
    candidate_order = [int(force_gpu)]
for idx in candidate_order:
    if idx not in gpus:
        continue
    gpu = gpus[idx]
    max_util = int(gpu.get("max_sample_utilization_gpu_pct") or 0)
    max_mem = int(gpu.get("max_sample_memory_used_mib") or 0)
    total_mem = int(gpu.get("memory_total_mib") or 0)
    free_mem = max(0, total_mem - max_mem)
    relaxed_usable = relaxed and max_util < 10 and free_mem >= relaxed_min_free
    if not (gpu.get("available") or relaxed_usable):
        continue
    slots_free = int(gpu.get("colocation_slots_free", 0))
    if slots_free <= 0 and relaxed_usable:
        slots_free = 1
    if slots_free <= 0:
        continue
    if len(active_user | {idx}) <= physical_budget:
        chosen = idx
        break
system = payload.get("system") or {}
relaxed_usable_gpus = []
for idx, gpu in gpus.items():
    max_util = int(gpu.get("max_sample_utilization_gpu_pct") or 0)
    max_mem = int(gpu.get("max_sample_memory_used_mib") or 0)
    total_mem = int(gpu.get("memory_total_mib") or 0)
    free_mem = max(0, total_mem - max_mem)
    if max_util < 10 and free_mem >= relaxed_min_free:
        relaxed_usable_gpus.append(
            {
                "index": idx,
                "free_mib_after_max_sample": free_mem,
                "max_sample_memory_used_mib": max_mem,
                "max_sample_utilization_gpu_pct": max_util,
                "reason": gpu.get("reason"),
                "compute_users": gpu.get("compute_users", []),
            }
        )
audit = {
    "status": "pass",
    "stable_light_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "active_user_gpus": sorted(active_user),
    "assigned_gpus": ([] if chosen is None else [chosen]),
    "force_gpu": force_gpu or None,
    "relaxed_gpu_selection": relaxed,
    "relaxed_gpu_min_free_mib": relaxed_min_free,
    "relaxed_usable_low_util_gpus": relaxed_usable_gpus,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if chosen is None and force_gpu:
    reasons.append(f"forced GPU {force_gpu} unavailable after stable-light/leave-one-empty/budget rules")
elif chosen is None:
    reasons.append("no GPU available after stable-light/leave-one-empty/budget rules")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

gpu="$("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["assigned_gpus"][0])
PY
)"

train_script="${run_root}/scripts/run_${RUN_NAME}.sh"
cat > "${train_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
export LATENT_BACKBONE=xverse
export DATA_DIR=${DATA_DIR}
export BIFLOW_DIR=${BIFLOW_DIR}
export SPLIT_FILE=${TRAINSELECT_SPLIT}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${LOG_ROOT}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${gpu}
export RUN_TAG=${RUN_NAME}
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=${INIT_CHECKPOINT_USE_EMA}
export ANCHOR_REPLAY_CHECKPOINT=${ANCHOR_CKPT}
export ANCHOR_REPLAY_CHECKPOINT_USE_EMA=${ANCHOR_REPLAY_CHECKPOINT_USE_EMA}
export FINETUNE_TRAINABLE_SCOPE=${FINETUNE_TRAINABLE_SCOPE}
export TOTAL_STEPS=${TOTAL_STEPS}
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export GAMMA=0.03
export GAMMA_WARMUP_START=500
export GAMMA_WARMUP_END=1500
export MMD_EVERY=4
export SELECTION_METRIC=pearson_pert_minus_mmd
export SELECTION_MMD_LAMBDA=0.5
export EVAL_MAX_CONDITIONS=0
export EVAL_MAX_CONDITIONS_PER_DATASET=0
export EVAL_MAX_MSE_CELLS=1024
export EVAL_MAX_MMD_CELLS=512
export EVAL_MAX_CHUNK=256
export CONDITION_DELTA_HEAD_USE_IN_MODEL=${CONDITION_DELTA_HEAD_USE_IN_MODEL}
export CONDITION_DELTA_IN_MODEL_FILTER=gene_multi
export CONDITION_DELTA_HEAD_LOSS_WEIGHT=0.0
export ADDITIVE_CONDITION_DELTA_LOSS_WEIGHT=0.0
export CONDITION_PRIOR_DELTA_LOSS_WEIGHT=0.0
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WEIGHT=0.0
export TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=${TRACKC_ROUTED_DISTILL_LOSS_WEIGHT}
export TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START=${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START}
export TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END=${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END}
export TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT=${TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT}
export TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START=${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START}
export TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END=${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END}
export TRACKC_ROUTED_DISTILL_ROUTE_FILE=${ROUTE_FILE}
export TRACKC_ROUTED_DISTILL_BANK_SPLIT_FILE=${TRACKC_BANK_SPLIT_FILE}
export TRACKC_ROUTED_DISTILL_TARGET_FRAME=endpoint_delta
export TRACKC_ROUTED_DISTILL_MEMORY_MODE=${TRACKC_ROUTED_DISTILL_MEMORY_MODE}
export TRACKC_ROUTED_DISTILL_MEMORY_K=${TRACKC_ROUTED_DISTILL_MEMORY_K}
export TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE=${TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE}
export TRACKC_ROUTED_DISTILL_MEMORY_SCOPE=${TRACKC_ROUTED_DISTILL_MEMORY_SCOPE}
export TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL=${TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL}
export TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL=${TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL}
export TRACKC_SUPPORT_FILM_USE_IN_MODEL=${TRACKC_SUPPORT_FILM_USE_IN_MODEL}
export TRACKC_SUPPORT_CONTEXT_DIM=${TRACKC_SUPPORT_CONTEXT_DIM}
export TRACKC_SUPPORT_CONTEXT_SOURCE=${TRACKC_SUPPORT_CONTEXT_SOURCE}
export CONDITION_PRIOR_BANK_MAX_CELLS=${CONDITION_PRIOR_BANK_MAX_CELLS}
export ANCHOR_REPLAY_LOSS_WEIGHT=${ANCHOR_REPLAY_LOSS_WEIGHT}
export ANCHOR_REPLAY_LOSS_WARMUP_START=${ANCHOR_REPLAY_LOSS_WARMUP_START}
export ANCHOR_REPLAY_LOSS_WARMUP_END=${ANCHOR_REPLAY_LOSS_WARMUP_END}
export ANCHOR_REPLAY_CONDITION_FILTER=${ANCHOR_REPLAY_CONDITION_FILTER}
export PERT_POOL_AGGREGATIONS="sum mean max min"
export PERT_POOL_SCALE_INIT="0.5 1.0 1.0 1.0"
export PERT_POOL_FUSION_MODE=sum
export PERT_PAIRWISE_MODE=${PERT_PAIRWISE_MODE}
export PERT_GENE_PROJECTOR_HIDDEN=1024
export PERT_CHEM_PROJECTOR_HIDDEN=1024
export PERT_TO_C_INIT_MODE=xavier_small
export USE_PERT_IN_FUSION=1
bash ${TRAIN_LAUNCHER}
EOF
chmod +x "${train_script}"

posthoc_script="${run_root}/scripts/posthoc_${RUN_NAME}.sh"
cat > "${posthoc_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
while [[ ! -f ${run_root}/${RUN_NAME}.EXIT_CODE ]]; do
  sleep 1800
done
code="\$(cat ${run_root}/${RUN_NAME}.EXIT_CODE)"
if [[ "\${code}" != "0" ]]; then
  echo "training failed for ${RUN_NAME}; skip posthoc" >&2
  exit "\${code}"
fi
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
out_eval=${run_root}/posthoc_eval
mkdir -p "\${out_eval}"
common_trainselect=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${TRAINSELECT_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 512)
common_canonical=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 256 --eval-max-conditions-per-dataset 12 --eval-max-mse-cells 1024 --eval-max-mmd-cells 512)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_multi --out "\${out_eval}/support_anchor_split_ode20.json" "\${common_trainselect[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene structure_multi test_multi --out "\${out_eval}/support_anchor_family_ode20.json" "\${common_trainselect[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_multi --out "\${out_eval}/support_candidate_split_ode20.json" "\${common_trainselect[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene structure_multi test_multi --out "\${out_eval}/support_candidate_family_ode20.json" "\${common_trainselect[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --force-support-context-absent --out "\${out_eval}/canonical_anchor_split_ode20_stablecaps.json" "\${common_canonical[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --force-support-context-absent --out "\${out_eval}/canonical_anchor_family_ode20_stablecaps.json" "\${common_canonical[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --force-support-context-absent --out "\${out_eval}/canonical_candidate_split_ode20_stablecaps.json" "\${common_canonical[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --force-support-context-absent --out "\${out_eval}/canonical_candidate_family_ode20_stablecaps.json" "\${common_canonical[@]}"
${PYTHON} ${ROOT}/ops/summarize_latentfm_trackc_routed_distill_smoke_20260622.py \
  --run-root ${run_root} \
  --out-json ${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json \
  --out-md ${ROOT}/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_${RUN_NAME}.md \
  --n-boot 2000 \
  --seed 42 \
  --python ${PYTHON}
date > ${run_root}/posthoc.FINISHED
EOF
chmod +x "${posthoc_script}"

train_log="${run_root}/logs/${RUN_NAME}.train.log"
posthoc_log="${run_root}/logs/${RUN_NAME}.posthoc.log"
train_session="trackc_route_train_${RUN_NAME}"
posthoc_session="trackc_route_posthoc_${RUN_NAME}"

cat > "${run_root}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
bash ${train_script}
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

Training tmux: \`${train_session}\`

Posthoc watcher tmux: \`${posthoc_session}\`

Assigned GPU: \`${gpu}\`

## Log path

\`${train_log}\`

\`${posthoc_log}\`

## Expected outputs

* \`${out_dir}/best.pt\`
* \`${out_dir}/trackc_routed_distill_bank_summary.json\`
* \`${run_root}/posthoc_eval/support_candidate_split_ode20.json\`
* \`${run_root}/posthoc_eval/canonical_candidate_split_ode20_stablecaps.json\`
* \`${ROOT}/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_${RUN_NAME}.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${train_log}
tail -n 50 ${posthoc_log}
cat ${run_root}/${RUN_NAME}.EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Track C smoke only. Hypothesis: ${TRACKC_SMOKE_HYPOTHESIS}.

Uses trainselect support split for training/selection and
support-val posthoc; canonical split is evaluated only as no-harm diagnostic.
Canonical posthoc forces support context absent.
Optional routed teacher bank split: \`${TRACKC_BANK_SPLIT_FILE:-same as training split}\`.
Route file: \`${ROUTE_FILE}\`.
Memory teacher: mode=\`${TRACKC_ROUTED_DISTILL_MEMORY_MODE}\`,
k=\`${TRACKC_ROUTED_DISTILL_MEMORY_K}\`,
min_score=\`${TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE}\`,
scope=\`${TRACKC_ROUTED_DISTILL_MEMORY_SCOPE}\`,
bank_max_cells=\`${CONDITION_PRIOR_BANK_MAX_CELLS}\`.
Support context: use_in_model=\`${TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL}\`,
residual_use_in_model=\`${TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL}\`,
film_use_in_model=\`${TRACKC_SUPPORT_FILM_USE_IN_MODEL}\`,
dim=\`${TRACKC_SUPPORT_CONTEXT_DIM}\`,
source=\`${TRACKC_SUPPORT_CONTEXT_SOURCE}\`.
The held-out query split is intentionally not evaluated here.

Loss settings: routed_distill=\`${TRACKC_ROUTED_DISTILL_LOSS_WEIGHT}\`
warmup \`${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START}-${TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END}\`;
routed_endpoint=\`${TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT}\` warmup
\`${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START}-${TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END}\`;
condition_delta_head_use_in_model=\`${CONDITION_DELTA_HEAD_USE_IN_MODEL}\`;
anchor_replay=\`${ANCHOR_REPLAY_LOSS_WEIGHT}\` warmup
\`${ANCHOR_REPLAY_LOSS_WARMUP_START}-${ANCHOR_REPLAY_LOSS_WARMUP_END}\`
filter=\`${ANCHOR_REPLAY_CONDITION_FILTER}\`.
Total steps: \`${TOTAL_STEPS}\`.
EMA alignment: init_checkpoint_use_ema=\`${INIT_CHECKPOINT_USE_EMA}\`,
anchor_replay_checkpoint_use_ema=\`${ANCHOR_REPLAY_CHECKPOINT_USE_EMA}\`.
EOF

tmux new -d -s "${train_session}" "bash ${train_script} > ${train_log} 2>&1; echo \$? > ${run_root}/${RUN_NAME}.EXIT_CODE; date > ${run_root}/${RUN_NAME}.FINISHED"
tmux new -d -s "${posthoc_session}" "bash ${posthoc_script} > ${posthoc_log} 2>&1; echo \$? > ${run_root}/${RUN_NAME}.POSTHOC_EXIT_CODE"

echo "Launched ${RUN_NAME}"
echo "Training session: ${train_session}"
echo "Posthoc session: ${posthoc_session}"
echo "Assigned GPU: ${gpu}"
tmux ls | grep -E "${train_session}|${posthoc_session}" || true
tail -n 20 "${train_log}" || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
