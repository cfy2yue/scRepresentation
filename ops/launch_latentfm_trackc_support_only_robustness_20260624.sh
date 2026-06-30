#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_TRACKC_SUPPORT_ONLY_ACK:-}" != "support_present_ablation_pass_fixed_robustness" ]]; then
  cat >&2 <<'EOF'
Refusing to launch Track C support-only robustness smoke.

Set:
  LATENTFM_TRACKC_SUPPORT_ONLY_ACK=support_present_ablation_pass_fixed_robustness

Boundary:
  - safe trainselect support-val only for selection/posthoc
  - no held-out Track C query
  - no canonical metrics or canonical multi selection
  - bounded fixed protocol, not a final multi claim
EOF
  exit 4
fi

RUN_NAME=${LATENTFM_TRACKC_SUPPORT_ONLY_RUN_NAME:-xverse_trackc_support_only_resfilm_ep050_replay2_2k_seed43}
RUN_ROOT=${LATENTFM_TRACKC_SUPPORT_ONLY_RUN_ROOT:-${ROOT}/runs/latentfm_trackc_support_only_robustness_20260624}
OUT_ROOT=${LATENTFM_TRACKC_SUPPORT_ONLY_OUT_ROOT:-${COUPLED}/output/latentfm_runs/trackc_support_only_robustness_20260624}
LOG_ROOT=${LATENTFM_TRACKC_SUPPORT_ONLY_LOG_ROOT:-${ROOT}/logs/latentfm_trackc_support_only_robustness_20260624}
PAIR_TYPE_FILTER=${LATENTFM_TRACKC_SUPPORT_ONLY_PAIR_TYPE_FILTER:-off}
RUN_SEED=${LATENTFM_TRACKC_SUPPORT_ONLY_SEED:-43}
TOTAL_STEPS=${LATENTFM_TRACKC_SUPPORT_ONLY_TOTAL_STEPS:-2000}
RUN_HYPOTHESIS=${LATENTFM_TRACKC_SUPPORT_ONLY_HYPOTHESIS:-Fixed Track C support-only robustness: after support-present ablation controls collapsed for the seed42 v2 support-context run, a capped repeat should retain material safe-trainselect support-val gain only when real support context is present.}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
TRAINSELECT_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
ROUTE_FILE=${ROOT}/reports/latentfm_trackc_residual_operator_route_teacher_20260623.json
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
GATE_JSON=${ROOT}/reports/latentfm_trackc_support_present_ablation_reproducibility_gate_20260624.json
SUMMARIZER=${ROOT}/ops/summarize_latentfm_trackc_support_only_robustness_20260624.py

run_dir=${RUN_ROOT}/${RUN_NAME}
out_dir=${OUT_ROOT}/${RUN_NAME}
log_dir=${LOG_ROOT}/${RUN_NAME}
mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${log_dir}" "${OUT_ROOT}" "${ROOT}/reports"

"${PYTHON}" - "${GATE_JSON}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "trackc_support_present_ablation_reproducibility_gate_pass_gpu_protocol_next":
    raise SystemExit(f"support-present ablation gate is not pass: {payload.get('status')}")
if payload.get("gpu_authorized") is not True:
    raise SystemExit("support-present ablation gate did not authorize GPU protocol")
boundary = payload.get("boundary") or {}
if boundary.get("reads_raw_heldout_query") or boundary.get("reads_canonical_multi_for_selection"):
    raise SystemExit("gate boundary unexpectedly reads query/canonical multi")
PY

for required in \
  "${DATA_DIR}/manifest.json" \
  "${TRAINSELECT_SPLIT}" \
  "${ROUTE_FILE}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}" \
  "${SUMMARIZER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${out_dir}" && "${FORCE_TRACKC_SUPPORT_ONLY_RERUN:-0}" != "1" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_TRACKC_SUPPORT_ONLY_RERUN=1 to relaunch" >&2
  exit 3
fi
if tmux has-session -t "trackc_support_only_${RUN_NAME}" 2>/dev/null; then
  echo "tmux session already exists: trackc_support_only_${RUN_NAME}" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] exact GPU status before Track C support-only launch" | tee "${run_dir}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${run_dir}/logs/gpu_launch_audit.log"
free -h | tee -a "${run_dir}/logs/gpu_launch_audit.log"
df -h "${ROOT}" | tee -a "${run_dir}/logs/gpu_launch_audit.log"
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 30 | tee -a "${run_dir}/logs/gpu_launch_audit.log"

gpu_json="${run_dir}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 4 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${run_dir}/logs/gpu_selection.stderr"

assignment_json="${run_dir}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
suggested = [int(x) for x in payload.get("suggested_job_gpus", [])]
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "assigned_gpus": suggested[:1],
    "allowed_physical_user_gpus": payload.get("allowed_physical_user_gpus"),
    "active_user_gpus": payload.get("active_user_gpus"),
    "new_physical_slots": payload.get("new_physical_slots"),
    "max_user_gpus": payload.get("max_user_gpus"),
    "max_jobs_per_gpu": payload.get("max_jobs_per_gpu"),
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if not suggested:
    reasons.append("no suggested GPU job slot for need=1")
if int(payload.get("max_user_gpus") or 0) > 4:
    reasons.append("max_user_gpus exceeds hard cap 4")
if int(payload.get("max_jobs_per_gpu") or 0) > 4:
    reasons.append("max_jobs_per_gpu exceeds cap 4")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
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

train_script="${run_dir}/scripts/run_${RUN_NAME}.sh"
posthoc_script="${run_dir}/scripts/posthoc_${RUN_NAME}.sh"

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
export LOG_ROOT=${log_dir}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${gpu}
export RUN_TAG=${RUN_NAME}
export SEED=${RUN_SEED}
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=1
export ANCHOR_REPLAY_CHECKPOINT=${ANCHOR_CKPT}
export ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
export FINETUNE_TRAINABLE_SCOPE=support_film_adapter
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
export CONDITION_DELTA_HEAD_USE_IN_MODEL=0
export CONDITION_DELTA_IN_MODEL_FILTER=gene_multi
export CONDITION_DELTA_HEAD_LOSS_WEIGHT=0.0
export ADDITIVE_CONDITION_DELTA_LOSS_WEIGHT=0.0
export CONDITION_PRIOR_DELTA_LOSS_WEIGHT=0.0
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WEIGHT=0.0
export TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=0.0
export TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START=0
export TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END=500
export TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT=0.50
export TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START=0
export TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END=500
export TRACKC_ROUTED_DISTILL_ROUTE_FILE=${ROUTE_FILE}
export TRACKC_ROUTED_DISTILL_BANK_SPLIT_FILE=${TRAINSELECT_SPLIT}
export TRACKC_ROUTED_DISTILL_TARGET_FRAME=endpoint_delta
export TRACKC_ROUTED_DISTILL_MEMORY_MODE=jaccard
export TRACKC_ROUTED_DISTILL_MEMORY_K=3
export TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE=0.25
export TRACKC_ROUTED_DISTILL_MEMORY_SCOPE=all_dataset
export TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL=0
export TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL=0
export TRACKC_SUPPORT_FILM_USE_IN_MODEL=1
export TRACKC_SUPPORT_CONTEXT_DIM=384
export TRACKC_SUPPORT_CONTEXT_SOURCE=routed_distill_target
export TRACKC_SUPPORT_CONTEXT_PAIR_TYPE_FILTER=${PAIR_TYPE_FILTER}
export CONDITION_PRIOR_BANK_MAX_CELLS=512
export ANCHOR_REPLAY_LOSS_WEIGHT=2.0
export ANCHOR_REPLAY_LOSS_WARMUP_START=0
export ANCHOR_REPLAY_LOSS_WARMUP_END=500
export ANCHOR_REPLAY_CONDITION_FILTER=all
export PERT_POOL_AGGREGATIONS="sum mean max min"
export PERT_POOL_SCALE_INIT="0.5 1.0 1.0 1.0"
export PERT_POOL_FUSION_MODE=sum
export PERT_PAIRWISE_MODE=off
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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
out_eval=${run_dir}/posthoc_eval
mkdir -p "\${out_eval}"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${TRAINSELECT_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 512)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_multi --out "\${out_eval}/support_anchor_split_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene structure_multi test_multi --out "\${out_eval}/support_anchor_family_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_multi --out "\${out_eval}/support_candidate_split_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene structure_multi test_multi --out "\${out_eval}/support_candidate_family_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_multi --support-context-control zero --out "\${out_eval}/support_zero_candidate_split_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_multi --support-context-control shuffle_condition --out "\${out_eval}/support_shuffle_condition_candidate_split_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_multi --force-support-context-absent --out "\${out_eval}/support_absent_support_candidate_split_ode20.json" "\${common[@]}"
${PYTHON} ${SUMMARIZER} \
  --run-root ${run_dir} \
  --out-json ${ROOT}/reports/latentfm_trackc_support_only_robustness_decision_${RUN_NAME}.json \
  --out-md ${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_ONLY_ROBUSTNESS_DECISION_${RUN_NAME}.md \
  --python ${PYTHON} \
  --n-boot 2000 \
  --seed ${RUN_SEED}
EOF
chmod +x "${posthoc_script}"

rm -f "${run_dir}/${RUN_NAME}.EXIT_CODE" "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/FINISHED" "${run_dir}/POSTHOC_FINISHED"
date '+%F %T %Z' > "${run_dir}/STARTED"
session="trackc_support_only_${RUN_NAME}"
tmux new -d -s "${session}" \
  "bash -lc 'bash ${train_script} > ${log_dir}/train.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/${RUN_NAME}.EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${log_dir}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"

cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Hypothesis

${RUN_HYPOTHESIS}

## Command

\`\`\`bash
LATENTFM_TRACKC_SUPPORT_ONLY_ACK=support_present_ablation_pass_fixed_robustness \\
LATENTFM_TRACKC_SUPPORT_ONLY_RUN_NAME=${RUN_NAME} \\
LATENTFM_TRACKC_SUPPORT_ONLY_PAIR_TYPE_FILTER=${PAIR_TYPE_FILTER} \\
LATENTFM_TRACKC_SUPPORT_ONLY_SEED=${RUN_SEED} \\
LATENTFM_TRACKC_SUPPORT_ONLY_TOTAL_STEPS=${TOTAL_STEPS} \\
bash ${ROOT}/ops/launch_latentfm_trackc_support_only_robustness_20260624.sh
\`\`\`

## Runtime classification

Long GPU training plus query-free support-val posthoc. Use 30-minute cadence
for result checks; use hourly lightweight resource checkpoints.

## Start time

$(cat "${run_dir}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${session}\`

Physical GPU: ${gpu}

## Log path

\`${log_dir}/train.log\`

Posthoc log:

\`${log_dir}/posthoc.log\`

## Expected outputs

* \`${out_dir}/best.pt\`
* \`${run_dir}/posthoc_eval/support_candidate_split_ode20.json\`
* \`${run_dir}/posthoc_eval/support_zero_candidate_split_ode20.json\`
* \`${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_ONLY_ROBUSTNESS_DECISION_${RUN_NAME}.md\`

## How to check manually

\`\`\`bash
tmux ls
cat ${run_dir}/${RUN_NAME}.EXIT_CODE 2>/dev/null || echo "still running"
cat ${run_dir}/POSTHOC_EXIT_CODE 2>/dev/null || echo "posthoc not complete"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Query-free Track C robustness smoke, not a final multi claim.
- Split: \`${TRAINSELECT_SPLIT}\`
- Held-out Track C query is not read.
- Canonical metrics and canonical multi are not read for selection or this decision.
- Seed: ${RUN_SEED}; total steps: ${TOTAL_STEPS}; trainable scope: support_film_adapter.
- Support pair-type filter: \`${PAIR_TYPE_FILTER}\`.
- Stop/promotion signal: actual support pp delta >= +0.04, support pp p_harm <= 0.20, support MMD delta <= 0, family no-harm, and zero/shuffle/forced-absent support controls collapse at least 0.02 below actual.
EOF

echo "Launched ${RUN_NAME} on GPU ${gpu} in tmux ${session}"
