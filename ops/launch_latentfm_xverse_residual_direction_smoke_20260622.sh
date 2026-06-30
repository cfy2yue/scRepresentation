#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM

if [[ "${LATENTFM_XVERSE_RESIDUAL_ACK:-}" != "after_active_single_background_gates" ]]; then
  cat >&2 <<'EOF'
Refusing to launch residual-direction smoke.

This is a gated P5 follow-up. First judge the active condition-delta and
metric-v2 single/background candidates, then relaunch with:

  LATENTFM_XVERSE_RESIDUAL_ACK=after_active_single_background_gates

Do not use this launcher for broad residual-loss sweeps.
EOF
  exit 4
fi

RUN_NAME=${LATENTFM_XVERSE_RESIDUAL_RUN_NAME:-xverse_residdir_trainonly_v2_w002_replay010_2k_seed42}
RUN_ROOT=${LATENTFM_XVERSE_RESIDUAL_RUN_ROOT:-${ROOT}/runs/latentfm_xverse_residual_direction_smoke_20260622}
OUT_ROOT=${LATENTFM_XVERSE_RESIDUAL_OUT_ROOT:-${COUPLED}/output/latentfm_runs/xverse_residual_direction_smoke_20260622}
LOG_ROOT=${LATENTFM_XVERSE_RESIDUAL_LOG_ROOT:-${ROOT}/logs/latentfm_xverse_residual_direction_smoke_20260622}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42.json
TRAINONLY_SPLIT=${LATENTFM_XVERSE_RESIDUAL_TRAIN_SPLIT:-${ROOT}/dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json}
TRAINONLY_PERT_MEANS=${LATENTFM_XVERSE_RESIDUAL_PERT_MEANS_FILE:-${ROOT}/runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz}
TRAIN_SEED=${LATENTFM_XVERSE_RESIDUAL_SEED:-42}
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
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
session=${LATENTFM_XVERSE_RESIDUAL_SESSION:-latentfm_xverse_residdir_20260622}
mkdir -p "${run_root}/logs" "${run_root}/scripts" "${log_root}" "${OUT_ROOT}" "${ROOT}/reports"

if [[ -z "${TRAINONLY_PERT_MEANS}" ]]; then
  echo "Refusing to launch: LATENTFM_XVERSE_RESIDUAL_PERT_MEANS_FILE is empty." >&2
  exit 4
fi
case "${TRAINONLY_PERT_MEANS}" in
  *xverse_trainonly_pert_means*) ;;
  *)
    cat >&2 <<EOF
Refusing to launch: residual losses require a train-only pert-means artifact.
Got: ${TRAINONLY_PERT_MEANS}
EOF
    exit 4
    ;;
esac

for required in \
  "${DATA_DIR}/manifest.json" \
  "${TRAINONLY_SPLIT}" \
  "${TRAINONLY_PERT_MEANS}" \
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}" \
  "${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${out_dir}" && "${FORCE_XVERSE_RESIDUAL_RERUN:-0}" != "1" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_XVERSE_RESIDUAL_RERUN=1 to relaunch" >&2
  exit 3
fi
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] exact GPU status before xverse residual-direction launch" | tee "${run_root}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${run_root}/logs/gpu_launch_audit.log"

assignment_json="${run_root}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
if [[ -n "${LATENTFM_XVERSE_RESIDUAL_GPU:-}" ]]; then
  "${PYTHON}" - "${LATENTFM_XVERSE_RESIDUAL_GPU}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
gpu = int(sys.argv[1])
payload = {
    "status": "pass",
    "assigned_gpus": [gpu],
    "manual_override": True,
    "note": "GPU must have been selected by an immediately preceding multi-sample audit.",
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
  gpu_json="manual_prelaunch_audit"
else
  gpu_json="${run_root}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" "${GPU_HELPER}" \
    --samples 3 \
    --interval-seconds 10 \
    --util-threshold-pct 10 \
    --memory-threshold-mib 4096 \
    --max-jobs-per-gpu 4 \
    --need 1 \
    --json-only \
    > "${gpu_json}" 2> "${run_root}/logs/gpu_selection.stderr"

  "${PYTHON}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
stable = [g for g in gpus.values() if g.get("stable_light")]
stable_count = len(stable)
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
physical_budget = min(4, stable_count) if stable_count >= 5 else max(0, min(4, stable_count - 1))
chosen = None
for idx in [int(x) for x in payload.get("candidate_order", [])]:
    gpu = gpus[idx]
    if not gpu.get("available"):
        continue
    if int(gpu.get("colocation_slots_free", 0)) <= 0:
        continue
    if len(active_user | {idx}) <= physical_budget:
        chosen = idx
        break
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "stable_light_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "active_user_gpus": sorted(active_user),
    "assigned_gpus": ([] if chosen is None else [chosen]),
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if chosen is None:
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
fi

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
export SPLIT_FILE=${TRAINONLY_SPLIT}
export PERT_MEANS_FILE=${TRAINONLY_PERT_MEANS}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${LOG_ROOT}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${gpu}
export RUN_TAG=${RUN_NAME}
export SEED=${TRAIN_SEED}
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export FINETUNE_TRAINABLE_SCOPE=all
export TOTAL_STEPS=${LATENTFM_XVERSE_RESIDUAL_TOTAL_STEPS:-2500}
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export GAMMA=0.03
export GAMMA_WARMUP_START=500
export GAMMA_WARMUP_END=1500
export MMD_EVERY=4
export SELECTION_METRIC=pearson_pert_minus_mmd
export SELECTION_MMD_LAMBDA=0.5
export PERT_RESIDUAL_DIRECTION_LOSS_WEIGHT=${LATENTFM_XVERSE_RESIDUAL_DIRECTION_WEIGHT:-0.02}
export PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_START=100
export PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_END=800
export PERT_RESIDUAL_CONTRASTIVE_LOSS_WEIGHT=${LATENTFM_XVERSE_RESIDUAL_CONTRASTIVE_WEIGHT:-0.0}
export PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_START=300
export PERT_RESIDUAL_CONTRASTIVE_LOSS_WARMUP_END=1000
export PERT_RESIDUAL_CONTRASTIVE_TEMPERATURE=0.10
export PERT_RESIDUAL_CONTRASTIVE_BANK_SIZE=256
export PERT_RESIDUAL_RELATIONAL_LOSS_WEIGHT=${LATENTFM_XVERSE_RESIDUAL_RELATIONAL_WEIGHT:-0.0}
export PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_START=300
export PERT_RESIDUAL_RELATIONAL_LOSS_WARMUP_END=1000
export ANCHOR_REPLAY_LOSS_WEIGHT=0.10
export ANCHOR_REPLAY_LOSS_WARMUP_START=100
export ANCHOR_REPLAY_LOSS_WARMUP_END=800
export ANCHOR_REPLAY_CONDITION_FILTER=all
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

posthoc_script="${run_root}/scripts/posthoc_${RUN_NAME}.sh"
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
eval_dir=${run_root}/posthoc_eval_canonical
mkdir -p "\${eval_dir}"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/split_group_eval_anchor_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/condition_family_eval_anchor_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/split_group_eval_candidate_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" "\${common[@]}"
${PYTHON} ${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py --candidate-split-json "\${eval_dir}/split_group_eval_candidate_ode20_canonical.json" --candidate-family-json "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" --n-boot 2000 --seed 42 --out-json "\${eval_dir}/single_background_candidate_gate.json" --out-md "\${eval_dir}/SINGLE_BACKGROUND_CANDIDATE_GATE.md"
${PYTHON} ${ROOT}/ops/summarize_latentfm_single_background_candidate_decision_20260622.py --gate-json "\${eval_dir}/single_background_candidate_gate.json" --label ${RUN_NAME} --title "LatentFM xverse residual-direction smoke candidate decision" --out-md "\${eval_dir}/SINGLE_BACKGROUND_CANDIDATE_DECISION.md"
EOF
chmod +x "${posthoc_script}"

rm -f "${run_root}/${RUN_NAME}.EXIT_CODE" "${run_root}/${RUN_NAME}.FINISHED" "${run_root}/POSTHOC_EXIT_CODE" "${run_root}/POSTHOC_FINISHED"
date '+%F %T %Z' > "${run_root}/${RUN_NAME}.STARTED"
tmux new -d -s "${session}" \
  "bash -lc 'bash ${train_script} > ${log_root}/train.log 2>&1; rc=\$?; echo \$rc > ${run_root}/${RUN_NAME}.EXIT_CODE; date \"+%F %T %Z\" > ${run_root}/${RUN_NAME}.FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${log_root}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${run_root}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_root}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"

cat > "${run_root}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
LATENTFM_XVERSE_RESIDUAL_ACK=after_active_single_background_gates bash ${ROOT}/ops/launch_latentfm_xverse_residual_direction_smoke_20260622.sh
\`\`\`

## Runtime classification

Long GPU training + posthoc task. Check at most every 30 minutes unless marker
files appear naturally.

## Start time

$(cat "${run_root}/${RUN_NAME}.STARTED")

## PID / tmux / scheduler ID

tmux session: \`${session}\`

## GPU assignment

Physical GPU: \`${gpu}\`

GPU selection audit:

* \`${gpu_json}\`
* \`${assignment_json}\`

## Log path

* train: \`${log_root}/train.log\`
* posthoc: \`${log_root}/posthoc.log\`

## Expected outputs

* \`${out_dir}/best.pt\`
* \`${run_root}/posthoc_eval_canonical/single_background_candidate_gate.json\`
* \`${run_root}/posthoc_eval_canonical/SINGLE_BACKGROUND_CANDIDATE_DECISION.md\`

## Current status

Started.

## Notes

Gated P5 residual-direction smoke. Uses train-only split
\`${TRAINONLY_SPLIT}\` and train-only pert means
\`${TRAINONLY_PERT_MEANS}\`. This branch cannot make a true-multi claim.
EOF

echo "Launched ${RUN_NAME} in tmux session ${session} on physical GPU ${gpu}"
echo "RUN_STATUS: ${run_root}/RUN_STATUS.md"
