#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_XVERSE_SCALING_SEED_ACK:-}" != "canonical_primary_cap120_pass" ]]; then
  cat >&2 <<'EOF'
Refusing to launch xverse scaling seed robustness.

Set:
  LATENTFM_XVERSE_SCALING_SEED_ACK=canonical_primary_cap120_pass

Required preread:
  reports/LATENTFM_XVERSE_SCALING_CANONICAL_NOHARM_DECISION_20260624.md
EOF
  exit 4
fi

CANON_JSON=${ROOT}/reports/latentfm_xverse_scaling_canonical_noharm_decision_20260624.json
if [[ ! -e "${CANON_JSON}" ]]; then
  echo "Missing canonical no-harm decision JSON: ${CANON_JSON}" >&2
  exit 2
fi
"${PYTHON}" - "${CANON_JSON}" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = (obj.get("decision") or {}).get("status")
primary = (obj.get("decision") or {}).get("primary_run")
if status != "canonical_noharm_pass" or primary != "xverse_scaling_cap120_all_3k_seed42":
    raise SystemExit(
        f"canonical primary gate not passed: status={status!r} primary={primary!r}"
    )
PY

RUN_ROOT=${LATENTFM_XVERSE_SCALING_SEED_RUN_ROOT:-${ROOT}/runs/latentfm_xverse_scaling_seed_robustness_20260624}
OUT_ROOT=${LATENTFM_XVERSE_SCALING_SEED_OUT_ROOT:-${COUPLED}/output/latentfm_runs/xverse_scaling_seed_robustness_20260624}
LOG_ROOT=${LATENTFM_XVERSE_SCALING_SEED_LOG_ROOT:-${ROOT}/logs/latentfm_xverse_scaling_seed_robustness_20260624}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
SPLIT_FILE=${BIFLOW_DIR}/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json
PERT_MEANS=${ROOT}/runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/xverse_trainonly_scaling_cap120_all_v2_pert_means.npz
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

SEEDS_STR=${LATENTFM_XVERSE_SCALING_SEEDS:-"43 44"}
read -r -a SEEDS <<< "${SEEDS_STR}"
if [[ ${#SEEDS[@]} -lt 1 ]]; then
  echo "No seeds requested" >&2
  exit 4
fi
need=${#SEEDS[@]}

for required in \
  "${DATA_DIR}/manifest.json" \
  "${CANONICAL_SPLIT}" \
  "${SPLIT_FILE}" \
  "${PERT_MEANS}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}" \
  "${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py" \
  "${ROOT}/ops/summarize_latentfm_single_background_candidate_decision_20260622.py" \
  "${ROOT}/ops/summarize_latentfm_xverse_scaling_seed_robustness_20260624.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

declare -a RUN_NAMES=()
for seed in "${SEEDS[@]}"; do
  if ! [[ "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: ${seed}" >&2
    exit 4
  fi
  run_name="xverse_scaling_cap120_all_3k_seed${seed}"
  RUN_NAMES+=("${run_name}")
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  session=lfm_${run_name}
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_XVERSE_SCALING_SEED_RERUN:-0}" != "1" ]]; then
    echo "Output exists for ${run_name}; set FORCE_LATENTFM_XVERSE_SCALING_SEED_RERUN=1 to relaunch" >&2
    exit 3
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

echo "[$(date '+%F %T %Z')] exact GPU status before scaling seed robustness launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

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
audit = {
    "status": "pass",
    "need": need,
    "assigned_gpus": suggested[:need],
    "allowed_physical_user_gpus": payload.get("allowed_physical_user_gpus"),
    "active_user_gpus": payload.get("active_user_gpus"),
    "new_physical_slots": payload.get("new_physical_slots"),
    "max_user_gpus": payload.get("max_user_gpus"),
    "max_jobs_per_gpu": payload.get("max_jobs_per_gpu"),
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if len(suggested) < need:
    reasons.append(f"only {len(suggested)} GPU job slots suggested for need={need}")
if int(payload.get("max_user_gpus") or 0) > 4:
    reasons.append("max_user_gpus exceeds user cap 4")
if int(payload.get("max_jobs_per_gpu") or 0) > 4:
    reasons.append("max_jobs_per_gpu exceeds per-GPU cap 4")
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

mapfile -t ASSIGNED_GPUS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for gpu in payload["assigned_gpus"]:
    print(int(gpu))
PY
)

for i in "${!RUN_NAMES[@]}"; do
  run_name=${RUN_NAMES[$i]}
  seed=${SEEDS[$i]}
  gpu=${ASSIGNED_GPUS[$i]}
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
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
export TOTAL_STEPS=3000
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
export DS_ALPHA=0.7
export DS_LOSS_ALPHA=0.0
export MIN_SELECTED_CONDITIONS_PER_DATASET=0
export CONDITION_VISIT_POWER=1.0
export CONDITION_VISIT_CAP=0
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
eval_dir=${run_dir}/posthoc_eval_canonical
mkdir -p "\${eval_dir}"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/split_group_eval_anchor_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/condition_family_eval_anchor_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/split_group_eval_candidate_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" "\${common[@]}"
${PYTHON} ${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py --candidate-split-json "\${eval_dir}/split_group_eval_candidate_ode20_canonical.json" --candidate-family-json "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" --n-boot 2000 --seed ${seed} --out-json "\${eval_dir}/single_background_candidate_gate.json" --out-md "\${eval_dir}/SINGLE_BACKGROUND_CANDIDATE_GATE.md"
${PYTHON} ${ROOT}/ops/summarize_latentfm_single_background_candidate_decision_20260622.py --gate-json "\${eval_dir}/single_background_candidate_gate.json" --label ${run_name} --title "LatentFM xverse scaling seed robustness decision" --out-md "\${eval_dir}/SINGLE_BACKGROUND_CANDIDATE_DECISION.md"
EOF
  chmod +x "${posthoc_script}"

  rm -f "${run_dir}/${run_name}.EXIT_CODE" "${run_dir}/${run_name}.FINISHED" "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  date '+%F %T %Z' > "${run_dir}/${run_name}.STARTED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${train_script} > ${log_dir}/launcher.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/${run_name}.EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/${run_name}.FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${log_dir}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"

  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: ${run_name}

## Hypothesis

If cap120_all is a real Track A scaling improvement, it should retain the canonical no-harm gate under independent fine-tune seed ${seed}.

## Command

\`\`\`bash
LATENTFM_XVERSE_SCALING_SEED_ACK=canonical_primary_cap120_pass bash ${ROOT}/ops/launch_latentfm_xverse_scaling_seed_robustness_20260624.sh
\`\`\`

## Runtime classification

Long GPU training plus canonical posthoc task. Use 30-minute cadence for checks.

## Start time

$(cat "${run_dir}/${run_name}.STARTED")

## PID / tmux / scheduler ID

tmux session: \`${session}\`

Physical GPU: ${gpu}

## Log path

\`${log_dir}/launcher.log\`

Posthoc log:

\`${log_dir}/posthoc.log\`

## Expected outputs

* \`${out_dir}/best.pt\`
* \`${run_dir}/posthoc_eval_canonical/single_background_candidate_gate.json\`
* \`${ROOT}/reports/LATENTFM_XVERSE_SCALING_SEED_ROBUSTNESS_DECISION_20260624.md\`

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

- Uses frozen train-only cap120_all split: \`${SPLIT_FILE}\`
- Canonical split is post-training no-harm only; canonical multi is diagnostic only.
- Resource policy: max 4 physical GPUs, max 4 LatentFM jobs/GPU, 48 CPU cores.
EOF
  echo "Launched ${run_name} on GPU ${gpu}"
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_scaling_seed_robustness_20260624

## Command

\`\`\`bash
LATENTFM_XVERSE_SCALING_SEED_ACK=canonical_primary_cap120_pass bash ${ROOT}/ops/launch_latentfm_xverse_scaling_seed_robustness_20260624.sh
\`\`\`

## Runtime classification

Long GPU seed-robustness batch. Each child run has its own RUN_STATUS.md.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

$(printf '* `%s`\n' "${RUN_NAMES[@]/#/lfm_}")

## Log path

\`${LOG_ROOT}/<run_name>/launcher.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_XVERSE_SCALING_SEED_ROBUSTNESS_DECISION_20260624.md\`

## Current status

Started ${need} seed robustness runs.

## Notes

- Guarded by canonical no-harm pass for seed42 cap120_all.
- Run summarizer after child posthoc jobs finish:
  \`${PYTHON} ${ROOT}/ops/summarize_latentfm_xverse_scaling_seed_robustness_20260624.py\`
EOF
