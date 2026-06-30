#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_NAME=${LATENTFM_XVERSE_PRIOR_RUN_NAME:-xverse_prior_adapter_global_genemean_w005_add002_replay1_4k}
RUN_ROOT=${LATENTFM_XVERSE_PRIOR_RUN_ROOT:-${ROOT}/runs/latentfm_xverse_condition_prior_adapter_smoke_20260621}
OUT_ROOT=${LATENTFM_XVERSE_PRIOR_OUT_ROOT:-${COUPLED}/output/latentfm_runs/xverse_condition_prior_adapter_smoke_20260621}
LOG_ROOT=${LATENTFM_XVERSE_PRIOR_LOG_ROOT:-${ROOT}/logs/latentfm_xverse_condition_prior_adapter_smoke_20260621}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
ANCHOR_CKPT=${LATENTFM_XVERSE_PRIOR_ANCHOR_CKPT:-${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt}
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
BOOTSTRAP_RUNNER=${ROOT}/ops/run_latentfm_posthoc_bootstrap_from_manifest_20260621.py
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
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${BOOTSTRAP_RUNNER}" \
  "${TRAIN_LAUNCHER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${out_dir}" && "${FORCE_XVERSE_PRIOR_RERUN:-0}" != "1" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_XVERSE_PRIOR_RERUN=1 to relaunch" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] exact GPU status before xverse condition-prior adapter launch" | tee "${run_root}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${run_root}/logs/gpu_launch_audit.log"

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

assignment_json="${run_root}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
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
for idx in payload.get("candidate_order", []):
    idx = int(idx)
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
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${LOG_ROOT}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${gpu}
export RUN_TAG=${RUN_NAME}
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export ANCHOR_REPLAY_CHECKPOINT=${ANCHOR_CKPT}
export FINETUNE_TRAINABLE_SCOPE=condition_prior_adapter
export TOTAL_STEPS=4000
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export GAMMA=0.03
export GAMMA_WARMUP_START=500
export GAMMA_WARMUP_END=1500
export MMD_EVERY=4
export SELECTION_METRIC=pearson_pert_minus_mmd
export SELECTION_MMD_LAMBDA=0.5
export EVAL_MAX_CONDITIONS=256
export EVAL_MAX_CONDITIONS_PER_DATASET=12
export EVAL_MAX_MSE_CELLS=1024
export EVAL_MAX_MMD_CELLS=1024
export EVAL_MAX_CHUNK=256
export CONDITION_DELTA_HEAD_USE_IN_MODEL=1
export CONDITION_DELTA_HEAD_LOSS_WEIGHT=0.0
export ADDITIVE_CONDITION_DELTA_LOSS_WEIGHT=0.0
export CONDITION_PRIOR_DELTA_LOSS_WEIGHT=0.05
export CONDITION_PRIOR_DELTA_LOSS_WARMUP_START=500
export CONDITION_PRIOR_DELTA_LOSS_WARMUP_END=1500
export CONDITION_PRIOR_DELTA_LOSS_EVERY=1
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WEIGHT=0.02
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_START=500
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_END=1500
export CONDITION_PRIOR_BANK_SCOPE=global
export CONDITION_PRIOR_BANK_SPLIT_FILE=${CANONICAL_SPLIT}
export CONDITION_PRIOR_BANK_AGGREGATION=gene_mean
export CONDITION_PRIOR_BANK_MAX_CELLS=512
export CONDITION_PRIOR_NUM_GENES=2
export ANCHOR_REPLAY_LOSS_WEIGHT=1.0
export ANCHOR_REPLAY_LOSS_WARMUP_START=500
export ANCHOR_REPLAY_LOSS_WARMUP_END=1500
export ANCHOR_REPLAY_CONDITION_FILTER=non_gene_multi
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

manifest="${run_root}/posthoc_manifest.json"
"${PYTHON}" - "${manifest}" "${RUN_NAME}" "${ANCHOR_CKPT}" "${out_dir}/best.pt" "${CANONICAL_SPLIT}" "${DATA_DIR}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "stage": "xverse_condition_prior_adapter_smoke",
    "run_name": sys.argv[2],
    "anchor_checkpoint": sys.argv[3],
    "candidate_checkpoint": sys.argv[4],
    "split_file": sys.argv[5],
    "data_dir": sys.argv[6],
    "launched_runs": [
        {
            "run_name": sys.argv[2],
            "anchor_checkpoint": sys.argv[3],
            "candidate_checkpoint": sys.argv[4],
            "split_file": sys.argv[5],
            "data_dir": sys.argv[6],
            "condition_prior_bank_scope": "global",
            "condition_prior_bank_aggregation": "gene_mean",
            "condition_prior_delta_loss_weight": 0.05,
            "condition_prior_additive_delta_loss_weight": 0.02,
            "anchor_replay_loss_weight": 1.0,
            "finetune_trainable_scope": "condition_prior_adapter",
        }
    ],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

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
out_eval=${run_root}/posthoc_eval_stablecaps
mkdir -p "\${out_eval}"
base_split="\${out_eval}/split_group_eval_anchor_ode20_stablecaps.json"
base_family="\${out_eval}/condition_family_eval_anchor_ode20_stablecaps.json"
cand_split="\${out_eval}/split_group_eval_candidate_ode20_stablecaps.json"
cand_family="\${out_eval}/condition_family_eval_candidate_ode20_stablecaps.json"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 256 --eval-max-conditions-per-dataset 12 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${base_split}" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${base_family}" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${cand_split}" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${cand_family}" "\${common[@]}"
${PYTHON} - ${manifest} "\${base_split}" "\${base_family}" "\${cand_split}" "\${cand_family}" <<PY
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
row = payload["launched_runs"][0]
row["baseline_split_json"] = sys.argv[2]
row["baseline_family_json"] = sys.argv[3]
row["run_split_json"] = sys.argv[4]
row["run_family_json"] = sys.argv[5]
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
${PYTHON} ${BOOTSTRAP_RUNNER} --manifest ${manifest} --out-dir ${ROOT}/reports/latentfm_xverse_condition_prior_adapter_${RUN_NAME}_bootstrap_20260621 --n-boot 2000 --seed 42 --split-groups test test_single test_multi test_multi_unseen2 --family-groups family_gene family_drug structure_multi
EOF
chmod +x "${posthoc_script}"

rm -f "${run_root}/${RUN_NAME}.EXIT_CODE" "${run_root}/${RUN_NAME}.FINISHED" "${run_root}/POSTHOC_EXIT_CODE" "${run_root}/POSTHOC_FINISHED"
session="lfm_${RUN_NAME}"
posthoc_session="lfm_${RUN_NAME}_posthoc"
tmux new -d -s "${session}" \
  "bash -lc 'bash ${train_script} > ${log_root}/${RUN_NAME}.log 2>&1; rc=\$?; echo \$rc > ${run_root}/${RUN_NAME}.EXIT_CODE; date \"+%F %T %Z\" > ${run_root}/${RUN_NAME}.FINISHED; exit \$rc'"
date '+%F %T %Z' > "${run_root}/${RUN_NAME}.STARTED"
tmux new -d -s "${posthoc_session}" \
  "bash -lc 'bash ${posthoc_script} > ${run_root}/logs/posthoc.log 2>&1; rc=\$?; echo \$rc > ${run_root}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_root}/POSTHOC_FINISHED; exit \$rc'"

cat > "${run_root}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_condition_prior_adapter_smoke_20260621/${RUN_NAME}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_xverse_condition_prior_adapter_smoke_20260621.sh
\`\`\`

## Runtime classification

Long LatentFM training task. Use 30-minute cadence for checks.

## Start time

$(cat "${run_root}/${RUN_NAME}.STARTED")

## tmux / GPU

* training: \`${session}\`, physical GPU${gpu}
* posthoc watcher: \`${posthoc_session}\`, physical GPU${gpu} after training finishes

## Log path

\`${log_root}/${RUN_NAME}.log\`

## Expected outputs

* \`${out_dir}/best.pt\`
* \`${out_dir}/condition_prior_bank_summary.json\`
* \`${manifest}\`
* \`${ROOT}/reports/latentfm_xverse_condition_prior_adapter_${RUN_NAME}_bootstrap_20260621/bootstrap_index.json\`

## Current status

Started training and low-frequency posthoc watcher.

## Notes

This capped smoke tests a global/gene_mean train-single synthetic-composition
prior under an anchor-preserving adapter. The base xverse 8k anchor and original
condition path are frozen by \`finetune_trainable_scope=condition_prior_adapter\`.
Only \`condition_delta_head.*\` and \`condition_delta_to_c.*\` train.

The route is diagnostic unless capped paired bootstrap improves
\`test_multi_unseen2\` pp without deterministic MMD, aggregate, family, single,
or drug harm.

GPU assignment audit: \`${assignment_json}\`.
EOF

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_condition_prior_adapter_smoke_20260621

Launched at $(date '+%F %T %Z').

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_xverse_condition_prior_adapter_smoke_20260621.sh
\`\`\`

## Runtime classification

Long LatentFM training task. Use 30-minute cadence for checks.

## Run

* \`${RUN_NAME}\`: RUN_STATUS \`${run_root}/RUN_STATUS.md\`

## Current status

Started training and low-frequency posthoc watcher.

## GPU assignment audit

\`${assignment_json}\`
EOF

echo "Launched xverse condition-prior adapter smoke"
echo "RUN_STATUS: ${run_root}/RUN_STATUS.md"
