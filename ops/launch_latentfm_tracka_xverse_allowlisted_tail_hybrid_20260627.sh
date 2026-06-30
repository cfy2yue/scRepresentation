#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_BLOCK=latentfm_tracka_xverse_allowlisted_tail_hybrid_20260627
RUN_NAME=${LATENTFM_XVERSE_ALLOWTAIL_RUN_NAME:-xverse_allowtail_hybrid_pertresid_prior_w003_p002_replay1_2k_seed42}
RUN_SEED=${LATENTFM_XVERSE_ALLOWTAIL_SEED:-42}
RUN_ROOT=${ROOT}/runs/${RUN_BLOCK}
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
OUT_ROOT=${COUPLED}/output/latentfm_runs/${RUN_BLOCK}
OUT_DIR=${OUT_ROOT}/${RUN_NAME}
LOG_ROOT=${ROOT}/logs/${RUN_BLOCK}
LOG_DIR=${LOG_ROOT}/${RUN_NAME}
REPORT_DIR=${ROOT}/reports

DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
TRAINONLY_SPLIT=${BIFLOW_DIR}/split_seed42_xverse_trainonly_crossbg_val_v2.json
TRAIN_PERT_MEANS=${ROOT}/runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz
ANCHOR_CKPT=${LATENTFM_XVERSE_ALLOWTAIL_ANCHOR_CKPT:-${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt}
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
ALLOWLIST=${ROOT}/reports/tracka_hybrid_touchset_preflight_20260627/tail_gene_allowlist.txt
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
TAIL_GATE=${ROOT}/ops/evaluate_latentfm_tracka_exact_tail_candidate_gate_20260627.py

CPU_THREADS=${LATENTFM_XVERSE_ALLOWTAIL_CPU_THREADS:-4}
if (( CPU_THREADS < 1 || CPU_THREADS > 24 )); then
  echo "Refusing CPU_THREADS=${CPU_THREADS}; keep this smoke within the active LatentFM CPU cap" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${LOG_DIR}" "${OUT_ROOT}" "${REPORT_DIR}"

for required in \
  "${PY}" \
  "${DATA_DIR}/manifest.json" \
  "${TRAINONLY_SPLIT}" \
  "${TRAIN_PERT_MEANS}" \
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${ALLOWLIST}" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}" \
  "${TAIL_GATE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 3
  fi
done

if [[ -e "${OUT_DIR}" && "${FORCE_LATENTFM_XVERSE_ALLOWTAIL_RERUN:-0}" != "1" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_LATENTFM_XVERSE_ALLOWTAIL_RERUN=1 to relaunch" >&2
  exit 4
fi
session=${LATENTFM_XVERSE_ALLOWTAIL_SESSION:-lfm_xverse_allowtail_seed${RUN_SEED}_20260627}
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 5
fi

echo "[$(date '+%F %T %Z')] GPU/CPU/RAM audit before xverse allowlisted-tail hybrid launch" | tee "${RUN_DIR}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_DIR}/logs/gpu_launch_audit.log"
free -h | tee "${RUN_DIR}/logs/free_launch_audit.log"
df -h "${ROOT}" | tee "${RUN_DIR}/logs/df_launch_audit.log"

gpu_json="${RUN_DIR}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PY}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct "${LATENTFM_XVERSE_ALLOWTAIL_UTIL_THRESHOLD:-30}" \
  --memory-threshold-mib "${LATENTFM_XVERSE_ALLOWTAIL_MEMORY_THRESHOLD_MIB:-10000}" \
  --max-user-gpus 2 \
  --max-jobs-per-gpu 2 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_DIR}/logs/gpu_selection.stderr"

assignment_json="${RUN_DIR}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PY}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
allowed = min(2, int(payload.get("allowed_physical_user_gpus", 0)))
chosen = None
for idx_raw in payload.get("candidate_order", []):
    idx = int(idx_raw)
    gpu = gpus[idx]
    if not gpu.get("available"):
        continue
    if int(gpu.get("colocation_slots_free", 0)) <= 0:
        continue
    if len(active_user | {idx}) <= allowed:
        chosen = idx
        break
system = payload.get("system") or {}
reasons = []
if chosen is None:
    reasons.append("need 1 GPU slot, got 0")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 1.5:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 1.500")
audit = {
    "status": "fail" if reasons else "pass",
    "assigned_gpus": ([] if chosen is None else [chosen]),
    "active_user_gpus": sorted(active_user),
    "allowed_physical_user_gpus": allowed,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
    "resource_note": "relaxed submittable threshold util<30%, memory<10GiB per user instruction; hard cap 2 GPUs/2 jobs.",
}
if reasons:
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if not reasons else 6)
PY

GPU="$("${PY}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(p["assigned_gpus"][0])
PY
)"

train_script=${RUN_DIR}/scripts/train_${RUN_NAME}.sh
posthoc_script=${RUN_DIR}/scripts/posthoc_${RUN_NAME}.sh
posthoc_dir=${RUN_DIR}/posthoc_canonical_tracka
gate_prefix=${REPORT_DIR}/tracka_exact_tail_candidate_gate_20260627/${RUN_NAME}

cat > "${train_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=${CPU_THREADS}
export MKL_NUM_THREADS=${CPU_THREADS}
export OPENBLAS_NUM_THREADS=${CPU_THREADS}
export NUMEXPR_NUM_THREADS=${CPU_THREADS}
export BLIS_NUM_THREADS=${CPU_THREADS}
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
export LATENT_BACKBONE=xverse
export DATA_DIR=${DATA_DIR}
export BIFLOW_DIR=${BIFLOW_DIR}
export SPLIT_FILE=${TRAINONLY_SPLIT}
export PERT_MEANS_FILE=${TRAIN_PERT_MEANS}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${LOG_DIR}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PY}
export GPU=${GPU}
export RUN_TAG=${RUN_NAME}
export SEED=${RUN_SEED}
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=1
export ANCHOR_REPLAY_CHECKPOINT=${ANCHOR_CKPT}
export ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
export FINETUNE_TRAINABLE_SCOPE=condition_prior_adapter
export CONDITION_DELTA_HEAD_USE_IN_MODEL=1
export CONDITION_DELTA_IN_MODEL_FILTER=allowlisted_gene_single
export CONDITION_DELTA_ALLOWLIST_GENE_FILE=${ALLOWLIST}
export CONDITION_DELTA_HEAD_LOSS_WEIGHT=0.03
export CONDITION_DELTA_HEAD_LOSS_WARMUP_START=100
export CONDITION_DELTA_HEAD_LOSS_WARMUP_END=800
export CONDITION_DELTA_HEAD_TARGET=pert_residual
export CONDITION_PRIOR_DELTA_LOSS_WEIGHT=0.02
export CONDITION_PRIOR_DELTA_LOSS_WARMUP_START=100
export CONDITION_PRIOR_DELTA_LOSS_WARMUP_END=800
export CONDITION_PRIOR_DELTA_LOSS_EVERY=1
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WEIGHT=0.01
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_START=100
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_END=800
export CONDITION_PRIOR_BANK_SCOPE=global
export CONDITION_PRIOR_BANK_SPLIT_FILE=${TRAINONLY_SPLIT}
export CONDITION_PRIOR_BANK_AGGREGATION=gene_mean
export CONDITION_PRIOR_BANK_MAX_CELLS=256
export CONDITION_PRIOR_NUM_GENES=1
export ANCHOR_REPLAY_LOSS_WEIGHT=1.0
export ANCHOR_REPLAY_LOSS_WARMUP_START=100
export ANCHOR_REPLAY_LOSS_WARMUP_END=800
export ANCHOR_REPLAY_CONDITION_FILTER=all
export TOTAL_STEPS=2000
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
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=${CPU_THREADS}
export MKL_NUM_THREADS=${CPU_THREADS}
export OPENBLAS_NUM_THREADS=${CPU_THREADS}
export NUMEXPR_NUM_THREADS=${CPU_THREADS}
export BLIS_NUM_THREADS=${CPU_THREADS}
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
mkdir -p ${posthoc_dir} ${REPORT_DIR}/tracka_exact_tail_candidate_gate_20260627
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PY} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug structure_single test_single --out ${posthoc_dir}/condition_family_eval_anchor_ode20_canonical.json "\${common[@]}"
${PY} -m model.latent.eval_condition_families --checkpoint ${OUT_DIR}/best.pt --groups test_all family_gene family_drug structure_single test_single --out ${posthoc_dir}/condition_family_eval_candidate_ode20_canonical.json "\${common[@]}"
${PY} ${TAIL_GATE} --anchor-json ${posthoc_dir}/condition_family_eval_anchor_ode20_canonical.json --candidate-json ${posthoc_dir}/condition_family_eval_candidate_ode20_canonical.json --out-prefix ${gate_prefix} --title "xverse allowlisted-tail hybrid exact-tail gate" --n-boot 5000 --seed 42
EOF
chmod +x "${posthoc_script}"

date '+%F %T %Z' > "${RUN_DIR}/STARTED"
echo "${session}" > "${RUN_DIR}/SESSION_NAME"
cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_tracka_xverse_allowlisted_tail_hybrid_20260627.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: ${session}

## Log path

\`${LOG_DIR}/train.log\`

## Expected outputs

* \`${OUT_DIR}/best.pt\`
* \`${posthoc_dir}/condition_family_eval_candidate_ode20_canonical.json\`
* \`${gate_prefix}.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_DIR}/train.log
cat ${RUN_DIR}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Hypothesis: a default-off allowlisted single-gene condition-delta/prior hybrid can
touch all seed-recurrent hard-tail Track A genes while avoiding the broad
single-gene MMD/no-harm failure seen in older xverse condition-delta/prior
adapters.

Stop rule: if exact simple-single, exact cross-background, canonical test_single,
or family_gene pp/MMD no-harm fails, or if recurrent cross hard-tail pp delta is
below +0.01 with p_improve <0.75, close the branch. Canonical multi and Track C
query are not used for selection.
EOF

tmux new -d -s "${session}" \
  "bash -lc 'bash ${train_script} > ${LOG_DIR}/train.log 2>&1; rc=\$?; echo \$rc > ${RUN_DIR}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_DIR}/FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${LOG_DIR}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${RUN_DIR}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${RUN_DIR}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"

tmux ls | tee "${RUN_DIR}/logs/tmux_after_launch.log"
tail -n 30 "${LOG_DIR}/train.log" || true
echo "Launched ${RUN_NAME} on GPU ${GPU}; RUN_STATUS=${RUN_DIR}/RUN_STATUS.md"
