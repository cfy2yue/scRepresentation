#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_BLOCK=${LATENTFM_TRACKA_GUARDED_RUN_BLOCK:-latentfm_tracka_scf_jiang_lowcount_adapter_20260623}
RUN_NAME=${LATENTFM_TRACKA_GUARDED_RUN_NAME:-${LATENTFM_TRACKA_JIANG_LOWCOUNT_RUN_NAME:-scfoundation_tracka_gene_shrink_k2_jiang_lowcount_adapter_2k_seed42}}
RUN_ROOT=${ROOT}/runs/${RUN_BLOCK}
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
OUT_ROOT=${COUPLED}/output/latentfm_runs/${RUN_BLOCK}
OUT_DIR=${OUT_ROOT}/${RUN_NAME}
LOG_ROOT=${ROOT}/logs/${RUN_BLOCK}
LOG_DIR=${LOG_ROOT}/${RUN_NAME}
REPORT_DIR=${ROOT}/reports
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
TRAINONLY_SPLIT=${BIFLOW_DIR}/split_seed42_xverse_trainonly_crossbg_val_v2.json
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
LATENT_BACKBONE_VALUE=${LATENTFM_TRACKA_GUARDED_LATENT_BACKBONE:-scfoundation}
DATA_DIR=${LATENTFM_TRACKA_GUARDED_DATA_DIR:-${ROOT}/dataset/latentfm_full/scfoundation}
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
GATE_SCRIPT=${ROOT}/ops/evaluate_latentfm_single_background_candidate_gate_20260623.py
DECISION_RENDERER=${ROOT}/ops/summarize_latentfm_single_background_candidate_decision_20260622.py
CPU_GATE_JSON=${LATENTFM_TRACKA_GUARDED_CPU_GATE_JSON:-${REPORT_DIR}/latentfm_tracka_scf_jiang_lowcount_mask_cpu_gate_20260623.json}
CODE_GATE_MD=${LATENTFM_TRACKA_GUARDED_CODE_GATE_MD:-${REPORT_DIR}/LATENTFM_TRACKA_SCF_JIANG_LOWCOUNT_CODE_GATE_20260623.md}
PROTOCOL_MD=${LATENTFM_TRACKA_GUARDED_PROTOCOL_MD:-${REPORT_DIR}/LATENTFM_TRACKA_SCF_GUARDED_FALLBACK_PROTOCOL_20260623.md}
ANCHOR_CKPT=${LATENTFM_TRACKA_GUARDED_ANCHOR_CKPT:-${COUPLED}/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt}
PERT_MEANS=${LATENTFM_TRACKA_GUARDED_PERT_MEANS:-${ROOT}/runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622/artifacts/scfoundation_trainonly_pert_means_split_seed42_crossbgval_v2.npz}
AGGREGATION=${LATENTFM_TRACKA_GUARDED_AGGREGATION:-gene_shrink_k2_jiang_lowcount_mask}
EXPECTED_POLICY=${LATENTFM_TRACKA_GUARDED_EXPECTED_POLICY:-fallback_to_dataset_mean_for_jiang_gene_train_count_le_1}
EXPECTED_CPU_GATE_STATUS=${LATENTFM_TRACKA_GUARDED_EXPECTED_CPU_GATE_STATUS:-tracka_scf_guarded_fallback_cpu_gate_pass_no_gpu_yet}
LABEL_SLUG=${LATENTFM_TRACKA_GUARDED_LABEL_SLUG:-scf_jiang_lowcount_adapter}
TITLE=${LATENTFM_TRACKA_GUARDED_TITLE:-LatentFM Track A scFoundation Jiang-Lowcount Adapter Decision}
BRANCH_LABEL=${LATENTFM_TRACKA_GUARDED_BRANCH_LABEL:-Jiang-lowcount}
HYPOTHESIS=${LATENTFM_TRACKA_GUARDED_HYPOTHESIS:-Narrow Jiang_IFNG/Jiang_TNFA low-count fallback reduces Track A no-harm failure while preserving the scFoundation cross-background near-miss signal.}
FAIL_CLOSE=${LATENTFM_TRACKA_GUARDED_FAIL_CLOSE:-If cross_background_seen_gene pp delta remains below +0.02, if all_test_single/family_gene pp or MMD harm persists, or if Jiang_IFNG/TNFA dataset-level pp delta is below -0.02, close this branch.}
CPU_THREADS=${LATENTFM_TRACKA_GUARDED_CPU_THREADS:-${LATENTFM_TRACKA_JIANG_LOWCOUNT_CPU_THREADS:-4}}
LAUNCH_COMMAND=${LATENTFM_TRACKA_GUARDED_LAUNCH_COMMAND:-bash ${ROOT}/ops/launch_latentfm_tracka_scf_jiang_lowcount_adapter_20260623.sh}

if (( CPU_THREADS < 1 || CPU_THREADS > 24 )); then
  echo "Refusing CPU_THREADS=${CPU_THREADS}; keep this single run below the 48-core LatentFM cap" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${LOG_DIR}" "${OUT_ROOT}" "${REPORT_DIR}"

for required in \
  "${PY}" \
  "${TRAINONLY_SPLIT}" \
  "${CANONICAL_SPLIT}" \
  "${DATA_DIR}/manifest.json" \
  "${DATA_DIR}/condition_metadata.json" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}" \
  "${GATE_SCRIPT}" \
  "${DECISION_RENDERER}" \
  "${CPU_GATE_JSON}" \
  "${CODE_GATE_MD}" \
  "${PROTOCOL_MD}" \
  "${ANCHOR_CKPT}" \
  "${PERT_MEANS}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 3
  fi
done

"${PY}" - "${CPU_GATE_JSON}" "${EXPECTED_POLICY}" "${EXPECTED_CPU_GATE_STATUS}" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = p.get("decision") or {}
expected_status = sys.argv[3]
if decision.get("status") != expected_status:
    raise SystemExit(f"CPU gate did not pass: {decision}")
expected_policy = sys.argv[2]
if decision.get("policy") != expected_policy:
    raise SystemExit(f"unexpected policy: {decision.get('policy')}")
if decision.get("gpu_authorization") != "none":
    raise SystemExit("CPU gate should not directly authorize GPU; launcher is separately protocol-gated")
PY

if [[ -e "${OUT_DIR}" && "${FORCE_LATENTFM_TRACKA_GUARDED_RERUN:-${FORCE_LATENTFM_TRACKA_JIANG_LOWCOUNT_RERUN:-0}}" != "1" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_LATENTFM_TRACKA_GUARDED_RERUN=1 to relaunch" >&2
  exit 4
fi
if tmux ls 2>/dev/null | grep -q "lfm_${RUN_NAME}"; then
  echo "Found existing lfm_${RUN_NAME} tmux session; refusing duplicate launch" >&2
  exit 5
fi

echo "[$(date '+%F %T %Z')] exact GPU status before Track A ${BRANCH_LABEL} launch" | tee "${RUN_DIR}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_DIR}/logs/gpu_launch_audit.log"
free -h | tee "${RUN_DIR}/logs/free_launch_audit.log"
df -h "${ROOT}" | tee "${RUN_DIR}/logs/df_launch_audit.log"

gpu_json="${RUN_DIR}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PY}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 5 \
  --max-jobs-per-gpu 4 \
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
allowed = int(payload.get("allowed_physical_user_gpus", 0))
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
gate_json=${REPORT_DIR}/latentfm_tracka_${LABEL_SLUG}_${RUN_NAME}_gate_20260623.json
decision_md=${REPORT_DIR}/LATENTFM_TRACKA_${LABEL_SLUG^^}_${RUN_NAME}_DECISION_20260623.md
manifest=${REPORT_DIR}/latentfm_tracka_${LABEL_SLUG}_manifest_20260623.json

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
export LATENT_BACKBONE=${LATENT_BACKBONE_VALUE}
export DATA_DIR=${DATA_DIR}
export BIFLOW_DIR=${BIFLOW_DIR}
export SPLIT_FILE=${TRAINONLY_SPLIT}
export PERT_MEANS_FILE=${PERT_MEANS}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${LOG_DIR}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PY}
export GPU=${GPU}
export RUN_TAG=${RUN_NAME}
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=1
export ANCHOR_REPLAY_CHECKPOINT=${ANCHOR_CKPT}
export ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
export FINETUNE_TRAINABLE_SCOPE=condition_prior_adapter
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
export CONDITION_DELTA_HEAD_USE_IN_MODEL=1
export CONDITION_DELTA_IN_MODEL_FILTER=gene_single
export CONDITION_DELTA_HEAD_LOSS_WEIGHT=0.0
export ADDITIVE_CONDITION_DELTA_LOSS_WEIGHT=0.0
export CONDITION_PRIOR_DELTA_LOSS_WEIGHT=0.05
export CONDITION_PRIOR_DELTA_LOSS_WARMUP_START=0
export CONDITION_PRIOR_DELTA_LOSS_WARMUP_END=500
export CONDITION_PRIOR_DELTA_LOSS_EVERY=1
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WEIGHT=0.02
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_START=0
export CONDITION_PRIOR_ADDITIVE_DELTA_LOSS_WARMUP_END=500
export CONDITION_PRIOR_BANK_SCOPE=global
export CONDITION_PRIOR_BANK_SPLIT_FILE=${TRAINONLY_SPLIT}
export CONDITION_PRIOR_BANK_AGGREGATION=${AGGREGATION}
export CONDITION_PRIOR_BANK_MAX_CELLS=256
export CONDITION_PRIOR_NUM_GENES=1
export ANCHOR_REPLAY_LOSS_WEIGHT=2.0
export ANCHOR_REPLAY_LOSS_WARMUP_START=0
export ANCHOR_REPLAY_LOSS_WARMUP_END=500
export ANCHOR_REPLAY_CONDITION_FILTER=all
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
while [[ ! -f ${RUN_DIR}/EXIT_CODE ]]; do
  sleep 1800
done
code="\$(cat ${RUN_DIR}/EXIT_CODE)"
if [[ "\${code}" != "0" ]]; then
  echo "training failed for ${RUN_NAME}; skip posthoc" >&2
  exit "\${code}"
fi
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
mkdir -p ${posthoc_dir}
anchor_split=${posthoc_dir}/split_group_eval_anchor_tracka_ode20.json
anchor_family=${posthoc_dir}/condition_family_eval_anchor_tracka_ode20.json
candidate_split=${posthoc_dir}/split_group_eval_candidate_tracka_ode20.json
candidate_family=${posthoc_dir}/condition_family_eval_candidate_tracka_ode20.json
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 2048 --eval-max-mmd-cells 2048)
${PY} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test_single --out "\${anchor_split}" "\${common[@]}"
${PY} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups family_gene --out "\${anchor_family}" "\${common[@]}"
${PY} -m model.latent.eval_split_groups --checkpoint ${OUT_DIR}/best.pt --groups test_single --out "\${candidate_split}" "\${common[@]}"
${PY} -m model.latent.eval_condition_families --checkpoint ${OUT_DIR}/best.pt --groups family_gene --out "\${candidate_family}" "\${common[@]}"
${PY} ${GATE_SCRIPT} \
  --anchor-split-json "\${anchor_split}" \
  --candidate-split-json "\${candidate_split}" \
  --anchor-family-json "\${anchor_family}" \
  --candidate-family-json "\${candidate_family}" \
  --split-file ${CANONICAL_SPLIT} \
  --data-dir ${DATA_DIR} \
  --n-boot 2000 \
  --seed 42 \
  --out-json ${gate_json}
${PY} ${DECISION_RENDERER} \
  --gate-json ${gate_json} \
  --label ${RUN_NAME} \
  --title "${TITLE}" \
  --out-md ${decision_md}
EOF
chmod +x "${posthoc_script}"

cat > "${manifest}" <<EOF
{
  "run_name": "${RUN_NAME}",
  "run_block": "${RUN_BLOCK}",
  "latent": "${LATENT_BACKBONE_VALUE}",
  "data_dir": "${DATA_DIR}",
  "aggregation": "${AGGREGATION}",
  "train_split": "${TRAINONLY_SPLIT}",
  "canonical_split": "${CANONICAL_SPLIT}",
  "cpu_gate_json": "${CPU_GATE_JSON}",
  "expected_cpu_gate_status": "${EXPECTED_CPU_GATE_STATUS}",
  "code_gate_md": "${CODE_GATE_MD}",
  "protocol_md": "${PROTOCOL_MD}",
  "assigned_gpu": ${GPU},
  "cpu_threads": ${CPU_THREADS},
  "gate_json": "${gate_json}",
  "decision_md": "${decision_md}",
  "hypothesis": "${HYPOTHESIS}",
  "fail_close": "${FAIL_CLOSE}"
}
EOF

rm -f "${RUN_DIR}/EXIT_CODE" "${RUN_DIR}/FINISHED" "${RUN_DIR}/POSTHOC_EXIT_CODE" "${RUN_DIR}/POSTHOC_FINISHED"
date '+%F %T %Z' > "${RUN_DIR}/STARTED"

train_session=lfm_${RUN_NAME}
posthoc_session=lfm_${RUN_NAME}_posthoc
train_log=${LOG_DIR}/train.log
posthoc_log=${RUN_DIR}/logs/posthoc.log

cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_BLOCK}/${RUN_NAME}

## Command

\`\`\`bash
${LAUNCH_COMMAND}
\`\`\`

## Runtime classification

Long LatentFM training task. Use 30-minute cadence for GPU/posthoc checks.

## Start time

$(cat "${RUN_DIR}/STARTED")

## PID / tmux / scheduler ID

* training tmux: \`${train_session}\`
* posthoc watcher tmux: \`${posthoc_session}\`
* physical GPU: \`${GPU}\`

## Log path

\`${train_log}\`

\`${posthoc_log}\`

## Expected outputs

* \`${OUT_DIR}/best.pt\`
* \`${OUT_DIR}/condition_prior_bank_summary.json\`
* \`${posthoc_dir}/split_group_eval_candidate_tracka_ode20.json\`
* \`${gate_json}\`
* \`${decision_md}\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${train_log}
cat ${RUN_DIR}/EXIT_CODE 2>/dev/null || echo "still running"
cat ${RUN_DIR}/POSTHOC_EXIT_CODE 2>/dev/null || echo "posthoc pending/running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Hypothesis: the train-only ${BRANCH_LABEL} fallback
\`${AGGREGATION}\` can reduce the scFoundation Track A no-harm failure while
preserving the near-miss cross-background gain. The base flow is warm-started
from the EMA anchor; only \`condition_delta_head.*\` and
\`condition_delta_to_c.*\` train. Training/selection use
\`${TRAINONLY_SPLIT}\`; canonical \`split_seed42.json\` is used only for
posthoc Track A single/background/family gates after checkpoint freeze.
Canonical multi selection weight is 0 and held-out Track C query is forbidden.

Promotion gate: posthoc paired bootstrap must improve
\`cross_background_seen_gene\` pearson_pert by at least +0.02 with
\`p_improve >= 0.90\`, while preserving \`all_test_single\` and
\`family_gene\` pp/MMD no-harm. Branch-specific fail-close rule:
${FAIL_CLOSE}

GPU assignment audit: \`${assignment_json}\`.
EOF

tmux new -d -s "${train_session}" "bash -lc 'bash ${train_script} > ${train_log} 2>&1; rc=\$?; echo \$rc > ${RUN_DIR}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_DIR}/FINISHED; exit \$rc'"
tmux new -d -s "${posthoc_session}" "bash -lc 'bash ${posthoc_script} > ${posthoc_log} 2>&1; rc=\$?; echo \$rc > ${RUN_DIR}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${RUN_DIR}/POSTHOC_FINISHED; exit \$rc'"

echo "Launched ${RUN_NAME}"
echo "RUN_STATUS=${RUN_DIR}/RUN_STATUS.md"
echo "Manifest=${manifest}"
echo "Training session=${train_session}"
echo "Posthoc session=${posthoc_session}"
echo "Assigned GPU=${GPU}"
tmux ls | grep -E "${train_session}|${posthoc_session}" || true
tail -n 20 "${train_log}" || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
