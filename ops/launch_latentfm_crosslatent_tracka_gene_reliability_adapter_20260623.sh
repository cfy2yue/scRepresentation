#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PY=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_BLOCK=latentfm_crosslatent_tracka_gene_reliability_adapter_20260623
RUN_ROOT=${ROOT}/runs/${RUN_BLOCK}
OUT_ROOT=${COUPLED}/output/latentfm_runs/crosslatent_tracka_gene_reliability_adapter_20260623
LOG_ROOT=${ROOT}/logs/${RUN_BLOCK}
REPORT_DIR=${ROOT}/reports
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
TRAINONLY_SPLIT=${BIFLOW_DIR}/split_seed42_xverse_trainonly_crossbg_val_v2.json
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
GATE_SCRIPT=${ROOT}/ops/evaluate_latentfm_single_background_candidate_gate_20260623.py
DECISION_RENDERER=${ROOT}/ops/summarize_latentfm_single_background_candidate_decision_20260622.py
CPU_THREADS=${LATENTFM_CPU_THREADS:-4}

if (( CPU_THREADS < 1 || CPU_THREADS > 24 )); then
  echo "Refusing CPU_THREADS=${CPU_THREADS}; use 1..24 per job so block stays below 48 cores" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/scripts" "${OUT_ROOT}" "${LOG_ROOT}" "${REPORT_DIR}"

latents=(scfoundation scldm)
run_names=(
  scfoundation_tracka_gene_shrink_k2_adapter_2k_seed42
  scldm_tracka_gene_shrink_k4_adapter_2k_seed42
)
shrinks=(gene_shrink_k2 gene_shrink_k4)
gate_jsons=(
  ${REPORT_DIR}/latentfm_crosslatent_scfoundation_gene_reliability_router_gate_20260622.json
  ${REPORT_DIR}/latentfm_crosslatent_scldm_gene_reliability_router_gate_20260622.json
)
selected_models=(shrink_k2 shrink_k4)
data_dirs=(
  ${ROOT}/dataset/latentfm_full/scfoundation
  ${ROOT}/dataset/latentfm_full/scldm
)
checkpoints=(
  ${COUPLED}/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt
  ${COUPLED}/output/latentfm_runs/full_scldm/20260617_scldm_comp006_delta_w5_12k/best.pt
)
pert_means=(
  ${ROOT}/runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622/artifacts/scfoundation_trainonly_pert_means_split_seed42_crossbgval_v2.npz
  ${ROOT}/runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622/artifacts/scldm_trainonly_pert_means_split_seed42_crossbgval_v2.npz
)

for required in "${PY}" "${TRAINONLY_SPLIT}" "${CANONICAL_SPLIT}" "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" "${TRAIN_LAUNCHER}" "${GATE_SCRIPT}" "${DECISION_RENDERER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 3
  fi
done

for i in "${!latents[@]}"; do
  latent=${latents[$i]}
  run=${run_names[$i]}
  out_dir=${OUT_ROOT}/${run}
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_TRACKA_GENE_REL_ADAPTER_RERUN:-0}" != "1" ]]; then
    echo "Output exists for ${run}; set FORCE_LATENTFM_TRACKA_GENE_REL_ADAPTER_RERUN=1 to relaunch" >&2
    exit 4
  fi
  for required in "${data_dirs[$i]}/manifest.json" "${data_dirs[$i]}/condition_metadata.json" \
    "${checkpoints[$i]}" "${pert_means[$i]}" "${gate_jsons[$i]}"; do
    if [[ ! -e "${required}" ]]; then
      echo "Missing ${latent} prerequisite: ${required}" >&2
      exit 5
    fi
  done
  status="$("${PY}" - "${gate_jsons[$i]}" "${selected_models[$i]}" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ok = (
    (p.get("decision") or {}).get("status") == "cpu_gate_pass_design_one_gene_reliability_adapter"
    and p.get("selected_model") == sys.argv[2]
)
print("ok" if ok else json.dumps({"decision": p.get("decision"), "selected_model": p.get("selected_model")}, sort_keys=True))
PY
)"
  if [[ "${status}" != "ok" ]]; then
    echo "CPU gate does not authorize ${latent}: ${status}" >&2
    exit 6
  fi
done

if tmux ls 2>/dev/null | grep -q "lfm_.*tracka_gene_shrink"; then
  echo "Found existing lfm_*tracka_gene_shrink tmux session; refusing duplicate launch" >&2
  exit 7
fi

echo "[$(date '+%F %T %Z')] exact GPU status before Track A gene-reliability adapter launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee "${RUN_ROOT}/logs/free_launch_audit.log"
df -h "${ROOT}" | tee "${RUN_ROOT}/logs/df_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PY}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 4 \
  --need 2 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PY}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
chosen = []
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
allowed = int(payload.get("allowed_physical_user_gpus", 0))
for idx_raw in payload.get("candidate_order", []):
    idx = int(idx_raw)
    gpu = gpus[idx]
    if not gpu.get("available"):
        continue
    if int(gpu.get("colocation_slots_free", 0)) <= 0:
        continue
    proposed = set(chosen) | active_user | {idx}
    if len(proposed) > allowed:
        continue
    chosen.append(idx)
    if len(chosen) >= 2:
        break
system = payload.get("system") or {}
reasons = []
if len(chosen) < 2:
    reasons.append(f"need 2 GPU slots, got {len(chosen)}")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 1.5:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 1.500")
audit = {
    "status": "fail" if reasons else "pass",
    "assigned_gpus": chosen,
    "active_user_gpus": sorted(active_user),
    "allowed_physical_user_gpus": allowed,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
    "notes": "Foreign stably-light GPUs are allowed by select_available_gpus.py; own occupied GPUs keep their slots.",
}
if reasons:
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if not reasons else 8)
PY

mapfile -t assigned_gpus < <("${PY}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in p["assigned_gpus"]:
    print(item)
PY
)

manifest=${REPORT_DIR}/latentfm_crosslatent_tracka_gene_reliability_adapter_manifest_20260623.jsonl
: > "${manifest}"

for i in "${!latents[@]}"; do
  latent=${latents[$i]}
  run=${run_names[$i]}
  shrink=${shrinks[$i]}
  gpu=${assigned_gpus[$i]}
  data_dir=${data_dirs[$i]}
  ckpt=${checkpoints[$i]}
  pert_mean=${pert_means[$i]}
  run_dir=${RUN_ROOT}/${run}
  out_dir=${OUT_ROOT}/${run}
  log_dir=${LOG_ROOT}/${run}
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${log_dir}"

  train_script=${run_dir}/scripts/train_${run}.sh
  posthoc_script=${run_dir}/scripts/posthoc_${run}.sh
  posthoc_dir=${run_dir}/posthoc_canonical_tracka
  gate_json=${REPORT_DIR}/latentfm_crosslatent_tracka_gene_reliability_adapter_${run}_gate_20260623.json
  decision_md=${REPORT_DIR}/LATENTFM_CROSSLATENT_TRACKA_GENE_RELIABILITY_ADAPTER_${run}_DECISION_20260623.md

  cat > "${train_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=${CPU_THREADS}
export MKL_NUM_THREADS=${CPU_THREADS}
export OPENBLAS_NUM_THREADS=${CPU_THREADS}
export NUMEXPR_NUM_THREADS=${CPU_THREADS}
export BLIS_NUM_THREADS=${CPU_THREADS}
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
export LATENT_BACKBONE=${latent}
export DATA_DIR=${data_dir}
export BIFLOW_DIR=${BIFLOW_DIR}
export SPLIT_FILE=${TRAINONLY_SPLIT}
export PERT_MEANS_FILE=${pert_mean}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${log_dir}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PY}
export GPU=${gpu}
export RUN_TAG=${run}
export INIT_CHECKPOINT=${ckpt}
export INIT_CHECKPOINT_USE_EMA=1
export ANCHOR_REPLAY_CHECKPOINT=${ckpt}
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
export CONDITION_PRIOR_BANK_AGGREGATION=${shrink}
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
while [[ ! -f ${run_dir}/EXIT_CODE ]]; do
  sleep 1800
done
code="\$(cat ${run_dir}/EXIT_CODE)"
if [[ "\${code}" != "0" ]]; then
  echo "training failed for ${run}; skip posthoc" >&2
  exit "\${code}"
fi
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${gpu}
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
common=(--data-dir ${data_dir} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 2048 --eval-max-mmd-cells 2048)
${PY} -m model.latent.eval_split_groups --checkpoint ${ckpt} --groups test_single --out "\${anchor_split}" "\${common[@]}"
${PY} -m model.latent.eval_condition_families --checkpoint ${ckpt} --groups family_gene --out "\${anchor_family}" "\${common[@]}"
${PY} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test_single --out "\${candidate_split}" "\${common[@]}"
${PY} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups family_gene --out "\${candidate_family}" "\${common[@]}"
${PY} ${GATE_SCRIPT} \
  --anchor-split-json "\${anchor_split}" \
  --candidate-split-json "\${candidate_split}" \
  --anchor-family-json "\${anchor_family}" \
  --candidate-family-json "\${candidate_family}" \
  --split-file ${CANONICAL_SPLIT} \
  --data-dir ${data_dir} \
  --n-boot 2000 \
  --seed 42 \
  --out-json ${gate_json}
${PY} ${DECISION_RENDERER} \
  --gate-json ${gate_json} \
  --label ${run} \
  --title "LatentFM Cross-Latent Track A Gene-Reliability Adapter Decision" \
  --out-md ${decision_md}
EOF
  chmod +x "${posthoc_script}"

  rm -f "${run_dir}/EXIT_CODE" "${run_dir}/FINISHED" "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  session=lfm_${run}
  posthoc_session=lfm_${run}_posthoc
  tmux new -d -s "${session}" "bash -lc 'bash ${train_script} > ${log_dir}/train.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/FINISHED; exit \$rc'"
  date '+%F %T %Z' > "${run_dir}/STARTED"
  tmux new -d -s "${posthoc_session}" "bash -lc 'bash ${posthoc_script} > ${run_dir}/logs/posthoc.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$rc'"

  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_BLOCK}/${run}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_crosslatent_tracka_gene_reliability_adapter_20260623.sh
\`\`\`

## Runtime classification

Long LatentFM training task. Use 30-minute cadence for GPU/posthoc checks.

## Start time

$(cat "${run_dir}/STARTED")

## PID / tmux / scheduler ID

* training tmux: \`${session}\`
* posthoc watcher tmux: \`${posthoc_session}\`
* physical GPU: \`${gpu}\`

## Log path

\`${log_dir}/train.log\`

## Expected outputs

* \`${out_dir}/best.pt\`
* \`${out_dir}/condition_prior_bank_summary.json\`
* \`${posthoc_dir}/split_group_eval_candidate_tracka_ode20.json\`
* \`${gate_json}\`
* \`${decision_md}\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${log_dir}/train.log
cat ${run_dir}/EXIT_CODE 2>/dev/null || echo "still running"
cat ${run_dir}/POSTHOC_EXIT_CODE 2>/dev/null || echo "posthoc pending/running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Hypothesis: the train-only \`${selected_models[$i]}\` gene-reliability CPU gate
can be converted into a tiny EMA-consistent Track A single-gene adapter. The
base flow is warm-started from the EMA anchor; only \`condition_delta_head.*\`
and \`condition_delta_to_c.*\` train. The condition-prior bank is built from
\`${TRAINONLY_SPLIT}\` with aggregation \`${shrink}\`; canonical multi is not
used for training or selection.

Promotion gate: posthoc paired bootstrap must improve
\`cross_background_seen_gene\` pearson_pert by at least +0.02 with
\`p_improve >= 0.90\`, while preserving \`all_test_single\` and
\`family_gene\` pp/MMD no-harm. Failure closes this adapter branch for this
latent. Canonical multi has selection weight 0 and is not evaluated here.

GPU assignment audit: \`${assignment_json}\`.
EOF

  "${PY}" - "${manifest}" "${run}" "${latent}" "${shrink}" "${gpu}" "${run_dir}/RUN_STATUS.md" "${gate_json}" "${decision_md}" <<'PY'
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
row = {
    "launched_at": __import__("datetime").datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    "run_name": sys.argv[2],
    "latent": sys.argv[3],
    "aggregation": sys.argv[4],
    "physical_gpu": int(sys.argv[5]),
    "run_status": sys.argv[6],
    "gate_json": sys.argv[7],
    "decision_md": sys.argv[8],
    "hypothesis": "train-only gene-reliability shrink prior can improve Track A cross-background single-gene behavior without all-single/family-gene harm",
}
with manifest.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_BLOCK}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_crosslatent_tracka_gene_reliability_adapter_20260623.sh
\`\`\`

## Runtime classification

Long LatentFM training block. Use 30-minute cadence for GPU/posthoc checks.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

* \`lfm_${run_names[0]}\`
* \`lfm_${run_names[0]}_posthoc\`
* \`lfm_${run_names[1]}\`
* \`lfm_${run_names[1]}_posthoc\`

## Log path

\`${LOG_ROOT}/<run>/train.log\`

## Expected outputs

* \`${manifest}\`
* per-run \`RUN_STATUS.md\`
* per-run Track A canonical candidate gate JSON and decision MD

## How to check manually

\`\`\`bash
tmux ls
cat ${RUN_ROOT}/*/EXIT_CODE 2>/dev/null || echo "some training still running"
cat ${RUN_ROOT}/*/POSTHOC_EXIT_CODE 2>/dev/null || echo "some posthoc pending/running"
nvidia-smi
\`\`\`

## Current status

Started two detached runs plus low-frequency posthoc watchers.

## Notes

Resource plan: two physical GPUs, ${CPU_THREADS} CPU threads per job, total
well below the user-approved 48-core LatentFM cap. This block uses only the two
latents whose train-only gene-reliability CPU gates passed:
\`scfoundation/shrink_k2\` and \`scldm/shrink_k4\`. Stack and xverse gates are
closed and are not launched.

GPU assignment audit: \`${assignment_json}\`.
EOF

echo "Launched ${RUN_BLOCK}"
echo "RUN_STATUS: ${RUN_ROOT}/RUN_STATUS.md"
echo "Manifest: ${manifest}"
