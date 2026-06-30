#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_XVERSE_SCALING_ACK:-}" != "nested_v2_split_gate_pass" ]]; then
  cat >&2 <<'EOF'
Refusing to launch xverse scaling count smokes.

Set:
  LATENTFM_XVERSE_SCALING_ACK=nested_v2_split_gate_pass

Required preread:
  reports/LATENTFM_XVERSE_SCALING_SPLITS_V2_20260624.md
EOF
  exit 4
fi

RUN_ROOT=${LATENTFM_XVERSE_SCALING_RUN_ROOT:-${ROOT}/runs/latentfm_xverse_scaling_count_smokes_20260624}
OUT_ROOT=${LATENTFM_XVERSE_SCALING_OUT_ROOT:-${COUPLED}/output/latentfm_runs/xverse_scaling_count_smokes_20260624}
LOG_ROOT=${LATENTFM_XVERSE_SCALING_LOG_ROOT:-${ROOT}/logs/latentfm_xverse_scaling_count_smokes_20260624}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
TRAINONLY_SPLIT=${BIFLOW_DIR}/split_seed42_xverse_trainonly_crossbg_val_v2.json
TRAINONLY_PERT_MEANS=${ROOT}/runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz
SPLIT_DIR=${BIFLOW_DIR}/xverse_scaling_splits_v2_20260624
ARTIFACT_DIR=${ROOT}/runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${DATA_DIR}/manifest.json" \
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}" \
  "${ROOT}/reports/LATENTFM_XVERSE_SCALING_SPLITS_V2_20260624.md" \
  "${ROOT}/ops/summarize_latentfm_xverse_scaling_count_smokes_20260624.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

declare -a RUN_NAMES=(
  "xverse_scaling_cap30_all_3k_seed42"
  "xverse_scaling_cap120_all_3k_seed42"
)
declare -a ARMS=("cap30_all" "cap120_all")
declare -a HYPOTHESES=(
  "Fixed-compute count-scaling low condition cap tests whether small nested train coverage is sufficient."
  "Fixed-compute count-scaling higher condition cap tests whether adding nested train conditions improves train-only cross-background validation without family harm."
)
declare -a SPLITS=(
  "${SPLIT_DIR}/split_seed42_xverse_trainonly_scaling_cap30_all_v2.json"
  "${SPLIT_DIR}/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
)
declare -a PERT_MEANS=(
  "${ARTIFACT_DIR}/xverse_trainonly_scaling_cap30_all_v2_pert_means.npz"
  "${ARTIFACT_DIR}/xverse_trainonly_scaling_cap120_all_v2_pert_means.npz"
)
declare -a TRAIN_CONDS=("586" "1582")

MODE_LABEL="count scaling"
if [[ "${LATENTFM_XVERSE_SCALING_ONLY_ARM:-}" == "gene_cap120_allbg" ]]; then
  RUN_NAMES=("xverse_scaling_gene_cap120_allbg_3k_seed42")
  ARMS=("gene_cap120_allbg")
  HYPOTHESES=("Exploratory nested-v2 gene-only arm tests whether removing chemical perturbations changes train-only internal validation behavior; do not make a formal perturbation-type claim before count gate resolves.")
  SPLITS=("${SPLIT_DIR}/split_seed42_xverse_trainonly_scaling_gene_cap120_allbg_v2.json")
  PERT_MEANS=("${ARTIFACT_DIR}/xverse_trainonly_scaling_gene_cap120_allbg_v2_pert_means.npz")
  TRAIN_CONDS=("1222")
  MODE_LABEL="gene-type exploratory scaling"
elif [[ "${LATENTFM_XVERSE_SCALING_ONLY_ARM:-}" == "type_balanced_cap120" ]]; then
  RUN_NAMES=("xverse_scaling_type_balanced_cap120_3k_seed42")
  ARMS=("type_balanced_cap120")
  HYPOTHESES=("Training-composition branch: type_balanced_cap120 keeps the nested cap120 condition universe but downweights dominant perturbation types, testing whether better perturbation-type balance preserves cross-background signal while reducing family/held-out no-harm risk.")
  SPLITS=("${SPLIT_DIR}/split_seed42_xverse_trainonly_scaling_type_balanced_cap120_v2.json")
  PERT_MEANS=("${ARTIFACT_DIR}/xverse_trainonly_scaling_type_balanced_cap120_v2_pert_means.npz")
  TRAIN_CONDS=("type-balanced")
  MODE_LABEL="type-balanced cap120 training-composition probe"
elif [[ "${LATENTFM_XVERSE_SCALING_ONLY_ARM:-}" == "jiang_exposure_capped" ]]; then
  JIANG_GATE_JSON=${ROOT}/reports/latentfm_xverse_jiang_exposure_capped_split_gate_20260624.json
  COUNT_JSON=${ROOT}/reports/latentfm_xverse_scaling_count_smokes_decision_20260624.json
  if [[ ! -e "${JIANG_GATE_JSON}" ]]; then
    echo "Missing Jiang exposure-capped split gate JSON: ${JIANG_GATE_JSON}" >&2
    exit 2
  fi
  "${PYTHON}" - "${JIANG_GATE_JSON}" "${COUNT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("status") != "jiang_exposure_capped_split_gate_pass":
    raise SystemExit(f"Jiang exposure-capped split gate not passed: {gate.get('status')!r}")
count_path = Path(sys.argv[2])
if count_path.is_file():
    count = json.loads(count_path.read_text(encoding="utf-8"))
    status = (count.get("type_balance_extension_decision") or {}).get("status")
    if status == "pending":
        raise SystemExit("type_balanced extension is still pending; do not launch backup GPU branch yet")
    if status == "type_balanced_extension_pass":
        raise SystemExit("type_balanced extension passed; do not launch Jiang backup before its frozen canonical decision")
PY
  RUN_NAMES=("xverse_scaling_jiang_exposure_capped_3k_seed42")
  ARMS=("jiang_exposure_capped")
  HYPOTHESES=("Training-composition backup branch: jiang_exposure_capped starts from type_balanced_cap120 and removes high-cell Jiang train conditions to reduce dataset/cell-count exposure while preserving validation groups; launch only after type_balanced internal gate is no longer pending.")
  SPLITS=("${SPLIT_DIR}/split_seed42_xverse_trainonly_scaling_jiang_exposure_capped_v2.json")
  PERT_MEANS=("${ARTIFACT_DIR}/xverse_trainonly_scaling_jiang_exposure_capped_v2_pert_means.npz")
  TRAIN_CONDS=("1085")
  MODE_LABEL="Jiang exposure-capped training-composition backup"
elif [[ "${LATENTFM_XVERSE_SCALING_ONLY_ARM:-}" == "general_exposure_cap_v2" ]]; then
  GENERAL_GATE_JSON=${ROOT}/reports/latentfm_xverse_general_exposure_cap_v2_gate_20260624.json
  COUNT_JSON=${ROOT}/reports/latentfm_xverse_scaling_count_smokes_decision_20260624.json
  JIANG_RUN_DIR=${RUN_ROOT}/xverse_scaling_jiang_exposure_capped_3k_seed42
  if [[ ! -e "${JIANG_RUN_DIR}/POSTHOC_EXIT_CODE" || "$(cat "${JIANG_RUN_DIR}/POSTHOC_EXIT_CODE")" != "0" ]]; then
    echo "Jiang exposure-capped posthoc is not complete with exit 0; refusing general exposure launch." >&2
    exit 2
  fi
  if [[ ! -e "${GENERAL_GATE_JSON}" ]]; then
    echo "Missing general exposure-cap v2 gate JSON: ${GENERAL_GATE_JSON}" >&2
    exit 2
  fi
  if [[ ! -e "${COUNT_JSON}" ]]; then
    echo "Missing scaling count decision JSON; Jiang decision must be terminal before general exposure backup launch: ${COUNT_JSON}" >&2
    exit 2
  fi
  "${PYTHON}" - "${GENERAL_GATE_JSON}" "${COUNT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("status") != "general_exposure_cap_v2_gate_pass_no_gpu":
    raise SystemExit(f"general exposure-cap v2 gate not passed: {gate.get('status')!r}")
count = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
status = (count.get("jiang_exposure_extension_decision") or {}).get("status")
if status != "jiang_exposure_extension_fail":
    raise SystemExit(
        "general exposure-cap v2 is standby only; requires "
        f"jiang_exposure_extension_fail, got {status!r}"
    )
PY
  RUN_NAMES=("xverse_scaling_general_exposure_cap_v2_3k_seed42")
  ARMS=("general_exposure_cap_v2")
  HYPOTHESES=("Training-composition standby branch: general_exposure_cap_v2 starts from the Jiang exposure-capped split and further caps broad dataset/cell microstep exposure while preserving perturbation-type/background coverage; launch only if Jiang internal decision fails.")
  SPLITS=("${SPLIT_DIR}/split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json")
  PERT_MEANS=("${ARTIFACT_DIR}/xverse_trainonly_scaling_general_exposure_cap_v2_pert_means.npz")
  TRAIN_CONDS=("1057")
  MODE_LABEL="general dataset/cell exposure-cap v2 standby"
elif [[ "${LATENTFM_XVERSE_SCALING_ONLY_ARM:-}" == "gene_cap120_k562bg" ]]; then
  RUN_NAMES=("xverse_scaling_gene_cap120_k562bg_3k_seed42")
  ARMS=("gene_cap120_k562bg")
  HYPOTHESES=("Exploratory nested-v2 K562-like gene-only background arm tests whether training on a narrower cell-background subset changes fixed train-only internal validation; do not make a formal background-scaling claim before count and gene-only gates resolve.")
  SPLITS=("${SPLIT_DIR}/split_seed42_xverse_trainonly_scaling_gene_cap120_k562bg_v2.json")
  PERT_MEANS=("${ARTIFACT_DIR}/xverse_trainonly_scaling_gene_cap120_k562bg_v2_pert_means.npz")
  TRAIN_CONDS=("360")
  MODE_LABEL="K562-like background exploratory scaling"
elif [[ "${LATENTFM_XVERSE_SCALING_ONLY_ARM:-}" == "full_trainonly" ]]; then
  RUN_NAMES=("xverse_scaling_full_trainonly_3k_seed42")
  ARMS=("full_trainonly")
  HYPOTHESES=("Nested-v2 extension: full train-only condition coverage tests whether scaling beyond cap120_all provides additional fixed internal validation gain under the same 3k fine-tune budget.")
  SPLITS=("${TRAINONLY_SPLIT}")
  PERT_MEANS=("${TRAINONLY_PERT_MEANS}")
  TRAIN_CONDS=("16823")
  MODE_LABEL="full train-only count-scaling extension"
elif [[ -n "${LATENTFM_XVERSE_SCALING_ONLY_ARM:-}" ]]; then
  echo "Unsupported LATENTFM_XVERSE_SCALING_ONLY_ARM=${LATENTFM_XVERSE_SCALING_ONLY_ARM}" >&2
  exit 4
fi

need=${#RUN_NAMES[@]}

for i in "${!RUN_NAMES[@]}"; do
  run_name=${RUN_NAMES[$i]}
  for required in "${SPLITS[$i]}" "${PERT_MEANS[$i]}"; do
    if [[ ! -e "${required}" ]]; then
      echo "Missing run artifact for ${run_name}: ${required}" >&2
      exit 2
    fi
  done
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  session=lfm_${run_name}
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_XVERSE_SCALING_RERUN:-0}" != "1" ]]; then
    echo "Output exists for ${run_name}; set FORCE_LATENTFM_XVERSE_SCALING_RERUN=1 to relaunch" >&2
    exit 3
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

echo "[$(date '+%F %T %Z')] exact GPU status before xverse scaling count launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
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
  arm=${ARMS[$i]}
  split_file=${SPLITS[$i]}
  pert_means=${PERT_MEANS[$i]}
  run_dir=${RUN_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  gpu=${ASSIGNED_GPUS[$i]}
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
export SPLIT_FILE=${split_file}
export PERT_MEANS_FILE=${pert_means}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${log_dir}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${gpu}
export RUN_TAG=${run_name}
export SEED=42
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
eval_dir=${run_dir}/posthoc_eval_internal
mkdir -p "\${eval_dir}"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${split_file} --pert-means-file ${pert_means} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out "\${eval_dir}/split_group_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out "\${eval_dir}/split_group_eval_candidate_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_candidate_internal_ode20.json" "\${common[@]}"
EOF
  chmod +x "${posthoc_script}"

  rm -f "${run_dir}/${run_name}.EXIT_CODE" "${run_dir}/${run_name}.FINISHED" "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  date '+%F %T %Z' > "${run_dir}/${run_name}.STARTED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${train_script} > ${log_dir}/launcher.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/${run_name}.EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/${run_name}.FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${log_dir}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"

  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: ${run_name}

## Hypothesis

${HYPOTHESES[$i]}

## Command

\`\`\`bash
LATENTFM_XVERSE_SCALING_ACK=nested_v2_split_gate_pass bash ${ROOT}/ops/launch_latentfm_xverse_scaling_count_smokes_20260624.sh
\`\`\`

## Runtime classification

Long GPU training plus posthoc task. Use 30-minute cadence for checks.

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
* \`${out_dir}/iid_eval_results.json\`
* \`${run_dir}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`

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

- Scaling arm: \`${arm}\`, train conditions: ${TRAIN_CONDS[$i]}.
- Uses nested v2 train-only split for checkpoint selection:
  \`${split_file}\`
- Uses matching train-only pert means:
  \`${pert_means}\`
- Canonical split is not evaluated in this first internal gate.
- Resource policy: max 4 physical GPUs, max 4 LatentFM jobs/GPU, 48 CPU cores.
EOF

  echo "Launched ${run_name} on GPU ${gpu} in tmux ${session}"
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_scaling_count_smokes_20260624

## Command

\`\`\`bash
LATENTFM_XVERSE_SCALING_ACK=nested_v2_split_gate_pass bash ${ROOT}/ops/launch_latentfm_xverse_scaling_count_smokes_20260624.sh
\`\`\`

## Runtime classification

Long GPU training batch. Each child run has its own RUN_STATUS.md.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

$(printf '* `%s`\n' "${RUN_NAMES[@]/#/lfm_}")

## Log path

\`${LOG_ROOT}/<run_name>/launcher.log\`

## Expected outputs

* \`${RUN_ROOT}/<run_name>/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`
* \`${ROOT}/reports/LATENTFM_XVERSE_SCALING_COUNT_SMOKES_DECISION_20260624.md\`

## Current status

Started ${need} scaling smoke(s).

## Notes

- Batch mode: ${MODE_LABEL}.
- Decision gate is train-only internal validation; canonical held-out is not used for this decision.
- Run summarizer after both posthoc jobs finish:
  \`${PYTHON} ${ROOT}/ops/summarize_latentfm_xverse_scaling_count_smokes_20260624.py\`
EOF
