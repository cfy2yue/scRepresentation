#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_CONDRES_SCALE_ACK:-}" != "condition_residual_scaling_robust_pass" ]]; then
  cat >&2 <<'EOF'
Refusing to launch condition-residual scaling slate.

Set:
  LATENTFM_CONDRES_SCALE_ACK=condition_residual_scaling_robust_pass

Required preread:
  reports/LATENTFM_CONDITION_RESIDUAL_SCALING_ROBUSTNESS_GATE_20260628.md
EOF
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_condition_residual_scaling_slate_20260628
OUT_ROOT=${COUPLED}/output/latentfm_runs/condition_residual_scaling_slate_20260628
LOG_ROOT=${ROOT}/logs/latentfm_condition_residual_scaling_slate_20260628
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
SPLIT_V2=${BIFLOW_DIR}/xverse_scaling_splits_v2_20260624
SPLIT_PROTOCOL=${BIFLOW_DIR}/xverse_scaling_protocol_splits_20260624
ARTIFACT_V2=${ROOT}/runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts
ARTIFACT_PROTOCOL=${ROOT}/runs/latentfm_scaling_protocol_splits_20260624/artifacts
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
ROBUST_JSON=${ROOT}/reports/latentfm_condition_residual_scaling_robustness_gate_20260628.json
SUMMARIZER=${ROOT}/ops/summarize_latentfm_condition_residual_scaling_slate_20260628.py
SEED_VALUE=${LATENTFM_CONDRES_SCALE_SEED:-43}
ONLY_PAIR=${LATENTFM_CONDRES_SCALE_ONLY_PAIR:-all}

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${DATA_DIR}/manifest.json" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}" \
  "${ROBUST_JSON}" \
  "${SUMMARIZER}" \
  "${ROOT}/reports/LATENTFM_CONDITION_RESIDUAL_SCALING_ROBUSTNESS_GATE_20260628.md"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - "${ROBUST_JSON}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("status") != "condition_residual_scaling_robust_pass_no_gpu":
    raise SystemExit(f"robustness gate status is not pass: {payload.get('status')!r}")
if int(payload.get("n_robust_signals") or 0) < 1:
    raise SystemExit("robustness gate has no robust signal")
PY

if [[ "${SEED_VALUE}" != "43" && "${SEED_VALUE}" != "44" ]]; then
  echo "Unsupported LATENTFM_CONDRES_SCALE_SEED=${SEED_VALUE}; supported: 43, 44" >&2
  exit 4
fi

declare -a RUN_NAMES=(
  "xverse_crscale_resp_gene_k562bg_3k_seed${SEED_VALUE}"
  "xverse_crscale_resp_breadth_manyshallow_3k_seed${SEED_VALUE}"
  "xverse_crscale_ptype_gene_allbg_3k_seed${SEED_VALUE}"
  "xverse_crscale_ptype_typebalanced_3k_seed${SEED_VALUE}"
)
declare -a ARMS=(
  "gene_cap120_k562bg"
  "breadth_many_shallow_19ds_cap30_budget480"
  "gene_cap120_allbg"
  "type_balanced_cap120"
)
declare -a PAIRS=(
  "response_strength_vs_breadth"
  "response_strength_vs_breadth"
  "perturbation_type_breadth"
  "perturbation_type_breadth"
)
declare -a HYPOTHESES=(
  "Matched-pair response-strength axis: narrow high-response K562-like gene arm should reproduce seed42 PP gains versus high-breadth shallow training, but must not introduce MMD/tail harm."
  "Matched-pair response-strength axis comparator: high dataset-breadth many-shallow arm tests whether breadth alone beats response magnitude at fixed-ish condition budget."
  "Matched-pair perturbation-type breadth baseline: gene-only all-background arm tests lower perturbation-type breadth at similar large train-support scale."
  "Matched-pair perturbation-type breadth arm: type-balanced cap120 tests whether added perturbation-type breadth can improve tails/generalization without PP regression."
)
declare -a SPLITS=(
  "${SPLIT_V2}/split_seed42_xverse_trainonly_scaling_gene_cap120_k562bg_v2.json"
  "${SPLIT_PROTOCOL}/split_seed42_xverse_scaling_protocol_breadth_many_shallow_19ds_cap30_budget480.json"
  "${SPLIT_V2}/split_seed42_xverse_trainonly_scaling_gene_cap120_allbg_v2.json"
  "${SPLIT_V2}/split_seed42_xverse_trainonly_scaling_type_balanced_cap120_v2.json"
)
declare -a PERT_MEANS=(
  "${ARTIFACT_V2}/xverse_trainonly_scaling_gene_cap120_k562bg_v2_pert_means.npz"
  "${ARTIFACT_PROTOCOL}/breadth_many_shallow_19ds_cap30_budget480_trainonly_pert_means.npz"
  "${ARTIFACT_V2}/xverse_trainonly_scaling_gene_cap120_allbg_v2_pert_means.npz"
  "${ARTIFACT_V2}/xverse_trainonly_scaling_type_balanced_cap120_v2_pert_means.npz"
)
declare -a TRAIN_CONDS=("360" "480" "1222" "1114")

case "${ONLY_PAIR}" in
  all)
    ;;
  response_strength_vs_breadth)
    RUN_NAMES=("${RUN_NAMES[0]}" "${RUN_NAMES[1]}")
    ARMS=("${ARMS[0]}" "${ARMS[1]}")
    PAIRS=("${PAIRS[0]}" "${PAIRS[1]}")
    HYPOTHESES=("${HYPOTHESES[0]}" "${HYPOTHESES[1]}")
    SPLITS=("${SPLITS[0]}" "${SPLITS[1]}")
    PERT_MEANS=("${PERT_MEANS[0]}" "${PERT_MEANS[1]}")
    TRAIN_CONDS=("${TRAIN_CONDS[0]}" "${TRAIN_CONDS[1]}")
    ;;
  perturbation_type_breadth)
    RUN_NAMES=("${RUN_NAMES[2]}" "${RUN_NAMES[3]}")
    ARMS=("${ARMS[2]}" "${ARMS[3]}")
    PAIRS=("${PAIRS[2]}" "${PAIRS[3]}")
    HYPOTHESES=("${HYPOTHESES[2]}" "${HYPOTHESES[3]}")
    SPLITS=("${SPLITS[2]}" "${SPLITS[3]}")
    PERT_MEANS=("${PERT_MEANS[2]}" "${PERT_MEANS[3]}")
    TRAIN_CONDS=("${TRAIN_CONDS[2]}" "${TRAIN_CONDS[3]}")
    ;;
  *)
    echo "Unsupported LATENTFM_CONDRES_SCALE_ONLY_PAIR=${ONLY_PAIR}" >&2
    exit 4
    ;;
esac

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
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_CONDRES_SCALE_RERUN:-0}" != "1" ]]; then
    echo "Output exists for ${run_name}; set FORCE_LATENTFM_CONDRES_SCALE_RERUN=1 to relaunch" >&2
    exit 3
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

audit_log=${RUN_ROOT}/logs/gpu_launch_audit.log
{
  echo "[$(date '+%F %T %Z')] resource audit before condition-residual scaling slate"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  free -h
  df -h "${ROOT}"
  ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 20
} | tee "${audit_log}"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 2 \
  --max-jobs-per-gpu 2 \
  --need "${need}" \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" "${need}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
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
if int(payload.get("max_user_gpus") or 0) > 2:
    reasons.append("max_user_gpus exceeds active cap 2")
if int(payload.get("max_jobs_per_gpu") or 0) > 2:
    reasons.append("max_jobs_per_gpu exceeds active cap 2")
if len(set(suggested[:need])) > 2:
    reasons.append("assigned physical GPU count exceeds active cap 2")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 1.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 1.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

mapfile -t ASSIGNED_GPUS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
for gpu in payload["assigned_gpus"]:
    print(int(gpu))
PY
)

for i in "${!RUN_NAMES[@]}"; do
  run_name=${RUN_NAMES[$i]}
  arm=${ARMS[$i]}
  pair=${PAIRS[$i]}
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
export SEED=${SEED_VALUE}
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
LATENTFM_CONDRES_SCALE_ACK=condition_residual_scaling_robust_pass bash ${ROOT}/ops/launch_latentfm_condition_residual_scaling_slate_20260628.sh
\`\`\`

## Runtime classification

Long GPU training plus posthoc task. Check at 30-minute cadence only.

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
* \`${run_dir}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`
* \`${run_dir}/posthoc_eval_internal/condition_family_eval_candidate_internal_ode20.json\`

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

- Pair: \`${pair}\`; arm: \`${arm}\`; seed: \`${SEED_VALUE}\`.
- Train conditions: ${TRAIN_CONDS[$i]}.
- Split: \`${split_file}\`
- Pert means: \`${pert_means}\`
- Selection and posthoc are train-only internal only; canonical multi and Track C query are not used.
- Fail-close: if pair-level PP gain is absent or MMD/tail harm appears in the summarizer, do not extend this arm.
EOF

  echo "Launched ${run_name} on GPU ${gpu} in tmux ${session}"
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_condition_residual_scaling_slate_20260628

## Command

\`\`\`bash
LATENTFM_CONDRES_SCALE_ACK=condition_residual_scaling_robust_pass bash ${ROOT}/ops/launch_latentfm_condition_residual_scaling_slate_20260628.sh
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
* \`${ROOT}/reports/LATENTFM_CONDITION_RESIDUAL_SCALING_SLATE_DECISION_20260628.md\`

## How to check manually

\`\`\`bash
tmux ls
cat ${RUN_ROOT}/*/*EXIT_CODE 2>/dev/null || true
cat ${RUN_ROOT}/*/POSTHOC_EXIT_CODE 2>/dev/null || true
${PYTHON} ${SUMMARIZER}
\`\`\`

## Current status

Started ${need} condition-residual scaling smokes.

## Notes

- Resource cap applied: max 2 physical GPUs, max 2 LatentFM training jobs per GPU, 24 project CPU cores.
- Requested seed: \`${SEED_VALUE}\`; requested pair filter: \`${ONLY_PAIR}\`.
- Hypotheses: response-strength vs breadth; perturbation-type breadth vs gene-only.
- GPU authorization remains exploratory only; promotion requires summarizer pass and later frozen no-harm gate.
EOF
