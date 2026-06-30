#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_strategy_probe_expanded_20260619
LOG_ROOT=${ROOT}/logs/latentfm_strategy_probe_expanded_20260619
mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy Probe Expanded 2026-06-19

Started: $(date '+%F %T %Z')
Status: running_training
Runtime classification: Long task.
Purpose: expand low-util LatentFM strategy search while keeping at most three training jobs per physical GPU.
Policy: stack two extra probes on each already-used GPU; keep CPU/RAM/I/O safe.
Expected outputs:
- ${ROOT}/reports/LATENTFM_STRATEGY_PROBE_EXPANDED_20260619.md
- ${ROOT}/reports/latentfm_strategy_probe_expanded_20260619.csv
- ${ROOT}/reports/latentfm_strategy_probe_expanded_20260619.json
EOF

SCF_INIT="${COUPLED}/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt"
STACK_INIT="${COUPLED}/output/latentfm_runs/stack_composite_selection/20260618_stack_comp006_selppmmd05_8k/best.pt"

run_probe() {
  local label="$1"
  local gpu="$2"
  local backbone="$3"
  local init_ckpt="$4"
  local out_subdir="$5"
  local comp_w="$6"
  local endpoint_w="$7"
  local cond_head_w="$8"
  local cond_target="$9"
  local cond_use="${10}"
  local additive_w="${11}"
  local gamma="${12}"
  local log="${RUN_ROOT}/logs/${label}.log"
  local exit_file="${RUN_ROOT}/logs/${label}.exit"
  local out_root="${COUPLED}/output/latentfm_runs/${out_subdir}"

  {
    echo "[$(date +%F_%T)] start ${label}"
    echo "gpu=${gpu} backbone=${backbone} comp_w=${comp_w} endpoint_w=${endpoint_w} cond_head_w=${cond_head_w} cond_target=${cond_target} cond_use=${cond_use} additive_w=${additive_w} gamma=${gamma}"
    echo "init_ckpt=${init_ckpt}"
    cd "${COUPLED}" || exit 1
    source "${ROOT}/init-scdfm.sh" >/dev/null
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    BLIS_NUM_THREADS=2 \
    GPU="${gpu}" \
    LATENT_BACKBONE="${backbone}" \
    DATA_DIR="${ROOT}/dataset/latentfm_full/${backbone}" \
    OUT_ROOT="${out_root}" \
    LOG_ROOT="${LOG_ROOT}" \
    TOTAL_STEPS=4000 \
    BATCH_SIZE=64 \
    GRAD_ACCUM_STEPS=1 \
    RUN_TAG="${label}" \
    INIT_CHECKPOINT="${init_ckpt}" \
    GAMMA="${gamma}" \
    GAMMA_WARMUP_START=300 \
    GAMMA_WARMUP_END=1200 \
    MMD_EVERY=4 \
    MMD_ESTIMATOR=unbiased \
    MMD_ODE_STEPS=0 \
    SELECTION_METRIC=pearson_pert_minus_mmd \
    SELECTION_MMD_LAMBDA=0.5 \
    COMPOSITION_DELTA_LOSS_WEIGHT="${comp_w}" \
    COMPOSITION_DELTA_LOSS_WARMUP_START=300 \
    COMPOSITION_DELTA_LOSS_WARMUP_END=1500 \
    COMPOSITION_DELTA_LOSS_EVERY=1 \
    COMPOSITION_DELTA_BANK_SIZE=1024 \
    ENDPOINT_DELTA_LOSS_WEIGHT="${endpoint_w}" \
    ENDPOINT_DELTA_LOSS_WARMUP_START=300 \
    ENDPOINT_DELTA_LOSS_WARMUP_END=1500 \
    PERT_RESIDUAL_DIRECTION_LOSS_WEIGHT=0.0 \
    PERT_RESIDUAL_CONTRASTIVE_LOSS_WEIGHT=0.0 \
    PERT_RESIDUAL_RELATIONAL_LOSS_WEIGHT=0.0 \
    CONDITION_DELTA_HEAD_LOSS_WEIGHT="${cond_head_w}" \
    CONDITION_DELTA_HEAD_LOSS_WARMUP_START=300 \
    CONDITION_DELTA_HEAD_LOSS_WARMUP_END=1500 \
    CONDITION_DELTA_HEAD_HIDDEN=1024 \
    CONDITION_DELTA_HEAD_TARGET="${cond_target}" \
    CONDITION_DELTA_HEAD_USE_IN_MODEL="${cond_use}" \
    ADDITIVE_CONDITION_DELTA_LOSS_WEIGHT="${additive_w}" \
    ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_START=300 \
    ADDITIVE_CONDITION_DELTA_LOSS_WARMUP_END=1500 \
    EVAL_MAX_CONDITIONS=256 \
    EVAL_MAX_CONDITIONS_PER_DATASET=12 \
    EVAL_MAX_MSE_CELLS=512 \
    EVAL_MAX_MMD_CELLS=512 \
    EVAL_MAX_CHUNK=256 \
    PERT_POOL_AGGREGATIONS="mean max min" \
    PERT_POOL_SCALE_INIT="1.0 1.0 1.0" \
    PERT_GENE_PROJECTOR_HIDDEN=1024 \
    PERT_CHEM_PROJECTOR_HIDDEN=1024 \
    bash model/latent/scripts/run_full_stack_latentfm.sh
    local code=$?
    echo "[$(date +%F_%T)] finish ${label} exit=${code}"
    echo "${code}" > "${exit_file}"
    return "${code}"
  } > "${log}" 2>&1
}

run_posthoc_one() {
  local label="$1"
  local gpu="$2"
  local backbone="$3"
  local out_subdir="$4"
  local run_dir="${COUPLED}/output/latentfm_runs/${out_subdir}/${label}"
  local data_dir="${ROOT}/dataset/latentfm_full/${backbone}"
  local posthoc_dir="${run_dir}/posthoc_eval"
  local log="${RUN_ROOT}/logs/${label}.posthoc.log"
  local exit_file="${RUN_ROOT}/logs/${label}.posthoc.exit"
  mkdir -p "${posthoc_dir}"
  {
    echo "[$(date +%F_%T)] start posthoc ${label}"
    if [[ ! -f "${run_dir}/best.pt" ]]; then
      echo "missing best.pt: ${run_dir}/best.pt"
      echo 3 > "${exit_file}"
      exit 3
    fi
    cd "${COUPLED}" || exit 1
    source "${ROOT}/init-scdfm.sh" >/dev/null
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export NUMEXPR_NUM_THREADS=2
    export BLIS_NUM_THREADS=2
    run_step() {
      local step_name="$1"
      shift
      echo "[$(date +%F_%T)] ${label} step=${step_name}"
      "$@"
      local code=$?
      if [[ "${code}" != "0" ]]; then
        echo "[$(date +%F_%T)] ${label} step=${step_name} failed exit=${code}"
        echo "${code}" > "${exit_file}"
        exit "${code}"
      fi
    }
    run_step split_groups python -m model.latent.eval_split_groups \
      --checkpoint "${run_dir}/best.pt" \
      --data-dir "${data_dir}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${posthoc_dir}/split_group_eval_best_ode20_mse1024_mmd1024.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    run_step condition_families python -m model.latent.eval_condition_families \
      --checkpoint "${run_dir}/best.pt" \
      --data-dir "${data_dir}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --groups test_all family_gene family_drug structure_single structure_multi type_CRISPRi type_CRISPRa type_CRISPRko type_Cas13 type_drug test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${posthoc_dir}/condition_family_eval_best_ode20_mse1024_mmd1024.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    run_step condition_residuals python -m model.latent.eval_condition_residuals \
      --checkpoint "${run_dir}/best.pt" \
      --data-dir "${data_dir}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 family_gene family_drug \
      --out-csv "${posthoc_dir}/condition_residual_full128_best.csv" \
      --out-json "${posthoc_dir}/condition_residual_full128_best.json" \
      --gpu 0 \
      --device cuda:0 \
      --ode-steps 20 \
      --max-chunk 256 \
      --eval-max-cells 128 \
      --skip-mmd
    echo "[$(date +%F_%T)] finish posthoc ${label}"
    echo 0 > "${exit_file}"
  } > "${log}" 2>&1
}

LABELS=(
  scf_e1_comp012
  scf_head_pert005
  scf_e3_comp012
  scf_add005
  stack_e1_comp012
  stack_head_pert005
  stack_e3_comp012
  stack_e2_comp020
)

run_probe scf_e1_comp012 0 scfoundation "${SCF_INIT}" scfoundation_strategy_probe_expanded_20260619 0.12 1.0 0.0 endpoint_delta 0 0.0 0.03 &
run_probe scf_head_pert005 0 scfoundation "${SCF_INIT}" scfoundation_strategy_probe_expanded_20260619 0.12 2.0 0.05 pert_residual 1 0.0 0.03 &
run_probe scf_e3_comp012 1 scfoundation "${SCF_INIT}" scfoundation_strategy_probe_expanded_20260619 0.12 3.0 0.0 endpoint_delta 0 0.0 0.03 &
run_probe scf_add005 1 scfoundation "${SCF_INIT}" scfoundation_strategy_probe_expanded_20260619 0.12 2.0 0.0 endpoint_delta 0 0.05 0.03 &
run_probe stack_e1_comp012 3 stack "${STACK_INIT}" stack_strategy_probe_expanded_20260619 0.12 1.0 0.0 endpoint_delta 0 0.0 0.03 &
run_probe stack_head_pert005 3 stack "${STACK_INIT}" stack_strategy_probe_expanded_20260619 0.12 2.0 0.05 pert_residual 1 0.0 0.03 &
run_probe stack_e3_comp012 4 stack "${STACK_INIT}" stack_strategy_probe_expanded_20260619 0.12 3.0 0.0 endpoint_delta 0 0.0 0.03 &
run_probe stack_e2_comp020 4 stack "${STACK_INIT}" stack_strategy_probe_expanded_20260619 0.20 2.0 0.0 endpoint_delta 0 0.0 0.03 &

wait

train_status=0
train_missing=0
for label in "${LABELS[@]}"; do
  exit_file="${RUN_ROOT}/logs/${label}.exit"
  if [[ ! -f "${exit_file}" ]]; then
    train_status=99
    train_missing=$((train_missing + 1))
    continue
  fi
  code="$(cat "${exit_file}")"
  if [[ "${code}" != "0" ]]; then
    train_status=1
  fi
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy Probe Expanded 2026-06-19

Started: see logs
Status: running_posthoc
Training finished: $(date '+%F %T %Z')
Training exit code: ${train_status}
Missing training exit files: ${train_missing}
EOF

if [[ "${train_status}" == "0" ]]; then
  run_posthoc_one scf_e1_comp012 0 scfoundation scfoundation_strategy_probe_expanded_20260619 &
  run_posthoc_one scf_head_pert005 0 scfoundation scfoundation_strategy_probe_expanded_20260619 &
  run_posthoc_one scf_e3_comp012 1 scfoundation scfoundation_strategy_probe_expanded_20260619 &
  run_posthoc_one scf_add005 1 scfoundation scfoundation_strategy_probe_expanded_20260619 &
  run_posthoc_one stack_e1_comp012 3 stack stack_strategy_probe_expanded_20260619 &
  run_posthoc_one stack_head_pert005 3 stack stack_strategy_probe_expanded_20260619 &
  run_posthoc_one stack_e3_comp012 4 stack stack_strategy_probe_expanded_20260619 &
  run_posthoc_one stack_e2_comp020 4 stack stack_strategy_probe_expanded_20260619 &
  wait
fi

posthoc_status=0
posthoc_missing=0
if [[ "${train_status}" != "0" ]]; then
  posthoc_status=98
else
  for label in "${LABELS[@]}"; do
    exit_file="${RUN_ROOT}/logs/${label}.posthoc.exit"
    if [[ ! -f "${exit_file}" ]]; then
      posthoc_status=99
      posthoc_missing=$((posthoc_missing + 1))
      continue
    fi
    code="$(cat "${exit_file}")"
    if [[ "${code}" != "0" ]]; then
      posthoc_status=1
    fi
  done
fi

summary_status=0
if [[ "${posthoc_status}" == "0" ]]; then
  /data/cyx/software/miniconda3/envs/scdfm/bin/python \
    "${ROOT}/ops/summarize_latentfm_strategy_probe_expanded_20260619.py" \
    > "${RUN_ROOT}/logs/summary.log" 2>&1
  summary_status=$?
fi

final_status=0
if [[ "${train_status}" != "0" || "${posthoc_status}" != "0" || "${summary_status}" != "0" ]]; then
  final_status=1
fi

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy Probe Expanded 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${final_status}
Training exit code: ${train_status}
Posthoc exit code: ${posthoc_status}
Summary exit code: ${summary_status}
Missing training exit files: ${train_missing}
Missing posthoc exit files: ${posthoc_missing}
Report: ${ROOT}/reports/LATENTFM_STRATEGY_PROBE_EXPANDED_20260619.md
CSV: ${ROOT}/reports/latentfm_strategy_probe_expanded_20260619.csv
JSON: ${ROOT}/reports/latentfm_strategy_probe_expanded_20260619.json
EOF

exit "${final_status}"
