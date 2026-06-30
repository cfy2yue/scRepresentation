#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_strategy_probe_20260619
LOG_ROOT=${ROOT}/logs/latentfm_strategy_probe_20260619
mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy Probe 2026-06-19

Started: $(date '+%F %T %Z')
Status: running
Purpose: short capped finetune probes after full128 residual audit.
Policy: four active physical GPUs maximum; do not exceed safe CPU/RAM/I/O limits.
EOF

run_probe() {
  local label="$1"
  local gpu="$2"
  local backbone="$3"
  local init_ckpt="$4"
  local out_subdir="$5"
  local comp_w="$6"
  local endpoint_w="$7"
  local pert_resid_w="$8"
  local log="${RUN_ROOT}/logs/${label}.log"
  local exit_file="${RUN_ROOT}/logs/${label}.exit"
  local out_root="${COUPLED}/output/latentfm_runs/${out_subdir}"

  {
    echo "[$(date +%F_%T)] start ${label}"
    echo "gpu=${gpu} backbone=${backbone} comp_w=${comp_w} endpoint_w=${endpoint_w} pert_resid_w=${pert_resid_w}"
    echo "init_ckpt=${init_ckpt}"
    cd "${COUPLED}" || exit 1
    source "${ROOT}/init-scdfm.sh" >/dev/null
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    BLIS_NUM_THREADS=4 \
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
    GAMMA=0.03 \
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
    PERT_RESIDUAL_DIRECTION_LOSS_WEIGHT="${pert_resid_w}" \
    PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_START=300 \
    PERT_RESIDUAL_DIRECTION_LOSS_WARMUP_END=1500 \
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

SCF_INIT="${COUPLED}/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt"
STACK_INIT="${COUPLED}/output/latentfm_runs/stack_composite_selection/20260618_stack_comp006_selppmmd05_8k/best.pt"

run_probe scf_e2_comp012_pr0 0 scfoundation "${SCF_INIT}" scfoundation_strategy_probe_20260619 0.12 2.0 0.0 &
run_probe scf_e2_comp020_pr0 1 scfoundation "${SCF_INIT}" scfoundation_strategy_probe_20260619 0.20 2.0 0.0 &
run_probe stack_e2_comp006_pr0 3 stack "${STACK_INIT}" stack_strategy_probe_20260619 0.06 2.0 0.0 &
run_probe stack_e2_comp012_pr0 4 stack "${STACK_INIT}" stack_strategy_probe_20260619 0.12 2.0 0.0 &

wait

status=0
missing=0
for label in \
  scf_e2_comp012_pr0 \
  scf_e2_comp020_pr0 \
  stack_e2_comp006_pr0 \
  stack_e2_comp012_pr0; do
  exit_file="${RUN_ROOT}/logs/${label}.exit"
  if [[ ! -f "${exit_file}" ]]; then
    status=99
    missing=$((missing + 1))
    continue
  fi
  code=$(cat "${exit_file}")
  if [[ "${code}" != "0" ]]; then
    status=1
  fi
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy Probe 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${status}
Missing exit files: ${missing}
Output roots:
- ${COUPLED}/output/latentfm_runs/scfoundation_strategy_probe_20260619
- ${COUPLED}/output/latentfm_runs/stack_strategy_probe_20260619
Notes:
- The initially considered weak pert_residual_direction controls
  scf_e2_comp012_pr002 and stack_e2_comp006_pr002 are intentionally not part
  of the active four-GPU probe matrix.
EOF

exit "${status}"
