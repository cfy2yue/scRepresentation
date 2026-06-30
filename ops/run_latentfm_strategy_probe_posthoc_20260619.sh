#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_strategy_probe_posthoc_20260619
LOG_ROOT=${ROOT}/logs/latentfm_strategy_probe_posthoc_20260619
mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy Probe Posthoc 2026-06-19

Started: $(date '+%F %T %Z')
Status: waiting_for_training
Runtime classification: Long task.
Training session: latentfm_strategy_probe_20260619
Outputs:
- ${ROOT}/reports/LATENTFM_STRATEGY_PROBE_20260619.md
- ${ROOT}/reports/latentfm_strategy_probe_20260619.csv
- ${ROOT}/reports/latentfm_strategy_probe_20260619.json
EOF

ACTIVE_LABELS=(
  scf_e2_comp012_pr0
  scf_e2_comp020_pr0
  stack_e2_comp006_pr0
  stack_e2_comp012_pr0
)

gpu_for_label() {
  case "$1" in
    scf_e2_comp012_pr0) echo 0 ;;
    scf_e2_comp020_pr0) echo 1 ;;
    stack_e2_comp006_pr0) echo 3 ;;
    stack_e2_comp012_pr0) echo 4 ;;
    *) echo 0 ;;
  esac
}

backbone_for_label() {
  case "$1" in
    scf_*) echo scfoundation ;;
    stack_*) echo stack ;;
    *) echo stack ;;
  esac
}

run_dir_for_label() {
  local label="$1"
  local backbone
  backbone="$(backbone_for_label "${label}")"
  if [[ "${backbone}" == "scfoundation" ]]; then
    echo "${COUPLED}/output/latentfm_runs/scfoundation_strategy_probe_20260619/${label}"
  else
    echo "${COUPLED}/output/latentfm_runs/stack_strategy_probe_20260619/${label}"
  fi
}

training_pids_remaining() {
  local count=0
  for label in "${ACTIVE_LABELS[@]}"; do
    if pgrep -af "model.latent.train" | grep -F "${label}" >/dev/null 2>&1; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

while true; do
  remaining="$(training_pids_remaining)"
  if [[ "${remaining}" == "0" ]]; then
    break
  fi
  {
    echo "# LatentFM Strategy Probe Posthoc 2026-06-19"
    echo
    echo "Started: see logs"
    echo "Status: waiting_for_training"
    echo "Remaining active training processes: ${remaining}"
    echo "Last checked: $(date '+%F %T %Z')"
    echo "Next internal check: about 30 minutes"
  } > "${RUN_ROOT}/RUN_STATUS.md"
  sleep 1800
done

{
  echo "# LatentFM Strategy Probe Posthoc 2026-06-19"
  echo
  echo "Started: see logs"
  echo "Status: running_posthoc"
  echo "Training finished by: $(date '+%F %T %Z')"
} > "${RUN_ROOT}/RUN_STATUS.md"

cd "${COUPLED}" || exit 1
source "${ROOT}/init-scdfm.sh" >/dev/null

run_posthoc_one() {
  local label="$1"
  local gpu="$2"
  local backbone="$3"
  local run_dir="$4"
  local data_dir="${ROOT}/dataset/latentfm_full/${backbone}"
  local posthoc_dir="${run_dir}/posthoc_eval"
  local log="${RUN_ROOT}/logs/${label}.posthoc.log"
  local exit_file="${RUN_ROOT}/logs/${label}.posthoc.exit"
  mkdir -p "${posthoc_dir}"
  {
    echo "[$(date +%F_%T)] start posthoc ${label} gpu=${gpu} backbone=${backbone}"
    if [[ ! -f "${run_dir}/best.pt" ]]; then
      echo "missing best.pt: ${run_dir}/best.pt"
      echo 3 > "${exit_file}"
      exit 3
    fi
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    export BLIS_NUM_THREADS=4
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

for label in "${ACTIVE_LABELS[@]}"; do
  gpu="$(gpu_for_label "${label}")"
  backbone="$(backbone_for_label "${label}")"
  run_dir="$(run_dir_for_label "${label}")"
  run_posthoc_one "${label}" "${gpu}" "${backbone}" "${run_dir}" &
done

wait

status=0
missing=0
for label in "${ACTIVE_LABELS[@]}"; do
  exit_file="${RUN_ROOT}/logs/${label}.posthoc.exit"
  if [[ ! -f "${exit_file}" ]]; then
    missing=$((missing + 1))
    status=99
    continue
  fi
  code="$(cat "${exit_file}")"
  if [[ "${code}" != "0" ]]; then
    status=1
  fi
done

if [[ "${status}" == "0" ]]; then
  /data/cyx/software/miniconda3/envs/scdfm/bin/python \
    "${ROOT}/ops/summarize_latentfm_strategy_probe_20260619.py" \
    > "${RUN_ROOT}/logs/summary.log" 2>&1
  summary_code=$?
  if [[ "${summary_code}" != "0" ]]; then
    status=2
  fi
fi

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Strategy Probe Posthoc 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${status}
Missing posthoc exit files: ${missing}
Report: ${ROOT}/reports/LATENTFM_STRATEGY_PROBE_20260619.md
CSV: ${ROOT}/reports/latentfm_strategy_probe_20260619.csv
JSON: ${ROOT}/reports/latentfm_strategy_probe_20260619.json
EOF

exit "${status}"
