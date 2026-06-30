#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_condition_prior_teacher_sister_posthoc_20260619
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

LABELS=(scf_prior002_e2_4k scf_prior010_e2_4k)
TRAIN_ROOTS=(
  "${ROOT}/runs/latentfm_condition_prior_teacher_prior002_20260619"
  "${ROOT}/runs/latentfm_condition_prior_teacher_prior010_20260619"
)

mkdir -p "${RUN_ROOT}/logs"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Sister Posthoc 2026-06-19

Started: $(date '+%F %T %Z')
Status: waiting_for_training_exit
Runtime classification: Long task.
Polling policy: checks only training EXIT_CODE files every 30 minutes; does not inspect training logs.
Labels:
- scf_prior002_e2_4k
- scf_prior010_e2_4k
EOF

all_done() {
  for root in "${TRAIN_ROOTS[@]}"; do
    [[ -f "${root}/EXIT_CODE" ]] || return 1
  done
  return 0
}

while ! all_done; do
  done_count=0
  for root in "${TRAIN_ROOTS[@]}"; do
    [[ -f "${root}/EXIT_CODE" ]] && done_count=$((done_count + 1))
  done
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Sister Posthoc 2026-06-19

Started: see logs
Status: waiting_for_training_exit
Last checked: $(date '+%F %T %Z')
Next internal check: about 30 minutes
Finished training markers: ${done_count} / ${#TRAIN_ROOTS[@]}
EOF
  sleep 1800
done

failed=0
for root in "${TRAIN_ROOTS[@]}"; do
  code="$(cat "${root}/EXIT_CODE" 2>/dev/null || echo 99)"
  if [[ "${code}" != "0" ]]; then
    failed=$((failed + 1))
  fi
done
if [[ "${failed}" != "0" ]]; then
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Sister Posthoc 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: skipped_training_failed
Failed training count: ${failed}
Exit code: 1
EOF
  echo 1 > "${RUN_ROOT}/EXIT_CODE"
  date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
  exit 1
fi

while true; do
  gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" "${GPU_HELPER}" \
    --samples 3 \
    --interval-seconds 10 \
    --need 1 \
    --json-only \
    > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"
  gpu="$("${PYTHON}" - "${gpu_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = payload.get("suggested_job_gpus") or []
print(chosen[0] if chosen else "")
PY
)"
  if [[ -n "${gpu}" ]]; then
    break
  fi
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Sister Posthoc 2026-06-19

Started: see logs
Status: waiting_for_gpu
Last checked: $(date '+%F %T %Z')
Next internal check: about 30 minutes
Latest GPU selection JSON: ${gpu_json}
EOF
  sleep 1800
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Sister Posthoc 2026-06-19

Started: see logs
Status: running_posthoc
Started posthoc: $(date '+%F %T %Z')
Selected physical GPU: ${gpu}
EOF

cd "${COUPLED}" || exit 1
source "${ROOT}/init-scdfm.sh" >/dev/null
export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4

run_posthoc_one() {
  local label="$1"
  local run_dir="${COUPLED}/output/latentfm_runs/condition_prior_teacher_probe_20260619/${label}"
  local data_dir="${ROOT}/dataset/latentfm_full/scfoundation"
  local posthoc_dir="${run_dir}/posthoc_eval"
  local log="${RUN_ROOT}/logs/${label}.posthoc.log"
  local exit_file="${RUN_ROOT}/logs/${label}.posthoc.exit"
  mkdir -p "${posthoc_dir}"
  {
    echo "[$(date +%F_%T)] start posthoc ${label} gpu=${gpu}"
    if [[ ! -f "${run_dir}/best.pt" ]]; then
      echo "missing best.pt: ${run_dir}/best.pt"
      echo 3 > "${exit_file}"
      exit 3
    fi
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
    echo 0 > "${exit_file}"
    echo "[$(date +%F_%T)] finish posthoc ${label}"
  } > "${log}" 2>&1
}

status=0
for label in "${LABELS[@]}"; do
  run_posthoc_one "${label}"
  code="$(cat "${RUN_ROOT}/logs/${label}.posthoc.exit" 2>/dev/null || echo 99)"
  if [[ "${code}" != "0" ]]; then
    status=1
  fi
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Sister Posthoc 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${status}
Outputs:
- ${COUPLED}/output/latentfm_runs/condition_prior_teacher_probe_20260619/scf_prior002_e2_4k/posthoc_eval
- ${COUPLED}/output/latentfm_runs/condition_prior_teacher_probe_20260619/scf_prior010_e2_4k/posthoc_eval
EOF

echo "${status}" > "${RUN_ROOT}/EXIT_CODE"
date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
exit "${status}"
