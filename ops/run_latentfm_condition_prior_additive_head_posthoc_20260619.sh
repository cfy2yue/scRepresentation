#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
TRAIN_RUN_ROOT=${ROOT}/runs/latentfm_condition_prior_additive_head_20260619
RUN_ROOT=${ROOT}/runs/latentfm_condition_prior_additive_head_posthoc_20260619
RUN_DIR=${COUPLED}/output/latentfm_runs/condition_prior_additive_head_20260619/scf_prioradd005_prior010_inject_e2_4k
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
POSTHOC_DIR=${RUN_DIR}/posthoc_eval
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${RUN_ROOT}/logs" "${POSTHOC_DIR}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Additive Head Posthoc 2026-06-19

Started: $(date '+%F %T %Z')
Status: waiting_for_training_exit
Runtime classification: Long task.
Polling policy: checks only training EXIT_CODE every 30 minutes; does not inspect training logs.
Training run: ${TRAIN_RUN_ROOT}
Expected checkpoint: ${RUN_DIR}/best.pt
Outputs:
- ${POSTHOC_DIR}/split_group_eval_best_ode20_mse1024_mmd1024.json
- ${POSTHOC_DIR}/condition_family_eval_best_ode20_mse1024_mmd1024.json
- ${POSTHOC_DIR}/condition_residual_full128_best.csv
- ${POSTHOC_DIR}/condition_residual_full128_best.json
- ${POSTHOC_DIR}/condition_delta_decomposition_full128_best.csv
- ${POSTHOC_DIR}/condition_delta_decomposition_full128_best.json
EOF

while true; do
  if [[ -f "${TRAIN_RUN_ROOT}/EXIT_CODE" ]]; then
    break
  fi
  if ! tmux has-session -t lfm_prioradd_20260619 2>/dev/null; then
    cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Additive Head Posthoc 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: failed
Exit code: 98
Reason: training tmux session is gone but ${TRAIN_RUN_ROOT}/EXIT_CODE is missing.
EOF
    echo 98 > "${RUN_ROOT}/EXIT_CODE"
    date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
    exit 98
  fi
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Additive Head Posthoc 2026-06-19

Started: see logs
Status: waiting_for_training_exit
Last checked: $(date '+%F %T %Z')
Next internal check: about 30 minutes
Training EXIT_CODE present: no
EOF
  sleep 1800
done

train_code="$(cat "${TRAIN_RUN_ROOT}/EXIT_CODE" 2>/dev/null || echo 99)"
if [[ "${train_code}" != "0" ]]; then
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Additive Head Posthoc 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: skipped_training_failed
Training exit code: ${train_code}
Exit code: ${train_code}
EOF
  echo "${train_code}" > "${RUN_ROOT}/EXIT_CODE"
  date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
  exit "${train_code}"
fi

while true; do
  gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" "${GPU_HELPER}" \
    --samples 3 \
    --interval-seconds 10 \
    --need 1 \
    --max-jobs-per-gpu 3 \
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
# LatentFM Condition-Prior Additive Head Posthoc 2026-06-19

Started: see logs
Status: waiting_for_gpu
Last checked: $(date '+%F %T %Z')
Next internal check: about 30 minutes
Latest GPU selection JSON: ${gpu_json}
EOF
  sleep 1800
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Additive Head Posthoc 2026-06-19

Started: see logs
Status: running_posthoc
Started posthoc: $(date '+%F %T %Z')
Selected physical GPU: ${gpu}
Run dir: ${RUN_DIR}
EOF

cd "${COUPLED}" || exit 1
source "${ROOT}/init-scdfm.sh" >/dev/null
export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4

posthoc_log="${RUN_ROOT}/logs/scf_prioradd005_prior010_inject_e2_4k.posthoc.log"
posthoc_exit="${RUN_ROOT}/logs/scf_prioradd005_prior010_inject_e2_4k.posthoc.exit"

{
  echo "[$(date +%F_%T)] start prior-additive posthoc gpu=${gpu}"
  if [[ ! -f "${RUN_DIR}/best.pt" ]]; then
    echo "missing best.pt: ${RUN_DIR}/best.pt"
    echo 3 > "${posthoc_exit}"
    exit 3
  fi
  run_step() {
    local step_name="$1"
    shift
    echo "[$(date +%F_%T)] step=${step_name}"
    "$@"
    local code=$?
    if [[ "${code}" != "0" ]]; then
      echo "[$(date +%F_%T)] step=${step_name} failed exit=${code}"
      echo "${code}" > "${posthoc_exit}"
      exit "${code}"
    fi
  }
  run_step split_groups python -m model.latent.eval_split_groups \
    --checkpoint "${RUN_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${ROOT}/dataset/biFlow_data" \
    --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
    --out "${POSTHOC_DIR}/split_group_eval_best_ode20_mse1024_mmd1024.json" \
    --gpu 0 \
    --ode-steps 20 \
    --max-chunk 512 \
    --eval-max-mse-cells 1024 \
    --eval-max-mmd-cells 1024
  run_step condition_families python -m model.latent.eval_condition_families \
    --checkpoint "${RUN_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${ROOT}/dataset/biFlow_data" \
    --groups test_all family_gene family_drug structure_single structure_multi type_CRISPRi type_CRISPRa type_CRISPRko type_Cas13 type_drug test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
    --out "${POSTHOC_DIR}/condition_family_eval_best_ode20_mse1024_mmd1024.json" \
    --gpu 0 \
    --ode-steps 20 \
    --max-chunk 512 \
    --eval-max-mse-cells 1024 \
    --eval-max-mmd-cells 1024
  run_step condition_residuals python -m model.latent.eval_condition_residuals \
    --checkpoint "${RUN_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${ROOT}/dataset/biFlow_data" \
    --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 family_gene family_drug \
    --out-csv "${POSTHOC_DIR}/condition_residual_full128_best.csv" \
    --out-json "${POSTHOC_DIR}/condition_residual_full128_best.json" \
    --gpu 0 \
    --device cuda:0 \
    --ode-steps 20 \
    --max-chunk 256 \
    --eval-max-cells 128 \
    --skip-mmd
  run_step condition_delta_decomposition python -m model.latent.eval_condition_delta_decomposition \
    --checkpoint "${RUN_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${ROOT}/dataset/biFlow_data" \
    --groups test_multi_seen test_multi_unseen1 test_multi_unseen2 family_gene family_drug \
    --out-csv "${POSTHOC_DIR}/condition_delta_decomposition_full128_best.csv" \
    --out-json "${POSTHOC_DIR}/condition_delta_decomposition_full128_best.json" \
    --gpu 0 \
    --device cuda:0 \
    --eval-max-cells 128
  echo "0" > "${posthoc_exit}"
} > "${posthoc_log}" 2>&1
posthoc_code="$(cat "${posthoc_exit}" 2>/dev/null || echo 99)"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Additive Head Posthoc 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${posthoc_code}
Selected physical GPU: ${gpu}
Outputs:
- ${POSTHOC_DIR}/split_group_eval_best_ode20_mse1024_mmd1024.json
- ${POSTHOC_DIR}/condition_family_eval_best_ode20_mse1024_mmd1024.json
- ${POSTHOC_DIR}/condition_residual_full128_best.csv
- ${POSTHOC_DIR}/condition_residual_full128_best.json
- ${POSTHOC_DIR}/condition_delta_decomposition_full128_best.csv
- ${POSTHOC_DIR}/condition_delta_decomposition_full128_best.json
EOF

echo "${posthoc_code}" > "${RUN_ROOT}/EXIT_CODE"
date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
exit "${posthoc_code}"
