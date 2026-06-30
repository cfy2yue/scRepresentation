#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
TRAIN_RUN_ROOT=${ROOT}/runs/latentfm_condition_prior_teacher_probe_20260619
RUN_ROOT=${ROOT}/runs/latentfm_condition_prior_teacher_posthoc_20260619
LOG_ROOT=${ROOT}/logs/latentfm_condition_prior_teacher_posthoc_20260619
RUN_DIR=${COUPLED}/output/latentfm_runs/condition_prior_teacher_probe_20260619/scf_prior005_e2_4k
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
POSTHOC_DIR=${RUN_DIR}/posthoc_eval
SUMMARY=${ROOT}/ops/summarize_latentfm_condition_prior_teacher_probe_20260619.py
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}" "${POSTHOC_DIR}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Posthoc 2026-06-19

Started: $(date '+%F %T %Z')
Status: waiting_for_training_exit
Runtime classification: Long task.
Polling policy: checks only training EXIT_CODE every 30 minutes; does not inspect training logs.
Training run: ${TRAIN_RUN_ROOT}
Expected checkpoint: ${RUN_DIR}/best.pt
Outputs:
- ${ROOT}/reports/LATENTFM_CONDITION_PRIOR_TEACHER_PROBE_20260619.md
- ${ROOT}/reports/latentfm_condition_prior_teacher_probe_20260619.csv
- ${ROOT}/reports/latentfm_condition_prior_teacher_probe_20260619.json
EOF

while true; do
  if [[ -f "${TRAIN_RUN_ROOT}/EXIT_CODE" ]]; then
    break
  fi
  if ! tmux has-session -t latentfm_condition_prior_teacher_probe_20260619 2>/dev/null; then
    cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Posthoc 2026-06-19

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
# LatentFM Condition-Prior Teacher Posthoc 2026-06-19

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
# LatentFM Condition-Prior Teacher Posthoc 2026-06-19

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
# LatentFM Condition-Prior Teacher Posthoc 2026-06-19

Started: see logs
Status: waiting_for_gpu
Last checked: $(date '+%F %T %Z')
Next internal check: about 30 minutes
Latest GPU selection JSON: ${gpu_json}
EOF
  sleep 1800
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Posthoc 2026-06-19

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

posthoc_log="${RUN_ROOT}/logs/scf_prior005_e2_4k.posthoc.log"
posthoc_exit="${RUN_ROOT}/logs/scf_prior005_e2_4k.posthoc.exit"

{
  echo "[$(date +%F_%T)] start condition-prior posthoc gpu=${gpu}"
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
  echo 0 > "${posthoc_exit}"
  echo "[$(date +%F_%T)] finished posthoc"
} > "${posthoc_log}" 2>&1
posthoc_code=$?

summary_code=NA
if [[ "${posthoc_code}" == "0" ]]; then
  "${PYTHON}" "${SUMMARY}" > "${RUN_ROOT}/logs/summary.log" 2>&1
  summary_code=$?
fi

status="${posthoc_code}"
if [[ "${posthoc_code}" == "0" && "${summary_code}" != "0" ]]; then
  status="${summary_code}"
fi

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Teacher Posthoc 2026-06-19

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${status}
Posthoc exit code: ${posthoc_code}
Summary exit code: ${summary_code}
Report: ${ROOT}/reports/LATENTFM_CONDITION_PRIOR_TEACHER_PROBE_20260619.md
CSV: ${ROOT}/reports/latentfm_condition_prior_teacher_probe_20260619.csv
JSON: ${ROOT}/reports/latentfm_condition_prior_teacher_probe_20260619.json
EOF

echo "${status}" > "${RUN_ROOT}/EXIT_CODE"
date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
exit "${status}"
