#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
TRAIN_RUN_ROOT=${ROOT}/runs/latentfm_dataset_upper_bound_20260620
RUN_ROOT=${ROOT}/runs/latentfm_dataset_upper_bound_posthoc_20260620
TRAIN_OUT_ROOT=${COUPLED}/output/latentfm_runs/dataset_upper_bound_20260620
BASELINE_DIR=${COUPLED}/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NORMAN=scf_prior010_upperbound_norman_4k
RUN_WESSELS=scf_prior010_upperbound_wessels_4k
RUN_GASPERINI=scf_prior010_upperbound_gasperini_4k

mkdir -p "${RUN_ROOT}/logs" "${ROOT}/reports"
trap 'rc=$?; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; echo "${rc}" > "${RUN_ROOT}/EXIT_CODE"; exit "${rc}"' EXIT

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_dataset_upper_bound_posthoc_20260620

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_dataset_upper_bound_posthoc_20260620.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`latentfm_dataset_upper_bound_posthoc_20260620\`

## Log path

\`${RUN_ROOT}/logs/run.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_DATASET_UPPER_BOUND_STABLECAPS_SUMMARY_20260620.md\`
* \`${ROOT}/reports/latentfm_dataset_upper_bound_summary_20260620.json\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo still-running
tail -n 80 ${RUN_ROOT}/logs/run.log
nvidia-smi
\`\`\`

## Current status

Waiting for dataset upper-bound training EXIT_CODE files.

## Notes

This script checks training completion at 30-minute cadence with \`sleep 1800\`.
EOF

log=${RUN_ROOT}/logs/run.log
{
  echo "[$(date '+%F %T %Z')] wait for dataset upper-bound training exits"
  while true; do
    all_done=1
    for run in "${RUN_NORMAN}" "${RUN_WESSELS}" "${RUN_GASPERINI}"; do
      if [[ ! -f "${TRAIN_RUN_ROOT}/${run}.EXIT_CODE" ]]; then
        all_done=0
      fi
    done
    if [[ "${all_done}" == "1" ]]; then
      break
    fi
    echo "[$(date '+%F %T %Z')] still waiting; next check in 1800s"
    sleep 1800
  done

  for run in "${RUN_NORMAN}" "${RUN_WESSELS}" "${RUN_GASPERINI}"; do
    code="$(cat "${TRAIN_RUN_ROOT}/${run}.EXIT_CODE")"
    echo "[$(date '+%F %T %Z')] train ${run} exit=${code}"
    if [[ "${code}" != "0" ]]; then
      echo "training failed for ${run}; skip posthoc" >&2
      exit "${code}"
    fi
  done

  echo "[$(date '+%F %T %Z')] exact GPU status before dataset upper-bound posthoc"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
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
  if [[ -z "${gpu}" ]]; then
    echo "No GPU selected; see ${gpu_json}" >&2
    exit 4
  fi

  resource_audit="${RUN_ROOT}/logs/resource_audit_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" - "${gpu_json}" "${resource_audit}" <<'PY'
import json
import os
import sys
from pathlib import Path

gpu_json = Path(sys.argv[1])
out = Path(sys.argv[2])
payload = json.loads(gpu_json.read_text(encoding="utf-8"))
system = payload.get("system") or {}
min_mem = float(os.environ.get("MIN_POSTHOC_MEM_AVAILABLE_GIB", "64"))
max_load = float(os.environ.get("MAX_POSTHOC_LOAD1_PER_CPU", "2.0"))
mem = float(system.get("mem_available_gib") or 0.0)
load = float(system.get("load1_per_cpu") or 0.0)
audit = {
    "status": "pass",
    "min_mem_available_gib": min_mem,
    "max_load1_per_cpu": max_load,
    "system": system,
    "gpu_selection_json": str(gpu_json),
}
reasons = []
if mem < min_mem:
    reasons.append(f"MemAvailable {mem:.1f} GiB < {min_mem:.1f} GiB")
if load > max_load:
    reasons.append(f"load1_per_cpu {load:.3f} > {max_load:.3f}")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 5)
PY

  cd "${COUPLED}"
  source "${ROOT}/init-scdfm.sh" >/dev/null
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  export BLIS_NUM_THREADS=4

  run_posthoc() {
    local name="$1"
    local checkpoint="$2"
    local split_file="$3"
    local out_dir="$4"
    mkdir -p "${out_dir}"
    echo "[$(date '+%F %T %Z')] posthoc ${name}"
    test -f "${checkpoint}"
    "${PYTHON}" -m model.latent.eval_split_groups \
      --checkpoint "${checkpoint}" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --split-file "${split_file}" \
      --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${out_dir}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    "${PYTHON}" -m model.latent.eval_condition_families \
      --checkpoint "${checkpoint}" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --split-file "${split_file}" \
      --groups test_all family_gene structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${out_dir}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    "${PYTHON}" "${ROOT}/ops/audit_latentfm_stablecaps_selection.py" \
      --split-json "${out_dir}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --family-json "${out_dir}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --out-json "${ROOT}/reports/latentfm_stablecaps_selection_audit_${name}_upperbound_20260620.json" \
      --out-md "${ROOT}/reports/LATENTFM_STABLECAPS_SELECTION_AUDIT_${name}_UPPERBOUND_20260620.md"
  }

  run_dataset() {
    local key="$1"
    local run_name="$2"
    local split_file="${TRAIN_RUN_ROOT}/latentfm_upperbound_${key}_split_seed42_20260620.json"
    run_posthoc "baseline_${key}" "${BASELINE_DIR}/best.pt" "${split_file}" "${BASELINE_DIR}/posthoc_eval_upperbound_${key}"
    run_posthoc "${run_name}" "${TRAIN_OUT_ROOT}/${run_name}/best.pt" "${split_file}" "${TRAIN_OUT_ROOT}/${run_name}/posthoc_eval_upperbound"
  }

  run_dataset norman "${RUN_NORMAN}"
  run_dataset wessels "${RUN_WESSELS}"
  run_dataset gasperini "${RUN_GASPERINI}"

  "${PYTHON}" "${ROOT}/ops/summarize_latentfm_dataset_upper_bound_20260620.py" \
    --root "${ROOT}" \
    --baseline-dir "${BASELINE_DIR}" \
    --out-root "${TRAIN_OUT_ROOT}" \
    --out-json "${ROOT}/reports/latentfm_dataset_upper_bound_summary_20260620.json" \
    --out-md "${ROOT}/reports/LATENTFM_DATASET_UPPER_BOUND_STABLECAPS_SUMMARY_20260620.md" \
    --out-csv "${ROOT}/reports/latentfm_dataset_upper_bound_summary_20260620.csv"

  echo "[$(date '+%F %T %Z')] finished dataset upper-bound posthoc"
} > "${log}" 2>&1
