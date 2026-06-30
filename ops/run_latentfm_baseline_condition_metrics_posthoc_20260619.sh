#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_baseline_condition_metrics_posthoc_20260619
BASELINE_DIR=${COUPLED}/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

POSTHOC_DIR=${BASELINE_DIR}/posthoc_eval_condition_metrics_20260619

mkdir -p "${RUN_ROOT}/logs" "${POSTHOC_DIR}" "${ROOT}/reports"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_baseline_condition_metrics_posthoc_20260619

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_baseline_condition_metrics_posthoc_20260619.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

Run in detached tmux by caller when launched.

## Log path

\`${RUN_ROOT}/logs/run.log\`

## Expected outputs

* \`${POSTHOC_DIR}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps_condition_metrics.json\`
* \`${POSTHOC_DIR}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps_condition_metrics.json\`
* \`${ROOT}/reports/latentfm_stablecaps_selection_audit_baseline_condition_metrics_20260619.json\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo still-running
tail -n 80 ${RUN_ROOT}/logs/run.log
nvidia-smi
\`\`\`

## Current status

Prepared / running when launched.

## Notes

This reruns the completed baseline stablecaps posthoc with the updated evaluator
that records \`condition_metrics\`. It writes to a separate posthoc directory and
does not overwrite the original stablecaps gate artifacts.
EOF

log=${RUN_ROOT}/logs/run.log
{
  echo "[$(date '+%F %T %Z')] wait 1800s before baseline condition-metrics posthoc"
  sleep 1800

  echo "[$(date '+%F %T %Z')] exact GPU status before baseline condition-metrics posthoc"
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

  test -f "${BASELINE_DIR}/best.pt"

  "${PYTHON}" -m model.latent.eval_split_groups \
    --checkpoint "${BASELINE_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${ROOT}/dataset/biFlow_data" \
    --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
    --out "${POSTHOC_DIR}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps_condition_metrics.json" \
    --gpu 0 \
    --ode-steps 20 \
    --max-chunk 512 \
    --eval-max-mse-cells 1024 \
    --eval-max-mmd-cells 1024

  "${PYTHON}" -m model.latent.eval_condition_families \
    --checkpoint "${BASELINE_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${ROOT}/dataset/biFlow_data" \
    --groups test_all family_gene family_drug structure_single structure_multi type_CRISPRi type_CRISPRa type_CRISPRko type_Cas13 type_drug test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
    --out "${POSTHOC_DIR}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps_condition_metrics.json" \
    --gpu 0 \
    --ode-steps 20 \
    --max-chunk 512 \
    --eval-max-mse-cells 1024 \
    --eval-max-mmd-cells 1024

  "${PYTHON}" "${ROOT}/ops/audit_latentfm_stablecaps_selection.py" \
    --split-json "${POSTHOC_DIR}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps_condition_metrics.json" \
    --family-json "${POSTHOC_DIR}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps_condition_metrics.json" \
    --out-json "${ROOT}/reports/latentfm_stablecaps_selection_audit_baseline_condition_metrics_20260619.json" \
    --out-md "${ROOT}/reports/LATENTFM_STABLECAPS_SELECTION_AUDIT_BASELINE_CONDITION_METRICS_20260619.md"

  echo "[$(date '+%F %T %Z')] finished baseline condition-metrics posthoc"
} > "${log}" 2>&1

echo 0 > "${RUN_ROOT}/EXIT_CODE"
date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"
