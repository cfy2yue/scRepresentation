#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_NAME=${RUN_NAME:-xverse_comp006_endpoint5_2k_smoke}
TRAIN_OUT_ROOT=${TRAIN_OUT_ROOT:-${COUPLED}/output/latentfm_runs/xverse_smoke_20260620}
RUN_DIR=${TRAIN_OUT_ROOT}/${RUN_NAME}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
SPLIT_FILE=${ROOT}/dataset/biFlow_data/split_seed42.json
RUN_ROOT=${RUN_ROOT:-${ROOT}/runs/latentfm_xverse_smoke_posthoc_20260620}
LOG_DIR=${RUN_ROOT}/logs
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${LOG_DIR}"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"
date "+%F %T %Z" > "${RUN_ROOT}/STARTED"
cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_smoke_posthoc_20260620

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_xverse_smoke_posthoc_20260620.sh
\`\`\`

## Runtime classification

GPU posthoc evaluation. Use 30-minute cadence for checks.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## Log path

\`${LOG_DIR}/run.log\`

## Expected outputs

* \`${RUN_DIR}/posthoc_eval/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json\`
* \`${RUN_DIR}/posthoc_eval/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json\`
* \`${ROOT}/reports/LATENTFM_STABLECAPS_SELECTION_AUDIT_${RUN_NAME}_20260620.md\`

## Current status

Started.
EOF

trap 'rc=$?; echo "$rc" > "${RUN_ROOT}/EXIT_CODE"; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; exit "$rc"' EXIT

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "${LOG_DIR}/run.log"
}

if [[ ! -s "${RUN_DIR}/best.pt" ]]; then
  echo "Missing best checkpoint: ${RUN_DIR}/best.pt" >&2
  exit 2
fi
if [[ ! -s "${DATA_DIR}/manifest.json" ]]; then
  echo "Missing xverse bundle manifest: ${DATA_DIR}/manifest.json" >&2
  exit 3
fi

log "exact GPU status before xverse smoke posthoc"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${LOG_DIR}/gpu_launch_audit.log" | tee -a "${LOG_DIR}/run.log"

gpu_json="${LOG_DIR}/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --need 1 \
  --max-jobs-per-gpu 3 \
  --json-only \
  > "${gpu_json}" 2> "${LOG_DIR}/gpu_selection.stderr"

gpu="$("${PYTHON}" - "${gpu_json}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = payload.get("suggested_job_gpus") or []
print(chosen[0] if chosen else "")
PY
)"
if [[ -z "${gpu}" ]]; then
  echo "No GPU selected by helper; see ${gpu_json}" >&2
  exit 4
fi

"${PYTHON}" - "${gpu_json}" <<'PY' | tee -a "${LOG_DIR}/run.log"
import json, os, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
system = payload.get("system") or {}
mem = float(system.get("mem_available_gib") or 0.0)
load = float(system.get("load1_per_cpu") or 0.0)
audit = {
    "status": "pass",
    "min_mem_available_gib": 64.0,
    "max_load1_per_cpu": 2.0,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if mem < 64.0:
    reasons.append(f"MemAvailable {mem:.1f} GiB < 64.0 GiB")
if load > 2.0:
    reasons.append(f"load1_per_cpu {load:.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 5)
PY

log "selected physical GPU${gpu}"

(
  source "${ROOT}/init-scdfm.sh" >/dev/null
  cd "${COUPLED}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  export BLIS_NUM_THREADS=4
  export PYTHONPATH="${COUPLED}:${PYTHONPATH:-}"
  export PERT_EMBED_SOURCE=scgpt_embed_gene

  out_dir="${RUN_DIR}/posthoc_eval"
  mkdir -p "${out_dir}"
  "${PYTHON}" -m model.latent.eval_split_groups \
    --checkpoint "${RUN_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${ROOT}/dataset/biFlow_data" \
    --split-file "${SPLIT_FILE}" \
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
    --checkpoint "${RUN_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${ROOT}/dataset/biFlow_data" \
    --split-file "${SPLIT_FILE}" \
    --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
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
    --out-json "${ROOT}/reports/latentfm_stablecaps_selection_audit_${RUN_NAME}_20260620.json" \
    --out-md "${ROOT}/reports/LATENTFM_STABLECAPS_SELECTION_AUDIT_${RUN_NAME}_20260620.md"
) 2>&1 | tee -a "${LOG_DIR}/run.log"

log "done"
