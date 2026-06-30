#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_corrected_dsloss_smoke_20260619
OUT_ROOT=${COUPLED}/output/latentfm_runs/corrected_dsloss_smoke_20260619
LOG_ROOT=${ROOT}/logs/latentfm_corrected_dsloss_smoke_20260619
AUDIT_JSON=${ROOT}/reports/latentfm_stablecaps_selection_audit_20260619.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
POSTHOC_SCRIPT=${ROOT}/ops/run_latentfm_corrected_dsloss_posthoc_after_finish_20260619.sh
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
BASE_INIT=${COUPLED}/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
RUN_NAME=scf_prior010_inject_visitcap8_power05_floor32_dsloss05_corrected_4k
SESSION=lfm_${RUN_NAME}
POSTHOC_SESSION=latentfm_corrected_dsloss_posthoc_20260619

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

if [[ ! -f "${AUDIT_JSON}" ]]; then
  echo "Metric gate missing: ${AUDIT_JSON}" >&2
  exit 2
fi

gate_status="$("${PYTHON}" - "${AUDIT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("status", ""))
PY
)"

if [[ "${gate_status}" != "pass" ]]; then
  echo "Metric gate is not pass: ${AUDIT_JSON} status=${gate_status}" >&2
  exit 3
fi

if [[ -e "${OUT_ROOT}/${RUN_NAME}" && "${FORCE_CORRECTED_DSLOSS_RERUN:-0}" != "1" ]]; then
  echo "Output exists: ${OUT_ROOT}/${RUN_NAME}; set FORCE_CORRECTED_DSLOSS_RERUN=1 to override" >&2
  exit 6
fi

echo "[$(date '+%F %T %Z')] exact GPU status before corrected dsloss launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

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

if [[ -z "${gpu}" ]]; then
  echo "No GPU selected by helper; see ${gpu_json}" >&2
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
min_mem = float(os.environ.get("MIN_LAUNCH_MEM_AVAILABLE_GIB", "64"))
max_load = float(os.environ.get("MAX_LAUNCH_LOAD1_PER_CPU", "2.0"))
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

run_script="${RUN_ROOT}/run_${RUN_NAME}.sh"
cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}

cd ${COUPLED}
source ${ROOT}/init-scdfm.sh >/dev/null

${PYTHON} -m model.latent.train \\
  --data-dir ${DATA_DIR} \\
  --biflow-dir ${BIFLOW_DIR} \\
  --latent-backbone scfoundation \\
  --emb-dim 3072 \\
  --save-dir ${OUT_ROOT}/${RUN_NAME} \\
  --log-file train.log \\
  --model-type control_mlp \\
  --init-checkpoint ${BASE_INIT} \\
  --use-pert-condition \\
  --pert-gene-emb-cache-dir ${GENE_CACHE} \\
  --pert-condition-embedding-source scgpt_embed_gene \\
  --pert-pool-aggregations mean max min \\
  --pert-pool-scale-init 1.0 1.0 1.0 \\
  --pert-pool-fusion-mode sum \\
  --pert-type-adapter-mode scalar \\
  --pert-chem-projector-hidden 1024 \\
  --pert-gene-projector-hidden 1024 \\
  --pert-to-c-init-mode xavier_small \\
  --use-pert-in-fusion \\
  --condition-delta-head-use-in-model \\
  --batch-size 64 \\
  --grad-accum-steps 1 \\
  --min-cells 32 \\
  --ds-alpha 0.7 \\
  --ds-loss-alpha 0.5 \\
  --ds-loss-warmup-start 0 \\
  --min-selected-conditions-per-dataset 32 \\
  --condition-visit-power 0.5 \\
  --condition-visit-cap 8 \\
  --total-steps 4000 \\
  --lr 0.0001 \\
  --warmup-steps 300 \\
  --lr-decay-steps 4000 \\
  --eval-max-conditions 256 \\
  --eval-max-conditions-per-dataset 12 \\
  --eval-max-mmd-cells 512 \\
  --eval-max-chunk 128 \\
  --selection-metric pearson_pert_minus_mmd \\
  --selection-mmd-lambda 0.5 \\
  --endpoint-delta-loss-weight 2.0 \\
  --endpoint-delta-loss-warmup-start 0 \\
  --endpoint-delta-loss-warmup-end 1000 \\
  --composition-delta-loss-weight 0.0 \\
  --condition-prior-delta-loss-weight 0.10 \\
  --condition-prior-delta-loss-warmup-start 0 \\
  --condition-prior-delta-loss-warmup-end 1000 \\
  --condition-prior-delta-loss-every 1 \\
  --condition-prior-bank-max-cells 512 \\
  --condition-prior-num-genes 2 \\
  --ot-method torch_sinkhorn \\
  --ot-sinkhorn-iter 50 \\
  --use-amp \\
  --amp-dtype bf16
EOF
chmod +x "${run_script}"

tmux new -d -s "${SESSION}" \
  "bash -lc 'bash ${run_script} > ${LOG_ROOT}/${RUN_NAME}.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/${RUN_NAME}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/${RUN_NAME}.FINISHED; exit \$rc'"

cat > "${RUN_ROOT}/RUN_STATUS_${RUN_NAME}.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
tmux new -d -s ${SESSION} \\
  "bash -lc 'bash ${run_script} > ${LOG_ROOT}/${RUN_NAME}.log 2>&1; rc=\\\$?; echo \\\$rc > ${RUN_ROOT}/${RUN_NAME}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/${RUN_NAME}.FINISHED; exit \\\$rc'"
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`${SESSION}\`

GPU: physical GPU${gpu}; selected after exact \`nvidia-smi\` and 3-sample helper audit.

## Log path

\`${LOG_ROOT}/${RUN_NAME}.log\`

## Expected outputs

* \`${OUT_ROOT}/${RUN_NAME}/best.pt\`
* \`${OUT_ROOT}/${RUN_NAME}/latest.pt\`
* \`${OUT_ROOT}/${RUN_NAME}/config.json\`

## How to check manually

\`\`\`bash
tmux has-session -t ${SESSION} && echo running || echo not-running
cat ${RUN_ROOT}/${RUN_NAME}.EXIT_CODE 2>/dev/null || echo still-running
tail -n 50 ${LOG_ROOT}/${RUN_NAME}.log
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Corrected dataset-loss smoke. Unlike the active old \`dsloss05\` run, this
command explicitly sets \`--ds-loss-warmup-start 0\` after code commit
\`c943d75\`.
EOF

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_corrected_dsloss_smoke_20260619

Started: $(date '+%F %T %Z')
Status: started
Metric gate: ${AUDIT_JSON} (${gate_status})
Selected physical GPU: ${gpu}
Run: ${RUN_NAME}
Log: ${LOG_ROOT}/${RUN_NAME}.log
GPU audit:
- ${RUN_ROOT}/logs/gpu_launch_audit.log
- ${gpu_json}
Resource audit:
- ${resource_audit}

Long-task policy: use 30-minute cadence for checks; do not tail continuously.
EOF

tmux new -d -s "${POSTHOC_SESSION}" \
  "bash -lc 'bash ${POSTHOC_SCRIPT} > ${RUN_ROOT}/logs/posthoc_launcher.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/POSTHOC_LAUNCHER_EXIT_CODE; exit \$rc'"

echo "Launched corrected dsloss smoke ${RUN_NAME} on physical GPU${gpu}"
