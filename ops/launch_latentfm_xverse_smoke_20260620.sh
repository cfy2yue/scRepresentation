#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_xverse_smoke_20260620
OUT_ROOT=${COUPLED}/output/latentfm_runs/xverse_smoke_20260620
LOG_ROOT=${ROOT}/logs/latentfm_xverse_smoke_20260620
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
VALIDATE_JSON=${ROOT}/reports/xverse_full_de5000_bundle_validation_20260620.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=${RUN_NAME:-xverse_comp006_endpoint5_2k_smoke}
SESSION=${SESSION:-lfm_${RUN_NAME}}
TOTAL_STEPS=${TOTAL_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}"
rm -f "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE" "${RUN_ROOT}/${RUN_NAME}.FINISHED"

if [[ ! -s "${VALIDATE_JSON}" ]]; then
  echo "Missing xverse validation report: ${VALIDATE_JSON}" >&2
  exit 2
fi

validation_ok="$("${PYTHON}" - "${VALIDATE_JSON}" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("1" if p.get("ok") is True else "0")
PY
)"
if [[ "${validation_ok}" != "1" ]]; then
  echo "xverse bundle validation is not ok: ${VALIDATE_JSON}" >&2
  exit 3
fi

if [[ ! -s "${DATA_DIR}/manifest.json" ]]; then
  echo "Missing xverse bundle manifest: ${DATA_DIR}/manifest.json" >&2
  exit 4
fi

EMB_DIM="$("${PYTHON}" - "${DATA_DIR}/manifest.json" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(p.get("emb_dim", ""))
PY
)"
if [[ "${EMB_DIM}" != "384" ]]; then
  echo "Unexpected xverse emb_dim=${EMB_DIM}; expected 384" >&2
  exit 5
fi

echo "[$(date '+%F %T %Z')] exact GPU status before xverse LatentFM smoke" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
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
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = payload.get("suggested_job_gpus") or []
print(chosen[0] if chosen else "")
PY
)"
if [[ -z "${gpu}" ]]; then
  echo "No GPU selected by helper; see ${gpu_json}" >&2
  exit 6
fi

resource_audit="${RUN_ROOT}/logs/resource_audit_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${resource_audit}" <<'PY'
import json, os, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
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
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if mem < min_mem:
    reasons.append(f"MemAvailable {mem:.1f} GiB < {min_mem:.1f} GiB")
if load > max_load:
    reasons.append(f"load1_per_cpu {load:.3f} > {max_load:.3f}")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 7)
PY

run_script="${RUN_ROOT}/run_${RUN_NAME}.sh"
cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}

export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene

${PYTHON} -m model.latent.train \\
  --data-dir ${DATA_DIR} \\
  --biflow-dir ${BIFLOW_DIR} \\
  --save-dir ${OUT_ROOT}/${RUN_NAME} \\
  --log-file train.log \\
  --latent-backbone xverse \\
  --model-type control_mlp \\
  --emb-dim ${EMB_DIM} \\
  --gpu 0 \\
  --batch-size ${BATCH_SIZE} \\
  --grad-accum-steps 1 \\
  --min-cells 16 \\
  --scale-noise 0.01 \\
  --lr 1e-4 \\
  --weight-decay 1e-4 \\
  --warmup-steps 300 \\
  --total-steps ${TOTAL_STEPS} \\
  --lr-decay-steps ${TOTAL_STEPS} \\
  --print-every 100 \\
  --eval-max-conditions 256 \\
  --eval-max-conditions-per-dataset 12 \\
  --eval-max-mse-cells 1024 \\
  --eval-max-mmd-cells 1024 \\
  --eval-max-chunk 256 \\
  --selection-metric test_mmd \\
  --ot-method torch_sinkhorn \\
  --ot-sinkhorn-reg 0.05 \\
  --ot-sinkhorn-iter 30 \\
  --use-mmd \\
  --gamma 0.03 \\
  --gamma-warmup-start 500 \\
  --gamma-warmup-end 1500 \\
  --mmd-every 4 \\
  --mmd-estimator unbiased \\
  --composition-delta-loss-weight 0.06 \\
  --composition-delta-loss-warmup-start 500 \\
  --composition-delta-loss-warmup-end 1500 \\
  --endpoint-delta-loss-weight 5.0 \\
  --endpoint-delta-loss-warmup-start 500 \\
  --endpoint-delta-loss-warmup-end 1500 \\
  --use-ema \\
  --ema-update-after 500 \\
  --ema-decay 0.999 \\
  --amp-dtype bf16 \\
  --use-pert-condition \\
  --pert-gene-emb-cache-dir ${GENE_CACHE} \\
  --pert-condition-embedding-source scgpt_embed_gene \\
  --pert-pool-aggregations sum mean max min \\
  --pert-pool-scale-init 0.5 1.0 1.0 1.0 \\
  --pert-pool-fusion-mode sum \\
  --pert-gene-projector-hidden 1024 \\
  --pert-chem-enabled \\
  --pert-chem-emb-dim 512 \\
  --pert-chem-projector-hidden 1024 \\
  --chem-fallback-embed-dim 512 \\
  --pert-to-c-init-mode xavier_small \\
  --use-pert-in-fusion \\
  --patience 4
EOF
chmod +x "${run_script}"

tmux new -d -s "${SESSION}" \
  "bash -lc 'bash ${run_script} > ${LOG_ROOT}/${RUN_NAME}.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/${RUN_NAME}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/${RUN_NAME}.FINISHED; exit \$rc'"

date '+%F %T %Z' > "${RUN_ROOT}/${RUN_NAME}.STARTED"
cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_smoke_20260620

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_xverse_smoke_20260620.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(cat "${RUN_ROOT}/${RUN_NAME}.STARTED")

## PID / tmux / scheduler ID

tmux: \`${SESSION}\`

Selected physical GPU: ${gpu}

## Log path

\`${LOG_ROOT}/${RUN_NAME}.log\`

## Expected outputs

* \`${OUT_ROOT}/${RUN_NAME}/best.pt\`
* \`${OUT_ROOT}/${RUN_NAME}/latest.pt\`
* \`${OUT_ROOT}/${RUN_NAME}/config.json\`
* \`${OUT_ROOT}/${RUN_NAME}/iid_eval_results.json\`

## How to check manually

\`\`\`bash
bash ${ROOT}/ops/check_latentfm_xverse_smoke_once_20260620.sh
\`\`\`

## Current status

Started.

## Notes

This is a short controlled xverse pipeline smoke after bundle validation, not a
promotion-grade latent comparison. Use 30-minute cadence for checks.
EOF

echo "Launched ${RUN_NAME} on physical GPU${gpu} (tmux ${SESSION})"
