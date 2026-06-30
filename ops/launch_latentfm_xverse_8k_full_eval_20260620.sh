#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${RUN_ROOT:-${ROOT}/runs/latentfm_xverse_8k_full_eval_20260620}
OUT_ROOT=${OUT_ROOT:-${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620}
LOG_ROOT=${LOG_ROOT:-${ROOT}/logs/latentfm_xverse_8k_full_eval_20260620}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
VALIDATE_JSON=${ROOT}/reports/xverse_full_de5000_bundle_validation_20260620.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=${RUN_NAME:-xverse_comp006_endpoint5_8k_seed42_fulleval}
SESSION=${SESSION:-lfm_${RUN_NAME}}
TOTAL_STEPS=${TOTAL_STEPS:-8000}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-42}

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}"
rm -f "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE" "${RUN_ROOT}/${RUN_NAME}.FINISHED"

validation_ok="$("${PYTHON}" - "${VALIDATE_JSON}" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("1" if p.get("ok") is True and (p.get("summary") or {}).get("emb_dim") == 384 else "0")
PY
)"
if [[ "${validation_ok}" != "1" ]]; then
  echo "xverse bundle validation is not ok/emb_dim=384: ${VALIDATE_JSON}" >&2
  exit 2
fi

echo "[$(date '+%F %T %Z')] exact GPU status before xverse 8k full-eval launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --need 1 \
  --max-jobs-per-gpu 4 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

resource_audit="${RUN_ROOT}/logs/resource_audit_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${resource_audit}" <<'PY'
import json, os, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
stable = [g for g in gpus.values() if g.get("stable_light")]
stable_count = len(stable)
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
physical_budget = min(4, stable_count) if stable_count >= 5 else max(0, min(4, stable_count - 1))
chosen = None
for idx in [int(x) for x in payload.get("candidate_order", [])]:
    gpu = gpus[idx]
    if not gpu.get("available"):
        continue
    if len(active_user | {idx}) <= physical_budget and int(gpu.get("colocation_slots_free", 0)) > 0:
        chosen = idx
        break
system = payload.get("system") or {}
mem = float(system.get("mem_available_gib") or 0.0)
load = float(system.get("load1_per_cpu") or 0.0)
audit = {
    "status": "pass",
    "stable_light_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "active_user_gpus": sorted(active_user),
    "chosen_gpu": chosen,
    "min_mem_available_gib": 64.0,
    "max_load1_per_cpu": 2.0,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if chosen is None:
    reasons.append("no GPU slot available under leave-one-empty and max-4-physical rules")
if mem < 64.0:
    reasons.append(f"MemAvailable {mem:.1f} GiB < 64.0 GiB")
if load > 2.0:
    reasons.append(f"load1_per_cpu {load:.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

gpu="$("${PYTHON}" - "${resource_audit}" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["chosen_gpu"])
PY
)"

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
  --emb-dim 384 \\
  --gpu 0 \\
  --batch-size ${BATCH_SIZE} \\
  --seed ${SEED} \\
  --grad-accum-steps 1 \\
  --min-cells 16 \\
  --scale-noise 0.01 \\
  --lr 1e-4 \\
  --weight-decay 1e-4 \\
  --warmup-steps 300 \\
  --total-steps ${TOTAL_STEPS} \\
  --lr-decay-steps ${TOTAL_STEPS} \\
  --print-every 200 \\
  --eval-max-conditions 0 \\
  --eval-max-conditions-per-dataset 0 \\
  --eval-max-mse-cells 2048 \\
  --eval-max-mmd-cells 2048 \\
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
# Run Status: latentfm_xverse_8k_full_eval_20260620

## Command

\`\`\`bash
RUN_ROOT=${RUN_ROOT} OUT_ROOT=${OUT_ROOT} LOG_ROOT=${LOG_ROOT} RUN_NAME=${RUN_NAME} SEED=${SEED} bash ${ROOT}/ops/launch_latentfm_xverse_8k_full_eval_20260620.sh
\`\`\`

## Runtime classification

Long LatentFM training task. Use 30-minute cadence for checks.

## Start time

$(cat "${RUN_ROOT}/${RUN_NAME}.STARTED")

## tmux

\`${SESSION}\`

Selected physical GPU: ${gpu}

## Log path

\`${LOG_ROOT}/${RUN_NAME}.log\`

## Expected outputs

* \`${OUT_ROOT}/${RUN_NAME}/best.pt\`
* \`${OUT_ROOT}/${RUN_NAME}/latest.pt\`
* \`${OUT_ROOT}/${RUN_NAME}/config.json\`
* \`${OUT_ROOT}/${RUN_NAME}/iid_eval_results.json\`

## Current status

Started.
EOF

echo "Launched ${RUN_NAME} on physical GPU${gpu} (tmux ${SESSION})"
