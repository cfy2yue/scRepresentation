#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_pairwise_condition_adapter_20260621
OUT_ROOT=${COUPLED}/output/latentfm_runs/pairwise_condition_adapter_20260621
LOG_ROOT=${ROOT}/logs/latentfm_pairwise_condition_adapter_20260621
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
POSTHOC_SCRIPT=${ROOT}/ops/run_latentfm_pairwise_condition_posthoc_20260621.sh
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=scf_pair_hadamard_condadapter_prior010_e2_4k
SESSION=lfm_${RUN_NAME}
POSTHOC_SESSION=latentfm_pairwise_condadapter_posthoc_20260621

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/scripts" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${DATA_DIR}/manifest.json" \
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${POSTHOC_SCRIPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ "${FORCE_PAIRWISE_CONDADAPTER_RERUN:-0}" != "1" && -e "${OUT_ROOT}/${RUN_NAME}" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_PAIRWISE_CONDADAPTER_RERUN=1 to relaunch" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] exact GPU status before pairwise-condition-adapter launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-jobs-per-gpu 4 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
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
audit = {
    "status": "pass",
    "stable_light_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "active_user_gpus": sorted(active_user),
    "chosen_gpu": chosen,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if chosen is None:
    reasons.append("no GPU slot available under leave-one-empty and max-4-physical rules")
if float(system.get("mem_available_gib") or 0.0) < 96.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 96.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

GPU="$("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["chosen_gpu"])
PY
)"

run_script="${RUN_ROOT}/scripts/run_${RUN_NAME}.sh"
cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${GPU}
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
  --split-file ${CANONICAL_SPLIT} \\
  --latent-backbone scfoundation \\
  --emb-dim 3072 \\
  --save-dir ${OUT_ROOT}/${RUN_NAME} \\
  --log-file train.log \\
  --model-type control_mlp \\
  --init-checkpoint ${ANCHOR_CKPT} \\
  --finetune-trainable-scope pairwise_condition_adapter \\
  --batch-size 64 \\
  --grad-accum-steps 1 \\
  --min-cells 32 \\
  --scale-noise 0.02 \\
  --ds-alpha 0.7 \\
  --ds-loss-alpha 0.0 \\
  --total-steps 4000 \\
  --lr 0.0001 \\
  --warmup-steps 300 \\
  --lr-decay-steps 4000 \\
  --print-every 200 \\
  --eval-every 2000 \\
  --eval-max-conditions 256 \\
  --eval-max-conditions-per-dataset 12 \\
  --eval-max-mse-cells 1024 \\
  --eval-max-mmd-cells 512 \\
  --eval-max-chunk 128 \\
  --selection-metric pearson_pert_minus_mmd \\
  --selection-mmd-lambda 0.5 \\
  --ot-method torch_sinkhorn \\
  --ot-sinkhorn-reg 0.05 \\
  --ot-sinkhorn-iter 50 \\
  --ot-threads 4 \\
  --prefetch 4 \\
  --n-ot-workers 4 \\
  --use-mmd \\
  --gamma 0.03 \\
  --gamma-warmup-start 50000 \\
  --gamma-warmup-end 100000 \\
  --mmd-every 1 \\
  --mmd-estimator unbiased \\
  --endpoint-delta-loss-weight 2.0 \\
  --endpoint-delta-loss-warmup-start 0 \\
  --endpoint-delta-loss-warmup-end 1000 \\
  --condition-prior-delta-loss-weight 0.10 \\
  --condition-prior-delta-loss-warmup-start 0 \\
  --condition-prior-delta-loss-warmup-end 1000 \\
  --condition-prior-delta-loss-every 1 \\
  --condition-prior-bank-max-cells 512 \\
  --condition-prior-num-genes 2 \\
  --condition-delta-head-use-in-model \\
  --use-ema \\
  --ema-update-after 1000 \\
  --ema-decay 0.999 \\
  --use-amp \\
  --amp-dtype bf16 \\
  --use-pert-condition \\
  --pert-gene-emb-cache-dir ${GENE_CACHE} \\
  --pert-condition-embedding-source scgpt_embed_gene \\
  --pert-pool-aggregations mean max min \\
  --pert-pool-scale-init 1.0 1.0 1.0 \\
  --pert-pool-fusion-mode sum \\
  --pert-type-adapter-mode scalar \\
  --pert-pairwise-mode hadamard_mean \\
  --pert-gene-projector-hidden 1024 \\
  --pert-chem-projector-hidden 1024 \\
  --pert-to-c-init-mode xavier_small \\
  --use-pert-in-fusion \\
  --patience 6
EOF
chmod +x "${run_script}"

rm -f "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE" "${RUN_ROOT}/${RUN_NAME}.FINISHED"
tmux new -d -s "${SESSION}" \
  "bash -lc 'bash ${run_script} > ${LOG_ROOT}/${RUN_NAME}.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/${RUN_NAME}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/${RUN_NAME}.FINISHED; exit \$rc'"
date '+%F %T %Z' > "${RUN_ROOT}/${RUN_NAME}.STARTED"

tmux new -d -s "${POSTHOC_SESSION}" \
  "bash -lc 'LATENTFM_PAIRWISE_RUN_ROOT=${RUN_ROOT} LATENTFM_PAIRWISE_OUT_ROOT=${OUT_ROOT} LATENTFM_PAIRWISE_RUN_NAME=${RUN_NAME} LATENTFM_PAIRWISE_SUMMARY_JSON=${ROOT}/reports/latentfm_pairwise_condition_adapter_smoke_summary_20260621.json LATENTFM_PAIRWISE_SUMMARY_MD=${ROOT}/reports/LATENTFM_PAIRWISE_CONDITION_ADAPTER_SMOKE_SUMMARY_20260621.md LATENTFM_PAIRWISE_BOOTSTRAP_DIR=${ROOT}/reports/latentfm_pairwise_condition_adapter_smoke_bootstrap_20260621 LATENTFM_PAIRWISE_POSTHOC_TITLE=latentfm_pairwise_condition_adapter_posthoc_20260621 bash ${POSTHOC_SCRIPT}'"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_pairwise_condition_adapter_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_pairwise_condition_adapter_smoke_20260621.sh
\`\`\`

## Runtime classification

Long LatentFM training task. Use 30-minute cadence for checks.

## Start time

$(cat "${RUN_ROOT}/${RUN_NAME}.STARTED")

## tmux / GPU

* \`${RUN_NAME}\`: \`${SESSION}\`, physical GPU${GPU}
* posthoc watcher: \`${POSTHOC_SESSION}\`

## Log path

\`${LOG_ROOT}/${RUN_NAME}.log\`

## Expected outputs

* \`${OUT_ROOT}/${RUN_NAME}/best.pt\`
* \`${OUT_ROOT}/${RUN_NAME}/latest.pt\`
* \`${OUT_ROOT}/${RUN_NAME}/config.json\`
* \`${ROOT}/reports/LATENTFM_PAIRWISE_CONDITION_ADAPTER_SMOKE_SUMMARY_20260621.md\`
* \`${ROOT}/reports/latentfm_pairwise_condition_adapter_smoke_bootstrap_20260621/bootstrap_index.json\`

## How to check manually

\`\`\`bash
tmux ls | grep '${SESSION}' || true
tail -n 50 ${LOG_ROOT}/${RUN_NAME}.log
cat ${RUN_ROOT}/${RUN_NAME}.EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started training and posthoc watcher.

## Notes

This is the anchor-preserving pairwise-condition adapter smoke. It warm-starts
from \`scf_prior010_inject_e2_4k\`, enables \`hadamard_mean\` pairwise condition
features, and freezes the base flow plus condition-delta head. Only
\`pert_encoder.pair_to_out.*\` and \`condition_delta_to_c.*\` are trainable under
\`--finetune-trainable-scope pairwise_condition_adapter\` for the current
scFoundation dimensions.

Gate: it must beat the matched canonical-refinetune control on multi-unseen pp
without MMD increase and must remove the anchor-relative single/drug/MMD harms
seen in full pairwise and adapter-only smokes before any uncapped promotion.
EOF

echo "Launched ${RUN_NAME} on physical GPU${GPU}"
echo "RUN_STATUS: ${RUN_ROOT}/RUN_STATUS.md"
