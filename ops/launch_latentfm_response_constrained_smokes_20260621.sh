#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
BASE_RUN_ROOT=${LATENTFM_RESPONSE_BASE_RUN_ROOT:-${ROOT}/runs/latentfm_response_constrained_20260621}
OUT_ROOT=${LATENTFM_RESPONSE_OUT_ROOT:-${COUPLED}/output/latentfm_runs/response_constrained_20260621}
LOG_ROOT=${LATENTFM_RESPONSE_LOG_ROOT:-${ROOT}/logs/latentfm_response_constrained_20260621}
RUN_TAG=$(basename "${BASE_RUN_ROOT}")
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
ARTIFACT=${ROOT}/runs/latentfm_response_normalization_20260621/artifacts/scfoundation_trainonly_dataset_scale_pca32.npz
RESPONSE_GEOMETRY_FILTER=${LATENTFM_RESPONSE_GEOMETRY_FILTER:-all}
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
POSTHOC_SCRIPT=${ROOT}/ops/run_latentfm_response_geometry_posthoc_20260621.sh
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${BASE_RUN_ROOT}/logs" "${BASE_RUN_ROOT}/scripts" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${DATA_DIR}/manifest.json" \
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${ARTIFACT}" \
  "${GPU_HELPER}" \
  "${POSTHOC_SCRIPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

RUN_NAMES=()
WEIGHTS=()
if [[ -n "${LATENTFM_RESPONSE_CONSTRAINED_SPECS:-}" ]]; then
  for spec in ${LATENTFM_RESPONSE_CONSTRAINED_SPECS}; do
    run_name="${spec%%:*}"
    weight="${spec#*:}"
    if [[ -z "${run_name}" || -z "${weight}" || "${run_name}" == "${weight}" ]]; then
      echo "Invalid LATENTFM_RESPONSE_CONSTRAINED_SPECS entry: ${spec}" >&2
      exit 2
    fi
    RUN_NAMES+=("${run_name}")
    WEIGHTS+=("${weight}")
  done
else
  RUN_NAMES=(
    scf_response_dataset_scale_pca32_aux025_4k
    scf_response_dataset_scale_pca32_aux05_4k
  )
  WEIGHTS=(0.25 0.5)
fi

if [[ "${FORCE_RESPONSE_CONSTRAINED_RERUN:-0}" != "1" ]]; then
  for run_name in "${RUN_NAMES[@]}"; do
    if [[ -e "${OUT_ROOT}/${run_name}" ]]; then
      echo "Output exists for ${run_name}; set FORCE_RESPONSE_CONSTRAINED_RERUN=1 to relaunch" >&2
      exit 3
    fi
  done
fi

echo "[$(date '+%F %T %Z')] exact GPU status before response-constrained launch" | tee "${BASE_RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${BASE_RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${BASE_RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-jobs-per-gpu 4 \
  --need "${#RUN_NAMES[@]}" \
  --json-only \
  > "${gpu_json}" 2> "${BASE_RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${BASE_RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" "${#RUN_NAMES[@]}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
need = int(sys.argv[3])
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
stable = [g for g in gpus.values() if g.get("stable_light")]
stable_count = len(stable)
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
physical_budget = min(4, stable_count) if stable_count >= 5 else max(0, min(4, stable_count - 1))
candidate_order = [int(x) for x in payload.get("candidate_order", [])]
assigned: list[int] = []
assigned_counts: dict[int, int] = {}
for _ in range(need):
    chosen = None
    for idx in candidate_order:
        gpu = gpus[idx]
        if not gpu.get("available"):
            continue
        slots = int(gpu.get("colocation_slots_free", 0)) - assigned_counts.get(idx, 0)
        if slots <= 0:
            continue
        would_use = active_user | set(assigned) | {idx}
        if len(would_use) <= physical_budget:
            chosen = idx
            break
    if chosen is None:
        break
    assigned.append(chosen)
    assigned_counts[chosen] = assigned_counts.get(chosen, 0) + 1
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "stable_light_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "active_user_gpus": sorted(active_user),
    "assigned_gpus": assigned,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if len(assigned) < need:
    reasons.append(f"assigned {len(assigned)} jobs < requested {need}")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

mapfile -t GPUS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
for gpu in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["assigned_gpus"]:
    print(gpu)
PY
)

for i in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$i]}"
  weight="${WEIGHTS[$i]}"
  gpu="${GPUS[$i]}"
  run_root="${BASE_RUN_ROOT}/${run_name}"
  log_root="${LOG_ROOT}/${run_name}"
  session="lfm_${run_name}"
  posthoc_session="latentfm_${RUN_TAG}_${run_name}_posthoc"
  mkdir -p "${run_root}/logs" "${run_root}/scripts" "${log_root}"

  run_script="${run_root}/scripts/run_${run_name}.sh"
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
  --split-file ${CANONICAL_SPLIT} \\
  --latent-backbone scfoundation \\
  --emb-dim 3072 \\
  --save-dir ${OUT_ROOT}/${run_name} \\
  --log-file train.log \\
  --model-type control_mlp \\
  --init-checkpoint ${ANCHOR_CKPT} \\
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
  --response-geometry-loss-weight ${weight} \\
  --response-geometry-loss-warmup-start 0 \\
  --response-geometry-loss-warmup-end 1000 \\
  --response-normalization-mode dataset_scale_pca \\
  --response-normalization-artifact ${ARTIFACT} \\
  --response-geometry-condition-filter ${RESPONSE_GEOMETRY_FILTER} \\
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
  --pert-gene-projector-hidden 1024 \\
  --pert-chem-projector-hidden 1024 \\
  --pert-to-c-init-mode xavier_small \\
  --use-pert-in-fusion \\
  --patience 6
EOF
  chmod +x "${run_script}"

  rm -f "${run_root}/${run_name}.EXIT_CODE" "${run_root}/${run_name}.FINISHED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${run_script} > ${log_root}/${run_name}.log 2>&1; rc=\$?; echo \$rc > ${run_root}/${run_name}.EXIT_CODE; date \"+%F %T %Z\" > ${run_root}/${run_name}.FINISHED; exit \$rc'"
  date '+%F %T %Z' > "${run_root}/${run_name}.STARTED"

  tmux new -d -s "${posthoc_session}" \
    "bash -lc 'LATENTFM_RESPONSE_RUN_ROOT=${run_root} LATENTFM_RESPONSE_OUT_ROOT=${OUT_ROOT} LATENTFM_RESPONSE_RUN_NAME=${run_name} LATENTFM_RESPONSE_ARTIFACT=${ARTIFACT} LATENTFM_RESPONSE_SUMMARY_JSON=${ROOT}/reports/latentfm_response_constrained_${run_name}_summary_20260621.json LATENTFM_RESPONSE_SUMMARY_MD=${ROOT}/reports/LATENTFM_RESPONSE_CONSTRAINED_${run_name}_SUMMARY_20260621.md LATENTFM_RESPONSE_BOOTSTRAP_DIR=${ROOT}/reports/latentfm_response_constrained_${run_name}_bootstrap_20260621 LATENTFM_RESPONSE_POSTHOC_TITLE=latentfm_response_constrained_${run_name}_posthoc_20260621 bash ${POSTHOC_SCRIPT}'"

  cat > "${run_root}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_response_constrained_20260621/${run_name}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_response_constrained_smokes_20260621.sh
\`\`\`

## Runtime classification

Long LatentFM training task. Use 30-minute cadence for checks.

## Start time

$(cat "${run_root}/${run_name}.STARTED")

## tmux / GPU

* \`${run_name}\`: \`${session}\`, physical GPU${gpu}
* posthoc watcher: \`${posthoc_session}\`

## Log path

\`${log_root}/${run_name}.log\`

## Expected outputs

* \`${OUT_ROOT}/${run_name}/best.pt\`
* \`${OUT_ROOT}/${run_name}/latest.pt\`
* \`${OUT_ROOT}/${run_name}/config.json\`
* \`${ROOT}/reports/LATENTFM_RESPONSE_CONSTRAINED_${run_name}_SUMMARY_20260621.md\`
* \`${ROOT}/reports/latentfm_response_constrained_${run_name}_bootstrap_20260621/bootstrap_index.json\`

## Current status

Started training and posthoc watcher.

## Notes

Constrained response geometry smoke with \`response_geometry_loss_weight=${weight}\`.
Response geometry condition filter: \`${RESPONSE_GEOMETRY_FILTER}\`.
Artifact is train-only:
\`${ARTIFACT}\`.
EOF
done

cat > "${BASE_RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_response_constrained_20260621

Launched response-constrained smokes at $(date '+%F %T %Z').

GPU assignment audit:
\`${assignment_json}\`

Runs:

$(for j in "${!RUN_NAMES[@]}"; do
  printf '* `%s`: weight `%s`, RUN_STATUS\n  `%s/%s/RUN_STATUS.md`\n' "${RUN_NAMES[$j]}" "${WEIGHTS[$j]}" "${BASE_RUN_ROOT}" "${RUN_NAMES[$j]}"
done)

Use 30-minute cadence for checks unless exit markers or summaries appear.
EOF

echo "Launched response constrained smokes"
echo "RUN_STATUS: ${BASE_RUN_ROOT}/RUN_STATUS.md"
