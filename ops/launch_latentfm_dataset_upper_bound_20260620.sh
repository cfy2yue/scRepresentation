#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_dataset_upper_bound_20260620
OUT_ROOT=${COUPLED}/output/latentfm_runs/dataset_upper_bound_20260620
LOG_ROOT=${ROOT}/logs/latentfm_dataset_upper_bound_20260620
DECISION_JSON=${ROOT}/reports/latentfm_focus_next_action_decision_20260619.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
POSTHOC_SCRIPT=${ROOT}/ops/run_latentfm_dataset_upper_bound_posthoc_20260620.sh
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
BASE_INIT=${COUPLED}/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
POSTHOC_SESSION=latentfm_dataset_upper_bound_posthoc_20260620

RUN_NORMAN=scf_prior010_upperbound_norman_4k
RUN_WESSELS=scf_prior010_upperbound_wessels_4k
RUN_GASPERINI=scf_prior010_upperbound_gasperini_4k

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in "${DATA_DIR}" "${CANONICAL_SPLIT}" "${BASE_INIT}" "${GENE_CACHE}/manifest.json" "${DECISION_JSON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

next_action="$("${PYTHON}" - "${DECISION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("next_action", ""))
PY
)"

case "${next_action}" in
  run_dataset_upper_bound_diagnostics|run_dataset_upper_bound_before_all_split_balance)
    ;;
  *)
    echo "Focus decision does not authorize dataset upper-bound launch: ${next_action}" >&2
    echo "Decision JSON: ${DECISION_JSON}" >&2
    exit 3
    ;;
esac

for run in "${RUN_NORMAN}" "${RUN_WESSELS}" "${RUN_GASPERINI}"; do
  if [[ -e "${OUT_ROOT}/${run}" && "${FORCE_DATASET_UPPER_BOUND_RERUN:-0}" != "1" ]]; then
    echo "Dataset upper-bound output exists for ${run}; set FORCE_DATASET_UPPER_BOUND_RERUN=1 to override" >&2
    exit 6
  fi
done

"${PYTHON}" - "${CANONICAL_SPLIT}" "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
run_root = Path(sys.argv[2])
datasets = {
    "norman": "NormanWeissman2019_filtered",
    "wessels": "Wessels",
    "gasperini": "GasperiniShendure2019_lowMOI",
}
payload = json.loads(src.read_text(encoding="utf-8"))
summary = {}
for key, dataset in datasets.items():
    if dataset not in payload:
        raise SystemExit(f"missing dataset in split: {dataset}")
    subset = {dataset: payload[dataset]}
    out = run_root / f"latentfm_upperbound_{key}_split_seed42_20260620.json"
    out.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    summary[key] = {
        "dataset": dataset,
        "split_file": str(out),
        "counts": {name: len(values) for name, values in payload[dataset].items() if isinstance(values, list)},
    }
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "[$(date '+%F %T %Z')] exact GPU status before dataset upper-bound launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
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

write_run_script() {
  local run_name="$1"
  local split_file="$2"
  local run_script="${RUN_ROOT}/run_${run_name}.sh"
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
  --split-file ${split_file} \\
  --latent-backbone scfoundation \\
  --emb-dim 3072 \\
  --save-dir ${OUT_ROOT}/${run_name} \\
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
  --ds-loss-alpha 0.0 \\
  --ds-loss-warmup-start 0 \\
  --min-selected-conditions-per-dataset 0 \\
  --condition-visit-power 1.0 \\
  --condition-visit-cap 0 \\
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
}

write_status() {
  local run_name="$1"
  local session="$2"
  local split_file="$3"
  cat > "${RUN_ROOT}/RUN_STATUS_${run_name}.md" <<EOF
# Run Status: ${run_name}

## Command

\`\`\`bash
tmux new -d -s ${session} \\
  "bash -lc 'bash ${RUN_ROOT}/run_${run_name}.sh > ${LOG_ROOT}/${run_name}.log 2>&1; rc=\\\$?; echo \\\$rc > ${RUN_ROOT}/${run_name}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/${run_name}.FINISHED; exit \\\$rc'"
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`${session}\`

GPU: physical GPU${gpu}; selected after exact \`nvidia-smi\` and 3-sample helper audit.

## Log path

\`${LOG_ROOT}/${run_name}.log\`

## Expected outputs

* \`${OUT_ROOT}/${run_name}/best.pt\`
* \`${OUT_ROOT}/${run_name}/latest.pt\`
* \`${OUT_ROOT}/${run_name}/config.json\`

## How to check manually

\`\`\`bash
tmux has-session -t ${session} && echo running || echo not-running
cat ${RUN_ROOT}/${run_name}.EXIT_CODE 2>/dev/null || echo still-running
tail -n 50 ${LOG_ROOT}/${run_name}.log
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Single-dataset upper-bound diagnostic using split:
\`${split_file}\`.
Long-task checks should use 30-minute cadence.
EOF
}

launch_one() {
  local run_name="$1"
  local split_file="$2"
  local session="lfm_${run_name}"
  write_run_script "${run_name}" "${split_file}"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${RUN_ROOT}/run_${run_name}.sh > ${LOG_ROOT}/${run_name}.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/${run_name}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/${run_name}.FINISHED; exit \$rc'"
  write_status "${run_name}" "${session}" "${split_file}"
}

launch_one "${RUN_NORMAN}" "${RUN_ROOT}/latentfm_upperbound_norman_split_seed42_20260620.json"
launch_one "${RUN_WESSELS}" "${RUN_ROOT}/latentfm_upperbound_wessels_split_seed42_20260620.json"
launch_one "${RUN_GASPERINI}" "${RUN_ROOT}/latentfm_upperbound_gasperini_split_seed42_20260620.json"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_dataset_upper_bound_20260620

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_dataset_upper_bound_20260620.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

* \`lfm_${RUN_NORMAN}\`
* \`lfm_${RUN_WESSELS}\`
* \`lfm_${RUN_GASPERINI}\`
* posthoc watcher: \`${POSTHOC_SESSION}\`

## Log path

* \`${LOG_ROOT}/${RUN_NORMAN}.log\`
* \`${LOG_ROOT}/${RUN_WESSELS}.log\`
* \`${LOG_ROOT}/${RUN_GASPERINI}.log\`
* \`${RUN_ROOT}/logs/posthoc_launcher.log\`

## Expected outputs

* \`${OUT_ROOT}/${RUN_NORMAN}/best.pt\`
* \`${OUT_ROOT}/${RUN_WESSELS}/best.pt\`
* \`${OUT_ROOT}/${RUN_GASPERINI}/best.pt\`
* \`${ROOT}/reports/LATENTFM_DATASET_UPPER_BOUND_STABLECAPS_SUMMARY_20260620.md\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/${RUN_NORMAN}.EXIT_CODE 2>/dev/null || echo ${RUN_NORMAN}: still-running
cat ${RUN_ROOT}/${RUN_WESSELS}.EXIT_CODE 2>/dev/null || echo ${RUN_WESSELS}: still-running
cat ${RUN_ROOT}/${RUN_GASPERINI}.EXIT_CODE 2>/dev/null || echo ${RUN_GASPERINI}: still-running
cat ${ROOT}/runs/latentfm_dataset_upper_bound_posthoc_20260620/EXIT_CODE 2>/dev/null || echo posthoc: still-running
tmux ls
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Authorized by \`${DECISION_JSON}\` with next_action=${next_action}.
Three FlowMatching single-dataset diagnostics were colocated on one physical GPU
because AGENTS permits up to 3 LatentFM training jobs/GPU when CPU/RAM/I/O are
safe. Do not poll more often than every 30 minutes.
EOF

tmux new -d -s "${POSTHOC_SESSION}" \
  "bash -lc 'bash ${POSTHOC_SCRIPT} > ${RUN_ROOT}/logs/posthoc_launcher.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/POSTHOC_LAUNCHER_EXIT_CODE; exit \$rc'"

echo "Launched dataset upper-bound diagnostics on physical GPU${gpu}: ${RUN_NORMAN}, ${RUN_WESSELS}, ${RUN_GASPERINI}"
tmux ls | grep -E 'lfm_scf_prior010_upperbound|latentfm_dataset_upper_bound_posthoc_20260620' || true
