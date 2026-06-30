#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_stronger_balance_after_focus_20260619
OUT_ROOT=${COUPLED}/output/latentfm_runs/stronger_balance_after_focus_20260619
LOG_ROOT=${ROOT}/logs/latentfm_stronger_balance_after_focus_20260619
DECISION_JSON=${ROOT}/reports/latentfm_focus_next_action_decision_20260619.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
POSTHOC_SCRIPT=${ROOT}/ops/run_latentfm_stronger_balance_posthoc_20260619.sh
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
BASE_INIT=${COUPLED}/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
POSTHOC_SESSION=latentfm_stronger_balance_posthoc_20260619

RUN_A=scf_prior010_balance_power025_cap4_floor32_4k
RUN_B=scf_prior010_balance_power025_cap4_floor32_dsloss05_4k

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

if [[ "${next_action}" != "launch_stronger_all_split_balance_4k" ]]; then
  echo "Focus decision does not authorize stronger balance launch: ${next_action}" >&2
  echo "Decision JSON: ${DECISION_JSON}" >&2
  exit 3
fi

if [[ -e "${OUT_ROOT}/${RUN_A}" || -e "${OUT_ROOT}/${RUN_B}" ]]; then
  if [[ "${FORCE_STRONGER_BALANCE_RERUN:-0}" != "1" ]]; then
    echo "Stronger-balance output exists under ${OUT_ROOT}; set FORCE_STRONGER_BALANCE_RERUN=1 to override" >&2
    exit 6
  fi
fi

echo "[$(date '+%F %T %Z')] exact GPU status before stronger-balance launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
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
  local ds_loss_alpha="$2"
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
  --ds-loss-alpha ${ds_loss_alpha} \\
  --ds-loss-warmup-start 0 \\
  --min-selected-conditions-per-dataset 32 \\
  --condition-visit-power 0.25 \\
  --condition-visit-cap 4 \\
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

Launched only after focus decision authorized stronger all-split balance.
Long-task checks should use 30-minute cadence.
EOF
}

launch_one() {
  local run_name="$1"
  local session="lfm_${run_name}"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${RUN_ROOT}/run_${run_name}.sh > ${LOG_ROOT}/${run_name}.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/${run_name}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/${run_name}.FINISHED; exit \$rc'"
  write_status "${run_name}" "${session}"
}

write_run_script "${RUN_A}" 0.0
write_run_script "${RUN_B}" 0.5

launch_one "${RUN_A}"
launch_one "${RUN_B}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_stronger_balance_after_focus_20260619

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_stronger_balance_after_focus_20260619.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

* \`lfm_${RUN_A}\`
* \`lfm_${RUN_B}\`
* posthoc watcher: \`${POSTHOC_SESSION}\`

## Log path

* \`${LOG_ROOT}/${RUN_A}.log\`
* \`${LOG_ROOT}/${RUN_B}.log\`
* \`${RUN_ROOT}/logs/posthoc_launcher.log\`

## Expected outputs

* \`${OUT_ROOT}/${RUN_A}/best.pt\`
* \`${OUT_ROOT}/${RUN_B}/best.pt\`
* \`${ROOT}/reports/LATENTFM_STRONGER_BALANCE_STABLECAPS_SUMMARY_20260619.md\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/${RUN_A}.EXIT_CODE 2>/dev/null || echo ${RUN_A}: still-running
cat ${RUN_ROOT}/${RUN_B}.EXIT_CODE 2>/dev/null || echo ${RUN_B}: still-running
cat ${ROOT}/runs/latentfm_stronger_balance_posthoc_20260619/EXIT_CODE 2>/dev/null || echo posthoc: still-running
tmux ls
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Authorized by \`${DECISION_JSON}\` with next_action=${next_action}.
Two FlowMatching full-split balance diagnostics were colocated on one physical GPU because AGENTS permits up to 3 training jobs/GPU when CPU/RAM/I/O are safe.
Do not poll more often than every 30 minutes.
EOF

tmux new -d -s "${POSTHOC_SESSION}" \
  "bash -lc 'bash ${POSTHOC_SCRIPT} > ${RUN_ROOT}/logs/posthoc_launcher.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/POSTHOC_LAUNCHER_EXIT_CODE; exit \$rc'"

echo "Launched stronger balance diagnostics ${RUN_A}, ${RUN_B} on physical GPU${gpu}"
