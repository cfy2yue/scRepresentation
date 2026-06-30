#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_SCALING_HT_CANONICAL_ACK:-}" != "internal_pass_frozen_noharm" ]]; then
  cat >&2 <<'EOF'
Refusing to launch high-throughput scaling canonical no-harm.

Set:
  LATENTFM_SCALING_HT_CANONICAL_ACK=internal_pass_frozen_noharm

Boundary:
  - only frozen candidates that passed train-only internal high-throughput gate
  - no canonical multi selection/evaluation
  - no held-out Track C query
EOF
  exit 4
fi

RUN_ROOT=${LATENTFM_SCALING_HT_CANONICAL_RUN_ROOT:-${ROOT}/runs/latentfm_scaling_highthroughput_canonical_noharm_20260624}
TRAIN_OUT_ROOT=${LATENTFM_SCALING_HT_TRAIN_OUT_ROOT:-${COUPLED}/output/latentfm_runs/scaling_highthroughput_smokes_20260624}
LOG_ROOT=${LATENTFM_SCALING_HT_CANONICAL_LOG_ROOT:-${ROOT}/logs/latentfm_scaling_highthroughput_canonical_noharm_20260624}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
INTERNAL_JSON=${LATENTFM_SCALING_HT_INTERNAL_JSON:-${ROOT}/reports/latentfm_scaling_highthroughput_smokes_decision_20260624.json}
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
AUDIT_SCRIPT=${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py
SUMMARIZER=${ROOT}/ops/summarize_latentfm_scaling_highthroughput_canonical_noharm_20260624.py
CANONICAL_DECISION_JSON=${LATENTFM_SCALING_HT_CANONICAL_DECISION_JSON:-${ROOT}/reports/latentfm_scaling_highthroughput_canonical_noharm_decision_20260624.json}
CANONICAL_DECISION_MD=${LATENTFM_SCALING_HT_CANONICAL_DECISION_MD:-${ROOT}/reports/LATENTFM_SCALING_HIGH_THROUGHPUT_CANONICAL_NOHARM_DECISION_20260624.md}

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${CANONICAL_SPLIT}" \
  "${INTERNAL_JSON}" \
  "${GPU_HELPER}" \
  "${AUDIT_SCRIPT}" \
  "${SUMMARIZER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

mapfile -t RUN_NAMES < <("${PYTHON}" - "${INTERNAL_JSON}" <<'PY'
import json
import sys
from pathlib import Path
obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name in (obj.get("decision") or {}).get("passed") or []:
    print(str(name))
PY
)

need=${#RUN_NAMES[@]}
if (( need <= 0 )); then
  echo "No internal-passed high-throughput scaling candidates to evaluate." >&2
  exit 2
fi

for run_name in "${RUN_NAMES[@]}"; do
  ckpt=${TRAIN_OUT_ROOT}/${run_name}/best.pt
  if [[ ! -e "${ckpt}" ]]; then
    echo "Missing frozen checkpoint for ${run_name}: ${ckpt}" >&2
    exit 2
  fi
  run_dir=${RUN_ROOT}/${run_name}
  session=lfm_scaling_ht_canon_${run_name}
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

echo "[$(date '+%F %T %Z')] exact GPU status before high-throughput canonical no-harm launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 4 \
  --need "${need}" \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" "${need}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
need = int(sys.argv[3])
suggested = [int(x) for x in payload.get("suggested_job_gpus", [])]
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "need": need,
    "assigned_gpus": suggested[:need],
    "allowed_physical_user_gpus": payload.get("allowed_physical_user_gpus"),
    "active_user_gpus": payload.get("active_user_gpus"),
    "max_user_gpus": payload.get("max_user_gpus"),
    "max_jobs_per_gpu": payload.get("max_jobs_per_gpu"),
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if len(suggested) < need:
    reasons.append(f"only {len(suggested)} GPU job slots suggested for need={need}")
if int(payload.get("max_user_gpus") or 0) > 4:
    reasons.append("max_user_gpus exceeds hard cap 4")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

mapfile -t ASSIGNED_GPUS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for gpu in payload["assigned_gpus"]:
    print(int(gpu))
PY
)

for i in "${!RUN_NAMES[@]}"; do
  run_name=${RUN_NAMES[$i]}
  gpu=${ASSIGNED_GPUS[$i]}
  run_dir=${RUN_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  session=lfm_scaling_ht_canon_${run_name}
  ckpt=${TRAIN_OUT_ROOT}/${run_name}/best.pt
  script=${run_dir}/scripts/posthoc_${run_name}.sh

  cat > "${script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
eval_dir=${run_dir}/posthoc_eval_canonical
mkdir -p "\${eval_dir}"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ckpt} --groups test_single --out "\${eval_dir}/split_group_eval_candidate_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ckpt} --groups family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" "\${common[@]}"
${PYTHON} ${AUDIT_SCRIPT} --candidate-split-json "\${eval_dir}/split_group_eval_candidate_ode20_canonical.json" --candidate-family-json "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" --n-boot 2000 --seed 42 --out-json "\${eval_dir}/single_background_candidate_gate.json" --out-md "\${eval_dir}/SINGLE_BACKGROUND_CANDIDATE_GATE.md"
LATENTFM_SCALING_HT_CANONICAL_RUN_ROOT=${RUN_ROOT} LATENTFM_SCALING_HT_INTERNAL_JSON=${INTERNAL_JSON} LATENTFM_SCALING_HT_CANONICAL_DECISION_JSON=${CANONICAL_DECISION_JSON} LATENTFM_SCALING_HT_CANONICAL_DECISION_MD=${CANONICAL_DECISION_MD} ${PYTHON} ${SUMMARIZER}
EOF
  chmod +x "${script}"
  rm -f "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  date '+%F %T %Z' > "${run_dir}/POSTHOC_STARTED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${script} > ${log_dir}/posthoc.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$rc'"

  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: high-throughput scaling canonical no-harm ${run_name}

## Command

\`\`\`bash
LATENTFM_SCALING_HT_CANONICAL_ACK=internal_pass_frozen_noharm bash ${ROOT}/ops/launch_latentfm_scaling_highthroughput_canonical_noharm_20260624.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation. Use 30-minute cadence for result checks.

## Start time

$(cat "${run_dir}/POSTHOC_STARTED")

## PID / tmux / scheduler ID

tmux session: \`${session}\`

Physical GPU: ${gpu}

## Log path

\`${log_dir}/posthoc.log\`

## Expected outputs

* \`${run_dir}/posthoc_eval_canonical/single_background_candidate_gate.json\`
* \`${run_dir}/posthoc_eval_canonical/SINGLE_BACKGROUND_CANDIDATE_GATE.md\`
* \`${CANONICAL_DECISION_MD}\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${log_dir}/posthoc.log
cat ${run_dir}/POSTHOC_EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Frozen candidate from train-only internal high-throughput gate.
- Canonical multi is neither selected nor evaluated here.
- Held-out Track C query is not read.
EOF
  echo "Launched canonical no-harm for ${run_name} on GPU ${gpu} in tmux ${session}"
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_scaling_highthroughput_canonical_noharm_20260624

## Command

\`\`\`bash
LATENTFM_SCALING_HT_CANONICAL_ACK=internal_pass_frozen_noharm bash ${ROOT}/ops/launch_latentfm_scaling_highthroughput_canonical_noharm_20260624.sh
\`\`\`

## Runtime classification

Long GPU posthoc batch. Each child run has its own RUN_STATUS.md.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

$(printf '* `%s`\n' "${RUN_NAMES[@]/#/lfm_scaling_ht_canon_}")

## Log path

\`${LOG_ROOT}/<run_name>/posthoc.log\`

## Expected outputs

* \`${RUN_ROOT}/<run_name>/posthoc_eval_canonical/single_background_candidate_gate.json\`
* \`${CANONICAL_DECISION_MD}\`

## How to check manually

\`\`\`bash
tmux ls
cat ${RUN_ROOT}/*/POSTHOC_EXIT_CODE 2>/dev/null || true
nvidia-smi
\`\`\`

## Current status

Started ${need} high-throughput scaling canonical no-harm posthoc jobs.

## Notes

- Evaluates only candidates listed as passed in:
  \`${INTERNAL_JSON}\`
- Canonical metrics are no-harm decision evidence only, not checkpoint selection.
EOF
