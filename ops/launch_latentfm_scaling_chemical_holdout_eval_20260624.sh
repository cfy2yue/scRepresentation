#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_CHEM_HOLDOUT_EVAL_ACK:-}" != "trainonly_chemical_eval_no_training" ]]; then
  cat >&2 <<'EOF'
Refusing to launch chemical holdout eval.

Set:
  LATENTFM_CHEM_HOLDOUT_EVAL_ACK=trainonly_chemical_eval_no_training

Boundary:
  - eval-only, no training
  - train-only SciPlex holdout split
  - no canonical multi or Track C query
EOF
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_scaling_chemical_holdout_eval_20260624
LOG_ROOT=${ROOT}/logs/latentfm_scaling_chemical_holdout_eval_20260624
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
SPLIT_FILE=${BIFLOW_DIR}/xverse_scaling_chemical_holdout_splits_20260624/split_seed42_xverse_scaling_cap120_chemical_holdout_v1.json
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
CAP30_CKPT=${COUPLED}/output/latentfm_runs/xverse_scaling_count_smokes_20260624/xverse_scaling_cap30_all_3k_seed42/best.pt
CAP120_CKPT=${COUPLED}/output/latentfm_runs/xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/best.pt
CAP30_PERT=${ROOT}/runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/xverse_trainonly_scaling_cap30_all_v2_pert_means.npz
CAP120_PERT=${ROOT}/runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/xverse_trainonly_scaling_cap120_all_v2_pert_means.npz
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
SUMMARY=${ROOT}/ops/summarize_latentfm_scaling_chemical_holdout_eval_20260624.py
SESSION=lfm_scaling_chemical_holdout_eval_20260624

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}" "${ROOT}/reports"

for required in "${SPLIT_FILE}" "${ANCHOR_CKPT}" "${CAP30_CKPT}" "${CAP120_CKPT}" "${CAP30_PERT}" "${CAP120_PERT}" "${GPU_HELPER}" "${SUMMARY}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

AUDIT_LOG=${RUN_ROOT}/logs/gpu_launch_audit.log
echo "[$(date '+%F %T %Z')] exact GPU status before chemical holdout eval" | tee "${AUDIT_LOG}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${AUDIT_LOG}"
free -h | tee -a "${AUDIT_LOG}"
df -h "${ROOT}" | tee -a "${AUDIT_LOG}"
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 30 | tee -a "${AUDIT_LOG}"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 1 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

GPU_ID=$("${PYTHON}" - "${gpu_json}" <<'PY'
import json, sys
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text())
suggested=payload.get("suggested_job_gpus") or []
if not suggested:
    raise SystemExit("no GPU suggested")
print(int(suggested[0]))
PY
)

run_script=${RUN_ROOT}/run_eval.sh
cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene

eval_pair() {
  local arm="\$1"
  local ckpt="\$2"
  local pert="\$3"
  local out_dir=${RUN_ROOT}/"\${arm}"
  mkdir -p "\${out_dir}"
  local common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${SPLIT_FILE} --gpu 0 --ode-steps 20 --max-chunk 256 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 512 --eval-max-mmd-cells 512 --pert-means-file "\${pert}")
  ${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_drug type_drug test_single --out "\${out_dir}/condition_family_eval_anchor_chemical_ode20.json" "\${common[@]}"
  ${PYTHON} -m model.latent.eval_condition_families --checkpoint "\${ckpt}" --groups test_all family_drug type_drug test_single --out "\${out_dir}/condition_family_eval_candidate_chemical_ode20.json" "\${common[@]}"
}

eval_pair cap30 ${CAP30_CKPT} ${CAP30_PERT}
eval_pair cap120 ${CAP120_CKPT} ${CAP120_PERT}
${PYTHON} ${SUMMARY}
EOF
chmod +x "${run_script}"

date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
tmux new -d -s "${SESSION}" "bash -lc 'bash ${run_script} > ${LOG_ROOT}/run.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/FINISHED; exit \$rc'"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_scaling_chemical_holdout_eval_20260624

## Hypothesis

Evaluate whether existing cap30/cap120 scaling checkpoints improve a train-only
SciPlex chemical holdout relative to the xverse anchor. This is a diagnostic
gate for the drug/dose semantic branch; it is not a descriptor-cache training
claim.

## Command

\`\`\`bash
LATENTFM_CHEM_HOLDOUT_EVAL_ACK=trainonly_chemical_eval_no_training bash ${ROOT}/ops/launch_latentfm_scaling_chemical_holdout_eval_20260624.sh
\`\`\`

## Runtime classification

Long/unknown GPU eval task. Detached to tmux; do not poll more frequently than
30 minutes unless there is evidence of a crash.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

Physical GPU: ${GPU_ID}

## Log path

\`${LOG_ROOT}/run.log\`

## Expected outputs

* \`${RUN_ROOT}/cap30/condition_family_eval_candidate_chemical_ode20.json\`
* \`${RUN_ROOT}/cap120/condition_family_eval_candidate_chemical_ode20.json\`
* \`${ROOT}/reports/LATENTFM_SCALING_CHEMICAL_HOLDOUT_EVAL_GATE_20260624.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_ROOT}/run.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Eval-only; no training.
- Split: \`${SPLIT_FILE}\`
- Canonical reference drugs are excluded by the split protocol.
- Canonical multi and Track C query are not used.
- Stop rule: if chemical pp gain, cap120-vs-cap30 gain, MMD, or dataset-tail
  gates fail, do not launch descriptor-cache smoke from this evidence.
EOF

echo "Launched ${SESSION} on GPU ${GPU_ID}"
tmux ls || true
tail -n 20 "${LOG_ROOT}/run.log" || true
