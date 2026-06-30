#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_ACK:-}" != "route_frozen_noharm_veto" ]]; then
  cat >&2 <<'EOF'
Refusing to launch budget128 6k canonical no-harm.

Set:
  LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_ACK=route_frozen_noharm_veto

Boundary:
  - route must be frozen before launch
  - evaluates all seeds 42/43/44
  - canonical single/family no-harm veto only
  - canonical multi is not evaluated or selected
  - held-out Track C query is not read
EOF
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625
TRAIN_OUT_ROOT=${COUPLED}/output/latentfm_runs/true_cell_count_budget128_tail_stability_6k_20260625
LOG_ROOT=${ROOT}/logs/latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
AUDIT_SCRIPT=${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py
SUMMARIZER=${ROOT}/ops/summarize_latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625.py
ROUTE_FREEZE=${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_ROUTE_FREEZE_20260625.md
INTERNAL_JSON=${ROOT}/reports/latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json
ARTIFACT_CONTROL_JSON=${ROOT}/reports/latentfm_true_cell_count_budget128_tail_stability_6k_artifact_control_20260625.json
ROUTE_SAFETY_JSON=${ROOT}/reports/latentfm_true_cell_count_budget128_6k_route_safety_controls_20260625.json

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${CANONICAL_SPLIT}" \
  "${DATA_DIR}/condition_metadata.json" \
  "${GPU_HELPER}" \
  "${AUDIT_SCRIPT}" \
  "${SUMMARIZER}" \
  "${ROUTE_FREEZE}" \
  "${INTERNAL_JSON}" \
  "${ARTIFACT_CONTROL_JSON}" \
  "${ROUTE_SAFETY_JSON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - "${INTERNAL_JSON}" "${ARTIFACT_CONTROL_JSON}" "${ROUTE_SAFETY_JSON}" <<'PY'
import json, sys
from pathlib import Path
internal, artifact, route = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
bad = []
if internal.get("status") != "nested_matrix_internal_pass":
    bad.append(f"internal status {internal.get('status')!r}")
if artifact.get("status") != "budget128_tail_stability_artifact_control_pass_no_gpu":
    bad.append(f"artifact status {artifact.get('status')!r}")
if route.get("status") not in {
    "budget128_6k_route_safety_controls_pass_no_gpu",
    "budget128_6k_route_safety_controls_pass_with_warnings_no_gpu",
}:
    bad.append(f"route safety status {route.get('status')!r}")
if bad:
    raise SystemExit("; ".join(bad))
PY

run_names=(
  xverse_truecell_nested_budget128_tailstable_seed42_6000
  xverse_truecell_nested_budget128_tailstable_seed43_6000
  xverse_truecell_nested_budget128_tailstable_seed44_6000
)

for run_name in "${run_names[@]}"; do
  ckpt=${TRAIN_OUT_ROOT}/${run_name}/best.pt
  if [[ ! -e "${ckpt}" ]]; then
    echo "Missing frozen checkpoint for ${run_name}: ${ckpt}" >&2
    exit 2
  fi
  run_dir=${RUN_ROOT}/${run_name}
  session=lfm_truecell_canon_${run_name}
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

echo "[$(date '+%F %T %Z')] exact GPU status before budget128 6k canonical no-harm launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 30 | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 4 \
  --need 3 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

mapfile -t ASSIGNED_GPUS < <("${PYTHON}" - "${gpu_json}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
slots = [int(x) for x in payload.get("suggested_job_gpus", [])]
if len(slots) < 3:
    raise SystemExit(f"need 3 GPU slots, got {slots}")
print("\n".join(map(str, slots[:3])))
PY
)

for i in "${!run_names[@]}"; do
  run_name=${run_names[$i]}
  gpu=${ASSIGNED_GPUS[$i]}
  run_dir=${RUN_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  session=lfm_truecell_canon_${run_name}
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
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ckpt} --groups family_gene test_single --out "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" "\${common[@]}"
${PYTHON} ${AUDIT_SCRIPT} --candidate-split-json "\${eval_dir}/split_group_eval_candidate_ode20_canonical.json" --candidate-family-json "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" --n-boot 2000 --seed 42 --out-json "\${eval_dir}/single_background_candidate_gate.json" --out-md "\${eval_dir}/SINGLE_BACKGROUND_CANDIDATE_GATE.md"
LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_RUN_ROOT=${RUN_ROOT} ${PYTHON} ${SUMMARIZER}
EOF
  chmod +x "${script}"
  rm -f "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  date '+%F %T %Z' > "${run_dir}/POSTHOC_STARTED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${script} > ${log_dir}/posthoc.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$rc'"

  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: true-cell-count budget128 6k canonical no-harm ${run_name}

## Command

\`\`\`bash
LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_ACK=route_frozen_noharm_veto bash ${ROOT}/ops/launch_latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625.sh
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
* \`${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md\`

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

- Frozen route: \`${ROUTE_FREEZE}\`
- Evaluates canonical \`test_single\` and \`family_gene\` no-harm only.
- Canonical multi is not evaluated or used for selection.
- Held-out Track C query is not read.
- All seeds 42/43/44 must pass; canonical results may not be used to choose a seed.
EOF
  echo "Launched budget128 6k canonical no-harm for ${run_name} on GPU ${gpu} in tmux ${session}"
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625

## Command

\`\`\`bash
LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_ACK=route_frozen_noharm_veto bash ${ROOT}/ops/launch_latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625.sh
\`\`\`

## Runtime classification

Long GPU posthoc batch. Each child run has its own RUN_STATUS.md.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

$(printf '* `%s`\n' "${run_names[@]/#/lfm_truecell_canon_}")

## Log path

\`${LOG_ROOT}/<run_name>/posthoc.log\`

## Expected outputs

* \`${RUN_ROOT}/<run_name>/posthoc_eval_canonical/single_background_candidate_gate.json\`
* \`${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md\`

## How to check manually

\`\`\`bash
tmux ls
cat ${RUN_ROOT}/*/POSTHOC_EXIT_CODE 2>/dev/null || true
nvidia-smi
\`\`\`

## Current status

Started 3 frozen canonical no-harm posthoc jobs.

## Notes

- Route freeze: \`${ROUTE_FREEZE}\`
- Canonical multi and Track C query are not read.
- This is a no-harm veto only, not checkpoint selection or promotion.
EOF

echo "Started 3 budget128 6k canonical no-harm posthoc jobs."
