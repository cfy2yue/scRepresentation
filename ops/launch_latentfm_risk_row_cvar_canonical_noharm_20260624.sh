#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_RISK_ROW_CANONICAL_ACK:-}" != "frozen_internal_pass_noharm_only" ]]; then
  cat >&2 <<'EOF'
Refusing to launch risk-row canonical no-harm.

Set:
  LATENTFM_RISK_ROW_CANONICAL_ACK=frozen_internal_pass_noharm_only

Boundary:
  - exactly one frozen risk-row latest.pt
  - canonical split post-freeze no-harm only
  - no canonical multi evaluation in this gate
  - no Track C query
EOF
  exit 4
fi

RUN_NAME=xverse_risk_row_cvar_allrisk_w020_2k_seed42
RUN_ROOT=${ROOT}/runs/latentfm_risk_row_cvar_canonical_noharm_20260624
SOURCE_RUN_DIR=${ROOT}/runs/latentfm_risk_row_cvar_trainonly_20260624/${RUN_NAME}
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
LOG_ROOT=${ROOT}/logs/latentfm_risk_row_cvar_canonical_noharm_20260624/${RUN_NAME}
EVAL_DIR=${RUN_DIR}/posthoc_eval_canonical
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
ANCHOR_SPLIT_JSON=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json
ANCHOR_FAMILY_JSON=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json
CANDIDATE_CKPT=${COUPLED}/output/latentfm_runs/risk_row_cvar_trainonly_20260624/${RUN_NAME}/latest.pt
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
AUDIT_SCRIPT=${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py
SUMMARIZER=${ROOT}/ops/summarize_latentfm_risk_row_cvar_canonical_noharm_20260624.py
INTERNAL_JSON=${ROOT}/reports/latentfm_risk_row_cvar_internal_posthoc_decision_20260624.json
LEIBNIZ_REPORT=${ROOT}/reports/LATENTFM_RISK_ROW_CVAR_EXTERNAL_AUDIT_LEIBNIZ_20260624.md
SESSION=lfm_riskrow_canonical_noharm_20260624

for required in \
  "${SOURCE_RUN_DIR}/RUN_STATUS.md" \
  "${INTERNAL_JSON}" \
  "${LEIBNIZ_REPORT}" \
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_SPLIT_JSON}" \
  "${ANCHOR_FAMILY_JSON}" \
  "${CANDIDATE_CKPT}" \
  "${GPU_HELPER}" \
  "${AUDIT_SCRIPT}" \
  "${SUMMARIZER}"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

"${PYTHON}" - "${INTERNAL_JSON}" <<'PY'
import json, sys
from pathlib import Path
j=json.loads(Path(sys.argv[1]).read_text())
if j.get("status") != "risk_row_cvar_internal_posthoc_pass_no_promotion":
    raise SystemExit(f"internal posthoc not pass: {j.get('status')!r}")
PY

if [[ -e "${RUN_DIR}" || -e "${LOG_ROOT}" ]]; then
  echo "Canonical run dir/log dir already exists; refusing rerun: ${RUN_NAME}" >&2
  exit 3
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${LOG_ROOT}" "${EVAL_DIR}"

echo "[$(date '+%F %T %Z')] exact GPU/CPU/RAM status before risk-row canonical no-harm" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 15 | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json=${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 1 \
  --need 1 \
  --json-only > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

GPU=$("${PYTHON}" - "${gpu_json}" <<'PY'
import json, sys
from pathlib import Path
j=json.loads(Path(sys.argv[1]).read_text())
suggested=j.get("suggested_job_gpus") or []
system=j.get("system") or {}
if not suggested:
    raise SystemExit("no GPU suggested")
if float(system.get("mem_available_gib") or 0) < 128:
    raise SystemExit("low RAM")
print(int(suggested[0]))
PY
)

script=${RUN_DIR}/scripts/posthoc_${RUN_NAME}.sh
cat > "${script}" <<EOF
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
mkdir -p ${EVAL_DIR}
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${CANDIDATE_CKPT} --groups test_single --out ${EVAL_DIR}/split_group_eval_candidate_ode20_canonical.json "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${CANDIDATE_CKPT} --groups family_gene family_drug test_single --out ${EVAL_DIR}/condition_family_eval_candidate_ode20_canonical.json "\${common[@]}"
${PYTHON} ${AUDIT_SCRIPT} --anchor-split-json ${ANCHOR_SPLIT_JSON} --anchor-family-json ${ANCHOR_FAMILY_JSON} --candidate-split-json ${EVAL_DIR}/split_group_eval_candidate_ode20_canonical.json --candidate-family-json ${EVAL_DIR}/condition_family_eval_candidate_ode20_canonical.json --n-boot 2000 --seed 42 --out-json ${EVAL_DIR}/single_background_candidate_gate.json --out-md ${EVAL_DIR}/SINGLE_BACKGROUND_CANDIDATE_GATE.md
${PYTHON} ${SUMMARIZER}
EOF
chmod +x "${script}"

cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: risk-row CVaR canonical no-harm ${RUN_NAME}

## Command

\`\`\`bash
LATENTFM_RISK_ROW_CANONICAL_ACK=frozen_internal_pass_noharm_only bash ${ROOT}/ops/launch_latentfm_risk_row_cvar_canonical_noharm_20260624.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation. Use 30-minute cadence for checks.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`; physical GPU: \`${GPU}\`

## Log path

\`${LOG_ROOT}/posthoc.log\`

## Expected outputs

* \`${EVAL_DIR}/single_background_candidate_gate.json\`
* \`${ROOT}/reports/LATENTFM_RISK_ROW_CVAR_CANONICAL_NOHARM_DECISION_20260624.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_ROOT}/posthoc.log
cat ${RUN_DIR}/POSTHOC_EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Frozen candidate checkpoint: \`${CANDIDATE_CKPT}\`.
- Canonical split is post-freeze no-harm only.
- Canonical multi is not evaluated in this gate.
- Track C query is not read.
- Gate pass requires \`single_background_candidate_gate.json\` status exactly
  \`candidate_gate_pass\`; otherwise close this recipe for promotion.
EOF

date '+%F %T %Z' > "${RUN_DIR}/POSTHOC_STARTED"
tmux new -d -s "${SESSION}" "bash -lc 'bash ${script} > ${LOG_ROOT}/posthoc.log 2>&1; rc=\$?; echo \$rc > ${RUN_DIR}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${RUN_DIR}/POSTHOC_FINISHED; exit \$rc'"
tmux ls
tail -n 30 "${LOG_ROOT}/posthoc.log" || true
