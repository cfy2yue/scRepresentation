#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=latentfm_crosslatent_stack_baseline_repair_20260622
RUN_ROOT=${ROOT}/runs/${RUN_NAME}
LOG_ROOT=${RUN_ROOT}/logs
SESSION=${RUN_NAME}
SCRIPT=${ROOT}/ops/repair_latentfm_crosslatent_stack_baseline_20260622.py
CPU_THREADS=${LATENTFM_CPU_THREADS:-32}

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${PYTHON}" \
  "${SCRIPT}" \
  "${ROOT}/ops/audit_latentfm_xverse_gene_reliability_router_gate_20260622.py" \
  "${ROOT}/runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622/artifacts/stack_trainonly_pert_means_split_seed42_crossbgval_v2.npz" \
  "${ROOT}/reports/latentfm_crosslatent_tracka_trainonly_baselines_20260622.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

cat > "${LOG_ROOT}/resource_audit.log" <<EOF
[$(date '+%F %T %Z')] CPU/RAM/disk audit before stack baseline repair

free -h:
$(free -h)

df -h /data/cyx/1030/scLatent:
$(df -h /data/cyx/1030/scLatent)

load:
$(uptime)

CPU thread cap:
${CPU_THREADS}
EOF

date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_crosslatent_stack_baseline_repair_20260622.sh
\`\`\`

## Runtime classification

Long CPU repair gate. Similar baseline gates can approach or exceed 10 minutes,
so this is detached even though it may finish quickly.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

## Log path

\`${LOG_ROOT}/run.log\`

## Expected outputs

* \`${ROOT}/reports/latentfm_crosslatent_stack_gene_reliability_router_gate_20260622.json\`
* \`${ROOT}/reports/LATENTFM_CROSSLATENT_STACK_GENE_RELIABILITY_ROUTER_GATE_20260622.md\`
* \`${ROOT}/reports/latentfm_crosslatent_tracka_trainonly_baselines_20260622.json\`
* \`${ROOT}/reports/LATENTFM_CROSSLATENT_TRACKA_TRAINONLY_BASELINES_20260622.md\`
* \`${RUN_ROOT}/EXIT_CODE\`
* \`${RUN_ROOT}/FINISHED\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_ROOT}/run.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
free -h
\`\`\`

## Current status

Started.

## Notes

Hypothesis: the failed stack row was an artifact-schema bug, not a scientific
baseline gate failure. The stack bundle has \`condition_metadata.json\`, but
its manifest lacks \`condition_metadata_file\`. The gate script now falls back
to \`<data-dir>/condition_metadata.json\` and records metadata provenance.

Promotion rule: success only restores the prerequisite baseline artifacts for
protocol review. It does not authorize GPU comparator launch by itself.
EOF

tmux new -d -s "${SESSION}" \
  "bash -lc 'source ${ROOT}/init-scdfm.sh >/dev/null 2>&1 || true; export OMP_NUM_THREADS=${CPU_THREADS} MKL_NUM_THREADS=${CPU_THREADS} OPENBLAS_NUM_THREADS=${CPU_THREADS} NUMEXPR_NUM_THREADS=${CPU_THREADS} BLIS_NUM_THREADS=${CPU_THREADS} CUDA_VISIBLE_DEVICES=\"\"; cd ${ROOT}; ${PYTHON} ${SCRIPT} > ${LOG_ROOT}/run.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/FINISHED; exit \$rc'"

tmux ls | tee "${LOG_ROOT}/tmux_ls_after_launch.txt"
sleep 2
tail -n 20 "${LOG_ROOT}/run.log" || true
