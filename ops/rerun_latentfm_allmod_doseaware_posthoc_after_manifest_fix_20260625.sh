#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625
LOG_ROOT=${ROOT}/logs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625
SUMMARIZER=${ROOT}/ops/summarize_latentfm_true_cell_count_allmodality_doseaware_smokes_20260625.py
SESSION=lfm_allmod_doseaware_posthoc_manifestfix_20260625
STATUS=${RUN_ROOT}/POSTHOC_MANIFEST_FIX_RUN_STATUS.md
GPU_ID=${LATENTFM_ALLMOD_POSTHOC_GPU:-5}

if [[ "${LATENTFM_ALLMOD_POSTHOC_MANIFEST_FIX_ACK:-}" != "rerun_allmod_posthoc_after_manifest_fix" ]]; then
  cat >&2 <<'EOF'
Refusing to rerun all-modality dose-aware posthoc.

Set:
  LATENTFM_ALLMOD_POSTHOC_MANIFEST_FIX_ACK=rerun_allmod_posthoc_after_manifest_fix

Boundary:
  - reruns posthoc eval only after eval manifest-conditions compatibility fix
  - no training
  - no canonical multi or Track C query
  - refreshes train-only/internal all-modality smoke decision
EOF
  exit 4
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}/posthoc_manifestfix"

cat > "${STATUS}" <<EOF
# Run Status: allmod_doseaware_posthoc_manifestfix_20260625

## Command

\`\`\`bash
LATENTFM_ALLMOD_POSTHOC_MANIFEST_FIX_ACK=rerun_allmod_posthoc_after_manifest_fix LATENTFM_ALLMOD_POSTHOC_GPU=${GPU_ID} bash ${ROOT}/ops/rerun_latentfm_allmod_doseaware_posthoc_after_manifest_fix_20260625.sh
\`\`\`

## Runtime classification

Unknown runtime task, treated as long because it performs GPU posthoc inference across four completed checkpoints.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

## Log path

\`${LOG_ROOT}/posthoc_manifestfix/rerun.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_SMOKE_DECISION_20260625.md\`
* \`${ROOT}/reports/latentfm_true_cell_count_allmodality_doseaware_smoke_decision_20260625.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_ROOT}/posthoc_manifestfix/rerun.log
cat ${RUN_ROOT}/POSTHOC_MANIFEST_FIX_EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Posthoc-only rerun after eval manifest loader was patched to populate missing
\`conditions\` from materialized H5 files.
EOF

worker=${RUN_ROOT}/logs/posthoc_manifestfix_worker.sh
cat > "${worker}" <<EOF
#!/usr/bin/env bash
set -u
cd ${ROOT}
for run_dir in \\
  ${RUN_ROOT}/xverse_allmod_doseaware_morgan512_budget16_seed42_2500 \\
  ${RUN_ROOT}/xverse_allmod_doseaware_morgan512_budget32_seed42_2500 \\
  ${RUN_ROOT}/xverse_allmod_doseaware_morgan512_budget64_seed42_2500 \\
  ${RUN_ROOT}/xverse_allmod_doseaware_morgan512_budget64_seed43_2500; do
  run_name=\$(basename "\${run_dir}")
  original=\${run_dir}/scripts/posthoc_\${run_name}.sh
  patched=\${run_dir}/scripts/posthoc_manifestfix_\${run_name}.sh
  if [[ ! -f "\${original}" ]]; then
    echo "missing posthoc script: \${original}" >&2
    exit 2
  fi
  sed 's/^export CUDA_VISIBLE_DEVICES=.*/export CUDA_VISIBLE_DEVICES=${GPU_ID}/' "\${original}" > "\${patched}"
  chmod +x "\${patched}"
  echo "[\$(date '+%F %T %Z')] rerun posthoc \${run_name} on GPU ${GPU_ID}"
  bash "\${patched}" > "${LOG_ROOT}/posthoc_manifestfix/\${run_name}.log" 2>&1
  code=\$?
  echo "\${code}" > "\${run_dir}/POSTHOC_MANIFEST_FIX_EXIT_CODE"
  date > "\${run_dir}/POSTHOC_MANIFEST_FIX_FINISHED"
  if [[ "\${code}" != "0" ]]; then
    exit "\${code}"
  fi
done
"${PYTHON}" "${SUMMARIZER}" > "${LOG_ROOT}/posthoc_manifestfix/summarizer.log" 2>&1
summary_code=\$?
echo "\${summary_code}" > "${RUN_ROOT}/POSTHOC_MANIFEST_FIX_SUMMARY_EXIT_CODE"
date > "${RUN_ROOT}/POSTHOC_MANIFEST_FIX_SUMMARY_FINISHED"
exit "\${summary_code}"
EOF
chmod +x "${worker}"

tmux new -d -s "${SESSION}" "bash '${worker}' > '${LOG_ROOT}/posthoc_manifestfix/rerun.log' 2>&1; code=\$?; echo \${code} > '${RUN_ROOT}/POSTHOC_MANIFEST_FIX_EXIT_CODE'; date > '${RUN_ROOT}/POSTHOC_MANIFEST_FIX_FINISHED'; exit \${code}"
echo "Launched ${SESSION} on GPU ${GPU_ID}"
tmux ls | grep "${SESSION}" || true
