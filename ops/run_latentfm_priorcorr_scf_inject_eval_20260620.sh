#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_priorcorr_scf_inject_eval_20260620
LOG_DIR=${RUN_ROOT}/logs
SESSION=latentfm_priorcorr_scf_inject_eval_20260620
SCRIPT=${ROOT}/ops/evaluate_latentfm_prior_correction_20260619.py
CHECKPOINT=${ROOT}/CoupledFM/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/best.pt
OUT_MD=${ROOT}/reports/LATENTFM_PRIOR_CORRECTION_EVAL_SCF_INJECT_20260620.md
OUT_CSV=${ROOT}/reports/latentfm_prior_correction_eval_scf_inject_20260620.csv
OUT_JSON=${ROOT}/reports/latentfm_prior_correction_eval_scf_inject_20260620.json

mkdir -p "${LOG_DIR}"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"

date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_priorcorr_scf_inject_eval_20260620

## Command

\`\`\`bash
bash ${SCRIPT} via ${BASH_SOURCE[0]}
\`\`\`

## Runtime classification

Long/unknown GPU evaluation task. Use 30-minute cadence for checks.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## tmux

\`${SESSION}\`

## GPU

Physical GPU0, selected by exact \`nvidia-smi\` plus 3-sample shared-GPU helper.

## Log path

\`${LOG_DIR}/run.log\`

## Expected outputs

* \`${OUT_MD}\`
* \`${OUT_CSV}\`
* \`${OUT_JSON}\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo still-running
tail -n 50 ${LOG_DIR}/run.log
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

This is an evaluator, not training. It tests prior-correction on the current
best scFoundation anchor \`scf_prior010_inject_e2_4k\`.
EOF

cat > "${RUN_ROOT}/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source /data/cyx/1030/scLatent/init-scdfm.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
python /data/cyx/1030/scLatent/ops/evaluate_latentfm_prior_correction_20260619.py \
  --checkpoint /data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/best.pt \
  --label scf_prior010_inject_e2_4k_priorcorr \
  --datasets NormanWeissman2019_filtered Wessels GasperiniShendure2019_lowMOI \
  --groups test_multi_seen test_multi_unseen1 test_multi_unseen2 \
  --alphas 0.0 0.25 0.5 0.75 1.0 \
  --k-values 5 10 \
  --ode-steps 20 \
  --eval-max-cells 128 \
  --prior-max-cells 512 \
  --max-chunk 256 \
  --gpu 0 \
  --out-md /data/cyx/1030/scLatent/reports/LATENTFM_PRIOR_CORRECTION_EVAL_SCF_INJECT_20260620.md \
  --out-csv /data/cyx/1030/scLatent/reports/latentfm_prior_correction_eval_scf_inject_20260620.csv \
  --out-json /data/cyx/1030/scLatent/reports/latentfm_prior_correction_eval_scf_inject_20260620.json
EOF
chmod +x "${RUN_ROOT}/run.sh"

tmux new-session -d -s "${SESSION}" "bash '${RUN_ROOT}/run.sh' > '${LOG_DIR}/run.log' 2>&1; code=\$?; echo \${code} > '${RUN_ROOT}/EXIT_CODE'; date '+%F %T %Z' > '${RUN_ROOT}/FINISHED'; exit \${code}"

echo "launched ${SESSION}"
echo "RUN_ROOT=${RUN_ROOT}"
