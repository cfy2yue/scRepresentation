#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_TRACKC_SUPPORT_ABLATION_ACK:-}" != "safe_trainselect_support_controls_only" ]]; then
  cat >&2 <<'EOF'
Refusing Track C support-present ablation control launch.

Set:
  LATENTFM_TRACKC_SUPPORT_ABLATION_ACK=safe_trainselect_support_controls_only

Boundary:
  - safe trainselect support-val controls only;
  - no held-out query;
  - no canonical multi selection;
  - no training or checkpoint selection.
EOF
  exit 4
fi

RUN_NAME=latentfm_trackc_support_present_ablation_controls_20260624
RUN_ROOT=${ROOT}/runs/${RUN_NAME}
LOG_DIR=${RUN_ROOT}/logs
STATUS=${RUN_ROOT}/RUN_STATUS.md
mkdir -p "${LOG_DIR}"

PRIMARY_RUN=xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42
PRIMARY_ROOT=${ROOT}/runs/latentfm_xverse_trackc_support_context_v2_20260623/${PRIMARY_RUN}
OUT_EVAL=${PRIMARY_ROOT}/posthoc_eval
CHECKPOINT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_support_context_v2_20260623/${PRIMARY_RUN}/best.pt
SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json

for required in "${CHECKPOINT}" "${SPLIT}" "${PRIMARY_ROOT}" \
  "${ROOT}/ops/audit_latentfm_trackc_support_present_ablation_reproducibility_gate_20260624.py"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

GPU_AUDIT=${LOG_DIR}/gpu_selection_$(date +%Y%m%d_%H%M%S).json
"${PYTHON}" "${ROOT}/ops/select_available_gpus.py" \
  --samples 3 --interval-seconds 10 \
  --max-user-gpus 4 --max-jobs-per-gpu 4 --need 1 \
  --json-only > "${GPU_AUDIT}"
PHYSICAL_GPU=$("${PYTHON}" - "${GPU_AUDIT}" <<'PY'
import json, sys
obj = json.loads(open(sys.argv[1], encoding="utf-8").read())
gpus = obj.get("suggested_job_gpus") or []
if not gpus:
    raise SystemExit("no GPU available for Track C support ablation controls")
print(int(gpus[0]))
PY
)

SCRIPT=${RUN_ROOT}/run_controls.sh
cat > "${SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${ROOT}/CoupledFM
export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${ROOT}/CoupledFM:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
mkdir -p "${OUT_EVAL}"
common=(--data-dir ${ROOT}/dataset/latentfm_full/xverse --biflow-dir ${ROOT}/dataset/biFlow_data --split-file ${SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 512)
ckpt=${CHECKPOINT}

${PYTHON} -m model.latent.eval_split_groups --checkpoint "\${ckpt}" --groups test test_multi --support-context-control zero --out "${OUT_EVAL}/support_zero_candidate_split_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint "\${ckpt}" --groups test_all family_gene structure_multi test_multi --support-context-control zero --out "${OUT_EVAL}/support_zero_candidate_family_ode20.json" "\${common[@]}"

${PYTHON} -m model.latent.eval_split_groups --checkpoint "\${ckpt}" --groups test test_multi --support-context-control shuffle_condition --out "${OUT_EVAL}/support_shuffle_condition_candidate_split_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint "\${ckpt}" --groups test_all family_gene structure_multi test_multi --support-context-control shuffle_condition --out "${OUT_EVAL}/support_shuffle_condition_candidate_family_ode20.json" "\${common[@]}"

${PYTHON} -m model.latent.eval_split_groups --checkpoint "\${ckpt}" --groups test test_multi --force-support-context-absent --out "${OUT_EVAL}/support_absent_support_candidate_split_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint "\${ckpt}" --groups test_all family_gene structure_multi test_multi --force-support-context-absent --out "${OUT_EVAL}/support_absent_support_candidate_family_ode20.json" "\${common[@]}"

cd ${ROOT}
${PYTHON} ${ROOT}/ops/audit_latentfm_trackc_support_present_ablation_reproducibility_gate_20260624.py
EOF
chmod +x "${SCRIPT}"

cat > "${STATUS}" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
LATENTFM_TRACKC_SUPPORT_ABLATION_ACK=safe_trainselect_support_controls_only bash ${ROOT}/ops/launch_latentfm_trackc_support_present_ablation_controls_20260624.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${RUN_NAME}\`, physical GPU${PHYSICAL_GPU}

## Log path

\`${LOG_DIR}/run.log\`

## Expected outputs

* \`${OUT_EVAL}/support_zero_candidate_split_ode20.json\`
* \`${OUT_EVAL}/support_shuffle_condition_candidate_split_ode20.json\`
* \`${OUT_EVAL}/support_absent_support_candidate_split_ode20.json\`
* \`${ROOT}/reports/LATENTFM_TRACKC_SUPPORT_PRESENT_ABLATION_REPRODUCIBILITY_GATE_20260624.md\`

## How to check manually

\`\`\`bash
tmux ls
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
tail -n 50 ${LOG_DIR}/run.log
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Hypothesis: the frozen support-context v2 support-val gain should collapse
  when support context is zeroed, shuffled across conditions, or forced absent.
- Boundary: safe trainselect support-val controls only; no held-out query,
  no canonical multi selection, no training, and no checkpoint selection.
- Stop rule: any nonzero exit, missing artifact, or non-collapsing control
  blocks Track C GPU/modeling continuation.
EOF

tmux new -d -s "${RUN_NAME}" "bash '${SCRIPT}' > '${LOG_DIR}/run.log' 2>&1; echo \$? > '${RUN_ROOT}/EXIT_CODE'; date '+%F %T %Z' > '${RUN_ROOT}/FINISHED'"
echo "${RUN_NAME}" > "${RUN_ROOT}/SESSION_NAME"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
echo "Started ${RUN_NAME} on physical GPU${PHYSICAL_GPU}"
