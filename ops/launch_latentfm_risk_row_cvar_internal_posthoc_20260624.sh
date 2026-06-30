#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_RISK_ROW_POSTHOC_ACK:-}" != "trainonly_internal_noharm_only" ]]; then
  echo "Set LATENTFM_RISK_ROW_POSTHOC_ACK=trainonly_internal_noharm_only" >&2
  exit 4
fi

RUN_NAME=xverse_risk_row_cvar_allrisk_w020_2k_seed42
RUN_ROOT=${ROOT}/runs/latentfm_risk_row_cvar_trainonly_20260624
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
LOG_ROOT=${ROOT}/logs/latentfm_risk_row_cvar_trainonly_20260624/${RUN_NAME}
EVAL_DIR=${RUN_DIR}/posthoc_eval_internal
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
SPLIT_FILE=${BIFLOW_DIR}/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json
PERT_MEANS=${ROOT}/runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/xverse_trainonly_scaling_general_exposure_cap_v2_pert_means.npz
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
CANDIDATE_CKPT=${COUPLED}/output/latentfm_runs/risk_row_cvar_trainonly_20260624/${RUN_NAME}/latest.pt
SUMMARIZER=${ROOT}/ops/summarize_latentfm_risk_row_cvar_internal_posthoc_20260624.py
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
SESSION=lfm_riskrow_internal_posthoc_20260624

for required in "${RUN_DIR}/RUN_STATUS.md" "${CANDIDATE_CKPT}" "${ANCHOR_CKPT}" "${SPLIT_FILE}" "${PERT_MEANS}" "${SUMMARIZER}" "${GPU_HELPER}"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

if [[ "$(cat "${RUN_DIR}/${RUN_NAME}.EXIT_CODE" 2>/dev/null || echo missing)" != "0" ]]; then
  echo "Training exit code is not 0; refusing posthoc" >&2
  exit 3
fi
if [[ -e "${RUN_DIR}/POSTHOC_EXIT_CODE" ]]; then
  echo "Posthoc marker already exists; refusing rerun" >&2
  exit 3
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}/logs" "${EVAL_DIR}" "${LOG_ROOT}" "${RUN_DIR}/scripts"

echo "[$(date '+%F %T %Z')] exact GPU/CPU/RAM status before risk-row internal posthoc" | tee "${RUN_ROOT}/logs/riskrow_internal_posthoc_resource_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/riskrow_internal_posthoc_resource_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/riskrow_internal_posthoc_resource_audit.log"
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 15 | tee -a "${RUN_ROOT}/logs/riskrow_internal_posthoc_resource_audit.log"

gpu_json=${RUN_ROOT}/logs/riskrow_internal_posthoc_gpu_selection_$(date +%Y%m%d_%H%M%S).json
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 1 \
  --need 1 \
  --json-only > "${gpu_json}" 2> "${RUN_ROOT}/logs/riskrow_internal_posthoc_gpu_selection.stderr"

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

posthoc_script=${RUN_DIR}/scripts/posthoc_internal_${RUN_NAME}.sh
cat > "${posthoc_script}" <<EOF
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
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${SPLIT_FILE} --pert-means-file ${PERT_MEANS} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out ${EVAL_DIR}/split_group_eval_anchor_internal_ode20.json "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug test_single --out ${EVAL_DIR}/condition_family_eval_anchor_internal_ode20.json "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${CANDIDATE_CKPT} --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out ${EVAL_DIR}/split_group_eval_candidate_internal_ode20.json "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${CANDIDATE_CKPT} --groups test_all family_gene family_drug test_single --out ${EVAL_DIR}/condition_family_eval_candidate_internal_ode20.json "\${common[@]}"
${PYTHON} ${SUMMARIZER}
EOF
chmod +x "${posthoc_script}"

cat >> "${RUN_DIR}/RUN_STATUS.md" <<EOF

## Internal posthoc addendum

Start time: $(date '+%F %T %Z')

tmux session: \`${SESSION}\`; physical GPU: \`${GPU}\`

Command:

\`\`\`bash
LATENTFM_RISK_ROW_POSTHOC_ACK=trainonly_internal_noharm_only bash ${ROOT}/ops/launch_latentfm_risk_row_cvar_internal_posthoc_20260624.sh
\`\`\`

Boundary: train-only/internal split only; no canonical metrics, canonical multi,
Track C query, or held-out query artifacts.

Expected decision:
\`${ROOT}/reports/LATENTFM_RISK_ROW_CVAR_INTERNAL_POSTHOC_DECISION_20260624.md\`
EOF

tmux new -d -s "${SESSION}" "bash -lc 'bash ${posthoc_script} > ${LOG_ROOT}/internal_posthoc.log 2>&1; rc=\$?; echo \$rc > ${RUN_DIR}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${RUN_DIR}/POSTHOC_FINISHED; exit \$rc'"
tmux ls
tail -n 30 "${LOG_ROOT}/internal_posthoc.log" || true
