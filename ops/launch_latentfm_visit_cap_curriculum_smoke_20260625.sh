#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_VISIT_CAP_CURRICULUM_ACK:-}" != "launch_visit_cap_curriculum_bounded_smoke" ]]; then
  cat >&2 <<'EOF'
Refusing to launch visit-cap curriculum GPU smoke.

Set:
  LATENTFM_VISIT_CAP_CURRICULUM_ACK=launch_visit_cap_curriculum_bounded_smoke

Boundary:
  - requires LATENTFM_VISIT_CAP_CURRICULUM_SLATE_GATE_20260625 pass
  - launches exactly one bounded train-only/internal smoke
  - no canonical multi, Track C query, or deployable promotion claim
  - canonical single/family no-harm may only be run later after route freeze
EOF
  exit 4
fi

GATE_JSON=${LATENTFM_VISIT_CAP_GATE_JSON:-${ROOT}/reports/latentfm_visit_cap_curriculum_slate_gate_20260625.json}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
TRAINONLY_SPLIT=${BIFLOW_DIR}/split_seed42_xverse_trainonly_crossbg_val_v2.json
TRAINONLY_PERT_MEANS=${ROOT}/runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
SUMMARIZER=${ROOT}/ops/summarize_latentfm_scaling_highthroughput_smokes_20260624.py

RUN_NAME=${LATENTFM_VISIT_CAP_RUN_NAME:-xverse_visitcap_p05_cap3_3k_seed42}
RUN_ROOT=${ROOT}/runs/latentfm_visit_cap_curriculum_smoke_20260625
OUT_ROOT=${COUPLED}/output/latentfm_runs/visit_cap_curriculum_smoke_20260625
LOG_ROOT=${ROOT}/logs/latentfm_visit_cap_curriculum_smoke_20260625
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
OUT_DIR=${OUT_ROOT}/${RUN_NAME}
LOG_DIR=${LOG_ROOT}/${RUN_NAME}
SESSION=lfm_${RUN_NAME}
TOTAL_STEPS=${LATENTFM_VISIT_CAP_TOTAL_STEPS:-3000}
VISIT_POWER=${LATENTFM_VISIT_CAP_POWER:-0.5}
VISIT_CAP=${LATENTFM_VISIT_CAP_CAP:-3}
VISIT_EXPECTED_CANDIDATE=${LATENTFM_VISIT_CAP_EXPECTED_CANDIDATE:-sublinear_visitpower0p5_cap3}
VISIT_HYPOTHESIS=${LATENTFM_VISIT_CAP_HYPOTHESIS:-"sublinear condition visits reduce large-condition/tail exposure while preserving condition coverage better than generic dataset loss or hard balancing."}

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${GATE_JSON}" \
  "${DATA_DIR}/manifest.json" \
  "${TRAINONLY_SPLIT}" \
  "${TRAINONLY_PERT_MEANS}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}" \
  "${SUMMARIZER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - "${GATE_JSON}" "${VISIT_EXPECTED_CANDIDATE}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sys.argv[2]
status = str(payload.get("status") or "")
allowed_statuses = {
    "visit_cap_curriculum_slate_gate_pass_one_bounded_smoke_candidate",
    "visit_cap_mild_mutation_gate_pass_one_bounded_smoke_candidate",
}
if status not in allowed_statuses:
    raise SystemExit(f"visit-cap gate not passed: {status!r}")
selected = payload.get("selected_candidate") or payload.get("mild_candidate") or {}
cfg = selected.get("config") if isinstance(selected.get("config"), dict) else selected
if cfg.get("name") != expected:
    raise SystemExit(f"unexpected visit-cap candidate {cfg}; expected {expected}")
PY

if [[ -e "${OUT_DIR}" && "${FORCE_LATENTFM_VISIT_CAP_RERUN:-0}" != "1" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_LATENTFM_VISIT_CAP_RERUN=1 to relaunch" >&2
  exit 3
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] exact GPU/CPU/RAM status before visit-cap launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
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
  --max-user-gpus 2 \
  --max-jobs-per-gpu 2 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
suggested = [int(x) for x in payload.get("suggested_job_gpus", [])]
system = payload.get("system") or {}
reasons = []
if not suggested:
    reasons.append("no GPU job slot available under temporary 2-GPU/2-jobs-per-GPU cap")
if int(payload.get("max_user_gpus") or 0) > 2:
    reasons.append("max_user_gpus exceeds temporary cap 2")
if int(payload.get("max_jobs_per_gpu") or 0) > 2:
    reasons.append("max_jobs_per_gpu exceeds temporary cap 2")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128 GiB")
audit = {
    "status": "fail" if reasons else "pass",
    "reasons": reasons,
    "assigned_gpus": suggested[:1],
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
    "temporary_caps": {"physical_gpus": 2, "jobs_per_gpu": 2, "cpu_threads_project": 24},
}
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

GPU=$("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["assigned_gpus"][0])
PY
)

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${LOG_DIR}"
train_script=${RUN_DIR}/scripts/run_${RUN_NAME}.sh
posthoc_script=${RUN_DIR}/scripts/posthoc_${RUN_NAME}.sh
wrapper_script=${RUN_DIR}/scripts/wrap_${RUN_NAME}.sh

cat > "${train_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
export LATENT_BACKBONE=xverse
export DATA_DIR=${DATA_DIR}
export BIFLOW_DIR=${BIFLOW_DIR}
export SPLIT_FILE=${TRAINONLY_SPLIT}
export PERT_MEANS_FILE=${TRAINONLY_PERT_MEANS}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${LOG_DIR}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${GPU}
export RUN_TAG=${RUN_NAME}
export SEED=42
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=1
export FINETUNE_TRAINABLE_SCOPE=all
export TOTAL_STEPS=${TOTAL_STEPS}
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export LR=1e-4
export DS_ALPHA=0.7
export DS_LOSS_ALPHA=0.0
export MIN_SELECTED_CONDITIONS_PER_DATASET=0
export CONDITION_VISIT_POWER=${VISIT_POWER}
export CONDITION_VISIT_CAP=${VISIT_CAP}
export OT_PAIR_MODE=multinomial
export GAMMA=0.03
export GAMMA_WARMUP_START=500
export GAMMA_WARMUP_END=1500
export MMD_EVERY=4
export SELECTION_METRIC=test_mmd
export SELECTION_MMD_LAMBDA=1.0
export COMPOSITION_DELTA_LOSS_WEIGHT=0.06
export COMPOSITION_DELTA_LOSS_WARMUP_START=500
export COMPOSITION_DELTA_LOSS_WARMUP_END=1500
export ENDPOINT_DELTA_LOSS_WEIGHT=5.0
export ENDPOINT_DELTA_LOSS_WARMUP_START=500
export ENDPOINT_DELTA_LOSS_WARMUP_END=1500
export EVAL_MAX_CONDITIONS=256
export EVAL_MAX_CONDITIONS_PER_DATASET=12
export EVAL_MAX_MSE_CELLS=1024
export EVAL_MAX_MMD_CELLS=1024
export EVAL_MAX_CHUNK=256
export PERT_POOL_AGGREGATIONS="sum mean max min"
export PERT_POOL_SCALE_INIT="0.5 1.0 1.0 1.0"
export PERT_POOL_FUSION_MODE=sum
export PERT_GENE_PROJECTOR_HIDDEN=1024
export PERT_CHEM_PROJECTOR_HIDDEN=1024
export PERT_TO_C_INIT_MODE=xavier_small
export USE_PERT_IN_FUSION=1
bash ${TRAIN_LAUNCHER}
EOF
chmod +x "${train_script}"

cat > "${posthoc_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
eval_dir=${RUN_DIR}/posthoc_eval_internal
mkdir -p "\${eval_dir}"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${TRAINONLY_SPLIT} --pert-means-file ${TRAINONLY_PERT_MEANS} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy test_single --out "\${eval_dir}/split_group_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups family_gene test_single --out "\${eval_dir}/condition_family_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${OUT_DIR}/best.pt --groups internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy test_single --out "\${eval_dir}/split_group_eval_candidate_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${OUT_DIR}/best.pt --groups family_gene test_single --out "\${eval_dir}/condition_family_eval_candidate_internal_ode20.json" "\${common[@]}"
EOF
chmod +x "${posthoc_script}"

cat > "${wrapper_script}" <<EOF
#!/usr/bin/env bash
set -u

bash '${train_script}' > '${LOG_DIR}/launcher.log' 2>&1
train_code=\$?
echo "\${train_code}" > '${RUN_DIR}/${RUN_NAME}.EXIT_CODE'
date > '${RUN_DIR}/${RUN_NAME}.FINISHED'

if [[ "\${train_code}" == "0" ]]; then
  bash '${posthoc_script}' > '${LOG_DIR}/posthoc.log' 2>&1
  posthoc_code=\$?
  echo "\${posthoc_code}" > '${RUN_DIR}/POSTHOC_EXIT_CODE'
  date > '${RUN_DIR}/POSTHOC_FINISHED'
  if [[ "\${posthoc_code}" == "0" ]]; then
    LATENTFM_SCALING_HT_RUN_ROOT='${RUN_ROOT}' \\
    LATENTFM_SCALING_HT_RUNS='${RUN_NAME}' \\
    LATENTFM_SCALING_HT_DECISION_JSON='${ROOT}/reports/latentfm_visit_cap_curriculum_smoke_decision_20260625.json' \\
    LATENTFM_SCALING_HT_DECISION_MD='${ROOT}/reports/LATENTFM_VISIT_CAP_CURRICULUM_SMOKE_DECISION_20260625.md' \\
    '${PYTHON}' '${SUMMARIZER}' > '${LOG_DIR}/summarizer.log' 2>&1
    summary_code=\$?
    echo "\${summary_code}" > '${RUN_DIR}/SUMMARY_EXIT_CODE'
    date > '${RUN_DIR}/SUMMARY_FINISHED'
  fi
fi

exit "\${train_code}"
EOF
chmod +x "${wrapper_script}"

cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
LATENTFM_VISIT_CAP_CURRICULUM_ACK=launch_visit_cap_curriculum_bounded_smoke bash ${ROOT}/ops/launch_latentfm_visit_cap_curriculum_smoke_20260625.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

## Log path

\`${LOG_DIR}/launcher.log\`

## Expected outputs

* \`${OUT_DIR}/best.pt\`
* \`${RUN_DIR}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`
* \`${RUN_DIR}/posthoc_eval_internal/condition_family_eval_candidate_internal_ode20.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_DIR}/launcher.log
cat ${RUN_DIR}/${RUN_NAME}.EXIT_CODE 2>/dev/null || echo "still running"
cat ${RUN_DIR}/POSTHOC_EXIT_CODE 2>/dev/null || echo "posthoc not done"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Hypothesis: ${VISIT_HYPOTHESIS}

Visit-cap candidate: \`CONDITION_VISIT_POWER=${VISIT_POWER}\`, \`CONDITION_VISIT_CAP=${VISIT_CAP}\`.

Gate file: \`${GATE_JSON}\`.

Gate: train-only/internal cross pp delta >= +0.010, internal/family pp no hard regression, family MMD <= +0.001, no dataset-tail harm. Canonical multi and held-out Track C query are not used.
EOF

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_visit_cap_curriculum_smoke_20260625

## Command

\`\`\`bash
LATENTFM_VISIT_CAP_CURRICULUM_ACK=launch_visit_cap_curriculum_bounded_smoke bash ${ROOT}/ops/launch_latentfm_visit_cap_curriculum_smoke_20260625.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

## Log path

\`${LOG_DIR}/launcher.log\`

## Expected outputs

* \`${RUN_DIR}/RUN_STATUS.md\`
* \`${RUN_DIR}/posthoc_eval_internal/\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_DIR}/launcher.log
cat ${RUN_DIR}/${RUN_NAME}.EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

One-run bounded visit-cap curriculum candidate selected by gate:
\`${GATE_JSON}\`.
EOF

tmux new -d -s "${SESSION}" "bash '${wrapper_script}'"

echo "${SESSION}" > "${RUN_DIR}/SESSION_NAME"
date > "${RUN_DIR}/${RUN_NAME}.STARTED"
echo "Launched ${RUN_NAME} on GPU ${GPU} in tmux ${SESSION}"
tmux ls | grep "${SESSION}" || true
