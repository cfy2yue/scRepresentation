#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_TRUE_CELL_COUNT_SMOKE_ACK:-}" != "bounded_capped_data_smoke" ]]; then
  cat >&2 <<'EOF'
Refusing to launch true cell-count GPU smoke.

Set:
  LATENTFM_TRUE_CELL_COUNT_SMOKE_ACK=bounded_capped_data_smoke
  LATENTFM_TRUE_CELL_COUNT_RUN_ID=<materialized run id>

Boundary:
  - requires materialized capped-H5 artifact from true cell-count materializer
  - uses capped DATA_DIR, matching split, and train-only pert_means.npz
  - requires schema/provenance, dry-load, and design-control gate reports
  - train-only internal selection only
  - no canonical multi or Track C query
EOF
  exit 4
fi

RUN_ID=${LATENTFM_TRUE_CELL_COUNT_RUN_ID:-}
if [[ -z "${RUN_ID}" ]]; then
  echo "Missing LATENTFM_TRUE_CELL_COUNT_RUN_ID" >&2
  exit 4
fi

MATERIALIZER_JSON=${LATENTFM_TRUE_CELL_COUNT_MATERIALIZER_JSON:-${ROOT}/reports/latentfm_true_cell_count_capped_h5_materializer_gate_20260624.json}
SCHEMA_JSON=${LATENTFM_TRUE_CELL_COUNT_SCHEMA_JSON:-${ROOT}/reports/latentfm_true_cell_count_capped_h5_schema_gate_20260624.json}
DRYLOAD_JSON=${LATENTFM_TRUE_CELL_COUNT_DRYLOAD_JSON:-${ROOT}/reports/latentfm_true_cell_count_dryload_gate_20260624.json}
DESIGN_JSON=${LATENTFM_TRUE_CELL_COUNT_DESIGN_JSON:-${ROOT}/reports/latentfm_true_cell_count_design_controls_gate_20260624.json}
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
BIFLOW_DIR=${ROOT}/dataset/biFlow_data

RUN_ROOT=${LATENTFM_TRUE_CELL_COUNT_RUN_ROOT:-${ROOT}/runs/latentfm_true_cell_count_smokes_20260624}
OUT_ROOT=${LATENTFM_TRUE_CELL_COUNT_OUT_ROOT:-${COUPLED}/output/latentfm_runs/true_cell_count_smokes_20260624}
LOG_ROOT=${LATENTFM_TRUE_CELL_COUNT_LOG_ROOT:-${ROOT}/logs/latentfm_true_cell_count_smokes_20260624}
TOTAL_STEPS=${LATENTFM_TRUE_CELL_COUNT_TOTAL_STEPS:-3000}
RUN_HYPOTHESIS=${LATENTFM_TRUE_CELL_COUNT_HYPOTHESIS:-"True cell-count capped-data smoke: with fixed condition identities and capped train cells, test whether training cell count alone produces a train-only internal signal. This is not a promotion claim."}
RUN_STOP_RULE=${LATENTFM_TRUE_CELL_COUNT_STOP_RULE:-"Promotion requires a separate train-only decision summary, then frozen canonical single/family no-harm only if train-only gate passes."}
LR_VALUE=${LATENTFM_TRUE_CELL_COUNT_LR:-1e-4}
FINETUNE_SCOPE=${LATENTFM_TRUE_CELL_COUNT_FINETUNE_SCOPE:-all}
ANCHOR_REPLAY_WEIGHT=${LATENTFM_TRUE_CELL_COUNT_ANCHOR_REPLAY_LOSS_WEIGHT:-0.0}

mapfile -t ARTIFACT_INFO < <("${PYTHON}" - "${MATERIALIZER_JSON}" "${RUN_ID}" <<'PY'
import json, sys
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text())
if not payload.get("materialized"):
    raise SystemExit("materializer JSON is not materialized")
for row in payload.get("materialized_rows", []):
    if row.get("run_id") == sys.argv[2]:
        data_dir=Path(row["data_dir"])
        split_file=Path(row["split_file"])
        for p in [data_dir/"manifest.json", data_dir/"pert_means.npz", split_file]:
            if not p.exists():
                raise SystemExit(f"missing artifact: {p}")
        print(data_dir)
        print(split_file)
        print(data_dir/"pert_means.npz")
        raise SystemExit(0)
raise SystemExit(f"run id not materialized: {sys.argv[2]}")
PY
)

"${PYTHON}" - "${SCHEMA_JSON}" "${RUN_ID}" <<'PY'
import json, sys
from pathlib import Path
schema_path = Path(sys.argv[1])
run_id = sys.argv[2]
if not schema_path.exists():
    raise SystemExit(f"schema gate report does not exist: {schema_path}")
payload = json.loads(schema_path.read_text())
if payload.get("status") != "capped_h5_schema_gate_pass_no_gpu":
    raise SystemExit(f"schema gate is not pass: {payload.get('status')}")
for row in payload.get("rows", []):
    if row.get("run_id") == run_id:
        if row.get("status") != "ok":
            raise SystemExit(f"schema gate row is not ok for {run_id}: {row.get('reasons')}")
        sample = row.get("sample_provenance") or {}
        if sample.get("status") != "ok":
            raise SystemExit(f"sample provenance is not ok for {run_id}: {sample.get('reasons')}")
        raise SystemExit(0)
raise SystemExit(f"schema gate has no row for run id: {run_id}")
PY

"${PYTHON}" - "${DRYLOAD_JSON}" "${RUN_ID}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
run_id = sys.argv[2]
if not path.exists():
    raise SystemExit(f"dry-load gate report does not exist: {path}")
payload = json.loads(path.read_text())
if payload.get("status") != "true_cell_count_dryload_pass_no_gpu":
    raise SystemExit(f"dry-load gate is not pass: {payload.get('status')}")
for row in payload.get("rows", []):
    if row.get("run_id") == run_id:
        if row.get("status") != "ok":
            raise SystemExit(f"dry-load gate row is not ok for {run_id}: {row.get('reasons')}")
        raise SystemExit(0)
raise SystemExit(f"dry-load gate has no row for run id: {run_id}")
PY

"${PYTHON}" - "${DESIGN_JSON}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"design-control gate report does not exist: {path}")
payload = json.loads(path.read_text())
if payload.get("status") != "true_cell_count_design_controls_pass_preliminary_only_no_gpu":
    raise SystemExit(f"design-control gate is not pass: {payload.get('status')}")
PY

DATA_DIR=${ARTIFACT_INFO[0]}
SPLIT_FILE=${ARTIFACT_INFO[1]}
PERT_MEANS=${ARTIFACT_INFO[2]}
SAFE_RUN_ID=$(echo "${RUN_ID}" | tr -c 'A-Za-z0-9_' '_' | cut -c1-80)
RUN_NAME=${LATENTFM_TRUE_CELL_COUNT_RUN_NAME:-xverse_truecell_${SAFE_RUN_ID}_${TOTAL_STEPS}}
SMOKE_SEED=${LATENTFM_TRUE_CELL_COUNT_SEED:-}
if [[ -z "${SMOKE_SEED}" ]]; then
  SMOKE_SEED=$("${PYTHON}" - "${RUN_ID}" <<'PY'
import re, sys
m = re.search(r"_seed(\d+)(?:_|$)", sys.argv[1])
if not m:
    raise SystemExit(f"could not parse seed from run id: {sys.argv[1]}")
print(m.group(1))
PY
)
fi
SESSION=lfm_${RUN_NAME}
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
OUT_DIR=${OUT_ROOT}/${RUN_NAME}
LOG_DIR=${LOG_ROOT}/${RUN_NAME}

for required in "${DATA_DIR}/manifest.json" "${SPLIT_FILE}" "${PERT_MEANS}" "${ANCHOR_CKPT}" "${GENE_CACHE}/manifest.json" "${GPU_HELPER}" "${TRAIN_LAUNCHER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${OUT_DIR}" && "${FORCE_LATENTFM_TRUE_CELL_COUNT_SMOKE:-0}" != "1" ]]; then
  echo "Output exists for ${RUN_NAME}; set FORCE_LATENTFM_TRUE_CELL_COUNT_SMOKE=1 to relaunch" >&2
  exit 3
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${RUN_ROOT}/logs" "${LOG_DIR}" "${OUT_ROOT}"

echo "[$(date '+%F %T %Z')] exact GPU status before true cell-count smoke launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit_${RUN_NAME}.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit_${RUN_NAME}.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit_${RUN_NAME}.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/gpu_launch_audit_${RUN_NAME}.log"
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 30 | tee -a "${RUN_ROOT}/logs/gpu_launch_audit_${RUN_NAME}.log"

GPU_OVERRIDE="${LATENTFM_TRUE_CELL_COUNT_GPU_OVERRIDE:-}"
if [[ -n "${GPU_OVERRIDE}" ]]; then
  if ! [[ "${GPU_OVERRIDE}" =~ ^[0-9]+$ ]]; then
    echo "LATENTFM_TRUE_CELL_COUNT_GPU_OVERRIDE must be a physical GPU index, got: ${GPU_OVERRIDE}" >&2
    exit 4
  fi
  GPU="${GPU_OVERRIDE}"
  gpu_json="${RUN_ROOT}/logs/gpu_selection_${RUN_NAME}_override_$(date +%Y%m%d_%H%M%S).json"
  {
    echo "{"
    echo "  \"override\": true,"
    echo "  \"selected_gpu\": ${GPU},"
    echo "  \"reason\": \"LATENTFM_TRUE_CELL_COUNT_GPU_OVERRIDE after external multi-sample resource audit\""
    echo "}"
  } > "${gpu_json}"
else
  gpu_json="${RUN_ROOT}/logs/gpu_selection_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" "${GPU_HELPER}" --samples 3 --interval-seconds 10 --util-threshold-pct 10 --memory-threshold-mib 4096 --max-user-gpus 4 --max-jobs-per-gpu 4 --need 1 --json-only > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection_${RUN_NAME}.stderr"

  GPU=$("${PYTHON}" - "${gpu_json}" <<'PY'
import json, sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
slots=[int(x) for x in p.get("suggested_job_gpus", [])]
if not slots:
    raise SystemExit("no GPU slot suggested")
print(slots[0])
PY
)
fi

train_script=${RUN_DIR}/scripts/run_${RUN_NAME}.sh
posthoc_script=${RUN_DIR}/scripts/posthoc_${RUN_NAME}.sh

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
export SPLIT_FILE=${SPLIT_FILE}
export PERT_MEANS_FILE=${PERT_MEANS}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${LOG_DIR}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${GPU}
export RUN_TAG=${RUN_NAME}
export SEED=${SMOKE_SEED}
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=1
export FINETUNE_TRAINABLE_SCOPE=${FINETUNE_SCOPE}
export TOTAL_STEPS=${TOTAL_STEPS}
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export LR=${LR_VALUE}
export GAMMA=0.03
export GAMMA_WARMUP_START=500
export GAMMA_WARMUP_END=1500
export MMD_EVERY=4
export OT_PAIR_MODE=multinomial
export SELECTION_METRIC=test_mmd
export SELECTION_MMD_LAMBDA=1.0
export COMPOSITION_DELTA_LOSS_WEIGHT=0.06
export COMPOSITION_DELTA_LOSS_WARMUP_START=500
export COMPOSITION_DELTA_LOSS_WARMUP_END=1500
export ENDPOINT_DELTA_LOSS_WEIGHT=5.0
export ENDPOINT_DELTA_LOSS_WARMUP_START=500
export ENDPOINT_DELTA_LOSS_WARMUP_END=1500
export DS_ALPHA=0.7
export DS_LOSS_ALPHA=0.0
export MIN_SELECTED_CONDITIONS_PER_DATASET=0
export CONDITION_VISIT_POWER=1.0
export CONDITION_VISIT_CAP=0
export ANCHOR_REPLAY_LOSS_WEIGHT=${ANCHOR_REPLAY_WEIGHT}
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
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${SPLIT_FILE} --pert-means-file ${PERT_MEANS} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out "\${eval_dir}/split_group_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${OUT_DIR}/best.pt --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out "\${eval_dir}/split_group_eval_candidate_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${OUT_DIR}/best.pt --groups test_all family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_candidate_internal_ode20.json" "\${common[@]}"
EOF
chmod +x "${posthoc_script}"

date '+%F %T %Z' > "${RUN_DIR}/STARTED"
rm -f "${RUN_DIR}/EXIT_CODE" "${RUN_DIR}/FINISHED" "${RUN_DIR}/POSTHOC_EXIT_CODE" "${RUN_DIR}/POSTHOC_FINISHED"

tmux new -d -s "${SESSION}" "bash -lc 'bash ${train_script} > ${LOG_DIR}/launcher.log 2>&1; rc=\$?; echo \$rc > ${RUN_DIR}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_DIR}/FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${LOG_DIR}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${RUN_DIR}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${RUN_DIR}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"

cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Hypothesis

${RUN_HYPOTHESIS}

## Command

\`\`\`bash
LATENTFM_TRUE_CELL_COUNT_SMOKE_ACK=bounded_capped_data_smoke LATENTFM_TRUE_CELL_COUNT_RUN_ID=${RUN_ID} bash ${ROOT}/ops/launch_latentfm_true_cell_count_single_smoke_20260624.sh
\`\`\`

## Runtime classification

Long GPU training plus posthoc task. Use 30-minute cadence for result checks.

## Start time

$(cat "${RUN_DIR}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

Physical GPU: ${GPU}

## Log path

\`${LOG_DIR}/launcher.log\`

Posthoc log: \`${LOG_DIR}/posthoc.log\`

## Expected outputs

* \`${OUT_DIR}/best.pt\`
* \`${RUN_DIR}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`
* \`${RUN_DIR}/posthoc_eval_internal/condition_family_eval_candidate_internal_ode20.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_DIR}/launcher.log
cat ${RUN_DIR}/EXIT_CODE 2>/dev/null || echo "still running"
cat ${RUN_DIR}/POSTHOC_EXIT_CODE 2>/dev/null || echo "posthoc not complete"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Capped DATA_DIR: \`${DATA_DIR}\`
- Split: \`${SPLIT_FILE}\`
- Train-only pert means: \`${PERT_MEANS}\`
- Materializer JSON: \`${MATERIALIZER_JSON}\`
- Schema gate JSON: \`${SCHEMA_JSON}\`
- Dry-load gate JSON: \`${DRYLOAD_JSON}\`
- Design-control gate JSON: \`${DESIGN_JSON}\`
- Canonical multi and Track C query are not used.
- Stop rule: ${RUN_STOP_RULE}
- LR: \`${LR_VALUE}\`; finetune scope: \`${FINETUNE_SCOPE}\`; anchor replay weight: \`${ANCHOR_REPLAY_WEIGHT}\`.
EOF

tmux ls
tail -n 20 "${LOG_DIR}/launcher.log" 2>/dev/null || true
