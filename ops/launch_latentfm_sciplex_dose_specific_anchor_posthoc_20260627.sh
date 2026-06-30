#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
COUPLED=${ROOT}/CoupledFM
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
DRUG_CACHE=${ROOT}/dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625
DATA_DIR=${ROOT}/runs/latentfm_true_cell_count_allmodality_doseaware_capped_h5_20260625/artifacts/all_modality_doseaware_fixed64_budget16_32_64_budget64_seed42
SPLIT_FILE=${ROOT}/dataset/biFlow_data/sciplex_dose_specific_splits_20260627/split_seed42_sciplex_logdose_cap120_all_doseeval.json
PERT_MEANS=${DATA_DIR}/pert_means.npz
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
RUN_ROOT=${ROOT}/runs/latentfm_sciplex_dose_specific_anchor_posthoc_20260627
LOG_ROOT=${ROOT}/logs/latentfm_sciplex_dose_specific_anchor_posthoc_20260627
RUN_NAME=xverse_anchor_sciplex_logdose_doseeval_seed42
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
LOG_DIR=${LOG_ROOT}/${RUN_NAME}
OUT_JSON=${RUN_DIR}/condition_family_eval_anchor_sciplex_doseeval_ode20.json
SESSION=lfm_sciplex_dose_anchor_posthoc_20260627
THREADS=${LATENTFM_SCIPLEX_DOSE_POSTHOC_THREADS:-4}

mkdir -p "${RUN_DIR}/logs" "${LOG_DIR}"

"${PYTHON}" "${ROOT}/ops/build_latentfm_sciplex_dose_specific_outcome_gate_20260627.py" > "${RUN_DIR}/logs/preflight_split_builder.log" 2>&1

for path in "${PYTHON}" "${ANCHOR_CKPT}" "${SPLIT_FILE}" "${DATA_DIR}/manifest.json" "${DATA_DIR}/condition_metadata.json" "${PERT_MEANS}" "${DRUG_CACHE}/drug_embeddings.npy" "${DRUG_CACHE}/drug_index.json"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done

GPU_ID=$("${PYTHON}" - <<'PY'
import csv
import subprocess
import sys
import time

samples = []
for i in range(3):
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, mem, util = [x.strip() for x in line.split(",")]
        rows.append((int(idx), int(mem), int(util)))
    samples.append(rows)
    if i < 2:
        time.sleep(10)

by_idx = {idx: [] for idx, _, _ in samples[0]}
for rows in samples:
    for idx, mem, util in rows:
        by_idx[idx].append((mem, util))

strict_empty = [
    idx
    for idx, vals in by_idx.items()
    if all(mem < 4096 and util < 10 for mem, util in vals)
]
submittable = [
    idx
    for idx, vals in by_idx.items()
    if all(mem < 10240 and util < 30 for mem, util in vals)
]
if len(submittable) < 2:
    print(
        f"Need at least 2 submittable GPUs to launch one diagnostic while leaving one free; strict_empty={strict_empty}, submittable={submittable}",
        file=sys.stderr,
    )
    sys.exit(3)

def score(idx: int) -> tuple[int, int, int]:
    vals = by_idx[idx]
    max_mem = max(mem for mem, _ in vals)
    max_util = max(util for _, util in vals)
    return (0 if idx in strict_empty else 1, max_mem, max_util)

print(sorted(submittable, key=score)[0])
PY
)

{
  echo "Launch time: $(date '+%F %T %Z')"
  echo "Selected GPU: ${GPU_ID}"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
  free -h
} > "${RUN_DIR}/logs/resource_audit_at_launch.log"

cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_sciplex_dose_specific_anchor_posthoc_20260627.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: ${SESSION}

## Log path

\`${LOG_DIR}/posthoc.log\`

## Expected outputs

* \`${OUT_JSON}\`
* \`/data/cyx/1030/scLatent/reports/LATENTFM_SCIPLEX_DOSE_SPECIFIC_OUTCOME_GATE_20260627.md\`
* \`/data/cyx/1030/scLatent/reports/latentfm_sciplex_dose_specific_outcome_gate_20260627.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_DIR}/posthoc.log
cat ${RUN_DIR}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Hypothesis: the existing SciPlex dose artifact can only be evaluated fairly at
dose-level. Existing condition metrics have 460 drug rows but zero within-drug
multi-dose groups, so this anchor-only diagnostic posthoc materializes the
1364-row dose-specific outcome table needed by the CPU gate.

Boundary: no training, no checkpoint selection, no canonical multi selection,
no Track C query. Uses a dedicated dose-specific split generated from
\`${ROOT}/reports/sciplex_dose_time_artifacts_20260627/sciplex_log_dose_condition_level.csv\`
and the dose-aware all-modality H5 artifact. Any future training still requires
the CPU gate to pass and external review.

Stop rule: if posthoc fails, fix only implementation/provenance issues or close
the diagnostic as not-ready. If the CPU gate fails after posthoc, no SciPlex
dose-aware training GPU is authorized.
EOF

echo "${SESSION}" > "${RUN_DIR}/SESSION_NAME"
date '+%F %T %Z' > "${RUN_DIR}/STARTED"
rm -f "${RUN_DIR}/EXIT_CODE" "${RUN_DIR}/FINISHED"

tmux new -d -s "${SESSION}" \
  "bash -lc 'set -euo pipefail; \
    source ${ROOT}/init-scdfm.sh >/dev/null; \
    cd ${COUPLED}; \
    export CUDA_VISIBLE_DEVICES=${GPU_ID}; \
    export OMP_NUM_THREADS=${THREADS}; \
    export MKL_NUM_THREADS=${THREADS}; \
    export OPENBLAS_NUM_THREADS=${THREADS}; \
    export NUMEXPR_NUM_THREADS=${THREADS}; \
    export BLIS_NUM_THREADS=${THREADS}; \
    export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}; \
    export PERT_EMBED_SOURCE=scgpt_embed_gene; \
    export RAW_DRUG_EMB_CACHE_DIR=${DRUG_CACHE}; \
    export LATENT_DRUG_EMB_CACHE_DIR=${DRUG_CACHE}; \
    set +e; \
    ( \
      set -euo pipefail; \
      echo \"[dose-posthoc] start=\$(date) gpu=${GPU_ID}\"; \
      common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${SPLIT_FILE} --pert-means-file ${PERT_MEANS} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024); \
      ${PYTHON} -m model.latent.eval_condition_families \
        --checkpoint ${ANCHOR_CKPT} \
        --groups test_all family_drug type_drug \
        --out ${OUT_JSON} \"\${common[@]}\"; \
      ${PYTHON} ${ROOT}/ops/build_latentfm_sciplex_dose_specific_outcome_gate_20260627.py \
        --eval-json ${OUT_JSON}; \
      echo \"[dose-posthoc] finished=\$(date)\"; \
    ) > ${LOG_DIR}/posthoc.log 2>&1; \
    code=\$?; set -e; echo \$code > ${RUN_DIR}/EXIT_CODE; date '+%F %T %Z' > ${RUN_DIR}/FINISHED; exit \$code'"

tmux ls | grep "${SESSION}" || true
tail -n 20 "${LOG_DIR}/posthoc.log" 2>/dev/null || true
