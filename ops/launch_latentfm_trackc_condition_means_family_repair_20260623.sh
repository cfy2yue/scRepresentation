#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
SOURCE_RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/xverse_support_film_retry1_condition_means_artifacts"
RUN_NAME=xverse_support_film_retry1_condition_means_family_repair
RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/logs"
OUT_DIR="${SOURCE_RUN_ROOT}/condition_means"
SESSION="lfm_trackc_condition_means_family_repair_20260623"

GPU_ID="${LATENTFM_CONDITION_MEANS_REPAIR_GPU:-0}"
DATA_DIR="${ROOT}/dataset/latentfm_full/xverse"
CANONICAL_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42.json"
ANCHOR_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
CANDIDATE_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_support_film_20260623/xverse_trackc_support_film_absroute_2k_seed42_retry1/best.pt"
PERT_MEANS="${DATA_DIR}/pert_means.npz"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

for path in \
  "${OUT_DIR}/support_anchor_split_condition_means_ode20.json" \
  "${OUT_DIR}/support_candidate_split_condition_means_ode20.json" \
  "${OUT_DIR}/canonical_anchor_split_test_single_condition_means_ode20.json" \
  "${OUT_DIR}/canonical_candidate_split_test_single_condition_means_ode20.json" \
  "${ANCHOR_CKPT}" \
  "${CANDIDATE_CKPT}" \
  "${CANONICAL_SPLIT}" \
  "${PERT_MEANS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required artifact: ${path}" >&2
    exit 2
  fi
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<STATUS
# Run Status: latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/${RUN_NAME}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_trackc_condition_means_family_repair_20260623.sh
\`\`\`

## Runtime classification

Long GPU posthoc artifact-generation repair task. Detached in tmux; use 30-minute cadence for checks.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`${SESSION}\`
physical GPU: \`${GPU_ID}\`

## Log path

\`${LOG_DIR}/run.log\`

## Expected outputs

* \`${OUT_DIR}/canonical_anchor_family_gene_condition_means_ode20.json\`
* \`${OUT_DIR}/canonical_candidate_family_gene_condition_means_ode20.json\`
* \`${ROOT}/reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_CPU_GATE_20260623.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_DIR}/run.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Repair for the first condition-means artifact job, which produced split/support
artifacts but failed when family eval lacked \`--pert-means-file\` support.
This repair only generates the missing canonical family_gene condition-means
artifacts, then runs the fail-closed CPU gate summarizer. It does not train,
read held-out Track C query, or use canonical multi selection.
STATUS

cat > "${RUN_ROOT}/run_family_repair.sh" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
SOURCE_RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/xverse_support_film_retry1_condition_means_artifacts"
OUT_DIR="${SOURCE_RUN_ROOT}/condition_means"
GPU_ID="${LATENTFM_CONDITION_MEANS_REPAIR_GPU:-0}"
DATA_DIR="${ROOT}/dataset/latentfm_full/xverse"
CANONICAL_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42.json"
ANCHOR_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
CANDIDATE_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_support_film_20260623/xverse_trackc_support_film_absroute_2k_seed42_retry1/best.pt"
PERT_MEANS="${DATA_DIR}/pert_means.npz"

export PYTHONPATH="${ROOT}/CoupledFM${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

common=(--data-dir "${DATA_DIR}" --gpu 0 --device cuda:0 --ode-steps 20 --max-chunk 512 --eval-max-mmd-cells 2048 --pert-means-file "${PERT_MEANS}" --save-condition-means)

echo "[condition-means-family-repair] start $(date '+%F %T %Z')"

"${PYTHON}" -m model.latent.eval_condition_families \
  --checkpoint "${ANCHOR_CKPT}" \
  --split-file "${CANONICAL_SPLIT}" \
  --groups family_gene \
  --out "${OUT_DIR}/canonical_anchor_family_gene_condition_means_ode20.json" \
  "${common[@]}"

"${PYTHON}" -m model.latent.eval_condition_families \
  --checkpoint "${CANDIDATE_CKPT}" \
  --split-file "${CANONICAL_SPLIT}" \
  --groups family_gene \
  --out "${OUT_DIR}/canonical_candidate_family_gene_condition_means_ode20.json" \
  "${common[@]}"

"${PYTHON}" "${ROOT}/ops/summarize_latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.py" \
  --run-root "${SOURCE_RUN_ROOT}"

echo "[condition-means-family-repair] finished $(date '+%F %T %Z')"
RUNNER
chmod +x "${RUN_ROOT}/run_family_repair.sh"

tmux new -d -s "${SESSION}" \
  "LATENTFM_CONDITION_MEANS_REPAIR_GPU=${GPU_ID} bash ${RUN_ROOT}/run_family_repair.sh > ${LOG_DIR}/run.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date '+%F %T %Z' > ${RUN_ROOT}/FINISHED"

echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

echo "launched ${SESSION} on GPU ${GPU_ID}"
tmux ls | grep "${SESSION}" || true
tail -n 20 "${LOG_DIR}/run.log" || true
