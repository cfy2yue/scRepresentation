#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
RUN_NAME=xverse_support_film_retry1_condition_means_artifacts
RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/logs"
OUT_DIR="${RUN_ROOT}/condition_means"
SESSION="lfm_trackc_condition_means_artifacts_20260623"

GPU_ID="${LATENTFM_CONDITION_MEANS_GPU:-0}"
DATA_DIR="${ROOT}/dataset/latentfm_full/xverse"
SUPPORT_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
CANONICAL_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42.json"
ANCHOR_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
CANDIDATE_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_support_film_20260623/xverse_trackc_support_film_absroute_2k_seed42_retry1/best.pt"
PERT_MEANS="${DATA_DIR}/pert_means.npz"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

if [[ ! -f "${ROOT}/reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_PROTOCOL_20260623.md" ]]; then
  echo "missing anchor-gated support-teacher protocol report" >&2
  exit 2
fi
if [[ ! -f "${ROOT}/reports/LATENTFM_TRACKC_CONDITION_MEANS_ARTIFACT_CODE_PREP_20260623.md" ]]; then
  echo "missing condition-means code-prep report" >&2
  exit 2
fi
for path in "${ANCHOR_CKPT}" "${CANDIDATE_CKPT}" "${SUPPORT_SPLIT}" "${CANONICAL_SPLIT}" "${PERT_MEANS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing required artifact: ${path}" >&2
    exit 2
  fi
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<STATUS
# Run Status: latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/${RUN_NAME}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_trackc_condition_means_artifacts_20260623.sh
\`\`\`

## Runtime classification

Long GPU posthoc artifact-generation task. Detached in tmux; use 30-minute cadence for checks.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`${SESSION}\`
physical GPU: \`${GPU_ID}\`

## Log path

\`${LOG_DIR}/run.log\`

## Expected outputs

* \`${OUT_DIR}/support_anchor_split_condition_means_ode20.json\`
* \`${OUT_DIR}/support_candidate_split_condition_means_ode20.json\`
* \`${OUT_DIR}/canonical_anchor_split_test_single_condition_means_ode20.json\`
* \`${OUT_DIR}/canonical_candidate_split_test_single_condition_means_ode20.json\`
* \`${OUT_DIR}/canonical_anchor_family_gene_condition_means_ode20.json\`
* \`${OUT_DIR}/canonical_candidate_family_gene_condition_means_ode20.json\`

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

Purpose: generate default-off per-condition mean artifacts for the Track C
anchor-gated support-teacher residual CPU gate. This does not train, does not
read held-out Track C query, and does not use canonical multi for selection.
Support artifacts use the safe trainselect split; canonical artifacts are for
single/background no-harm after route design, not checkpoint selection.
STATUS

cat > "${RUN_ROOT}/run_condition_means_artifacts.sh" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/xverse_support_film_retry1_condition_means_artifacts"
OUT_DIR="${RUN_ROOT}/condition_means"
GPU_ID="${LATENTFM_CONDITION_MEANS_GPU:-0}"
DATA_DIR="${ROOT}/dataset/latentfm_full/xverse"
SUPPORT_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
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

echo "[condition-means] start $(date '+%F %T %Z')"

"${PYTHON}" -m model.latent.eval_split_groups \
  --checkpoint "${ANCHOR_CKPT}" \
  --split-file "${SUPPORT_SPLIT}" \
  --groups test test_multi \
  --out "${OUT_DIR}/support_anchor_split_condition_means_ode20.json" \
  "${common[@]}"

"${PYTHON}" -m model.latent.eval_split_groups \
  --checkpoint "${CANDIDATE_CKPT}" \
  --split-file "${SUPPORT_SPLIT}" \
  --groups test test_multi \
  --out "${OUT_DIR}/support_candidate_split_condition_means_ode20.json" \
  "${common[@]}"

"${PYTHON}" -m model.latent.eval_split_groups \
  --checkpoint "${ANCHOR_CKPT}" \
  --split-file "${CANONICAL_SPLIT}" \
  --groups test_single \
  --out "${OUT_DIR}/canonical_anchor_split_test_single_condition_means_ode20.json" \
  "${common[@]}"

"${PYTHON}" -m model.latent.eval_split_groups \
  --checkpoint "${CANDIDATE_CKPT}" \
  --split-file "${CANONICAL_SPLIT}" \
  --groups test_single \
  --out "${OUT_DIR}/canonical_candidate_split_test_single_condition_means_ode20.json" \
  "${common[@]}"

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

echo "[condition-means] finished $(date '+%F %T %Z')"
RUNNER
chmod +x "${RUN_ROOT}/run_condition_means_artifacts.sh"

tmux new -d -s "${SESSION}" \
  "LATENTFM_CONDITION_MEANS_GPU=${GPU_ID} bash ${RUN_ROOT}/run_condition_means_artifacts.sh > ${LOG_DIR}/run.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date '+%F %T %Z' > ${RUN_ROOT}/FINISHED"

echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

echo "launched ${SESSION} on GPU ${GPU_ID}"
tmux ls | grep "${SESSION}" || true
tail -n 20 "${LOG_DIR}/run.log" || true
