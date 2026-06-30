#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${LATENTFM_RESPONSE_RUN_ROOT:-${ROOT}/runs/latentfm_response_normalization_20260621}
OUT_ROOT=${LATENTFM_RESPONSE_OUT_ROOT:-${COUPLED}/output/latentfm_runs/response_normalization_20260621}
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
ANCHOR_DIR=${COUPLED}/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k
ARTIFACT=${LATENTFM_RESPONSE_ARTIFACT:-${ROOT}/runs/latentfm_response_normalization_20260621/artifacts/scfoundation_trainonly_dataset_scale_pca32.npz}
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
SUMMARY=${ROOT}/ops/summarize_latentfm_response_geometry_smoke_20260621.py
BOOTSTRAP_RUNNER=${ROOT}/ops/run_latentfm_posthoc_bootstrap_from_manifest_20260621.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=${LATENTFM_RESPONSE_RUN_NAME:-scf_response_dataset_scale_pca32_aux1_4k}
SUMMARY_JSON=${LATENTFM_RESPONSE_SUMMARY_JSON:-${ROOT}/reports/latentfm_response_geometry_smoke_summary_20260621.json}
SUMMARY_MD=${LATENTFM_RESPONSE_SUMMARY_MD:-${ROOT}/reports/LATENTFM_RESPONSE_GEOMETRY_SMOKE_SUMMARY_20260621.md}
BOOTSTRAP_DIR=${LATENTFM_RESPONSE_BOOTSTRAP_DIR:-${ROOT}/reports/latentfm_response_geometry_smoke_bootstrap_20260621}
POSTHOC_TITLE=${LATENTFM_RESPONSE_POSTHOC_TITLE:-latentfm_response_geometry_posthoc_20260621}
LOG_DIR=${RUN_ROOT}/logs
mkdir -p "${LOG_DIR}" "${ROOT}/reports"
rm -f "${RUN_ROOT}/POSTHOC_EXIT_CODE" "${RUN_ROOT}/POSTHOC_FINISHED"
date '+%F %T %Z' > "${RUN_ROOT}/POSTHOC_STARTED"

cat > "${RUN_ROOT}/POSTHOC_RUN_STATUS.md" <<EOF
# Run Status: ${POSTHOC_TITLE}

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_response_geometry_posthoc_20260621.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation. Check at most every 30 minutes.

## Start time

$(cat "${RUN_ROOT}/POSTHOC_STARTED")

## Log path

\`${LOG_DIR}/posthoc.log\`

## Expected outputs

* \`${SUMMARY_MD}\`
* \`${SUMMARY_JSON}\`

## Current status

Waiting for response-geometry smoke training to finish.
EOF

trap 'rc=$?; echo "$rc" > "${RUN_ROOT}/POSTHOC_EXIT_CODE"; date "+%F %T %Z" > "${RUN_ROOT}/POSTHOC_FINISHED"; exit "$rc"' EXIT

{
  echo "[$(date '+%F %T %Z')] response geometry posthoc watcher start"
  while [[ ! -f "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE" ]]; do
    echo "[$(date '+%F %T %Z')] training still running; next posthoc check in 1800s"
    sleep 1800
  done
  code="$(cat "${RUN_ROOT}/${RUN_NAME}.EXIT_CODE")"
  echo "[$(date '+%F %T %Z')] train ${RUN_NAME} exit=${code}"
  if [[ "${code}" != "0" ]]; then
    echo "training failed for ${RUN_NAME}; skip posthoc" >&2
    exit "${code}"
  fi

  echo "[$(date '+%F %T %Z')] exact GPU status before response posthoc"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  gpu_json="${LOG_DIR}/posthoc_gpu_selection_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" "${GPU_HELPER}" \
    --samples 3 \
    --interval-seconds 10 \
    --util-threshold-pct 10 \
    --memory-threshold-mib 4096 \
    --max-jobs-per-gpu 4 \
    --need 1 \
    --json-only \
    > "${gpu_json}" 2> "${LOG_DIR}/posthoc_gpu_selection.stderr"
  gpu="$("${PYTHON}" - "${gpu_json}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = payload.get("suggested_job_gpus") or []
print(chosen[0] if chosen else "")
PY
)"
  if [[ -z "${gpu}" ]]; then
    echo "No GPU selected for posthoc; see ${gpu_json}" >&2
    exit 3
  fi

  source "${ROOT}/init-scdfm.sh" >/dev/null
  cd "${COUPLED}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  export BLIS_NUM_THREADS=4
  export PYTHONPATH="${COUPLED}:${PYTHONPATH:-}"
  export PERT_EMBED_SOURCE=scgpt_embed_gene

  run_dir="${OUT_ROOT}/${RUN_NAME}"
  cand_posthoc="${run_dir}/posthoc_eval"
  base_posthoc="${RUN_ROOT}/baseline_posthoc/${RUN_NAME}"
  mkdir -p "${cand_posthoc}" "${base_posthoc}"
  test -s "${run_dir}/best.pt"
  test -s "${ANCHOR_DIR}/best.pt"

  base_split_json="${base_posthoc}/split_group_eval_anchor_canonical_ode20_mse1024_mmd1024_stablecaps.json"
  base_family_json="${base_posthoc}/condition_family_eval_anchor_canonical_ode20_mse1024_mmd1024_stablecaps.json"
  run_split_json="${cand_posthoc}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
  run_family_json="${cand_posthoc}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json"

  "${PYTHON}" -m model.latent.eval_split_groups \
    --checkpoint "${ANCHOR_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${BIFLOW_DIR}" \
    --split-file "${CANONICAL_SPLIT}" \
    --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
    --out "${base_split_json}" \
    --gpu 0 \
    --ode-steps 20 \
    --max-chunk 512 \
    --eval-max-conditions 256 \
    --eval-max-conditions-per-dataset 12 \
    --eval-max-mse-cells 1024 \
    --eval-max-mmd-cells 1024

  "${PYTHON}" -m model.latent.eval_condition_families \
    --checkpoint "${ANCHOR_DIR}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${BIFLOW_DIR}" \
    --split-file "${CANONICAL_SPLIT}" \
    --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
    --out "${base_family_json}" \
    --gpu 0 \
    --ode-steps 20 \
    --max-chunk 512 \
    --eval-max-conditions 256 \
    --eval-max-conditions-per-dataset 12 \
    --eval-max-mse-cells 1024 \
    --eval-max-mmd-cells 1024

  "${PYTHON}" -m model.latent.eval_split_groups \
    --checkpoint "${run_dir}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${BIFLOW_DIR}" \
    --split-file "${CANONICAL_SPLIT}" \
    --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
    --out "${run_split_json}" \
    --gpu 0 \
    --ode-steps 20 \
    --max-chunk 512 \
    --eval-max-conditions 256 \
    --eval-max-conditions-per-dataset 12 \
    --eval-max-mse-cells 1024 \
    --eval-max-mmd-cells 1024

  "${PYTHON}" -m model.latent.eval_condition_families \
    --checkpoint "${run_dir}/best.pt" \
    --data-dir "${DATA_DIR}" \
    --biflow-dir "${BIFLOW_DIR}" \
    --split-file "${CANONICAL_SPLIT}" \
    --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
    --out "${run_family_json}" \
    --gpu 0 \
    --ode-steps 20 \
    --max-chunk 512 \
    --eval-max-conditions 256 \
    --eval-max-conditions-per-dataset 12 \
    --eval-max-mse-cells 1024 \
    --eval-max-mmd-cells 1024

  manifest="${RUN_ROOT}/posthoc_manifest.json"
  "${PYTHON}" - "${manifest}" "${base_split_json}" "${base_family_json}" "${run_split_json}" "${run_family_json}" <<PY
import json
import sys
from pathlib import Path
payload = {
    "run_name": "${RUN_NAME}",
    "anchor_checkpoint": "${ANCHOR_DIR}/best.pt",
    "candidate_checkpoint": "${OUT_ROOT}/${RUN_NAME}/best.pt",
    "split_file": "${CANONICAL_SPLIT}",
    "response_normalization_artifact": "${ARTIFACT}",
    "baseline_split_json": sys.argv[2],
    "baseline_family_json": sys.argv[3],
    "run_split_json": sys.argv[4],
    "run_family_json": sys.argv[5],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  "${PYTHON}" "${SUMMARY}" \
    --manifest "${manifest}" \
    --out-json "${SUMMARY_JSON}" \
    --out-md "${SUMMARY_MD}"
  "${PYTHON}" "${BOOTSTRAP_RUNNER}" \
    --manifest "${manifest}" \
    --out-dir "${BOOTSTRAP_DIR}" \
    --n-boot 2000 \
    --seed 42
  echo "[$(date '+%F %T %Z')] response geometry posthoc done"
} 2>&1 | tee "${LOG_DIR}/posthoc.log"
