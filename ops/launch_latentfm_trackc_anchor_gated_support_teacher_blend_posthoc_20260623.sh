#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
RUN_NAME=xverse_support_film_retry1_anchor_gated_blend_posthoc_ode20_retry1
RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/logs"
OUT_DIR="${RUN_ROOT}/posthoc_eval"
SESSION=lfm_trackc_anchor_gate_blend_posthoc_retry1_20260623

ANCHOR_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
TEACHER_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_support_film_20260623/xverse_trackc_support_film_absroute_2k_seed42_retry1/best.pt"
SUPPORT_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
CANONICAL_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42.json"
DATA_DIR="${ROOT}/dataset/latentfm_full/xverse"
PERT_MEANS="${DATA_DIR}/pert_means.npz"
EVAL_SCRIPT="${ROOT}/ops/evaluate_latentfm_trackc_anchor_gated_support_teacher_blend_20260623.py"
SUMMARY_SCRIPT="${ROOT}/ops/summarize_latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623.py"
REPORT_JSON="${ROOT}/reports/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_gate_20260623.json"
REPORT_MD="${ROOT}/reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_BLEND_POSTHOC_GATE_20260623.md"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${RUN_ROOT}/scripts"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

cat > "${RUN_ROOT}/RUN_STATUS.md" <<STATUS
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`${SESSION}\`

## Log path

\`${LOG_DIR}/run.log\`

## Expected outputs

* \`${OUT_DIR}/support_trainselect_support_val_multi_blend_ode20.json\`
* \`${OUT_DIR}/canonical_test_single_blend_ode20.json\`
* \`${OUT_DIR}/canonical_family_gene_blend_ode20.json\`
* \`${REPORT_MD}\`
* \`${REPORT_JSON}\`

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

Hypothesis: the frozen support-teacher residual is useful on safe Track C
support-val rows when applied as \`anchor + 0.75 * residual\`, while canonical
single/family no-harm is exact because canonical scope uses \`gate=0\`.

Resource plan: three detached posthoc eval processes, one each on physical
GPU0/GPU1/GPU2, with 4 CPU threads per process (12 total LatentFM threads).
This follows the 2026-06-23 06:35-06:36 CST audit: GPU0-5 were empty across
three samples; GPU6/7 had other-user compute or >4GiB memory; RAM available was
about 420GiB; /data had about 2.2T free. CPU load was high but on a 384-core
host, and LatentFM is capped here to 12 threads.

Validation gate:

* support split only: \`${SUPPORT_SPLIT}\`
* no held-out Track C query, no canonical multi selection
* support equal-dataset mean pearson_pert delta >= +0.02
* support pp bootstrap p_harm <= 0.10
* support unbiased and biased MMD deltas <= +0.005
* support MMD bootstrap p_harm <= 0.10
* canonical test_single/family_gene gate=0 max absolute MMD/Pearson delta <= 1e-8

Failure close rule: if the report status is not
\`trackc_anchor_gated_support_teacher_blend_posthoc_gate_pass\`, do not run
held-out query and do not claim multi capability.
STATUS

cat > "${RUN_ROOT}/scripts/run_blend_posthoc.sh" <<'RUNNER'
#!/usr/bin/env bash
set -u

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
RUN_ROOT="${ROOT}/runs/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623/xverse_support_film_retry1_anchor_gated_blend_posthoc_ode20_retry1"
LOG_DIR="${RUN_ROOT}/logs"
OUT_DIR="${RUN_ROOT}/posthoc_eval"
ANCHOR_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
TEACHER_CKPT="${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_support_film_20260623/xverse_trackc_support_film_absroute_2k_seed42_retry1/best.pt"
SUPPORT_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
CANONICAL_SPLIT="${ROOT}/dataset/biFlow_data/split_seed42.json"
DATA_DIR="${ROOT}/dataset/latentfm_full/xverse"
PERT_MEANS="${DATA_DIR}/pert_means.npz"
EVAL_SCRIPT="${ROOT}/ops/evaluate_latentfm_trackc_anchor_gated_support_teacher_blend_20260623.py"
SUMMARY_SCRIPT="${ROOT}/ops/summarize_latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623.py"
REPORT_JSON="${ROOT}/reports/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_gate_20260623.json"
REPORT_MD="${ROOT}/reports/LATENTFM_TRACKC_ANCHOR_GATED_SUPPORT_TEACHER_BLEND_POSTHOC_GATE_20260623.md"

export PYTHONPATH="${ROOT}/CoupledFM${PYTHONPATH:+:${PYTHONPATH}}"

common=(
  --anchor-checkpoint "${ANCHOR_CKPT}"
  --support-teacher-checkpoint "${TEACHER_CKPT}"
  --data-dir "${DATA_DIR}"
  --alpha 0.75
  --ode-steps 20
  --max-chunk 512
  --eval-max-mmd-cells 2048
  --pert-means-file "${PERT_MEANS}"
)

run_eval() {
  local label="$1"
  local gpu="$2"
  local scope="$3"
  local kind="$4"
  local group="$5"
  local split="$6"
  local out="$7"
  local log="$8"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    echo "[${label}] start $(date '+%F %T %Z') physical_gpu=${gpu}"
    "${PYTHON}" "${EVAL_SCRIPT}" \
      "${common[@]}" \
      --scope "${scope}" \
      --group-kind "${kind}" \
      --groups "${group}" \
      --split-file "${split}" \
      --gpu 0 \
      --device cuda:0 \
      --out "${out}"
    local rc=$?
    echo "[${label}] finished rc=${rc} $(date '+%F %T %Z')"
    echo "${rc}" > "${RUN_ROOT}/${label}.EXIT_CODE"
    exit "${rc}"
  ) > "${log}" 2>&1 &
  RUN_EVAL_PID=$!
}

echo "[blend-posthoc] start $(date '+%F %T %Z')"

run_eval support_val_multi 0 support_trainselect split support_val_multi "${SUPPORT_SPLIT}" "${OUT_DIR}/support_trainselect_support_val_multi_blend_ode20.json" "${LOG_DIR}/support_val_multi.log"
pid_support="${RUN_EVAL_PID}"
run_eval canonical_test_single 1 canonical_noharm split test_single "${CANONICAL_SPLIT}" "${OUT_DIR}/canonical_test_single_blend_ode20.json" "${LOG_DIR}/canonical_test_single.log"
pid_test_single="${RUN_EVAL_PID}"
run_eval canonical_family_gene 2 canonical_noharm family family_gene "${CANONICAL_SPLIT}" "${OUT_DIR}/canonical_family_gene_blend_ode20.json" "${LOG_DIR}/canonical_family_gene.log"
pid_family_gene="${RUN_EVAL_PID}"

rc=0
wait "${pid_support}" || rc=1
wait "${pid_test_single}" || rc=1
wait "${pid_family_gene}" || rc=1

if [[ "${rc}" != "0" ]]; then
  echo "[blend-posthoc] at least one eval failed; skipping summarizer"
  exit "${rc}"
fi

"${PYTHON}" "${SUMMARY_SCRIPT}" \
  --support-json "${OUT_DIR}/support_trainselect_support_val_multi_blend_ode20.json" \
  --canonical-test-single-json "${OUT_DIR}/canonical_test_single_blend_ode20.json" \
  --canonical-family-gene-json "${OUT_DIR}/canonical_family_gene_blend_ode20.json" \
  --support-group support_val_multi \
  --out-json "${REPORT_JSON}" \
  --out-md "${REPORT_MD}"

echo "[blend-posthoc] finished $(date '+%F %T %Z')"
RUNNER

chmod +x "${RUN_ROOT}/scripts/run_blend_posthoc.sh"

tmux new -d -s "${SESSION}" \
  "bash ${RUN_ROOT}/scripts/run_blend_posthoc.sh > ${LOG_DIR}/run.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date '+%F %T %Z' > ${RUN_ROOT}/FINISHED"

echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

echo "launched ${SESSION}"
tmux ls | grep "${SESSION}" || true
tail -n 20 "${LOG_DIR}/run.log" || true
