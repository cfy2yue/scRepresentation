#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
COUPLED=${ROOT}/CoupledFM

POSTHOC_GATE_JSON=${ROOT}/reports/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_gate_20260623.json
REQUIRED_POSTHOC_STATUS=trackc_anchor_gated_support_teacher_blend_posthoc_gate_pass

LABEL=latentfm_trackc_anchor_gated_blend_query_once_20260623_retry1
RUN_ROOT=${ROOT}/runs/${LABEL}
LOG_DIR=${RUN_ROOT}/logs
EVAL_DIR=${RUN_ROOT}/eval
SESSION=lfm_trackc_anchor_blend_query_once_retry1_20260623

ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
TEACHER_CKPT=${COUPLED}/output/latentfm_runs/xverse_trackc_support_film_20260623/xverse_trackc_support_film_absroute_2k_seed42_retry1/best.pt
TRAINSELECT_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
PERT_MEANS=${DATA_DIR}/pert_means.npz
EVALUATOR=${ROOT}/ops/evaluate_latentfm_trackc_anchor_gated_support_teacher_blend_20260623.py
SUMMARIZER=${ROOT}/ops/summarize_latentfm_trackc_anchor_gated_blend_query_once_20260623.py

QUERY_JSON=${EVAL_DIR}/anchor_gated_blend_query_once_ode20.json
QUERY_DECISION_JSON=${ROOT}/reports/latentfm_trackc_anchor_gated_blend_query_once_decision_20260623.json
QUERY_DECISION_MD=${ROOT}/reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_QUERY_ONCE_DECISION_20260623.md

NOT_BEFORE_EPOCH=$(date -d '2026-06-23 07:10:00 CST' +%s)
NOW_EPOCH=$(date +%s)
if (( NOW_EPOCH < NOT_BEFORE_EPOCH )); then
  echo "Refusing query launcher before 2026-06-23 07:10:00 CST" >&2
  exit 3
fi

if [[ ! -f "${POSTHOC_GATE_JSON}" ]]; then
  echo "Refusing query: full blend posthoc gate JSON missing: ${POSTHOC_GATE_JSON}" >&2
  exit 2
fi

posthoc_status=$(
  "${PYTHON}" - <<PY
import json
from pathlib import Path
p = Path("${POSTHOC_GATE_JSON}")
obj = json.loads(p.read_text(encoding="utf-8"))
print(obj.get("status", "missing_status"))
PY
)
if [[ "${posthoc_status}" != "${REQUIRED_POSTHOC_STATUS}" ]]; then
  echo "Refusing query: posthoc gate status is '${posthoc_status}', not '${REQUIRED_POSTHOC_STATUS}'." >&2
  exit 2
fi

if [[ -e "${RUN_ROOT}" || -e "${QUERY_DECISION_JSON}" || -e "${QUERY_DECISION_MD}" ]]; then
  echo "Refusing one-shot query because run or decision artifact already exists." >&2
  exit 2
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Refusing one-shot query because tmux session already exists: ${SESSION}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${EVAL_DIR}" "${RUN_ROOT}/scripts"

AUDIT_DIR=${RUN_ROOT}/resource_audit
mkdir -p "${AUDIT_DIR}"
{
  date '+%F %T %Z'
  free -h
  df -h /data /
  uptime
} > "${AUDIT_DIR}/cpu_ram_disk.txt"

for i in 1 2 3; do
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    > "${AUDIT_DIR}/gpu_sample_${i}.csv"
  if [[ "${i}" != "3" ]]; then
    sleep 10
  fi
done

GPU=$(
  "${PYTHON}" - <<PY
from pathlib import Path
audit = Path("${AUDIT_DIR}")
samples = []
for i in (1, 2, 3):
    rows = []
    for line in (audit / f"gpu_sample_{i}.csv").read_text().strip().splitlines():
        idx, mem, util = [x.strip() for x in line.split(",")]
        rows.append((int(idx), int(mem), int(util)))
    samples.append(rows)
by_gpu = {}
for rows in samples:
    for idx, mem, util in rows:
        by_gpu.setdefault(idx, []).append((mem, util))
empty = []
for idx, vals in sorted(by_gpu.items()):
    if len(vals) == 3 and all(mem < 4096 and util < 10 for mem, util in vals):
        empty.append(idx)
if len(empty) < 2:
    raise SystemExit("insufficient empty GPUs for one-shot query while leaving one empty")
print(empty[0])
PY
)

cat > "${RUN_ROOT}/RUN_STATUS.md" <<STATUS
# Run Status: ${LABEL}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_trackc_anchor_gated_blend_query_once_if_pass_20260623.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`${SESSION}\`

## Log path

\`${LOG_DIR}/query_eval.log\`

## Expected outputs

* \`${QUERY_JSON}\`
* \`${QUERY_DECISION_JSON}\`
* \`${QUERY_DECISION_MD}\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_DIR}/query_eval.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

One-shot held-out Track C query diagnostic for the frozen anchor-gated blend
only.  This launcher refuses unless the full support/canonical blend posthoc
gate has already passed.  Query results must not tune alpha, route, checkpoint,
threshold, or future branches.

Resource plan: one GPU selected by a fresh 3-sample empty-card audit, 8 CPU
threads. Selected physical GPU: \`${GPU}\`.
STATUS

cat > "${RUN_ROOT}/scripts/run_query_once.sh" <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${COUPLED}\${PYTHONPATH:+:\${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

echo "[query-once] start \$(date '+%F %T %Z') physical_gpu=${GPU}"
"${PYTHON}" "${EVALUATOR}" \
  --anchor-checkpoint "${ANCHOR_CKPT}" \
  --support-teacher-checkpoint "${TEACHER_CKPT}" \
  --scope heldout_query_once \
  --group-kind split \
  --groups heldout_query_multi_final_only heldout_query_multi_seen_final_only heldout_query_multi_unseen1_final_only heldout_query_multi_unseen2_final_only \
  --split-file "${TRAINSELECT_SPLIT}" \
  --data-dir "${DATA_DIR}" \
  --alpha 0.75 \
  --gpu 0 \
  --device cuda:0 \
  --ode-steps 20 \
  --max-chunk 512 \
  --eval-max-mmd-cells 2048 \
  --pert-means-file "${PERT_MEANS}" \
  --out "${QUERY_JSON}"

"${PYTHON}" "${SUMMARIZER}" \
  --query-json "${QUERY_JSON}" \
  --posthoc-gate-json "${POSTHOC_GATE_JSON}" \
  --out-json "${QUERY_DECISION_JSON}" \
  --out-md "${QUERY_DECISION_MD}"

echo "[query-once] finished \$(date '+%F %T %Z')"
RUNNER

chmod +x "${RUN_ROOT}/scripts/run_query_once.sh"

tmux new -d -s "${SESSION}" \
  "bash ${RUN_ROOT}/scripts/run_query_once.sh > ${LOG_DIR}/query_eval.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date '+%F %T %Z' > ${RUN_ROOT}/FINISHED"

echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

echo "Launched ${SESSION} on physical GPU${GPU}"
tmux ls | grep "${SESSION}" || true
tail -n 20 "${LOG_DIR}/query_eval.log" || true
