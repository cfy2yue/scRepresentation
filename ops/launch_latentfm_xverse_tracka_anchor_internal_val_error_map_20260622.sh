#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=latentfm_xverse_tracka_anchor_internal_val_error_map_20260622
RUN_ROOT=${ROOT}/runs/${RUN_NAME}
LOG_ROOT=${RUN_ROOT}/logs
SESSION=${RUN_NAME}

DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
SPLIT_FILE=${ROOT}/dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json
CHECKPOINT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
BASELINE_JSON=${ROOT}/reports/latentfm_xverse_gene_reliability_router_gate_20260622.json
SUMMARY=${ROOT}/ops/summarize_latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.py
GPU_HELPER=${ROOT}/ops/select_available_gpus.py

ANCHOR_EVAL_JSON=${RUN_ROOT}/anchor_internal_val_split_eval.json
SUMMARY_JSON=${ROOT}/reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json
SUMMARY_MD=${ROOT}/reports/LATENTFM_XVERSE_TRACKA_ANCHOR_INTERNAL_VAL_ERROR_MAP_20260622.md

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${PYTHON}" \
  "${CHECKPOINT}" \
  "${DATA_DIR}/manifest.json" \
  "${SPLIT_FILE}" \
  "${BASELINE_JSON}" \
  "${SUMMARY}" \
  "${GPU_HELPER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

if [[ -e "${ANCHOR_EVAL_JSON}" || -e "${SUMMARY_JSON}" || -e "${SUMMARY_MD}" ]]; then
  echo "Refusing to overwrite existing anchor internal-val outputs" >&2
  exit 4
fi

echo "[$(date '+%F %T %Z')] exact GPU status before Track A anchor internal-val eval" | tee "${LOG_ROOT}/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${LOG_ROOT}/gpu_launch_audit.log"

gpu_json="${LOG_ROOT}/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-jobs-per-gpu 4 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${LOG_ROOT}/gpu_selection.stderr"

assignment_json="${LOG_ROOT}/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
stable = [g for g in gpus.values() if g.get("stable_light")]
stable_count = len(stable)
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
physical_budget = min(4, stable_count) if stable_count >= 5 else max(0, min(4, stable_count - 1))
chosen = None
for idx in [int(x) for x in payload.get("candidate_order", [])]:
    gpu = gpus[idx]
    if not gpu.get("available"):
        continue
    if len(active_user | {idx}) <= physical_budget and int(gpu.get("colocation_slots_free", 0)) > 0:
        chosen = idx
        break
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "stable_light_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "active_user_gpus": sorted(active_user),
    "chosen_gpu": chosen,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if chosen is None:
    reasons.append("no GPU slot available under leave-one-empty and max-4-physical rules")
if float(system.get("mem_available_gib") or 0.0) < 64.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 64.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 5)
PY

GPU="$("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["chosen_gpu"])
PY
)"

run_script="${RUN_ROOT}/run_anchor_internal_val_error_map.sh"
cat > "${run_script}" <<EOF
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

${PYTHON} -m model.latent.eval_split_groups \\
  --checkpoint ${CHECKPOINT} \\
  --data-dir ${DATA_DIR} \\
  --biflow-dir ${BIFLOW_DIR} \\
  --split-file ${SPLIT_FILE} \\
  --groups internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy \\
  --out ${ANCHOR_EVAL_JSON} \\
  --gpu 0 \\
  --ode-steps 20 \\
  --max-chunk 256 \\
  --eval-max-conditions 0 \\
  --eval-max-conditions-per-dataset 0 \\
  --eval-max-mse-cells 1024 \\
  --eval-max-mmd-cells 1024

${PYTHON} ${SUMMARY} \\
  --anchor-eval-json ${ANCHOR_EVAL_JSON} \\
  --baseline-json ${BASELINE_JSON} \\
  --out-json ${SUMMARY_JSON} \\
  --out-md ${SUMMARY_MD} \\
  --n-boot 2000 \\
  --seed 42
EOF
chmod +x "${run_script}"

date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

Physical GPU: \`${GPU}\`

## Log path

\`${LOG_ROOT}/run.log\`

## Expected outputs

* \`${ANCHOR_EVAL_JSON}\`
* \`${SUMMARY_JSON}\`
* \`${SUMMARY_MD}\`
* \`${RUN_ROOT}/EXIT_CODE\`
* \`${RUN_ROOT}/FINISHED\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_ROOT}/run.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Hypothesis: inspect whether the xverse 8k anchor is worse than train-only
\`gene_raw_mean\`/\`dataset_mean\` on the train-only Track A internal-val split
and whether any non-closed mechanism is visible. This is an audit/posthoc job,
not checkpoint selection. It evaluates only
\`internal_val_cross_background_seen_gene_proxy\` and
\`internal_val_family_gene_proxy\` from
\`split_seed42_xverse_trainonly_crossbg_val_v2.json\`. It does not evaluate
canonical test, canonical multi, or Track C query.

Failure/stop rule: if the summary status is
\`anchor_internal_val_map_no_gpu_mechanism\`, do not launch a GPU training smoke
from this error map.
EOF

tmux new -d -s "${SESSION}" \
  "bash -lc 'bash ${run_script} > ${LOG_ROOT}/run.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/FINISHED; exit \$rc'"

tmux ls | tee "${LOG_ROOT}/tmux_ls_after_launch.txt"
tail -n 20 "${LOG_ROOT}/run.log" || true
