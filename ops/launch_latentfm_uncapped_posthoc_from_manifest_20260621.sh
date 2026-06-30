#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUNNER=${ROOT}/ops/run_latentfm_uncapped_posthoc_from_manifest_20260621.py
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

MANIFEST=${MANIFEST:?Set MANIFEST=/path/to/posthoc_or_launch_manifest.json}
LABEL=${LABEL:-latentfm_uncapped_posthoc_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${ROOT}/reports/${LABEL}}
RUN_ROOT=${ROOT}/runs/${LABEL}
LOG_DIR=${RUN_ROOT}/logs
ONLY_RUN_NAME=${ONLY_RUN_NAME:-}
EVAL_MAX_MSE_CELLS=${EVAL_MAX_MSE_CELLS:-0}
EVAL_MAX_MMD_CELLS=${EVAL_MAX_MMD_CELLS:-0}
SPLIT_GROUPS=${SPLIT_GROUPS:-}
FAMILY_GROUPS=${FAMILY_GROUPS:-}

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
for required in "${RUNNER}" "${GPU_HELPER}" "${MANIFEST}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

echo "[$(date '+%F %T %Z')] exact GPU status before uncapped posthoc launch" | tee "${LOG_DIR}/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${LOG_DIR}/gpu_launch_audit.log"

gpu_json="${LOG_DIR}/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-jobs-per-gpu 4 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${LOG_DIR}/gpu_selection.stderr"

assignment_json="${LOG_DIR}/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
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
if float(system.get("mem_available_gib") or 0.0) < 96.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 96.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

GPU="$("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["chosen_gpu"])
PY
)"

run_script="${RUN_ROOT}/run_uncapped_posthoc.sh"
cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${ROOT}/CoupledFM
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${ROOT}/CoupledFM:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene

cmd=(
  ${PYTHON} ${RUNNER}
  --manifest ${MANIFEST}
  --out-dir ${OUT_DIR}
  --python ${PYTHON}
  --gpu 0
  --ode-steps 20
  --max-chunk 512
  --eval-max-mse-cells ${EVAL_MAX_MSE_CELLS}
  --eval-max-mmd-cells ${EVAL_MAX_MMD_CELLS}
)
if [[ -n "${ONLY_RUN_NAME}" ]]; then
  cmd+=(--only-run-name "${ONLY_RUN_NAME}")
fi
if [[ -n "${SPLIT_GROUPS}" ]]; then
  # shellcheck disable=SC2206
  split_groups_arr=(${SPLIT_GROUPS})
  cmd+=(--split-groups "\${split_groups_arr[@]}")
fi
if [[ -n "${FAMILY_GROUPS}" ]]; then
  # shellcheck disable=SC2206
  family_groups_arr=(${FAMILY_GROUPS})
  cmd+=(--family-groups "\${family_groups_arr[@]}")
fi
"\${cmd[@]}"
EOF
chmod +x "${run_script}"

SESSION=${LABEL}
tmux new -d -s "${SESSION}" \
  "bash -lc 'bash ${run_script} > ${LOG_DIR}/uncapped_posthoc.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/FINISHED; exit \$rc'"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: ${LABEL}

## Command

\`\`\`bash
MANIFEST=${MANIFEST} LABEL=${LABEL} OUT_DIR=${OUT_DIR} ONLY_RUN_NAME=${ONLY_RUN_NAME} EVAL_MAX_MSE_CELLS=${EVAL_MAX_MSE_CELLS} EVAL_MAX_MMD_CELLS=${EVAL_MAX_MMD_CELLS} SPLIT_GROUPS="${SPLIT_GROUPS}" FAMILY_GROUPS="${FAMILY_GROUPS}" bash ${ROOT}/ops/launch_latentfm_uncapped_posthoc_from_manifest_20260621.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation. Check at most every 30 minutes.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`, physical GPU${GPU}

## Log path

\`${LOG_DIR}/uncapped_posthoc.log\`

## Expected outputs

* \`${OUT_DIR}/uncapped_posthoc_index.json\`

## How to check manually

\`\`\`bash
tmux ls | grep '${SESSION}' || true
tail -n 50 ${LOG_DIR}/uncapped_posthoc.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Promotion-stage uncapped posthoc. All eval caps are set to zero in the runner.
Condition caps are always zero. Cell caps are controlled by
\`EVAL_MAX_MSE_CELLS\` and \`EVAL_MAX_MMD_CELLS\`.
Optional \`SPLIT_GROUPS\` and \`FAMILY_GROUPS\` restrict evaluated groups;
empty values preserve the historical default including canonical multi
diagnostic groups.
EOF

echo "Launched uncapped posthoc ${LABEL} on physical GPU${GPU}"
echo "RUN_STATUS: ${RUN_ROOT}/RUN_STATUS.md"
