#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_priorcorr_shifted_mmd_gate_20260621
LOG_DIR=${RUN_ROOT}/logs
SESSION=latentfm_priorcorr_shifted_mmd_gate_20260621
SCRIPT=${ROOT}/ops/evaluate_latentfm_prior_correction_20260619.py
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

CHECKPOINT=${ROOT}/CoupledFM/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/best.pt
OUT_MD=${ROOT}/reports/LATENTFM_PRIOR_CORRECTION_SHIFTED_MMD_GATE_SCF_INJECT_20260621.md
OUT_CSV=${ROOT}/reports/latentfm_prior_correction_shifted_mmd_gate_scf_inject_20260621.csv
OUT_JSON=${ROOT}/reports/latentfm_prior_correction_shifted_mmd_gate_scf_inject_20260621.json

mkdir -p "${LOG_DIR}" "${ROOT}/reports"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"

for required in "${SCRIPT}" "${CHECKPOINT}" "${GPU_HELPER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

echo "[$(date '+%F %T %Z')] exact GPU status before shifted-MMD prior-correction evaluator" | tee "${LOG_DIR}/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${LOG_DIR}/gpu_launch_audit.log"

gpu_json="${LOG_DIR}/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --need 1 \
  --max-jobs-per-gpu 3 \
  --json-only \
  > "${gpu_json}" 2> "${LOG_DIR}/gpu_selection.stderr"

gpu="$("${PYTHON}" - "${gpu_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = payload.get("suggested_job_gpus") or []
print(chosen[0] if chosen else "")
PY
)"
if [[ -z "${gpu}" ]]; then
  echo "No GPU selected by helper; see ${gpu_json}" >&2
  exit 4
fi

resource_audit="${LOG_DIR}/resource_audit_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${resource_audit}" <<'PY'
import json
import os
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
system = payload.get("system") or {}
min_mem = float(os.environ.get("MIN_LAUNCH_MEM_AVAILABLE_GIB", "64"))
max_load = float(os.environ.get("MAX_LAUNCH_LOAD1_PER_CPU", "2.0"))
mem = float(system.get("mem_available_gib") or 0.0)
load = float(system.get("load1_per_cpu") or 0.0)
audit = {
    "status": "pass",
    "min_mem_available_gib": min_mem,
    "max_load1_per_cpu": max_load,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if mem < min_mem:
    reasons.append(f"MemAvailable {mem:.1f} GiB < {min_mem:.1f} GiB")
if load > max_load:
    reasons.append(f"load1_per_cpu {load:.3f} > {max_load:.3f}")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 5)
PY

date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"

cat > "${RUN_ROOT}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
${PYTHON} ${SCRIPT} \\
  --checkpoint ${CHECKPOINT} \\
  --label scf_prior010_inject_e2_4k_priorcorr_shifted_mmd \\
  --datasets NormanWeissman2019_filtered Wessels GasperiniShendure2019_lowMOI \\
  --groups test_multi_seen test_multi_unseen1 test_multi_unseen2 \\
  --alphas 0.0 0.25 0.5 0.75 1.0 \\
  --k-values 5 10 \\
  --ode-steps 20 \\
  --eval-max-cells 128 \\
  --prior-max-cells 512 \\
  --max-chunk 256 \\
  --gpu 0 \\
  --compute-shifted-mmd \\
  --out-md ${OUT_MD} \\
  --out-csv ${OUT_CSV} \\
  --out-json ${OUT_JSON}
EOF
chmod +x "${RUN_ROOT}/run.sh"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_priorcorr_shifted_mmd_gate_20260621

## Command

\`\`\`bash
bash ${BASH_SOURCE[0]}
\`\`\`

## Runtime classification

Long/unknown GPU evaluation task. Use 30-minute cadence for checks.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux: \`${SESSION}\`

## GPU

Physical GPU${gpu}, selected by exact \`nvidia-smi\` plus 3-sample shared-GPU helper.

## Log path

\`${LOG_DIR}/run.log\`

## Expected outputs

* \`${OUT_MD}\`
* \`${OUT_CSV}\`
* \`${OUT_JSON}\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo still-running
tail -n 50 ${LOG_DIR}/run.log
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

This evaluator applies prior-correction as a distribution shift:
\`corrected_cells = predicted_cells + corrected_mean - model_mean\`.
It is not training and not a promotion claim. It is the MMD-aware pre-GPU gate
for deciding whether a narrow priorcorr pp-frame training smoke is justified.
EOF

tmux new-session -d -s "${SESSION}" \
  "bash '${RUN_ROOT}/run.sh' > '${LOG_DIR}/run.log' 2>&1; code=\$?; echo \${code} > '${RUN_ROOT}/EXIT_CODE'; date '+%F %T %Z' > '${RUN_ROOT}/FINISHED'; exit \${code}"

echo "launched ${SESSION} on physical GPU${gpu}"
echo "RUN_ROOT=${RUN_ROOT}"
