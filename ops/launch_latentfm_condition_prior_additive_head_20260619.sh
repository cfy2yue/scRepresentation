#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_condition_prior_additive_head_20260619
RUN_SCRIPT=${RUN_ROOT}/run_scf_prioradd005_prior010_inject_e2_4k.sh
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
SESSION=lfm_prioradd_20260619
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${RUN_ROOT}/logs"

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv \
  | tee "${RUN_ROOT}/logs/prelaunch_nvidia_smi.csv"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --need 1 \
  --max-jobs-per-gpu 3 \
  --json-only \
  > "${gpu_json}"

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
  cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Additive Head 2026-06-19

Started: $(date '+%F %T %Z')
Status: not_launched_no_gpu
GPU selection JSON: ${gpu_json}
EOF
  echo 97 > "${RUN_ROOT}/EXIT_CODE"
  date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
  exit 97
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"
chmod +x "${RUN_SCRIPT}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition-Prior Additive Head 2026-06-19

Started: $(cat "${RUN_ROOT}/STARTED")
Status: running
Runtime classification: Long GPU task.
Polling policy: no frequent polling; downstream watchers sleep 1800 seconds between checks.
tmux session: ${SESSION}
Selected physical GPU: ${gpu}
GPU selection JSON: ${gpu_json}
Run: scf_prioradd005_prior010_inject_e2_4k
Output dir: ${ROOT}/CoupledFM/output/latentfm_runs/condition_prior_additive_head_20260619/scf_prioradd005_prior010_inject_e2_4k
Log: ${RUN_ROOT}/logs/scf_prioradd005_prior010_inject_e2_4k.log
Key changes vs injection baseline:
- condition_prior_delta_loss_weight=0.10
- condition_delta_head_use_in_model=True
- condition_prior_additive_delta_loss_weight=0.05
EOF

tmux new-session -d -s "${SESSION}" \
  "cd '${RUN_ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' bash '${RUN_SCRIPT}' > '${RUN_ROOT}/logs/scf_prioradd005_prior010_inject_e2_4k.log' 2>&1; code=\$?; echo \${code} > '${RUN_ROOT}/EXIT_CODE'; date '+%F %T %Z' > '${RUN_ROOT}/FINISHED'; if [[ \${code} == 0 ]]; then status=finished; else status=failed; fi; cat > '${RUN_ROOT}/RUN_STATUS.md' <<EOF
# LatentFM Condition-Prior Additive Head 2026-06-19

Started: $(cat "${RUN_ROOT}/STARTED")
Finished: \$(cat '${RUN_ROOT}/FINISHED')
Status: \${status}
Exit code: \${code}
tmux session: ${SESSION}
Selected physical GPU: ${gpu}
Run: scf_prioradd005_prior010_inject_e2_4k
Output dir: ${ROOT}/CoupledFM/output/latentfm_runs/condition_prior_additive_head_20260619/scf_prioradd005_prior010_inject_e2_4k
Log: ${RUN_ROOT}/logs/scf_prioradd005_prior010_inject_e2_4k.log
EOF"

tmux ls | grep "${SESSION}" || true
