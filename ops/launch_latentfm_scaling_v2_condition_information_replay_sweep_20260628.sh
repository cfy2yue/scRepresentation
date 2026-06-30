#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

BASE_LAUNCHER=${ROOT}/ops/launch_latentfm_scaling_v2_condition_information_highlow_smoke_20260628.sh
DECISION_JSON=${ROOT}/reports/scaling_v2_condition_information_highlow_smoke_20260628/latentfm_scaling_v2_condition_information_highlow_decision_20260628.json
HIGH_SPLIT=${ROOT}/reports/scaling_v2_condition_information_draft_splits_20260628/draft_split_seed42_xverse_info_composite_high_from_cap120_all_v2.json
LOW_SPLIT=${ROOT}/reports/scaling_v2_condition_information_draft_splits_20260628/draft_split_seed42_xverse_info_composite_low_from_cap120_all_v2.json
PACKET_JSON=${ROOT}/reports/scaling_v2_condition_information_packet_audit_20260628/latentfm_scaling_v2_condition_information_packet_audit_20260628.json

SWEEP_ROOT=${ROOT}/runs/latentfm_scaling_v2_condition_information_replay_sweep_20260628
LOG_ROOT=${ROOT}/logs/latentfm_scaling_v2_condition_information_replay_sweep_20260628
REPORT_ROOT=${ROOT}/reports/scaling_v2_condition_information_replay_sweep_20260628
OUT_ROOT_BASE=${COUPLED}/output/latentfm_runs/scaling_v2_condition_information_replay_sweep_20260628

mkdir -p "${SWEEP_ROOT}/logs" "${LOG_ROOT}" "${REPORT_ROOT}" "${OUT_ROOT_BASE}"

for required in "${BASE_LAUNCHER}" "${DECISION_JSON}" "${HIGH_SPLIT}" "${LOW_SPLIT}" "${PACKET_JSON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - "${DECISION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = (payload.get("decision") or {}).get("status")
if status != "scaling_v2_condition_information_highlow_fail_or_mechanism_only_no_gpu":
    raise SystemExit(f"baseline high/low decision is not the expected fail/mutate state: {status!r}")
PY

cat > "${SWEEP_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_scaling_v2_condition_information_replay_sweep_20260628

## Hypothesis

The seed42 condition-information high/low smoke failed partly because full
finetuning drifted away from the xVERSE anchor. Anchor replay plus lower LR may
preserve no-harm behavior; if the high split still trails low after replay,
the axis/design rather than simple drift is the likely problem.

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_scaling_v2_condition_information_replay_sweep_20260628.sh
\`\`\`

## Runtime classification

Long GPU sweep. Detached child jobs are launched through the base high/low
launcher. Do not poll the same child logs more often than every 30 minutes.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

Expected sessions:

* \`lfm_scv2rep05_high_2000_s42\`
* \`lfm_scv2rep05_low_2000_s42\`
* \`lfm_scv2rep10_high_2000_s42\`
* \`lfm_scv2rep10_low_2000_s42\`

## Log path

\`${LOG_ROOT}/<batch>/<run_name>/launcher.log\`

## Expected outputs

* \`${REPORT_ROOT}/w05/latentfm_scaling_v2_condition_information_replay_w05_decision_20260628.md\`
* \`${REPORT_ROOT}/w10/latentfm_scaling_v2_condition_information_replay_w10_decision_20260628.md\`

## How to check manually

\`\`\`bash
tmux ls
cat ${SWEEP_ROOT}/w05/*/EXIT_CODE 2>/dev/null || true
cat ${SWEEP_ROOT}/w10/*/EXIT_CODE 2>/dev/null || true
nvidia-smi
\`\`\`

## Current status

Starting.

## Notes

- Resource policy: max 2 physical GPUs, max 2 LatentFM train jobs/GPU, CPU <=24 cores.
- Each job uses OMP/MKL/OPENBLAS/NUMEXPR/BLIS=3, OT_THREADS=2, N_OT_WORKERS=2, PREFETCH=4.
- No canonical multi or Track C query selection.
- Gate: high must beat low on cross/family pp and pass no-harm vs anchor; otherwise close or mutate the axis.
EOF

launch_batch() {
  local label="$1"
  local replay_weight="$2"
  local session_prefix="$3"
  local run_prefix="$4"

  local run_root="${SWEEP_ROOT}/${label}"
  local out_root="${OUT_ROOT_BASE}/${label}"
  local log_root="${LOG_ROOT}/${label}"
  local report_dir="${REPORT_ROOT}/${label}"
  mkdir -p "${run_root}" "${out_root}" "${log_root}" "${report_dir}"

  LATENTFM_SCALING_V2_INFO_PACKET_JSON="${PACKET_JSON}" \
  LATENTFM_SCALING_V2_INFO_HIGH_SPLIT="${HIGH_SPLIT}" \
  LATENTFM_SCALING_V2_INFO_LOW_SPLIT="${LOW_SPLIT}" \
  LATENTFM_SCALING_V2_INFO_RUN_ROOT="${run_root}" \
  LATENTFM_SCALING_V2_INFO_OUT_ROOT="${out_root}" \
  LATENTFM_SCALING_V2_INFO_LOG_ROOT="${log_root}" \
  LATENTFM_SCALING_V2_INFO_REPORT_DIR="${report_dir}" \
  LATENTFM_SCALING_V2_INFO_RUN_PREFIX="${run_prefix}" \
  LATENTFM_SCALING_V2_INFO_SESSION_PREFIX="${session_prefix}" \
  LATENTFM_SCALING_V2_INFO_RUN_STATUS_TITLE="latentfm_scaling_v2_condition_information_replay_${label}_20260628" \
  LATENTFM_SCALING_V2_INFO_LAUNCH_COMMAND="bash ${ROOT}/ops/launch_latentfm_scaling_v2_condition_information_replay_sweep_20260628.sh # batch ${label}" \
  LATENTFM_SCALING_V2_INFO_STEPS=2000 \
  LATENTFM_SCALING_V2_INFO_SEED=42 \
  LATENTFM_SCALING_V2_INFO_LR=5e-5 \
  LATENTFM_SCALING_V2_INFO_ANCHOR_REPLAY_LOSS_WEIGHT="${replay_weight}" \
  LATENTFM_SCALING_V2_INFO_ANCHOR_REPLAY_LOSS_WARMUP_START=0 \
  LATENTFM_SCALING_V2_INFO_ANCHOR_REPLAY_LOSS_WARMUP_END=500 \
  LATENTFM_SCALING_V2_INFO_COMPOSITION_DELTA_LOSS_WEIGHT=0.03 \
  LATENTFM_SCALING_V2_INFO_ENDPOINT_DELTA_LOSS_WEIGHT=2.5 \
  LATENTFM_SCALING_V2_INFO_DS_ALPHA=1.0 \
  LATENTFM_SCALING_V2_INFO_DS_LOSS_ALPHA=0.0 \
  LATENTFM_SCALING_V2_INFO_OT_THREADS=2 \
  LATENTFM_SCALING_V2_INFO_PREFETCH=4 \
  LATENTFM_SCALING_V2_INFO_N_OT_WORKERS=2 \
  bash "${BASE_LAUNCHER}" > "${SWEEP_ROOT}/logs/${label}_launch.log" 2>&1
}

launch_batch "w05" "0.5" "lfm_scv2rep05" "xverse_scaling_v2_info_replay_w05"
launch_batch "w10" "1.0" "lfm_scv2rep10" "xverse_scaling_v2_info_replay_w10"

{
  echo
  echo "## Launch sanity check"
  echo
  echo "\`\`\`"
  date '+%F %T %Z'
  tmux ls 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  echo "\`\`\`"
  echo
  echo "## Current status"
  echo
  echo "Launched w05 and w10 high/low child jobs."
} >> "${SWEEP_ROOT}/RUN_STATUS.md"

echo "Launched replay sweep. RUN_STATUS: ${SWEEP_ROOT}/RUN_STATUS.md"
