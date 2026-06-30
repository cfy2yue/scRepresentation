#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
RUN_ROOT=${ROOT}/runs/latentfm_active_posthoc_bootstrap_20260621
LOG_DIR=${RUN_ROOT}/logs
BOOTSTRAP_RUNNER=${ROOT}/ops/run_latentfm_posthoc_bootstrap_from_manifest_20260621.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${LOG_DIR}" "${ROOT}/reports"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_active_posthoc_bootstrap_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_active_posthoc_bootstrap_watcher_20260621.sh
\`\`\`

## Runtime classification

Long CPU watcher. It checks for posthoc manifests every 30 minutes and runs
paired condition-level bootstrap once per manifest.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`latentfm_active_posthoc_bootstrap_20260621\`

## Log path

\`${LOG_DIR}/bootstrap_watcher.log\`

## Expected outputs

* \`${ROOT}/reports/latentfm_fewshot_multi_calibration_bootstrap_20260621/bootstrap_index.json\`
* \`${ROOT}/reports/latentfm_response_geometry_smoke_bootstrap_20260621/bootstrap_index.json\`

## How to check manually

\`\`\`bash
tmux ls | grep latentfm_active_posthoc_bootstrap_20260621 || true
tail -n 50 ${LOG_DIR}/bootstrap_watcher.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
\`\`\`

## Current status

Started.

## Notes

Does not inspect training logs or GPUs. It only watches for completed posthoc
manifest files and then runs \`run_latentfm_posthoc_bootstrap_from_manifest_20260621.py\`.
EOF

trap 'rc=$?; echo "$rc" > "${RUN_ROOT}/EXIT_CODE"; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; exit "$rc"' EXIT

run_target() {
  local label="$1"
  local manifest="$2"
  local out_dir="$3"
  local index="${out_dir}/bootstrap_index.json"
  if [[ -s "${index}" ]]; then
    echo "[$(date '+%F %T %Z')] ${label}: bootstrap already exists: ${index}"
    return 0
  fi
  if [[ ! -s "${manifest}" ]]; then
    echo "[$(date '+%F %T %Z')] ${label}: waiting for manifest ${manifest}"
    return 1
  fi
  echo "[$(date '+%F %T %Z')] ${label}: running paired bootstrap"
  "${PYTHON}" "${BOOTSTRAP_RUNNER}" \
    --manifest "${manifest}" \
    --out-dir "${out_dir}" \
    --n-boot 2000 \
    --seed 42
  echo "[$(date '+%F %T %Z')] ${label}: bootstrap complete"
  return 0
}

{
  echo "[$(date '+%F %T %Z')] active posthoc bootstrap watcher start"
  while true; do
    pending=0
    run_target \
      "fewshot_multi_calibration" \
      "${ROOT}/runs/latentfm_fewshot_multi_calibration_20260621/posthoc_manifest.json" \
      "${ROOT}/reports/latentfm_fewshot_multi_calibration_bootstrap_20260621" \
      || pending=1
    run_target \
      "response_geometry_smoke" \
      "${ROOT}/runs/latentfm_response_normalization_20260621/posthoc_manifest.json" \
      "${ROOT}/reports/latentfm_response_geometry_smoke_bootstrap_20260621" \
      || pending=1
    if [[ "${pending}" == "0" ]]; then
      echo "[$(date '+%F %T %Z')] all active posthoc bootstraps complete"
      exit 0
    fi
    echo "[$(date '+%F %T %Z')] pending manifests remain; next check in 1800s"
    sleep 1800
  done
} 2>&1 | tee "${LOG_DIR}/bootstrap_watcher.log"
