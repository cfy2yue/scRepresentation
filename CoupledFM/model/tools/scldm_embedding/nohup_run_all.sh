#!/usr/bin/env bash
# Sequential control embedding then GT. Detach from the ssh session by wrapping once:
#   nohup bash /path/to/nohup_run_all.sh >/dev/null 2>&1 &
#
# Repo-local TMPDIR and logs/scldm_embedding/run_<UTC>/ (no writes under HOME).
set -euo pipefail

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
PYTHON="${SCLDM_PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

export TMPDIR="${ROOT}/tmp/scldm_embedding"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

RUN_LOGROOT="${ROOT}/logs/scldm_embedding/run_${STAMP}"
mkdir -p "${RUN_LOGROOT}"

# Example: encode on seven GPUs (-j parallelizes buckets; sequential within each GPU bucket).
CONTROL_ARGS=(
  "--gpus" "0,1,2,3,4,5,6"
  "-j" "7"
  "--device" "cuda"
)
GT_ARGS=(
  "--gpus" "0,1,2,3,4,5,6"
  "-j" "7"
  "--device" "cuda"
)

echo "Logging to ${RUN_LOGROOT}"
(
  exec >"${RUN_LOGROOT}/step1_stdout.log" 2>"${RUN_LOGROOT}/step1_stderr.log"
  "${PYTHON}" "${ROOT}/model/tools/scldm_embedding/step1_control_embedding.py" "${CONTROL_ARGS[@]}" "$@"
)

(
  exec >"${RUN_LOGROOT}/step3_stdout.log" 2>"${RUN_LOGROOT}/step3_stderr.log"
  "${PYTHON}" "${ROOT}/model/tools/scldm_embedding/step3_gt_embedding.py" "${GT_ARGS[@]}" "$@"
)

echo "Finished sequential runs; inspect ${RUN_LOGROOT}/step*.log and script logs/"
