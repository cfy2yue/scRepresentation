#!/usr/bin/env bash
# Seven-GPU scFoundation control + GT runner. Detach with:
#   nohup bash /path/to/CoupledFM/model/tools/scfoundation_embedding/run_7gpu_nohup.sh both >/dev/null 2>&1 &
#
# The script writes all tmp/log artifacts under CoupledFM, not $HOME.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${SCFOUNDATION_PYTHON:-python}"
TMP="${ROOT}/tmp/scfoundation_embedding"
mkdir -p "${TMP}" "${ROOT}/logs/scfoundation_embedding"

export PYTHONUNBUFFERED=1
export TMPDIR="${TMP}"
export TEMP="${TMP}"
export TMP="${TMP}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MODE="${1:-control}"  # control | gt | both
RUN_LOGROOT="${ROOT}/logs/scfoundation_embedding/run_${STAMP}"
mkdir -p "${RUN_LOGROOT}"

run_control() {
  local logdir="${ROOT}/logs/scfoundation_embedding/control_${STAMP}"
  mkdir -p "${logdir}"
  (
    exec >"${RUN_LOGROOT}/step1_stdout.log" 2>"${RUN_LOGROOT}/step1_stderr.log"
    env TMPDIR="${TMP}" TEMP="${TMP}" TMP="${TMP}" \
      "${PY}" "${ROOT}/model/tools/scfoundation_embedding/step1_control_embedding.py" \
    --schedule greedy --gpus 0,1,2,3,4,5,6 -j 7 \
    --log-dir "${logdir}" --tmp-dir "${TMP}" \
    "$@"
  )
  echo "control done logdir=${logdir}"
}

run_gt() {
  local logdir="${ROOT}/logs/scfoundation_embedding/gt_${STAMP}"
  mkdir -p "${logdir}"
  (
    exec >"${RUN_LOGROOT}/step3_stdout.log" 2>"${RUN_LOGROOT}/step3_stderr.log"
    env TMPDIR="${TMP}" TEMP="${TMP}" TMP="${TMP}" \
      "${PY}" "${ROOT}/model/tools/scfoundation_embedding/step3_gt_embedding.py" \
    --schedule greedy --gpus 0,1,2,3,4,5,6 -j 7 \
    --log-dir "${logdir}" --tmp-dir "${TMP}" \
    "$@"
  )
  echo "gt done logdir=${logdir}"
}

case "${MODE}" in
  control) shift || true; run_control "$@" ;;
  gt) shift || true; run_gt "$@" ;;
  both)
    shift || true
    run_control "$@"
    run_gt "$@"
    ;;
  *) echo "usage: $0 [control|gt|both]"; exit 2 ;;
esac
