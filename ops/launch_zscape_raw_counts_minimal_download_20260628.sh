#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
DATA_DIR="${ROOT}/dataset/external/zscape_20260628"
RUN_ROOT="${ROOT}/runs/zscape_raw_counts_minimal_download_20260628"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="zscape_raw_counts_minimal_download_${TS}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
SESSION_NAME="${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
LOG_PATH="${LOG_DIR}/run.log"
WORKER="${ROOT}/ops/run_zscape_raw_counts_minimal_download_20260628.sh"

mkdir -p "${LOG_DIR}" "${RUN_DIR}/outputs" "${DATA_DIR}"

date > "${RUN_DIR}/STARTED"
echo "${SESSION_NAME}" > "${RUN_DIR}/SESSION_NAME"

cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
${WORKER} ${RUN_DIR} ${DATA_DIR} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED
\`\`\`

## Runtime classification

Long task.

## Start time

$(date)

## PID / tmux / scheduler ID

tmux session: \`${SESSION_NAME}\`

## Log path

\`${LOG_PATH}\`

## Expected outputs

* \`${DATA_DIR}/GSE202639_zperturb_full_raw_counts.RDS.gz\`
* \`${RUN_DIR}/SHA256SUMS\`
* \`${RUN_DIR}/outputs/LATENTFM_ZSCAPE_RAW_COUNTS_MINIMAL_DOWNLOAD_20260628.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -c 4000 ${LOG_PATH} | tr '\r' '\n' | tail -n 50
cat ${RUN_DIR}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Hypothesis: the ZSCAPE metadata/coordinate gates justify downloading the
  minimal ZPERTURB raw-count source needed for expression-space OT validation.
- Resource plan: network/disk only, OMP-equivalent CPU use 1, no GPU.
- Boundary: no CDS/reference matrix, no training, no inference, no embedding,
  no canonical multi, no Track C query.
- Promotion gate after download: expression-cell join/provenance gate and
  CPU-only OT/continuity gate; this run alone authorizes no GPU.
- Fail-close: if download or gzip validation fails repeatedly, do not infer
  biological failure; record transport/env issue and redesign access path.
EOF

tmux new -d -s "${SESSION_NAME}" \
  "${WORKER} ${RUN_DIR} ${DATA_DIR} > ${LOG_PATH} 2>&1; echo \$? > ${RUN_DIR}/EXIT_CODE; date > ${RUN_DIR}/FINISHED"

echo "${RUN_DIR}"
