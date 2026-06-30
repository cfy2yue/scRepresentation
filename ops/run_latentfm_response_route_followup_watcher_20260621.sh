#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_ROOT=${RUN_ROOT:-${ROOT}/runs/latentfm_response_route_followup_watcher_20260621}
LOG_DIR=${RUN_ROOT}/logs
REPORT_DIR=${ROOT}/reports
INTERVAL_SECONDS=${INTERVAL_SECONDS:-1800}
MAX_ROUNDS=${MAX_ROUNDS:-48}
mkdir -p "${LOG_DIR}" "${REPORT_DIR}"

UNCAPPED_RUN_ROOT=${UNCAPPED_RUN_ROOT:-${ROOT}/runs/latentfm_response_aux05_uncapped_posthoc_20260621}
UNCAPPED_INDEX=${UNCAPPED_INDEX:-${REPORT_DIR}/latentfm_response_aux05_uncapped_posthoc_20260621/uncapped_posthoc_index.json}
UNCAPPED_PREFIX=${UNCAPPED_PREFIX:-${REPORT_DIR}/latentfm_response_aux05_uncapped_route_audit_20260621}
UNCAPPED_BOOT_JSON=${UNCAPPED_BOOT_JSON:-${REPORT_DIR}/latentfm_response_aux05_uncapped_route_bootstrap_20260621.json}
UNCAPPED_BOOT_MD=${UNCAPPED_BOOT_MD:-${REPORT_DIR}/LATENTFM_RESPONSE_AUX05_UNCAPPED_ROUTE_BOOTSTRAP_20260621.md}
UNCAPPED_DECISION_JSON=${UNCAPPED_DECISION_JSON:-${REPORT_DIR}/latentfm_response_aux05_uncapped_route_decision_20260621.json}
UNCAPPED_DECISION_MD=${UNCAPPED_DECISION_MD:-${REPORT_DIR}/LATENTFM_RESPONSE_AUX05_UNCAPPED_ROUTE_DECISION_20260621.md}

SWEET_ROOT=${SWEET_ROOT:-${ROOT}/runs/latentfm_response_route_sweetspot_20260621}
SWEET_PREFIX=${SWEET_PREFIX:-${REPORT_DIR}/latentfm_response_route_sweetspot_route_audit_20260621}
SWEET_BOOT_JSON=${SWEET_BOOT_JSON:-${REPORT_DIR}/latentfm_response_route_sweetspot_route_bootstrap_20260621.json}
SWEET_BOOT_MD=${SWEET_BOOT_MD:-${REPORT_DIR}/LATENTFM_RESPONSE_ROUTE_SWEETSPOT_BOOTSTRAP_20260621.md}
SWEET_DECISION_JSON=${SWEET_DECISION_JSON:-${REPORT_DIR}/latentfm_response_route_sweetspot_route_decision_20260621.json}
SWEET_DECISION_MD=${SWEET_DECISION_MD:-${REPORT_DIR}/LATENTFM_RESPONSE_ROUTE_SWEETSPOT_DECISION_20260621.md}
SWEET_RUNS=(
  scf_response_dataset_scale_pca32_aux0625_4k
  scf_response_dataset_scale_pca32_aux075_4k
  scf_response_dataset_scale_pca32_aux0875_4k
)

ANALYZE=${ROOT}/ops/analyze_latentfm_condition_route_audit_20260621.py
BOOTSTRAP=${ROOT}/ops/bootstrap_latentfm_route_audit_20260621.py
DECISION=${ROOT}/ops/summarize_latentfm_route_decision_20260621.py

date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_response_route_followup_watcher_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_response_route_followup_watcher_20260621.sh
\`\`\`

## Runtime classification

Long CPU watcher. It checks at most once every ${INTERVAL_SECONDS}s.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## Log path

\`${LOG_DIR}/watcher.log\`

## Expected outputs

* \`${UNCAPPED_PREFIX}.md\`
* \`${UNCAPPED_BOOT_MD}\`
* \`${UNCAPPED_DECISION_MD}\`
* \`${SWEET_PREFIX}.md\`
* \`${SWEET_BOOT_MD}\`
* \`${SWEET_DECISION_MD}\`

## Current status

Started.
EOF

run_uncapped_followup() {
  if [[ -s "${UNCAPPED_BOOT_MD}" && -s "${UNCAPPED_PREFIX}.md" ]]; then
    return 0
  fi
  if [[ ! -f "${UNCAPPED_RUN_ROOT}/EXIT_CODE" ]]; then
    return 1
  fi
  local rc
  rc="$(cat "${UNCAPPED_RUN_ROOT}/EXIT_CODE")"
  if [[ "${rc}" != "0" ]]; then
    echo "uncapped posthoc failed with exit=${rc}; skip uncapped route followup"
    return 0
  fi
  if [[ ! -s "${UNCAPPED_INDEX}" ]]; then
    echo "uncapped exit=0 but index missing: ${UNCAPPED_INDEX}"
    return 1
  fi
  echo "[$(date '+%F %T %Z')] running uncapped route audit"
  "${PYTHON}" "${ANALYZE}" \
    --no-include-defaults \
    --uncapped-index "${UNCAPPED_INDEX}" \
    --out-prefix "${UNCAPPED_PREFIX}"
  echo "[$(date '+%F %T %Z')] running uncapped route bootstrap"
  "${PYTHON}" "${BOOTSTRAP}" \
    --condition-csv "${UNCAPPED_PREFIX}.conditions.csv" \
    --comparisons scf_response_dataset_scale_pca32_aux05_4k_uncapped_vs_anchor \
    --routes candidate_gene_multi candidate_multi_not_drug \
    --out-json "${UNCAPPED_BOOT_JSON}" \
    --out-md "${UNCAPPED_BOOT_MD}"
  echo "[$(date '+%F %T %Z')] running uncapped route decision"
  "${PYTHON}" "${DECISION}" \
    --route-bootstrap-json "${UNCAPPED_BOOT_JSON}" \
    --out-json "${UNCAPPED_DECISION_JSON}" \
    --out-md "${UNCAPPED_DECISION_MD}"
  return 0
}

sweet_manifests_ready() {
  local run
  for run in "${SWEET_RUNS[@]}"; do
    if [[ ! -s "${SWEET_ROOT}/${run}/posthoc_manifest.json" ]]; then
      return 1
    fi
  done
  return 0
}

run_sweet_followup() {
  if [[ -s "${SWEET_BOOT_MD}" && -s "${SWEET_PREFIX}.md" ]]; then
    return 0
  fi
  if ! sweet_manifests_ready; then
    return 1
  fi
  echo "[$(date '+%F %T %Z')] running sweetspot route audit"
  "${PYTHON}" "${ANALYZE}" \
    --no-include-defaults \
    --posthoc-manifest "${SWEET_ROOT}/scf_response_dataset_scale_pca32_aux0625_4k/posthoc_manifest.json" \
    --posthoc-manifest "${SWEET_ROOT}/scf_response_dataset_scale_pca32_aux075_4k/posthoc_manifest.json" \
    --posthoc-manifest "${SWEET_ROOT}/scf_response_dataset_scale_pca32_aux0875_4k/posthoc_manifest.json" \
    --out-prefix "${SWEET_PREFIX}"
  echo "[$(date '+%F %T %Z')] running sweetspot route bootstrap"
  "${PYTHON}" "${BOOTSTRAP}" \
    --condition-csv "${SWEET_PREFIX}.conditions.csv" \
    --comparisons \
      scf_response_dataset_scale_pca32_aux0625_4k_vs_anchor \
      scf_response_dataset_scale_pca32_aux075_4k_vs_anchor \
      scf_response_dataset_scale_pca32_aux0875_4k_vs_anchor \
    --routes candidate_gene_multi candidate_multi_not_drug \
    --out-json "${SWEET_BOOT_JSON}" \
    --out-md "${SWEET_BOOT_MD}"
  echo "[$(date '+%F %T %Z')] running sweetspot route decision"
  "${PYTHON}" "${DECISION}" \
    --route-bootstrap-json "${SWEET_BOOT_JSON}" \
    --out-json "${SWEET_DECISION_JSON}" \
    --out-md "${SWEET_DECISION_MD}"
  return 0
}

{
  echo "[$(date '+%F %T %Z')] response route followup watcher start"
  uncapped_done=0
  sweet_done=0
  for ((round=1; round<=MAX_ROUNDS; round++)); do
    echo "[$(date '+%F %T %Z')] watcher round ${round}/${MAX_ROUNDS}"
    if [[ "${uncapped_done}" == "0" ]]; then
      if run_uncapped_followup; then
        uncapped_done=1
      fi
    fi
    if [[ "${sweet_done}" == "0" ]]; then
      if run_sweet_followup; then
        sweet_done=1
      fi
    fi
    if [[ "${uncapped_done}" == "1" && "${sweet_done}" == "1" ]]; then
      echo "[$(date '+%F %T %Z')] all followups complete"
      break
    fi
    echo "[$(date '+%F %T %Z')] followups pending; next check in ${INTERVAL_SECONDS}s"
    sleep "${INTERVAL_SECONDS}"
  done
} 2>&1 | tee "${LOG_DIR}/watcher.log"

date '+%F %T %Z' > "${RUN_ROOT}/FINISHED"
echo 0 > "${RUN_ROOT}/EXIT_CODE"
