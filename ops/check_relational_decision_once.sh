#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
RUN_ROOT="${ROOT}/runs/latentfm_scfoundation_relational_residual_20260619"
REPORT="${ROOT}/reports/LATENTFM_RELATIONAL_ONE_SHOT_STATUS_20260619.md"

now="$(date '+%F %T %Z')"

marker() {
  local stem="$1"
  local exit_file="${RUN_ROOT}/${stem}_EXIT_CODE"
  local finished_file="${RUN_ROOT}/${stem}_FINISHED"
  local status_file="${RUN_ROOT}/${stem}_STATUS.md"
  local exit_code="pending"
  local finished="pending"
  local status_state="missing"
  [[ -f "${exit_file}" ]] && exit_code="$(tr -d '\n' < "${exit_file}")"
  [[ -f "${finished_file}" ]] && finished="$(tr -d '\n' < "${finished_file}")"
  [[ -f "${status_file}" ]] && status_state="present"
  printf '| `%s` | `%s` | `%s` | `%s` |\n' "${stem}" "${exit_code}" "${finished}" "${status_state}"
}

json_status() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "missing"
    return 0
  fi
  python - "$path" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    obj = json.load(handle)
print(obj.get("status", "present_no_status"))
PY
}

{
  echo "# LatentFM Relational One-Shot Status"
  echo
  echo "Generated: ${now}"
  echo
  echo "This is a single lightweight check. It does not tail training logs,"
  echo "inspect GPU utilization, attach to tmux, or launch new jobs."
  echo
  echo "## Scheduled Markers"
  echo
  echo "| Marker | Exit code | Finished | Status file |"
  echo "|---|---:|---|---|"
  marker "SCHEDULED_POSTHOC_0745"
  marker "SCHEDULED_SUMMARY_0835"
  marker "SCHEDULED_POSTHOC_0845"
  marker "SCHEDULED_SUMMARY_0935"
  marker "SCHEDULED_DECISION_0945"
  marker "SCHEDULED_FINALIZE_0950"
  echo
  echo "## Report States"
  echo
  echo "| Artifact | State |"
  echo "|---|---|"
  for path in \
    "${ROOT}/reports/latentfm_scfoundation_relational_residual_status_20260619.json" \
    "${ROOT}/reports/LATENTFM_SCFOUNDATION_RELATIONAL_RESIDUAL_REPORT_20260619.md" \
    "${ROOT}/reports/latentfm_scfoundation_relational_residual_decision_20260619.json" \
    "${ROOT}/reports/LATENTFM_SCFOUNDATION_RELATIONAL_RESIDUAL_DECISION_20260619.md" \
    "${ROOT}/reports/WORKSPACE_STATUS.md"; do
    if [[ -f "${path}" ]]; then
      printf '| `%s` | present (%s bytes) |\n' "${path}" "$(stat -c '%s' "${path}")"
    else
      printf '| `%s` | missing |\n' "${path}"
    fi
  done
  echo
  echo "## JSON Status"
  echo
  printf -- '- Summary status: `%s`\n' "$(json_status "${ROOT}/reports/latentfm_scfoundation_relational_residual_status_20260619.json")"
  printf -- '- Decision status: `%s`\n' "$(json_status "${ROOT}/reports/latentfm_scfoundation_relational_residual_decision_20260619.json")"
  echo
  echo "## Recommended Read Commands After 09:50"
  echo
  echo '```bash'
  echo "sed -n '1,180p' ${ROOT}/reports/LATENTFM_SCFOUNDATION_RELATIONAL_RESIDUAL_DECISION_20260619.md"
  echo "sed -n '1,220p' ${ROOT}/reports/WORKSPACE_STATUS.md"
  echo "sed -n '1,200p' ${ROOT}/reports/LATENTFM_POST_RELATIONAL_NEXT_ACTIONS_20260619.md"
  echo '```'
} > "${REPORT}"

echo "${REPORT}"
sed -n '1,220p' "${REPORT}"
