#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
RUN_ROOT="${ROOT}/runs/latentfm_fullcap_posthoc_20260618"
SESSION="latentfm_fullcap_posthoc_20260618"
LOG="${ROOT}/logs/latentfm_fullcap_posthoc_20260618/run.log"
REPORT="${ROOT}/reports/LATENTFM_FULLCAP_ONE_SHOT_STATUS_20260619.md"
FULLCAP_REPORT="${ROOT}/reports/LATENTFM_FULLCAP_POSTHOC_REPORT_20260618.md"

session_state="not running"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  session_state="running"
fi

{
  echo "# LatentFM Full-Cap One-Shot Status"
  echo
  echo "Generated: $(date '+%F %T %Z')"
  echo
  echo "This is one lightweight check. It does not attach to tmux or continuously tail logs."
  echo
  echo "## Session"
  echo
  echo "| Session | State |"
  echo "|---|---|"
  echo "| \`${SESSION}\` | ${session_state} |"
  echo
  echo "## Markers"
  echo
  echo "| Marker | State | Value |"
  echo "|---|---|---|"
  for m in STARTED FINISHED EXIT_CODE SESSION_NAME; do
    p="${RUN_ROOT}/${m}"
    if [[ -f "${p}" ]]; then
      echo "| \`${m}\` | present | \`$(tr '\n' ' ' < "${p}")\` |"
    else
      echo "| \`${m}\` | missing | NA |"
    fi
  done
  echo
  echo "## Full-Cap Report Head"
  echo
  if [[ -f "${FULLCAP_REPORT}" ]]; then
    echo '```text'
    sed -n '1,120p' "${FULLCAP_REPORT}"
    echo '```'
  else
    echo "Missing: \`${FULLCAP_REPORT}\`"
  fi
  echo
  echo "## Log Tail"
  echo
  if [[ -f "${LOG}" ]]; then
    echo '```text'
    tail -n 80 "${LOG}"
    echo '```'
  else
    echo "Missing: \`${LOG}\`"
  fi
  echo
  echo "## Workspace Status Refresh"
  python "${ROOT}/ops/generate_workspace_status.py" >/dev/null || true
  echo
  echo "Refreshed: \`${ROOT}/reports/WORKSPACE_STATUS.md\`"
} > "${REPORT}"

echo "${REPORT}"
