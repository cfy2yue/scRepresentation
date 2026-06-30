#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
FOLLOWUP_ROOT="${ROOT}/runs/latentfm_followup_posthoc_20260617"
ALIGN_ROOT="${ROOT}/runs/latentfm_alignment_smoke_posthoc_20260617"
OUT="${ROOT}/reports/LATENTFM_FOLLOWUP_ONE_SHOT_STATUS_20260617.md"

now="$(date '+%F %T %Z')"

{
  echo "# LatentFM Follow-Up One-Shot Status"
  echo
  echo "Generated: ${now}"
  echo
  echo "This is a single lightweight status check. It does not attach to tmux, tail"
  echo "logs continuously, or launch new GPU work."
  echo
  echo "## tmux Sessions"
  echo
  for session in \
    posthoc_20260617_scfoundation_conddelta_addatom_3k_smoke \
    posthoc_20260617_scfoundation_conddelta_inject_3k_smoke \
    watcher_latentfm_followup_posthoc_20260617; do
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "- \`${session}\`: running"
    else
      echo "- \`${session}\`: not running"
    fi
  done
  echo
  echo "## Watcher Tail"
  echo
  echo '```text'
  tail -n 80 "${ROOT}/logs/latentfm_followup_posthoc_20260617/watcher.log" 2>/dev/null || true
  echo '```'
  echo
  echo "## Posthoc JSON Outputs"
  echo
  echo '```text'
  find "${ROOT}/CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke" \
    -maxdepth 3 \
    -type f \
    \( -path '*conddelta005_addatom005*/posthoc_eval/*.json' \
       -o -path '*conddelta005_inject*/posthoc_eval/*.json' \) \
    -printf '%p %s %TY-%Tm-%Td %TH:%TM\n' 2>/dev/null | sort || true
  echo '```'
} > "$OUT"

if [[ -x "${ALIGN_ROOT}/summarize_alignment_smoke.py" ]]; then
  source "${ROOT}/init-scdfm.sh" >/dev/null 2>&1 || true
  python "${ALIGN_ROOT}/summarize_alignment_smoke.py" >/dev/null
fi

if [[ -x "${ROOT}/ops/latentfm_followup_decision.py" ]]; then
  python "${ROOT}/ops/latentfm_followup_decision.py" >/dev/null
fi

echo "$OUT"
if [[ -f "${ROOT}/reports/LATENTFM_ALIGNMENT_SMOKE_REPORT_20260617.md" ]]; then
  echo "${ROOT}/reports/LATENTFM_ALIGNMENT_SMOKE_REPORT_20260617.md"
fi
if [[ -f "${ROOT}/reports/LATENTFM_FOLLOWUP_DECISION_STATUS_20260617.md" ]]; then
  echo "${ROOT}/reports/LATENTFM_FOLLOWUP_DECISION_STATUS_20260617.md"
fi
if [[ -f "${ROOT}/reports/latentfm_followup_decision_status_20260617.json" ]]; then
  echo "${ROOT}/reports/latentfm_followup_decision_status_20260617.json"
fi

if [[ -f "$OUT" ]]; then
  sed -n '1,180p' "$OUT"
fi
