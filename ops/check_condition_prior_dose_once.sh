#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
REPORT="${ROOT}/reports/LATENTFM_CONDITION_PRIOR_DOSE_ONE_SHOT_STATUS_20260619.md"

TRAIN_RUNS=(
  "${ROOT}/runs/latentfm_condition_prior_teacher_prior002_20260619"
  "${ROOT}/runs/latentfm_condition_prior_teacher_probe_20260619"
  "${ROOT}/runs/latentfm_condition_prior_teacher_prior010_20260619"
)
WATCHER_RUNS=(
  "${ROOT}/runs/latentfm_condition_prior_teacher_posthoc_20260619"
  "${ROOT}/runs/latentfm_condition_prior_teacher_sister_posthoc_20260619"
  "${ROOT}/runs/latentfm_condition_prior_teacher_dose_summary_20260619"
)
DOSE_REPORT="${ROOT}/reports/LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md"
DOSE_JSON="${ROOT}/reports/latentfm_condition_prior_teacher_dose_20260619.json"
DOSE_CSV="${ROOT}/reports/latentfm_condition_prior_teacher_dose_20260619.csv"
FIG_BASE="${ROOT}/reports/latentfm_condition_prior_teacher_dose_20260619"

file_state() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    printf 'present (%s bytes)' "$(stat -c '%s' "${path}")"
  else
    printf 'missing'
  fi
}

marker_value() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    tr '\n' ' ' < "${path}"
  else
    printf 'missing'
  fi
}

json_summary() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "missing"
    return 0
  fi
  python - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
rows = payload.get("rows", [])
complete = sum(1 for row in rows if row.get("complete"))
repeat = sum(1 for row in rows if row.get("decision") == "repeat_candidate")
best = payload.get("best") or {}
best_run = best.get("run", "NA") if isinstance(best, dict) else "NA"
print(
    f"status={payload.get('status', 'present_no_status')}; "
    f"complete={complete}/{len(rows)}; "
    f"repeat_candidates={repeat}; "
    f"best={best_run}"
)
PY
}

{
  echo "# LatentFM Condition-Prior Dose One-Shot Status"
  echo
  echo "Generated: $(date '+%F %T %Z')"
  echo
  echo "This is a single lightweight check. It reads marker/status/report files"
  echo "only. It does not tail training logs, inspect GPU utilization, attach to"
  echo "tmux, or launch new training/posthoc jobs."
  echo
  echo "## Training Markers"
  echo
  echo "| Run | EXIT_CODE | FINISHED | RUN_STATUS |"
  echo "|---|---:|---|---|"
  for run in "${TRAIN_RUNS[@]}"; do
    printf '| `%s` | `%s` | `%s` | %s |\n' \
      "$(basename "${run}")" \
      "$(marker_value "${run}/EXIT_CODE")" \
      "$(marker_value "${run}/FINISHED")" \
      "$(file_state "${run}/RUN_STATUS.md")"
  done
  echo
  echo "## Watcher Markers"
  echo
  echo "| Run | EXIT_CODE | FINISHED | RUN_STATUS |"
  echo "|---|---:|---|---|"
  for run in "${WATCHER_RUNS[@]}"; do
    printf '| `%s` | `%s` | `%s` | %s |\n' \
      "$(basename "${run}")" \
      "$(marker_value "${run}/EXIT_CODE")" \
      "$(marker_value "${run}/FINISHED")" \
      "$(file_state "${run}/RUN_STATUS.md")"
  done
  echo
  echo "## Dose Artifacts"
  echo
  echo "| Artifact | State |"
  echo "|---|---|"
  for path in \
    "${DOSE_REPORT}" \
    "${DOSE_JSON}" \
    "${DOSE_CSV}" \
    "${FIG_BASE}.pdf" \
    "${FIG_BASE}.svg" \
    "${FIG_BASE}.png" \
    "${FIG_BASE}.figure_meta.json"; do
    printf '| `%s` | %s |\n' "${path}" "$(file_state "${path}")"
  done
  echo
  echo "## Dose JSON Summary"
  echo
  printf -- '- `%s`\n' "$(json_summary "${DOSE_JSON}")"
  echo
  echo "## Report Head"
  if [[ -f "${DOSE_REPORT}" ]]; then
    echo '```text'
    sed -n '1,90p' "${DOSE_REPORT}"
    echo '```'
  else
    echo "Missing: \`${DOSE_REPORT}\`"
  fi
  echo
  echo "## Next Read Commands"
  echo
  echo '```bash'
  echo "sed -n '1,220p' ${DOSE_REPORT}"
  echo "python -m json.tool ${DOSE_JSON} | sed -n '1,220p'"
  echo "sed -n '1,180p' ${ROOT}/reports/WORKSPACE_STATUS.md"
  echo '```'
} > "${REPORT}"

echo "${REPORT}"
sed -n '1,220p' "${REPORT}"
