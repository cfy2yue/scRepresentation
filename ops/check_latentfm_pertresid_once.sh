#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
RUN_ROOT="${ROOT}/runs/latentfm_condition_delta_pertresid_smoke_20260617"
RUN_TAG="20260617_scfoundation_conddelta005_pertresidtarget_comp006_endpoint5_3k_smoke"
TRAIN_SESSION="latentfm_20260618_scfoundation_conddelta_pertresid_3k_smoke"
WATCHER_SESSION="watcher_20260618_scfoundation_conddelta_pertresid_posthoc"
POSTHOC_SESSION="posthoc_20260618_scfoundation_conddelta_pertresid_3k_smoke"
RUN_DIR="${ROOT}/CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke/${RUN_TAG}"
OUT_DIR="${RUN_DIR}/posthoc_eval"
REPORT="${ROOT}/reports/LATENTFM_PERTRESID_ONE_SHOT_STATUS_20260618.md"
WORKSPACE_STATUS="${ROOT}/reports/WORKSPACE_STATUS.md"

session_state() {
  local name="$1"
  if tmux has-session -t "${name}" 2>/dev/null; then
    printf 'running'
  else
    printf 'not running'
  fi
}

file_row() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    printf '| `%s` | present | %s | %s |\n' \
      "${path}" \
      "$(stat -c '%s' "${path}")" \
      "$(stat -c '%y' "${path}" | cut -d'.' -f1)"
  else
    printf '| `%s` | missing | NA | NA |\n' "${path}"
  fi
}

{
  printf '# LatentFM Pert-Residual One-Shot Status\n\n'
  printf 'Generated: %s\n\n' "$(date '+%F %T %Z')"
  printf 'This is a single lightweight check. It does not attach to tmux, tail logs continuously, or launch training.\n\n'

  printf '## Sessions\n\n'
  printf '| Session | State |\n'
  printf '|---|---|\n'
  printf '| `%s` | %s |\n' "${TRAIN_SESSION}" "$(session_state "${TRAIN_SESSION}")"
  printf '| `%s` | %s |\n' "${WATCHER_SESSION}" "$(session_state "${WATCHER_SESSION}")"
  printf '| `%s` | %s |\n\n' "${POSTHOC_SESSION}" "$(session_state "${POSTHOC_SESSION}")"

  printf '## Markers\n\n'
  printf '| Marker | State | Value |\n'
  printf '|---|---|---|\n'
  for marker in STARTED FINISHED EXIT_CODE SESSION_NAME; do
    path="${RUN_ROOT}/${marker}"
    if [[ -f "${path}" ]]; then
      printf '| `%s` | present | `%s` |\n' "${marker}" "$(tr '\n' ' ' < "${path}" | sed 's/[[:space:]]*$//')"
    else
      printf '| `%s` | missing | NA |\n' "${marker}"
    fi
  done

  printf '\n## Posthoc Outputs\n\n'
  printf '| File | State | Bytes | Mtime |\n'
  printf '|---|---|---:|---|\n'
  file_row "${OUT_DIR}/split_group_eval_best_ode20_mse2048_mmd2048.json"
  file_row "${OUT_DIR}/condition_family_eval_best_ode20_mse2048_mmd2048.json"
  file_row "${OUT_DIR}/condition_delta_head_gene_test.json"

  printf '\n## Watcher Tail\n\n'
  printf '```text\n'
  tail -n 40 "${ROOT}/logs/latentfm_condition_delta_pertresid_smoke/posthoc_watcher.log" 2>/dev/null || true
  printf '```\n\n'

  printf '## Workspace Status Refresh\n\n'
  if python "${ROOT}/ops/generate_workspace_status.py" >/dev/null; then
    printf 'Refreshed: `%s`\n' "${WORKSPACE_STATUS}"
  else
    printf 'Refresh failed: `%s`\n' "${WORKSPACE_STATUS}"
  fi
} > "${REPORT}"

printf '%s\n' "${REPORT}"
