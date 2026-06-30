#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
DRY_RUN="${DRY_RUN:-1}"
ARCHIVE_DIR="${ROOT}/logs/archive"
TRANSFER_STATUS_FILE="${TRANSFER_STATUS_FILE:-${ROOT}/logs/transfer_from_lilab.status}"
TRANSFER_LOG="${TRANSFER_LOG:-${ROOT}/logs/transfer_from_lilab.log}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_or_print() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_post_sync_state() {
  if ! grep -q $'\tALL DONE' "$TRANSFER_STATUS_FILE" 2>/dev/null && \
     ! grep -q 'ALL DONE' "$TRANSFER_LOG" 2>/dev/null; then
    log "transfer has not reported ALL DONE; cleanup is intentionally disabled"
    exit 10
  fi
}

archive_old_transfer_logs() {
  log "archive old transfer logs"
  run_or_print mkdir -p "$ARCHIVE_DIR"
  find "${ROOT}/logs" -maxdepth 1 -type f \
    \( -name 'transfer_from_lilab.20*.log' -o -name 'nohup_probe.*' \) \
    -print | while read -r path; do
      run_or_print mv "$path" "$ARCHIVE_DIR/"
    done
}

list_temp_candidates() {
  log "list temporary or backup candidates under dataset"
  find "${ROOT}/dataset" -type f \
    \( -name '*.tmp.h5ad' -o -name '*.bak*' -o -name '*.before_*' -o -name '.*.??????' \) \
    -print 2>/dev/null || true
}

list_empty_dirs() {
  log "list empty directories under output/log roots"
  find \
    "${ROOT}/logs" \
    "${ROOT}/scFM_output" \
    "${ROOT}/CoupledFM/output" \
    -type d -empty -print 2>/dev/null || true
}

cleanup_python_caches() {
  log "remove Python cache directories and .pyc files"
  find \
    "${ROOT}/CoupledFM" \
    "${ROOT}/scFMBench" \
    -type d -name '__pycache__' -print 2>/dev/null | while read -r path; do
      run_or_print rm -rf "$path"
    done
  find \
    "${ROOT}/CoupledFM" \
    "${ROOT}/scFMBench" \
    -type f -name '*.pyc' -print 2>/dev/null | while read -r path; do
      run_or_print rm -f "$path"
    done
}

main() {
  require_post_sync_state
  archive_old_transfer_logs
  list_temp_candidates
  list_empty_dirs
  cleanup_python_caches
  log "cleanup pass finished (DRY_RUN=${DRY_RUN})"
}

main "$@"
