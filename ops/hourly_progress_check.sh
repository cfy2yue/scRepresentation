#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
MIN_INTERVAL_SECONDS="${MIN_INTERVAL_SECONDS:-3600}"
STATE_FILE="${ROOT}/logs/progress_last_check.epoch"
HISTORY_FILE="${ROOT}/logs/progress_history.tsv"
TRANSFER_PID_FILE="${ROOT}/logs/transfer_from_lilab.pid"
WATCHER_PID_FILE="${ROOT}/logs/post_sync_validate.pid"
TRANSFER_STATUS_FILE="${ROOT}/logs/transfer_from_lilab.status"

now="$(date +%s)"
mkdir -p "${ROOT}/logs"

if [[ -f "$STATE_FILE" ]]; then
  last="$(cat "$STATE_FILE")"
  elapsed="$((now - last))"
  if (( elapsed < MIN_INTERVAL_SECONDS )); then
    echo "skip: only ${elapsed}s since last progress check; min=${MIN_INTERVAL_SECONDS}s"
    exit 0
  fi
fi

transfer_pid="$(cat "$TRANSFER_PID_FILE" 2>/dev/null || true)"
watcher_pid="$(cat "$WATCHER_PID_FILE" 2>/dev/null || true)"
transfer_alive=0
watcher_alive=0
[[ -n "$transfer_pid" ]] && kill -0 "$transfer_pid" 2>/dev/null && transfer_alive=1
[[ -n "$watcher_pid" ]] && kill -0 "$watcher_pid" 2>/dev/null && watcher_alive=1

status="$(tr '\t' ' ' < "$TRANSFER_STATUS_FILE" 2>/dev/null || true)"
dataset_size="$(du -sh "${ROOT}/dataset" 2>/dev/null | awk '{print $1}' || true)"
biflow_size="$(du -sh "${ROOT}/dataset/biFlow_data" 2>/dev/null | awk '{print $1}' || true)"
pretrained_size="$(du -sh "${ROOT}/scFM_pretrained" 2>/dev/null | awk '{print $1}' || true)"
third_party_size="$(du -sh "${ROOT}/scFM_third_party" 2>/dev/null | awk '{print $1}' || true)"
pretrainckpt_size="$(du -sh "${ROOT}/pretrainckpt" 2>/dev/null | awk '{print $1}' || true)"
mem_avail_gib="$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"
data_free="$(df -h "$ROOT" | awk 'NR==2 {print $4}')"
loadavg="$(awk '{print $1","$2","$3}' /proc/loadavg)"

if [[ ! -f "$HISTORY_FILE" ]]; then
  printf 'timestamp\ttransfer_alive\twatcher_alive\tstatus\tdataset\tbiFlow\tscFM_pretrained\tscFM_third_party\tpretrainckpt\tmem_available_gib\tdata_free\tloadavg\n' > "$HISTORY_FILE"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date '+%F %T')" \
  "$transfer_alive" \
  "$watcher_alive" \
  "$status" \
  "${dataset_size:-NA}" \
  "${biflow_size:-NA}" \
  "${pretrained_size:-NA}" \
  "${third_party_size:-NA}" \
  "${pretrainckpt_size:-NA}" \
  "$mem_avail_gib" \
  "$data_free" \
  "$loadavg" | tee -a "$HISTORY_FILE"

printf '%s\n' "$now" > "$STATE_FILE"
