#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
SESSION="${SESSION:-sync_top3_training_data}"
LOG="${ROOT}/logs/rsync/sync_top3_training_data_monitor_20260616.log"

mkdir -p "$(dirname "${LOG}")"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"
}

log "monitor started for tmux session ${SESSION}"
while tmux has-session -t "${SESSION}" 2>/dev/null; do
  progress="$(tmux capture-pane -pt "${SESSION}" -S -20 2>/dev/null | grep -E '[0-9.]+[GM] +[0-9]+%|done (scldm|scfoundation)' | tail -n 1 || true)"
  log "progress=${progress}"
  sleep 600
done
log "session ${SESSION} finished"
