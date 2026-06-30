#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
exec bash "${ROOT}/ops/restart_latentfm_trackc_trainonly_memory_mc256_posthoc_if_ready_20260622.sh"
