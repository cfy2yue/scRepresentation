#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent

# Endpoint-routed Track C smoke. This intentionally reuses the routed-distill
# launcher/posthoc gate but disables the old head-only distill loss and enables
# direct endpoint supervision against the same trainselect routed teacher.
export LATENTFM_TRACKC_RUN_NAME=xverse_trackc_endpoint_route_w05_replay1_2k_seed42
export LATENTFM_TRACKC_RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_endpoint_routed_20260622
export LATENTFM_TRACKC_OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_endpoint_routed_20260622
export LATENTFM_TRACKC_LOG_ROOT=${ROOT}/logs/latentfm_xverse_trackc_endpoint_routed_20260622

export LATENTFM_TRACKC_TRAINSELECT_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_route_datasets_trainselect.json
export LATENTFM_TRACKC_BANK_SPLIT_FILE=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json

export LATENTFM_TRACKC_INIT_CHECKPOINT_USE_EMA=1
export LATENTFM_TRACKC_ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
export LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE=condition_prior_adapter

export LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=0.0
export LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START=0
export LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END=500
export LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT=0.5
export LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START=0
export LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END=500

export LATENTFM_TRACKC_SMOKE_HYPOTHESIS="endpoint-routed teacher supervision directly trains x1_hat toward src + routed_teacher_delta, avoiding the failed branch's head-only route supervision while preserving route-focused trainselect support exposure and EMA-aligned anchor replay"

bash ${ROOT}/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh
