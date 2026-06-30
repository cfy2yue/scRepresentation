#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

CPU_GATE_JSON=${ROOT}/reports/latentfm_trackc_support_context_v2_cpu_gate_20260623.json
LAUNCH_GATE_JSON=${ROOT}/reports/latentfm_trackc_support_context_v2_launcher_provenance_gate_20260623.json
GENERIC_LAUNCHER=${ROOT}/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh

"${PYTHON}" - "${CPU_GATE_JSON}" "${LAUNCH_GATE_JSON}" <<'PY'
import json
import sys
from pathlib import Path

cpu_gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
launch_gate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if cpu_gate.get("status") != "trackc_support_context_v2_cpu_gate_pass_launcher_gate_next_no_gpu":
    raise SystemExit("v2 CPU gate is not in launcher-gate-ready status")
if launch_gate.get("status") != "trackc_support_context_v2_launcher_provenance_gate_pass_launch_allowed":
    raise SystemExit("v2 launcher/provenance gate has not passed")
if launch_gate.get("gpu_authorization") != "one_capped_smoke_allowed_after_fresh_resource_audit":
    raise SystemExit("v2 launcher/provenance gate did not authorize a capped smoke")
PY

export LATENTFM_TRACKC_RUN_NAME=xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42
export LATENTFM_TRACKC_RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_support_context_v2_20260623
export LATENTFM_TRACKC_OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_support_context_v2_20260623
export LATENTFM_TRACKC_LOG_ROOT=${ROOT}/logs/latentfm_xverse_trackc_support_context_v2_20260623
export LATENTFM_TRACKC_TRAINSELECT_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
export LATENTFM_TRACKC_BANK_SPLIT_FILE=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
export LATENTFM_TRACKC_ROUTE_FILE=${ROOT}/reports/latentfm_trackc_residual_operator_route_teacher_20260623.json
export LATENTFM_TRACKC_ANCHOR_CKPT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt

export LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE=support_film_adapter
export LATENTFM_TRACKC_SUPPORT_CONTEXT_USE_IN_MODEL=0
export LATENTFM_TRACKC_SUPPORT_RESIDUAL_USE_IN_MODEL=0
export LATENTFM_TRACKC_SUPPORT_FILM_USE_IN_MODEL=1
export LATENTFM_TRACKC_SUPPORT_CONTEXT_DIM=384
export LATENTFM_TRACKC_SUPPORT_CONTEXT_SOURCE=routed_distill_target

export LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=0.0
export LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT=0.50
export LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START=0
export LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END=500
export LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MODE=jaccard
export LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_K=3
export LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE=0.25
export LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_SCOPE=all_dataset
export LATENTFM_TRACKC_CONDITION_PRIOR_BANK_MAX_CELLS=512

export LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WEIGHT=2.0
export LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_START=0
export LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_END=500
export LATENTFM_TRACKC_ANCHOR_REPLAY_CONDITION_FILTER=all
export LATENTFM_TRACKC_INIT_CHECKPOINT_USE_EMA=1
export LATENTFM_TRACKC_ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
export LATENTFM_TRACKC_CONDITION_DELTA_HEAD_USE_IN_MODEL=0
export LATENTFM_TRACKC_TOTAL_STEPS=2000
export LATENTFM_TRACKC_SMOKE_HYPOTHESIS="Track C support-context v2: explicit support context present on safe trainselect support-val and forced absent on canonical Track A; biasless support-FiLM residual/scale should absorb support teacher signal while preserving canonical exact no-op."

bash "${GENERIC_LAUNCHER}"
