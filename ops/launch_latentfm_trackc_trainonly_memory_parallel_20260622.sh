#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
LAUNCHER=${ROOT}/ops/launch_latentfm_xverse_trackc_routed_distill_smoke_20260622.sh
RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_trainonly_memory_parallel_mc256_20260622
OUT_ROOT=${ROOT}/CoupledFM/output/latentfm_runs/xverse_trackc_trainonly_memory_parallel_mc256_20260622
LOG_ROOT=${ROOT}/logs/latentfm_xverse_trackc_trainonly_memory_parallel_mc256_20260622
TRAINSELECT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_route_datasets_trainselect.json
BANK_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
ROUTE_FILE=${ROOT}/reports/latentfm_trackc_trainonly_memory_route_teacher_20260622.json
VALIDATION_JSON=${ROOT}/reports/latentfm_trackc_trainonly_memory_teacher_validation_20260622.json
MANIFEST=${ROOT}/reports/latentfm_trackc_trainonly_memory_parallel_mc256_manifest_20260622.jsonl
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${RUN_ROOT}" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"
if [[ -e "${MANIFEST}" ]]; then
  echo "Manifest already exists: ${MANIFEST}" >&2
  exit 2
fi

"${PYTHON}" - "${VALIDATION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "trainonly_memory_teacher_validation_pass":
    raise SystemExit(f"memory teacher validation is not pass: {payload.get('status')}")
PY

"${PYTHON}" - "${ROOT}/reports" <<'PY'
import json
import sys
from pathlib import Path

reports = Path(sys.argv[1])
runs = [
    "xverse_trackc_noharm_pc_ep050_replay2_all_2k_seed42",
    "xverse_trackc_noharm_pc_ep050_replay4_nongm_2k_seed42",
    "xverse_trackc_noharm_pc_ep050del_replay2_all_2k_seed42",
    "xverse_trackc_noharm_pc_ep100_replay2_all_2k_seed42",
    "xverse_trackc_noharm_pc_ep100del_replay4_all_2k_seed42",
    "xverse_trackc_noharm_pc_ep100del_replay4_nongm_2k_seed42",
]
statuses = {}
for run in runs:
    path = reports / f"latentfm_trackc_pairwise_latest_decision_{run}.json"
    if not path.is_file():
        raise SystemExit(f"latest decision missing, do not launch memory branch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    decision = payload.get("decision") or {}
    status = str(payload.get("status") or payload.get("decision_status") or decision.get("status") or "")
    statuses[run] = status
    if status == "trackc_smoke_support_pass_needs_uncapped_noharm_before_query":
        raise SystemExit(f"latest checkpoint passed for {run}; do not launch memory branch")
    if not status.startswith("trackc_smoke_fail_"):
        raise SystemExit(f"latest checkpoint status not closed for {run}: {status}")
print(json.dumps({"latest_statuses": statuses}, indent=2))
PY

launch_one() {
  local run_name="$1"
  local forced_gpu="$2"
  local scope="$3"
  local pairwise_mode="$4"
  local endpoint_weight="$5"
  local endpoint_start="$6"
  local endpoint_end="$7"
  local replay_weight="$8"
  local replay_filter="$9"
  local hypothesis="${10}"

  echo "[$(date '+%F %T %Z')] launching ${run_name} on GPU${forced_gpu}"
  LATENTFM_TRACKC_RUN_NAME="${run_name}" \
  LATENTFM_TRACKC_RUN_ROOT="${RUN_ROOT}" \
  LATENTFM_TRACKC_OUT_ROOT="${OUT_ROOT}" \
  LATENTFM_TRACKC_LOG_ROOT="${LOG_ROOT}" \
  LATENTFM_TRACKC_TRAINSELECT_SPLIT="${TRAINSELECT}" \
  LATENTFM_TRACKC_BANK_SPLIT_FILE="${BANK_SPLIT}" \
  LATENTFM_TRACKC_ROUTE_FILE="${ROUTE_FILE}" \
  LATENTFM_TRACKC_INIT_CHECKPOINT_USE_EMA=1 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1 \
  LATENTFM_TRACKC_FINETUNE_TRAINABLE_SCOPE="${scope}" \
  LATENTFM_TRACKC_PERT_PAIRWISE_MODE="${pairwise_mode}" \
  LATENTFM_TRACKC_FORCE_GPU="${forced_gpu}" \
  LATENTFM_TRACKC_RELAXED_GPU_SELECTION=1 \
  LATENTFM_TRACKC_RELAXED_GPU_MIN_FREE_MIB=8192 \
  LATENTFM_TRACKC_TOTAL_STEPS=2000 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WEIGHT=0.0 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ROUTED_DISTILL_LOSS_WARMUP_END=500 \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WEIGHT="${endpoint_weight}" \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_START="${endpoint_start}" \
  LATENTFM_TRACKC_ROUTED_ENDPOINT_LOSS_WARMUP_END="${endpoint_end}" \
  LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MODE=jaccard \
  LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_K=3 \
  LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_MIN_SCORE=0.25 \
  LATENTFM_TRACKC_ROUTED_DISTILL_MEMORY_SCOPE=same_dataset \
  LATENTFM_TRACKC_CONDITION_PRIOR_BANK_MAX_CELLS=256 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WEIGHT="${replay_weight}" \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_START=0 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_LOSS_WARMUP_END=500 \
  LATENTFM_TRACKC_ANCHOR_REPLAY_CONDITION_FILTER="${replay_filter}" \
  LATENTFM_TRACKC_SMOKE_HYPOTHESIS="${hypothesis}" \
  bash "${LAUNCHER}"

  "${PYTHON}" - "${MANIFEST}" "${run_name}" "${forced_gpu}" "${scope}" "${pairwise_mode}" \
    "${endpoint_weight}" "${endpoint_start}" "${endpoint_end}" "${replay_weight}" "${replay_filter}" "${hypothesis}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

manifest = Path(sys.argv[1])
run_name = sys.argv[2]
row = {
    "launched_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    "run_name": run_name,
    "forced_gpu": int(sys.argv[3]),
    "finetune_trainable_scope": sys.argv[4],
    "pert_pairwise_mode": sys.argv[5],
    "endpoint_weight": float(sys.argv[6]),
    "endpoint_warmup_start": int(sys.argv[7]),
    "endpoint_warmup_end": int(sys.argv[8]),
    "anchor_replay_weight": float(sys.argv[9]),
    "anchor_replay_filter": sys.argv[10],
    "memory_rule": "jaccard/k3/same_dataset/min_score0.25",
    "hypothesis": sys.argv[11],
    "run_status": f"/data/cyx/1030/scLatent/runs/latentfm_xverse_trackc_trainonly_memory_parallel_mc256_20260622/{run_name}/RUN_STATUS.md",
    "decision_json": f"/data/cyx/1030/scLatent/reports/latentfm_trackc_routed_distill_smoke_decision_{run_name}.json",
    "decision_md": f"/data/cyx/1030/scLatent/reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_{run_name}.md",
}
with manifest.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

launch_one xverse_trackc_mem256_cp_ep050_replay2_all_2k_seed42 1 \
  condition_prior_adapter off 0.50 0 500 2.0 all \
  "Train-only memory endpoint w0.5 tests whether the frozen train_multi memory teacher transfers support-val signal without canonical harm."

launch_one xverse_trackc_mem256_cp_ep100_replay2_all_2k_seed42 2 \
  condition_prior_adapter off 1.00 0 500 2.0 all \
  "Train-only memory endpoint w1.0 tests support-gain dose response under all-condition replay."

launch_one xverse_trackc_mem256_pc_ep050_replay2_all_2k_seed42 3 \
  pairwise_condition_adapter hadamard_mean 0.50 0 500 2.0 all \
  "Pairwise adapter plus train-only memory endpoint w0.5 tests whether interaction capacity improves memory-transfer absorption."

launch_one xverse_trackc_mem256_pc_ep050_replay4_nongm_2k_seed42 2 \
  pairwise_condition_adapter hadamard_mean 0.50 0 500 4.0 non_gene_multi \
  "Pairwise adapter plus canonical-only replay tests whether non-gene-multi replay protects canonical strata while leaving support multi trainable."

echo "trainonly_memory_parallel_mc256_launched"
echo "manifest=${MANIFEST}"
