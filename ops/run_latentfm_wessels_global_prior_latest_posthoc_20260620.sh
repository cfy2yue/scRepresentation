#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_wessels_global_prior_latest_posthoc_20260620
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
SPLIT_FILE=${ROOT}/runs/latentfm_dataset_upper_bound_20260620/latentfm_upperbound_wessels_split_seed42_20260620.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

BASELINE_DIR=${COUPLED}/output/latentfm_runs/dataset_upper_bound_20260620/scf_prior010_upperbound_wessels_4k
RUN010_DIR=${COUPLED}/output/latentfm_runs/wessels_global_prior_20260620/scf_globalprior010_add005_wessels_4k
RUN020_DIR=${COUPLED}/output/latentfm_runs/wessels_global_prior_sweep_20260620/scf_globalprior020_add010_wessels_4k
RUN000_DIR=${COUPLED}/output/latentfm_runs/wessels_global_prior_sweep_20260620/scf_globalprior000_add010_wessels_4k

mkdir -p "${RUN_ROOT}/logs" "${ROOT}/reports"
trap 'rc=$?; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; echo "${rc}" > "${RUN_ROOT}/EXIT_CODE"; exit "${rc}"' EXIT

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_wessels_global_prior_latest_posthoc_20260620

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_wessels_global_prior_latest_posthoc_20260620.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`latentfm_wessels_global_prior_latest_posthoc_20260620\`

## Log path

\`${RUN_ROOT}/logs/run.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_LATEST_GATE_AUDIT_20260620.md\`
* \`${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_LATEST_SUMMARY_20260620.md\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo still-running
tail -n 80 ${RUN_ROOT}/logs/run.log
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Evaluates \`latest.pt\` checkpoints only. This is a diagnostic for checkpoint
selection behavior and does not replace promotion-grade uncapped posthoc.
EOF

log=${RUN_ROOT}/logs/run.log
{
  echo "[$(date '+%F %T %Z')] exact GPU status before Wessels global-prior latest posthoc"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" "${GPU_HELPER}" --samples 3 --interval-seconds 10 --need 1 --json-only \
    > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"
  gpu="$("${PYTHON}" - "${gpu_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = payload.get("suggested_job_gpus") or []
print(chosen[0] if chosen else "")
PY
)"
  if [[ -z "${gpu}" ]]; then
    echo "No GPU selected; see ${gpu_json}" >&2
    exit 4
  fi

  resource_audit="${RUN_ROOT}/logs/resource_audit_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" - "${gpu_json}" "${resource_audit}" <<'PY'
import json
import os
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
system = payload.get("system") or {}
min_mem = float(os.environ.get("MIN_POSTHOC_MEM_AVAILABLE_GIB", "64"))
max_load = float(os.environ.get("MAX_POSTHOC_LOAD1_PER_CPU", "2.0"))
mem = float(system.get("mem_available_gib") or 0.0)
load = float(system.get("load1_per_cpu") or 0.0)
audit = {"status": "pass", "min_mem_available_gib": min_mem, "max_load1_per_cpu": max_load, "system": system, "gpu_selection_json": str(sys.argv[1])}
reasons = []
if mem < min_mem:
    reasons.append(f"MemAvailable {mem:.1f} GiB < {min_mem:.1f} GiB")
if load > max_load:
    reasons.append(f"load1_per_cpu {load:.3f} > {max_load:.3f}")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 5)
PY

  cd "${COUPLED}"
  source "${ROOT}/init-scdfm.sh" >/dev/null
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  export BLIS_NUM_THREADS=4

  eval_one() {
    local name="$1"
    local run_dir="$2"
    local out_dir="$3"
    mkdir -p "${out_dir}"
    echo "[$(date '+%F %T %Z')] latest posthoc ${name}"
    "${PYTHON}" -m model.latent.eval_split_groups \
      --checkpoint "${run_dir}/latest.pt" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --split-file "${SPLIT_FILE}" \
      --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${out_dir}/split_group_eval_latest_ode20_mse1024_mmd1024_stablecaps.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    "${PYTHON}" -m model.latent.eval_condition_families \
      --checkpoint "${run_dir}/latest.pt" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --split-file "${SPLIT_FILE}" \
      --groups test_all family_gene structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${out_dir}/condition_family_eval_latest_ode20_mse1024_mmd1024_stablecaps.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    "${PYTHON}" "${ROOT}/ops/audit_latentfm_stablecaps_selection.py" \
      --split-json "${out_dir}/split_group_eval_latest_ode20_mse1024_mmd1024_stablecaps.json" \
      --family-json "${out_dir}/condition_family_eval_latest_ode20_mse1024_mmd1024_stablecaps.json" \
      --out-json "${ROOT}/reports/latentfm_stablecaps_selection_audit_${name}_latest_20260620.json" \
      --out-md "${ROOT}/reports/LATENTFM_STABLECAPS_SELECTION_AUDIT_${name}_LATEST_20260620.md"
  }

  eval_one "baseline_latest_scf_prior010_upperbound_wessels_4k" "${BASELINE_DIR}" "${BASELINE_DIR}/posthoc_eval_latest_global_prior"
  eval_one "scf_globalprior010_add005_wessels_4k" "${RUN010_DIR}" "${RUN010_DIR}/posthoc_eval_latest_global_prior"
  eval_one "scf_globalprior020_add010_wessels_4k" "${RUN020_DIR}" "${RUN020_DIR}/posthoc_eval_latest_global_prior"
  eval_one "scf_globalprior000_add010_wessels_4k" "${RUN000_DIR}" "${RUN000_DIR}/posthoc_eval_latest_global_prior"

  "${PYTHON}" "${ROOT}/ops/audit_wessels_global_prior_gate_20260620.py" \
    --baseline-name scf_prior010_upperbound_wessels_4k_latest \
    --baseline-json "${BASELINE_DIR}/posthoc_eval_latest_global_prior/split_group_eval_latest_ode20_mse1024_mmd1024_stablecaps.json" \
    --run scf_globalprior010_add005_wessels_4k_latest "${RUN010_DIR}/posthoc_eval_latest_global_prior/split_group_eval_latest_ode20_mse1024_mmd1024_stablecaps.json" \
    --run scf_globalprior020_add010_wessels_4k_latest "${RUN020_DIR}/posthoc_eval_latest_global_prior/split_group_eval_latest_ode20_mse1024_mmd1024_stablecaps.json" \
    --run scf_globalprior000_add010_wessels_4k_latest "${RUN000_DIR}/posthoc_eval_latest_global_prior/split_group_eval_latest_ode20_mse1024_mmd1024_stablecaps.json" \
    --out-json "${ROOT}/reports/latentfm_wessels_global_prior_latest_gate_audit_20260620.json" \
    --out-md "${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_LATEST_GATE_AUDIT_20260620.md"

  "${PYTHON}" - "${ROOT}/reports/latentfm_wessels_global_prior_latest_gate_audit_20260620.json" \
    "${ROOT}/reports/latentfm_wessels_global_prior_latest_summary_20260620.json" \
    "${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_LATEST_SUMMARY_20260620.md" <<'PY'
import json
import sys
from pathlib import Path

audit_path = Path(sys.argv[1])
out_json = Path(sys.argv[2])
out_md = Path(sys.argv[3])
payload = json.loads(audit_path.read_text(encoding="utf-8"))

rows = []
for run in payload["runs"]:
    groups = run["groups"]
    rows.append({
        "run": run["run"],
        "status": run["status"],
        "unseen2_pp_delta": run["unseen2_pp_delta"],
        "test_mmd_ratio": run["test_mmd_ratio"],
        "test_pp_delta": groups["test"]["pearson_pert"]["delta"],
        "unseen1_pp_delta": groups["test_multi_unseen1"]["pearson_pert"]["delta"],
        "seen_pp_delta": groups["test_multi_seen"]["pearson_pert"]["delta"],
    })

summary = {"source_gate_audit": str(audit_path), "decision": payload["decision"], "rows": rows}
out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

def fmt(v):
    return "NA" if v is None else f"{float(v):.6f}"

lines = [
    "# Wessels Global Prior Latest-Checkpoint Summary",
    "",
    "This evaluates `latest.pt` for baseline and global-prior arms after `best.pt` posthoc collapsed to early checkpoints.",
    "",
    f"Decision: `{payload['decision']['status']}`",
    f"Next action: `{payload['decision']['next_action']}`",
    "",
    "| run | status | test pp delta | seen pp delta | unseen1 pp delta | unseen2 pp delta | test MMD ratio |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| `{row['run']}` | {row['status']} | {fmt(row['test_pp_delta'])} | "
        f"{fmt(row['seen_pp_delta'])} | {fmt(row['unseen1_pp_delta'])} | "
        f"{fmt(row['unseen2_pp_delta'])} | {fmt(row['test_mmd_ratio'])} |"
    )
lines.extend([
    "",
    "Latest-checkpoint posthoc is diagnostic only. If it differs from `best.pt`, the immediate issue is checkpoint selection, not necessarily model capacity.",
    "",
])
out_md.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "decision": payload["decision"]}, indent=2))
PY

  echo "[$(date '+%F %T %Z')] finished Wessels global-prior latest posthoc"
} > "${log}" 2>&1
