#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
TRAIN_RUN_ROOT=${ROOT}/runs/latentfm_wessels_global_prior_sweep_20260620
RUN_ROOT=${ROOT}/runs/latentfm_wessels_global_prior_sweep_posthoc_20260620
TRAIN_OUT_ROOT=${COUPLED}/output/latentfm_runs/wessels_global_prior_sweep_20260620
BASELINE_DIR=${COUPLED}/output/latentfm_runs/dataset_upper_bound_20260620/scf_prior010_upperbound_wessels_4k
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
SPLIT_FILE=${ROOT}/runs/latentfm_dataset_upper_bound_20260620/latentfm_upperbound_wessels_split_seed42_20260620.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUNS=(
  "scf_globalprior020_add010_wessels_4k"
  "scf_globalprior000_add010_wessels_4k"
)

mkdir -p "${RUN_ROOT}/logs" "${ROOT}/reports"
trap 'rc=$?; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; echo "${rc}" > "${RUN_ROOT}/EXIT_CODE"; exit "${rc}"' EXIT

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_wessels_global_prior_sweep_posthoc_20260620

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_wessels_global_prior_sweep_posthoc_20260620.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`latentfm_wessels_global_prior_sweep_posthoc_20260620\`

## Log path

\`${RUN_ROOT}/logs/run.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_SWEEP_SUMMARY_20260620.md\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo still-running
tail -n 80 ${RUN_ROOT}/logs/run.log
nvidia-smi
\`\`\`

## Current status

Waiting for training EXIT_CODE files.

## Notes

Checks training completion at 30-minute cadence with \`sleep 1800\`.
EOF

log=${RUN_ROOT}/logs/run.log
{
  echo "[$(date '+%F %T %Z')] wait for Wessels global-prior sweep training exits"
  while true; do
    all_done=1
    for run in "${RUNS[@]}"; do
      if [[ ! -f "${TRAIN_RUN_ROOT}/${run}.EXIT_CODE" ]]; then
        all_done=0
      fi
    done
    if [[ "${all_done}" == "1" ]]; then
      break
    fi
    echo "[$(date '+%F %T %Z')] still waiting; next check in 1800s"
    sleep 1800
  done
  for run in "${RUNS[@]}"; do
    code="$(cat "${TRAIN_RUN_ROOT}/${run}.EXIT_CODE")"
    echo "[$(date '+%F %T %Z')] train ${run} exit=${code}"
    if [[ "${code}" != "0" ]]; then
      exit "${code}"
    fi
  done

  echo "[$(date '+%F %T %Z')] exact GPU status before Wessels global-prior sweep posthoc"
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

  for run in "${RUNS[@]}"; do
    out_dir="${TRAIN_OUT_ROOT}/${run}/posthoc_eval_global_prior_sweep"
    mkdir -p "${out_dir}"
    "${PYTHON}" -m model.latent.eval_split_groups \
      --checkpoint "${TRAIN_OUT_ROOT}/${run}/best.pt" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --split-file "${SPLIT_FILE}" \
      --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${out_dir}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    "${PYTHON}" -m model.latent.eval_condition_families \
      --checkpoint "${TRAIN_OUT_ROOT}/${run}/best.pt" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --split-file "${SPLIT_FILE}" \
      --groups test_all family_gene structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${out_dir}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    "${PYTHON}" "${ROOT}/ops/audit_latentfm_stablecaps_selection.py" \
      --split-json "${out_dir}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --family-json "${out_dir}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --out-json "${ROOT}/reports/latentfm_stablecaps_selection_audit_${run}_20260620.json" \
      --out-md "${ROOT}/reports/LATENTFM_STABLECAPS_SELECTION_AUDIT_${run}_20260620.md"
  done

  "${PYTHON}" - "${ROOT}" "${BASELINE_DIR}" "${TRAIN_OUT_ROOT}" "${RUNS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base_dir = Path(sys.argv[2])
out_root = Path(sys.argv[3])
runs = sys.argv[4:]
base_path = base_dir / "posthoc_eval_upperbound/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
base = json.loads(base_path.read_text(encoding="utf-8"))
groups = ["test", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"]

def fval(g, key):
    try:
        return float(g.get(key))
    except (TypeError, ValueError):
        return None

def group(obj, name):
    item = obj.get("groups", {}).get(name, {})
    return item if isinstance(item, dict) else {}

rows = []
for run in runs:
    run_path = out_root / run / "posthoc_eval_global_prior_sweep/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    for name in groups:
        bg = group(base, name)
        rg = group(payload, name)
        bpp = fval(bg, "pearson_pert")
        rpp = fval(rg, "pearson_pert")
        bpc = fval(bg, "pearson_ctrl")
        rpc = fval(rg, "pearson_ctrl")
        bmmd = fval(bg, "test_mmd_clamped") if bg.get("test_mmd_clamped") is not None else fval(bg, "test_mmd")
        rmmd = fval(rg, "test_mmd_clamped") if rg.get("test_mmd_clamped") is not None else fval(rg, "test_mmd")
        rows.append({
            "run": run,
            "group": name,
            "n": rg.get("n_conds"),
            "pp_baseline": bpp,
            "pp": rpp,
            "pp_delta": None if bpp is None or rpp is None else rpp - bpp,
            "pc_baseline": bpc,
            "pc": rpc,
            "pc_delta": None if bpc is None or rpc is None else rpc - bpc,
            "mmd_baseline": bmmd,
            "mmd": rmmd,
            "mmd_ratio": None if bmmd is None or rmmd is None else rmmd / max(bmmd, 1e-12),
        })

gate_rows = []
for run in runs:
    u2 = next((r for r in rows if r["run"] == run and r["group"] == "test_multi_unseen2"), {})
    test = next((r for r in rows if r["run"] == run and r["group"] == "test"), {})
    status = (
        "pass"
        if u2.get("pp_delta") is not None
        and u2["pp_delta"] >= 0.05
        and test.get("mmd_ratio") is not None
        and test["mmd_ratio"] <= 1.15
        else "fail"
    )
    gate_rows.append({
        "run": run,
        "status": status,
        "unseen2_pp_delta": u2.get("pp_delta"),
        "test_mmd_ratio": test.get("mmd_ratio"),
    })

out_csv = root / "reports/latentfm_wessels_global_prior_sweep_summary_20260620.csv"
out_json = root / "reports/latentfm_wessels_global_prior_sweep_summary_20260620.json"
out_md = root / "reports/LATENTFM_WESSELS_GLOBAL_PRIOR_SWEEP_SUMMARY_20260620.md"
with out_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
payload = {"baseline": "scf_prior010_upperbound_wessels_4k", "gate_rows": gate_rows, "rows": rows}
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def fmt(v):
    return "NA" if v is None else f"{float(v):.6f}"

lines = [
    "# LatentFM Wessels Global Prior Sweep Summary",
    "",
    "Baseline: `scf_prior010_upperbound_wessels_4k`.",
    "",
    "## Gate",
    "",
    "| run | status | unseen2 pp delta | test MMD ratio |",
    "|---|---|---:|---:|",
]
for row in gate_rows:
    lines.append(f"| `{row['run']}` | {row['status']} | {fmt(row['unseen2_pp_delta'])} | {fmt(row['test_mmd_ratio'])} |")
lines.extend([
    "",
    "## Group Details",
    "",
    "| run | group | n | pp baseline | pp | delta pp | pc baseline | pc | delta pc | MMD baseline | MMD | MMD ratio |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
])
for row in rows:
    lines.append(
        f"| `{row['run']}` | `{row['group']}` | {row['n']} | {fmt(row['pp_baseline'])} | {fmt(row['pp'])} | {fmt(row['pp_delta'])} | "
        f"{fmt(row['pc_baseline'])} | {fmt(row['pc'])} | {fmt(row['pc_delta'])} | {fmt(row['mmd_baseline'])} | {fmt(row['mmd'])} | {fmt(row['mmd_ratio'])} |"
    )
lines.extend(["", "Stable-caps diagnostic only; promotion requires uncapped posthoc and bootstrap CI.", ""])
out_md.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "out_csv": str(out_csv), "gate_rows": gate_rows}, indent=2))
PY

  echo "[$(date '+%F %T %Z')] finished Wessels global-prior sweep posthoc"
} > "${log}" 2>&1
