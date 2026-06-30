#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
TRAIN_RUN_ROOT=${ROOT}/runs/latentfm_wessels_global_prior_20260620
RUN_ROOT=${ROOT}/runs/latentfm_wessels_global_prior_posthoc_20260620
TRAIN_OUT_ROOT=${COUPLED}/output/latentfm_runs/wessels_global_prior_20260620
BASELINE_DIR=${COUPLED}/output/latentfm_runs/dataset_upper_bound_20260620/scf_prior010_upperbound_wessels_4k
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
SPLIT_FILE=${ROOT}/runs/latentfm_dataset_upper_bound_20260620/latentfm_upperbound_wessels_split_seed42_20260620.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=scf_globalprior010_add005_wessels_4k
mkdir -p "${RUN_ROOT}/logs" "${ROOT}/reports"
trap 'rc=$?; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; echo "${rc}" > "${RUN_ROOT}/EXIT_CODE"; exit "${rc}"' EXIT

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_wessels_global_prior_posthoc_20260620

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_wessels_global_prior_posthoc_20260620.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`latentfm_wessels_global_prior_posthoc_20260620\`

## Log path

\`${RUN_ROOT}/logs/run.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_WESSELS_GLOBAL_PRIOR_SUMMARY_20260620.md\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo still-running
tail -n 80 ${RUN_ROOT}/logs/run.log
nvidia-smi
\`\`\`

## Current status

Waiting for training EXIT_CODE.

## Notes

Checks training completion at 30-minute cadence with \`sleep 1800\`.
EOF

log=${RUN_ROOT}/logs/run.log
{
  echo "[$(date '+%F %T %Z')] wait for Wessels global-prior training exit"
  while [[ ! -f "${TRAIN_RUN_ROOT}/${RUN_NAME}.EXIT_CODE" ]]; do
    echo "[$(date '+%F %T %Z')] still waiting; next check in 1800s"
    sleep 1800
  done
  code="$(cat "${TRAIN_RUN_ROOT}/${RUN_NAME}.EXIT_CODE")"
  echo "[$(date '+%F %T %Z')] train ${RUN_NAME} exit=${code}"
  if [[ "${code}" != "0" ]]; then
    exit "${code}"
  fi

  echo "[$(date '+%F %T %Z')] exact GPU status before Wessels global-prior posthoc"
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

  out_dir="${TRAIN_OUT_ROOT}/${RUN_NAME}/posthoc_eval_global_prior"
  mkdir -p "${out_dir}"
  "${PYTHON}" -m model.latent.eval_split_groups \
    --checkpoint "${TRAIN_OUT_ROOT}/${RUN_NAME}/best.pt" \
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
    --checkpoint "${TRAIN_OUT_ROOT}/${RUN_NAME}/best.pt" \
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
    --out-json "${ROOT}/reports/latentfm_stablecaps_selection_audit_${RUN_NAME}_20260620.json" \
    --out-md "${ROOT}/reports/LATENTFM_STABLECAPS_SELECTION_AUDIT_${RUN_NAME}_20260620.md"

  "${PYTHON}" - "${ROOT}" \
    "${BASELINE_DIR}/posthoc_eval_upperbound/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
    "${out_dir}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
    "${RUN_NAME}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base_path = Path(sys.argv[2])
run_path = Path(sys.argv[3])
run_name = sys.argv[4]
base = json.loads(base_path.read_text(encoding="utf-8"))
run = json.loads(run_path.read_text(encoding="utf-8"))
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
for name in groups:
    bg = group(base, name)
    rg = group(run, name)
    bpp = fval(bg, "pearson_pert")
    rpp = fval(rg, "pearson_pert")
    bpc = fval(bg, "pearson_ctrl")
    rpc = fval(rg, "pearson_ctrl")
    bmmd = fval(bg, "test_mmd_clamped") if bg.get("test_mmd_clamped") is not None else fval(bg, "test_mmd")
    rmmd = fval(rg, "test_mmd_clamped") if rg.get("test_mmd_clamped") is not None else fval(rg, "test_mmd")
    rows.append({
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

unseen2 = next((row for row in rows if row["group"] == "test_multi_unseen2"), {})
test = next((row for row in rows if row["group"] == "test"), {})
gate = {
    "pp_delta_min": 0.05,
    "mmd_ratio_max": 1.15,
    "pp_delta": unseen2.get("pp_delta"),
    "test_mmd_ratio": test.get("mmd_ratio"),
}
gate["status"] = (
    "pass"
    if gate["pp_delta"] is not None
    and gate["pp_delta"] >= gate["pp_delta_min"]
    and gate["test_mmd_ratio"] is not None
    and gate["test_mmd_ratio"] <= gate["mmd_ratio_max"]
    else "fail"
)

out_csv = root / "reports" / "latentfm_wessels_global_prior_summary_20260620.csv"
out_json = root / "reports" / "latentfm_wessels_global_prior_summary_20260620.json"
out_md = root / "reports" / "LATENTFM_WESSELS_GLOBAL_PRIOR_SUMMARY_20260620.md"
with out_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
payload = {
    "baseline": "scf_prior010_upperbound_wessels_4k",
    "run": run_name,
    "gate": gate,
    "rows": rows,
}
out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def fmt(v):
    return "NA" if v is None else f"{float(v):.6f}"

lines = [
    "# LatentFM Wessels Global Train-Only Prior Summary",
    "",
    "Baseline: `scf_prior010_upperbound_wessels_4k`.",
    "",
    f"Gate: `{gate['status']}`; unseen2 pp delta {fmt(gate['pp_delta'])} vs >= {gate['pp_delta_min']}, test MMD ratio {fmt(gate['test_mmd_ratio'])} vs <= {gate['mmd_ratio_max']}.",
    "",
    "| group | n | pp baseline | pp | delta pp | pc baseline | pc | delta pc | MMD baseline | MMD | MMD ratio |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| `{row['group']}` | {row['n']} | {fmt(row['pp_baseline'])} | {fmt(row['pp'])} | {fmt(row['pp_delta'])} | "
        f"{fmt(row['pc_baseline'])} | {fmt(row['pc'])} | {fmt(row['pc_delta'])} | {fmt(row['mmd_baseline'])} | {fmt(row['mmd'])} | {fmt(row['mmd_ratio'])} |"
    )
lines.extend(
    [
        "",
        "This is a Wessels-only diagnostic of a global train-only gene-response prior teacher.",
        "Promotion still requires full all-split retraining, uncapped posthoc, bootstrap CI, and leakage audit.",
        "",
    ]
)
out_md.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "out_csv": str(out_csv), "gate": gate}, indent=2))
PY

  echo "[$(date '+%F %T %Z')] finished Wessels global-prior posthoc"
} > "${log}" 2>&1
