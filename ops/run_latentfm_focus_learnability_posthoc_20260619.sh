#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
TRAIN_RUN_ROOT=${ROOT}/runs/latentfm_focus_learnability_20260619
RUN_ROOT=${ROOT}/runs/latentfm_focus_learnability_posthoc_20260619
TRAIN_OUT_ROOT=${COUPLED}/output/latentfm_runs/focus_learnability_20260619
BASELINE_DIR=${COUPLED}/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
FOCUS_SPLIT=${TRAIN_RUN_ROOT}/latentfm_focus_nwg_split_seed42_20260619.json
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_A=scf_prior010_inject_nwg_focus_4k
RUN_B=scf_prior010_inject_nwg_focus_dsloss05_4k

mkdir -p "${RUN_ROOT}/logs" "${ROOT}/reports"
trap 'rc=$?; date "+%F %T %Z" > "${RUN_ROOT}/FINISHED"; echo "${rc}" > "${RUN_ROOT}/EXIT_CODE"; exit "${rc}"' EXIT

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_focus_learnability_posthoc_20260619

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_focus_learnability_posthoc_20260619.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux: \`latentfm_focus_learnability_posthoc_20260619\`

## Log path

\`${RUN_ROOT}/logs/run.log\`

## Expected outputs

* \`${TRAIN_OUT_ROOT}/${RUN_A}/posthoc_eval_focus_nwg/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json\`
* \`${TRAIN_OUT_ROOT}/${RUN_B}/posthoc_eval_focus_nwg/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json\`
* \`${ROOT}/reports/LATENTFM_FOCUS_LEARNABILITY_STABLECAPS_SUMMARY_20260619.md\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo still-running
tail -n 80 ${RUN_ROOT}/logs/run.log
nvidia-smi
\`\`\`

## Current status

Waiting for focus training EXIT_CODE files.

## Notes

This script checks training completion at 30-minute cadence with \`sleep 1800\`.
EOF

log=${RUN_ROOT}/logs/run.log
{
  echo "[$(date '+%F %T %Z')] wait for focus training exits"
  while true; do
    all_done=1
    for run in "${RUN_A}" "${RUN_B}"; do
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

  for run in "${RUN_A}" "${RUN_B}"; do
    code="$(cat "${TRAIN_RUN_ROOT}/${run}.EXIT_CODE")"
    echo "[$(date '+%F %T %Z')] train ${run} exit=${code}"
    if [[ "${code}" != "0" ]]; then
      echo "training failed for ${run}; skip posthoc" >&2
      exit "${code}"
    fi
  done

  echo "[$(date '+%F %T %Z')] exact GPU status before focus posthoc"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" "${GPU_HELPER}" \
    --samples 3 \
    --interval-seconds 10 \
    --need 1 \
    --json-only \
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

gpu_json = Path(sys.argv[1])
out = Path(sys.argv[2])
payload = json.loads(gpu_json.read_text(encoding="utf-8"))
system = payload.get("system") or {}
min_mem = float(os.environ.get("MIN_POSTHOC_MEM_AVAILABLE_GIB", "64"))
max_load = float(os.environ.get("MAX_POSTHOC_LOAD1_PER_CPU", "2.0"))
mem = float(system.get("mem_available_gib") or 0.0)
load = float(system.get("load1_per_cpu") or 0.0)
audit = {
    "status": "pass",
    "min_mem_available_gib": min_mem,
    "max_load1_per_cpu": max_load,
    "system": system,
    "gpu_selection_json": str(gpu_json),
}
reasons = []
if mem < min_mem:
    reasons.append(f"MemAvailable {mem:.1f} GiB < {min_mem:.1f} GiB")
if load > max_load:
    reasons.append(f"load1_per_cpu {load:.3f} > {max_load:.3f}")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
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

  run_posthoc() {
    local name="$1"
    local run_dir="$2"
    local posthoc="${run_dir}/posthoc_eval_focus_nwg"
    mkdir -p "${posthoc}"
    echo "[$(date '+%F %T %Z')] posthoc ${name}"
    test -f "${run_dir}/best.pt"
    "${PYTHON}" -m model.latent.eval_split_groups \
      --checkpoint "${run_dir}/best.pt" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --split-file "${FOCUS_SPLIT}" \
      --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${posthoc}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    "${PYTHON}" -m model.latent.eval_condition_families \
      --checkpoint "${run_dir}/best.pt" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${ROOT}/dataset/biFlow_data" \
      --split-file "${FOCUS_SPLIT}" \
      --groups test_all family_gene structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${posthoc}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    "${PYTHON}" "${ROOT}/ops/audit_latentfm_stablecaps_selection.py" \
      --split-json "${posthoc}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --family-json "${posthoc}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json" \
      --out-json "${ROOT}/reports/latentfm_stablecaps_selection_audit_${name}_focus_nwg_20260619.json" \
      --out-md "${ROOT}/reports/LATENTFM_STABLECAPS_SELECTION_AUDIT_${name}_FOCUS_NWG_20260619.md"
  }

  BASE_FOCUS_POSTHOC="${BASELINE_DIR}/posthoc_eval_focus_nwg_20260619"
  run_posthoc "baseline_scf_prior010_inject_e2_4k" "${BASELINE_DIR}"
  mkdir -p "${BASE_FOCUS_POSTHOC}"
  cp "${BASELINE_DIR}/posthoc_eval_focus_nwg/"*.json "${BASE_FOCUS_POSTHOC}/"

  run_posthoc "${RUN_A}" "${TRAIN_OUT_ROOT}/${RUN_A}"
  run_posthoc "${RUN_B}" "${TRAIN_OUT_ROOT}/${RUN_B}"

  "${PYTHON}" - "${ROOT}" "${BASELINE_DIR}" "${TRAIN_OUT_ROOT}" "${RUN_A}" "${RUN_B}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_dir = Path(sys.argv[2])
train_out = Path(sys.argv[3])
runs = [sys.argv[4], sys.argv[5]]
baseline_name = "scf_prior010_inject_e2_4k"
paths = {
    baseline_name: baseline_dir / "posthoc_eval_focus_nwg" / "split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json",
}
for run in runs:
    paths[run] = train_out / run / "posthoc_eval_focus_nwg" / "split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def metric(obj, group, key):
    val = obj.get("groups", {}).get(group, {}).get(key)
    return None if val is None else float(val)

def fmt(x, nd=6):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{float(x):.{nd}f}"

loaded = {name: load(path) for name, path in paths.items()}
base = loaded[baseline_name]
groups = ["test", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"]
rows = []
for name, obj in loaded.items():
    for group in groups:
        g = obj.get("groups", {}).get(group, {})
        if not g or g.get("skipped"):
            continue
        b_pp = metric(base, group, "pearson_pert")
        pp = metric(obj, group, "pearson_pert")
        b_mmd = metric(base, group, "test_mmd")
        mmd = metric(obj, group, "test_mmd")
        rows.append({
            "run": name,
            "group": group,
            "n": g.get("n_conds"),
            "pp": pp,
            "delta_pp": None if pp is None or b_pp is None else pp - b_pp,
            "mmd": mmd,
            "mmd_ratio": None if mmd is None or b_mmd in (None, 0.0) else mmd / b_mmd,
            "pc": metric(obj, group, "pearson_ctrl"),
            "dp": metric(obj, group, "direct_pearson"),
        })

csv_path = root / "reports" / "latentfm_focus_learnability_stablecaps_summary_20260619.csv"
md_path = root / "reports" / "LATENTFM_FOCUS_LEARNABILITY_STABLECAPS_SUMMARY_20260619.md"
gate_path = root / "reports" / "latentfm_focus_learnability_stablecaps_summary_20260619_gate.json"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["run", "group", "n", "pp", "delta_pp", "mmd", "mmd_ratio", "pc", "dp"])
    writer.writeheader()
    writer.writerows(rows)

gate = {"baseline": baseline_name, "runs": []}
for run in runs:
    r_test = next((r for r in rows if r["run"] == run and r["group"] == "test"), {})
    r_unseen2 = next((r for r in rows if r["run"] == run and r["group"] == "test_multi_unseen2"), {})
    gate["runs"].append({
        "run": run,
        "test_pp_delta": r_test.get("delta_pp"),
        "test_mmd_ratio": r_test.get("mmd_ratio"),
        "unseen2_pp_delta": r_unseen2.get("delta_pp"),
        "unseen2_pp": r_unseen2.get("pp"),
        "diagnostic_interpretation": "focus_learnability_signal" if (r_unseen2.get("delta_pp") or -999) > 0 else "no_focus_unseen2_rescue",
    })
gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")

lines = [
    "# LatentFM Focus Learnability Stable-Caps Summary",
    "",
    f"Baseline: `{baseline_name}` re-evaluated on Norman/Wessels/Gasperini focus split.",
    "",
    "| run | group | n | pp | delta_pp | MMD | MMD_ratio | pc | dp |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| `{r['run']}` | `{r['group']}` | {r['n']} | {fmt(r['pp'])} | {fmt(r['delta_pp'])} | {fmt(r['mmd'])} | {fmt(r['mmd_ratio'])} | {fmt(r['pc'])} | {fmt(r['dp'])} |"
    )
lines.extend([
    "",
    "## Gate",
    "",
    "This is a learnability diagnostic, not a promotion gate. A useful signal is improved focus `test_multi_unseen2` pp without a large MMD regression versus the re-evaluated baseline.",
    "",
])
md_path.write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"out_csv": str(csv_path), "out_md": str(md_path), "out_gate_json": str(gate_path)}, indent=2))
PY

  echo "[$(date '+%F %T %Z')] finished focus learnability posthoc"
} > "${log}" 2>&1
