#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_fewshot_multi_calibration_20260621
OUT_ROOT=${COUPLED}/output/latentfm_runs/fewshot_multi_calibration_20260621
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
ANCHOR_DIR=${COUPLED}/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
SUMMARY=${ROOT}/ops/summarize_latentfm_fewshot_multi_calibration_20260621.py
BOOTSTRAP_RUNNER=${ROOT}/ops/run_latentfm_posthoc_bootstrap_from_manifest_20260621.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

MANIFEST=${RUN_ROOT}/launch_manifest.json
LOG_DIR=${RUN_ROOT}/logs
mkdir -p "${LOG_DIR}" "${ROOT}/reports"
rm -f "${RUN_ROOT}/POSTHOC_EXIT_CODE" "${RUN_ROOT}/POSTHOC_FINISHED"
date '+%F %T %Z' > "${RUN_ROOT}/POSTHOC_STARTED"

cat > "${RUN_ROOT}/POSTHOC_RUN_STATUS.md" <<EOF
# Run Status: latentfm_fewshot_multi_calibration_posthoc_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/run_latentfm_fewshot_multi_calibration_posthoc_20260621.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation. Check at most every 30 minutes.

## Start time

$(cat "${RUN_ROOT}/POSTHOC_STARTED")

## Log path

\`${LOG_DIR}/posthoc.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_FEWSHOT_MULTI_CALIBRATION_SUMMARY_20260621.md\`
* \`${ROOT}/reports/latentfm_fewshot_multi_calibration_summary_20260621.json\`

## Current status

Waiting for launched training runs to finish.
EOF

trap 'rc=$?; echo "$rc" > "${RUN_ROOT}/POSTHOC_EXIT_CODE"; date "+%F %T %Z" > "${RUN_ROOT}/POSTHOC_FINISHED"; exit "$rc"' EXIT

{
  echo "[$(date '+%F %T %Z')] few-shot posthoc watcher start"
  if [[ ! -s "${MANIFEST}" ]]; then
    echo "Missing launch manifest: ${MANIFEST}" >&2
    exit 2
  fi
  while true; do
    mapfile -t runs < <("${PYTHON}" - "${MANIFEST}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in payload.get("launched_runs", []):
    print(row["run_name"])
PY
)
    all_done=1
    for run in "${runs[@]}"; do
      if [[ ! -f "${RUN_ROOT}/${run}.EXIT_CODE" ]]; then
        all_done=0
      fi
    done
    if [[ "${all_done}" == "1" ]]; then
      break
    fi
    echo "[$(date '+%F %T %Z')] training still running; next posthoc check in 1800s"
    sleep 1800
  done

  for run in "${runs[@]}"; do
    code="$(cat "${RUN_ROOT}/${run}.EXIT_CODE")"
    echo "[$(date '+%F %T %Z')] train ${run} exit=${code}"
    if [[ "${code}" != "0" ]]; then
      echo "training failed for ${run}; skip posthoc" >&2
      exit "${code}"
    fi
  done

  echo "[$(date '+%F %T %Z')] exact GPU status before few-shot posthoc"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  gpu_json="${LOG_DIR}/posthoc_gpu_selection_$(date +%Y%m%d_%H%M%S).json"
  "${PYTHON}" "${GPU_HELPER}" \
    --samples 3 \
    --interval-seconds 10 \
    --util-threshold-pct 10 \
    --memory-threshold-mib 4096 \
    --max-jobs-per-gpu 4 \
    --need 1 \
    --json-only \
    > "${gpu_json}" 2> "${LOG_DIR}/posthoc_gpu_selection.stderr"
  gpu="$("${PYTHON}" - "${gpu_json}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = payload.get("suggested_job_gpus") or []
print(chosen[0] if chosen else "")
PY
)"
  if [[ -z "${gpu}" ]]; then
    echo "No GPU selected for posthoc; see ${gpu_json}" >&2
    exit 3
  fi

  "${PYTHON}" - "${gpu_json}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
system = payload.get("system") or {}
mem = float(system.get("mem_available_gib") or 0.0)
load = float(system.get("load1_per_cpu") or 0.0)
audit = {
    "status": "pass",
    "min_mem_available_gib": 64.0,
    "max_load1_per_cpu": 2.0,
    "system": system,
}
reasons = []
if mem < 64.0:
    reasons.append(f"MemAvailable {mem:.1f} GiB < 64.0 GiB")
if load > 2.0:
    reasons.append(f"load1_per_cpu {load:.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

  source "${ROOT}/init-scdfm.sh" >/dev/null
  cd "${COUPLED}"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  export BLIS_NUM_THREADS=4
  export PYTHONPATH="${COUPLED}:${PYTHONPATH:-}"
  export PERT_EMBED_SOURCE=scgpt_embed_gene

  tmp_manifest="${RUN_ROOT}/posthoc_manifest.tmp.json"
  "${PYTHON}" - "${MANIFEST}" "${tmp_manifest}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in payload.get("launched_runs", []):
    row["baseline_split_json"] = ""
    row["baseline_family_json"] = ""
    row["run_split_json"] = ""
    row["run_family_json"] = ""
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  for idx in "${!runs[@]}"; do
    run="${runs[$idx]}"
    echo "[$(date '+%F %T %Z')] posthoc ${run}"
    split_file="$("${PYTHON}" - "${MANIFEST}" "${run}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
name = sys.argv[2]
for row in payload.get("launched_runs", []):
    if row["run_name"] == name:
        print(row["split_file"])
        break
PY
)"
    run_dir="${OUT_ROOT}/${run}"
    cand_posthoc="${run_dir}/posthoc_eval"
    base_posthoc="${RUN_ROOT}/baseline_posthoc/${run}"
    mkdir -p "${cand_posthoc}" "${base_posthoc}"
    test -s "${run_dir}/best.pt"
    test -s "${ANCHOR_DIR}/best.pt"

    base_split_json="${base_posthoc}/split_group_eval_anchor_same_split_ode20_mse1024_mmd1024_stablecaps.json"
    base_family_json="${base_posthoc}/condition_family_eval_anchor_same_split_ode20_mse1024_mmd1024_stablecaps.json"
    run_split_json="${cand_posthoc}/split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
    run_family_json="${cand_posthoc}/condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json"

    "${PYTHON}" -m model.latent.eval_split_groups \
      --checkpoint "${ANCHOR_DIR}/best.pt" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${BIFLOW_DIR}" \
      --split-file "${split_file}" \
      --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${base_split_json}" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024

    "${PYTHON}" -m model.latent.eval_condition_families \
      --checkpoint "${ANCHOR_DIR}/best.pt" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${BIFLOW_DIR}" \
      --split-file "${split_file}" \
      --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${base_family_json}" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024

    "${PYTHON}" -m model.latent.eval_split_groups \
      --checkpoint "${run_dir}/best.pt" \
      --data-dir "${DATA_DIR}" \
      --biflow-dir "${BIFLOW_DIR}" \
      --split-file "${split_file}" \
      --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${run_split_json}" \
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
      --biflow-dir "${BIFLOW_DIR}" \
      --split-file "${split_file}" \
      --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \
      --out "${run_family_json}" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 512 \
      --eval-max-conditions 256 \
      --eval-max-conditions-per-dataset 12 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024

    "${PYTHON}" - "${tmp_manifest}" "${run}" "${base_split_json}" "${base_family_json}" "${run_split_json}" "${run_family_json}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
name = sys.argv[2]
for row in payload.get("launched_runs", []):
    if row["run_name"] == name:
        row["baseline_split_json"] = sys.argv[3]
        row["baseline_family_json"] = sys.argv[4]
        row["run_split_json"] = sys.argv[5]
        row["run_family_json"] = sys.argv[6]
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  done

  mv "${tmp_manifest}" "${RUN_ROOT}/posthoc_manifest.json"
  "${PYTHON}" "${SUMMARY}" \
    --manifest "${RUN_ROOT}/posthoc_manifest.json" \
    --out-json "${ROOT}/reports/latentfm_fewshot_multi_calibration_summary_20260621.json" \
    --out-csv "${ROOT}/reports/latentfm_fewshot_multi_calibration_summary_20260621.csv" \
    --out-md "${ROOT}/reports/LATENTFM_FEWSHOT_MULTI_CALIBRATION_SUMMARY_20260621.md"
  "${PYTHON}" "${BOOTSTRAP_RUNNER}" \
    --manifest "${RUN_ROOT}/posthoc_manifest.json" \
    --out-dir "${ROOT}/reports/latentfm_fewshot_multi_calibration_bootstrap_20260621" \
    --n-boot 2000 \
    --seed 42
  echo "[$(date '+%F %T %Z')] few-shot posthoc done"
} 2>&1 | tee "${LOG_DIR}/posthoc.log"
