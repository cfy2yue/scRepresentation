#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_xverse_uncapped_posthoc_20260621
LOG_ROOT=${ROOT}/logs/latentfm_xverse_uncapped_posthoc_20260621
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
SPLIT_FILE=${ROOT}/dataset/biFlow_data/split_seed42.json
RUN_DIR=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
BOOTSTRAP=${ROOT}/ops/bootstrap_latentfm_single_posthoc_ci_20260621.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

SESSION=latentfm_xverse_uncapped_posthoc_20260621
OUT_DIR=${RUN_DIR}/posthoc_eval_uncapped_20260621

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}" "${OUT_DIR}" "${ROOT}/reports"

for required in \
  "${RUN_DIR}/best.pt" \
  "${DATA_DIR}/manifest.json" \
  "${SPLIT_FILE}" \
  "${GPU_HELPER}" \
  "${BOOTSTRAP}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

echo "[$(date '+%F %T %Z')] exact GPU status before xverse uncapped posthoc" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-jobs-per-gpu 4 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
stable = [g for g in gpus.values() if g.get("stable_light")]
stable_count = len(stable)
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
physical_budget = min(4, stable_count) if stable_count >= 5 else max(0, min(4, stable_count - 1))
chosen = None
for idx in [int(x) for x in payload.get("candidate_order", [])]:
    gpu = gpus[idx]
    if not gpu.get("available"):
        continue
    if len(active_user | {idx}) <= physical_budget and int(gpu.get("colocation_slots_free", 0)) > 0:
        chosen = idx
        break
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "stable_light_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "active_user_gpus": sorted(active_user),
    "chosen_gpu": chosen,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if chosen is None:
    reasons.append("no GPU slot available under leave-one-empty and max-4-physical rules")
if float(system.get("mem_available_gib") or 0.0) < 64.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 64.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

GPU="$("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["chosen_gpu"])
PY
)"

run_script="${RUN_ROOT}/run_xverse_uncapped_posthoc.sh"
cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene

${PYTHON} -m model.latent.eval_split_groups \\
  --checkpoint ${RUN_DIR}/best.pt \\
  --data-dir ${DATA_DIR} \\
  --biflow-dir ${BIFLOW_DIR} \\
  --split-file ${SPLIT_FILE} \\
  --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \\
  --out ${OUT_DIR}/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json \\
  --gpu 0 \\
  --ode-steps 20 \\
  --max-chunk 512 \\
  --eval-max-conditions 0 \\
  --eval-max-conditions-per-dataset 0 \\
  --eval-max-mse-cells 2048 \\
  --eval-max-mmd-cells 2048

${PYTHON} -m model.latent.eval_condition_families \\
  --checkpoint ${RUN_DIR}/best.pt \\
  --data-dir ${DATA_DIR} \\
  --biflow-dir ${BIFLOW_DIR} \\
  --split-file ${SPLIT_FILE} \\
  --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 \\
  --out ${OUT_DIR}/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json \\
  --gpu 0 \\
  --ode-steps 20 \\
  --max-chunk 512 \\
  --eval-max-conditions 0 \\
  --eval-max-conditions-per-dataset 0 \\
  --eval-max-mse-cells 2048 \\
  --eval-max-mmd-cells 2048

${PYTHON} ${BOOTSTRAP} \\
  --eval-json ${OUT_DIR}/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json \\
  --groups test test_multi_unseen2 \\
  --n-boot 2000 \\
  --seed 42 \\
  --title "LatentFM xverse 8k Condition-Uncapped Split CI" \\
  --out-json ${ROOT}/reports/latentfm_xverse_8k_condition_uncapped_split_ci_20260621.json \\
  --out-md ${ROOT}/reports/LATENTFM_XVERSE_8K_CONDITION_UNCAPPED_SPLIT_CI_20260621.md

${PYTHON} ${BOOTSTRAP} \\
  --eval-json ${OUT_DIR}/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json \\
  --groups test_all family_gene family_drug structure_multi \\
  --n-boot 2000 \\
  --seed 42 \\
  --title "LatentFM xverse 8k Condition-Uncapped Family CI" \\
  --out-json ${ROOT}/reports/latentfm_xverse_8k_condition_uncapped_family_ci_20260621.json \\
  --out-md ${ROOT}/reports/LATENTFM_XVERSE_8K_CONDITION_UNCAPPED_FAMILY_CI_20260621.md
EOF
chmod +x "${run_script}"

rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
tmux new -d -s "${SESSION}" \
  "bash -lc 'bash ${run_script} > ${LOG_ROOT}/xverse_uncapped_posthoc.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/FINISHED; exit \$rc'"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_uncapped_posthoc_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_xverse_uncapped_posthoc_20260621.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation. Use 30-minute cadence for checks.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## tmux / GPU

* \`${SESSION}\`, physical GPU${GPU}

## Log path

\`${LOG_ROOT}/xverse_uncapped_posthoc.log\`

## Expected outputs

* \`${OUT_DIR}/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json\`
* \`${OUT_DIR}/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json\`
* \`${ROOT}/reports/LATENTFM_XVERSE_8K_CONDITION_UNCAPPED_SPLIT_CI_20260621.md\`
* \`${ROOT}/reports/LATENTFM_XVERSE_8K_CONDITION_UNCAPPED_FAMILY_CI_20260621.md\`

## How to check manually

\`\`\`bash
tmux ls | grep ${SESSION} || true
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
tail -n 50 ${LOG_ROOT}/xverse_uncapped_posthoc.log
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

This is condition-uncapped (`eval_max_conditions=0`,
`eval_max_conditions_per_dataset=0`) but cell-capped at 2048 for MSE/MMD,
matching the xverse 8k train-time full-eval cell caps.
EOF

echo "RUN_STATUS: ${RUN_ROOT}/RUN_STATUS.md"
