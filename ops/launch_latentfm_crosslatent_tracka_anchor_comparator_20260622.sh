#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=latentfm_crosslatent_tracka_anchor_comparator_20260622
RUN_ROOT=${ROOT}/runs/${RUN_NAME}
LOG_ROOT=${RUN_ROOT}/logs
SESSION=${RUN_NAME}
BASELINE_RUN=${ROOT}/runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622
BASELINE_REPAIR_RUN=${ROOT}/runs/latentfm_crosslatent_stack_baseline_repair_20260622
BASELINE_SUMMARY_JSON=${ROOT}/reports/latentfm_crosslatent_tracka_trainonly_baselines_20260622.json
BASELINE_SUMMARY_MD=${ROOT}/reports/LATENTFM_CROSSLATENT_TRACKA_TRAINONLY_BASELINES_20260622.md
PROTOCOL=${ROOT}/reports/LATENTFM_CROSSLATENT_TRACKA_ANCHOR_COMPARATOR_PROTOCOL_20260622.md
SUMMARY=${ROOT}/ops/summarize_latentfm_crosslatent_tracka_anchor_internal_val_20260622.py
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
SPLIT_FILE=${ROOT}/dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CPU_THREADS=${LATENTFM_CPU_THREADS:-32}

LATENTS=(stack scfoundation scldm)

data_dir_for() {
  case "$1" in
    stack) echo "${ROOT}/dataset/latentfm_full/stack" ;;
    scfoundation) echo "${ROOT}/dataset/latentfm_full/scfoundation" ;;
    scldm) echo "${ROOT}/dataset/latentfm_full/scldm" ;;
    *) return 2 ;;
  esac
}

checkpoint_for() {
  case "$1" in
    stack) echo "${ROOT}/CoupledFM/output/latentfm_runs/full_stack/20260617_stack_comp006_delta_w5_12k/best.pt" ;;
    scfoundation) echo "${ROOT}/CoupledFM/output/latentfm_runs/full_scfoundation/20260617_scfoundation_comp006_delta_w5_12k/best.pt" ;;
    scldm) echo "${ROOT}/CoupledFM/output/latentfm_runs/full_scldm/20260617_scldm_comp006_delta_w5_12k/best.pt" ;;
    *) return 2 ;;
  esac
}

pert_means_for() {
  echo "${BASELINE_RUN}/artifacts/${1}_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
}

baseline_json_for() {
  echo "${ROOT}/reports/latentfm_crosslatent_${1}_gene_reliability_router_gate_20260622.json"
}

anchor_eval_for() {
  echo "${RUN_ROOT}/${1}/anchor_internal_val_split_eval.json"
}

summary_json_for() {
  echo "${ROOT}/reports/latentfm_crosslatent_${1}_tracka_anchor_internal_val_20260622.json"
}

summary_md_for() {
  local upper
  upper=$(echo "$1" | tr '[:lower:]' '[:upper:]')
  echo "${ROOT}/reports/LATENTFM_CROSSLATENT_${upper}_TRACKA_ANCHOR_INTERNAL_VAL_20260622.md"
}

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in "${PYTHON}" "${SUMMARY}" "${GPU_HELPER}" "${PROTOCOL}" "${SPLIT_FILE}" "${BASELINE_RUN}/RUN_STATUS.md"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ ! -f "${BASELINE_SUMMARY_MD}" || ! -f "${BASELINE_SUMMARY_JSON}" ]]; then
  echo "Missing cross-latent baseline summary report; refusing launch" >&2
  exit 4
fi

baseline_ready="$("${PYTHON}" - "${BASELINE_RUN}" "${BASELINE_REPAIR_RUN}" "${BASELINE_SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

baseline_run = Path(sys.argv[1])
repair_run = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
payload = json.loads(summary_json.read_text(encoding="utf-8"))
status = payload.get("status")
original_ok = (baseline_run / "EXIT_CODE").read_text(encoding="utf-8").strip() == "0" if (baseline_run / "EXIT_CODE").is_file() else False
repair_ok = (repair_run / "EXIT_CODE").read_text(encoding="utf-8").strip() == "0" if (repair_run / "EXIT_CODE").is_file() else False
rows = payload.get("latents") or []
all_rows_ok = bool(rows) and all(
    row.get("pert_mean_status") == "ok"
    and (row.get("baseline_gate") or {}).get("returncode") == 0
    for row in rows
)
if status == "crosslatent_trainonly_baselines_ready_for_protocol_review" and all_rows_ok and (original_ok or repair_ok):
    print("ready")
else:
    print(json.dumps({
        "status": status,
        "original_ok": original_ok,
        "repair_ok": repair_ok,
        "all_rows_ok": all_rows_ok,
    }, sort_keys=True))
PY
)"
if [[ "${baseline_ready}" != "ready" ]]; then
  echo "Cross-latent baseline prerequisites are not ready: ${baseline_ready}" >&2
  exit 3
fi

for latent in "${LATENTS[@]}"; do
  for required in \
    "$(data_dir_for "${latent}")/manifest.json" \
    "$(checkpoint_for "${latent}")" \
    "$(pert_means_for "${latent}")" \
    "$(baseline_json_for "${latent}")"; do
    if [[ ! -e "${required}" ]]; then
      echo "Missing ${latent} prerequisite: ${required}" >&2
      exit 5
    fi
  done
  if [[ -e "$(anchor_eval_for "${latent}")" || -e "$(summary_json_for "${latent}")" || -e "$(summary_md_for "${latent}")" ]]; then
    echo "Refusing to overwrite existing comparator outputs for ${latent}" >&2
    exit 6
  fi
  mkdir -p "${RUN_ROOT}/${latent}"
done

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 7
fi

echo "[$(date '+%F %T %Z')] exact GPU status before cross-latent comparator" | tee "${LOG_ROOT}/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${LOG_ROOT}/gpu_launch_audit.log"

gpu_json="${LOG_ROOT}/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-jobs-per-gpu 1 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${LOG_ROOT}/gpu_selection.stderr"

assignment_json="${LOG_ROOT}/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
stable_empty = [
    g for g in gpus.values()
    if g.get("stable_light")
    and int(g.get("foreign_process_count", 0)) == 0
    and int(g.get("own_process_count", 0)) == 0
]
stable_count = len(stable_empty)
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
physical_budget = min(4, stable_count) if stable_count >= 5 else max(0, min(4, stable_count - 1))
chosen = []
for idx in [int(x) for x in payload.get("candidate_order", [])]:
    gpu = gpus[idx]
    if not gpu.get("available"):
        continue
    if not gpu.get("stable_light"):
        continue
    if int(gpu.get("foreign_process_count", 0)) > 0:
        continue
    if int(gpu.get("own_process_count", 0)) > 0:
        continue
    proposed = active_user | set(chosen) | {idx}
    if len(proposed) <= physical_budget and int(gpu.get("colocation_slots_free", 0)) > 0:
        chosen.append(idx)
    if len(chosen) >= 3:
        break
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "stable_empty_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "active_user_gpus": sorted(active_user),
    "chosen_gpus": chosen,
    "parallel": len(chosen) >= 3,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if not chosen:
    reasons.append("no GPU slot available under AGENTS policy")
if float(system.get("mem_available_gib") or 0.0) < 64.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 64.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 8)
PY

GPU_LIST="$("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(",".join(str(x) for x in payload["chosen_gpus"]))
PY
)"

PARALLEL="$("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
print("1" if json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("parallel") else "0")
PY
)"

run_script="${RUN_ROOT}/run_crosslatent_anchor_comparator.sh"
cat > "${run_script}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

run_latent() {
  local latent="$1"
  local gpu="$2"
  local data_dir="$3"
  local checkpoint="$4"
  local pert_means="$5"
  local baseline_json="$6"
  local anchor_eval="$7"
  local out_json="$8"
  local out_md="$9"
  local log_path="${LOG_ROOT}/${latent}.log"
  {
    echo "[$(date '+%F %T %Z')] starting ${latent} on physical GPU ${gpu}"
    cd "${COUPLED}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m model.latent.eval_split_groups \
      --checkpoint "${checkpoint}" \
      --data-dir "${data_dir}" \
      --biflow-dir "${BIFLOW_DIR}" \
      --split-file "${SPLIT_FILE}" \
      --groups internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy \
      --pert-means-file "${pert_means}" \
      --out "${anchor_eval}" \
      --gpu 0 \
      --ode-steps 20 \
      --max-chunk 256 \
      --eval-max-conditions 0 \
      --eval-max-conditions-per-dataset 0 \
      --eval-max-mse-cells 1024 \
      --eval-max-mmd-cells 1024
    "${PYTHON}" "${SUMMARY}" \
      --latent "${latent}" \
      --anchor-eval-json "${anchor_eval}" \
      --baseline-json "${baseline_json}" \
      --out-json "${out_json}" \
      --out-md "${out_md}" \
      --n-boot 2000 \
      --seed 42
    "${PYTHON}" - "${anchor_eval}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
means = payload.get("means_files") or {}
if not means.get("pert_means_override"):
    raise SystemExit("missing explicit pert_means override")
print(json.dumps(means, indent=2))
PY
    echo "[$(date '+%F %T %Z')] finished ${latent}"
  } > "${log_path}" 2>&1
}
EOF

for latent in "${LATENTS[@]}"; do
  cat >> "${run_script}" <<EOF
DATA_DIR_${latent}="$(data_dir_for "${latent}")"
CHECKPOINT_${latent}="$(checkpoint_for "${latent}")"
PERT_${latent}="$(pert_means_for "${latent}")"
BASELINE_${latent}="$(baseline_json_for "${latent}")"
ANCHOR_${latent}="$(anchor_eval_for "${latent}")"
SUMMARY_JSON_${latent}="$(summary_json_for "${latent}")"
SUMMARY_MD_${latent}="$(summary_md_for "${latent}")"
EOF
done

cat >> "${run_script}" <<EOF
export ROOT="${ROOT}"
export COUPLED="${COUPLED}"
export PYTHON="${PYTHON}"
export BIFLOW_DIR="${BIFLOW_DIR}"
export SPLIT_FILE="${SPLIT_FILE}"
export SUMMARY="${SUMMARY}"
export LOG_ROOT="${LOG_ROOT}"
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export BLIS_NUM_THREADS="${CPU_THREADS}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${PARALLEL}" == "1" ]]; then
  run_latent stack "\${GPUS[0]}" "\${DATA_DIR_stack}" "\${CHECKPOINT_stack}" "\${PERT_stack}" "\${BASELINE_stack}" "\${ANCHOR_stack}" "\${SUMMARY_JSON_stack}" "\${SUMMARY_MD_stack}" &
  p1=\$!
  run_latent scfoundation "\${GPUS[1]}" "\${DATA_DIR_scfoundation}" "\${CHECKPOINT_scfoundation}" "\${PERT_scfoundation}" "\${BASELINE_scfoundation}" "\${ANCHOR_scfoundation}" "\${SUMMARY_JSON_scfoundation}" "\${SUMMARY_MD_scfoundation}" &
  p2=\$!
  run_latent scldm "\${GPUS[2]}" "\${DATA_DIR_scldm}" "\${CHECKPOINT_scldm}" "\${PERT_scldm}" "\${BASELINE_scldm}" "\${ANCHOR_scldm}" "\${SUMMARY_JSON_scldm}" "\${SUMMARY_MD_scldm}" &
  p3=\$!
  rc=0
  wait "\${p1}" || rc=\$?
  wait "\${p2}" || rc=\$?
  wait "\${p3}" || rc=\$?
  exit "\${rc}"
else
  gpu="\${GPUS[0]}"
  run_latent stack "\${gpu}" "\${DATA_DIR_stack}" "\${CHECKPOINT_stack}" "\${PERT_stack}" "\${BASELINE_stack}" "\${ANCHOR_stack}" "\${SUMMARY_JSON_stack}" "\${SUMMARY_MD_stack}"
  run_latent scfoundation "\${gpu}" "\${DATA_DIR_scfoundation}" "\${CHECKPOINT_scfoundation}" "\${PERT_scfoundation}" "\${BASELINE_scfoundation}" "\${ANCHOR_scfoundation}" "\${SUMMARY_JSON_scfoundation}" "\${SUMMARY_MD_scfoundation}"
  run_latent scldm "\${gpu}" "\${DATA_DIR_scldm}" "\${CHECKPOINT_scldm}" "\${PERT_scldm}" "\${BASELINE_scldm}" "\${ANCHOR_scldm}" "\${SUMMARY_JSON_scldm}" "\${SUMMARY_MD_scldm}"
fi
EOF
chmod +x "${run_script}"

date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
echo "${SESSION}" > "${RUN_ROOT}/SESSION_NAME"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_crosslatent_tracka_anchor_comparator_20260622.sh
\`\`\`

## Runtime classification

Long GPU posthoc comparator audit.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`

Chosen physical GPUs: \`${GPU_LIST}\`

Parallel mode: \`${PARALLEL}\`

## Log path

\`${LOG_ROOT}/<latent>.log\`

## Expected outputs

* \`${RUN_ROOT}/stack/anchor_internal_val_split_eval.json\`
* \`${RUN_ROOT}/scfoundation/anchor_internal_val_split_eval.json\`
* \`${RUN_ROOT}/scldm/anchor_internal_val_split_eval.json\`
* \`${ROOT}/reports/latentfm_crosslatent_stack_tracka_anchor_internal_val_20260622.json\`
* \`${ROOT}/reports/latentfm_crosslatent_scfoundation_tracka_anchor_internal_val_20260622.json\`
* \`${ROOT}/reports/latentfm_crosslatent_scldm_tracka_anchor_internal_val_20260622.json\`
* \`${RUN_ROOT}/EXIT_CODE\`
* \`${RUN_ROOT}/FINISHED\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_ROOT}/stack.log
tail -n 50 ${LOG_ROOT}/scfoundation.log
tail -n 50 ${LOG_ROOT}/scldm.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Hypothesis: test whether Track A anchor weakness is xverse-specific or shared
across full latent spaces by evaluating frozen stack/scfoundation/scldm
checkpoints on the same train-only internal-val groups. This is posthoc audit
only, not training. It uses latent-specific train-only pert means via explicit
\`--pert-means-file\` and latent-specific baseline JSONs. It does not use
canonical test, canonical multi, or Track C query for selection.

Promotion rule: passing an anchor gate only nominates a latent/checkpoint for
cross-latent interpretation and mechanism review; it does not authorize
training.
EOF

tmux new -d -s "${SESSION}" \
  "bash -lc 'source ${ROOT}/init-scdfm.sh >/dev/null 2>&1 || true; bash ${run_script}; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/FINISHED; exit \$rc'"

tmux ls | tee "${LOG_ROOT}/tmux_ls_after_launch.txt"
sleep 2
for latent in "${LATENTS[@]}"; do
  echo "--- ${latent}"
  tail -n 10 "${LOG_ROOT}/${latent}.log" 2>/dev/null || true
done
