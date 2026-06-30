#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_XVERSE_SOFT_CANONICAL_ACK:-}" != "soft_exposure_internal_pass_frozen" ]]; then
  cat >&2 <<'EOF'
Refusing to launch soft-exposure canonical no-harm posthoc.

Set:
  LATENTFM_XVERSE_SOFT_CANONICAL_ACK=soft_exposure_internal_pass_frozen

Required preread:
  reports/LATENTFM_XVERSE_SOFT_EXPOSURE_SMOKES_DECISION_20260624.md
EOF
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_xverse_soft_exposure_canonical_noharm_20260624
TRAIN_OUT_ROOT=${COUPLED}/output/latentfm_runs/xverse_soft_exposure_smokes_20260624
LOG_ROOT=${ROOT}/logs/latentfm_xverse_soft_exposure_canonical_noharm_20260624
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
DECISION_JSON=${ROOT}/reports/latentfm_xverse_soft_exposure_smokes_decision_20260624.json
SUMMARIZER=${ROOT}/ops/summarize_latentfm_xverse_soft_exposure_canonical_noharm_20260624.py

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GPU_HELPER}" \
  "${DECISION_JSON}" \
  "${ROOT}/reports/LATENTFM_XVERSE_SOFT_EXPOSURE_SMOKES_DECISION_20260624.md" \
  "${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py" \
  "${ROOT}/ops/summarize_latentfm_single_background_candidate_decision_20260622.py" \
  "${SUMMARIZER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

mapfile -t RUN_NAMES < <("${PYTHON}" - "${DECISION_JSON}" <<'PY'
import json
import os
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
decision = obj.get("decision") or {}
if decision.get("status") != "soft_exposure_internal_pass":
    raise SystemExit(f"soft exposure internal decision not pass: {decision.get('status')!r}")
requested = os.environ.get("LATENTFM_XVERSE_SOFT_CANONICAL_ONLY_RUN", "").strip()
passed = set(decision.get("passed_runs") or [])
best = str(decision.get("best_run") or "")
if requested:
    if requested not in passed:
        raise SystemExit(f"requested run is not in passed_runs: {requested!r}")
    print(requested)
elif best:
    print(best)
else:
    raise SystemExit("missing best_run in soft exposure decision")
PY
)

need=${#RUN_NAMES[@]}
if (( need < 1 )); then
  echo "No soft-exposure canonical run selected" >&2
  exit 4
fi

for run_name in "${RUN_NAMES[@]}"; do
  ckpt=${TRAIN_OUT_ROOT}/${run_name}/best.pt
  if [[ ! -e "${ckpt}" ]]; then
    echo "Missing frozen checkpoint for ${run_name}: ${ckpt}" >&2
    exit 2
  fi
  run_dir=${RUN_ROOT}/${run_name}
  session=lfm_softcanon_${run_name}
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

echo "[$(date '+%F %T %Z')] exact GPU status before soft canonical no-harm launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 4 \
  --need "${need}" \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" "${need}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
need = int(sys.argv[3])
suggested = [int(x) for x in payload.get("suggested_job_gpus", [])]
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "need": need,
    "assigned_gpus": suggested[:need],
    "allowed_physical_user_gpus": payload.get("allowed_physical_user_gpus"),
    "active_user_gpus": payload.get("active_user_gpus"),
    "max_user_gpus": payload.get("max_user_gpus"),
    "max_jobs_per_gpu": payload.get("max_jobs_per_gpu"),
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if len(suggested) < need:
    reasons.append(f"only {len(suggested)} GPU job slots suggested for need={need}")
if int(payload.get("max_user_gpus") or 0) > 4:
    reasons.append("max_user_gpus exceeds current cap 4")
if int(payload.get("max_jobs_per_gpu") or 0) > 4:
    reasons.append("max_jobs_per_gpu exceeds per-GPU cap 4")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

mapfile -t ASSIGNED_GPUS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for gpu in payload["assigned_gpus"]:
    print(int(gpu))
PY
)

for i in "${!RUN_NAMES[@]}"; do
  run_name=${RUN_NAMES[$i]}
  run_dir=${RUN_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  gpu=${ASSIGNED_GPUS[$i]}
  session=lfm_softcanon_${run_name}
  ckpt=${TRAIN_OUT_ROOT}/${run_name}/best.pt
  script=${run_dir}/scripts/posthoc_${run_name}.sh

  cat > "${script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
eval_dir=${run_dir}/posthoc_eval_canonical
mkdir -p "\${eval_dir}"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${CANONICAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/split_group_eval_anchor_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/condition_family_eval_anchor_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ckpt} --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/split_group_eval_candidate_ode20_canonical.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ckpt} --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" "\${common[@]}"
${PYTHON} ${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py --candidate-split-json "\${eval_dir}/split_group_eval_candidate_ode20_canonical.json" --candidate-family-json "\${eval_dir}/condition_family_eval_candidate_ode20_canonical.json" --n-boot 2000 --seed 42 --out-json "\${eval_dir}/single_background_candidate_gate.json" --out-md "\${eval_dir}/SINGLE_BACKGROUND_CANDIDATE_GATE.md"
${PYTHON} ${ROOT}/ops/summarize_latentfm_single_background_candidate_decision_20260622.py --gate-json "\${eval_dir}/single_background_candidate_gate.json" --label ${run_name} --title "LatentFM xverse soft-exposure canonical no-harm decision" --out-md "\${eval_dir}/SINGLE_BACKGROUND_CANDIDATE_DECISION.md"
${PYTHON} ${SUMMARIZER}
EOF
  chmod +x "${script}"
  rm -f "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  date '+%F %T %Z' > "${run_dir}/POSTHOC_STARTED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${script} > ${log_dir}/posthoc.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$rc'"

  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: soft-exposure canonical no-harm ${run_name}

## Command

\`\`\`bash
LATENTFM_XVERSE_SOFT_CANONICAL_ACK=soft_exposure_internal_pass_frozen bash ${ROOT}/ops/launch_latentfm_xverse_soft_exposure_canonical_noharm_20260624.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation. Use 30-minute cadence for checks.

## Start time

$(cat "${run_dir}/POSTHOC_STARTED")

## PID / tmux / scheduler ID

tmux session: \`${session}\`

Physical GPU: ${gpu}

## Log path

\`${log_dir}/posthoc.log\`

## Expected outputs

* \`${run_dir}/posthoc_eval_canonical/single_background_candidate_gate.json\`
* \`${ROOT}/reports/LATENTFM_XVERSE_SOFT_EXPOSURE_CANONICAL_NOHARM_DECISION_20260624.md\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${log_dir}/posthoc.log
cat ${run_dir}/POSTHOC_EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Frozen checkpoint: \`${ckpt}\`
- Canonical split is post-freeze no-harm only; canonical multi groups are diagnostic only.
- Resource policy: current cap max 4 physical GPUs, max 4 LatentFM jobs/GPU, 48 CPU cores.
EOF
  echo "Launched soft canonical no-harm ${run_name} on GPU ${gpu}"
done
