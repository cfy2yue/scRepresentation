#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_XVERSE_SCALING_CANONICAL_ACK:-}" != "internal_count_gate_pass_frozen" ]]; then
  cat >&2 <<'EOF'
Refusing to launch canonical no-harm posthoc.

Set:
  LATENTFM_XVERSE_SCALING_CANONICAL_ACK=internal_count_gate_pass_frozen

Required preread:
  reports/LATENTFM_XVERSE_SCALING_COUNT_SMOKES_DECISION_20260624.md
EOF
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_xverse_scaling_canonical_noharm_20260624
TRAIN_OUT_ROOT=${COUPLED}/output/latentfm_runs/xverse_scaling_count_smokes_20260624
LOG_ROOT=${ROOT}/logs/latentfm_xverse_scaling_canonical_noharm_20260624
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GPU_HELPER=${ROOT}/ops/select_available_gpus.py

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}" "${ROOT}/reports"

declare -a RUN_NAMES=(
  "xverse_scaling_cap120_all_3k_seed42"
  "xverse_scaling_gene_cap120_allbg_3k_seed42"
)
declare -a HYPOTHESES=(
  "Frozen count-scaling winner cap120_all should pass canonical Track A no-harm before any promotion."
  "Frozen exploratory gene-only arm is checked as a parallel canonical diagnostic, not as training-time selection."
)

if [[ "${LATENTFM_XVERSE_SCALING_CANONICAL_ONLY_RUN:-}" == "xverse_scaling_gene_cap120_k562bg_3k_seed42" ]]; then
  RUN_NAMES=("xverse_scaling_gene_cap120_k562bg_3k_seed42")
  HYPOTHESES=("Frozen exploratory K562-like background arm is checked as a canonical diagnostic only; do not use it as formal background-scaling evidence unless upstream count and gene-only gates remain acceptable.")
elif [[ "${LATENTFM_XVERSE_SCALING_CANONICAL_ONLY_RUN:-}" == "xverse_scaling_full_trainonly_3k_seed42" ]]; then
  COUNT_JSON=${ROOT}/reports/latentfm_xverse_scaling_count_smokes_decision_20260624.json
  if [[ ! -e "${COUNT_JSON}" ]]; then
    echo "Missing count/full extension decision JSON: ${COUNT_JSON}" >&2
    exit 2
  fi
  "${PYTHON}" - "${COUNT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = (obj.get("full_extension_decision") or {}).get("status")
if status != "full_trainonly_extension_pass":
    raise SystemExit(f"full_trainonly extension gate not passed: {status!r}")
PY
  RUN_NAMES=("xverse_scaling_full_trainonly_3k_seed42")
  HYPOTHESES=("Frozen full train-only extension is checked as a separate canonical no-harm diagnostic after full_extension_decision pass; it must not replace the primary cap120_all candidate without a separate decision.")
elif [[ "${LATENTFM_XVERSE_SCALING_CANONICAL_ONLY_RUN:-}" == "xverse_scaling_type_balanced_cap120_3k_seed42" ]]; then
  COUNT_JSON=${ROOT}/reports/latentfm_xverse_scaling_count_smokes_decision_20260624.json
  if [[ ! -e "${COUNT_JSON}" ]]; then
    echo "Missing type-balance extension decision JSON: ${COUNT_JSON}" >&2
    exit 2
  fi
  "${PYTHON}" - "${COUNT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = (obj.get("type_balance_extension_decision") or {}).get("status")
if status != "type_balanced_extension_pass":
    raise SystemExit(f"type_balanced extension gate not passed: {status!r}")
PY
  RUN_NAMES=("xverse_scaling_type_balanced_cap120_3k_seed42")
  HYPOTHESES=("Frozen type-balanced cap120 extension is checked as a separate canonical no-harm diagnostic only after type_balance_extension_decision pass; canonical metrics must not be used to select or tune the arm.")
elif [[ "${LATENTFM_XVERSE_SCALING_CANONICAL_ONLY_RUN:-}" == "xverse_scaling_jiang_exposure_capped_3k_seed42" ]]; then
  COUNT_JSON=${ROOT}/reports/latentfm_xverse_scaling_count_smokes_decision_20260624.json
  JIANG_RUN_DIR=${ROOT}/runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_jiang_exposure_capped_3k_seed42
  if [[ ! -e "${JIANG_RUN_DIR}/POSTHOC_EXIT_CODE" || "$(cat "${JIANG_RUN_DIR}/POSTHOC_EXIT_CODE")" != "0" ]]; then
    echo "Jiang exposure-capped posthoc is not complete with exit 0; refusing canonical no-harm launch." >&2
    exit 2
  fi
  if [[ ! -e "${COUNT_JSON}" ]]; then
    echo "Missing Jiang extension decision JSON: ${COUNT_JSON}" >&2
    exit 2
  fi
  "${PYTHON}" - "${COUNT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = (obj.get("jiang_exposure_extension_decision") or {}).get("status")
if status != "jiang_exposure_extension_pass":
    raise SystemExit(f"Jiang exposure extension gate not passed: {status!r}")
PY
  RUN_NAMES=("xverse_scaling_jiang_exposure_capped_3k_seed42")
  HYPOTHESES=("Frozen Jiang exposure-capped extension is checked as a separate canonical no-harm diagnostic only after jiang_exposure_extension_decision pass; canonical metrics must not be used to select or tune the arm.")
elif [[ "${LATENTFM_XVERSE_SCALING_CANONICAL_ONLY_RUN:-}" == "xverse_scaling_general_exposure_cap_v2_3k_seed42" ]]; then
  COUNT_JSON=${ROOT}/reports/latentfm_xverse_scaling_count_smokes_decision_20260624.json
  GENERAL_RUN_DIR=${ROOT}/runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_general_exposure_cap_v2_3k_seed42
  if [[ ! -e "${GENERAL_RUN_DIR}/POSTHOC_EXIT_CODE" || "$(cat "${GENERAL_RUN_DIR}/POSTHOC_EXIT_CODE")" != "0" ]]; then
    echo "general exposure-cap v2 posthoc is not complete with exit 0; refusing canonical no-harm launch." >&2
    exit 2
  fi
  if [[ ! -e "${COUNT_JSON}" ]]; then
    echo "Missing general exposure extension decision JSON: ${COUNT_JSON}" >&2
    exit 2
  fi
  "${PYTHON}" - "${COUNT_JSON}" <<'PY'
import json
import sys
from pathlib import Path

obj = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = (obj.get("general_exposure_extension_decision") or {}).get("status")
if status != "general_exposure_extension_pass":
    raise SystemExit(f"general exposure extension gate not passed: {status!r}")
PY
  RUN_NAMES=("xverse_scaling_general_exposure_cap_v2_3k_seed42")
  HYPOTHESES=("Frozen general exposure-cap v2 extension is checked as a separate canonical no-harm diagnostic only after general_exposure_extension_decision pass; canonical metrics must not be used to select or tune the arm.")
elif [[ -n "${LATENTFM_XVERSE_SCALING_CANONICAL_ONLY_RUN:-}" ]]; then
  echo "Unsupported LATENTFM_XVERSE_SCALING_CANONICAL_ONLY_RUN=${LATENTFM_XVERSE_SCALING_CANONICAL_ONLY_RUN}" >&2
  exit 4
fi

need=${#RUN_NAMES[@]}

for required in \
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GPU_HELPER}" \
  "${ROOT}/reports/LATENTFM_XVERSE_SCALING_COUNT_SMOKES_DECISION_20260624.md" \
  "${ROOT}/ops/audit_latentfm_xverse_single_background_candidate_20260622.py" \
  "${ROOT}/ops/summarize_latentfm_single_background_candidate_decision_20260622.py" \
  "${ROOT}/ops/summarize_latentfm_xverse_scaling_canonical_noharm_20260624.py"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

for run_name in "${RUN_NAMES[@]}"; do
  ckpt=${TRAIN_OUT_ROOT}/${run_name}/best.pt
  if [[ ! -e "${ckpt}" ]]; then
    echo "Missing frozen checkpoint for ${run_name}: ${ckpt}" >&2
    exit 2
  fi
  run_dir=${RUN_ROOT}/${run_name}
  session=lfm_scaling_canon_${run_name}
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${LOG_ROOT}/${run_name}"
done

echo "[$(date '+%F %T %Z')] exact GPU status before scaling canonical no-harm launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
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
    "new_physical_slots": payload.get("new_physical_slots"),
    "max_user_gpus": payload.get("max_user_gpus"),
    "max_jobs_per_gpu": payload.get("max_jobs_per_gpu"),
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if len(suggested) < need:
    reasons.append(f"only {len(suggested)} GPU job slots suggested for need={need}")
if int(payload.get("max_user_gpus") or 0) > 4:
    reasons.append("max_user_gpus exceeds user cap 4")
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
  session=lfm_scaling_canon_${run_name}
  ckpt=${TRAIN_OUT_ROOT}/${run_name}/best.pt
  script=${run_dir}/scripts/posthoc_${run_name}.sh

  cat > "${script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export BLIS_NUM_THREADS=3
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
${PYTHON} ${ROOT}/ops/summarize_latentfm_single_background_candidate_decision_20260622.py --gate-json "\${eval_dir}/single_background_candidate_gate.json" --label ${run_name} --title "LatentFM xverse scaling canonical no-harm decision" --out-md "\${eval_dir}/SINGLE_BACKGROUND_CANDIDATE_DECISION.md"
EOF
  chmod +x "${script}"
  rm -f "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  date '+%F %T %Z' > "${run_dir}/POSTHOC_STARTED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${script} > ${log_dir}/posthoc.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$rc'"

  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: scaling canonical no-harm ${run_name}

## Hypothesis

${HYPOTHESES[$i]}

## Command

\`\`\`bash
LATENTFM_XVERSE_SCALING_CANONICAL_ACK=internal_count_gate_pass_frozen bash ${ROOT}/ops/launch_latentfm_xverse_scaling_canonical_noharm_20260624.sh
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
* \`${run_dir}/posthoc_eval_canonical/SINGLE_BACKGROUND_CANDIDATE_DECISION.md\`
* \`${ROOT}/reports/LATENTFM_XVERSE_SCALING_CANONICAL_NOHARM_DECISION_20260624.md\`

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
- Canonical split is post-freeze no-harm only; it was not used for training selection.
- Canonical multi groups are diagnostic only.
- Resource policy: max 4 physical GPUs, max 4 LatentFM jobs/GPU, 48 CPU cores.
EOF
  echo "Launched canonical no-harm ${run_name} on GPU ${gpu}"
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_scaling_canonical_noharm_20260624

## Command

\`\`\`bash
LATENTFM_XVERSE_SCALING_CANONICAL_ACK=internal_count_gate_pass_frozen bash ${ROOT}/ops/launch_latentfm_xverse_scaling_canonical_noharm_20260624.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation batch. Each child run has its own RUN_STATUS.md.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

$(printf '* `%s`\n' "${RUN_NAMES[@]/#/lfm_scaling_canon_}")

## Log path

\`${LOG_ROOT}/<run_name>/posthoc.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_XVERSE_SCALING_CANONICAL_NOHARM_DECISION_20260624.md\`

## Current status

Started ${need} canonical no-harm posthoc jobs.

## Notes

- Run summarizer after posthoc jobs finish:
  \`${PYTHON} ${ROOT}/ops/summarize_latentfm_xverse_scaling_canonical_noharm_20260624.py\`
EOF
