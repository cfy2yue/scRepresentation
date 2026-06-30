#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_STATIC_BEST_LATEST_ACK:-}" != "posthoc_internal_only" ]]; then
  echo "Set LATENTFM_STATIC_BEST_LATEST_ACK=posthoc_internal_only" >&2
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_static_best_latest_update_audit_20260624
LOG_ROOT=${ROOT}/logs/latentfm_static_best_latest_update_audit_20260624
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
SUMMARIZER=${ROOT}/ops/summarize_latentfm_static_best_latest_update_audit_20260624.py

mkdir -p "${RUN_ROOT}/logs" "${LOG_ROOT}" "${ROOT}/reports"

for required in "${DATA_DIR}/manifest.json" "${GPU_HELPER}" "${SUMMARIZER}"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

need=1
echo "[$(date '+%F %T %Z')] exact GPU status before static best-vs-latest audit launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" --samples 3 --interval-seconds 10 --util-threshold-pct 10 --memory-threshold-mib 4096 --max-user-gpus 4 --max-jobs-per-gpu 4 --need "${need}" --json-only > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"
assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" "${need}" <<'PY'
import json, sys
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
need=int(sys.argv[3])
suggested=[int(x) for x in payload.get("suggested_job_gpus", [])]
system=payload.get("system") or {}
audit={"status":"pass","need":need,"assigned_gpus":suggested[:need],"active_user_gpus":payload.get("active_user_gpus"),"allowed_physical_user_gpus":payload.get("allowed_physical_user_gpus"),"system":system,"gpu_selection_json":str(sys.argv[1])}
reasons=[]
if len(suggested)<need:
    reasons.append(f"only {len(suggested)} GPU slots suggested for need={need}")
if float(system.get("mem_available_gib") or 0)<128:
    reasons.append("low_mem")
if float(system.get("load1_per_cpu") or 0)>2:
    reasons.append("high_cpu_load")
if reasons:
    audit["status"]="fail"
    audit["reasons"]=reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
raise SystemExit(0 if audit["status"]=="pass" else 4)
PY
GPU=$("${PYTHON}" - "${assignment_json}" <<'PY'
import json, sys
from pathlib import Path
print(int(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["assigned_gpus"][0]))
PY
)

declare -a SPECS=(
  "cap60_resp010_seed42|/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/scaling_cap60_response_repair_20260624/xverse_scaling_cap60_resp010_replay05_4k_seed42/latest.pt|/data/cyx/1030/dataset/biFlow_data/xverse_scaling_protocol_splits_20260624/split_seed42_xverse_scaling_protocol_cap60_primary19.json|/data/cyx/1030/scLatent/runs/latentfm_scaling_protocol_splits_20260624/artifacts/cap60_primary19_trainonly_pert_means.npz"
  "cap60_resp025_seed42|/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/scaling_cap60_response_repair_20260624/xverse_scaling_cap60_resp025_replay05_4k_seed42/latest.pt|/data/cyx/1030/dataset/biFlow_data/xverse_scaling_protocol_splits_20260624/split_seed42_xverse_scaling_protocol_cap60_primary19.json|/data/cyx/1030/scLatent/runs/latentfm_scaling_protocol_splits_20260624/artifacts/cap60_primary19_trainonly_pert_means.npz"
  "general_exposure_mmdguard|/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/general_exposure_mmdguard_repair_20260624/xverse_general_exposure_mmdguard_replay05_mmd05_3k_seed42/latest.pt|/data/cyx/1030/dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json|/data/cyx/1030/scLatent/runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/xverse_trainonly_scaling_general_exposure_cap_v2_pert_means.npz"
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r name ckpt split_file pert_means <<< "${spec}"
  [[ -e "${ckpt}" ]] || { echo "Missing checkpoint ${ckpt}" >&2; exit 2; }
  [[ -e "${split_file}" ]] || { echo "Missing split ${split_file}" >&2; exit 2; }
  [[ -e "${pert_means}" ]] || { echo "Missing pert means ${pert_means}" >&2; exit 2; }
  mkdir -p "${RUN_ROOT}/${name}/posthoc_eval_internal_latest" "${LOG_ROOT}/${name}"
done

driver=${RUN_ROOT}/scripts/run_static_best_latest_posthoc.sh
mkdir -p "${RUN_ROOT}/scripts"
cat > "${driver}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
EOF
for spec in "${SPECS[@]}"; do
  IFS='|' read -r name ckpt split_file pert_means <<< "${spec}"
  cat >> "${driver}" <<EOF
echo "[\$(date '+%F %T %Z')] evaluating latest ${name}"
eval_dir=${RUN_ROOT}/${name}/posthoc_eval_internal_latest
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${split_file} --pert-means-file ${pert_means} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ckpt} --groups test test_single internal_val_cross_background_seen_gene_proxy internal_val_family_gene_proxy --out "\${eval_dir}/split_group_eval_latest_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ckpt} --groups test_all family_gene family_drug test_single --out "\${eval_dir}/condition_family_eval_latest_internal_ode20.json" "\${common[@]}"
echo 0 > ${RUN_ROOT}/${name}/POSTHOC_EXIT_CODE
date '+%F %T %Z' > ${RUN_ROOT}/${name}/POSTHOC_FINISHED
EOF
done
cat >> "${driver}" <<EOF
${PYTHON} ${SUMMARIZER}
EOF
chmod +x "${driver}"

session=lfm_static_best_latest_update_audit_20260624
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session already exists: ${session}" >&2
  exit 3
fi
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"
rm -f "${RUN_ROOT}/EXIT_CODE" "${RUN_ROOT}/FINISHED"
tmux new -d -s "${session}" "bash -lc 'bash ${driver} > ${LOG_ROOT}/posthoc.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/FINISHED; exit \$rc'"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_static_best_latest_update_audit_20260624

## Hypothesis

Recent repair branches may have failed because the selected \`best.pt\` over-updated or selected a drifted checkpoint; evaluating \`latest.pt\` on the same train-only internal gates can test whether update magnitude/checkpoint selection is a plausible rescue before any more training.

## Command

\`\`\`bash
LATENTFM_STATIC_BEST_LATEST_ACK=posthoc_internal_only bash ${ROOT}/ops/launch_latentfm_static_best_latest_update_audit_20260624.sh
\`\`\`

## Runtime classification

Long GPU posthoc batch. Use 30-minute cadence for result checks.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${session}\`; physical GPU: ${GPU}

## Log path

\`${LOG_ROOT}/posthoc.log\`

## Expected outputs

* \`${ROOT}/reports/LATENTFM_STATIC_BEST_LATEST_UPDATE_AUDIT_20260624.md\`
* \`${ROOT}/reports/latentfm_static_best_latest_update_audit_20260624.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_ROOT}/posthoc.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Train-only internal posthoc only.
- Does not read canonical metrics, canonical multi, or Track C held-out query.
- Evaluates latest checkpoints for cap60 response and general exposure repair branches.
EOF

echo "Launched static best-vs-latest audit on GPU ${GPU} in tmux ${session}"
