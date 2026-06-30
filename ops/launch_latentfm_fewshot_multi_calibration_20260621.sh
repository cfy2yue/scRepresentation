#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_fewshot_multi_calibration_20260621
OUT_ROOT=${COUPLED}/output/latentfm_runs/fewshot_multi_calibration_20260621
LOG_ROOT=${ROOT}/logs/latentfm_fewshot_multi_calibration_20260621
SPLIT_DIR=${RUN_ROOT}/splits
DATA_DIR=${ROOT}/dataset/latentfm_full/scfoundation
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
SPLIT_BUILDER=${ROOT}/ops/create_latentfm_fewshot_multi_splits_20260621.py
POSTHOC_SCRIPT=${ROOT}/ops/run_latentfm_fewshot_multi_calibration_posthoc_20260621.sh
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/scripts" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${DATA_DIR}/manifest.json" \
  "${CANONICAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${SPLIT_BUILDER}" \
  "${POSTHOC_SCRIPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" "${SPLIT_BUILDER}"

declare -a ARMS=(
  "canonical_refinetune|scf_prior010_inject_fewshot_canonical_refinetune_4k|${CANONICAL_SPLIT}|${RUN_ROOT}/canonical_refinetune_metadata.json|0"
  "wessels_multi16|scf_prior010_inject_fewshot_wessels_multi16_4k|${SPLIT_DIR}/wessels_multi16_split_seed42_fewshot20260621.json|${SPLIT_DIR}/wessels_multi16_metadata.json|16"
  "norman_wessels_multi32|scf_prior010_inject_fewshot_nw_multi32_4k|${SPLIT_DIR}/norman_wessels_multi32_split_seed42_fewshot20260621.json|${SPLIT_DIR}/norman_wessels_multi32_metadata.json|32"
  "norman_multi16|scf_prior010_inject_fewshot_norman_multi16_4k|${SPLIT_DIR}/norman_multi16_split_seed42_fewshot20260621.json|${SPLIT_DIR}/norman_multi16_metadata.json|16"
  "norman_wessels_gasperini_multi33|scf_prior010_inject_fewshot_nwg_multi33_4k|${SPLIT_DIR}/norman_wessels_gasperini_multi33_split_seed42_fewshot20260621.json|${SPLIT_DIR}/norman_wessels_gasperini_multi33_metadata.json|33"
)

cat > "${RUN_ROOT}/canonical_refinetune_metadata.json" <<EOF
{
  "arm": "canonical_refinetune",
  "recipe": {},
  "moved_by_dataset": {},
  "interpretation": "control run: continue fine-tuning the anchor for 4k steps on the canonical zero-shot split, with no train_multi added"
}
EOF

if [[ "${FORCE_FEWSHOT_MULTI_RERUN:-0}" != "1" ]]; then
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r _arm run_name _split _meta _moved <<< "${spec}"
    if [[ -e "${OUT_ROOT}/${run_name}" ]]; then
      echo "Output exists for ${run_name}; set FORCE_FEWSHOT_MULTI_RERUN=1 to relaunch" >&2
      exit 3
    fi
  done
fi

echo "[$(date '+%F %T %Z')] exact GPU status before few-shot launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-jobs-per-gpu 4 \
  --need "${#ARMS[@]}" \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignments_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${#ARMS[@]}" "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
need = int(sys.argv[2])
out = Path(sys.argv[3])
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
stable = [g for g in gpus.values() if g.get("stable_light")]
stable_count = len(stable)
if stable_count >= 5:
    physical_budget = min(4, stable_count)
else:
    physical_budget = max(0, min(4, stable_count - 1))
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
candidate_order = [int(x) for x in payload.get("candidate_order", [])]

selected: list[int] = []
new_physical_used = 0
for idx in candidate_order:
    gpu = gpus[idx]
    if not gpu.get("available"):
        continue
    if idx in selected:
        continue
    if idx in active_user:
        if len(set(selected) | {idx}) <= physical_budget:
            selected.append(idx)
        continue
    if len((set(selected) | active_user) | {idx}) <= physical_budget:
        selected.append(idx)
        new_physical_used += 1
    if len(selected) >= physical_budget:
        break

assignments: list[int] = []
slots: dict[int, int] = {}
for idx in selected:
    slots[idx] = int(gpus[idx].get("colocation_slots_free", 0))

while len(assignments) < need:
    progressed = False
    for idx in list(selected):
        if slots.get(idx, 0) <= 0:
            continue
        assignments.append(idx)
        slots[idx] -= 1
        progressed = True
        if len(assignments) >= need:
            break
    if not progressed:
        break

system = payload.get("system") or {}
audit = {
    "status": "pass",
    "stable_light_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "selected_unique_gpus": selected,
    "assignments": assignments,
    "requested_jobs": need,
    "min_jobs_required": 3,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if len(assignments) < 3:
    reasons.append(f"only {len(assignments)} job slots available, need at least 3")
mem = float(system.get("mem_available_gib") or 0.0)
load = float(system.get("load1_per_cpu") or 0.0)
if mem < 96.0:
    reasons.append(f"MemAvailable {mem:.1f} GiB < 96.0 GiB")
if load > 2.0:
    reasons.append(f"load1_per_cpu {load:.3f} > 2.000")
if physical_budget <= 0:
    reasons.append("no physical GPU budget under empty-card leave-one-free rule")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

mapfile -t ASSIGNMENTS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for idx in payload.get("assignments", []):
    print(idx)
PY
)

launch_count="${#ASSIGNMENTS[@]}"
if (( launch_count > ${#ARMS[@]} )); then
  launch_count="${#ARMS[@]}"
fi

rm -f "${RUN_ROOT}"/*.EXIT_CODE "${RUN_ROOT}"/*.FINISHED

write_run_script() {
  local arm="$1"
  local run_name="$2"
  local split_file="$3"
  local gpu="$4"
  local run_script="${RUN_ROOT}/scripts/run_${run_name}.sh"
  cat > "${run_script}" <<EOF
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

${PYTHON} -m model.latent.train \\
  --data-dir ${DATA_DIR} \\
  --biflow-dir ${BIFLOW_DIR} \\
  --split-file ${split_file} \\
  --latent-backbone scfoundation \\
  --emb-dim 3072 \\
  --save-dir ${OUT_ROOT}/${run_name} \\
  --log-file train.log \\
  --model-type control_mlp \\
  --init-checkpoint ${ANCHOR_CKPT} \\
  --batch-size 64 \\
  --grad-accum-steps 1 \\
  --min-cells 32 \\
  --scale-noise 0.02 \\
  --ds-alpha 0.7 \\
  --ds-loss-alpha 0.0 \\
  --ds-loss-warmup-start 0 \\
  --total-steps 4000 \\
  --lr 0.0001 \\
  --warmup-steps 300 \\
  --lr-decay-steps 4000 \\
  --print-every 200 \\
  --eval-every 2000 \\
  --eval-max-conditions 256 \\
  --eval-max-conditions-per-dataset 12 \\
  --eval-max-mse-cells 1024 \\
  --eval-max-mmd-cells 512 \\
  --eval-max-chunk 128 \\
  --selection-metric pearson_pert_minus_mmd \\
  --selection-mmd-lambda 0.5 \\
  --ot-method torch_sinkhorn \\
  --ot-sinkhorn-reg 0.05 \\
  --ot-sinkhorn-iter 50 \\
  --ot-threads 4 \\
  --prefetch 4 \\
  --n-ot-workers 4 \\
  --use-mmd \\
  --gamma 0.03 \\
  --gamma-warmup-start 50000 \\
  --gamma-warmup-end 100000 \\
  --mmd-every 1 \\
  --mmd-estimator unbiased \\
  --endpoint-delta-loss-weight 2.0 \\
  --endpoint-delta-loss-warmup-start 0 \\
  --endpoint-delta-loss-warmup-end 1000 \\
  --composition-delta-loss-weight 0.0 \\
  --condition-prior-delta-loss-weight 0.10 \\
  --condition-prior-delta-loss-warmup-start 0 \\
  --condition-prior-delta-loss-warmup-end 1000 \\
  --condition-prior-delta-loss-every 1 \\
  --condition-prior-bank-max-cells 512 \\
  --condition-prior-num-genes 2 \\
  --condition-delta-head-use-in-model \\
  --use-ema \\
  --ema-update-after 1000 \\
  --ema-decay 0.999 \\
  --use-amp \\
  --amp-dtype bf16 \\
  --use-pert-condition \\
  --pert-gene-emb-cache-dir ${GENE_CACHE} \\
  --pert-condition-embedding-source scgpt_embed_gene \\
  --pert-pool-aggregations mean max min \\
  --pert-pool-scale-init 1.0 1.0 1.0 \\
  --pert-pool-fusion-mode sum \\
  --pert-type-adapter-mode scalar \\
  --pert-gene-projector-hidden 1024 \\
  --pert-chem-projector-hidden 1024 \\
  --pert-to-c-init-mode xavier_small \\
  --use-pert-in-fusion \\
  --patience 6
EOF
  chmod +x "${run_script}"
}

manifest_tmp="${RUN_ROOT}/launch_manifest.tmp.jsonl"
: > "${manifest_tmp}"

for ((i=0; i<launch_count; i++)); do
  spec="${ARMS[$i]}"
  gpu="${ASSIGNMENTS[$i]}"
  IFS='|' read -r arm run_name split_file metadata_file moved_multi <<< "${spec}"
  write_run_script "${arm}" "${run_name}" "${split_file}" "${gpu}"
  session="lfm_${run_name}"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${RUN_ROOT}/scripts/run_${run_name}.sh > ${LOG_ROOT}/${run_name}.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/${run_name}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/${run_name}.FINISHED; exit \$rc'"
  date '+%F %T %Z' > "${RUN_ROOT}/${run_name}.STARTED"
  "${PYTHON}" - "${manifest_tmp}" <<PY
import json
from pathlib import Path
path = Path("${manifest_tmp}")
row = {
    "arm": "${arm}",
    "run_name": "${run_name}",
    "split_file": "${split_file}",
    "metadata_file": "${metadata_file}",
    "moved_multi": int("${moved_multi}"),
    "session": "${session}",
    "gpu": int("${gpu}"),
    "out_dir": "${OUT_ROOT}/${run_name}",
    "log": "${LOG_ROOT}/${run_name}.log",
    "started": Path("${RUN_ROOT}/${run_name}.STARTED").read_text(encoding="utf-8").strip(),
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\\n")
PY
  echo "Launched ${run_name} on physical GPU${gpu}"
done

"${PYTHON}" - "${manifest_tmp}" "${RUN_ROOT}/launch_manifest.json" "${assignment_json}" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
assign = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
payload = {
    "created": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    "experiment": "latentfm_fewshot_multi_calibration_20260621",
    "anchor_checkpoint": "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/condition_prior_teacher_injection_20260619/scf_prior010_inject_e2_4k/best.pt",
    "data_dir": "/data/cyx/1030/dataset/latentfm_full/scfoundation",
    "launched_runs": rows,
    "gpu_assignment_audit": assign,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
rm -f "${manifest_tmp}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_fewshot_multi_calibration_20260621

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_fewshot_multi_calibration_20260621.sh
\`\`\`

## Runtime classification

Long LatentFM training tasks. Use 30-minute cadence for checks.

## Start time

$(date '+%F %T %Z')

## tmux / GPUs

$( "${PYTHON}" - "${RUN_ROOT}/launch_manifest.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in payload.get("launched_runs", []):
    print(f"* `{row['run_name']}`: `{row['session']}`, physical GPU{row['gpu']}")
PY
)

## Logs

$( "${PYTHON}" - "${RUN_ROOT}/launch_manifest.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for row in payload.get("launched_runs", []):
    print(f"* `{row['log']}`")
PY
)

## Expected outputs

* \`${RUN_ROOT}/launch_manifest.json\`
* \`${ROOT}/reports/LATENTFM_FEWSHOT_MULTI_CALIBRATION_SUMMARY_20260621.md\`

## How to check manually

\`\`\`bash
cat ${RUN_ROOT}/*.EXIT_CODE 2>/dev/null || echo training still running
cat ${RUN_ROOT}/POSTHOC_EXIT_CODE 2>/dev/null || echo posthoc not finished
tmux ls
nvidia-smi
\`\`\`

## Current status

Started training and posthoc watcher.

## Notes

Diagnostic few-shot multi-calibration. Not a zero-shot promotion experiment.
Each candidate will be compared with the unchanged anchor evaluated on the same
custom split.
EOF

tmux new -d -s latentfm_fewshot_multi_calibration_posthoc_20260621 \
  "bash -lc 'bash ${POSTHOC_SCRIPT}'"

echo "Launch manifest: ${RUN_ROOT}/launch_manifest.json"
echo "RUN_STATUS: ${RUN_ROOT}/RUN_STATUS.md"
