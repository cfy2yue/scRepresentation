#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_CHEM_UNSEEN_SMOKE_ACK:-}" != "launch_chemical_unseen_bounded_smoke" ]]; then
  cat >&2 <<'EOF'
Refusing to launch chemical unseen-drug/scaffold GPU smoke.

Set:
  LATENTFM_CHEM_UNSEEN_SMOKE_ACK=launch_chemical_unseen_bounded_smoke

Boundary:
  - requires CPU protocol, split, dryload, and train-only pert-means gates
  - train-only/internal SciPlex unseen-drug or unseen-scaffold split
  - no canonical multi, Track C query, or deployable promotion claim
EOF
  exit 4
fi

SPLIT_GATE_JSON=${ROOT}/reports/latentfm_chemical_unseen_drug_scaffold_loader_splits_20260625.json
DRYLOAD_JSON=${ROOT}/reports/latentfm_chemical_unseen_drug_scaffold_dryload_pertmeans_20260625.json
DRUG_CACHE=${ROOT}/dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse

RUN_ROOT=${ROOT}/runs/latentfm_chemical_unseen_drug_scaffold_smokes_20260625
OUT_ROOT=${COUPLED}/output/latentfm_runs/chemical_unseen_drug_scaffold_smokes_20260625
LOG_ROOT=${ROOT}/logs/latentfm_chemical_unseen_drug_scaffold_smokes_20260625
TOTAL_STEPS=${LATENTFM_CHEM_UNSEEN_TOTAL_STEPS:-2500}

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path
checks = {
    "splits": (Path("/data/cyx/1030/scLatent/reports/latentfm_chemical_unseen_drug_scaffold_loader_splits_20260625.json"), "chemical_unseen_drug_scaffold_loader_splits_ready_no_gpu"),
    "dryload": (Path("/data/cyx/1030/scLatent/reports/latentfm_chemical_unseen_drug_scaffold_dryload_pertmeans_20260625.json"), "chemical_unseen_drug_scaffold_dryload_pertmeans_ready_gpu_candidate"),
}
for name, (path, expected) in checks.items():
    if not path.exists():
        raise SystemExit(f"{name} report missing: {path}")
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
    if status != expected:
        raise SystemExit(f"{name} status {status!r} != {expected!r}")
PY

if [[ -n "${LATENTFM_CHEM_UNSEEN_MODES:-}" ]]; then
  IFS=',' read -r -a MODES <<< "${LATENTFM_CHEM_UNSEEN_MODES}"
else
  MODES=("unseen_scaffold" "unseen_drug")
fi
need=${#MODES[@]}
if (( need < 1 || need > 2 )); then
  echo "Need 1-2 smoke modes under current temporary cap, got ${need}" >&2
  exit 4
fi

run_table=${RUN_ROOT}/logs/selected_modes_$(date +%Y%m%d_%H%M%S).jsonl
"${PYTHON}" - "${SPLIT_GATE_JSON}" "${DRYLOAD_JSON}" "${run_table}" "${MODES[@]}" <<'PY'
import json
import sys
from pathlib import Path
split_gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
dry = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
out = Path(sys.argv[3])
want = sys.argv[4:]
split_by = {r["mode"]: r for r in split_gate["rows"]}
dry_by = {r["mode"]: r for r in dry["rows"]}
lines = []
for mode in want:
    if mode not in split_by or mode not in dry_by:
        raise SystemExit(f"missing mode in gates: {mode}")
    if dry_by[mode].get("status") != "ok":
        raise SystemExit(f"dryload/pertmean not ok for {mode}")
    row = {
        "mode": mode,
        "split_file": split_by[mode]["split_file"],
        "pert_means_file": dry_by[mode]["pert_means_file"],
    }
    for key in ("split_file", "pert_means_file"):
        if not Path(row[key]).exists():
            raise SystemExit(f"missing {key} for {mode}: {row[key]}")
    lines.append(json.dumps(row, sort_keys=True))
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "[$(date '+%F %T %Z')] exact GPU/CPU/RAM status before chemical unseen smoke launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 30 | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 2 \
  --max-jobs-per-gpu 1 \
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
reasons = []
if len(suggested) < need:
    reasons.append(f"only {len(suggested)} GPU job slots suggested for need={need}")
if int(payload.get("max_user_gpus") or 0) > 2:
    reasons.append("max_user_gpus exceeds temporary cap 2")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128 GiB")
audit = {
    "status": "fail" if reasons else "pass",
    "reasons": reasons,
    "need": need,
    "assigned_gpus": suggested[:need],
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
    "temporary_caps": {"physical_gpus": 2, "jobs_per_gpu": 1, "cpu_threads_project": 24},
}
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
raise SystemExit(0 if audit["status"] == "pass" else 4)
PY

mapfile -t ASSIGNED_GPUS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
for gpu in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["assigned_gpus"]:
    print(int(gpu))
PY
)

i=0
while IFS= read -r line; do
  mode=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["mode"])
PY
)
  split_file=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["split_file"])
PY
)
  pert_means=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["pert_means_file"])
PY
)
  gpu=${ASSIGNED_GPUS[$i]}
  run_name="xverse_chemical_${mode}_morgan512_2500_seed42"
  session="lfm_${run_name}"
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_CHEM_UNSEEN_SMOKE:-0}" != "1" ]]; then
    echo "Output exists for ${run_name}; set FORCE_LATENTFM_CHEM_UNSEEN_SMOKE=1 to relaunch" >&2
    exit 3
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 3
  fi
  mkdir -p "${run_dir}/logs" "${run_dir}/scripts" "${log_dir}"
  train_script=${run_dir}/scripts/run_${run_name}.sh
  posthoc_script=${run_dir}/scripts/posthoc_${run_name}.sh
  cat > "${train_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export BLIS_NUM_THREADS=3
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
export RAW_DRUG_EMB_CACHE_DIR=${DRUG_CACHE}
export LATENT_DRUG_EMB_CACHE_DIR=${DRUG_CACHE}
export LATENT_BACKBONE=xverse
export EMB_DIM=384
export DATA_DIR=${DATA_DIR}
export BIFLOW_DIR=${BIFLOW_DIR}
export SPLIT_FILE=${split_file}
export PERT_MEANS_FILE=${pert_means}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${log_dir}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${gpu}
export RUN_TAG=${run_name}
export SEED=42
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=1
export FINETUNE_TRAINABLE_SCOPE=all
export TOTAL_STEPS=${TOTAL_STEPS}
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export LR=1e-4
export GAMMA=0.03
export GAMMA_WARMUP_START=500
export GAMMA_WARMUP_END=1500
export MMD_EVERY=4
export OT_PAIR_MODE=multinomial
export SELECTION_METRIC=test_mmd
export SELECTION_MMD_LAMBDA=1.0
export COMPOSITION_DELTA_LOSS_WEIGHT=0.06
export COMPOSITION_DELTA_LOSS_WARMUP_START=500
export COMPOSITION_DELTA_LOSS_WARMUP_END=1500
export ENDPOINT_DELTA_LOSS_WEIGHT=5.0
export ENDPOINT_DELTA_LOSS_WARMUP_START=500
export ENDPOINT_DELTA_LOSS_WARMUP_END=1500
export DS_ALPHA=0.7
export DS_LOSS_ALPHA=0.0
export EVAL_MAX_CONDITIONS=256
export EVAL_MAX_CONDITIONS_PER_DATASET=12
export EVAL_MAX_MSE_CELLS=1024
export EVAL_MAX_MMD_CELLS=1024
export EVAL_MAX_CHUNK=256
export PERT_POOL_AGGREGATIONS="sum mean max min"
export PERT_POOL_SCALE_INIT="0.5 1.0 1.0 1.0"
export PERT_POOL_FUSION_MODE=sum
export PERT_GENE_PROJECTOR_HIDDEN=1024
export PERT_CHEM_PROJECTOR_HIDDEN=1024
export PERT_CHEM_EMB_DIM=512
export CHEM_FALLBACK_EMBED_DIM=512
export PERT_TO_C_INIT_MODE=xavier_small
export USE_PERT_IN_FUSION=1
bash ${TRAIN_LAUNCHER}
EOF
  chmod +x "${train_script}"
  cat > "${posthoc_script}" <<EOF
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
export RAW_DRUG_EMB_CACHE_DIR=${DRUG_CACHE}
export LATENT_DRUG_EMB_CACHE_DIR=${DRUG_CACHE}
eval_dir=${run_dir}/posthoc_eval_internal
mkdir -p "\${eval_dir}"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${split_file} --pert-means-file ${pert_means} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test --out "\${eval_dir}/split_group_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug type_drug --out "\${eval_dir}/condition_family_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${out_dir}/best.pt --groups test --out "\${eval_dir}/split_group_eval_candidate_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${out_dir}/best.pt --groups test_all family_gene family_drug type_drug --out "\${eval_dir}/condition_family_eval_candidate_internal_ode20.json" "\${common[@]}"
EOF
  chmod +x "${posthoc_script}"
  rm -f "${run_dir}/EXIT_CODE" "${run_dir}/FINISHED" "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  date '+%F %T %Z' > "${run_dir}/STARTED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${train_script} > ${log_dir}/launcher.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${log_dir}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"
  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: ${run_name}

## Hypothesis

Chemical scaling tests whether Morgan512-conditioned xverse LatentFM fine-tuning on
train-only SciPlex chemical perturbations improves held-out ${mode}
generalization relative to xverse_8k_anchor, while preserving gene-family
behavior. This is a bounded mechanism smoke, not a deployable claim.

## Command

\`\`\`bash
LATENTFM_CHEM_UNSEEN_SMOKE_ACK=launch_chemical_unseen_bounded_smoke bash ${ROOT}/ops/launch_latentfm_chemical_unseen_drug_scaffold_smokes_20260625.sh
\`\`\`

## Runtime classification

Long GPU training plus posthoc task. Use 30-minute cadence for result checks.

## Start time

$(cat "${run_dir}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${session}\`

Physical GPU: ${gpu}

## Log path

\`${log_dir}/launcher.log\`

Posthoc log: \`${log_dir}/posthoc.log\`

## Expected outputs

* \`${out_dir}/best.pt\`
* \`${run_dir}/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json\`
* \`${run_dir}/posthoc_eval_internal/condition_family_eval_candidate_internal_ode20.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${log_dir}/launcher.log
cat ${run_dir}/EXIT_CODE 2>/dev/null || echo "still running"
cat ${run_dir}/POSTHOC_EXIT_CODE 2>/dev/null || echo "posthoc not complete"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Mode: \`${mode}\`
- Split: \`${split_file}\`
- Train-only pert means: \`${pert_means}\`
- Canonical multi and Track C query are not used.
- Gate after posthoc: prefer candidate only if family_drug/test_all pp improves without family_gene hard harm and MMD hard harm; otherwise close or mutate.
EOF
  echo "Launched ${session} on GPU ${gpu}"
  i=$((i + 1))
done < "${run_table}"

tmux ls || true
for mode in "${MODES[@]}"; do
  run_name="xverse_chemical_${mode}_morgan512_2500_seed42"
  tail -n 8 "${LOG_ROOT}/${run_name}/launcher.log" || true
done
