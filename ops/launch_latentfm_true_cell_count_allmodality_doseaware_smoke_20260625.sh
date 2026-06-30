#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_ALLMODALITY_DOSEAWARE_SMOKE_ACK:-}" != "launch_allmodality_doseaware_bounded_smoke" ]]; then
  cat >&2 <<'EOF'
Refusing to launch all-modality dose-aware GPU smoke.

Set:
  LATENTFM_ALLMODALITY_DOSEAWARE_SMOKE_ACK=launch_allmodality_doseaware_bounded_smoke

Boundary:
  - requires completed dose-aware all-modality materialization
  - requires metadata/schema/structural dryload/chemical-conditioning/design/loader-split gates
  - uses loader-compatible train/test split derived from internal_val_allmodality_doseaware
  - uses Morgan512 projected chemical cache to match xverse_8k_anchor chemical branch
  - train-only/internal selection only
  - no canonical multi, Track C query, or deployable promotion claim
EOF
  exit 4
fi

MATERIALIZER_JSON=${ROOT}/reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json
METADATA_JSON=${ROOT}/reports/latentfm_true_cell_count_allmodality_doseaware_condition_metadata_backfill_20260625.json
SCHEMA_JSON=${ROOT}/reports/latentfm_true_cell_count_allmodality_doseaware_schema_gate_20260625.json
DRYLOAD_JSON=${ROOT}/reports/latentfm_true_cell_count_allmodality_doseaware_dryload_gate_20260625.json
CHEM_GATE_JSON=${ROOT}/reports/latentfm_true_cell_count_allmodality_doseaware_chemical_conditioning_gate_20260625.json
DESIGN_JSON=${ROOT}/reports/latentfm_true_cell_count_allmodality_doseaware_design_controls_20260625.json
LOADER_SPLITS_JSON=${ROOT}/reports/latentfm_true_cell_count_allmodality_doseaware_loader_splits_20260625.json
MORGAN512_REPORT=${ROOT}/reports/latentfm_sciplex_morgan512_projected_cache_20260625.json
DRUG_CACHE=${ROOT}/dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
BIFLOW_DIR=${ROOT}/dataset/biFlow_data

RUN_ROOT=${ROOT}/runs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625
OUT_ROOT=${COUPLED}/output/latentfm_runs/true_cell_count_allmodality_doseaware_smokes_20260625
LOG_ROOT=${ROOT}/logs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625
TOTAL_STEPS=${LATENTFM_ALLMODALITY_DOSEAWARE_TOTAL_STEPS:-2500}

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}"

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

checks = {
    "materializer": (Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json"), "allmodality_doseaware_materialized_no_gpu"),
    "metadata": (Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_condition_metadata_backfill_20260625.json"), "allmodality_doseaware_condition_metadata_written_no_gpu"),
    "schema": (Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_schema_gate_20260625.json"), "allmodality_doseaware_schema_pass_no_gpu"),
    "dryload": (Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_dryload_gate_20260625.json"), "allmodality_doseaware_dryload_pass_no_gpu"),
    "chemical_conditioning": (Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_chemical_conditioning_gate_20260625.json"), "allmodality_doseaware_chemical_conditioning_pass_no_gpu"),
    "loader_splits": (Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_loader_splits_20260625.json"), "allmodality_doseaware_loader_splits_ready_no_gpu"),
    "morgan512": (Path("/data/cyx/1030/scLatent/reports/latentfm_sciplex_morgan512_projected_cache_20260625.json"), "sciplex_morgan512_projected_cache_ready_no_gpu"),
}
for name, (path, expected) in checks.items():
    if not path.exists():
        raise SystemExit(f"{name} gate/report missing: {path}")
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
    if status != expected:
        raise SystemExit(f"{name} status {status!r} != {expected!r}")
design = json.loads(Path("/data/cyx/1030/scLatent/reports/latentfm_true_cell_count_allmodality_doseaware_design_controls_20260625.json").read_text(encoding="utf-8"))
if not design.get("smoke_ready_after_schema_dryload"):
    raise SystemExit(f"design gate is not smoke-ready: {design.get('status')}")
PY

declare -a DEFAULT_RUN_IDS=(
  "all_modality_doseaware_fixed64_budget16_32_64_budget16_seed42"
  "all_modality_doseaware_fixed64_budget16_32_64_budget32_seed42"
  "all_modality_doseaware_fixed64_budget16_32_64_budget64_seed42"
  "all_modality_doseaware_fixed64_budget16_32_64_budget64_seed43"
)

if [[ -n "${LATENTFM_ALLMODALITY_DOSEAWARE_RUN_IDS:-}" ]]; then
  IFS=',' read -r -a RUN_IDS <<< "${LATENTFM_ALLMODALITY_DOSEAWARE_RUN_IDS}"
else
  RUN_IDS=("${DEFAULT_RUN_IDS[@]}")
fi

need=${#RUN_IDS[@]}
if (( need < 1 || need > 4 )); then
  echo "Need 1-4 runs under temporary cap, got ${need}" >&2
  exit 4
fi

run_table=${RUN_ROOT}/logs/selected_runs_$(date +%Y%m%d_%H%M%S).jsonl
"${PYTHON}" - "${MATERIALIZER_JSON}" "${LOADER_SPLITS_JSON}" "${run_table}" "${RUN_IDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

mat = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
loader = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
out = Path(sys.argv[3])
want = sys.argv[4:]
mat_by = {r["run_id"]: r for r in mat.get("materialized_rows", [])}
loader_by = {r["run_id"]: r for r in loader.get("rows", [])}
lines = []
for run_id in want:
    if run_id not in mat_by:
        raise SystemExit(f"missing materialized row: {run_id}")
    if run_id not in loader_by:
        raise SystemExit(f"missing loader split row: {run_id}")
    m = mat_by[run_id]
    l = loader_by[run_id]
    if l.get("status") != "ok":
        raise SystemExit(f"loader row not ok for {run_id}: {l.get('reasons')}")
    row = {
        "run_id": run_id,
        "budget": int(m["budget"]),
        "seed": int(m["seed"]),
        "data_dir": m["data_dir"],
        "source_split_file": m["split_file"],
        "loader_split_file": l["loader_split_file"],
        "pert_means_file": str(Path(m["data_dir"]) / "pert_means.npz"),
    }
    for p in [Path(row["data_dir"]) / "manifest.json", Path(row["data_dir"]) / "condition_metadata.json", Path(row["pert_means_file"]), Path(row["loader_split_file"])]:
        if not p.exists():
            raise SystemExit(f"missing selected artifact for {run_id}: {p}")
    lines.append(json.dumps(row, sort_keys=True))
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "[$(date '+%F %T %Z')] exact GPU/CPU/RAM status before all-modality smoke launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
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
  --max-jobs-per-gpu 2 \
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
if int(payload.get("max_jobs_per_gpu") or 0) > 2:
    reasons.append("max_jobs_per_gpu exceeds temporary cap 2")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 128 GiB")
audit = {
    "status": "fail" if reasons else "pass",
    "reasons": reasons,
    "need": need,
    "assigned_gpus": suggested[:need],
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
    "temporary_caps": {"physical_gpus": 2, "jobs_per_gpu": 2, "cpu_threads_project": 24},
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
  run_id=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["run_id"])
PY
)
  budget=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["budget"])
PY
)
  seed=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["seed"])
PY
)
  data_dir=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["data_dir"])
PY
)
  split_file=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["loader_split_file"])
PY
)
  pert_means=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["pert_means_file"])
PY
)
  run_name="xverse_allmod_doseaware_morgan512_budget${budget}_seed${seed}_${TOTAL_STEPS}"
  gpu=${ASSIGNED_GPUS[$i]}
  session=lfm_${run_name}
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_ALLMODALITY_DOSEAWARE_SMOKE:-0}" != "1" ]]; then
    echo "Output exists for ${run_name}; set FORCE_LATENTFM_ALLMODALITY_DOSEAWARE_SMOKE=1 to relaunch" >&2
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
export DATA_DIR=${data_dir}
export BIFLOW_DIR=${BIFLOW_DIR}
export SPLIT_FILE=${split_file}
export PERT_MEANS_FILE=${pert_means}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${log_dir}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${gpu}
export RUN_TAG=${run_name}
export SEED=${seed}
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
common=(--data-dir ${data_dir} --biflow-dir ${BIFLOW_DIR} --split-file ${split_file} --pert-means-file ${pert_means} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
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

Dose-aware all-modality scaling tests whether adding valid chemical dose-level conditions and gene perturbation conditions under fixed train-cell budget ${budget} improves train-only/internal gene+drug generalization relative to xverse_8k_anchor. This is a bounded smoke, not a deployable claim.

## Command

\`\`\`bash
LATENTFM_ALLMODALITY_DOSEAWARE_SMOKE_ACK=launch_allmodality_doseaware_bounded_smoke bash ${ROOT}/ops/launch_latentfm_true_cell_count_allmodality_doseaware_smoke_20260625.sh
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

- RUN_ID: \`${run_id}\`
- Capped DATA_DIR: \`${data_dir}\`
- Loader split: \`${split_file}\`
- Train-only pert means: \`${pert_means}\`
- Drug cache: \`${DRUG_CACHE}\`
- Temporary resource cap: max 2 physical GPUs, max 2 training jobs/GPU, <=24 CPU threads project-wide.
- Canonical multi and Track C query are not used.
- Stop rule: summarize train-only/internal test/family_gene/family_drug only; any gene or drug family hard harm, unsafe tails, or failed posthoc closes/mutates before more GPU.
EOF
  echo "Launched ${run_name} on GPU ${gpu} in tmux ${session}"
  i=$((i + 1))
done < "${run_table}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_true_cell_count_allmodality_doseaware_smokes_20260625

## Command

\`\`\`bash
LATENTFM_ALLMODALITY_DOSEAWARE_SMOKE_ACK=launch_allmodality_doseaware_bounded_smoke bash ${ROOT}/ops/launch_latentfm_true_cell_count_allmodality_doseaware_smoke_20260625.sh
\`\`\`

## Runtime classification

Long GPU training batch. Each child run has its own RUN_STATUS.md.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

$(printf '* `%s`\n' "${RUN_IDS[@]/#/lfm_xverse_allmod_doseaware_morgan512_}")

## Log path

\`${LOG_ROOT}/<run_name>/launcher.log\`

## Expected outputs

* \`${RUN_ROOT}/<run_name>/posthoc_eval_internal/*.json\`
* all-modality smoke decision report after summarization

## How to check manually

\`\`\`bash
tmux ls
find ${RUN_ROOT} -name EXIT_CODE -o -name POSTHOC_EXIT_CODE
nvidia-smi
\`\`\`

## Current status

Started ${need} all-modality dose-aware smokes.

## Notes

- Uses Morgan512 projected cache:
  \`${DRUG_CACHE}\`
- Current temporary cap: max 2 physical GPUs, max 2 training jobs per GPU, <=24 CPU threads.
- No canonical multi or Track C query.
EOF

tmux ls
