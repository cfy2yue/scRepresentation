#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_CHEM_V2_FIXEDSTEP_ACK:-}" != "launch_v2_fixedstep_controls_after_protocol_review" ]]; then
  cat >&2 <<'EOF'
Refusing to launch chemical unseen-scaffold V2 fixed-step controls.

Set:
  LATENTFM_CHEM_V2_FIXEDSTEP_ACK=launch_v2_fixedstep_controls_after_protocol_review

Boundary:
  - requires V2 CPU unlock and an external/protocol review
  - independent scaffold V2 seeds 43/44 only
  - descriptor arms: real_morgan512, shuffled_morgan512, random_morgan512
  - candidate posthoc uses fixed final/latest checkpoint only, not best.pt
  - no canonical multi, Track C query, or deployable promotion claim

Example, first two arms under the temporary cap:
  LATENTFM_CHEM_V2_ARMS=real_morgan512:43,real_morgan512:44 \
  LATENTFM_CHEM_V2_FIXEDSTEP_ACK=launch_v2_fixedstep_controls_after_protocol_review \
  bash ops/launch_latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625.sh
EOF
  exit 4
fi

CPU_UNLOCK_JSON=${ROOT}/reports/latentfm_chemical_unseen_scaffold_v2_cpu_unlock_20260625.json
LORENTZ_AUDIT=${ROOT}/reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_V2_EXTERNAL_AUDIT_LORENTZ_20260625.md
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse

RUN_ROOT=${ROOT}/runs/latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625
OUT_ROOT=${COUPLED}/output/latentfm_runs/chemical_unseen_scaffold_v2_fixedstep_controls_20260625
LOG_ROOT=${ROOT}/logs/latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625
TOTAL_STEPS=${LATENTFM_CHEM_V2_TOTAL_STEPS:-2500}

mkdir -p "${RUN_ROOT}/logs" "${OUT_ROOT}" "${LOG_ROOT}"

for required in "${CPU_UNLOCK_JSON}" "${LORENTZ_AUDIT}" "${GPU_HELPER}" "${TRAIN_LAUNCHER}" "${ANCHOR_CKPT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ -n "${LATENTFM_CHEM_V2_ARMS:-}" ]]; then
  IFS=',' read -r -a ARMS <<< "${LATENTFM_CHEM_V2_ARMS}"
else
  ARMS=("real_morgan512:43" "real_morgan512:44")
fi

need=${#ARMS[@]}
if (( need < 1 || need > 2 )); then
  echo "Need 1-2 V2 fixed-step arms per launch under the current temporary cap, got ${need}" >&2
  exit 4
fi

run_table=${RUN_ROOT}/logs/selected_v2_arms_$(date +%Y%m%d_%H%M%S).jsonl
"${PYTHON}" - "${CPU_UNLOCK_JSON}" "${run_table}" "${ARMS[@]}" <<'PY'
import json
import sys
from pathlib import Path

cpu_unlock = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if cpu_unlock.get("status") != "chemical_unseen_scaffold_v2_cpu_unlock_ready_protocol_next_no_gpu":
    raise SystemExit(f"V2 CPU unlock status is not ready: {cpu_unlock.get('status')}")

root = Path("/data/cyx/1030/scLatent")
cache_by_arm = {
    "real_morgan512": root / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625",
}
for row in cpu_unlock.get("control_caches", []):
    cache_by_arm[row["name"]] = Path(row["cache_dir"])
row_by_seed = {int(row["split_seed"]): row for row in cpu_unlock.get("rows", []) if row.get("status") == "ok"}

out = []
for spec in sys.argv[3:]:
    try:
        arm, seed_s = spec.split(":", 1)
        seed = int(seed_s)
    except Exception as exc:
        raise SystemExit(f"invalid arm spec {spec!r}; expected arm:seed") from exc
    if arm not in cache_by_arm:
        raise SystemExit(f"unknown descriptor arm {arm!r}; known={sorted(cache_by_arm)}")
    if seed not in row_by_seed:
        raise SystemExit(f"unknown/invalid V2 seed {seed}; valid={sorted(row_by_seed)}")
    cache = cache_by_arm[arm]
    for required in [cache / "drug_embeddings.npy", cache / "drug_index.json"]:
        if not required.exists():
            raise SystemExit(f"missing cache artifact for {arm}: {required}")
    row = row_by_seed[seed]
    split = Path(row["split_file"])
    means = Path(row["pert_means_file"])
    if not split.exists():
        raise SystemExit(f"missing split file: {split}")
    if not means.exists():
        raise SystemExit(f"missing pert-means file: {means}")
    out.append(
        {
            "arm": arm,
            "seed": seed,
            "drug_cache": str(cache),
            "split_file": str(split),
            "pert_means_file": str(means),
        }
    )

Path(sys.argv[2]).write_text("\n".join(json.dumps(r, sort_keys=True) for r in out) + "\n", encoding="utf-8")
PY

echo "[$(date '+%F %T %Z')] exact GPU/CPU/RAM status before V2 fixed-step controls" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
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

mapfile -t ASSIGNED_GPUS < <("${PYTHON}" - "${gpu_json}" "${need}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
need = int(sys.argv[2])
suggested = [int(x) for x in payload.get("suggested_job_gpus", [])]
system = payload.get("system") or {}
if len(suggested) < need:
    raise SystemExit(f"only {len(suggested)} slots for need={need}")
if int(payload.get("max_user_gpus") or 0) > 2:
    raise SystemExit("temporary cap violation: max_user_gpus > 2")
if float(system.get("mem_available_gib") or 0.0) < 128.0:
    raise SystemExit(f"MemAvailable too low: {system.get('mem_available_gib')}")
for gpu in suggested[:need]:
    print(gpu)
PY
)

i=0
while IFS= read -r line; do
  arm=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["arm"])
PY
)
  seed=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["seed"])
PY
)
  drug_cache=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["drug_cache"])
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
  run_name="xverse_chemical_unseen_scaffold_v2_${arm}_fixedlatest_2500_seed${seed}"
  session="lfm_${run_name}"
  run_dir=${RUN_ROOT}/${run_name}
  out_dir=${OUT_ROOT}/${run_name}
  log_dir=${LOG_ROOT}/${run_name}
  if [[ -e "${out_dir}" && "${FORCE_LATENTFM_CHEM_V2_FIXEDSTEP:-0}" != "1" ]]; then
    echo "Output exists for ${run_name}; set FORCE_LATENTFM_CHEM_V2_FIXEDSTEP=1 to relaunch" >&2
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
export RAW_DRUG_EMB_CACHE_DIR=${drug_cache}
export LATENT_DRUG_EMB_CACHE_DIR=${drug_cache}
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
export TRAIN_EVAL_ENABLED=0
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
export RAW_DRUG_EMB_CACHE_DIR=${drug_cache}
export LATENT_DRUG_EMB_CACHE_DIR=${drug_cache}
candidate_ckpt=${out_dir}/latest.pt
if [[ ! -f "\${candidate_ckpt}" ]]; then
  echo "Missing fixed final/latest checkpoint: \${candidate_ckpt}" >&2
  exit 8
fi
eval_dir=${run_dir}/posthoc_eval_internal
mkdir -p "\${eval_dir}"
echo "\${candidate_ckpt}" > ${run_dir}/FIXED_CANDIDATE_CHECKPOINT
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${split_file} --pert-means-file ${pert_means} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test --out "\${eval_dir}/split_group_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug type_drug --out "\${eval_dir}/condition_family_eval_anchor_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint "\${candidate_ckpt}" --groups test --out "\${eval_dir}/split_group_eval_candidate_internal_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint "\${candidate_ckpt}" --groups test_all family_gene family_drug type_drug --out "\${eval_dir}/condition_family_eval_candidate_internal_ode20.json" "\${common[@]}"
EOF
  chmod +x "${posthoc_script}"
  rm -f "${run_dir}/EXIT_CODE" "${run_dir}/FINISHED" "${run_dir}/POSTHOC_EXIT_CODE" "${run_dir}/POSTHOC_FINISHED"
  date '+%F %T %Z' > "${run_dir}/STARTED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${train_script} > ${log_dir}/launcher.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/FINISHED; if [[ \$rc -eq 0 ]]; then bash ${posthoc_script} > ${log_dir}/posthoc.log 2>&1; prc=\$?; echo \$prc > ${run_dir}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_dir}/POSTHOC_FINISHED; exit \$prc; else exit \$rc; fi'"
  cat > "${run_dir}/RUN_STATUS.md" <<EOF
# Run Status: ${run_name}

## Hypothesis

Independent V2 chemical unseen-scaffold fixed-step control. Real Morgan512
should beat shuffled/random descriptor controls on the same independent
scaffold split if chemical descriptor semantics, not split noise or checkpoint
selection, drives the signal.

## Command

\`\`\`bash
LATENTFM_CHEM_V2_ARMS=${arm}:${seed} \\
LATENTFM_CHEM_V2_FIXEDSTEP_ACK=launch_v2_fixedstep_controls_after_protocol_review \\
bash ${ROOT}/ops/launch_latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625.sh
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

* \`${out_dir}/latest.pt\`
* \`${run_dir}/FIXED_CANDIDATE_CHECKPOINT\`
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

- V2 split seed: ${seed}
- Descriptor arm: ${arm}
- Split: \`${split_file}\`
- Train-only pert means: \`${pert_means}\`
- Drug cache: \`${drug_cache}\`
- Candidate checkpoint is fixed to \`${out_dir}/latest.pt\`; \`best.pt\` is not used for V2 control adjudication.
- Train-time eval/best-checkpoint selection is disabled with \`TRAIN_EVAL_ENABLED=0\`; the separate posthoc block is the only V2 evaluation.
- Canonical multi and Track C query are not used.
- Failure-close: real arms must pass family_drug/type_drug pp >= +0.005,
  test_all pp >= +0.005, family_gene pp >= -0.002, key MMD deltas <= +0.00025,
  median real family_drug pp >= +0.008 across V2 seeds, and real must clearly
  beat shuffled/random controls.
EOF
  echo "Launched ${session} on GPU ${gpu}"
  i=$((i + 1))
done < "${run_table}"

tmux ls || true
while IFS= read -r line; do
  arm=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["arm"])
PY
)
  seed=$("${PYTHON}" - "${line}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["seed"])
PY
)
  run_name="xverse_chemical_unseen_scaffold_v2_${arm}_fixedlatest_2500_seed${seed}"
  tail -n 8 "${LOG_ROOT}/${run_name}/launcher.log" || true
done < "${run_table}"
