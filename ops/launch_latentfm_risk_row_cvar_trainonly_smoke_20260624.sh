#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

if [[ "${LATENTFM_RISK_ROW_CVAR_ACK:-}" != "trainonly_noeval_one_capped_smoke" ]]; then
  echo "Set LATENTFM_RISK_ROW_CVAR_ACK=trainonly_noeval_one_capped_smoke" >&2
  exit 4
fi

RUN_ROOT=${ROOT}/runs/latentfm_risk_row_cvar_trainonly_20260624
OUT_ROOT=${COUPLED}/output/latentfm_runs/risk_row_cvar_trainonly_20260624
LOG_ROOT=${ROOT}/logs/latentfm_risk_row_cvar_trainonly_20260624
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
SPLIT_FILE=${BIFLOW_DIR}/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json
PERT_MEANS=${ROOT}/runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/xverse_trainonly_scaling_general_exposure_cap_v2_pert_means.npz
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
TRAIN_LAUNCHER=${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh
CODE_GATE_JSON=${ROOT}/reports/latentfm_risk_row_cvar_loss_code_gate_20260624.json
EXTERNAL_AUDIT=${ROOT}/reports/LATENTFM_RISK_ROW_CVAR_EXTERNAL_AUDIT_BERNOULLI_20260624.md

RUN_NAME=${RISK_ROW_RUN_NAME:-xverse_risk_row_cvar_allrisk_w020_2k_seed42}
SESSION=lfm_${RUN_NAME}
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
OUT_DIR=${OUT_ROOT}/${RUN_NAME}
LOG_DIR=${LOG_ROOT}/${RUN_NAME}
RISK_FILTER=${RISK_ROW_CVAR_DATASET_FILTER_VALUE:-Nadig_hepg2,Nadig_jurket,NormanWeissman2019_filtered,ReplogleWeissman2022_K562_gwps,Replogle_RPE1essential,TianActivation}

if [[ -e "${RUN_DIR}" || -e "${OUT_DIR}" || -e "${LOG_DIR}" ]]; then
  echo "Run/output/log dir already exists for ${RUN_NAME}; choose a fresh run name" >&2
  exit 3
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 3
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_DIR}/logs" "${RUN_DIR}/scripts" "${OUT_ROOT}" "${LOG_DIR}" "${ROOT}/reports"

for required in \
  "${DATA_DIR}/manifest.json" \
  "${SPLIT_FILE}" \
  "${PERT_MEANS}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${GPU_HELPER}" \
  "${TRAIN_LAUNCHER}" \
  "${CODE_GATE_JSON}" \
  "${EXTERNAL_AUDIT}"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

"${PYTHON}" - "${CODE_GATE_JSON}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
status = payload.get("status")
if status != "risk_row_cvar_loss_code_gate_pass_unit_validated_no_gpu":
    raise SystemExit(f"code gate is not pass: {status!r}")
PY

split_audit=${RUN_ROOT}/logs/split_riskrow_audit_$(date +%Y%m%d_%H%M%S).json
"${PYTHON}" - "${SPLIT_FILE}" "${split_audit}" "${RISK_FILTER}" <<'PY'
import json, sys
from pathlib import Path
split_path = Path(sys.argv[1])
out = Path(sys.argv[2])
risk_filter = sys.argv[3]
split = json.loads(split_path.read_text())
risk_datasets = [item.strip() for chunk in risk_filter.split(";") for item in chunk.split(",") if item.strip()]
forbidden = {"test_multi", "test_multi_unseen", "query", "heldout_query", "support_query"}
audit = {
    "split_file": str(split_path),
    "risk_filter": risk_filter,
    "risk_datasets": risk_datasets,
    "dataset_counts": {},
    "forbidden_keys_seen": sorted({k for entry in split.values() for k in entry if k in forbidden}),
    "status": "pass",
    "reasons": [],
}
if "split_seed42.json" in split_path.name:
    audit["reasons"].append("canonical_split_path")
if not risk_datasets:
    audit["reasons"].append("empty_risk_filter")
for ds in risk_datasets:
    entry = split.get(ds) or {}
    counts = {k: len(v or []) for k, v in entry.items() if isinstance(v, list)}
    audit["dataset_counts"][ds] = counts
    if counts.get("train", 0) <= 0:
        audit["reasons"].append(f"no_train_rows:{ds}")
if audit["forbidden_keys_seen"]:
    audit["reasons"].append("forbidden_query_or_multi_keys_present")
if audit["reasons"]:
    audit["status"] = "fail"
out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
raise SystemExit(0 if audit["status"] == "pass" else 5)
PY

echo "[$(date '+%F %T %Z')] exact GPU/CPU/RAM status before risk-row launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
free -h | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
df -h "${ROOT}" | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"
ps -u cyx -o pid,pcpu,pmem,comm,args --sort=-pcpu | head -n 15 | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json=${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-user-gpus 4 \
  --max-jobs-per-gpu 4 \
  --need 1 \
  --json-only > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json=${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json
"${PYTHON}" - "${gpu_json}" "${assignment_json}" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
suggested = [int(x) for x in payload.get("suggested_job_gpus", [])]
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "need": 1,
    "assigned_gpus": suggested[:1],
    "active_user_gpus": payload.get("active_user_gpus"),
    "allowed_physical_user_gpus": payload.get("allowed_physical_user_gpus"),
    "new_physical_slots": payload.get("new_physical_slots"),
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
    "reasons": [],
}
if len(suggested) < 1:
    audit["reasons"].append("no_gpu_slot_suggested")
if float(system.get("mem_available_gib") or 0) < 128:
    audit["reasons"].append("low_ram")
if float(system.get("load1_per_cpu") or 0) > 2:
    audit["reasons"].append("high_cpu_load")
if audit["reasons"]:
    audit["status"] = "fail"
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
raise SystemExit(0 if audit["status"] == "pass" else 6)
PY

GPU=$("${PYTHON}" - "${assignment_json}" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["assigned_gpus"][0])
PY
)

train_script=${RUN_DIR}/scripts/run_${RUN_NAME}.sh

cat > "${train_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
export LATENT_BACKBONE=xverse
export DATA_DIR=${DATA_DIR}
export BIFLOW_DIR=${BIFLOW_DIR}
export SPLIT_FILE=${SPLIT_FILE}
export PERT_MEANS_FILE=${PERT_MEANS}
export OUT_ROOT=${OUT_ROOT}
export LOG_ROOT=${LOG_DIR}
export GENE_CACHE=${GENE_CACHE}
export PYTHON_BIN=${PYTHON}
export GPU=${GPU}
export RUN_TAG=${RUN_NAME}
export SEED=42
export INIT_CHECKPOINT=${ANCHOR_CKPT}
export INIT_CHECKPOINT_USE_EMA=1
export FINETUNE_TRAINABLE_SCOPE=all
export TOTAL_STEPS=2000
export BATCH_SIZE=64
export GRAD_ACCUM_STEPS=1
export LR=1e-4
export GAMMA=0.03
export GAMMA_WARMUP_START=300
export GAMMA_WARMUP_END=1200
export MMD_EVERY=1
export MMD_ESTIMATOR=biased
export MMD_DATASET_FILTER=
export RISK_ROW_CVAR_LOSS_WEIGHT=0.20
export RISK_ROW_CVAR_LOSS_WARMUP_START=100
export RISK_ROW_CVAR_LOSS_WARMUP_END=500
export RISK_ROW_CVAR_DATASET_FILTER=${RISK_FILTER}
export RISK_ROW_CVAR_HISTORY_SIZE=256
export RISK_ROW_CVAR_MIN_HISTORY=4
export RISK_ROW_CVAR_TOP_FRAC=0.25
export RISK_ROW_CVAR_MMD_THRESHOLD=0.005
export TRAIN_EVAL_ENABLED=0
export SELECTION_METRIC=pearson_pert_minus_mmd
export SELECTION_MMD_LAMBDA=1.0
export COMPOSITION_DELTA_LOSS_WEIGHT=0.06
export COMPOSITION_DELTA_LOSS_WARMUP_START=500
export COMPOSITION_DELTA_LOSS_WARMUP_END=1500
export ENDPOINT_DELTA_LOSS_WEIGHT=5.0
export ENDPOINT_DELTA_LOSS_WARMUP_START=500
export ENDPOINT_DELTA_LOSS_WARMUP_END=1500
export ANCHOR_REPLAY_LOSS_WEIGHT=0.5
export ANCHOR_REPLAY_LOSS_WARMUP_START=300
export ANCHOR_REPLAY_LOSS_WARMUP_END=1200
export ANCHOR_REPLAY_CONDITION_FILTER=all
export ANCHOR_REPLAY_DATASET_FILTER=
export ANCHOR_REPLAY_CHECKPOINT=${ANCHOR_CKPT}
export ANCHOR_REPLAY_CHECKPOINT_USE_EMA=1
export EVAL_MAX_CONDITIONS=0
export EVAL_MAX_CONDITIONS_PER_DATASET=0
export EVAL_MAX_MSE_CELLS=0
export EVAL_MAX_MMD_CELLS=2048
export EVAL_MAX_CHUNK=256
export PERT_POOL_AGGREGATIONS="sum mean max min"
export PERT_POOL_SCALE_INIT="0.5 1.0 1.0 1.0"
export PERT_POOL_FUSION_MODE=sum
export PERT_GENE_PROJECTOR_HIDDEN=1024
export PERT_CHEM_PROJECTOR_HIDDEN=1024
export PERT_TO_C_INIT_MODE=xavier_small
export USE_PERT_IN_FUSION=1
bash ${TRAIN_LAUNCHER}
EOF
chmod +x "${train_script}"

"${PYTHON}" - "${RUN_DIR}" "${COUPLED}" \
  "${COUPLED}/model/latent/config.py" \
  "${COUPLED}/model/latent/train.py" \
  "${COUPLED}/model/latent/scripts/run_full_stack_latentfm.sh" \
  "${ROOT}/ops/launch_latentfm_risk_row_cvar_trainonly_smoke_20260624.sh" \
  "${ROOT}/ops/audit_latentfm_risk_row_cvar_loss_code_gate_20260624.py" \
  "${ROOT}/reports/LATENTFM_RISK_ROW_CVAR_EXTERNAL_AUDIT_BERNOULLI_20260624.md" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
repo = Path(sys.argv[2])
files = [Path(p) for p in sys.argv[3:]]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

try:
    git_status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"],
        text=True,
        stderr=subprocess.STDOUT,
    )
except Exception as exc:
    git_status = f"git status failed: {exc}"
try:
    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
except Exception as exc:
    git_head = f"git rev-parse failed: {exc}"

payload = {
    "git_head": git_head,
    "git_status_short": git_status.splitlines(),
    "file_sha256": {str(path): sha256(path) for path in files},
}
(run_dir / "provenance.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
(run_dir / "git_status_short.txt").write_text(git_status, encoding="utf-8")
PY

date '+%F %T %Z' > "${RUN_DIR}/${RUN_NAME}.STARTED"
cat > "${RUN_DIR}/RUN_STATUS.md" <<EOF
# Run Status: ${RUN_NAME}

## Hypothesis

Online train-only risk-row CVaR/top-tail MMD can add no-harm pressure only on
previously high-tail rows in known risk datasets, instead of applying a scalar
dataset-filtered MMD weight to every target batch. This smoke tests mechanism
activation and bounded train-only feasibility, not final model promotion.

## Command

\`\`\`bash
LATENTFM_RISK_ROW_CVAR_ACK=trainonly_noeval_one_capped_smoke bash ${ROOT}/ops/launch_latentfm_risk_row_cvar_trainonly_smoke_20260624.sh
\`\`\`

## Runtime classification

Long GPU training task. Use 30-minute cadence for result checks after launch
sanity verification.

## Start time

$(date '+%F %T %Z')

## PID / tmux / scheduler ID

tmux session: \`${SESSION}\`; physical GPU: \`${GPU}\`

## Log path

\`${LOG_DIR}/launcher.log\`

## Expected outputs

* \`${OUT_DIR}/latest.pt\`
* \`${OUT_DIR}/config.json\`
* \`${RUN_DIR}/${RUN_NAME}.EXIT_CODE\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 ${LOG_DIR}/launcher.log
cat ${RUN_DIR}/${RUN_NAME}.EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

- Train-only/no-eval smoke: \`TRAIN_EVAL_ENABLED=0\`, so no epoch/final IID/OOD
  evaluation, no best checkpoint selection, and no canonical no-harm.
- Split: \`${SPLIT_FILE}\`.
- Risk-row filter: \`${RISK_FILTER}\`.
- Gate before launch:
  \`${ROOT}/reports/LATENTFM_RISK_ROW_CVAR_LOSS_CODE_GATE_20260624.md\`.
- External audit:
  \`${EXTERNAL_AUDIT}\`.
- Resource assignment: \`${assignment_json}\`.
- Stop signal: training must finish with exit code 0 and logs must show
  \`risk_row_obs>0\` and \`risk_row_apply>0\` before any extension or posthoc.
EOF

tmux new -d -s "${SESSION}" "bash -lc 'bash ${train_script} > ${LOG_DIR}/launcher.log 2>&1; rc=\$?; echo \$rc > ${RUN_DIR}/${RUN_NAME}.EXIT_CODE; date \"+%F %T %Z\" > ${RUN_DIR}/${RUN_NAME}.FINISHED; exit \$rc'"
echo "${SESSION}" > "${RUN_DIR}/SESSION_NAME"

tmux ls
tail -n 30 "${LOG_DIR}/launcher.log" || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
