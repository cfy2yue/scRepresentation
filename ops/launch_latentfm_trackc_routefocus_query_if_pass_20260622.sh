#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

RUN_NAME=${LATENTFM_TRACKC_QUERY_RUN_NAME:-xverse_trackc_routefocus_condprior_w05_replay1_2k_seed42}
LABEL=${LATENTFM_TRACKC_QUERY_LABEL:-latentfm_trackc_routefocus_query_once_20260622}
RUN_ROOT=${ROOT}/runs/${LABEL}
LOG_DIR=${RUN_ROOT}/logs
OUT_DIR=${ROOT}/reports/${LABEL}
EVAL_DIR=${OUT_DIR}/eval
FULL_V2_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2.json
TRAINSELECT_SPLIT=${ROOT}/dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
ANCHOR_CKPT=${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt
CANDIDATE_CKPT=${LATENTFM_TRACKC_QUERY_CANDIDATE_CKPT:-${COUPLED}/output/latentfm_runs/xverse_trackc_routefocused_distill_20260622/${RUN_NAME}/best.pt}
SMOKE_DECISION=${LATENTFM_TRACKC_QUERY_SMOKE_DECISION:-${ROOT}/reports/latentfm_trackc_routed_distill_smoke_decision_${RUN_NAME}.json}
UNCAPPED_DECISION=${LATENTFM_TRACKC_QUERY_UNCAPPED_DECISION:-${ROOT}/reports/latentfm_trackc_routefocus_uncapped_noharm_decision_20260622.json}
SUMMARIZER=${ROOT}/ops/summarize_latentfm_trackc_routefocus_query_once_20260622.py
QUERY_DECISION_JSON=${LATENTFM_TRACKC_QUERY_DECISION_JSON:-${ROOT}/reports/latentfm_trackc_routefocus_query_once_decision_20260622.json}
QUERY_DECISION_MD=${LATENTFM_TRACKC_QUERY_DECISION_MD:-${ROOT}/reports/LATENTFM_TRACKC_ROUTEFOCUS_QUERY_ONCE_DECISION_20260622.md}
QUERY_BOOT_DIR=${LATENTFM_TRACKC_QUERY_BOOT_DIR:-${ROOT}/reports/latentfm_trackc_routefocus_query_once_bootstrap_20260622}
QUERY_REPORT_TITLE=${LATENTFM_TRACKC_QUERY_REPORT_TITLE:-LatentFM Track C Route-Focused One-Shot Query Decision}
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
CPU_THREADS=${LATENTFM_CPU_THREADS:-48}
QUERY_EVAL_MAX_MSE_CELLS=${QUERY_EVAL_MAX_MSE_CELLS:-2048}
QUERY_EVAL_MAX_MMD_CELLS=${QUERY_EVAL_MAX_MMD_CELLS:-2048}

if ! [[ "${CPU_THREADS}" =~ ^[0-9]+$ ]]; then
  echo "LATENTFM_CPU_THREADS must be a positive integer, got '${CPU_THREADS}'" >&2
  exit 2
fi
if (( CPU_THREADS < 1 )); then
  echo "LATENTFM_CPU_THREADS must be >= 1" >&2
  exit 2
fi
if (( CPU_THREADS > 48 )); then
  CPU_THREADS=48
fi

for required in \
  "${PYTHON}" \
  "${SUMMARIZER}" \
  "${GPU_HELPER}" \
  "${ANCHOR_CKPT}" \
  "${CANDIDATE_CKPT}" \
  "${SMOKE_DECISION}" \
  "${UNCAPPED_DECISION}" \
  "${FULL_V2_SPLIT}" \
  "${TRAINSELECT_SPLIT}" \
  "${DATA_DIR}/manifest.json"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required query-gated artifact: ${required}" >&2
    exit 2
  fi
done

status_json="$("${PYTHON}" - "${SMOKE_DECISION}" "${UNCAPPED_DECISION}" <<'PY'
import json
import sys
from pathlib import Path

smoke = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
uncapped = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
print(json.dumps({
    "smoke": (smoke.get("decision") or {}).get("status", ""),
    "uncapped": (uncapped.get("decision") or {}).get("status", ""),
}))
PY
)"
smoke_status="$("${PYTHON}" - "${status_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["smoke"])
PY
)"
uncapped_status="$("${PYTHON}" - "${status_json}" <<'PY'
import json, sys
print(json.loads(sys.argv[1])["uncapped"])
PY
)"

if [[ "${smoke_status}" != "trackc_smoke_support_pass_needs_uncapped_noharm_before_query" ]]; then
  echo "Refusing query eval: smoke decision status is '${smoke_status}', not pass." >&2
  exit 5
fi
if [[ "${uncapped_status}" != "trackc_uncapped_canonical_noharm_pass_query_allowed_once" ]]; then
  echo "Refusing query eval: uncapped no-harm status is '${uncapped_status}', not pass." >&2
  exit 6
fi

if [[ -e "${RUN_ROOT}" || -e "${OUT_DIR}" ]]; then
  echo "Refusing one-shot query eval because run/output path already exists: ${RUN_ROOT} or ${OUT_DIR}" >&2
  exit 7
fi
if [[ -e "${QUERY_DECISION_JSON}" || -e "${QUERY_DECISION_MD}" || -e "${QUERY_BOOT_DIR}" ]]; then
  echo "Refusing one-shot query eval because final query decision already exists." >&2
  exit 8
fi
if tmux has-session -t "${LABEL}" 2>/dev/null; then
  echo "tmux session already exists: ${LABEL}" >&2
  exit 9
fi

"${PYTHON}" - "${FULL_V2_SPLIT}" "${TRAINSELECT_SPLIT}" <<'PY'
import json
import sys
from pathlib import Path

full = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
trainselect = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
query = set()
support = set()
train_multi = set()
for ds, obj in full.items():
    if not isinstance(obj, dict):
        continue
    for cond in obj.get("query_multi") or []:
        query.add((str(ds), str(cond)))
    for cond in obj.get("support_val_multi") or []:
        support.add((str(ds), str(cond)))
    for cond in obj.get("train_multi") or []:
        train_multi.add((str(ds), str(cond)))
if not query:
    raise SystemExit("full v2 split contains no query_multi conditions")
if query & support:
    raise SystemExit("query/support overlap is non-empty")
if query & train_multi:
    raise SystemExit("query/train_multi overlap is non-empty")
trainselect_test = {
    (str(ds), str(cond))
    for ds, obj in trainselect.items()
    if isinstance(obj, dict)
    for cond in (obj.get("test") or [])
}
if support != trainselect_test:
    raise SystemExit("full v2 support_val_multi does not match trainselect test support-val")
print(json.dumps({
    "query_multi": len(query),
    "support_val_multi": len(support),
    "train_multi": len(train_multi),
    "status": "split_guard_pass",
}, sort_keys=True))
PY

PRELAUNCH_ROOT=${ROOT}/runs/${LABEL}_prelaunch_$(date +%Y%m%d_%H%M%S)
PRELAUNCH_LOG_DIR=${PRELAUNCH_ROOT}/logs
mkdir -p "${PRELAUNCH_LOG_DIR}"

echo "[$(date '+%F %T %Z')] exact GPU status before Track C one-shot query eval" | tee "${PRELAUNCH_LOG_DIR}/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${PRELAUNCH_LOG_DIR}/gpu_launch_audit.log"

gpu_json="${PRELAUNCH_LOG_DIR}/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-jobs-per-gpu 4 \
  --need 1 \
  --json-only \
  > "${gpu_json}" 2> "${PRELAUNCH_LOG_DIR}/gpu_selection.stderr"

assignment_json="${PRELAUNCH_LOG_DIR}/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
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
if float(system.get("mem_available_gib") or 0.0) < 96.0:
    reasons.append(f"MemAvailable {float(system.get('mem_available_gib') or 0.0):.1f} GiB < 96.0 GiB")
if float(system.get("load1_per_cpu") or 0.0) > 2.0:
    reasons.append(f"load1_per_cpu {float(system.get('load1_per_cpu') or 0.0):.3f} > 2.000")
if reasons:
    audit["status"] = "fail"
    audit["reasons"] = reasons
Path(sys.argv[2]).write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
raise SystemExit(0 if audit["status"] == "pass" else 10)
PY

GPU="$("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["chosen_gpu"])
PY
)"

mkdir -p "${RUN_ROOT}" "${EVAL_DIR}"
mv "${PRELAUNCH_LOG_DIR}" "${LOG_DIR}"
rmdir "${PRELAUNCH_ROOT}" 2>/dev/null || true

run_script="${RUN_ROOT}/run_query_eval.sh"
cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=${CPU_THREADS}
export MKL_NUM_THREADS=${CPU_THREADS}
export OPENBLAS_NUM_THREADS=${CPU_THREADS}
export NUMEXPR_NUM_THREADS=${CPU_THREADS}
export BLIS_NUM_THREADS=${CPU_THREADS}
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene

common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${FULL_V2_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells ${QUERY_EVAL_MAX_MSE_CELLS} --eval-max-mmd-cells ${QUERY_EVAL_MAX_MMD_CELLS})
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups query_multi query_multi_seen query_multi_unseen1 query_multi_unseen2 test_multi --out "${EVAL_DIR}/query_anchor_split_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${CANDIDATE_CKPT} --groups query_multi query_multi_seen query_multi_unseen1 query_multi_unseen2 test_multi --out "${EVAL_DIR}/query_candidate_split_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_multi --out "${EVAL_DIR}/query_anchor_family_ode20.json" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${CANDIDATE_CKPT} --groups test_multi --out "${EVAL_DIR}/query_candidate_family_ode20.json" "\${common[@]}"
${PYTHON} ${SUMMARIZER} \
  --eval-dir "${EVAL_DIR}" \
  --smoke-decision-json "${SMOKE_DECISION}" \
  --uncapped-decision-json "${UNCAPPED_DECISION}" \
  --out-json "${QUERY_DECISION_JSON}" \
  --out-md "${QUERY_DECISION_MD}" \
  --boot-dir "${QUERY_BOOT_DIR}" \
  --report-title "${QUERY_REPORT_TITLE}" \
  --python ${PYTHON}
EOF
chmod +x "${run_script}"

tmux new -d -s "${LABEL}" \
  "bash -lc 'bash ${run_script} > ${LOG_DIR}/query_eval.log 2>&1; rc=\$?; echo \$rc > ${RUN_ROOT}/EXIT_CODE; date \"+%F %T %Z\" > ${RUN_ROOT}/FINISHED; exit \$rc'"
date '+%F %T %Z' > "${RUN_ROOT}/STARTED"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: ${LABEL}

## Command

\`\`\`bash
LATENTFM_CPU_THREADS=${CPU_THREADS} QUERY_EVAL_MAX_MSE_CELLS=${QUERY_EVAL_MAX_MSE_CELLS} QUERY_EVAL_MAX_MMD_CELLS=${QUERY_EVAL_MAX_MMD_CELLS} bash ${ROOT}/ops/launch_latentfm_trackc_routefocus_query_if_pass_20260622.sh
\`\`\`

## Runtime classification

Long GPU posthoc evaluation. Check at most every 30 minutes.

## Start time

$(cat "${RUN_ROOT}/STARTED")

## PID / tmux / scheduler ID

tmux session: \`${LABEL}\`, physical GPU${GPU}

## Log path

\`${LOG_DIR}/query_eval.log\`

## Expected outputs

* \`${EVAL_DIR}/query_anchor_split_ode20.json\`
* \`${EVAL_DIR}/query_candidate_split_ode20.json\`
* \`${QUERY_DECISION_MD}\`

## How to check manually

\`\`\`bash
tmux ls | grep '${LABEL}' || true
tail -n 50 ${LOG_DIR}/query_eval.log
cat ${RUN_ROOT}/EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Final one-shot Track C held-out query diagnostic for a frozen checkpoint only.
This launcher refuses unless both the smoke support/canonical gate
and the uncapped canonical no-harm gate pass. The query result must not be used
for route/checkpoint selection.
EOF

echo "Launched one-shot Track C query eval ${LABEL} on physical GPU${GPU}"
echo "RUN_STATUS: ${RUN_ROOT}/RUN_STATUS.md"
