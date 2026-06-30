#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${LATENTFM_XVERSE_RESPONSE_RUN_ROOT:-${ROOT}/runs/latentfm_xverse_response_repair_smoke_20260621}
OUT_ROOT=${LATENTFM_XVERSE_RESPONSE_OUT_ROOT:-${COUPLED}/output/latentfm_runs/xverse_response_repair_smoke_20260621}
LOG_ROOT=${LATENTFM_XVERSE_RESPONSE_LOG_ROOT:-${ROOT}/logs/latentfm_xverse_response_repair_smoke_20260621}
DATA_DIR=${ROOT}/dataset/latentfm_full/xverse
BIFLOW_DIR=${ROOT}/dataset/biFlow_data
CANONICAL_SPLIT=${BIFLOW_DIR}/split_seed42.json
TRAIN_SPLIT=${LATENTFM_XVERSE_RESPONSE_TRAIN_SPLIT:-${CANONICAL_SPLIT}}
EVAL_SPLIT=${LATENTFM_XVERSE_RESPONSE_EVAL_SPLIT:-${CANONICAL_SPLIT}}
TRAIN_PERT_MEANS=${LATENTFM_XVERSE_RESPONSE_TRAIN_PERT_MEANS:-}
ANCHOR_CKPT=${LATENTFM_XVERSE_RESPONSE_ANCHOR_CKPT:-${COUPLED}/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt}
GENE_CACHE=${ROOT}/pretrainckpt/genepert_cache/scgpt_embed_gene
ARTIFACT=${LATENTFM_XVERSE_RESPONSE_ARTIFACT:-${ROOT}/runs/latentfm_xverse_response_normalization_20260621/artifacts/xverse_trainonly_dataset_scale_pca32.npz}
RESPONSE_MODE=${LATENTFM_XVERSE_RESPONSE_MODE:-dataset_scale_pca}
POSTHOC_WAIT_SECONDS=${LATENTFM_XVERSE_RESPONSE_POSTHOC_WAIT_SECONDS:-1800}
GPU_HELPER=${ROOT}/ops/select_available_gpus.py
BOOTSTRAP_RUNNER=${ROOT}/ops/run_latentfm_posthoc_bootstrap_from_manifest_20260621.py
STABLECAPS_SUMMARIZER=${ROOT}/ops/summarize_latentfm_stablecaps_uncapped_readiness_20260622.py
PYTHON=${ROOT}/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=/data/cyx/software/miniconda3/envs/scdfm/bin/python
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/scripts" "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/reports"

for required in \
  "${DATA_DIR}/manifest.json" \
  "${TRAIN_SPLIT}" \
  "${EVAL_SPLIT}" \
  "${ANCHOR_CKPT}" \
  "${GENE_CACHE}/manifest.json" \
  "${ARTIFACT}" \
  "${GPU_HELPER}" \
  "${BOOTSTRAP_RUNNER}" \
  "${STABLECAPS_SUMMARIZER}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done
if [[ -n "${TRAIN_PERT_MEANS}" && ! -e "${TRAIN_PERT_MEANS}" ]]; then
  echo "Missing train-only pert means artifact: ${TRAIN_PERT_MEANS}" >&2
  exit 2
fi

"${PYTHON}" - "${ARTIFACT}" "${TRAIN_SPLIT}" "${RESPONSE_MODE}" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

artifact = Path(sys.argv[1])
split = Path(sys.argv[2])
mode = sys.argv[3]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

obj = np.load(str(artifact), allow_pickle=False)
meta = json.loads(str(obj["metadata_json"].item()))
reasons = []
if meta.get("fit_scope") != "train_only":
    reasons.append(f"fit_scope={meta.get('fit_scope')!r}")
if int(meta.get("emb_dim", -1)) != 384:
    reasons.append(f"emb_dim={meta.get('emb_dim')!r}")
if mode not in {"dataset_scale", "pca_subspace", "dataset_scale_pca"}:
    reasons.append(f"unsupported_response_mode={mode!r}")
expected = str(meta.get("split_sha256") or "")
actual = sha256_file(split)
if expected and expected != actual:
    reasons.append("split_sha256_mismatch")
forbidden = meta.get("forbidden_inputs_used") or {}
if any(bool(v) for v in forbidden.values()):
    reasons.append(f"forbidden_inputs_used={forbidden}")
if reasons:
    raise SystemExit("Invalid xverse response-normalizer artifact: " + "; ".join(reasons))
print(json.dumps({
    "artifact": str(artifact),
    "fit_scope": meta.get("fit_scope"),
    "emb_dim": meta.get("emb_dim"),
    "mode": mode,
    "n_train_residuals": meta.get("n_train_residuals"),
    "pca_cumulative_ev": meta.get("pca_cumulative_ev"),
}, indent=2))
PY

RUN_NAMES=()
WEIGHTS=()
ANCHOR_REPLAY_WEIGHTS=()
if [[ -n "${LATENTFM_XVERSE_RESPONSE_SPECS:-}" ]]; then
  for spec in ${LATENTFM_XVERSE_RESPONSE_SPECS}; do
    IFS=: read -r run_name resp_weight replay_weight <<<"${spec}"
    if [[ -z "${run_name}" || -z "${resp_weight}" || -z "${replay_weight}" ]]; then
      echo "Invalid LATENTFM_XVERSE_RESPONSE_SPECS entry: ${spec}; expected run:resp_weight:replay_weight" >&2
      exit 2
    fi
    RUN_NAMES+=("${run_name}")
    WEIGHTS+=("${resp_weight}")
    ANCHOR_REPLAY_WEIGHTS+=("${replay_weight}")
  done
else
  RUN_NAMES=(
    xverse_response_pca32_aux025_replay1_4k
    xverse_response_pca32_aux05_replay1_4k
  )
  WEIGHTS=(0.25 0.5)
  ANCHOR_REPLAY_WEIGHTS=(1.0 1.0)
fi

if [[ "${FORCE_XVERSE_RESPONSE_RERUN:-0}" != "1" ]]; then
  for run_name in "${RUN_NAMES[@]}"; do
    if [[ -e "${OUT_ROOT}/${run_name}" ]]; then
      echo "Output exists for ${run_name}; set FORCE_XVERSE_RESPONSE_RERUN=1 to relaunch" >&2
      exit 3
    fi
  done
fi

echo "[$(date '+%F %T %Z')] exact GPU status before xverse response-repair launch" | tee "${RUN_ROOT}/logs/gpu_launch_audit.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "${RUN_ROOT}/logs/gpu_launch_audit.log"

gpu_json="${RUN_ROOT}/logs/gpu_selection_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" "${GPU_HELPER}" \
  --samples 3 \
  --interval-seconds 10 \
  --util-threshold-pct 10 \
  --memory-threshold-mib 4096 \
  --max-jobs-per-gpu 4 \
  --need "${#RUN_NAMES[@]}" \
  --json-only \
  > "${gpu_json}" 2> "${RUN_ROOT}/logs/gpu_selection.stderr"

assignment_json="${RUN_ROOT}/logs/gpu_assignment_$(date +%Y%m%d_%H%M%S).json"
"${PYTHON}" - "${gpu_json}" "${assignment_json}" "${#RUN_NAMES[@]}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
need = int(sys.argv[3])
gpus = {int(g["index"]): g for g in payload.get("gpus", [])}
stable = [g for g in gpus.values() if g.get("stable_light")]
stable_count = len(stable)
active_user = set(int(x) for x in payload.get("active_user_gpus", []))
physical_budget = min(4, stable_count) if stable_count >= 5 else max(0, min(4, stable_count - 1))
candidate_order = [int(x) for x in payload.get("candidate_order", [])]
assigned = []
assigned_counts = {}
for _ in range(need):
    chosen = None
    for idx in candidate_order:
        gpu = gpus[idx]
        if not gpu.get("available"):
            continue
        slots = int(gpu.get("colocation_slots_free", 0)) - assigned_counts.get(idx, 0)
        if slots <= 0:
            continue
        if len(active_user | set(assigned) | {idx}) <= physical_budget:
            chosen = idx
            break
    if chosen is None:
        break
    assigned.append(chosen)
    assigned_counts[chosen] = assigned_counts.get(chosen, 0) + 1
system = payload.get("system") or {}
audit = {
    "status": "pass",
    "stable_light_gpu_count": stable_count,
    "physical_budget": physical_budget,
    "active_user_gpus": sorted(active_user),
    "assigned_gpus": assigned,
    "system": system,
    "gpu_selection_json": str(sys.argv[1]),
}
reasons = []
if len(assigned) < need:
    reasons.append(f"assigned {len(assigned)} jobs < requested {need}")
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

mapfile -t GPUS < <("${PYTHON}" - "${assignment_json}" <<'PY'
import json
import sys
from pathlib import Path
for gpu in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["assigned_gpus"]:
    print(gpu)
PY
)

manifest="${RUN_ROOT}/posthoc_manifest.json"
"${PYTHON}" - "${manifest}" "${ANCHOR_CKPT}" "${TRAIN_SPLIT}" "${EVAL_SPLIT}" "${DATA_DIR}" "${ARTIFACT}" "${RESPONSE_MODE}" "${TRAIN_PERT_MEANS}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "stage": "xverse_response_repair_smoke",
    "anchor_checkpoint": sys.argv[2],
    "train_split_file": sys.argv[3],
    "eval_split_file": sys.argv[4],
    "data_dir": sys.argv[5],
    "response_normalization_artifact": sys.argv[6],
    "response_normalization_mode": sys.argv[7],
    "train_pert_means_file": sys.argv[8],
    "launched_runs": [],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

for i in "${!RUN_NAMES[@]}"; do
  run_name="${RUN_NAMES[$i]}"
  weight="${WEIGHTS[$i]}"
  replay_weight="${ANCHOR_REPLAY_WEIGHTS[$i]}"
  gpu="${GPUS[$i]}"
  run_root="${RUN_ROOT}/${run_name}"
  log_root="${LOG_ROOT}/${run_name}"
  session="lfm_${run_name}"
  posthoc_session="latentfm_xverse_response_${run_name}_posthoc"
  mkdir -p "${run_root}/logs" "${run_root}/scripts" "${log_root}"

  run_script="${run_root}/scripts/run_${run_name}.sh"
  cat > "${run_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
echo "[\$(date '+%F %T %Z')] exact GPU status before xverse response posthoc"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
posthoc_gpu_json=${run_root}/logs/posthoc_gpu_selection_\$(date +%Y%m%d_%H%M%S).json
${PYTHON} ${GPU_HELPER} --samples 3 --interval-seconds 10 --util-threshold-pct 10 --memory-threshold-mib 4096 --max-jobs-per-gpu 4 --need 1 --json-only > "\${posthoc_gpu_json}" 2> ${run_root}/logs/posthoc_gpu_selection.stderr
posthoc_gpu="\$(${PYTHON} - "\${posthoc_gpu_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
chosen = payload.get("suggested_job_gpus") or []
print(chosen[0] if chosen else "")
PY
)"
if [[ -z "\${posthoc_gpu}" ]]; then
  echo "No GPU selected for xverse response posthoc; see \${posthoc_gpu_json}" >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES="\${posthoc_gpu}"
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
  --split-file ${TRAIN_SPLIT} \\
  --save-dir ${OUT_ROOT}/${run_name} \\
  --log-file train.log \\
  --latent-backbone xverse \\
  --model-type control_mlp \\
  --emb-dim 384 \\
  ${TRAIN_PERT_MEANS:+--pert-means-file ${TRAIN_PERT_MEANS} }\\
  --gpu 0 \\
  --batch-size 64 \\
  --seed 42 \\
  --grad-accum-steps 1 \\
  --min-cells 16 \\
  --scale-noise 0.01 \\
  --lr 5e-5 \\
  --weight-decay 1e-4 \\
  --warmup-steps 300 \\
  --total-steps 4000 \\
  --lr-decay-steps 4000 \\
  --print-every 200 \\
  --eval-every 2000 \\
  --eval-max-conditions 256 \\
  --eval-max-conditions-per-dataset 12 \\
  --eval-max-mse-cells 1024 \\
  --eval-max-mmd-cells 1024 \\
  --eval-max-chunk 256 \\
  --selection-metric pearson_pert_minus_mmd \\
  --selection-mmd-lambda 0.5 \\
  --ot-method torch_sinkhorn \\
  --ot-sinkhorn-reg 0.05 \\
  --ot-sinkhorn-iter 30 \\
  --use-mmd \\
  --gamma 0.03 \\
  --gamma-warmup-start 500 \\
  --gamma-warmup-end 1500 \\
  --mmd-every 4 \\
  --mmd-estimator unbiased \\
  --composition-delta-loss-weight 0.06 \\
  --composition-delta-loss-warmup-start 500 \\
  --composition-delta-loss-warmup-end 1500 \\
  --endpoint-delta-loss-weight 5.0 \\
  --endpoint-delta-loss-warmup-start 500 \\
  --endpoint-delta-loss-warmup-end 1500 \\
  --response-geometry-loss-weight ${weight} \\
  --response-geometry-loss-warmup-start 500 \\
  --response-geometry-loss-warmup-end 1500 \\
  --response-normalization-mode ${RESPONSE_MODE} \\
  --response-normalization-artifact ${ARTIFACT} \\
  --response-normalization-strict-split \\
  --response-geometry-condition-filter all \\
  --anchor-replay-loss-weight ${replay_weight} \\
  --anchor-replay-loss-warmup-start 500 \\
  --anchor-replay-loss-warmup-end 1500 \\
  --anchor-replay-condition-filter non_gene_multi \\
  --anchor-replay-checkpoint ${ANCHOR_CKPT} \\
  --init-checkpoint ${ANCHOR_CKPT} \\
  --use-ema \\
  --ema-update-after 500 \\
  --ema-decay 0.999 \\
  --amp-dtype bf16 \\
  --use-pert-condition \\
  --pert-gene-emb-cache-dir ${GENE_CACHE} \\
  --pert-condition-embedding-source scgpt_embed_gene \\
  --pert-pool-aggregations sum mean max min \\
  --pert-pool-scale-init 0.5 1.0 1.0 1.0 \\
  --pert-pool-fusion-mode sum \\
  --pert-gene-projector-hidden 1024 \\
  --pert-chem-enabled \\
  --pert-chem-emb-dim 512 \\
  --pert-chem-projector-hidden 1024 \\
  --chem-fallback-embed-dim 512 \\
  --pert-to-c-init-mode xavier_small \\
  --use-pert-in-fusion \\
  --patience 4
EOF
  chmod +x "${run_script}"

  rm -f "${run_root}/${run_name}.EXIT_CODE" "${run_root}/${run_name}.FINISHED"
  tmux new -d -s "${session}" \
    "bash -lc 'bash ${run_script} > ${log_root}/${run_name}.log 2>&1; rc=\$?; echo \$rc > ${run_root}/${run_name}.EXIT_CODE; date \"+%F %T %Z\" > ${run_root}/${run_name}.FINISHED; exit \$rc'"
  date '+%F %T %Z' > "${run_root}/${run_name}.STARTED"

  "${PYTHON}" - "${manifest}" "${run_name}" "${OUT_ROOT}/${run_name}/best.pt" "${weight}" "${replay_weight}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload.setdefault("launched_runs", []).append({
    "run_name": sys.argv[2],
    "candidate_checkpoint": sys.argv[3],
    "response_geometry_loss_weight": float(sys.argv[4]),
    "anchor_replay_loss_weight": float(sys.argv[5]),
    "train_split_file": payload["train_split_file"],
    "eval_split_file": payload["eval_split_file"],
    "anchor_checkpoint": payload["anchor_checkpoint"],
})
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

  posthoc_script="${run_root}/scripts/posthoc_${run_name}.sh"
  cat > "${posthoc_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
while [[ ! -f ${run_root}/${run_name}.EXIT_CODE ]]; do
  sleep ${POSTHOC_WAIT_SECONDS}
done
code="\$(cat ${run_root}/${run_name}.EXIT_CODE)"
if [[ "\${code}" != "0" ]]; then
  echo "training failed for ${run_name}; skip posthoc" >&2
  exit "\${code}"
fi
source ${ROOT}/init-scdfm.sh >/dev/null
cd ${COUPLED}
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=${COUPLED}:\${PYTHONPATH:-}
out_dir=${run_root}/posthoc_eval_stablecaps
mkdir -p "\${out_dir}"
base_split="\${out_dir}/split_group_eval_anchor_ode20_stablecaps.json"
base_family="\${out_dir}/condition_family_eval_anchor_ode20_stablecaps.json"
cand_split="\${out_dir}/split_group_eval_candidate_ode20_stablecaps.json"
cand_family="\${out_dir}/condition_family_eval_candidate_ode20_stablecaps.json"
common=(--data-dir ${DATA_DIR} --biflow-dir ${BIFLOW_DIR} --split-file ${EVAL_SPLIT} --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 256 --eval-max-conditions-per-dataset 12 --eval-max-mse-cells 1024 --eval-max-mmd-cells 1024)
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${ANCHOR_CKPT} --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${base_split}" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${ANCHOR_CKPT} --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${base_family}" "\${common[@]}"
${PYTHON} -m model.latent.eval_split_groups --checkpoint ${OUT_ROOT}/${run_name}/best.pt --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${cand_split}" "\${common[@]}"
${PYTHON} -m model.latent.eval_condition_families --checkpoint ${OUT_ROOT}/${run_name}/best.pt --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${cand_family}" "\${common[@]}"
${PYTHON} - ${run_root}/posthoc_manifest.json "\${base_split}" "\${base_family}" "\${cand_split}" "\${cand_family}" <<PY
import json
import sys
from pathlib import Path
payload = {
    "run_name": "${run_name}",
    "anchor_checkpoint": "${ANCHOR_CKPT}",
    "candidate_checkpoint": "${OUT_ROOT}/${run_name}/best.pt",
    "train_split_file": "${TRAIN_SPLIT}",
    "eval_split_file": "${EVAL_SPLIT}",
    "data_dir": "${DATA_DIR}",
    "response_normalization_artifact": "${ARTIFACT}",
    "response_normalization_mode": "${RESPONSE_MODE}",
    "train_pert_means_file": "${TRAIN_PERT_MEANS}",
    "baseline_split_json": sys.argv[2],
    "baseline_family_json": sys.argv[3],
    "run_split_json": sys.argv[4],
    "run_family_json": sys.argv[5],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
${PYTHON} ${BOOTSTRAP_RUNNER} --manifest ${run_root}/posthoc_manifest.json --out-dir ${ROOT}/reports/latentfm_xverse_response_repair_${run_name}_bootstrap_20260621 --n-boot 2000 --seed 42 --split-groups test test_single test_multi test_multi_unseen2 --family-groups family_gene family_drug structure_multi
${PYTHON} ${STABLECAPS_SUMMARIZER} --bootstrap-index ${ROOT}/reports/latentfm_xverse_response_repair_${run_name}_bootstrap_20260621/bootstrap_index.json --label ${run_name} --out-json ${ROOT}/reports/latentfm_xverse_response_repair_${run_name}_stablecaps_decision_20260622.json --out-md ${ROOT}/reports/LATENTFM_XVERSE_RESPONSE_REPAIR_${run_name}_STABLECAPS_DECISION_20260622.md
EOF
  chmod +x "${posthoc_script}"
  tmux new -d -s "${posthoc_session}" \
    "bash -lc 'bash ${posthoc_script} > ${run_root}/logs/posthoc.log 2>&1; rc=\$?; echo \$rc > ${run_root}/POSTHOC_EXIT_CODE; date \"+%F %T %Z\" > ${run_root}/POSTHOC_FINISHED; exit \$rc'"

  cat > "${run_root}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_response_repair_smoke_20260621/${run_name}

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_xverse_response_repair_smoke_20260621.sh
\`\`\`

## Runtime classification

Long LatentFM training task. Use 30-minute cadence for checks.

## Start time

$(cat "${run_root}/${run_name}.STARTED")

## tmux / GPU

* training: \`${session}\`, physical GPU${gpu}
* posthoc watcher: \`${posthoc_session}\`

## Log path

\`${log_root}/${run_name}.log\`

## Expected outputs

* \`${OUT_ROOT}/${run_name}/best.pt\`
* \`${OUT_ROOT}/${run_name}/iid_eval_results.json\`
* \`${run_root}/posthoc_manifest.json\`
* \`${ROOT}/reports/latentfm_xverse_response_repair_${run_name}_bootstrap_20260621/bootstrap_index.json\`

## Current status

Started training and posthoc watcher.

## Notes

xverse response-normalized repair smoke. Warm-start and anchor replay both use:
\`${ANCHOR_CKPT}\`.

response_geometry_loss_weight=${weight}; anchor_replay_loss_weight=${replay_weight}.
Artifact:
\`${ARTIFACT}\`.
Response mode: \`${RESPONSE_MODE}\`.
Train split: \`${TRAIN_SPLIT}\`.
Eval split: \`${EVAL_SPLIT}\`.
Train pert means: \`${TRAIN_PERT_MEANS:-default}\`.

Promotion requires capped paired bootstrap vs xverse anchor to improve
\`test_multi_unseen2\` pp without MMD hard harm and without aggregate/family/single/drug
regression. Passing capped smoke is not a paper claim; it only permits uncapped
posthoc and seed/anchor robustness.
EOF
done

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_xverse_response_repair_smoke_20260621

Launched at $(date '+%F %T %Z').

## Command

\`\`\`bash
bash ${ROOT}/ops/launch_latentfm_xverse_response_repair_smoke_20260621.sh
\`\`\`

## Runtime classification

Long LatentFM training sweep. Use 30-minute cadence for checks.

## GPU assignment audit

\`${assignment_json}\`

## Runs

$(for j in "${!RUN_NAMES[@]}"; do
  printf '* `%s`: response weight `%s`, anchor replay `%s`, RUN_STATUS `%s/%s/RUN_STATUS.md`\n' "${RUN_NAMES[$j]}" "${WEIGHTS[$j]}" "${ANCHOR_REPLAY_WEIGHTS[$j]}" "${RUN_ROOT}" "${RUN_NAMES[$j]}"
done)

## Current status

Started training jobs and low-frequency posthoc watchers.

## Notes

This sweep uses the train-only xverse response-normalizer artifact and the
xverse 8k seed42 anchor. Do not check more often than every 30 minutes unless
exit/bootstrap markers appear naturally.

Posthoc watcher sleep interval: \`${POSTHOC_WAIT_SECONDS}\` seconds.
EOF

echo "Launched xverse response-repair smokes"
echo "RUN_STATUS: ${RUN_ROOT}/RUN_STATUS.md"
