#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
if [[ ! -x "${PY}" ]]; then
  PY=/data/cyx/1030/software/miniconda3/envs/scdfm/bin/python
fi

SRC_RUN_ROOT=${ROOT}/runs/latentfm_xverse_trackc_noharm_adapter_parallel_c_20260622
SRC_OUT_ROOT=${COUPLED}/output/latentfm_runs/xverse_trackc_noharm_adapter_parallel_c_20260622
RUN_ROOT=${ROOT}/runs/latentfm_trackc_pairwise_latest_posthoc_20260622
REPORT_DIR=${ROOT}/reports
AUDIT_DIR=${RUN_ROOT}/audit
LOG_ROOT=${RUN_ROOT}/logs
mkdir -p "${AUDIT_DIR}" "${LOG_ROOT}"

RUNS=(
  xverse_trackc_noharm_pc_ep050_replay2_all_2k_seed42
  xverse_trackc_noharm_pc_ep050_replay4_nongm_2k_seed42
  xverse_trackc_noharm_pc_ep050del_replay2_all_2k_seed42
  xverse_trackc_noharm_pc_ep100_replay2_all_2k_seed42
  xverse_trackc_noharm_pc_ep100del_replay4_all_2k_seed42
  xverse_trackc_noharm_pc_ep100del_replay4_nongm_2k_seed42
)

for run in "${RUNS[@]}"; do
  for path in \
    "${SRC_RUN_ROOT}/${run}/posthoc_eval/support_anchor_split_ode20.json" \
    "${SRC_RUN_ROOT}/${run}/posthoc_eval/support_anchor_family_ode20.json" \
    "${SRC_RUN_ROOT}/${run}/posthoc_eval/canonical_anchor_split_ode20_stablecaps.json" \
    "${SRC_RUN_ROOT}/${run}/posthoc_eval/canonical_anchor_family_ode20_stablecaps.json" \
    "${SRC_OUT_ROOT}/${run}/latest.pt"; do
    if [[ ! -s "${path}" ]]; then
      echo "missing required input: ${path}" >&2
      exit 2
    fi
  done
done

audit_stamp=$(date +%Y%m%d_%H%M%S)
for sample in 1 2 3; do
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    > "${AUDIT_DIR}/gpu_sample_${audit_stamp}_${sample}.csv"
  if [[ "${sample}" != "3" ]]; then
    sleep 10
  fi
done
free -h > "${AUDIT_DIR}/free_${audit_stamp}.txt"
df -h "${ROOT}" > "${AUDIT_DIR}/df_${audit_stamp}.txt"
uptime > "${AUDIT_DIR}/uptime_${audit_stamp}.txt"

GPU_SELECTION_JSON=${AUDIT_DIR}/gpu_selection_${audit_stamp}.json
"${PY}" - "${AUDIT_DIR}" "${audit_stamp}" "${GPU_SELECTION_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

audit_dir = Path(sys.argv[1])
stamp = sys.argv[2]
out = Path(sys.argv[3])

samples = []
for idx in range(1, 4):
    rows = []
    with (audit_dir / f"gpu_sample_{stamp}_{idx}.csv").open(newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            rows.append(
                {
                    "index": int(row[0].strip()),
                    "memory_mib": int(row[1].strip()),
                    "util_pct": int(row[2].strip()),
                }
            )
    samples.append(rows)

by_gpu: dict[int, list[dict[str, int]]] = {}
for rows in samples:
    for row in rows:
        by_gpu.setdefault(row["index"], []).append(row)

stable_empty = []
for gpu, rows in sorted(by_gpu.items()):
    if len(rows) == 3 and all(r["memory_mib"] < 4096 and r["util_pct"] < 10 for r in rows):
        stable_empty.append(gpu)

if len(stable_empty) >= 5:
    allowed = min(4, len(stable_empty))
else:
    allowed = max(0, len(stable_empty) - 1)

selected = stable_empty[: min(3, allowed)]
payload = {
    "samples": samples,
    "stable_empty": stable_empty,
    "allowed_by_policy": allowed,
    "selected": selected,
    "policy": "3 samples, memory.used < 4096 MiB and util < 10%; if fewer than 5 empty, leave at least 1 empty; this launcher uses at most 3 GPUs",
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
if len(selected) < 3:
    raise SystemExit(f"need 3 selected GPUs for this parallel posthoc gate, got {selected}")
print(json.dumps(payload, indent=2))
PY

mapfile -t SELECTED_GPUS < <("${PY}" - "${GPU_SELECTION_JSON}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
for gpu in payload["selected"]:
    print(gpu)
PY
)

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# Run Status: latentfm_trackc_pairwise_latest_posthoc_20260622

## Command

\`\`\`bash
bash /data/cyx/1030/scLatent/ops/launch_latentfm_trackc_pairwise_latest_posthoc_20260622.sh
\`\`\`

## Runtime classification

Long task.

## Start time

$(date "+%Y-%m-%d %H:%M:%S %Z")

## PID / tmux / scheduler ID

Block launcher; per-run tmux sessions:

$(for run in "${RUNS[@]}"; do echo "* \`trackc_latest_${run}\`"; done)

Selected physical GPUs: \`${SELECTED_GPUS[*]}\`

## Log path

\`/data/cyx/1030/scLatent/runs/latentfm_trackc_pairwise_latest_posthoc_20260622/logs/<run>.latest_posthoc.log\`

## Expected outputs

* \`/data/cyx/1030/scLatent/reports/LATENTFM_TRACKC_PAIRWISE_LATEST_DECISION_<run>.md\`
* \`/data/cyx/1030/scLatent/reports/latentfm_trackc_pairwise_latest_decision_<run>.json\`
* \`/data/cyx/1030/scLatent/runs/latentfm_trackc_pairwise_latest_posthoc_20260622/<run>/posthoc_eval/support_candidate_split_ode20.json\`
* \`/data/cyx/1030/scLatent/runs/latentfm_trackc_pairwise_latest_posthoc_20260622/<run>/posthoc_eval/canonical_candidate_split_ode20_stablecaps.json\`

## How to check manually

\`\`\`bash
tmux ls
tail -n 50 /data/cyx/1030/scLatent/runs/latentfm_trackc_pairwise_latest_posthoc_20260622/logs/<run>.latest_posthoc.log
cat /data/cyx/1030/scLatent/runs/latentfm_trackc_pairwise_latest_posthoc_20260622/<run>/<run>.LATEST_POSTHOC_EXIT_CODE 2>/dev/null || echo "still running"
nvidia-smi
\`\`\`

## Current status

Started.

## Notes

Hypothesis: the C-block pairwise-condition endpoint partial positives may be
limited by support checkpoint selection. This posthoc-only gate evaluates each
run's \`latest.pt\` at step 2000 against the same support trainselect and
canonical no-harm protocol used for \`best.pt\`. It does not read held-out
Track C query outputs and does not authorize E-block training by itself.

Resource plan: 6 detached posthoc jobs, assigned round-robin across 3 selected
GPUs, \`OMP/MKL/OPENBLAS/NUMEXPR/BLIS_NUM_THREADS=4\` per job for a total of
about 24 LatentFM CPU threads.

Gate: same Track C support/canonical smoke decision rules. If latest fails the
support gate or canonical no-harm gate, close the checkpoint-selection branch.
If latest passes, it only authorizes protocol review for uncapped canonical
no-harm; it is not a multi success claim and still does not authorize query.
EOF

for i in "${!RUNS[@]}"; do
  run="${RUNS[$i]}"
  gpu="${SELECTED_GPUS[$((i % ${#SELECTED_GPUS[@]}))]}"
  run_dir="${RUN_ROOT}/${run}"
  out_eval="${run_dir}/posthoc_eval"
  script="${run_dir}/run_latest_posthoc.sh"
  mkdir -p "${out_eval}" "${run_dir}/scripts"
  cp "${SRC_RUN_ROOT}/${run}/posthoc_eval/support_anchor_split_ode20.json" "${out_eval}/support_anchor_split_ode20.json"
  cp "${SRC_RUN_ROOT}/${run}/posthoc_eval/support_anchor_family_ode20.json" "${out_eval}/support_anchor_family_ode20.json"
  cp "${SRC_RUN_ROOT}/${run}/posthoc_eval/canonical_anchor_split_ode20_stablecaps.json" "${out_eval}/canonical_anchor_split_ode20_stablecaps.json"
  cp "${SRC_RUN_ROOT}/${run}/posthoc_eval/canonical_anchor_family_ode20_stablecaps.json" "${out_eval}/canonical_anchor_family_ode20_stablecaps.json"
  cat > "${script}" <<EOS
#!/usr/bin/env bash
set -euo pipefail
source /data/cyx/1030/scLatent/init-scdfm.sh >/dev/null
cd /data/cyx/1030/scLatent/CoupledFM
export CUDA_VISIBLE_DEVICES=${gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export BLIS_NUM_THREADS=4
export PYTHONPATH=/data/cyx/1030/scLatent/CoupledFM:\${PYTHONPATH:-}
export PERT_EMBED_SOURCE=scgpt_embed_gene
out_eval=${out_eval}
common_trainselect=(--data-dir /data/cyx/1030/dataset/latentfm_full/xverse --biflow-dir /data/cyx/1030/dataset/biFlow_data --split-file /data/cyx/1030/dataset/biFlow_data/split_seed42_multi_support_v2_route_datasets_trainselect.json --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 0 --eval-max-conditions-per-dataset 0 --eval-max-mse-cells 1024 --eval-max-mmd-cells 512)
common_canonical=(--data-dir /data/cyx/1030/dataset/latentfm_full/xverse --biflow-dir /data/cyx/1030/dataset/biFlow_data --split-file /data/cyx/1030/dataset/biFlow_data/split_seed42.json --gpu 0 --ode-steps 20 --max-chunk 512 --eval-max-conditions 256 --eval-max-conditions-per-dataset 12 --eval-max-mse-cells 1024 --eval-max-mmd-cells 512)
${PY} -m model.latent.eval_split_groups --checkpoint ${SRC_OUT_ROOT}/${run}/latest.pt --groups test test_multi --out "\${out_eval}/support_candidate_split_ode20.json" "\${common_trainselect[@]}"
${PY} -m model.latent.eval_condition_families --checkpoint ${SRC_OUT_ROOT}/${run}/latest.pt --groups test_all family_gene structure_multi test_multi --out "\${out_eval}/support_candidate_family_ode20.json" "\${common_trainselect[@]}"
${PY} -m model.latent.eval_split_groups --checkpoint ${SRC_OUT_ROOT}/${run}/latest.pt --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${out_eval}/canonical_candidate_split_ode20_stablecaps.json" "\${common_canonical[@]}"
${PY} -m model.latent.eval_condition_families --checkpoint ${SRC_OUT_ROOT}/${run}/latest.pt --groups test_all family_gene family_drug structure_single structure_multi test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 --out "\${out_eval}/canonical_candidate_family_ode20_stablecaps.json" "\${common_canonical[@]}"
${PY} /data/cyx/1030/scLatent/ops/summarize_latentfm_trackc_routed_distill_smoke_20260622.py --run-root ${run_dir} --out-json ${REPORT_DIR}/latentfm_trackc_pairwise_latest_decision_${run}.json --out-md ${REPORT_DIR}/LATENTFM_TRACKC_PAIRWISE_LATEST_DECISION_${run}.md --n-boot 2000 --seed 42 --python ${PY}
EOS
  chmod +x "${script}"
  tmux new -d -s "trackc_latest_${run}" \
    "bash ${script} > ${LOG_ROOT}/${run}.latest_posthoc.log 2>&1; rc=\$?; echo \$rc > ${run_dir}/${run}.LATEST_POSTHOC_EXIT_CODE; date > ${run_dir}/${run}.LATEST_POSTHOC_FINISHED; exit \$rc"
done

echo "launched ${#RUNS[@]} latest-checkpoint posthoc jobs"
echo "run root: ${RUN_ROOT}"
echo "gpu selection: ${GPU_SELECTION_JSON}"
