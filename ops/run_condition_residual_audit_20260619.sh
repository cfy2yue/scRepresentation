#!/usr/bin/env bash
set -u

PY=/data/cyx/software/miniconda3/envs/scdfm/bin/python
ROOT=/data/cyx/1030/scLatent
COUPLED=${ROOT}/CoupledFM
RUN_ROOT=${ROOT}/runs/latentfm_condition_residual_audit_20260619
REPORT_ROOT=${ROOT}/reports/latentfm_condition_residual_audit_20260619/main

mkdir -p "${RUN_ROOT}/logs" "${REPORT_ROOT}"

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition Residual Audit

Started: $(date '+%F %T %Z')
Status: running
Purpose: capped per-condition residual/ranking audit for existing checkpoints before launching more LatentFM branches.
Scope: primary scFoundation, relational rel002/rel005, strong comp020, stack comp006.
EOF

cd "${COUPLED}" || exit 1

COMMON=(
  --ode-steps 10
  --max-chunk 256
  --eval-max-cells 256
  --max-conditions-per-group 8
  --skip-mmd
  --groups test test_single test_multi test_multi_seen test_multi_unseen1 test_multi_unseen2 family_gene family_drug
)

run_one() {
  local label="$1"
  local gpu="$2"
  local ckpt="$3"
  local log="${RUN_ROOT}/logs/${label}.log"
  local exit_file="${RUN_ROOT}/logs/${label}.exit"
  rm -f "${exit_file}"
  {
    echo "[$(date +%F_%T)] start ${label} gpu=${gpu} ckpt=${ckpt}"
    "${PY}" -m model.latent.eval_condition_residuals \
      --checkpoint "${ckpt}" \
      --gpu "${gpu}" --device "cuda:${gpu}" \
      "${COMMON[@]}" \
      --out-csv "${REPORT_ROOT}/${label}.csv" \
      --out-json "${REPORT_ROOT}/${label}.json"
    local code=$?
    echo "[$(date +%F_%T)] finish ${label} exit=${code}"
    echo "${code}" > "${exit_file}"
    return "${code}"
  } > "${log}" 2>&1
}

run_one primary_scfoundation 0 "${COUPLED}/output/latentfm_runs/top3_pertcond_baseline/20260616_top3_pertcond_v4/scfoundation/best.pt" &
run_one rel002 1 "${COUPLED}/output/latentfm_runs/scfoundation_relational_residual_20260619/20260619_scfoundation_rel002_comp006_endpoint5_8k/best.pt" &
run_one rel005 2 "${COUPLED}/output/latentfm_runs/scfoundation_relational_residual_20260619/20260619_scfoundation_rel005_comp006_endpoint5_8k/best.pt" &
run_one strong_comp020 3 "${COUPLED}/output/latentfm_runs/scfoundation_strong_composition_20260619/20260619_scfoundation_comp020_endpoint5_8k/best.pt" &
run_one stack_comp006 4 "${COUPLED}/output/latentfm_runs/stack_composite_selection/20260618_stack_comp006_selppmmd05_8k/best.pt" &

wait

status=0
missing=0
for label in primary_scfoundation rel002 rel005 strong_comp020 stack_comp006; do
  exit_file="${RUN_ROOT}/logs/${label}.exit"
  if [[ ! -f "${exit_file}" ]]; then
    status=99
    missing=$((missing + 1))
    continue
  fi
  code=$(cat "${exit_file}")
  if [[ "${code}" != "0" ]]; then
    status=1
  fi
done

if [[ "${status}" == "0" ]]; then
  "${PY}" "${ROOT}/ops/summarize_condition_residual_audit.py" \
    --input-dir "${REPORT_ROOT}" \
    --out-md "${ROOT}/reports/LATENTFM_CONDITION_RESIDUAL_AUDIT_20260619.md" \
    --out-json "${ROOT}/reports/latentfm_condition_residual_audit_20260619.json" \
    > "${RUN_ROOT}/logs/summary.log" 2>&1
  summary_code=$?
  if [[ "${summary_code}" != "0" ]]; then
    status=2
  fi
fi

cat > "${RUN_ROOT}/RUN_STATUS.md" <<EOF
# LatentFM Condition Residual Audit

Started: see logs
Finished: $(date '+%F %T %Z')
Status: finished
Exit code: ${status}
Missing exit files: ${missing}
Outputs: ${REPORT_ROOT}
Report: ${ROOT}/reports/LATENTFM_CONDITION_RESIDUAL_AUDIT_20260619.md
EOF

exit "${status}"
