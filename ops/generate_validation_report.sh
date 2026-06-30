#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
RUN_LOG="${1:-${ROOT}/logs/post_sync_validate/current.log}"
REPORT="${REPORT:-${ROOT}/reports/VALIDATION_REPORT.md}"
TRANSFER_STATUS_FILE="${TRANSFER_STATUS_FILE:-${ROOT}/logs/transfer_from_lilab.status}"
TRANSFER_LOG="${TRANSFER_LOG:-${ROOT}/logs/transfer_from_lilab.log}"

status_from_log() {
  local pattern="$1"
  if grep -q "$pattern" "$RUN_LOG" 2>/dev/null; then
    if grep -q 'POST-SYNC VALIDATION PASSED' "$RUN_LOG" 2>/dev/null; then
      printf 'passed'
    elif grep -q 'POST-SYNC VALIDATION FAILED' "$RUN_LOG" 2>/dev/null; then
      printf 'ran before failure'
    else
      printf 'ran'
    fi
  else
    printf 'pending'
  fi
}

overall_status() {
  if grep -q 'POST-SYNC VALIDATION PASSED' "$RUN_LOG" 2>/dev/null; then
    printf 'passed'
  elif grep -q 'POST-SYNC VALIDATION FAILED' "$RUN_LOG" 2>/dev/null; then
    printf 'failed'
  else
    printf 'pending'
  fi
}

sync_status() {
  if grep -q $'\tALL DONE' "$TRANSFER_STATUS_FILE" 2>/dev/null || \
     grep -q 'ALL DONE' "$TRANSFER_LOG" 2>/dev/null; then
    printf 'complete'
  else
    printf 'pending'
  fi
}

evidence() {
  if [[ -f "$RUN_LOG" ]]; then
    printf '`%s`' "$RUN_LOG"
  else
    printf 'pending log'
  fi
}

main() {
  mkdir -p "$(dirname "$REPORT")"
  local generated_at sync_state log_ref
  generated_at="$(date '+%F %T')"
  sync_state="$(sync_status)"
  log_ref="$(evidence)"

  {
    cat <<EOF
# Validation Report

Status: $(overall_status)

Generated: ${generated_at}

Primary evidence log: ${log_ref}

## Environment

| Item | Evidence |
| --- | --- |
| Conda env | \`/data/cyx/software/miniconda3/envs/scdfm\` via \`init-scdfm.sh\` |
| Python | captured in validation commands |
| Torch/CUDA | validated earlier by environment setup; post-sync smoke uses CUDA when data is ready |
| flash-attn | installed in \`scdfm\`; import verified during setup |
| Visible GPUs | select with \`nvidia-smi\` before each GPU run; shared-GPU policy is recorded in \`${ROOT}/AGENTS.md\` |
| CPU/resource caps | post-sync default: \`OMP/MKL/OPENBLAS/NUMEXPR/BLIS=4\`, \`MIN_MEM_AVAILABLE_GIB=16\`, \`nice=3\`, one smoke GPU unless overridden |

## Data Sync

| Resource | Expected local path | Status | Evidence |
| --- | --- | --- | --- |
| CoupledFM biFlow data | \`${ROOT}/dataset/biFlow_data\` | ${sync_state} | transfer status/log |
| CoupledFM cellgene census | \`${ROOT}/dataset/cellgene_census\` | ${sync_state} | transfer status/log |
| scFMBench data | \`${ROOT}/dataset/scFM_data\` | ${sync_state} | transfer status/log |
| raw source data | \`${ROOT}/dataset/raw\` | ${sync_state} | transfer status/log |
| scFM pretrained | \`${ROOT}/scFM_pretrained\` | ${sync_state} | transfer status/log |
| scFM third party | \`${ROOT}/scFM_third_party\` | ${sync_state} | transfer status/log |
| CoupledFM gene caches | \`${ROOT}/pretrainckpt/genepert_cache\` | ${sync_state} | transfer status/log |
| Training-ready dataset package | \`${ROOT}/dataset/{README.md,latentfm_full,biFlow_data,scFM_data,drug_cache,cellgene_census}\` | $(if [[ -x "${ROOT}/ops/validate_dataset_package.sh" ]]; then printf 'checkable'; else printf 'pending'; fi) | \`${ROOT}/ops/validate_dataset_package.sh\` |

## CoupledFM Validation

| Check | Command | Status | Evidence |
| --- | --- | --- | --- |
| Resource validation, local-smoke | \`python -m model.tools.validate_resources --mode local-smoke --datasets Adamson\` | $(status_from_log 'validate_resources --mode local-smoke') | ${log_ref} |
| Resource validation, full | \`python -m model.tools.validate_resources\` | $(status_from_log 'RUN: python -m model.tools.validate_resources$') | ${log_ref} |
| CPU regression tests | \`python -m pytest ... -q\` | $(status_from_log 'RUN: python -m pytest') | ${log_ref} |
| Core smoke | \`python model/tools/smoke_test.py\` | $(status_from_log 'RUN: python model/tools/smoke_test.py') | ${log_ref} |
| Historical 4-GPU launcher dry-run | \`bash model/scripts/submit_pert_embed_compare_8gpu.sh --dry-run --gpus 0,1,2,3 ...\` | $(status_from_log 'submit_pert_embed_compare_8gpu.sh') | ${log_ref}; future runs must select available GPUs first |
| Raw-pretrain smoke | \`bash model/tests/local_single_gpu_smoke.sh\` | $(status_from_log 'local_single_gpu_smoke.sh') | ${log_ref} |
| CoupledFM train smoke | \`bash model/tests/local_single_gpu_smoke.sh\` | $(status_from_log 'local_single_gpu_smoke.sh') | ${log_ref} |
| Inference smoke | \`python -m model.inference --ckpt ... --max-cells-per-cond 4 --max-conditions 2\` | $(status_from_log 'RUN: python -m model.inference') | ${log_ref} |
| Training-ready dataset package presence | \`${ROOT}/ops/validate_dataset_package.sh\` | $(if [[ -x "${ROOT}/ops/validate_dataset_package.sh" ]]; then printf 'available'; else printf 'pending'; fi) | run manually after data package restore |

## scFMBench Validation

| Check | Command | Status | Evidence |
| --- | --- | --- | --- |
| Resource validation | \`python fm/tools/validate_resources.py --models scgpt cellnavi stack\` | $(status_from_log 'fm/tools/validate_resources.py') | ${log_ref} |
| Preflight manifest | \`python fm/tools/preflight_embedding.py --models scgpt cellnavi stack --require-materialized-x\` | $(status_from_log 'fm/tools/preflight_embedding.py') | ${log_ref} |
| PCA baseline smoke | \`python fm/smoke/test_pca_baseline.py\` | $(status_from_log 'fm/smoke/test_pca_baseline.py') | ${log_ref} |
| Metrics function smoke | \`python benchmark/smoke/test_metrics_pipeline.py\` | $(status_from_log 'benchmark/smoke/test_metrics_pipeline.py') | ${log_ref} |
| Metrics CLI smoke | \`python benchmark/smoke/test_metrics_cli.py\` | $(status_from_log 'benchmark/smoke/test_metrics_cli.py') | ${log_ref} |
| One-model embedding smoke | \`python fm/tools/export_embedding_one.py --model cellnavi ... --max-cells 8\` | $(status_from_log 'fm/tools/export_embedding_one.py') | ${log_ref} |
| Metrics on generated latent | \`python benchmark/cli/run_metrics_one.py --emb-dir ... --skip atlas\` | $(status_from_log 'benchmark/cli/run_metrics_one.py') | ${log_ref} |
| Benchmark tiny-run or dry-run | synthetic CLI + generated-latent metric run | $(status_from_log 'benchmark/smoke/test_metrics_cli.py') | ${log_ref} |

## Known Constraints

- Download phase is intentionally checked at most once per hour.
- Smoke tests run with shared-server settings: one GPU by default, moderate
  CPU/I/O priority, four-thread CPU math libraries, and a 16GiB
  available-memory guard unless overridden.
- GPU IDs in runbooks are examples. Always select available cards before launch
  according to \`${ROOT}/AGENTS.md\`.
- Full model benchmark coverage is not claimed until command evidence is added
  to this report.
- \`validate_dataset_package.sh\` defaults to presence checks only; set
  \`RUN_FM_BUNDLE_VALIDATION=1\` to run the three LatentFM HDF5 bundle validators.

## Final Result

$(case "$(overall_status)" in
  passed) printf 'Post-sync validation passed.' ;;
  failed) printf 'Post-sync validation failed; inspect the primary evidence log above for the failing command.' ;;
  *) printf 'Pending sync completion or validation result.' ;;
esac)
EOF
  } > "$REPORT"

  echo "wrote ${REPORT}"
}

main "$@"
