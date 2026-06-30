#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/dataset}"
RUN_FM_BUNDLE_VALIDATION="${RUN_FM_BUNDLE_VALIDATION:-0}"

usage() {
  cat <<'EOF'
Usage:
  validate_dataset_package.sh [--dataset-root PATH] [--run-fm-bundle-validation]

Environment:
  ROOT=/data/cyx/1030/scLatent
  DATASET_ROOT=$ROOT/dataset
  RUN_FM_BUNDLE_VALIDATION=0

Examples:
  /data/cyx/1030/scLatent/ops/validate_dataset_package.sh
  DATASET_ROOT=/path/to/restored/dataset /data/cyx/1030/scLatent/ops/validate_dataset_package.sh
  RUN_FM_BUNDLE_VALIDATION=1 /data/cyx/1030/scLatent/ops/validate_dataset_package.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --run-fm-bundle-validation)
      RUN_FM_BUNDLE_VALIDATION=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

need_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing: $path" >&2
    return 1
  fi
  echo "present: $path"
}

main() {
  local failed=0

  need_path "${DATASET_ROOT}/README.md" || failed=1
  need_path "${DATASET_ROOT}/latentfm_full/stack/manifest.json" || failed=1
  need_path "${DATASET_ROOT}/latentfm_full/scfoundation/manifest.json" || failed=1
  need_path "${DATASET_ROOT}/latentfm_full/scldm/manifest.json" || failed=1
  need_path "${DATASET_ROOT}/latentfm_full/stack/condition_metadata.json" || failed=1
  need_path "${DATASET_ROOT}/latentfm_full/scfoundation/condition_metadata.json" || failed=1
  need_path "${DATASET_ROOT}/latentfm_full/scldm/condition_metadata.json" || failed=1
  need_path "${DATASET_ROOT}/biFlow_data/split_seed42.json" || failed=1
  need_path "${DATASET_ROOT}/scFM_data" || failed=1
  need_path "${DATASET_ROOT}/drug_cache" || failed=1
  need_path "${DATASET_ROOT}/cellgene_census" || failed=1

  if [[ "$failed" != "0" ]]; then
    echo "dataset package presence validation failed" >&2
    exit 1
  fi

  if [[ "$RUN_FM_BUNDLE_VALIDATION" == "1" ]]; then
    cd "${ROOT}/CoupledFM"
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    python -m model.latent.validate_fm_bundle --data-dir "${DATASET_ROOT}/latentfm_full/stack" --require-metadata
    python -m model.latent.validate_fm_bundle --data-dir "${DATASET_ROOT}/latentfm_full/scfoundation" --require-metadata
    python -m model.latent.validate_fm_bundle --data-dir "${DATASET_ROOT}/latentfm_full/scldm" --require-metadata
  else
    echo "skipped FM bundle validation; set RUN_FM_BUNDLE_VALIDATION=1 to enable"
  fi

  echo "dataset package validation passed"
}

main "$@"
