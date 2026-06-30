#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/dataset}"
DEST_ROOT=""
DRY_RUN=0
RUN_FM_BUNDLE_VALIDATION=0

usage() {
  cat <<'EOF'
Usage:
  stage_training_dataset_package.sh --dest PATH [options]

Options:
  --dataset-root PATH          Source dataset root. Default: /data/cyx/1030/dataset
  --dest PATH                  Destination staging root. Creates PATH/dataset
  --dry-run                    Show rsync actions without copying
  --run-fm-bundle-validation   Run full LatentFM bundle validation before staging
  -h, --help                   Show this help

This script stages only the training-ready package:
  dataset/README.md
  dataset/latentfm_full/
  dataset/biFlow_data/
  dataset/scFM_data/
  dataset/drug_cache/
  dataset/cellgene_census/

It intentionally does not stage source/rebuild archives:
  dataset/Training_data/
  dataset/raw/
  dataset/latentfm_staging/
  dataset/latentfm/

The script does not pass rsync --delete.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --dest)
      DEST_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if [[ -z "$DEST_ROOT" ]]; then
  echo "missing required --dest PATH" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "dataset root does not exist: $DATASET_ROOT" >&2
  exit 1
fi

STAGE_DATASET="${DEST_ROOT%/}/dataset"
RSYNC_FLAGS=(-aH --info=progress2)
if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_FLAGS+=(--dry-run)
fi

echo "source dataset: $DATASET_ROOT"
echo "stage dataset:  $STAGE_DATASET"

RUN_FM_BUNDLE_VALIDATION="$RUN_FM_BUNDLE_VALIDATION" \
  "${ROOT}/ops/validate_dataset_package.sh" --dataset-root "$DATASET_ROOT"

mkdir -p "$STAGE_DATASET"

rsync "${RSYNC_FLAGS[@]}" \
  "${DATASET_ROOT}/README.md" \
  "${DATASET_ROOT}/latentfm_full" \
  "${DATASET_ROOT}/biFlow_data" \
  "${DATASET_ROOT}/scFM_data" \
  "${DATASET_ROOT}/drug_cache" \
  "${DATASET_ROOT}/cellgene_census" \
  "$STAGE_DATASET/"

if [[ "$DRY_RUN" == "0" ]]; then
  {
    printf "path\tsize\n"
    for rel in README.md latentfm_full biFlow_data scFM_data drug_cache cellgene_census; do
      du -sh "${STAGE_DATASET}/${rel}" 2>/dev/null | awk -v rel="$rel" '{print rel "\t" $1}'
    done
  } > "${STAGE_DATASET}/PACKAGE_CONTENTS.tsv"

  "${ROOT}/ops/generate_dataset_package_manifest.sh" \
    --dataset-root "$STAGE_DATASET" \
    --package training \
    --out "${STAGE_DATASET}/PACKAGE_MANIFEST.tsv" >/dev/null

  cat > "${STAGE_DATASET}/VALIDATION.md" <<EOF
# Dataset Package Validation

Run after restore:

\`\`\`bash
source ${ROOT}/init-scdfm.sh
${ROOT}/ops/validate_dataset_package.sh --dataset-root /path/to/restored/dataset
RUN_FM_BUNDLE_VALIDATION=1 ${ROOT}/ops/validate_dataset_package.sh --dataset-root /path/to/restored/dataset
\`\`\`

This package intentionally excludes source/rebuild archives:

\`\`\`text
Training_data/
raw/
latentfm_staging/
latentfm/
\`\`\`
EOF
fi

echo "staging command finished"
