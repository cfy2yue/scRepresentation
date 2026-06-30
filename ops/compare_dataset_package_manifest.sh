#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/dataset}"
PACKAGE="training"
REFERENCE="${ROOT}/reports/dataset_training_package_manifest.tsv"

usage() {
  cat <<'EOF'
Usage:
  compare_dataset_package_manifest.sh [options]

Options:
  --dataset-root PATH   Dataset root to inspect. Default: /data/cyx/1030/dataset
  --package NAME        training | rebuild | legacy | all. Default: training
  --reference PATH      Reference manifest TSV. Default: reports/dataset_training_package_manifest.tsv
  -h, --help            Show this help

Compares package manifests by relative_path, type, size_bytes, and
symlink_target. mtime_epoch is intentionally ignored because cloud/download
round trips may not preserve modification times.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --package)
      PACKAGE="$2"
      shift 2
      ;;
    --reference)
      REFERENCE="$2"
      shift 2
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

if [[ ! -f "$REFERENCE" ]]; then
  echo "reference manifest not found: $REFERENCE" >&2
  exit 1
fi

tmp_current="$(mktemp)"
tmp_ref_key="$(mktemp)"
tmp_cur_key="$(mktemp)"
trap 'rm -f "$tmp_current" "$tmp_ref_key" "$tmp_cur_key"' EXIT

"${ROOT}/ops/generate_dataset_package_manifest.sh" \
  --dataset-root "$DATASET_ROOT" \
  --package "$PACKAGE" \
  --out "$tmp_current" >/dev/null

awk -F'\t' 'NR == 1 {next} {print $1 "\t" $2 "\t" $3 "\t" $5}' "$REFERENCE" | sort > "$tmp_ref_key"
awk -F'\t' 'NR == 1 {next} {print $1 "\t" $2 "\t" $3 "\t" $5}' "$tmp_current" | sort > "$tmp_cur_key"

if diff -u "$tmp_ref_key" "$tmp_cur_key"; then
  echo "dataset package manifest comparison passed"
  echo "reference: $REFERENCE"
  echo "dataset:   $DATASET_ROOT"
else
  echo "dataset package manifest comparison failed" >&2
  exit 1
fi
