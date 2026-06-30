#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/dataset}"
PACKAGE="training"
OUT=""
WITH_SHA256=0

usage() {
  cat <<'EOF'
Usage:
  generate_dataset_package_manifest.sh [options]

Options:
  --dataset-root PATH   Dataset root. Default: /data/cyx/1030/dataset
  --package NAME        training | rebuild | legacy | all. Default: training
  --out PATH            Output TSV path. Default: reports/dataset_<package>_package_manifest.tsv
  --with-sha256         Also compute sha256 for each file. This reads file contents and can be slow.
  -h, --help            Show this help

The default training package is intentionally lightweight and records path,
type, size, mtime, and symlink target without reading full file contents.
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
    --out)
      OUT="$2"
      shift 2
      ;;
    --with-sha256)
      WITH_SHA256=1
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

case "$PACKAGE" in
  training)
    REL_PATHS=(README.md latentfm_full biFlow_data scFM_data drug_cache cellgene_census)
    ;;
  rebuild)
    REL_PATHS=(Training_data raw latentfm_staging)
    ;;
  legacy)
    REL_PATHS=(latentfm)
    ;;
  all)
    REL_PATHS=(.)
    ;;
  *)
    echo "unknown package: $PACKAGE" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "$OUT" ]]; then
  OUT="${ROOT}/reports/dataset_${PACKAGE}_package_manifest.tsv"
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "dataset root does not exist: $DATASET_ROOT" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
tmp="$(mktemp "${OUT}.tmp.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

if [[ "$WITH_SHA256" == "1" ]]; then
  printf "relative_path\ttype\tsize_bytes\tmtime_epoch\tsymlink_target\tsha256\n" > "$tmp"
else
  printf "relative_path\ttype\tsize_bytes\tmtime_epoch\tsymlink_target\n" > "$tmp"
fi

emit_one() {
  local path="$1"
  local rel type size mtime target checksum

  rel="${path#${DATASET_ROOT}/}"
  if [[ "$path" == "$DATASET_ROOT" ]]; then
    rel="."
  fi

  if [[ -L "$path" ]]; then
    type="symlink"
    target="$(readlink "$path")"
  elif [[ -d "$path" ]]; then
    type="dir"
    target=""
  elif [[ -f "$path" ]]; then
    type="file"
    target=""
  else
    type="other"
    target=""
  fi

  size="$(stat -c '%s' "$path")"
  mtime="$(stat -c '%Y' "$path")"

  if [[ "$WITH_SHA256" == "1" && -f "$path" && ! -L "$path" ]]; then
    checksum="$(sha256sum "$path" | awk '{print $1}')"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$rel" "$type" "$size" "$mtime" "$target" "$checksum" >> "$tmp"
  elif [[ "$WITH_SHA256" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\t\n" "$rel" "$type" "$size" "$mtime" "$target" >> "$tmp"
  else
    printf "%s\t%s\t%s\t%s\t%s\n" "$rel" "$type" "$size" "$mtime" "$target" >> "$tmp"
  fi
}

for rel in "${REL_PATHS[@]}"; do
  root_path="${DATASET_ROOT%/}/${rel}"
  if [[ ! -e "$root_path" && ! -L "$root_path" ]]; then
    echo "missing package path: $root_path" >&2
    exit 1
  fi

  emit_one "$root_path"
  if [[ -d "$root_path" && ! -L "$root_path" ]]; then
    while IFS= read -r -d '' path; do
      emit_one "$path"
    done < <(find "$root_path" -mindepth 1 -print0 | sort -z)
  fi
done

mv "$tmp" "$OUT"
trap - EXIT

echo "wrote package manifest: $OUT"
wc -l "$OUT"
