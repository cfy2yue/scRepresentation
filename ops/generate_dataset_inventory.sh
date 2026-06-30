#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/cyx/1030/scLatent}"
REPORT="${REPORT:-${ROOT}/reports/DATASET_INVENTORY.md}"
TRANSFER_STATUS_FILE="${TRANSFER_STATUS_FILE:-${ROOT}/logs/transfer_from_lilab.status}"
TRANSFER_LOG="${TRANSFER_LOG:-${ROOT}/logs/transfer_from_lilab.log}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-4}"
RUN_NICE="${RUN_NICE:-3}"

require_sync_complete() {
  if ! grep -q $'\tALL DONE' "$TRANSFER_STATUS_FILE" 2>/dev/null && \
     ! grep -q 'ALL DONE' "$TRANSFER_LOG" 2>/dev/null; then
    echo "transfer has not reported ALL DONE; dataset inventory is intentionally disabled" >&2
    exit 10
  fi
}

run_low_priority() {
  if command -v ionice >/dev/null 2>&1; then
    nice -n "$RUN_NICE" ionice -c 2 -n 4 "$@"
  else
    nice -n "$RUN_NICE" "$@"
  fi
}

path_size() {
  local path="$1"
  if [[ -e "$path" ]]; then
    run_low_priority du -sh "$path" 2>/dev/null | awk '{print $1}'
  else
    printf 'missing'
  fi
}

file_count() {
  local path="$1"
  if [[ -d "$path" ]]; then
    run_low_priority find "$path" -type f -printf '.' 2>/dev/null | wc -c
  else
    printf '0'
  fi
}

status_for() {
  local path="$1"
  if [[ -e "$path" ]]; then
    printf 'present'
  else
    printf 'missing'
  fi
}

write_summary_row() {
  local rel="$1"
  local note="$2"
  local abs="${ROOT}/${rel}"
  printf '| `%s` | %s | %s | %s |\n' \
    "$rel" \
    "$(path_size "$abs")" \
    "$(file_count "$abs")" \
    "$note"
}

write_required_row() {
  local rel="$1"
  printf '| `%s` | %s |\n' "$rel" "$(status_for "${ROOT}/${rel}")"
}

main() {
  require_sync_complete
  mkdir -p "$(dirname "$REPORT")"

  local generated_at
  generated_at="$(date '+%F %T')"

  {
    cat <<EOF
# Dataset Inventory

Status: generated after sync completion and LatentFM bundle preparation

Generated: ${generated_at}

Canonical data root:

\`\`\`text
${ROOT}/dataset
\`\`\`

## Current Tree

\`\`\`text
dataset/
  README.md
  Training_data/
    scfoundation/
    scldm/
  biFlow_data/
    control_stack/
    gt_stack/
    control_center_stack/
    split_seed42.json
  cellgene_census/
    processed/
  drug_cache/
    sciplex_label_identity_561/
  latentfm_full/
    stack/
    scfoundation/
    scldm/
  latentfm/
    stack/
    scfoundation/
    state/
  latentfm_staging/
    scfm_embeddings/
  scFM_data/
    staging/
    raw/
  raw/
\`\`\`

## Size and Count Summary

| Path | Size | Files | Notes |
| --- | ---: | ---: | --- |
EOF
    write_summary_row "dataset" "canonical local dataset root; includes training-ready bundles plus source/rebuild archives"
    write_summary_row "dataset/latentfm_full" "current training-ready LatentFM bundles for stack, scfoundation, and scldm"
    write_summary_row "dataset/biFlow_data" "CoupledFM/RawFM perturbation stacks and canonical split"
    write_summary_row "dataset/scFM_data" "scFMBench h5ad data"
    write_summary_row "dataset/cellgene_census" "CoupledFM raw pretrain data"
    write_summary_row "dataset/drug_cache" "SciPlex label-identity drug condition cache"
    write_summary_row "dataset/Training_data" "LiLab-synced source h5ad trees used to build latentfm_full; not normally read by active training"
    write_summary_row "dataset/raw" "raw source h5ads for rebuild/release"
    write_summary_row "dataset/latentfm_staging" "intermediate scFM embedding staging"
    write_summary_row "dataset/latentfm" "older/partial LatentFM bundles retained for legacy compatibility checks"

    cat <<'EOF'

## Required Example Files

| File | Status |
| --- | --- |
EOF
    write_required_row "dataset/README.md"
    write_required_row "dataset/biFlow_data/control_stack/Adamson.h5ad"
    write_required_row "dataset/biFlow_data/gt_stack/Adamson.h5ad"
    write_required_row "dataset/biFlow_data/control_center_stack/Adamson.h5ad"
    write_required_row "dataset/biFlow_data/split_seed42.json"
    write_required_row "dataset/cellgene_census/processed/kidney/kidney_top6000var.h5ad"
    write_required_row "dataset/latentfm_full/stack/manifest.json"
    write_required_row "dataset/latentfm_full/scfoundation/manifest.json"
    write_required_row "dataset/latentfm_full/scldm/manifest.json"
    write_required_row "dataset/latentfm_full/stack/condition_metadata.json"
    write_required_row "dataset/latentfm_full/scfoundation/condition_metadata.json"
    write_required_row "dataset/latentfm_full/scldm/condition_metadata.json"

    cat <<'EOF'

## Current Training-Ready Package

For current LatentFM/CoupledFM/scFMBench execution, the compact training-ready
package is:

```text
dataset/README.md
dataset/latentfm_full/
dataset/biFlow_data/
dataset/scFM_data/
dataset/drug_cache/
dataset/cellgene_census/
```

## Source / Rebuild Archive Candidates

These are useful for provenance and regeneration, but are not normally needed
for direct training once `latentfm_full` exists:

```text
dataset/Training_data/
dataset/raw/
dataset/latentfm_staging/
```

`dataset/latentfm/` is an older/partial compatibility tree and should not be
included in the main cloud training package unless a specific legacy run needs
it.

## Temporary File Candidates

These are listed for review before any deletion.

```text
EOF
    run_low_priority find "${ROOT}/dataset" -type f \
      \( -name '*.tmp.h5ad' -o -name '*.bak*' -o -name '*.before_*' -o -name '.*.??????' \) \
      -print 2>/dev/null | head -n 200

    cat <<'EOF'
```

## Packaging Notes

- `dataset/` is intended to be compressible and publishable separately from
  code.
- For cloud/Zenodo release, prefer a primary "train/evaluate immediately"
  package containing only the training-ready package above, and keep source
  rebuild material in a separate optional archive.
- Temporary or backup h5ad variants should not be part of the canonical release
  unless explicitly documented.
- Pretrained model weights live outside `dataset/` because they are model
  resources, not raw/benchmark data.
- Release documentation should state expected sibling paths for
  `scFM_pretrained/`, `scFM_third_party/`, and `pretrainckpt/`.
- `dataset/raw/cellgene_census` is a symlink to the old LiLab-style location;
  portable packages should use the real copied `dataset/cellgene_census`
  directory instead.

## Retention Plan

Detailed backup and cleanup guidance is recorded in:

```text
reports/DATASET_RETENTION_PLAN_20260617.md
```

## Final Rsync Evidence

Final dry-run comparison is recorded in `reports/FINAL_RSYNC_AUDIT.md`.
Content resources match the LiLab source, with only documented metadata
regeneration differences for local scFMBench embedding manifests.
EOF
  } > "$REPORT"

  echo "wrote ${REPORT}"
}

main "$@"
