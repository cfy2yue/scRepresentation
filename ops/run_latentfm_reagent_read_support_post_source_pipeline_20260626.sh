#!/usr/bin/env bash
# Run the CPU-only post-source reagent/read-support pipeline.
#
# Boundary: this script extracts external source metadata and runs CPU gates.
# It does not train, infer, read checkpoints, read canonical multi, read Track C
# query, or use GPU.

set -euo pipefail

ROOT="/data/cyx/1030/scLatent"
cd "${ROOT}"

FRANGIEH_SOURCE="reports/external_artifact_sources_20260626/frangieh_figshare/Frangieh_2021.h5ad"
DIXIT_TAR="reports/external_artifact_sources_20260626/dixit_geo/GSE90063_RAW.tar"
DIXIT_FIGSHARE="reports/external_artifact_sources_20260626/dixit_figshare/Dixit_2016.h5ad"

FRANGIEH_MIN_BYTES=1000000000
DIXIT_MIN_BYTES=1000000000

echo "[post-source] boundary: CPU-only metadata extraction/gates; no training, no GPU"

if [[ -f "${FRANGIEH_SOURCE}" ]]; then
  frangieh_size="$(stat -c '%s' "${FRANGIEH_SOURCE}")"
  if [[ "${frangieh_size}" -ge "${FRANGIEH_MIN_BYTES}" ]]; then
    echo "[post-source] extracting Frangieh artifacts (${frangieh_size} bytes)"
    conda run -n scdfm python ops/extract_latentfm_frangieh_figshare_reagent_artifacts_20260626.py
  else
    echo "[post-source] Frangieh source incomplete (${frangieh_size} bytes); skipping extraction"
  fi
else
  echo "[post-source] Frangieh source missing; skipping extraction"
fi

if [[ -f "${DIXIT_TAR}" ]]; then
  dixit_size="$(stat -c '%s' "${DIXIT_TAR}")"
  if [[ "${dixit_size}" -ge "${DIXIT_MIN_BYTES}" ]]; then
    echo "[post-source] extracting Dixit artifacts (${dixit_size} bytes)"
    conda run -n scdfm python ops/extract_latentfm_dixit_geo_reagent_artifacts_20260626.py
  else
    echo "[post-source] Dixit tar incomplete (${dixit_size} bytes); skipping extraction"
  fi
else
  echo "[post-source] Dixit tar missing; skipping extraction"
fi

if [[ -f "${DIXIT_FIGSHARE}" ]]; then
  dixit_figshare_size="$(stat -c '%s' "${DIXIT_FIGSHARE}")"
  if [[ "${dixit_figshare_size}" -ge "${DIXIT_MIN_BYTES}" ]]; then
    echo "[post-source] extracting Dixit figshare artifacts (${dixit_figshare_size} bytes)"
    conda run -n scdfm python ops/extract_latentfm_dixit_figshare_reagent_artifacts_20260626.py
  else
    echo "[post-source] Dixit figshare source incomplete (${dixit_figshare_size} bytes); skipping extraction"
  fi
else
  echo "[post-source] Dixit figshare source missing; skipping extraction"
fi

echo "[post-source] refreshing combined manifest"
conda run -n scdfm python ops/build_latentfm_reagent_read_support_combined_manifest_20260626.py

echo "[post-source] running combined external-artifact preflight"
LATENTFM_EXTERNAL_ARTIFACT_CONFIG="configs/latentfm_reagent_read_support_combined_manifest_20260626.json" \
LATENTFM_EXTERNAL_ARTIFACT_OUT_PREFIX="latentfm_reagent_read_support_combined_preflight_20260626" \
LATENTFM_EXTERNAL_ARTIFACT_OUT_TITLE="LATENTFM_REAGENT_READ_SUPPORT_COMBINED_PREFLIGHT_20260626" \
conda run -n scdfm python ops/audit_latentfm_external_artifact_preflight_20260626.py

echo "[post-source] running combined signal gate"
conda run -n scdfm python ops/audit_latentfm_reagent_read_support_combined_signal_gate_20260626.py

echo "[post-source] localizing MMD blocker"
conda run -n scdfm python ops/audit_latentfm_reagent_read_support_mmd_blocker_20260626.py

echo "[post-source] running MMD-safe residual gate"
conda run -n scdfm python ops/audit_latentfm_reagent_read_support_mmd_safe_residual_gate_20260626.py

echo "[post-source] running source-block LODO confound gate"
conda run -n scdfm python ops/audit_latentfm_reagent_read_support_source_block_lodo_gate_20260626.py

echo "[post-source] refreshing pre-registered scaling axis matrix"
conda run -n scdfm python ops/build_latentfm_scaling_preregistered_axis_matrix_20260626.py

echo "[post-source] done"
