# Data Pipeline

## Canonical Root

The canonical local data root is:

```text
/data/cyx/1030/dataset
```

This directory is intentionally self-contained so it can later be staged,
compressed, backed up, or uploaded separately from the code repositories.

## Training-Ready Package

The current direct-use package is about 444G:

| Path | Role |
|---|---|
| `dataset/latentfm_full` | Formal LatentFM HDF5 bundles for Stack, scLDM, and scFoundation. |
| `dataset/biFlow_data` | Canonical split/control/ground-truth source used by CoupledFM-style workflows. |
| `dataset/scFM_data` | scFMBench benchmark datasets. |
| `dataset/drug_cache` | Current small drug-condition caches. |
| `dataset/cellgene_census` | Support resource used by benchmark/model tooling. |
| `dataset/README.md` | Portable dataset-root instructions. |

Formal LatentFM bundles currently expected under `latentfm_full`:

```text
latentfm_full/stack/
latentfm_full/scldm/
latentfm_full/scfoundation/
```

Each bundle should contain:

```text
manifest.json
condition_metadata.json
*.h5
```

## Source, Rebuild, And Legacy Layers

The full local data tree is about 850G because it also contains source and
rebuild layers:

| Path | Size | Status |
|---|---:|---|
| `dataset/Training_data` | about 351G | LiLab-synced rebuild source; not read by normal training once `latentfm_full` is validated. |
| `dataset/raw` | about 30G | Rebuild/provenance source layer. |
| `dataset/latentfm_staging` | about 11G | Intermediate staging. |
| `dataset/latentfm` | about 17G | Legacy compatibility tree. |

Do not delete these layers until active LatentFM probes finish, package
validation is repeated, and the staged/cloud backup is verified.

## Validation Commands

Lightweight package validation:

```bash
/data/cyx/1030/scLatent/ops/validate_dataset_package.sh
```

Full bundle validation before final cloud or Zenodo packaging:

```bash
source /data/cyx/1030/scLatent/init-scdfm.sh
RUN_FM_BUNDLE_VALIDATION=1 /data/cyx/1030/scLatent/ops/validate_dataset_package.sh
```

If validating a restored copy:

```bash
/data/cyx/1030/scLatent/ops/validate_dataset_package.sh --dataset-root /path/to/restored/dataset
```

## Backup Package Boundary

Primary direct-use backup package:

```text
dataset/latentfm_full/
dataset/biFlow_data/
dataset/scFM_data/
dataset/drug_cache/
dataset/cellgene_census/
dataset/README.md
```

Reference manifests and staging helpers:

```text
/data/cyx/1030/scLatent/reports/dataset_training_package_manifest.tsv
/data/cyx/1030/scLatent/reports/DATASET_CLOUD_BACKUP_MANIFEST_20260617.md
/data/cyx/1030/scLatent/ops/stage_training_dataset_package.sh
/data/cyx/1030/scLatent/ops/compare_dataset_package_manifest.sh
```

## Preprocessing Invariants

The benchmark h5ad inputs under `scFM_data` are already log1p-processed where
the benchmark expects log1p data. Adapters that require count-like inputs must
use explicit count sources such as `raw.X` or `layers['counts']`.

Current NicheFormer and TranscriptFormer policy:

- use explicit count/raw source;
- do not apply duplicate `log1p`;
- do not silently reconstruct pseudo-counts with `expm1(X)`.

Evidence:

```text
/data/cyx/1030/scLatent/reports/SCFMBENCH_CONTINUATION_CHECK_20260619.md
```

## Current Cleanup Rule

No data cleanup should happen before:

1. active relational-residual LatentFM probes and posthoc evaluations finish;
2. lightweight dataset validation passes again;
3. full HDF5 validation passes if preparing a final public package;
4. staged/cloud backup is verified against the package manifest.
