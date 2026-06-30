# scFM benchmark status

## Models

The active benchmark includes **11** model names under `output/embeddings/*/`: nine foundation encoders plus fitted baselines **pca** and **scvi** (each with `latent_dim = 128` after export). Rebuild the run manifest after adding baseline exports:

`python benchmark/cli/build_metrics_manifest.py`

## Dual latent_space metrics

Per `model` × `dataset`, metrics are written to:

- `output/metrics/<model>/<dataset>/raw/` — native exported latents.
- `output/metrics/<model>/<dataset>/pca128/` — per-task PCA to `K = min(128, latent_dim)` after scaling (see `benchmark/docs/metric_latent_space_recommendations.md`).

Batch runner: `benchmark/cli/run_metrics_batch.py --latent-space raw|pca128|both`.

## Inventory

- `output/benchmark_inventory/raw_inventory.{md,csv}` — embedding/export audit.
- `output/benchmark_inventory/pca128_eligibility.csv` — intended PCA fit scope per `(model, dataset)` for the pca128 metric branch.

Run: `python benchmark/cli/audit_raw_inventory.py`

## §6 Summaries

Aggregate CSVs from all `summary.json` files:

`python benchmark/cli/aggregate_report.py --write-scfm-benchmark-csvs`
