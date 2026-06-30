# scFM: Foundation Model Embedding Benchmark

`scFM/` is the lightweight code repository for foundation-model adapters,
embedding export, and benchmark metrics. Large resources live beside the repo
and are intentionally not committed.

```text
<delivery_root>/
  scFM/              # code, scripts, docs, schemas
  scFM_data/         # h5ad inputs and staging files
  scFM_pretrained/   # model weights, model configs, NicheNet assets
  scFM_third_party/  # upstream source mirrors
  scFM_output/       # embeddings, metrics, logs, figures, tmp
  scFM_envs/         # optional unpacked/env reference files
```

## Quickstart

Run from `<delivery_root>/scFM` unless noted:

```bash
export PYTHONPATH="$PWD/fm:${PYTHONPATH:-}"
python -m tools.validate_resources --print-only
python -m tools.validate_resources --models scgpt cellnavi stack --skip-import-test
python fm/tools/preflight_embedding.py --skip-import-test --models scgpt cellnavi stack
python fm/tools/submit_embedding_queue.py \
  --manifest ../scFM_output/embedding_runs/manifest.jsonl \
  --models scgpt --gpus 0 --dry-run
```

After embeddings exist:

```bash
python benchmark/cli/run_metrics_one.py \
  --emb-dir ../scFM_output/embeddings/<model>/<dataset>/raw
```

## Environment Overrides

Recommended new variables:

```bash
export SCFM_DATA_ROOT=/path/to/scFM_data
export SCFM_PRETRAINED_ROOT=/path/to/scFM_pretrained
export SCFM_THIRD_PARTY_ROOT=/path/to/scFM_third_party
export SCFM_OUTPUT_ROOT=/path/to/scFM_output
export SCFM_ENVS_ROOT=/path/to/scFM_envs
```

Legacy explicit overrides such as `COUPLEDFM_PRETRAINED_ROOT`,
`LATENT_BENCH_OUTPUT_ROOT`, and `LATENT_BENCH_*` model-specific variables still
work. New documentation and defaults use `SCFM_*`.

## Resource Contract

Minimum expected resource layout:

```text
scFM_data/
  staging/{atlas,chempert,genepert}/*.h5ad
  raw/{atlas_TS,chemicalpert_bench,genepert_bench}/*.h5ad  # optional

scFM_pretrained/
  geneformer/Geneformer-V2-316M/
  uce/model_files/
  state/SE-600M/
  stack/{bc_large.ckpt,basecount_1000per_15000max.pkl}
  scdlm/vae_census/
  xVerse/xVERSE_384.pth
  scgpt/{best_model.pt,vocab.json,args.json}
  cellnavi/data/{gene_name.txt,Nichenet/,pretrain/pretrain_weights.pth}
  scFoundation/models.ckpt
  nichenet/{node2idx.json,idx2node.json,graph.pkl,graph.pt}

scFM_third_party/
  Geneformer/ uce/ state/ stack/ scGPT-main/ scFoundation/ scldm/
  xVERSE_code/ CellNavi/ dataset_fitted_baseline/
```

The code repo should not contain `.h5ad`, `.npy`, `.pth`, `.pt`, `.pkl`,
`.ckpt`, `.safetensors`, or other large benchmark artifacts.

## Directory Guide

| Path | Role |
| --- | --- |
| `fm/adapters/` | Thin wrappers around each foundation model |
| `fm/tools/` | Preflight, resource validation, embedding queue/export tools |
| `benchmark/` | Metric CLIs, schemas, plots, and aggregation |
| `scripts/` | Local nohup helpers using the same `SCFM_*` defaults |
| `docs/` | Architecture and operational notes |

See `fm/docs/pretrained_index.md` for the model asset checklist and
`docs/architecture.md` for the data flow.
