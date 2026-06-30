# Foundation Model Adapters

`fm/` contains thin adapters and export tools. Upstream source mirrors and model
weights are external:

- Source mirrors: `SCFM_THIRD_PARTY_ROOT` (`<delivery_root>/scFM_third_party`)
- Weights/configs: `SCFM_PRETRAINED_ROOT` (`<delivery_root>/scFM_pretrained`)
- Outputs: `SCFM_OUTPUT_ROOT` (`<delivery_root>/scFM_output`)
- Inputs: `SCFM_DATA_ROOT` (`<delivery_root>/scFM_data`)

## Adapter Matrix

| Model | External source dir | Adapter |
| --- | --- | --- |
| stack | `scFM_third_party/stack` | `adapters/stack` |
| state | `scFM_third_party/state` | `adapters/state` |
| scGPT | `scFM_third_party/scGPT-main` | `adapters/scgpt` |
| xVERSE | `scFM_third_party/xVERSE_code` | `adapters/xverse` |
| Geneformer | `scFM_third_party/Geneformer` | `adapters/geneformer` |
| UCE | `scFM_third_party/uce` | `adapters/uce` |
| scLDM | `scFM_third_party/scldm` | `adapters/scldm` |
| scFoundation | `scFM_third_party/scFoundation` | `adapters/scfoundation` |
| CellNavi | `scFM_third_party/CellNavi` | `adapters/cellnavi` |
| NicheFormer | `scFM_third_party/nicheformer` | `adapters/nicheformer` |
| TranscriptFormer | `scFM_third_party/transcriptformer` | `adapters/transcriptformer` |
| PCA baseline | `scFM_third_party/dataset_fitted_baseline` | `adapters/pca_baseline` |

## Tools

| Script | Purpose |
| --- | --- |
| `tools/validate_resources.py` | Print/check resolved external resource roots |
| `tools/preflight_embedding.py` | Scan h5ad roots and write preflight/manifest |
| `tools/submit_embedding_queue.py` | Multi-GPU embedding queue |
| `tools/export_embedding_one.py` | Single model x single h5ad export |
| `tools/validate_embedding_outputs.py` | Export completeness checks |

## Quickstart

```bash
cd <delivery_root>/scFM
export PYTHONPATH="$PWD/fm:${PYTHONPATH:-}"
python -m tools.validate_resources --print-only
python fm/tools/preflight_embedding.py --skip-import-test --models scgpt cellnavi stack
python fm/tools/submit_embedding_queue.py \
  --manifest ../scFM_output/embedding_runs/manifest.jsonl \
  --models scgpt --gpus 0 --dry-run
```

Legacy `LATENT_BENCH_*` and `COUPLEDFM_PRETRAINED_ROOT` overrides still work,
but new defaults are `SCFM_*`.
