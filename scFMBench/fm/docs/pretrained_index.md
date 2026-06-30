# Pretrained Asset Index

Default root: `SCFM_PRETRAINED_ROOT`, falling back to
`<delivery_root>/scFM_pretrained`. Legacy `COUPLEDFM_PRETRAINED_ROOT` remains an
explicit override.

| Model | Required path under `SCFM_PRETRAINED_ROOT` |
| --- | --- |
| Geneformer V2-316M | `geneformer/Geneformer-V2-316M/` |
| UCE | `uce/model_files/33layer_model.torch` and companion token/protein files |
| State SE-600M | `state/SE-600M/` with `.ckpt` or `.safetensors`; optional `protein_embeddings.pt` |
| arc-stack | `stack/bc_large.ckpt`, `stack/basecount_1000per_15000max.pkl` |
| scLDM | `scdlm/vae_census/{70M.ckpt,70M.yaml,concatenated_unique_genes.parquet}` |
| xVERSE | `xVerse/xVERSE_384.pth` |
| scGPT | `scgpt/{best_model.pt,vocab.json,args.json}` |
| CellNavi | `cellnavi/data/pretrain/pretrain_weights.pth`, `cellnavi/data/gene_name.txt`, `cellnavi/data/Nichenet/{node2idx.json,graph.pkl}` |
| scFoundation | `scFoundation/models.ckpt` |
| NicheFormer | `nicheformer/nicheformer.ckpt` or `nicheformer/theislab_Nicheformer/{config.json,model.safetensors}`; model mean defaults to `SCFM_THIRD_PARTY_ROOT/nicheformer/data/model_means/model.h5ad` |
| TranscriptFormer | `transcriptformer/tf_sapiens/{config.json,model_weights.pt,vocabs/}` by default; set `LATENT_BENCH_TRANSCRIPTFORMER_MODEL=tf_exemplar|tf_metazoa` or `LATENT_BENCH_TRANSCRIPTFORMER_CKPT=/path/to/checkpoint_dir` for another official checkpoint |
| NicheNet standalone | `nichenet/{node2idx.json,idx2node.json,graph.pkl,graph.pt}` |

Third-party source code is separate and should be placed under
`SCFM_THIRD_PARTY_ROOT` (`<delivery_root>/scFM_third_party` by default).

Official download notes:

- NicheFormer README points legacy pretrained weights to Mendeley; place the
  resulting `.ckpt` at `SCFM_PRETRAINED_ROOT/nicheformer/nicheformer.ckpt` or set
  `LATENT_BENCH_NICHEFORMER_CKPT`. The adapter also supports the HuggingFace
  `theislab/Nicheformer` snapshot at
  `SCFM_PRETRAINED_ROOT/nicheformer/theislab_Nicheformer` or
  `LATENT_BENCH_NICHEFORMER_HF_DIR`.
- TranscriptFormer weights are downloaded with the official CLI, e.g.
  `transcriptformer download tf-sapiens --checkpoint-dir $SCFM_PRETRAINED_ROOT/transcriptformer`.
  The adapter uses `tf_sapiens` by default because the current benchmark data are
  human-centric; use `tf_exemplar` for cross-species data.

Environment notes:

- Use a dedicated `transcriptformer` env. The official package requires Python
  >=3.11 and pins `torch==2.5.1`; the shared `scdfm` env may carry newer torch
  versions that the README warns can trigger checkpoint pickle errors.
- The HuggingFace NicheFormer path runs in the shared `scdfm` env. Use a
  dedicated `nicheformer` env only if you need the legacy Lightning `.ckpt`
  workflow; the upstream package pins older numpy/pandas and includes
  Merlin/Dask dependencies that should not be mixed into `scdfm`.

Validate with:

```bash
PYTHONPATH=/path/to/scFM/fm python -m tools.validate_resources --skip-import-test
```
