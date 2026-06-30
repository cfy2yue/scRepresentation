# scFM Architecture

`scFM/` is code only. Inputs, checkpoints, third-party source mirrors, and
outputs are sibling directories.

```mermaid
flowchart LR
  data[scFM_data/staging or scFM_data/raw]
  ckpt[scFM_pretrained]
  tp[scFM_third_party]
  preflight[fm/tools/preflight_embedding.py]
  manifest[scFM_output/embedding_runs/manifest.jsonl]
  export[fm/tools/submit_embedding_queue.py]
  emb[scFM_output/embeddings/model/dataset/raw]
  metrics[benchmark/cli/run_metrics_one.py]
  mout[scFM_output/metrics/model/dataset]
  data --> preflight --> manifest --> export --> emb --> metrics --> mout
  ckpt --> export
  tp --> export
```

- `fm/paths.py` is the source of truth for default roots.
- `SCFM_DATA_ROOT`, `SCFM_PRETRAINED_ROOT`, `SCFM_THIRD_PARTY_ROOT`, and
  `SCFM_OUTPUT_ROOT` override the sibling defaults.
- `python -m tools.validate_resources` reports the resolved layout and missing
  resources before live export.
