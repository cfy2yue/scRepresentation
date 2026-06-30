# Environment Map

The code no longer hard-codes machine-local legacy paths. Runtime paths resolve in this
order:

1. Explicit model-specific env vars, for example `LATENT_BENCH_SCGPT_MODEL_DIR`,
   `LATENT_BENCH_STACK_CKPT`, `LATENT_BENCH_STATE_CKPT`, or
   `SCFM_SCGPT_PYTHON`.
2. Shared `SCFM_*` roots:
   - `SCFM_DATA_ROOT`
   - `SCFM_PRETRAINED_ROOT`
   - `SCFM_THIRD_PARTY_ROOT`
   - `SCFM_OUTPUT_ROOT`
   - `SCFM_ENVS_ROOT`
   - `SCFM_CACHE_ROOT`
3. Sibling defaults under `<delivery_root>/scFM_*`.

## Python Resolution

`fm/tools/model_registry.py` resolves model interpreters as:

1. `LATENT_BENCH_<MODEL>_PYTHON`
2. `SCFM_<MODEL>_PYTHON`
3. `SCFM_ENVS_ROOT/<env_name>/bin/python`
4. `/data3/chenfy/miniconda3/envs/<env_name>/bin/python`
5. `python3`

Default env names:

| Model | Env name |
| --- | --- |
| scGPT, xVERSE, scFoundation, UCE, scLDM, stack, Geneformer | `scdfm` |
| CellNavi | `cellnavi` |

Current local `scdfm` additions for scFM benchmark:

- `flash-attn==2.5.9.post1` installed from the official Dao-AILab wheel
  `cu122torch2.4cxx11abiFALSE-cp312`; this is the newest tested prebuilt wheel
  here that loads on the host `glibc 2.17`.
- `transformers==4.46.3` and `ipython` are installed for Geneformer/scGPT import
  compatibility.
- `datasets` is intentionally not installed in `scdfm`; on this Python 3.12 host
  it pulls a slow `pyarrow` source build. The scGPT adapter provides a small import
  shim for upstream modules that only need `datasets` at import time.

Use `SCFM_<MODEL>_PYTHON=/path/to/python` if a model needs a stricter upstream
environment later.

## Cache

Queue workers default cache variables to `SCFM_CACHE_ROOT`:

```text
SCFM_CACHE_ROOT/
  huggingface/
  huggingface/transformers/
  xdg/
  torch/
```

Override these directly if your cluster requires a different cache location.
