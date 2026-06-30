# Environment lock notes

The canonical training environment for `CoupledFM/model` is `environment.yml` in this directory (no machine-specific `prefix:`; install with `conda env create -f environment.yml` on your host).

Pinned highlights from the last export (verify after `conda install`):

- Python 3.11
- `torch` 2.7.x + CUDA 12.6 pip wheels
- `scanpy`, `anndata`, `pot` (POT/EMD for OT baselines)
- `pytest` for tests

Optional H100/H20 / flash path: install `flash-attn` only if needed via `pip install "flash-attn>=2.5,<3"` or `pip install -e ".[flash]"` from `CoupledFM/model/pyproject.toml` when present.
