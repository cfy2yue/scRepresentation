"""
scldm VAE adapter: encodes through the pretrained Census TransformerVAE (Hydra + Lightning).

Benchmark protocol: the only expression input is ``adata.X`` (aligned to the Census
vocabulary). No perturbation metadata is fed as a separate conditional stream.

``TransformerVAE.encode`` is invoked with both a full per-cell vector and a
**subset** tensor packed to ``genes_seq_len``; the subset is where expressed genes
(and, when ``force_pert=True``, protected genes from ``obsm['pert_var_idx']``) are
carried so truncation does not drop them—**coverage only**, not a separate
perturbation condition stream.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import paths

from .._common import histogram_pert_kept




def _ensure_scldm_path() -> Path:
    src = paths.third_party_root() / "scldm" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _default_ckpt_dir() -> Path:
    return paths.pretrained_root() / "scdlm" / "vae_census"


def _build_module(config_path: Path, ckpt_path: Path, device: torch.device):
    """Instantiate ``scldm.models.VAE`` from the pretrained Hydra yaml + ckpt."""
    _ensure_scldm_path()
    from omegaconf import OmegaConf
    import hydra
    from scldm._utils import remap_config, remap_pickle

    try:
        OmegaConf.register_new_resolver("eval", eval)
    except Exception:
        pass

    cfg = OmegaConf.load(config_path)
    remap_config(cfg)
    # Resolve every `${...}` (eval resolver + interpolation to datamodule.n_genes, etc.)
    OmegaConf.resolve(cfg)
    module_cfg = cfg.model.module
    # torch.compile is only used in on_fit_start; we are inference-only.
    OmegaConf.update(module_cfg, "compile", False, merge=False)

    module = hydra.utils.instantiate(module_cfg)

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        pickle_module=remap_pickle,
        weights_only=False,
    )
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    module_keys = set(module.state_dict().keys())
    # Census ckpts contain both the raw `vae_model.*` weights and a duplicated
    # `vae_model_compiled._orig_mod.*` copy produced by torch.compile during
    # training; only the raw keys are relevant for direct inference.
    filtered = {k: v for k, v in state_dict.items() if k in module_keys}
    missing = module_keys - set(filtered.keys())
    if missing:
        raise RuntimeError(f"scldm ckpt missing {len(missing)} required keys: sample={list(missing)[:5]}")
    module.load_state_dict(filtered, strict=True)
    module.eval()
    module.to(device)
    return module


def _align_expression_to_vocab(
    adata: ad.AnnData,
    vocab_feature_ids: List[str],
    vocab_feature_names: List[str],
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Project ``adata.X`` onto the model's gene vocabulary (ENSG id then symbol).

    Returns ``(X_full, hits, old_to_vocab)`` where ``X_full`` has shape ``(n_cells, n_vocab)``;
    missing vocab genes stay 0.
    """
    n_cells = adata.n_obs
    n_vocab = len(vocab_feature_ids)
    X_src = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X_src = X_src.astype(np.float32, copy=False)

    X_full = np.zeros((n_cells, n_vocab), dtype=np.float32)
    old_to_vocab = np.full(adata.n_vars, -1, dtype=np.int64)

    # Try ENSG ids first (Ensembl_ID var column or ENSG-like var_names).
    ens_source: Optional[List[str]] = None
    if "Ensembl_ID" in adata.var.columns:
        ens_source = adata.var["Ensembl_ID"].astype(str).tolist()
    elif len(adata.var_names) and str(adata.var_names[0]).startswith("ENSG"):
        ens_source = [str(g) for g in adata.var_names]

    hits = 0
    if ens_source is not None:
        ens_to_col = {g: i for i, g in enumerate(ens_source)}
        for j, ens in enumerate(vocab_feature_ids):
            if ens in ens_to_col:
                src_col = ens_to_col[ens]
                X_full[:, j] = X_src[:, src_col]
                if old_to_vocab[src_col] < 0:
                    old_to_vocab[src_col] = j
                hits += 1

    if hits < 100:
        # Fallback: match by gene symbol (feature_name).
        name_to_col = {str(g).upper(): i for i, g in enumerate(adata.var_names)}
        X_full.fill(0.0)
        old_to_vocab.fill(-1)
        hits = 0
        for j, nm in enumerate(vocab_feature_names):
            k = str(nm).upper()
            if k in name_to_col:
                src_col = name_to_col[k]
                X_full[:, j] = X_src[:, src_col]
                if old_to_vocab[src_col] < 0:
                    old_to_vocab[src_col] = j
                hits += 1
    return X_full, hits, old_to_vocab


def _build_expressed_subset(
    X_full: np.ndarray,
    genes_seq_len: int,
    protected_vocab_indices: Optional[List[List[int]]] = None,
    mask_token_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pack expressed-gene tokens/counts for the encoder with optional protected genes."""
    n_cells, n_vocab = X_full.shape
    expressed = X_full > 0
    num_expressed = expressed.sum(axis=1)
    cap = int(num_expressed.max()) if n_cells else 0
    max_protected = max((len(x) for x in (protected_vocab_indices or [])), default=0)
    seq_len = max(genes_seq_len, cap, max_protected)

    genes_out = np.full((n_cells, seq_len), mask_token_idx, dtype=np.int64)
    counts_out = np.zeros((n_cells, seq_len), dtype=np.float32)
    for i in range(n_cells):
        idx = np.nonzero(expressed[i])[0].tolist()
        protected = []
        if protected_vocab_indices is not None and i < len(protected_vocab_indices):
            protected = [int(j) for j in protected_vocab_indices[i] if 0 <= int(j) < n_vocab]
        protected_unique = list(dict.fromkeys(protected))
        if len(protected_unique) > seq_len:
            raise ValueError(
                f"scldm protected gene count ({len(protected_unique)}) exceeds genes_seq_len ({seq_len})."
            )
        candidate = sorted(
            set(idx).union(protected_unique),
            key=lambda j: (-float(X_full[i, j]), int(j)),
        )
        if len(candidate) > seq_len:
            selected = candidate[:seq_len]
            missing = [j for j in protected_unique if j not in selected]
            if missing:
                keep = [j for j in selected if j in protected_unique]
                rest = [j for j in selected if j not in protected_unique]
                n_rest = max(0, seq_len - len(keep) - len(missing))
                selected = keep + rest[:n_rest] + missing
        else:
            selected = candidate
        idx = np.asarray(selected, dtype=np.int64)
        k = len(idx)
        # Vocab tokens are 1-indexed; slot 0 is the mask/pad embedding.
        genes_out[i, :k] = idx.astype(np.int64) + 1
        counts_out[i, :k] = X_full[i, idx]
    return genes_out, counts_out


def encode(
    adata: ad.AnnData,
    *,
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
    force_pert: bool = True,  # if False, skip protected-gene logic for subset packing
    input_is_log1p: bool = True,
    batch_size: int = 8,
    genes_seq_len: int = 8000,
    device: Optional[str] = None,
    gene_parquet: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """Encode each cell with the scldm TransformerVAE and return mean-posterior latents.

    Returns:
        ``(n_cells, 256*16)`` float32 latents flattened from the VAE's deterministic
        ``(B, n_inducing_points, n_embed_latent)`` encoder output, plus metadata.

    Notes:
        * scldm's ``TransformerVAE.encode`` is deterministic (no reparameterization);
          the output is directly the mean of the learned posterior.
        * ``force_pert``: if True and ``obsm['pert_var_idx']`` exists, map those
          indices to vocab slots and use them as the protected set in
          ``_build_expressed_subset`` (subset packing / truncation). If False,
          no protected list is applied and ``pert_kept_histogram`` is omitted.
          Never a separate perturbation condition—only coverage in the expressed subset.
        * For the 70M Census VAE (``agg_func=projconcat`` in ``70M.yaml``), the
          input pathway is ``TransformerVAE.encode`` → ``InputTransformerVAE`` →
          ``ProjectionConcat``, which applies ``torch.log1p(counts)``
          internally (``scldm/layers.py`` ~line 62). Benchmark convention is
          ``adata.X`` is already ``log1p``-transformed, so we ``expm1`` before
          feeding the VAE when ``input_is_log1p=True`` (default) to avoid
          double-log1p. The ``nnets.EncoderScvi`` ``torch.log1p`` at
          ``scldm/nnets.py`` ~line 43 and the ``log1p_transform`` helper at
          ``scldm/layers.py`` ~line 28 are **not** on this config's code path.
    """
    _ensure_scldm_path()

    ckpt_dir = _default_ckpt_dir()
    ckpt_path = Path(checkpoint or os.environ.get("LATENT_BENCH_SCLDM_CKPT", ckpt_dir / "70M.ckpt"))
    config_path = Path(config or os.environ.get("LATENT_BENCH_SCLDM_CFG", ckpt_dir / "70M.yaml"))
    parquet_path = Path(
        gene_parquet
        or os.environ.get("LATENT_BENCH_SCLDM_GENES", ckpt_dir / "concatenated_unique_genes.parquet")
    )
    for p, label in ((ckpt_path, "checkpoint"), (config_path, "config"), (parquet_path, "gene parquet")):
        if not p.is_file():
            raise FileNotFoundError(f"scldm {label} not found: {p}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module = _build_module(config_path, ckpt_path, dev)

    genes_df = pd.read_parquet(parquet_path)
    vocab_ids = genes_df["feature_id"].astype(str).tolist()
    vocab_names = genes_df["feature_name"].astype(str).tolist()
    n_vocab = len(vocab_ids)

    X_full, hits, old_to_vocab = _align_expression_to_vocab(adata, vocab_ids, vocab_names)
    if hits == 0:
        raise RuntimeError(
            "scldm: no adata gene matched the Census vocabulary via Ensembl_ID nor feature_name."
        )

    if input_is_log1p:
        # Undo benchmark-level log1p; scldm's encoder re-applies log1p internally.
        X_full = np.expm1(np.clip(X_full, 0.0, None)).astype(np.float32, copy=False)

    protected_rows: List[List[int]] = []
    pv_raw = adata.obsm.get("pert_var_idx", None)
    pert_present = pv_raw is not None
    if force_pert and pv_raw is not None:
        pv = np.asarray(pv_raw, dtype=np.int64)
        for i in range(adata.n_obs):
            row = pv[i] if i < pv.shape[0] else []
            mapped: List[int] = []
            seen: set[int] = set()
            for x in np.asarray(row).ravel():
                j = int(x)
                if j < 0 or j >= len(old_to_vocab):
                    continue
                jj = int(old_to_vocab[j])
                if jj < 0 or jj in seen:
                    continue
                seen.add(jj)
                mapped.append(jj)
            protected_rows.append(mapped)
    else:
        protected_rows = [[] for _ in range(adata.n_obs)]

    genes_subset_np, counts_subset_np = _build_expressed_subset(
        X_full,
        genes_seq_len,
        protected_vocab_indices=protected_rows,
    )

    n_cells = adata.n_obs
    # Full-vector tokens: 1..n_vocab (mask_idx=0 kept for the subset padding).
    genes_full_row = torch.arange(1, n_vocab + 1, dtype=torch.long)

    vae_model = module.vae_model
    vae_model.eval()
    latent_dim = vae_model.encoder.latent_dim * vae_model.encoder.latent_embedding  # 256*16
    out = np.zeros((n_cells, latent_dim), dtype=np.float32)

    use_amp = dev.type == "cuda"
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        counts_t = torch.from_numpy(X_full[start:end]).to(dev)
        genes_t = genes_full_row.unsqueeze(0).expand(end - start, -1).to(dev)
        counts_sub_t = torch.from_numpy(counts_subset_np[start:end]).to(dev)
        genes_sub_t = torch.from_numpy(genes_subset_np[start:end]).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            z = vae_model.encode(counts_t, genes_t, counts_sub_t, genes_sub_t)
        out[start:end] = z.flatten(start_dim=1).float().cpu().numpy()

    meta: dict = {
        "encoder_role": "ExpressionOnlyEncoder",
        "latent_dim": int(latent_dim),
        "hidden_dim": int(latent_dim),
        "vocab_hits": int(hits),
        "vocab_size": int(n_vocab),
        "force_pert": bool(force_pert),
        "pert_var_idx_present": bool(pert_present),
        # Protected-gene coverage for the expressed-subset path (not a perturbation condition).
        "force_pert_effective": bool(force_pert and pert_present),
        "pert_source": "obsm_pert_var_idx" if pert_present else None,
        "input_is_log1p": bool(input_is_log1p),
    }
    if force_pert and pert_present:
        m = np.asarray(pv_raw, dtype=np.int32)
        counts_per_cell = [int(np.sum(row >= 0)) for row in m]
        meta["pert_kept_histogram"] = histogram_pert_kept(counts_per_cell)
    return out, meta
