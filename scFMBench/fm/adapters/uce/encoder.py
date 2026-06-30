"""
UCE cell embedding adapter (latent_bench expression-only protocol).

Tokenizer input is ``adata.X`` only. When ``obsm['pert_var_idx']`` exists and
``force_pert=True``, row entries are mapped to per-cell **protected index sets**
so top-K / sentence packing retains those genes; they are **not** a separate
condition input to the transformer. This adapter does **not** parse
``obs['perturbation']`` or other metadata (unlike optional helpers in UCE
third_party ``encode_adata``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import torch
import paths
from scipy.sparse import csr_matrix, issparse

from .._common import histogram_pert_kept




def _load_uce_inference_module():
    uce_root = paths.third_party_root() / "uce"
    uce_exp = uce_root / "exp_emb" / "uce_inference.py"
    if not uce_exp.is_file():
        raise FileNotFoundError(f"UCE inference not found: {uce_exp}")
    os.environ.setdefault("COUPLEDFM_ROOT", str(paths.delivery_root()))
    os.environ.setdefault("COUPLEDFM_UCE_ROOT", str(paths.pretrained_root() / "uce"))
    os.environ["COUPLEDFM_UCE_SRC"] = str(uce_root)
    spec = importlib.util.spec_from_file_location("latent_bench_uce_inference", uce_exp)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def encode(
    adata: ad.AnnData,
    *,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    batch_size: int = 32,
    species: str = "human",
    model_ckpt: Optional[Path] = None,
    token_file: Optional[Path] = None,
    spec_chrom_csv: Optional[Path] = None,
    species_offsets_pkl: Optional[Path] = None,
    pe_dir: Optional[Path] = None,
    n_collate_workers: int = 0,
    show_progress: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Returns:
        embeddings (n_cells, 1280) float32
        meta dict may include ``pert_kept_histogram`` when protected coverage is active

    Notes:
        * UCE's ``build_cell_sentence`` is strict rank-based
          (``argsort(-counts)``) and the forward pass L2-normalizes each token
          embedding, so the output is invariant to any positive monotone
          transform of ``counts``. ``input_is_log1p`` is therefore a no-op here
          (kept for API uniformity).
    """
    mod = _load_uce_inference_module()
    UCEInference = mod.UCEInference
    DEFAULT_MODEL_CKPT = mod.DEFAULT_MODEL_CKPT
    DEFAULT_TOKEN_FILE = mod.DEFAULT_TOKEN_FILE
    DEFAULT_CHROM_CSV = mod.DEFAULT_CHROM_CSV
    DEFAULT_OFFSET_PKL = mod.DEFAULT_OFFSET_PKL
    DEFAULT_PE_DIR = mod.DEFAULT_PE_DIR

    inf = UCEInference(
        species=species,
        model_ckpt=model_ckpt or DEFAULT_MODEL_CKPT,
        token_file=token_file or DEFAULT_TOKEN_FILE,
        spec_chrom_csv=spec_chrom_csv or DEFAULT_CHROM_CSV,
        species_offsets_pkl=species_offsets_pkl or DEFAULT_OFFSET_PKL,
        pe_dir=pe_dir or DEFAULT_PE_DIR,
    )

    n_cells = adata.n_obs
    var_names = adata.var_names.tolist()
    gl = inf.build_gene_lookup(var_names)

    pert_var_idx = adata.obsm.get("pert_var_idx", None)

    # Per-cell protected gene column indices for tokenizer coverage (not a condition stream).
    protected_sets_per_cell: list[Optional[list[int]]] = [None] * n_cells
    meta: dict = {}
    if force_pert and pert_var_idx is not None:
        kept = []
        for k in range(n_cells):
            row = np.asarray(pert_var_idx[k]).ravel()
            ordered = [int(g) for g in row if int(g) >= 0]
            if ordered:
                counts_k = np.asarray(
                    adata.X[k].todense() if issparse(adata.X) else adata.X[k],
                    dtype=np.float32,
                ).ravel()
                ordered = sorted(
                    dict.fromkeys(ordered),
                    key=lambda j: (-float(counts_k[j]), int(j)),
                )
            protected_sets_per_cell[k] = ordered if ordered else None
            kept.append(len(ordered))
        meta["pert_kept_histogram"] = histogram_pert_kept(kept)

    if not issparse(adata.X):
        X_csr = csr_matrix(adata.X)
    else:
        X_csr = adata.X.tocsr()

    tokenizer = inf.tokenizer
    from torch.utils.data import DataLoader, Dataset

    class _CellDataset(Dataset):
        def __init__(self, X, gloc, tok, bps):
            self.X = X
            self.gl = gloc
            self.tok = tok
            self.bps = bps

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            row = self.X.getrow(idx)
            counts = np.asarray(row.todense(), dtype=np.float32).ravel()
            pset = self.bps[idx]
            sent, msk, _ = self.tok.build_cell_sentence(counts, self.gl, pset)
            return (torch.from_numpy(sent.astype(np.int64)), torch.from_numpy(msk))

    dataset = _CellDataset(X_csr, gl, tokenizer, protected_sets_per_cell)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_collate_workers,
        prefetch_factor=(2 if n_collate_workers > 0 else None),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(n_collate_workers > 0),
    )

    all_embs = []
    for tokens_b, masks_b in loader:
        emb = inf.encode_batch(tokens_b, masks_b)
        all_embs.append(emb)

    out = np.concatenate(all_embs, axis=0).astype(np.float32)
    meta["input_is_log1p"] = bool(input_is_log1p)
    meta["force_pert"] = bool(force_pert)
    meta["pert_var_idx_present"] = pert_var_idx is not None
    meta["force_pert_effective"] = bool(force_pert and pert_var_idx is not None)
    meta["encoder_role"] = "ExpressionOnlyEncoder"
    meta["pert_source"] = "obsm_pert_var_idx" if pert_var_idx is not None else None
    meta["hidden_dim"] = int(out.shape[1])
    return out, meta
