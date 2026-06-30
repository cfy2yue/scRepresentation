"""
CellNavi sparse graph encoder adapter (expression-only).

Benchmark protocol: only ``adata.X`` is encoded; ``obsm['pert_var_idx']`` may
constrain which genes enter the sparse subgraph when ``force_pert=True`` (genes
with zero count are otherwise dropped by upstream ``prepare_cell_input`` logic).

Outputs the pretrained **CLS** cell embedding from ``SparseCellNaviEncoder`` forward
(``cls_out``), not an ad-hoc pooled vector.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import anndata as ad
import numpy as np
import paths

from .._common import histogram_pert_kept




def _ensure_cellnavi_path() -> Path:
    cellnavi_root = paths.third_party_root() / "CellNavi"
    if str(cellnavi_root) not in sys.path:
        sys.path.insert(0, str(cellnavi_root))
    return cellnavi_root


# CoupledFM 统一部署的 CellNavi / NicheNet 数据根（graph、node2idx 在 Nichenet 子目录）
_CANONICAL_CELLNAVI_DATA = paths.pretrained_root() / "cellnavi" / "data"
_CANONICAL_NICHENET_DATA = _CANONICAL_CELLNAVI_DATA / "Nichenet"


def _resolve_cellnavi_file(
    explicit: Optional[str],
    env_key: str,
    canonical: Path,
    legacy: Path,
) -> str:
    """Resolve one asset path: explicit > env > canonical (if exists) > legacy (if exists) > canonical.

    If neither canonical nor legacy exists, return canonical so error messages point at the preferred location.
    """
    if explicit:
        return explicit
    env_v = os.environ.get(env_key)
    if env_v:
        return env_v
    if canonical.is_file():
        return str(canonical)
    if legacy.is_file():
        return str(legacy)
    return str(canonical)


def _row_expression_vector(adata: ad.AnnData, row_i: int) -> np.ndarray:
    import scipy.sparse as sp

    x = adata.X[row_i]
    if sp.issparse(x):
        x = np.asarray(x.todense()).ravel()
    else:
        x = np.asarray(x).ravel()
    return x.astype(np.float64, copy=False)


def _prepare_cell_input_bench(
    adata: ad.AnnData,
    row_i: int,
    vocab,
    *,
    normalize: bool = True,
    input_is_log1p: bool = True,
    protected_var_indices: Optional[Set[int]] = None,
    pseudo_count: int = 1,
) -> dict:
    """Delegate to ``cellnavi.data_provider.data_utils.prepare_cell_input_log1p`` (single source of truth)."""
    _ensure_cellnavi_path()
    from cellnavi.data_provider.data_utils import prepare_cell_input_log1p

    return prepare_cell_input_log1p(
        adata[row_i : row_i + 1],
        vocab,
        normalize=normalize,
        input_is_log1p=input_is_log1p,
        protected_var_indices=protected_var_indices,
        pseudo_count=pseudo_count,
    )


def _pad_pert_var_idx(pert_var_idx: Optional[np.ndarray], n_obs: int) -> Optional[np.ndarray]:
    if pert_var_idx is None:
        return None
    m = np.asarray(pert_var_idx, dtype=np.int64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.shape[0] < n_obs:
        pad = np.full((n_obs - int(m.shape[0]), m.shape[1]), -1, dtype=np.int64)
        m = np.vstack([m, pad])
    return m


def encode(
    adata: ad.AnnData,
    *,
    checkpoint: Optional[str] = None,
    gene_name_file: Optional[str] = None,
    nichenet_node2idx: Optional[str] = None,
    graph_pkl: Optional[str] = None,
    normalize: bool = True,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    device: Optional[str] = None,
    ratio: int = 5,
    show_progress: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Encode AnnData rows with CellNavi ``SparseCellNaviEncoder``; return CLS embeddings.

    **Precondition:** each cell must have at least one gene that is both in the CellNavi
    vocabulary and either **expressed** (rounded count > 0 after ``input_is_log1p`` handling) or
    listed in ``obsm['pert_var_idx']`` when ``force_pert=True`` and that matrix is present.
    Otherwise ``_prepare_cell_input_bench`` raises (no valid subgraph).

    Args:
        adata: Single-cell matrix; gene symbols in ``var_names`` must overlap CellNavi vocab.
        checkpoint: ``pretrain_weights.pth`` (original CellNavi encoder weights).
        gene_name_file: ``gene_name.txt`` (see path resolution below).
        nichenet_node2idx: ``Nichenet/node2idx.json`` (see path resolution below).
        graph_pkl: Pickled NicheNet graph (**required** for inference; default under ``pretrained/cellnavi/data/Nichenet``).
        normalize: If True, apply ``log1p(count / sum * 10000)`` per cell (CellNavi default).
        force_pert: If True and ``obsm['pert_var_idx']`` exists, force those genes into the subgraph
            even when the expression round-trip count is 0 (uses ``pseudo_count``).
        input_is_log1p: If True (benchmark default), undo ``log1p`` on ``X`` before integer rounding
            so we do not treat log-space values as raw counts.
        ratio: CLS ratio embedding index (upstream default 5).
        show_progress: Print cell index progress.

    Path resolution (each asset independently):
        explicit argument > ``LATENT_BENCH_CELLNAVI_*`` env >
        ``SCFM_PRETRAINED_ROOT/cellnavi/data`` (``Nichenet/`` for graph / node2idx) >
        legacy ``third_party/CellNavi`` mirror paths if the file exists there; otherwise prefer the canonical path for errors.

    Environment overrides:
        ``LATENT_BENCH_CELLNAVI_CKPT``, ``LATENT_BENCH_CELLNAVI_GENE_NAME``,
        ``LATENT_BENCH_CELLNAVI_NODE2IDX``, ``LATENT_BENCH_CELLNAVI_GRAPH_PKL``
    """
    import torch

    base = _ensure_cellnavi_path()
    from cellnavi import GeneVocab, NicheNetGraph, SparseCellNaviEncoder

    ckpt = _resolve_cellnavi_file(
        checkpoint,
        "LATENT_BENCH_CELLNAVI_CKPT",
        _CANONICAL_CELLNAVI_DATA / "pretrain" / "pretrain_weights.pth",
        base / "data" / "pretrain" / "pretrain_weights.pth",
    )
    gfile = _resolve_cellnavi_file(
        gene_name_file,
        "LATENT_BENCH_CELLNAVI_GENE_NAME",
        _CANONICAL_CELLNAVI_DATA / "gene_name.txt",
        base / "data" / "gene_name.txt",
    )
    n2i = _resolve_cellnavi_file(
        nichenet_node2idx,
        "LATENT_BENCH_CELLNAVI_NODE2IDX",
        _CANONICAL_NICHENET_DATA / "node2idx.json",
        base / "Nichenet" / "node2idx.json",
    )
    gpkl = _resolve_cellnavi_file(
        graph_pkl,
        "LATENT_BENCH_CELLNAVI_GRAPH_PKL",
        _CANONICAL_NICHENET_DATA / "graph.pkl",
        base / "Nichenet" / "graph.pkl",
    )

    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"CellNavi checkpoint not found: {ckpt}")
    if not Path(gfile).is_file():
        raise FileNotFoundError(f"CellNavi gene_name.txt not found: {gfile}")
    if not Path(n2i).is_file():
        raise FileNotFoundError(f"CellNavi node2idx.json not found: {n2i}")
    if not Path(gpkl).is_file():
        raise FileNotFoundError(
            f"CellNavi NicheNet graph.pkl not found: {gpkl}. "
            "Download or place graph.pkl next to node2idx.json (see CellNavi docs / release assets)."
        )

    vocab = GeneVocab(gfile, n2i)
    graph = NicheNetGraph(gpkl, vocab)

    model = SparseCellNaviEncoder()
    model.load_pretrained_weights(ckpt)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()

    n_obs = int(adata.n_obs)
    pert_var_idx = adata.obsm.get("pert_var_idx", None)
    pert_var_idx = _pad_pert_var_idx(pert_var_idx, n_obs)
    has_pert_matrix = pert_var_idx is not None

    kept_per_cell: List[int] = []
    if has_pert_matrix:
        for i in range(n_obs):
            row = pert_var_idx[i] if i < pert_var_idx.shape[0] else []
            prot = {int(x) for x in np.asarray(row).ravel() if int(x) >= 0}
            n_ok = 0
            gnames = list(adata.var_names)
            for j in prot:
                if 0 <= j < len(gnames) and gnames[j] in vocab:
                    n_ok += 1
            kept_per_cell.append(n_ok)

    meta: dict = {
        "input_is_log1p": bool(input_is_log1p),
        "force_pert": bool(force_pert),
        "encoder_role": "ExpressionOnlyEncoder",
        "hidden_dim": int(model.d_model),
        "normalize": bool(normalize),
        "pert_var_idx_present": bool(has_pert_matrix),
        "pert_source": "obsm_pert_var_idx" if has_pert_matrix else None,
        # Zero-count genes are dropped unless protected-gene coverage runs → effective when
        # force_pert and obsm matrix exist (same convention as scGPT / Geneformer).
        "force_pert_effective": bool(force_pert and has_pert_matrix),
    }
    if has_pert_matrix:
        meta["pert_kept_histogram"] = histogram_pert_kept(kept_per_cell)

    out_list: List[np.ndarray] = []
    for i in range(n_obs):
        if show_progress and (i % max(1, n_obs // 10) == 0):
            print(f"[CellNavi] {i}/{n_obs}", flush=True)
        prot_set: Optional[Set[int]] = None
        if force_pert and has_pert_matrix:
            row = pert_var_idx[i]
            prot_set = {int(x) for x in np.asarray(row).ravel() if int(x) >= 0}
        cell_data = _prepare_cell_input_bench(
            adata,
            i,
            vocab,
            normalize=normalize,
            input_is_log1p=input_is_log1p,
            protected_var_indices=prot_set,
        )
        gene_tokens = cell_data["gene_token_ids"].to(dev)
        expression = cell_data["expression"].to(dev)
        rawcount = cell_data["rawcount"].to(dev)
        edge_index = graph.build_edge_index(gene_tokens, device=dev)
        with torch.no_grad():
            _, cls_emb = model(gene_tokens, expression, rawcount, edge_index, ratio=ratio)
        out_list.append(cls_emb.detach().float().cpu().numpy())

    emb = np.stack(out_list, axis=0).astype(np.float32, copy=False)
    return emb, meta
