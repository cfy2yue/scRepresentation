"""
scFoundation cell-embedding adapter (expression-only).

Follows the official ``get_embedding.py`` path for ``output_type=cell``:
``gatherData`` → ``token_emb`` + ``pos_emb`` → ``encoder`` → official pooling.

Benchmark protocol: only ``adata.X`` is encoded. ``obsm['pert_var_idx']`` may
force gene columns into the ``gatherData`` selection mask when ``force_pert=True``,
so zero-expression protected genes are not dropped (upstream uses
``value_labels = pretrain_gene_x > 0`` only).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch
import paths

from .._common import histogram_pert_kept




def _ensure_scfoundation_model_dir() -> Path:
    model_dir = paths.third_party_root() / "scFoundation" / "model"
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    return model_dir


def _default_ckpt() -> str:
    return str(
        os.environ.get(
            "LATENT_BENCH_SCFOUNDATION_CKPT",
            str(paths.pretrained_root() / "scFoundation" / "models.ckpt"),
        )
    )


def _default_gene_tsv(model_dir: Path) -> str:
    return str(
        os.environ.get(
            "LATENT_BENCH_SCFOUNDATION_GENE_TSV",
            str(model_dir / "OS_scRNA_gene_index.19264.tsv"),
        )
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


def _main_gene_selection(X_df: pd.DataFrame, gene_list: List[str]) -> pd.DataFrame:
    """Same as upstream ``main_gene_selection``; local copy because ``load.py`` omits ``import pandas``."""
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(
        np.zeros((X_df.shape[0], len(to_fill_columns))),
        columns=to_fill_columns,
        index=X_df.index,
    )
    X_df = pd.DataFrame(
        np.concatenate([X_df.values, padding_df.values], axis=1),
        index=X_df.index,
        columns=list(X_df.columns) + list(padding_df.columns),
    )
    return X_df[gene_list]


def _adata_to_dataframe(adata: ad.AnnData) -> pd.DataFrame:
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    cols = list(adata.var_names.astype(str))
    idx = list(adata.obs_names.astype(str))
    return pd.DataFrame(X, index=idx, columns=cols)


def _build_pretrain_gene_x_singlecell(
    row: pd.Series,
    *,
    pre_normalized: str,
    tgthighres: str,
    device: torch.device,
) -> torch.Tensor:
    if pre_normalized == "F":
        vals = row.to_numpy(dtype=np.float64, copy=True)
        s = float(vals.sum()) + 1e-12
        tmpdata = np.log1p(vals / s * 1e4).tolist()
        totalcount = float(vals.sum())
    elif pre_normalized == "T":
        vals = row.to_numpy(dtype=np.float64, copy=True)
        tmpdata = vals.tolist()
        totalcount = float(vals.sum())
    elif pre_normalized == "A":
        arr = row.to_numpy(dtype=np.float64, copy=True)
        tmpdata = arr[:-1].tolist()
        totalcount = float(arr[-1])
    else:
        raise ValueError("pre_normalized must be F, T, or A")

    if tgthighres[0] == "f":
        tail = [np.log10(totalcount * float(tgthighres[1:])), np.log10(totalcount)]
    elif tgthighres[0] == "a":
        tail = [np.log10(totalcount) + float(tgthighres[1:]), np.log10(totalcount)]
    elif tgthighres[0] == "t":
        tail = [float(tgthighres[1:]), np.log10(totalcount)]
    else:
        raise ValueError("tgthighres must start with f, a, or t")

    vec = tmpdata + tail
    return torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)


def _protected_positions_in_gene_list(
    adata: ad.AnnData,
    prot_var_idx: Set[int],
    gene_list: List[str],
) -> List[int]:
    sym_to_pos = {str(g): i for i, g in enumerate(gene_list)}
    out: List[int] = []
    vn = list(adata.var_names.astype(str))
    for j in prot_var_idx:
        if j < 0 or j >= len(vn):
            continue
        sym = vn[j]
        p = sym_to_pos.get(sym)
        if p is not None:
            out.append(int(p))
    return sorted(set(out))


def _load_model_to_device(ckpt_path: str, key: str, device: torch.device):
    """Like ``load.load_model_frommmf`` but supports CPU and avoids unconditional ``.cuda()``."""
    from load import convertconfig
    from pretrainmodels import select_model

    try:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt_path, map_location="cpu")
    model_data = raw[key]
    model_data = convertconfig(model_data)
    config = model_data["config"]
    if not config.__contains__("qv_dim"):
        if config.get("model") != "mae_autobin":
            if config.__contains__("dim_head"):
                config["qv_dim"] = config["dim_head"]
            else:
                config["qv_dim"] = 64
    if not config.__contains__("ppi_edge"):
        config["ppi_edge"] = None
    model = select_model(config)
    model.load_state_dict(model_data["model_state_dict"])
    return model.to(device).eval(), config


def _encode_one_cell(
    model: torch.nn.Module,
    *,
    pad_id: int,
    gene_list: List[str],
    adata: ad.AnnData,
    row: pd.Series,
    global_obs_index: int,
    pert_var_idx: Optional[np.ndarray],
    has_pert_matrix: bool,
    force_pert: bool,
    pre_normalized: str,
    tgthighres: str,
    pool_type: str,
    dev: torch.device,
    gatherData: Callable[..., Any],
) -> np.ndarray:
    pretrain_gene_x = _build_pretrain_gene_x_singlecell(
        row, pre_normalized=pre_normalized, tgthighres=tgthighres, device=dev
    )
    data_gene_ids = torch.arange(19266, device=dev, dtype=torch.long).unsqueeze(0)

    value_labels = pretrain_gene_x > 0
    if force_pert and has_pert_matrix and pert_var_idx is not None:
        prot_set = {int(x) for x in np.asarray(pert_var_idx[global_obs_index]).ravel() if int(x) >= 0}
        for p in _protected_positions_in_gene_list(adata, prot_set, gene_list):
            value_labels[0, p] = True

    x, x_padding = gatherData(pretrain_gene_x, value_labels, pad_id)
    position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pad_id)

    with torch.no_grad():
        x_tok = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_emb = model.pos_emb(position_gene_ids.long())
        x_tok = x_tok + position_emb
        geneemb = model.encoder(x_tok, x_padding)

        geneemb1 = geneemb[:, -1, :]
        geneemb2 = geneemb[:, -2, :]
        geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
        geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
        if pool_type == "all":
            geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)
        elif pool_type == "max":
            geneembmerge, _ = torch.max(geneemb, dim=1)
        else:
            raise ValueError("pool_type must be all or max")

    return geneembmerge.detach().float().cpu().numpy().reshape(-1)


def encode(
    adata: ad.AnnData,
    *,
    checkpoint: Optional[str] = None,
    gene_tsv: Optional[str] = None,
    version: str = "ce",
    pool_type: str = "all",
    tgthighres: str = "t4",
    pre_normalized: Optional[str] = None,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    device: Optional[str] = None,
    show_progress: bool = False,
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Encode cells with scFoundation official **cell** embedding path.

    Args:
        adata: Gene symbols (or consistent names) in ``var_names``; values in ``X``.
        checkpoint: Path to ``models.ckpt`` (multi-key pickle).
        gene_tsv: Path to ``OS_scRNA_gene_index.19264.tsv``.
        version: ``ce`` (key ``cell``) or ``rde`` (key ``rde``) — official ``get_embedding.py`` convention.
        pool_type: ``all`` or ``max`` (cell mode only).
        tgthighres: T/S token construction (see upstream README).
        pre_normalized: ``F`` / ``T`` / ``A``; default ``T`` if ``input_is_log1p`` else ``F``.
        force_pert: If True and ``obsm['pert_var_idx']`` exists, OR protected gene
            positions into the ``gatherData`` mask (coverage only).
        input_is_log1p: If True, treat ``X`` as normalized+log1p (``pre_normalized='T'``).
        device: Torch device string; default cuda if available.
        chunk_size: Subset ``adata.X`` into chunks of this many cells when building the
            expression matrix (``None`` ⇒ one chunk, previous behavior). Use 256–1024 for large runs.
    """
    model_dir = _ensure_scfoundation_model_dir()
    from load import gatherData

    ckpt = checkpoint or _default_ckpt()
    gtsv = gene_tsv or _default_gene_tsv(model_dir)
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"scFoundation checkpoint not found: {ckpt}")
    if not Path(gtsv).is_file():
        raise FileNotFoundError(f"scFoundation gene TSV not found: {gtsv}")

    if pre_normalized is None:
        pre_normalized = "T" if input_is_log1p else "F"
    if version == "ce":
        ckpt_key = "cell"
    elif version == "rde":
        ckpt_key = "rde"
    else:
        raise ValueError("version must be 'ce' or 'rde'")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, pretrainconfig = _load_model_to_device(ckpt, ckpt_key, dev)
    pad_id = pretrainconfig["pad_token_id"]

    gene_list_df = pd.read_csv(gtsv, header=0, delimiter="\t")
    gene_list = list(gene_list_df["gene_name"])

    n_obs = int(adata.n_obs)
    pert_var_idx = adata.obsm.get("pert_var_idx", None)
    pert_var_idx = _pad_pert_var_idx(pert_var_idx, n_obs)
    has_pert_matrix = pert_var_idx is not None

    kept_per_cell: List[int] = []
    if has_pert_matrix:
        for i in range(n_obs):
            row = pert_var_idx[i] if i < pert_var_idx.shape[0] else []
            prot = {int(x) for x in np.asarray(row).ravel() if int(x) >= 0}
            gnames = list(adata.var_names.astype(str))
            n_ok = 0
            sym_set = set(gene_list)
            for j in prot:
                if 0 <= j < len(gnames) and gnames[j] in sym_set:
                    n_ok += 1
            kept_per_cell.append(n_ok)

    csize = int(n_obs if chunk_size is None else max(1, chunk_size))

    meta: dict = {
        "encoder_role": "ExpressionOnlyEncoder",
        "input_is_log1p": bool(input_is_log1p),
        "pre_normalized": pre_normalized,
        "version": version,
        "pool_type": pool_type,
        "tgthighres": tgthighres,
        "force_pert": bool(force_pert),
        "pert_source": "obsm_pert_var_idx" if has_pert_matrix else None,
        "force_pert_effective": bool(force_pert and has_pert_matrix),
    }
    if has_pert_matrix:
        meta["pert_kept_histogram"] = histogram_pert_kept(kept_per_cell)

    meta["chunk_size_cells"] = int(csize)

    out_rows: List[np.ndarray] = []
    for start in range(0, n_obs, csize):
        end = min(start + csize, n_obs)
        sub = adata[start:end]
        df = _adata_to_dataframe(sub)
        if df.shape[1] != 19264 or list(df.columns.astype(str)) != gene_list:
            df = _main_gene_selection(df, gene_list)
        if df.shape[1] != 19264:
            raise ValueError(f"Expected 19264 genes after alignment, got {df.shape[1]}")

        for local_i in range(end - start):
            gidx = start + local_i
            if show_progress and gidx % max(1, n_obs // 10) == 0:
                print(f"[scFoundation] {gidx}/{n_obs}", flush=True)

            out_rows.append(
                _encode_one_cell(
                    model,
                    pad_id=pad_id,
                    gene_list=gene_list,
                    adata=adata,
                    row=df.iloc[local_i],
                    global_obs_index=gidx,
                    pert_var_idx=pert_var_idx,
                    has_pert_matrix=has_pert_matrix,
                    force_pert=force_pert,
                    pre_normalized=pre_normalized,
                    tgthighres=tgthighres,
                    pool_type=pool_type,
                    dev=dev,
                    gatherData=gatherData,
                )
            )

    emb = np.stack(out_rows, axis=0).astype(np.float32, copy=False)
    meta["hidden_dim"] = int(emb.shape[1])
    return emb, meta


def encode_to_memmap(
    adata: ad.AnnData,
    memmap_path: Path | str,
    *,
    checkpoint: Optional[str] = None,
    gene_tsv: Optional[str] = None,
    version: str = "ce",
    pool_type: str = "all",
    tgthighres: str = "t4",
    pre_normalized: Optional[str] = None,
    force_pert: bool = True,
    input_is_log1p: bool = True,
    device: Optional[str] = None,
    chunk_size: int = 512,
    show_progress: bool = False,
    progress_logger: Optional[Any] = None,
) -> Tuple[np.memmap, dict]:
    """
    Encode like ``encode`` but stream rows to a float32 memmap of shape ``(n_obs, hidden_dim)``.
    The caller should detach ``obsm['emb']`` views, flush, and unlink ``meta['memmap_path']`` after IO.
    """
    mmap_p = Path(memmap_path)
    model_dir = _ensure_scfoundation_model_dir()
    from load import gatherData

    ckpt = checkpoint or _default_ckpt()
    gtsv = gene_tsv or _default_gene_tsv(model_dir)
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"scFoundation checkpoint not found: {ckpt}")
    if not Path(gtsv).is_file():
        raise FileNotFoundError(f"scFoundation gene TSV not found: {gtsv}")

    if pre_normalized is None:
        pre_normalized = "T" if input_is_log1p else "F"
    if version == "ce":
        ckpt_key = "cell"
    elif version == "rde":
        ckpt_key = "rde"
    else:
        raise ValueError("version must be 'ce' or 'rde'")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, pretrainconfig = _load_model_to_device(ckpt, ckpt_key, dev)
    pad_id = pretrainconfig["pad_token_id"]
    gene_list_df = pd.read_csv(gtsv, header=0, delimiter="\t")
    gene_list = list(gene_list_df["gene_name"])

    n_obs = int(adata.n_obs)
    pert_var_idx = adata.obsm.get("pert_var_idx", None)
    pert_var_idx = _pad_pert_var_idx(pert_var_idx, n_obs)
    has_pert_matrix = pert_var_idx is not None

    kept_per_cell: List[int] = []
    if has_pert_matrix:
        for i in range(n_obs):
            row = pert_var_idx[i] if i < pert_var_idx.shape[0] else []
            prot = {int(x) for x in np.asarray(row).ravel() if int(x) >= 0}
            gnames = list(adata.var_names.astype(str))
            n_ok = 0
            sym_set = set(gene_list)
            for j in prot:
                if 0 <= j < len(gnames) and gnames[j] in sym_set:
                    n_ok += 1
            kept_per_cell.append(n_ok)

    csize = max(1, int(chunk_size))

    sub0 = adata[:1]
    df0 = _adata_to_dataframe(sub0)
    if df0.shape[1] != 19264 or list(df0.columns.astype(str)) != gene_list:
        df0 = _main_gene_selection(df0, gene_list)
    if df0.shape[1] != 19264:
        raise ValueError(f"Expected 19264 genes after alignment, got {df0.shape[1]}")
    hid = int(
        _encode_one_cell(
            model,
            pad_id=pad_id,
            gene_list=gene_list,
            adata=adata,
            row=df0.iloc[0],
            global_obs_index=0,
            pert_var_idx=pert_var_idx,
            has_pert_matrix=has_pert_matrix,
            force_pert=force_pert,
            pre_normalized=pre_normalized,
            tgthighres=tgthighres,
            pool_type=pool_type,
            dev=dev,
            gatherData=gatherData,
        ).shape[0]
    )

    meta: dict = {
        "encoder_role": "ExpressionOnlyEncoder",
        "input_is_log1p": bool(input_is_log1p),
        "pre_normalized": pre_normalized,
        "version": version,
        "pool_type": pool_type,
        "tgthighres": tgthighres,
        "force_pert": bool(force_pert),
        "pert_source": "obsm_pert_var_idx" if has_pert_matrix else None,
        "force_pert_effective": bool(force_pert and has_pert_matrix),
        "memmap_path": str(mmap_p.resolve()),
        "chunk_size_cells": int(csize),
        "hidden_dim": hid,
    }
    if has_pert_matrix:
        meta["pert_kept_histogram"] = histogram_pert_kept(kept_per_cell)

    mmap_p.parent.mkdir(parents=True, exist_ok=True)
    out: np.memmap = np.memmap(str(mmap_p), dtype=np.float32, mode="w+", shape=(n_obs, hid))

    try:
        for start in range(0, n_obs, csize):
            end = min(start + csize, n_obs)
            sub = adata[start:end]
            df = _adata_to_dataframe(sub)
            if df.shape[1] != 19264 or list(df.columns.astype(str)) != gene_list:
                df = _main_gene_selection(df, gene_list)
            if df.shape[1] != 19264:
                raise ValueError(f"Expected 19264 genes after alignment, got {df.shape[1]}")

            for local_i in range(end - start):
                gidx = start + local_i
                if progress_logger is not None and (
                    gidx == 0 or gidx % max(1, n_obs // 20) == 0 or gidx == n_obs - 1
                ):
                    progress_logger.info("scFoundation encoded cells %d/%d", gidx + 1, n_obs)
                elif show_progress and gidx % max(1, n_obs // 10) == 0:
                    print(f"[scFoundation] {gidx}/{n_obs}", flush=True)

                out[gidx] = _encode_one_cell(
                    model,
                    pad_id=pad_id,
                    gene_list=gene_list,
                    adata=adata,
                    row=df.iloc[local_i],
                    global_obs_index=gidx,
                    pert_var_idx=pert_var_idx,
                    has_pert_matrix=has_pert_matrix,
                    force_pert=force_pert,
                    pre_normalized=pre_normalized,
                    tgthighres=tgthighres,
                    pool_type=pool_type,
                    dev=dev,
                    gatherData=gatherData,
                )
            out.flush()
    except BaseException:
        try:
            del out
        except Exception:
            pass
        mmap_p.unlink(missing_ok=True)
        raise

    return out, meta
