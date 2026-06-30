"""Lazy h5ad access: low memory; supports obsm['emb'] (preferred) or legacy 'exp_emb1'."""

import gc
from typing import Optional, Tuple

import anndata as ad
import h5py
import numpy as np
from scipy.sparse import issparse


def read_obs_meta(h5ad_path: str, read_index: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read perturbation labels (and optionally cell index) from h5ad."""
    try:
        with h5py.File(h5ad_path, "r") as f:
            if "__categories" in f["obs"]:
                cats = f["obs"]["__categories"]
                pert_codes = f["obs"]["perturbation"][:]
                pert_cats = cats["perturbation"].asstr()[:]
                perturbations = pert_cats[pert_codes]
            elif "perturbation" in f["obs"] and "categories" in f["obs"]["perturbation"]:
                pert_codes = f["obs"]["perturbation"]["codes"][:]
                pert_cats = f["obs"]["perturbation"]["categories"].asstr()[:]
                perturbations = pert_cats[pert_codes]
            else:
                perturbations = f["obs"]["perturbation"].asstr()[:]

            index_data = None
            if read_index:
                if "_index" in f["obs"]:
                    index_data = f["obs"]["_index"].asstr()[:]
                elif "index" in f["obs"]:
                    index_data = f["obs"]["index"].asstr()[:]

        return perturbations, index_data
    except Exception:
        adata = ad.read_h5ad(h5ad_path, backed="r")
        perturbations = np.array(adata.obs["perturbation"].values)
        index_data = np.array(adata.obs.index) if read_index else None
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        del adata
        gc.collect()
        return perturbations, index_data


_EMB_KEY_CANDIDATES = ("emb", "exp_emb1", "exp_emb", "X_emb", "latent")


def _resolve_emb_key(f: h5py.File) -> Optional[str]:
    if "obsm" not in f:
        return None
    g = f["obsm"]
    for k in _EMB_KEY_CANDIDATES:
        if k in g:
            return k
    return None


class LazyH5AnnData:
    """Lazy h5py handle for X and obsm embeddings (CSR-friendly)."""

    def __init__(self, h5ad_path: str, load_latent: bool = False):
        self._f = h5py.File(h5ad_path, "r")

        x_grp = self._f["X"]
        if isinstance(x_grp, h5py.Dataset):
            self._x_sparse = False
            self._x_ds = x_grp
            self.n_rows = x_grp.shape[0]
            self._n_cols = x_grp.shape[1]
        elif "data" in x_grp:
            self._x_sparse = True
            self._x_data = x_grp["data"]
            self._x_indices = x_grp["indices"]
            self._x_indptr = x_grp["indptr"][:]
            shape = x_grp.attrs.get("shape", x_grp.attrs.get("h5sparse_shape"))
            self.n_rows = int(shape[0])
            self._n_cols = int(shape[1])
        else:
            self._f.close()
            raise ValueError(f"Unknown X layout in {h5ad_path}")

        self._emb_key: Optional[str] = None
        self._z_ds = None
        if load_latent:
            self._emb_key = _resolve_emb_key(self._f)
            if self._emb_key is not None:
                self._z_ds = self._f["obsm"][self._emb_key]

    @property
    def has_latent(self) -> bool:
        return self._z_ds is not None

    @property
    def emb_key(self) -> Optional[str]:
        return self._emb_key

    def read_X_rows(self, row_indices: np.ndarray, col_mask: np.ndarray) -> np.ndarray:
        if self._x_sparse:
            return self._read_sparse_X(row_indices, col_mask)
        return self._read_dense_X(row_indices, col_mask)

    def _read_dense_X(self, row_indices, col_mask):
        order = np.argsort(row_indices)
        sorted_rows = row_indices[order]
        raw = np.asarray(self._x_ds[sorted_rows.tolist()], dtype=np.float32)
        result = raw[:, col_mask]
        out = np.empty_like(result)
        out[order] = result
        return out

    def _read_sparse_X(self, row_indices, col_mask):
        order = np.argsort(row_indices)
        sorted_rows = row_indices[order]
        starts = self._x_indptr[sorted_rows]
        ends = self._x_indptr[sorted_rows + 1]

        n_out_cols = int(col_mask.sum())
        result = np.zeros((len(sorted_rows), n_out_cols), dtype=np.float32)

        vocab_cols = np.where(col_mask)[0]
        col_remap = np.full(self._n_cols, -1, dtype=np.int32)
        col_remap[vocab_cols] = np.arange(n_out_cols, dtype=np.int32)

        _GAP = 65536
        i = 0
        while i < len(sorted_rows):
            seg_begin = i
            seg_lo = int(starts[i])
            seg_hi = int(ends[i])
            i += 1
            while i < len(sorted_rows):
                nxt_lo = int(starts[i])
                if nxt_lo > seg_hi + _GAP:
                    break
                seg_hi = max(seg_hi, int(ends[i]))
                i += 1

            if seg_lo >= seg_hi:
                continue

            chunk_vals = np.asarray(self._x_data[seg_lo:seg_hi])
            chunk_cols = np.asarray(self._x_indices[seg_lo:seg_hi])

            for j in range(seg_begin, i):
                s = int(starts[j]) - seg_lo
                e = int(ends[j]) - seg_lo
                if s >= e:
                    continue
                rc = chunk_cols[s:e]
                rv = chunk_vals[s:e]
                mapped = col_remap[rc]
                valid = mapped >= 0
                if valid.any():
                    result[j, mapped[valid]] = rv[valid].astype(np.float32)

        out = np.empty_like(result)
        out[order] = result
        return out

    def read_z_rows(self, indices: np.ndarray) -> np.ndarray:
        if self._z_ds is None:
            raise RuntimeError("No latent in file")
        order = np.argsort(indices)
        sorted_idx = indices[order]
        data = np.asarray(self._z_ds[sorted_idx.tolist()], dtype=np.float32)
        out = np.empty_like(data)
        out[order] = data
        return out

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None


# Backward alias for code that imports _LazyH5
_LazyH5 = LazyH5AnnData
