"""
Coupled FM dataset: loads raw expression + latent embeddings from biFlow_data.

Data sources (no IR):
  - control_center/{ds}.h5ad  → control pool (X + obsm['emb'] for z_ctrl)
  - gt/{ds}.h5ad              → perturbed GT (X + obsm['emb'] for z_gt)

Flow: x_t = (1-t)*x_ctrl_start + t*x_gt with x_ctrl_start sampled from control_center.
"""

import dataclasses
import gc
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import anndata as ad
import torch
from scipy.sparse import issparse, csr_matrix
from torch.utils.data import IterableDataset

from .vocab import GeneVocab
from model.utils.data.biflow_paths import (
    normalize_latent_backbone,
    resolve_biflow_control_gt_h5ad,
    resolve_gt_h5ad_for_pert_metadata,
)
from model.utils.train.time_sampling import sample_t
from model.utils.io.lazy_loader import read_obs_meta
from model.condition_emb.chempert.chem_resolver import (
    load_chemical_embed_backend,
    resolve_chemical_embeddings_for_metadata,
)
from model.condition_emb.genepert.metainfo import apply_pert_metainfo_fallback, load_dataset_metainfo
from model.condition_emb.genepert.perturbation import ConditionMetadata, PerturbationBatch
from model.condition_emb.genepert.gene_cache import GeneEmbeddingCache



def _condition_metadata_from_cond_string(cond: str) -> ConditionMetadata:
    """Parse condition string (same contract as latent perturbation helpers)."""
    return ConditionMetadata.from_obs_fields(cond, perturbation_field=None)


def _build_obs_condition_metadata_lookup(
    h5ad_path: Path,
    wanted_conds: List[str],
    *,
    chem_obs_column: str = "",
) -> Dict[str, ConditionMetadata]:
    """One ``cond -> ConditionMetadata`` map from gt h5ad ``obs`` (built once per file)."""
    if not h5ad_path.is_file():
        return {}
    try:
        import anndata as ad  # noqa: WPS433
    except ImportError:
        return {}
    lookup: Dict[str, ConditionMetadata] = {}
    try:
        ada = ad.read_h5ad(str(h5ad_path), backed="r")
        obs = ada.obs.copy()
        if hasattr(ada, "file") and ada.file is not None:
            try:
                ada.file.close()
            except Exception:
                pass
        del ada
        from model.condition_emb.genepert.h5ad_obs import (
            condition_metadata_from_obs_row,
            pick_obs_columns,
        )

        cols = pick_obs_columns(obs)
        pert_col = cols.get("perturbation")
        gene_col = cols.get("gene")
        for cond in sorted(wanted_conds):
            row_idx: Optional[int] = None
            if pert_col is not None:
                s = obs[pert_col].astype(str).values == str(cond)
                if np.any(s):
                    row_idx = int(np.argmax(s.astype(np.int32)))
            if row_idx is None and gene_col is not None:
                s = obs[gene_col].astype(str).values == str(cond)
                if np.any(s):
                    row_idx = int(np.argmax(s.astype(np.int32)))
            if row_idx is not None:
                meta = condition_metadata_from_obs_row(obs, row_idx, columns=cols)
                ccol = str(chem_obs_column or "").strip()
                if ccol and ccol in obs.columns:
                    try:
                        import pandas as pd  # noqa: WPS433

                        vcell = obs.iloc[int(row_idx)][ccol]
                        if vcell is not None and not bool(pd.isna(vcell)):
                            vs = str(vcell).strip()
                            if vs and vs.lower() not in ("nan", "none", "<na>"):
                                meta = dataclasses.replace(meta, chem_obs_value=vs)
                    except Exception:
                        pass
                lookup[cond] = meta
    except Exception:
        return {}
    return lookup


class _LazyH5:
    """Unified lazy h5py handle for X matrix and obsm embeddings.

    Memory footprint per file:
      - Sparse CSR X: only ``indptr`` loaded (~30 MB for gwps 3.8M rows).
        ``data`` and ``indices`` stay on disk, read per-batch on demand.
      - Dense X: nothing pre-loaded, rows read via h5py fancy indexing.
      - obsm/exp_emb1: nothing pre-loaded, rows read on demand.

    I/O is optimised by sorting row indices and merging adjacent read
    segments, so per-condition reads (100–500 clustered cells) typically
    result in just 1–2 contiguous disk reads.
    """

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
            self._x_indptr = x_grp["indptr"][:]          # small, keep in RAM
            shape = x_grp.attrs.get("shape", x_grp.attrs.get("h5sparse_shape"))
            self.n_rows = int(shape[0])
            self._n_cols = int(shape[1])
        else:
            self._f.close()
            raise ValueError(f"Unknown X layout in {h5ad_path}")

        self._z_ds = None
        if load_latent and "obsm" in self._f:
            og = self._f["obsm"]
            if "emb" in og:
                self._z_ds = og["emb"]
            elif "exp_emb1" in og:
                self._z_ds = og["exp_emb1"]

    @property
    def has_latent(self) -> bool:
        return self._z_ds is not None

    # ── X rows ────────────────────────────────────────────────────

    def read_X_rows(self, row_indices: np.ndarray,
                    col_mask: np.ndarray) -> np.ndarray:
        """Read rows from X, apply col_mask, return dense float32."""
        n_out_cols = int(np.asarray(col_mask, dtype=bool).sum())
        if self._n_cols == 1 and n_out_cols > 1:
            return np.zeros((len(row_indices), n_out_cols), dtype=np.float32)
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
        """Read sparse CSR rows with merged-segment I/O."""
        order = np.argsort(row_indices)
        sorted_rows = row_indices[order]

        starts = self._x_indptr[sorted_rows]
        ends = self._x_indptr[sorted_rows + 1]

        n_out_cols = int(col_mask.sum())
        result = np.zeros((len(sorted_rows), n_out_cols), dtype=np.float32)

        vocab_cols = np.where(col_mask)[0]
        col_remap = np.full(self._n_cols, -1, dtype=np.int32)
        col_remap[vocab_cols] = np.arange(n_out_cols, dtype=np.int32)

        _GAP = 65536       # merge segments within 256 KB gap
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

    # ── obsm rows (latent embeddings) ─────────────────────────────

    def read_z_rows(self, indices: np.ndarray) -> np.ndarray:
        """Read obsm/exp_emb1 rows on demand (sorted-index I/O)."""
        order = np.argsort(indices)
        sorted_idx = indices[order]
        data = np.asarray(self._z_ds[sorted_idx.tolist()], dtype=np.float32)
        out = np.empty_like(data)
        out[order] = data
        return out

    # ── lifecycle ─────────────────────────────────────────────────

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None


class _DatasetHandle:
    """control_center + gt (no IR / no perturbed h5ad)."""

    def __init__(
        self,
        ds_name: str,
        cc_path: str,
        gt_path: str,
        vocab: GeneVocab,
        load_latent: bool = False,
        de_gene_list: Optional[List[str]] = None,
        nichenet_graph: Optional[object] = None,
    ):
        self.ds_name = ds_name

        cc = ad.read_h5ad(cc_path)
        var_names = list(cc.var_names)
        self.G = len(var_names)

        self.gene_ids = np.array(
            [vocab.gene2token.get(g, -1) for g in var_names], dtype=np.int64
        )
        self.in_vocab = self.gene_ids >= 0
        self.gene_ids_valid = torch.from_numpy(self.gene_ids[self.in_vocab])

        self.edge_index: Optional[torch.Tensor] = None
        if nichenet_graph is not None:
            self.edge_index = nichenet_graph.build_edge_index(
                self.gene_ids_valid, device="cpu", add_cls=True,
            )

        # DE genes for OT feature space (subset of var_names; need not be in CellNavi vocab).
        self.de_mask: Optional[np.ndarray] = None
        if de_gene_list:
            var_set = set(var_names)
            ordered: List[str] = []
            for g in de_gene_list:
                if g in var_set and g not in ordered:
                    ordered.append(g)
            if ordered:
                self.de_mask = np.isin(np.asarray(var_names), np.asarray(ordered))

        self.X_ctrl = self._force_dense(cc.X)
        self.n_ctrl = cc.n_obs
        self._z_cc: Optional[np.ndarray] = None
        if load_latent:
            # UCE 用 exp_emb，State 用 emb；其它历史 key 也兼容，和 LazyH5AnnData 一致
            for _k in ("emb", "exp_emb1", "exp_emb", "X_emb", "latent"):
                if _k in cc.obsm:
                    self._z_cc = np.asarray(cc.obsm[_k], dtype=np.float32)
                    break
        del cc
        gc.collect()

        self.cc_pool = np.arange(self.n_ctrl, dtype=np.int64)
        self.pert_ctrl_map = np.arange(self.n_ctrl, dtype=np.int64)

        gt_labels, _ = read_obs_meta(gt_path, read_index=False)
        self._gt_h5 = _LazyH5(gt_path, load_latent=load_latent)
        self.n_gt = self._gt_h5.n_rows
        self.n_pert = self.n_ctrl  # legacy name: pool size

        self.gt_cond2idx: Dict[str, np.ndarray] = {}
        for cond in np.unique(gt_labels):
            self.gt_cond2idx[cond] = np.where(gt_labels == cond)[0]

        self.gt_cond2idx = {k: v for k, v in self.gt_cond2idx.items() if k != "control"}
        self.pert_cond2idx = {c: self.cc_pool for c in self.gt_cond2idx}

        self.conditions = sorted(
            set(self.pert_cond2idx) & set(self.gt_cond2idx)
        )

        del gt_labels
        gc.collect()

        try:
            import psutil
            proc = psutil.Process()
            rss_gb = proc.memory_info().rss / (1024 ** 3)
            print(f"    [{ds_name}] RSS={rss_gb:.1f}GB", end=" ", flush=True)
        except ImportError:
            pass

        if self.de_mask is not None:
            n_de = int(self.de_mask.sum())
            n_de_vocab = int((self.de_mask & self.in_vocab).sum())
            print(
                f"de_genes={n_de}  de∩vocab={n_de_vocab}",
                end=" ",
                flush=True,
            )

    def get_pert_rows(self, indices: np.ndarray, col_mask: np.ndarray) -> np.ndarray:
        """Expression from control_center (flow start x0)."""
        return self.X_ctrl[indices][:, col_mask]

    def get_gt_rows(self, indices: np.ndarray, col_mask: np.ndarray) -> np.ndarray:
        return self._gt_h5.read_X_rows(indices, col_mask)

    def get_z_src_rows(self, indices: np.ndarray) -> np.ndarray:
        """Latent z for control-pool cells used as flow source (same storage as legacy z_ctrl)."""
        if self._z_cc is None:
            raise RuntimeError("no z_ctrl")
        return self._z_cc[indices]

    def get_z_ir_rows(self, indices: np.ndarray) -> np.ndarray:
        import warnings
        warnings.warn(
            "get_z_ir_rows is deprecated; use get_z_src_rows",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_z_src_rows(indices)

    def get_z_gt_rows(self, indices: np.ndarray) -> np.ndarray:
        return self._gt_h5.read_z_rows(indices)

    def get_de_ctrl_rows(self, indices: np.ndarray) -> np.ndarray:
        if self.de_mask is None:
            raise RuntimeError("no DE mask for OT feature space")
        return self.X_ctrl[indices][:, self.de_mask]

    def get_de_gt_rows(self, indices: np.ndarray) -> np.ndarray:
        if self.de_mask is None:
            raise RuntimeError("no DE mask for OT feature space")
        return self._gt_h5.read_X_rows(indices, self.de_mask)

    @property
    def has_latent(self) -> bool:
        return self._z_cc is not None and self._gt_h5.has_latent

    @property
    def has_de(self) -> bool:
        return self.de_mask is not None and bool(self.de_mask.any())

    def ctrl_mean_gene(self) -> np.ndarray:
        return self.X_ctrl[:, self.in_vocab].mean(axis=0).astype(np.float32)

    def compute_gt_mean_gene(self, max_cells: int = 200_000) -> np.ndarray:
        all_idx = np.concatenate(list(self.gt_cond2idx.values())).astype(np.int64)
        if len(all_idx) > max_cells:
            rng = np.random.default_rng(42)
            all_idx = rng.choice(all_idx, max_cells, replace=False)
        x = self._gt_h5.read_X_rows(all_idx, self.in_vocab)
        return x.mean(axis=0).astype(np.float32)

    def compute_gt_mean_gene_cond(
        self, cond: str, max_cells: int = 200_000,
    ) -> np.ndarray:
        idx = self.gt_cond2idx.get(cond)
        if idx is None or len(idx) == 0:
            return self.ctrl_mean_gene()
        ii = idx.astype(np.int64)
        if len(ii) > max_cells:
            rng = np.random.default_rng(42)
            ii = rng.choice(ii, max_cells, replace=False)
        x = self._gt_h5.read_X_rows(ii, self.in_vocab)
        return x.mean(axis=0).astype(np.float32)

    def compute_dx_prior_gene(
        self, cond: str, max_cells: int = 200_000,
    ) -> np.ndarray:
        gtm = self.compute_gt_mean_gene_cond(cond, max_cells=max_cells)
        cm = self.ctrl_mean_gene()
        return (gtm - cm).astype(np.float32)

    def close(self):
        self._gt_h5.close()

    @staticmethod
    def _force_dense(X) -> np.ndarray:
        if issparse(X):
            return np.asarray(X.toarray(), dtype=np.float32)
        return np.asarray(X, dtype=np.float32)


class CoupledFMDataset(IterableDataset):
    """Yields per-step training data with mode-dependent contents.

    Modes:
      - ``baseline``: random pairing, no latent
      - ``ot``: OT-paired in latent space, no latent output
      - ``coupled``: OT-paired, yields latent z_t (or z_src for ODE mode)

    **Batch tuple layouts** (``__iter__``):

    *Default* (no raw perturbation batch; ``use_raw_pert_condition=False``)::

        (x_t, x_ctrl, t, gene_ids_valid, dx_t, gene_mask, pert_idx,
         edge_index, ds_name, cond, dx_prior_t, latent_data)

    **With perturbation conditioning** (``use_raw_pert_condition=True`` and cache dir set)::

        ( ..., dx_prior_t, perturbation_batch, latent_data )

    where ``perturbation_batch`` is ``PerturbationBatch.as_tuple_full()`` (**7-tuple**;
    ``chem_emb`` / ``chem_mask`` commonly ``None`` when ``pert_chem_enabled=False``).
    The last slot is always ``latent_data`` (``None`` for baseline / ot, or tensors for coupled).
    """

    def __init__(
        self,
        biflow_dir: str,
        vocab: GeneVocab,
        split: Dict[str, Dict[str, List[str]]],
        mode: str = "train",
        coupling_mode: str = "coupled",
        latent_z_mode: str = "interp",
        batch_size: int = 64,
        min_cells: int = 16,
        ds_alpha: float = 0.7,
        ot_method: str = "torch_sinkhorn",
        ot_threads: int = 4,
        ot_device: Optional[str] = None,
        ot_sinkhorn_reg: float = 0.05,
        ot_sinkhorn_iter: int = 50,
        seed: int = 42,
        rank: int = 0,
        dataset_names: Optional[List[str]] = None,
        shared_handles: Optional[Dict[str, "_DatasetHandle"]] = None,
        ot_emb_cap_src: Optional[int] = None,
        ot_emb_cap_gt: Optional[int] = None,
        ot_feature: str = "latent",
        de_dir: Optional[str] = None,
        time_sampling: str = "logit_normal",
        ot_cost: str = "cosine",
        ot_sample_mode: str = "assignment",
        gene_mask_prob: float = 0.0,
        gene_mask_all_prob: float = 0.0,
        gene_budget_manifest_path: str = "",
        gene_budget_label: str = "",
        use_graph: bool = False,
        nichenet_graph_pkl: str = "",
        pert_idx_mode: str = "zero",
        num_pert_ids: int = 10000,
        use_residual_flow: bool = False,
        use_raw_pert_condition: bool = False,
        max_pert_genes: int = 16,
        pert_gene_emb_cache_dir: str = "",
        use_h5ad_pert_metadata: bool = False,
        pert_metainfo_path: str = "",
        chemical_metainfo_path: str = "",
        chem_emb_source_dir: str = "",
        chem_obs_column: str = "",
        drug_emb_cache_dir: str = "",
        max_chem_keys: int = 4,
        chem_fallback_embed_dim: int = 512,
        latent_backbone: str = "state",
        pert_chem_enabled: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.ot_emb_cap_src = ot_emb_cap_src
        self.ot_emb_cap_gt = ot_emb_cap_gt
        self._cap_src_eff = ot_emb_cap_src if ot_emb_cap_src is not None else batch_size
        self._cap_gt_eff = ot_emb_cap_gt if ot_emb_cap_gt is not None else batch_size
        self.time_sampling = time_sampling
        self.ot_cost = ot_cost
        self.ot_sample_mode = (ot_sample_mode or "assignment").lower()
        self.gene_mask_prob = float(gene_mask_prob)
        self.gene_mask_all_prob = float(gene_mask_all_prob)
        self.gene_budget_manifest_path = str(gene_budget_manifest_path or "").strip()
        self.gene_budget_label = str(gene_budget_label or "").strip()
        self._gene_budget_entries: Dict[str, object] = {}
        self._gene_budget_mask_cache: Dict[str, np.ndarray] = {}
        self.pert_idx_mode = (pert_idx_mode or "zero").lower()
        self.num_pert_ids = int(num_pert_ids)
        self.use_residual_flow = bool(use_residual_flow)
        self._dx_prior_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self.seed = seed
        self.rank = rank
        self.mode = mode
        self.coupling_mode = coupling_mode
        self.latent_z_mode = latent_z_mode
        self.ds_alpha = ds_alpha
        self._epoch = 0
        self.ot_feature = ot_feature
        self._latent_ot_warned: set = set()

        self._use_raw_pert = bool(use_raw_pert_condition)
        self.max_pert_genes = int(max_pert_genes)
        self._gene_cache: Optional[GeneEmbeddingCache] = None
        self._obs_cond_meta: Dict[str, Dict[str, ConditionMetadata]] = {}
        self._condition_meta_cache: Dict[Tuple[str, str], ConditionMetadata] = {}
        self._pert_metainfo: Dict[str, str] = {}
        self.chem_emb_source_dir = str(chem_emb_source_dir or "").strip()
        self.chem_obs_column = str(chem_obs_column or "").strip()
        self.drug_emb_cache_dir = str(drug_emb_cache_dir or "").strip()
        self.max_chem_keys = int(max_chem_keys)
        self.chemical_metainfo_path = str(chemical_metainfo_path or "").strip()
        self.chem_fallback_embed_dim = int(chem_fallback_embed_dim)
        self._pert_chem_enabled = bool(pert_chem_enabled)
        self._chem_embed_backend = load_chemical_embed_backend(
            self, fallback_dim=max(8, int(self.chem_fallback_embed_dim)),
        )

        use_latent = coupling_mode == "coupled" or (
            coupling_mode == "ot" and str(ot_feature or "latent").lower() == "latent"
        )
        biflow_dir = Path(biflow_dir)
        self._latent_backbone_s = normalize_latent_backbone(latent_backbone)
        biflow_p = biflow_dir

        if self._use_raw_pert:
            cached = str(pert_gene_emb_cache_dir or "").strip()
            if not cached:
                raise ValueError(
                    "CoupledFMDataset: pert_gene_emb_cache_dir is required when "
                    "use_raw_pert_condition=True"
                )
            self._gene_cache = GeneEmbeddingCache(Path(cached).expanduser())
            gene_mp = load_dataset_metainfo(pert_metainfo_path, allow_missing=True)
            chem_mp = load_dataset_metainfo(self.chemical_metainfo_path, allow_missing=True)
            self._pert_metainfo = {**gene_mp, **chem_mp}

        de_lists: Dict[str, List[str]] = {}
        if ot_feature == "de" and de_dir:
            de_root = Path(de_dir)
            for _ds in sorted(split.keys()):
                jp = de_root / f"{_ds}.json"
                if jp.exists():
                    de_lists[_ds] = json.loads(jp.read_text(encoding="utf-8"))

        # OT backend 选择：默认 GPU torch_sinkhorn，彻底消除 CPU bound。
        self._ot_pairer = None
        self._ot_method = ot_method
        # 数据集 __iter__ 运行在训练主进程，可安全使用 CUDA（不是 DataLoader worker）。
        if ot_device is not None:
            self._ot_device = torch.device(ot_device)
        else:
            self._ot_device = torch.device(
                f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
            )
        if coupling_mode in ("ot", "coupled"):
            from model.utils.data.ot_pairer import LatentOTPairer
            self._ot_pairer = LatentOTPairer(
                method=ot_method,
                num_threads=ot_threads,
                reg=ot_sinkhorn_reg,
                n_iter=ot_sinkhorn_iter,
                device=self._ot_device,
                cost_fn=ot_cost,
            )

        self._nichenet = None
        if use_graph and nichenet_graph_pkl:
            gp = Path(nichenet_graph_pkl)
            if gp.is_file():
                from model.utils.data.graph import NicheNetGraph
                self._nichenet = NicheNetGraph(str(gp), vocab)
            else:
                print(
                    f"[CoupledFMDataset] use_graph but graph missing: {gp}",
                    flush=True,
                )

        self.handles: Dict[str, _DatasetHandle] = {}
        self.ds_conds: Dict[str, List[str]] = {}

        ds_list = sorted(split.keys())
        if dataset_names:
            allowed = set(dataset_names)
            ds_list = [d for d in ds_list if d in allowed]
            print(f"[CoupledFM] filtering to {len(ds_list)} datasets: {ds_list}")
        n_total = len(ds_list)
        for di, ds_name in enumerate(ds_list):
            sp = split[ds_name]

            if shared_handles and ds_name in shared_handles:
                h = shared_handles[ds_name]
                print(f"  [{di+1}/{n_total}] reusing {ds_name} ...", end=" ", flush=True)
            else:
                pair = resolve_biflow_control_gt_h5ad(
                    biflow_p,
                    ds_name,
                    latent_backbone=self._latent_backbone_s,
                )
                if pair is None:
                    continue
                cc_p, gt_p = pair
                print(
                    f"  [{di+1}/{n_total}] loading {ds_name} | "
                    f"control={cc_p} | gt={gt_p}",
                    flush=True,
                )

                de_gl = de_lists.get(ds_name) if ot_feature == "de" else None
                h = _DatasetHandle(
                    ds_name,
                    str(cc_p), str(gt_p),
                    vocab,
                    load_latent=use_latent,
                    de_gene_list=de_gl,
                    nichenet_graph=self._nichenet,
                )

            if coupling_mode == "ot" and self.ot_feature == "de" and not h.has_de:
                if not shared_handles or ds_name not in shared_handles:
                    h.close()
                print("skipped (no DE genes / JSON)", flush=True)
                continue
            if coupling_mode == "ot" and self.ot_feature == "latent" and not h.has_latent:
                if not shared_handles or ds_name not in shared_handles:
                    h.close()
                print("skipped (no latent for OT)", flush=True)
                continue

            valid_conds = []
            for cond in sp.get(mode, []):
                if cond not in h.gt_cond2idx or cond not in h.pert_cond2idx:
                    continue
                if coupling_mode == "coupled" and not h.has_latent:
                    continue
                n_src = len(h.pert_cond2idx[cond])  # control pool size (flow source)
                n_gt = len(h.gt_cond2idx[cond])
                if n_src >= min_cells and n_gt >= min_cells:
                    valid_conds.append(cond)

            if valid_conds:
                self.handles[ds_name] = h
                self.ds_conds[ds_name] = valid_conds
                print(f"{len(valid_conds)} conds, "
                      f"ctrl={h.n_ctrl} pert={h.n_pert} gt={h.n_gt}"
                      f"{' latent=✓' if h.has_latent else ''}",
                      flush=True)
            else:
                if not shared_handles or ds_name not in shared_handles:
                    h.close()
                print("skipped (no valid conds)", flush=True)

        self.ds_names = sorted(self.ds_conds.keys())
        if self.gene_budget_manifest_path:
            self._load_gene_budget_manifest()
            for ds_name in self.ds_names:
                h = self.handles[ds_name]
                self._gene_budget_mask_for_dataset(ds_name, len(h.gene_ids_valid))
        total_conds = sum(len(v) for v in self.ds_conds.values())
        print(
            f"[CoupledFM {mode}/{coupling_mode}] "
            f"ot_feature={self.ot_feature} "
            f"{len(self.ds_names)} datasets, {total_conds} conditions"
        )

        if self._use_raw_pert and use_h5ad_pert_metadata:
            for ds_name in self.handles:
                h5_path = resolve_gt_h5ad_for_pert_metadata(
                    biflow_p,
                    ds_name,
                    latent_backbone=self._latent_backbone_s,
                )
                if h5_path is None or not h5_path.is_file():
                    continue
                h5_p = h5_path
                tr = set(split.get(ds_name, {}).get("train", []))
                te = set(split.get(ds_name, {}).get("test", []))
                wanted = sorted(tr | te)
                lk = _build_obs_condition_metadata_lookup(
                    h5_p, wanted, chem_obs_column=self.chem_obs_column,
                )
                if lk:
                    self._obs_cond_meta[ds_name] = lk

    def metadata_for_condition(self, ds_name: str, cond: str) -> ConditionMetadata:
        ck = (str(ds_name), str(cond))
        hit = self._condition_meta_cache.get(ck)
        if hit is not None:
            return hit
        ds_mp = self._obs_cond_meta.get(str(ds_name))
        if ds_mp is not None and cond in ds_mp:
            meta = ds_mp[cond]
        else:
            meta = _condition_metadata_from_cond_string(cond)
        meta = apply_pert_metainfo_fallback(
            meta,
            str(ds_name),
            self._pert_metainfo,
            use_pert_condition=self._use_raw_pert,
        )
        self._condition_meta_cache[ck] = meta
        return meta

    def enrich_metadata_with_chem(self, meta: ConditionMetadata) -> ConditionMetadata:
        if not self._pert_chem_enabled:
            return meta
        legacy = [self.chem_emb_source_dir] if self.chem_emb_source_dir else None
        vecs = resolve_chemical_embeddings_for_metadata(
            meta,
            self,
            backend=self._chem_embed_backend,
            legacy_chem_dirs=legacy,
            max_keys=self.max_chem_keys,
        )
        if not vecs:
            return meta
        return dataclasses.replace(meta, chem_emb_list=vecs, chem_emb=None)

    def perturbation_batch_tensors(
        self,
        ds_name: str,
        cond: str,
        batch_size: int,
        *,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """Build **7-tuple** perturb tensors on *device* (chem slots optional)."""
        if not self._use_raw_pert or self._gene_cache is None:
            return None
        meta = self.enrich_metadata_with_chem(self.metadata_for_condition(ds_name, cond))
        rows = [meta] * int(batch_size)
        pb = PerturbationBatch.from_metadata_list(
            rows,
            self._gene_cache,
            max_genes=self.max_pert_genes,
            max_chem_slots=int(self.max_chem_keys),
            device=device,
        )
        assert pb.combo_ids is not None
        return pb.as_tuple_full()

    def _n_eff(self, n: int) -> int:
        if self.ds_alpha >= 1.0:
            return n
        return max(1, min(int(math.ceil(n ** self.ds_alpha)), n))

    @property
    def epoch_steps(self) -> int:
        total = 0
        for ds in self.ds_names:
            h = self.handles[ds]
            n_eff = self._n_eff(len(self.ds_conds[ds]))
            ds_total = 0
            for cond in self.ds_conds[ds]:
                n_gt = len(h.gt_cond2idx[cond])
                ds_total += max(1, math.ceil(n_gt / self.batch_size))
            total += ds_total * n_eff // max(len(self.ds_conds[ds]), 1)
        return max(total, 1)

    def _load_gene_budget_manifest(self) -> None:
        path = Path(self.gene_budget_manifest_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"gene budget manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        entries = obj.get("datasets", obj) if isinstance(obj, dict) else {}
        if not isinstance(entries, dict) or not entries:
            raise ValueError(
                "gene budget manifest must contain a non-empty dataset map "
                "or top-level `datasets` object"
            )
        self._gene_budget_entries = entries
        label = self.gene_budget_label or str(obj.get("label", "")) if isinstance(obj, dict) else ""
        print(
            f"[CoupledFM] loaded gene budget manifest: {path}"
            f"{' label=' + label if label else ''}",
            flush=True,
        )

    def _gene_budget_mask_for_dataset(self, ds_name: str, n_genes: int) -> Optional[np.ndarray]:
        if not self.gene_budget_manifest_path:
            return None
        if ds_name in self._gene_budget_mask_cache:
            return self._gene_budget_mask_cache[ds_name]
        if ds_name not in self._gene_budget_entries:
            raise KeyError(f"gene budget manifest missing dataset {ds_name!r}")
        entry = self._gene_budget_entries[ds_name]
        if isinstance(entry, dict):
            keep = (
                entry.get("keep_indices")
                or entry.get("keep_gene_indices")
                or entry.get("indices")
                or entry.get("keep_idx")
            )
        else:
            keep = entry
        if keep is None:
            raise ValueError(
                f"gene budget manifest entry for {ds_name!r} must provide keep_indices"
            )
        keep_arr = np.asarray(keep, dtype=np.int64)
        if keep_arr.ndim != 1 or keep_arr.size == 0:
            raise ValueError(f"gene budget keep_indices for {ds_name!r} is empty or not 1-D")
        if keep_arr.min() < 0 or keep_arr.max() >= int(n_genes):
            raise ValueError(
                f"gene budget keep_indices for {ds_name!r} out of range for G={n_genes}"
            )
        mask = np.ones(int(n_genes), dtype=np.float32)
        mask[np.unique(keep_arr)] = 0.0
        self._gene_budget_mask_cache[ds_name] = mask
        return mask

    def budget_mask_for_eval(self, ds_name: str) -> Optional[np.ndarray]:
        if not self.gene_budget_manifest_path:
            return None
        h = self.handles[ds_name]
        return self._gene_budget_mask_for_dataset(ds_name, len(h.gene_ids_valid))

    def ot_feature_effective_label(self) -> str:
        """Resolved ``train.ot_feature`` accounting for missing latent columns in h5ad.

        When ``ot_feature == \"latent\"`` but a dataset handle lacks ``obsm`` embeddings,
        the iterator silently falls back to non-latent pairing; this label surfaces that.
        """
        req = str(self.ot_feature or "")
        if req != "latent":
            return req
        if not self.handles:
            return req
        if any(not getattr(h, "has_latent", False) for h in self.handles.values()):
            return "latent_degraded_missing_obsm_then_random_pairing"
        return "latent"

    def __iter__(self):
        ep = self._epoch
        self._epoch += 1

        iter_rng = np.random.RandomState(self.seed + ep)
        batch_rng = np.random.RandomState(self.seed * 1000 + ep * 100 + self.rank)

        all_pairs: List[Tuple[str, str]] = []
        for ds in self.ds_names:
            conds = list(self.ds_conds[ds])
            iter_rng.shuffle(conds)
            n_eff = self._n_eff(len(conds))
            selected = conds[:n_eff]
            h = self.handles[ds]
            for cond in selected:
                n_gt = len(h.gt_cond2idx[cond])
                n_visits = max(1, math.ceil(n_gt / self.batch_size))
                all_pairs.extend([(ds, cond)] * n_visits)
        iter_rng.shuffle(all_pairs)

        for ds_name, cond in all_pairs:
            h = self.handles[ds_name]
            B = self.batch_size
            in_vocab = h.in_vocab
            budget_mask_1d = self._gene_budget_mask_for_dataset(
                ds_name, len(h.gene_ids_valid),
            )
            budget_keep_1d = None
            if budget_mask_1d is not None:
                budget_keep_1d = 1.0 - budget_mask_1d
            if self._ot_pairer is None:
                use_ot = False
            elif self.ot_feature == "latent":
                use_ot = h.has_latent
                if not use_ot and ds_name not in self._latent_ot_warned:
                    self._latent_ot_warned.add(ds_name)
                    warnings.warn(
                        f"ot_feature='latent' but dataset {ds_name!r} has no latent embeddings "
                        f"in h5ad (obsm); OT pairing falls back to random (same as baseline).",
                        UserWarning,
                        stacklevel=2,
                    )
            elif self.ot_feature == "de":
                use_ot = h.has_de
            elif self.ot_feature == "raw":
                use_ot = True
            else:
                use_ot = False

            src_pool = h.pert_cond2idx[cond]
            gt_pool = h.gt_cond2idx[cond]

            if use_ot:
                # 潜空间 OT：先截断 → GPU Sinkhorn 配对 → 按 plan 抽 B 对（见 utils.data.ot_pairer）。
                # torch_sinkhorn 全程在 GPU 上进行，把原 CPU bound 的 pot.emd 彻底解除。
                cap_src, cap_gt = self._cap_src_eff, self._cap_gt_eff
                if len(src_pool) > cap_src:
                    src_sub = batch_rng.choice(len(src_pool), size=cap_src, replace=False)
                    src_ot = src_pool[src_sub]
                else:
                    src_ot = src_pool
                if len(gt_pool) > cap_gt:
                    gt_sub = batch_rng.choice(len(gt_pool), size=cap_gt, replace=False)
                    gt_ot = gt_pool[gt_sub]
                else:
                    gt_ot = gt_pool

                if self.ot_feature == "latent":
                    z_ctrl_s = h.get_z_src_rows(src_ot)
                    z_gt_s = h.get_z_gt_rows(gt_ot)
                elif self.ot_feature == "de":
                    z_ctrl_s = h.get_de_ctrl_rows(src_ot)
                    z_gt_s = h.get_de_gt_rows(gt_ot)
                else:
                    z_ctrl_s = h.get_pert_rows(src_ot, in_vocab)
                    z_gt_s = h.get_gt_rows(gt_ot, in_vocab)
                    if budget_keep_1d is not None:
                        z_ctrl_s = z_ctrl_s * budget_keep_1d[None, :]
                        z_gt_s = z_gt_s * budget_keep_1d[None, :]

                if self._ot_method == "torch_sinkhorn":
                    z_ctrl_t = torch.from_numpy(np.ascontiguousarray(z_ctrl_s)).to(
                        self._ot_device, dtype=torch.float32, non_blocking=True
                    )
                    z_gt_t = torch.from_numpy(np.ascontiguousarray(z_gt_s)).to(
                        self._ot_device, dtype=torch.float32, non_blocking=True
                    )
                    use_asg = self.ot_sample_mode == "assignment"
                    i_t, j_t = self._ot_pairer.pair_torch(
                        z_ctrl_t, z_gt_t, B,
                        cost_fn=self.ot_cost,
                        use_assignment=use_asg,
                    )
                    src_ot_idx = i_t.detach().cpu().numpy()
                    gt_ot_idx = j_t.detach().cpu().numpy()
                else:
                    use_asg_cpu = self.ot_sample_mode == "assignment"
                    src_ot_idx, gt_ot_idx = self._ot_pairer.pair(
                        z_ctrl_s, z_gt_s, B, use_assignment=use_asg_cpu,
                    )
                if len(src_ot_idx) != B or len(gt_ot_idx) != B:
                    raise RuntimeError(
                        f"OT pairing length mismatch: expected B={B}, got "
                        f"src={len(src_ot_idx)} gt={len(gt_ot_idx)} "
                        f"ds={ds_name} cond={cond} ot_mode={self.ot_sample_mode} "
                        f"cap_src={self._cap_src_eff} cap_gt={self._cap_gt_eff} "
                        f"n_src_ot={len(src_ot)} n_gt_ot={len(gt_ot)}"
                    )

                src_idx = src_ot[src_ot_idx]
                gt_idx = gt_ot[gt_ot_idx]

                z_ctrl_batch = z_ctrl_s[src_ot_idx] if self.coupling_mode == "coupled" else None
                z_gt_batch = z_gt_s[gt_ot_idx] if self.coupling_mode == "coupled" else None
            else:
                src_idx = batch_rng.choice(src_pool, size=B, replace=(len(src_pool) < B))
                gt_idx = batch_rng.choice(gt_pool, size=B, replace=(len(gt_pool) < B))
                z_ctrl_batch = None
                z_gt_batch = None

            ctrl_idx = h.pert_ctrl_map[src_idx]
            ctrl_idx = np.clip(ctrl_idx, 0, h.n_ctrl - 1)

            x0 = h.get_pert_rows(src_idx, in_vocab)
            x_ctrl_ref = h.X_ctrl[ctrl_idx][:, in_vocab]
            x_gt = h.get_gt_rows(gt_idx, in_vocab)

            t = sample_t(batch_rng, B, self.time_sampling)
            t_col = t[:, None]
            x_t = (1.0 - t_col) * x0 + t_col * x_gt
            dx_t = x_gt - x0

            G = x0.shape[1]
            gene_mask = np.zeros((B, G), dtype=np.float32)
            if self.gene_mask_prob > 0.0 or self.gene_mask_all_prob > 0.0:
                gene_mask = (batch_rng.rand(B, G) < self.gene_mask_prob).astype(
                    np.float32
                )
                for b in range(B):
                    if batch_rng.rand() < self.gene_mask_all_prob:
                        gene_mask[b, :] = 1.0
                x_t = x_t * (1.0 - gene_mask)
            if budget_mask_1d is not None:
                budget_mask = np.broadcast_to(budget_mask_1d[None, :], (B, G)).astype(
                    np.float32,
                    copy=True,
                )
                gene_mask = np.maximum(gene_mask, budget_mask)
                keep = 1.0 - gene_mask
                x_t = x_t * keep
                x_ctrl_ref = x_ctrl_ref * keep
                dx_t = dx_t * keep

            if self.pert_idx_mode == "random":
                pert_idx = batch_rng.randint(
                    0, max(1, self.num_pert_ids), size=B,
                ).astype(np.int64)
            else:
                pert_idx = np.zeros(B, dtype=np.int64)

            if self.use_residual_flow:
                dpk = (ds_name, cond)
                if dpk not in self._dx_prior_cache:
                    self._dx_prior_cache[dpk] = h.compute_dx_prior_gene(cond)
                dx_prior_t = torch.from_numpy(
                    self._dx_prior_cache[dpk].astype(np.float32, copy=True),
                )
            else:
                dx_prior_t = None

            batch = [
                torch.from_numpy(x_t),
                torch.from_numpy(x_ctrl_ref),
                torch.from_numpy(t),
                h.gene_ids_valid,
                torch.from_numpy(dx_t),
                torch.from_numpy(gene_mask),
                torch.from_numpy(pert_idx),
                h.edge_index,
                ds_name,
                cond,
                dx_prior_t,
            ]

            if self._use_raw_pert and self._gene_cache is not None:
                meta = self.enrich_metadata_with_chem(self.metadata_for_condition(ds_name, cond))
                pb = PerturbationBatch.from_metadata_list(
                    [meta] * B,
                    self._gene_cache,
                    max_genes=self.max_pert_genes,
                    max_chem_slots=int(self.max_chem_keys),
                    device=torch.device("cpu"),
                )
                assert pb.combo_ids is not None
                batch.append(pb.as_tuple_full())

            if self.coupling_mode == "coupled" and z_ctrl_batch is not None:
                if self.latent_z_mode == "interp":
                    z_t = (1.0 - t_col) * z_ctrl_batch + t_col * z_gt_batch
                    batch.append(torch.from_numpy(z_t.astype(np.float32)))
                elif self.latent_z_mode == "curriculum":
                    # 训练循环按 curriculum 概率动态选择 interp / ode；
                    # 这里同时给出两份并拼在最后一个维度上（shape: (B, 2*d_latent)）。
                    # - 前半 d_latent：interp 版本 z_t = (1-t)*z_ctrl + t*z_gt
                    # - 后半 d_latent：z_ctrl（供 latent FM ode_at_t 从 t=0 演化到 t）
                    z_t_interp = (1.0 - t_col) * z_ctrl_batch + t_col * z_gt_batch
                    merged = np.concatenate(
                        [z_t_interp.astype(np.float32),
                         z_ctrl_batch.astype(np.float32)],
                        axis=-1,
                    )
                    batch.append(torch.from_numpy(merged))
                else:  # "ode"
                    batch.append(torch.from_numpy(z_ctrl_batch.astype(np.float32)))
            else:
                batch.append(None)

            yield tuple(batch)

    def close(self):
        for h in self.handles.values():
            h.close()


__all__ = ["CoupledFMDataset"]
