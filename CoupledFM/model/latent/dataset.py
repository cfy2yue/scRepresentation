"""
Cross-dataset FM data loading.

Reads condition-sorted HDF5 files produced by prepare_fm_data.py,
performs train/test split, and provides a globally-shuffled multi-visit
sampler that covers all GT samples per epoch.
"""

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset

from model.condition_emb.chempert.chem_resolver import (
    load_chemical_embed_backend,
    resolve_chemical_embeddings_for_metadata,
)
from model.utils.conditioning.metainfo import apply_pert_metainfo_fallback, load_dataset_metainfo
from model.utils.conditioning.perturbation import ConditionMetadata, PerturbationBatch
from model.utils.embeddings.gene_cache import GeneEmbeddingCache
from model.utils.data.biflow_paths import normalize_latent_backbone, resolve_gt_h5ad_for_pert_metadata

from model.latent.perturb_helpers import condition_metadata_from_cond_string


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def _split_conditions(
    conditions: List[str],
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split conditions into train / test.

    Rules (from the plan):
      - multi-perturbation conditions (containing '+') -> test
      - single-perturbation conditions: 10% random -> test, rest -> train
    """
    multi = [c for c in conditions if "+" in c]
    single = [c for c in conditions if "+" not in c]

    rng = np.random.RandomState(seed)
    n_test = max(1, int(len(single) * test_ratio)) if single else 0
    perm = rng.permutation(len(single))
    single_test = [single[i] for i in perm[:n_test]]
    single_train = [single[i] for i in perm[n_test:]]

    train_conds = sorted(single_train)
    test_conds = sorted(single_test + multi)
    return train_conds, test_conds


def load_or_create_split(
    data_dir: str,
    manifest: dict,
    test_ratio: float = 0.1,
    seed: int = 42,
    biflow_dir: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Return ``{dataset: {train: [...], test: [...]}}``.

    **优先使用 canonical 统一 split**（``biflow_dir/split_seed{seed}.json``），
    与 coupled / raw 保持一致，避免实验对比或 finetune 时出现泄露。
    仅在 canonical 不存在时回退到旧的 latent 专用划分（基于 manifest conditions），
    并把结果写到 canonical 路径，供后续 coupled / raw 复用。
    """
    # 1) 尝试 canonical（统一真相源）
    if biflow_dir is None:
        biflow_dir = str(Path(data_dir).resolve().parent / "biFlow_data")
    canonical = Path(biflow_dir) / f"split_seed{seed}.json"
    if canonical.exists():
        with open(canonical) as f:
            full_split = json.load(f)
        # 仅保留 manifest 中实际存在的数据集和 condition。Canonical split
        # 可能来自更完整的 biFlow/raw 数据，scFMBench 子集或 quick bundle
        # 只包含其中一部分 condition；训练时必须以当前 HDF5 manifest 为准。
        out: Dict[str, Dict[str, List[str]]] = {}
        for ds, ds_meta in manifest["datasets"].items():
            allowed = set(map(str, ds_meta.get("conditions", [])))
            if ds in full_split:
                sp = full_split.get(ds, {})
                train = [c for c in sp.get("train", []) if str(c) in allowed]
                test = [c for c in sp.get("test", []) if str(c) in allowed]
            else:
                train, test = [], []
            if not train and not test and allowed:
                train, test = _split_conditions(sorted(allowed), test_ratio, seed)
            if train or test:
                out[ds] = {"train": train, "test": test}
        return out

    # 2) 旧回退路径（与历史兼容）
    legacy_path = Path(data_dir) / f"split_seed{seed}.json"
    if legacy_path.exists():
        with open(legacy_path) as f:
            return json.load(f)

    # 3) 首次生成：按 latent 的简单策略（单扰 10% test；多扰全部 test）——
    #    写到 canonical 位置，让 coupled / raw 下次直接复用；同时留一份 legacy 副本。
    split = {}
    for ds_name, ds_meta in manifest["datasets"].items():
        conds = ds_meta["conditions"]
        tr, te = _split_conditions(conds, test_ratio, seed)
        split[ds_name] = {"train": tr, "test": te}

    try:
        canonical.parent.mkdir(parents=True, exist_ok=True)
        with open(canonical, "w") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)
        print(
            f"[latent] canonical split 不存在，已根据 manifest 生成并写入 {canonical}\n"
            f"         后续 coupled / raw 训练将共享此 split。若需策略（单扰 cap=30 等）版本，"
            f"请跑 `python tools/build_split.py --biflow-dir {biflow_dir} --seed {seed}` 覆盖它。",
            flush=True,
        )
    except OSError as e:
        print(f"[latent] WARN: 写 canonical split 失败（{e}），回退到 {legacy_path}")
        with open(legacy_path, "w") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)
    return split


# ---------------------------------------------------------------------------
# Optional h5ad → condition metadata lookup (built once per dataset)
# ---------------------------------------------------------------------------


def _build_obs_condition_metadata_lookup(
    h5ad_path: Path,
    wanted_conds: List[str],
    *,
    chem_obs_column: str = "",
) -> Dict[str, ConditionMetadata]:
    """One-time map ``cond_name -> ConditionMetadata`` from biFlow-style ``obs``."""
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
        from model.utils.conditioning.h5ad_obs import condition_metadata_from_obs_row, pick_obs_columns

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


def _infer_biflow_dir(data_dir: str, explicit: Optional[str]) -> Optional[Path]:
    if explicit and str(explicit).strip():
        return Path(explicit).expanduser()
    return Path(data_dir).resolve().parent.parent / "biFlow_data"


def _metadata_from_json_entry(entry: Dict[str, Any], fallback_cond: str) -> ConditionMetadata:
    """Build condition metadata persisted by ``prepare_scfm_fm_data.py``."""
    genes_raw = entry.get("genes", None)
    if isinstance(genes_raw, (list, tuple)):
        genes = tuple(str(g).strip().upper() for g in genes_raw if str(g).strip())
    elif genes_raw is None:
        genes = condition_metadata_from_cond_string(fallback_cond).genes
    else:
        genes = condition_metadata_from_cond_string(str(genes_raw)).genes

    nperts = entry.get("nperts_obs", None)
    try:
        nperts_obs = None if nperts is None else int(nperts)
    except (TypeError, ValueError):
        nperts_obs = None

    combo = entry.get("combo_id", 0)
    try:
        combo_id = int(combo)
    except (TypeError, ValueError):
        combo_id = 0

    def _clean(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s and s.lower() not in ("nan", "none", "<na>") else None

    return ConditionMetadata(
        genes=genes,
        perturbation_type_raw=_clean(entry.get("perturbation_type_raw", entry.get("perturbation_type"))),
        combo_id=combo_id,
        nperts_obs=nperts_obs,
        chem_emb=None,
        chem_emb_list=None,
        chem_source=_clean(entry.get("chem_source")),
        chem_obs_value=_clean(entry.get("chem_obs_value")),
    )


def _load_condition_metadata_sidecar(data_dir: Path) -> Dict[str, Dict[str, ConditionMetadata]]:
    """Load optional per-condition metadata next to converted LatentFM HDF5 files."""
    path = data_dir / "condition_metadata.json"
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, Dict[str, ConditionMetadata]] = {}
    for ds_name, ds_obj in obj.items():
        if not isinstance(ds_obj, dict):
            continue
        ds_map: Dict[str, ConditionMetadata] = {}
        for cond, entry in ds_obj.items():
            if not isinstance(entry, dict):
                continue
            ds_map[str(cond)] = _metadata_from_json_entry(entry, str(cond))
        if ds_map:
            out[str(ds_name)] = ds_map
    return out


def _metadata_is_drug(meta: ConditionMetadata, ds_name: str) -> bool:
    """Return whether one condition should be treated as chemical/drug perturbation."""
    typ = str(meta.perturbation_type_raw or "").strip().lower()
    if typ in {"drug", "chemical", "compound", "small molecule", "small-molecule"}:
        return True
    if meta.chem_obs_value or meta.chem_source:
        return True
    dsl = str(ds_name).strip().lower()
    return any(tok in dsl for tok in ("sciplex", "chempert", "chemical", "drug"))


def _keep_by_perturbation_family(meta: ConditionMetadata, ds_name: str, family_filter: str) -> bool:
    filt = str(family_filter or "all").strip().lower()
    if filt in {"", "all", "any"}:
        return True
    is_drug = _metadata_is_drug(meta, ds_name)
    if filt in {"drug", "chemical", "chem"}:
        return is_drug
    if filt in {"gene", "genetic"}:
        return not is_drug
    raise ValueError(
        "perturbation_family_filter must be one of: all, gene, drug; "
        f"got {family_filter!r}"
    )


# ---------------------------------------------------------------------------
# Lightweight per-dataset handle (h5py)
# ---------------------------------------------------------------------------

class _DatasetHandle:
    """Lazy h5py reader for one dataset's HDF5."""

    def __init__(self, h5_path: str):
        self.path = h5_path
        self._f: Optional[h5py.File] = None
        self._conditions: Optional[List[str]] = None
        self._cond2idx: Optional[Dict[str, int]] = None
        self._ctrl_offsets: Optional[np.ndarray] = None
        self._gt_offsets: Optional[np.ndarray] = None
        self._ctrl_key: str = "ctrl"  # or "ir" for legacy HDF5

    def _open(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
            self._conditions = self._f["conditions"].asstr()[:].tolist()
            self._cond2idx = {c: i for i, c in enumerate(self._conditions)}
            if "ctrl/offsets" in self._f:
                self._ctrl_offsets = self._f["ctrl/offsets"][:]
                self._ctrl_key = "ctrl"
            else:
                self._ctrl_offsets = self._f["ir/offsets"][:]
                self._ctrl_key = "ir"
            self._gt_offsets = self._f["gt/offsets"][:]

    @property
    def conditions(self):
        self._open()
        return self._conditions

    def read_ctrl(self, cond: str) -> np.ndarray:
        self._open()
        idx = self._cond2idx[cond]
        off = self._ctrl_offsets
        assert off is not None
        s, e = int(off[idx]), int(off[idx + 1])
        return self._f[f"{self._ctrl_key}/emb"][s:e]

    def read_src(self, cond: str) -> np.ndarray:
        """Embeddings for the control-pool / flow-source cells for this condition."""
        return self.read_ctrl(cond)

    def _read_rows(self, key: str, start: int, rel_idx: np.ndarray) -> np.ndarray:
        """Read selected relative rows from a contiguous condition slice.

        h5py requires fancy indices to be sorted and unique.  The dataset
        sampler can request duplicate rows when sampling with replacement, so
        we read sorted unique absolute rows and reconstruct the requested order.
        """
        rel_idx = np.asarray(rel_idx, dtype=np.int64)
        if rel_idx.ndim != 1:
            raise ValueError(f"row indices must be 1D, got shape={rel_idx.shape}")
        if rel_idx.size == 0:
            ds = self._f[key]
            return np.empty((0, int(ds.shape[1])), dtype=np.float32)
        uniq, inverse = np.unique(rel_idx, return_inverse=True)
        abs_idx = uniq + int(start)
        block = self._f[key][abs_idx]
        return np.asarray(block[inverse], dtype=np.float32)

    def read_src_rows(self, cond: str, rel_idx: np.ndarray) -> np.ndarray:
        """Read selected control/source rows for ``cond`` without loading all cells."""
        self._open()
        idx = self._cond2idx[cond]
        off = self._ctrl_offsets
        assert off is not None
        s = int(off[idx])
        return self._read_rows(f"{self._ctrl_key}/emb", s, rel_idx)

    def read_ir(self, cond: str) -> np.ndarray:
        import warnings
        warnings.warn(
            "read_ir is deprecated; use read_src",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.read_src(cond)

    def read_gt(self, cond: str) -> np.ndarray:
        self._open()
        idx = self._cond2idx[cond]
        s, e = int(self._gt_offsets[idx]), int(self._gt_offsets[idx + 1])
        return self._f["gt/emb"][s:e]

    def read_gt_rows(self, cond: str, rel_idx: np.ndarray) -> np.ndarray:
        """Read selected GT rows for ``cond`` without loading all cells."""
        self._open()
        idx = self._cond2idx[cond]
        s = int(self._gt_offsets[idx])
        return self._read_rows("gt/emb", s, rel_idx)

    def cond_sizes(self, cond: str) -> Tuple[int, int]:
        """Return (n_ctrl, n_gt) for a condition without reading embeddings."""
        self._open()
        idx = self._cond2idx[cond]
        off = self._ctrl_offsets
        assert off is not None
        n_ctrl = int(off[idx + 1] - off[idx])
        n_gt = int(self._gt_offsets[idx + 1] - self._gt_offsets[idx])
        return n_ctrl, n_gt

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


# ---------------------------------------------------------------------------
# Cross-dataset FM Dataset (iterable, yields one condition per step)
# ---------------------------------------------------------------------------

class CrossDatasetFMDataset(IterableDataset):
    """Yields ``(src_batch, gt_batch, ds_name, cond_name, perturbation_batch|None)`` per step.

    When ``use_pert_condition`` is enabled, ``perturbation_batch`` is a frozen dict of CPU tensors
    (``pert_gene_ids``, ``pert_mask``, ``pert_type_id``, ``nperts``, optional ``combo_id``)
    with batch dim ``batch_size``.  Disabled → ``None`` (legacy callers unpack first four fields).

    One full pass = one epoch.  With ``ds_alpha < 1`` the number of conditions
    selected from large datasets is capped (different subset each epoch).
    Each selected condition appears ceil(n_gt / batch_size) times so every GT
    sample is visited once.  All entries are globally shuffled to eliminate
    tail-concentration of any single dataset.
    """

    def __init__(
        self,
        data_dir: str,
        split: Dict[str, Dict[str, List[str]]],
        batch_size: int = 256,
        seed: int = 42,
        mode: str = "train",
        min_cells: int = 32,
        ds_alpha: float = 1.0,
        scale_noise: float = 0.0,
        min_selected_conditions_per_dataset: int = 0,
        condition_visit_power: float = 1.0,
        condition_visit_cap: int = 0,
        *,
        use_pert_condition: bool = False,
        max_pert_genes: int = 16,
        gene_embedding_cache_dir: str = "",
        biflow_dir: Optional[str] = None,
        use_h5ad_pert_metadata: bool = False,
        pert_metainfo_path: str = "",
        chem_emb_source_dir: str = "",
        chem_obs_column: str = "",
        drug_emb_cache_dir: str = "",
        max_chem_keys: int = 4,
        chemical_metainfo_path: str = "",
        chem_fallback_embed_dim: int = 512,
        latent_backbone: str = "state",
        pert_chem_enabled: bool = False,
        perturbation_family_filter: str = "all",
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        ddp_sync_min_len: bool = True,
        silent: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.min_cells = min_cells
        self.seed = seed
        self.mode = mode
        self.ds_alpha = ds_alpha
        self.scale_noise = scale_noise
        self.min_selected_conditions_per_dataset = max(0, int(min_selected_conditions_per_dataset))
        self.condition_visit_power = max(0.0, float(condition_visit_power))
        self.condition_visit_cap = max(0, int(condition_visit_cap))
        self._ddp_rank = max(0, int(ddp_rank))
        self._ddp_world_size = max(1, int(ddp_world_size))
        self._ddp_sync_min_len = bool(ddp_sync_min_len)
        self._silent = bool(silent)
        self._epoch_counter = 0
        self.chem_emb_source_dir = str(chem_emb_source_dir or "").strip()
        self.chem_obs_column = str(chem_obs_column or "").strip()
        self.drug_emb_cache_dir = str(drug_emb_cache_dir or "").strip()
        self.max_chem_keys = int(max_chem_keys)
        self.chemical_metainfo_path = str(chemical_metainfo_path or "").strip()
        self.chem_fallback_embed_dim = int(chem_fallback_embed_dim)
        self._pert_chem_enabled = bool(pert_chem_enabled)
        self.perturbation_family_filter = str(perturbation_family_filter or "all").strip().lower()
        self._chem_embed_backend = load_chemical_embed_backend(
            self, fallback_dim=max(8, int(self.chem_fallback_embed_dim)),
        )

        self.use_pert_condition = bool(use_pert_condition)
        self.max_pert_genes = int(max_pert_genes)
        self._gene_cache: Optional[GeneEmbeddingCache] = None
        self._obs_cond_meta: Dict[str, Dict[str, ConditionMetadata]] = {}
        self._sidecar_cond_meta: Dict[str, Dict[str, ConditionMetadata]] = _load_condition_metadata_sidecar(self.data_dir)
        self._pert_metainfo: Dict[str, str] = {}
        self._pert_batch_cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, ...]] = {}

        if self.use_pert_condition:
            d = str(gene_embedding_cache_dir or "").strip()
            if not d:
                raise ValueError(
                    "CrossDatasetFMDataset: gene_embedding_cache_dir is required when use_pert_condition=True"
                )
            self._gene_cache = GeneEmbeddingCache(Path(d))
            pmp = str(pert_metainfo_path or "").strip()
            if pmp or self.chemical_metainfo_path:
                gene_mp = load_dataset_metainfo(pmp, allow_missing=True) if pmp else {}
                chem_mp = load_dataset_metainfo(self.chemical_metainfo_path, allow_missing=True)
                self._pert_metainfo = {**gene_mp, **chem_mp}

        bi_root = _infer_biflow_dir(str(self.data_dir), biflow_dir)
        self._latent_backbone_s = normalize_latent_backbone(latent_backbone)
        if bi_root is not None and bool(use_h5ad_pert_metadata) and split:
            bi_path = Path(bi_root)
            for ds_name in split.keys():
                conds_here = sorted(split.get(ds_name, {}).get(mode, []))
                if not conds_here:
                    continue
                h5_path = resolve_gt_h5ad_for_pert_metadata(
                    bi_path,
                    ds_name,
                    latent_backbone=self._latent_backbone_s,
                )
                if h5_path is None or not h5_path.is_file():
                    continue
                if not self._silent:
                    print(
                        f"[CrossDatasetFMDataset] h5ad perturbation metadata: "
                        f"{ds_name} -> {h5_path}",
                        flush=True,
                    )
                lk = _build_obs_condition_metadata_lookup(
                    h5_path, conds_here, chem_obs_column=self.chem_obs_column,
                )
                if lk:
                    self._obs_cond_meta[str(ds_name)] = lk

        self.handles: Dict[str, _DatasetHandle] = {}
        self.ds_conds: Dict[str, List[str]] = {}
        self._cond_sizes: Dict[str, Dict[str, Tuple[int, int]]] = {}

        skipped = 0
        family_filtered = 0
        for ds_name, sp in split.items():
            h5_path = self.data_dir / f"{ds_name}.h5"
            if not h5_path.exists():
                continue
            handle = _DatasetHandle(str(h5_path))
            valid_conds = []
            sizes = {}
            for cond in sp[mode]:
                n_src, n_gt = handle.cond_sizes(cond)
                if n_src >= min_cells and n_gt >= min_cells:
                    meta = self.metadata_for_condition(ds_name, cond)
                    if not _keep_by_perturbation_family(
                        meta, ds_name, self.perturbation_family_filter,
                    ):
                        family_filtered += 1
                        continue
                    valid_conds.append(cond)
                    sizes[cond] = (n_src, n_gt)
                else:
                    skipped += 1
            if valid_conds:
                self.handles[ds_name] = handle
                self.ds_conds[ds_name] = valid_conds
                self._cond_sizes[ds_name] = sizes
            else:
                handle.close()

        self.ds_names = sorted(self.ds_conds.keys())
        if skipped > 0 and not self._silent:
            print(f"[{mode}] Skipped {skipped} conditions with source pool or GT < {min_cells}")
        if self.perturbation_family_filter not in {"", "all", "any"} and not self._silent:
            print(
                f"[{mode}] perturbation_family_filter={self.perturbation_family_filter}; "
                f"filtered={family_filtered}",
                flush=True,
            )

        if not self._silent:
            self._print_ds_balance()

    @property
    def gene_embedding_cache(self) -> Optional[GeneEmbeddingCache]:
        """Filled when ``use_pert_condition`` is True."""
        return self._gene_cache

    def _print_ds_balance(self):
        for ds in self.ds_names:
            n = len(self.ds_conds[ds])
            n_eff = self._n_eff(n)
            sizes = self._cond_sizes[ds]
            avg_visits = sum(
                self._condition_visits(s[1]) for s in sizes.values()
            ) / max(len(sizes), 1)
            est_batches = int(n_eff * avg_visits)
            print(f"  [{self.mode}] {ds}: {n} conds, {n_eff} selected, ~{est_batches} batches/epoch")
        print(f"  [{self.mode}] total_conds={self.total_conditions}, epoch_steps~{self.epoch_steps}")

    def _n_eff(self, n: int) -> int:
        if self.ds_alpha >= 1.0:
            base = n
        else:
            base = max(1, min(int(math.ceil(n ** self.ds_alpha)), n))
        if self.min_selected_conditions_per_dataset > 0:
            base = max(base, min(self.min_selected_conditions_per_dataset, n))
        return base

    def _condition_visits(self, n_gt: int) -> int:
        visits = max(1, math.ceil(int(n_gt) / self.batch_size))
        if self.condition_visit_power != 1.0:
            visits = max(1, int(math.ceil(float(visits) ** self.condition_visit_power)))
        if self.condition_visit_cap > 0:
            visits = min(visits, self.condition_visit_cap)
        return visits

    @property
    def total_conditions(self) -> int:
        return sum(len(v) for v in self.ds_conds.values())

    @property
    def epoch_steps(self) -> int:
        """Expected steps per epoch (random condition selection → use average visits)."""
        total = 0
        for ds in self.ds_names:
            sizes = self._cond_sizes[ds]
            n_eff = self._n_eff(len(sizes))
            avg_visits = sum(
                self._condition_visits(s[1]) for s in sizes.values()
            ) / max(len(sizes), 1)
            total += int(n_eff * avg_visits)
        if self._ddp_world_size > 1:
            return (total + self._ddp_world_size - 1) // self._ddp_world_size
        return total

    def _build_epoch_order(self, rng: np.random.RandomState) -> List[Tuple[str, str]]:
        """Global-shuffle with multi-visit to cover all GT samples.

        1. ds_alpha caps the number of conditions selected from large datasets.
        2. Each selected condition appears ceil(n_gt / batch_size) times so
           every GT sample is visited exactly once per epoch.
        3. All (ds, cond) entries are globally shuffled — no tail concentration.
        """
        all_pairs: List[Tuple[str, str]] = []

        for ds in self.ds_names:
            conds = list(self.ds_conds[ds])
            rng.shuffle(conds)
            n_eff = self._n_eff(len(conds))
            selected = conds[:n_eff]

            for cond in selected:
                _, n_gt = self._cond_sizes[ds][cond]
                n_visits = self._condition_visits(n_gt)
                all_pairs.extend([(ds, cond)] * n_visits)

        rng.shuffle(all_pairs)
        return all_pairs

    def metadata_for_condition(self, ds_name: str, cond: str) -> ConditionMetadata:
        """Resolved :class:`ConditionMetadata` (h5ad cache when configured, else string parse)."""
        ds_mp = self._obs_cond_meta.get(str(ds_name))
        if ds_mp is not None and cond in ds_mp:
            meta = ds_mp[cond]
        elif cond in self._sidecar_cond_meta.get(str(ds_name), {}):
            meta = self._sidecar_cond_meta[str(ds_name)][cond]
        else:
            meta = condition_metadata_from_cond_string(cond)
        return apply_pert_metainfo_fallback(
            meta,
            str(ds_name),
            self._pert_metainfo,
            use_pert_condition=self.use_pert_condition,
        )

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

    def _perturbation_batch_for_condition(self, ds_name: str, cond: str) -> Tuple[torch.Tensor, ...]:
        """Return a cached CPU perturbation tuple for one repeated condition batch."""
        key = (str(ds_name), str(cond), int(self.batch_size))
        cached = self._pert_batch_cache.get(key)
        if cached is not None:
            return cached
        meta = self.metadata_for_condition(ds_name, cond)
        meta = self.enrich_metadata_with_chem(meta)
        rows = [meta] * int(self.batch_size)
        pb = PerturbationBatch.from_metadata_list(
            rows,
            self._gene_cache,
            max_genes=self.max_pert_genes,
            max_chem_slots=int(self.max_chem_keys),
            device=torch.device("cpu"),
        )
        cached = pb.as_tuple_full()
        self._pert_batch_cache[key] = cached
        return cached

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, str, str, Optional[Tuple[torch.Tensor, ...]]]]:
        worker_info = torch.utils.data.get_worker_info()
        epoch_seed = self.seed + self._epoch_counter
        self._epoch_counter += 1
        if worker_info is not None:
            epoch_seed += worker_info.id * 100000

        rng = np.random.RandomState(epoch_seed)
        order = self._build_epoch_order(rng)

        if worker_info is not None:
            n_workers = worker_info.num_workers
            wid = worker_info.id
            order = order[wid::n_workers]

        if self._ddp_world_size > 1:
            order = order[self._ddp_rank :: self._ddp_world_size]
            if self._ddp_sync_min_len:
                try:
                    import torch.distributed as dist
                except ImportError:
                    dist = None  # type: ignore
                if dist is not None and dist.is_available() and dist.is_initialized():
                    if torch.cuda.is_available():
                        dev = torch.device("cuda", torch.cuda.current_device())
                    else:
                        dev = torch.device("cpu")
                    t = torch.tensor([len(order)], dtype=torch.long, device=dev)
                    dist.all_reduce(t, op=dist.ReduceOp.MIN)
                    min_len = int(t.item())
                    order = order[:min_len]

        gt_perms: Dict[Tuple[str, str], np.ndarray] = {}
        gt_cursors: Dict[Tuple[str, str], int] = {}

        for ds_name, cond in order:
            key = (ds_name, cond)
            h = self.handles[ds_name]

            n_src_total, n_gt_total = self._cond_sizes[ds_name][cond]

            if n_gt_total < self.batch_size:
                gt_idx = rng.choice(n_gt_total, size=self.batch_size, replace=True)
            else:
                if key not in gt_perms:
                    gt_perms[key] = rng.permutation(n_gt_total)
                    gt_cursors[key] = 0
                perm = gt_perms[key]
                cursor = gt_cursors[key]
                end = min(cursor + self.batch_size, n_gt_total)
                gt_idx = perm[cursor:end]
                if len(gt_idx) < self.batch_size:
                    shortfall = self.batch_size - len(gt_idx)
                    gt_idx = np.concatenate([
                        gt_idx,
                        rng.choice(n_gt_total, size=shortfall, replace=True),
                    ])
                gt_cursors[key] = end if end < n_gt_total else 0

            gt_batch = h.read_gt_rows(cond, gt_idx)

            src_idx = rng.choice(n_src_total, size=self.batch_size, replace=(n_src_total < self.batch_size))
            src_batch = h.read_src_rows(cond, src_idx)

            if self.scale_noise > 0 and self.mode == "train":
                src_batch = src_batch * (1.0 + self.scale_noise * rng.randn(*src_batch.shape))
                gt_batch = gt_batch * (1.0 + self.scale_noise * rng.randn(*gt_batch.shape))

            perturbation_batch = None
            if self.use_pert_condition:
                perturbation_batch = self._perturbation_batch_for_condition(ds_name, cond)

            yield (
                torch.from_numpy(src_batch.astype(np.float32)),
                torch.from_numpy(gt_batch.astype(np.float32)),
                ds_name,
                cond,
                perturbation_batch,
            )

    def close(self):
        for h in getattr(self, "handles", {}).values():
            h.close()

    def __del__(self):
        self.close()
