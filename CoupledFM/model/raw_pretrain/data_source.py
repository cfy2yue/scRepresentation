"""Per-tissue shard readers for cellgene-census pairwise pretraining."""

from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from model.utils.data.vocab import GeneVocab

# CellNavi SparseCellNaviEncoder padding idx (matches RawExprVelocityField.embed_gene).
PADDING_GENE_TOKEN = 40000


class TissueShardSource(ABC):
    """One tissue → one backed h5ad view."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def n_units(self) -> int:
        """Number of cells (rows)."""

    @abstractmethod
    def gene_ids_cellnavi(self) -> np.ndarray:
        """Length G int64 gene embedding ids (OOV → ``PADDING_GENE_TOKEN``)."""

    @abstractmethod
    def get_expr(self, row_idx: int) -> np.ndarray:
        """Gene expression bins normalized to approximately ``[0, 1]`` (÷ (num_bins-1))."""

    @abstractmethod
    def schema_summary(self) -> dict[str, Any]:
        """Serializable shard metadata for audit logs."""


class H5adTissueShard(TissueShardSource):
    """Lazy ``backed='r'`` anndata slice for one ``*_top6000var.h5ad``."""

    def __init__(
        self,
        h5ad_path: str | Path,
        vocab: GeneVocab,
        *,
        num_bins: int = 50,
        gene_symbol_column: str = "feature_name",
        min_gene_hit_rate: float = 0.80,
        root_dir: str | Path | None = None,
    ):
        import anndata as ad

        self._path = Path(h5ad_path).expanduser().resolve()
        self._root_dir = Path(root_dir).expanduser().resolve() if root_dir is not None else None
        self._num_bins = int(num_bins)
        self._name = self._path.parent.name
        self._adata = ad.read_h5ad(self._path, backed="r")

        xprobe = self._adata.X[0]
        if not (hasattr(xprobe, "toarray") or hasattr(xprobe, "shape")):
            raise TypeError(
                f"H5adTissueShard: row 0 of X has unexpected type {type(xprobe)} "
                f"(expected sparse matrix row or dense ndarray)"
            )

        var_names = list(map(str, self._adata.var_names))
        var = self._adata.var
        if gene_symbol_column and gene_symbol_column in var.columns:
            gene_symbols = [str(x) for x in var[gene_symbol_column].tolist()]
            gene_source = f"var[{gene_symbol_column!r}]"
        else:
            gene_symbols = var_names
            gene_source = "var_names"

        self._G = len(gene_symbols)
        gid = []
        hit = 0
        for g in gene_symbols:
            if g in vocab.gene2token:
                gid.append(int(vocab.gene2token[g]))
                hit += 1
            else:
                gid.append(PADDING_GENE_TOKEN)
        self._gene_ids = np.asarray(gid, dtype=np.int64)
        hit_rate = float(hit) / float(max(len(gene_symbols), 1))
        if hit_rate < float(min_gene_hit_rate):
            raise ValueError(
                f"H5adTissueShard {self._path}: CellNavi gene hit rate {hit_rate:.3f} "
                f"below min_gene_hit_rate={float(min_gene_hit_rate):.3f}; "
                f"used {gene_source}. Check gene_symbol_column.",
            )

        obs_cols = list(map(str, self._adata.obs.columns))
        var_cols = list(map(str, self._adata.var.columns))
        required_obs = (
            "tissue",
            "cell_type",
            "cell_type_ontology_term_id",
            "cluster_id",
            "cluster_size",
            "cluster_frac",
        )
        required_var = (gene_symbol_column,) if gene_symbol_column else ()
        missing_obs = [c for c in required_obs if c not in obs_cols]
        missing_var = [c for c in required_var if c not in var_cols]
        h = hashlib.sha1()
        for g in gene_symbols:
            h.update(g.encode("utf-8", errors="replace"))
            h.update(b"\0")

        x0 = xprobe.toarray() if hasattr(xprobe, "toarray") else xprobe
        x0_arr = np.asarray(x0, dtype=np.float32).ravel()
        rel_path = str(self._path)
        if self._root_dir is not None:
            try:
                rel_path = str(self._path.relative_to(self._root_dir))
            except ValueError:
                rel_path = str(self._path)
        self._summary: dict[str, Any] = {
            "name": self._name,
            "path": rel_path,
            "n_obs": int(self._adata.n_obs),
            "n_vars": int(self._adata.n_vars),
            "gene_symbol_source": gene_source,
            "gene_hit_count": int(hit),
            "gene_oov_count": int(len(gene_symbols) - hit),
            "gene_hit_rate": hit_rate,
            "gene_axis_sha1": h.hexdigest(),
            "missing_obs_columns": missing_obs,
            "missing_var_columns": missing_var,
            "x_probe_min": float(x0_arr.min()) if x0_arr.size else 0.0,
            "x_probe_max": float(x0_arr.max()) if x0_arr.size else 0.0,
            "x_probe_nnz": int(np.count_nonzero(x0_arr)),
        }

        scale = max(float(self._num_bins) - 1.0, 1.0)

        def _read_row(r: int) -> np.ndarray:
            row = self._adata.X[int(r)]
            if hasattr(row, "toarray"):
                x = np.asarray(row.toarray(), dtype=np.float32).ravel()
            else:
                x = np.asarray(row, dtype=np.float32).ravel()
            x = np.clip(x, 0.0, float(self._num_bins - 1)) / scale
            return x.astype(np.float32, copy=False)

        self._read_row = _read_row

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_units(self) -> int:
        return int(self._adata.n_obs)

    def gene_ids_cellnavi(self) -> np.ndarray:
        return self._gene_ids.copy()

    def get_expr(self, row_idx: int) -> np.ndarray:
        return self._read_row(int(row_idx))

    def schema_summary(self) -> dict[str, Any]:
        return dict(self._summary)


def _candidate_paths_from_metainfo_row(root: Path, row: dict[str, str]) -> list[Path]:
    tissue = str(row.get("tissue", "")).strip()
    raw_path = str(row.get("path", "")).strip()
    out: list[Path] = []
    if raw_path:
        p = Path(raw_path)
        if p.is_absolute():
            try:
                out.append(root / p.relative_to(root))
            except ValueError:
                if tissue:
                    out.append(root / tissue / p.name)
            out.append(p)
        else:
            out.append(root / p)
    if tissue:
        out.extend(sorted((root / tissue).glob("*_top6000var.h5ad")))
    dedup: list[Path] = []
    seen: set[str] = set()
    for p in out:
        key = str(p)
        if key not in seen:
            dedup.append(p)
            seen.add(key)
    return dedup


def _paths_from_metainfo(root: Path, metainfo_path: Path | None) -> list[Path]:
    path = metainfo_path or (root / "tissue_metainfo.csv")
    if not path.is_file():
        return []
    hits: list[Path] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            chosen = None
            for cand in _candidate_paths_from_metainfo_row(root, row):
                if cand.is_file():
                    chosen = cand
                    break
            if chosen is not None:
                hits.append(chosen)
    return hits


def _paths_from_glob(root: Path, glob_leaf: str) -> list[Path]:
    hits: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("__"):
            continue
        found = sorted(entry.glob(glob_leaf))
        if found:
            hits.append(found[0])
    return hits


def discover_h5ad_shards(
    processed_dir: str | Path,
    vocab: GeneVocab,
    *,
    glob_leaf: str = "*_top6000var.h5ad",
    num_bins: int = 50,
    strict_same_genes: bool = True,
    tissue_metainfo_path: str | Path | None = None,
    gene_symbol_column: str = "feature_name",
    min_gene_hit_rate: float = 0.80,
) -> List[H5adTissueShard]:
    """Return one shard per tissue, preferring ``tissue_metainfo.csv`` discovery."""
    root = Path(processed_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(processed_dir)

    meta = Path(tissue_metainfo_path).expanduser() if tissue_metainfo_path else None
    if meta is not None and not meta.is_absolute() and not meta.is_file():
        meta = root / meta
    paths = _paths_from_metainfo(root, meta)
    if not paths:
        paths = _paths_from_glob(root, glob_leaf)

    shards: List[H5adTissueShard] = [
        H5adTissueShard(
            p,
            vocab,
            num_bins=num_bins,
            gene_symbol_column=gene_symbol_column,
            min_gene_hit_rate=min_gene_hit_rate,
            root_dir=root,
        )
        for p in paths
    ]

    if not shards:
        raise FileNotFoundError(f"No {glob_leaf} under {root}")

    ref = shards[0].gene_ids_cellnavi()
    if strict_same_genes:
        for s in shards[1:]:
            cur = s.gene_ids_cellnavi()
            if cur.shape != ref.shape or not np.array_equal(cur, ref):
                raise ValueError(
                    f"gene axis mismatch between shards ({shards[0].name} vs {s.name}); "
                    "rebuild matching gene lists or set strict_same_genes=False.",
                )
    return shards


def write_shard_summaries(shards: Sequence[TissueShardSource], path: str | Path) -> None:
    """Write schema summaries for reproducible pretrain data audits."""
    payload = [s.schema_summary() for s in shards]
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "PADDING_GENE_TOKEN",
    "TissueShardSource",
    "H5adTissueShard",
    "discover_h5ad_shards",
    "write_shard_summaries",
]
