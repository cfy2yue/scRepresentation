"""Load precomputed small-molecule vectors (SMILES / ChEMBL / drug key) from a disk cache.

When :func:`resolve_chem_embedding` is wired from dataset code, set
``chem_emb_source_dir`` on the trainer / dataset config to a directory containing
``manifest.json``, ``embeddings.npy``, and ``index.tsv`` (see
``condition_emb/genepert/tools/export_chem_embedding_cache.py``). Empty directory → hook returns
``None`` (default-off).

Reserved for UniMol / MolFormer exports: use ``export_chem_embedding_cache.py
--format passthrough_dict`` or implement ``--format unimol`` at the marked
insertion point.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .perturbation import ConditionMetadata

_CHEM_CACHE_BY_DIR: dict[str, "ChemEmbeddingCache"] = {}
_RESOLVE_HIT_LOGGED = False


def _normalize_smiles_key(s: str) -> str:
    return s.strip()


def _normalize_chembl_key(s: str) -> str:
    return s.strip()


def _normalize_drug_name_key(s: str) -> str:
    return s.strip()


def parse_chem_source_fields(chem_source: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (smiles, chembl_id, drug_name) from a ``chem_source`` string."""
    if not chem_source:
        return None, None, None
    sm: Optional[str] = None
    ch: Optional[str] = None
    dr: Optional[str] = None
    for part in str(chem_source).split("|"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k_st = k.strip().lower()
        v = v.strip()
        if k_st == "smiles":
            sm = v
        elif k_st in ("chembl_id", "chembl-id"):
            ch = v
        elif k_st == "drug":
            dr = v
    return sm, ch, dr


class ChemEmbeddingCache:
    """Filesystem-backed table: ``manifest.json`` + ``embeddings.npy`` + ``index.tsv``.

    Row ``pad_index`` is reserved (typically zeros); keys in ``index.tsv`` map a
    stripped SMILES string, ChEMBL id, and/or drug label to embedding rows.
    Multiple keys may reference the same row. Design mirrors
    :class:`~condition_emb.genepert.gene_cache.GeneEmbeddingCache`, with richer string
    keys instead of gene symbols.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        pad_index: int = 0,
    ):
        self.root = Path(root)
        self.pad_index = int(pad_index)

        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        self.manifest: Dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))

        npy_path = self.root / "embeddings.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"missing embeddings: {npy_path}")
        self._emb = np.load(str(npy_path), mmap_mode="r")
        if self._emb.ndim != 2:
            raise ValueError(f"embeddings.npy expects 2D array, got {self._emb.shape}")

        idx_path = self.root / "index.tsv"
        if not idx_path.is_file():
            raise FileNotFoundError(f"missing index.tsv under {self.root}")
        self._key_to_row = self._load_index_tsv(idx_path)
        self.validate_index_bounds()

    @staticmethod
    def _load_index_tsv(path: Path) -> Dict[str, int]:
        mp: Dict[str, int] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(lines):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) >= 2 and parts[0].lower() in ("key", "keys", "smiles", "id"):
                continue
            if len(parts) < 2:
                parts = ln.split(None, 1)
            if len(parts) < 2:
                raise ValueError(f"bad index.tsv line {i + 1}: {ln}")
            key, idx_s = parts[0], parts[-1].strip().split()[0]
            mp[key.strip()] = int(idx_s)
        return mp

    @property
    def embed_dim(self) -> int:
        return int(self._emb.shape[1])

    @property
    def num_rows(self) -> int:
        return int(self._emb.shape[0])

    def row_as_numpy(self, row_index: int, *, copy: bool = False) -> np.ndarray:
        vec = np.asarray(self._emb[int(row_index)], dtype=np.float32)
        return vec.copy() if copy else vec

    def lookup_row_index(self, key: str) -> Optional[int]:
        """Return row index for an exact cache key (after caller-side normalization), or ``None``."""
        if key not in self._key_to_row:
            return None
        ri = int(self._key_to_row[key])
        if ri == self.pad_index:
            return None
        if ri < 0 or ri >= self.num_rows:
            return None
        return ri

    def lookup_embedding(self, key: str) -> Optional[np.ndarray]:
        ri = self.lookup_row_index(key)
        if ri is None:
            return None
        return self.row_as_numpy(ri, copy=True)

    def validate_index_bounds(self) -> None:
        if not self._key_to_row:
            return
        mx = max(self._key_to_row.values())
        if mx >= self.num_rows:
            raise ValueError(f"max index {mx} >= num_rows {self.num_rows}")
        if self.pad_index >= self.num_rows:
            raise ValueError(f"pad_index {self.pad_index} >= num_rows {self.num_rows}")


def _get_chem_cache(root: Path) -> ChemEmbeddingCache:
    key = str(root.resolve())
    if key not in _CHEM_CACHE_BY_DIR:
        _CHEM_CACHE_BY_DIR[key] = ChemEmbeddingCache(root)
    return _CHEM_CACHE_BY_DIR[key]


def _resolve_from_cache(cache: ChemEmbeddingCache, meta: "ConditionMetadata") -> Optional[np.ndarray]:
    sm, ch, dr = parse_chem_source_fields(meta.chem_source)
    tries: list[tuple[str, str]] = []
    if sm is not None:
        tries.append(("smiles", _normalize_smiles_key(sm)))
    if ch is not None:
        tries.append(("chembl_id", _normalize_chembl_key(ch)))
    if dr is not None:
        tries.append(("drug", _normalize_drug_name_key(dr)))
    for _kind, k in tries:
        if not k:
            continue
        vec = cache.lookup_embedding(k)
        if vec is not None:
            return vec
    return None


def resolve_chem_embedding(meta: "ConditionMetadata", cfg: Any) -> Optional[np.ndarray]:
    """Return a single float32 vector (slot 1) for ``meta`` if resolvable."""
    # Unified resolver (DrugEmbeddingCache-shaped table + deterministic OOV fallback + cocktail).
    if bool(getattr(cfg, "pert_chem_enabled", False)):
        from model.condition_emb.chempert.chem_resolver import resolve_first_chemical_embedding

        return resolve_first_chemical_embedding(meta, cfg)

    # Legacy: ``chem_emb_source_dir`` alone (no unified drug cache / FLAGS).
    d = str(getattr(cfg, "chem_emb_source_dir", "") or "").strip()
    if not d:
        return None
    root = Path(d)
    if not root.is_dir():
        return None
    cache = _get_chem_cache(root)
    vec = _resolve_from_cache(cache, meta)
    global _RESOLVE_HIT_LOGGED  # noqa: PLW0603
    if vec is not None and not _RESOLVE_HIT_LOGGED:
        warnings.warn(
            "chem_emb_source_dir: matched a precomputed molecule embedding (suppressing further hit logs).",
            UserWarning,
            stacklevel=2,
        )
        _RESOLVE_HIT_LOGGED = True
    return vec
