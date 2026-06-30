"""Drug / small-molecule embedding cache shaped like ``GeneEmbeddingCache`` (``.npy`` + index + manifest)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
DrugBackend = Union["DrugEmbeddingCache", "RandomDrugEmbeddingFallback"]


def deterministic_standard_normal_vec(key: str, dim: int, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Reproducible ``N(0,1)^{dim}`` from UTF-8 key (fallback when cache misses)."""
    d = int(dim)
    if d <= 0:
        raise ValueError(f"dim must be positive, got {d}")
    h = hashlib.sha256(str(key).encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(d, dtype=dtype)


class DrugEmbeddingCache:
    """Load ``drug_embeddings.npy`` plus ``drug_index.tsv`` or ``drug_index.json`` + ``manifest.json``.

    Row ``pad_index`` (default 0) is reserved — lookups never return it from ``lookup``.
    OOV strings use deterministic Gaussian fallback (:func:`deterministic_standard_normal_vec`).
    """

    def __init__(
        self,
        cache_dir: PathLike,
        *,
        unk_index: int = 1,
        pad_index: int = 0,
    ):
        root = Path(cache_dir).expanduser()
        self.cache_dir = root
        self.unk_index = int(unk_index)
        self.pad_index = int(pad_index)

        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        self.manifest: Dict[str, object] = json.loads(manifest_path.read_text(encoding="ascii"))

        npy_path = root / "drug_embeddings.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"missing embeddings: {npy_path}")
        self._emb = np.load(str(npy_path), mmap_mode="r")
        if self._emb.ndim != 2:
            raise ValueError(f"drug_embeddings.npy expects 2D array, got {self._emb.shape}")

        idx_tsv = root / "drug_index.tsv"
        idx_json = root / "drug_index.json"
        if idx_tsv.is_file():
            self._key_to_index = self._load_tsv_index(idx_tsv)
        elif idx_json.is_file():
            self._key_to_index = self._load_json_index(idx_json)
        else:
            raise FileNotFoundError(f"need drug_index.tsv or drug_index.json under {root}")

        n_rows = int(self._emb.shape[0])
        if len({self.pad_index, self.unk_index}) != 2:
            raise ValueError("pad_index and unk_index must differ")
        if n_rows <= max(self.pad_index, self.unk_index):
            raise ValueError("embedding rows fewer than pad/unk indices")

    @staticmethod
    def _normalize_key(sym: str) -> str:
        return sym.strip()

    @classmethod
    def _load_tsv_index(cls, path: Path) -> Dict[str, int]:
        mp: Dict[str, int] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(lines):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) >= 2 and parts[0].lower() in ("key", "drug", "compound", "id"):
                continue
            if len(parts) < 2:
                parts = ln.split(None, 1)
            if len(parts) < 2:
                raise ValueError(f"bad drug_index.tsv line {i + 1}: {ln}")
            sym, idx_s = parts[0], parts[-1].strip().split()[0]
            mp[cls._normalize_key(sym)] = int(idx_s)
        return mp

    @classmethod
    def _load_json_index(cls, path: Path) -> Dict[str, int]:
        obj = json.loads(path.read_text(encoding="utf-8"))
        mp: Dict[str, int] = {}
        if isinstance(obj, Mapping):
            if "symbols" in obj and isinstance(obj["symbols"], list):
                for it in obj["symbols"]:
                    if not isinstance(it, Mapping):
                        continue
                    sym = str(it.get("key", it.get("drug", it.get("id", ""))))
                    idx = int(it.get("index", -1))
                    mp[cls._normalize_key(sym)] = idx
            else:
                for k, v in obj.items():
                    if isinstance(k, str) and isinstance(v, int):
                        mp[cls._normalize_key(k)] = int(v)
        return mp

    @property
    def embed_dim(self) -> int:
        return int(self._emb.shape[1])

    @property
    def num_embeddings(self) -> int:
        return int(self._emb.shape[0])

    def row_as_numpy(self, row_index: int, *, copy: bool = False) -> np.ndarray:
        vec = np.asarray(self._emb[int(row_index)], dtype=np.float32)
        return vec.copy() if copy else vec

    def lookup(self, key: str) -> Tuple[np.ndarray, bool]:
        """Return ``(dim,)`` vector and whether the on-disk cache was hit."""
        k = self._normalize_key(key)
        if not k:
            z = deterministic_standard_normal_vec("<empty>", self.embed_dim)
            return z, False
        ri = self._key_to_index.get(k)
        if ri is None:
            ri = self.unk_index
        ri = int(ri)
        if ri == self.pad_index or ri < 0 or ri >= self.num_embeddings:
            z = deterministic_standard_normal_vec(k, self.embed_dim)
            return z, False
        if ri == self.unk_index:
            z = deterministic_standard_normal_vec(k, self.embed_dim)
            return z, False
        return self.row_as_numpy(ri, copy=True), True

    def lookup_many(self, keys: Sequence[str]) -> Tuple[np.ndarray, Sequence[bool]]:
        vecs = [self.lookup(str(k)) for k in keys]
        if not vecs:
            return np.zeros((0, self.embed_dim), dtype=np.float32), ()
        mats = np.stack([v for v, _ in vecs], axis=0)
        hits = [h for _, h in vecs]
        return mats, hits

    @classmethod
    def from_dir_or_random(cls, dir_path: Optional[PathLike], *, dim: int = 512) -> DrugBackend:
        """Prefer on-disk layout; invalid / empty ``dir_path`` → :class:`RandomDrugEmbeddingFallback`."""
        s = "" if dir_path is None else str(dir_path).strip()
        if not s:
            return RandomDrugEmbeddingFallback(dim=int(dim))
        root = Path(s).expanduser()
        manifest = root / "manifest.json"
        npy = root / "drug_embeddings.npy"
        if not root.is_dir() or not manifest.is_file() or not npy.is_file():
            return RandomDrugEmbeddingFallback(dim=int(dim))
        return cls(root)


class RandomDrugEmbeddingFallback:
    """Pure fallback: every ``lookup`` is a deterministic Gaussian (``hit=False``)."""

    def __init__(self, *, dim: int = 512):
        self.embed_dim = int(dim)
        if self.embed_dim <= 0:
            raise ValueError(f"dim must be positive, got {self.embed_dim}")

    def lookup(self, key: str) -> Tuple[np.ndarray, bool]:
        k = "" if key is None else str(key).strip()
        kk = k if k else "<empty>"
        z = deterministic_standard_normal_vec(kk, self.embed_dim)
        return z, False
