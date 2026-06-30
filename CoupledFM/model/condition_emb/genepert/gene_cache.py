"""Gene symbol embedding table backed by filesystem cache (numpy + index + manifest)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


PathLike = Union[str, Path]


def _normalize_symbol(sym: str) -> str:
    return sym.strip().upper()


class GeneEmbeddingCache:
    """Loads ``gene_embeddings.npy`` plus ``gene_index.tsv`` or ``gene_index.json``.

    Rows are aligned with sequential indices stored in the index files.
    Index 0 must be pad/null (zero embedding). Index 1 is ``<unk>`` for OOV symbols.
    ``lookup()`` returns integers suitable for embedding modules.

    For training (``latent``, ``coupled``, ``raw_independent``): set dataset / trainer config
    field ``pert_gene_emb_cache_dir`` to this directory when using
    :class:`~condition_emb.genepert.perturbation_encoder.PerturbationConditionEncoder` in
    ``pretrained_*`` modes — build caches with ``condition_emb/genepert/tools/export_gene_embedding_cache.py``.
    """

    def __init__(
        self,
        cache_dir: PathLike,
        *,
        unk_index: int = 1,
        pad_index: int = 0,
    ):
        root = Path(cache_dir)
        self.cache_dir = root
        self.unk_index = int(unk_index)
        self.pad_index = int(pad_index)

        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        self.manifest: Dict[str, object] = json.loads(manifest_path.read_text(encoding="ascii"))

        npy_path = root / "gene_embeddings.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"missing embeddings: {npy_path}")
        self._emb = np.load(str(npy_path), mmap_mode="r")
        if self._emb.ndim != 2:
            raise ValueError(f"gene_embeddings.npy expects 2D array, got {self._emb.shape}")

        idx_tsv = root / "gene_index.tsv"
        idx_json = root / "gene_index.json"
        if idx_tsv.is_file():
            self._symbol_to_index = self._load_tsv_index(idx_tsv)
        elif idx_json.is_file():
            self._symbol_to_index = self._load_json_index(idx_json)
        else:
            raise FileNotFoundError(f"need gene_index.tsv or gene_index.json under {root}")

        n_rows = int(self._emb.shape[0])
        if len({self.pad_index, self.unk_index}) != 2:
            raise ValueError("pad_index and unk_index must differ")
        if n_rows <= max(self.pad_index, self.unk_index):
            raise ValueError("embedding rows fewer than pad/unk indices")

    @staticmethod
    def _load_tsv_index(path: Path) -> Dict[str, int]:
        mp: Dict[str, int] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, ln in enumerate(lines):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) >= 2 and parts[0].lower() in ("symbol", "gene", "gene_symbol"):
                continue
            if len(parts) < 2:
                parts = ln.split(None, 1)
            if len(parts) < 2:
                raise ValueError(f"bad gene_index.tsv line {i + 1}: {ln}")
            sym, idx_s = parts[0], parts[-1].strip().split()[0]
            mp[_normalize_symbol(sym)] = int(idx_s)
        return mp

    @staticmethod
    def _load_json_index(path: Path) -> Dict[str, int]:
        obj = json.loads(path.read_text(encoding="utf-8"))
        # Allow {"symbols": [{"symbol":"TP53","index":2}, ...]} or flat {"TP53":2}
        mp: Dict[str, int] = {}
        if isinstance(obj, Mapping):
            if "symbols" in obj and isinstance(obj["symbols"], list):
                for it in obj["symbols"]:
                    if not isinstance(it, Mapping):
                        continue
                    sym = str(it.get("symbol", ""))
                    idx = int(it.get("index", -1))
                    mp[_normalize_symbol(sym)] = idx
            else:
                for k, v in obj.items():
                    if isinstance(k, str) and isinstance(v, int):
                        mp[_normalize_symbol(k)] = int(v)
        return mp

    @property
    def embed_dim(self) -> int:
        return int(self._emb.shape[1])

    @property
    def num_embeddings(self) -> int:
        return int(self._emb.shape[0])

    def embedding_numpy(self, copy: bool = False) -> np.ndarray:
        arr = np.array(self._emb) if copy else np.asarray(self._emb)
        return arr

    def lookup(self, symbol: str) -> int:
        s = _normalize_symbol(symbol)
        return int(self._symbol_to_index.get(s, self.unk_index))

    def lookup_many(self, symbols: Sequence[str]) -> List[int]:
        return [self.lookup(s) for s in symbols]

    def symbols_dict(self) -> Mapping[str, int]:
        """Read-only mapping upper_symbol -> row index."""
        return self._symbol_to_index

    def validate_index_bounds(self) -> None:
        """Raise if any symbol index is out of range for ``gene_embeddings.npy`` rows."""
        if not self._symbol_to_index:
            return
        mx = max(self._symbol_to_index.values())
        if mx >= self.num_embeddings:
            raise ValueError(f"max gene index {mx} >= num_embeddings {self.num_embeddings}")
        for name, idx in (("pad", self.pad_index), ("unk", self.unk_index)):
            if idx >= self.num_embeddings:
                raise ValueError(f"{name}_index {idx} >= num_embeddings {self.num_embeddings}")


class GeneEmbeddingTable(nn.Module):
    """``nn.Embedding`` filled from ``GeneEmbeddingCache`` or arbitrary weight tensor."""

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int,
        *,
        weights: Optional[torch.Tensor] = None,
        padding_idx: int = 0,
        freeze: bool = False,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
        if weights is not None:
            if weights.shape != (num_embeddings, embed_dim):
                raise ValueError(f"weights shape {weights.shape} != ({num_embeddings}, {embed_dim})")
            self.embed.weight.data.copy_(weights)
        else:
            nn.init.normal_(self.embed.weight.data, mean=0.0, std=0.02)
            with torch.no_grad():
                self.embed.weight[padding_idx].zero_()
        if freeze:
            self.embed.weight.requires_grad_(False)

    @classmethod
    def from_cache(cls, cache: GeneEmbeddingCache, *, freeze: bool = False) -> "GeneEmbeddingTable":
        w = torch.from_numpy(cache.embedding_numpy(copy=True)).to(dtype=torch.float32)
        return cls(
            w.shape[0],
            w.shape[1],
            weights=w,
            padding_idx=cache.pad_index,
            freeze=freeze,
        )

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(gene_ids)
