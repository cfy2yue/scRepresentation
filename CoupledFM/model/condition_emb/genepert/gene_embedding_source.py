"""Configuration helpers for filesystem-backed gene embeddings used by encoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from .gene_cache import GeneEmbeddingCache

PathLike = Union[str, Path]

# Modes consumed by :class:`~condition_emb.genepert.perturbation_encoder.PerturbationConditionEncoder`.
EncoderMode = Literal[
    "random_learned",
    "pretrained_frozen",
    "pretrained_tunable",
    "pretrained_with_type_gate",
    "combo_id_baseline",
]


@dataclass(frozen=True)
class GeneEmbeddingLoaderConfig:
    """Filesystem gene embedding bundle (``gene_embeddings.npy`` + index + manifest).

    This does **not** instantiate Torch modules; pair with ``PerturbationConditionEncoder``
    by passing ``cache=load_gene_embedding_cache(cfg)`` for ``pretrained_*`` modes.
    """

    cache_dir: PathLike
    pad_index: int = 0
    unk_index: int = 1

    def load_cache(self) -> GeneEmbeddingCache:
        return load_gene_embedding_cache(self)


def load_gene_embedding_cache(cfg: GeneEmbeddingLoaderConfig) -> GeneEmbeddingCache:
    """Load :class:`~condition_emb.genepert.gene_cache.GeneEmbeddingCache` from ``cfg.cache_dir``."""
    return GeneEmbeddingCache(cfg.cache_dir, pad_index=cfg.pad_index, unk_index=cfg.unk_index)


def describe_encoder_pairing(mode: EncoderMode) -> str:
    """Human-readable note linking loader config to encoder ``mode``."""
    if mode == "random_learned":
        return "GeneEmbeddingLoaderConfig optional; encoder builds learned Embedding."
    if mode == "combo_id_baseline":
        return "GeneEmbeddingLoaderConfig unused; combo_id embedding only."
    return "GeneEmbeddingLoaderConfig.load_cache() required for pretrained weights."
