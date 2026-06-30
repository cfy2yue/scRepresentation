"""Benchmark encoders with per-cell perturbation conditioning (latent_bench).

Submodules are loaded lazily so ``import ...adapters`` does not require every
model dependency (e.g. Geneformer needs ``transformers``).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "uce",
    "state",
    "scgpt",
    "geneformer",
    "stack",
    "scldm",
    "xverse",
    "cellnavi",
    "scfoundation",
    "nicheformer",
    "transcriptformer",
    "pca_baseline",
    "scvi_baseline",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f".{name}", __package__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, *globals().keys()])
