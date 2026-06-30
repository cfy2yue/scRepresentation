"""Metric runners for scFM benchmark (tier1/2/3)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "atlas_scib",
    "geometry",
    "perturb_geom",
    "perturb_xcellline",
    "post_process",
    "tx_eval_ported",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
