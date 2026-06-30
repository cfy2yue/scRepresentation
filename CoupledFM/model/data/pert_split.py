"""DEPRECATED: real implementation lives in ``utils/data/split.py``.

Thin shim for legacy ``from data.pert_split import ...``. New code should import from ``utils.data.split``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

from model.utils.data.split import (
    build_canonical_split as _build_canonical_split,
    canonical_split_path,
    is_multi_pert,
    load_split_json as _load_split_json,
    n_single_pert_holdout,
    save_split,
)


def build_explicit_pert_split(
    biflow_dir,
    vocab,
    seed: int,
    min_cells: int = 16,
    coupling_mode: str = "coupled",
    dataset_names=None,
    ot_feature: str = "latent",
    de_dir=None,
) -> Dict[str, Dict[str, List[str]]]:
    """Back-compat wrapper → ``utils.data.split.build_canonical_split``。"""
    return _build_canonical_split(
        biflow_dir=biflow_dir,
        vocab=vocab,
        seed=seed,
        min_cells=min_cells,
        coupling_mode=coupling_mode,
        dataset_names=dataset_names,
        ot_feature=ot_feature,
        de_dir=de_dir,
    )


def load_explicit_json(path) -> Dict[str, Dict[str, List[str]]]:
    """Load split written by :func:`save_split`。"""
    return _load_split_json(path)


__all__ = [
    "build_explicit_pert_split",
    "canonical_split_path",
    "is_multi_pert",
    "load_explicit_json",
    "n_single_pert_holdout",
    "save_split",
]
