"""Collect smoke_*.py entrypoints so ``pytest tests/`` runs integration scripts."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]   # model/
_TESTS = Path(__file__).resolve().parent       # model/tests/


def _load_module(stem: str):
    path = _TESTS / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_chem_condition():
    _load_module("smoke_chem_condition").main()


def test_smoke_metainfo_fallback():
    _load_module("smoke_metainfo_fallback").main()


_CACHE = _ROOT / "condition_emb" / "genepert" / "cache" / "cellnavi_embed_gene"
_CACHE_NPY = _CACHE / "gene_embeddings.npy"


@pytest.mark.skipif(not _CACHE_NPY.is_file(), reason=f"missing {_CACHE_NPY}")
def test_smoke_pert_condition_e2e():
    _load_module("smoke_pert_condition_e2e").main()


def test_smoke_biflow_state_genepert():
    _load_module("smoke_biflow_state_genepert").main()


_SCIPLEX3_H5AD = Path(
    os.environ.get(
        "SCIPLEX3_H5AD",
        str(_ROOT / "data" / "raw" / "chemicalpert_DE5000" / "sciplex3_A549.h5ad"),
    )
).expanduser()


@pytest.mark.skipif(not _SCIPLEX3_H5AD.is_file(), reason=f"optional fixture missing {_SCIPLEX3_H5AD}")
def test_smoke_chem_metainfo_sciplex3():
    _load_module("smoke_chem_metainfo_sciplex3").main()
