"""Shared helpers for Stack control/GT embedding scripts.

NOTE (delivery scope): this module is NOT exercised by the two delivered
flows (raw flow pretrain & CoupledFM sweep / CellNavi-vs-scGPT compare).
It depends on a sibling ``scFM/`` checkout and is kept only for offline
regeneration of pre-exported embedding caches under
``<delivery_root>/pretrainckpt/genepert_cache/``. Skip unless rebuilding
caches from raw Stack checkpoints.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STACK_SRC = _REPO_ROOT / "scFM" / "fm" / "third_party" / "stack" / "src"

_CONTROL_TOKENS = frozenset(
    {
        "",
        "nan",
        "none",
        "null",
        "control",
        "ctrl",
        "ntc",
        "non-targeting",
        "nontargeting",
        "pbs",
        "unperturbed",
        "wildtype",
        "wt",
    }
)


def coupled_fm_root() -> Path:
    return _REPO_ROOT.resolve()


def default_stack_tmp_dir() -> Path:
    """Scratch under the CoupledFM repo (typically on ``/data2``, not ``$HOME``)."""
    d = coupled_fm_root() / "tmp" / "stack_embedding"
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def apply_temp_dir_env(tmp_dir: Path) -> None:
    """Point libc/Python temp files at ``tmp_dir`` (``TMPDIR`` / ``tempfile.tempdir``)."""
    p = str(Path(tmp_dir).expanduser().resolve())
    Path(p).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = p
    os.environ["TEMP"] = p
    os.environ["TMP"] = p
    tempfile.tempdir = p


def stack_package_src() -> Path:
    return _STACK_SRC.resolve()


def ensure_stack_on_syspath() -> None:
    """Insert vendored ``stack`` package (``.../third_party/stack/src``) first."""
    s = str(stack_package_src())
    if s not in sys.path:
        sys.path.insert(0, s)


def stack_obs_mask(adata) -> np.ndarray:
    """Human-like cells for CoupledFM-side filtering (align with Stack loader)."""
    if "organism" not in adata.obs.columns:
        return np.ones(adata.n_obs, dtype=bool)
    org = adata.obs["organism"].astype(str).str.strip()
    lower = org.str.lower()
    mask = (org == "Homo sapiens") | lower.isin({"human", "homo sapiens"})
    if bool(mask.any()):
        return mask.to_numpy(dtype=bool)
    return np.ones(adata.n_obs, dtype=bool)


def normalize_organism_for_stack_loader(adata):
    """Stack ``TestSamplerDataset`` requires ``organism == 'Homo sapiens'`` when present."""
    out = adata.copy()
    if "organism" not in out.obs.columns:
        return out
    vals = out.obs["organism"].astype(str).copy()
    lower = vals.str.lower().str.strip()
    vals.loc[lower.isin({"human", "homo sapiens"})] = "Homo sapiens"
    out.obs["organism"] = vals
    return out


def infer_control_mask_from_adata(adata) -> np.ndarray:
    """Return length ``n_obs`` boolean mask: True = control / unperturbed."""
    obs = adata.obs
    n = adata.n_obs

    if "control" in obs.columns:
        s = pd.to_numeric(obs["control"], errors="coerce")
        if bool(s.eq(1).any()):
            return s.eq(1).to_numpy(dtype=bool)

    if "perturbation" in obs.columns:
        pert = obs["perturbation"].astype(str).str.strip().str.lower()
        m = pert.isin(_CONTROL_TOKENS) | pert.eq("<na>")
        if bool(m.any()):
            return m.to_numpy(dtype=bool)

    if "gene" in obs.columns:
        g = obs["gene"].astype(str).str.strip().str.upper()
        m = g.isin({"CTRL", "CONTROL", "NTC", ""})
        if bool(m.any()):
            return m.to_numpy(dtype=bool)

    return np.zeros(n, dtype=bool)


def resolve_raw_paths(dataset_stem: str, raw_dirs: Tuple[Path, ...]) -> Optional[Path]:
    for d in raw_dirs:
        p = d / f"{dataset_stem}.h5ad"
        if p.is_file():
            return p.resolve()
    return None


def stack_embedding_uns(
    *,
    checkpoint: str,
    genelist: str,
    source_path: str,
    batch_size: int,
    num_workers: int,
    embedding_dim: int,
    mode: str,
) -> dict:
    return {
        "encoder": "stack",
        "checkpoint": str(checkpoint),
        "genelist": str(genelist),
        "source_path": str(source_path),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "embedding_dim": int(embedding_dim),
        "mode": str(mode),
    }


_GENE_COL_CANDIDATES = (
    "gene_symbol",
    "Gene_symbol",
    "gene_name",
    "Gene_name",
    "gene_names",
    "symbol",
    "Symbol",
    "rawSYMBOL",
    "lincsSYMBOL",
)


def resolve_gene_name_col(adata, override: Optional[str]) -> Optional[str]:
    """Column in ``adata.var`` holding gene symbols for Stack (or ``None``)."""
    if override:
        return str(override)
    for c in _GENE_COL_CANDIDATES:
        if c in adata.var.columns:
            return c
    return None


_STACK_TMP_GENE_COL = "__coupled_fm_stack_gene__"


def prepare_adata_for_stack_tmp(adata, gene_name_col: Optional[str]) -> tuple:
    """Return ``(adata_to_write, gene_col_for_stack_cli)`` for a temp ``.h5ad``.

    Vendored Stack reads gene names only from ``var`` datasets ``_index`` / ``index``
    or from ``gene_name_col``. Many CoupledFM exports omit ``var/_index`` in HDF5;
    we then fall back to a known symbol column or inject symbols from ``var_names``.

    Chemical perturbation datasets often use ``organism='human'``; Stack only accepts
    ``Homo sapiens`` when filtering is enabled, so we normalize before writing.
    """
    out = normalize_organism_for_stack_loader(adata.copy())
    col = resolve_gene_name_col(out, gene_name_col)
    if col is None:
        out.var[_STACK_TMP_GENE_COL] = out.var_names.astype(str)
        col = _STACK_TMP_GENE_COL
    return out, col


def discover_raw_stems(*raw_dirs: Path) -> list[str]:
    stems: set[str] = set()
    for d in raw_dirs:
        if not d.is_dir():
            continue
        stems.update(p.stem for p in d.glob("*.h5ad"))
    return sorted(stems)


def configure_logging(
    *,
    log_file: Optional[Path],
    console: bool,
    level: int = logging.INFO,
) -> None:
    """Attach handlers to the root logger (file-only by default)."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [pid=%(process)d] %(name)s: %(message)s",
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    logging.captureWarnings(True)
    warnings.simplefilter("default")


def install_tqdm_file_sink(progress_log: Optional[Path]) -> Callable[[], None]:
    """Force Stack/tqdm progress lines into a log file (non‑TTY friendly).

    Returns an uninstall callable for tests / subprocess hygiene.
    """

    if progress_log is None:

        def _noop():
            return

        return _noop

    progress_log.parent.mkdir(parents=True, exist_ok=True)
    tqdm_mod = __import__("tqdm.auto", fromlist=["tqdm"])
    tqdm_cls = getattr(tqdm_mod, "tqdm")

    fh = open(progress_log, "a", encoding="utf-8", buffering=1)
    orig_init = tqdm_cls.__init__

    @functools.wraps(orig_init)
    def _wrapped_init(self, *args, **kwargs):
        kwargs.setdefault("file", fh)
        kwargs.setdefault("mininterval", 2.0)
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("ascii", True)
        kwargs.setdefault("leave", True)
        return orig_init(self, *args, **kwargs)

    tqdm_cls.__init__ = _wrapped_init

    def _uninstall():
        tqdm_cls.__init__ = orig_init
        fh.close()

    return _uninstall
