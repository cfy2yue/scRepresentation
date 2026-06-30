"""Shared helpers for scLDM control/GT embedding scripts.

NOTE (delivery scope): this module is NOT exercised by the two delivered
flows (raw flow pretrain & CoupledFM sweep / CellNavi-vs-scGPT compare).
It depends on a sibling ``scFM/`` checkout and is kept only for offline
regeneration of pre-exported embedding caches under
``<delivery_root>/pretrainckpt/genepert_cache/``. Skip unless rebuilding
caches from raw scLDM checkpoints.
"""

from __future__ import annotations

import gc
import functools
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FM_ROOT = _REPO_ROOT / "scFM" / "fm"
_SCLDM_SRC = _FM_ROOT / "third_party" / "scldm" / "src"

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


def default_scldm_tmp_dir() -> Path:
    d = coupled_fm_root() / "tmp" / "scldm_embedding"
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def scldm_fm_root() -> Path:
    return _FM_ROOT.resolve()


def scldm_third_party_src() -> Path:
    return _SCLDM_SRC.resolve()


def apply_temp_dir_env(tmp_dir: Path) -> None:
    p = str(Path(tmp_dir).expanduser().resolve())
    Path(p).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = p
    os.environ["TEMP"] = p
    os.environ["TMP"] = p
    tempfile.tempdir = p


def ensure_scldm_paths() -> None:
    for p in (str(_FM_ROOT), str(_SCLDM_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)


def configure_logging(
    *,
    log_file: Optional[Path],
    console: bool,
    level: int = logging.INFO,
) -> None:
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
    if progress_log is None:

        def _noop() -> None:
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
        kwargs.setdefault("mininterval", 5.0)
        kwargs.setdefault("dynamic_ncols", False)
        kwargs.setdefault("ascii", True)
        kwargs.setdefault("leave", True)
        return orig_init(self, *args, **kwargs)

    tqdm_cls.__init__ = _wrapped_init

    def _uninstall() -> None:
        tqdm_cls.__init__ = orig_init
        fh.close()

    return _uninstall


def discover_raw_stems(*raw_dirs: Path) -> list[str]:
    stems: set[str] = set()
    for d in raw_dirs:
        if not d.is_dir():
            continue
        stems.update(p.stem for p in d.glob("*.h5ad"))
    return sorted(stems)


def resolve_raw_path(dataset_stem: str, raw_dirs: Tuple[Path, ...]) -> Optional[Path]:
    for d in raw_dirs:
        p = d / f"{dataset_stem}.h5ad"
        if p.is_file():
            return p.resolve()
    return None


def resolve_raw_paths(dataset_stem: str, raw_dirs: Tuple[Path, ...]) -> Optional[Path]:
    """Alias for ``resolve_raw_path`` (same basename as Stack scripts)."""

    return resolve_raw_path(dataset_stem, raw_dirs)


def stack_obs_mask(adata) -> np.ndarray:
    if "organism" not in adata.obs.columns:
        return np.ones(adata.n_obs, dtype=bool)
    org = adata.obs["organism"].astype(str).str.strip()
    lower = org.str.lower()
    mask = (org == "Homo sapiens") | lower.isin({"human", "homo sapiens"})
    if bool(mask.any()):
        return mask.to_numpy(dtype=bool)
    return np.ones(adata.n_obs, dtype=bool)


def infer_control_mask_from_adata(adata) -> np.ndarray:
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


def infer_control_mask_for_source(
    adata,
    *,
    source_path: Path,
    chemical_raw_dir: Path,
) -> np.ndarray:
    """Control mask aligned with Stack, with a chemical‑pert explicit rule."""

    chemical_root = chemical_raw_dir.expanduser().resolve()
    src = source_path.expanduser().resolve()
    obs = adata.obs
    is_chemical = src.parent.resolve() == chemical_root

    if is_chemical and "control" in obs.columns:
        s = pd.to_numeric(obs["control"], errors="coerce")
        return s.eq(1).to_numpy(dtype=bool)

    return infer_control_mask_from_adata(adata)


def normalize_organism_for_scldm(adata):
    out = adata.copy()
    if "organism" not in out.obs.columns:
        return out
    vals = out.obs["organism"].astype(str).copy()
    lower = vals.str.lower().str.strip()
    vals.loc[lower.isin({"human", "homo sapiens"})] = "Homo sapiens"
    out.obs["organism"] = vals
    return out


def default_scldm_checkpoint_dir() -> Path:
    return (coupled_fm_root() / "scFM" / "pretrained" / "scdlm" / "vae_census").resolve()


def scldm_embedding_uns(
    *,
    checkpoint: str,
    config: str,
    gene_parquet: str,
    source_path: str,
    batch_size: int,
    latent_dim: int,
    mode: str,
    encoder_meta: Mapping[str, Any],
) -> dict:
    meta = dict(encoder_meta)
    return {
        "encoder": "scldm",
        "checkpoint": str(checkpoint),
        "config": str(config),
        "gene_parquet": str(gene_parquet),
        "source_path": str(source_path),
        "source": str(source_path),
        "batch": int(batch_size),
        "batch_size": int(batch_size),
        "latent_dim": int(latent_dim),
        "embedding_dim": int(latent_dim),
        "mode": str(mode),
        "meta": meta,
    }


def merge_uns_optional_pert_histogram(
    uns_dict: MutableMapping[str, Any],
    *,
    adata_slice,
    force_pert: bool,
) -> None:
    """Attach ``pert_kept_histogram`` into ``meta`` when pert indices exist."""

    if not force_pert:
        return
    pv = adata_slice.obsm.get("pert_var_idx", None)
    if pv is None:
        return
    ensure_scldm_paths()
    from adapters._common import histogram_pert_kept

    m = np.asarray(pv, dtype=np.int32)
    counts_per_cell = [int(np.sum(row >= 0)) for row in m]
    meta_block = uns_dict.get("meta")
    if not isinstance(meta_block, dict):
        return
    meta_block["pert_kept_histogram"] = histogram_pert_kept(counts_per_cell)


def unlink_embedding_scratch(mmap_path: Optional[Path]) -> None:
    """Invoke after flushing and deleting the mmap: reclaim refs and unlink the temp backing file."""

    gc.collect()
    if mmap_path is not None:
        mmap_path.unlink(missing_ok=True)


class ScldmChunkEncoder:
    """Load scLDM VAE once; encode AnnData chunks with batch‑sized vocab matrices only."""

    def __init__(
        self,
        *,
        checkpoint: Optional[str],
        config: Optional[str],
        gene_parquet: Optional[str],
        batch_size: int,
        genes_seq_len: int,
        force_pert: bool,
        input_is_log1p: bool,
        device: str,
        tmp_dir: Path,
    ) -> None:
        ensure_scldm_paths()
        import torch

        self.torch = torch
        from adapters.scldm import encoder as adapter

        self.adapter = adapter
        ckpt_root = default_scldm_checkpoint_dir()
        self.checkpoint = Path(
            checkpoint
            or os.environ.get("LATENT_BENCH_SCLDM_CKPT", str(ckpt_root / "70M.ckpt"))
        )
        self.config = Path(
            config or os.environ.get("LATENT_BENCH_SCLDM_CFG", str(ckpt_root / "70M.yaml"))
        )
        self.gene_parquet = Path(
            gene_parquet
            or os.environ.get("LATENT_BENCH_SCLDM_GENES", str(ckpt_root / "concatenated_unique_genes.parquet"))
        )
        for p, label in (
            (self.checkpoint, "checkpoint"),
            (self.config, "config"),
            (self.gene_parquet, "gene parquet"),
        ):
            if not p.is_file():
                raise FileNotFoundError(f"scLDM {label} not found: {p}")

        self.batch_size = int(batch_size)
        self.genes_seq_len = int(genes_seq_len)
        self.force_pert = bool(force_pert)
        self.input_is_log1p = bool(input_is_log1p)

        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved_device)

        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.module = adapter._build_module(self.config, self.checkpoint, self.device)
        self.vae_model = self.module.vae_model
        self.vae_model.eval()

        genes_df = pd.read_parquet(self.gene_parquet)
        self.vocab_ids = genes_df["feature_id"].astype(str).tolist()
        self.vocab_names = genes_df["feature_name"].astype(str).tolist()
        self.n_vocab = len(self.vocab_ids)
        self.genes_full_row = torch.arange(1, self.n_vocab + 1, dtype=torch.long)
        self.latent_dim = int(self.vae_model.encoder.latent_dim * self.vae_model.encoder.latent_embedding)

    def _protected_rows(
        self, adata_batch, old_to_vocab: np.ndarray
    ) -> tuple[list[list[int]], bool]:
        pv_raw = adata_batch.obsm.get("pert_var_idx", None)
        pert_present = pv_raw is not None
        if not self.force_pert or pv_raw is None:
            return [[] for _ in range(adata_batch.n_obs)], pert_present
        pv = np.asarray(pv_raw, dtype=np.int64)
        rows: list[list[int]] = []
        for i in range(adata_batch.n_obs):
            row = pv[i] if i < pv.shape[0] else []
            mapped: list[int] = []
            seen: set[int] = set()
            for x in np.asarray(row).ravel():
                j = int(x)
                if j < 0 or j >= len(old_to_vocab):
                    continue
                jj = int(old_to_vocab[j])
                if jj < 0 or jj in seen:
                    continue
                seen.add(jj)
                mapped.append(jj)
            rows.append(mapped)
        return rows, pert_present

    def encode_to_memmap(
        self, adata, *, dataset_stem: str, progress_logger: logging.Logger
    ) -> tuple[np.memmap, dict]:
        from tqdm.auto import tqdm

        n_cells = adata.n_obs
        emb_path = self.tmp_dir / f"scldm_{dataset_stem}_{os.getpid()}_{id(adata)}.emb.memmap"
        out = np.memmap(emb_path, dtype=np.float32, mode="w+", shape=(n_cells, self.latent_dim))

        vocab_hits = 0
        n_batches = (n_cells + self.batch_size - 1) // self.batch_size
        any_pert = False
        use_amp = self.device.type == "cuda"
        pbar = tqdm(total=n_batches, desc=f"scldm:{dataset_stem}", unit="batch")

        try:
            for bidx, start in enumerate(range(0, n_cells, self.batch_size), start=1):
                end = min(start + self.batch_size, n_cells)
                bs = end - start
                batch = adata[start:end].copy()
                X_full, hits, old_to_vocab = self.adapter._align_expression_to_vocab(
                    batch,
                    self.vocab_ids,
                    self.vocab_names,
                )
                if hits == 0:
                    emb_path.unlink(missing_ok=True)
                    raise RuntimeError(f"scLDM: no genes matched for {dataset_stem} batch {bidx}/{n_batches}")
                if bidx == 1:
                    vocab_hits = int(hits)

                if self.input_is_log1p:
                    X_full = np.expm1(np.clip(X_full, 0.0, None)).astype(np.float32, copy=False)

                protected_rows, pert_present = self._protected_rows(batch, old_to_vocab)
                any_pert = any_pert or pert_present

                genes_subset_np, counts_subset_np = self.adapter._build_expressed_subset(
                    X_full,
                    self.genes_seq_len,
                    protected_vocab_indices=protected_rows,
                )

                counts_t = self.torch.from_numpy(X_full).to(self.device)
                genes_t = self.genes_full_row.unsqueeze(0).expand(bs, -1).to(self.device)
                counts_sub_t = self.torch.from_numpy(counts_subset_np).to(self.device)
                genes_sub_t = self.torch.from_numpy(genes_subset_np).to(self.device)

                with self.torch.no_grad(), self.torch.amp.autocast(
                    "cuda",
                    enabled=use_amp,
                    dtype=self.torch.bfloat16,
                ):
                    z = self.vae_model.encode(counts_t, genes_t, counts_sub_t, genes_sub_t)
                out[start:end] = z.flatten(start_dim=1).float().cpu().numpy()
                out.flush()
                pbar.update(1)
                if bidx == 1 or bidx % 25 == 0 or bidx == n_batches:
                    progress_logger.info(
                        "encoded dataset=%s batch=%d/%d cells=%d/%d",
                        dataset_stem,
                        bidx,
                        n_batches,
                        end,
                        n_cells,
                    )
        except BaseException:
            try:
                pbar.close()
            finally:
                try:
                    out.flush()
                    del out
                    gc.collect()
                except Exception:
                    pass
                emb_path.unlink(missing_ok=True)
            raise

        pbar.close()
        meta: dict[str, Any] = {
            "encoder_role": "ExpressionOnlyEncoder",
            "latent_dim": int(self.latent_dim),
            "hidden_dim": int(self.latent_dim),
            "vocab_hits": int(vocab_hits),
            "vocab_size": int(self.n_vocab),
            "force_pert": bool(self.force_pert),
            "pert_var_idx_present": bool(any_pert),
            "force_pert_effective": bool(self.force_pert and any_pert),
            "pert_source": "obsm_pert_var_idx" if any_pert else None,
            "input_is_log1p": bool(self.input_is_log1p),
            "memmap_path": str(emb_path),
        }
        return out, meta
