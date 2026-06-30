"""Shared helpers for scFoundation control / GT embedding scripts.

NOTE (delivery scope): this module is NOT exercised by the two delivered
flows (raw flow pretrain & CoupledFM sweep / CellNavi-vs-scGPT compare).
It depends on a sibling ``scFM/`` checkout and is kept only for offline
regeneration of pre-exported embedding caches under
``<delivery_root>/pretrainckpt/genepert_cache/``. Skip unless rebuilding
caches from raw scFoundation checkpoints.
"""

from __future__ import annotations

import sys
import gc
from pathlib import Path
from collections import defaultdict
from math import ceil
from typing import Any, Mapping, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FM_ROOT = _REPO_ROOT / "scFM" / "fm"


def coupled_fm_root() -> Path:
    return _REPO_ROOT.resolve()


def default_scfoundation_tmp_dir() -> Path:
    d = coupled_fm_root() / "tmp" / "scfoundation_embedding"
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def ensure_scfoundation_fm_paths() -> None:
    p = str(_FM_ROOT.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def default_scfoundation_checkpoint() -> Path:
    return (coupled_fm_root() / "scFM" / "pretrained" / "scFoundation" / "models.ckpt").resolve()


def default_scfoundation_gene_tsv() -> Path:
    return (_FM_ROOT / "third_party" / "scFoundation" / "model" / "OS_scRNA_gene_index.19264.tsv").resolve()


def estimate_raw_n_obs(path: Optional[Path]) -> int:
    if path is None or not path.is_file():
        return 0
    import anndata as ad

    b = ad.read_h5ad(path, backed="r")
    try:
        return int(b.n_obs)
    finally:
        b.file.close()


def estimate_selected_n_obs(
    dataset_stem: str,
    *,
    raw_dirs: Tuple[Path, ...],
    raw_chemical_dir: Path,
    mode: str,
) -> int:
    """Estimate target cells for scheduling without loading expression ``X``."""

    from model.tools.scldm_embedding import common as c

    path = c.resolve_raw_path(dataset_stem, raw_dirs)
    if path is None or not path.is_file():
        return 0

    import anndata as ad

    b = ad.read_h5ad(path, backed="r")
    try:
        ctrl = c.infer_control_mask_for_source(
            b,
            source_path=path,
            chemical_raw_dir=raw_chemical_dir,
        )
        mask_hs = c.stack_obs_mask(b)
        if mode == "control":
            keep = ctrl & mask_hs
        elif mode == "gt":
            keep = (~ctrl) & mask_hs
        else:
            raise ValueError("mode must be 'control' or 'gt'")
        return int(keep.sum())
    finally:
        b.file.close()


def plan_gpu_buckets_round_robin(names: list[str], gpu_ids: list[int]) -> list[tuple[int, list[str]]]:
    buckets: list[list[str]] = [[] for _ in gpu_ids]
    for i, ds in enumerate(sorted(names)):
        buckets[i % len(gpu_ids)].append(ds)
    return [(gpu_ids[i], buckets[i]) for i in range(len(gpu_ids)) if buckets[i]]


def plan_gpu_buckets_greedy(
    names: list[str],
    *,
    raw_dirs: Tuple[Path, ...],
    raw_chemical_dir: Optional[Path] = None,
    gpu_ids: list[int],
    mode: Optional[str] = None,
) -> list[tuple[int, list[str]]]:
    from model.tools.scldm_embedding.common import resolve_raw_path

    weights: dict[str, int] = {}
    for ds in names:
        if mode is not None and raw_chemical_dir is not None:
            weights[ds] = estimate_selected_n_obs(
                ds,
                raw_dirs=raw_dirs,
                raw_chemical_dir=raw_chemical_dir,
                mode=mode,
            )
        else:
            weights[ds] = estimate_raw_n_obs(resolve_raw_path(ds, raw_dirs))

    order = sorted(names, key=lambda d: weights.get(d, 0), reverse=True)
    buckets: dict[int, list[str]] = {g: [] for g in gpu_ids}
    load: dict[int, int] = {g: 0 for g in gpu_ids}
    for ds in order:
        g = min(gpu_ids, key=lambda x: load[x])
        buckets[g].append(ds)
        load[g] += int(weights.get(ds, 0))
    return [(g, buckets[g]) for g in gpu_ids if buckets[g]]


def plan_cell_shard_buckets(
    names: list[str],
    *,
    raw_dirs: Tuple[Path, ...],
    raw_chemical_dir: Path,
    gpu_ids: list[int],
    mode: str,
    target_cells_per_shard: Optional[int] = None,
) -> tuple[list[tuple[int, list[dict[str, int | str]]]], dict[str, int]]:
    """Split large datasets into selected-cell shards and greedily balance GPUs."""

    weights = {
        ds: estimate_selected_n_obs(
            ds,
            raw_dirs=raw_dirs,
            raw_chemical_dir=raw_chemical_dir,
            mode=mode,
        )
        for ds in names
    }
    total = sum(weights.values())
    if not gpu_ids:
        gpu_ids = [0]
    if target_cells_per_shard is None or target_cells_per_shard <= 0:
        target_cells_per_shard = max(1, int(ceil(total / max(1, len(gpu_ids)))))

    shards: list[dict[str, int | str]] = []
    for ds in sorted(names):
        n = int(weights.get(ds, 0))
        if n <= 0:
            continue
        n_shards = max(1, int(ceil(n / target_cells_per_shard)))
        shard_size = int(ceil(n / n_shards))
        for shard_index in range(n_shards):
            start = shard_index * shard_size
            end = min(n, start + shard_size)
            if end <= start:
                continue
            shards.append(
                {
                    "dataset_stem": ds,
                    "shard_index": shard_index,
                    "shard_count": n_shards,
                    "selected_start": start,
                    "selected_end": end,
                    "n_cells_est": end - start,
                }
            )

    buckets: dict[int, list[dict[str, int | str]]] = {g: [] for g in gpu_ids}
    load: dict[int, int] = {g: 0 for g in gpu_ids}
    for shard in sorted(shards, key=lambda x: int(x["n_cells_est"]), reverse=True):
        gpu = min(gpu_ids, key=lambda x: load[x])
        buckets[gpu].append(shard)
        load[gpu] += int(shard["n_cells_est"])

    return [(g, buckets[g]) for g in gpu_ids if buckets[g]], weights


def merge_sharded_outputs(
    *,
    out_root: Path,
    mode_dir: str,
    task_specs: list[dict],
    overwrite: bool,
    logger: Any,
) -> None:
    """Merge per-dataset shard h5ads into final ``mode_dir/{dataset}.h5ad`` files."""

    grouped: dict[str, dict[int, Path]] = defaultdict(dict)
    expected: dict[str, int] = {}
    shard_root = out_root / "_shards" / mode_dir
    for spec in task_specs:
        shard_count = int(spec.get("shard_count", 1))
        if shard_count <= 1:
            continue
        ds = str(spec["dataset_stem"])
        idx = int(spec["shard_index"])
        expected[ds] = shard_count
        grouped[ds][idx] = shard_root / ds / f"part_{idx:04d}_of_{shard_count:04d}.h5ad"

    if not grouped:
        return

    import anndata as ad

    final_dir = out_root / mode_dir
    final_dir.mkdir(parents=True, exist_ok=True)
    for ds, parts_by_idx in grouped.items():
        out_path = final_dir / f"{ds}.h5ad"
        if out_path.is_file() and not overwrite:
            logger.info("merge skip final exists: %s", out_path)
            continue

        n_expected = expected[ds]
        missing = [i for i in range(n_expected) if not parts_by_idx.get(i, Path()).is_file()]
        if missing:
            raise FileNotFoundError(f"{ds}: missing shard outputs {missing}")

        paths = [parts_by_idx[i] for i in range(n_expected)]
        logger.info("merge_shards dataset=%s parts=%d out=%s", ds, len(paths), out_path)
        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_out.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)

        # ``concat_on_disk`` in anndata 0.11 can fail when merged ``var`` columns
        # are represented as pandas Series.  Shards are few and already selected,
        # so in-memory concat is more reliable and still bounded by shard size.
        parts = [ad.read_h5ad(p) for p in paths]
        merged = ad.concat(parts, axis=0, join="outer", merge="first", uns_merge="first")
        if "scfoundation_embedding" not in merged.uns and parts and "scfoundation_embedding" in parts[0].uns:
            merged.uns["scfoundation_embedding"] = dict(parts[0].uns["scfoundation_embedding"])
        merged.write_h5ad(tmp_out)
        tmp_out.replace(out_path)
        del merged
        del parts
        gc.collect()
        for p in paths:
            p.unlink(missing_ok=True)
        shard_dir = shard_root / ds
        try:
            shard_dir.rmdir()
        except OSError:
            pass
        logger.info("merge_done dataset=%s out=%s", ds, out_path)


def scfoundation_embedding_uns(
    *,
    checkpoint: str,
    gene_tsv: str,
    source_path: str,
    version: str,
    pool_type: str,
    tgthighres: str,
    chunk_size: int,
    latent_dim: int,
    mode: str,
    encoder_meta: Mapping[str, Any],
) -> dict:
    meta = dict(encoder_meta)
    return {
        "encoder": "scfoundation",
        "checkpoint": str(checkpoint),
        "gene_tsv": str(gene_tsv),
        "source_path": str(source_path),
        "source": str(source_path),
        "version": str(version),
        "pool_type": str(pool_type),
        "tgthighres": str(tgthighres),
        "chunk_size_cells": int(chunk_size),
        "latent_dim": int(latent_dim),
        "embedding_dim": int(latent_dim),
        "mode": str(mode),
        "meta": meta,
    }


def default_log_dir(repo_root: Path, mode_prefix: str) -> Path:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = repo_root / "logs" / "scfoundation_embedding" / f"{mode_prefix}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def import_scldm_common():
    """Deferred import to reuse logging, masks, and raw discovery."""

    from model.tools.scldm_embedding import common as c

    return c
