#!/usr/bin/env python3
"""Encode control cells with scLDM → ``latent_data/scldm/control_scldm/{ds}.h5ad``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.tools.scldm_embedding.common import (
    apply_temp_dir_env,
    coupled_fm_root,
    configure_logging,
    default_scldm_checkpoint_dir,
    default_scldm_tmp_dir,
    discover_raw_stems,
    infer_control_mask_for_source,
    install_tqdm_file_sink,
    merge_uns_optional_pert_histogram,
    normalize_organism_for_scldm,
    unlink_embedding_scratch,
    resolve_raw_paths,
    scldm_embedding_uns,
    ScldmChunkEncoder,
    stack_obs_mask,
)

LOGGER = logging.getLogger("scldm_embed.control")


def _run_one(
    dataset_stem: str,
    *,
    out_root: Path,
    raw_dirs: tuple[Path, ...],
    raw_chemical_dir: Path,
    checkpoint: Path,
    config_path: Path,
    gene_parquet: Path,
    batch_size: int,
    genes_seq_len: int,
    force_pert: bool,
    input_is_log1p: bool,
    device: str,
    tmp_dir: Path,
    overwrite: bool,
) -> None:
    import anndata as ad

    src = resolve_raw_paths(dataset_stem, raw_dirs)
    if src is None:
        raise FileNotFoundError(f"No raw h5ad for {dataset_stem!r} under {raw_dirs}")

    out_dir = out_root / "control_scldm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_stem}.h5ad"
    if out_path.is_file() and not overwrite:
        LOGGER.info("skip (exists): %s", out_path)
        return

    t0 = perf_counter()
    LOGGER.info("start dataset=%s mode=control_scldm source=%s", dataset_stem, src)

    # scLDM's environment currently drops backed sparse X on ``to_memory()`` for
    # these h5ad files, so use normal read_h5ad and keep X sparse until encoding.
    adata = ad.read_h5ad(src)
    ctrl = infer_control_mask_for_source(adata, source_path=src, chemical_raw_dir=raw_chemical_dir)
    mask_hs = stack_obs_mask(adata)
    keep = ctrl & mask_hs
    n_keep = int(keep.sum())
    LOGGER.info(
        "selected dataset=%s mode=control cells=%d/%d",
        dataset_stem,
        n_keep,
        adata.n_obs,
    )
    if n_keep == 0:
        LOGGER.warning("%s: zero control cells after masks; skipping", dataset_stem)
        return
    sub = adata[keep].copy()

    if sub.n_obs == 0:
        LOGGER.warning("%s: zero control cells after mask; skipping", dataset_stem)
        return

    sub = normalize_organism_for_scldm(sub)

    encoder = ScldmChunkEncoder(
        checkpoint=str(checkpoint),
        config=str(config_path),
        gene_parquet=str(gene_parquet),
        batch_size=batch_size,
        genes_seq_len=genes_seq_len,
        force_pert=force_pert,
        input_is_log1p=input_is_log1p,
        device=device,
        tmp_dir=tmp_dir,
    )
    mmap_arr, encoder_meta = encoder.encode_to_memmap(sub, dataset_stem=dataset_stem, progress_logger=LOGGER)
    mmap_path_str = encoder_meta.pop("memmap_path", None)
    mmap_path = Path(mmap_path_str) if mmap_path_str else None

    emb_uns = scldm_embedding_uns(
        checkpoint=str(checkpoint),
        config=str(config_path),
        gene_parquet=str(gene_parquet),
        source_path=str(src),
        batch_size=batch_size,
        latent_dim=encoder.latent_dim,
        mode="control_scldm",
        encoder_meta=encoder_meta,
    )
    merge_uns_optional_pert_histogram(emb_uns, adata_slice=sub, force_pert=force_pert)

    mmap_ref = mmap_arr
    emb_view = mmap_ref.view(np.ndarray)
    try:
        sub.obsm["emb"] = emb_view
        sub.uns["scldm_embedding"] = emb_uns
        LOGGER.info(
            "write_h5ad dataset=%s cells=%s emb_dim=%s out=%s",
            dataset_stem,
            sub.n_obs,
            encoder.latent_dim,
            out_path,
        )
        sub.write_h5ad(out_path)
    finally:
        sub.obsm.pop("emb", None)
        mmap_ref.flush()
        del emb_view
        del mmap_ref
        unlink_embedding_scratch(mmap_path)

    LOGGER.info(
        "done dataset=%s mode=control cells=%d out=%s elapsed=%.1fs",
        dataset_stem,
        sub.n_obs,
        out_path,
        perf_counter() - t0,
    )


def _worker_entry(
    gpu_id: int,
    task_kwargs_list: list[dict],
    *,
    log_file: Path,
    progress_log: Path | None,
    console: bool,
    tmp_dir: Path,
) -> None:
    configure_logging(log_file=log_file, console=console)
    uninstall_tqdm = install_tqdm_file_sink(progress_log)
    try:
        apply_temp_dir_env(tmp_dir)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        LOGGER.info("worker_start gpu=%s datasets=%d log=%s", gpu_id, len(task_kwargs_list), log_file)
        if progress_log is not None:
            LOGGER.info("worker_progress_log gpu=%s path=%s", gpu_id, progress_log)
        for i, kwargs in enumerate(task_kwargs_list, start=1):
            LOGGER.info(
                "worker_progress gpu=%s dataset_index=%d/%d dataset=%s",
                gpu_id,
                i,
                len(task_kwargs_list),
                kwargs["dataset_stem"],
            )
            _run_one(**kwargs)
        LOGGER.info("worker_done gpu=%s datasets=%d", gpu_id, len(task_kwargs_list))
    finally:
        uninstall_tqdm()


def main() -> None:
    root = coupled_fm_root()
    ck_root = default_scldm_checkpoint_dir()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--latent-root",
        default=str(root / "data" / "latent_data" / "scldm"),
        type=Path,
        help="Output root containing control_scldm/ and gt_scldm/",
    )
    p.add_argument("--raw-genepert", default=str(root / "data" / "raw" / "genepert_DE5000"), type=Path)
    p.add_argument("--raw-chemical", default=str(root / "data" / "raw" / "chemicalpert_DE5000"), type=Path)
    p.add_argument("--checkpoint", default=str(ck_root / "70M.ckpt"), type=Path)
    p.add_argument("--config", default=str(ck_root / "70M.yaml"), type=Path)
    p.add_argument(
        "--gene-parquet",
        default=str(ck_root / "concatenated_unique_genes.parquet"),
        type=Path,
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--genes-seq-len", type=int, default=8000)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--force-pert", dest="force_pert", action="store_true", default=None)
    g.add_argument("--no-force-pert", dest="force_pert", action="store_false")
    p.set_defaults(force_pert=True)
    p.add_argument(
        "--raw-counts-input",
        action="store_true",
        help="If set, expressions are counts (disable expm1 preprocessing).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm batch progress lines (omit progress_*.log output)",
    )
    p.add_argument(
        "--log-dir",
        default=None,
        type=Path,
        help=f"defaults to {coupled_fm_root() / 'logs' / 'scldm_embedding'} /control_<UTC>/",
    )
    p.add_argument("--console", action="store_true")
    p.add_argument("--tmp-dir", default=None, type=Path)
    p.add_argument("--gpus", default="0")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-j", "--jobs", type=int, default=1)
    args = p.parse_args()

    raw_dirs = (args.raw_genepert.expanduser().resolve(), args.raw_chemical.expanduser().resolve())
    raw_chem = raw_dirs[1]
    names = args.datasets if args.datasets else discover_raw_stems(*raw_dirs)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_root = (
        args.log_dir.expanduser().resolve()
        if args.log_dir is not None
        else (coupled_fm_root() / "logs" / "scldm_embedding" / f"control_{stamp}")
    )
    log_root.mkdir(parents=True, exist_ok=True)
    main_log = log_root / "main.log"
    configure_logging(log_file=main_log, console=args.console)

    resolved_tmp = (
        args.tmp_dir.expanduser().resolve() if args.tmp_dir is not None else default_scldm_tmp_dir()
    )
    apply_temp_dir_env(resolved_tmp)

    if not names:
        LOGGER.warning("No datasets under raw dirs %s", raw_dirs)
        return

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_ids:
        gpu_ids = [0]

    if args.dry_run:
        LOGGER.info(
            "dry_run mode=control latent_root=%s tmp=%s datasets=%d",
            args.latent_root,
            resolved_tmp,
            len(names),
        )
        for ds in names:
            LOGGER.info(
                "dry_run dataset=%s raw_path=%s",
                ds,
                resolve_raw_paths(ds, raw_dirs),
            )
        return

    chk = args.checkpoint.expanduser().resolve()
    cfg_p = args.config.expanduser().resolve()
    pq = args.gene_parquet.expanduser().resolve()
    for path, lbl in (
        (chk, "checkpoint"),
        (cfg_p, "config"),
        (pq, "gene parquet"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{lbl} missing: {path}")

    latent_out = args.latent_root.expanduser().resolve()
    latent_out.mkdir(parents=True, exist_ok=True)

    buckets: list[list[dict]] = [[] for _ in gpu_ids]
    input_is_log1p = not args.raw_counts_input

    shared_kwargs_base = dict(
        out_root=latent_out,
        raw_dirs=raw_dirs,
        raw_chemical_dir=raw_chem,
        checkpoint=chk,
        config_path=cfg_p,
        gene_parquet=pq,
        batch_size=args.batch_size,
        genes_seq_len=args.genes_seq_len,
        force_pert=bool(args.force_pert),
        input_is_log1p=input_is_log1p,
        device=args.device,
        tmp_dir=resolved_tmp,
        overwrite=args.overwrite,
    )
    for i, ds in enumerate(sorted(names)):
        buckets[i % len(gpu_ids)].append({"dataset_stem": ds, **shared_kwargs_base})

    gpu_task_pairs = [(gpu, bucket) for gpu, bucket in zip(gpu_ids, buckets) if bucket]

    LOGGER.info(
        "plan mode=control_scldm datasets=%d gpus=%s jobs=%d batch=%d latent_root=%s log=%s",
        len(names),
        ",".join(str(g) for g in gpu_ids),
        args.jobs,
        args.batch_size,
        latent_out,
        main_log,
    )
    LOGGER.info("tmp=%s TMPDIR redirected", resolved_tmp)

    show_progress = not args.no_progress
    jobs = args.jobs

    if jobs <= 1 or len(gpu_task_pairs) <= 1:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        for gpu, bucket in gpu_task_pairs:
            worker_log = log_root / f"gpu_{gpu}.log"
            progress_log = None if not show_progress else log_root / f"progress_gpu_{gpu}.log"
            configure_logging(log_file=worker_log, console=args.console)
            uninstall_tqdm = install_tqdm_file_sink(progress_log if show_progress else None)
            try:
                apply_temp_dir_env(resolved_tmp)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                for task_kw in bucket:
                    _run_one(**task_kw)
            finally:
                uninstall_tqdm()
                configure_logging(log_file=main_log, console=args.console)
        return

    max_workers = min(jobs, len(gpu_task_pairs))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [
            pool.submit(
                _worker_entry,
                gpu,
                list(bucket),
                log_file=log_root / f"gpu_{gpu}.log",
                progress_log=None if not show_progress else log_root / f"progress_gpu_{gpu}.log",
                console=args.console,
                tmp_dir=resolved_tmp,
            )
            for gpu, bucket in gpu_task_pairs
        ]
        for f in as_completed(futs):
            f.result()


if __name__ == "__main__":
    main()
