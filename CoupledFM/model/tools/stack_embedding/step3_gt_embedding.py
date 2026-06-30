#!/usr/bin/env python3
"""Encode non-control (GT) cells with Stack → ``biFlow_data/gt_stack/{ds}.h5ad``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.tools.stack_embedding.common import (
    apply_temp_dir_env,
    configure_logging,
    coupled_fm_root,
    default_stack_tmp_dir,
    discover_raw_stems,
    ensure_stack_on_syspath,
    infer_control_mask_from_adata,
    install_tqdm_file_sink,
    prepare_adata_for_stack_tmp,
    resolve_raw_paths,
    stack_embedding_uns,
    stack_obs_mask,
)

LOGGER = logging.getLogger("stack_embed.gt")


def _run_one(
    dataset_stem: str,
    *,
    biflow_dir: Path,
    raw_dirs: tuple[Path, ...],
    checkpoint: Path,
    genelist: Path,
    batch_size: int,
    num_workers: int,
    gene_name_col: str | None,
    device: str,
    show_progress: bool,
    tmp_dir: Path,
    overwrite: bool,
) -> None:
    import anndata as ad

    ensure_stack_on_syspath()
    from stack.cli.embedding import extract_embeddings

    src = resolve_raw_paths(dataset_stem, raw_dirs)
    if src is None:
        raise FileNotFoundError(f"No raw h5ad for {dataset_stem!r} under {raw_dirs}")

    out_dir = biflow_dir / "gt_stack"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_stem}.h5ad"
    if out_path.is_file() and not overwrite:
        LOGGER.info("skip (exists): %s", out_path)
        return

    t0 = perf_counter()
    LOGGER.info("start dataset=%s mode=gt source=%s", dataset_stem, src)

    # Backed read keeps X lazy while we inspect obs/var and build the row mask.
    adata = ad.read_h5ad(src, backed="r")
    try:
        ctrl = infer_control_mask_from_adata(adata)
        mask_hs = stack_obs_mask(adata)
        keep = (~ctrl) & mask_hs
        n_keep = int(keep.sum())
        LOGGER.info(
            "selected dataset=%s mode=gt cells=%d/%d",
            dataset_stem,
            n_keep,
            adata.n_obs,
        )
        if n_keep == 0:
            warnings.warn(f"{dataset_stem}: zero GT cells after masks; skipping", stacklevel=2)
            return
        sub = adata[keep].to_memory()
    finally:
        adata.file.close()

    if sub.n_obs == 0:
        warnings.warn(f"{dataset_stem}: zero GT cells after mask; skipping", stacklevel=2)
        return

    td = tmp_dir
    td.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False, dir=str(td)) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_ad, gcol_stack = prepare_adata_for_stack_tmp(sub, gene_name_col)
        write_ad.write_h5ad(tmp_path)
        LOGGER.info("embedding dataset=%s mode=gt temp_h5ad=%s", dataset_stem, tmp_path)
        emb, _ = extract_embeddings(
            str(checkpoint),
            str(tmp_path),
            str(genelist),
            gene_name_col=gcol_stack,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            show_progress=show_progress,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if emb.shape[0] != sub.n_obs:
        raise RuntimeError(
            f"{dataset_stem}: embedding rows {emb.shape[0]} != obs {sub.n_obs} "
            "(organism / Stack filtering mismatch — check raw ``obs['organism']``)."
        )

    sub.obsm["emb"] = emb.astype("float32", copy=False)
    sub.uns["stack_embedding"] = stack_embedding_uns(
        checkpoint=str(checkpoint),
        genelist=str(genelist),
        source_path=str(src),
        batch_size=batch_size,
        num_workers=num_workers,
        embedding_dim=int(emb.shape[1]),
        mode="gt",
    )
    sub.write_h5ad(out_path)
    LOGGER.info(
        "done dataset=%s mode=gt cells=%d emb=%s out=%s elapsed=%.1fs",
        dataset_stem,
        sub.n_obs,
        emb.shape,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--biflow-dir",
        default=str(root / "data" / "latent_data" / "stack"),
        help="Stack embedding output root (contains control_stack/ and gt_stack/)",
        type=Path,
    )
    parser.add_argument(
        "--raw-genepert",
        default=str(root / "data" / "raw" / "genepert_DE5000"),
        type=Path,
    )
    parser.add_argument(
        "--raw-chemical",
        default=str(root / "data" / "raw" / "chemicalpert_DE5000"),
        type=Path,
    )
    parser.add_argument(
        "--checkpoint",
        default=str(root / "pretrainckpt" / "stack" / "bc_large.ckpt"),
        type=Path,
    )
    parser.add_argument(
        "--genelist",
        default=str(root / "pretrainckpt" / "stack" / "basecount_1000per_15000max.pkl"),
        type=Path,
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gene-name-col", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable Stack/tqdm embedding progress bars in logs",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="write logs here (default: logs/stack_embedding/gt_<UTCstamp>/)",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="also mirror logs to stderr (default: file-only)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="temp directory for Stack/anndata (default: <repo>/tmp/stack_embedding; large GWPS GT needs tens of GB free here—not $HOME)",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-j", "--jobs", type=int, default=1)
    args = parser.parse_args()

    raw_dirs = (args.raw_genepert.expanduser(), args.raw_chemical.expanduser())
    names = args.datasets if args.datasets else discover_raw_stems(*raw_dirs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_root = (
        args.log_dir.expanduser().resolve()
        if args.log_dir is not None
        else (coupled_fm_root() / "logs" / "stack_embedding" / f"gt_{stamp}")
    )
    log_root.mkdir(parents=True, exist_ok=True)
    main_log = log_root / "main.log"
    configure_logging(log_file=main_log, console=args.console)

    resolved_tmp = (
        args.tmp_dir.expanduser().resolve()
        if args.tmp_dir is not None
        else default_stack_tmp_dir()
    )
    apply_temp_dir_env(resolved_tmp)

    if not names:
        LOGGER.warning("No datasets found under raw dirs %s", raw_dirs)
        return

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_ids:
        gpu_ids = [0]

    if args.dry_run:
        LOGGER.info(
            "dry_run mode=gt log_root=%s tmp_dir=%s datasets=%d",
            log_root,
            resolved_tmp,
            len(names),
        )
        for ds in names:
            LOGGER.info("dry_run dataset=%s raw_path=%s", ds, resolve_raw_paths(ds, raw_dirs))
        return

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.genelist.is_file():
        raise FileNotFoundError(f"genelist not found: {args.genelist}")

    buckets: list[list[dict]] = [[] for _ in gpu_ids]
    for i, ds in enumerate(sorted(names)):
        buckets[i % len(gpu_ids)].append(
            {
                "dataset_stem": ds,
                "biflow_dir": args.biflow_dir.expanduser().resolve(),
                "raw_dirs": tuple(d.expanduser().resolve() for d in raw_dirs),
                "checkpoint": args.checkpoint.expanduser().resolve(),
                "genelist": args.genelist.expanduser().resolve(),
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "gene_name_col": args.gene_name_col,
                "device": args.device,
                "show_progress": not args.no_progress,
                "tmp_dir": resolved_tmp,
                "overwrite": args.overwrite,
            }
        )

    gpu_task_pairs = [(gpu, bucket) for gpu, bucket in zip(gpu_ids, buckets) if bucket]

    LOGGER.info(
        "plan mode=gt datasets=%d gpus=%s jobs=%d batch_size=%d num_workers=%d progress=%s log_root=%s main_log=%s",
        len(names),
        ",".join(str(g) for g in gpu_ids),
        args.jobs,
        args.batch_size,
        args.num_workers,
        not args.no_progress,
        log_root,
        main_log,
    )
    LOGGER.info("tmp_dir=%s (TMPDIR/TEMP/TMP redirected)", resolved_tmp)
    for gpu, bucket in gpu_task_pairs:
        LOGGER.info(
            "assignment gpu=%s datasets=%s",
            gpu,
            ",".join(t["dataset_stem"] for t in bucket),
        )

    show_progress = not args.no_progress

    if args.jobs <= 1 or len(gpu_task_pairs) <= 1:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        apply_temp_dir_env(resolved_tmp)
        for gpu, bucket in gpu_task_pairs:
            worker_log = log_root / f"gpu_{gpu}.log"
            progress_log = None if not show_progress else log_root / f"progress_gpu_{gpu}.log"
            configure_logging(log_file=worker_log, console=args.console)
            uninstall_tqdm = install_tqdm_file_sink(progress_log)
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                LOGGER.info(
                    "sequential_worker gpu=%s datasets=%d worker_log=%s",
                    gpu,
                    len(bucket),
                    worker_log,
                )
                if progress_log is not None:
                    LOGGER.info("progress_log gpu=%s path=%s", gpu, progress_log)
                for task_kwargs in bucket:
                    _run_one(**task_kwargs)
            finally:
                uninstall_tqdm()
                configure_logging(log_file=main_log, console=args.console)
        return

    max_workers = min(args.jobs, len(gpu_task_pairs))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
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
        for fut in as_completed(futures):
            fut.result()


if __name__ == "__main__":
    main()
