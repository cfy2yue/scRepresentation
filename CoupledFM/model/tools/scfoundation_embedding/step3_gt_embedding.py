#!/usr/bin/env python3
"""Encode non-control (GT) cells with scFoundation → ``latent_data/scfoundation/gt_scfoundation/{ds}.h5ad``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model.tools.scfoundation_embedding.common import (
    coupled_fm_root,
    default_log_dir,
    default_scfoundation_checkpoint,
    default_scfoundation_gene_tsv,
    default_scfoundation_tmp_dir,
    ensure_scfoundation_fm_paths,
    estimate_raw_n_obs,
    estimate_selected_n_obs,
    import_scldm_common,
    merge_sharded_outputs,
    plan_cell_shard_buckets,
    plan_gpu_buckets_greedy,
    plan_gpu_buckets_round_robin,
    scfoundation_embedding_uns,
)

LOGGER = logging.getLogger("scfoundation_embed.gt")


def _run_one(
    dataset_stem: str,
    *,
    out_root: Path,
    raw_dirs: tuple[Path, ...],
    raw_chemical_dir: Path,
    checkpoint: Path,
    gene_tsv: Path,
    chunk_size: int,
    version: str,
    pool_type: str,
    tgthighres: str,
    force_pert: bool,
    input_is_log1p: bool,
    device: str,
    tmp_dir: Path,
    overwrite: bool,
    shard_index: int = 0,
    shard_count: int = 1,
    selected_start: int = 0,
    selected_end: int = 0,
    n_cells_est: int = 0,
) -> None:
    c = import_scldm_common()
    import anndata as ad

    from adapters.scfoundation.encoder import encode_to_memmap

    src = c.resolve_raw_paths(dataset_stem, raw_dirs)
    if src is None:
        raise FileNotFoundError(f"No raw h5ad for {dataset_stem!r} under {raw_dirs}")

    out_dir = out_root / "gt_scfoundation"
    shard_dir = out_root / "_shards" / "gt_scfoundation" / dataset_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if shard_count > 1:
        shard_dir.mkdir(parents=True, exist_ok=True)
        out_path = shard_dir / f"part_{shard_index:04d}_of_{shard_count:04d}.h5ad"
        final_path = out_dir / f"{dataset_stem}.h5ad"
    else:
        out_path = out_dir / f"{dataset_stem}.h5ad"
        final_path = out_path
    if final_path.is_file() and not overwrite:
        LOGGER.info("skip (final exists): %s", final_path)
        return
    if out_path.is_file() and not overwrite:
        LOGGER.info("skip (shard exists): %s", out_path)
        return

    t0 = perf_counter()
    LOGGER.info(
        "start dataset=%s mode=gt_scfoundation shard=%d/%d selected_range=%d:%d source=%s",
        dataset_stem,
        shard_index + 1,
        shard_count,
        selected_start,
        selected_end,
        src,
    )

    adata = ad.read_h5ad(src)
    ctrl = c.infer_control_mask_for_source(adata, source_path=src, chemical_raw_dir=raw_chemical_dir)
    mask_hs = c.stack_obs_mask(adata)
    keep = (~ctrl) & mask_hs
    n_keep = int(keep.sum())
    LOGGER.info(
        "selected dataset=%s mode=gt_scfoundation cells=%d/%d",
        dataset_stem,
        n_keep,
        adata.n_obs,
    )
    if n_keep == 0:
        LOGGER.warning("%s: zero GT cells after masks; skipping", dataset_stem)
        return
    selected_idx = np.flatnonzero(keep)
    if shard_count > 1:
        end = selected_end if selected_end > selected_start else n_keep
        selected_idx = selected_idx[selected_start:end]
        LOGGER.info(
            "selected_shard dataset=%s shard=%d/%d cells=%d estimated=%d",
            dataset_stem,
            shard_index + 1,
            shard_count,
            len(selected_idx),
            n_cells_est,
        )
    sub = adata[selected_idx].copy()
    del adata

    if sub.n_obs == 0:
        LOGGER.warning("%s: zero GT cells after mask; skipping", dataset_stem)
        return

    sub = c.normalize_organism_for_scldm(sub)

    mmap_path = tmp_dir / f"scfoundation_gt_{dataset_stem}_{os.getpid()}_{id(sub)}.emb.memmap"
    mmap_arr, encoder_meta = encode_to_memmap(
        sub,
        mmap_path,
        checkpoint=str(checkpoint),
        gene_tsv=str(gene_tsv),
        version=version,
        pool_type=pool_type,
        tgthighres=tgthighres,
        force_pert=force_pert,
        input_is_log1p=input_is_log1p,
        device=device,
        chunk_size=chunk_size,
        progress_logger=LOGGER,
    )
    mmap_path_str = encoder_meta.pop("memmap_path", None)
    mmap_path = Path(mmap_path_str) if mmap_path_str else mmap_path

    hid = int(encoder_meta.get("hidden_dim", mmap_arr.shape[1]))
    emb_uns = scfoundation_embedding_uns(
        checkpoint=str(checkpoint),
        gene_tsv=str(gene_tsv),
        source_path=str(src),
        version=version,
        pool_type=pool_type,
        tgthighres=tgthighres,
        chunk_size=chunk_size,
        latent_dim=hid,
        mode="gt_scfoundation",
        encoder_meta=encoder_meta,
    )
    c.merge_uns_optional_pert_histogram(emb_uns, adata_slice=sub, force_pert=force_pert)

    mmap_ref = mmap_arr
    emb_view = mmap_ref.view(np.ndarray)
    try:
        sub.obsm["emb"] = emb_view
        sub.uns["scfoundation_embedding"] = emb_uns
        LOGGER.info(
            "write_h5ad dataset=%s cells=%s emb_dim=%s out=%s",
            dataset_stem,
            sub.n_obs,
            hid,
            out_path,
        )
        sub.write_h5ad(out_path)
    finally:
        sub.obsm.pop("emb", None)
        mmap_ref.flush()
        del emb_view
        del mmap_ref
        c.unlink_embedding_scratch(mmap_path)

    LOGGER.info(
        "done dataset=%s mode=gt_scfoundation cells=%d out=%s elapsed=%.1fs",
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
    console: bool,
    tmp_dir: Path,
) -> None:
    c = import_scldm_common()
    c.configure_logging(log_file=log_file, console=console)
    try:
        c.apply_temp_dir_env(tmp_dir)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        LOGGER.info("worker_start gpu=%s datasets=%d log=%s", gpu_id, len(task_kwargs_list), log_file)
        for i, kwargs in enumerate(task_kwargs_list, start=1):
            LOGGER.info(
                "worker_progress gpu=%s dataset_index=%d/%d dataset=%s",
                gpu_id,
                i,
                len(task_kwargs_list),
                kwargs["dataset_stem"],
            )
            ensure_scfoundation_fm_paths()
            _run_one(**kwargs)
        LOGGER.info("worker_done gpu=%s datasets=%d", gpu_id, len(task_kwargs_list))
    finally:
        pass


def main() -> None:
    root = coupled_fm_root()
    ck_def = default_scfoundation_checkpoint()
    gtsv_def = default_scfoundation_gene_tsv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--latent-root",
        default=str(root / "data" / "latent_data" / "scfoundation"),
        type=Path,
    )
    p.add_argument("--raw-genepert", default=str(root / "data" / "raw" / "genepert_DE5000"), type=Path)
    p.add_argument("--raw-chemical", default=str(root / "data" / "raw" / "chemicalpert_DE5000"), type=Path)
    p.add_argument("--checkpoint", default=str(ck_def), type=Path)
    p.add_argument("--gene-tsv", default=str(gtsv_def), type=Path)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--version", default="ce", choices=("ce", "rde"))
    p.add_argument("--pool-type", default="all", choices=("all", "max"))
    p.add_argument("--tgthighres", default="t4")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--force-pert", dest="force_pert", action="store_true", default=None)
    g.add_argument("--no-force-pert", dest="force_pert", action="store_false")
    p.set_defaults(force_pert=True)
    p.add_argument("--raw-counts-input", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-dir", default=None, type=Path)
    p.add_argument("--console", action="store_true")
    p.add_argument("--tmp-dir", default=None, type=Path)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6")
    p.add_argument(
        "--schedule",
        choices=("round-robin", "greedy"),
        default="greedy",
        help="Dataset→GPU assignment. Greedy splits large datasets into selected-cell shards.",
    )
    p.add_argument(
        "--target-cells-per-shard",
        type=int,
        default=0,
        help="0 = total selected cells / GPU count; lower values create more shards.",
    )
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-j", "--jobs", type=int, default=1)
    args = p.parse_args()

    c = import_scldm_common()
    raw_dirs = (args.raw_genepert.expanduser().resolve(), args.raw_chemical.expanduser().resolve())
    raw_chem = raw_dirs[1]
    names = args.datasets if args.datasets else c.discover_raw_stems(*raw_dirs)

    log_root = (
        args.log_dir.expanduser().resolve()
        if args.log_dir is not None
        else default_log_dir(coupled_fm_root(), "gt")
    )
    log_root.mkdir(parents=True, exist_ok=True)
    main_log = log_root / "main.log"
    c.configure_logging(log_file=main_log, console=args.console)

    resolved_tmp = (
        args.tmp_dir.expanduser().resolve() if args.tmp_dir is not None else default_scfoundation_tmp_dir()
    )
    c.apply_temp_dir_env(resolved_tmp)

    if not names:
        LOGGER.warning("No datasets under raw dirs %s", raw_dirs)
        return

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_ids:
        gpu_ids = [0]

    if args.dry_run:
        LOGGER.info(
            "dry_run mode=gt_scfoundation latent_root=%s tmp=%s datasets=%d schedule=%s target_cells_per_shard=%s",
            args.latent_root,
            resolved_tmp,
            len(names),
            args.schedule,
            args.target_cells_per_shard,
        )
        if args.schedule == "greedy":
            plan, weights = plan_cell_shard_buckets(
                names,
                raw_dirs=raw_dirs,
                raw_chemical_dir=raw_chem,
                gpu_ids=gpu_ids,
                mode="gt",
                target_cells_per_shard=args.target_cells_per_shard,
            )
            LOGGER.info("dry_run selected_total_cells=%d", sum(weights.values()))
            for gpu, specs in plan:
                LOGGER.info(
                    "dry_run gpu=%s cells=%d shards=%d shardspecs=%s",
                    gpu,
                    sum(int(s["n_cells_est"]) for s in specs),
                    len(specs),
                    [
                        f"{s['dataset_stem']}[{int(s['shard_index']) + 1}/{s['shard_count']}:{s['n_cells_est']}]"
                        for s in specs
                    ],
                )
        else:
            plan = plan_gpu_buckets_round_robin(names, gpu_ids)
            for gpu, dss in plan:
                LOGGER.info("dry_run gpu=%s n_datasets=%d datasets=%s", gpu, len(dss), dss)
        for ds in sorted(names):
            rp = c.resolve_raw_paths(ds, raw_dirs)
            LOGGER.info(
                "dry_run dataset=%s raw_path=%s raw_n_obs_est=%d selected_gt_est=%d",
                ds,
                rp,
                estimate_raw_n_obs(rp),
                estimate_selected_n_obs(
                    ds,
                    raw_dirs=raw_dirs,
                    raw_chemical_dir=raw_chem,
                    mode="gt",
                ),
            )
        return

    chk = args.checkpoint.expanduser().resolve()
    gtsv = args.gene_tsv.expanduser().resolve()
    for path, lbl in ((chk, "checkpoint"), (gtsv, "gene TSV")):
        if not path.is_file():
            raise FileNotFoundError(f"{lbl} missing: {path}")

    latent_out = args.latent_root.expanduser().resolve()
    latent_out.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        final_dir = latent_out / "gt_scfoundation"
        before = len(names)
        names = [ds for ds in names if not (final_dir / f"{ds}.h5ad").is_file()]
        skipped = before - len(names)
        if skipped:
            LOGGER.info("skip_existing_final_outputs mode=gt_scfoundation skipped=%d remaining=%d", skipped, len(names))
        if not names:
            LOGGER.info("all gt_scfoundation outputs already exist; nothing to do")
            return

    input_is_log1p = not args.raw_counts_input
    shared_kwargs_base = dict(
        out_root=latent_out,
        raw_dirs=raw_dirs,
        raw_chemical_dir=raw_chem,
        checkpoint=chk,
        gene_tsv=gtsv,
        chunk_size=int(args.chunk_size),
        version=args.version,
        pool_type=args.pool_type,
        tgthighres=args.tgthighres,
        force_pert=bool(args.force_pert),
        input_is_log1p=input_is_log1p,
        device=args.device,
        tmp_dir=resolved_tmp,
        overwrite=args.overwrite,
    )

    all_task_specs: list[dict] = []
    if args.schedule == "greedy":
        shard_plan, weights = plan_cell_shard_buckets(
            names,
            raw_dirs=raw_dirs,
            raw_chemical_dir=raw_chem,
            gpu_ids=gpu_ids,
            mode="gt",
            target_cells_per_shard=args.target_cells_per_shard,
        )
        gpu_task_pairs = [
            (gpu, [{**spec, **shared_kwargs_base} for spec in bucket])
            for gpu, bucket in shard_plan
        ]
        all_task_specs = [task for _, bucket in gpu_task_pairs for task in bucket]
        LOGGER.info("selected_total_cells=%d target_cells_per_shard=%s", sum(weights.values()), args.target_cells_per_shard)
    else:
        gpu_task_pairs = [
            (gpu, [{"dataset_stem": ds, **shared_kwargs_base} for ds in bucket])
            for gpu, bucket in plan_gpu_buckets_round_robin(names, gpu_ids)
        ]
        all_task_specs = [task for _, bucket in gpu_task_pairs for task in bucket]

    LOGGER.info(
        "plan mode=gt_scfoundation datasets=%d gpus=%s jobs=%d chunk=%s latent_root=%s log=%s schedule=%s",
        len(names),
        ",".join(str(g) for g in gpu_ids),
        args.jobs,
        args.chunk_size,
        latent_out,
        main_log,
        args.schedule,
    )
    LOGGER.info("tmp=%s TMPDIR redirected", resolved_tmp)
    for gpu, bucket in gpu_task_pairs:
        LOGGER.info(
            "bucket gpu=%s cells=%d tasks=%d specs=%s",
            gpu,
            sum(int(t.get("n_cells_est", 0)) for t in bucket),
            len(bucket),
            [
                f"{t['dataset_stem']}[{int(t.get('shard_index', 0)) + 1}/{int(t.get('shard_count', 1))}:{int(t.get('n_cells_est', 0))}]"
                for t in bucket
            ],
        )

    jobs = args.jobs
    if jobs <= 1 or len(gpu_task_pairs) <= 1:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        for gpu, bucket in gpu_task_pairs:
            worker_log = log_root / f"gpu_{gpu}.log"
            c.configure_logging(log_file=worker_log, console=args.console)
            try:
                c.apply_temp_dir_env(resolved_tmp)
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
                for task_kw in bucket:
                    ensure_scfoundation_fm_paths()
                    _run_one(**task_kw)
            finally:
                c.configure_logging(log_file=main_log, console=args.console)
        merge_sharded_outputs(
            out_root=latent_out,
            mode_dir="gt_scfoundation",
            task_specs=all_task_specs,
            overwrite=args.overwrite,
            logger=LOGGER,
        )
        return

    max_workers = min(jobs, len(gpu_task_pairs))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [
            pool.submit(
                _worker_entry,
                gpu,
                list(bucket),
                log_file=log_root / f"gpu_{gpu}.log",
                console=args.console,
                tmp_dir=resolved_tmp,
            )
            for gpu, bucket in gpu_task_pairs
        ]
        for f in as_completed(futs):
            f.result()
    merge_sharded_outputs(
        out_root=latent_out,
        mode_dir="gt_scfoundation",
        task_specs=all_task_specs,
        overwrite=args.overwrite,
        logger=LOGGER,
    )


if __name__ == "__main__":
    main()
