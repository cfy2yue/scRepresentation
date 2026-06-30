#!/usr/bin/env python3
"""Evaluate a LatentFM checkpoint at per-condition residual granularity.

The standard LatentFM evaluators report aggregate split/family metrics.  This
CLI writes one row per condition so we can inspect which perturbations improve
or fail, and whether the predicted residual retrieves the true target residual
from the condition bank.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from model.latent.config import Config
from model.latent.dataset import CrossDatasetFMDataset
from model.latent.eval_condition_families import (
    build_family_group_splits,
    _genes,
    _is_drug,
    _load_condition_metadata,
    _pert_type,
)
from model.latent.eval_split_groups import (
    DEFAULT_GROUPS as SPLIT_GROUPS,
    _group_as_test_split,
    _json_default,
    _load_cfg,
    _load_manifest,
    _load_means,
    _load_split,
)
from model.latent.fm_ot import median_sigmas, mmd2_unbiased
from model.latent.train import (
    _cross_dataset_kw,
    _model_uses_pert,
    _pearson_np,
    _pert_chunk,
    _pert_for_eval_batch,
    _pert_to_device,
    build_model,
    ode_integrate,
)
from model.utils.train.ema import ModelEMA


DEFAULT_GROUPS = (
    "test",
    "test_single",
    "test_multi",
    "test_multi_seen",
    "test_multi_unseen1",
    "test_multi_unseen2",
    "family_gene",
    "family_drug",
)


def _as_float(v: float | np.floating | None) -> float | None:
    if v is None:
        return None
    out = float(v)
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float32)))


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return 0.0
    return float(np.dot(a, b) / (denom + eps))


def _residual_retrieval_metrics(
    pred: np.ndarray,
    targets: np.ndarray,
    *,
    true_index: int,
    top_ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Rank the true residual by cosine similarity against a residual bank."""
    pred_v = np.asarray(pred, dtype=np.float32).reshape(-1)
    target_m = np.asarray(targets, dtype=np.float32)
    if target_m.ndim != 2:
        raise ValueError(f"targets must be 2D, got shape={target_m.shape}")
    if not 0 <= int(true_index) < int(target_m.shape[0]):
        raise IndexError(f"true_index out of range: {true_index}")
    if target_m.shape[1] != pred_v.shape[0]:
        raise ValueError(
            f"pred dim {pred_v.shape[0]} does not match target dim {target_m.shape[1]}"
        )

    pred_norm = np.linalg.norm(pred_v)
    target_norms = np.linalg.norm(target_m, axis=1)
    denom = np.maximum(pred_norm * target_norms, 1e-12)
    sims = (target_m @ pred_v) / denom
    order = np.argsort(-sims, kind="mergesort")
    rank_pos = np.where(order == int(true_index))[0]
    rank = int(rank_pos[0]) + 1 if rank_pos.size else int(target_m.shape[0]) + 1

    out: dict[str, Any] = {
        "retrieval_rank": rank,
        "retrieval_best_index": int(order[0]) if order.size else None,
        "retrieval_best_similarity": _as_float(sims[order[0]]) if order.size else None,
        "retrieval_true_similarity": _as_float(sims[int(true_index)]),
    }
    for k in top_ks:
        kk = max(1, int(k))
        out[f"retrieval_top{kk}"] = bool(rank <= kk)
    return out


def _split_group_splits(
    *,
    manifest: dict[str, Any],
    split: dict[str, dict[str, list[str]]],
    groups: Iterable[str],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    out: dict[str, dict[str, dict[str, list[str]]]] = {}
    for group in groups:
        if group in SPLIT_GROUPS:
            gs = _group_as_test_split(split=split, manifest=manifest, group=group)
            if gs:
                out[group] = gs
    return out


def _condition_group_index(
    group_splits: dict[str, dict[str, dict[str, list[str]]]],
) -> dict[tuple[str, str], list[str]]:
    idx: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group, by_ds in group_splits.items():
        for ds_name, parts in by_ds.items():
            for cond in parts.get("test", []):
                idx[(str(ds_name), str(cond))].append(str(group))
    return {k: sorted(set(v)) for k, v in idx.items()}


def _selected_split(
    cond_groups: dict[tuple[str, str], list[str]],
    *,
    max_conditions: int | None,
    max_conditions_per_group: int | None,
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    if max_conditions_per_group and max_conditions_per_group > 0:
        by_group: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for pair, groups in cond_groups.items():
            for group in groups:
                by_group[group].append(pair)
        for group in sorted(by_group):
            vals = sorted(set(by_group[group]))
            rng = np.random.RandomState(seed + 17011 + sum(map(ord, group)))
            if len(vals) > max_conditions_per_group:
                keep = rng.permutation(len(vals))[:max_conditions_per_group]
                vals = [vals[int(i)] for i in sorted(keep)]
            for pair in vals:
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
    else:
        pairs = sorted(cond_groups)

    if max_conditions and max_conditions > 0 and len(pairs) > max_conditions:
        rng = np.random.RandomState(seed + 17021)
        keep = set(rng.permutation(len(pairs))[:max_conditions].tolist())
        pairs = [p for i, p in enumerate(pairs) if i in keep]

    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"train": [], "test": []})
    for ds_name, cond in pairs:
        out[ds_name]["test"].append(cond)
    return {
        ds_name: {"train": [], "test": sorted(set(parts["test"]))}
        for ds_name, parts in sorted(out.items())
        if parts["test"]
    }


def _metadata_row(
    *,
    condition_metadata: dict[str, dict[str, dict[str, Any]]],
    ds_name: str,
    cond: str,
) -> dict[str, Any]:
    entry = condition_metadata.get(ds_name, {}).get(cond, {})
    genes = _genes(entry)
    is_drug = _is_drug(entry, ds_name)
    return {
        "perturbation_type": _pert_type(entry),
        "perturbation_family": "drug" if is_drug else ("gene" if genes else "unknown"),
        "n_genes": int(len(genes)),
        "genes": "+".join(genes),
        "is_multi": bool(len(genes) > 1 or "+" in str(cond)),
    }


def _sample_rows(
    arr: np.ndarray,
    *,
    max_rows: int,
    rng: np.random.RandomState,
) -> torch.Tensor:
    n = int(arr.shape[0])
    if max_rows > 0:
        n = min(n, int(max_rows))
    idx = rng.permutation(int(arr.shape[0]))[:n]
    return torch.from_numpy(np.asarray(arr[idx], dtype=np.float32))


@torch.no_grad()
def _evaluate_condition_rows(
    *,
    model: torch.nn.Module,
    dataset: CrossDatasetFMDataset,
    cfg: Config,
    device: torch.device,
    cond_groups: dict[tuple[str, str], list[str]],
    condition_metadata: dict[str, dict[str, dict[str, Any]]],
    ctrl_means: dict[str, np.ndarray] | None,
    pert_means: dict[str, np.ndarray] | None,
    ode_steps: int,
    max_chunk: int,
    eval_max_cells: int,
    skip_mmd: bool,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    use_pe = _model_uses_pert(model)
    count = 0

    for ds_name in dataset.ds_names:
        handle = dataset.handles[ds_name]
        for cond in dataset.ds_conds[ds_name]:
            src_np = handle.read_src(cond)
            gt_np = handle.read_gt(cond)
            rng = np.random.RandomState(int(cfg.seed) + 18011 + count)
            src_eval = _sample_rows(src_np, max_rows=eval_max_cells, rng=rng)
            gt_eval = _sample_rows(gt_np, max_rows=eval_max_cells, rng=rng)

            pred_parts: list[torch.Tensor] = []
            pb_dev_full = None
            if use_pe:
                pb_cpu = _pert_for_eval_batch(dataset, ds_name, cond, int(src_eval.size(0)))
                pb_dev_full = _pert_to_device(pb_cpu, device)

            for start in range(0, int(src_eval.size(0)), int(max_chunk)):
                end = min(start + int(max_chunk), int(src_eval.size(0)))
                src_c = src_eval[start:end].to(device, non_blocking=True)
                pb_use = None if pb_dev_full is None else _pert_chunk(pb_dev_full, start, end)
                pred_parts.append(
                    ode_integrate(
                        model,
                        src_c,
                        src_c,
                        cfg,
                        n_steps=int(ode_steps),
                        perturbation_batch=pb_use if use_pe else None,
                    ).detach().cpu()
                )

            pred = torch.cat(pred_parts, dim=0) if pred_parts else torch.empty(0, int(cfg.emb_dim))
            pred_mean = pred.mean(dim=0).numpy()
            gt_mean = gt_eval.mean(dim=0).numpy()

            ref = None
            ref_name = "zero"
            if pert_means and ds_name in pert_means:
                ref = np.asarray(pert_means[ds_name], dtype=np.float32)
                ref_name = "pert_mean"
            elif ctrl_means and ds_name in ctrl_means:
                ref = np.asarray(ctrl_means[ds_name], dtype=np.float32)
                ref_name = "ctrl_mean"
            if ref is None:
                ref = np.zeros_like(gt_mean, dtype=np.float32)

            pred_resid = pred_mean - ref
            target_resid = gt_mean - ref

            mmd_val = None
            if not skip_mmd and pred.numel() and gt_eval.numel():
                pred_dev = pred.to(device)
                gt_dev = gt_eval.to(device)
                sigmas, dyy = median_sigmas(gt_dev.float(), return_D2=True)
                mmd_val = float(mmd2_unbiased(pred_dev.float(), gt_dev.float(), sigmas, Dyy=dyy).item())
                del pred_dev, gt_dev

            meta = _metadata_row(
                condition_metadata=condition_metadata,
                ds_name=ds_name,
                cond=cond,
            )
            rows.append(
                {
                    "dataset": ds_name,
                    "condition": cond,
                    "groups": ",".join(cond_groups.get((ds_name, cond), [])),
                    "n_src_eval": int(src_eval.size(0)),
                    "n_gt_eval": int(gt_eval.size(0)),
                    "reference": ref_name,
                    "pred_norm": _norm(pred_resid),
                    "target_norm": _norm(target_resid),
                    "pred_target_cosine": _cosine(pred_resid, target_resid),
                    "pred_target_pearson": _as_float(_pearson_np(pred_resid, target_resid)),
                    "mmd": _as_float(mmd_val),
                    "_pred_residual": pred_resid.astype(np.float32),
                    "_target_residual": target_resid.astype(np.float32),
                    **meta,
                }
            )
            count += 1

    return rows


def _attach_retrieval(rows: list[dict[str, Any]], top_ks: Sequence[int]) -> None:
    if not rows:
        return
    targets = np.stack([np.asarray(r["_target_residual"], dtype=np.float32) for r in rows], axis=0)
    for i, row in enumerate(rows):
        row.update(
            _residual_retrieval_metrics(
                np.asarray(row["_pred_residual"], dtype=np.float32),
                targets,
                true_index=i,
                top_ks=top_ks,
            )
        )


def _clean_public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        out.append(clean)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, default=None, help="Default: <biflow_dir>/split_seed<seed>.json")
    ap.add_argument("--data-dir", type=str, default="", help="Override checkpoint config data_dir")
    ap.add_argument("--biflow-dir", type=str, default="", help="Override checkpoint config biflow_dir")
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--max-chunk", type=int, default=512)
    ap.add_argument("--eval-max-cells", type=int, default=2048)
    ap.add_argument("--max-conditions", type=int, default=None)
    ap.add_argument("--max-conditions-per-group", type=int, default=None)
    ap.add_argument("--top-k", nargs="*", type=int, default=[1, 5, 10])
    ap.add_argument("--skip-mmd", action="store_true")
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    ckpt_path = args.checkpoint.expanduser().resolve()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"checkpoint must contain a 'model' key: {ckpt_path}")

    cfg = _load_cfg(ckpt, data_dir=args.data_dir, biflow_dir=args.biflow_dir)
    cfg.gpu = int(args.gpu)
    data_dir = Path(cfg.data_dir).expanduser().resolve()
    split_path = args.split_file
    if split_path is None:
        split_path = Path(cfg.biflow_dir).expanduser().resolve() / f"split_seed{cfg.split_seed}.json"
    else:
        split_path = split_path.expanduser().resolve()

    device_s = args.device.strip() or (f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    manifest = _load_manifest(data_dir, cfg.manifest)
    split = _load_split(split_path)
    condition_metadata = _load_condition_metadata(data_dir)
    family_splits = build_family_group_splits(
        manifest=manifest,
        split=split,
        condition_metadata=condition_metadata,
    )
    split_splits = _split_group_splits(manifest=manifest, split=split, groups=args.groups)
    all_group_splits = {**family_splits, **split_splits}
    wanted = {g: all_group_splits[g] for g in args.groups if g in all_group_splits}
    cond_groups = _condition_group_index(wanted)
    selected = _selected_split(
        cond_groups,
        max_conditions=args.max_conditions,
        max_conditions_per_group=args.max_conditions_per_group,
        seed=int(cfg.seed),
    )
    if not selected:
        raise ValueError("no conditions selected from requested groups")

    ctrl_means = _load_means(data_dir, "ctrl_means.npz")
    pert_means = _load_means(data_dir, "pert_means.npz")

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"], strict=True)

    ema = None
    if not args.no_ema and "ema" in ckpt and bool(getattr(cfg, "use_ema", False)):
        ema = ModelEMA(
            model,
            decay=float(getattr(cfg, "ema_decay", 0.999)),
            update_after=int(getattr(cfg, "ema_update_after", 0)),
            update_every=int(getattr(cfg, "ema_update_every", 1)),
            device=device,
        )
        ema.load_state_dict(ckpt["ema"], strict=False)

    ds = CrossDatasetFMDataset(
        str(data_dir),
        selected,
        cfg.batch_size,
        cfg.seed,
        mode="test",
        min_cells=16,
        ds_alpha=1.0,
        silent=False,
        **_cross_dataset_kw(cfg),
    )

    eval_kwargs = dict(
        model=model,
        dataset=ds,
        cfg=cfg,
        device=device,
        cond_groups=cond_groups,
        condition_metadata=condition_metadata,
        ctrl_means=ctrl_means,
        pert_means=pert_means,
        ode_steps=int(args.ode_steps),
        max_chunk=int(args.max_chunk),
        eval_max_cells=int(args.eval_max_cells),
        skip_mmd=bool(args.skip_mmd),
    )
    if ema is not None:
        with ema.apply_to(model):
            rows = _evaluate_condition_rows(**eval_kwargs)
    else:
        rows = _evaluate_condition_rows(**eval_kwargs)

    _attach_retrieval(rows, top_ks=tuple(args.top_k))
    public_rows = _clean_public_rows(rows)

    out_csv = args.out_csv or (ckpt_path.parent / "condition_residual_eval.csv")
    out_json = args.out_json or (ckpt_path.parent / "condition_residual_eval.json")
    _write_csv(out_csv, public_rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "groups": list(args.groups),
        "selected_conditions": int(len(public_rows)),
        "used_ema": ema is not None,
        "config": dataclasses.asdict(cfg),
        "rows": public_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {"out_csv": str(out_csv), "out_json": str(out_json), "rows": len(public_rows)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
