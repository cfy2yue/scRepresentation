#!/usr/bin/env python3
"""Evaluate condition-delta head combo/additive/interaction decomposition.

This is a lightweight diagnostic for checkpoints with ``condition_delta_head``.
It does not run ODE integration and does not compute MMD.  It asks whether the
condition head's direct combo prediction, additive atom prediction, and their
combo-minus-additive residual align with observed condition-level response
deltas.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from model.latent.dataset import CrossDatasetFMDataset
from model.latent.eval_condition_families import (
    _genes,
    _is_drug,
    _load_condition_metadata,
    _pert_type,
    build_family_group_splits,
)
from model.latent.eval_condition_residuals import (
    DEFAULT_GROUPS,
    _condition_group_index,
    _metadata_row,
    _selected_split,
    _split_group_splits,
)
from model.latent.eval_split_groups import (
    _json_default,
    _load_cfg,
    _load_manifest,
    _load_means,
    _load_split,
)
from model.latent.train import (
    _cross_dataset_kw,
    _pearson_np,
    _pert_for_eval_batch,
    _pert_to_device,
    _unpack_pert_up_to7,
    build_model,
)
from model.utils.train.ema import ModelEMA


def _as_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float32).reshape(-1)))


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= eps:
        return 0.0
    return float(np.dot(aa, bb) / (denom + eps))


def _mean_rows(arr: np.ndarray, *, max_cells: int, seed: int) -> np.ndarray:
    n = int(arr.shape[0])
    if max_cells > 0 and n > max_cells:
        rng = np.random.RandomState(int(seed))
        idx = np.sort(rng.choice(n, size=int(max_cells), replace=False))
        arr = arr[idx]
    return np.asarray(arr, dtype=np.float32).mean(axis=0)


def _public_row_metrics(prefix: str, pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    return {
        f"{prefix}_norm": _norm(pred),
        f"{prefix}_cosine": _cosine(pred, target),
        f"{prefix}_pearson": _as_float(_pearson_np(pred, target)),
    }


def _predict_decomposition(
    model: torch.nn.Module,
    perturbation_batch: tuple,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inner = model.module if hasattr(model, "module") else model
    if getattr(inner, "condition_delta_head", None) is None:
        raise RuntimeError("checkpoint model does not have condition_delta_head enabled")
    pb = _pert_to_device(perturbation_batch, device)
    gid, mk, tid, npt, cid, ce, cm = _unpack_pert_up_to7(pb)
    combo = inner.predict_condition_delta(
        pert_gene_ids=gid,
        pert_mask=mk,
        pert_type_id=tid,
        nperts=npt,
        combo_id=cid,
        chem_emb=ce,
        chem_mask=cm,
    )
    additive = inner.predict_additive_condition_delta(
        pert_gene_ids=gid,
        pert_mask=mk,
        pert_type_id=tid,
        nperts=npt,
        combo_id=cid,
        chem_emb=ce,
        chem_mask=cm,
    )
    interaction = inner.predict_interaction_condition_delta(
        pert_gene_ids=gid,
        pert_mask=mk,
        pert_type_id=tid,
        nperts=npt,
        combo_id=cid,
        chem_emb=ce,
        chem_mask=cm,
    )
    return (
        combo.float().mean(dim=0).detach().cpu().numpy(),
        additive.float().mean(dim=0).detach().cpu().numpy(),
        interaction.float().mean(dim=0).detach().cpu().numpy(),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for group in str(row.get("groups", "")).split(","):
            if group:
                grouped[(str(row["dataset"]), group)].append(row)
    out: list[dict[str, Any]] = []
    for (dataset, group), vals in sorted(grouped.items()):
        out.append({
            "dataset": dataset,
            "group": group,
            "n": len(vals),
            "mean_combo_endpoint_cosine": float(np.mean([v["combo_endpoint_cosine"] for v in vals])),
            "mean_additive_endpoint_cosine": float(np.mean([v["additive_endpoint_cosine"] for v in vals])),
            "mean_interaction_endpoint_cosine": float(np.mean([v["interaction_endpoint_cosine"] for v in vals])),
            "mean_combo_pert_residual_cosine": float(np.mean([v["combo_pert_residual_cosine"] for v in vals])),
            "mean_additive_pert_residual_cosine": float(np.mean([v["additive_pert_residual_cosine"] for v in vals])),
            "mean_interaction_pert_residual_cosine": float(np.mean([v["interaction_pert_residual_cosine"] for v in vals])),
            "mean_combo_additive_cosine": float(np.mean([v["combo_additive_cosine"] for v in vals])),
            "mean_additive_norm_ratio": float(np.mean([v["additive_norm_ratio"] for v in vals])),
            "mean_interaction_norm_ratio": float(np.mean([v["interaction_norm_ratio"] for v in vals])),
        })
    return out


@torch.no_grad()
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
    ap.add_argument("--eval-max-cells", type=int, default=512)
    ap.add_argument("--max-conditions", type=int, default=None)
    ap.add_argument("--max-conditions-per-group", type=int, default=None)
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

    ctrl_means = _load_means(data_dir, "ctrl_means.npz") or {}
    pert_means = _load_means(data_dir, "pert_means.npz") or {}

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
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

    dataset = CrossDatasetFMDataset(
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

    rows: list[dict[str, Any]] = []
    ctx = ema.apply_to(model) if ema is not None else torch.no_grad()
    with ctx:
        for ds_name in dataset.ds_names:
            handle = dataset.handles[ds_name]
            for cond in dataset.ds_conds[ds_name]:
                src = handle.read_src(cond)
                gt = handle.read_gt(cond)
                seed = int(cfg.seed) + sum(ord(c) for c in f"{ds_name}\t{cond}")
                src_mean = _mean_rows(src, max_cells=int(args.eval_max_cells), seed=seed)
                gt_mean = _mean_rows(gt, max_cells=int(args.eval_max_cells), seed=seed + 17)
                endpoint_target = gt_mean - src_mean
                pert_ref = np.asarray(pert_means.get(ds_name, np.zeros_like(gt_mean)), dtype=np.float32)
                pert_residual_target = gt_mean - pert_ref

                pb = _pert_for_eval_batch(dataset, ds_name, cond, 1)
                combo, additive, interaction = _predict_decomposition(model, pb, device)
                meta = _metadata_row(
                    condition_metadata=condition_metadata,
                    ds_name=ds_name,
                    cond=cond,
                )
                combo_norm = _norm(combo)
                additive_norm = _norm(additive)
                row = {
                    "dataset": ds_name,
                    "condition": cond,
                    "groups": ",".join(cond_groups.get((ds_name, cond), [])),
                    "n_src_total": int(src.shape[0]),
                    "n_gt_total": int(gt.shape[0]),
                    "n_src_eval": min(int(src.shape[0]), int(args.eval_max_cells)),
                    "n_gt_eval": min(int(gt.shape[0]), int(args.eval_max_cells)),
                    **meta,
                    "endpoint_target_norm": _norm(endpoint_target),
                    "pert_residual_target_norm": _norm(pert_residual_target),
                    **_public_row_metrics("combo_endpoint", combo, endpoint_target),
                    **_public_row_metrics("additive_endpoint", additive, endpoint_target),
                    **_public_row_metrics("interaction_endpoint", interaction, endpoint_target),
                    **_public_row_metrics("combo_pert_residual", combo, pert_residual_target),
                    **_public_row_metrics("additive_pert_residual", additive, pert_residual_target),
                    **_public_row_metrics("interaction_pert_residual", interaction, pert_residual_target),
                    "combo_additive_cosine": _cosine(combo, additive),
                    "additive_norm_ratio": additive_norm / max(combo_norm, 1e-12),
                    "interaction_norm_ratio": _norm(interaction) / max(combo_norm, 1e-12),
                }
                rows.append(row)

    summary = _summarize(rows)
    out_csv = args.out_csv or (ckpt_path.parent / "condition_delta_decomposition.csv")
    out_json = args.out_json or (ckpt_path.parent / "condition_delta_decomposition.json")
    _write_csv(out_csv, rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "groups": list(args.groups),
        "selected_conditions": len(rows),
        "used_ema": ema is not None,
        "device": str(device),
        "summary": summary,
        "rows": rows,
        "config": dataclasses.asdict(cfg),
    }
    out_json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out_csv": str(out_csv), "out_json": str(out_json), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
