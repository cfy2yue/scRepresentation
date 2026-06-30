#!/usr/bin/env python3
"""Audit whether a LatentFM checkpoint is sensitive to the correct condition.

This is a train-only/internal-val gate for Track A single/background work. It
compares predictions from the same frozen checkpoint under:

* true condition metadata;
* shuffled condition metadata;
* zero/no condition metadata.

It is intentionally capped by default and writes a report rather than launching
training. Use it to decide whether a tiny condition-path adapter is justified.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.condition_emb.genepert.perturbation import ConditionMetadata, PerturbationBatch
from model.latent.dataset import CrossDatasetFMDataset
from model.latent.eval_split_groups import _load_cfg, _load_manifest, _load_means, _load_split
from model.latent.fm_ot import CondOTPath
from model.latent.train import (
    _cross_dataset_kw,
    _pearson_np,
    _pert_for_eval_batch,
    _pert_to_device,
    build_model,
    checkpoint_ema_is_active,
    load_model_weights_only,
    ode_integrate,
)
from model.utils.train.ema import ModelEMA


DEFAULT_GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def group_as_test_split(
    *,
    split: dict[str, dict[str, list[str]]],
    manifest: dict[str, Any],
    group: str,
) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for ds_name, ds_meta in manifest.get("datasets", {}).items():
        allowed = set(map(str, ds_meta.get("conditions", [])))
        conds = [str(c) for c in split.get(str(ds_name), {}).get(group, []) if str(c) in allowed]
        if conds:
            out[str(ds_name)] = {"train": [], "test": sorted(set(conds))}
    return out


def limit_group_split(
    group_split: dict[str, dict[str, list[str]]],
    *,
    max_conditions_per_dataset: int,
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    if max_conditions_per_dataset <= 0:
        return group_split
    out: dict[str, dict[str, list[str]]] = {}
    for ds_name, parts in sorted(group_split.items()):
        conds = list(map(str, parts.get("test", [])))
        if len(conds) > max_conditions_per_dataset:
            rng = np.random.RandomState(stable_int(f"{seed}\t{ds_name}") % (2**32 - 1))
            keep = sorted(rng.choice(conds, size=max_conditions_per_dataset, replace=False).tolist())
        else:
            keep = sorted(conds)
        if keep:
            out[ds_name] = {"train": [], "test": keep}
    return out


def make_shuffle_map(group_split: dict[str, dict[str, list[str]]], *, seed: int) -> dict[tuple[str, str], tuple[str, str]]:
    pairs = [(ds, cond) for ds, parts in sorted(group_split.items()) for cond in sorted(parts.get("test", []))]
    if not pairs:
        return {}
    out: dict[tuple[str, str], tuple[str, str]] = {}
    by_ds: dict[str, list[str]] = defaultdict(list)
    for ds, cond in pairs:
        by_ds[ds].append(cond)
    for ds, conds in by_ds.items():
        conds = sorted(set(conds))
        if len(conds) > 1:
            shift = stable_int(f"{seed}\t{ds}\tshuffle") % (len(conds) - 1) + 1
            for i, cond in enumerate(conds):
                out[(ds, cond)] = (ds, conds[(i + shift) % len(conds)])
        else:
            idx = pairs.index((ds, conds[0]))
            out[(ds, conds[0])] = pairs[(idx + 1) % len(pairs)]
    return out


def zero_pert_batch(dataset: CrossDatasetFMDataset, batch_size: int) -> tuple:
    cache = dataset.gene_embedding_cache
    if cache is None:
        raise RuntimeError("zero perturbation batch requires dataset gene_embedding_cache")
    rows = [ConditionMetadata(genes=(), perturbation_type_raw=None)] * int(batch_size)
    pb = PerturbationBatch.from_metadata_list(
        rows,
        cache,
        max_genes=int(dataset.max_pert_genes),
        max_chem_slots=int(getattr(dataset, "max_chem_keys", 4)),
        device=torch.device("cpu"),
    )
    return pb.as_tuple_full()


def perturb_batch_for_mode(
    dataset: CrossDatasetFMDataset,
    ds_name: str,
    cond: str,
    batch_size: int,
    *,
    mode: str,
    shuffle_map: dict[tuple[str, str], tuple[str, str]],
) -> tuple:
    if mode == "true":
        return _pert_for_eval_batch(dataset, ds_name, cond, batch_size)
    if mode == "zero":
        return zero_pert_batch(dataset, batch_size)
    if mode == "shuffle":
        sds, scond = shuffle_map[(ds_name, cond)]
        return _pert_for_eval_batch(dataset, sds, scond, batch_size)
    raise ValueError(f"unknown mode: {mode}")


@torch.no_grad()
def eval_group_modes(
    *,
    model: torch.nn.Module,
    dataset: CrossDatasetFMDataset,
    path: CondOTPath,
    cfg: Any,
    device: torch.device,
    ctrl_means: dict[str, np.ndarray] | None,
    pert_means: dict[str, np.ndarray] | None,
    group_split: dict[str, dict[str, list[str]]],
    modes: tuple[str, ...],
    max_cells: int,
    max_chunk: int,
    ode_steps: int,
) -> dict[str, Any]:
    model.eval()
    shuffle_map = make_shuffle_map(group_split, seed=int(getattr(cfg, "seed", 0) or 0))
    rows = []
    per_ds: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in modes}
    for ds_name, parts in sorted(group_split.items()):
        handle = dataset.handles[ds_name]
        for cond in sorted(parts.get("test", [])):
            src_full = torch.from_numpy(handle.read_src(cond)).float()
            gt_full = torch.from_numpy(handle.read_gt(cond)).float()
            seed = stable_int(f"{getattr(cfg, 'seed', 0)}\t{ds_name}\t{cond}\tcondition_path") % (2**32 - 1)
            rng = np.random.RandomState(seed)
            n_src = min(src_full.size(0), max_cells)
            n_gt = min(gt_full.size(0), max_cells)
            src_eval = src_full[rng.permutation(src_full.size(0))[:n_src]]
            gt_eval = gt_full[rng.permutation(gt_full.size(0))[:n_gt]]
            gt_mean = gt_eval.mean(dim=0).cpu().numpy()
            ctrl_mean = None if ctrl_means is None else ctrl_means.get(ds_name)
            pert_mean = None if pert_means is None else pert_means.get(ds_name)

            for mode in modes:
                pred_parts = []
                pb_cpu = perturb_batch_for_mode(
                    dataset,
                    ds_name,
                    cond,
                    int(n_src),
                    mode=mode,
                    shuffle_map=shuffle_map,
                )
                pb_dev_full = _pert_to_device(pb_cpu, device)
                for st in range(0, int(n_src), max_chunk):
                    en = min(st + max_chunk, int(n_src))
                    src = src_eval[st:en].to(device)
                    pb_use = tuple(
                        None if x is None else x[st:en]
                        for x in pb_dev_full
                    )
                    pred = ode_integrate(
                        model,
                        src,
                        src,
                        cfg,
                        n_steps=ode_steps,
                        perturbation_batch=pb_use,
                    )
                    pred_parts.append(pred.cpu())
                pred_mean = torch.cat(pred_parts, dim=0).mean(dim=0).numpy()
                direct = _pearson_np(pred_mean, gt_mean)
                pc = None if ctrl_mean is None else _pearson_np(pred_mean - ctrl_mean, gt_mean - ctrl_mean)
                pp = None if pert_mean is None else _pearson_np(pred_mean - pert_mean, gt_mean - pert_mean)
                if pp is not None:
                    per_ds[mode][ds_name].append(float(pp))
                rows.append(
                    {
                        "dataset": ds_name,
                        "condition": cond,
                        "mode": mode,
                        "direct_pearson": float(direct),
                        "pearson_ctrl": None if pc is None else float(pc),
                        "pearson_pert": None if pp is None else float(pp),
                        "n_src_eval": int(n_src),
                        "n_gt_eval": int(n_gt),
                        "shuffle_condition": None if mode != "shuffle" else list(shuffle_map[(ds_name, cond)]),
                    }
                )
    mode_summary = {}
    for mode in modes:
        ds_means = {ds: float(np.mean(vals)) for ds, vals in sorted(per_ds[mode].items()) if vals}
        mode_summary[mode] = {
            "pearson_pert_dataset_equal": float(np.mean(list(ds_means.values()))) if ds_means else None,
            "per_dataset_pearson_pert": ds_means,
        }
    return {
        "mode_summary": mode_summary,
        "condition_rows": rows,
    }


def paired_mode_delta(mode_summary: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    ad = mode_summary.get(a, {}).get("per_dataset_pearson_pert", {}) or {}
    bd = mode_summary.get(b, {}).get("per_dataset_pearson_pert", {}) or {}
    common = sorted(set(ad) & set(bd))
    vals = [float(ad[d]) - float(bd[d]) for d in common]
    return {
        "comparison": f"{a}_minus_{b}",
        "n_datasets": len(common),
        "delta_mean": None if not vals else float(np.mean(vals)),
        "per_dataset_delta": {d: float(ad[d]) - float(bd[d]) for d in common},
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Condition-Path Sensitivity Audit",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Provenance",
        "",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- data_dir: `{payload['data_dir']}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- groups: `{payload['groups']}`",
        f"- modes: `{payload['modes']}`",
        f"- max_conditions_per_dataset: `{payload['max_conditions_per_dataset']}`",
        f"- max_cells: `{payload['max_cells']}`",
        "- leakage note: intended for train-only/internal-val groups; do not use canonical test reference for CPU gating.",
        "",
        "## Group Summaries",
        "",
    ]
    for group, res in payload["results"].items():
        lines += [
            f"### {group}",
            "",
            "| mode | dataset-equal pp |",
            "|---|---:|",
        ]
        for mode, summ in res["mode_summary"].items():
            val = summ.get("pearson_pert_dataset_equal")
            lines.append(f"| `{mode}` | {'NA' if val is None else f'{val:+.6f}'} |")
        lines += ["", "| comparison | n datasets | delta |", "|---|---:|---:|"]
        for delta in res["deltas"]:
            val = delta.get("delta_mean")
            lines.append(f"| `{delta['comparison']}` | {delta['n_datasets']} | {'NA' if val is None else f'{val:+.6f}'} |")
        lines.append("")
    lines += [
        "## Gate Interpretation",
        "",
        "- This script only checks whether the frozen model is sensitive to correct condition metadata.",
        "- A GPU adapter smoke is justified only if true condition beats shuffled and zero condition on the train-only proxy groups.",
        "- If true condition is not better, do not train a condition-path adapter; inspect data/model conditioning first.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, required=True)
    ap.add_argument("--data-dir", type=str, default="")
    ap.add_argument("--biflow-dir", type=str, default="")
    ap.add_argument("--pert-means-file", type=Path, required=True)
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--modes", nargs="*", default=["true", "shuffle", "zero"])
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--device", default="")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--max-chunk", type=int, default=256)
    ap.add_argument("--max-cells", type=int, default=256)
    ap.add_argument("--max-conditions-per-dataset", type=int, default=4)
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    ckpt_path = args.checkpoint.expanduser().resolve()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = _load_cfg(ckpt, data_dir=args.data_dir, biflow_dir=args.biflow_dir)
    cfg.gpu = int(args.gpu)
    cfg.eval_max_conditions = 0
    cfg.eval_max_conditions_per_dataset = 0
    cfg.eval_max_mse_cells = int(args.max_cells)
    cfg.eval_max_mmd_cells = int(args.max_cells)

    data_dir = Path(cfg.data_dir).expanduser().resolve()
    split_path = args.split_file.expanduser().resolve()
    device_s = args.device.strip() or (f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    manifest = _load_manifest(data_dir, cfg.manifest)
    split = _load_split(split_path)
    ctrl_means = _load_means(data_dir, "ctrl_means.npz")
    pert_means = {k: v for k, v in np.load(str(args.pert_means_file.expanduser().resolve())).items()}

    model = build_model(cfg, device)
    load_model_weights_only(ckpt_path, model, device, strict=False)
    if not args.no_ema and checkpoint_ema_is_active(ckpt, cfg):
        ema = ModelEMA(
            model,
            decay=float(getattr(cfg, "ema_decay", 0.999)),
            update_after=int(getattr(cfg, "ema_update_after", 0)),
            update_every=int(getattr(cfg, "ema_update_every", 1)),
            device=device,
        )
        ema.load_state_dict(ckpt["ema"], strict=False)
        ema.copy_to(model)

    path = CondOTPath()
    cd_kw = _cross_dataset_kw(cfg)
    results = {}
    for group in args.groups:
        group_split = group_as_test_split(split=split, manifest=manifest, group=group)
        group_split = limit_group_split(
            group_split,
            max_conditions_per_dataset=int(args.max_conditions_per_dataset),
            seed=int(getattr(cfg, "seed", 0) or 0),
        )
        if not group_split:
            results[group] = {"skipped": True, "reason": "no group conditions"}
            continue
        ds = CrossDatasetFMDataset(
            str(data_dir),
            group_split,
            cfg.batch_size,
            cfg.seed,
            mode="test",
            min_cells=16,
            ds_alpha=1.0,
            silent=True,
            **cd_kw,
        )
        res = eval_group_modes(
            model=model,
            dataset=ds,
            path=path,
            cfg=cfg,
            device=device,
            ctrl_means=ctrl_means,
            pert_means=pert_means,
            group_split=group_split,
            modes=tuple(args.modes),
            max_cells=int(args.max_cells),
            max_chunk=int(args.max_chunk),
            ode_steps=int(args.ode_steps),
        )
        res["deltas"] = [
            paired_mode_delta(res["mode_summary"], "true", "shuffle"),
            paired_mode_delta(res["mode_summary"], "true", "zero"),
        ]
        res["skipped"] = False
        results[group] = res

    status = "condition_path_sensitivity_audit_complete"
    payload = {
        "status": status,
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "pert_means_file": str(args.pert_means_file.expanduser().resolve()),
        "groups": list(args.groups),
        "modes": list(args.modes),
        "max_conditions_per_dataset": int(args.max_conditions_per_dataset),
        "max_cells": int(args.max_cells),
        "ode_steps": int(args.ode_steps),
        "config": dataclasses.asdict(cfg),
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "status": status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
