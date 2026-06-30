#!/usr/bin/env python3
"""Evaluate a LatentFM checkpoint on canonical split subgroups.

The training loop reports the canonical ``test`` group as one IID aggregate.
For multi-perturbation claims we also need explicit subgroup metrics:

* ``test_single``
* ``test_multi``
* ``test_multi_seen``
* ``test_multi_unseen1``
* ``test_multi_unseen2``

This CLI reuses ``model.latent.train.evaluate`` so metric definitions stay
identical to training-time evaluation.  It only changes which split key is
presented as ``mode="test"`` to ``CrossDatasetFMDataset``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import h5py

from model.latent.config import Config
from model.latent.dataset import CrossDatasetFMDataset
from model.latent.fm_ot import CondOTPath
from model.latent.train import (
    _cross_dataset_kw,
    build_model,
    checkpoint_ema_is_active,
    evaluate,
    load_model_weights_only,
)
from model.utils.train.ema import ModelEMA


DEFAULT_GROUPS = (
    "test",
    "test_single",
    "test_multi",
    "test_multi_seen",
    "test_multi_unseen1",
    "test_multi_unseen2",
)


def _load_cfg(ckpt: dict[str, Any], *, data_dir: str = "", biflow_dir: str = "") -> Config:
    cfg = Config()
    raw = ckpt.get("config", {})
    if isinstance(raw, dict):
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if data_dir:
        cfg.data_dir = data_dir
    if biflow_dir:
        cfg.biflow_dir = biflow_dir
    drug_cache_override = (
        os.environ.get("LATENT_DRUG_EMB_CACHE_DIR", "")
        or os.environ.get("RAW_DRUG_EMB_CACHE_DIR", "")
    ).strip()
    if drug_cache_override:
        cfg.drug_emb_cache_dir = drug_cache_override
    return cfg


def _load_manifest(data_dir: Path, manifest_name: str) -> dict[str, Any]:
    with open(data_dir / manifest_name) as f:
        manifest = json.load(f)
    datasets = manifest.get("datasets")
    if isinstance(datasets, dict):
        for ds_name, ds_meta in datasets.items():
            if not isinstance(ds_meta, dict) or ds_meta.get("conditions"):
                continue
            h5_path = data_dir / f"{ds_name}.h5"
            if not h5_path.is_file():
                continue
            with h5py.File(str(h5_path), "r") as handle:
                if "conditions" not in handle:
                    continue
                ds_meta["conditions"] = [
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in handle["conditions"][()]
                ]
    return manifest


def _load_split(path: Path) -> dict[str, dict[str, list[str]]]:
    with open(path) as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"split must be a dataset -> groups mapping: {path}")
    return obj


def _group_as_test_split(
    *,
    split: dict[str, dict[str, list[str]]],
    manifest: dict[str, Any],
    group: str,
) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for ds_name, ds_meta in manifest.get("datasets", {}).items():
        allowed = set(map(str, ds_meta.get("conditions", [])))
        conds = [str(c) for c in split.get(ds_name, {}).get(group, []) if str(c) in allowed]
        if conds:
            out[str(ds_name)] = {"train": [], "test": conds}
    return out


def _load_means(data_dir: Path, name: str) -> dict[str, np.ndarray] | None:
    path = data_dir / name
    if not path.is_file():
        return None
    return {k: v for k, v in np.load(str(path)).items()}


def _resolve_means_file(path_s: str, *, data_dir: Path, default_name: str) -> Path:
    if path_s.strip():
        path = Path(path_s).expanduser()
        return path if path.is_absolute() else path.resolve()
    return data_dir / default_name


def _load_means_file(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    return {k: v for k, v in np.load(str(path)).items()}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="best.pt or latest.pt")
    ap.add_argument("--split-file", type=Path, default=None, help="Default: <biflow_dir>/split_seed<seed>.json")
    ap.add_argument("--data-dir", type=str, default="", help="Override checkpoint config data_dir")
    ap.add_argument("--biflow-dir", type=str, default="", help="Override checkpoint config biflow_dir")
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", type=str, default="", help="Override device, e.g. cuda:0 or cpu")
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--max-chunk", type=int, default=512)
    ap.add_argument("--eval-max-conditions", type=int, default=None)
    ap.add_argument("--eval-max-conditions-per-dataset", type=int, default=None)
    ap.add_argument("--eval-max-mse-cells", type=int, default=None)
    ap.add_argument("--eval-max-mmd-cells", type=int, default=None)
    ap.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "Override cfg.seed for deterministic eval cell sub-sampling. "
            "Use this when comparing checkpoints whose training seeds differ."
        ),
    )
    ap.add_argument(
        "--save-condition-means",
        action="store_true",
        help="Default-off: include per-condition pred/gt/ctrl/pert means for posthoc residual audits.",
    )
    ap.add_argument(
        "--force-support-context-absent",
        action="store_true",
        help=(
            "Evaluate support-context checkpoints with support context absent. "
            "Use this for canonical Track A no-op diagnostics."
        ),
    )
    ap.add_argument(
        "--support-context-control",
        choices=("actual", "zero", "shuffle_condition"),
        default="actual",
        help="Eval-only Track C support-context control. Default keeps checkpoint behavior.",
    )
    ap.add_argument(
        "--support-set-task-control",
        choices=("actual", "zero", "shuffle_condition", "absent"),
        default="actual",
        help="Eval-only Track C support-set task control. Default keeps checkpoint behavior.",
    )
    ap.add_argument("--no-ema", action="store_true", help="Use raw model weights even if EMA is present")
    ap.add_argument(
        "--pert-means-file",
        type=str,
        default="",
        help="Override dataset perturbed mean reference for pearson_pert; default: <data-dir>/pert_means.npz",
    )
    args = ap.parse_args()

    ckpt_path = args.checkpoint.expanduser().resolve()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"checkpoint must contain a 'model' key: {ckpt_path}")

    cfg = _load_cfg(ckpt, data_dir=args.data_dir, biflow_dir=args.biflow_dir)
    cfg.gpu = int(args.gpu)
    if args.eval_seed is not None:
        cfg.seed = int(args.eval_seed)
    for attr, val in (
        ("eval_max_conditions", args.eval_max_conditions),
        ("eval_max_conditions_per_dataset", args.eval_max_conditions_per_dataset),
        ("eval_max_mse_cells", args.eval_max_mse_cells),
        ("eval_max_mmd_cells", args.eval_max_mmd_cells),
    ):
        if val is not None:
            setattr(cfg, attr, int(val))
    if args.save_condition_means:
        cfg.eval_save_condition_means = True
    if args.force_support_context_absent:
        cfg.trackc_support_context_source = "off"
    cfg.trackc_support_context_eval_control = str(args.support_context_control)
    cfg.trackc_support_set_task_eval_control = str(args.support_set_task_control)

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
    ctrl_means_path = data_dir / "ctrl_means.npz"
    pert_means_path = _resolve_means_file(
        str(args.pert_means_file or ""),
        data_dir=data_dir,
        default_name="pert_means.npz",
    )
    ctrl_means = _load_means_file(ctrl_means_path)
    pert_means = _load_means_file(pert_means_path)

    model = build_model(cfg, device)
    missing_keys, unexpected_keys, skipped_shape_mismatch = load_model_weights_only(
        ckpt_path, model, device, strict=False
    )

    ema = None
    if not args.no_ema and checkpoint_ema_is_active(ckpt, cfg):
        ema = ModelEMA(
            model,
            decay=float(getattr(cfg, "ema_decay", 0.999)),
            update_after=int(getattr(cfg, "ema_update_after", 0)),
            update_every=int(getattr(cfg, "ema_update_every", 1)),
            device=device,
        )
        ema.load_state_dict(ckpt["ema"], strict=False)
    elif not args.no_ema and "ema" in ckpt and bool(getattr(cfg, "use_ema", False)):
        print(
            "[eval_split_groups] EMA state present but inactive for checkpoint "
            f"step={ckpt.get('step')} < ema_update_after={getattr(cfg, 'ema_update_after', 0)}; using raw weights"
        )

    path = CondOTPath()
    cd_kw = _cross_dataset_kw(cfg)
    results: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "groups": {},
        "config": dataclasses.asdict(cfg),
        "means_files": {
            "ctrl_means": str(ctrl_means_path),
            "pert_means": str(pert_means_path),
            "pert_means_override": bool(str(args.pert_means_file or "").strip()),
            "ctrl_means_loaded": ctrl_means is not None,
            "pert_means_loaded": pert_means is not None,
        },
        "support_context_forced_absent": bool(args.force_support_context_absent),
        "support_context_control": str(args.support_context_control),
        "support_set_task_control": str(args.support_set_task_control),
        "eval_seed_override": None if args.eval_seed is None else int(args.eval_seed),
        "used_ema": ema is not None,
        "load_state": {
            "strict": False,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "skipped_shape_mismatch": skipped_shape_mismatch,
        },
    }

    empty_eval_caps = {
        "max_conditions": cfg.eval_max_conditions,
        "max_conditions_per_dataset": cfg.eval_max_conditions_per_dataset,
        "max_mse_cells": cfg.eval_max_mse_cells,
        "max_mmd_cells": cfg.eval_max_mmd_cells,
        "aggregation": "condition_mean_then_dataset_equal_mean",
    }

    for group in args.groups:
        group_split = _group_as_test_split(split=split, manifest=manifest, group=group)
        n_requested = sum(len(v["test"]) for v in group_split.values())
        if n_requested == 0:
            results["groups"][group] = {
                "skipped": True,
                "reason": "no conditions in split group after manifest filtering",
                "n_requested": 0,
                "n_available_conditions": 0,
                "n_conds": 0,
                "eval_caps": empty_eval_caps,
                "selected_conditions": [],
                "condition_metrics": [],
                "per_ds_mmd": {},
                "per_ds_mmd_biased": {},
                "per_ds_mmd_clamped": {},
                "per_ds_direct": {},
                "per_ds_p_ctrl": {},
                "per_ds_p_pert": {},
            }
            continue

        ds = CrossDatasetFMDataset(
            str(data_dir),
            group_split,
            cfg.batch_size,
            cfg.seed,
            mode="test",
            min_cells=16,
            ds_alpha=1.0,
            silent=False,
            **cd_kw,
        )
        if ema is not None:
            with ema.apply_to(model):
                res = evaluate(
                    model, ds, path, cfg, device,
                    ctrl_means=ctrl_means, pert_means=pert_means,
                    ode_steps=int(args.ode_steps),
                    max_chunk=int(args.max_chunk),
                )
        else:
            res = evaluate(
                model, ds, path, cfg, device,
                ctrl_means=ctrl_means, pert_means=pert_means,
                ode_steps=int(args.ode_steps),
                max_chunk=int(args.max_chunk),
            )
        res["skipped"] = False
        res["n_requested"] = int(n_requested)
        results["groups"][group] = res

    out_path = args.out or (ckpt_path.parent / "split_group_eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "groups": list(results["groups"].keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
