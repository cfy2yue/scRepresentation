#!/usr/bin/env python3
"""Evaluate a LatentFM checkpoint by perturbation family/type groups.

This CLI reuses ``model.latent.train.evaluate`` so metric definitions stay
identical to training-time evaluation. It only builds alternative test splits
from ``condition_metadata.json`` and the canonical split file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch

from model.latent.dataset import CrossDatasetFMDataset
from model.latent.eval_split_groups import (
    _json_default,
    _load_cfg,
    _load_manifest,
    _load_means,
    _load_means_file,
    _load_split,
    _resolve_means_file,
)
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
    "test_all",
    "family_gene",
    "family_drug",
    "structure_single",
    "structure_multi",
    "type_CRISPRi",
    "type_CRISPRa",
    "type_CRISPRko",
    "type_Cas13",
    "type_drug",
)


def _clean(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "<na>"} else s


def _load_condition_metadata(data_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = data_dir / "condition_metadata.json"
    if not path.is_file():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for ds, ds_obj in obj.items():
        if not isinstance(ds_obj, dict):
            continue
        ds_map: dict[str, dict[str, Any]] = {}
        for cond, entry in ds_obj.items():
            if isinstance(entry, dict):
                ds_map[str(cond)] = entry
        if ds_map:
            out[str(ds)] = ds_map
    return out


def _pert_type(entry: dict[str, Any]) -> str:
    raw = _clean(entry.get("perturbation_type_raw", entry.get("perturbation_type")))
    rl = raw.lower()
    if rl in {"crispri", "knockdown", "kd"}:
        return "CRISPRi"
    if rl in {"crispra", "activation", "overexpression"}:
        return "CRISPRa"
    if rl in {"crisprko", "ko", "knockout"}:
        return "CRISPRko"
    if rl == "cas13":
        return "Cas13"
    if rl in {"drug", "chemical", "compound", "small molecule", "small-molecule"}:
        return "drug"
    return raw or "unknown"


def _genes(entry: dict[str, Any]) -> list[str]:
    g = entry.get("genes")
    if not isinstance(g, list):
        return []
    return [str(x).strip() for x in g if str(x).strip()]


def _is_drug(entry: dict[str, Any], ds_name: str) -> bool:
    typ = _pert_type(entry).lower()
    if typ == "drug":
        return True
    dsl = ds_name.lower()
    return any(tok in dsl for tok in ("sciplex", "chempert", "chemical", "drug"))


def _base_test_conditions(
    *,
    manifest: dict[str, Any],
    split: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ds_name, ds_meta in manifest.get("datasets", {}).items():
        allowed = set(map(str, ds_meta.get("conditions", [])))
        conds = [str(c) for c in split.get(str(ds_name), {}).get("test", []) if str(c) in allowed]
        if conds:
            out[str(ds_name)] = conds
    return out


def build_family_group_splits(
    *,
    manifest: dict[str, Any],
    split: dict[str, dict[str, list[str]]],
    condition_metadata: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    base = _base_test_conditions(manifest=manifest, split=split)
    buckets: dict[str, dict[str, list[str]]] = {g: {} for g in DEFAULT_GROUPS}

    def add(group: str, ds: str, cond: str) -> None:
        buckets.setdefault(group, {}).setdefault(ds, []).append(cond)

    for ds, conds in base.items():
        ds_meta = condition_metadata.get(ds, {})
        for cond in conds:
            entry = ds_meta.get(cond, {})
            genes = _genes(entry)
            typ = _pert_type(entry)
            is_drug = _is_drug(entry, ds)

            add("test_all", ds, cond)
            if is_drug:
                add("family_drug", ds, cond)
            elif genes:
                add("family_gene", ds, cond)
            else:
                add("family_unknown", ds, cond)

            add(f"type_{typ}", ds, cond)
            if len(genes) > 1 or "+" in str(cond):
                add("structure_multi", ds, cond)
            elif genes or not is_drug:
                add("structure_single", ds, cond)

    # Preserve canonical split-group labels when present, intersected to this manifest.
    for group in ("test_single", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"):
        for ds_name, ds_meta in manifest.get("datasets", {}).items():
            allowed = set(map(str, ds_meta.get("conditions", [])))
            conds = [str(c) for c in split.get(str(ds_name), {}).get(group, []) if str(c) in allowed]
            if conds:
                buckets.setdefault(group, {})[str(ds_name)] = conds

    out: dict[str, dict[str, dict[str, list[str]]]] = {}
    for group, by_ds in buckets.items():
        clean = {ds: {"train": [], "test": sorted(set(conds))} for ds, conds in sorted(by_ds.items()) if conds}
        if clean:
            out[group] = clean
    return out


def _count_group(group_split: dict[str, dict[str, list[str]]]) -> int:
    return sum(len(v.get("test", [])) for v in group_split.values())


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-file", type=Path, default=None, help="Default: <biflow_dir>/split_seed<seed>.json")
    ap.add_argument("--data-dir", type=str, default="", help="Override checkpoint config data_dir")
    ap.add_argument("--biflow-dir", type=str, default="", help="Override checkpoint config biflow_dir")
    ap.add_argument("--groups", nargs="*", default=list(DEFAULT_GROUPS))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
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
    ap.add_argument(
        "--pert-means-file",
        type=str,
        default="",
        help="Override dataset perturbed mean reference for pearson_pert; default: <data-dir>/pert_means.npz",
    )
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

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
    condition_metadata = _load_condition_metadata(data_dir)
    group_splits = build_family_group_splits(
        manifest=manifest,
        split=split,
        condition_metadata=condition_metadata,
    )
    ctrl_means = _load_means(data_dir, "ctrl_means.npz")
    pert_means_path = _resolve_means_file(
        str(args.pert_means_file or ""),
        data_dir=data_dir,
        default_name="pert_means.npz",
    )
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
            "[eval_condition_families] EMA state present but inactive for checkpoint "
            f"step={ckpt.get('step')} < ema_update_after={getattr(cfg, 'ema_update_after', 0)}; using raw weights"
        )

    path = CondOTPath()
    cd_kw = _cross_dataset_kw(cfg)
    wanted = list(args.groups)
    results: dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step"),
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "groups": {},
        "available_group_counts": {k: _count_group(v) for k, v in sorted(group_splits.items())},
        "config": dataclasses.asdict(cfg),
        "means_files": {
            "ctrl_means": str(data_dir / "ctrl_means.npz"),
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

    for group in wanted:
        group_split = group_splits.get(group)
        n_requested = 0 if group_split is None else _count_group(group_split)
        if not group_split or n_requested == 0:
            results["groups"][group] = {
                "skipped": True,
                "reason": "no conditions in family group after manifest/split filtering",
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

    out_path = args.out or (ckpt_path.parent / "condition_family_eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "groups": list(results["groups"].keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
