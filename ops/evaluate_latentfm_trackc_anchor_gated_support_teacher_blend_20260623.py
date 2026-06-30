#!/usr/bin/env python3
"""Evaluate an anchor-gated support-teacher endpoint blend.

This is a default-off Track C posthoc evaluator for the frozen mechanism:

    pred = anchor_pred + gate * alpha * (support_teacher_pred - anchor_pred)

It evaluates the blend on already-authorized scopes only:

* ``support_trainselect``: safe Track C support-val rows, gate=1.
* ``canonical_noharm``: canonical single/background rows, gate=0.
* ``heldout_query_once``: final query rows only after a frozen route pass,
  gate=1.

Held-out Track C query groups are rejected unless the explicit
``heldout_query_once`` scope is selected.  Canonical multi groups are not
allowed in the no-harm gate because they are not selection evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from model.latent.config import Config
from model.latent.dataset import CrossDatasetFMDataset
from model.latent.eval_condition_families import build_family_group_splits
from model.latent.eval_split_groups import (
    _group_as_test_split,
    _json_default,
    _load_cfg,
    _load_manifest,
    _load_means_file,
    _load_split,
    _resolve_means_file,
)
from model.latent.fm_ot import median_sigmas, mmd2_biased, mmd2_unbiased
from model.latent.train import (
    _cross_dataset_kw,
    _eval_rank,
    _model_uses_pert,
    _model_uses_support_context,
    _pearson_np,
    _pert_chunk,
    _pert_for_eval_batch,
    _pert_to_device,
    _select_eval_condition_pairs,
    build_model,
    build_trackc_routed_distill_bank,
    checkpoint_ema_is_active,
    load_model_weights_only,
    make_trackc_support_context_batch,
    ode_integrate,
    validate_support_context_config,
)
from model.utils.train.ema import ModelEMA


ROOT = Path("/data/cyx/1030/scLatent")
SUPPORT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
CANONICAL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"

SUPPORT_GROUPS = {"support_val_multi", "test", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"}
CANONICAL_NOHARM_GROUPS = {"test_single", "family_gene", "test_all"}
QUERY_GROUPS = {
    "heldout_query_multi_final_only",
    "heldout_query_multi_seen_final_only",
    "heldout_query_multi_unseen1_final_only",
    "heldout_query_multi_unseen2_final_only",
}
FORBIDDEN_TOKENS = ("heldout_query", "query")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"checkpoint must contain a 'model' key: {path}")
    return ckpt


def _apply_overrides(cfg: Config, args: argparse.Namespace, *, checkpoint_label: str) -> None:
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.biflow_dir:
        cfg.biflow_dir = args.biflow_dir
    cfg.gpu = int(args.gpu)
    for attr, val in (
        ("eval_max_conditions", args.eval_max_conditions),
        ("eval_max_conditions_per_dataset", args.eval_max_conditions_per_dataset),
        ("eval_max_mse_cells", args.eval_max_mse_cells),
        ("eval_max_mmd_cells", args.eval_max_mmd_cells),
    ):
        if val is not None:
            setattr(cfg, attr, int(val))
    if int(getattr(cfg, "emb_dim", 0) or 0) <= 0:
        raise ValueError(f"{checkpoint_label} config has invalid emb_dim={getattr(cfg, 'emb_dim', None)!r}")


def _load_model_bundle(
    *,
    ckpt_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    ckpt = _load_checkpoint(ckpt_path)
    cfg = _load_cfg(ckpt, data_dir=args.data_dir, biflow_dir=args.biflow_dir)
    _apply_overrides(cfg, args, checkpoint_label=label)
    model = build_model(cfg, device)
    missing_keys, unexpected_keys, skipped_shape_mismatch = load_model_weights_only(
        ckpt_path,
        model,
        device,
        strict=False,
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
    model.eval()
    return {
        "path": str(ckpt_path),
        "checkpoint": ckpt,
        "cfg": cfg,
        "model": model,
        "ema": ema,
        "used_ema": ema is not None,
        "load_state": {
            "strict": False,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "skipped_shape_mismatch": skipped_shape_mismatch,
        },
    }


def _resolve_split(args: argparse.Namespace) -> Path:
    if args.split_file is not None:
        return args.split_file.expanduser().resolve()
    if args.scope in {"support_trainselect", "heldout_query_once"}:
        return SUPPORT_SPLIT
    return CANONICAL_SPLIT


def _validate_scope(*, scope: str, group_kind: str, groups: list[str], split_path: Path) -> None:
    lowered = [g.lower() for g in groups]
    if scope != "heldout_query_once" and any(tok in g for g in lowered for tok in FORBIDDEN_TOKENS):
        raise ValueError("held-out query groups are forbidden in this evaluator")
    if scope == "support_trainselect":
        if split_path.name != SUPPORT_SPLIT.name:
            raise ValueError(f"support_trainselect scope requires {SUPPORT_SPLIT}")
        bad = sorted(set(groups) - SUPPORT_GROUPS)
        if bad:
            raise ValueError(f"support_trainselect scope received unsupported groups: {bad}")
        if group_kind != "split":
            raise ValueError("support_trainselect scope requires --group-kind split")
        return
    if scope == "canonical_noharm":
        bad_multi = [g for g in groups if "multi" in g.lower()]
        if bad_multi:
            raise ValueError(f"canonical_noharm scope forbids canonical multi groups: {bad_multi}")
        bad = sorted(set(groups) - CANONICAL_NOHARM_GROUPS)
        if bad:
            raise ValueError(f"canonical_noharm scope received unsupported groups: {bad}")
        if group_kind == "family" and any(g not in {"family_gene", "test_all"} for g in groups):
            raise ValueError("family group-kind currently supports family_gene/test_all for no-harm")
        return
    if scope == "heldout_query_once":
        if split_path.name != SUPPORT_SPLIT.name:
            raise ValueError(f"heldout_query_once scope requires {SUPPORT_SPLIT}")
        if group_kind != "split":
            raise ValueError("heldout_query_once scope requires --group-kind split")
        bad = sorted(set(groups) - QUERY_GROUPS)
        if bad:
            raise ValueError(f"heldout_query_once scope received unsupported groups: {bad}")
        return
    raise ValueError(f"unsupported scope: {scope!r}")


def _build_group_splits(
    *,
    group_kind: str,
    groups: list[str],
    manifest: dict[str, Any],
    split: dict[str, dict[str, list[str]]],
    data_dir: Path,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    if group_kind == "split":
        return {
            group: _group_as_test_split(split=split, manifest=manifest, group=group)
            for group in groups
        }
    if group_kind == "family":
        family_splits = build_family_group_splits(
            manifest=manifest,
            split=split,
            condition_metadata=_load_condition_metadata(data_dir),
        )
        return {group: family_splits.get(group, {}) for group in groups}
    raise ValueError(f"unsupported group kind: {group_kind!r}")


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
        entries = {str(cond): meta for cond, meta in ds_obj.items() if isinstance(meta, dict)}
        if entries:
            out[str(ds)] = entries
    return out


def _gate_for_scope(scope: str) -> float:
    if scope in {"support_trainselect", "heldout_query_once"}:
        return 1.0
    if scope == "canonical_noharm":
        return 0.0
    raise ValueError(f"unsupported scope: {scope!r}")


def _support_bank_for(bundle: dict[str, Any], dataset: CrossDatasetFMDataset) -> dict[str, Any]:
    cfg = bundle["cfg"]
    model = bundle["model"]
    if not _model_uses_support_context(model):
        return {}
    validate_support_context_config(cfg)
    return build_trackc_routed_distill_bank(dataset, cfg)


def _empty_group_result(reason: str, cfg: Config, group_split: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    return {
        "skipped": True,
        "reason": reason,
        "n_requested": int(sum(len(v.get("test", [])) for v in group_split.values())),
        "n_available_conditions": 0,
        "n_conds": 0,
        "eval_caps": {
            "eval_max_conditions": int(getattr(cfg, "eval_max_conditions", 0) or 0),
            "eval_max_conditions_per_dataset": int(getattr(cfg, "eval_max_conditions_per_dataset", 0) or 0),
            "eval_max_mmd_cells": int(getattr(cfg, "eval_max_mmd_cells", 0) or 0),
            "condition_selection": "stable_hash_dataset_condition",
            "cell_selection": "stable_hash_dataset_condition_metric",
            "aggregation": "condition_mean_then_dataset_equal_mean",
        },
        "selected_conditions": [],
        "condition_metrics": [],
        "per_ds": {},
    }


def _ds_mean(d: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) for k, v in sorted(d.items()) if v}


def _equal_ds_mean(d: dict[str, float]) -> float:
    return float(np.mean(list(d.values()))) if d else float("nan")


@torch.no_grad()
def evaluate_anchor_teacher_blend(
    *,
    anchor: dict[str, Any],
    teacher: dict[str, Any],
    dataset: CrossDatasetFMDataset,
    scope: str,
    alpha: float,
    device: torch.device,
    ctrl_means: dict[str, np.ndarray] | None,
    pert_means: dict[str, np.ndarray] | None,
    ode_steps: int,
    max_chunk: int,
) -> dict[str, Any]:
    anchor_model = anchor["model"]
    teacher_model = teacher["model"]
    anchor_cfg = anchor["cfg"]
    teacher_cfg = teacher["cfg"]
    anchor_model.eval()
    teacher_model.eval()

    if int(getattr(anchor_cfg, "emb_dim", 0) or 0) != int(getattr(teacher_cfg, "emb_dim", 0) or 0):
        raise ValueError("anchor and support-teacher emb_dim mismatch")

    gate = _gate_for_scope(scope)
    cond_pairs, n_available_conditions = _select_eval_condition_pairs(dataset, anchor_cfg)
    max_mmd_cells = int(getattr(anchor_cfg, "eval_max_mmd_cells", 2048) or 2048)
    emb_dim = int(getattr(anchor_cfg, "emb_dim", 0) or 0)

    anchor_use_pe = _model_uses_pert(anchor_model)
    teacher_use_pe = _model_uses_pert(teacher_model)
    anchor_use_support = _model_uses_support_context(anchor_model)
    teacher_use_support = _model_uses_support_context(teacher_model)
    anchor_bank = _support_bank_for(anchor, dataset)
    teacher_bank = _support_bank_for(teacher, dataset)

    per_ds: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    condition_metrics: list[dict[str, Any]] = []

    for ds_name, cond in cond_pairs:
        handle = dataset.handles[ds_name]
        ctrl_mean = None if ctrl_means is None else ctrl_means.get(ds_name)
        pert_mean = None if pert_means is None else pert_means.get(ds_name)

        src_full = torch.from_numpy(handle.read_src(cond)).float()
        gt_full = torch.from_numpy(handle.read_gt(cond)).float()
        seed = int(getattr(anchor_cfg, "seed", 0) or 0)
        rng_mmd = np.random.RandomState(_eval_rank(seed, "mmd_cells", ds_name, cond) % (2**32 - 1))
        n_src_eval = min(int(src_full.size(0)), max_mmd_cells)
        n_gt_eval = min(int(gt_full.size(0)), max_mmd_cells)
        src_eval = src_full[rng_mmd.permutation(src_full.size(0))[:n_src_eval]]
        gt_eval = gt_full[rng_mmd.permutation(gt_full.size(0))[:n_gt_eval]].to(device)

        pb_cpu = _pert_for_eval_batch(dataset, ds_name, cond, n_src_eval) if (anchor_use_pe or teacher_use_pe) else None
        pb_dev = None if pb_cpu is None else _pert_to_device(pb_cpu, device)

        anchor_support_cpu = None
        if anchor_use_support:
            anchor_support_cpu = make_trackc_support_context_batch(
                anchor_bank,
                dataset,
                ds_name,
                cond,
                n_src_eval,
                anchor_cfg,
                torch.device("cpu"),
            )
        teacher_support_cpu = None
        if teacher_use_support:
            teacher_support_cpu = make_trackc_support_context_batch(
                teacher_bank,
                dataset,
                ds_name,
                cond,
                n_src_eval,
                teacher_cfg,
                torch.device("cpu"),
            )

        anchor_pred = torch.empty(n_src_eval, emb_dim, device=device)
        teacher_pred = torch.empty(n_src_eval, emb_dim, device=device)
        for st in range(0, n_src_eval, max_chunk):
            en = min(st + max_chunk, n_src_eval)
            s = src_eval[st:en].to(device)
            pb_use = None if pb_dev is None else _pert_chunk(pb_dev, st, en)
            anchor_support = None if anchor_support_cpu is None else anchor_support_cpu[st:en].to(device)
            teacher_support = None if teacher_support_cpu is None else teacher_support_cpu[st:en].to(device)
            anchor_pred[st:en] = ode_integrate(
                anchor_model,
                s,
                s,
                anchor_cfg,
                n_steps=ode_steps,
                perturbation_batch=pb_use if anchor_use_pe else None,
                support_context=anchor_support,
            )
            teacher_pred[st:en] = ode_integrate(
                teacher_model,
                s,
                s,
                teacher_cfg,
                n_steps=ode_steps,
                perturbation_batch=pb_use if teacher_use_pe else None,
                support_context=teacher_support,
            )

        blend_pred = anchor_pred + float(gate) * float(alpha) * (teacher_pred - anchor_pred)

        sigmas, Dyy = median_sigmas(gt_eval, return_D2=True)
        metrics: dict[str, float] = {}
        for prefix, pred in (("anchor", anchor_pred), ("teacher", teacher_pred), ("blend", blend_pred)):
            mmd = float(mmd2_unbiased(pred, gt_eval, sigmas, Dyy=Dyy).item())
            mmd_biased = float(mmd2_biased(pred, gt_eval, sigmas, Dyy=Dyy).item())
            pred_mean = pred.mean(dim=0).cpu().numpy()
            gt_mean = gt_eval.mean(dim=0).cpu().numpy()
            metrics[f"{prefix}_test_mmd"] = mmd
            metrics[f"{prefix}_test_mmd_biased"] = mmd_biased
            metrics[f"{prefix}_test_mmd_clamped"] = max(mmd, 0.0)
            metrics[f"{prefix}_direct_pearson"] = _pearson_np(pred_mean, gt_mean)
            if ctrl_mean is not None:
                metrics[f"{prefix}_pearson_ctrl"] = _pearson_np(pred_mean - ctrl_mean, gt_mean - ctrl_mean)
            if pert_mean is not None:
                metrics[f"{prefix}_pearson_pert"] = _pearson_np(pred_mean - pert_mean, gt_mean - pert_mean)

        record = {
            "dataset": str(ds_name),
            "condition": str(cond),
            "gate": float(gate),
            "alpha": float(alpha),
            "effective_alpha": float(gate) * float(alpha),
            "n_src_eval": int(n_src_eval),
            "n_gt_eval": int(n_gt_eval),
            **metrics,
        }
        for key in (
            "test_mmd",
            "test_mmd_biased",
            "test_mmd_clamped",
            "direct_pearson",
            "pearson_ctrl",
            "pearson_pert",
        ):
            a_key = f"anchor_{key}"
            t_key = f"teacher_{key}"
            b_key = f"blend_{key}"
            if a_key in metrics and b_key in metrics:
                record[f"blend_delta_vs_anchor_{key}"] = metrics[b_key] - metrics[a_key]
            if a_key in metrics and t_key in metrics:
                record[f"teacher_delta_vs_anchor_{key}"] = metrics[t_key] - metrics[a_key]
        condition_metrics.append(record)

        for key, value in record.items():
            if key in {"dataset", "condition"} or value is None:
                continue
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                per_ds[str(ds_name)][key].append(float(value))

        del gt_eval, anchor_pred, teacher_pred, blend_pred

    per_ds_out: dict[str, dict[str, float]] = {}
    overall: dict[str, float] = {}
    metric_keys = sorted({k for ds_vals in per_ds.values() for k in ds_vals})
    for metric in metric_keys:
        ds_metric = _ds_mean({ds: vals.get(metric, []) for ds, vals in per_ds.items()})
        if ds_metric:
            overall[metric] = _equal_ds_mean(ds_metric)
            for ds, value in ds_metric.items():
                per_ds_out.setdefault(ds, {})[metric] = value

    return {
        "skipped": False,
        "scope": scope,
        "gate": float(gate),
        "alpha": float(alpha),
        "n_conds": int(len(condition_metrics)),
        "n_available_conditions": int(n_available_conditions),
        "eval_caps": {
            "eval_max_conditions": int(getattr(anchor_cfg, "eval_max_conditions", 0) or 0),
            "eval_max_conditions_per_dataset": int(getattr(anchor_cfg, "eval_max_conditions_per_dataset", 0) or 0),
            "eval_max_mmd_cells": int(max_mmd_cells),
            "condition_selection": "stable_hash_dataset_condition",
            "cell_selection": "stable_hash_dataset_condition_metric",
            "aggregation": "condition_mean_then_dataset_equal_mean",
            "endpoint_blend": "anchor + gate * alpha * (support_teacher - anchor)",
            "velocity_mse": "not_defined_for_endpoint_blend",
        },
        "selected_conditions": [{"dataset": ds, "condition": cond} for ds, cond in cond_pairs],
        "per_ds": per_ds_out,
        "overall": overall,
        "condition_metrics": condition_metrics,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-checkpoint", type=Path, required=True)
    ap.add_argument("--support-teacher-checkpoint", type=Path, required=True)
    ap.add_argument("--scope", choices=("support_trainselect", "canonical_noharm", "heldout_query_once"), required=True)
    ap.add_argument("--group-kind", choices=("split", "family"), default="split")
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--split-file", type=Path, default=None)
    ap.add_argument("--data-dir", type=str, default="")
    ap.add_argument("--biflow-dir", type=str, default="")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--ode-steps", type=int, default=20)
    ap.add_argument("--max-chunk", type=int, default=512)
    ap.add_argument("--eval-max-conditions", type=int, default=None)
    ap.add_argument("--eval-max-conditions-per-dataset", type=int, default=None)
    ap.add_argument("--eval-max-mse-cells", type=int, default=None)
    ap.add_argument("--eval-max-mmd-cells", type=int, default=None)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--pert-means-file", type=str, default="")
    args = ap.parse_args()

    split_path = _resolve_split(args)
    _validate_scope(
        scope=str(args.scope),
        group_kind=str(args.group_kind),
        groups=[str(g) for g in args.groups],
        split_path=split_path,
    )
    if not (0.0 <= float(args.alpha) <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    device_s = args.device.strip() or (f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    anchor_path = args.anchor_checkpoint.expanduser().resolve()
    teacher_path = args.support_teacher_checkpoint.expanduser().resolve()

    anchor = _load_model_bundle(ckpt_path=anchor_path, args=args, device=device, label="anchor")
    teacher = _load_model_bundle(ckpt_path=teacher_path, args=args, device=device, label="support_teacher")
    anchor_cfg = anchor["cfg"]
    data_dir = Path(anchor_cfg.data_dir).expanduser().resolve()
    manifest = _load_manifest(data_dir, anchor_cfg.manifest)
    split = _load_split(split_path)
    group_splits = _build_group_splits(
        group_kind=str(args.group_kind),
        groups=[str(g) for g in args.groups],
        manifest=manifest,
        split=split,
        data_dir=data_dir,
    )

    ctrl_means_path = data_dir / "ctrl_means.npz"
    pert_means_path = _resolve_means_file(str(args.pert_means_file or ""), data_dir=data_dir, default_name="pert_means.npz")
    ctrl_means = _load_means_file(ctrl_means_path)
    pert_means = _load_means_file(pert_means_path)
    cd_kw = _cross_dataset_kw(anchor_cfg)

    results: dict[str, Any] = {
        "status": "anchor_gated_support_teacher_blend_posthoc_complete",
        "scope": str(args.scope),
        "group_kind": str(args.group_kind),
        "alpha": float(args.alpha),
        "split_file": str(split_path),
        "data_dir": str(data_dir),
        "anchor_checkpoint": str(anchor_path),
        "support_teacher_checkpoint": str(teacher_path),
        "used_ema": {
            "anchor": bool(anchor["used_ema"]),
            "support_teacher": bool(teacher["used_ema"]),
        },
        "load_state": {
            "anchor": anchor["load_state"],
            "support_teacher": teacher["load_state"],
        },
        "means_files": {
            "ctrl_means": str(ctrl_means_path),
            "pert_means": str(pert_means_path),
            "pert_means_override": bool(str(args.pert_means_file or "").strip()),
            "ctrl_means_loaded": ctrl_means is not None,
            "pert_means_loaded": pert_means is not None,
        },
        "safety": {
            "heldout_query_read": str(args.scope) == "heldout_query_once",
            "canonical_multi_selection": False,
            "support_scope_split_required": str(SUPPORT_SPLIT),
            "query_scope_split_required": str(SUPPORT_SPLIT),
            "canonical_scope_allows_multi": False,
            "query_result_may_select_or_tune": False,
        },
        "config": {
            "anchor": dataclasses.asdict(anchor_cfg),
            "support_teacher": dataclasses.asdict(teacher["cfg"]),
        },
        "groups": {},
    }

    with contextlib.ExitStack() as stack:
        if anchor["ema"] is not None:
            stack.enter_context(anchor["ema"].apply_to(anchor["model"]))
        if teacher["ema"] is not None:
            stack.enter_context(teacher["ema"].apply_to(teacher["model"]))
        for group, group_split in group_splits.items():
            n_requested = sum(len(v.get("test", [])) for v in group_split.values())
            if n_requested == 0:
                results["groups"][group] = _empty_group_result("no conditions after manifest filtering", anchor_cfg, group_split)
                continue
            ds = CrossDatasetFMDataset(
                str(data_dir),
                group_split,
                anchor_cfg.batch_size,
                anchor_cfg.seed,
                mode="test",
                min_cells=16,
                ds_alpha=1.0,
                silent=False,
                **cd_kw,
            )
            group_result = evaluate_anchor_teacher_blend(
                anchor=anchor,
                teacher=teacher,
                dataset=ds,
                scope=str(args.scope),
                alpha=float(args.alpha),
                device=device,
                ctrl_means=ctrl_means,
                pert_means=pert_means,
                ode_steps=int(args.ode_steps),
                max_chunk=int(args.max_chunk),
            )
            group_result["n_requested"] = int(n_requested)
            results["groups"][group] = group_result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "groups": list(results["groups"].keys())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
