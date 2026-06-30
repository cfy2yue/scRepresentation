#!/usr/bin/env python3
"""One-step CPU gradient/path dry-run for the allowlisted-tail route.

This is a pre-GPU gate. It loads an existing allowlisted-tail config/checkpoint,
selects a train-only allowlisted single-gene condition, and compares one-step
gradients/virtual parameter movement with route losses disabled vs enabled.

It does not run training, inference, canonical evaluation, canonical multi, or
Track C query. It performs backward passes only on one or a few train batches.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
if str(COUPLEDFM) not in sys.path:
    sys.path.insert(0, str(COUPLEDFM))

import torch  # noqa: E402

from model.latent.config import Config  # noqa: E402
from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402
from model.latent.eval_split_groups import _load_manifest, _load_split  # noqa: E402
from model.latent.fm_ot import CondOTPath  # noqa: E402
from model.latent.train import (  # noqa: E402
    _cross_dataset_kw,
    _model_condition_delta,
    build_condition_prior_delta_bank,
    build_model,
    condition_delta_head_loss_schedule,
    condition_prior_additive_delta_loss_schedule,
    condition_prior_delta_loss_schedule,
    endpoint_delta_loss_schedule,
    gamma_schedule,
    load_model_weights_only,
    sample_condition_prior_teacher,
    train_step,
)


DEFAULT_RUN = (
    ROOT
    / "CoupledFM/output/latentfm_runs/latentfm_tracka_xverse_allowlisted_tail_hybrid_20260627/"
    "xverse_allowtail_hybrid_pertresid_prior_w003_p002_replay1_2k_seed42"
)
OUT_DIR = ROOT / "reports/latentfm_allowtail_gradient_path_dryrun_20260627"
JSON_PATH = ROOT / "reports/latentfm_allowtail_gradient_path_dryrun_20260627.json"
MD_PATH = ROOT / "reports/LATENTFM_ALLOWTAIL_GRADIENT_PATH_DRYRUN_20260627.md"


def load_cfg(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = Config()
    valid = {f.name for f in dataclasses.fields(Config)}
    for k, v in raw.items():
        if k in valid:
            setattr(cfg, k, v)
    cfg.use_amp = False
    cfg.batch_size = min(int(getattr(cfg, "batch_size", 64) or 64), 16)
    cfg.scale_noise = 0.0
    cfg.n_ot_workers = 0
    cfg.prefetch = 1
    return cfg


def load_allowlist(path_s: str) -> set[str]:
    path = Path(path_s).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().split(",")[0].split("\t")[0].strip().upper()
        if s and not s.startswith("#"):
            out.add(s)
    return out


def cfg_split(cfg: Config) -> dict[str, dict[str, list[str]]]:
    split_file = str(getattr(cfg, "split_file", "") or "").strip()
    if not split_file:
        raise ValueError("dry-run requires an explicit split_file in config")
    return _load_split(Path(split_file))


def build_train_dataset(cfg: Config) -> CrossDatasetFMDataset:
    manifest = _load_manifest(Path(cfg.data_dir), str(getattr(cfg, "manifest", "manifest.json") or "manifest.json"))
    split = cfg_split(cfg)
    iid_split = {
        ds: {
            "train": [c for c in sp.get("train", []) if c in set(manifest.get("datasets", {}).get(ds, {}).get("conditions", []))],
            "test": [c for c in sp.get("test", []) if c in set(manifest.get("datasets", {}).get(ds, {}).get("conditions", []))],
        }
        for ds, sp in split.items()
        if ds in manifest.get("datasets", {}) and sp.get("train")
    }
    return CrossDatasetFMDataset(
        cfg.data_dir,
        iid_split,
        batch_size=int(cfg.batch_size),
        seed=int(cfg.seed),
        mode="train",
        min_cells=int(cfg.min_cells),
        ds_alpha=float(cfg.ds_alpha),
        scale_noise=0.0,
        min_selected_conditions_per_dataset=int(getattr(cfg, "min_selected_conditions_per_dataset", 0) or 0),
        condition_visit_power=float(getattr(cfg, "condition_visit_power", 1.0) or 1.0),
        condition_visit_cap=int(getattr(cfg, "condition_visit_cap", 0) or 0),
        perturbation_family_filter=str(getattr(cfg, "perturbation_family_filter", "all") or "all"),
        silent=True,
        **_cross_dataset_kw(cfg),
    )


def find_allowlisted_batch(
    ds: CrossDatasetFMDataset,
    allowlist: set[str],
    *,
    max_scan: int,
) -> tuple[torch.Tensor, torch.Tensor, str, str, tuple[Any, ...]]:
    for idx, item in enumerate(ds):
        if idx >= max_scan:
            break
        src, gt, ds_name, cond, pb = item
        meta = ds.metadata_for_condition(ds_name, cond)
        genes = {str(g).strip().upper() for g in getattr(meta, "genes", ()) if str(g).strip()}
        if pb is not None and len(genes) == 1 and (not allowlist or genes & allowlist):
            return src.float(), gt.float(), str(ds_name), str(cond), pb
    raise RuntimeError(f"no allowlisted train batch found after scanning {max_scan} batches")


def load_pert_means(path_s: str) -> dict[str, torch.Tensor]:
    path = Path(path_s).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pert_means_file not found: {path}")
    obj = np.load(str(path))
    return {str(k): torch.from_numpy(np.asarray(v)).float() for k, v in obj.items()}


def grad_group(name: str) -> str:
    if name.startswith("condition_delta_head.") or name.startswith("condition_delta_to_c."):
        return "condition_delta_head_or_to_c"
    if "pert_encoder" in name or "pert_to_c" in name:
        return "pert_conditioning"
    return "fm_body"


def grad_norms(model: torch.nn.Module) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        sums[grad_group(name)] += float(torch.sum(g * g).item())
    return {k: math.sqrt(v) for k, v in sorted(sums.items())}


def total_grad_norm(norms: dict[str, float]) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in norms.values()))


def condition_delta_mean(model: torch.nn.Module, pb: tuple[Any, ...], device: torch.device) -> torch.Tensor | None:
    pb_dev = tuple(None if x is None else x.to(device) if hasattr(x, "to") else x for x in pb)
    out = _model_condition_delta(model, pb_dev)
    if out is None:
        return None
    return out.detach().float().mean(dim=0).cpu()


def ensure_explicit_no_chem_mask(pb: tuple[Any, ...]) -> tuple[Any, ...]:
    """Make no-chemical evidence explicit for fail-closed allowlist gates."""
    if len(pb) == 5:
        gid, mk, tid, npt, cid = pb
        ce = None
        cm = None
    else:
        gid, mk, tid, npt, cid, ce, cm = pb
    if cm is None:
        batch = int(gid.shape[0])
        cm = torch.zeros((batch, 1), dtype=torch.bool, device=gid.device)
    return gid, mk, tid, npt, cid, ce, cm


def run_backward_with_cache(
    *,
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device,
    src: torch.Tensor,
    gt: torch.Tensor,
    ds_name: str,
    cond: str,
    pb: tuple[Any, ...],
    pert_mean_ref: torch.Tensor,
    condition_prior_bank: dict[str, Any],
    gene_cache: Any,
    step: int,
    enable_routes: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.zero_grad(set_to_none=True)
    fm_path = CondOTPath()
    route_prior_w = condition_prior_delta_loss_schedule(step, cfg) if enable_routes else 0.0
    route_prior_add_w = condition_prior_additive_delta_loss_schedule(step, cfg) if enable_routes else 0.0
    route_head_w = condition_delta_head_loss_schedule(step, cfg) if enable_routes else 0.0
    condition_prior_delta_target = None
    condition_prior_perturbation_batch = None
    if (route_prior_w > 0 or route_prior_add_w > 0) and condition_prior_bank:
        condition_prior_delta_target, condition_prior_perturbation_batch = sample_condition_prior_teacher(
            bank=condition_prior_bank,
            ds_name=ds_name,
            step=step,
            cond=cond,
            batch_size=int(src.size(0)),
            cache=gene_cache,
            max_genes=int(getattr(cfg, "max_pert_genes", 16)),
            max_chem_keys=int(getattr(cfg, "max_chem_keys", 4)),
            num_genes=int(getattr(cfg, "condition_prior_num_genes", 1)),
        )
        if condition_prior_perturbation_batch is not None:
            condition_prior_perturbation_batch = ensure_explicit_no_chem_mask(condition_prior_perturbation_batch)
    out = train_step(
        src,
        gt,
        model,
        fm_path,
        cfg,
        device,
        ds_name=ds_name,
        gamma_t=gamma_schedule(step, cfg) if int(step) % max(1, int(getattr(cfg, "mmd_every", 1) or 1)) == 0 else 0.0,
        endpoint_delta_weight_t=endpoint_delta_loss_schedule(step, cfg),
        condition_prior_delta_weight_t=route_prior_w,
        condition_prior_delta_target=condition_prior_delta_target,
        condition_prior_perturbation_batch=condition_prior_perturbation_batch,
        condition_prior_additive_delta_weight_t=route_prior_add_w,
        condition_delta_head_weight_t=route_head_w,
        pert_mean_ref=pert_mean_ref,
        perturbation_batch=pb,
    )
    out["loss"].backward()
    scalars = {}
    for key, value in out.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            scalars[key] = float(value.detach().cpu().item())
    scalars.update(
        {
            "route_prior_weight": float(route_prior_w),
            "route_prior_additive_weight": float(route_prior_add_w),
            "route_head_weight": float(route_head_w),
        }
    )
    return scalars, grad_norms(model)


def virtual_step_delta(
    *,
    model: torch.nn.Module,
    pb: tuple[Any, ...],
    device: torch.device,
    lr: float,
) -> float | None:
    before = condition_delta_mean(model, pb, device)
    if before is None:
        return None
    originals = []
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            originals.append((p, p.detach().clone()))
            p.add_(p.grad, alpha=-float(lr))
        after = condition_delta_mean(model, pb, device)
        for p, old in originals:
            p.copy_(old)
    if after is None:
        return None
    return float(torch.norm(after - before).item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--step", type=int, default=1000)
    ap.add_argument("--max-scan-batches", type=int, default=512)
    ap.add_argument("--virtual-lr", type=float, default=1e-4)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir
    cfg = load_cfg(run_dir / "config.json")
    checkpoint = args.checkpoint or (run_dir / "best.pt")
    device = torch.device("cpu")
    allowlist = load_allowlist(str(getattr(cfg, "condition_delta_allowlist_gene_file", "") or ""))

    ds = build_train_dataset(cfg)
    src, gt, ds_name, cond, pb = find_allowlisted_batch(ds, allowlist, max_scan=int(args.max_scan_batches))
    pb = ensure_explicit_no_chem_mask(pb)
    pert_means = load_pert_means(str(getattr(cfg, "pert_means_file", "") or ""))
    if ds_name not in pert_means:
        raise RuntimeError(f"pert mean missing for dataset {ds_name}")
    pert_mean_ref = pert_means[ds_name]

    model = build_model(cfg, device)
    missing, unexpected, skipped = load_model_weights_only(
        checkpoint,
        model,
        device,
        strict=False,
        prefer_ema=True,
    )
    model.train()

    condition_prior_bank = build_condition_prior_delta_bank(ds, cfg, log=None)
    gene_cache = ds.gene_embedding_cache
    base_scalars, base_norms = run_backward_with_cache(
        model=model,
        cfg=cfg,
        device=device,
        src=src,
        gt=gt,
        ds_name=ds_name,
        cond=cond,
        pb=pb,
        pert_mean_ref=pert_mean_ref,
        condition_prior_bank=condition_prior_bank,
        gene_cache=gene_cache,
        step=int(args.step),
        enable_routes=False,
    )
    base_move = virtual_step_delta(model=model, pb=pb, device=device, lr=float(args.virtual_lr))
    route_scalars, route_norms = run_backward_with_cache(
        model=model,
        cfg=cfg,
        device=device,
        src=src,
        gt=gt,
        ds_name=ds_name,
        cond=cond,
        pb=pb,
        pert_mean_ref=pert_mean_ref,
        condition_prior_bank=condition_prior_bank,
        gene_cache=gene_cache,
        step=int(args.step),
        enable_routes=True,
    )
    route_move = virtual_step_delta(model=model, pb=pb, device=device, lr=float(args.virtual_lr))
    ds.close()

    base_total = total_grad_norm(base_norms)
    route_total = total_grad_norm(route_norms)
    head_base = base_norms.get("condition_delta_head_or_to_c", 0.0)
    head_route = route_norms.get("condition_delta_head_or_to_c", 0.0)
    body_base = base_norms.get("fm_body", 0.0)
    body_route = route_norms.get("fm_body", 0.0)
    head_gain_ratio = head_route / max(head_base, 1e-12)
    total_gain_ratio = route_total / max(base_total, 1e-12)
    movement_gain = (route_move or 0.0) / max((base_move or 0.0), 1e-12)

    pass_gate = (
        route_total > 0
        and head_route > 1e-8
        and head_gain_ratio >= 1.25
        and movement_gain >= 1.25
        and route_scalars.get("condition_delta_head", 0.0) > 0
    )
    status = "pass_needs_metric_noharm_gate_no_gpu" if pass_gate else "fail_or_inconclusive_no_gpu"
    reasons = []
    if route_total <= 0:
        reasons.append("route_total_grad_zero")
    if head_route <= 1e-8:
        reasons.append("condition_delta_head_grad_near_zero")
    if head_gain_ratio < 1.25:
        reasons.append(f"head_grad_gain_ratio_{head_gain_ratio:.6g}_lt_1p25")
    if movement_gain < 1.25:
        reasons.append(f"virtual_head_movement_gain_{movement_gain:.6g}_lt_1p25")
    if route_scalars.get("condition_delta_head", 0.0) <= 0:
        reasons.append("condition_delta_head_loss_nonpositive")
    if pass_gate:
        reasons.append("gradient_path_material_but_requires_metric_noharm_gate")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "one_step_backward_only": True,
            "optimizer_step": False,
            "training_loop": False,
            "inference_or_posthoc_eval": False,
            "canonical_multi_used": False,
            "trackc_query_used": False,
            "split_file": str(getattr(cfg, "split_file", "")),
        },
        "inputs": {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "config": str(run_dir / "config.json"),
            "allowlist": str(getattr(cfg, "condition_delta_allowlist_gene_file", "")),
            "pert_means_file": str(getattr(cfg, "pert_means_file", "")),
            "step": int(args.step),
            "virtual_lr": float(args.virtual_lr),
        },
        "selected_batch": {
            "dataset": ds_name,
            "condition": cond,
            "batch_size": int(src.size(0)),
            "emb_dim": int(src.size(1)),
        },
        "load_state": {
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "skipped_shape_mismatch": skipped,
        },
        "condition_prior_bank": {
            "datasets": len(condition_prior_bank),
            "records": sum(len(v) for v in condition_prior_bank.values()),
        },
        "base_scalars": base_scalars,
        "route_scalars": route_scalars,
        "base_grad_norms": base_norms,
        "route_grad_norms": route_norms,
        "summary": {
            "base_total_grad_norm": base_total,
            "route_total_grad_norm": route_total,
            "total_grad_gain_ratio": total_gain_ratio,
            "base_head_grad_norm": head_base,
            "route_head_grad_norm": head_route,
            "head_grad_gain_ratio": head_gain_ratio,
            "base_body_grad_norm": body_base,
            "route_body_grad_norm": body_route,
            "base_virtual_condition_delta_movement": base_move,
            "route_virtual_condition_delta_movement": route_move,
            "virtual_condition_delta_movement_gain": movement_gain,
        },
        "reasons": reasons,
        "next_action": (
            "design metric/no-harm CPU gate before GPU" if pass_gate
            else "close current loss-schedule route or inspect with stronger architecture/unit gate"
        ),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    md = f"""# LatentFM Allowlisted-Tail Gradient Path Dry-Run

## Status

`{status}`

GPU authorized: `False`

## Boundary

CPU-only one-step backward dry-run on a train-only batch. No optimizer step, no
training loop, no inference/posthoc evaluation, no canonical multi selection,
and no Track C query use.

## Selected Batch

* Dataset: `{ds_name}`
* Condition: `{cond}`
* Batch size: `{int(src.size(0))}`
* Split: `{getattr(cfg, 'split_file', '')}`

## Gradient Summary

| Quantity | Baseline route-off | Route-on | Ratio |
|---|---:|---:|---:|
| total grad norm | {base_total:.6e} | {route_total:.6e} | {total_gain_ratio:.6f} |
| head/to-c grad norm | {head_base:.6e} | {head_route:.6e} | {head_gain_ratio:.6f} |
| body grad norm | {body_base:.6e} | {body_route:.6e} | {(body_route / max(body_base, 1e-12)):.6f} |
| virtual condition-delta movement | {(base_move or 0.0):.6e} | {(route_move or 0.0):.6e} | {movement_gain:.6f} |

## Route Loss Terms

* condition_delta_head: `{route_scalars.get('condition_delta_head', 0.0):.6e}`
* condition_prior_delta: `{route_scalars.get('condition_prior_delta', 0.0):.6e}`
* condition_prior_additive_delta: `{route_scalars.get('condition_prior_additive_delta', 0.0):.6e}`
* route weights: head `{route_scalars.get('route_head_weight', 0.0):.6g}`,
  prior `{route_scalars.get('route_prior_weight', 0.0):.6g}`,
  prior-add `{route_scalars.get('route_prior_additive_weight', 0.0):.6g}`

## Decision Reasons

{chr(10).join(f'- `{r}`' for r in reasons)}

## Decision

This dry-run can only authorize the next CPU metric/no-harm gate. It cannot
authorize GPU training by itself.

## Outputs

* JSON: `{JSON_PATH}`
"""
    MD_PATH.write_text(md)
    print(json.dumps({"status": status, "json": str(JSON_PATH), "md": str(MD_PATH)}, indent=2))


if __name__ == "__main__":
    main()
