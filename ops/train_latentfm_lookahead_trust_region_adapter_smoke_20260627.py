#!/usr/bin/env python3
"""Tiny lookahead/trust-region adapter training smoke for LatentFM.

Detached GPU smoke target. This script fine-tunes only the zero-initialized
condition-delta adapter path on a safe train-only split. It uses a conservative
line-search update:

1. compute task gradient on a train batch;
2. take a virtual unprojected step;
3. compute held-back train no-harm anchor gradient at the virtual point;
4. project the original task gradient against the lookahead no-harm gradient;
5. apply the first/strongest candidate step that improves task loss while
   keeping held-back anchor replay drift below a strict threshold.

It does not read canonical multi, Track C query, or held-out query data. It
saves a smoke checkpoint for posthoc internal/canonical no-harm evaluation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
sys.path.insert(0, str(COUPLEDFM))
sys.path.insert(0, str(ROOT))

from model.latent.config import Config  # noqa: E402
from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402
from model.latent.fm_ot import CondOTPath, OTPlanSampler  # noqa: E402
from model.latent.train import (  # noqa: E402
    _model_latent_velocity,
    _pert_to_device,
    apply_finetune_freeze,
    build_model,
    checkpoint_ema_is_active,
    load_model_weights_only,
)
from model.utils.train.ema import ModelEMA  # noqa: E402


ANCHOR_CKPT = (
    COUPLEDFM
    / "output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
STEP_GRID_DEFAULT = "1,3,10,30,100,300"
EPS = 1e-12


def parse_step_grid(text: str) -> list[float]:
    vals = [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if not vals:
        raise ValueError("empty step grid")
    return vals


def cfg_from_checkpoint(path: Path, *, batch_size: int, seed: int) -> Config:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    raw = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    cfg = Config()
    valid = {field.name for field in dataclasses.fields(Config)}
    for key, value in raw.items():
        if key in valid:
            setattr(cfg, key, value)
    cfg.data_dir = str(ROOT / "dataset/latentfm_full/xverse")
    cfg.biflow_dir = str(ROOT / "dataset/biFlow_data")
    cfg.split_file = str(SAFE_SPLIT)
    cfg.latent_backbone = "xverse"
    cfg.batch_size = int(batch_size)
    cfg.min_cells = 16
    cfg.scale_noise = 0.0
    cfg.seed = int(seed)
    cfg.use_amp = False
    cfg.mmd_ode_steps = 0
    cfg.use_pert_condition = True
    cfg.pert_gene_emb_cache_dir = str(ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene")
    cfg.pert_metainfo_path = str(COUPLEDFM / "data/raw/genepert_DE5000/metainfo.json")
    cfg.chemical_metainfo_path = str(COUPLEDFM / "data/raw/chemicalpert_DE5000/metainfo.json")
    return cfg


def candidate_cfg(anchor_cfg: Config, *, adapter_kind: str = "condition_delta", lowrank_rank: int = 32) -> Config:
    cfg = deepcopy(anchor_cfg)
    adapter_kind = str(adapter_kind or "condition_delta").strip().lower()
    if adapter_kind == "condition_delta":
        cfg.condition_delta_head_use_in_model = True
        cfg.condition_lowrank_residual_use_in_model = False
        cfg.finetune_trainable_scope = "condition_prior_adapter"
    elif adapter_kind == "lowrank_residual":
        cfg.condition_delta_head_use_in_model = False
        cfg.condition_lowrank_residual_use_in_model = True
        cfg.condition_lowrank_residual_rank = int(lowrank_rank)
        cfg.finetune_trainable_scope = "condition_lowrank_residual_adapter"
    else:
        raise ValueError(f"unknown adapter_kind: {adapter_kind}")
    cfg.condition_delta_head_loss_weight = 0.0
    cfg.additive_condition_delta_loss_weight = 0.0
    cfg.condition_prior_additive_delta_loss_weight = 0.0
    cfg.trackc_routed_distill_loss_weight = 0.0
    cfg.trackc_routed_endpoint_loss_weight = 0.0
    cfg.anchor_replay_loss_weight = 1.0
    cfg.anchor_replay_condition_filter = "all"
    cfg.anchor_replay_checkpoint = str(ANCHOR_CKPT)
    cfg.anchor_replay_checkpoint_use_ema = True
    return cfg


def load_raw_then_ema(path: Path, model: torch.nn.Module, cfg: Config, device: torch.device) -> dict[str, Any]:
    missing, unexpected, skipped = load_model_weights_only(path, model, device, strict=False, prefer_ema=False)
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    ema_applied = False
    if isinstance(ckpt, dict) and checkpoint_ema_is_active(ckpt, cfg):
        ema = ModelEMA(
            model,
            decay=float(getattr(cfg, "ema_decay", 0.999)),
            update_after=int(getattr(cfg, "ema_update_after", 0)),
            update_every=int(getattr(cfg, "ema_update_every", 1)),
            device=device,
        )
        ema.load_state_dict(ckpt["ema"], strict=False)
        ema.copy_to(model)
        ema_applied = True
    return {
        "missing": missing,
        "unexpected": unexpected,
        "skipped_shape_mismatch": skipped,
        "ema_applied": ema_applied,
    }


def trainable_items(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def flat_params(items: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for _, p in items])


def set_flat_params(items: list[tuple[str, torch.nn.Parameter]], vec: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for _, p in items:
            n = p.numel()
            p.copy_(vec[offset : offset + n].reshape_as(p))
            offset += n


def grad_vector(loss: torch.Tensor, items: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    params = [p for _, p in items]
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    chunks = []
    for (_, p), g in zip(items, grads):
        chunks.append(torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1))
    return torch.cat(chunks)


def pcgrad(task_grad: torch.Tensor, anchor_grad: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    anchor_norm_sq = float(torch.dot(anchor_grad, anchor_grad).item())
    dot = float(torch.dot(task_grad, anchor_grad).item())
    if anchor_norm_sq <= EPS:
        return task_grad.clone(), {
            "dot_before": dot,
            "dot_after": dot,
            "anchor_norm": math.sqrt(anchor_norm_sq),
            "projection_coeff": 0.0,
        }
    coeff = min(0.0, dot / anchor_norm_sq)
    projected = task_grad - coeff * anchor_grad
    return projected, {
        "dot_before": dot,
        "dot_after": float(torch.dot(projected, anchor_grad).item()),
        "anchor_norm": math.sqrt(anchor_norm_sq),
        "projection_coeff": coeff,
    }


def make_dataset(cfg: Config, *, seed: int, batch_size: int) -> CrossDatasetFMDataset:
    split = json.loads(SAFE_SPLIT.read_text(encoding="utf-8"))
    return CrossDatasetFMDataset(
        cfg.data_dir,
        split,
        batch_size=batch_size,
        seed=seed,
        mode="train",
        min_cells=16,
        ds_alpha=1.0,
        scale_noise=0.0,
        condition_visit_cap=1,
        use_pert_condition=True,
        max_pert_genes=int(getattr(cfg, "max_pert_genes", 16)),
        gene_embedding_cache_dir=str(getattr(cfg, "pert_gene_emb_cache_dir", "")),
        biflow_dir=str(getattr(cfg, "biflow_dir", "")),
        use_h5ad_pert_metadata=bool(getattr(cfg, "use_h5ad_pert_metadata", False)),
        pert_metainfo_path=str(getattr(cfg, "pert_metainfo_path", "")),
        chem_emb_source_dir=str(getattr(cfg, "chem_emb_source_dir", "")),
        chem_obs_column=str(getattr(cfg, "chem_obs_column", "")),
        drug_emb_cache_dir=str(getattr(cfg, "drug_emb_cache_dir", "")),
        max_chem_keys=int(getattr(cfg, "max_chem_keys", 4)),
        chemical_metainfo_path=str(getattr(cfg, "chemical_metainfo_path", "")),
        chem_fallback_embed_dim=int(getattr(cfg, "chem_fallback_embed_dim", 512)),
        latent_backbone="xverse",
        pert_chem_enabled=bool(getattr(cfg, "pert_chem_enabled", True)),
        perturbation_family_filter="all",
        silent=True,
    )


def to_device_batch(raw: tuple, sampler: OTPlanSampler, device: torch.device, *, idx: int) -> dict[str, Any]:
    src, gt, ds_name, cond, pb = raw
    src_paired, gt_paired = sampler.sample_plan(src.float(), gt.float())
    bsz = int(src_paired.shape[0])
    t = torch.linspace(0.05, 0.95, bsz, dtype=torch.float32)
    if idx % 2:
        t = torch.flip(t, dims=[0])
    return {
        "src": src_paired.to(device=device, dtype=torch.float32, non_blocking=True).contiguous(),
        "gt": gt_paired.to(device=device, dtype=torch.float32, non_blocking=True).contiguous(),
        "t": t.to(device=device, dtype=torch.float32, non_blocking=True).contiguous(),
        "pb": _pert_to_device(pb, device),
        "dataset": str(ds_name),
        "condition": str(cond),
    }


def next_batch(iterator, dataset: CrossDatasetFMDataset, sampler: OTPlanSampler, device: torch.device, *, idx: int):
    try:
        raw = next(iterator)
    except StopIteration:
        iterator = iter(dataset)
        raw = next(iterator)
    return iterator, to_device_batch(raw, sampler, device, idx=idx)


def batch_losses(
    *,
    model: torch.nn.Module,
    anchor: torch.nn.Module,
    batch: dict[str, Any],
    path: CondOTPath,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src = batch["src"]
    gt = batch["gt"]
    t = batch["t"]
    pb = batch["pb"]
    ps = path.sample(x_0=src, x_1=gt, t=t)
    v_pred = _model_latent_velocity(model, ps.x_t, ps.t, src, pb)
    task = F.mse_loss(v_pred.float(), ps.dx_t.float())
    x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
    with torch.no_grad():
        anchor_v = _model_latent_velocity(anchor, ps.x_t, ps.t, src, pb)
        anchor_x1 = ps.x_t + anchor_v * (1.0 - t).unsqueeze(-1)
    anchor_loss = F.mse_loss(x1_hat.float(), anchor_x1.float())
    row_l2 = torch.linalg.norm((x1_hat - anchor_x1).float(), dim=1)
    return task, anchor_loss, row_l2.mean(), row_l2


def evaluate(model: torch.nn.Module, anchor: torch.nn.Module, batch: dict[str, Any], path: CondOTPath) -> dict[str, float]:
    with torch.no_grad():
        task, anchor_loss, footprint, row_l2 = batch_losses(model=model, anchor=anchor, batch=batch, path=path)
    return {
        "task_loss": float(task.item()),
        "anchor_loss": float(anchor_loss.item()),
        "footprint_mean_l2": float(footprint.item()),
        "material_row_frac": float((row_l2 > 1e-6).float().mean().item()),
    }


def csv_writer(path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return handle, writer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-attempts", type=int, default=80)
    ap.add_argument("--max-accepted", type=int, default=40)
    ap.add_argument("--step-grid", type=str, default=STEP_GRID_DEFAULT)
    ap.add_argument("--anchor-threshold", type=float, default=1e-6)
    ap.add_argument("--min-task-delta", type=float, default=1e-10)
    ap.add_argument("--min-footprint", type=float, default=1e-7)
    ap.add_argument("--max-reject-streak", type=int, default=30)
    ap.add_argument("--adapter-kind", choices=["condition_delta", "lowrank_residual"], default="condition_delta")
    ap.add_argument("--lowrank-rank", type=int, default=32)
    args = ap.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_csv = args.save_dir / "train_metrics.csv"
    summary_json = args.save_dir / "summary.json"
    ckpt_path = args.save_dir / "latest.pt"
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    anchor_cfg = cfg_from_checkpoint(ANCHOR_CKPT, batch_size=int(args.batch_size), seed=int(args.seed))
    cfg = candidate_cfg(anchor_cfg, adapter_kind=args.adapter_kind, lowrank_rank=int(args.lowrank_rank))
    anchor = build_model(anchor_cfg, device)
    candidate = build_model(cfg, device)
    anchor_load = load_raw_then_ema(ANCHOR_CKPT, anchor, anchor_cfg, device)
    cand_load = load_raw_then_ema(ANCHOR_CKPT, candidate, cfg, device)
    for p in anchor.parameters():
        p.requires_grad = False
    anchor.eval()
    candidate.eval()
    apply_finetune_freeze(candidate, cfg)
    trainable = trainable_items(candidate)
    trainable_names = [name for name, _ in trainable]
    if str(args.adapter_kind) == "lowrank_residual":
        allowed_trainable = all(
            name.startswith("condition_lowrank_residual_down.")
            or name.startswith("condition_lowrank_residual_up.")
            for name in trainable_names
        )
        allowed_missing = all(
            key == "condition_delta_prior_gene_allowlist"
            or key.startswith("condition_lowrank_residual_down.")
            or key.startswith("condition_lowrank_residual_up.")
            for key in cand_load["missing"]
        )
    else:
        allowed_trainable = all(
            name.startswith("condition_delta_head.") or name.startswith("condition_delta_to_c.")
            for name in trainable_names
        )
        allowed_missing = all(
            key == "condition_delta_prior_gene_allowlist"
            or key.startswith("condition_delta_head.")
            or key.startswith("condition_delta_to_c.")
            for key in cand_load["missing"]
        )
    if not allowed_trainable or not allowed_missing or cand_load["unexpected"] or cand_load["skipped_shape_mismatch"]:
        raise RuntimeError(
            "provenance/scope failure: "
            f"allowed_trainable={allowed_trainable} allowed_missing={allowed_missing} "
            f"unexpected={cand_load['unexpected']} skipped={cand_load['skipped_shape_mismatch']}"
        )

    dataset = make_dataset(cfg, seed=int(args.seed), batch_size=int(args.batch_size))
    iterator = iter(dataset)
    sampler = OTPlanSampler(method="exact", num_threads=4)
    path = CondOTPath()
    steps = parse_step_grid(args.step_grid)

    # Init no-op check on two fresh train batches.
    noop_drifts: list[float] = []
    for i in range(2):
        iterator, batch = next_batch(iterator, dataset, sampler, device, idx=i)
        with torch.no_grad():
            src = batch["src"]
            gt = batch["gt"]
            t = batch["t"]
            pb = batch["pb"]
            ps = path.sample(src, gt, t)
            cv = _model_latent_velocity(candidate, ps.x_t, ps.t, src, pb)
            av = _model_latent_velocity(anchor, ps.x_t, ps.t, src, pb)
            noop_drifts.append(float((cv - av).abs().max().item()))
    max_noop_drift = max(noop_drifts) if noop_drifts else 999.0
    if max_noop_drift > 1e-6:
        raise RuntimeError(f"initial no-op drift too large: {max_noop_drift}")

    fields = [
        "attempt",
        "accepted_step",
        "accepted",
        "task_dataset",
        "task_condition",
        "noharm_dataset",
        "noharm_condition",
        "task_grad_norm",
        "anchor_grad0_norm",
        "base_task_loss",
        "base_noharm_anchor_loss",
        "best_step",
        "proj_task_delta",
        "proj_anchor_delta",
        "proj_footprint_mean_l2",
        "proj_material_row_frac",
        "task_retention_vs_unprojected",
        "projection_reduced_anchor_delta_frac",
        "reject_reason",
    ]
    handle, writer = csv_writer(log_csv, fields)
    accepted = 0
    reject_streak = 0
    attempts = 0
    try:
        for attempt in range(int(args.max_attempts)):
            attempts = attempt + 1
            iterator, task_batch = next_batch(iterator, dataset, sampler, device, idx=2 * attempt)
            iterator, noharm_batch = next_batch(iterator, dataset, sampler, device, idx=2 * attempt + 1)
            p0 = flat_params(trainable).clone()
            base_task = evaluate(candidate, anchor, task_batch, path)
            base_noharm = evaluate(candidate, anchor, noharm_batch, path)
            task_loss, _, _, _ = batch_losses(model=candidate, anchor=anchor, batch=task_batch, path=path)
            task_grad = grad_vector(task_loss, trainable)
            task_grad_norm = float(torch.linalg.norm(task_grad).item())
            _, anchor_loss0, _, _ = batch_losses(model=candidate, anchor=anchor, batch=noharm_batch, path=path)
            anchor_grad0 = grad_vector(anchor_loss0, trainable)
            anchor_grad0_norm = float(torch.linalg.norm(anchor_grad0).item())
            best: dict[str, Any] | None = None
            for step in steps:
                set_flat_params(trainable, p0 - float(step) * task_grad)
                unproj_task = evaluate(candidate, anchor, task_batch, path)
                unproj_noharm = evaluate(candidate, anchor, noharm_batch, path)
                _, probe_anchor_loss, _, _ = batch_losses(model=candidate, anchor=anchor, batch=noharm_batch, path=path)
                probe_anchor_grad = grad_vector(probe_anchor_loss, trainable)
                set_flat_params(trainable, p0)
                proj_grad, proj_stats = pcgrad(task_grad, probe_anchor_grad)
                set_flat_params(trainable, p0 - float(step) * proj_grad)
                proj_task = evaluate(candidate, anchor, task_batch, path)
                proj_noharm = evaluate(candidate, anchor, noharm_batch, path)
                set_flat_params(trainable, p0)
                unproj_task_delta = unproj_task["task_loss"] - base_task["task_loss"]
                proj_task_delta = proj_task["task_loss"] - base_task["task_loss"]
                unproj_anchor_delta = unproj_noharm["anchor_loss"] - base_noharm["anchor_loss"]
                proj_anchor_delta = proj_noharm["anchor_loss"] - base_noharm["anchor_loss"]
                retention = (
                    abs(proj_task_delta) / max(abs(unproj_task_delta), 1e-12)
                    if proj_task_delta < 0 and unproj_task_delta < 0
                    else 0.0
                )
                reduction = (
                    1.0 - (proj_anchor_delta / max(unproj_anchor_delta, 1e-12))
                    if unproj_anchor_delta > 0
                    else 0.0
                )
                candidate_ok = (
                    proj_task_delta < -float(args.min_task_delta)
                    and proj_anchor_delta <= float(args.anchor_threshold)
                    and (reduction >= 0.50 or proj_anchor_delta <= float(args.anchor_threshold) * 0.01)
                    and retention >= 0.20
                    and proj_noharm["footprint_mean_l2"] > float(args.min_footprint)
                    and proj_noharm["material_row_frac"] >= 0.15
                )
                row = {
                    "best_step": step,
                    "proj_task_delta": proj_task_delta,
                    "proj_anchor_delta": proj_anchor_delta,
                    "proj_footprint_mean_l2": proj_noharm["footprint_mean_l2"],
                    "proj_material_row_frac": proj_noharm["material_row_frac"],
                    "task_retention_vs_unprojected": retention,
                    "projection_reduced_anchor_delta_frac": reduction,
                    "probe_anchor_grad_norm": proj_stats["anchor_norm"],
                }
                if candidate_ok and (
                    best is None
                    or (row["proj_anchor_delta"], row["proj_task_delta"]) < (best["proj_anchor_delta"], best["proj_task_delta"])
                ):
                    best = row

            out = {
                "attempt": attempt,
                "accepted_step": accepted,
                "accepted": best is not None,
                "task_dataset": task_batch["dataset"],
                "task_condition": task_batch["condition"],
                "noharm_dataset": noharm_batch["dataset"],
                "noharm_condition": noharm_batch["condition"],
                "task_grad_norm": task_grad_norm,
                "anchor_grad0_norm": anchor_grad0_norm,
                "base_task_loss": base_task["task_loss"],
                "base_noharm_anchor_loss": base_noharm["anchor_loss"],
                "reject_reason": "" if best is not None else "no_step_passed_line_search",
            }
            if best is None:
                set_flat_params(trainable, p0)
                reject_streak += 1
            else:
                # Recompute projected vector for the selected step and apply it.
                selected = float(best["best_step"])
                set_flat_params(trainable, p0 - selected * task_grad)
                _, probe_anchor_loss, _, _ = batch_losses(model=candidate, anchor=anchor, batch=noharm_batch, path=path)
                probe_anchor_grad = grad_vector(probe_anchor_loss, trainable)
                set_flat_params(trainable, p0)
                proj_grad, _ = pcgrad(task_grad, probe_anchor_grad)
                set_flat_params(trainable, p0 - selected * proj_grad)
                accepted += 1
                reject_streak = 0
                out.update(best)
            writer.writerow({field: out.get(field, "") for field in fields})
            handle.flush()
            if accepted >= int(args.max_accepted):
                break
            if reject_streak >= int(args.max_reject_streak):
                break
    finally:
        handle.close()
        dataset.close()

    torch.save(
        {
            "step": accepted,
            "model": candidate.state_dict(),
            "optimizer": {"type": "lookahead_trust_region_line_search", "step_grid": steps},
            "best_score": float("nan"),
            "config": dataclasses.asdict(cfg),
            "smoke_metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
                "anchor_checkpoint": str(ANCHOR_CKPT),
                "safe_split": str(SAFE_SPLIT),
                "attempts": attempts,
                "accepted": accepted,
                "max_noop_drift": max_noop_drift,
            "anchor_load": anchor_load,
            "candidate_load": cand_load,
            "trainable_names": trainable_names,
            "adapter_kind": str(args.adapter_kind),
            "lowrank_rank": int(args.lowrank_rank),
        },
        },
        ckpt_path,
    )
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": "finished" if accepted > 0 else "failed_no_accepted_steps",
        "attempts": attempts,
        "accepted": accepted,
        "max_noop_drift": max_noop_drift,
        "checkpoint": str(ckpt_path),
        "train_metrics": str(log_csv),
        "anchor_checkpoint": str(ANCHOR_CKPT),
        "safe_split": str(SAFE_SPLIT),
        "trainable_names": trainable_names,
        "adapter_kind": str(args.adapter_kind),
        "lowrank_rank": int(args.lowrank_rank),
        "step_grid": steps,
        "anchor_threshold": float(args.anchor_threshold),
        "min_task_delta": float(args.min_task_delta),
        "min_footprint": float(args.min_footprint),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if accepted > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
