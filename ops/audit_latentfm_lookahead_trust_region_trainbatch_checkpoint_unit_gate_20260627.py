#!/usr/bin/env python3
"""Lookahead/trust-region adapter train-batch checkpoint unit gate.

CPU-only, short-running gate. This is the real-batch follow-up to the
frozen-means lookahead/trust-region adapter unit pass. It uses the actual
xverse anchor checkpoint, actual train-only split, real perturbation
conditioning, CPU exact OT pairing, and the existing zero-initialized
``condition_delta_to_c`` bridge.

It does not train, save checkpoints, evaluate canonical multi, read Track C
query, or use GPU. Passing this gate only authorizes external audit and design
of a tiny bounded training smoke.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

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


REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "lookahead_trust_region_trainbatch_checkpoint_unit_gate_20260627"
OUT_ROWS = OUT_DIR / "lookahead_trust_region_trainbatch_rows.csv"
OUT_JSON = REPORTS / "latentfm_lookahead_trust_region_trainbatch_checkpoint_unit_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_LOOKAHEAD_TRUST_REGION_TRAINBATCH_CHECKPOINT_UNIT_GATE_20260627.md"

ANCHOR_CKPT = (
    COUPLEDFM
    / "output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
N_BATCHES = 6
BATCH_SIZE = 16
STEP_SIZES = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
RNG_SEED = 20260627
EPS = 1e-12


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def cfg_from_checkpoint(path: Path) -> Config:
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
    cfg.batch_size = BATCH_SIZE
    cfg.min_cells = 16
    cfg.scale_noise = 0.0
    cfg.seed = RNG_SEED
    cfg.use_amp = False
    cfg.mmd_ode_steps = 0
    cfg.use_pert_condition = True
    cfg.pert_gene_emb_cache_dir = str(ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene")
    cfg.pert_metainfo_path = str(COUPLEDFM / "data/raw/genepert_DE5000/metainfo.json")
    cfg.chemical_metainfo_path = str(COUPLEDFM / "data/raw/chemicalpert_DE5000/metainfo.json")
    return cfg


def candidate_cfg(anchor_cfg: Config) -> Config:
    cfg = deepcopy(anchor_cfg)
    cfg.condition_delta_head_use_in_model = True
    cfg.condition_delta_head_loss_weight = 0.0
    cfg.additive_condition_delta_loss_weight = 0.0
    cfg.condition_prior_additive_delta_loss_weight = 0.0
    cfg.trackc_routed_distill_loss_weight = 0.0
    cfg.trackc_routed_endpoint_loss_weight = 0.0
    cfg.finetune_trainable_scope = "condition_prior_adapter"
    cfg.anchor_replay_loss_weight = 1.0
    cfg.anchor_replay_condition_filter = "all"
    cfg.anchor_replay_checkpoint = str(ANCHOR_CKPT)
    cfg.anchor_replay_checkpoint_use_ema = True
    return cfg


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


def tensor_hash(items: list[tuple[str, torch.nn.Parameter]]) -> str:
    h = hashlib.sha256()
    for name, p in items:
        h.update(name.encode("utf-8"))
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def nontrainable_hash(model: torch.nn.Module) -> str:
    return tensor_hash([(name, p) for name, p in model.named_parameters() if not p.requires_grad])


def load_raw_then_ema(
    path: Path,
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device,
    *,
    strict: bool,
) -> dict[str, Any]:
    missing, unexpected, skipped = load_model_weights_only(
        path, model, device, strict=strict, prefer_ema=False
    )
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


def load_models() -> tuple[torch.nn.Module, torch.nn.Module, Config, dict[str, Any]]:
    device = torch.device("cpu")
    anchor_cfg = cfg_from_checkpoint(ANCHOR_CKPT)
    cand_cfg = candidate_cfg(anchor_cfg)
    anchor = build_model(anchor_cfg, device)
    candidate = build_model(cand_cfg, device)
    anchor_load = load_raw_then_ema(ANCHOR_CKPT, anchor, anchor_cfg, device, strict=False)
    cand_load = load_raw_then_ema(ANCHOR_CKPT, candidate, cand_cfg, device, strict=False)
    for p in anchor.parameters():
        p.requires_grad = False
    anchor.eval()
    candidate.eval()
    apply_finetune_freeze(candidate, cand_cfg)
    meta = {
        "anchor_missing": anchor_load["missing"],
        "anchor_unexpected": anchor_load["unexpected"],
        "anchor_skipped_shape_mismatch": anchor_load["skipped_shape_mismatch"],
        "anchor_ema_applied": anchor_load["ema_applied"],
        "candidate_missing": cand_load["missing"],
        "candidate_unexpected": cand_load["unexpected"],
        "candidate_skipped_shape_mismatch": cand_load["skipped_shape_mismatch"],
        "candidate_ema_applied": cand_load["ema_applied"],
        "trainable_names": [name for name, _ in trainable_items(candidate)],
    }
    return anchor, candidate, cand_cfg, meta


def collect_batches(cfg: Config, n_batches: int) -> list[dict[str, Any]]:
    split = json.loads(SAFE_SPLIT.read_text(encoding="utf-8"))
    ds = CrossDatasetFMDataset(
        cfg.data_dir,
        split,
        batch_size=BATCH_SIZE,
        seed=RNG_SEED,
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
    sampler = OTPlanSampler(method="exact", num_threads=4)
    out: list[dict[str, Any]] = []
    rng_state = np.random.get_state()
    np.random.seed(RNG_SEED)
    try:
        for idx, (src, gt, ds_name, cond, pb) in enumerate(ds):
            if idx >= n_batches:
                break
            src_paired, gt_paired = sampler.sample_plan(src.float(), gt.float())
            # Deterministic, non-degenerate path times.
            t = torch.linspace(0.05, 0.95, src_paired.shape[0], dtype=torch.float32)
            if idx % 2:
                t = torch.flip(t, dims=[0])
            out.append(
                {
                    "src": src_paired.detach().contiguous(),
                    "gt": gt_paired.detach().contiguous(),
                    "t": t.contiguous(),
                    "pb": _pert_to_device(pb, torch.device("cpu")),
                    "dataset": str(ds_name),
                    "condition": str(cond),
                }
            )
    finally:
        np.random.set_state(rng_state)
        ds.close()
    if len(out) < n_batches:
        raise RuntimeError(f"Collected only {len(out)} train batches; expected {n_batches}")
    return out


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
    footprint = row_l2.mean()
    return task, anchor_loss, footprint, row_l2


def evaluate(
    *,
    model: torch.nn.Module,
    anchor: torch.nn.Module,
    batch: dict[str, Any],
    path: CondOTPath,
) -> dict[str, float]:
    with torch.no_grad():
        task, anchor_loss, footprint, row_l2 = batch_losses(
            model=model,
            anchor=anchor,
            batch=batch,
            path=path,
        )
    return {
        "task_loss": float(task.item()),
        "anchor_loss": float(anchor_loss.item()),
        "footprint_mean_l2": float(footprint.item()),
        "material_row_frac": float((row_l2 > 1e-6).float().mean().item()),
    }


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    anchor, candidate, cfg, load_meta = load_models()
    trainable = trainable_items(candidate)
    p0 = flat_params(trainable).clone()
    nontrain_hash0 = nontrainable_hash(candidate)
    batches = collect_batches(cfg, N_BATCHES)
    path = CondOTPath()

    allowed_trainable = all(
        name.startswith("condition_delta_head.") or name.startswith("condition_delta_to_c.")
        for name, _ in trainable
    )
    allowed_missing = all(
        key == "condition_delta_prior_gene_allowlist"
        or key.startswith("condition_delta_head.")
        or key.startswith("condition_delta_to_c.")
        for key in load_meta["candidate_missing"]
    )
    no_op_drifts: list[float] = []
    for batch in batches:
        with torch.no_grad():
            src = batch["src"]
            gt = batch["gt"]
            t = batch["t"]
            pb = batch["pb"]
            ps = path.sample(src, gt, t)
            cv = _model_latent_velocity(candidate, ps.x_t, ps.t, src, pb)
            av = _model_latent_velocity(anchor, ps.x_t, ps.t, src, pb)
            no_op_drifts.append(float((cv - av).abs().max().item()))
    max_noop_drift = max(no_op_drifts) if no_op_drifts else 999.0

    rows: list[dict[str, Any]] = []
    best_by_pair: list[dict[str, Any] | None] = []
    for pair_idx in range(N_BATCHES - 1):
        task_batch = batches[pair_idx]
        noharm_batch = batches[pair_idx + 1]
        set_flat_params(trainable, p0)
        base_task = evaluate(model=candidate, anchor=anchor, batch=task_batch, path=path)
        base_noharm = evaluate(model=candidate, anchor=anchor, batch=noharm_batch, path=path)
        task_loss, _, _, _ = batch_losses(model=candidate, anchor=anchor, batch=task_batch, path=path)
        task_grad = grad_vector(task_loss, trainable)
        task_grad_norm = float(torch.linalg.norm(task_grad).item())
        _, anchor_loss0, _, _ = batch_losses(model=candidate, anchor=anchor, batch=noharm_batch, path=path)
        anchor_grad0 = grad_vector(anchor_loss0, trainable)
        anchor_grad0_norm = float(torch.linalg.norm(anchor_grad0).item())
        best: dict[str, Any] | None = None
        for step in STEP_SIZES:
            set_flat_params(trainable, p0 - float(step) * task_grad)
            unproj_task = evaluate(model=candidate, anchor=anchor, batch=task_batch, path=path)
            unproj_noharm = evaluate(model=candidate, anchor=anchor, batch=noharm_batch, path=path)
            _, anchor_probe_loss, _, _ = batch_losses(
                model=candidate,
                anchor=anchor,
                batch=noharm_batch,
                path=path,
            )
            anchor_grad_probe = grad_vector(anchor_probe_loss, trainable)
            set_flat_params(trainable, p0)
            proj_grad, proj_stats = pcgrad(task_grad, anchor_grad_probe)
            set_flat_params(trainable, p0 - float(step) * proj_grad)
            proj_task = evaluate(model=candidate, anchor=anchor, batch=task_batch, path=path)
            proj_noharm = evaluate(model=candidate, anchor=anchor, batch=noharm_batch, path=path)
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
            row = {
                "pair_idx": pair_idx,
                "task_dataset": task_batch["dataset"],
                "task_condition": task_batch["condition"],
                "noharm_dataset": noharm_batch["dataset"],
                "noharm_condition": noharm_batch["condition"],
                "step": step,
                "task_grad_norm": task_grad_norm,
                "anchor_grad0_norm": anchor_grad0_norm,
                "probe_anchor_grad_norm": proj_stats["anchor_norm"],
                "probe_dot_before": proj_stats["dot_before"],
                "probe_dot_after": proj_stats["dot_after"],
                "probe_projection_coeff": proj_stats["projection_coeff"],
                "base_task_loss": base_task["task_loss"],
                "base_noharm_anchor_loss": base_noharm["anchor_loss"],
                "unproj_task_delta": unproj_task_delta,
                "unproj_anchor_delta": unproj_anchor_delta,
                "unproj_footprint_mean_l2": unproj_noharm["footprint_mean_l2"],
                "proj_task_delta": proj_task_delta,
                "proj_anchor_delta": proj_anchor_delta,
                "proj_footprint_mean_l2": proj_noharm["footprint_mean_l2"],
                "proj_material_row_frac": proj_noharm["material_row_frac"],
                "task_retention_vs_unprojected": retention,
                "projection_reduced_anchor_delta_frac": (
                    1.0 - (proj_anchor_delta / max(unproj_anchor_delta, 1e-12))
                    if unproj_anchor_delta > 0
                    else 0.0
                ),
            }
            rows.append(row)
            candidate_ok = (
                row["proj_task_delta"] < -1e-9
                and row["proj_anchor_delta"] <= 1e-6
                and (
                    row["projection_reduced_anchor_delta_frac"] >= 0.50
                    or row["proj_anchor_delta"] <= 1e-8
                )
                and row["task_retention_vs_unprojected"] >= 0.20
                and row["proj_footprint_mean_l2"] > 1e-7
                and row["proj_material_row_frac"] >= 0.15
            )
            if candidate_ok and (
                best is None
                or (row["proj_anchor_delta"], -row["proj_task_delta"]) < (best["proj_anchor_delta"], -best["proj_task_delta"])
            ):
                best = row
        best_by_pair.append(best)

    set_flat_params(trainable, p0)
    nontrain_hash1 = nontrainable_hash(candidate)
    passed_pairs = [row for row in best_by_pair if row is not None]
    reasons: list[str] = []
    if not allowed_trainable:
        reasons.append("trainable_scope_includes_disallowed_base_params")
    if not allowed_missing:
        reasons.append("candidate_missing_keys_not_limited_to_new_adapter")
    if load_meta["candidate_unexpected"] or load_meta["candidate_skipped_shape_mismatch"]:
        reasons.append("candidate_checkpoint_load_unexpected_or_shape_skipped")
    if max_noop_drift > 1e-6:
        reasons.append("candidate_anchor_noop_drift_above_1e-6")
    if nontrain_hash0 != nontrain_hash1:
        reasons.append("nontrainable_base_param_hash_changed")
    if len(passed_pairs) < len(best_by_pair):
        reasons.append("one_or_more_trainbatch_pairs_failed")
    if passed_pairs:
        if min(float(row["task_retention_vs_unprojected"]) for row in passed_pairs) < 0.20:
            reasons.append("min_task_retention_below_0p20")
        if max(float(row["proj_anchor_delta"]) for row in passed_pairs) > 1e-6:
            reasons.append("max_projected_anchor_delta_above_1e-6")
    status = (
        "lookahead_trust_region_trainbatch_checkpoint_unit_gate_pass_external_audit_only_no_gpu"
        if not reasons
        else "lookahead_trust_region_trainbatch_checkpoint_unit_gate_fail_no_gpu"
    )

    write_csv(
        OUT_ROWS,
        rows,
        [
            "pair_idx",
            "task_dataset",
            "task_condition",
            "noharm_dataset",
            "noharm_condition",
            "step",
            "task_grad_norm",
            "anchor_grad0_norm",
            "probe_anchor_grad_norm",
            "probe_dot_before",
            "probe_dot_after",
            "probe_projection_coeff",
            "base_task_loss",
            "base_noharm_anchor_loss",
            "unproj_task_delta",
            "unproj_anchor_delta",
            "unproj_footprint_mean_l2",
            "proj_task_delta",
            "proj_anchor_delta",
            "proj_footprint_mean_l2",
            "proj_material_row_frac",
            "task_retention_vs_unprojected",
            "projection_reduced_anchor_delta_frac",
        ],
    )
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "anchor_checkpoint": str(ANCHOR_CKPT),
        "safe_split": str(SAFE_SPLIT),
        "n_batches": N_BATCHES,
        "n_pairs": len(best_by_pair),
        "n_passed_pairs": len(passed_pairs),
        "max_noop_drift": max_noop_drift,
        "allowed_trainable": allowed_trainable,
        "allowed_missing": allowed_missing,
        "nontrainable_hash_unchanged": nontrain_hash0 == nontrain_hash1,
        "load_meta": load_meta,
        "best_by_pair": best_by_pair,
        "outputs": {
            "rows": str(OUT_ROWS),
            "report": str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Lookahead Trust-Region Train-Batch Checkpoint Unit Gate",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M %Z')}`",
        "",
        f"Status: `{status}`",
        "",
        "## Scope",
        "",
        "CPU-only real-batch/checkpoint unit gate. Uses safe train-only split, "
        "actual xverse anchor checkpoint loaded with EMA provenance, CPU exact "
        "OT pairing, and the existing zero-initialized condition-delta bridge. "
        "It does not train, save a checkpoint, read canonical multi for "
        "selection, read Track C query, or use GPU.",
        "",
        "## Gate Summary",
        "",
        f"- train batches / pairs: `{N_BATCHES}` / `{len(best_by_pair)}`",
        f"- passed pairs: `{len(passed_pairs)}`",
        f"- max no-op velocity drift: `{max_noop_drift:.6g}`",
        f"- allowed trainable scope: `{allowed_trainable}`",
        f"- allowed missing keys: `{allowed_missing}`",
        f"- nontrainable base hash unchanged: `{nontrain_hash0 == nontrain_hash1}`",
        f"- fail/pass reasons: `{reasons}`",
        "",
        "## Best Pair Steps",
        "",
        "| pair | step | task delta | anchor delta | footprint | retention | anchor reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(best_by_pair):
        if row is None:
            lines.append(f"| {idx} |  |  |  |  |  |  |")
        else:
            lines.append(
                f"| {idx} | `{row['step']}` | `{row['proj_task_delta']:.6g}` | "
                f"`{row['proj_anchor_delta']:.6g}` | `{row['proj_footprint_mean_l2']:.6g}` | "
                f"`{row['task_retention_vs_unprojected']:.6g}` | "
                f"`{row['projection_reduced_anchor_delta_frac']:.6g}` |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Passing this gate does not authorize GPU by itself. It only supports "
            "external audit and design of a tiny capped detached training smoke "
            "with RUN_STATUS, strict internal no-harm gates, and frozen-route "
            "checkpoint-selection rules.",
            "",
            "## Outputs",
            "",
            f"- Rows: `{OUT_ROWS}`",
            f"- JSON: `{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "reasons", "n_pairs", "n_passed_pairs", "max_noop_drift"]}, indent=2))


if __name__ == "__main__":
    main()
