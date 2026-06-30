#!/usr/bin/env python3
"""Gradient-conflict preflight gate for a possible no-harm projection branch.

The gate samples a small number of train-only batches from the xverse
general-exposure v2 split and computes parameter-gradient cosine between the FM
MSE objective and the MMD endpoint objective on the anchor checkpoint. It does
not train, does not read canonical metrics, and does not touch Track C query.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
import sys

if str(COUPLED) not in sys.path:
    sys.path.insert(0, str(COUPLED))

from model.latent.config import Config
from model.latent.dataset import CrossDatasetFMDataset
from model.latent.fm_ot import CondOTPath, median_sigmas, mmd2_biased
from model.latent.train import (
    _cross_dataset_kw,
    _model_latent_velocity,
    _pert_to_device,
    build_model,
    load_model_weights_only,
)
from model.utils.train.time_sampling import sample_t_torch


SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json"
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
ANCHOR = ROOT / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
GENE_CACHE = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
OUT_JSON = ROOT / "reports/latentfm_gradient_conflict_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_GRADIENT_CONFLICT_GATE_20260624.md"
RISK_DATASETS = {
    "Nadig_hepg2",
    "Nadig_jurket",
    "NormanWeissman2019_filtered",
    "ReplogleWeissman2022_K562_gwps",
    "Replogle_RPE1essential",
    "TianActivation",
}


def grad_vector(model: torch.nn.Module) -> torch.Tensor:
    parts = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        # Exclude very large frozen/mostly lookup-style perturbation tables from
        # the diagnostic so cosine reflects shared FM body conflict.
        if "gene_table" in name or "condition_delta_prior_gene_allowlist" in name:
            continue
        parts.append(p.grad.detach().float().flatten().cpu())
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 0 or not math.isfinite(denom):
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def grad_cosine_for_batch(
    model: torch.nn.Module,
    path: CondOTPath,
    cfg: Config,
    device: torch.device,
    src: torch.Tensor,
    gt: torch.Tensor,
    pb: tuple | None,
) -> dict[str, float]:
    src = src.to(device)
    gt = gt.to(device)
    pb_dev = _pert_to_device(pb, device)
    bsz = src.size(0)

    torch.manual_seed(12345)
    t = sample_t_torch(bsz, device, mode=cfg.time_sampling)
    ps = path.sample(x_0=src, x_1=gt, t=t)

    def forward_losses() -> tuple[torch.Tensor, torch.Tensor]:
        v_pred = _model_latent_velocity(model, ps.x_t, ps.t, src, pb_dev)
        mse = F.mse_loss(v_pred, ps.dx_t)
        x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
        sigmas = median_sigmas(gt.float())
        mmd = mmd2_biased(x1_hat.float(), gt.float(), sigmas)
        return mse, mmd

    model.zero_grad(set_to_none=True)
    mse, _ = forward_losses()
    mse.backward()
    g_mse = grad_vector(model)

    model.zero_grad(set_to_none=True)
    _, mmd = forward_losses()
    mmd.backward()
    g_mmd = grad_vector(model)
    model.zero_grad(set_to_none=True)

    return {
        "grad_cosine_mse_mmd": cosine(g_mse, g_mmd),
        "mse_grad_norm": float(g_mse.norm().item()),
        "mmd_grad_norm": float(g_mmd.norm().item()),
        "mse": float(mse.detach().cpu().item()),
        "mmd": float(mmd.detach().cpu().item()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["grad_cosine_mse_mmd"]) for r in rows if math.isfinite(float(r["grad_cosine_mse_mmd"]))]
    if not vals:
        return {"n": 0, "mean_cos": None, "neg_frac": None, "p10_cos": None}
    return {
        "n": len(vals),
        "mean_cos": float(np.mean(vals)),
        "neg_frac": float(np.mean([v < 0 for v in vals])),
        "p10_cos": float(np.percentile(vals, 10)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-risk-batches", type=int, default=8)
    ap.add_argument("--max-nonrisk-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu", help="cpu or cuda:<n>; default CPU for no-GPU gate")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    cfg = Config(
        data_dir=str(DATA_DIR),
        biflow_dir=str(ROOT / "dataset/biFlow_data"),
        latent_backbone="xverse",
        split_file=str(SPLIT),
        model_type="control_mlp",
        emb_dim=384,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        min_cells=16,
        scale_noise=0.0,
        ds_alpha=0.7,
        use_mmd=True,
        mmd_estimator="biased",
        use_pert_condition=True,
        pert_gene_emb_cache_dir=str(GENE_CACHE),
        pert_chem_enabled=True,
        pert_pool_aggregations=("sum", "mean", "max", "min"),
        pert_pool_scale_init=(0.5, 1.0, 1.0, 1.0),
        pert_pool_fusion_mode="sum",
        pert_to_c_init_mode="xavier_small",
        use_pert_in_fusion=True,
        use_amp=False,
    )
    device = torch.device(args.device)
    model = build_model(cfg, device)
    load_model_weights_only(ANCHOR, model, device, strict=False, prefer_ema=True)
    model.train(False)
    path = CondOTPath()

    train_ds = CrossDatasetFMDataset(
        cfg.data_dir,
        split,
        cfg.batch_size,
        cfg.seed,
        mode="train",
        min_cells=cfg.min_cells,
        ds_alpha=cfg.ds_alpha,
        scale_noise=0.0,
        perturbation_family_filter="all",
        silent=True,
        **_cross_dataset_kw(cfg),
    )

    rng = random.Random(args.seed)
    risk_rows: list[dict[str, Any]] = []
    nonrisk_rows: list[dict[str, Any]] = []
    for src, gt, ds_name, cond, pb in train_ds:
        is_risk = str(ds_name) in RISK_DATASETS
        if is_risk and len(risk_rows) >= args.max_risk_batches:
            continue
        if (not is_risk) and len(nonrisk_rows) >= args.max_nonrisk_batches:
            continue
        row = grad_cosine_for_batch(model, path, cfg, device, src, gt, pb)
        row.update({"dataset": str(ds_name), "condition": str(cond), "risk": bool(is_risk)})
        if is_risk:
            risk_rows.append(row)
        else:
            nonrisk_rows.append(row)
        if len(risk_rows) >= args.max_risk_batches and len(nonrisk_rows) >= args.max_nonrisk_batches:
            break
    train_ds.close()

    shuffled = []
    all_rows = risk_rows + nonrisk_rows
    labels = [r["risk"] for r in all_rows]
    rng.shuffle(labels)
    for row, label in zip(all_rows, labels):
        rr = dict(row)
        rr["shuffled_risk"] = bool(label)
        shuffled.append(rr)
    shuffled_risk = [r for r in shuffled if r["shuffled_risk"]]
    shuffled_nonrisk = [r for r in shuffled if not r["shuffled_risk"]]

    risk_summary = summarize(risk_rows)
    nonrisk_summary = summarize(nonrisk_rows)
    shuffled_risk_summary = summarize(shuffled_risk)
    shuffled_nonrisk_summary = summarize(shuffled_nonrisk)

    risk_mean = risk_summary.get("mean_cos")
    nonrisk_mean = nonrisk_summary.get("mean_cos")
    shuffled_gap = None
    if shuffled_risk_summary.get("mean_cos") is not None and shuffled_nonrisk_summary.get("mean_cos") is not None:
        shuffled_gap = float(shuffled_risk_summary["mean_cos"] - shuffled_nonrisk_summary["mean_cos"])
    gap = None if risk_mean is None or nonrisk_mean is None else float(risk_mean - nonrisk_mean)

    checks = {
        "enough_risk_batches": len(risk_rows) >= max(4, min(args.max_risk_batches, 4)),
        "enough_nonrisk_batches": len(nonrisk_rows) >= max(4, min(args.max_nonrisk_batches, 4)),
        "risk_gradients_more_conflicted": gap is not None and gap < -0.05,
        "risk_negative_fraction_high": float(risk_summary.get("neg_frac") or 0.0) >= 0.25,
        "shuffled_label_gap_collapses": shuffled_gap is not None and abs(shuffled_gap) < 0.05,
    }
    status = (
        "gradient_conflict_gate_pass_gpu_projection_design_next_no_gpu"
        if all(checks.values())
        else "gradient_conflict_gate_fail_no_gpu"
    )

    payload = {
        "status": status,
        "boundary": {
            "train_only_batches": True,
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
            "training_performed": False,
            "device": str(device),
        },
        "config": {
            "split": str(SPLIT),
            "anchor": str(ANCHOR),
            "max_risk_batches": args.max_risk_batches,
            "max_nonrisk_batches": args.max_nonrisk_batches,
            "batch_size": args.batch_size,
        },
        "checks": checks,
        "gap_risk_minus_nonrisk_mean_cos": gap,
        "shuffled_gap": shuffled_gap,
        "risk_summary": risk_summary,
        "nonrisk_summary": nonrisk_summary,
        "shuffled_risk_summary": shuffled_risk_summary,
        "shuffled_nonrisk_summary": shuffled_nonrisk_summary,
        "risk_rows": risk_rows,
        "nonrisk_rows": nonrisk_rows,
        "decision": {
            "gpu_authorized": False,
            "next_if_pass": "External/code design review for a single default-off gradient projection hook.",
            "next_if_fail": "Do not launch gradient-projection GPU smoke.",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Gradient-Conflict Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- Samples train-only xverse batches and anchor checkpoint gradients.",
        "- Does not train, read canonical metrics, read canonical multi, or read Track C query.",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, value in checks.items():
        lines.append(f"| `{name}` | `{bool(value)}` |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- risk mean cosine: `{risk_summary.get('mean_cos')}`",
            f"- non-risk mean cosine: `{nonrisk_summary.get('mean_cos')}`",
            f"- risk-minus-nonrisk gap: `{gap}`",
            f"- shuffled gap: `{shuffled_gap}`",
            "",
            "## Decision",
            "",
            "- No GPU training is authorized by this gate.",
            "- A pass only authorizes external/code review for a default-off projection hook.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
