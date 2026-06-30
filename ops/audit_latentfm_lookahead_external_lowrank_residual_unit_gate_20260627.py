#!/usr/bin/env python3
"""CPU gate for an external low-rank condition residual adapter.

This is a non-invasive architecture probe. It freezes the xverse anchor,
extracts the anchor perturbation conditioning vector, and adds a zero-initialized
low-rank condition-only residual directly to the predicted velocity. It tests
whether a larger controlled footprint is even possible before editing the core
model/checkpoint schema.

No checkpoint training, canonical metrics, canonical multi, Track C query, or
GPU is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data/cyx/1030/scLatent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CoupledFM"))

from model.latent.fm_ot import CondOTPath, OTPlanSampler  # noqa: E402
from model.latent.train import (  # noqa: E402
    _model_latent_velocity,
    _unpack_pert_up_to7,
    _unwrap_model,
    build_model,
)
from ops.train_latentfm_lookahead_trust_region_adapter_smoke_20260627 import (  # noqa: E402
    ANCHOR_CKPT,
    SAFE_SPLIT,
    cfg_from_checkpoint,
    flat_params,
    grad_vector,
    load_raw_then_ema,
    make_dataset,
    next_batch,
    parse_step_grid,
    pcgrad,
    set_flat_params,
    to_device_batch,
    trainable_items,
)
from ops.audit_latentfm_lookahead_balanced_schedule_unit_gate_20260627 import (  # noqa: E402
    build_balanced_pairs,
    build_sequential_pairs,
    collect_raws,
)

REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "lookahead_external_lowrank_residual_unit_gate_20260627"
OUT_JSON = REPORTS / "latentfm_lookahead_external_lowrank_residual_unit_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_LOOKAHEAD_EXTERNAL_LOWRANK_RESIDUAL_UNIT_GATE_20260627.md"
ROWS_CSV = OUT_DIR / "lowrank_rows.csv"
SUMMARY_CSV = OUT_DIR / "lowrank_summary.csv"


class ExternalLowRankConditionResidual(nn.Module):
    def __init__(self, d_model: int, emb_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(int(d_model), int(rank), bias=False)
        self.up = nn.Linear(int(rank), int(emb_dim), bias=True)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, p_d: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(p_d)))


def condition_features(anchor: torch.nn.Module, pb: tuple) -> torch.Tensor:
    inner = _unwrap_model(anchor)
    gid, mk, tid, npt, cid, chem_emb, chem_mask = _unpack_pert_up_to7(pb)
    with torch.no_grad():
        return inner._pert_projection(
            pert_gene_ids=gid,
            pert_mask=mk,
            pert_type_id=tid,
            nperts=npt,
            combo_id=cid,
            chem_emb=chem_emb,
            chem_mask=chem_mask,
        ).detach()


def lowrank_batch_losses(
    *,
    anchor: torch.nn.Module,
    adapter: ExternalLowRankConditionResidual,
    batch: dict[str, Any],
    path: CondOTPath,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src = batch["src"]
    gt = batch["gt"]
    t = batch["t"]
    pb = batch["pb"]
    ps = path.sample(x_0=src, x_1=gt, t=t)
    with torch.no_grad():
        anchor_v = _model_latent_velocity(anchor, ps.x_t, ps.t, src, pb)
        p_d = condition_features(anchor, pb)
    delta_v = adapter(p_d)
    v_pred = anchor_v.detach() + delta_v
    task = F.mse_loss(v_pred.float(), ps.dx_t.float())
    x1_hat = ps.x_t + v_pred * (1.0 - t).unsqueeze(-1)
    anchor_x1 = ps.x_t + anchor_v.detach() * (1.0 - t).unsqueeze(-1)
    anchor_loss = F.mse_loss(x1_hat.float(), anchor_x1.float())
    row_l2 = torch.linalg.norm((x1_hat - anchor_x1).float(), dim=1)
    return task, anchor_loss, row_l2.mean(), row_l2


def evaluate_lowrank(anchor, adapter, batch, path) -> dict[str, float]:
    with torch.no_grad():
        task, anchor_loss, footprint, row_l2 = lowrank_batch_losses(
            anchor=anchor,
            adapter=adapter,
            batch=batch,
            path=path,
        )
    return {
        "task_loss": float(task.item()),
        "anchor_loss": float(anchor_loss.item()),
        "footprint_mean_l2": float(footprint.item()),
        "material_row_frac": float((row_l2 > 1e-6).float().mean().item()),
    }


def run_trial(
    *,
    anchor: torch.nn.Module,
    adapter: ExternalLowRankConditionResidual,
    trainable: list[tuple[str, torch.nn.Parameter]],
    task_batch: dict[str, Any],
    noharm_batch: dict[str, Any],
    path: CondOTPath,
    steps: list[float],
    anchor_threshold: float,
    min_task_delta: float,
    min_footprint: float,
) -> dict[str, Any]:
    p0 = flat_params(trainable).clone()
    base_task = evaluate_lowrank(anchor, adapter, task_batch, path)
    base_noharm = evaluate_lowrank(anchor, adapter, noharm_batch, path)
    task_loss, _, _, _ = lowrank_batch_losses(anchor=anchor, adapter=adapter, batch=task_batch, path=path)
    task_grad = grad_vector(task_loss, trainable)
    task_grad_norm = float(torch.linalg.norm(task_grad).item())
    _, anchor_loss0, _, _ = lowrank_batch_losses(anchor=anchor, adapter=adapter, batch=noharm_batch, path=path)
    anchor_grad0 = grad_vector(anchor_loss0, trainable)
    anchor_grad0_norm = float(torch.linalg.norm(anchor_grad0).item())
    best: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None
    for step in steps:
        set_flat_params(trainable, p0 - float(step) * task_grad)
        unproj_task = evaluate_lowrank(anchor, adapter, task_batch, path)
        unproj_noharm = evaluate_lowrank(anchor, adapter, noharm_batch, path)
        _, probe_anchor_loss, _, _ = lowrank_batch_losses(anchor=anchor, adapter=adapter, batch=noharm_batch, path=path)
        probe_anchor_grad = grad_vector(probe_anchor_loss, trainable)
        set_flat_params(trainable, p0)
        proj_grad, proj_stats = pcgrad(task_grad, probe_anchor_grad)
        set_flat_params(trainable, p0 - float(step) * proj_grad)
        proj_task = evaluate_lowrank(anchor, adapter, task_batch, path)
        proj_noharm = evaluate_lowrank(anchor, adapter, noharm_batch, path)
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
        row = {
            "best_step": float(step),
            "proj_task_delta": float(proj_task_delta),
            "unproj_task_delta": float(unproj_task_delta),
            "proj_anchor_delta": float(proj_anchor_delta),
            "unproj_anchor_delta": float(unproj_anchor_delta),
            "proj_footprint_mean_l2": float(proj_noharm["footprint_mean_l2"]),
            "proj_material_row_frac": float(proj_noharm["material_row_frac"]),
            "task_retention_vs_unprojected": float(retention),
            "projection_reduced_anchor_delta_frac": float(reduction),
            "probe_anchor_grad_norm": float(proj_stats["anchor_norm"]),
        }
        if best_any is None or (row["proj_task_delta"], -row["proj_anchor_delta"]) < (
            best_any["proj_task_delta"],
            -best_any["proj_anchor_delta"],
        ):
            best_any = row
        ok = (
            proj_task_delta < -float(min_task_delta)
            and proj_anchor_delta <= float(anchor_threshold)
            and (reduction >= 0.50 or proj_anchor_delta <= float(anchor_threshold) * 0.01)
            and retention >= 0.20
            and proj_noharm["footprint_mean_l2"] > float(min_footprint)
            and proj_noharm["material_row_frac"] >= 0.15
        )
        if ok and (best is None or (row["proj_anchor_delta"], row["proj_task_delta"]) < (best["proj_anchor_delta"], best["proj_task_delta"])):
            best = row
    set_flat_params(trainable, p0)
    out = {
        "accepted": best is not None,
        "task_grad_norm": task_grad_norm,
        "anchor_grad0_norm": anchor_grad0_norm,
        "base_task_loss": base_task["task_loss"],
        "base_noharm_anchor_loss": base_noharm["anchor_loss"],
    }
    if best is None and best_any is not None:
        out.update({f"best_any_{k}": v for k, v in best_any.items()})
    if best is not None:
        out.update(best)
    return out


def fmean(vals: list[float]) -> float | None:
    return float(mean(vals)) if vals else None


def fmedian(vals: list[float]) -> float | None:
    return float(median(vals)) if vals else None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--schedule", choices=["iterator", "sequential_pool", "balanced_pool"], default="iterator")
    ap.add_argument("--pool-size", type=int, default=96)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--step-grid", default="1,3,10,30,100,300")
    ap.add_argument("--anchor-threshold", type=float, default=1e-6)
    ap.add_argument("--min-task-delta", type=float, default=1e-10)
    ap.add_argument("--min-footprint", type=float, default=5e-6)
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()
    if str(args.device).startswith("cuda"):
        raise ValueError("CPU-only gate: do not use CUDA")
    torch.set_num_threads(max(1, int(args.num_threads)))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg = cfg_from_checkpoint(ANCHOR_CKPT, batch_size=int(args.batch_size), seed=int(args.seed))
    anchor = build_model(cfg, device)
    load_info = load_raw_then_ema(ANCHOR_CKPT, anchor, cfg, device)
    for p in anchor.parameters():
        p.requires_grad = False
    anchor.eval()
    inner = _unwrap_model(anchor)
    d_model = int(getattr(inner, "d_model"))
    emb_dim = int(getattr(cfg, "emb_dim"))
    adapter = ExternalLowRankConditionResidual(d_model=d_model, emb_dim=emb_dim, rank=int(args.rank)).to(device)
    adapter.train()
    trainable = trainable_items(adapter)
    trainable_names = [name for name, _ in trainable]

    dataset = make_dataset(cfg, seed=int(args.seed), batch_size=int(args.batch_size))
    iterator = iter(dataset)
    sampler = OTPlanSampler(method="exact", num_threads=min(4, int(args.num_threads)))
    path = CondOTPath()
    steps = parse_step_grid(args.step_grid)
    p0 = flat_params(trainable).clone()
    rows: list[dict[str, Any]] = []
    noop_drifts: list[float] = []
    try:
        for idx in range(2):
            iterator, batch = next_batch(iterator, dataset, sampler, device, idx=idx)
            with torch.no_grad():
                src, gt, t, pb = batch["src"], batch["gt"], batch["t"], batch["pb"]
                ps = path.sample(src, gt, t)
                anchor_v = _model_latent_velocity(anchor, ps.x_t, ps.t, src, pb)
                p_d = condition_features(anchor, pb)
                v_pred = anchor_v + adapter(p_d)
                noop_drifts.append(float((v_pred - anchor_v).abs().max().item()))
        set_flat_params(trainable, p0)
        scheduled_pairs: list[tuple[tuple, tuple]] | None = None
        if args.schedule != "iterator":
            raws = collect_raws(dataset, pool_size=int(args.pool_size))
            if args.schedule == "balanced_pool":
                scheduled_pairs = build_balanced_pairs(raws, int(args.max_pairs))
            else:
                scheduled_pairs = build_sequential_pairs(raws, int(args.max_pairs))
        for pair_idx in range(int(args.max_pairs)):
            if scheduled_pairs is None:
                iterator, task_batch = next_batch(iterator, dataset, sampler, device, idx=2 * pair_idx + 2)
                iterator, noharm_batch = next_batch(iterator, dataset, sampler, device, idx=2 * pair_idx + 3)
            else:
                if pair_idx >= len(scheduled_pairs):
                    break
                task_raw, noharm_raw = scheduled_pairs[pair_idx]
                task_batch = to_device_batch(task_raw, sampler, device, idx=2 * pair_idx + 2)
                noharm_batch = to_device_batch(noharm_raw, sampler, device, idx=2 * pair_idx + 3)
            trial = run_trial(
                anchor=anchor,
                adapter=adapter,
                trainable=trainable,
                task_batch=task_batch,
                noharm_batch=noharm_batch,
                path=path,
                steps=steps,
                anchor_threshold=float(args.anchor_threshold),
                min_task_delta=float(args.min_task_delta),
                min_footprint=float(args.min_footprint),
            )
            rows.append(
                {
                    "pair_idx": pair_idx,
                    "task_dataset": task_batch["dataset"],
                    "task_condition": task_batch["condition"],
                    "noharm_dataset": noharm_batch["dataset"],
                    "noharm_condition": noharm_batch["condition"],
                    **trial,
                }
            )
    finally:
        set_flat_params(trainable, p0)
        dataset.close()

    acc = [r for r in rows if r["accepted"]]
    summary = {
        "attempts": len(rows),
        "accepted": len(acc),
        "accepted_rate": len(acc) / max(len(rows), 1),
        "task_dataset_coverage": len({r["task_dataset"] for r in rows}),
        "noharm_dataset_coverage": len({r["noharm_dataset"] for r in rows}),
        "mean_task_delta": fmean([float(r["proj_task_delta"]) for r in acc if "proj_task_delta" in r]),
        "median_task_delta": fmedian([float(r["proj_task_delta"]) for r in acc if "proj_task_delta" in r]),
        "mean_anchor_delta": fmean([float(r["proj_anchor_delta"]) for r in acc if "proj_anchor_delta" in r]),
        "mean_footprint": fmean([float(r["proj_footprint_mean_l2"]) for r in acc if "proj_footprint_mean_l2" in r]),
        "mean_material_row_frac": fmean([float(r["proj_material_row_frac"]) for r in acc if "proj_material_row_frac" in r]),
        "mean_task_grad_norm": fmean([float(r["task_grad_norm"]) for r in rows]),
    }
    reasons: list[str] = []
    if max(noop_drifts or [999.0]) > 1e-7:
        reasons.append("initial_noop_drift_above_1e-7")
    if summary["accepted_rate"] < 0.75:
        reasons.append("accepted_rate_below_0p75")
    if (summary["mean_footprint"] or 0.0) < 5e-6:
        reasons.append("mean_footprint_below_5e-6")
    if (summary["mean_anchor_delta"] or 0.0) > 1e-6:
        reasons.append("mean_anchor_delta_above_1e-6")
    if summary["task_dataset_coverage"] < min(int(args.max_pairs), 4):
        reasons.append("task_dataset_coverage_too_low")
    status = (
        "lookahead_external_lowrank_residual_unit_pass_model_patch_candidate"
        if not reasons
        else "lookahead_external_lowrank_residual_unit_fail_no_gpu"
    )

    row_fields = sorted({key for row in rows for key in row})
    write_csv(ROWS_CSV, rows, row_fields)
    write_csv(SUMMARY_CSV, [summary], list(summary.keys()))
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "model_patch_candidate": not reasons,
        "reasons": reasons,
        "boundary": {
            "cpu_only": True,
            "safe_split": str(SAFE_SPLIT),
            "anchor_checkpoint": str(ANCHOR_CKPT),
            "external_adapter_only": True,
            "edits_core_model": False,
            "trains_checkpoint": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
        },
        "args": vars(args),
        "load_info": load_info,
        "trainable_names": trainable_names,
        "max_noop_drift": max(noop_drifts or [999.0]),
        "summary": summary,
        "rows_csv": str(ROWS_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "next_action": (
            "design a guarded core-model patch and CPU provenance gate before any GPU"
            if not reasons
            else "do not patch core model or launch GPU from this evidence"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# LatentFM Lookahead External Low-Rank Residual Unit Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `false`",
        "",
        "## Boundary",
        "",
        "- CPU-only non-invasive architecture probe.",
        "- Freezes the anchor and adds a zero-initialized external low-rank condition residual to velocity.",
        "- No checkpoint training, no canonical metrics, no canonical multi, no Track C query, and no GPU.",
        "",
        "## Summary",
        "",
        "| attempts | accepted | rate | task ds | noharm ds | mean task delta | mean anchor delta | mean footprint | mean material rows |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {summary['attempts']} | {summary['accepted']} | {summary['accepted_rate']:.6g} | "
        f"{summary['task_dataset_coverage']} | {summary['noharm_dataset_coverage']} | "
        f"{summary['mean_task_delta']} | {summary['mean_anchor_delta']} | "
        f"{summary['mean_footprint']} | {summary['mean_material_row_frac']} |",
        "",
        "## Decision",
        "",
        f"- max no-op drift: `{payload['max_noop_drift']}`",
        f"- reasons: `{reasons}`",
        f"- next action: {payload['next_action']}",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{ROWS_CSV}`",
        f"- summary: `{SUMMARY_CSV}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(OUT_MD)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
