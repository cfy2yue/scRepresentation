#!/usr/bin/env python3
"""CPU gate for response-normalized lookahead adapter gradients.

This tests a narrow reopen of the closed response-normalization family: use a
train-only normalizer only as a default-off task-loss preconditioner inside the
lookahead adapter unit step, while measuring raw-space anchor/no-harm drift.
It does not train a checkpoint, evaluate canonical metrics, read canonical
multi, read Track C query, or use GPU by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data/cyx/1030/scLatent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CoupledFM"))

from model.latent.fm_ot import CondOTPath, OTPlanSampler  # noqa: E402
from model.latent.response_normalizer import ResponseNormalizer  # noqa: E402
from model.latent.train import (  # noqa: E402
    _model_latent_velocity,
    apply_finetune_freeze,
    build_model,
)
from ops.train_latentfm_lookahead_trust_region_adapter_smoke_20260627 import (  # noqa: E402
    ANCHOR_CKPT,
    SAFE_SPLIT,
    batch_losses,
    candidate_cfg,
    cfg_from_checkpoint,
    evaluate,
    flat_params,
    grad_vector,
    load_raw_then_ema,
    make_dataset,
    next_batch,
    parse_step_grid,
    pcgrad,
    set_flat_params,
    trainable_items,
)

REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "lookahead_response_normalized_unit_gate_20260627"
OUT_JSON = REPORTS / "latentfm_lookahead_response_normalized_unit_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_LOOKAHEAD_RESPONSE_NORMALIZED_UNIT_GATE_20260627.md"
ROWS_CSV = OUT_DIR / "normalizer_rows.csv"
SUMMARY_CSV = OUT_DIR / "normalizer_summary.csv"


def shuffled_dataset(ds: str, all_datasets: list[str]) -> str:
    if not all_datasets:
        return str(ds)
    ordered = sorted(str(x) for x in all_datasets)
    if str(ds) not in ordered:
        return ordered[0]
    idx = ordered.index(str(ds))
    return ordered[(idx + 7) % len(ordered)]


def normalized_task_loss(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    path: CondOTPath,
    normalizer: ResponseNormalizer | None,
    norm_dataset: str | None = None,
) -> torch.Tensor:
    src = batch["src"]
    gt = batch["gt"]
    t = batch["t"]
    pb = batch["pb"]
    ps = path.sample(x_0=src, x_1=gt, t=t)
    v_pred = _model_latent_velocity(model, ps.x_t, ps.t, src, pb)
    target = ps.dx_t
    if normalizer is not None:
        ds = str(norm_dataset or batch["dataset"])
        v_pred = normalizer.transform_delta(ds, v_pred.float())
        target = normalizer.transform_delta(ds, target.float())
    return F.mse_loss(v_pred.float(), target.float())


def mode_trial(
    *,
    model: torch.nn.Module,
    anchor: torch.nn.Module,
    trainable: list[tuple[str, torch.nn.Parameter]],
    task_batch: dict[str, Any],
    noharm_batch: dict[str, Any],
    path: CondOTPath,
    steps: list[float],
    anchor_threshold: float,
    min_task_delta: float,
    min_footprint: float,
    normalizer: ResponseNormalizer | None,
    norm_dataset: str | None,
) -> dict[str, Any]:
    p0 = flat_params(trainable).clone()
    base_task_eval = evaluate(model, anchor, task_batch, path)
    base_noharm = evaluate(model, anchor, noharm_batch, path)
    base_task_loss = normalized_task_loss(
        model=model,
        batch=task_batch,
        path=path,
        normalizer=normalizer,
        norm_dataset=norm_dataset,
    )
    task_grad = grad_vector(base_task_loss, trainable)
    task_grad_norm = float(torch.linalg.norm(task_grad).item())
    _, anchor_loss0, _, _ = batch_losses(model=model, anchor=anchor, batch=noharm_batch, path=path)
    anchor_grad0 = grad_vector(anchor_loss0, trainable)
    anchor_grad0_norm = float(torch.linalg.norm(anchor_grad0).item())
    best: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None
    for step in steps:
        set_flat_params(trainable, p0 - float(step) * task_grad)
        unproj_norm_task = normalized_task_loss(
            model=model,
            batch=task_batch,
            path=path,
            normalizer=normalizer,
            norm_dataset=norm_dataset,
        )
        unproj_task_eval = evaluate(model, anchor, task_batch, path)
        unproj_noharm = evaluate(model, anchor, noharm_batch, path)
        _, probe_anchor_loss, _, _ = batch_losses(model=model, anchor=anchor, batch=noharm_batch, path=path)
        probe_anchor_grad = grad_vector(probe_anchor_loss, trainable)
        set_flat_params(trainable, p0)
        proj_grad, proj_stats = pcgrad(task_grad, probe_anchor_grad)
        set_flat_params(trainable, p0 - float(step) * proj_grad)
        proj_norm_task = normalized_task_loss(
            model=model,
            batch=task_batch,
            path=path,
            normalizer=normalizer,
            norm_dataset=norm_dataset,
        )
        proj_task_eval = evaluate(model, anchor, task_batch, path)
        proj_noharm = evaluate(model, anchor, noharm_batch, path)
        set_flat_params(trainable, p0)
        unproj_norm_delta = float(unproj_norm_task.item() - base_task_loss.item())
        proj_norm_delta = float(proj_norm_task.item() - base_task_loss.item())
        raw_proj_task_delta = float(proj_task_eval["task_loss"] - base_task_eval["task_loss"])
        raw_unproj_task_delta = float(unproj_task_eval["task_loss"] - base_task_eval["task_loss"])
        unproj_anchor_delta = float(unproj_noharm["anchor_loss"] - base_noharm["anchor_loss"])
        proj_anchor_delta = float(proj_noharm["anchor_loss"] - base_noharm["anchor_loss"])
        retention = (
            abs(proj_norm_delta) / max(abs(unproj_norm_delta), 1e-12)
            if proj_norm_delta < 0 and unproj_norm_delta < 0
            else 0.0
        )
        reduction = (
            1.0 - (proj_anchor_delta / max(unproj_anchor_delta, 1e-12))
            if unproj_anchor_delta > 0
            else 0.0
        )
        row = {
            "best_step": float(step),
            "proj_norm_task_delta": proj_norm_delta,
            "unproj_norm_task_delta": unproj_norm_delta,
            "raw_proj_task_delta": raw_proj_task_delta,
            "raw_unproj_task_delta": raw_unproj_task_delta,
            "proj_anchor_delta": proj_anchor_delta,
            "proj_footprint_mean_l2": float(proj_noharm["footprint_mean_l2"]),
            "proj_material_row_frac": float(proj_noharm["material_row_frac"]),
            "task_retention_vs_unprojected": float(retention),
            "projection_reduced_anchor_delta_frac": float(reduction),
            "probe_anchor_grad_norm": float(proj_stats["anchor_norm"]),
        }
        if best_any is None or (row["proj_norm_task_delta"], -row["proj_anchor_delta"]) < (
            best_any["proj_norm_task_delta"],
            -best_any["proj_anchor_delta"],
        ):
            best_any = row
        candidate_ok = (
            proj_norm_delta < -float(min_task_delta)
            and proj_anchor_delta <= float(anchor_threshold)
            and (reduction >= 0.50 or proj_anchor_delta <= float(anchor_threshold) * 0.01)
            and retention >= 0.20
            and proj_noharm["footprint_mean_l2"] > float(min_footprint)
            and proj_noharm["material_row_frac"] >= 0.15
        )
        if candidate_ok and (
            best is None
            or (row["proj_anchor_delta"], row["proj_norm_task_delta"]) < (
                best["proj_anchor_delta"],
                best["proj_norm_task_delta"],
            )
        ):
            best = row
    set_flat_params(trainable, p0)
    out = {
        "accepted": best is not None,
        "task_grad_norm": task_grad_norm,
        "anchor_grad0_norm": anchor_grad0_norm,
        "base_raw_task_loss": float(base_task_eval["task_loss"]),
        "base_norm_task_loss": float(base_task_loss.item()),
        "base_noharm_anchor_loss": float(base_noharm["anchor_loss"]),
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


def summarize(rows: list[dict[str, Any]], modes: list[str]) -> list[dict[str, Any]]:
    out = []
    for mode in modes:
        sub = [r for r in rows if r["mode"] == mode]
        acc = [r for r in sub if r["accepted"]]
        out.append(
            {
                "mode": mode,
                "attempts": len(sub),
                "accepted": len(acc),
                "accepted_rate": len(acc) / max(len(sub), 1),
                "mean_norm_task_delta": fmean([float(r["proj_norm_task_delta"]) for r in acc if "proj_norm_task_delta" in r]),
                "median_norm_task_delta": fmedian(
                    [float(r["proj_norm_task_delta"]) for r in acc if "proj_norm_task_delta" in r]
                ),
                "mean_raw_task_delta": fmean([float(r["raw_proj_task_delta"]) for r in acc if "raw_proj_task_delta" in r]),
                "mean_anchor_delta": fmean([float(r["proj_anchor_delta"]) for r in acc if "proj_anchor_delta" in r]),
                "mean_footprint": fmean([float(r["proj_footprint_mean_l2"]) for r in acc if "proj_footprint_mean_l2" in r]),
                "mean_material_row_frac": fmean([float(r["proj_material_row_frac"]) for r in acc if "proj_material_row_frac" in r]),
                "mean_task_grad_norm": fmean([float(r["task_grad_norm"]) for r in sub]),
            }
        )
    return out


def decide(summary: list[dict[str, Any]]) -> tuple[str, list[str], str | None]:
    by_mode = {row["mode"]: row for row in summary}
    raw = by_mode.get("raw") or {}
    control = by_mode.get("dataset_scale_pca_shuffled") or {}
    raw_foot = float(raw.get("mean_footprint") or 0.0)
    raw_rate = float(raw.get("accepted_rate") or 0.0)
    control_foot = float(control.get("mean_footprint") or 0.0)
    reasons: list[str] = []
    winner = None
    for mode in ("dataset_scale", "pca_subspace", "dataset_scale_pca"):
        cand = by_mode.get(mode) or {}
        cand_rate = float(cand.get("accepted_rate") or 0.0)
        cand_foot = float(cand.get("mean_footprint") or 0.0)
        cand_anchor = float(cand.get("mean_anchor_delta") or 0.0)
        cand_raw_delta = cand.get("mean_raw_task_delta")
        raw_delta = raw.get("mean_raw_task_delta")
        rate_ok = cand_rate >= max(0.75, raw_rate - 0.10)
        footprint_ok = cand_foot >= max(raw_foot * 1.50, control_foot * 1.20, 1e-6)
        anchor_ok = cand_anchor <= 1e-6
        raw_not_harmed = cand_raw_delta is not None and (raw_delta is None or float(cand_raw_delta) <= max(float(raw_delta), 0.0) + 1e-10)
        if rate_ok and footprint_ok and anchor_ok and raw_not_harmed:
            winner = mode
            break
        if not rate_ok:
            reasons.append(f"{mode}:accepted_rate_too_low")
        if not footprint_ok:
            reasons.append(f"{mode}:footprint_not_larger_than_raw_or_shuffled")
        if not anchor_ok:
            reasons.append(f"{mode}:anchor_delta_above_threshold")
        if not raw_not_harmed:
            reasons.append(f"{mode}:raw_task_delta_not_safe")
    if winner:
        return "lookahead_response_normalized_unit_pass_gpu_smoke_candidate", [], winner
    return "lookahead_response_normalized_unit_fail_no_gpu", reasons or ["no_response_normalizer_winner"], None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        if math.isnan(x):
            return "NA"
        return f"{x:.6g}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--normalizer", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["raw", "dataset_scale", "pca_subspace", "dataset_scale_pca", "dataset_scale_pca_shuffled"],
    )
    ap.add_argument("--step-grid", default="1,3,10,30,100,300")
    ap.add_argument("--anchor-threshold", type=float, default=1e-6)
    ap.add_argument("--min-task-delta", type=float, default=1e-10)
    ap.add_argument("--min-footprint", type=float, default=1e-7)
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()

    if str(args.device).startswith("cuda"):
        raise ValueError("This audit is CPU-only by default; do not use CUDA for this gate.")
    torch.set_num_threads(max(1, int(args.num_threads)))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    normalizers: dict[str, ResponseNormalizer | None] = {"raw": None}
    for mode in ("dataset_scale", "pca_subspace", "dataset_scale_pca"):
        normalizers[mode] = ResponseNormalizer.from_npz(
            args.normalizer,
            mode=mode,
            device=device,
            strict_split_file=SAFE_SPLIT,
        )
    normalizers["dataset_scale_pca_shuffled"] = normalizers["dataset_scale_pca"]
    norm_meta = normalizers["dataset_scale_pca"].metadata or {}
    all_datasets = sorted((norm_meta.get("dataset_train_residual_counts") or {}).keys())

    anchor_cfg = cfg_from_checkpoint(ANCHOR_CKPT, batch_size=int(args.batch_size), seed=int(args.seed))
    cfg = candidate_cfg(anchor_cfg)
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
    if not (allowed_trainable and allowed_missing and not cand_load["unexpected"] and not cand_load["skipped_shape_mismatch"]):
        raise RuntimeError("provenance/scope failure")

    dataset = make_dataset(cfg, seed=int(args.seed), batch_size=int(args.batch_size))
    iterator = iter(dataset)
    sampler = OTPlanSampler(method="exact", num_threads=min(4, int(args.num_threads)))
    path = CondOTPath()
    steps = parse_step_grid(args.step_grid)
    p0 = flat_params(trainable).clone()
    rows: list[dict[str, Any]] = []
    try:
        for pair_idx in range(int(args.max_pairs)):
            iterator, task_batch = next_batch(iterator, dataset, sampler, device, idx=2 * pair_idx)
            iterator, noharm_batch = next_batch(iterator, dataset, sampler, device, idx=2 * pair_idx + 1)
            for mode in args.modes:
                set_flat_params(trainable, p0)
                norm_ds = None
                if mode == "dataset_scale_pca_shuffled":
                    norm_ds = shuffled_dataset(task_batch["dataset"], all_datasets)
                trial = mode_trial(
                    model=candidate,
                    anchor=anchor,
                    trainable=trainable,
                    task_batch=task_batch,
                    noharm_batch=noharm_batch,
                    path=path,
                    steps=steps,
                    anchor_threshold=float(args.anchor_threshold),
                    min_task_delta=float(args.min_task_delta),
                    min_footprint=float(args.min_footprint),
                    normalizer=normalizers.get(mode),
                    norm_dataset=norm_ds,
                )
                rows.append(
                    {
                        "pair_idx": pair_idx,
                        "mode": mode,
                        "norm_dataset": norm_ds or task_batch["dataset"],
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

    summary = summarize(rows, list(args.modes))
    status, reasons, winner = decide(summary)
    row_fields = sorted({key for row in rows for key in row})
    summary_fields = [
        "mode",
        "attempts",
        "accepted",
        "accepted_rate",
        "mean_norm_task_delta",
        "median_norm_task_delta",
        "mean_raw_task_delta",
        "mean_anchor_delta",
        "mean_footprint",
        "mean_material_row_frac",
        "mean_task_grad_norm",
    ]
    write_csv(ROWS_CSV, rows, row_fields)
    write_csv(SUMMARY_CSV, summary, summary_fields)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": status.endswith("gpu_smoke_candidate"),
        "winner": winner,
        "reasons": reasons,
        "boundary": {
            "cpu_only": True,
            "device": str(device),
            "safe_split": str(SAFE_SPLIT),
            "normalizer": str(args.normalizer.resolve()),
            "anchor_checkpoint": str(ANCHOR_CKPT),
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_checkpoint": False,
            "uses_canonical_metrics": False,
        },
        "normalizer_metadata": norm_meta,
        "args": {**vars(args), "normalizer": str(args.normalizer)},
        "provenance": {
            "anchor_ema_applied": bool(anchor_load.get("ema_applied")),
            "candidate_ema_applied": bool(cand_load.get("ema_applied")),
            "trainable_names": trainable_names,
        },
        "summary": summary,
        "rows_csv": str(ROWS_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "next_action": (
            "if winner remains stable on extended CPU gate, prepare one bounded GPU smoke using normalized task loss"
            if winner
            else "do not launch a response-normalized GPU smoke from this evidence"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Lookahead Response-Normalized Unit Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{str(payload['gpu_authorized']).lower()}`",
        "",
        "## Boundary",
        "",
        "- CPU-only adapter-gradient audit on safe train-only batches.",
        "- Normalizer was fit from the same train-only split and hash-checked.",
        "- No checkpoint training, no canonical metrics, no canonical multi, no Track C query, and no GPU.",
        "",
        "## Summary",
        "",
        "| mode | attempts | accepted | rate | mean norm task delta | mean raw task delta | mean anchor delta | mean footprint | mean grad norm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['mode']}` | {row['attempts']} | {row['accepted']} | {fmt(row['accepted_rate'])} | "
            f"{fmt(row['mean_norm_task_delta'])} | {fmt(row['mean_raw_task_delta'])} | "
            f"{fmt(row['mean_anchor_delta'])} | {fmt(row['mean_footprint'])} | "
            f"{fmt(row['mean_task_grad_norm'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- winner: `{winner or 'none'}`",
            f"- reasons: `{reasons}`",
            f"- next action: {payload['next_action']}",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- rows: `{ROWS_CSV}`",
            f"- summary: `{SUMMARY_CSV}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "winner": winner, "report": str(OUT_MD)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
