#!/usr/bin/env python3
"""CPU gate for OT pair modes inside the lookahead adapter update.

This is a focused audit of the current lookahead/trust-region bottleneck. It
does not train a checkpoint, evaluate canonical metrics, read canonical multi,
read Track C query, or use GPU by default. It compares pair modes on the same
train-only task/no-harm batches and asks whether a deterministic assignment
mode gives a stronger projected adapter step than the current exact-OT
multinomial sampler while a random control underperforms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch

ROOT = Path("/data/cyx/1030/scLatent")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CoupledFM"))

from model.latent.fm_ot import CondOTPath, OTPlanSampler  # noqa: E402
from model.latent.train import _pert_to_device, apply_finetune_freeze, build_model  # noqa: E402
from model.utils.data.ot_pairer import hungarian_pair  # noqa: E402
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
    parse_step_grid,
    pcgrad,
    set_flat_params,
    trainable_items,
)

REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "lookahead_ot_pairmode_unit_gate_20260627"
OUT_JSON = REPORTS / "latentfm_lookahead_ot_pairmode_unit_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_LOOKAHEAD_OT_PAIRMODE_UNIT_GATE_20260627.md"
ROWS_CSV = OUT_DIR / "pairmode_rows.csv"
SUMMARY_CSV = OUT_DIR / "pairmode_summary.csv"


def stable_seed(seed: int, idx: int, mode: str) -> int:
    return abs(hash((int(seed), int(idx), str(mode)))) % (2**31 - 1)


def pair_tensors(
    src: torch.Tensor,
    gt: torch.Tensor,
    *,
    mode: str,
    seed: int,
    num_threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    src = src.detach().to(dtype=torch.float32).contiguous()
    gt = gt.detach().to(dtype=torch.float32).contiguous()
    n = int(src.shape[0])
    if n <= 0:
        raise ValueError("empty source batch")
    mode = mode.lower()
    if mode == "multinomial":
        np.random.seed(seed)
        sampler = OTPlanSampler(method="exact", num_threads=int(num_threads))
        x0, x1 = sampler.sample_plan_np(src.numpy(), gt.numpy(), use_assignment=False)
        return torch.from_numpy(np.asarray(x0, dtype=np.float32)), torch.from_numpy(np.asarray(x1, dtype=np.float32))
    if mode == "assignment":
        np.random.seed(seed)
        sampler = OTPlanSampler(method="exact", num_threads=int(num_threads))
        x0, x1 = sampler.sample_plan_np(src.numpy(), gt.numpy(), use_assignment=True)
        return torch.from_numpy(np.asarray(x0, dtype=np.float32)), torch.from_numpy(np.asarray(x1, dtype=np.float32))
    if mode == "hungarian":
        torch.manual_seed(seed)
        i, j = hungarian_pair(src, gt, n_samples=n, cost_fn="l2")
        return src[i].clone(), gt[j].clone()
    if mode == "random":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        i = torch.randint(0, int(src.shape[0]), (n,), generator=gen)
        j = torch.randint(0, int(gt.shape[0]), (n,), generator=gen)
        return src[i].clone(), gt[j].clone()
    raise ValueError(f"unknown pair mode: {mode}")


def make_batch(raw: tuple, *, mode: str, device: torch.device, idx: int, seed: int, num_threads: int) -> dict[str, Any]:
    src, gt, ds_name, cond, pb = raw
    src_paired, gt_paired = pair_tensors(
        src.float(),
        gt.float(),
        mode=mode,
        seed=stable_seed(seed, idx, mode),
        num_threads=num_threads,
    )
    bsz = int(src_paired.shape[0])
    t = torch.linspace(0.05, 0.95, bsz, dtype=torch.float32)
    if idx % 2:
        t = torch.flip(t, dims=[0])
    return {
        "src": src_paired.to(device=device, dtype=torch.float32).contiguous(),
        "gt": gt_paired.to(device=device, dtype=torch.float32).contiguous(),
        "t": t.to(device=device, dtype=torch.float32).contiguous(),
        "pb": _pert_to_device(pb, device),
        "dataset": str(ds_name),
        "condition": str(cond),
    }


def next_raw(iterator, dataset):
    try:
        raw = next(iterator)
    except StopIteration:
        iterator = iter(dataset)
        raw = next(iterator)
    return iterator, raw


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
) -> dict[str, Any]:
    p0 = flat_params(trainable).clone()
    base_task = evaluate(model, anchor, task_batch, path)
    base_noharm = evaluate(model, anchor, noharm_batch, path)
    task_loss, _, _, _ = batch_losses(model=model, anchor=anchor, batch=task_batch, path=path)
    task_grad = grad_vector(task_loss, trainable)
    task_grad_norm = float(torch.linalg.norm(task_grad).item())
    _, anchor_loss0, _, _ = batch_losses(model=model, anchor=anchor, batch=noharm_batch, path=path)
    anchor_grad0 = grad_vector(anchor_loss0, trainable)
    anchor_grad0_norm = float(torch.linalg.norm(anchor_grad0).item())
    best: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None
    for step in steps:
        set_flat_params(trainable, p0 - float(step) * task_grad)
        unproj_task = evaluate(model, anchor, task_batch, path)
        unproj_noharm = evaluate(model, anchor, noharm_batch, path)
        _, probe_anchor_loss, _, _ = batch_losses(model=model, anchor=anchor, batch=noharm_batch, path=path)
        probe_anchor_grad = grad_vector(probe_anchor_loss, trainable)
        set_flat_params(trainable, p0)
        proj_grad, proj_stats = pcgrad(task_grad, probe_anchor_grad)
        set_flat_params(trainable, p0 - float(step) * proj_grad)
        proj_task = evaluate(model, anchor, task_batch, path)
        proj_noharm = evaluate(model, anchor, noharm_batch, path)
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
            "proj_anchor_delta": float(proj_anchor_delta),
            "proj_footprint_mean_l2": float(proj_noharm["footprint_mean_l2"]),
            "proj_material_row_frac": float(proj_noharm["material_row_frac"]),
            "task_retention_vs_unprojected": float(retention),
            "projection_reduced_anchor_delta_frac": float(reduction),
            "probe_anchor_grad_norm": float(proj_stats["anchor_norm"]),
            "unproj_task_delta": float(unproj_task_delta),
            "unproj_anchor_delta": float(unproj_anchor_delta),
        }
        if best_any is None or (row["proj_task_delta"], -row["proj_anchor_delta"]) < (
            best_any["proj_task_delta"],
            -best_any["proj_anchor_delta"],
        ):
            best_any = row
        candidate_ok = (
            proj_task_delta < -float(min_task_delta)
            and proj_anchor_delta <= float(anchor_threshold)
            and (reduction >= 0.50 or proj_anchor_delta <= float(anchor_threshold) * 0.01)
            and retention >= 0.20
            and proj_noharm["footprint_mean_l2"] > float(min_footprint)
            and proj_noharm["material_row_frac"] >= 0.15
        )
        if candidate_ok and (
            best is None
            or (row["proj_anchor_delta"], row["proj_task_delta"]) < (best["proj_anchor_delta"], best["proj_task_delta"])
        ):
            best = row
    set_flat_params(trainable, p0)
    out = {
        "accepted": best is not None,
        "task_grad_norm": task_grad_norm,
        "anchor_grad0_norm": anchor_grad0_norm,
        "base_task_loss": float(base_task["task_loss"]),
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
                "mean_task_delta": fmean([float(r["proj_task_delta"]) for r in acc if "proj_task_delta" in r]),
                "median_task_delta": fmedian([float(r["proj_task_delta"]) for r in acc if "proj_task_delta" in r]),
                "mean_anchor_delta": fmean([float(r["proj_anchor_delta"]) for r in acc if "proj_anchor_delta" in r]),
                "mean_footprint": fmean([float(r["proj_footprint_mean_l2"]) for r in acc if "proj_footprint_mean_l2" in r]),
                "mean_material_row_frac": fmean([float(r["proj_material_row_frac"]) for r in acc if "proj_material_row_frac" in r]),
                "mean_best_any_task_delta": fmean(
                    [float(r["best_any_proj_task_delta"]) for r in sub if "best_any_proj_task_delta" in r]
                ),
            }
        )
    return out


def decide(summary: list[dict[str, Any]]) -> tuple[str, list[str], str | None]:
    by_mode = {row["mode"]: row for row in summary}
    base = by_mode.get("multinomial") or {}
    rand = by_mode.get("random") or {}
    reasons: list[str] = []
    winner = None
    for mode in ("assignment", "hungarian"):
        cand = by_mode.get(mode) or {}
        rate_gain = float(cand.get("accepted_rate") or 0.0) - float(base.get("accepted_rate") or 0.0)
        cand_task = cand.get("mean_task_delta")
        base_task = base.get("mean_task_delta")
        rand_rate = float(rand.get("accepted_rate") or 0.0)
        if cand_task is None:
            reasons.append(f"{mode}:no_accepted_steps")
            continue
        task_better = base_task is None or float(cand_task) <= float(base_task) * 1.10
        rate_ok = rate_gain >= 0.10 or float(cand.get("accepted_rate") or 0.0) >= 0.75
        random_control = float(cand.get("accepted_rate") or 0.0) >= rand_rate + 0.10
        anchor_ok = float(cand.get("mean_anchor_delta") or 0.0) <= 1e-6
        if task_better and rate_ok and random_control and anchor_ok:
            winner = mode
            break
        if not task_better:
            reasons.append(f"{mode}:task_delta_not_better_than_multinomial")
        if not rate_ok:
            reasons.append(f"{mode}:accepted_rate_not_better_than_multinomial")
        if not random_control:
            reasons.append(f"{mode}:not_separated_from_random_control")
        if not anchor_ok:
            reasons.append(f"{mode}:anchor_delta_above_threshold")
    if winner:
        return "lookahead_ot_pairmode_unit_pass_gpu_smoke_candidate", [], winner
    return "lookahead_ot_pairmode_unit_fail_no_gpu", reasons or ["no_pairmode_winner"], None


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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=["multinomial", "assignment", "hungarian", "random"])
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
    provenance_ok = bool(
        allowed_trainable
        and allowed_missing
        and not cand_load["unexpected"]
        and not cand_load["skipped_shape_mismatch"]
    )
    if not provenance_ok:
        raise RuntimeError(
            "provenance/scope failure: "
            f"allowed_trainable={allowed_trainable} allowed_missing={allowed_missing} "
            f"unexpected={cand_load['unexpected']} skipped={cand_load['skipped_shape_mismatch']}"
        )

    dataset = make_dataset(cfg, seed=int(args.seed), batch_size=int(args.batch_size))
    iterator = iter(dataset)
    path = CondOTPath()
    steps = parse_step_grid(args.step_grid)
    p0 = flat_params(trainable).clone()
    rows: list[dict[str, Any]] = []
    try:
        for pair_idx in range(int(args.max_pairs)):
            iterator, raw_task = next_raw(iterator, dataset)
            iterator, raw_noharm = next_raw(iterator, dataset)
            for mode in args.modes:
                set_flat_params(trainable, p0)
                task_batch = make_batch(
                    raw_task,
                    mode=mode,
                    device=device,
                    idx=2 * pair_idx,
                    seed=int(args.seed),
                    num_threads=int(args.num_threads),
                )
                noharm_batch = make_batch(
                    raw_noharm,
                    mode=mode,
                    device=device,
                    idx=2 * pair_idx + 1,
                    seed=int(args.seed),
                    num_threads=int(args.num_threads),
                )
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
                )
                rows.append(
                    {
                        "pair_idx": pair_idx,
                        "mode": mode,
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
        "mean_task_delta",
        "median_task_delta",
        "mean_anchor_delta",
        "mean_footprint",
        "mean_material_row_frac",
        "mean_best_any_task_delta",
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
            "anchor_checkpoint": str(ANCHOR_CKPT),
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_checkpoint": False,
            "uses_canonical_metrics": False,
        },
        "args": vars(args),
        "provenance": {
            "anchor_ema_applied": bool(anchor_load.get("ema_applied")),
            "candidate_ema_applied": bool(cand_load.get("ema_applied")),
            "trainable_names": trainable_names,
        },
        "summary": summary,
        "rows_csv": str(ROWS_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "next_action": (
            "if winner remains stable on an extended CPU gate, prepare exactly one bounded GPU smoke with that pair mode"
            if winner
            else "do not launch a pair-mode GPU smoke from this evidence"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Lookahead OT Pairmode Unit Gate",
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
        "- No checkpoint training, no canonical metrics, no canonical multi, no Track C query, and no GPU.",
        "",
        "## Summary",
        "",
        "| mode | attempts | accepted | rate | mean task delta | mean anchor delta | mean footprint | mean material rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['mode']}` | {row['attempts']} | {row['accepted']} | {fmt(row['accepted_rate'])} | "
            f"{fmt(row['mean_task_delta'])} | {fmt(row['mean_anchor_delta'])} | "
            f"{fmt(row['mean_footprint'])} | {fmt(row['mean_material_row_frac'])} |"
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
