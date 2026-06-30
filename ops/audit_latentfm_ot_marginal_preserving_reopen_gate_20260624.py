#!/usr/bin/env python3
"""CPU-only gate for marginal-preserving OT reopen ideas.

This diagnostic uses only the train-only split. It does not train, select a
checkpoint, or read canonical / Track C query artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
sys.path.insert(0, str(COUPLED))

from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402
from model.utils.data.ot_pairer import compute_ot_cost, sinkhorn_pair  # noqa: E402


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _finite(vals: list[float]) -> list[float]:
    return [v for v in vals if math.isfinite(v)]


def _mean(vals: list[float]) -> float:
    vals = _finite(vals)
    return float(statistics.fmean(vals)) if vals else float("nan")


def _median(vals: list[float]) -> float:
    vals = _finite(vals)
    return float(statistics.median(vals)) if vals else float("nan")


def _quantile(vals: list[float], q: float) -> float:
    vals = sorted(_finite(vals))
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def _diag_cost(src: torch.Tensor, gt: torch.Tensor, cost_fn: str) -> float:
    cost = compute_ot_cost(src.float(), gt.float(), cost_fn=cost_fn)
    return _f(torch.diagonal(cost, 0).mean().item())


def _pair_cost(src: torch.Tensor, gt: torch.Tensor, i: torch.Tensor, j: torch.Tensor, cost_fn: str) -> float:
    paired = compute_ot_cost(src[i].float(), gt[j].float(), cost_fn=cost_fn)
    return _f(torch.diagonal(paired, 0).mean().item())


def _delta_rel_error(src: torch.Tensor, gt: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> float:
    raw_delta = gt.float().mean(dim=0) - src.float().mean(dim=0)
    pair_delta = gt[j].float().mean(dim=0) - src[i].float().mean(dim=0)
    denom = max(raw_delta.norm().item(), 1e-8)
    return _f((pair_delta - raw_delta).norm().item() / denom)


def _hungarian_pair(src: torch.Tensor, gt: torch.Tensor, cost_fn: str) -> tuple[torch.Tensor, torch.Tensor]:
    cost = compute_ot_cost(src.float(), gt.float(), cost_fn=cost_fn)
    med = cost.median().clamp_min(1e-12)
    cost_np = (cost / med).cpu().numpy()
    row, col = linear_sum_assignment(cost_np)
    return (
        torch.as_tensor(row, device=src.device, dtype=torch.long),
        torch.as_tensor(col, device=src.device, dtype=torch.long),
    )


def _candidate_metrics(
    src: torch.Tensor,
    gt: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    *,
    identity_cost: float,
    cost_fn: str,
) -> dict[str, float]:
    b = int(src.shape[0])
    pair_cost = _pair_cost(src, gt, i, j, cost_fn)
    return {
        "cost": _f(pair_cost),
        "cost_delta_frac": _f((pair_cost - identity_cost) / max(identity_cost, 1e-8)),
        "unique_src_frac": _f(torch.unique(i).numel() / b),
        "unique_gt_frac": _f(torch.unique(j).numel() / b),
        "delta_rel_error": _delta_rel_error(src, gt, i, j),
    }


def _row(
    src: torch.Tensor,
    gt: torch.Tensor,
    *,
    ds_name: str,
    cond: str,
    cost_fn: str,
    reg: float,
    n_iter: int,
    seed: int,
    row_id: int,
) -> dict[str, Any]:
    b = int(src.shape[0])
    identity_cost = _diag_cost(src, gt, cost_fn)

    gen = torch.Generator(device=src.device)
    gen.manual_seed(int(seed) + int(row_id) * 1009)
    im, jm = sinkhorn_pair(
        src,
        gt,
        b,
        reg=reg,
        n_iter=n_iter,
        generator=gen,
        cost_fn=cost_fn,
        use_assignment=False,
    )
    ia, ja = sinkhorn_pair(
        src,
        gt,
        b,
        reg=reg,
        n_iter=n_iter,
        cost_fn=cost_fn,
        use_assignment=True,
    )
    ih, jh = _hungarian_pair(src, gt, cost_fn)

    return {
        "dataset": ds_name,
        "condition": cond,
        "batch_size": b,
        "cost_fn": cost_fn,
        "identity_cost": _f(identity_cost),
        "multinomial": _candidate_metrics(src, gt, im, jm, identity_cost=identity_cost, cost_fn=cost_fn),
        "sinkhorn_greedy_assignment": _candidate_metrics(src, gt, ia, ja, identity_cost=identity_cost, cost_fn=cost_fn),
        "hungarian_min_cost_assignment": _candidate_metrics(src, gt, ih, jh, identity_cost=identity_cost, cost_fn=cost_fn),
    }


def _summarize_candidate(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    fields = ["cost_delta_frac", "unique_src_frac", "unique_gt_frac", "delta_rel_error"]
    out: dict[str, Any] = {"n_batches": len(rows)}
    for field in fields:
        vals = [_f(row[name][field]) for row in rows]
        out[field] = {
            "mean": _mean(vals),
            "median": _median(vals),
            "p10": _quantile(vals, 0.10),
            "p90": _quantile(vals, 0.90),
        }
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(row)
    out["by_dataset"] = {
        ds: {
            "n_batches": len(ds_rows),
            "cost_delta_frac_mean": _mean([_f(r[name]["cost_delta_frac"]) for r in ds_rows]),
            "delta_rel_error_mean": _mean([_f(r[name]["delta_rel_error"]) for r in ds_rows]),
            "unique_gt_frac_mean": _mean([_f(r[name]["unique_gt_frac"]) for r in ds_rows]),
        }
        for ds, ds_rows in sorted(by_ds.items())
    }
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_batches": len(rows),
        "datasets": sorted({str(r["dataset"]) for r in rows}),
        "candidates": {
            "multinomial": _summarize_candidate(rows, "multinomial"),
            "sinkhorn_greedy_assignment": _summarize_candidate(rows, "sinkhorn_greedy_assignment"),
            "hungarian_min_cost_assignment": _summarize_candidate(rows, "hungarian_min_cost_assignment"),
        },
    }


def _decision(summary: dict[str, Any]) -> dict[str, Any]:
    candidates = summary["candidates"]
    greedy = candidates["sinkhorn_greedy_assignment"]
    hung = candidates["hungarian_min_cost_assignment"]
    greedy_reduction = max(0.0, -_f(greedy["cost_delta_frac"]["mean"]))
    hung_reduction = max(0.0, -_f(hung["cost_delta_frac"]["mean"]))
    coverage = len(summary.get("datasets") or [])
    reasons: list[str] = []
    if coverage < 10:
        reasons.append("insufficient_dataset_coverage")
    if _f(hung["unique_src_frac"]["mean"]) < 0.90 or _f(hung["unique_gt_frac"]["mean"]) < 0.90:
        reasons.append("hungarian_not_marginal_preserving")
    if _f(hung["delta_rel_error"]["mean"]) > 0.05:
        reasons.append("hungarian_mean_delta_error_above_gate")
    if greedy_reduction <= 0:
        reasons.append("greedy_assignment_has_no_reference_reduction")
    elif hung_reduction < 0.95 * greedy_reduction:
        reasons.append("hungarian_cost_reduction_not_within_5pct_of_greedy_assignment")
    status = "ot_marginal_preserving_reopen_gate_pass_no_gpu" if not reasons else "ot_marginal_preserving_reopen_gate_fail_no_gpu"
    return {
        "status": status,
        "reasons": reasons,
        "gate": {
            "datasets_min": 10,
            "unique_src_gt_mean_min": 0.90,
            "delta_rel_error_mean_max": 0.05,
            "cost_reduction_vs_greedy_assignment_min_ratio": 0.95,
        },
        "reference": {
            "greedy_assignment_cost_reduction_mean": greedy_reduction,
            "hungarian_cost_reduction_mean": hung_reduction,
        },
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    d = payload["decision"]
    lines = [
        "# LatentFM OT Marginal-Preserving Reopen Gate",
        "",
        "## Boundary",
        "",
        "- CPU-only train-split diagnostic.",
        "- No model training, checkpoint selection, canonical split, or Track C query.",
        "- Tests whether a stricter marginal-preserving assignment mechanism is worth later GPU implementation.",
        "",
        "## Decision",
        "",
        f"Status: `{d['status']}`",
        "",
    ]
    if d["reasons"]:
        lines.append("Reasons:")
        for reason in d["reasons"]:
            lines.append(f"- `{reason}`")
        lines.append("")
    lines.extend([
        "## Key Metrics",
        "",
        f"- batches audited: `{s['n_batches']}`",
        f"- datasets covered: `{len(s['datasets'])}`",
        "",
        "| candidate | cost delta frac mean | unique src mean | unique gt mean | delta rel err mean |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, row in s["candidates"].items():
        lines.append(
            f"| {name} | "
            f"{row['cost_delta_frac']['mean']:+.4f} | "
            f"{row['unique_src_frac']['mean']:.4f} | "
            f"{row['unique_gt_frac']['mean']:.4f} | "
            f"{row['delta_rel_error']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- A pass here is only permission to design a default-off GPU smoke after the running no-OT random control resolves.",
        "- It is not evidence that OT improves Track A metrics.",
        "- A fail closes marginal-preserving OT redesign until a stronger CPU mechanism is proposed.",
        "",
        "## JSON",
        "",
        f"`{payload['out_json']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "dataset/latentfm_full/xverse"))
    ap.add_argument("--split-file", default=str(ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-batches", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reg", type=float, default=0.05)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--cost-fn", default="l2")
    ap.add_argument("--out-json", default=str(ROOT / "reports/latentfm_ot_marginal_preserving_reopen_gate_20260624.json"))
    ap.add_argument("--out-md", default=str(ROOT / "reports/LATENTFM_OT_MARGINAL_PRESERVING_REOPEN_GATE_20260624.md"))
    args = ap.parse_args()

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    dataset = CrossDatasetFMDataset(
        args.data_dir,
        split,
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        mode="train",
        min_cells=16,
        ds_alpha=0.7,
        scale_noise=0.0,
        latent_backbone="xverse",
        use_pert_condition=False,
        silent=True,
    )

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        if idx >= int(args.max_batches):
            break
        src, gt, ds_name, cond, *_ = item
        rows.append(
            _row(
                src.float(),
                gt.float(),
                ds_name=str(ds_name),
                cond=str(cond),
                cost_fn=str(args.cost_fn),
                reg=float(args.reg),
                n_iter=int(args.n_iter),
                seed=int(args.seed),
                row_id=idx,
            )
        )

    summary = _summarize(rows)
    payload = {
        "boundary": {
            "data_dir": str(args.data_dir),
            "split_file": str(args.split_file),
            "batch_size": int(args.batch_size),
            "max_batches": int(args.max_batches),
            "seed": int(args.seed),
            "reg": float(args.reg),
            "n_iter": int(args.n_iter),
            "cost_fn": str(args.cost_fn),
            "no_training": True,
            "no_canonical_or_query": True,
        },
        "summary": summary,
        "decision": _decision(summary),
        "rows": rows,
        "out_json": str(args.out_json),
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_md(out_md, payload)
    print(json.dumps({"out_md": str(out_md), "out_json": str(out_json), "status": payload["decision"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
