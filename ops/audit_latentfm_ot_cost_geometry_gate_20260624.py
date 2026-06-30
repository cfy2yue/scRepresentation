#!/usr/bin/env python3
"""CPU gate for LatentFM OT cost-geometry variants.

This is a train-split-only diagnostic. It does not train, select checkpoints,
or read canonical/query evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
import sys

sys.path.insert(0, str(COUPLED))

from model.latent.dataset import CrossDatasetFMDataset  # noqa: E402
from model.utils.data.ot_pairer import compute_ot_cost, sinkhorn_pair  # noqa: E402


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.fmean(vals)) if vals else float("nan")


def _median(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.median(vals)) if vals else float("nan")


def _quantile(vals: list[float], q: float) -> float:
    vals = sorted(v for v in vals if math.isfinite(v))
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


def _delta_rel_error(src: torch.Tensor, gt: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> float:
    raw_delta = gt.float().mean(dim=0) - src.float().mean(dim=0)
    pair_delta = gt[j].float().mean(dim=0) - src[i].float().mean(dim=0)
    denom = max(raw_delta.norm().item(), 1e-8)
    return _f((pair_delta - raw_delta).norm().item() / denom)


def _row_for_cost(
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
    gen = torch.Generator(device=src.device)
    gen.manual_seed(int(seed) + int(row_id) * 1009 + sum(ord(c) for c in cost_fn))
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
    identity_cost = _diag_cost(src, gt, cost_fn)
    mult_cost = _diag_cost(src[im], gt[jm], cost_fn)
    assign_cost = _diag_cost(src[ia], gt[ja], cost_fn)
    denom = max(identity_cost, 1e-8)
    return {
        "dataset": ds_name,
        "condition": cond,
        "batch_size": b,
        "cost_fn": cost_fn,
        "identity_cost": _f(identity_cost),
        "multinomial_cost": _f(mult_cost),
        "assignment_cost": _f(assign_cost),
        "multinomial_cost_delta_frac": _f((mult_cost - identity_cost) / denom),
        "assignment_cost_delta_frac": _f((assign_cost - identity_cost) / denom),
        "multinomial_unique_src_frac": _f(torch.unique(im).numel() / b),
        "multinomial_unique_gt_frac": _f(torch.unique(jm).numel() / b),
        "assignment_unique_src_frac": _f(torch.unique(ia).numel() / b),
        "assignment_unique_gt_frac": _f(torch.unique(ja).numel() / b),
        "multinomial_delta_rel_error": _delta_rel_error(src, gt, im, jm),
        "assignment_delta_rel_error": _delta_rel_error(src, gt, ia, ja),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "multinomial_cost_delta_frac",
        "assignment_cost_delta_frac",
        "multinomial_unique_gt_frac",
        "assignment_unique_gt_frac",
        "multinomial_delta_rel_error",
        "assignment_delta_rel_error",
    ]
    by_cost: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cost[str(row["cost_fn"])].append(row)
    out: dict[str, Any] = {"n_rows": len(rows), "by_cost_fn": {}}
    for cost_fn, cost_rows in sorted(by_cost.items()):
        entry: dict[str, Any] = {"n_batches": len(cost_rows)}
        for field in fields:
            vals = [_f(r.get(field)) for r in cost_rows]
            entry[field] = {
                "mean": _mean(vals),
                "median": _median(vals),
                "p10": _quantile(vals, 0.10),
                "p90": _quantile(vals, 0.90),
            }
        out["by_cost_fn"][cost_fn] = entry
    return out


def _candidate(summary: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for cost_fn, entry in (summary.get("by_cost_fn") or {}).items():
        assign_cost = entry["assignment_cost_delta_frac"]["mean"]
        assign_delta_err = entry["assignment_delta_rel_error"]["mean"]
        mult_delta_err = entry["multinomial_delta_rel_error"]["mean"]
        rows.append(
            {
                "cost_fn": cost_fn,
                "assignment_cost_delta_frac_mean": assign_cost,
                "assignment_delta_rel_error_mean": assign_delta_err,
                "multinomial_delta_rel_error_mean": mult_delta_err,
                "mechanism_score": assign_cost + 0.25 * assign_delta_err,
            }
        )
    rows = sorted(rows, key=lambda r: r["mechanism_score"])
    best = rows[0] if rows else None
    if best is None:
        return {"status": "fail_no_rows", "best": None, "rows": rows}
    l2 = next((r for r in rows if r["cost_fn"] == "l2"), None)
    reasons = []
    if best["cost_fn"] == "l2":
        reasons.append("best_cost_fn_is_existing_l2")
    if best["assignment_cost_delta_frac_mean"] > -0.20:
        reasons.append("weak_assignment_cost_reduction")
    if best["assignment_delta_rel_error_mean"] > 0.05:
        reasons.append("assignment_mean_delta_not_preserved")
    if l2 and best["mechanism_score"] > l2["mechanism_score"] - 0.02:
        reasons.append("best_not_materially_better_than_l2_assignment")
    status = "ot_cost_geometry_reopen_gate_pass" if not reasons else "ot_cost_geometry_reopen_gate_fail"
    return {"status": status, "best": best, "rows": rows, "reasons": reasons}


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM OT Cost-Geometry Gate",
        "",
        "## Boundary",
        "",
        "- CPU-only diagnostic over the train-only split.",
        "- No model training, no checkpoint selection, no canonical/test/query artifacts.",
        "- Tests whether any cost geometry is strong enough to justify exposing `ot_cost_fn` for a future default-off GPU smoke.",
        "",
        "## Decision",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
    ]
    if payload["decision"].get("reasons"):
        lines.append("Reasons:")
        for reason in payload["decision"]["reasons"]:
            lines.append(f"- `{reason}`")
        lines.append("")
    lines.extend([
        "## Rows",
        "",
        "| cost_fn | n | assignment cost delta | assignment delta err | multinomial cost delta | multinomial delta err |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for cost_fn, entry in (payload["summary"].get("by_cost_fn") or {}).items():
        lines.append(
            f"| {cost_fn} | {entry['n_batches']} | "
            f"{entry['assignment_cost_delta_frac']['mean']:+.4f} | "
            f"{entry['assignment_delta_rel_error']['mean']:.4f} | "
            f"{entry['multinomial_cost_delta_frac']['mean']:+.4f} | "
            f"{entry['multinomial_delta_rel_error']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- A pass here is only permission to consider one capped GPU smoke; it is not model evidence.",
        "- A fail keeps OT demoted while Track A scaling/training-data branches proceed.",
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
    ap.add_argument("--cost-fns", nargs="+", default=["l2", "cosine", "zscore_l2", "rank_l2"])
    ap.add_argument("--out-json", default=str(ROOT / "reports/latentfm_ot_cost_geometry_gate_20260624.json"))
    ap.add_argument("--out-md", default=str(ROOT / "reports/LATENTFM_OT_COST_GEOMETRY_GATE_20260624.md"))
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
        src_t = src.float()
        gt_t = gt.float()
        for cost_fn in args.cost_fns:
            rows.append(
                _row_for_cost(
                    src_t,
                    gt_t,
                    ds_name=str(ds_name),
                    cond=str(cond),
                    cost_fn=str(cost_fn),
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
            "cost_fns": list(map(str, args.cost_fns)),
            "no_training": True,
            "no_canonical_or_query": True,
        },
        "summary": summary,
        "decision": _candidate(summary),
        "rows": rows,
        "out_json": str(args.out_json),
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_md(out_md, payload)
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
