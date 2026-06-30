#!/usr/bin/env python3
"""CPU audit for LatentFM mini-batch OT pairing behavior.

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
from model.utils.data.ot_pairer import sinkhorn_pair  # noqa: E402


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


def _sq_cost(src: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (src.float() - gt.float()).pow(2).sum(dim=1)


def _row(
    src: torch.Tensor,
    gt: torch.Tensor,
    *,
    ds_name: str,
    cond: str,
    reg: float,
    n_iter: int,
    seed: int,
    row_id: int,
) -> dict[str, Any]:
    b = int(src.shape[0])
    gen = torch.Generator(device=src.device)
    gen.manual_seed(int(seed) + int(row_id) * 1009)
    im, jm = sinkhorn_pair(src, gt, b, reg=reg, n_iter=n_iter, generator=gen, use_assignment=False)
    ia, ja = sinkhorn_pair(src, gt, b, reg=reg, n_iter=n_iter, use_assignment=True)

    identity_cost = _sq_cost(src, gt).mean().item()
    mult_cost = _sq_cost(src[im], gt[jm]).mean().item()
    assign_cost = _sq_cost(src[ia], gt[ja]).mean().item()
    raw_delta = gt.float().mean(dim=0) - src.float().mean(dim=0)
    raw_norm = raw_delta.norm().item()
    mult_delta = gt[jm].float().mean(dim=0) - src[im].float().mean(dim=0)
    assign_delta = gt[ja].float().mean(dim=0) - src[ia].float().mean(dim=0)
    denom = max(raw_norm, 1e-8)
    return {
        "dataset": ds_name,
        "condition": cond,
        "batch_size": b,
        "identity_cost": _f(identity_cost),
        "sinkhorn_multinomial_cost": _f(mult_cost),
        "sinkhorn_assignment_cost": _f(assign_cost),
        "multinomial_cost_delta_frac": _f((mult_cost - identity_cost) / max(identity_cost, 1e-8)),
        "assignment_cost_delta_frac": _f((assign_cost - identity_cost) / max(identity_cost, 1e-8)),
        "multinomial_unique_src_frac": _f(torch.unique(im).numel() / b),
        "multinomial_unique_gt_frac": _f(torch.unique(jm).numel() / b),
        "assignment_unique_src_frac": _f(torch.unique(ia).numel() / b),
        "assignment_unique_gt_frac": _f(torch.unique(ja).numel() / b),
        "raw_delta_norm": _f(raw_norm),
        "multinomial_delta_rel_error": _f((mult_delta - raw_delta).norm().item() / denom),
        "assignment_delta_rel_error": _f((assign_delta - raw_delta).norm().item() / denom),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "identity_cost",
        "sinkhorn_multinomial_cost",
        "sinkhorn_assignment_cost",
        "multinomial_cost_delta_frac",
        "assignment_cost_delta_frac",
        "multinomial_unique_src_frac",
        "multinomial_unique_gt_frac",
        "assignment_unique_src_frac",
        "assignment_unique_gt_frac",
        "multinomial_delta_rel_error",
        "assignment_delta_rel_error",
    ]
    out: dict[str, Any] = {"n_batches": len(rows)}
    for field in fields:
        vals = [_f(r.get(field)) for r in rows]
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
            "multinomial_cost_delta_frac_mean": _mean([_f(r["multinomial_cost_delta_frac"]) for r in ds_rows]),
            "multinomial_delta_rel_error_mean": _mean([_f(r["multinomial_delta_rel_error"]) for r in ds_rows]),
            "multinomial_unique_gt_frac_mean": _mean([_f(r["multinomial_unique_gt_frac"]) for r in ds_rows]),
        }
        for ds, ds_rows in sorted(by_ds.items())
    }
    return out


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# LatentFM OT Pairing Signal Audit",
        "",
        "## Boundary",
        "",
        "- CPU-only diagnostic over the train-only split.",
        "- No model training, no checkpoint selection, no canonical/test/query artifacts.",
        "- Compares current GPU-path semantics (`torch_sinkhorn` multinomial plan sampling) against identity random pairing and greedy assignment from the same mini-batch plan.",
        "",
        "## Key Results",
        "",
        f"- batches audited: `{s['n_batches']}`",
        f"- multinomial cost delta frac mean/median: `{s['multinomial_cost_delta_frac']['mean']:+.4f}` / `{s['multinomial_cost_delta_frac']['median']:+.4f}`",
        f"- assignment cost delta frac mean/median: `{s['assignment_cost_delta_frac']['mean']:+.4f}` / `{s['assignment_cost_delta_frac']['median']:+.4f}`",
        f"- multinomial unique GT frac mean/median: `{s['multinomial_unique_gt_frac']['mean']:.4f}` / `{s['multinomial_unique_gt_frac']['median']:.4f}`",
        f"- multinomial delta relative error mean/median: `{s['multinomial_delta_rel_error']['mean']:.4f}` / `{s['multinomial_delta_rel_error']['median']:.4f}`",
        f"- assignment delta relative error mean/median: `{s['assignment_delta_rel_error']['mean']:.4f}` / `{s['assignment_delta_rel_error']['median']:.4f}`",
        "",
        "## Interpretation",
        "",
        "- If Sinkhorn cost deltas are strongly negative, OT materially changes the per-cell path coupling.",
        "- If multinomial uniqueness is low or delta relative error is high, current replacement sampling may inject extra mean-delta noise even when the transport plan is useful.",
        "- Assignment preserves one-to-one mini-batch marginals when batch sizes match; it is a plausible capped ablation if multinomial noise looks large.",
        "",
        "## Dataset Rows",
        "",
        "| dataset | n | mult cost frac mean | mult delta err mean | mult unique GT mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, row in (s.get("by_dataset") or {}).items():
        lines.append(
            f"| {ds} | {row['n_batches']} | "
            f"{row['multinomial_cost_delta_frac_mean']:+.4f} | "
            f"{row['multinomial_delta_rel_error_mean']:.4f} | "
            f"{row['multinomial_unique_gt_frac_mean']:.4f} |"
        )
    lines.extend([
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
    ap.add_argument("--max-batches", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reg", type=float, default=0.05)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--out-json", default=str(ROOT / "reports/latentfm_ot_pairing_signal_audit_20260624.json"))
    ap.add_argument("--out-md", default=str(ROOT / "reports/LATENTFM_OT_PAIRING_SIGNAL_AUDIT_20260624.md"))
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
                reg=float(args.reg),
                n_iter=int(args.n_iter),
                seed=int(args.seed),
                row_id=idx,
            )
        )

    payload = {
        "boundary": {
            "data_dir": str(args.data_dir),
            "split_file": str(args.split_file),
            "batch_size": int(args.batch_size),
            "max_batches": int(args.max_batches),
            "seed": int(args.seed),
            "reg": float(args.reg),
            "n_iter": int(args.n_iter),
            "no_training": True,
            "no_canonical_or_query": True,
        },
        "summary": _summarize(rows),
        "rows": rows,
        "out_json": str(args.out_json),
    }
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_md(out_md, payload)
    print(json.dumps({"out_md": str(out_md), "out_json": str(out_json), "n_batches": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
