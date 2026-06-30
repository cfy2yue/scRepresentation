#!/usr/bin/env python3
"""Condition-overlap OT reliability gate for LatentFM.

CPU-only. Recomputes OT pair-quality features on the exact internal validation
conditions used by the Track A anchor error map, then checks whether pair
quality predicts internal failures strongly enough to reopen OT. No training,
inference, canonical multi selection, Track C query, or GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
MANIFEST = DATA_DIR / "manifest.json"
ERROR_MAP = ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json"
OUT_DIR = ROOT / "reports/ot_condition_overlap_reliability_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_ot_condition_overlap_reliability_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_OT_CONDITION_OVERLAP_RELIABILITY_GATE_20260627.md"
OUT_ROWS = OUT_DIR / "ot_condition_overlap_rows.csv"
SEED = 424242


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return pearson(rank(xs), rank(ys))


def read_batch(manifest: dict[str, Any], ds: str, cond: str, *, seed: int, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor] | None:
    h5_path = Path(manifest["datasets"][ds]["out_path"])
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as handle:
        conds = [decode(x) for x in handle["conditions"][()]]
        if cond not in conds:
            return None
        idx = conds.index(cond)
        gt_offsets = [int(x) for x in handle["gt/offsets"][()]]
        ctrl_offsets = [int(x) for x in handle["ctrl/offsets"][()]]
        gt_start, gt_end = gt_offsets[idx], gt_offsets[idx + 1]
        ctrl_start, ctrl_end = ctrl_offsets[idx], ctrl_offsets[idx + 1]
        n_gt = gt_end - gt_start
        n_ctrl = ctrl_end - ctrl_start
        n = min(batch_size, n_gt, n_ctrl)
        if n < 16:
            return None
        gt_idx = np.sort(rng.choice(n_gt, size=n, replace=False) + gt_start)
        ctrl_idx = np.sort(rng.choice(n_ctrl, size=n, replace=False) + ctrl_start)
        gt = np.asarray(handle["gt/emb"][gt_idx], dtype=np.float32)
        ctrl = np.asarray(handle["ctrl/emb"][ctrl_idx], dtype=np.float32)
    return torch.from_numpy(ctrl), torch.from_numpy(gt)


def dup_rate(idx: torch.Tensor) -> float:
    n = int(idx.numel())
    return math.nan if n == 0 else 1.0 - (int(torch.unique(idx).numel()) / float(n))


def summarize_pairing(src: torch.Tensor, gt: torch.Tensor, *, seed: int) -> dict[str, Any]:
    sys.path.insert(0, str(COUPLED))
    from model.utils.data.ot_pairer import compute_ot_cost, hungarian_pair, sinkhorn_pair

    torch.manual_seed(seed)
    n = int(src.shape[0])
    cost = compute_ot_cost(src.float(), gt.float())
    baseline_diag = cost[torch.arange(n), torch.arange(n)]
    out: dict[str, Any] = {
        "n": n,
        "full_cost_mean": float(cost.mean().item()),
        "random_index_cost_mean": float(baseline_diag.mean().item()),
    }
    for mode in ("multinomial", "assignment", "hungarian"):
        if mode == "hungarian":
            i, j = hungarian_pair(src.float(), gt.float(), n_samples=n)
        else:
            i, j = sinkhorn_pair(
                src.float(),
                gt.float(),
                n_samples=n,
                reg=0.05,
                n_iter=30,
                use_assignment=(mode == "assignment"),
            )
        paired = cost[i, j]
        out[f"{mode}_paired_cost_mean"] = float(paired.mean().item())
        out[f"{mode}_cost_delta_vs_random_index"] = float(paired.mean().item() - baseline_diag.mean().item())
        out[f"{mode}_src_duplicate_rate"] = dup_rate(i.cpu())
        out[f"{mode}_gt_duplicate_rate"] = dup_rate(j.cpu())
    out["assignment_minus_multinomial_cost"] = out["assignment_paired_cost_mean"] - out["multinomial_paired_cost_mean"]
    out["hungarian_minus_multinomial_cost"] = out["hungarian_paired_cost_mean"] - out["multinomial_paired_cost_mean"]
    out["multinomial_mean_dup_rate"] = (out["multinomial_src_duplicate_rate"] + out["multinomial_gt_duplicate_rate"]) / 2.0
    return out


def finite_pairs(rows: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = to_float(row.get(x_key))
        y = to_float(row.get(y_key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def within_dataset_shuffle_p(rows: list[dict[str, Any]], x_key: str, y_key: str, observed: float, *, n_perm: int = 1000) -> float | None:
    rng = random.Random(SEED)
    base_x = [float(row[x_key]) for row in rows]
    base_y = [float(row[y_key]) for row in rows]
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_dataset[str(row["dataset"])].append(i)
    valid = 0
    ge = 0
    for _ in range(n_perm):
        xs = list(base_x)
        for idxs in by_dataset.values():
            vals = [xs[i] for i in idxs]
            rng.shuffle(vals)
            for i, val in zip(idxs, vals):
                xs[i] = val
        rho = spearman(xs, base_y)
        if rho is None:
            continue
        valid += 1
        if abs(rho) >= abs(observed):
            ge += 1
    return (ge + 1) / (valid + 1) if valid else None


def evaluate_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "multinomial_paired_cost_mean",
        "multinomial_mean_dup_rate",
        "assignment_minus_multinomial_cost",
        "hungarian_minus_multinomial_cost",
        "full_cost_mean",
        "random_index_cost_mean",
        "gene_train_count",
    ]
    out = []
    for feat in features:
        xs, bad = finite_pairs(rows, feat, "bad_pp")
        rho = spearman(xs, bad)
        mx, my = finite_pairs(rows, feat, "anchor_mmd_clamped")
        mmd_rho = spearman(mx, my)
        p = within_dataset_shuffle_p(rows, feat, "bad_pp", rho) if rho is not None else None
        ds_rhos = {}
        for ds in sorted({str(row["dataset"]) for row in rows}):
            sub = [row for row in rows if row["dataset"] == ds]
            sx, sy = finite_pairs(sub, feat, "bad_pp")
            r = spearman(sx, sy)
            if r is not None:
                ds_rhos[ds] = r
        out.append(
            {
                "feature": feat,
                "n": len(xs),
                "rho_bad_pp": rho,
                "abs_rho_bad_pp": None if rho is None else abs(rho),
                "shuffle_p_abs": p,
                "rho_mmd": mmd_rho,
                "abs_rho_mmd": None if mmd_rho is None else abs(mmd_rho),
                "dataset_min_abs_rho": min((abs(v) for v in ds_rhos.values()), default=None),
                "dataset_rhos": ds_rhos,
            }
        )
    return sorted(out, key=lambda r: (r["abs_rho_bad_pp"] is not None, r["abs_rho_bad_pp"] or -1), reverse=True)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM OT Condition-Overlap Reliability Gate 2026-06-27",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only OT pair-quality recomputation on internal validation/error-map conditions.",
        "- No training, inference, canonical multi selection, Track C query, or GPU.",
        "",
        "## Coverage",
        "",
        f"- requested internal rows: `{payload['requested_rows']}`",
        f"- computed pair-quality rows: `{payload['computed_rows']}`",
        f"- datasets: `{payload['datasets']}`",
        "",
        "## Feature Correlations",
        "",
        "| feature | n | rho vs bad pp | shuffle p(abs) | rho vs MMD | dataset min abs rho |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["feature_summary"]:
        lines.append(
            f"| `{row['feature']}` | {row['n']} | {fmt(row['rho_bad_pp'])} | {fmt(row['shuffle_p_abs'])} | {fmt(row['rho_mmd'])} | {fmt(row['dataset_min_abs_rho'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- best feature: `{payload['best_feature']['feature'] if payload['best_feature'] else 'NA'}`",
        f"- reasons: `{payload['reasons']}`",
        "- Existing OT GPU pair-mode evidence remains binding unless this gate passes and an external audit approves a single default-off intervention.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_ROWS}`",
        "",
    ]
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:+.6f}"
    return str(value)


def write_rows(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "group",
        "dataset",
        "condition",
        "gene",
        "gene_train_count",
        "anchor_pearson_pert",
        "bad_pp",
        "anchor_mmd_clamped",
        "multinomial_paired_cost_mean",
        "multinomial_mean_dup_rate",
        "assignment_minus_multinomial_cost",
        "hungarian_minus_multinomial_cost",
        "full_cost_mean",
        "random_index_cost_mean",
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    manifest = load_json(MANIFEST)
    error_map = load_json(ERROR_MAP)
    requested = [
        row for row in error_map.get("condition_rows", [])
        if row.get("group") == "internal_val_cross_background_seen_gene_proxy"
    ]
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(requested):
        ds = str(rec["dataset"])
        cond = str(rec["condition"])
        batch = read_batch(manifest, ds, cond, seed=SEED + idx)
        if batch is None:
            continue
        src, gt = batch
        pq = summarize_pairing(src, gt, seed=SEED + idx)
        out = {
            "group": rec.get("group"),
            "dataset": ds,
            "condition": cond,
            "gene": rec.get("gene"),
            "gene_train_count": to_float(rec.get("gene_train_count")),
            "anchor_pearson_pert": to_float(rec.get("anchor_pearson_pert")),
            "bad_pp": -float(rec.get("anchor_pearson_pert")),
            "anchor_mmd_clamped": to_float(rec.get("anchor_mmd_clamped")),
        }
        out.update(pq)
        rows.append(out)
    feature_summary = evaluate_features(rows)
    best = feature_summary[0] if feature_summary else None
    reasons: list[str] = []
    if len(rows) < 80:
        reasons.append("computed_condition_overlap_below_80")
    if not best or (best.get("abs_rho_bad_pp") or 0.0) < 0.35:
        reasons.append("best_pair_quality_abs_rho_below_0p35")
    if not best or best.get("shuffle_p_abs") is None or best["shuffle_p_abs"] > 0.05:
        reasons.append("best_pair_quality_shuffle_p_gt_0p05")
    if best and best.get("abs_rho_mmd") is not None and best["abs_rho_mmd"] >= (best.get("abs_rho_bad_pp") or 0.0):
        reasons.append("mmd_correlation_matches_or_exceeds_bad_pp_signal")
    # Full/random cost and gene count are controls; if they tie the best
    # pair-quality feature, the signal is not specific enough to reopen OT.
    control_features = {"full_cost_mean", "random_index_cost_mean", "gene_train_count"}
    best_pair_rho = max(
        (row["abs_rho_bad_pp"] or 0.0 for row in feature_summary if row["feature"] not in control_features),
        default=0.0,
    )
    best_control_rho = max(
        (row["abs_rho_bad_pp"] or 0.0 for row in feature_summary if row["feature"] in control_features),
        default=0.0,
    )
    if best_control_rho >= best_pair_rho - 0.03:
        reasons.append("control_feature_matches_pair_quality_signal")
    status = "ot_condition_overlap_reliability_gate_fail_no_gpu" if reasons else "ot_condition_overlap_reliability_gate_pass_external_audit_next_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "requested_rows": len(requested),
        "computed_rows": len(rows),
        "datasets": len({row["dataset"] for row in rows}),
        "feature_summary": feature_summary,
        "best_feature": best,
        "reasons": reasons,
        "boundary": {
            "cpu_only": True,
            "internal_error_map": str(ERROR_MAP),
            "canonical_multi_selection": False,
            "trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
    }
    write_rows(rows)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
