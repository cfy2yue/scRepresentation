#!/usr/bin/env python3
"""Control-state support geometry v2 gate.

CPU/report-only. Computes dataset-level control-state geometry summaries from
local control-center h5ad files and tests whether they explain train/internal
tail risk. This is a bounded preflight; it does not train, infer, select
checkpoints, read canonical multi for selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse


ROOT = Path("/data/cyx/1030/scLatent")
CONTROL_DIR = ROOT / "dataset/Training_data/scfoundation/control_center_scfoundation"
LATENT_DIR = ROOT / "dataset/latentfm_staging/scfm_embeddings/stack"
TAIL_ROWS = ROOT / "reports/train_internal_recurrent_tail_analogue_gate_20260627/train_internal_recurrent_tail_analogue_rows.csv"
OUT_DIR = ROOT / "reports/control_state_support_geometry_v2_gate_20260627"
OUT_GEOM = OUT_DIR / "control_state_geometry_rows.csv"
OUT_JOIN = OUT_DIR / "control_state_geometry_internal_tail_join.csv"
OUT_JSON = ROOT / "reports/latentfm_control_state_support_geometry_v2_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_CONTROL_STATE_SUPPORT_GEOMETRY_V2_GATE_20260627.md"
SEED = 20260627


def norm(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def fnum(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys)) if len(xs) >= 3 else None


def entropy(labels: list[str]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values() if count)


def summarize_h5ad(path: Path) -> dict[str, Any]:
    dataset = path.stem
    adata = ad.read_h5ad(path)
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    n_obs, n_vars = X.shape
    mean_vec = X.mean(axis=0)
    centered = X - mean_vec
    row_radius = np.sqrt(np.maximum(0.0, np.sum(centered * centered, axis=1)))
    # A small SVD on control-only files is bounded; files are <9MB each.
    sv = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    sv2 = sv * sv
    eff_rank = float((sv2.sum() ** 2) / np.sum(sv2 * sv2)) if np.sum(sv2 * sv2) > 0 else 0.0
    latent_path = LATENT_DIR / dataset / "raw/latent.npy"
    if not latent_path.exists() and dataset.startswith("sciplex3_"):
        latent_path = LATENT_DIR / "sciplex3_xCellLine/raw/latent.npy"
    obs = adata.obs
    cluster_labels = [norm(v) for v in obs["cluster"].tolist()] if "cluster" in obs.columns else []
    return {
        "dataset": dataset,
        "control_cells": n_obs,
        "n_vars": n_vars,
        "radius_mean": float(np.mean(row_radius)) if len(row_radius) else None,
        "radius_p95": float(np.percentile(row_radius, 95)) if len(row_radius) else None,
        "radius_cv": float(np.std(row_radius) / np.mean(row_radius)) if len(row_radius) and np.mean(row_radius) else None,
        "effective_rank": eff_rank,
        "cluster_entropy": entropy(cluster_labels),
        "n_clusters": len(set(cluster_labels)) if cluster_labels else 0,
        "total_counts_mean": float(obs["total_counts"].mean()) if "total_counts" in obs.columns else None,
        "n_genes_by_counts_mean": float(obs["n_genes_by_counts"].mean()) if "n_genes_by_counts" in obs.columns else None,
        "latent_available": latent_path.exists(),
        "latent_path": str(latent_path) if latent_path.exists() else "",
    }


def load_tail_by_dataset() -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(TAIL_ROWS):
        grouped.setdefault(norm(row.get("dataset")), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for dataset, rows in grouped.items():
        bad = [fnum(row.get("bad_pp_mean")) for row in rows]
        bad = [float(v) for v in bad if v is not None]
        mmd = [fnum(row.get("mmd_max")) for row in rows]
        mmd = [float(v) for v in mmd if v is not None]
        hard = [norm(row.get("internal_recurrent_hard_tail")).lower() == "true" for row in rows]
        if bad:
            out[dataset] = {
                "dataset": dataset,
                "internal_rows": len(rows),
                "bad_pp_mean": mean(bad),
                "bad_pp_p90": sorted(bad)[int(0.9 * (len(bad) - 1))],
                "hard_tail_frac": sum(hard) / len(hard) if hard else 0.0,
                "mmd_mean": mean(mmd) if mmd else None,
            }
    return out


def permutation_p(xs: list[float], ys: list[float], actual: float, *, n_perm: int = 1000) -> float | None:
    rng = random.Random(SEED)
    hits = 0
    total = 0
    for _ in range(n_perm):
        shuffled = xs[:]
        rng.shuffle(shuffled)
        rho = spearman(shuffled, ys)
        if rho is None:
            continue
        total += 1
        if abs(rho) >= abs(actual):
            hits += 1
    return (hits + 1) / (total + 1) if total else None


def lodo_min(rows: list[dict[str, Any]], feature: str, target: str) -> float | None:
    vals = []
    for dataset in sorted({row["dataset"] for row in rows}):
        sub = [row for row in rows if row["dataset"] != dataset]
        xs = [float(row[feature]) for row in sub if isinstance(row.get(feature), (int, float)) and isinstance(row.get(target), (int, float))]
        ys = [float(row[target]) for row in sub if isinstance(row.get(feature), (int, float)) and isinstance(row.get(target), (int, float))]
        rho = spearman(xs, ys)
        if rho is not None:
            vals.append(rho)
    return min(vals) if vals else None


def main() -> None:
    geom_rows = [summarize_h5ad(path) for path in sorted(CONTROL_DIR.glob("*.h5ad"))]
    tail = load_tail_by_dataset()
    join_rows: list[dict[str, Any]] = []
    for row in geom_rows:
        trec = tail.get(row["dataset"])
        if trec:
            join_rows.append({**row, **trec})
    features = [
        "control_cells",
        "radius_mean",
        "radius_p95",
        "radius_cv",
        "effective_rank",
        "cluster_entropy",
        "n_clusters",
        "total_counts_mean",
        "n_genes_by_counts_mean",
    ]
    feature_results: list[dict[str, Any]] = []
    for feature in features:
        pairs = [(float(row[feature]), float(row["bad_pp_mean"])) for row in join_rows if isinstance(row.get(feature), (int, float))]
        if len(pairs) < 5:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        rho = spearman(xs, ys)
        if rho is None:
            continue
        mmd_pairs = [(float(row[feature]), float(row["mmd_mean"])) for row in join_rows if isinstance(row.get(feature), (int, float)) and isinstance(row.get("mmd_mean"), (int, float))]
        mmd_rho = spearman([p[0] for p in mmd_pairs], [p[1] for p in mmd_pairs]) if len(mmd_pairs) >= 3 else None
        feature_results.append(
            {
                "feature": feature,
                "n": len(pairs),
                "rho_bad_pp": rho,
                "shuffle_p_abs": permutation_p(xs, ys, rho),
                "lodo_min": lodo_min(join_rows, feature, "bad_pp_mean"),
                "abs_mmd_rho": None if mmd_rho is None else abs(mmd_rho),
            }
        )
    feature_results.sort(key=lambda row: -abs(float(row["rho_bad_pp"])))
    best = feature_results[0] if feature_results else None
    reasons = []
    if len(join_rows) < 8:
        reasons.append("joined_datasets_below_8")
    if best is None:
        reasons.append("no_feature_signal")
    else:
        if abs(float(best["rho_bad_pp"])) < 0.25:
            reasons.append("best_abs_rho_below_0p25")
        if best.get("lodo_min") is None or abs(float(best["lodo_min"])) < 0.10:
            reasons.append("lodo_abs_min_below_0p10")
        if best.get("shuffle_p_abs") is None or float(best["shuffle_p_abs"]) > 0.01:
            reasons.append("shuffle_p_abs_gt_0p01")
        if best.get("abs_mmd_rho") is not None and float(best["abs_mmd_rho"]) >= 0.15:
            reasons.append("mmd_abs_rho_ge_0p15")
    status = "control_state_support_geometry_v2_pass_needs_external_audit_no_gpu" if not reasons else "control_state_support_geometry_v2_fail_no_gpu"

    write_csv(OUT_GEOM, geom_rows, list(geom_rows[0].keys()) if geom_rows else ["dataset"])
    write_csv(
        OUT_JOIN,
        join_rows,
        list(join_rows[0].keys()) if join_rows else ["dataset"],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "n_geometry_datasets": len(geom_rows),
        "n_joined_datasets": len(join_rows),
        "best_feature": best,
        "feature_results": feature_results,
        "reasons": reasons,
        "outputs": {"geometry": str(OUT_GEOM), "join": str(OUT_JOIN), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = []
    for row in feature_results[:10]:
        lines.append(
            "| {feature} | {n} | {rho} | {sp} | {lodo} | {mmd} |".format(
                feature=row["feature"],
                n=row["n"],
                rho=f"{row['rho_bad_pp']:+.6f}",
                sp=f"{row['shuffle_p_abs']:.6f}" if isinstance(row.get("shuffle_p_abs"), float) else "NA",
                lodo=f"{row['lodo_min']:+.6f}" if isinstance(row.get("lodo_min"), float) else "NA",
                mmd=f"{row['abs_mmd_rho']:+.6f}" if isinstance(row.get("abs_mmd_rho"), float) else "NA",
            )
        )
    md = f"""# LatentFM Control-State Support Geometry V2 Gate 2026-06-27

Timestamp: `{payload['timestamp']}`

Status: `{status}`

GPU authorized: `False`

## Boundary

- CPU/report-only control-state geometry gate over local control-center h5ad
  files and train/internal recurrent-tail analogue rows.
- Latent stack availability is recorded as provenance; this bounded preflight
  does not load full latent matrices.
- No training, inference, checkpoint selection, canonical multi selection,
  Track C query, or GPU.

## Coverage

- control geometry datasets: `{len(geom_rows)}`
- joined internal-tail datasets: `{len(join_rows)}`
- reasons: `{reasons}`

## Feature Signals

| feature | n | rho vs dataset bad pp | shuffle p(abs) | LODO min | abs MMD rho |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

## Decision

Control-state geometry does not authorize GPU unless it explains internal tail
risk robustly after shuffle/LODO and without MMD coupling. Current status:
`{status}`.

## Outputs

- JSON: `{OUT_JSON}`
- geometry: `{OUT_GEOM}`
- joined rows: `{OUT_JOIN}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
