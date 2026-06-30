#!/usr/bin/env python3
"""Select encoder candidates for downstream LatentFM experiments.

Inputs:
- ``SCFM_OUTPUT_ROOT/metrics/summary_all.csv`` from the benchmark metrics pass.
- Optional ``SCFM_OUTPUT_ROOT/embeddings/**/meta.json`` throughput metadata.

The score intentionally reuses the benchmark metric registry so metric
directions stay consistent with publication figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FM_ROOT = ROOT / "fm"
for p in (ROOT, FM_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import paths
from benchmark.plot import data as D
from benchmark.plot import metrics as M
from benchmark.plot import style as ST


def _load_throughput() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for meta_path in sorted((paths.output_root() / "embeddings").glob("**/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        model = str(meta.get("model", "")).lower().strip()
        if model not in ST.FM_MODELS:
            continue
        n_obs = float(meta.get("n_obs") or 0)
        wall = float(meta.get("wall_time_s") or 0)
        if n_obs <= 0 or wall <= 0:
            continue
        rows.append(
            {
                "model": model,
                "cells_per_s": n_obs / wall,
                "n_obs": n_obs,
                "wall_time_s": wall,
                "meta_path": str(meta_path),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["model", "median_cells_per_s", "n_runs"])
    df = pd.DataFrame(rows)
    return (
        df.groupby("model")
        .agg(median_cells_per_s=("cells_per_s", "median"), n_runs=("cells_per_s", "size"))
        .reset_index()
    )


def _efficiency_score(tp: pd.DataFrame) -> pd.DataFrame:
    if tp.empty:
        return pd.DataFrame({"model": list(ST.FM_MODELS), "efficiency_score": 0.5, "median_cells_per_s": np.nan, "n_runs": 0})
    out = tp.copy()
    vals = np.log1p(out["median_cells_per_s"].astype(float))
    if vals.max() == vals.min():
        out["efficiency_score"] = 0.5
    else:
        out["efficiency_score"] = (vals - vals.min()) / (vals.max() - vals.min())
    return out


def select_top(
    *,
    latent_space: str,
    perf_weight: float,
    eff_weight: float,
    top_k: int,
    min_datasets: int,
) -> pd.DataFrame:
    summary = paths.output_root() / "metrics" / "summary_all.csv"
    if not summary.is_file():
        raise FileNotFoundError(
            f"Missing benchmark summary: {summary}. Run benchmark metrics before selecting LatentFM encoders."
        )

    wide = D.load_wide(paths.scfm_root())
    wide = wide[wide["model"].isin(ST.FM_MODELS)].copy()
    if latent_space != "any":
        wide = wide[wide["latent_space"].eq(latent_space)].copy()
    if wide.empty:
        raise ValueError(f"No benchmark rows for latent_space={latent_space!r}")
    coverage = (
        wide.groupby("model")
        .agg(
            n_datasets=("dataset_id", "nunique"),
            n_rows=("dataset_id", "size"),
            n_categories=("category", "nunique"),
            categories=("category", lambda s: ",".join(sorted(set(map(str, s))))),
        )
        .reset_index()
    )
    if min_datasets > 0:
        keep = set(coverage.loc[coverage["n_datasets"] >= min_datasets, "model"])
        wide = wide[wide["model"].isin(keep)].copy()
        coverage = coverage[coverage["model"].isin(keep)].copy()
        if wide.empty:
            raise ValueError(
                f"No benchmark rows left after --min-datasets {min_datasets}; "
                f"relax coverage filtering or run more datasets."
            )

    metric_cols = set(wide.columns)
    headline = [m for m in M.ALL_METRICS if m.column in metric_cols and m.column in (set(M.HEADLINE_ATLAS) | set(M.HEADLINE_GEOMETRY) | set(M.HEADLINE_PERTURB))]
    metrics = headline or [m for m in M.ALL_METRICS if m.column in metric_cols]
    long = D.melt_metrics(wide, metrics=metrics)
    normed = D.normalize_per_dataset(long, method="rank")
    perf = D.aggregate_model_score(normed, by=("model",)).rename(columns={"mean_score": "performance_score"})

    eff = _efficiency_score(_load_throughput())
    out = perf.merge(eff, on="model", how="left")
    out = out.merge(coverage, on="model", how="left")
    out["efficiency_score"] = out["efficiency_score"].fillna(0.5)
    out["median_cells_per_s"] = out["median_cells_per_s"].astype(float)
    out["n_runs"] = out["n_runs"].fillna(0).astype(int)
    out["n_datasets"] = out["n_datasets"].fillna(0).astype(int)
    out["n_rows"] = out["n_rows"].fillna(0).astype(int)
    out["n_categories"] = out["n_categories"].fillna(0).astype(int)
    out["categories"] = out["categories"].fillna("")
    out["composite_score"] = perf_weight * out["performance_score"] + eff_weight * out["efficiency_score"]
    out["latent_space_filter"] = latent_space
    out["min_datasets_filter"] = int(min_datasets)
    out["n_metric_values"] = normed.groupby("model")["score"].size().reindex(out["model"]).fillna(0).astype(int).values
    out = out.sort_values(["composite_score", "performance_score", "efficiency_score"], ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out.head(top_k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent-space", default="raw", help="raw|pca128|any; default raw for LatentFM source encoders")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--perf-weight", type=float, default=0.75)
    ap.add_argument("--eff-weight", type=float, default=0.25)
    ap.add_argument(
        "--min-datasets",
        type=int,
        default=0,
        help="Drop models with fewer covered datasets after latent-space filtering; 0 disables filtering.",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    total = args.perf_weight + args.eff_weight
    if total <= 0:
        raise SystemExit("perf/eff weights must sum to a positive value")
    perf_w = args.perf_weight / total
    eff_w = args.eff_weight / total

    try:
        out = select_top(
            latent_space=args.latent_space,
            perf_weight=perf_w,
            eff_weight=eff_w,
            top_k=args.top_k,
            min_datasets=args.min_datasets,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    out_path = args.out or (paths.output_root() / "metrics" / "top_encoder_candidates.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
