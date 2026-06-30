#!/usr/bin/env python3
"""
Read-only inventory of raw embedding exports under output/embeddings/*/*/raw/.

Writes:
  output/benchmark_inventory/raw_inventory.csv
  output/benchmark_inventory/raw_inventory.md
  output/benchmark_inventory/pca128_eligibility.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

BENCH_ROOT = Path(__file__).resolve().parents[1]
SCFM_ROOT = BENCH_ROOT.parent

import sys

sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, str(SCFM_ROOT / "fm"))

from metrics.obs_io import read_obs_table
import paths


def _load_meta(emb_dir: Path) -> Dict[str, Any]:
    with open(emb_dir / "meta.json") as f:
        return json.load(f)


def _resolve_obs_path(emb_dir: Path, meta: Dict[str, Any]) -> Path | None:
    oa = meta.get("obs_artifact")
    if oa:
        cand = emb_dir / str(oa)
        if cand.is_file():
            return cand
    for name in ("obs.parquet", "obs.csv.gz", "obs.csv"):
        p = emb_dir / name
        if p.is_file():
            return p
    return None


def _category_from_meta(meta: Dict[str, Any], dataset_id: str) -> str:
    src = str(meta.get("source_adata") or "")
    if "/chempert/" in src or "/chemicalpert_bench/" in src:
        return "chempert"
    if "/genepert/" in src or "/genepert_bench/" in src:
        return "genepert"
    if "/atlas_TS/" in src or "/raw/atlas_TS/" in src:
        return "atlas"
    if "/staging/atlas/" in src:
        return "atlas"
    if dataset_id.startswith("sciplex3_"):
        return "chempert"
    if dataset_id.startswith("TS_") and dataset_id.endswith("_filtered"):
        return "atlas"
    if dataset_id in {"Blood", "BoneMarrow", "Heart", "Lung", "LymphNode", "Skin", "TS_Immune_xtissue"}:
        return "atlas"
    return "other"


def _metrics_status(scfm: Path, model: str, dataset: str) -> Tuple[str, str]:
    base = paths.output_root() / "metrics" / model / dataset
    out: List[str] = []
    for sp in ("raw", "pca128"):
        p = base / sp / "summary.json"
        out.append("yes" if p.is_file() else "no")
    return out[0], out[1]


def _pca128_eligibility_row(
    category: str,
    obs: pd.DataFrame,
) -> Tuple[str, str | None, int]:
    """Returns (pca128_fit_scope, control_col_used, n_control)."""
    ctrl_col = None
    for c in ("control", "is_control"):
        if c in obs.columns:
            ctrl_col = c
            break
    n_ctrl = 0
    if ctrl_col is not None:
        n_ctrl = int(obs[ctrl_col].astype(bool).sum())
    if category != "chempert":
        return "all_cells", ctrl_col, n_ctrl
    if ctrl_col is None:
        return "all_cells", None, 0
    if n_ctrl >= 200:
        return "control_only", ctrl_col, n_ctrl
    return "all_cells", ctrl_col, n_ctrl


def _audit_one(emb_dir: Path, scfm: Path) -> Dict[str, Any]:
    meta = _load_meta(emb_dir)
    raw = emb_dir.resolve()
    model = raw.parent.parent.name
    dataset = raw.parent.name
    category = _category_from_meta(meta, dataset)

    obs_path = _resolve_obs_path(emb_dir, meta)
    latent_p = emb_dir / "latent.npy"
    row: Dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "category": category,
        "n_cells": "",
        "latent_dim": "",
        "has_meta": True,
        "has_obs": obs_path is not None,
        "obs_format": obs_path.suffix if obs_path else "",
        "latent_dtype": "",
        "latent_min": "",
        "latent_max": "",
        "latent_mean_norm": "",
        "n_unique_batch": "",
        "n_unique_cell_type": "",
        "n_control_cells": "",
        "n_perturbations": "",
        "metrics_raw": "",
        "metrics_pca128": "",
    }
    z = None
    if latent_p.is_file():
        z = np.load(latent_p)
        row["n_cells"] = int(z.shape[0])
        row["latent_dim"] = int(z.shape[1])
        row["latent_dtype"] = str(z.dtype)
        row["latent_min"] = float(z.min())
        row["latent_max"] = float(z.max())
        row["latent_mean_norm"] = float(np.linalg.norm(z, axis=1).mean())

    obs: pd.DataFrame | None = None
    if obs_path is not None:
        obs = read_obs_table(obs_path)
        row["n_unique_batch"] = int(obs["batch"].nunique()) if "batch" in obs.columns else ""
        row["n_unique_cell_type"] = int(obs["cell_type"].nunique()) if "cell_type" in obs.columns else ""
        for c in ("control", "is_control"):
            if c in obs.columns:
                row["n_control_cells"] = int(obs[c].astype(bool).sum())
                break
        if "perturbation" in obs.columns:
            row["n_perturbations"] = int(obs["perturbation"].nunique())

    mr, mp = _metrics_status(scfm, model, dataset)
    row["metrics_raw"] = mr
    row["metrics_pca128"] = mp

    if z is not None and obs is not None and z.shape[0] != len(obs):
        row["obs_format"] = f"MISMATCH_ROWS:{row['obs_format']}"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scfm-root", type=Path, default=SCFM_ROOT)
    args = ap.parse_args()
    scfm: Path = args.scfm_root.resolve()
    out_dir = paths.output_root() / "benchmark_inventory"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    elig: List[Dict[str, Any]] = []
    emb_root = paths.output_root() / "embeddings"

    for meta_path in sorted(emb_root.glob("*/*/raw/meta.json")):
        raw_d = meta_path.parent
        if not (raw_d / "latent.npy").is_file():
            continue
        r = _audit_one(raw_d, scfm)
        rows.append(r)
        meta = _load_meta(raw_d)
        category = _category_from_meta(meta, raw_d.parent.name)
        obs_path = _resolve_obs_path(raw_d, meta)
        if obs_path and obs_path.is_file():
            obs = read_obs_table(obs_path)
            scope, cused, nc = _pca128_eligibility_row(category, obs)
            elig.append(
                {
                    "model": raw_d.parent.parent.name,
                    "dataset": raw_d.parent.name,
                    "control_col_used": cused or "",
                    "n_control": nc,
                    "pca128_fit_scope": scope,
                }
            )

    cols = [
        "model",
        "dataset",
        "category",
        "n_cells",
        "latent_dim",
        "has_meta",
        "has_obs",
        "obs_format",
        "latent_dtype",
        "latent_min",
        "latent_max",
        "latent_mean_norm",
        "n_unique_batch",
        "n_unique_cell_type",
        "n_control_cells",
        "n_perturbations",
        "metrics_raw",
        "metrics_pca128",
    ]
    df = pd.DataFrame(rows)[cols]
    df.to_csv(out_dir / "raw_inventory.csv", index=False)

    with open(out_dir / "pca128_eligibility.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "dataset", "control_col_used", "n_control", "pca128_fit_scope"],
        )
        w.writeheader()
        for e in elig:
            w.writerow(e)

    md_lines = [
        "# Raw embedding inventory",
        "",
        f"Scanned: `{emb_root}`",
        f"Rows: {len(df)} (model × dataset)",
        "",
        "## Per-model summary",
        "",
    ]
    for m in sorted(df["model"].unique()):
        sub = df[df["model"] == m]
        md_lines.append(f"### {m}")
        md_lines.append(
            f"- latent_dim: {sub['latent_dim'].iloc[0] if not sub.empty else 'n/a'}; "
            f"datasets: {len(sub)}; n_cells range {sub['n_cells'].min()}–{sub['n_cells'].max()}"
        )
        md_lines.append("")

    md_lines.append("## Metrics coverage (from output/metrics)")
    cov = df.groupby("model").agg(
        raw_done=("metrics_raw", lambda s: int((s == "yes").sum())),
        pca_done=("metrics_pca128", lambda s: int((s == "yes").sum())),
        n=("model", "count"),
    )
    try:
        md_lines.append(cov.to_markdown())
    except ImportError:
        md_lines.append(cov.to_string())
    md_lines.append("")
    md_lines.append("## Full table (CSV)")
    md_lines.append("")
    md_lines.append("See `raw_inventory.csv` and `pca128_eligibility.csv`.")

    with open(out_dir / "raw_inventory.md", "w") as f:
        f.write("\n".join(md_lines))

    print(json.dumps({"raw_inventory_csv": str(out_dir / "raw_inventory.csv"), "n_rows": len(df)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
