#!/usr/bin/env python3
"""
One-shot atlas + latent-geometry + perturb metrics from a single embedding export.

Inputs:  --emb-dir ``output/embeddings/<model>/<dataset>/raw``
          (``latent.npy`` + ``obs.{parquet|csv.gz}`` + ``meta.json``)

Outputs: ``--out-dir`` (default inferred as ``output/metrics/<model>/<dataset>/<latent_space>/``):
          ``atlas.json``, ``geometry.json``, ``perturb.json``, ``summary.json``

Behaviour:
  - Resolves obs via ``meta.json`` ``obs_artifact`` or ``obs.parquet`` / ``obs.csv.gz``.
  - Calls ``run_atlas_metrics``, ``run_geometry_metrics``, ``centroid_shift_metrics``,
    and optionally ``summarize_xcellline_by_line`` when ``cell_line`` (or ``--cell-line-col``) exists.
  - ``summarize_ot_deltas`` runs when controls are present; requires POT for non-null EMD values.
  - ``--skip atlas|geometry|perturb`` disables subsets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))
FM_ROOT = BENCH_ROOT.parent / "fm"
sys.path.insert(0, str(FM_ROOT))

from metrics.obs_io import read_obs_table
import paths


def _load_meta(emb_dir: Path) -> Dict[str, Any]:
    p = emb_dir / "meta.json"
    if not p.is_file():
        return {}
    with open(p) as f:
        return json.load(f)


def _resolve_obs_path(emb_dir: Path, meta: Dict[str, Any]) -> Path:
    oa = meta.get("obs_artifact")
    if oa:
        cand = emb_dir / str(oa)
        if cand.is_file():
            return cand
    for name in ("obs.parquet", "obs.csv.gz", "obs.csv"):
        p = emb_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No obs table in {emb_dir} (expected meta.obs_artifact or obs.parquet|obs.csv.gz)"
    )


def _infer_out_dir(emb_dir: Path, meta: Dict[str, Any], latent_space: str) -> Path:
    """Infer ``output/metrics/<model>/<dataset_id>/<latent_space>/`` from ``.../embeddings/<m>/<ds>/raw``."""
    raw = emb_dir.resolve()
    if raw.name != "raw":
        raise ValueError("Cannot infer --out-dir: --emb-dir must end with /raw")
    dataset_id = raw.parent.name
    model = raw.parent.parent.name
    p = raw.parent
    out_root: Optional[Path] = None
    output_root = paths.output_root().resolve()
    while p != p.parent:
        if p == output_root or p.name in {"output", "scFM_output"}:
            out_root = p
            break
        p = p.parent
    if out_root is None:
        model = meta.get("model", model)
        raise ValueError(
            "Cannot find output root ancestor; pass --out-dir explicitly "
            f"(model={model!r}, dataset_id={dataset_id!r})"
        )
    if latent_space not in ("raw", "pca128"):
        raise ValueError(f"latent_space must be raw|pca128, got {latent_space!r}")
    return out_root / "metrics" / model / dataset_id / latent_space


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--emb-dir",
        type=Path,
        required=True,
        help="Directory with latent.npy, meta.json, obs sidecar",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: sibling output/metrics/<model>/<dataset>/ under output/",
    )
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=("atlas", "geometry", "perturb"),
        help="Metric families to skip",
    )
    ap.add_argument("--batch-col", type=str, default="batch")
    ap.add_argument("--label-col", type=str, default="cell_type")
    ap.add_argument("--pert-col", type=str, default="pert")
    ap.add_argument("--is-control-col", type=str, default="is_control")
    ap.add_argument("--cell-line-col", type=str, default="cell_line")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--latent-space",
        choices=("raw", "pca128"),
        default="raw",
        help="raw: native latent; pca128: per-task PCA after StandardScaler on chosen fit mask",
    )
    args = ap.parse_args()

    skip: Set[str] = set(args.skip)
    emb_dir: Path = args.emb_dir.resolve()
    latent_p = emb_dir / "latent.npy"
    if not latent_p.is_file():
        raise SystemExit(f"missing {latent_p}")

    meta = _load_meta(emb_dir)
    obs_path = _resolve_obs_path(emb_dir, meta)
    obs = read_obs_table(obs_path)
    z = np.load(latent_p)
    if z.shape[0] != len(obs):
        raise SystemExit(f"latent rows {z.shape[0]} != obs rows {len(obs)}")

    latent_space = args.latent_space
    z_orig_dim = int(z.shape[1])
    summary: Dict[str, Any] = {
        "latent_space": latent_space,
        "emb_dir": str(emb_dir),
        "obs_path": str(obs_path),
        "n_cells": int(z.shape[0]),
        "latent_dim": int(z.shape[1]),
        "latent_dim_input": z_orig_dim,
        "skipped": sorted(skip),
    }

    if latent_space == "pca128":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        k = min(128, z.shape[1], z.shape[0])
        ctrl_col = next((c for c in ("control", "is_control") if c in obs.columns), None)
        n_ctrl = int(obs[ctrl_col].astype(bool).sum()) if ctrl_col else 0
        if ctrl_col and n_ctrl >= max(k + 10, 200):
            fit_mask = obs[ctrl_col].astype(bool).to_numpy()
            fit_scope = "control_only"
        else:
            fit_mask = np.ones(len(obs), dtype=bool)
            fit_scope = "all_cells"
        n_fit = int(fit_mask.sum())
        k = min(k, n_fit)
        if k < 1:
            raise SystemExit("pca128 needs at least one cell")
        z_fit = z[fit_mask].astype(np.float64)
        scaler = StandardScaler().fit(z_fit)
        pca = PCA(n_components=k, random_state=args.seed).fit(scaler.transform(z_fit))
        z = pca.transform(scaler.transform(z.astype(np.float64))).astype(np.float32)
        summary["pca128_fit_scope"] = fit_scope
        summary["pca128_n_fit_cells"] = int(fit_mask.sum())
        summary["pca128_explained_variance"] = float(pca.explained_variance_ratio_.sum())
        summary["pca128_k"] = int(k)
        summary["latent_dim"] = int(z.shape[1])

    out_dir = (args.out_dir.resolve() if args.out_dir else _infer_out_dir(emb_dir, meta, latent_space))
    out_dir.mkdir(parents=True, exist_ok=True)

    if "atlas" not in skip:
        from metrics.atlas_scib import run_atlas_metrics

        atlas_payload: Dict[str, Any]
        if args.batch_col not in obs.columns or args.label_col not in obs.columns:
            atlas_payload = {
                "skipped": True,
                "reason": f"need columns {args.batch_col!r} and {args.label_col!r}",
            }
        else:
            atlas_payload = run_atlas_metrics(
                z,
                obs,
                batch_col=args.batch_col,
                label_col=args.label_col,
                seed=args.seed,
                trust_random_state=args.seed,
            )
            atlas_payload["n_cells"] = int(z.shape[0])
            atlas_payload["latent_dim"] = int(z.shape[1])
        summary["atlas"] = atlas_payload
        with open(out_dir / "atlas.json", "w") as f:
            json.dump(atlas_payload, f, indent=2)

    if "geometry" not in skip:
        from metrics.geometry import run_geometry_metrics

        geom_payload = run_geometry_metrics(
            z,
            obs,
            label_col=args.label_col if args.label_col in obs.columns else None,
            batch_col=args.batch_col if args.batch_col in obs.columns else None,
            seed=args.seed,
        )
        geom_payload["n_cells"] = int(z.shape[0])
        geom_payload["latent_dim"] = int(z.shape[1])
        summary["geometry"] = geom_payload
        with open(out_dir / "geometry.json", "w") as f:
            json.dump(geom_payload, f, indent=2)

    if "perturb" not in skip:
        from metrics.perturb_geom import centroid_shift_metrics, summarize_ot_deltas
        from metrics.perturb_xcellline import summarize_xcellline_by_line

        pert_payload: Dict[str, Any] = {}
        if args.pert_col not in obs.columns:
            pert_payload["note"] = f"missing column {args.pert_col!r}"
        elif args.is_control_col not in obs.columns:
            pert_payload["centroid_shift"] = {
                "skipped": True,
                "reason": f"missing {args.is_control_col!r}",
            }
            pert_payload["ot_summary"] = {"note": "no is_control column"}
        else:
            pert_payload["centroid_shift"] = centroid_shift_metrics(
                z,
                obs,
                args.pert_col,
                is_control_col=args.is_control_col,
            )
            pert_payload["ot_summary"] = summarize_ot_deltas(
                z,
                obs,
                args.pert_col,
                is_control_col=args.is_control_col,
                seed=args.seed,
            )
        if args.cell_line_col in obs.columns and args.pert_col in obs.columns:
            if args.is_control_col in obs.columns:
                pert_payload["xcellline"] = summarize_xcellline_by_line(
                    z,
                    obs,
                    args.cell_line_col,
                    args.pert_col,
                    is_control_col=args.is_control_col,
                    seed=args.seed,
                )
            else:
                pert_payload["xcellline"] = {"skipped": True, "reason": "no is_control"}
        summary["perturb"] = pert_payload
        with open(out_dir / "perturb.json", "w") as f:
            json.dump(pert_payload, f, indent=2)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
