#!/usr/bin/env python3
"""
Tier-1 atlas metrics (A1–A6) on exported latent + obs sidecar.

Expects log1p-normalized expression reference only for A6 expr-ref UMAP (optional --adata-ref).
Latent embedding: latent.npy aligned row-wise with obs.parquet.

Metrics:
  A1 NMI (Leiden), A2 cLISI, A3 iLISI, A4 graph_connectivity, A5 trustworthiness (2D UMAP on latent),
  A6 figure panels (expr ref UMAP + latent UMAPs for one or more latents).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import trustworthiness

import anndata as ad
from scib_metrics import (
    clisi_knn,
    ilisi_knn,
    nmi_ari_cluster_labels_leiden,
)
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent

try:
    from .obs_io import read_obs_table
except ImportError:  # noqa: PERF203
    from obs_io import read_obs_table


def graph_connectivity_compat(nnr: NeighborsResults, labels: np.ndarray) -> float:
    """Same as scib_metrics.graph_connectivity; pandas>=2.4 removed pd.value_counts."""
    clust_res: List[float] = []
    graph = nnr.knn_graph_distances
    for label in np.unique(labels):
        mask = labels == label
        graph_sub = graph[mask]
        graph_sub = graph_sub[:, mask]
        _, comps = connected_components(graph_sub, connection="strong")
        tab = pd.Series(comps).value_counts()
        clust_res.append(float(tab.max() / tab.sum()))
    return float(np.mean(clust_res))


def _nn(X: np.ndarray, n_neighbors: int = 90, n_jobs: int = 1, seed: int = 0):
    X = np.asarray(X, dtype=np.float64)
    return pynndescent(X, n_neighbors=n_neighbors, random_state=seed, n_jobs=n_jobs)


def run_atlas_metrics(
    latent: np.ndarray,
    obs: pd.DataFrame,
    *,
    batch_col: str = "batch",
    label_col: str = "cell_type",
    n_neighbors: int = 90,
    seed: int = 42,
    trust_max_cells: int = 15000,
    trust_random_state: int = 0,
) -> Dict[str, Any]:
    assert len(obs) == latent.shape[0], (len(obs), latent.shape[0])
    labels = obs[label_col].astype(str).to_numpy()
    batches = obs[batch_col].astype(str).to_numpy()
    n_batch = len(np.unique(batches))

    # pynndescent requires n_neighbors < n_samples; clamp so small atlas
    # subsets (e.g. debug runs) do not crash. No effect for real atlases
    # where n_cells >> n_neighbors.
    n_cells = latent.shape[0]
    n_neighbors = max(2, min(n_neighbors, n_cells - 1))

    nnr = _nn(latent, n_neighbors=n_neighbors, seed=seed)
    out: Dict[str, Any] = {}

    nmi_res = nmi_ari_cluster_labels_leiden(nnr, labels, seed=seed)
    out["A1_nmi"] = float(nmi_res.get("nmi", np.nan))
    out["A1_ari"] = float(nmi_res.get("ari", np.nan))

    out["A2_clisi"] = float(clisi_knn(nnr, labels))

    if n_batch >= 2:
        out["A3_ilisi"] = float(ilisi_knn(nnr, batches))
        out["A4_graph_connectivity"] = graph_connectivity_compat(nnr, labels)
    else:
        out["A3_ilisi"] = None
        out["A4_graph_connectivity"] = None

    # A5: trustworthiness in 2D UMAP space (subsample for speed)
    rng = np.random.default_rng(trust_random_state)
    n = latent.shape[0]
    if n > trust_max_cells:
        idx = rng.choice(n, size=trust_max_cells, replace=False)
        Z = latent[idx]
        high = latent[idx]
    else:
        idx = np.arange(n)
        Z = latent
        high = latent

    ad_um = ad.AnnData(X=Z.astype(np.float32))
    sc.pp.neighbors(ad_um, n_neighbors=15, use_rep="X")
    sc.tl.umap(ad_um)
    u2 = ad_um.obsm["X_umap"].astype(np.float64)
    tw = trustworthiness(high, u2, n_neighbors=15)
    out["A5_trustworthiness"] = float(tw)
    return out


def _umap_panel(
    adata_expr: ad.AnnData,
    latent_paths: List[Path],
    latent_labels: List[str],
    color_col: str,
    out_png: Path,
    title_suffix: str = "",
) -> None:
    """Subplot 0: expr HVG+PCA+UMAP; rest: latent UMAPs."""
    n = 1 + len(latent_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axr = axes[0]

    ade = adata_expr.copy()
    sc.pp.highly_variable_genes(ade, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(ade, use_highly_variable=True, n_comps=50)
    sc.pp.neighbors(ade, n_neighbors=15, use_rep="X_pca")
    sc.tl.umap(ade)
    sc.pl.umap(ade, color=color_col, ax=axr[0], show=False, title=f"expr ref{title_suffix}")

    for j, (lp, lab) in enumerate(zip(latent_paths, latent_labels), start=1):
        Z = np.load(lp)
        adz = ad.AnnData(X=Z.astype(np.float32))
        adz.obs[color_col] = ade.obs[color_col].values
        sc.pp.neighbors(adz, n_neighbors=15, use_rep="X")
        sc.tl.umap(adz)
        sc.pl.umap(adz, color=color_col, ax=axr[j], show=False, title=lab)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latent", type=Path, required=True, help="latent.npy (n_cells, d)")
    ap.add_argument("--obs", type=Path, required=True, help="obs table (.parquet or .csv.gz; batch, cell_type, …)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--adata-ref", type=Path, default=None, help="Optional h5ad for A6 expr panel (same cells/order as obs)")
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--extra-latents", type=Path, nargs="*", default=[], help="More latent.npy for A6 multi-panel")
    ap.add_argument("--extra-labels", type=str, nargs="*", default=[])
    ap.add_argument("--batch-col", type=str, default="batch")
    ap.add_argument("--label-col", type=str, default="cell_type")
    args = ap.parse_args()

    Z = np.load(args.latent)
    obs = read_obs_table(args.obs)
    metrics = run_atlas_metrics(
        Z,
        obs,
        batch_col=args.batch_col,
        label_col=args.label_col,
    )
    metrics["n_cells"] = int(Z.shape[0])
    metrics["latent_dim"] = int(Z.shape[1])

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    if args.fig_dir and args.adata_ref:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        ad_ref = ad.read_h5ad(args.adata_ref)
        # align obs order
        if not ad_ref.n_obs == len(obs):
            raise ValueError("adata-ref n_obs mismatch obs parquet")
        latents = [args.latent, *args.extra_latents]
        labels = ["latent_main", *args.extra_labels]
        if len(labels) != len(latents):
            labels = [f"m{i}" for i in range(len(latents))]
        _umap_panel(
            ad_ref,
            latents,
            labels,
            args.label_col,
            args.fig_dir / "umap_panel_by_celltype.png",
        )
        _umap_panel(
            ad_ref,
            latents,
            labels,
            args.batch_col,
            args.fig_dir / "umap_panel_by_batch.png",
        )
        if "compartment" in ad_ref.obs.columns:
            _umap_panel(
                ad_ref,
                latents,
                labels,
                "compartment",
                args.fig_dir / "umap_panel_by_compartment.png",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
