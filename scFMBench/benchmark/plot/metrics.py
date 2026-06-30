"""Metric registry for the scFM benchmark.

Each entry describes one column in ``summary_all*.csv``:
- column: dotted path as it appears in the wide CSV
- short:  short label for tick / heatmap headers
- long:   one-line description for legends / captions
- family: one of {atlas, geometry, perturb}
- direction: ``up`` if higher is better, ``down`` if lower is better
- categories: which dataset categories the metric applies to
- prefer_space: ``raw`` | ``pca128`` | ``either`` (recommendation, not enforced)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Metric:
    column: str
    short: str
    long: str
    family: str  # atlas | geometry | perturb
    direction: str  # up | down
    categories: Tuple[str, ...] = field(default=("atlas", "atlas_TS", "chempert"))
    prefer_space: str = "either"  # raw | pca128 | either


# --- atlas (cell-type / batch integration on staging atlas datasets) -----
ATLAS_METRICS: Tuple[Metric, ...] = (
    Metric("atlas.A1_nmi", "NMI", "Cluster-celltype NMI", "atlas", "up", ("atlas",), "either"),
    Metric("atlas.A1_ari", "ARI", "Cluster-celltype ARI", "atlas", "up", ("atlas",), "either"),
    Metric("atlas.A2_clisi", "cLISI", "Cell-type local LISI conservation (scib scaled, higher is better)", "atlas", "up", ("atlas",), "either"),
    Metric("atlas.A3_ilisi", "iLISI", "Batch local LISI (higher = better mixing)", "atlas", "up", ("atlas",), "either"),
    Metric("atlas.A4_graph_connectivity", "GC", "Graph connectivity per label", "atlas", "up", ("atlas",), "either"),
    Metric("atlas.A5_trustworthiness", "Trust", "Trustworthiness (kNN preserved)", "atlas", "up", ("atlas",), "pca128"),
)

# --- geometry (latent geometry / dimensionality / stability) -------------
GEOMETRY_METRICS: Tuple[Metric, ...] = (
    Metric("geometry.G1_participation_ratio", "PR", "Participation ratio (effective dim)", "geometry", "up", prefer_space="raw"),
    Metric("geometry.G1_effective_rank_cov", "rk_eff", "Effective rank of covariance", "geometry", "up", prefer_space="raw"),
    Metric("geometry.G1_pca_90_ratio", "PCA90", "Fraction of dims for 90% variance", "geometry", "down", prefer_space="raw"),
    Metric("geometry.G2_knn_label_consistency", "kNN-LC", "kNN label consistency", "geometry", "up", prefer_space="either"),
    Metric("geometry.G3_lambda_max_over_trace", "anisoλ", "Top-eigenvalue / trace (lower is more isotropic)", "geometry", "down", prefer_space="raw"),
    Metric("geometry.G3_condition_eig", "cond", "Eigenvalue condition number (lower better)", "geometry", "down", prefer_space="raw"),
    Metric("geometry.G4_silhouette_euclidean", "Sil", "Silhouette under provided labels", "geometry", "up", prefer_space="either"),
    Metric("geometry.G5_dist_spearman_under_noise", "G5stab", "Distance Spearman under noise", "geometry", "up", prefer_space="either"),
    Metric("geometry.G6_laplacian_energy", "Lap", "Laplacian energy (graph regularity)", "geometry", "up", prefer_space="either"),
    Metric("geometry.LDM_proxy_score", "LDM*", "LDM proxy composite", "geometry", "up", prefer_space="either"),
)

# --- perturb (chempert + compatible genepert) ----------------------------
_PERTURB_FAMILIES = ("chempert", "genepert")

PERTURB_METRICS: Tuple[Metric, ...] = (
    Metric("perturb.centroid_shift.mean_l2_to_control", "mean L2", "Mean centroid shift (perturbed vs control)", "perturb", "up", _PERTURB_FAMILIES, "raw"),
    Metric("perturb.centroid_shift.median_l2_to_control", "median L2", "Median centroid shift", "perturb", "up", _PERTURB_FAMILIES, "raw"),
    Metric("perturb.ot_summary.emd_mean", "EMD mean", "Optimal transport EMD (mean across pert)", "perturb", "up", _PERTURB_FAMILIES, "raw"),
    Metric("perturb.ot_summary.emd_median", "EMD med", "EMD median across perturbations", "perturb", "up", _PERTURB_FAMILIES, "raw"),
    Metric("perturb.xcellline.xcellline_mean_l2_across_lines", "xL2", "Cross cell-line mean L2 (xCellLine only)", "perturb", "up", ("chempert",), "raw"),
    Metric("perturb.xcellline.xcellline_mean_emd_across_lines", "xEMD", "Cross cell-line mean EMD (xCellLine only)", "perturb", "up", ("chempert",), "raw"),
    # Top-K Spearman between scale-normalized latent centroid shifts and raw
    # expression-space centroid L2 (ground-truth perturbation strength). K is
    # the per-dataset max available (sciplex3 → 50, genepert → 30). Injected
    # into the wide table by ``data.augment_with_topk_spearman`` at build time.
    Metric("perturb.topk_spearman_vs_gt", "Top-K ρ", "Top-K Spearman vs raw-expr GT (max K per dataset)", "perturb", "up", _PERTURB_FAMILIES, "raw"),
    # Mantel-Spearman of cosine perturbation-similarity matrices between the
    # raw expression GT (top-10 perturbations) and the model's latent
    # representation. Captures preservation of inter-perturbation relationship
    # structure (pathway co-functionality). Injected by
    # ``data.augment_with_mantel_spearman``.
    Metric("perturb.mantel_spearman_cos_vs_gt", "Mantel ρ_S", "Top-10 cosine sim matrix Mantel-Spearman vs raw GT", "perturb", "up", _PERTURB_FAMILIES, "raw"),
)

ALL_METRICS: Tuple[Metric, ...] = ATLAS_METRICS + GEOMETRY_METRICS + PERTURB_METRICS


# Headline metrics curated for figure 1 (one per "axis of evaluation")
HEADLINE_ATLAS: Tuple[str, ...] = (
    "atlas.A1_nmi",
    "atlas.A1_ari",
    "atlas.A3_ilisi",
    "atlas.A2_clisi",
    "atlas.A5_trustworthiness",
)
HEADLINE_GEOMETRY: Tuple[str, ...] = (
    "geometry.G1_participation_ratio",
    "geometry.G3_lambda_max_over_trace",
    "geometry.G2_knn_label_consistency",
    "geometry.G5_dist_spearman_under_noise",
)
HEADLINE_PERTURB: Tuple[str, ...] = (
    "perturb.topk_spearman_vs_gt",
    "perturb.mantel_spearman_cos_vs_gt",
)


def by_column() -> Dict[str, Metric]:
    return {m.column: m for m in ALL_METRICS}


def filter_metrics(
    metrics: Sequence[Metric],
    *,
    family: str | None = None,
    direction: str | None = None,
    category: str | None = None,
) -> List[Metric]:
    out = list(metrics)
    if family is not None:
        out = [m for m in out if m.family == family]
    if direction is not None:
        out = [m for m in out if m.direction == direction]
    if category is not None:
        out = [m for m in out if category in m.categories]
    return out
