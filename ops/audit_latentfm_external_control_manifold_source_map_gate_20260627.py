#!/usr/bin/env python3
"""External control-manifold source-map / anti-proxy CPU gate.

This gate tests whether local cellgene-census blood/lung control manifolds can
provide a condition-level, non-proxy artifact for Track A. It uses only local
processed h5ad files and internal-val residual forensics rows. No model is run.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


ROOT = Path("/data/cyx/1030/scLatent")
S0 = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
RESIDUAL_ROWS = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
METAINFO = ROOT / "dataset/cellgene_census/processed/celltype_metainfo.csv"
H5ADS = {
    "blood": ROOT / "dataset/cellgene_census/processed/blood/blood_top6000var.h5ad",
    "lung": ROOT / "dataset/cellgene_census/processed/lung/lung_top6000var.h5ad",
}

OUT_DIR = ROOT / "reports/external_control_manifold_source_map_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_external_control_manifold_source_map_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_EXTERNAL_CONTROL_MANIFOLD_SOURCE_MAP_GATE_20260627.md"
OUT_JOIN = OUT_DIR / "control_manifold_join_rows.csv"
OUT_FEATURES = OUT_DIR / "control_manifold_gene_features.csv"

FEATURES = [
    "atlas_gene_present",
    "atlas_rank_mean",
    "atlas_rank_nonzero_frac",
    "atlas_rank_max",
    "atlas_cluster_weighted_mean",
    "atlas_var_nnz_frac",
    "atlas_log_n_measured_obs",
]
TARGETS = [
    "anchor_pearson_pert",
    "anchor_mmd_clamped",
    "anchor_minus_dataset_mean",
    "anchor_minus_gene_raw_mean",
    "target_residual_norm",
]


def norm_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def gene_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(value).lower())


def to_float(value: object) -> float | None:
    text = norm_text(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def read_s0_backgrounds() -> tuple[pd.DataFrame, dict[str, str]]:
    s0 = pd.read_csv(S0, sep="\t", dtype=str).fillna("")
    s0["n_cells_float"] = pd.to_numeric(s0.get("n_cells", ""), errors="coerce")
    dataset_bg: dict[str, str] = {}
    for dataset, part in s0.groupby("dataset"):
        vals = [v for v in part["cell_background_source"].astype(str).tolist() if v]
        if vals:
            dataset_bg[str(dataset)] = Counter(vals).most_common(1)[0][0]
    return s0, dataset_bg


def map_background(background: str) -> tuple[list[str], str, str]:
    bg = background.lower()
    tissues: list[str] = []
    reasons: list[str] = []
    if any(token in bg for token in ["k562", "thp-1", "thp1", "primary t", "t cell", "jurkat"]):
        tissues.append("blood")
        reasons.append("hematopoietic_or_t_cell_background")
    if "a549" in bg or "lung" in bg:
        tissues.append("lung")
        reasons.append("lung_or_a549_background")
    tissues = sorted(set(tissues))
    if not background:
        return [], "unmapped", "missing_background"
    if len(tissues) == 1:
        return tissues, "mapped_single_tissue", ";".join(reasons)
    if len(tissues) > 1:
        return tissues, "ambiguous_multi_tissue", ";".join(reasons)
    return [], "unmapped", "no_blood_lung_mapping_rule"


def load_gene_features(wanted_genes: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metainfo = pd.read_csv(METAINFO)
    tissue_meta = {
        tissue: {
            "atlas_cell_type_count": int(len(part)),
            "atlas_total_source_n_cells": float(part["n_cells"].sum()),
            "atlas_median_planned_k": float(part["planned_k"].median()),
        }
        for tissue, part in metainfo.groupby("tissue")
    }
    for tissue, path in H5ADS.items():
        atlas = ad.read_h5ad(path)
        var = atlas.var.copy()
        var["gene_key"] = var.get("feature_name", pd.Series(index=var.index, dtype=str)).astype(str).map(gene_key)
        index_by_gene = {g: int(i) for i, g in enumerate(var["gene_key"].tolist()) if g}
        cluster_sizes = pd.to_numeric(atlas.obs.get("cluster_size", 1), errors="coerce").fillna(1.0).to_numpy(dtype=float)
        cluster_weights = cluster_sizes / cluster_sizes.sum() if cluster_sizes.sum() else np.ones_like(cluster_sizes) / len(cluster_sizes)
        x = atlas.X
        for g in sorted(wanted_genes):
            idx = index_by_gene.get(g)
            base = {
                "tissue": tissue,
                "gene_key": g,
                **tissue_meta.get(tissue, {}),
            }
            if idx is None:
                rows.append(
                    {
                        **base,
                        "atlas_gene_present": 0.0,
                        "atlas_rank_mean": 0.0,
                        "atlas_rank_nonzero_frac": 0.0,
                        "atlas_rank_max": 0.0,
                        "atlas_cluster_weighted_mean": 0.0,
                        "atlas_var_nnz_frac": 0.0,
                        "atlas_log_n_measured_obs": 0.0,
                    }
                )
                continue
            col = x[:, idx]
            if sparse.issparse(col):
                arr = np.asarray(col.toarray()).reshape(-1)
                nnz = int(col.nnz)
            else:
                arr = np.asarray(col).reshape(-1)
                nnz = int(np.count_nonzero(arr))
            n_measured = to_float(var.iloc[idx].get("n_measured_obs"))
            rows.append(
                {
                    **base,
                    "feature_name": norm_text(var.iloc[idx].get("feature_name")),
                    "atlas_gene_present": 1.0,
                    "atlas_rank_mean": float(np.mean(arr)),
                    "atlas_rank_nonzero_frac": float(nnz / max(1, arr.shape[0])),
                    "atlas_rank_max": float(np.max(arr)) if arr.size else 0.0,
                    "atlas_cluster_weighted_mean": float(np.dot(arr, cluster_weights)),
                    "atlas_var_nnz_frac": float((to_float(var.iloc[idx].get("nnz")) or 0.0) / max(1.0, n_measured or arr.shape[0])),
                    "atlas_log_n_measured_obs": float(math.log1p(n_measured or 0.0)),
                }
            )
    return pd.DataFrame(rows)


def residualize(values: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    if not mask.any():
        return np.full_like(y, np.nan, dtype=float)
    cols = [np.ones(len(y))]
    for col in controls.columns:
        if controls[col].dtype.kind in "bifc":
            arr = pd.to_numeric(controls[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            cols.append(arr)
        else:
            dummies = pd.get_dummies(controls[col].astype(str), prefix=col, drop_first=True)
            for dcol in dummies.columns[:50]:
                cols.append(dummies[dcol].to_numpy(dtype=float))
    x = np.vstack(cols).T
    finite = mask & np.isfinite(x).all(axis=1)
    if finite.sum() <= x.shape[1] + 2:
        return y - np.nanmean(y)
    beta, *_ = np.linalg.lstsq(x[finite], y[finite], rcond=None)
    pred = x @ beta
    out = y - pred
    out[~mask] = np.nan
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return None, None, int(mask.sum())
    rho, p = spearmanr(x[mask], y[mask])
    if not np.isfinite(rho):
        return None, None, int(mask.sum())
    return float(rho), float(p), int(mask.sum())


def shuffle_pvalue(df: pd.DataFrame, feature: str, target: str, actual_abs: float, *, n_perm: int = 500) -> float:
    rng = np.random.default_rng(20260627)
    vals = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
    target_vals = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    hits = 0
    total = 0
    groups = df.groupby("dataset").indices
    for _ in range(n_perm):
        shuffled = vals.copy()
        for idx in groups.values():
            idx_arr = np.asarray(list(idx), dtype=int)
            shuffled[idx_arr] = rng.permutation(shuffled[idx_arr])
        rho, _, n = spearman(shuffled, target_vals)
        if rho is None or n < 10:
            continue
        total += 1
        if abs(rho) >= actual_abs:
            hits += 1
    return float((hits + 1) / (total + 1)) if total else 1.0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s0, dataset_bg = read_s0_backgrounds()
    residual = pd.read_csv(RESIDUAL_ROWS)
    residual["gene_key"] = residual["gene"].map(gene_key)
    residual["s0_key"] = list(zip(residual["dataset"].astype(str), residual["condition"].astype(str)))
    s0_small = s0[["dataset", "condition", "cell_background_source", "n_cells_float", "canonical_seed42_membership"]].copy()
    joined = residual.merge(s0_small, on=["dataset", "condition"], how="left")
    joined["background_source_used"] = joined["cell_background_source"].fillna("")
    missing_bg = joined["background_source_used"].astype(str).str.len() == 0
    joined.loc[missing_bg, "background_source_used"] = joined.loc[missing_bg, "dataset"].map(dataset_bg).fillna("")
    joined["background_mapping_source"] = np.where(missing_bg, "dataset_modal_s0_background", "exact_s0_condition_background")

    wanted = set(joined["gene_key"].dropna().astype(str))
    feature_df = load_gene_features(wanted)
    feature_df.to_csv(OUT_FEATURES, index=False)

    rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        tissues, status, reason = map_background(norm_text(row.get("background_source_used")))
        if not tissues:
            rows.append({**row.to_dict(), "mapped_tissue": "", "map_status": status, "map_reason": reason})
            continue
        for tissue in tissues:
            rows.append({**row.to_dict(), "mapped_tissue": tissue, "map_status": status, "map_reason": reason})
    expanded = pd.DataFrame(rows)
    merged = expanded.merge(feature_df, left_on=["mapped_tissue", "gene_key"], right_on=["tissue", "gene_key"], how="left")
    for feat in FEATURES:
        merged[feat] = pd.to_numeric(merged.get(feat), errors="coerce").fillna(0.0)
    merged.to_csv(OUT_JOIN, index=False)

    analyzable = merged[(merged["map_status"] == "mapped_single_tissue") & (merged["atlas_gene_present"] > 0)].copy()
    controls = pd.DataFrame(
        {
            "dataset": analyzable["dataset"].astype(str),
            "background": analyzable["background_source_used"].astype(str),
            "group": analyzable["group"].astype(str),
            "n_cells": pd.to_numeric(analyzable["n_cells_float"], errors="coerce").fillna(0.0),
        }
    )
    assoc_rows = []
    for feature in FEATURES:
        if feature == "atlas_gene_present":
            continue
        x_raw = pd.to_numeric(analyzable[feature], errors="coerce").to_numpy(dtype=float)
        x_resid = residualize(analyzable[feature], controls)
        for target in TARGETS:
            y_raw = pd.to_numeric(analyzable[target], errors="coerce").to_numpy(dtype=float)
            y_resid = residualize(analyzable[target], controls)
            rho, p, n = spearman(x_raw, y_raw)
            rrho, rp, rn = spearman(x_resid, y_resid)
            shuffle_p = shuffle_pvalue(analyzable, feature, target, abs(rho or 0.0)) if rho is not None else 1.0
            assoc_rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "n": n,
                    "raw_spearman": rho,
                    "raw_p": p,
                    "resid_spearman": rrho,
                    "resid_p": rp,
                    "resid_n": rn,
                    "within_dataset_shuffle_p_abs_ge_actual": shuffle_p,
                }
            )
    assoc = pd.DataFrame(assoc_rows)
    assoc_path = OUT_DIR / "control_manifold_association_summary.csv"
    assoc.to_csv(assoc_path, index=False)
    best = assoc.sort_values(
        ["resid_p", "within_dataset_shuffle_p_abs_ge_actual"],
        na_position="last",
    ).head(10)

    map_counts = Counter(merged["map_status"].astype(str))
    map_tissue_counts = Counter(merged["mapped_tissue"].astype(str))
    analyzable_summary = {
        "rows": int(len(analyzable)),
        "datasets": int(analyzable["dataset"].nunique()) if len(analyzable) else 0,
        "backgrounds": int(analyzable["background_source_used"].nunique()) if len(analyzable) else 0,
        "genes": int(analyzable["gene_key"].nunique()) if len(analyzable) else 0,
        "groups": dict(Counter(analyzable["group"].astype(str))),
    }
    nonconstant_within_background = 0
    for feature in FEATURES:
        if feature == "atlas_gene_present":
            continue
        varying = 0
        for _, part in analyzable.groupby("background_source_used"):
            if pd.to_numeric(part[feature], errors="coerce").nunique(dropna=True) > 1:
                varying += 1
        nonconstant_within_background += int(varying > 0)

    pass_reasons = []
    fail_reasons = []
    if analyzable_summary["rows"] >= 50 and analyzable_summary["datasets"] >= 3:
        pass_reasons.append("source_map_size_screen_pass")
    else:
        fail_reasons.append("source_map_size_screen_fail")
    if nonconstant_within_background > 0:
        pass_reasons.append("nonconstant_within_background_features_present")
    else:
        fail_reasons.append("no_nonconstant_within_background_feature")
    strong = assoc[
        (assoc["n"] >= 50)
        & (assoc["raw_spearman"].abs() >= 0.20)
        & (assoc["resid_spearman"].abs() >= 0.20)
        & (assoc["resid_p"] <= 0.05)
        & (assoc["within_dataset_shuffle_p_abs_ge_actual"] <= 0.05)
    ]
    if len(strong):
        pass_reasons.append("association_survives_residual_and_shuffle")
    else:
        fail_reasons.append("no_association_survives_residual_and_shuffle")
    if map_counts.get("ambiguous_multi_tissue", 0) or map_counts.get("unmapped", 0):
        fail_reasons.append("background_mapping_incomplete_or_ambiguous")
    fail_reasons.append("no_candidate_pp_delta_no_mmd_noharm_from_source_map_only")

    status = "external_control_manifold_source_map_gate_fail_no_gpu"
    gpu_authorized = False
    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_only": True,
            "training": False,
            "inference": False,
            "gpu": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "profile": "cellgene_census_processed_blood_lung_rank_binned_h5ad_only",
        },
        "map_counts": dict(map_counts),
        "mapped_tissue_counts": dict(map_tissue_counts),
        "analyzable_summary": analyzable_summary,
        "nonconstant_within_background_feature_count": nonconstant_within_background,
        "best_associations": best.to_dict(orient="records"),
        "pass_reasons": pass_reasons,
        "fail_reasons": fail_reasons,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "join_rows": str(OUT_JOIN),
            "gene_features": str(OUT_FEATURES),
            "association_summary": str(assoc_path),
        },
        "decision": "No GPU. External census control manifolds are useful source-map diagnostics, but this gate does not yield a candidate no-harm training signal.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# External Control-Manifold Source-Map Gate",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU-only source-map / anti-proxy gate over local processed cellgene-census blood/lung h5ad files.",
        "- No training, inference, GPU, canonical multi selection, or Track C query.",
        "- Tabula Sapiens large h5ad files were not read.",
        "",
        "## Mapping",
        "",
        f"- map counts: `{dict(map_counts)}`",
        f"- mapped tissue counts: `{dict(map_tissue_counts)}`",
        f"- analyzable rows: `{analyzable_summary['rows']}`",
        f"- analyzable datasets/backgrounds/genes: `{analyzable_summary['datasets']}` / `{analyzable_summary['backgrounds']}` / `{analyzable_summary['genes']}`",
        f"- nonconstant-within-background feature count: `{nonconstant_within_background}`",
        "",
        "## Best Associations",
        "",
        "| feature | target | n | raw rho | resid rho | resid p | shuffle p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| `{row['feature']}` | `{row['target']}` | {int(row['n']) if pd.notna(row['n']) else 0} | "
            f"{row['raw_spearman'] if pd.notna(row['raw_spearman']) else 'NA'} | "
            f"{row['resid_spearman'] if pd.notna(row['resid_spearman']) else 'NA'} | "
            f"{row['resid_p'] if pd.notna(row['resid_p']) else 'NA'} | "
            f"{row['within_dataset_shuffle_p_abs_ge_actual'] if pd.notna(row['within_dataset_shuffle_p_abs_ge_actual']) else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No GPU is authorized. The source map has some analyzable gene-level overlap, but it is incomplete/ambiguous for many perturbation backgrounds and this gate produces no candidate pp/MMD no-harm delta. Any future use would need a predeclared training policy and a separate no-harm gate.",
            "",
            "## Reasons",
            "",
            *[f"- `{reason}`" for reason in fail_reasons],
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- join rows: `{OUT_JOIN}`",
            f"- gene features: `{OUT_FEATURES}`",
            f"- association summary: `{assoc_path}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": gpu_authorized, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
