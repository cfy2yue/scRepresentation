#!/usr/bin/env python3
"""Train-set residual cluster/density information gate.

CPU/report-only. This tests whether train-only residual cluster coverage and
local density metrics explain frozen downstream outcomes beyond exact coverage,
condition count, and source/background controls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

for _key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_key, "4")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from materialize_latentfm_trainonly_condition_residual_information_20260628 import (  # noqa: E402
    ConditionVectors,
    collect_needed_conditions,
    load_condition_vector_cache,
    load_json,
)


OUTCOME_JOIN = (
    ROOT
    / "reports/exact_response_information_clustered_ci_combined_20260628/"
    / "exact_response_information_outcome_join_rows.csv"
)
TRAINONLY_RESIDUAL_ROWS = (
    ROOT
    / "reports/trainonly_condition_residual_information_20260628/"
    / "trainonly_condition_residual_information_rows.csv"
)
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
COND_META = DATA_DIR / "condition_metadata.json"
OUT_DIR = ROOT / "reports/trainset_cluster_density_information_gate_20260629"
PARENT_SPLIT_NAME = "split_seed42_xverse_trainonly_scaling_cap120_all_v2"
OUTCOMES = ["family_mmd_delta", "tail_score", "family_pp_delta", "cross_pp_delta"]
CLUSTER_PREDICTORS = [
    "residual_cluster_effective_count",
    "cluster_entropy_norm",
    "parent_cluster_coverage_fraction",
    "density_per_condition",
    "median_local_kernel_density",
    "low_density_tail_fraction",
    "max_cluster_share",
]
RANDOM_PREDICTORS = [
    "random_cluster_effective_count_mean",
    "random_cluster_entropy_norm_mean",
    "random_parent_cluster_coverage_fraction_mean",
    "random_density_per_condition_mean",
    "random_median_local_kernel_density_mean",
    "random_low_density_tail_fraction_mean",
]
CONTROL_SETS = {
    "base_exact": [
        "n_train_conditions",
        "base_dataset_effective_count",
        "base_background_effective_count",
        "exact_condition_fraction",
    ],
    "base_exact_type_target": [
        "n_train_conditions",
        "base_dataset_effective_count",
        "base_background_effective_count",
        "base_perturbation_type_effective_count",
        "base_target_gene_effective_count",
        "exact_condition_fraction",
    ],
}


def set_low_thread_env() -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ.setdefault(key, "1")


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def entropy_from_counts(counts: np.ndarray) -> tuple[float, float, float]:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0, 0.0, 0.0
    probs = counts.astype(float) / total
    probs = probs[probs > 0]
    ent = -float(np.sum(probs * np.log(probs)))
    effective = math.exp(ent)
    entropy_norm = ent / math.log(len(counts)) if len(counts) > 1 else 0.0
    max_share = float(np.max(counts) / total)
    return effective, entropy_norm, max_share


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    cols = []
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.isfinite(col).any():
            col = np.where(np.isfinite(col), col, float(np.nanmean(col)))
        if float(np.nanstd(col)) > 1e-12:
            cols.append((col - float(np.nanmean(col))) / float(np.nanstd(col)))
    design = np.column_stack([np.ones(len(y)), *cols]) if cols else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def residual_spearman(df: pd.DataFrame, predictor: str, outcome: str, controls: list[str]) -> dict[str, Any]:
    cols = [predictor, outcome, *controls]
    part = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(part) < max(8, len(controls) + 5) or part[predictor].nunique() < 3 or part[outcome].nunique() < 3:
        return {"n": int(len(part)), "rho": float("nan"), "p_value": float("nan")}
    x_resid = residualize(part[predictor].to_numpy(dtype=float), part[controls].to_numpy(dtype=float))
    y_resid = residualize(part[outcome].to_numpy(dtype=float), part[controls].to_numpy(dtype=float))
    if float(np.std(x_resid)) <= 1e-12 or float(np.std(y_resid)) <= 1e-12:
        return {"n": int(len(part)), "rho": float("nan"), "p_value": float("nan")}
    rho, p_value = spearmanr(x_resid, y_resid)
    return {"n": int(len(part)), "rho": float(rho), "p_value": float(p_value)}


def choose_split_rows(outcome: pd.DataFrame, residual_rows: pd.DataFrame) -> list[dict[str, str]]:
    needed = set(outcome["split_name"].astype(str))
    rows: list[dict[str, str]] = []
    for split_name in sorted(needed | {PARENT_SPLIT_NAME}):
        hits = residual_rows[residual_rows["split_name"].astype(str) == split_name].copy()
        if hits.empty:
            raise RuntimeError(f"missing split_file mapping for split_name={split_name}")
        hits["prefer"] = hits["split_file"].astype(str).str.contains("nested").astype(int)
        hit = hits.sort_values(["prefer", "split_file"]).iloc[0]
        rows.append(
            {
                "split_name": str(hit["split_name"]),
                "split_file": str(hit["split_file"]),
                "n_train_conditions": str(hit.get("n_train_conditions_declared", "")),
            }
        )
    return rows


def collect_split_matrix(
    split_row: dict[str, str],
    cache: dict[str, dict[str, ConditionVectors]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    split = load_json(ROOT / split_row["split_file"])
    residuals: list[np.ndarray] = []
    condition_rows: list[dict[str, Any]] = []
    for dataset, groups in split.items():
        for condition in groups.get("train", []):
            condition_s = str(condition)
            vectors = cache.get(dataset, {}).get(condition_s)
            if vectors is None:
                continue
            residuals.append(vectors.residual)
            condition_rows.append(
                {
                    "split_name": split_row["split_name"],
                    "dataset": dataset,
                    "condition": condition_s,
                    "n_ctrl": vectors.n_ctrl,
                    "n_gt": vectors.n_gt,
                    "response_norm": float(np.linalg.norm(vectors.residual)),
                }
            )
    matrix = np.vstack(residuals).astype(np.float64) if residuals else np.zeros((0, 0), dtype=np.float64)
    return matrix, condition_rows


def fit_reference_space(parent_matrix: np.ndarray, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray, KMeans, float, float]:
    if parent_matrix.shape[0] < 4:
        raise RuntimeError("parent matrix has too few rows for reference clustering")
    mean = parent_matrix.mean(axis=0, keepdims=True)
    std = parent_matrix.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    parent_z = (parent_matrix - mean) / std
    k = min(n_clusters, max(2, int(parent_z.shape[0])))
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=seed, max_iter=300)
    kmeans.fit(parent_z)
    sigma2 = estimate_sigma2(parent_z, max_points=768, seed=seed)
    parent_density = local_density(parent_z, sigma2, max_points=768, seed=seed)
    parent_low_density_threshold = float(np.quantile(parent_density, 0.25)) if len(parent_density) else 0.0
    return mean, std, kmeans, sigma2, parent_low_density_threshold


def estimate_sigma2(matrix: np.ndarray, max_points: int, seed: int) -> float:
    x = matrix
    if x.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(x.shape[0], size=max_points, replace=False))
        x = x[idx]
    diffs = x[:, None, :] - x[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    tri = d2[np.triu_indices_from(d2, k=1)]
    positive = tri[tri > 1e-12]
    return float(np.median(positive)) if positive.size else 1.0


def local_density(matrix: np.ndarray, sigma2: float, max_points: int, seed: int) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.zeros((matrix.shape[0],), dtype=float)
    x = matrix
    if x.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(x.shape[0], size=max_points, replace=False))
        x = x[idx]
    diffs = x[:, None, :] - x[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    kernel = np.exp(-d2 / max(2.0 * sigma2, 1e-12))
    np.fill_diagonal(kernel, np.nan)
    return np.nanmean(kernel, axis=1)


def summarize_labels(labels: np.ndarray, k: int) -> tuple[float, float, float, float]:
    counts = np.bincount(labels, minlength=k)
    effective, entropy_norm, max_share = entropy_from_counts(counts)
    coverage = float(np.count_nonzero(counts) / max(k, 1))
    return effective, entropy_norm, coverage, max_share


def random_cluster_controls(
    n: int,
    parent_probs: np.ndarray,
    parent_density: np.ndarray,
    parent_low_density_threshold: float,
    n_reps: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    k = len(parent_probs)
    effs: list[float] = []
    ents: list[float] = []
    covs: list[float] = []
    maxs: list[float] = []
    med_density: list[float] = []
    tail_density: list[float] = []
    for _ in range(n_reps):
        labels = rng.choice(k, size=n, replace=True, p=parent_probs)
        eff, ent, cov, max_share = summarize_labels(labels, k)
        effs.append(eff)
        ents.append(ent)
        covs.append(cov)
        maxs.append(max_share)
        dens = rng.choice(parent_density, size=min(n, max(len(parent_density), 1)), replace=True)
        med_density.append(float(np.median(dens)) if len(dens) else 0.0)
        tail_density.append(float(np.mean(dens < parent_low_density_threshold)) if len(dens) else 0.0)
    return {
        "random_cluster_effective_count_mean": float(np.mean(effs)),
        "random_cluster_entropy_norm_mean": float(np.mean(ents)),
        "random_parent_cluster_coverage_fraction_mean": float(np.mean(covs)),
        "random_max_cluster_share_mean": float(np.mean(maxs)),
        "random_density_per_condition_mean": float(np.mean(effs) / max(n, 1)),
        "random_median_local_kernel_density_mean": float(np.mean(med_density)),
        "random_low_density_tail_fraction_mean": float(np.mean(tail_density)),
    }


def build_density_metrics(
    split_rows: list[dict[str, str]],
    cache: dict[str, dict[str, ConditionVectors]],
    n_clusters: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matrices: dict[str, np.ndarray] = {}
    condition_rows: list[dict[str, Any]] = []
    for row in split_rows:
        matrix, cond_rows = collect_split_matrix(row, cache)
        matrices[row["split_name"]] = matrix
        condition_rows.extend(cond_rows)
    parent_matrix = matrices[PARENT_SPLIT_NAME]
    mean, std, kmeans, sigma2, low_threshold = fit_reference_space(parent_matrix, n_clusters, seed)
    parent_z = (parent_matrix - mean) / std
    parent_labels = kmeans.predict(parent_z)
    parent_counts = np.bincount(parent_labels, minlength=kmeans.n_clusters).astype(float)
    parent_probs = parent_counts / parent_counts.sum()
    parent_density = local_density(parent_z, sigma2, max_points=768, seed=seed)

    metric_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(split_rows):
        matrix = matrices[row["split_name"]]
        n = int(matrix.shape[0])
        if n == 0:
            continue
        z = (matrix - mean) / std
        labels = kmeans.predict(z)
        eff, ent, coverage, max_share = summarize_labels(labels, kmeans.n_clusters)
        density = local_density(z, sigma2, max_points=768, seed=seed + idx + 1)
        random_controls = random_cluster_controls(
            n=n,
            parent_probs=parent_probs,
            parent_density=parent_density,
            parent_low_density_threshold=low_threshold,
            n_reps=100,
            seed=seed + 1000 + idx,
        )
        metric_rows.append(
            {
                "split_name": row["split_name"],
                "split_file_for_density": row["split_file"],
                "n_train_conditions_cluster_density": n,
                "reference_parent_split_name": PARENT_SPLIT_NAME,
                "reference_n_clusters": int(kmeans.n_clusters),
                "reference_sigma2": sigma2,
                "reference_low_density_q25": low_threshold,
                "residual_cluster_effective_count": eff,
                "cluster_entropy_norm": ent,
                "parent_cluster_coverage_fraction": coverage,
                "max_cluster_share": max_share,
                "density_per_condition": eff / max(n, 1),
                "median_local_kernel_density": float(np.median(density)) if len(density) else 0.0,
                "mean_local_kernel_density": float(np.mean(density)) if len(density) else 0.0,
                "low_density_tail_fraction": float(np.mean(density < low_threshold)) if len(density) else 0.0,
                **random_controls,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(condition_rows), {
        "parent_split_name": PARENT_SPLIT_NAME,
        "parent_train_conditions": int(parent_matrix.shape[0]),
        "n_clusters": int(kmeans.n_clusters),
        "sigma2": sigma2,
        "low_density_q25": low_threshold,
        "parent_cluster_counts": parent_counts.astype(int).tolist(),
    }


def load_join_with_metrics(outcome_join: Path, residual_rows_csv: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    outcome = pd.read_csv(outcome_join)
    residual_rows = pd.read_csv(residual_rows_csv)
    residual_one = (
        residual_rows.sort_values("split_file")
        .drop_duplicates(subset=["split_name"], keep="first")
        .loc[:, ["split_name", "split_file"]]
    )
    joined = outcome.merge(metrics, on="split_name", how="left", validate="many_to_one")
    joined = joined.merge(residual_one, on="split_name", how="left", validate="many_to_one", suffixes=("", "_residual_source"))
    if "n_train_conditions" not in joined.columns:
        train_cols = [c for c in ["n_train_conditions_y", "n_train_conditions_x"] if c in joined.columns]
        if train_cols:
            joined["n_train_conditions"] = joined[train_cols[0]]
    return joined


def lodo_same_sign(joined: pd.DataFrame, predictor: str, outcome: str, controls: list[str], full_rho: float) -> float:
    if not math.isfinite(full_rho) or full_rho == 0:
        return float("nan")
    full_sign = math.copysign(1.0, full_rho)
    signs: list[bool] = []
    for leave_col in ["source_family", "axis_family"]:
        for leave_value in sorted(joined[leave_col].astype(str).unique()):
            sub = joined[joined[leave_col].astype(str) != leave_value]
            rho = residual_spearman(sub, predictor, outcome, controls)["rho"]
            if math.isfinite(rho) and rho != 0:
                signs.append(math.copysign(1.0, rho) == full_sign)
    return float(np.mean(signs)) if signs else float("nan")


def bootstrap_ci(
    joined: pd.DataFrame,
    predictor: str,
    outcome: str,
    controls: list[str],
    seed: int,
    n_boot: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = sorted(joined["source_family"].astype(str).unique())
    values: list[float] = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        parts = []
        for group in sampled_groups:
            part = joined[joined["source_family"].astype(str) == group]
            parts.append(part.sample(n=len(part), replace=True, random_state=int(rng.integers(0, 2**31 - 1))))
        boot = pd.concat(parts, ignore_index=True)
        rho = residual_spearman(boot, predictor, outcome, controls)["rho"]
        if math.isfinite(rho):
            values.append(float(rho))
    if len(values) < max(50, n_boot // 10):
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def stratified_permutation_p(
    joined: pd.DataFrame,
    predictor: str,
    outcome: str,
    controls: list[str],
    observed_rho: float,
    seed: int,
    n_perm: int,
) -> float:
    if not math.isfinite(observed_rho):
        return float("nan")
    rng = np.random.default_rng(seed)
    null_abs: list[float] = []
    for _ in range(n_perm):
        perm = joined.copy()
        parts = []
        for _, part in perm.groupby("source_family", sort=False):
            part = part.copy()
            if len(part) > 1:
                part[predictor] = rng.permutation(part[predictor].to_numpy())
            parts.append(part)
        perm = pd.concat(parts, ignore_index=True)
        rho = residual_spearman(perm, predictor, outcome, controls)["rho"]
        if math.isfinite(rho):
            null_abs.append(abs(float(rho)))
    if not null_abs:
        return float("nan")
    return float((1 + sum(v >= abs(observed_rho) for v in null_abs)) / (len(null_abs) + 1))


def build_associations(joined: pd.DataFrame, seed: int, n_boot: int, n_perm: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    predictor_meta = {p: "cluster_density" for p in CLUSTER_PREDICTORS}
    predictor_meta.update({p: "random_cluster_control" for p in RANDOM_PREDICTORS})
    for predictor, family in predictor_meta.items():
        if predictor not in joined.columns:
            continue
        for outcome in OUTCOMES:
            for control_name, controls in CONTROL_SETS.items():
                result = residual_spearman(joined, predictor, outcome, controls)
                rho = result["rho"]
                ci_low, ci_high = bootstrap_ci(
                    joined,
                    predictor,
                    outcome,
                    controls,
                    seed=seed + len(rows) * 17,
                    n_boot=n_boot,
                )
                perm_p = stratified_permutation_p(
                    joined,
                    predictor,
                    outcome,
                    controls,
                    observed_rho=rho,
                    seed=seed + len(rows) * 31,
                    n_perm=n_perm,
                )
                rows.append(
                    {
                        "predictor": predictor,
                        "predictor_family": family,
                        "outcome": outcome,
                        "control_set": control_name,
                        "controls": ";".join(controls),
                        "n": result["n"],
                        "residual_spearman_rho": rho,
                        "p_value": result["p_value"],
                        "bootstrap_ci_low": ci_low,
                        "bootstrap_ci_high": ci_high,
                        "ci_excludes_zero": bool(
                            math.isfinite(ci_low)
                            and math.isfinite(ci_high)
                            and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))
                        ),
                        "lodo_same_sign_rate": lodo_same_sign(joined, predictor, outcome, controls, rho),
                        "source_stratified_permutation_p": perm_p,
                    }
                )
    return pd.DataFrame(rows)


def decide(assoc: pd.DataFrame) -> tuple[str, list[str], list[dict[str, Any]]]:
    random = assoc[assoc["predictor_family"] == "random_cluster_control"].copy()
    passing: list[dict[str, Any]] = []
    for _, row in assoc[assoc["predictor_family"] == "cluster_density"].iterrows():
        if row["outcome"] not in {"family_mmd_delta", "tail_score"}:
            continue
        if row["control_set"] != "base_exact_type_target":
            continue
        rho = float(row["residual_spearman_rho"])
        p_value = float(row["p_value"])
        perm_p = float(row["source_stratified_permutation_p"])
        lodo = float(row["lodo_same_sign_rate"])
        random_same = random[
            (random["outcome"] == row["outcome"])
            & (random["control_set"] == row["control_set"])
            & (random["residual_spearman_rho"].abs() >= abs(rho) * 0.9)
            & (random["source_stratified_permutation_p"] <= 0.1)
        ]
        pass_gate = (
            math.isfinite(rho)
            and abs(rho) >= 0.55
            and p_value <= 0.05
            and bool(row["ci_excludes_zero"])
            and math.isfinite(lodo)
            and lodo >= 0.8
            and math.isfinite(perm_p)
            and perm_p <= 0.1
            and random_same.empty
        )
        if pass_gate:
            passing.append(row.to_dict())
    reasons: list[str] = []
    if not passing:
        reasons.append("no_cluster_density_axis_passes_strict_incremental_gate")
    status = (
        "trainset_cluster_density_information_gate_pass_design_only_no_gpu"
        if passing
        else "trainset_cluster_density_information_gate_no_passing_axis_no_gpu"
    )
    return status, reasons, passing


def write_report(
    out_md: Path,
    joined: pd.DataFrame,
    metrics: pd.DataFrame,
    assoc: pd.DataFrame,
    payload: dict[str, Any],
) -> None:
    primary = assoc[
        (assoc["control_set"] == "base_exact_type_target")
        & (assoc["outcome"].isin(["family_mmd_delta", "tail_score", "family_pp_delta", "cross_pp_delta"]))
    ].copy()
    primary["abs_rho"] = primary["residual_spearman_rho"].abs()
    primary = primary.sort_values("abs_rho", ascending=False).head(18)
    metric_display = metrics.sort_values("residual_cluster_effective_count", ascending=False).head(12)
    lines = [
        "# LatentFM Train-Set Cluster Density Information Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only gate over train conditions from existing split JSONs and xVERSE latent H5 bundles.",
        "- Reference clusters and kernel scale are fitted on the parent train split only.",
        "- Downstream outcomes are frozen posthoc rows; no training, inference, canonical multi, Track C query, or checkpoint selection.",
        "- This tests information-density as a scaling descriptor; even a pass authorizes only matched split-design work, not a model claim.",
        "",
        "## Reference Space",
        "",
        f"- Parent split: `{payload['reference']['parent_split_name']}`.",
        f"- Parent train conditions with vectors: `{payload['reference']['parent_train_conditions']}`.",
        f"- Reference clusters: `{payload['reference']['n_clusters']}`.",
        f"- Reference low-density threshold: `{fmt(payload['reference']['low_density_q25'])}`.",
        "",
        "## Split Metrics",
        "",
        "| split | n | cluster eff. | entropy | coverage | density/condition | median density | low-density tail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metric_display.iterrows():
        lines.append(
            f"| `{row['split_name']}` | `{int(row['n_train_conditions_cluster_density'])}` | "
            f"`{fmt(row['residual_cluster_effective_count'])}` | `{fmt(row['cluster_entropy_norm'])}` | "
            f"`{fmt(row['parent_cluster_coverage_fraction'])}` | `{fmt(row['density_per_condition'])}` | "
            f"`{fmt(row['median_local_kernel_density'])}` | `{fmt(row['low_density_tail_fraction'])}` |"
        )
    lines.extend(
        [
            "",
            "## Primary Associations",
            "",
            "| predictor | family | outcome | n | rho | p | boot CI | LODO sign | source-perm p |",
            "|---|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for _, row in primary.iterrows():
        lines.append(
            f"| `{row['predictor']}` | `{row['predictor_family']}` | `{row['outcome']}` | `{int(row['n'])}` | "
            f"`{fmt(row['residual_spearman_rho'])}` | `{fmt(row['p_value'])}` | "
            f"`[{fmt(row['bootstrap_ci_low'])}, {fmt(row['bootstrap_ci_high'])}]` | "
            f"`{fmt(row['lodo_same_sign_rate'])}` | `{fmt(row['source_stratified_permutation_p'])}` |"
        )
    lines.extend(["", "## Decision", ""])
    if payload["passing_axes"]:
        lines.extend(
            [
                "- At least one cluster/density axis passed the strict incremental association gate.",
                "- This still does not authorize GPU. Next action is a matched high/low split-design feasibility audit with dual controls and frozen no-harm.",
            ]
        )
    else:
        lines.extend(
            [
                "- No cluster/density axis passed the strict incremental gate after exact/count/source/background/type/target controls plus bootstrap, LODO, source-stratified permutation, and random-density controls.",
                "- Use these metrics as covariates/failure-analysis descriptors unless a later matched design finds stronger evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "## Subagent Integration",
            "",
            "- This implements Dirac rank-1: residual cluster/density split-association gate.",
            "- It intentionally does not relaunch generic OT minibatch pairing or exact/analog observability GPU.",
            "",
            "## Outputs",
            "",
            f"- Join rows: `{payload['outputs']['join_rows']}`",
            f"- Split density metrics: `{payload['outputs']['split_density_metrics']}`",
            f"- Condition rows: `{payload['outputs']['condition_rows']}`",
            f"- Associations: `{payload['outputs']['association_rows']}`",
            f"- JSON: `{payload['outputs']['json']}`",
            "",
            f"Outcome rows analyzed: `{len(joined)}`.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome-join", type=Path, default=OUTCOME_JOIN)
    parser.add_argument("--trainonly-residual-rows", type=Path, default=TRAINONLY_RESIDUAL_ROWS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-clusters", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-perm", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    set_low_thread_env()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_md = args.out_dir / "LATENTFM_TRAINSET_CLUSTER_DENSITY_INFORMATION_GATE_20260629.md"
    out_json = args.out_dir / "latentfm_trainset_cluster_density_information_gate_20260629.json"
    out_metrics = args.out_dir / "trainset_cluster_density_metrics.csv"
    out_conditions = args.out_dir / "trainset_cluster_density_condition_rows.csv"
    out_join = args.out_dir / "trainset_cluster_density_join_rows.csv"
    out_assoc = args.out_dir / "trainset_cluster_density_association_rows.csv"

    outcome = pd.read_csv(args.outcome_join)
    residual_rows = pd.read_csv(args.trainonly_residual_rows)
    split_rows = choose_split_rows(outcome, residual_rows)
    needed = collect_needed_conditions(split_rows)
    cache, missing_rows = load_condition_vector_cache(args.data_dir, needed)
    metrics, condition_rows, reference = build_density_metrics(split_rows, cache, args.n_clusters, args.seed)
    joined = load_join_with_metrics(args.outcome_join, args.trainonly_residual_rows, metrics)
    assoc = build_associations(joined, args.seed, args.n_boot, args.n_perm)
    status, reasons, passing = decide(assoc)

    metrics.to_csv(out_metrics, index=False)
    condition_rows.to_csv(out_conditions, index=False)
    joined.to_csv(out_join, index=False)
    assoc.to_csv(out_assoc, index=False)

    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "passing_axes": passing,
        "n_outcome_rows": int(len(joined)),
        "n_metric_rows": int(len(metrics)),
        "missing_condition_vectors": len(missing_rows),
        "reference": reference,
        "inputs": {
            "outcome_join": str(args.outcome_join),
            "trainonly_residual_rows": str(args.trainonly_residual_rows),
            "data_dir": str(args.data_dir),
        },
        "outputs": {
            "report": str(out_md),
            "json": str(out_json),
            "join_rows": str(out_join),
            "split_density_metrics": str(out_metrics),
            "condition_rows": str(out_conditions),
            "association_rows": str(out_assoc),
        },
        "closed_routes_preserved": [
            "generic_ot_minibatch_pairmode",
            "exact_or_analog_observability_gpu_axis",
            "generic_hvg_full_gene_smokes",
            "direct_state_context_gpu_smokes",
            "current_zscape_module_pathway_loss",
        ],
    }
    write_json(out_json, payload)
    write_report(out_md, joined, metrics, assoc, payload)
    print(json.dumps({"status": status, "reasons": reasons, "passing_axes": passing, "report": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
