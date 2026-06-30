#!/usr/bin/env python3
"""Train-set OT/sliced-Wasserstein coverage information gate.

CPU/report-only. This treats OT as aggregate coverage geometry to a parent
train distribution, not as minibatch pairing or a training loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
OUT_DIR = ROOT / "reports/trainset_ot_coverage_information_gate_20260629"
PARENT_SPLIT_NAME = "split_seed42_xverse_trainonly_scaling_cap120_all_v2"
OUTCOMES = ["family_mmd_delta", "tail_score", "family_pp_delta", "cross_pp_delta"]
PREDICTORS = [
    "residual_sliced_wasserstein_to_parent",
    "residual_ot_coverage_score",
    "residual_sliced_wasserstein_delta_vs_random",
    "residual_sliced_wasserstein_ratio_vs_random",
    "ctrl_sliced_wasserstein_to_parent",
    "gt_sliced_wasserstein_to_parent",
    "gt_minus_ctrl_sliced_wasserstein",
]
CONTROL_PREDICTORS = [
    "random_same_n_sliced_wasserstein_mean",
    "random_same_n_sliced_wasserstein_std",
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


def collect_split_matrices(
    split_row: dict[str, str],
    cache: dict[str, dict[str, ConditionVectors]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = load_json(ROOT / split_row["split_file"])
    residuals: list[np.ndarray] = []
    ctrls: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    for dataset, groups in split.items():
        for condition in groups.get("train", []):
            vectors = cache.get(dataset, {}).get(str(condition))
            if vectors is None:
                continue
            residuals.append(vectors.residual)
            ctrls.append(vectors.ctrl_mean)
            gts.append(vectors.gt_mean)
    if not residuals:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
        )
    return (
        np.vstack(residuals).astype(np.float64),
        np.vstack(ctrls).astype(np.float64),
        np.vstack(gts).astype(np.float64),
    )


def standardize(parent: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = parent.mean(axis=0, keepdims=True)
    std = parent.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (parent - mean) / std, (matrix - mean) / std, (mean, std)


def make_projection(dim: int, n_proj: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(dim, n_proj))
    norms = np.linalg.norm(proj, axis=0, keepdims=True)
    return proj / np.maximum(norms, 1e-12)


def quantile_grid(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((n,), dtype=float)
    qs = (np.arange(n) + 0.5) / n
    return np.quantile(values, qs)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, projection: np.ndarray, n_quantiles: int) -> float:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    x_proj = x @ projection
    y_proj = y @ projection
    distances: list[float] = []
    for j in range(projection.shape[1]):
        xq = quantile_grid(x_proj[:, j], n_quantiles)
        yq = quantile_grid(y_proj[:, j], n_quantiles)
        distances.append(float(np.sqrt(np.mean((xq - yq) ** 2))))
    return float(np.mean(distances))


def random_same_n_parent_distance(
    parent_z: np.ndarray,
    projection: np.ndarray,
    n: int,
    n_quantiles: int,
    n_reps: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_reps):
        idx = rng.choice(parent_z.shape[0], size=min(n, parent_z.shape[0]), replace=False)
        vals.append(sliced_wasserstein(parent_z[idx], parent_z, projection, n_quantiles))
    return float(np.mean(vals)), float(np.std(vals))


def build_ot_metrics(
    split_rows: list[dict[str, str]],
    cache: dict[str, dict[str, ConditionVectors]],
    n_proj: int,
    n_quantiles: int,
    n_random_reps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for row in split_rows:
        matrices[row["split_name"]] = collect_split_matrices(row, cache)
    parent_resid, parent_ctrl, parent_gt = matrices[PARENT_SPLIT_NAME]
    if parent_resid.shape[0] < 4:
        raise RuntimeError("parent split has too few residual vectors")
    parent_resid_z, _, (resid_mean, resid_std) = standardize(parent_resid, parent_resid)
    parent_ctrl_z, _, (ctrl_mean, ctrl_std) = standardize(parent_ctrl, parent_ctrl)
    parent_gt_z, _, (gt_mean, gt_std) = standardize(parent_gt, parent_gt)
    proj = make_projection(parent_resid_z.shape[1], n_proj, seed)

    rows: list[dict[str, Any]] = []
    parent_reference_distance = sliced_wasserstein(parent_resid_z, parent_resid_z, proj, n_quantiles)
    for idx, row in enumerate(split_rows):
        split_name = row["split_name"]
        residuals, ctrls, gts = matrices[split_name]
        if residuals.shape[0] == 0:
            continue
        residual_z = (residuals - resid_mean) / resid_std
        ctrl_z = (ctrls - ctrl_mean) / ctrl_std
        gt_z = (gts - gt_mean) / gt_std
        resid_dist = sliced_wasserstein(residual_z, parent_resid_z, proj, n_quantiles)
        ctrl_dist = sliced_wasserstein(ctrl_z, parent_ctrl_z, proj, n_quantiles)
        gt_dist = sliced_wasserstein(gt_z, parent_gt_z, proj, n_quantiles)
        random_mean, random_std = random_same_n_parent_distance(
            parent_resid_z,
            proj,
            n=residuals.shape[0],
            n_quantiles=n_quantiles,
            n_reps=n_random_reps,
            seed=seed + 1000 + idx,
        )
        rows.append(
            {
                "split_name": split_name,
                "split_file_for_ot": row["split_file"],
                "n_train_conditions_ot": int(residuals.shape[0]),
                "residual_sliced_wasserstein_to_parent": resid_dist,
                "ctrl_sliced_wasserstein_to_parent": ctrl_dist,
                "gt_sliced_wasserstein_to_parent": gt_dist,
                "gt_minus_ctrl_sliced_wasserstein": gt_dist - ctrl_dist,
                "random_same_n_sliced_wasserstein_mean": random_mean,
                "random_same_n_sliced_wasserstein_std": random_std,
                "residual_sliced_wasserstein_delta_vs_random": resid_dist - random_mean,
                "residual_sliced_wasserstein_ratio_vs_random": resid_dist / max(random_mean, 1e-12),
                "residual_ot_coverage_score": 1.0 - resid_dist / max(random_mean, 1e-12),
            }
        )
    reference = {
        "parent_split_name": PARENT_SPLIT_NAME,
        "parent_train_conditions": int(parent_resid.shape[0]),
        "n_projections": int(n_proj),
        "n_quantiles": int(n_quantiles),
        "n_random_reps": int(n_random_reps),
        "parent_self_distance": parent_reference_distance,
    }
    return pd.DataFrame(rows), reference


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
    if len(values) < max(30, n_boot // 10):
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
        parts = []
        for _, part in joined.groupby("source_family", sort=False):
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
    families = {p: "ot_coverage" for p in PREDICTORS}
    families.update({p: "random_same_n_control" for p in CONTROL_PREDICTORS})
    for predictor, family in families.items():
        if predictor not in joined.columns:
            continue
        for outcome in OUTCOMES:
            for control_name, controls in CONTROL_SETS.items():
                result = residual_spearman(joined, predictor, outcome, controls)
                rho = result["rho"]
                ci_low, ci_high = bootstrap_ci(joined, predictor, outcome, controls, seed + len(rows) * 17, n_boot)
                perm_p = stratified_permutation_p(
                    joined, predictor, outcome, controls, observed_rho=rho, seed=seed + len(rows) * 31, n_perm=n_perm
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
    passing: list[dict[str, Any]] = []
    control = assoc[assoc["predictor_family"] == "random_same_n_control"]
    for _, row in assoc[assoc["predictor_family"] == "ot_coverage"].iterrows():
        if row["outcome"] not in {"family_mmd_delta", "tail_score"}:
            continue
        if row["control_set"] != "base_exact_type_target":
            continue
        rho = float(row["residual_spearman_rho"])
        p_value = float(row["p_value"])
        perm_p = float(row["source_stratified_permutation_p"])
        lodo = float(row["lodo_same_sign_rate"])
        strong_control = control[
            (control["outcome"] == row["outcome"])
            & (control["control_set"] == row["control_set"])
            & (control["residual_spearman_rho"].abs() >= abs(rho) * 0.9)
            & (control["source_stratified_permutation_p"] <= 0.1)
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
            and strong_control.empty
        )
        if pass_gate:
            passing.append(row.to_dict())
    reasons = [] if passing else ["no_ot_coverage_axis_passes_strict_incremental_gate"]
    status = (
        "trainset_ot_coverage_information_gate_pass_design_only_no_gpu"
        if passing
        else "trainset_ot_coverage_information_gate_no_passing_axis_no_gpu"
    )
    return status, reasons, passing


def write_report(out_md: Path, metrics: pd.DataFrame, assoc: pd.DataFrame, payload: dict[str, Any]) -> None:
    metric_rows = metrics.sort_values("residual_sliced_wasserstein_to_parent").head(14)
    primary = assoc[assoc["control_set"] == "base_exact_type_target"].copy()
    primary["abs_rho"] = primary["residual_spearman_rho"].abs()
    primary = primary.sort_values("abs_rho", ascending=False).head(18)
    lines = [
        "# LatentFM Train-Set OT Coverage Information Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only aggregate sliced-Wasserstein coverage geometry.",
        "- Uses only train conditions from split JSONs and existing xVERSE latent H5 bundles.",
        "- Parent reference is the train-only cap120 all split; held-out Track C query and canonical multi are not read or selected.",
        "- This is not OT minibatch pairing and does not authorize generic OT pairmode relaunch.",
        "",
        "## Reference",
        "",
        f"- Parent split: `{payload['reference']['parent_split_name']}`.",
        f"- Parent train conditions: `{payload['reference']['parent_train_conditions']}`.",
        f"- Projections / quantiles / random reps: `{payload['reference']['n_projections']}` / `{payload['reference']['n_quantiles']}` / `{payload['reference']['n_random_reps']}`.",
        "",
        "## Closest Train Distributions To Parent",
        "",
        "| split | n | residual SW | random same-n SW | delta vs random | coverage score | ctrl SW | gt SW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metric_rows.iterrows():
        lines.append(
            f"| `{row['split_name']}` | `{int(row['n_train_conditions_ot'])}` | "
            f"`{fmt(row['residual_sliced_wasserstein_to_parent'])}` | "
            f"`{fmt(row['random_same_n_sliced_wasserstein_mean'])}` | "
            f"`{fmt(row['residual_sliced_wasserstein_delta_vs_random'])}` | "
            f"`{fmt(row['residual_ot_coverage_score'])}` | "
            f"`{fmt(row['ctrl_sliced_wasserstein_to_parent'])}` | "
            f"`{fmt(row['gt_sliced_wasserstein_to_parent'])}` |"
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
                "- At least one OT coverage axis passed the strict CPU gate.",
                "- This authorizes matched split-design work only; GPU requires high/low split feasibility and frozen no-harm protocol.",
            ]
        )
    else:
        lines.extend(
            [
                "- No OT coverage axis passed the strict incremental gate after exact/count/source/background/type/target controls plus bootstrap, LODO, source-stratified permutation, and same-n random controls.",
                "- Keep aggregate OT coverage as a diagnostic/covariate. Do not relaunch generic OT minibatch pairmode from this evidence.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Metrics: `{payload['outputs']['metrics']}`",
            f"- Join rows: `{payload['outputs']['join_rows']}`",
            f"- Associations: `{payload['outputs']['association_rows']}`",
            f"- JSON: `{payload['outputs']['json']}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome-join", type=Path, default=OUTCOME_JOIN)
    parser.add_argument("--trainonly-residual-rows", type=Path, default=TRAINONLY_RESIDUAL_ROWS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-projections", type=int, default=128)
    parser.add_argument("--n-quantiles", type=int, default=128)
    parser.add_argument("--n-random-reps", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    set_low_thread_env()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_md = args.out_dir / "LATENTFM_TRAINSET_OT_COVERAGE_INFORMATION_GATE_20260629.md"
    out_json = args.out_dir / "latentfm_trainset_ot_coverage_information_gate_20260629.json"
    out_metrics = args.out_dir / "trainset_ot_coverage_metrics.csv"
    out_join = args.out_dir / "trainset_ot_coverage_join_rows.csv"
    out_assoc = args.out_dir / "trainset_ot_coverage_association_rows.csv"

    outcome = pd.read_csv(args.outcome_join)
    residual_rows = pd.read_csv(args.trainonly_residual_rows)
    split_rows = choose_split_rows(outcome, residual_rows)
    needed = collect_needed_conditions(split_rows)
    cache, missing_rows = load_condition_vector_cache(args.data_dir, needed)
    metrics, reference = build_ot_metrics(
        split_rows,
        cache,
        n_proj=args.n_projections,
        n_quantiles=args.n_quantiles,
        n_random_reps=args.n_random_reps,
        seed=args.seed,
    )
    joined = load_join_with_metrics(args.outcome_join, args.trainonly_residual_rows, metrics)
    assoc = build_associations(joined, args.seed, args.n_boot, args.n_perm)
    status, reasons, passing = decide(assoc)

    metrics.to_csv(out_metrics, index=False)
    joined.to_csv(out_join, index=False)
    assoc.to_csv(out_assoc, index=False)
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "passing_axes": passing,
        "n_outcome_rows": int(len(joined)),
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
            "metrics": str(out_metrics),
            "join_rows": str(out_join),
            "association_rows": str(out_assoc),
        },
    }
    write_json(out_json, payload)
    write_report(out_md, metrics, assoc, payload)
    print(json.dumps({"status": status, "reasons": reasons, "passing_axes": passing, "report": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
