#!/usr/bin/env python3
"""Missingness-adjusted gate for exact response-information coverage."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
JOIN_CSV = ROOT / "reports/exact_response_information_clustered_ci_20260628/exact_response_information_outcome_join_rows.csv"
OUT_DIR = ROOT / "reports/exact_response_information_missingness_adjusted_20260628"
OUT_MD = OUT_DIR / "LATENTFM_EXACT_RESPONSE_INFORMATION_MISSINGNESS_ADJUSTED_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_exact_response_information_missingness_adjusted_20260628.json"
OUT_ASSOC = OUT_DIR / "missingness_adjusted_association_rows.csv"
OUT_LODO = OUT_DIR / "missingness_adjusted_lodo_rows.csv"
OUT_DIAG = OUT_DIR / "coverage_confound_diagnostics.csv"

PREDICTOR = "exact_condition_fraction"
OUTCOMES = ["tail_score", "family_mmd_delta", "cross_pp_delta", "family_pp_delta"]
CONFOUNDS = [
    "n_train_conditions_y",
    "base_dataset_effective_count",
    "base_background_effective_count",
    "base_perturbation_type_effective_count",
    "base_target_gene_effective_count",
    "drug_condition_fraction",
    "gene_condition_fraction",
]
CATEGORICAL = ["source_family", "axis_family"]
BOOT_REPEATS = 5000
PERM_REPEATS = 5000
SEED = 47


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        out[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x = x.astype(float) - float(np.mean(x))
    y = y.astype(float) - float(np.mean(y))
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(ranks(x), ranks(y))


def design_matrix(frame: pd.DataFrame, include_categories: bool) -> np.ndarray:
    cols = []
    for col in CONFOUNDS:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        std = float(np.nanstd(values))
        if std > 0 and np.isfinite(std):
            values = (values - float(np.nanmean(values))) / std
            cols.append(values)
    if include_categories:
        for col in CATEGORICAL:
            dummies = pd.get_dummies(frame[col].astype(str), prefix=col, drop_first=True)
            for dummy_col in dummies.columns:
                vals = dummies[dummy_col].to_numpy(dtype=float)
                if vals.std() > 0:
                    cols.append(vals)
    if not cols:
        return np.empty((frame.shape[0], 0), dtype=float)
    return np.column_stack(cols)


def residualize(values: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    y = values.astype(float)
    if confounds.size == 0:
        return y - float(np.mean(y))
    x = np.column_stack([np.ones(len(y)), confounds])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ coef


def partial_spearman(frame: pd.DataFrame, outcome: str, include_categories: bool) -> float:
    need = [PREDICTOR, outcome] + CONFOUNDS + CATEGORICAL
    part = frame[need].replace([np.inf, -np.inf], np.nan).dropna()
    if part.shape[0] < 5:
        return float("nan")
    conf = design_matrix(part, include_categories=include_categories)
    x = residualize(ranks(part[PREDICTOR].to_numpy(dtype=float)), conf)
    y = residualize(ranks(part[outcome].to_numpy(dtype=float)), conf)
    return pearson(x, y)


def raw_spearman(frame: pd.DataFrame, outcome: str) -> float:
    part = frame[[PREDICTOR, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    if part.shape[0] < 3:
        return float("nan")
    return spearman(part[PREDICTOR].to_numpy(dtype=float), part[outcome].to_numpy(dtype=float))


def cluster_boot(frame: pd.DataFrame, outcome: str, include_categories: bool) -> tuple[float | None, float | None]:
    part = frame[[PREDICTOR, outcome, "split_name"] + CONFOUNDS + CATEGORICAL].replace([np.inf, -np.inf], np.nan).dropna()
    clusters = sorted(part["split_name"].astype(str).unique())
    if len(clusters) < 4:
        return None, None
    grouped = {cluster: part[part["split_name"].astype(str) == cluster] for cluster in clusters}
    rng = np.random.default_rng(SEED)
    vals: list[float] = []
    for _ in range(BOOT_REPEATS):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[c] for c in picked], ignore_index=True)
        rho = partial_spearman(boot, outcome, include_categories=include_categories)
        if np.isfinite(rho):
            vals.append(rho)
    if len(vals) < 10:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def stratified_permutation_p(frame: pd.DataFrame, outcome: str, include_categories: bool) -> float:
    part = frame[[PREDICTOR, outcome] + CONFOUNDS + CATEGORICAL].replace([np.inf, -np.inf], np.nan).dropna().copy()
    observed = partial_spearman(part, outcome, include_categories=include_categories)
    if part.shape[0] < 5 or not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(SEED + 1)
    extreme = 1
    strata = [g.index.to_numpy() for _, g in part.groupby(CATEGORICAL, sort=True)]
    for _ in range(PERM_REPEATS):
        shuffled = part.copy()
        values = shuffled[PREDICTOR].to_numpy(copy=True)
        for idx in strata:
            if len(idx) > 1:
                values[idx] = rng.permutation(values[idx])
        shuffled[PREDICTOR] = values
        val = partial_spearman(shuffled, outcome, include_categories=include_categories)
        if np.isfinite(val) and abs(val) >= abs(observed):
            extreme += 1
    return extreme / (PERM_REPEATS + 1)


def association_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        part = frame[[PREDICTOR, outcome, "split_name"] + CONFOUNDS + CATEGORICAL].replace([np.inf, -np.inf], np.nan).dropna()
        raw = raw_spearman(part, outcome)
        basic = partial_spearman(part, outcome, include_categories=False)
        basic_low, basic_high = cluster_boot(part, outcome, include_categories=False)
        full = partial_spearman(part, outcome, include_categories=True)
        full_low, full_high = cluster_boot(part, outcome, include_categories=True)
        rows.append(
            {
                "predictor": PREDICTOR,
                "outcome": outcome,
                "n_rows": int(part.shape[0]),
                "n_clusters": int(part["split_name"].astype(str).nunique()),
                "raw_spearman": raw,
                "partial_basic": basic,
                "partial_basic_ci95_low": basic_low,
                "partial_basic_ci95_high": basic_high,
                "partial_full": full,
                "partial_full_ci95_low": full_low,
                "partial_full_ci95_high": full_high,
                "partial_full_stratified_perm_p": stratified_permutation_p(part, outcome, include_categories=True),
                "full_ci_excludes_zero": bool(full_low is not None and full_high is not None and (full_low > 0 or full_high < 0)),
            }
        )
    return rows


def lodo_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        full = partial_spearman(frame, outcome, include_categories=True)
        for col in CATEGORICAL:
            for value in sorted(frame[col].astype(str).unique()):
                part = frame[frame[col].astype(str) != value]
                rho = partial_spearman(part, outcome, include_categories=True)
                rows.append(
                    {
                        "predictor": PREDICTOR,
                        "outcome": outcome,
                        "leave_col": col,
                        "leave_value": value,
                        "n_rows": int(part[[PREDICTOR, outcome]].dropna().shape[0]),
                        "partial_full": full,
                        "partial_leaveout": rho,
                        "same_sign": bool(
                            np.isfinite(full)
                            and np.isfinite(rho)
                            and (full == 0 or math.copysign(1.0, full) == math.copysign(1.0, rho))
                        ),
                    }
                )
    return rows


def diagnostics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for col in CONFOUNDS + ["exact_abundance_share_top1000_mean", "exact_hvg_share_top1000_mean"]:
        part = frame[[PREDICTOR, col]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "predictor": PREDICTOR,
                "confound": col,
                "n": int(part.shape[0]),
                "spearman": spearman(part[PREDICTOR].to_numpy(dtype=float), part[col].to_numpy(dtype=float))
                if part.shape[0] >= 3
                else float("nan"),
            }
        )
    return rows


def decide(rows: list[dict[str, Any]], lodo: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    reasons = []
    primary_tail = next(row for row in rows if row["outcome"] == "tail_score")
    primary_mmd = next(row for row in rows if row["outcome"] == "family_mmd_delta")
    for row, name in [(primary_tail, "tail_score"), (primary_mmd, "family_mmd_delta")]:
        if not row["full_ci_excludes_zero"]:
            reasons.append(f"{name}_full_adjusted_ci_crosses_zero")
    for outcome in ["tail_score", "family_mmd_delta"]:
        part = [row for row in lodo if row["outcome"] == outcome]
        if part:
            rate = sum(1 for row in part if row["same_sign"]) / len(part)
            if rate < 0.75:
                reasons.append(f"{outcome}_lodo_same_sign_rate_below_0.75")
    if reasons:
        return (
            "exact_response_information_missingness_adjusted_partial_no_gpu",
            reasons,
            "keep exact condition fraction as a strong but confounded scaling candidate; design matched splits or IPW/permutation controls before any GPU",
        )
    return (
        "exact_response_information_missingness_adjusted_pass_no_gpu",
        [],
        "promote exact condition fraction to a robust CPU scaling-law candidate; still no GPU without a new matched split launcher and no-harm gate",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(JOIN_CSV)
    rows = association_rows(frame)
    lodo = lodo_rows(frame)
    diag = diagnostics(frame)
    status, reasons, next_action = decide(rows, lodo)

    assoc_fields = [
        "predictor",
        "outcome",
        "n_rows",
        "n_clusters",
        "raw_spearman",
        "partial_basic",
        "partial_basic_ci95_low",
        "partial_basic_ci95_high",
        "partial_full",
        "partial_full_ci95_low",
        "partial_full_ci95_high",
        "partial_full_stratified_perm_p",
        "full_ci_excludes_zero",
    ]
    lodo_fields = ["predictor", "outcome", "leave_col", "leave_value", "n_rows", "partial_full", "partial_leaveout", "same_sign"]
    diag_fields = ["predictor", "confound", "n", "spearman"]
    write_csv(OUT_ASSOC, rows, assoc_fields)
    write_csv(OUT_LODO, lodo, lodo_fields)
    write_csv(OUT_DIAG, diag, diag_fields)

    payload = {
        "created_at": now_cst(),
        "status": status,
        "reasons": reasons,
        "next_action": next_action,
        "association_csv": str(OUT_ASSOC),
        "lodo_csv": str(OUT_LODO),
        "diagnostics_csv": str(OUT_DIAG),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Exact Response-Information Missingness-Adjusted Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only adjustment gate over exact response-information joined outcomes.",
        "* Partial Spearman controls numeric split confounds, and the full model also includes source/axis family dummies.",
        "* Cluster bootstrap samples by split cluster; permutation shuffles within source/axis strata.",
        "* No train/infer/GPU/canonical multi/Track C query/checkpoint selection.",
        "",
        "## Adjusted Associations",
        "",
        "| outcome | raw rho | basic partial | basic CI | full partial | full CI | stratified perm p |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["outcome"]),
                    fmt_float(row["raw_spearman"]),
                    fmt_float(row["partial_basic"]),
                    f"[{fmt_float(row['partial_basic_ci95_low'])}, {fmt_float(row['partial_basic_ci95_high'])}]",
                    fmt_float(row["partial_full"]),
                    f"[{fmt_float(row['partial_full_ci95_low'])}, {fmt_float(row['partial_full_ci95_high'])}]",
                    fmt_float(row["partial_full_stratified_perm_p"]),
                ]
            )
            + " |"
        )
    for outcome in ["tail_score", "family_mmd_delta"]:
        part = [row for row in lodo if row["outcome"] == outcome]
        rate = sum(1 for row in part if row["same_sign"]) / len(part) if part else float("nan")
        lines.append(f"\n* LODO same-sign rate for `{outcome}`: `{fmt_float(rate)}`.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"* Status: `{status}`.",
            f"* Reasons: `{', '.join(reasons) if reasons else 'none'}`.",
            f"* Next action: {next_action}.",
            "",
            "## Outputs",
            "",
            f"* Association rows: `{OUT_ASSOC}`",
            f"* LODO rows: `{OUT_LODO}`",
            f"* Confound diagnostics: `{OUT_DIAG}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
