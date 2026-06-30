#!/usr/bin/env python3
"""Clustered robustness gate for exact response-information scaling axes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
POSTHOC_DIR = ROOT / "reports/exact_response_information_posthoc_20260628"
SPLIT_MATRIX = POSTHOC_DIR / "exact_response_information_split_matrix.csv"
OUTCOME_CSV = ROOT / "reports/downstream_condition_residual_association_gate_20260628/condition_residual_information_outcome_join_rows.csv"
OUT_DIR = ROOT / "reports/exact_response_information_clustered_ci_20260628"
OUT_MD = OUT_DIR / "LATENTFM_EXACT_RESPONSE_INFORMATION_CLUSTERED_CI_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_exact_response_information_clustered_ci_20260628.json"
OUT_JOIN = OUT_DIR / "exact_response_information_outcome_join_rows.csv"
OUT_ASSOC = OUT_DIR / "exact_response_information_clustered_association_rows.csv"
OUT_LODO = OUT_DIR / "exact_response_information_lodo_rows.csv"

PREDICTORS = [
    "exact_condition_fraction",
    "exact_abundance_share_top1000_mean",
    "exact_hvg_share_top1000_mean",
    "exact_abundance_k80_mean",
    "exact_abundance_k90_mean",
]
OUTCOMES = ["cross_pp_delta", "family_pp_delta", "family_mmd_delta", "tail_score"]
BOOT_REPEATS = 5000
SEED = 46


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


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def corr_for_rows(rows: pd.DataFrame, predictor: str, outcome: str) -> float:
    part = rows[[predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    if part.shape[0] < 3:
        return float("nan")
    return spearman(part[predictor].to_numpy(dtype=float), part[outcome].to_numpy(dtype=float))


def clustered_bootstrap_ci(rows: pd.DataFrame, predictor: str, outcome: str) -> tuple[float | None, float | None, float | None]:
    part = rows[["split_name", predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    clusters = sorted(part["split_name"].astype(str).unique())
    if len(clusters) < 4:
        return None, None, None
    grouped = {cluster: part[part["split_name"].astype(str) == cluster] for cluster in clusters}
    rng = np.random.default_rng(SEED)
    vals: list[float] = []
    for _ in range(BOOT_REPEATS):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[c] for c in picked], ignore_index=True)
        rho = corr_for_rows(boot, predictor, outcome)
        if np.isfinite(rho):
            vals.append(rho)
    if len(vals) < 10:
        return None, None, None
    arr = np.asarray(vals, dtype=float)
    p_sign = float(min((arr <= 0).mean(), (arr >= 0).mean()) * 2.0)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), p_sign


def build_join() -> pd.DataFrame:
    matrix = pd.read_csv(SPLIT_MATRIX)
    outcomes = pd.read_csv(OUTCOME_CSV)
    predictor_cols = ["split_name"] + PREDICTORS + [
        "n_train_conditions",
        "exact_condition_rows",
        "base_dataset_effective_count",
        "base_background_effective_count",
        "base_perturbation_type_effective_count",
        "base_target_gene_effective_count",
    ]
    predictor_frame = matrix[predictor_cols].drop_duplicates()
    return outcomes.merge(predictor_frame, on="split_name", how="left", validate="many_to_one")


def association_rows(joined: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for predictor in PREDICTORS:
        for outcome in OUTCOMES:
            part = joined[[predictor, outcome, "split_name"]].replace([np.inf, -np.inf], np.nan).dropna()
            rho = corr_for_rows(part, predictor, outcome)
            ci_low, ci_high, p_sign = clustered_bootstrap_ci(part, predictor, outcome)
            out.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_rows": int(part.shape[0]),
                    "n_clusters": int(part["split_name"].astype(str).nunique()),
                    "rho": rho,
                    "cluster_boot_ci95_low": ci_low,
                    "cluster_boot_ci95_high": ci_high,
                    "cluster_boot_p_sign": p_sign,
                    "cluster_ci_excludes_zero": bool(
                        ci_low is not None and ci_high is not None and (ci_low > 0 or ci_high < 0)
                    ),
                }
            )
    return out


def lodo_rows(joined: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for predictor in PREDICTORS:
        for outcome in OUTCOMES:
            full = corr_for_rows(joined, predictor, outcome)
            for col in ["source_family", "axis_family"]:
                for value in sorted(joined[col].astype(str).unique()):
                    part = joined[joined[col].astype(str) != value]
                    rho = corr_for_rows(part, predictor, outcome)
                    out.append(
                        {
                            "predictor": predictor,
                            "outcome": outcome,
                            "leave_col": col,
                            "leave_value": value,
                            "n_rows": int(part[[predictor, outcome]].dropna().shape[0]),
                            "rho_full": full,
                            "rho_leaveout": rho,
                            "same_sign": bool(
                                np.isfinite(full)
                                and np.isfinite(rho)
                                and (full == 0 or math.copysign(1.0, full) == math.copysign(1.0, rho))
                            ),
                        }
                    )
    return out


def decide(assoc: list[dict[str, Any]], lodo: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    primary = next(
        (
            row
            for row in assoc
            if row["predictor"] == "exact_condition_fraction" and row["outcome"] == "family_mmd_delta"
        ),
        None,
    )
    if not primary:
        reasons.append("missing_primary_exact_condition_fraction_family_mmd_delta")
    elif not primary["cluster_ci_excludes_zero"]:
        reasons.append("primary_cluster_ci_does_not_exclude_zero")
    lodo_primary = [
        row
        for row in lodo
        if row["predictor"] == "exact_condition_fraction" and row["outcome"] == "family_mmd_delta"
    ]
    if lodo_primary:
        same_sign_rate = sum(1 for row in lodo_primary if row["same_sign"]) / len(lodo_primary)
        if same_sign_rate < 0.75:
            reasons.append("primary_lodo_same_sign_rate_below_0.75")
    else:
        reasons.append("missing_primary_lodo_rows")
    if reasons:
        return (
            "exact_response_information_clustered_ci_partial_no_gpu",
            reasons,
            "treat exact response-information associations as hypothesis-generating; add more independent outcome families before manuscript-level scaling claim",
        )
    return (
        "exact_response_information_clustered_ci_pass_no_gpu",
        [],
        "promote exact response-information coverage to a formal scaling-law candidate; next run clustered/LODO figure generation and preregistered interpretation",
    )


def main() -> None:
    global POSTHOC_DIR, SPLIT_MATRIX, OUT_DIR, OUT_MD, OUT_JSON, OUT_JOIN, OUT_ASSOC, OUT_LODO

    parser = argparse.ArgumentParser()
    parser.add_argument("--posthoc-dir", type=Path, default=POSTHOC_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    POSTHOC_DIR = args.posthoc_dir
    SPLIT_MATRIX = POSTHOC_DIR / "exact_response_information_split_matrix.csv"
    OUT_DIR = args.out_dir
    OUT_MD = OUT_DIR / "LATENTFM_EXACT_RESPONSE_INFORMATION_CLUSTERED_CI_20260628.md"
    OUT_JSON = OUT_DIR / "latentfm_exact_response_information_clustered_ci_20260628.json"
    OUT_JOIN = OUT_DIR / "exact_response_information_outcome_join_rows.csv"
    OUT_ASSOC = OUT_DIR / "exact_response_information_clustered_association_rows.csv"
    OUT_LODO = OUT_DIR / "exact_response_information_lodo_rows.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined = build_join()
    assoc = association_rows(joined)
    lodo = lodo_rows(joined)
    status, reasons, next_action = decide(assoc, lodo)

    joined_fields = list(joined.columns)
    write_csv(OUT_JOIN, joined.to_dict("records"), joined_fields)
    assoc_fields = [
        "predictor",
        "outcome",
        "n_rows",
        "n_clusters",
        "rho",
        "cluster_boot_ci95_low",
        "cluster_boot_ci95_high",
        "cluster_boot_p_sign",
        "cluster_ci_excludes_zero",
    ]
    lodo_fields = ["predictor", "outcome", "leave_col", "leave_value", "n_rows", "rho_full", "rho_leaveout", "same_sign"]
    write_csv(OUT_ASSOC, assoc, assoc_fields)
    write_csv(OUT_LODO, lodo, lodo_fields)

    payload = {
        "created_at": now_cst(),
        "status": status,
        "reasons": reasons,
        "next_action": next_action,
        "joined_rows": int(joined.shape[0]),
        "joined_clusters": int(joined["split_name"].astype(str).nunique()),
        "association_csv": str(OUT_ASSOC),
        "lodo_csv": str(OUT_LODO),
        "join_csv": str(OUT_JOIN),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    top = sorted(
        [row for row in assoc if np.isfinite(safe_float(row["rho"]))],
        key=lambda r: abs(safe_float(r["rho"])),
        reverse=True,
    )[:10]
    lines = [
        "# LatentFM Exact Response-Information Clustered CI Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only robustness gate over exact response-information posthoc outputs and frozen downstream outcomes.",
        "* Cluster bootstrap samples by `split_name`; LODO leaves out `source_family` and `axis_family`.",
        "* Does not train, infer, use canonical multi, use Track C query, or select checkpoints.",
        "",
        "## Summary",
        "",
        f"* Joined outcome rows: `{payload['joined_rows']}`.",
        f"* Split clusters: `{payload['joined_clusters']}`.",
        "",
        "## Top Clustered Associations",
        "",
        "| predictor | outcome | rows | clusters | rho | CI low | CI high | excludes 0 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in top:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["predictor"]),
                    str(row["outcome"]),
                    str(row["n_rows"]),
                    str(row["n_clusters"]),
                    fmt_float(row["rho"]),
                    fmt_float(row["cluster_boot_ci95_low"]),
                    fmt_float(row["cluster_boot_ci95_high"]),
                    str(row["cluster_ci_excludes_zero"]),
                ]
            )
            + " |"
        )
    primary_lodo = [
        row
        for row in lodo
        if row["predictor"] == "exact_condition_fraction" and row["outcome"] == "family_mmd_delta"
    ]
    if primary_lodo:
        same_sign_rate = sum(1 for row in primary_lodo if row["same_sign"]) / len(primary_lodo)
    else:
        same_sign_rate = float("nan")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"* Status: `{status}`.",
            f"* Reasons: `{', '.join(reasons) if reasons else 'none'}`.",
            f"* Primary LODO same-sign rate: `{fmt_float(same_sign_rate)}`.",
            f"* Next action: {next_action}.",
            "",
            "## Outputs",
            "",
            f"* Join rows: `{OUT_JOIN}`",
            f"* Association rows: `{OUT_ASSOC}`",
            f"* LODO rows: `{OUT_LODO}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
