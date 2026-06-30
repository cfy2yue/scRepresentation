#!/usr/bin/env python3
"""Condition-level missingness/IPW gate for exact response information."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
INFO_CSV = ROOT / "reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv"
OUTCOME_CSV = ROOT / "reports/downstream_condition_residual_association_gate_20260628/condition_residual_information_outcome_join_rows.csv"
COVERAGE_CSV = ROOT / "runs/latentfm_exact_response_information_coverage_20260628/latentfm_exact_response_information_coverage_20260628_20260628_144814/outputs/exact_response_information_condition_rows.csv"

OUT_DIR = ROOT / "reports/exact_response_information_ipw_missingness_20260628"
OUT_MD = OUT_DIR / "LATENTFM_EXACT_RESPONSE_INFORMATION_IPW_MISSINGNESS_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_exact_response_information_ipw_missingness_20260628.json"
OUT_SPLIT = OUT_DIR / "ipw_missingness_split_rows.csv"
OUT_JOIN = OUT_DIR / "ipw_missingness_outcome_join_rows.csv"
OUT_ASSOC = OUT_DIR / "ipw_missingness_association_rows.csv"
OUT_PERM = OUT_DIR / "dataset_stratified_permutation_rows.csv"
OUT_DATASET = OUT_DIR / "coverage_dataset_rates.csv"

PREDICTORS = [
    "exact_condition_fraction",
    "expected_coverage_fraction",
    "residual_coverage_fraction",
    "standardized_residual_coverage",
    "ipw_coverage_fraction",
]
OUTCOMES = ["tail_score", "family_mmd_delta", "cross_pp_delta", "family_pp_delta"]
BOOT_REPEATS = 3000
PERM_REPEATS = 2000
SEED = 48


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


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def condition_type(meta: dict[str, Any]) -> str:
    raw = str(meta.get("perturbation_type_raw", "unknown"))
    return raw if raw and raw != "nan" else "unknown"


def gene_count_bin(meta: dict[str, Any]) -> str:
    genes = meta.get("genes", [])
    if not isinstance(genes, list):
        return "unknown"
    n = len(genes)
    if n <= 1:
        return "single"
    if n == 2:
        return "double"
    return "multi3plus"


def smoothed_rate(num: int, den: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    return (num + alpha) / max(den + alpha + beta, 1e-12)


def build_condition_table() -> pd.DataFrame:
    meta = load_json(COND_META)
    covered = {
        (str(row.dataset), str(row.condition))
        for row in pd.read_csv(COVERAGE_CSV).itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for dataset, conditions in meta.items():
        for condition, cond_meta in conditions.items():
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "covered": int((dataset, condition) in covered),
                    "perturbation_type": condition_type(cond_meta),
                    "gene_count_bin": gene_count_bin(cond_meta),
                    "target_gene_count": len(cond_meta.get("genes", [])) if isinstance(cond_meta.get("genes", []), list) else 0,
                }
            )
    frame = pd.DataFrame(rows)
    dataset_rates = frame.groupby("dataset")["covered"].agg(["sum", "count"]).reset_index()
    dataset_rate_map = {
        row.dataset: smoothed_rate(int(row["sum"]), int(row["count"]))
        for _, row in dataset_rates.iterrows()
    }
    type_rates = frame.groupby("perturbation_type")["covered"].agg(["sum", "count"]).reset_index()
    type_rate_map = {
        row.perturbation_type: smoothed_rate(int(row["sum"]), int(row["count"]))
        for _, row in type_rates.iterrows()
    }
    bin_rates = frame.groupby("gene_count_bin")["covered"].agg(["sum", "count"]).reset_index()
    bin_rate_map = {
        row.gene_count_bin: smoothed_rate(int(row["sum"]), int(row["count"]))
        for _, row in bin_rates.iterrows()
    }
    global_rate = smoothed_rate(int(frame["covered"].sum()), int(frame.shape[0]))
    p_hat = []
    for row in frame.itertuples(index=False):
        p = (
            0.70 * dataset_rate_map.get(row.dataset, global_rate)
            + 0.20 * type_rate_map.get(row.perturbation_type, global_rate)
            + 0.10 * bin_rate_map.get(row.gene_count_bin, global_rate)
        )
        p_hat.append(float(np.clip(p, 0.02, 0.98)))
    frame["p_hat"] = p_hat
    return frame


def build_split_rows(condition_frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, list[str]]]:
    cond_lookup = {
        (str(row.dataset), str(row.condition)): row
        for row in condition_frame.itertuples(index=False)
    }
    split_rows = []
    split_to_keys: dict[str, list[tuple[str, str]]] = {}
    split_info = pd.read_csv(INFO_CSV)
    seen_splits: set[str] = set()
    for row in split_info.itertuples(index=False):
        split_name = str(row.split_name)
        if split_name in seen_splits:
            continue
        seen_splits.add(split_name)
        split = load_json(ROOT / str(row.split_file))
        covered_values: list[float] = []
        p_values: list[float] = []
        keys: list[tuple[str, str]] = []
        missing_meta = 0
        for dataset, groups in split.items():
            for condition in groups.get("train", []):
                key = (str(dataset), str(condition))
                keys.append(key)
                cond = cond_lookup.get(key)
                if cond is None:
                    missing_meta += 1
                    continue
                covered_values.append(float(cond.covered))
                p_values.append(float(cond.p_hat))
        covered_arr = np.asarray(covered_values, dtype=float)
        p_arr = np.asarray(p_values, dtype=float)
        if covered_arr.size:
            residual = covered_arr - p_arr
            denom = math.sqrt(float(np.sum(p_arr * (1.0 - p_arr))))
            ipw_weights = 1.0 / np.clip(p_arr, 0.02, 0.98)
            ipw_fraction = float(np.sum(covered_arr * ipw_weights) / np.sum(ipw_weights))
            split_rows.append(
                {
                    "split_file": str(row.split_file),
                    "split_name": split_name,
                    "n_train_conditions": int(len(keys)),
                    "n_train_conditions_with_meta": int(covered_arr.size),
                    "missing_condition_metadata": int(missing_meta),
                    "exact_condition_fraction": float(covered_arr.mean()),
                    "expected_coverage_fraction": float(p_arr.mean()),
                    "residual_coverage_fraction": float(residual.mean()),
                    "standardized_residual_coverage": float(np.sum(residual) / denom) if denom > 0 else float("nan"),
                    "ipw_coverage_fraction": ipw_fraction,
                }
            )
        split_to_keys[split_name] = keys
    # Maps used by dataset-stratified permutation.
    condition_index_by_dataset: dict[str, dict[str, int]] = {}
    coverage_by_dataset: dict[str, list[int]] = {}
    for dataset, part in condition_frame.groupby("dataset", sort=True):
        condition_index_by_dataset[str(dataset)] = {
            str(condition): idx for idx, condition in enumerate(part["condition"].astype(str).tolist())
        }
        coverage_by_dataset[str(dataset)] = part["covered"].astype(int).tolist()
    return split_rows, coverage_by_dataset, {k: [f"{d}::{c}" for d, c in v] for k, v in split_to_keys.items()}


def join_outcomes(split_rows: list[dict[str, Any]]) -> pd.DataFrame:
    split_frame = pd.DataFrame(split_rows)
    outcomes = pd.read_csv(OUTCOME_CSV)
    return outcomes.merge(split_frame, on="split_name", how="left", validate="many_to_one")


def corr_for(frame: pd.DataFrame, predictor: str, outcome: str) -> float:
    part = frame[[predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    if part.shape[0] < 3:
        return float("nan")
    return spearman(part[predictor].to_numpy(dtype=float), part[outcome].to_numpy(dtype=float))


def clustered_ci(frame: pd.DataFrame, predictor: str, outcome: str) -> tuple[float | None, float | None]:
    part = frame[["split_name", predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
    clusters = sorted(part["split_name"].astype(str).unique())
    if len(clusters) < 4:
        return None, None
    grouped = {cluster: part[part["split_name"].astype(str) == cluster] for cluster in clusters}
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(BOOT_REPEATS):
        picked = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([grouped[c] for c in picked], ignore_index=True)
        val = corr_for(boot, predictor, outcome)
        if np.isfinite(val):
            vals.append(val)
    if len(vals) < 10:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def association_rows(joined: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for predictor in PREDICTORS:
        for outcome in OUTCOMES:
            part = joined[["split_name", predictor, outcome]].replace([np.inf, -np.inf], np.nan).dropna()
            ci_low, ci_high = clustered_ci(part, predictor, outcome)
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_rows": int(part.shape[0]),
                    "n_clusters": int(part["split_name"].astype(str).nunique()),
                    "rho": corr_for(part, predictor, outcome),
                    "cluster_ci95_low": ci_low,
                    "cluster_ci95_high": ci_high,
                    "cluster_ci_excludes_zero": bool(ci_low is not None and ci_high is not None and (ci_low > 0 or ci_high < 0)),
                }
            )
    return rows


def dataset_permutation_rows(
    joined: pd.DataFrame,
    coverage_by_dataset: dict[str, list[int]],
    split_to_keys: dict[str, list[str]],
) -> list[dict[str, Any]]:
    # Precompute per-split dataset/condition references.
    split_refs: dict[str, list[tuple[str, str]]] = {}
    for split, keys in split_to_keys.items():
        refs = []
        for key in keys:
            dataset, condition = key.split("::", 1)
            refs.append((dataset, condition))
        split_refs[split] = refs

    # Condition order must match build_condition_table grouping order.
    cond_meta = load_json(COND_META)
    index_by_dataset = {
        dataset: {condition: idx for idx, condition in enumerate(conditions.keys())}
        for dataset, conditions in cond_meta.items()
    }
    observed = {
        outcome: corr_for(joined, "exact_condition_fraction", outcome)
        for outcome in OUTCOMES
    }
    rng = np.random.default_rng(SEED + 1)
    nulls = {outcome: [] for outcome in OUTCOMES}
    split_names = joined["split_name"].astype(str).unique().tolist()
    for _ in range(PERM_REPEATS):
        permuted = {
            dataset: rng.permutation(np.asarray(values, dtype=float))
            for dataset, values in coverage_by_dataset.items()
        }
        perm_fraction = {}
        for split_name in split_names:
            vals = []
            for dataset, condition in split_refs.get(split_name, []):
                idx = index_by_dataset.get(dataset, {}).get(condition)
                if idx is None or dataset not in permuted:
                    continue
                vals.append(float(permuted[dataset][idx]))
            perm_fraction[split_name] = float(np.mean(vals)) if vals else float("nan")
        temp = joined.copy()
        temp["perm_exact_condition_fraction"] = temp["split_name"].astype(str).map(perm_fraction)
        for outcome in OUTCOMES:
            val = corr_for(temp, "perm_exact_condition_fraction", outcome)
            if np.isfinite(val):
                nulls[outcome].append(val)
    rows = []
    for outcome in OUTCOMES:
        arr = np.asarray(nulls[outcome], dtype=float)
        obs = observed[outcome]
        if arr.size:
            p = (1 + float(np.sum(np.abs(arr) >= abs(obs)))) / (1 + arr.size)
            rows.append(
                {
                    "outcome": outcome,
                    "observed_rho": obs,
                    "null_mean": float(arr.mean()),
                    "null_ci95_low": float(np.percentile(arr, 2.5)),
                    "null_ci95_high": float(np.percentile(arr, 97.5)),
                    "dataset_stratified_perm_p": p,
                    "perm_repeats": int(arr.size),
                }
            )
    return rows


def dataset_rate_rows(condition_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for dataset, part in condition_frame.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": dataset,
                "conditions": int(part.shape[0]),
                "covered_conditions": int(part["covered"].sum()),
                "coverage_rate": float(part["covered"].mean()),
                "mean_p_hat": float(part["p_hat"].mean()),
                "perturbation_types": ",".join(sorted(part["perturbation_type"].astype(str).unique())),
            }
        )
    return rows


def decide(assoc: list[dict[str, Any]], perm: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    reasons = []
    residual_tail = next(row for row in assoc if row["predictor"] == "residual_coverage_fraction" and row["outcome"] == "tail_score")
    residual_mmd = next(row for row in assoc if row["predictor"] == "residual_coverage_fraction" and row["outcome"] == "family_mmd_delta")
    for row, name in [(residual_tail, "tail_score"), (residual_mmd, "family_mmd_delta")]:
        if not row["cluster_ci_excludes_zero"]:
            reasons.append(f"residual_{name}_cluster_ci_crosses_zero")
    for outcome in ["tail_score", "family_mmd_delta"]:
        prow = next((row for row in perm if row["outcome"] == outcome), None)
        if not prow or float(prow["dataset_stratified_perm_p"]) >= 0.05:
            reasons.append(f"{outcome}_dataset_stratified_permutation_not_significant")
    if reasons:
        return (
            "exact_response_information_ipw_missingness_partial_no_gpu",
            reasons,
            "treat exact coverage as confounded CPU scaling evidence; create strict matched splits or stronger IPW controls before GPU",
        )
    return (
        "exact_response_information_ipw_missingness_pass_no_gpu",
        [],
        "promote residual exact coverage as a robust CPU scaling-law candidate; still no GPU until a leakage-safe matched launcher/no-harm gate exists",
    )


def main() -> None:
    global COVERAGE_CSV, OUT_DIR, OUT_MD, OUT_JSON, OUT_SPLIT, OUT_JOIN, OUT_ASSOC, OUT_PERM, OUT_DATASET

    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-csv", type=Path, default=COVERAGE_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    COVERAGE_CSV = args.coverage_csv
    OUT_DIR = args.out_dir
    OUT_MD = OUT_DIR / "LATENTFM_EXACT_RESPONSE_INFORMATION_IPW_MISSINGNESS_20260628.md"
    OUT_JSON = OUT_DIR / "latentfm_exact_response_information_ipw_missingness_20260628.json"
    OUT_SPLIT = OUT_DIR / "ipw_missingness_split_rows.csv"
    OUT_JOIN = OUT_DIR / "ipw_missingness_outcome_join_rows.csv"
    OUT_ASSOC = OUT_DIR / "ipw_missingness_association_rows.csv"
    OUT_PERM = OUT_DIR / "dataset_stratified_permutation_rows.csv"
    OUT_DATASET = OUT_DIR / "coverage_dataset_rates.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    condition_frame = build_condition_table()
    split_rows, coverage_by_dataset, split_to_keys = build_split_rows(condition_frame)
    joined = join_outcomes(split_rows)
    assoc = association_rows(joined)
    perm = dataset_permutation_rows(joined, coverage_by_dataset, split_to_keys)
    dataset_rows = dataset_rate_rows(condition_frame)
    status, reasons, next_action = decide(assoc, perm)

    split_fields = [
        "split_file",
        "split_name",
        "n_train_conditions",
        "n_train_conditions_with_meta",
        "missing_condition_metadata",
        "exact_condition_fraction",
        "expected_coverage_fraction",
        "residual_coverage_fraction",
        "standardized_residual_coverage",
        "ipw_coverage_fraction",
    ]
    join_fields = list(joined.columns)
    assoc_fields = ["predictor", "outcome", "n_rows", "n_clusters", "rho", "cluster_ci95_low", "cluster_ci95_high", "cluster_ci_excludes_zero"]
    perm_fields = ["outcome", "observed_rho", "null_mean", "null_ci95_low", "null_ci95_high", "dataset_stratified_perm_p", "perm_repeats"]
    dataset_fields = ["dataset", "conditions", "covered_conditions", "coverage_rate", "mean_p_hat", "perturbation_types"]
    write_csv(OUT_SPLIT, split_rows, split_fields)
    write_csv(OUT_JOIN, joined.to_dict("records"), join_fields)
    write_csv(OUT_ASSOC, assoc, assoc_fields)
    write_csv(OUT_PERM, perm, perm_fields)
    write_csv(OUT_DATASET, dataset_rows, dataset_fields)

    payload = {
        "created_at": now_cst(),
        "status": status,
        "reasons": reasons,
        "next_action": next_action,
        "condition_universe_rows": int(condition_frame.shape[0]),
        "covered_conditions": int(condition_frame["covered"].sum()),
        "split_rows": len(split_rows),
        "joined_rows": int(joined.shape[0]),
        "association_csv": str(OUT_ASSOC),
        "permutation_csv": str(OUT_PERM),
        "split_csv": str(OUT_SPLIT),
        "dataset_csv": str(OUT_DATASET),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    top_assoc = sorted(assoc, key=lambda r: abs(float(r["rho"])) if np.isfinite(float(r["rho"])) else -1, reverse=True)[:12]
    lines = [
        "# LatentFM Exact Response-Information IPW Missingness Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only condition-level missingness/IPW gate.",
        "* Coverage probability is estimated from dataset, perturbation type, and target-count class; residual/IPW split metrics are then joined to frozen outcomes.",
        "* Dataset-stratified permutation shuffles coverage indicators within each dataset, preserving dataset-level raw-expression availability.",
        "* No train/infer/GPU/canonical multi/Track C query/checkpoint selection.",
        "",
        "## Summary",
        "",
        f"* Condition universe rows: `{payload['condition_universe_rows']}`; covered: `{payload['covered_conditions']}`.",
        f"* Split rows: `{payload['split_rows']}`; joined outcome rows: `{payload['joined_rows']}`.",
        "",
        "## Top Associations",
        "",
        "| predictor | outcome | rows | rho | CI low | CI high | excludes 0 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in top_assoc:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["predictor"]),
                    str(row["outcome"]),
                    str(row["n_rows"]),
                    fmt_float(row["rho"]),
                    fmt_float(row["cluster_ci95_low"]),
                    fmt_float(row["cluster_ci95_high"]),
                    str(row["cluster_ci_excludes_zero"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Dataset-Stratified Permutation", "", "| outcome | observed rho | null CI | p |", "|---|---:|---|---:|"])
    for row in perm:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["outcome"]),
                    fmt_float(row["observed_rho"]),
                    f"[{fmt_float(row['null_ci95_low'])}, {fmt_float(row['null_ci95_high'])}]",
                    fmt_float(row["dataset_stratified_perm_p"]),
                ]
            )
            + " |"
        )
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
            f"* Split rows: `{OUT_SPLIT}`",
            f"* Outcome join rows: `{OUT_JOIN}`",
            f"* Association rows: `{OUT_ASSOC}`",
            f"* Dataset-stratified permutation rows: `{OUT_PERM}`",
            f"* Dataset coverage rates: `{OUT_DATASET}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
