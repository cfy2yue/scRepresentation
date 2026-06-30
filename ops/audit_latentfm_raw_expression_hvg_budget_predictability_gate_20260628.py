#!/usr/bin/env python3
"""CPU-only raw-expression HVG budget predictability gate.

This gate tests whether downstream perturbation-response signal is concentrated
in small gene budgets strongly enough to motivate a formal information-scaling
axis. It does not train, infer, select checkpoints, or authorize GPU use.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/raw_expression_hvg_budget_predictability_gate_20260628"
OUT_MD = OUT_DIR / "LATENTFM_RAW_EXPRESSION_HVG_BUDGET_PREDICTABILITY_GATE_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_raw_expression_hvg_budget_predictability_gate_20260628.json"
OUT_CONDITION_CSV = OUT_DIR / "condition_budget_rows.csv"
OUT_SUMMARY_CSV = OUT_DIR / "budget_summary_rows.csv"

BUDGETS = (500, 1000, 2000, 4000)
RANDOM_REPEATS = 25
MIN_CONTROL_CELLS = 50
MIN_PERT_CELLS = 25
MAX_CONDITIONS_PER_FILE = 20
BOOT_REPEATS = 500
SEED = 42


@dataclass(frozen=True)
class DatasetSpec:
    group: str
    path: Path
    perturbation_col: str = "perturbation"
    control_col: str | None = None


DATASETS = [
    DatasetSpec("genepert_DE5000_small", ROOT / "dataset/raw/genepert_DE5000/TianActivation.h5ad"),
    DatasetSpec("genepert_DE5000_small", ROOT / "dataset/raw/genepert_DE5000/GasperiniShendure2019_lowMOI.h5ad"),
    DatasetSpec("genepert_DE5000_small", ROOT / "dataset/raw/genepert_DE5000/Papalexi.h5ad"),
    DatasetSpec("genepert_DE5000_small", ROOT / "dataset/raw/genepert_DE5000/TianInhibition.h5ad"),
    DatasetSpec("chemicalpert_bench", ROOT / "dataset/raw/chemicalpert_bench/sciplex3_A549.h5ad", control_col="control"),
    DatasetSpec("chemicalpert_bench", ROOT / "dataset/raw/chemicalpert_bench/sciplex3_K562.h5ad", control_col="control"),
    DatasetSpec("chemicalpert_bench", ROOT / "dataset/raw/chemicalpert_bench/sciplex3_MCF7.h5ad", control_col="control"),
]


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
    if value is None:
        return "NA"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def dense_row_mean(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    return np.asarray(mean).ravel().astype(np.float64)


def dense_row_var(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    mean = dense_row_mean(matrix)
    if sparse.issparse(matrix):
        mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel().astype(np.float64)
    else:
        arr = np.asarray(matrix)
        mean_sq = np.mean(arr * arr, axis=0).astype(np.float64)
    return np.maximum(mean_sq - mean * mean, 0.0)


def get_matrix(adata: ad.AnnData) -> tuple[sparse.spmatrix | np.ndarray, str, str]:
    if "logNor" in adata.layers:
        matrix = adata.layers["logNor"]
        source = "layer:logNor"
        policy = "use_existing_log_normalized_matrix_no_second_log1p"
    elif adata.X is not None:
        matrix = adata.X
        source = "X"
        policy = "use_existing_X_no_second_log1p_counts_layer_absent"
    else:
        raise ValueError("no expression matrix found")
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
    else:
        matrix = np.asarray(matrix)
    return matrix, source, policy


def control_mask(obs: pd.DataFrame, spec: DatasetSpec) -> np.ndarray:
    if spec.control_col and spec.control_col in obs.columns:
        vals = obs[spec.control_col].astype(str).str.lower()
        return vals.isin({"1", "true", "yes", "control", "ctrl"}).to_numpy()
    if spec.perturbation_col in obs.columns:
        vals = obs[spec.perturbation_col].astype(str).str.lower()
        return vals.isin({"control", "ctrl", "vehicle", "dmso", "non-targeting", "non_targeting"}).to_numpy()
    if "gene" in obs.columns:
        vals = obs["gene"].astype(str).str.lower()
        return vals.isin({"ctrl", "control"}).to_numpy()
    return np.zeros(obs.shape[0], dtype=bool)


def choose_conditions(obs: pd.DataFrame, mask_control: np.ndarray, spec: DatasetSpec) -> list[str]:
    if spec.perturbation_col not in obs.columns:
        return []
    counts = obs.loc[~mask_control, spec.perturbation_col].astype(str).value_counts()
    counts = counts[counts >= MIN_PERT_CELLS]
    if MAX_CONDITIONS_PER_FILE and MAX_CONDITIONS_PER_FILE > 0:
        counts = counts.head(MAX_CONDITIONS_PER_FILE)
    return sorted(counts.index.tolist())


def bootstrap_ci(values: list[float], rng: np.random.Generator) -> tuple[float | None, float | None]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if arr.size < 3:
        return None, None
    samples = rng.choice(arr, size=(BOOT_REPEATS, arr.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def budget_rows_for_dataset(spec: DatasetSpec, rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not spec.path.is_file():
        return [], {
            "dataset": spec.path.stem,
            "group": spec.group,
            "status": "missing_file",
            "path": str(spec.path),
        }

    adata = ad.read_h5ad(spec.path)
    try:
        matrix, matrix_source, log1p_policy = get_matrix(adata)
        obs = adata.obs.copy()
        mask_control = control_mask(obs, spec)
        n_control = int(mask_control.sum())
        conditions = choose_conditions(obs, mask_control, spec)
        meta: dict[str, Any] = {
            "dataset": spec.path.stem,
            "group": spec.group,
            "status": "ok",
            "path": str(spec.path),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "matrix_source": matrix_source,
            "log1p_policy": log1p_policy,
            "n_control": n_control,
            "conditions_selected": len(conditions),
            "conditions_available_min_cells": len(
                obs.loc[~mask_control, spec.perturbation_col].astype(str).value_counts()[
                    lambda s: s >= MIN_PERT_CELLS
                ]
            )
            if spec.perturbation_col in obs.columns
            else 0,
        }
        if n_control < MIN_CONTROL_CELLS or not conditions:
            meta["status"] = "insufficient_control_or_conditions"
            return [], meta

        control_matrix = matrix[mask_control]
        control_mean = dense_row_mean(control_matrix)
        control_var = dense_row_var(control_matrix)
        hvg_order = np.argsort(-control_var, kind="mergesort")
        random_indices = {
            k: [rng.choice(adata.n_vars, size=min(k, adata.n_vars), replace=False) for _ in range(RANDOM_REPEATS)]
            for k in BUDGETS
        }

        rows: list[dict[str, Any]] = []
        perturb_values = obs[spec.perturbation_col].astype(str).to_numpy()
        for condition in conditions:
            mask_condition = perturb_values == condition
            pert_matrix = matrix[mask_condition]
            n_pert = int(mask_condition.sum())
            pert_mean = dense_row_mean(pert_matrix)
            response_sq = np.square(pert_mean - control_mean)
            total_energy = float(response_sq.sum())
            if not np.isfinite(total_energy) or total_energy <= 0:
                continue
            response_order = np.argsort(-response_sq, kind="mergesort")
            for budget in BUDGETS:
                k_eff = min(int(budget), int(adata.n_vars))
                hvg_idx = hvg_order[:k_eff]
                oracle_idx = response_order[:k_eff]
                random_shares = [
                    float(response_sq[idx].sum() / total_energy)
                    for idx in random_indices[budget]
                ]
                hvg_share = float(response_sq[hvg_idx].sum() / total_energy)
                oracle_share = float(response_sq[oracle_idx].sum() / total_energy)
                random_mean = float(np.mean(random_shares))
                random_p95 = float(np.percentile(random_shares, 95))
                rows.append(
                    {
                        "dataset": spec.path.stem,
                        "group": spec.group,
                        "condition": condition,
                        "n_vars": int(adata.n_vars),
                        "budget": budget,
                        "effective_budget": k_eff,
                        "budget_clipped": int(k_eff < budget),
                        "n_control": n_control,
                        "n_pert": n_pert,
                        "matrix_source": matrix_source,
                        "log1p_policy": log1p_policy,
                        "response_energy": total_energy,
                        "control_hvg_share": hvg_share,
                        "random_share_mean": random_mean,
                        "random_share_p95": random_p95,
                        "oracle_response_share": oracle_share,
                        "hvg_minus_random_mean": hvg_share - random_mean,
                        "hvg_minus_random_p95": hvg_share - random_p95,
                        "hvg_over_random_mean": hvg_share / random_mean if random_mean > 0 else np.nan,
                    }
                )
        return rows, meta
    finally:
        adata.file.close() if getattr(adata, "isbacked", False) else None


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 1)
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    group_cols = ["group", "dataset", "budget"]
    for keys, part in frame.groupby(group_cols, sort=True):
        group, dataset, budget = keys
        values = part["control_hvg_share"].astype(float).tolist()
        advantage = part["hvg_minus_random_mean"].astype(float).tolist()
        ci_low, ci_high = bootstrap_ci(values, rng)
        adv_low, adv_high = bootstrap_ci(advantage, rng)
        summaries.append(
            {
                "group": group,
                "dataset": dataset,
                "budget": int(budget),
                "condition_rows": int(part.shape[0]),
                "n_vars": int(part["n_vars"].iloc[0]),
                "budget_clipped_fraction": float(part["budget_clipped"].mean()),
                "control_hvg_share_mean": float(part["control_hvg_share"].mean()),
                "control_hvg_share_median": float(part["control_hvg_share"].median()),
                "control_hvg_share_ci95_low": ci_low,
                "control_hvg_share_ci95_high": ci_high,
                "random_share_mean": float(part["random_share_mean"].mean()),
                "oracle_response_share_mean": float(part["oracle_response_share"].mean()),
                "hvg_minus_random_mean": float(part["hvg_minus_random_mean"].mean()),
                "hvg_minus_random_ci95_low": adv_low,
                "hvg_minus_random_ci95_high": adv_high,
                "hvg_minus_random_p95_mean": float(part["hvg_minus_random_p95"].mean()),
            }
        )
    for keys, part in frame.groupby(["group", "budget"], sort=True):
        group, budget = keys
        values = part["control_hvg_share"].astype(float).tolist()
        advantage = part["hvg_minus_random_mean"].astype(float).tolist()
        ci_low, ci_high = bootstrap_ci(values, rng)
        adv_low, adv_high = bootstrap_ci(advantage, rng)
        summaries.append(
            {
                "group": group,
                "dataset": "__GROUP__",
                "budget": int(budget),
                "condition_rows": int(part.shape[0]),
                "n_vars": "mixed",
                "budget_clipped_fraction": float(part["budget_clipped"].mean()),
                "control_hvg_share_mean": float(part["control_hvg_share"].mean()),
                "control_hvg_share_median": float(part["control_hvg_share"].median()),
                "control_hvg_share_ci95_low": ci_low,
                "control_hvg_share_ci95_high": ci_high,
                "random_share_mean": float(part["random_share_mean"].mean()),
                "oracle_response_share_mean": float(part["oracle_response_share"].mean()),
                "hvg_minus_random_mean": float(part["hvg_minus_random_mean"].mean()),
                "hvg_minus_random_ci95_low": adv_low,
                "hvg_minus_random_ci95_high": adv_high,
                "hvg_minus_random_p95_mean": float(part["hvg_minus_random_p95"].mean()),
            }
        )
    for budget, part in frame.groupby("budget", sort=True):
        values = part["control_hvg_share"].astype(float).tolist()
        advantage = part["hvg_minus_random_mean"].astype(float).tolist()
        ci_low, ci_high = bootstrap_ci(values, rng)
        adv_low, adv_high = bootstrap_ci(advantage, rng)
        summaries.append(
            {
                "group": "__ALL__",
                "dataset": "__ALL__",
                "budget": int(budget),
                "condition_rows": int(part.shape[0]),
                "n_vars": "mixed",
                "budget_clipped_fraction": float(part["budget_clipped"].mean()),
                "control_hvg_share_mean": float(part["control_hvg_share"].mean()),
                "control_hvg_share_median": float(part["control_hvg_share"].median()),
                "control_hvg_share_ci95_low": ci_low,
                "control_hvg_share_ci95_high": ci_high,
                "random_share_mean": float(part["random_share_mean"].mean()),
                "oracle_response_share_mean": float(part["oracle_response_share"].mean()),
                "hvg_minus_random_mean": float(part["hvg_minus_random_mean"].mean()),
                "hvg_minus_random_ci95_low": adv_low,
                "hvg_minus_random_ci95_high": adv_high,
                "hvg_minus_random_p95_mean": float(part["hvg_minus_random_p95"].mean()),
            }
        )
    return summaries


def decide_status(metas: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    ok_datasets = [m for m in metas if m.get("status") == "ok" and int(m.get("conditions_selected", 0)) > 0]
    all_1000 = next((s for s in summaries if s["group"] == "__ALL__" and int(s["budget"]) == 1000), None)
    chemical_1000 = next((s for s in summaries if s["group"] == "chemicalpert_bench" and s["dataset"] == "__GROUP__" and int(s["budget"]) == 1000), None)
    gene_500 = next((s for s in summaries if s["group"] == "genepert_DE5000_small" and s["dataset"] == "__GROUP__" and int(s["budget"]) == 500), None)
    reasons: list[str] = []
    if len(ok_datasets) < 4:
        reasons.append("fewer_than_four_usable_raw_expression_datasets")
    if not all_1000 or int(all_1000["condition_rows"]) < 60:
        reasons.append("too_few_condition_rows_for_global_budget_signal")
    if all_1000 and float(all_1000["hvg_minus_random_mean"]) <= 0.05:
        reasons.append("global_control_hvg_top1000_not_clearly_above_random")
    if chemical_1000 and float(chemical_1000["hvg_minus_random_mean"]) <= 0.05:
        reasons.append("chemical_control_hvg_top1000_not_clearly_above_random")
    if gene_500 and float(gene_500["hvg_minus_random_mean"]) <= 0.02:
        reasons.append("genepert_control_hvg_top500_not_clearly_above_random")
    if reasons:
        status = "raw_expression_hvg_budget_predictability_partial_no_gpu"
        next_action = (
            "treat HVG budget as a plausible but not yet formal scaling axis; "
            "expand CPU matrix coverage and add train-split-only HVG ranking before any GPU"
        )
    else:
        status = "raw_expression_hvg_budget_predictability_pass_no_gpu"
        next_action = (
            "promote HVG/response-information budget to a formal scaling-law CPU design matrix; "
            "still no immediate GPU without a leakage-safe train-only launcher and no-harm gate"
        )
    return status, reasons, next_action


def main() -> None:
    global OUT_DIR, OUT_MD, OUT_JSON, OUT_CONDITION_CSV, OUT_SUMMARY_CSV
    global RANDOM_REPEATS, MAX_CONDITIONS_PER_FILE, BOOT_REPEATS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for the report artifacts.",
    )
    parser.add_argument(
        "--max-conditions-per-file",
        type=int,
        default=MAX_CONDITIONS_PER_FILE,
        help="Cap selected perturbation conditions per h5ad file; <=0 uses all eligible conditions.",
    )
    parser.add_argument(
        "--random-repeats",
        type=int,
        default=RANDOM_REPEATS,
        help="Number of random gene-set repeats per condition/budget.",
    )
    parser.add_argument(
        "--boot-repeats",
        type=int,
        default=BOOT_REPEATS,
        help="Bootstrap repeats for report confidence intervals.",
    )
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    OUT_MD = OUT_DIR / "LATENTFM_RAW_EXPRESSION_HVG_BUDGET_PREDICTABILITY_GATE_20260628.md"
    OUT_JSON = OUT_DIR / "latentfm_raw_expression_hvg_budget_predictability_gate_20260628.json"
    OUT_CONDITION_CSV = OUT_DIR / "condition_budget_rows.csv"
    OUT_SUMMARY_CSV = OUT_DIR / "budget_summary_rows.csv"
    RANDOM_REPEATS = int(args.random_repeats)
    MAX_CONDITIONS_PER_FILE = int(args.max_conditions_per_file)
    BOOT_REPEATS = int(args.boot_repeats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    condition_rows: list[dict[str, Any]] = []
    dataset_meta: list[dict[str, Any]] = []
    for spec in DATASETS:
        rows, meta = budget_rows_for_dataset(spec, rng)
        condition_rows.extend(rows)
        dataset_meta.append(meta)

    summary_rows = summarize_rows(condition_rows)
    status, reasons, next_action = decide_status(dataset_meta, summary_rows)

    condition_fields = [
        "group",
        "dataset",
        "condition",
        "n_vars",
        "budget",
        "effective_budget",
        "budget_clipped",
        "n_control",
        "n_pert",
        "matrix_source",
        "log1p_policy",
        "response_energy",
        "control_hvg_share",
        "random_share_mean",
        "random_share_p95",
        "oracle_response_share",
        "hvg_minus_random_mean",
        "hvg_minus_random_p95",
        "hvg_over_random_mean",
    ]
    summary_fields = [
        "group",
        "dataset",
        "budget",
        "condition_rows",
        "n_vars",
        "budget_clipped_fraction",
        "control_hvg_share_mean",
        "control_hvg_share_median",
        "control_hvg_share_ci95_low",
        "control_hvg_share_ci95_high",
        "random_share_mean",
        "oracle_response_share_mean",
        "hvg_minus_random_mean",
        "hvg_minus_random_ci95_low",
        "hvg_minus_random_ci95_high",
        "hvg_minus_random_p95_mean",
    ]
    write_csv(OUT_CONDITION_CSV, condition_rows, condition_fields)
    write_csv(OUT_SUMMARY_CSV, summary_rows, summary_fields)

    payload = {
        "created_at": now_cst(),
        "status": status,
        "reasons": reasons,
        "next_action": next_action,
        "dataset_meta": dataset_meta,
        "summary_rows": summary_rows,
        "condition_csv": str(OUT_CONDITION_CSV),
        "summary_csv": str(OUT_SUMMARY_CSV),
        "limits": {
            "budgets": BUDGETS,
            "random_repeats": RANDOM_REPEATS,
            "min_control_cells": MIN_CONTROL_CELLS,
            "min_pert_cells": MIN_PERT_CELLS,
            "max_conditions_per_file": MAX_CONDITIONS_PER_FILE,
            "seed": SEED,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    all_rows = {int(s["budget"]): s for s in summary_rows if s["group"] == "__ALL__" and s["dataset"] == "__ALL__"}
    chemical_rows = {
        int(s["budget"]): s
        for s in summary_rows
        if s["group"] == "chemicalpert_bench" and s["dataset"] == "__GROUP__"
    }
    gene_rows = {
        int(s["budget"]): s
        for s in summary_rows
        if s["group"] == "genepert_DE5000_small" and s["dataset"] == "__GROUP__"
    }

    def bullet_summary(rows: dict[int, dict[str, Any]], label: str) -> list[str]:
        out = [f"### {label}", ""]
        out.append("| budget | rows | clipped | control-HVG share | random share | HVG-random | oracle upper bound |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|")
        for budget in BUDGETS:
            row = rows.get(budget)
            if not row:
                continue
            out.append(
                "| "
                + " | ".join(
                    [
                        str(budget),
                        str(row["condition_rows"]),
                        fmt_float(row["budget_clipped_fraction"]),
                        fmt_float(row["control_hvg_share_mean"]),
                        fmt_float(row["random_share_mean"]),
                        fmt_float(row["hvg_minus_random_mean"]),
                        fmt_float(row["oracle_response_share_mean"]),
                    ]
                )
                + " |"
            )
        out.append("")
        return out

    if MAX_CONDITIONS_PER_FILE and MAX_CONDITIONS_PER_FILE > 0:
        condition_cap_line = (
            f"* Conditions were capped at {MAX_CONDITIONS_PER_FILE} per file "
            "with deterministic top-count selection."
        )
    else:
        condition_cap_line = "* All eligible conditions per file were selected deterministically."

    lines = [
        "# LatentFM Raw-Expression HVG Budget Predictability Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Purpose",
        "",
        "Test whether perturbation-response energy in local raw-expression panels is concentrated in control-HVG gene budgets strongly enough to support an information-scaling x-axis. This is CPU-only and does not authorize GPU training, checkpoint selection, canonical multi selection, or Track C query use.",
        "",
        "## Boundary",
        "",
        "* Used selected real expression matrices from `raw/genepert_DE5000` and `raw/chemicalpert_bench`; skipped `raw/genepert_bench` because its small h5ad files expose null `X` matrices despite obs/var metadata.",
        "* For chemical bench, used existing `layer:logNor`; for genepert DE5000, used existing `X`. No second `log1p` was applied.",
        "* Control-HVG ranking uses control cells only. The oracle response ranking is reported only as an upper bound and is not a legal training-time feature selector.",
        condition_cap_line,
        "",
        "## Dataset Meta",
        "",
        "| group | dataset | status | cells | genes | control cells | selected conditions | matrix | log1p policy |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for meta in dataset_meta:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(meta.get("group", "")),
                    str(meta.get("dataset", "")),
                    str(meta.get("status", "")),
                    str(meta.get("n_obs", "")),
                    str(meta.get("n_vars", "")),
                    str(meta.get("n_control", "")),
                    str(meta.get("conditions_selected", "")),
                    str(meta.get("matrix_source", "")),
                    str(meta.get("log1p_policy", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.extend(bullet_summary(all_rows, "All Usable Conditions"))
    lines.extend(bullet_summary(gene_rows, "Gene Perturbation DE5000 Small Files"))
    lines.extend(bullet_summary(chemical_rows, "Chemical Perturbation Bench"))
    lines.extend(
        [
            "## Decision",
            "",
            f"* Status: `{status}`.",
            f"* Reasons: `{', '.join(reasons) if reasons else 'none'}`.",
            f"* Next action: {next_action}.",
            "",
            "## Outputs",
            "",
            f"* Condition rows: `{OUT_CONDITION_CSV}`",
            f"* Summary rows: `{OUT_SUMMARY_CSV}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
