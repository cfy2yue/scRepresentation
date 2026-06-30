#!/usr/bin/env python3
"""Compare control-HVG and control-abundance gene budgets for response energy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
sys.path.append(str(ROOT / "ops"))
import audit_latentfm_hvg_meanmatched_negative_controls_20260628 as neg  # noqa: E402

INPUT_CONDITION_CSV = ROOT / "reports/raw_expression_hvg_budget_predictability_gate_20260628/condition_budget_rows.csv"
OUT_DIR = ROOT / "reports/hvg_vs_abundance_baseline_20260628"
OUT_MD = OUT_DIR / "LATENTFM_HVG_VS_ABUNDANCE_BASELINE_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_hvg_vs_abundance_baseline_20260628.json"
OUT_CONDITION_CSV = OUT_DIR / "condition_hvg_vs_abundance_rows.csv"
OUT_SUMMARY_CSV = OUT_DIR / "hvg_vs_abundance_summary_rows.csv"

BUDGETS = (500, 1000)
SEED = 44
BOOT_REPEATS = 500


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


def bootstrap_ci(values: list[float], rng: np.random.Generator) -> tuple[float | None, float | None]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size < 3:
        return None, None
    samples = rng.choice(arr, size=(BOOT_REPEATS, arr.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def condition_rows_for_dataset(group: str, dataset: str, conditions: list[str]) -> list[dict[str, Any]]:
    path = neg.raw_path(dataset, group)
    adata = ad.read_h5ad(path)
    try:
        matrix, matrix_source, log1p_policy = neg.get_matrix(adata, group)
        obs = adata.obs.copy()
        mask_control = neg.control_mask(obs, group)
        control_matrix = matrix[mask_control]
        control_mean = neg.dense_mean(control_matrix)
        control_var = neg.dense_var(control_matrix)
        hvg_order = np.argsort(-control_var, kind="mergesort")
        abundance_order = np.argsort(-control_mean, kind="mergesort")
        perturb_values = obs["perturbation"].astype(str).to_numpy()

        rows: list[dict[str, Any]] = []
        for condition in conditions:
            mask_condition = perturb_values == condition
            if int(mask_condition.sum()) <= 0:
                continue
            pert_mean = neg.dense_mean(matrix[mask_condition])
            response_sq = np.square(pert_mean - control_mean)
            total = float(response_sq.sum())
            if total <= 0 or not np.isfinite(total):
                continue
            for budget in BUDGETS:
                k = min(budget, adata.n_vars)
                hvg_share = neg.response_share(response_sq, hvg_order[:k], total)
                abundance_share = neg.response_share(response_sq, abundance_order[:k], total)
                overlap = len(set(map(int, hvg_order[:k])) & set(map(int, abundance_order[:k])))
                rows.append(
                    {
                        "group": group,
                        "dataset": dataset,
                        "condition": condition,
                        "budget": budget,
                        "effective_budget": k,
                        "n_vars": int(adata.n_vars),
                        "n_control": int(mask_control.sum()),
                        "n_pert": int(mask_condition.sum()),
                        "matrix_source": matrix_source,
                        "log1p_policy": log1p_policy,
                        "control_hvg_share": hvg_share,
                        "control_abundance_share": abundance_share,
                        "hvg_minus_abundance": hvg_share - abundance_share,
                        "hvg_abundance_overlap_fraction": overlap / max(k, 1),
                    }
                )
        return rows
    finally:
        del adata


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED + 1)
    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    groupings = [(["group", "dataset", "budget"], "dataset"), (["group", "budget"], "group"), (["budget"], "all")]
    for cols, level in groupings:
        for keys, part in frame.groupby(cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(cols, keys))
            diff = part["hvg_minus_abundance"].astype(float).tolist()
            ci_low, ci_high = bootstrap_ci(diff, rng)
            out.append(
                {
                    "level": level,
                    "group": key_map.get("group", "__ALL__"),
                    "dataset": key_map.get("dataset", "__ALL__"),
                    "budget": int(key_map["budget"]),
                    "condition_rows": int(part.shape[0]),
                    "control_hvg_share_mean": float(part["control_hvg_share"].mean()),
                    "control_abundance_share_mean": float(part["control_abundance_share"].mean()),
                    "hvg_minus_abundance_mean": float(part["hvg_minus_abundance"].mean()),
                    "hvg_minus_abundance_ci95_low": ci_low,
                    "hvg_minus_abundance_ci95_high": ci_high,
                    "hvg_abundance_overlap_fraction_mean": float(part["hvg_abundance_overlap_fraction"].mean()),
                }
            )
    return out


def decide(summary_rows: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    groups = [r for r in summary_rows if r["level"] == "group" and int(r["budget"]) == 1000]
    for row in groups:
        if abs(float(row["hvg_minus_abundance_mean"])) < 0.03:
            reasons.append(f"{row['group']}_hvg_and_abundance_nearly_equivalent_top1000")
    if reasons:
        return (
            "hvg_vs_abundance_baseline_abundance_equivalent_no_gpu",
            reasons,
            "redefine the scaling axis as expressed-gene/response-information budget, not HVG-specific superiority",
        )
    return (
        "hvg_vs_abundance_baseline_hvg_specific_no_gpu",
        [],
        "keep HVG-specific coverage as a candidate, pending split-level expansion",
    )


def main() -> None:
    global INPUT_CONDITION_CSV, OUT_DIR, OUT_MD, OUT_JSON, OUT_CONDITION_CSV, OUT_SUMMARY_CSV, BOOT_REPEATS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-condition-csv", type=Path, default=INPUT_CONDITION_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--boot-repeats", type=int, default=BOOT_REPEATS)
    args = parser.parse_args()

    INPUT_CONDITION_CSV = args.input_condition_csv
    OUT_DIR = args.out_dir
    OUT_MD = OUT_DIR / "LATENTFM_HVG_VS_ABUNDANCE_BASELINE_20260628.md"
    OUT_JSON = OUT_DIR / "latentfm_hvg_vs_abundance_baseline_20260628.json"
    OUT_CONDITION_CSV = OUT_DIR / "condition_hvg_vs_abundance_rows.csv"
    OUT_SUMMARY_CSV = OUT_DIR / "hvg_vs_abundance_summary_rows.csv"
    BOOT_REPEATS = int(args.boot_repeats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = (
        pd.read_csv(INPUT_CONDITION_CSV)
        .query("budget == 1000")[["group", "dataset", "condition"]]
        .drop_duplicates()
        .sort_values(["group", "dataset", "condition"])
    )
    condition_rows: list[dict[str, Any]] = []
    for (group, dataset), part in selected.groupby(["group", "dataset"], sort=True):
        condition_rows.extend(condition_rows_for_dataset(str(group), str(dataset), part["condition"].astype(str).tolist()))
    summary_rows = summarize(condition_rows)
    status, reasons, next_action = decide(summary_rows)

    condition_fields = [
        "group",
        "dataset",
        "condition",
        "budget",
        "effective_budget",
        "n_vars",
        "n_control",
        "n_pert",
        "matrix_source",
        "log1p_policy",
        "control_hvg_share",
        "control_abundance_share",
        "hvg_minus_abundance",
        "hvg_abundance_overlap_fraction",
    ]
    summary_fields = [
        "level",
        "group",
        "dataset",
        "budget",
        "condition_rows",
        "control_hvg_share_mean",
        "control_abundance_share_mean",
        "hvg_minus_abundance_mean",
        "hvg_minus_abundance_ci95_low",
        "hvg_minus_abundance_ci95_high",
        "hvg_abundance_overlap_fraction_mean",
    ]
    write_csv(OUT_CONDITION_CSV, condition_rows, condition_fields)
    write_csv(OUT_SUMMARY_CSV, summary_rows, summary_fields)

    payload = {
        "created_at": now_cst(),
        "status": status,
        "reasons": reasons,
        "next_action": next_action,
        "condition_csv": str(OUT_CONDITION_CSV),
        "summary_csv": str(OUT_SUMMARY_CSV),
        "summary_rows": summary_rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM HVG Versus Abundance Baseline",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU-only baseline over the same selected raw-expression conditions.",
        "* Compares control-variance HVG top-k genes with control-mean-abundance top-k genes.",
        "* Does not train, infer, evaluate canonical multi, use Track C query, select checkpoints, or authorize GPU.",
        "",
        "## Summary",
        "",
        "| group | budget | rows | HVG share | abundance share | HVG - abundance | CI low | overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["level"] != "group" or int(row["budget"]) not in BUDGETS:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    str(row["budget"]),
                    str(row["condition_rows"]),
                    fmt_float(row["control_hvg_share_mean"]),
                    fmt_float(row["control_abundance_share_mean"]),
                    fmt_float(row["hvg_minus_abundance_mean"]),
                    fmt_float(row["hvg_minus_abundance_ci95_low"]),
                    fmt_float(row["hvg_abundance_overlap_fraction_mean"]),
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
