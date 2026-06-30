#!/usr/bin/env python3
"""Compare control-HVG and control-detection gene budgets for response energy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path("/data/cyx/1030/scLatent")
sys.path.append(str(ROOT / "ops"))
import audit_latentfm_hvg_meanmatched_negative_controls_20260628 as neg  # noqa: E402

INPUT_CONDITION_CSV = ROOT / "reports/raw_expression_hvg_budget_expanded_gate_20260629/condition_budget_rows.csv"
OUT_DIR = ROOT / "reports/hvg_vs_detection_expanded_baseline_20260629"
BUDGETS = (500, 1000)
SEED = 45
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


def detection_fraction(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray((matrix > 0).mean(axis=0)).ravel().astype(np.float64)
    arr = np.asarray(matrix)
    return np.mean(arr > 0, axis=0).astype(np.float64)


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
        control_detect = detection_fraction(control_matrix)
        hvg_order = np.argsort(-control_var, kind="mergesort")
        detection_order = np.argsort(-control_detect, kind="mergesort")
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
                hvg_idx = hvg_order[:k]
                detection_idx = detection_order[:k]
                hvg_share = neg.response_share(response_sq, hvg_idx, total)
                detection_share = neg.response_share(response_sq, detection_idx, total)
                overlap = len(set(map(int, hvg_idx)) & set(map(int, detection_idx)))
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
                        "control_detection_share": detection_share,
                        "hvg_minus_detection": hvg_share - detection_share,
                        "hvg_detection_overlap_fraction": overlap / max(k, 1),
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
            diff = part["hvg_minus_detection"].astype(float).tolist()
            ci_low, ci_high = bootstrap_ci(diff, rng)
            out.append(
                {
                    "level": level,
                    "group": key_map.get("group", "__ALL__"),
                    "dataset": key_map.get("dataset", "__ALL__"),
                    "budget": int(key_map["budget"]),
                    "condition_rows": int(part.shape[0]),
                    "control_hvg_share_mean": float(part["control_hvg_share"].mean()),
                    "control_detection_share_mean": float(part["control_detection_share"].mean()),
                    "hvg_minus_detection_mean": float(part["hvg_minus_detection"].mean()),
                    "hvg_minus_detection_ci95_low": ci_low,
                    "hvg_minus_detection_ci95_high": ci_high,
                    "hvg_detection_overlap_fraction_mean": float(part["hvg_detection_overlap_fraction"].mean()),
                }
            )
    return out


def decide(summary_rows: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    reasons: list[str] = []
    groups = [r for r in summary_rows if r["level"] == "group" and int(r["budget"]) == 1000]
    for row in groups:
        if abs(float(row["hvg_minus_detection_mean"])) < 0.03:
            reasons.append(f"{row['group']}_hvg_and_detection_nearly_equivalent_top1000")
    if reasons:
        return (
            "hvg_vs_detection_baseline_detection_equivalent_no_gpu",
            reasons,
            "reframe as observable/detected-gene budget rather than HVG-specific superiority",
        )
    return (
        "hvg_vs_detection_baseline_hvg_specific_no_gpu",
        [],
        "keep HVG-specific coverage as a candidate pending matched controls",
    )


def main() -> int:
    global BOOT_REPEATS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-condition-csv", type=Path, default=INPUT_CONDITION_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--boot-repeats", type=int, default=BOOT_REPEATS)
    args = parser.parse_args()

    BOOT_REPEATS = int(args.boot_repeats)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "LATENTFM_HVG_VS_DETECTION_BASELINE_20260629.md"
    out_json = out_dir / "latentfm_hvg_vs_detection_baseline_20260629.json"
    out_condition = out_dir / "condition_hvg_vs_detection_rows.csv"
    out_summary = out_dir / "hvg_vs_detection_summary_rows.csv"

    selected = (
        pd.read_csv(args.input_condition_csv)
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
        "control_detection_share",
        "hvg_minus_detection",
        "hvg_detection_overlap_fraction",
    ]
    summary_fields = [
        "level",
        "group",
        "dataset",
        "budget",
        "condition_rows",
        "control_hvg_share_mean",
        "control_detection_share_mean",
        "hvg_minus_detection_mean",
        "hvg_minus_detection_ci95_low",
        "hvg_minus_detection_ci95_high",
        "hvg_detection_overlap_fraction_mean",
    ]
    write_csv(out_condition, condition_rows, condition_fields)
    write_csv(out_summary, summary_rows, summary_fields)
    payload = {
        "created_at": now_cst(),
        "status": status,
        "reasons": reasons,
        "next_action": next_action,
        "condition_csv": str(out_condition),
        "summary_csv": str(out_summary),
        "summary_rows": summary_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM HVG Versus Detection Baseline",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU-only baseline over the expanded selected raw-expression conditions.",
        "* Compares control-variance HVG top-k genes with control-detection-rate top-k genes.",
        "* Does not train, infer, evaluate canonical multi, use Track C query, select checkpoints, or authorize GPU.",
        "",
        "## Summary",
        "",
        "| group | budget | rows | HVG share | detection share | HVG - detection | CI low | overlap |",
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
                    fmt_float(row["control_detection_share_mean"]),
                    fmt_float(row["hvg_minus_detection_mean"]),
                    fmt_float(row["hvg_minus_detection_ci95_low"]),
                    fmt_float(row["hvg_detection_overlap_fraction_mean"]),
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
            f"* Condition rows: `{out_condition}`",
            f"* Summary rows: `{out_summary}`",
            f"* JSON: `{out_json}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
