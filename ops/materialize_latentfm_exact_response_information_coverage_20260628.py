#!/usr/bin/env python3
"""Expand exact raw-expression response-information coverage.

CPU-only. This materializes condition-level response-energy coverage for more
raw-expression datasets using control-HVG and control-abundance gene budgets.
It does not train, infer, evaluate held-out routes, or authorize GPU use.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
sys.path.append(str(ROOT / "ops"))
import audit_latentfm_hvg_meanmatched_negative_controls_20260628 as neg  # noqa: E402

BUDGETS = (250, 500, 1000, 2000)


@dataclass(frozen=True)
class RawDataset:
    group: str
    dataset: str
    path: Path
    size_mb: float


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


def discover_raw_datasets(max_file_mb: float, min_file_mb: float) -> list[RawDataset]:
    specs: list[RawDataset] = []
    for path in sorted((ROOT / "dataset/raw/genepert_DE5000").glob("*.h5ad")):
        size_mb = path.stat().st_size / (1024 * 1024)
        if min_file_mb <= size_mb <= max_file_mb:
            specs.append(RawDataset("genepert_DE5000", path.stem, path, size_mb))
    for path in sorted((ROOT / "dataset/raw/chemicalpert_bench").glob("*.h5ad")):
        size_mb = path.stat().st_size / (1024 * 1024)
        if min_file_mb <= size_mb <= max_file_mb:
            specs.append(RawDataset("chemicalpert_bench", path.stem, path, size_mb))
    return specs


def selected_conditions(obs: pd.DataFrame, mask_control: np.ndarray, max_conditions: int, min_pert_cells: int) -> list[str]:
    if "perturbation" not in obs.columns:
        return []
    counts = obs.loc[~mask_control, "perturbation"].astype(str).value_counts()
    counts = counts[counts >= min_pert_cells]
    return sorted(counts.head(max_conditions).index.tolist())


def first_budget_for(cumulative: np.ndarray, threshold: float) -> int:
    if cumulative.size == 0:
        return 0
    idx = np.searchsorted(cumulative, threshold, side="left")
    return int(min(idx + 1, cumulative.size))


def dataset_coverage_rows(
    spec: RawDataset,
    max_conditions: int,
    min_pert_cells: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    adata = ad.read_h5ad(spec.path)
    try:
        matrix, matrix_source, log1p_policy = neg.get_matrix(adata, "chemicalpert_bench" if spec.group == "chemicalpert_bench" else "genepert_DE5000_small")
        obs = adata.obs.copy()
        mask_control = neg.control_mask(obs, "chemicalpert_bench" if spec.group == "chemicalpert_bench" else "genepert_DE5000_small")
        n_control = int(mask_control.sum())
        conditions = selected_conditions(obs, mask_control, max_conditions, min_pert_cells)
        meta: dict[str, Any] = {
            "group": spec.group,
            "dataset": spec.dataset,
            "path": str(spec.path),
            "size_mb": spec.size_mb,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "n_control": n_control,
            "conditions_selected": len(conditions),
            "status": "ok",
            "matrix_source": matrix_source,
            "log1p_policy": log1p_policy,
        }
        if n_control <= 0 or not conditions:
            meta["status"] = "insufficient_control_or_conditions"
            return [], [], meta

        control_matrix = matrix[mask_control]
        control_mean = neg.dense_mean(control_matrix)
        control_var = neg.dense_var(control_matrix)
        hvg_order = np.argsort(-control_var, kind="mergesort")
        abundance_order = np.argsort(-control_mean, kind="mergesort")
        perturb_values = obs["perturbation"].astype(str).to_numpy()

        condition_rows: list[dict[str, Any]] = []
        budget_rows: list[dict[str, Any]] = []
        for condition in conditions:
            mask_condition = perturb_values == condition
            n_pert = int(mask_condition.sum())
            if n_pert < min_pert_cells:
                continue
            pert_mean = neg.dense_mean(matrix[mask_condition])
            response_sq = np.square(pert_mean - control_mean)
            total_energy = float(response_sq.sum())
            if total_energy <= 0 or not np.isfinite(total_energy):
                continue
            hvg_cum = np.cumsum(response_sq[hvg_order]) / total_energy
            abundance_cum = np.cumsum(response_sq[abundance_order]) / total_energy
            condition_rows.append(
                {
                    "group": spec.group,
                    "dataset": spec.dataset,
                    "condition": condition,
                    "n_vars": int(adata.n_vars),
                    "n_control": n_control,
                    "n_pert": n_pert,
                    "response_energy": total_energy,
                    "hvg_k80": first_budget_for(hvg_cum, 0.80),
                    "hvg_k90": first_budget_for(hvg_cum, 0.90),
                    "abundance_k80": first_budget_for(abundance_cum, 0.80),
                    "abundance_k90": first_budget_for(abundance_cum, 0.90),
                    "matrix_source": matrix_source,
                    "log1p_policy": log1p_policy,
                }
            )
            for budget in BUDGETS:
                k = min(budget, int(adata.n_vars))
                hvg_share = neg.response_share(response_sq, hvg_order[:k], total_energy)
                abundance_share = neg.response_share(response_sq, abundance_order[:k], total_energy)
                overlap = len(set(map(int, hvg_order[:k])) & set(map(int, abundance_order[:k]))) / max(k, 1)
                budget_rows.append(
                    {
                        "group": spec.group,
                        "dataset": spec.dataset,
                        "condition": condition,
                        "budget": budget,
                        "effective_budget": k,
                        "budget_clipped": int(k < budget),
                        "n_vars": int(adata.n_vars),
                        "n_control": n_control,
                        "n_pert": n_pert,
                        "response_energy": total_energy,
                        "hvg_share": hvg_share,
                        "abundance_share": abundance_share,
                        "hvg_minus_abundance": hvg_share - abundance_share,
                        "hvg_abundance_overlap_fraction": overlap,
                        "matrix_source": matrix_source,
                        "log1p_policy": log1p_policy,
                    }
                )
        meta["condition_rows"] = len(condition_rows)
        return condition_rows, budget_rows, meta
    finally:
        del adata


def summarize_budget_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for cols, level in [
        (["group", "dataset", "budget"], "dataset"),
        (["group", "budget"], "group"),
        (["budget"], "all"),
    ]:
        for keys, part in frame.groupby(cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(cols, keys))
            out.append(
                {
                    "level": level,
                    "group": key_map.get("group", "__ALL__"),
                    "dataset": key_map.get("dataset", "__ALL__"),
                    "budget": int(key_map["budget"]),
                    "condition_rows": int(part.shape[0]),
                    "hvg_share_mean": float(part["hvg_share"].mean()),
                    "abundance_share_mean": float(part["abundance_share"].mean()),
                    "hvg_minus_abundance_mean": float(part["hvg_minus_abundance"].mean()),
                    "hvg_abundance_overlap_fraction_mean": float(part["hvg_abundance_overlap_fraction"].mean()),
                    "budget_clipped_fraction": float(part["budget_clipped"].mean()),
                }
            )
    return out


def summarize_condition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []
    for cols, level in [(["group", "dataset"], "dataset"), (["group"], "group"), ([], "all")]:
        grouped = [(("__ALL__",), frame)] if not cols else frame.groupby(cols, sort=True)
        for keys, part in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(cols, keys))
            out.append(
                {
                    "level": level,
                    "group": key_map.get("group", "__ALL__"),
                    "dataset": key_map.get("dataset", "__ALL__"),
                    "condition_rows": int(part.shape[0]),
                    "hvg_k80_median": float(part["hvg_k80"].median()),
                    "hvg_k90_median": float(part["hvg_k90"].median()),
                    "abundance_k80_median": float(part["abundance_k80"].median()),
                    "abundance_k90_median": float(part["abundance_k90"].median()),
                    "response_energy_mean": float(part["response_energy"].mean()),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/exact_response_information_coverage_20260628")
    parser.add_argument("--min-file-mb", type=float, default=0.0)
    parser.add_argument("--max-file-mb", type=float, default=800.0)
    parser.add_argument("--max-conditions-per-dataset", type=int, default=160)
    parser.add_argument("--min-pert-cells", type=int, default=25)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = discover_raw_datasets(args.max_file_mb, args.min_file_mb)
    all_condition_rows: list[dict[str, Any]] = []
    all_budget_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for spec in datasets:
        condition_rows, budget_rows, meta = dataset_coverage_rows(
            spec,
            max_conditions=args.max_conditions_per_dataset,
            min_pert_cells=args.min_pert_cells,
        )
        all_condition_rows.extend(condition_rows)
        all_budget_rows.extend(budget_rows)
        meta_rows.append(meta)
        print(f"[{now_cst()}] {spec.dataset}: {meta['status']} rows={meta.get('condition_rows', 0)}", flush=True)

    budget_summary = summarize_budget_rows(all_budget_rows)
    condition_summary = summarize_condition_rows(all_condition_rows)
    status = "exact_response_information_coverage_partial_no_gpu"
    if len(all_condition_rows) >= 1000 and sum(1 for m in meta_rows if m.get("status") == "ok") >= 10:
        status = "exact_response_information_coverage_expanded_no_gpu"

    condition_csv = args.out_dir / "exact_response_information_condition_rows.csv"
    budget_csv = args.out_dir / "exact_response_information_budget_rows.csv"
    budget_summary_csv = args.out_dir / "exact_response_information_budget_summary.csv"
    condition_summary_csv = args.out_dir / "exact_response_information_condition_summary.csv"
    meta_csv = args.out_dir / "exact_response_information_dataset_meta.csv"
    report_md = args.out_dir / "LATENTFM_EXACT_RESPONSE_INFORMATION_COVERAGE_20260628.md"
    json_path = args.out_dir / "latentfm_exact_response_information_coverage_20260628.json"

    write_csv(
        condition_csv,
        all_condition_rows,
        [
            "group",
            "dataset",
            "condition",
            "n_vars",
            "n_control",
            "n_pert",
            "response_energy",
            "hvg_k80",
            "hvg_k90",
            "abundance_k80",
            "abundance_k90",
            "matrix_source",
            "log1p_policy",
        ],
    )
    write_csv(
        budget_csv,
        all_budget_rows,
        [
            "group",
            "dataset",
            "condition",
            "budget",
            "effective_budget",
            "budget_clipped",
            "n_vars",
            "n_control",
            "n_pert",
            "response_energy",
            "hvg_share",
            "abundance_share",
            "hvg_minus_abundance",
            "hvg_abundance_overlap_fraction",
            "matrix_source",
            "log1p_policy",
        ],
    )
    write_csv(
        budget_summary_csv,
        budget_summary,
        [
            "level",
            "group",
            "dataset",
            "budget",
            "condition_rows",
            "hvg_share_mean",
            "abundance_share_mean",
            "hvg_minus_abundance_mean",
            "hvg_abundance_overlap_fraction_mean",
            "budget_clipped_fraction",
        ],
    )
    write_csv(
        condition_summary_csv,
        condition_summary,
        [
            "level",
            "group",
            "dataset",
            "condition_rows",
            "hvg_k80_median",
            "hvg_k90_median",
            "abundance_k80_median",
            "abundance_k90_median",
            "response_energy_mean",
        ],
    )
    write_csv(
        meta_csv,
        meta_rows,
        [
            "group",
            "dataset",
            "path",
            "size_mb",
            "n_obs",
            "n_vars",
            "n_control",
            "conditions_selected",
            "condition_rows",
            "status",
            "matrix_source",
            "log1p_policy",
        ],
    )
    payload = {
        "created_at": now_cst(),
        "status": status,
        "out_dir": str(args.out_dir),
        "datasets_considered": len(datasets),
        "datasets_ok": sum(1 for m in meta_rows if m.get("status") == "ok"),
        "condition_rows": len(all_condition_rows),
        "budget_rows": len(all_budget_rows),
        "max_file_mb": args.max_file_mb,
        "min_file_mb": args.min_file_mb,
        "max_conditions_per_dataset": args.max_conditions_per_dataset,
        "min_pert_cells": args.min_pert_cells,
        "condition_csv": str(condition_csv),
        "budget_csv": str(budget_csv),
        "budget_summary_csv": str(budget_summary_csv),
        "condition_summary_csv": str(condition_summary_csv),
        "meta_csv": str(meta_csv),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    group_budget1000 = [
        row
        for row in budget_summary
        if row["level"] == "group" and int(row["budget"]) == 1000
    ]
    all_budget1000 = next(
        (row for row in budget_summary if row["level"] == "all" and int(row["budget"]) == 1000),
        None,
    )
    all_condition = next((row for row in condition_summary if row["level"] == "all"), None)
    lines = [
        "# LatentFM Exact Response-Information Coverage",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU-only expansion of exact raw-expression response-information coverage.",
        "* Uses existing raw expression matrices only; no training, inference, GPU, canonical multi, Track C query, or checkpoint selection.",
        "* Skips files above the configured size cap and caps conditions per dataset to bound I/O.",
        "",
        "## Summary",
        "",
        f"* Datasets considered: `{payload['datasets_considered']}`; ok: `{payload['datasets_ok']}`.",
        f"* Condition rows: `{payload['condition_rows']}`; budget rows: `{payload['budget_rows']}`.",
        f"* Max file size: `{args.max_file_mb}` MB; max conditions/dataset: `{args.max_conditions_per_dataset}`.",
        f"* Min file size: `{args.min_file_mb}` MB.",
    ]
    if all_budget1000:
        lines.append(
            f"* Overall top-1000 HVG share `{fmt_float(all_budget1000['hvg_share_mean'])}`, "
            f"abundance share `{fmt_float(all_budget1000['abundance_share_mean'])}`, "
            f"HVG-abundance `{fmt_float(all_budget1000['hvg_minus_abundance_mean'])}`."
        )
    if all_condition:
        lines.append(
            f"* Overall median abundance k80/k90: `{fmt_float(all_condition['abundance_k80_median'])}`/"
            f"`{fmt_float(all_condition['abundance_k90_median'])}` genes."
        )
    lines.extend(
        [
            "",
            "## Group Top-1000 Summary",
            "",
            "| group | rows | HVG share | abundance share | HVG - abundance | overlap | clipped |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in group_budget1000:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["group"]),
                    str(row["condition_rows"]),
                    fmt_float(row["hvg_share_mean"]),
                    fmt_float(row["abundance_share_mean"]),
                    fmt_float(row["hvg_minus_abundance_mean"]),
                    fmt_float(row["hvg_abundance_overlap_fraction_mean"]),
                    fmt_float(row["budget_clipped_fraction"]),
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
            "* This output is a CPU evidence table for the scaling-law branch; it does not authorize GPU.",
            "* Use it to replace group-level priors in the split-level design matrix once complete.",
            "",
            "## Outputs",
            "",
            f"* Condition CSV: `{condition_csv}`",
            f"* Budget CSV: `{budget_csv}`",
            f"* Budget summary: `{budget_summary_csv}`",
            f"* Condition summary: `{condition_summary_csv}`",
            f"* Dataset meta: `{meta_csv}`",
            f"* JSON: `{json_path}`",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_md}", flush=True)
    print(f"status {status}", flush=True)


if __name__ == "__main__":
    main()
