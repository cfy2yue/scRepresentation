#!/usr/bin/env python3
"""Complete exact response-information rows for parent train conditions.

CPU-only. This fills parent-train conditions missing from the existing exact
response-information coverage table, instead of rescanning arbitrary top
conditions. It does not train, infer, use GPU, select checkpoints, read
canonical multi for selection, or read Track C query.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
sys.path.append(str(ROOT / "ops"))
import audit_latentfm_hvg_meanmatched_negative_controls_20260628 as neg  # noqa: E402
import materialize_latentfm_exact_response_information_coverage_20260628 as cov  # noqa: E402


BUDGETS = (250, 500, 1000, 2000)
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
EXISTING_COVERAGE = ROOT / "reports/exact_response_information_combined_coverage_20260628/exact_response_information_condition_rows.csv"
OUT_DIR = ROOT / "reports/parent_train_exact_response_completion_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parent_train_conditions(path: Path) -> dict[str, set[str]]:
    split = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(dataset): {str(cond) for cond in (groups or {}).get("train", [])}
        for dataset, groups in split.items()
    }


def existing_covered(path: Path) -> set[tuple[str, str]]:
    frame = pd.read_csv(path)
    return set(zip(frame["dataset"].astype(str), frame["condition"].astype(str)))


def first_budget_for(cumulative: np.ndarray, threshold: float) -> int:
    if cumulative.size == 0:
        return 0
    idx = np.searchsorted(cumulative, threshold, side="left")
    return int(min(idx + 1, cumulative.size))


def compute_dataset_rows(
    spec: cov.RawDataset,
    targets: set[str],
    min_pert_cells: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    adata = ad.read_h5ad(spec.path)
    try:
        group_key = "chemicalpert_bench" if spec.group == "chemicalpert_bench" else "genepert_DE5000_small"
        matrix, matrix_source, log1p_policy = neg.get_matrix(adata, group_key)
        obs = adata.obs.copy()
        if "perturbation" not in obs.columns:
            return [], [], {
                "group": spec.group,
                "dataset": spec.dataset,
                "path": str(spec.path),
                "size_mb": spec.size_mb,
                "status": "missing_perturbation_column",
                "targets_requested": len(targets),
                "condition_rows": 0,
            }
        mask_control = neg.control_mask(obs, group_key)
        n_control = int(mask_control.sum())
        if n_control <= 0:
            return [], [], {
                "group": spec.group,
                "dataset": spec.dataset,
                "path": str(spec.path),
                "size_mb": spec.size_mb,
                "status": "no_control_cells",
                "targets_requested": len(targets),
                "condition_rows": 0,
            }
        perturb_values = obs["perturbation"].astype(str).to_numpy()
        present_counts = pd.Series(perturb_values[~mask_control]).value_counts()
        targets_present = [cond for cond in sorted(targets) if int(present_counts.get(cond, 0)) >= min_pert_cells]
        missing_or_low = sorted(set(targets) - set(targets_present))
        control_matrix = matrix[mask_control]
        control_mean = neg.dense_mean(control_matrix)
        control_var = neg.dense_var(control_matrix)
        hvg_order = np.argsort(-control_var, kind="mergesort")
        abundance_order = np.argsort(-control_mean, kind="mergesort")
        condition_rows: list[dict[str, Any]] = []
        budget_rows: list[dict[str, Any]] = []
        for condition in targets_present:
            mask_condition = perturb_values == condition
            n_pert = int(mask_condition.sum())
            pert_mean = neg.dense_mean(matrix[mask_condition])
            response_sq = np.square(pert_mean - control_mean)
            total_energy = float(response_sq.sum())
            if total_energy <= 0 or not np.isfinite(total_energy):
                missing_or_low.append(condition)
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
        meta = {
            "group": spec.group,
            "dataset": spec.dataset,
            "path": str(spec.path),
            "size_mb": spec.size_mb,
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "n_control": n_control,
            "targets_requested": len(targets),
            "targets_present_min_cells": len(targets_present),
            "missing_or_low_pert_cells": len(set(missing_or_low)),
            "condition_rows": len(condition_rows),
            "status": "ok",
            "matrix_source": matrix_source,
            "log1p_policy": log1p_policy,
        }
        return condition_rows, budget_rows, meta
    finally:
        del adata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-split", type=Path, default=PARENT_SPLIT)
    parser.add_argument("--existing-coverage", type=Path, default=EXISTING_COVERAGE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-file-mb", type=float, default=0.0)
    parser.add_argument("--max-file-mb", type=float, default=20000.0)
    parser.add_argument("--min-pert-cells", type=int, default=25)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parent = parent_train_conditions(args.parent_split)
    covered = existing_covered(args.existing_coverage)
    raw_specs = {spec.dataset: spec for spec in cov.discover_raw_datasets(args.max_file_mb, args.min_file_mb)}

    all_condition_rows: list[dict[str, Any]] = []
    all_budget_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    missing_raw_rows: list[dict[str, Any]] = []

    for dataset, train_conditions in sorted(parent.items()):
        targets = {condition for condition in train_conditions if (dataset, condition) not in covered}
        if not targets:
            meta_rows.append(
                {
                    "dataset": dataset,
                    "status": "already_complete_for_parent_train",
                    "targets_requested": 0,
                    "condition_rows": 0,
                }
            )
            continue
        spec = raw_specs.get(dataset)
        if spec is None:
            missing_raw_rows.append({"dataset": dataset, "missing_parent_train_conditions": len(targets)})
            meta_rows.append(
                {
                    "dataset": dataset,
                    "status": "missing_raw_h5ad",
                    "targets_requested": len(targets),
                    "condition_rows": 0,
                }
            )
            continue
        condition_rows, budget_rows, meta = compute_dataset_rows(spec, targets, args.min_pert_cells)
        all_condition_rows.extend(condition_rows)
        all_budget_rows.extend(budget_rows)
        meta_rows.append(meta)
        print(
            f"[{now_cst()}] {dataset}: targets={len(targets)} rows={len(condition_rows)} status={meta['status']}",
            flush=True,
        )

    condition_csv = args.out_dir / "parent_train_exact_response_completion_condition_rows.csv"
    budget_csv = args.out_dir / "parent_train_exact_response_completion_budget_rows.csv"
    meta_csv = args.out_dir / "parent_train_exact_response_completion_dataset_meta.csv"
    missing_csv = args.out_dir / "parent_train_exact_response_completion_missing_raw.csv"
    report_md = args.out_dir / "LATENTFM_PARENT_TRAIN_EXACT_RESPONSE_COMPLETION_20260628.md"
    json_path = args.out_dir / "parent_train_exact_response_completion_20260628.json"

    cov.write_csv(
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
    cov.write_csv(
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
    cov.write_csv(
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
            "targets_requested",
            "targets_present_min_cells",
            "missing_or_low_pert_cells",
            "condition_rows",
            "status",
            "matrix_source",
            "log1p_policy",
        ],
    )
    cov.write_csv(missing_csv, missing_raw_rows, ["dataset", "missing_parent_train_conditions"])

    total_parent_train = sum(len(v) for v in parent.values())
    missing_before = sum(
        1 for dataset, conditions in parent.items() for condition in conditions if (dataset, condition) not in covered
    )
    payload = {
        "timestamp": now_cst(),
        "status": "parent_train_exact_response_completion_done_no_gpu",
        "gpu_authorized_next": False,
        "parent_split": str(args.parent_split),
        "existing_coverage": str(args.existing_coverage),
        "total_parent_train_conditions": total_parent_train,
        "missing_before": missing_before,
        "new_condition_rows": len(all_condition_rows),
        "new_budget_rows": len(all_budget_rows),
        "missing_raw_datasets": len(missing_raw_rows),
        "min_pert_cells": args.min_pert_cells,
        "max_file_mb": args.max_file_mb,
        "outputs": {
            "condition_csv": str(condition_csv),
            "budget_csv": str(budget_csv),
            "meta_csv": str(meta_csv),
            "missing_raw_csv": str(missing_csv),
        },
    }
    write_json(json_path, payload)

    lines = [
        "# LatentFM Parent-Train Exact Response Completion",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized next: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only exact response-information completion for parent train conditions missing from the combined coverage table.",
        "- Uses raw h5ad matrices only; no training, inference, GPU, checkpoint selection, canonical multi selection, or Track C query.",
        "- Raw counts/expression are processed through the existing audited matrix helper and log1p policy.",
        "",
        "## Summary",
        "",
        f"- parent train conditions: `{total_parent_train}`",
        f"- missing before completion: `{missing_before}`",
        f"- new condition rows: `{len(all_condition_rows)}`",
        f"- new budget rows: `{len(all_budget_rows)}`",
        f"- missing raw datasets: `{len(missing_raw_rows)}`",
        "",
        "## Decision",
        "",
        "This completion table is an input to the next clean scaling-x matrix. It does not authorize GPU by itself.",
        "",
        "## Outputs",
        "",
        f"- Condition rows: `{condition_csv}`",
        f"- Budget rows: `{budget_csv}`",
        f"- Dataset meta: `{meta_csv}`",
        f"- Missing raw: `{missing_csv}`",
        f"- JSON: `{json_path}`",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
