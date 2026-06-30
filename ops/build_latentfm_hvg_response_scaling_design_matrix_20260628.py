#!/usr/bin/env python3
"""Build a split-level design matrix for HVG/response-information scaling.

CPU/report-only. This joins split train-condition composition with the
raw-expression HVG-budget gate. It separates exact measured coverage from
dataset/group priors and does not train, infer, or authorize GPU use.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
INFO_CSV = ROOT / "reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv"
OUTCOME_CSV = ROOT / "reports/downstream_condition_residual_association_gate_20260628/condition_residual_information_outcome_join_rows.csv"
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
HVG_COND_CSV = ROOT / "reports/raw_expression_hvg_budget_predictability_gate_20260628/condition_budget_rows.csv"
HVG_SUMMARY_CSV = ROOT / "reports/raw_expression_hvg_budget_predictability_gate_20260628/budget_summary_rows.csv"

OUT_DIR = ROOT / "reports/hvg_response_scaling_design_matrix_20260628"
OUT_MATRIX = OUT_DIR / "hvg_response_scaling_design_matrix.csv"
OUT_RAW_SCHEMA = OUT_DIR / "raw_expression_matrix_schema.csv"
OUT_MD = OUT_DIR / "LATENTFM_HVG_RESPONSE_SCALING_DESIGN_MATRIX_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_hvg_response_scaling_design_matrix_20260628.json"

BUDGET = 1000


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def raw_path_for_dataset(dataset: str) -> Path | None:
    if dataset.startswith("sciplex3_"):
        bench = ROOT / f"dataset/raw/chemicalpert_bench/{dataset}.h5ad"
        de5000 = ROOT / f"dataset/raw/chemicalpert_DE5000/{dataset}.h5ad"
        return bench if bench.exists() else de5000 if de5000.exists() else None
    de5000 = ROOT / f"dataset/raw/genepert_DE5000/{dataset}.h5ad"
    return de5000 if de5000.exists() else None


def h5_shape(obj: h5py.Dataset | h5py.Group) -> tuple[int, int] | tuple[None, None]:
    if isinstance(obj, h5py.Group):
        shape = obj.attrs.get("shape")
        if shape is not None and len(shape) == 2:
            return int(shape[0]), int(shape[1])
        return None, None
    if obj.shape is not None and len(obj.shape) == 2:
        return int(obj.shape[0]), int(obj.shape[1])
    return None, None


def matrix_schema(dataset: str) -> dict[str, Any]:
    path = raw_path_for_dataset(dataset)
    row: dict[str, Any] = {
        "dataset": dataset,
        "path": str(path) if path else "",
        "exists": bool(path and path.exists()),
        "has_real_matrix": False,
        "preferred_matrix": "",
        "n_obs": "",
        "n_vars": "",
        "notes": "",
    }
    if not path or not path.exists():
        row["notes"] = "missing_raw_h5ad"
        return row
    with h5py.File(path, "r") as h5:
        candidates = []
        for key in ("layers/logNor", "X", "layers/counts"):
            if key not in h5:
                continue
            obj = h5[key]
            enc = obj.attrs.get("encoding-type", "")
            if isinstance(enc, bytes):
                enc = enc.decode()
            if enc == "null":
                continue
            n_obs, n_vars = h5_shape(obj)
            if n_obs and n_vars:
                candidates.append((key, n_obs, n_vars, str(enc)))
        if candidates:
            key, n_obs, n_vars, enc = candidates[0]
            row.update(
                {
                    "has_real_matrix": True,
                    "preferred_matrix": key,
                    "n_obs": n_obs,
                    "n_vars": n_vars,
                    "notes": f"encoding={enc}",
                }
            )
        else:
            row["notes"] = "no_nonnull_matrix_candidate"
    return row


def perturbation_family(meta: dict[str, Any]) -> str:
    raw = str(meta.get("perturbation_type_raw", "")).lower()
    if "drug" in raw or "chemical" in raw:
        return "chemicalpert_bench"
    return "genepert_DE5000_small"


def load_hvg_priors() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], set[tuple[str, str]]]:
    summary = pd.read_csv(HVG_SUMMARY_CSV)
    summary = summary[summary["budget"] == BUDGET]
    dataset_prior: dict[str, dict[str, float]] = {}
    group_prior: dict[str, dict[str, float]] = {}
    for _, row in summary.iterrows():
        target = {
            "hvg_share": float(row["control_hvg_share_mean"]),
            "random_share": float(row["random_share_mean"]),
            "hvg_minus_random": float(row["hvg_minus_random_mean"]),
            "oracle_share": float(row["oracle_response_share_mean"]),
        }
        dataset = str(row["dataset"])
        group = str(row["group"])
        if dataset == "__GROUP__":
            group_prior[group] = target
        elif dataset not in {"__ALL__"}:
            dataset_prior[dataset] = target
    cond = pd.read_csv(HVG_COND_CSV)
    cond = cond[cond["budget"] == BUDGET]
    measured_conditions = {(str(r["dataset"]), str(r["condition"])) for _, r in cond.iterrows()}
    return dataset_prior, group_prior, measured_conditions


def read_split(path: str) -> dict[str, dict[str, list[str]]]:
    with (ROOT / path).open(encoding="utf-8") as fh:
        return json.load(fh)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def weighted_mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def build_matrix() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with COND_META.open(encoding="utf-8") as fh:
        cond_meta: dict[str, dict[str, dict[str, Any]]] = json.load(fh)
    schema_rows = [matrix_schema(dataset) for dataset in sorted(cond_meta)]
    schema_by_dataset = {row["dataset"]: row for row in schema_rows}
    dataset_prior, group_prior, measured_conditions = load_hvg_priors()
    split_rows = load_csv(INFO_CSV)
    outcome_rows = {row["split_name"]: row for row in load_csv(OUTCOME_CSV)} if OUTCOME_CSV.exists() else {}

    matrix_rows: list[dict[str, Any]] = []
    for split_info in split_rows:
        split = read_split(split_info["split_file"])
        train_items: list[tuple[str, str, dict[str, Any]]] = []
        for dataset, groups in split.items():
            for condition in groups.get("train", []):
                meta = cond_meta.get(dataset, {}).get(str(condition), {})
                train_items.append((dataset, str(condition), meta))

        counts = Counter()
        values_exact_dataset: list[float] = []
        values_exact_condition: list[float] = []
        values_group_imputed: list[float] = []
        advantages_group_imputed: list[float] = []
        random_group_imputed: list[float] = []
        oracle_group_imputed: list[float] = []

        for dataset, condition, meta in train_items:
            counts["total"] += 1
            schema = schema_by_dataset.get(dataset, {})
            has_raw = schema.get("has_real_matrix") is True
            counts["raw_available"] += int(has_raw)
            family = perturbation_family(meta)
            counts[family] += 1
            if dataset in dataset_prior:
                counts["dataset_prior_available"] += 1
                values_exact_dataset.append(dataset_prior[dataset]["hvg_share"])
            if (dataset, condition) in measured_conditions:
                counts["condition_exact_measured"] += 1
                values_exact_condition.append(dataset_prior.get(dataset, group_prior.get(family, {})).get("hvg_share", float("nan")))
            prior = dataset_prior.get(dataset) or group_prior.get(family)
            if prior:
                counts["group_or_dataset_prior_available"] += 1
                values_group_imputed.append(prior["hvg_share"])
                advantages_group_imputed.append(prior["hvg_minus_random"])
                random_group_imputed.append(prior["random_share"])
                oracle_group_imputed.append(prior["oracle_share"])

        total = max(counts["total"], 1)
        outcome = outcome_rows.get(split_info["split_name"], {})
        row: dict[str, Any] = {
            "split_file": split_info["split_file"],
            "split_name": split_info["split_name"],
            "n_train_conditions": counts["total"],
            "raw_expression_available_fraction": counts["raw_available"] / total,
            "hvg_dataset_prior_fraction": counts["dataset_prior_available"] / total,
            "hvg_condition_exact_fraction": counts["condition_exact_measured"] / total,
            "hvg_group_or_dataset_prior_fraction": counts["group_or_dataset_prior_available"] / total,
            "gene_condition_fraction_from_meta": counts["genepert_DE5000_small"] / total,
            "chemical_condition_fraction_from_meta": counts["chemicalpert_bench"] / total,
            "hvg_top1000_dataset_exact_mean": weighted_mean(values_exact_dataset),
            "hvg_top1000_condition_exact_mean": weighted_mean(values_exact_condition),
            "hvg_top1000_group_or_dataset_prior_mean": weighted_mean(values_group_imputed),
            "hvg_top1000_random_group_or_dataset_prior_mean": weighted_mean(random_group_imputed),
            "hvg_top1000_advantage_group_or_dataset_prior_mean": weighted_mean(advantages_group_imputed),
            "hvg_top1000_oracle_group_or_dataset_prior_mean": weighted_mean(oracle_group_imputed),
            "cross_pp_delta": safe_float(outcome.get("cross_pp_delta")),
            "family_pp_delta": safe_float(outcome.get("family_pp_delta")),
            "family_mmd_delta": safe_float(outcome.get("family_mmd_delta")),
            "tail_score": safe_float(outcome.get("tail_score")),
            "has_downstream_outcome": bool(outcome),
            "base_dataset_effective_count": safe_float(split_info.get("dataset_effective_count")),
            "base_background_effective_count": safe_float(split_info.get("background_effective_count")),
            "base_perturbation_type_effective_count": safe_float(split_info.get("perturbation_type_effective_count")),
            "base_target_gene_effective_count": safe_float(split_info.get("target_gene_effective_count")),
        }
        matrix_rows.append(row)

    payload = {
        "created_at": now_cst(),
        "budget": BUDGET,
        "split_rows": len(matrix_rows),
        "raw_schema_rows": len(schema_rows),
        "splits_with_downstream_outcomes": sum(1 for row in matrix_rows if row["has_downstream_outcome"]),
        "mean_raw_available_fraction": weighted_mean([row["raw_expression_available_fraction"] for row in matrix_rows]),
        "mean_hvg_dataset_prior_fraction": weighted_mean([row["hvg_dataset_prior_fraction"] for row in matrix_rows]),
        "mean_hvg_condition_exact_fraction": weighted_mean([row["hvg_condition_exact_fraction"] for row in matrix_rows]),
        "mean_hvg_group_or_dataset_prior_fraction": weighted_mean(
            [row["hvg_group_or_dataset_prior_fraction"] for row in matrix_rows]
        ),
    }
    return matrix_rows, schema_rows, payload


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_rows, schema_rows, payload = build_matrix()
    status = "hvg_response_scaling_design_matrix_partial_no_gpu"
    if payload["mean_hvg_condition_exact_fraction"] >= 0.5:
        status = "hvg_response_scaling_design_matrix_ready_no_gpu"
    payload["status"] = status
    payload["matrix_csv"] = str(OUT_MATRIX)
    payload["schema_csv"] = str(OUT_RAW_SCHEMA)

    matrix_fields = [
        "split_file",
        "split_name",
        "n_train_conditions",
        "raw_expression_available_fraction",
        "hvg_dataset_prior_fraction",
        "hvg_condition_exact_fraction",
        "hvg_group_or_dataset_prior_fraction",
        "gene_condition_fraction_from_meta",
        "chemical_condition_fraction_from_meta",
        "hvg_top1000_dataset_exact_mean",
        "hvg_top1000_condition_exact_mean",
        "hvg_top1000_group_or_dataset_prior_mean",
        "hvg_top1000_random_group_or_dataset_prior_mean",
        "hvg_top1000_advantage_group_or_dataset_prior_mean",
        "hvg_top1000_oracle_group_or_dataset_prior_mean",
        "cross_pp_delta",
        "family_pp_delta",
        "family_mmd_delta",
        "tail_score",
        "has_downstream_outcome",
        "base_dataset_effective_count",
        "base_background_effective_count",
        "base_perturbation_type_effective_count",
        "base_target_gene_effective_count",
    ]
    schema_fields = ["dataset", "path", "exists", "has_real_matrix", "preferred_matrix", "n_obs", "n_vars", "notes"]
    write_csv(OUT_MATRIX, matrix_rows, matrix_fields)
    write_csv(OUT_RAW_SCHEMA, schema_rows, schema_fields)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    exact_sorted = sorted(matrix_rows, key=lambda r: safe_float(r["hvg_condition_exact_fraction"]), reverse=True)[:8]
    dataset_sorted = sorted(matrix_rows, key=lambda r: safe_float(r["hvg_dataset_prior_fraction"]), reverse=True)[:8]
    lines = [
        "# LatentFM HVG/Response-Information Scaling Design Matrix",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only split-level design matrix.",
        "* Reads split JSON, xVERSE condition metadata, raw-expression HDF5 schema, and the HVG-budget gate outputs.",
        "* Separates exact measured condition coverage from dataset/group priors; imputed values are not model evidence.",
        "* Does not train, infer, evaluate canonical multi, use Track C query, select checkpoints, or authorize GPU.",
        "",
        "## Summary",
        "",
        f"* Split rows: `{payload['split_rows']}`.",
        f"* Splits with downstream outcomes: `{payload['splits_with_downstream_outcomes']}`.",
        f"* Mean raw-expression availability fraction: `{fmt_float(payload['mean_raw_available_fraction'])}`.",
        f"* Mean exact dataset-prior fraction from the current HVG gate: `{fmt_float(payload['mean_hvg_dataset_prior_fraction'])}`.",
        f"* Mean exact condition-measured fraction from the current HVG gate: `{fmt_float(payload['mean_hvg_condition_exact_fraction'])}`.",
        f"* Mean group-or-dataset prior coverage: `{fmt_float(payload['mean_hvg_group_or_dataset_prior_fraction'])}`.",
        "",
        "## Highest Exact Condition Coverage Splits",
        "",
        "| split | train conditions | condition-exact fraction | dataset-prior fraction | group/dataset prior fraction | top1000 prior mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in exact_sorted:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split_name"]),
                    str(row["n_train_conditions"]),
                    fmt_float(row["hvg_condition_exact_fraction"]),
                    fmt_float(row["hvg_dataset_prior_fraction"]),
                    fmt_float(row["hvg_group_or_dataset_prior_fraction"]),
                    fmt_float(row["hvg_top1000_group_or_dataset_prior_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Highest Dataset-Prior Coverage Splits",
            "",
            "| split | train conditions | dataset-prior fraction | condition-exact fraction | top1000 dataset exact mean | top1000 group/dataset prior mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dataset_sorted:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split_name"]),
                    str(row["n_train_conditions"]),
                    fmt_float(row["hvg_dataset_prior_fraction"]),
                    fmt_float(row["hvg_condition_exact_fraction"]),
                    fmt_float(row["hvg_top1000_dataset_exact_mean"]),
                    fmt_float(row["hvg_top1000_group_or_dataset_prior_mean"]),
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
            "* The current HVG gate supports the existence of a strong information-budget prior, but exact per-split/per-condition coverage is still sparse.",
            "* A formal scaling-law fit should either expand the CPU raw-expression computation to all train datasets/conditions or explicitly model group priors as priors, not observations.",
            "* No immediate GPU launch is authorized by this matrix.",
            "",
            "## Outputs",
            "",
            f"* Matrix CSV: `{OUT_MATRIX}`",
            f"* Raw schema CSV: `{OUT_RAW_SCHEMA}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
