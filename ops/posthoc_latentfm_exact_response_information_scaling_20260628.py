#!/usr/bin/env python3
"""Posthoc integration for exact response-information coverage expansion."""

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
INFO_CSV = ROOT / "reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv"
OUTCOME_CSV = ROOT / "reports/downstream_condition_residual_association_gate_20260628/condition_residual_information_outcome_join_rows.csv"


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


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


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


def permutation_p(x: np.ndarray, y: np.ndarray, observed: float, n_perm: int = 2000) -> float:
    if len(x) < 5 or not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(45)
    extreme = 1
    for _ in range(n_perm):
        val = spearman(x, rng.permutation(y))
        if np.isfinite(val) and abs(val) >= abs(observed):
            extreme += 1
    return extreme / (n_perm + 1)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def build_split_matrix(coverage_dir: Path, budget: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budget_rows_path = coverage_dir / "exact_response_information_budget_rows.csv"
    condition_rows_path = coverage_dir / "exact_response_information_condition_rows.csv"
    if not budget_rows_path.is_file() or not condition_rows_path.is_file():
        raise FileNotFoundError(f"missing exact coverage outputs in {coverage_dir}")

    budget_rows = pd.read_csv(budget_rows_path)
    budget_rows = budget_rows[budget_rows["budget"] == budget]
    condition_rows = pd.read_csv(condition_rows_path)
    budget_by_key = {
        (str(row.dataset), str(row.condition)): row
        for row in budget_rows.itertuples(index=False)
    }
    condition_by_key = {
        (str(row.dataset), str(row.condition)): row
        for row in condition_rows.itertuples(index=False)
    }
    split_rows = pd.read_csv(INFO_CSV)
    outcome_rows = {
        str(row.split_name): row
        for row in pd.read_csv(OUTCOME_CSV).itertuples(index=False)
    } if OUTCOME_CSV.exists() else {}

    out: list[dict[str, Any]] = []
    for split_info in split_rows.itertuples(index=False):
        split = load_json(ROOT / str(split_info.split_file))
        total = 0
        exact = 0
        hvg_shares: list[float] = []
        abundance_shares: list[float] = []
        hvg_minus_abundance: list[float] = []
        hvg_k80: list[float] = []
        abundance_k80: list[float] = []
        hvg_k90: list[float] = []
        abundance_k90: list[float] = []
        for dataset, groups in split.items():
            for condition in groups.get("train", []):
                total += 1
                key = (str(dataset), str(condition))
                brow = budget_by_key.get(key)
                crow = condition_by_key.get(key)
                if brow is None or crow is None:
                    continue
                exact += 1
                hvg_shares.append(safe_float(getattr(brow, "hvg_share")))
                abundance_shares.append(safe_float(getattr(brow, "abundance_share")))
                hvg_minus_abundance.append(safe_float(getattr(brow, "hvg_minus_abundance")))
                hvg_k80.append(safe_float(getattr(crow, "hvg_k80")))
                hvg_k90.append(safe_float(getattr(crow, "hvg_k90")))
                abundance_k80.append(safe_float(getattr(crow, "abundance_k80")))
                abundance_k90.append(safe_float(getattr(crow, "abundance_k90")))
        outcome = outcome_rows.get(str(split_info.split_name))
        row = {
            "split_file": str(split_info.split_file),
            "split_name": str(split_info.split_name),
            "n_train_conditions": int(total),
            "exact_condition_rows": int(exact),
            "exact_condition_fraction": exact / max(total, 1),
            f"exact_hvg_share_top{budget}_mean": finite_mean(hvg_shares),
            f"exact_abundance_share_top{budget}_mean": finite_mean(abundance_shares),
            f"exact_hvg_minus_abundance_top{budget}_mean": finite_mean(hvg_minus_abundance),
            "exact_hvg_k80_mean": finite_mean(hvg_k80),
            "exact_hvg_k90_mean": finite_mean(hvg_k90),
            "exact_abundance_k80_mean": finite_mean(abundance_k80),
            "exact_abundance_k90_mean": finite_mean(abundance_k90),
            "base_dataset_effective_count": safe_float(getattr(split_info, "dataset_effective_count")),
            "base_background_effective_count": safe_float(getattr(split_info, "background_effective_count")),
            "base_perturbation_type_effective_count": safe_float(getattr(split_info, "perturbation_type_effective_count")),
            "base_target_gene_effective_count": safe_float(getattr(split_info, "target_gene_effective_count")),
            "has_downstream_outcome": outcome is not None,
            "cross_pp_delta": safe_float(getattr(outcome, "cross_pp_delta", float("nan"))) if outcome else float("nan"),
            "family_pp_delta": safe_float(getattr(outcome, "family_pp_delta", float("nan"))) if outcome else float("nan"),
            "family_mmd_delta": safe_float(getattr(outcome, "family_mmd_delta", float("nan"))) if outcome else float("nan"),
            "tail_score": safe_float(getattr(outcome, "tail_score", float("nan"))) if outcome else float("nan"),
        }
        out.append(row)
    payload = {
        "split_rows": len(out),
        "budget": budget,
        "mean_exact_condition_fraction": finite_mean([row["exact_condition_fraction"] for row in out]),
        "splits_with_downstream_outcomes": sum(1 for row in out if row["has_downstream_outcome"]),
    }
    return out, payload


def association_rows(matrix_rows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    predictors = [
        f"exact_hvg_share_top{budget}_mean",
        f"exact_abundance_share_top{budget}_mean",
        "exact_abundance_k80_mean",
        "exact_abundance_k90_mean",
        "exact_condition_fraction",
    ]
    outcomes = ["cross_pp_delta", "family_pp_delta", "family_mmd_delta", "tail_score"]
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for outcome in outcomes:
            part = [
                row
                for row in matrix_rows
                if row["has_downstream_outcome"]
                and row["exact_condition_fraction"] > 0
                and np.isfinite(safe_float(row.get(predictor)))
                and np.isfinite(safe_float(row.get(outcome)))
            ]
            x = np.asarray([safe_float(row[predictor]) for row in part], dtype=float)
            y = np.asarray([safe_float(row[outcome]) for row in part], dtype=float)
            rho = spearman(x, y)
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n": int(len(part)),
                    "rho": rho,
                    "p_perm": permutation_p(x, y, rho) if len(part) >= 5 else float("nan"),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matrix_rows, payload = build_split_matrix(args.coverage_dir, args.budget)
    assoc_rows = association_rows(matrix_rows, args.budget)
    status = "exact_response_information_posthoc_partial_no_gpu"
    if payload["mean_exact_condition_fraction"] >= 0.5:
        status = "exact_response_information_posthoc_ready_for_clustered_ci_no_gpu"

    matrix_csv = args.out_dir / "exact_response_information_split_matrix.csv"
    assoc_csv = args.out_dir / "exact_response_information_association_rows.csv"
    json_path = args.out_dir / "latentfm_exact_response_information_posthoc_20260628.json"
    report_md = args.out_dir / "LATENTFM_EXACT_RESPONSE_INFORMATION_POSTHOC_20260628.md"
    matrix_fields = [
        "split_file",
        "split_name",
        "n_train_conditions",
        "exact_condition_rows",
        "exact_condition_fraction",
        f"exact_hvg_share_top{args.budget}_mean",
        f"exact_abundance_share_top{args.budget}_mean",
        f"exact_hvg_minus_abundance_top{args.budget}_mean",
        "exact_hvg_k80_mean",
        "exact_hvg_k90_mean",
        "exact_abundance_k80_mean",
        "exact_abundance_k90_mean",
        "base_dataset_effective_count",
        "base_background_effective_count",
        "base_perturbation_type_effective_count",
        "base_target_gene_effective_count",
        "has_downstream_outcome",
        "cross_pp_delta",
        "family_pp_delta",
        "family_mmd_delta",
        "tail_score",
    ]
    assoc_fields = ["predictor", "outcome", "n", "rho", "p_perm"]
    write_csv(matrix_csv, matrix_rows, matrix_fields)
    write_csv(assoc_csv, assoc_rows, assoc_fields)
    payload.update(
        {
            "created_at": now_cst(),
            "status": status,
            "coverage_dir": str(args.coverage_dir),
            "matrix_csv": str(matrix_csv),
            "association_csv": str(assoc_csv),
        }
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    top_assoc = sorted(
        [row for row in assoc_rows if np.isfinite(safe_float(row["rho"]))],
        key=lambda r: abs(safe_float(r["rho"])),
        reverse=True,
    )[:8]
    lines = [
        "# LatentFM Exact Response-Information Posthoc",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only posthoc integration of completed exact coverage outputs.",
        "* Does not train, infer, use canonical multi, use Track C query, or select checkpoints.",
        "",
        "## Summary",
        "",
        f"* Split rows: `{payload['split_rows']}`.",
        f"* Mean exact condition fraction: `{fmt_float(payload['mean_exact_condition_fraction'])}`.",
        f"* Splits with downstream outcomes: `{payload['splits_with_downstream_outcomes']}`.",
        "",
        "## Top Associations",
        "",
        "| predictor | outcome | n | rho | p_perm |",
        "|---|---|---:|---:|---:|",
    ]
    for row in top_assoc:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["predictor"]),
                    str(row["outcome"]),
                    str(row["n"]),
                    fmt_float(row["rho"]),
                    fmt_float(row["p_perm"]),
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
            "* If ready, the next step is clustered CI / LODO analysis; otherwise expand exact coverage further.",
            "* No GPU is authorized by this posthoc.",
            "",
            "## Outputs",
            "",
            f"* Split matrix: `{matrix_csv}`",
            f"* Association rows: `{assoc_csv}`",
            f"* JSON: `{json_path}`",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_md}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
