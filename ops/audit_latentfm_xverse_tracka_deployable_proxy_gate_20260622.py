#!/usr/bin/env python3
"""Gate whether residual-forensics signals have deployable Track A proxies.

This is intentionally CPU-only and reads only the train-only internal-val
condition table produced by the residual-forensics audit.  It does not read
canonical test, canonical multi, or Track C query artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
INPUT_CSV = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
OUT_JSON = ROOT / "reports/latentfm_xverse_tracka_deployable_proxy_gate_20260622.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TRACKA_DEPLOYABLE_PROXY_GATE_20260622.md"

FEATURES = [
    "gene_train_count",
    "gene_pred_norm",
    "dataset_pred_norm",
    "global_pred_norm",
    "gene_dataset_cosine",
]
TARGET = "anchor_minus_gene_raw_mean"
BASELINES = ["anchor_pearson_pert", "gene_raw_mean", "dataset_mean"]
BOOT_N = 2000
SEED = 42


def parse_float(value: str) -> float:
    if value is None or value == "" or value.lower() == "nan":
        return float("nan")
    return float(value)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    xr = rankdata_average(x[mask])
    yr = rankdata_average(y[mask])
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def r2_score(y: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return None
    yy = y[mask]
    pp = pred[mask]
    denom = float(np.sum((yy - yy.mean()) ** 2))
    if denom == 0:
        return None
    return float(1.0 - np.sum((yy - pp) ** 2) / denom)


def linear_r2_from_feature(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    xx = x[mask]
    yy = y[mask]
    if float(np.std(xx)) == 0:
        return None
    design = np.c_[np.ones(len(xx)), xx]
    coef, *_ = np.linalg.lstsq(design, yy, rcond=None)
    pred = design @ coef
    return r2_score(yy, pred)


def equal_dataset_values(values: np.ndarray, datasets: list[str]) -> np.ndarray:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for value, ds in zip(values, datasets):
        if np.isfinite(value):
            by_ds[str(ds)].append(float(value))
    return np.asarray([float(np.mean(by_ds[ds])) for ds in sorted(by_ds) if by_ds[ds]], dtype=float)


def bootstrap_delta(a: np.ndarray, b: np.ndarray, datasets: list[str]) -> dict:
    delta = equal_dataset_values(a - b, datasets)
    rng = np.random.default_rng(SEED)
    boots = []
    n = len(delta)
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, size=n)
        boots.append(float(delta[idx].mean()))
    arr = np.asarray(boots)
    return {
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_harm": float(np.mean(arr <= 0.0)),
    }


def fit_predict_lodo(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, dict]:
    datasets = sorted({r["dataset"] for r in rows})
    y = np.asarray([r[TARGET] for r in rows], dtype=float)
    X_full = np.asarray([[r[f] for f in feature_names] for r in rows], dtype=float)
    pred = np.full(len(rows), np.nan, dtype=float)
    fold_notes = []
    for ds in datasets:
        test = np.asarray([r["dataset"] == ds for r in rows], dtype=bool)
        train = ~test
        X_train = X_full[train]
        y_train = y[train]
        finite_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        finite_test = np.isfinite(X_full[test]).all(axis=1)
        usable = []
        for j in range(X_train.shape[1]):
            col = X_train[finite_train, j]
            if len(col) >= 3 and float(np.std(col)) > 1e-12:
                usable.append(j)
        if len(usable) == 0 or finite_train.sum() < 5:
            fold_notes.append({"dataset": ds, "status": "fallback_train_mean"})
            pred[test] = float(np.nanmean(y_train))
            continue
        Xtr = X_train[finite_train][:, usable]
        ytr = y_train[finite_train]
        mean = Xtr.mean(axis=0)
        std = Xtr.std(axis=0)
        std[std < 1e-12] = 1.0
        Xz = (Xtr - mean) / std
        design = np.c_[np.ones(len(Xz)), Xz]
        alpha = 1.0
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ ytr)
        fold_pred = np.full(test.sum(), float(np.mean(ytr)))
        if finite_test.any():
            Xte = X_full[test][:, usable]
            Xte_z = (Xte[finite_test] - mean) / std
            fold_pred[finite_test] = np.c_[np.ones(finite_test.sum()), Xte_z] @ coef
        pred[test] = fold_pred
        fold_notes.append(
            {
                "dataset": ds,
                "status": "ridge_lodo",
                "n_train": int(finite_train.sum()),
                "n_test": int(test.sum()),
                "features": [feature_names[j] for j in usable],
            }
        )
    return pred, {"folds": fold_notes}


def summarize_group(rows: list[dict]) -> dict:
    datasets = [str(r["dataset"]) for r in rows]
    y = np.asarray([r[TARGET] for r in rows], dtype=float)
    anchor = np.asarray([r["anchor_pearson_pert"] for r in rows], dtype=float)
    gene = np.asarray([r["gene_raw_mean"] for r in rows], dtype=float)
    dataset = np.asarray([r["dataset_mean"] for r in rows], dtype=float)
    feature_tests = []
    for feat in FEATURES:
        x = np.asarray([r[feat] for r in rows], dtype=float)
        feature_tests.append(
            {
                "feature": feat,
                "spearman_to_anchor_minus_gene": spearman(x, y),
                "linear_r2_to_anchor_minus_gene": linear_r2_from_feature(x, y),
            }
        )

    pred, lodo_meta = fit_predict_lodo(rows, FEATURES)
    route = np.where(pred > 0.0, anchor, gene)
    oracle_route = np.where(y > 0.0, anchor, gene)
    dataset_deltas = {}
    for ds in sorted({r["dataset"] for r in rows}):
        mask = np.asarray([r["dataset"] == ds for r in rows], dtype=bool)
        dataset_deltas[ds] = {
            "n": int(mask.sum()),
            "route_minus_gene": float(np.mean(route[mask] - gene[mask])),
            "oracle_minus_gene": float(np.mean(oracle_route[mask] - gene[mask])),
            "predicted_delta_mean": float(np.mean(pred[mask])),
            "actual_anchor_minus_gene_mean": float(np.mean(y[mask])),
        }
    material_harm_datasets = [
        ds for ds, vals in dataset_deltas.items() if vals["route_minus_gene"] < -0.02
    ]
    return {
        "n": len(rows),
        "feature_tests": feature_tests,
        "lodo_model": {
            "features": FEATURES,
            "target": TARGET,
            "spearman": spearman(pred, y),
            "r2": r2_score(y, pred),
            "predicted_positive_fraction": float(np.mean(pred > 0.0)),
            "meta": lodo_meta,
        },
        "means": {
            "anchor_pearson_pert": float(np.mean(equal_dataset_values(anchor, datasets))),
            "gene_raw_mean": float(np.mean(equal_dataset_values(gene, datasets))),
            "dataset_mean": float(np.mean(equal_dataset_values(dataset, datasets))),
            "deployable_route": float(np.mean(equal_dataset_values(route, datasets))),
            "oracle_route": float(np.mean(equal_dataset_values(oracle_route, datasets))),
        },
        "deltas": {
            "deployable_route_vs_gene": bootstrap_delta(route, gene, datasets),
            "deployable_route_vs_dataset": bootstrap_delta(route, dataset, datasets),
            "deployable_route_vs_anchor": bootstrap_delta(route, anchor, datasets),
            "oracle_route_vs_gene": bootstrap_delta(oracle_route, gene, datasets),
        },
        "dataset_deltas": dataset_deltas,
        "material_harm_datasets": material_harm_datasets,
    }


def fmt(value: float | None, nd: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and not math.isfinite(value):
        return "NA"
    return f"{value:+.{nd}f}"


def main() -> None:
    rows = []
    with INPUT_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {
                "group": raw["group"],
                "dataset": raw["dataset"],
                "condition": raw["condition"],
                "gene": raw["gene"],
            }
            for key in FEATURES + [TARGET] + BASELINES:
                row[key] = parse_float(raw[key])
            rows.append(row)

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)

    group_summaries = {group: summarize_group(group_rows) for group, group_rows in by_group.items()}
    pass_groups = []
    gate_reasons = []
    for group, summary in group_summaries.items():
        model = summary["lodo_model"]
        route_delta = summary["deltas"]["deployable_route_vs_gene"]
        corr_ok = (
            (model["spearman"] is not None and abs(model["spearman"]) >= 0.35)
            or (model["r2"] is not None and model["r2"] >= 0.15)
        )
        route_ok = route_delta["delta"] >= 0.02 and route_delta["p_harm"] <= 0.20
        harm_ok = len(summary["material_harm_datasets"]) == 0
        if corr_ok and route_ok and harm_ok:
            pass_groups.append(group)
        if not corr_ok:
            gate_reasons.append(f"{group}: deployable LODO proxy does not explain anchor-minus-gene risk")
        if not route_ok:
            gate_reasons.append(f"{group}: deployable route does not beat gene_raw_mean by +0.02 with low harm")
        if not harm_ok:
            gate_reasons.append(f"{group}: deployable route has dataset-level material harms")

    decision = {
        "status": "cpu_proxy_gate_fail_close_residual_forensics_as_oracle_only"
        if len(pass_groups) != len(group_summaries)
        else "cpu_proxy_gate_pass_candidate_for_separate_protocol",
        "recommended_action": "do_not_launch_gpu_from_residual_forensics; consider cross-latent upper-bound audit or new source evidence"
        if len(pass_groups) != len(group_summaries)
        else "write_separate_frozen_cpu_routing_protocol_before_any_gpu",
        "pass_groups": pass_groups,
        "gate_reasons": gate_reasons,
        "gate": {
            "prediction": "leave-one-dataset-out ridge using deployable features only",
            "correlation": "abs(Spearman)>=0.35 or R2>=0.15 in both Track A groups",
            "route": "deployable route equal-dataset score delta vs gene_raw_mean >= +0.02 with p_harm<=0.20",
            "harm": "no dataset-level route-minus-gene delta < -0.02",
        },
    }

    output = {
        "input_csv": str(INPUT_CSV),
        "leakage_status": "train_only_internal_val_condition_table_no_canonical_no_query",
        "deployable_features": FEATURES,
        "nondeployable_oracle_features_excluded": [
            "gene_target_cosine",
            "dataset_target_cosine",
            "target_residual_norm",
            "anchor_mmd_clamped",
            "anchor_pearson_pert as a selector feature",
            "gene_minus_dataset_score",
        ],
        "n_rows": len(rows),
        "groups": group_summaries,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM xverse Track A Deployable Proxy Gate",
        "",
        f"Status: `{decision['status']}`",
        f"Recommended action: `{decision['recommended_action']}`",
        "",
        "## Provenance",
        "",
        f"- input CSV: `{INPUT_CSV}`",
        f"- output JSON: `{OUT_JSON}`",
        "- leakage status: `train_only_internal_val_condition_table_no_canonical_no_query`",
        f"- condition rows: `{len(rows)}`",
        "- target: `anchor_minus_gene_raw_mean`",
        "- deployable features: `" + "`, `".join(FEATURES) + "`",
        "- excluded oracle/non-deployable fields: `gene_target_cosine`, `dataset_target_cosine`, `target_residual_norm`, `anchor_mmd_clamped`, `gene_minus_dataset_score`",
        "",
        "## Gate",
        "",
        "- Leave-one-dataset-out ridge must explain anchor-minus-gene risk in both groups: `abs(Spearman)>=0.35` or `R2>=0.15`.",
        "- The induced deployable route chooses anchor only when predicted anchor-minus-gene is positive.",
        "- Route must beat `gene_raw_mean` by at least `+0.02` equal-dataset score in both groups with `p_harm<=0.20`.",
        "- No dataset may have route-minus-gene delta `<-0.02`.",
        "",
        "## Group Summary",
        "",
        "| group | n | LODO Spearman | LODO R2 | route score | gene score | route-gene delta | p_harm | oracle-gene delta | harmed datasets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, summary in group_summaries.items():
        model = summary["lodo_model"]
        means = summary["means"]
        route_delta = summary["deltas"]["deployable_route_vs_gene"]
        oracle_delta = summary["deltas"]["oracle_route_vs_gene"]
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(summary["n"]),
                    fmt(model["spearman"]),
                    fmt(model["r2"]),
                    fmt(means["deployable_route"]),
                    fmt(means["gene_raw_mean"]),
                    fmt(route_delta["delta"]),
                    fmt(route_delta["p_harm"]),
                    fmt(oracle_delta["delta"]),
                    str(len(summary["material_harm_datasets"])),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Deployable Feature Tests", ""])
    lines.extend(
        [
            "| group | feature | Spearman to anchor-gene | linear R2 |",
            "|---|---|---:|---:|",
        ]
    )
    for group, summary in group_summaries.items():
        for row in summary["feature_tests"]:
            lines.append(
                f"| {group} | `{row['feature']}` | {fmt(row['spearman_to_anchor_minus_gene'])} | {fmt(row['linear_r2_to_anchor_minus_gene'])} |"
            )
    lines.extend(["", "## Gate Reasons", ""])
    for reason in gate_reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The residual-forensics signal is real as an explanatory/oracle target-residual geometry signal, but the tested deployable proxies do not carry enough leave-dataset-out predictive information.",
            "- This closes the residual-forensics branch for GPU promotion unless a new deployable information source is introduced and passes a separate train-only gate.",
            "- No canonical test, canonical multi, or Track C query evidence is used.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    print(decision["status"])


if __name__ == "__main__":
    main()
