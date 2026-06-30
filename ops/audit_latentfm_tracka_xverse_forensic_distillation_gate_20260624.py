#!/usr/bin/env python3
"""CPU gate for deployable distillation of nondeployable forensic risk.

The diagnostic response-covariate router has strong internal signal but uses
target-derived features. This gate asks whether its use-anchor decisions can be
distilled, under nested leave-one-dataset-out, into deployable covariates only.
It does not launch training or read canonical/query artifacts.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import audit_latentfm_tracka_xverse_response_covariate_router_gate_20260624 as base


OUT_JSON = ROOT / "reports/latentfm_tracka_xverse_forensic_distillation_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_XVERSE_FORENSIC_DISTILLATION_GATE_20260624.md"

DEPLOYABLE_FEATURES = (
    "gene_dataset_cosine",
    "gene_pred_norm",
    "dataset_pred_norm",
    "global_pred_norm",
    "gene_train_count",
)
COUNT_FEATURES = ("gene_train_count",)
ALPHAS = (0.01, 0.1, 1.0, 10.0)
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
BOOT_N = 2000
SEED = 20260624


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def matrix(rows: list[dict[str, Any]], features: tuple[str, ...], *, shuffled: bool = False, seed: int = SEED) -> np.ndarray:
    x = np.asarray([[as_float(row.get(k)) for k in features] for row in rows], dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = np.isfinite(col)
        fill = float(np.median(col[finite])) if finite.any() else 0.0
        col[~finite] = fill
        x[:, j] = col
    if shuffled:
        rng = np.random.default_rng(seed)
        for j in range(x.shape[1]):
            x[:, j] = rng.permutation(x[:, j])
    return x


def zfit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x, axis=0)
    sd = np.std(x, axis=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    mu, sd = zfit(x)
    z = (x - mu) / sd
    design = np.c_[np.ones(len(z)), z]
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return np.r_[coef[0], coef[1:], mu, sd]


def ridge_predict(model: np.ndarray, x: np.ndarray, n_feat: int) -> np.ndarray:
    intercept = model[0]
    coef = model[1 : 1 + n_feat]
    mu = model[1 + n_feat : 1 + 2 * n_feat]
    sd = model[1 + 2 * n_feat :]
    return intercept + ((x - mu) / sd) @ coef


def full_forensics_oracle(rows: list[dict[str, Any]], group: str) -> dict[tuple[str, str], bool]:
    old_features = base.FEATURES
    old_thresholds = base.THRESHOLDS
    try:
        base.FEATURES = (
            "gene_target_cosine",
            "dataset_target_cosine",
            "target_residual_norm",
            "gene_dataset_cosine",
            "gene_minus_dataset_score",
            "gene_pred_norm",
            "dataset_pred_norm",
            "global_pred_norm",
            "gene_train_count",
        )
        base.THRESHOLDS = (-0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
        result = base.evaluate_group(rows, group)
    finally:
        base.FEATURES = old_features
        base.THRESHOLDS = old_thresholds
    return {
        (str(row["dataset"]), str(row["condition"])): bool(row["router_use_anchor"])
        for row in result["scored_rows"]
    }


def score_rows(rows: list[dict[str, Any]], pred: np.ndarray, threshold: float, *, label: str) -> list[dict[str, Any]]:
    out = []
    for row, p in zip(rows, pred, strict=True):
        use_anchor = float(p) >= float(threshold)
        item = dict(row)
        item[f"{label}_pred_oracle_use_anchor"] = float(p)
        item[f"{label}_use_anchor"] = bool(use_anchor)
        item[f"{label}_anchor_or_gene"] = as_float(row["anchor_pearson_pert"]) if use_anchor else as_float(row["gene_raw_mean"])
        out.append(item)
    return out


def ds_delta(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(as_float(row[candidate]) - as_float(row[baseline]))
    return {k: float(np.mean(v)) for k, v in by_ds.items() if v}


def metrics_no_boot(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, float]:
    vals = ds_delta(rows, candidate, baseline)
    use = [bool(row.get("distill_use_anchor", False)) for row in rows]
    return {
        "delta": float(np.mean(list(vals.values()))) if vals else float("nan"),
        "dataset_min": float(min(vals.values())) if vals else float("nan"),
        "harm_frac": float(np.mean([v < 0.0 for v in vals.values()])) if vals else 1.0,
        "use_anchor_fraction": float(np.mean(use)) if use else 0.0,
    }


def inner_select(rows: list[dict[str, Any]], x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, features: tuple[str, ...]) -> tuple[float, float]:
    best = None
    train_rows = [rows[int(i)] for i in train_idx]
    train_ds = sorted({str(r["dataset"]) for r in train_rows})
    for alpha in ALPHAS:
        pred = np.full(len(train_idx), np.nan, dtype=float)
        for ds in train_ds:
            fit_local = np.asarray([i for i, idx in enumerate(train_idx) if str(rows[int(idx)]["dataset"]) != ds], dtype=int)
            val_local = np.asarray([i for i, idx in enumerate(train_idx) if str(rows[int(idx)]["dataset"]) == ds], dtype=int)
            if len(fit_local) < 8 or len(val_local) == 0:
                continue
            model = ridge_fit(x[train_idx[fit_local]], y[train_idx[fit_local]], alpha)
            pred[val_local] = ridge_predict(model, x[train_idx[val_local]], len(features))
        if not np.isfinite(pred).all():
            model = ridge_fit(x[train_idx], y[train_idx], alpha)
            pred = ridge_predict(model, x[train_idx], len(features))
        for threshold in THRESHOLDS:
            scored = score_rows(train_rows, pred, threshold, label="distill")
            m = metrics_no_boot(scored, "distill_anchor_or_gene", "gene_raw_mean")
            key = (
                0.05 <= m["use_anchor_fraction"] <= 0.20,
                m["dataset_min"] >= -0.02,
                m["harm_frac"] <= 0.20,
                m["delta"],
                -abs(m["use_anchor_fraction"] - 0.125),
                -alpha,
            )
            if best is None or key > best[0]:
                best = (key, alpha, threshold)
    if best is None:
        return 1.0, 0.5
    return float(best[1]), float(best[2])


def nested_distill(rows: list[dict[str, Any]], oracle: dict[tuple[str, str], bool], features: tuple[str, ...], *, shuffled_features: bool = False, shuffled_labels: bool = False) -> dict[str, Any]:
    work = [dict(row) for row in rows if (str(row["dataset"]), str(row["condition"])) in oracle]
    x = matrix(work, features, shuffled=shuffled_features, seed=SEED + 31)
    y = np.asarray([1.0 if oracle[(str(row["dataset"]), str(row["condition"]))] else 0.0 for row in work], dtype=float)
    if shuffled_labels:
        rng = np.random.default_rng(SEED + 37)
        y = rng.permutation(y)
    pred = np.full(len(work), np.nan, dtype=float)
    specs = {}
    for ds in sorted({str(r["dataset"]) for r in work}):
        train_idx = np.asarray([i for i, row in enumerate(work) if str(row["dataset"]) != ds], dtype=int)
        val_idx = np.asarray([i for i, row in enumerate(work) if str(row["dataset"]) == ds], dtype=int)
        alpha, threshold = inner_select(work, x, y, train_idx, features)
        model = ridge_fit(x[train_idx], y[train_idx], alpha)
        pred[val_idx] = ridge_predict(model, x[val_idx], len(features))
        specs[ds] = {"alpha": alpha, "threshold": threshold, "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}
    scored = []
    for ds in sorted({str(r["dataset"]) for r in work}):
        idx = [i for i, row in enumerate(work) if str(row["dataset"]) == ds]
        threshold = specs[ds]["threshold"]
        scored.extend(score_rows([work[i] for i in idx], pred[idx], threshold, label="distill"))
    return {"rows": scored, "specs_by_heldout_dataset": specs}


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, *, seed: int) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(as_float(row[candidate]) - as_float(row[baseline]))
    keys = sorted(k for k, v in by_ds.items() if v)
    point_by_ds = {k: float(np.mean(by_ds[k])) for k in keys}
    point = float(np.mean(list(point_by_ds.values()))) if point_by_ds else float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(BOOT_N):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        vals = []
        for ds in sampled:
            arr = np.asarray(by_ds[str(ds)], dtype=float)
            vals.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        boot.append(float(np.mean(vals)))
    arr = np.asarray(boot, dtype=float)
    return {
        "baseline": baseline,
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_harm": float(np.mean(arr < 0.0)),
        "dataset_min": float(min(point_by_ds.values())) if point_by_ds else float("nan"),
        "dataset_harm_fraction": float(np.mean([v < 0.0 for v in point_by_ds.values()])) if point_by_ds else float("nan"),
    }


def evaluate_group(all_rows: list[dict[str, Any]], group: str, features: tuple[str, ...], control: str) -> dict[str, Any]:
    rows = [dict(row) for row in all_rows if row.get("group") == group]
    oracle = full_forensics_oracle(all_rows, group)
    dist = nested_distill(
        rows,
        oracle,
        features,
        shuffled_features=(control == "shuffled_features"),
        shuffled_labels=(control == "shuffled_labels"),
    )
    scored = dist["rows"]
    paired = [
        paired_bootstrap(scored, "distill_anchor_or_gene", baseline, seed=SEED + i)
        for i, baseline in enumerate(("gene_raw_mean", "anchor_pearson_pert", "dataset_mean", "global_mean"))
    ]
    oracle_use = [oracle[(str(row["dataset"]), str(row["condition"]))] for row in rows if (str(row["dataset"]), str(row["condition"])) in oracle]
    return {
        "group": group,
        "control": control,
        "features": list(features),
        "n_rows": len(scored),
        "oracle_use_anchor_fraction": float(np.mean(oracle_use)) if oracle_use else 0.0,
        "use_anchor_fraction": float(np.mean([row["distill_use_anchor"] for row in scored])) if scored else 0.0,
        "paired_deltas": paired,
        "specs_by_heldout_dataset": dist["specs_by_heldout_dataset"],
    }


def paired_row(result: dict[str, Any], baseline: str) -> dict[str, Any]:
    return next(row for row in result["paired_deltas"] if row["baseline"] == baseline)


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    main = {(r["group"], r["control"]): r for r in results}
    for group in base.GROUPS:
        result = main[(group, "main")]
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        if float(result["use_anchor_fraction"]) < 0.05:
            reasons.append(f"{group}_uses_anchor_too_rarely")
        if float(result["use_anchor_fraction"]) > 0.20:
            reasons.append(f"{group}_uses_anchor_too_often")
        if float(vs_gene["delta_mean"]) < 0.025:
            reasons.append(f"{group}_delta_vs_gene_below_0p025")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_p_harm_vs_gene_above_0p20")
        if float(vs_anchor["delta_mean"]) < -0.005:
            reasons.append(f"{group}_material_loss_vs_anchor")
        for control in ("shuffled_features", "shuffled_labels", "count_only"):
            crow = paired_row(main[(group, control)], "gene_raw_mean")
            if float(crow["delta_mean"]) >= float(vs_gene["delta_mean"]) - 0.005:
                reasons.append(f"{group}_{control}_not_separated")
    return {
        "status": "tracka_xverse_forensic_distillation_gate_pass_code_gate_next_no_gpu" if not reasons else "tracka_xverse_forensic_distillation_gate_fail_no_gpu",
        "gpu_authorization": "none",
        "next_authorization": "code_gate_only_if_pass_else_none",
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A xverse Forensic-Risk Distillation Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- CPU-only nested leave-one-dataset-out gate.",
        "- Trains only on xverse train-only/internal residual-forensics rows.",
        "- Held-out datasets use only deployable covariates, not target-derived forensic features.",
        "- Does not read canonical outcomes, canonical multi, Track C query, active logs, or GPU artifacts.",
        "",
        "## Gate Rule",
        "",
        "A deployable distillation must pass both internal groups with use-anchor in `[0.05, 0.20]`, delta vs gene `>= +0.025`, dataset min `>= -0.02`, p_harm `<= 0.20`, and separated shuffled/count controls.",
        "",
        "## Results",
        "",
        "| group | control | features | oracle use anchor | distill use anchor | delta vs gene | p harm | dataset min | delta vs anchor |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        lines.append(
            f"| {result['group']} | `{result['control']}` | `{'+'.join(result['features'])}` | "
            f"{result['oracle_use_anchor_fraction']:.3f} | {result['use_anchor_fraction']:.3f} | "
            f"{vs_gene['delta_mean']:+.6f} | {vs_gene['p_harm']:.3f} | {vs_gene['dataset_min']:+.6f} | "
            f"{vs_anchor['delta_mean']:+.6f} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in payload["decision"]["reasons"]] or ["- none"])
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`", ""])
    return "\n".join(lines)


def main() -> int:
    source = base.load_json(base.XVERSE_ROWS)
    rows = source["condition_rows"]
    variants = {
        "main": DEPLOYABLE_FEATURES,
        "shuffled_features": DEPLOYABLE_FEATURES,
        "shuffled_labels": DEPLOYABLE_FEATURES,
        "count_only": COUNT_FEATURES,
    }
    results = []
    for group in base.GROUPS:
        for control, features in variants.items():
            results.append(evaluate_group(rows, group, features, control))
    decision = decide(results)
    payload = {
        "status": decision["status"],
        "boundary": {
            "xverse_residual_forensics": str(base.XVERSE_ROWS),
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
            "nested_lodo": True,
        },
        "results": results,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
