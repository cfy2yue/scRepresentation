#!/usr/bin/env python3
"""Track A xverse response-covariate learned abstain/router CPU gate.

This gate turns the train-only/internal xverse residual forensics signal into a
strict nested leave-one-dataset-out router.  For each held-out proxy dataset,
the ridge model and anchor-vs-gene threshold are selected using only the other
proxy datasets.  Canonical Track A outcomes, canonical multi, held-out query,
active logs, and GPU artifacts are not read.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
XVERSE_ROWS = REPORTS / "latentfm_xverse_tracka_residual_forensics_20260622.json"
OUT_JSON = REPORTS / "latentfm_tracka_xverse_response_covariate_router_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKA_XVERSE_RESPONSE_COVARIATE_ROUTER_GATE_20260624.md"
GROUPS = ("internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy")
FEATURES = (
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
ALPHAS = (0.01, 0.1, 1.0, 10.0)
THRESHOLDS = (-0.05, -0.02, 0.0, 0.02, 0.05)
BOOT_N = 2000
SEED = 20260624


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def matrix(rows: list[dict[str, Any]], *, shuffled: bool = False, seed: int = SEED) -> np.ndarray:
    x = np.asarray([[as_float(row.get(k)) for k in FEATURES] for row in rows], dtype=float)
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


def ridge_predict(model: np.ndarray, x: np.ndarray) -> np.ndarray:
    n_feat = len(FEATURES)
    intercept = model[0]
    coef = model[1 : 1 + n_feat]
    mu = model[1 + n_feat : 1 + 2 * n_feat]
    sd = model[1 + 2 * n_feat :]
    return intercept + ((x - mu) / sd) @ coef


def score_policy(rows: list[dict[str, Any]], pred: np.ndarray, threshold: float, *, label: str) -> list[dict[str, Any]]:
    out = []
    for row, p in zip(rows, pred, strict=True):
        use_anchor = float(p) >= float(threshold)
        item = dict(row)
        item[f"{label}_pred_anchor_minus_gene"] = float(p)
        item[f"{label}_use_anchor"] = bool(use_anchor)
        item[f"{label}_anchor_or_gene"] = as_float(row["anchor_pearson_pert"]) if use_anchor else as_float(row["gene_raw_mean"])
        out.append(item)
    return out


def paired_delta_no_boot(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(as_float(row[candidate]) - as_float(row[baseline]))
    return {ds: float(np.mean(vals)) for ds, vals in by_ds.items() if vals}


def inner_select(rows: list[dict[str, Any]], x: np.ndarray, train_idx: np.ndarray) -> tuple[float, float]:
    train_rows = [rows[int(i)] for i in train_idx]
    train_ds = sorted({str(row["dataset"]) for row in train_rows})
    y = np.asarray([as_float(rows[int(i)]["anchor_minus_gene_raw_mean"]) for i in train_idx], dtype=float)
    best = None
    for alpha in ALPHAS:
        pred_train = np.full(len(train_idx), np.nan, dtype=float)
        for ds in train_ds:
            inner_fit = np.asarray([i for i, idx in enumerate(train_idx) if str(rows[int(idx)]["dataset"]) != ds], dtype=int)
            inner_val = np.asarray([i for i, idx in enumerate(train_idx) if str(rows[int(idx)]["dataset"]) == ds], dtype=int)
            if len(inner_fit) < 5 or len(inner_val) == 0:
                continue
            model = ridge_fit(x[train_idx[inner_fit]], y[inner_fit], alpha)
            pred_train[inner_val] = ridge_predict(model, x[train_idx[inner_val]])
        if not np.isfinite(pred_train).all():
            model = ridge_fit(x[train_idx], y, alpha)
            pred_train = ridge_predict(model, x[train_idx])
        for threshold in THRESHOLDS:
            scored = score_policy(train_rows, pred_train, threshold, label="inner")
            ds_delta = paired_delta_no_boot(scored, "inner_anchor_or_gene", "gene_raw_mean")
            ds_anchor = paired_delta_no_boot(scored, "inner_anchor_or_gene", "anchor_pearson_pert")
            mean_delta = float(np.mean(list(ds_delta.values()))) if ds_delta else -999.0
            min_delta = float(min(ds_delta.values())) if ds_delta else -999.0
            mean_anchor = float(np.mean(list(ds_anchor.values()))) if ds_anchor else -999.0
            harm_frac = float(np.mean([v < 0.0 for v in ds_delta.values()])) if ds_delta else 1.0
            key = (min_delta >= -0.02, harm_frac <= 0.25, mean_delta, mean_anchor, -abs(threshold), -alpha)
            if best is None or key > best[0]:
                best = (key, alpha, threshold)
    if best is None:
        return 1.0, 0.0
    return float(best[1]), float(best[2])


def nested_oof(rows: list[dict[str, Any]], *, shuffled: bool = False, seed: int = SEED) -> dict[str, Any]:
    x = matrix(rows, shuffled=shuffled, seed=seed)
    y = np.asarray([as_float(row["anchor_minus_gene_raw_mean"]) for row in rows], dtype=float)
    pred = np.full(len(rows), np.nan, dtype=float)
    specs = {}
    datasets = sorted({str(row["dataset"]) for row in rows})
    for ds in datasets:
        train_idx = np.asarray([i for i, row in enumerate(rows) if str(row["dataset"]) != ds], dtype=int)
        val_idx = np.asarray([i for i, row in enumerate(rows) if str(row["dataset"]) == ds], dtype=int)
        alpha, threshold = inner_select(rows, x, train_idx)
        model = ridge_fit(x[train_idx], y[train_idx], alpha)
        pred[val_idx] = ridge_predict(model, x[val_idx])
        specs[ds] = {"alpha": alpha, "threshold": threshold, "n_train": int(len(train_idx)), "n_val": int(len(val_idx))}
    scored = []
    for ds in datasets:
        idx = [i for i, row in enumerate(rows) if str(row["dataset"]) == ds]
        threshold = specs[ds]["threshold"]
        ds_scored = score_policy([rows[i] for i in idx], pred[idx], threshold, label="router")
        scored.extend(ds_scored)
    return {"rows": scored, "specs_by_heldout_dataset": specs}


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, *, seed: int) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        cand = as_float(row.get(candidate))
        base = as_float(row.get(baseline))
        if np.isfinite(cand) and np.isfinite(base):
            by_ds[str(row["dataset"])].append(float(cand - base))
    keys = sorted(ds for ds, vals in by_ds.items() if vals)
    point_by_ds = {ds: float(np.mean(by_ds[ds])) for ds in keys}
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
    return {"candidate": candidate, "baseline": baseline, "delta_mean": point, "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))], "p_improve": float(np.mean(arr > 0.0)), "p_harm": float(np.mean(arr < 0.0)), "dataset_deltas": point_by_ds, "dataset_min": float(min(point_by_ds.values())) if point_by_ds else float("nan")}


def evaluate_group(all_rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    rows = [dict(row) for row in all_rows if row.get("group") == group]
    real = nested_oof(rows, shuffled=False, seed=SEED)
    shuf = nested_oof(rows, shuffled=True, seed=SEED + 77)
    scored = real["rows"]
    shuffled_rows = shuf["rows"]
    by_key = {(row["dataset"], row["condition"]): row for row in shuffled_rows}
    for row in scored:
        other = by_key[(row["dataset"], row["condition"])]
        row["shuffled_router_anchor_or_gene"] = other["router_anchor_or_gene"]
        row["shuffled_router_use_anchor"] = other["router_use_anchor"]
        row["shuffled_router_pred_anchor_minus_gene"] = other["router_pred_anchor_minus_gene"]
    paired = [paired_bootstrap(scored, "router_anchor_or_gene", baseline, seed=SEED + i) for i, baseline in enumerate(("gene_raw_mean", "anchor_pearson_pert", "dataset_mean", "global_mean", "shuffled_router_anchor_or_gene"))]
    return {
        "group": group,
        "n_rows": len(scored),
        "features": FEATURES,
        "use_anchor_fraction": float(np.mean([row["router_use_anchor"] for row in scored])) if scored else 0.0,
        "shuffled_use_anchor_fraction": float(np.mean([row["shuffled_router_use_anchor"] for row in scored])) if scored else 0.0,
        "paired_deltas": paired,
        "specs_by_heldout_dataset": real["specs_by_heldout_dataset"],
        "shuffled_specs_by_heldout_dataset": shuf["specs_by_heldout_dataset"],
        "scored_rows": scored,
    }


def paired_row(result: dict[str, Any], baseline: str) -> dict[str, Any]:
    return next(row for row in result["paired_deltas"] if row["baseline"] == baseline)


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    for result in results:
        group = result["group"]
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        vs_shuf = paired_row(result, "shuffled_router_anchor_or_gene")
        if float(result["use_anchor_fraction"]) < 0.05:
            reasons.append(f"{group}_uses_anchor_too_rarely")
        if float(result["use_anchor_fraction"]) > 0.95:
            reasons.append(f"{group}_uses_anchor_too_often")
        if float(vs_gene["delta_mean"]) < 0.02:
            reasons.append(f"{group}_delta_vs_gene_below_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_harm_vs_gene_above_0p20")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_anchor["delta_mean"]) < -0.005:
            reasons.append(f"{group}_material_loss_vs_anchor")
        if float(vs_shuf["delta_mean"]) < 0.01:
            reasons.append(f"{group}_shuffled_router_not_beaten_by_0p01")
    status = "tracka_xverse_response_covariate_router_gate_pass_code_gate_next_no_gpu" if not reasons else "tracka_xverse_response_covariate_router_gate_fail_no_gpu"
    return {"status": status, "gpu_authorization": "none", "next_authorization": "code_gate_only_if_pass_else_none", "reasons": reasons}


def render(payload: dict[str, Any]) -> str:
    lines = ["# Track A xverse Response-Covariate Router Gate", "", f"Status: `{payload['decision']['status']}`", "GPU authorization: `none`", "", "## Boundary", "", "- Uses only xverse internal proxy residual-forensics rows.", "- Nested leave-one-dataset-out: each held-out dataset uses ridge alpha and threshold selected only from other datasets.", "- Does not read canonical Track A outcomes, canonical multi, held-out query, active logs, or GPU artifacts.", "", "## Results", "", "| group | use anchor | delta vs gene | p harm | dataset min | delta vs anchor | delta vs shuffled |", "|---|---:|---:|---:|---:|---:|---:|"]
    for result in payload["results"]:
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        vs_shuf = paired_row(result, "shuffled_router_anchor_or_gene")
        lines.append(f"| {result['group']} | {result['use_anchor_fraction']:.3f} | {fmt(vs_gene['delta_mean'])} | {fmt(vs_gene['p_harm'])} | {fmt(vs_gene['dataset_min'])} | {fmt(vs_anchor['delta_mean'])} | {fmt(vs_shuf['delta_mean'])} |")
    lines.extend(["", "## Gate Reasons", ""])
    lines.extend([f"- `{r}`" for r in payload["decision"].get("reasons", [])] or ["- none"])
    lines.extend(["", "## Feature Set", "", ", ".join(f"`{x}`" for x in FEATURES), ""])
    return "\n".join(lines)


def main() -> int:
    source = load_json(XVERSE_ROWS)
    rows = source["condition_rows"]
    results = [evaluate_group(rows, group) for group in GROUPS]
    decision = decide(results)
    payload = {
        "status": decision["status"],
        "timestamp": "2026-06-24 00:35 CST",
        "inputs": {"xverse_residual_forensics": str(XVERSE_ROWS)},
        "boundary": {"canonical_test_read": False, "canonical_multi_read": False, "heldout_query_read": False, "active_log_read": False, "gpu_artifact_read": False, "nested_lodo": True},
        "features": FEATURES,
        "alphas": ALPHAS,
        "thresholds": THRESHOLDS,
        "results": results,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
