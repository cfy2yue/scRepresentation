#!/usr/bin/env python3
"""Train-only Track A CPU gate for OmniPath TF-target prior features.

This gate uses only xverse train-only/internal proxy rows and the frozen
OmniPath TF-target prior. It tests whether directed TF/target degree and sign
features predict when the xverse anchor should be routed instead of the
train-only gene baseline. It does not read canonical tests, canonical multi,
held-out query artifacts, active logs, or GPU artifacts.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
PRIOR_DIR = ROOT / "dataset" / "external_priors" / "omnipath_tf_20260623"

PRIOR_SUMMARY = PRIOR_DIR / "omnipath_tf_prior_summary.json"
PRIOR_FEATURES = PRIOR_DIR / "omnipath_tf_target_gene_features.tsv"
XVERSE_ROWS = REPORTS / "latentfm_xverse_tracka_residual_forensics_20260622.json"
OUT_JSON = REPORTS / "latentfm_tracka_omnipath_tf_prior_metric_gate_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKA_OMNIPATH_TF_PRIOR_METRIC_GATE_20260623.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
BOOT_N = 2000
SEED = 20260623
RIDGE_ALPHA = 10.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_gene_features(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = str(row["gene"]).upper()
            vals = {k: as_float(v) for k, v in row.items() if k != "gene"}
            tf_total = vals.get("tf_out_degree", 0.0)
            target_total = vals.get("target_in_degree", 0.0)
            vals["tf_signed_balance"] = (
                (vals.get("tf_activation_out_degree", 0.0) - vals.get("tf_inhibition_out_degree", 0.0)) / max(tf_total, 1.0)
            )
            vals["target_signed_balance"] = (
                (vals.get("target_activation_in_degree", 0.0) - vals.get("target_inhibition_in_degree", 0.0)) / max(target_total, 1.0)
            )
            out[gene] = vals
    return out


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
    if int(mask.sum()) < 3:
        return None
    xr = rankdata_average(x[mask])
    yr = rankdata_average(y[mask])
    if float(np.std(xr)) <= 1e-12 or float(np.std(yr)) <= 1e-12:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def r2_score(y: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(y) & np.isfinite(pred)
    if int(mask.sum()) < 3:
        return None
    yy = y[mask]
    pp = pred[mask]
    denom = float(np.sum((yy - yy.mean()) ** 2))
    if denom <= 1e-12:
        return None
    return float(1.0 - np.sum((yy - pp) ** 2) / denom)


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
            arr = by_ds[str(ds)]
            vals.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        boot.append(float(np.mean(vals)))
    arr = np.asarray(boot, dtype=float)
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "dataset_deltas": point_by_ds,
        "dataset_min": float(min(point_by_ds.values())) if point_by_ds else float("nan"),
    }


def feature_vector(vals: dict[str, float] | None) -> np.ndarray:
    if vals is None:
        return np.zeros(10, dtype=float)
    raw = [
        vals.get("tf_out_degree", 0.0),
        vals.get("target_in_degree", 0.0),
        vals.get("tf_activation_out_degree", 0.0),
        vals.get("tf_inhibition_out_degree", 0.0),
        vals.get("target_activation_in_degree", 0.0),
        vals.get("target_inhibition_in_degree", 0.0),
    ]
    log_feats = [np.log1p(max(v, 0.0)) for v in raw]
    return np.asarray(
        [
            *log_feats,
            vals.get("tf_signed_balance", 0.0),
            vals.get("target_signed_balance", 0.0),
            float(vals.get("tf_out_degree", 0.0) > 0.0),
            float(vals.get("target_in_degree", 0.0) > 0.0),
        ],
        dtype=float,
    )


def build_feature_matrix(
    rows: list[dict[str, Any]],
    gene_features: dict[str, dict[str, float]],
    *,
    shuffled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    genes = [str(row["gene"]).upper() for row in rows]
    unique = sorted(set(genes))
    if shuffled:
        rng = np.random.default_rng(SEED + 31)
        shuffled_genes = list(unique)
        rng.shuffle(shuffled_genes)
        gene_map = dict(zip(unique, shuffled_genes))
    else:
        gene_map = {g: g for g in unique}
    X = []
    covered = 0
    for gene in genes:
        mapped = gene_map.get(gene, gene)
        vals = gene_features.get(mapped)
        covered += int(vals is not None)
        X.append(feature_vector(vals))
    return np.vstack(X), {
        "n_rows": len(rows),
        "n_unique_row_genes": len(unique),
        "n_covered_rows": covered,
        "coverage_fraction": float(covered / len(rows)) if rows else 0.0,
        "n_features": int(len(X[0])) if X else 0,
        "shuffled": shuffled,
    }


def lodo_ridge(rows: list[dict[str, Any]], X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray([as_float(row["anchor_minus_gene_raw_mean"]) for row in rows], dtype=float)
    pred = np.full(len(rows), np.nan, dtype=float)
    folds = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        test_mask = np.asarray([str(row["dataset"]) == ds for row in rows], dtype=bool)
        train_mask = ~test_mask
        y_train = y[train_mask]
        finite = np.isfinite(y_train)
        if int(finite.sum()) < 5:
            pred[test_mask] = float(np.nanmean(y_train))
            folds.append({"dataset": ds, "status": "fallback", "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())})
            continue
        Xtr0 = X[train_mask][finite]
        Xte0 = X[test_mask]
        ytr = y_train[finite]
        mean = Xtr0.mean(axis=0)
        std = Xtr0.std(axis=0)
        keep = std > 1e-12
        if int(keep.sum()) == 0:
            pred[test_mask] = float(np.mean(ytr))
            folds.append({"dataset": ds, "status": "constant_features", "n_train": int(finite.sum()), "n_test": int(test_mask.sum())})
            continue
        Xtr = (Xtr0[:, keep] - mean[keep]) / std[keep]
        Xte = (Xte0[:, keep] - mean[keep]) / std[keep]
        y_mean = float(np.mean(ytr))
        yc = ytr - y_mean
        coef = np.linalg.solve(Xtr.T @ Xtr + RIDGE_ALPHA * np.eye(Xtr.shape[1]), Xtr.T @ yc)
        pred[test_mask] = Xte @ coef + y_mean
        folds.append({"dataset": ds, "status": "ridge", "n_train": int(finite.sum()), "n_test": int(test_mask.sum()), "n_features": int(keep.sum())})
    return pred, {"folds": folds, "spearman": spearman(y, pred), "r2": r2_score(y, pred)}


def evaluate_group(all_rows: list[dict[str, Any]], gene_features: dict[str, dict[str, float]], group: str) -> dict[str, Any]:
    rows = [dict(row) for row in all_rows if row.get("group") == group]
    X, meta = build_feature_matrix(rows, gene_features, shuffled=False)
    pred, lodo = lodo_ridge(rows, X)
    Xs, smeta = build_feature_matrix(rows, gene_features, shuffled=True)
    spred, slodo = lodo_ridge(rows, Xs)
    scored = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["omnipath_tf_pred_anchor_minus_gene"] = float(pred[i])
        item["omnipath_tf_shuffled_pred_anchor_minus_gene"] = float(spred[i])
        item["omnipath_tf_routed_xverse_or_gene"] = as_float(row["anchor_pearson_pert"]) if pred[i] > 0 else as_float(row["gene_raw_mean"])
        item["omnipath_tf_shuffled_routed_xverse_or_gene"] = as_float(row["anchor_pearson_pert"]) if spred[i] > 0 else as_float(row["gene_raw_mean"])
        scored.append(item)
    paired = [
        paired_bootstrap(scored, "omnipath_tf_routed_xverse_or_gene", baseline, seed=SEED + i)
        for i, baseline in enumerate(("gene_raw_mean", "dataset_mean", "global_mean", "anchor_pearson_pert", "omnipath_tf_shuffled_routed_xverse_or_gene"))
    ]
    return {
        "group": group,
        "feature_meta": meta,
        "shuffled_feature_meta": smeta,
        "lodo": lodo,
        "shuffled_lodo": slodo,
        "paired_deltas": paired,
    }


def paired_row(result: dict[str, Any], baseline: str) -> dict[str, Any]:
    return next(row for row in result["paired_deltas"] if row["baseline"] == baseline)


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    for result in results:
        group = result["group"]
        if result["feature_meta"]["coverage_fraction"] < 0.80:
            reasons.append(f"{group}_coverage_below_0p80")
        if (result["lodo"].get("spearman") or 0.0) < 0.20 and (result["lodo"].get("r2") or -999.0) < 0.05:
            reasons.append(f"{group}_predictive_signal_below_gate")
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_shuf = paired_row(result, "omnipath_tf_shuffled_routed_xverse_or_gene")
        if float(vs_gene["delta_mean"]) < 0.02:
            reasons.append(f"{group}_delta_vs_gene_below_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_harm_vs_gene_above_0p20")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_shuf["delta_mean"]) < 0.02:
            reasons.append(f"{group}_shuffled_control_not_separated")
    status = "tracka_omnipath_tf_prior_metric_gate_pass_code_gate_next_no_gpu" if not reasons else "tracka_omnipath_tf_prior_metric_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "code_gate_only_if_pass_else_none",
        "reasons": reasons,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A OmniPath TF Prior Metric Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses only OmniPath TF-target prior and xverse train-only/internal proxy residual rows.",
        "- Does not read canonical test, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- Passing would authorize only a later code/provenance gate, not immediate GPU.",
        "",
        "## Prior",
        "",
        f"- source report: `{payload['prior']['acquisition_report']}`",
        f"- raw TSV SHA256: `{payload['prior']['raw_tsv_sha256']}`",
        f"- deduplicated edges: `{payload['prior']['deduplicated_edges']}`",
        f"- genes with features: `{payload['prior']['genes_with_features']}`",
        "",
        "## Group Results",
        "",
        "| group | coverage | features | LODO Spearman | LODO R2 | delta vs gene | p harm | dataset min | delta vs shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_shuf = paired_row(result, "omnipath_tf_shuffled_routed_xverse_or_gene")
        lines.append(
            f"| {result['group']} | {result['feature_meta']['coverage_fraction']:.3f} | "
            f"{result['feature_meta']['n_features']} | {fmt(result['lodo'].get('spearman'))} | "
            f"{fmt(result['lodo'].get('r2'))} | {fmt(vs_gene['delta_mean'])} | "
            f"{fmt(vs_gene['p_harm'])} | {fmt(vs_gene['dataset_min'])} | {fmt(vs_shuf['delta_mean'])} |"
        )
    lines.extend(["", "## Gate Reasons", ""])
    reasons = payload["decision"].get("reasons") or []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Decision", "", payload["decision_text"], ""])
    return "\n".join(lines)


def main() -> int:
    prior_summary = load_json(PRIOR_SUMMARY)
    rows_payload = load_json(XVERSE_ROWS)
    all_rows = rows_payload["condition_rows"]
    gene_features = load_gene_features(PRIOR_FEATURES)
    results = [evaluate_group(all_rows, gene_features, group) for group in GROUPS]
    decision = decide(results)
    payload = {
        "status": decision["status"],
        "timestamp": "2026-06-23 13:35 CST",
        "boundary": {
            "query_free": True,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_authorization": "none",
            "selection_or_tuning": False,
        },
        "prior": {
            "acquisition_report": str(REPORTS / "LATENTFM_OMNIPATH_TF_PRIOR_ACQUISITION_20260623.md"),
            "raw_tsv_sha256": prior_summary["hashes"]["raw_tsv"],
            "gene_features_sha256": prior_summary["hashes"]["gene_features_tsv"],
            "deduplicated_edges": prior_summary["deduplicated_edges"],
            "genes_with_features": prior_summary["genes_with_features"],
        },
        "results": results,
        "decision": decision,
        "decision_text": (
            "The OmniPath TF prior metric gate may proceed only if both internal proxy groups pass coverage, "
            "predictive-signal, improvement, no-harm, dataset-min, and shuffled-control gates. A failure keeps "
            "this directed regulatory prior as a hashed input/negative result and does not authorize GPU."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
