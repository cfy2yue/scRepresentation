#!/usr/bin/env python3
"""Train-only Track A CPU gate for the GOA human external prior.

This gate uses only the xverse train-only/internal proxy rows and the frozen
GOA human prior. It asks whether GO term membership can predict when the xverse
anchor should be trusted over the train-only gene baseline. It does not read
canonical tests, canonical multi, held-out query artifacts, active logs, or GPU.
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
PRIOR_DIR = ROOT / "dataset" / "external_priors" / "goa_human_20260519"

GOA_SUMMARY = PRIOR_DIR / "goa_human_prior_summary.json"
GOA_GENE_TERMS = PRIOR_DIR / "goa_human_gene_terms.tsv"
XVERSE_ROWS = REPORTS / "latentfm_xverse_tracka_residual_forensics_20260622.json"
TRACKA_INVENTORY = REPORTS / "latentfm_tracka_external_source_prior_inventory_20260623.json"

OUT_JSON = REPORTS / "latentfm_tracka_goa_prior_metric_gate_20260623.json"
OUT_MD = REPORTS / "LATENTFM_TRACKA_GOA_PRIOR_METRIC_GATE_20260623.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
BOOT_N = 2000
SEED = 20260623
RIDGE_ALPHA = 10.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gene_terms(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            terms = set(filter(None, str(row.get("go_terms") or "").split(";")))
            if terms:
                out[str(row["gene"]).upper()] = terms
    return out


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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


def build_feature_matrix(rows: list[dict[str, Any]], gene_terms: dict[str, set[str]], *, shuffled: bool) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    row_genes = [str(row["gene"]).upper() for row in rows]
    source_genes = sorted(set(row_genes))
    feature_terms = sorted(set().union(*(gene_terms.get(g, set()) for g in source_genes)))
    if shuffled:
        rng = np.random.default_rng(SEED + 17)
        shuffled_source = list(source_genes)
        rng.shuffle(shuffled_source)
        gene_map = dict(zip(source_genes, shuffled_source))
    else:
        gene_map = {g: g for g in source_genes}

    raw = np.zeros((len(rows), len(feature_terms)), dtype=float)
    term_to_idx = {term: i for i, term in enumerate(feature_terms)}
    covered = 0
    for i, gene in enumerate(row_genes):
        mapped = gene_map.get(gene, gene)
        terms = gene_terms.get(mapped, set())
        if terms:
            covered += 1
        for term in terms:
            j = term_to_idx.get(term)
            if j is not None:
                raw[i, j] = 1.0
    freq = raw.sum(axis=0)
    keep = np.where((freq >= 2) & (freq <= max(2, len(rows) - 2)))[0]
    X = raw[:, keep]
    kept_terms = [feature_terms[int(j)] for j in keep]
    meta = {
        "n_rows": len(rows),
        "n_unique_row_genes": len(source_genes),
        "n_covered_rows": covered,
        "coverage_fraction": float(covered / len(rows)) if rows else 0.0,
        "n_raw_terms": len(feature_terms),
        "n_kept_terms": len(kept_terms),
        "shuffled": shuffled,
    }
    return X, kept_terms, meta


def lodo_kernel_ridge(rows: list[dict[str, Any]], X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray([as_float(row["anchor_minus_gene_raw_mean"]) for row in rows], dtype=float)
    pred = np.full(len(rows), np.nan, dtype=float)
    datasets = sorted({str(row["dataset"]) for row in rows})
    folds = []
    for ds in datasets:
        test_mask = np.asarray([str(row["dataset"]) == ds for row in rows], dtype=bool)
        train_mask = ~test_mask
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        finite_train = np.isfinite(y_train)
        if X_train.shape[1] == 0 or int(finite_train.sum()) < 5:
            fallback = float(np.nanmean(y_train))
            pred[test_mask] = fallback
            folds.append({"dataset": ds, "status": "fallback", "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())})
            continue
        Xtr = X_train[finite_train]
        ytr = y_train[finite_train]
        mean = Xtr.mean(axis=0)
        std = Xtr.std(axis=0)
        keep = std > 1e-12
        if int(keep.sum()) == 0:
            fallback = float(np.mean(ytr))
            pred[test_mask] = fallback
            folds.append({"dataset": ds, "status": "constant_features", "n_train": int(finite_train.sum()), "n_test": int(test_mask.sum())})
            continue
        Xtr = (Xtr[:, keep] - mean[keep]) / std[keep]
        Xte = (X_test[:, keep] - mean[keep]) / std[keep]
        y_mean = float(np.mean(ytr))
        yc = ytr - y_mean
        K = Xtr @ Xtr.T
        coef = np.linalg.solve(K + RIDGE_ALPHA * np.eye(K.shape[0]), yc)
        pred[test_mask] = Xte @ Xtr.T @ coef + y_mean
        folds.append({"dataset": ds, "status": "kernel_ridge", "n_train": int(finite_train.sum()), "n_test": int(test_mask.sum()), "n_features": int(keep.sum())})
    return pred, {"folds": folds, "spearman": spearman(y, pred), "r2": r2_score(y, pred)}


def evaluate_group(all_rows: list[dict[str, Any]], gene_terms: dict[str, set[str]], group: str) -> dict[str, Any]:
    base_rows = [dict(row) for row in all_rows if row.get("group") == group]
    X, terms, feature_meta = build_feature_matrix(base_rows, gene_terms, shuffled=False)
    pred, lodo = lodo_kernel_ridge(base_rows, X)
    Xs, _, shuffled_meta = build_feature_matrix(base_rows, gene_terms, shuffled=True)
    spred, slodo = lodo_kernel_ridge(base_rows, Xs)
    scored = []
    for i, row in enumerate(base_rows):
        item = dict(row)
        item["goa_pred_anchor_minus_gene"] = float(pred[i])
        item["goa_shuffled_pred_anchor_minus_gene"] = float(spred[i])
        item["goa_routed_xverse_or_gene"] = as_float(row["anchor_pearson_pert"]) if pred[i] > 0 else as_float(row["gene_raw_mean"])
        item["goa_shuffled_routed_xverse_or_gene"] = as_float(row["anchor_pearson_pert"]) if spred[i] > 0 else as_float(row["gene_raw_mean"])
        scored.append(item)
    paired = [
        paired_bootstrap(scored, "goa_routed_xverse_or_gene", baseline, seed=SEED + i)
        for i, baseline in enumerate(("gene_raw_mean", "dataset_mean", "global_mean", "anchor_pearson_pert", "goa_shuffled_routed_xverse_or_gene"))
    ]
    return {
        "group": group,
        "feature_meta": feature_meta,
        "shuffled_feature_meta": shuffled_meta,
        "kept_terms_preview": terms[:20],
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
        vs_shuffled = paired_row(result, "goa_shuffled_routed_xverse_or_gene")
        if float(vs_gene["delta_mean"]) < 0.02:
            reasons.append(f"{group}_delta_vs_gene_below_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_harm_vs_gene_above_0p20")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_shuffled["delta_mean"]) < 0.02:
            reasons.append(f"{group}_shuffled_control_not_separated")
    status = "tracka_goa_prior_metric_gate_pass_code_gate_next_no_gpu" if not reasons else "tracka_goa_prior_metric_gate_fail_no_gpu"
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
        "# Track A GOA Prior Metric Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses only GOA human prior and xverse train-only/internal proxy residual rows.",
        "- Does not read canonical test, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- Passing would authorize only a later code/provenance gate, not immediate GPU.",
        "",
        "## Prior",
        "",
        f"- source report: `{payload['prior']['acquisition_report']}`",
        f"- raw GAF SHA256: `{payload['prior']['raw_gaf_sha256']}`",
        f"- genes with GO terms: `{payload['prior']['n_genes']}`",
        f"- GO terms: `{payload['prior']['n_go_terms']}`",
        "",
        "## Group Results",
        "",
        "| group | coverage | kept terms | LODO Spearman | LODO R2 | delta vs gene | p harm | dataset min | delta vs shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_shuf = paired_row(result, "goa_shuffled_routed_xverse_or_gene")
        lines.append(
            f"| {result['group']} | {result['feature_meta']['coverage_fraction']:.3f} | "
            f"{result['feature_meta']['n_kept_terms']} | {fmt(result['lodo'].get('spearman'))} | "
            f"{fmt(result['lodo'].get('r2'))} | {fmt(vs_gene['delta_mean'])} | "
            f"{fmt(vs_gene['p_harm'])} | {fmt(vs_gene['dataset_min'])} | {fmt(vs_shuf['delta_mean'])} |"
        )
    lines.extend(["", "## Gate Reasons", ""])
    reasons = payload["decision"].get("reasons") or []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines.extend(["", "## Decision", "", payload["decision_text"], ""])
    return "\n".join(lines)


def main() -> int:
    prior_summary = load_json(GOA_SUMMARY)
    inventory = load_json(TRACKA_INVENTORY)
    rows_payload = load_json(XVERSE_ROWS)
    all_rows = rows_payload["condition_rows"]
    gene_terms = load_gene_terms(GOA_GENE_TERMS)
    results = [evaluate_group(all_rows, gene_terms, group) for group in GROUPS]
    decision = decide(results)
    payload = {
        "status": decision["status"],
        "timestamp": "2026-06-23 13:05 CST",
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
            "acquisition_report": str(REPORTS / "LATENTFM_GOA_HUMAN_PRIOR_ACQUISITION_20260623.md"),
            "inventory_status_before_acquisition": inventory.get("status"),
            "raw_gaf_sha256": prior_summary["hashes"]["raw_gaf_gz"],
            "gene_terms_sha256": prior_summary["hashes"]["gene_terms_tsv"],
            "n_genes": prior_summary["n_genes"],
            "n_go_terms": prior_summary["n_go_terms"],
        },
        "results": results,
        "decision": decision,
        "decision_text": (
            "The GOA prior metric gate may proceed only if both internal proxy groups pass coverage, "
            "predictive-signal, improvement, no-harm, and shuffled-control gates. A failure keeps this "
            "external prior as a hashed input/negative result and does not authorize GPU."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
