#!/usr/bin/env python3
"""Train-response local reliability CPU gate for Track A xverse.

This is a deployability-oriented CPU gate.  It asks whether response reliability
can be predicted from gene-embedding neighborhoods built only from other
datasets in nested LODO, instead of using target-derived residual covariates.
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
GENE_CACHE = Path(
    __import__("os").environ.get(
        "LATENTFM_LOCAL_RELIABILITY_GENE_CACHE",
        str(ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"),
    )
)
REPORT_STEM = __import__("os").environ.get(
    "LATENTFM_LOCAL_RELIABILITY_REPORT_STEM",
    "latentfm_tracka_xverse_train_response_local_reliability_gate_20260624",
)
OUT_JSON = REPORTS / f"{REPORT_STEM}.json"
OUT_MD = REPORTS / f"{REPORT_STEM.upper()}.md"

GROUPS = ("internal_val_cross_background_seen_gene_proxy", "internal_val_family_gene_proxy")
ALPHAS = (0.01, 0.1, 1.0, 10.0)
THRESHOLDS = (-0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
K_LIST = (8, 16, 32)
BOOT_N = 2000
SEED = 20260624


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def load_gene_embeddings() -> dict[str, np.ndarray]:
    emb = np.load(GENE_CACHE / "gene_embeddings.npy")
    out: dict[str, np.ndarray] = {}
    for line in (GENE_CACHE / "gene_index.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lower().startswith("symbol"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        symbol = parts[0].upper()
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        vec = np.asarray(emb[idx], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            out[symbol] = vec / norm
    return out


def row_gene(row: dict[str, Any]) -> str:
    return str(row.get("gene") or row.get("condition") or "").split("+")[0].upper()


def make_feature_vector(
    row: dict[str, Any],
    train_rows: list[dict[str, Any]],
    gene_emb: dict[str, np.ndarray],
    *,
    shuffled: bool = False,
    seed: int = SEED,
) -> dict[str, float]:
    gene = row_gene(row)
    q = gene_emb.get(gene)
    labels = []
    sims = []
    counts = []
    ds_seen = set()
    same_gene = 0
    if q is not None:
        for tr in train_rows:
            tg = row_gene(tr)
            tv = gene_emb.get(tg)
            if tv is None:
                continue
            sim = float(np.dot(q, tv))
            labels.append(float(tr["anchor_minus_gene_raw_mean"]))
            sims.append(sim)
            counts.append(float(tr.get("gene_train_count") or 0.0))
            ds_seen.add(str(tr["dataset"]))
            if tg == gene:
                same_gene += 1
    if not labels:
        return {
            "local_mean_k8": 0.0,
            "local_abs_mean_k8": 0.0,
            "local_var_k8": 0.0,
            "local_sign_consistency_k8": 0.0,
            "local_density_k8": 0.0,
            "local_mean_k16": 0.0,
            "local_abs_mean_k16": 0.0,
            "local_var_k16": 0.0,
            "local_sign_consistency_k16": 0.0,
            "local_density_k16": 0.0,
            "local_mean_k32": 0.0,
            "local_abs_mean_k32": 0.0,
            "local_var_k32": 0.0,
            "local_sign_consistency_k32": 0.0,
            "local_density_k32": 0.0,
            "same_gene_support": 0.0,
            "mean_gene_train_count_neighbors": 0.0,
            "num_neighbor_datasets": 0.0,
            "query_gene_train_count": float(row.get("gene_train_count") or 0.0),
        }
    labels_arr = np.asarray(labels, dtype=float)
    sims_arr = np.asarray(sims, dtype=float)
    if shuffled:
        rng = np.random.default_rng(seed)
        labels_arr = rng.permutation(labels_arr)
    order = np.argsort(-sims_arr)
    features: dict[str, float] = {}
    for k in K_LIST:
        idx = order[: min(k, len(order))]
        lab = labels_arr[idx]
        sim = sims_arr[idx]
        weights = np.maximum(sim, 0.0)
        if float(weights.sum()) <= 1e-12:
            weights = np.ones_like(lab)
        weights = weights / float(weights.sum())
        mean = float(np.sum(weights * lab))
        features[f"local_mean_k{k}"] = mean
        features[f"local_abs_mean_k{k}"] = abs(mean)
        features[f"local_var_k{k}"] = float(np.sum(weights * (lab - mean) ** 2))
        features[f"local_sign_consistency_k{k}"] = abs(float(np.sum(weights * np.sign(lab))))
        features[f"local_density_k{k}"] = float(np.mean(sim))
    features["same_gene_support"] = float(same_gene)
    features["mean_gene_train_count_neighbors"] = float(np.mean(counts)) if counts else 0.0
    features["num_neighbor_datasets"] = float(len(ds_seen))
    features["query_gene_train_count"] = float(row.get("gene_train_count") or 0.0)
    return features


def feature_matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    x = np.asarray([[float(row["features"].get(name, 0.0)) for name in names] for row in rows], dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = np.isfinite(col)
        fill = float(np.median(col[finite])) if finite.any() else 0.0
        col[~finite] = fill
        x[:, j] = col
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
    n_feat = (len(model) - 1) // 3
    intercept = model[0]
    coef = model[1 : 1 + n_feat]
    mu = model[1 + n_feat : 1 + 2 * n_feat]
    sd = model[1 + 2 * n_feat :]
    return intercept + ((x - mu) / sd) @ coef


def paired_delta_no_boot(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(float(row[candidate]) - float(row[baseline]))
    return {ds: float(np.mean(vals)) for ds, vals in by_ds.items() if vals}


def score_policy(rows: list[dict[str, Any]], pred: np.ndarray, threshold: float, label: str) -> list[dict[str, Any]]:
    out = []
    for row, score in zip(rows, pred, strict=True):
        use_anchor = float(score) >= float(threshold)
        item = dict(row)
        item[f"{label}_pred_anchor_minus_gene"] = float(score)
        item[f"{label}_use_anchor"] = bool(use_anchor)
        item[f"{label}_anchor_or_gene"] = float(row["anchor_pearson_pert"]) if use_anchor else float(row["gene_raw_mean"])
        out.append(item)
    return out


def inner_select(rows: list[dict[str, Any]], x: np.ndarray, train_idx: np.ndarray) -> tuple[float, float]:
    train_rows = [rows[int(i)] for i in train_idx]
    train_ds = sorted({str(row["dataset"]) for row in train_rows})
    y = np.asarray([float(rows[int(i)]["anchor_minus_gene_raw_mean"]) for i in train_idx], dtype=float)
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
            scored = score_policy(train_rows, pred_train, threshold, "inner")
            ds_delta = paired_delta_no_boot(scored, "inner_anchor_or_gene", "gene_raw_mean")
            ds_anchor = paired_delta_no_boot(scored, "inner_anchor_or_gene", "anchor_pearson_pert")
            vals = list(ds_delta.values())
            mean_delta = float(np.mean(vals)) if vals else -999.0
            min_delta = float(min(vals)) if vals else -999.0
            mean_anchor = float(np.mean(list(ds_anchor.values()))) if ds_anchor else -999.0
            harm_frac = float(np.mean([v < 0.0 for v in vals])) if vals else 1.0
            use_frac = float(np.mean([r["inner_use_anchor"] for r in scored])) if scored else 0.0
            key = (
                use_frac >= 0.05,
                min_delta >= -0.02,
                harm_frac <= 0.20,
                mean_delta,
                mean_anchor,
                -abs(threshold),
                -alpha,
            )
            if best is None or key > best[0]:
                best = (key, alpha, threshold)
    if best is None:
        return 1.0, 0.0
    return float(best[1]), float(best[2])


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, seed: int) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(float(row[candidate]) - float(row[baseline]))
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


def evaluate_group(all_rows: list[dict[str, Any]], group: str, gene_emb: dict[str, np.ndarray], *, shuffled: bool = False) -> dict[str, Any]:
    rows = [dict(row) for row in all_rows if row.get("group") == group]
    datasets = sorted({str(row["dataset"]) for row in rows})
    scored_all = []
    specs: dict[str, Any] = {}
    for ds in datasets:
        train_rows = [row for row in rows if str(row["dataset"]) != ds]
        val_rows = [row for row in rows if str(row["dataset"]) == ds]
        train_feat_rows = []
        for row in train_rows:
            other = [r for r in train_rows if r is not row and str(r["dataset"]) != str(row["dataset"])]
            item = dict(row)
            item["features"] = make_feature_vector(row, other, gene_emb, shuffled=shuffled, seed=SEED + len(train_feat_rows))
            train_feat_rows.append(item)
        val_feat_rows = []
        for row in val_rows:
            item = dict(row)
            item["features"] = make_feature_vector(row, train_rows, gene_emb, shuffled=shuffled, seed=SEED + 1000 + len(val_feat_rows))
            val_feat_rows.append(item)
        feature_names = sorted(train_feat_rows[0]["features"].keys()) if train_feat_rows else []
        train_idx = np.arange(len(train_feat_rows), dtype=int)
        x_train = feature_matrix(train_feat_rows, feature_names)
        y_train = np.asarray([float(row["anchor_minus_gene_raw_mean"]) for row in train_feat_rows], dtype=float)
        alpha, threshold = inner_select(train_feat_rows, x_train, train_idx)
        model = ridge_fit(x_train, y_train, alpha)
        x_val = feature_matrix(val_feat_rows, feature_names)
        pred = ridge_predict(model, x_val)
        scored = score_policy(val_feat_rows, pred, threshold, "local")
        scored_all.extend(scored)
        specs[ds] = {"alpha": alpha, "threshold": threshold, "n_train": len(train_feat_rows), "n_val": len(val_feat_rows)}
    paired = [
        paired_bootstrap(scored_all, "local_anchor_or_gene", baseline, SEED + i)
        for i, baseline in enumerate(("gene_raw_mean", "anchor_pearson_pert"))
    ]
    return {
        "group": group,
        "n_rows": len(scored_all),
        "use_anchor_fraction": float(np.mean([row["local_use_anchor"] for row in scored_all])) if scored_all else 0.0,
        "paired_deltas": paired,
        "specs_by_heldout_dataset": specs,
        "scored_rows": scored_all,
    }


def paired_row(result: dict[str, Any], baseline: str) -> dict[str, Any]:
    return next(row for row in result["paired_deltas"] if row["baseline"] == baseline)


def decide(results: list[dict[str, Any]], shuffled_results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    for result, shuf in zip(results, shuffled_results, strict=True):
        group = result["group"]
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        shuf_vs_gene = paired_row(shuf, "gene_raw_mean")
        if float(result["use_anchor_fraction"]) < 0.05:
            reasons.append(f"{group}_uses_anchor_too_rarely")
        if float(vs_gene["delta_mean"]) < 0.02:
            reasons.append(f"{group}_delta_vs_gene_below_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_p_harm_vs_gene_above_0p20")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_anchor["delta_mean"]) < -0.005:
            reasons.append(f"{group}_material_loss_vs_anchor")
        if float(vs_gene["delta_mean"]) - float(shuf_vs_gene["delta_mean"]) < 0.01:
            reasons.append(f"{group}_shuffled_neighborhood_not_beaten_by_0p01")
    return {
        "status": "tracka_xverse_train_response_local_reliability_gate_pass_code_gate_next_no_gpu"
        if not reasons
        else "tracka_xverse_train_response_local_reliability_gate_fail_no_gpu",
        "gpu_authorization": "none",
        "action": "design_small_gpu_smoke_only_if_pass_else_close_local_reliability",
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A xverse Train-Response Local Reliability Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        f"- Uses only train-only/internal residual-forensics rows and `{GENE_CACHE}` gene embeddings.",
        "- Nested leave-one-dataset-out: held-out dataset labels are never used for its feature construction, model fit, or threshold selection.",
        "- Does not read canonical outcomes, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- Shuffled-neighborhood control permutes neighbor labels while preserving feature/query structure.",
        "",
        "## Gate Rule",
        "",
        "Both internal groups must satisfy delta vs gene `>= +0.02`, p_harm `<= 0.20`, dataset min `>= -0.02`, no material loss vs anchor, use-anchor `>=0.05`, and beat shuffled-neighborhood by `>= +0.01`.",
        "",
        "## Rows",
        "",
        "| group | use anchor | delta vs gene | p harm | dataset min | delta vs anchor | shuffled delta vs gene |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result, shuf in zip(payload["results"], payload["shuffled_results"], strict=True):
        vs_gene = paired_row(result, "gene_raw_mean")
        vs_anchor = paired_row(result, "anchor_pearson_pert")
        shuf_gene = paired_row(shuf, "gene_raw_mean")
        lines.append(
            f"| {result['group']} | {result['use_anchor_fraction']:.3f} | "
            f"{fmt(vs_gene['delta_mean'])} | {fmt(vs_gene['p_harm'])} | "
            f"{fmt(vs_gene['dataset_min'])} | {fmt(vs_anchor['delta_mean'])} | "
            f"{fmt(shuf_gene['delta_mean'])} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in payload["decision"]["reasons"]] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    rows = load_json(XVERSE_ROWS)["condition_rows"]
    gene_emb = load_gene_embeddings()
    results = [evaluate_group(rows, group, gene_emb, shuffled=False) for group in GROUPS]
    shuffled_results = [evaluate_group(rows, group, gene_emb, shuffled=True) for group in GROUPS]
    decision = decide(results, shuffled_results)
    payload = {
        "status": decision["status"],
        "inputs": {
            "xverse_residual_forensics": str(XVERSE_ROWS),
            "gene_cache": str(GENE_CACHE),
        },
        "boundary": {
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
            "nested_lodo": True,
        },
        "results": results,
        "shuffled_results": shuffled_results,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
