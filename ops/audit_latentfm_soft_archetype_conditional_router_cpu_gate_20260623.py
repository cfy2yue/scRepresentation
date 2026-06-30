#!/usr/bin/env python3
"""CPU gate for a conditional soft-archetype Track A router.

The previous soft-archetype gate showed stable K16 soft state memberships but
failed aggregate dataset-mean no-harm. This script tests a conservative
conditional router:

* rebuild the frozen K16 soft-archetype spec on the train-only internal split;
* use only train rows to decide whether a dataset has positive archetype margin;
* apply archetype predictions on validation rows only for those datasets;
* compare against dataset/gene/raw-archetype/shuffled controls.

Only the predeclared primary rule can authorize GPU. Other rules are reported
as diagnostics and must not be used for post-hoc selection.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
sys.path.insert(0, str(OPS))

from audit_latentfm_soft_archetype_predictive_gate_20260623 import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_GENE_CACHE,
    DEFAULT_PERT_MEANS,
    DEFAULT_SPLIT,
    dataset_stats,
    features,
    fit_pca,
    fit_ridge,
    load_gene_embeddings,
    load_json,
    normalized_entropy,
    predict_ridge,
    residualized_ctrl,
    simple_kmeans,
    soft_assign,
    transform_pca,
)
from audit_latentfm_xverse_background_state_residual_consensus_gate_20260622 import (  # noqa: E402
    GROUPS,
    build_baselines,
    collect_rows,
    paired_bootstrap,
    score,
)


OUT_JSON = ROOT / "reports/latentfm_soft_archetype_conditional_router_cpu_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_CONDITIONAL_ROUTER_CPU_GATE_20260623.md"

CANDIDATE = "soft_archetype_gene_interact_ridge"
SHUFFLED = "soft_archetype_gene_shuffled_ridge"
ROUTER = "conditional_archetype_router"
SHUFFLED_ROUTER = "conditional_shuffled_archetype_router"
PRIMARY_RULE = "dataset_train_margin_positive"
HARM_POCKETS = {"Jiang_IFNG", "Jiang_TNFA", "NormanWeissman2019_filtered", "Jiang_TGFB"}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def score_rows_with_soft_features(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    pert_means: dict[str, np.ndarray],
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_ctrl = np.vstack([r["ctrl"] for r in train_rows]).astype(np.float32)
    train_gene = np.vstack([r["gene_emb"] for r in train_rows]).astype(np.float32)
    val_gene = np.vstack([r["gene_emb"] for r in val_rows]).astype(np.float32)
    train_y = np.vstack([r["residual"] for r in train_rows]).astype(np.float32)

    bg_pca = fit_pca(train_ctrl, 24)
    gene_pca = fit_pca(train_gene, 16)
    stats = dataset_stats(train_rows, bg_pca)
    train_state = residualized_ctrl(train_rows, bg_pca, stats)
    val_state = residualized_ctrl(val_rows, bg_pca, stats)
    train_gene_p = transform_pca(train_gene, gene_pca)
    val_gene_p = transform_pca(val_gene, gene_pca)

    centroids = simple_kmeans(train_state, k=16, seed=42, max_iter=120)
    train_w = soft_assign(train_state, centroids)
    val_w = soft_assign(val_state, centroids)
    rng = np.random.default_rng(seed + 16)
    train_w_shuf = train_w[rng.permutation(train_w.shape[0])]
    val_w_shuf = val_w[rng.permutation(val_w.shape[0])]

    specs = {
        CANDIDATE: ("background_gene_interact", train_w, train_gene_p, val_w, val_gene_p),
        "gene_only_ridge": ("gene_only", train_w, train_gene_p, val_w, val_gene_p),
        SHUFFLED: ("background_gene_interact", train_w_shuf, train_gene_p, val_w_shuf, val_gene_p),
    }
    pred_train: dict[str, np.ndarray] = {}
    pred_val: dict[str, np.ndarray] = {}
    for name, (mode, tr_state, tr_gene, va_state, va_gene) in specs.items():
        x_train = features(tr_state, tr_gene, mode, 8)
        x_val = features(va_state, va_gene, mode, 8)
        model = fit_ridge(x_train, train_y, 20.0)
        pred_train[name] = predict_ridge(model, x_train)
        pred_val[name] = predict_ridge(model, x_val)

    baselines = build_baselines(train_rows)
    train_counts = Counter(str(r["dataset"]) for r in train_rows)
    gene_counts = Counter(str(r["gene"]) for r in train_rows)

    def score_one(row: dict[str, Any], i: int, weights: np.ndarray, preds: dict[str, np.ndarray], group: str) -> dict[str, Any]:
        ds = str(row["dataset"])
        gene = str(row["gene"])
        out: dict[str, Any] = {
            "dataset": ds,
            "condition": row["condition"],
            "gene": gene,
            "group": row.get("group", group),
            "soft_entropy": float(normalized_entropy(weights[i : i + 1])[0]),
            "soft_max_mass": float(weights[i].max()),
            "dataset_train_count": int(train_counts[ds]),
            "gene_train_count": int(gene_counts[gene]),
        }
        baseline_preds = {
            "dataset_mean": baselines["dataset_mean"].get(ds, baselines["global_mean"]),
            "gene_raw_mean": baselines["gene_raw_mean"].get(gene, baselines["global_mean"]),
            "global_mean": baselines["global_mean"],
        }
        for name, arr in preds.items():
            out[name] = score(row, arr[i], pert_means)
        for name, pred in baseline_preds.items():
            out[name] = score(row, pred, pert_means)
        return out

    train_scored = [score_one(row, i, train_w, pred_train, "train_internal_fit_proxy") for i, row in enumerate(train_rows)]
    val_scored = [score_one(row, i, val_w, pred_val, "") for i, row in enumerate(val_rows)]
    return train_scored, val_scored


def dataset_margins(train_rows: list[dict[str, Any]], candidate: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in train_rows:
        if row.get(candidate) is None or row.get("dataset_mean") is None:
            continue
        grouped[str(row["dataset"])].append(float(row[candidate]) - float(row["dataset_mean"]))
    return {ds: mean(vals) for ds, vals in grouped.items() if vals}


def apply_rule(
    rows: list[dict[str, Any]],
    *,
    name: str,
    margin_by_dataset: dict[str, float],
    shuffled_margin_by_dataset: dict[str, float],
    train_soft_max_median: float,
    train_entropy_median: float,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ds = str(row["dataset"])
        use = False
        if name == "dataset_train_margin_positive":
            use = margin_by_dataset.get(ds, -999.0) > 0.0
        elif name == "dataset_train_margin_ge_002":
            use = margin_by_dataset.get(ds, -999.0) >= 0.02
        elif name == "dataset_train_margin_positive_soft_conf":
            use = (
                margin_by_dataset.get(ds, -999.0) > 0.0
                and float(row["soft_max_mass"]) >= train_soft_max_median
                and float(row["soft_entropy"]) <= train_entropy_median
            )
        elif name == "soft_conf_only_no_dataset_margin":
            use = float(row["soft_max_mass"]) >= train_soft_max_median and float(row["soft_entropy"]) <= train_entropy_median
        else:
            raise ValueError(name)

        item = dict(row)
        item[ROUTER] = item[CANDIDATE] if use else item["dataset_mean"]
        use_shuf = shuffled_margin_by_dataset.get(ds, -999.0) > 0.0 if "dataset_train_margin" in name else use
        item[SHUFFLED_ROUTER] = item[SHUFFLED] if use_shuf else item["dataset_mean"]
        item["router_used_archetype"] = bool(use)
        out.append(item)
    return out


def summarize_rule(rows: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    paired = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        for baseline in ("dataset_mean", "gene_raw_mean", "gene_only_ridge", CANDIDATE, SHUFFLED_ROUTER):
            row = paired_bootstrap(group_rows, ROUTER, baseline, n_boot=2000, seed=42 + len(paired))
            row["group"] = group
            row["baseline"] = baseline
            paired.append(row)
    harm = []
    for group in GROUPS:
        for dataset in sorted(HARM_POCKETS):
            items = [r for r in rows if r["group"] == group and r["dataset"] == dataset]
            if not items:
                continue
            deltas = [float(r[ROUTER]) - float(r["dataset_mean"]) for r in items if r.get(ROUTER) is not None]
            if deltas:
                harm.append(
                    {
                        "group": group,
                        "dataset": dataset,
                        "n": len(deltas),
                        "delta_vs_dataset_mean": mean(deltas),
                        "median_delta_vs_dataset_mean": median(deltas),
                        "negative_fraction": sum(x < 0.0 for x in deltas) / len(deltas),
                    }
                )
    return {
        "rule": rule,
        "coverage_fraction": sum(bool(r.get("router_used_archetype")) for r in rows) / max(len(rows), 1),
        "paired_deltas": paired,
        "harm_pockets": harm,
    }


def decide(primary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    paired = {(r["group"], r["baseline"]): r for r in primary["paired_deltas"]}
    for group in GROUPS:
        ds = paired.get((group, "dataset_mean")) or {}
        if ds.get("status") != "ok" or float(ds.get("delta_mean") or -999.0) < 0.02:
            reasons.append(f"{group}_dataset_mean_delta_below_0p02")
        if float(ds.get("p_harm") if ds.get("p_harm") is not None else 1.0) > 0.20:
            reasons.append(f"{group}_dataset_mean_p_harm_above_0p20")
        if ds.get("leave_one_min") is None or float(ds["leave_one_min"]) < -0.02:
            reasons.append(f"{group}_leave_one_dataset_below_minus_0p02")
        raw = paired.get((group, CANDIDATE)) or {}
        if raw.get("status") != "ok" or float(raw.get("delta_mean") or -999.0) <= 0.0:
            reasons.append(f"{group}_does_not_beat_raw_archetype")
        if float(raw.get("p_harm") if raw.get("p_harm") is not None else 1.0) > 0.20:
            reasons.append(f"{group}_raw_archetype_comparison_p_harm_above_0p20")
        gene = paired.get((group, "gene_only_ridge")) or {}
        if gene.get("status") != "ok" or float(gene.get("delta_mean") or -999.0) <= 0.0:
            reasons.append(f"{group}_does_not_beat_gene_only")
        if float(gene.get("p_harm") if gene.get("p_harm") is not None else 1.0) > 0.20:
            reasons.append(f"{group}_gene_only_comparison_p_harm_above_0p20")
        shuf = paired.get((group, SHUFFLED_ROUTER)) or {}
        if shuf.get("status") != "ok" or float(shuf.get("delta_mean") or -999.0) < 0.02:
            reasons.append(f"{group}_shuffled_router_did_not_collapse")
        if float(shuf.get("p_harm") if shuf.get("p_harm") is not None else 1.0) > 0.20:
            reasons.append(f"{group}_shuffled_router_comparison_p_harm_above_0p20")

    for row in primary["harm_pockets"]:
        if float(row["delta_vs_dataset_mean"]) < -0.02 or float(row["negative_fraction"]) > 0.50:
            reasons.append(f"harm_pocket_not_protected_{row['group']}_{row['dataset']}")

    return {
        "status": "soft_archetype_conditional_router_cpu_gate_pass_authorize_one_capped_smoke"
        if not reasons
        else "soft_archetype_conditional_router_cpu_gate_fail_no_gpu",
        "gpu_authorization": "one_capped_smoke" if not reasons else "none",
        "primary_rule": PRIMARY_RULE,
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Soft-Archetype Conditional Router CPU Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"GPU authorization: `{payload['decision']['gpu_authorization']}`",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- primary rule: `{payload['decision']['primary_rule']}`",
        f"- train rows: `{payload['n_train_rows']}`",
        f"- validation rows: `{payload['n_val_rows']}`",
        "",
        "## Rule Summary",
        "",
        "| rule | coverage | status if primary? | reasons |",
        "|---|---:|---|---|",
    ]
    for rule in payload["rules"]:
        status = payload["decision"]["status"] if rule["rule"] == payload["decision"]["primary_rule"] else "diagnostic_only"
        reasons = ";".join(payload["decision"]["reasons"]) if rule["rule"] == payload["decision"]["primary_rule"] else "not eligible for posthoc selection"
        lines.append(f"| {rule['rule']} | {fmt(rule['coverage_fraction'])} | {status} | {reasons or 'none'} |")

    primary = next(r for r in payload["rules"] if r["rule"] == payload["decision"]["primary_rule"])
    lines += [
        "",
        "## Primary Rule Paired Deltas",
        "",
        "| group | baseline | delta | 95% CI | p improve | p harm | leave-one min | status |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in primary["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {row['group']} | {row['baseline']} | {fmt(row.get('delta_mean'))} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row.get('p_improvement'))} | "
            f"{fmt(row.get('p_harm'))} | {fmt(row.get('leave_one_min'))} | {row.get('status')} |"
        )
    lines += [
        "",
        "## Primary Rule Harm Pockets",
        "",
        "| group | dataset | n | delta vs dataset_mean | neg frac |",
        "|---|---|---:|---:|---:|",
    ]
    for row in primary["harm_pockets"]:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['n']} | "
            f"{fmt(row['delta_vs_dataset_mean'])} | {fmt(row['negative_fraction'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Only the predeclared primary rule can authorize GPU; diagnostic rules are not valid for post-hoc selection.",
        "- This gate uses train-only/internal proxy data and does not read canonical, multi/query, or active GPU run artifacts.",
        "- If the primary rule fails, keep archetype as a diagnostic feature rather than a GPU adapter branch.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    data_dir = DEFAULT_DATA_DIR.resolve()
    split = load_json(DEFAULT_SPLIT)
    manifest = load_json(data_dir / "manifest.json")
    metadata = load_json(Path(manifest["condition_metadata_file"]))
    gene_mapping, gene_emb, unk_index = load_gene_embeddings(DEFAULT_GENE_CACHE)
    pert_means = {k: v.astype(np.float32) for k, v in np.load(DEFAULT_PERT_MEANS).items()}
    train_rows, val_rows = collect_rows(
        data_dir,
        split,
        metadata,
        gene_mapping,
        gene_emb,
        unk_index,
        max_train_per_dataset=160,
        max_cells=128,
    )
    train_scored, val_scored = score_rows_with_soft_features(train_rows, val_rows, pert_means)
    margin_by_dataset = dataset_margins(train_scored, CANDIDATE)
    shuffled_margin_by_dataset = dataset_margins(train_scored, SHUFFLED)
    train_soft_max_median = median(float(r["soft_max_mass"]) for r in train_scored)
    train_entropy_median = median(float(r["soft_entropy"]) for r in train_scored)
    rules = []
    for rule in (
        PRIMARY_RULE,
        "dataset_train_margin_ge_002",
        "dataset_train_margin_positive_soft_conf",
        "soft_conf_only_no_dataset_margin",
    ):
        routed = apply_rule(
            val_scored,
            name=rule,
            margin_by_dataset=margin_by_dataset,
            shuffled_margin_by_dataset=shuffled_margin_by_dataset,
            train_soft_max_median=train_soft_max_median,
            train_entropy_median=train_entropy_median,
        )
        rules.append(summarize_rule(routed, rule))

    primary = next(r for r in rules if r["rule"] == PRIMARY_RULE)
    payload = {
        "data_dir": str(data_dir),
        "split_file": str(DEFAULT_SPLIT),
        "pert_means_file": str(DEFAULT_PERT_MEANS),
        "leakage_status": "trainonly_internal_proxy_no_canonical_no_multi_no_query_no_active_run_artifacts",
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "train_soft_max_median": train_soft_max_median,
        "train_entropy_median": train_entropy_median,
        "dataset_margin_summary": {
            "positive_margin_datasets": sorted(ds for ds, value in margin_by_dataset.items() if value > 0.0),
            "n_positive_margin_datasets": sum(value > 0.0 for value in margin_by_dataset.values()),
            "n_datasets": len(margin_by_dataset),
        },
        "rules": rules,
        "decision": decide(primary),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
