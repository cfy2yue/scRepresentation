#!/usr/bin/env python3
"""CPU-only multi-prior tail-risk mask gate for true-cell scaling."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
POSTHOC_ROOT = ROOT / "runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625"
S0_TSV = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
TARGET_JSON = ROOT / "reports/latentfm_scaling_target_activity_gate_20260624.json"
CANONICAL_ANCHOR = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_canonical/condition_family_eval_anchor_ode20_canonical.json"

GOA = ROOT / "dataset/external_priors/goa_human_20260519/goa_human_gene_terms.tsv"
REACTOME = ROOT / "dataset/external_priors/reactome_pathways_current_20260623/reactome_gene_pathways.tsv"
CORUM = ROOT / "dataset/external_priors/corum_complexes_20260624/corum_human_gene_complexes.tsv"
OMNIPATH = ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_target_gene_features.tsv"

OUT_JSON = ROOT / "reports/latentfm_multiprior_tailrisk_mask_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_MULTIPRIOR_TAILRISK_MASK_GATE_20260625.md"
OUT_CSV = ROOT / "reports/latentfm_multiprior_tailrisk_mask_rows_20260625.csv"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def genes(condition: str) -> tuple[str, ...]:
    out = tuple(tok.strip() for tok in condition.replace(",", "+").split("+") if tok.strip())
    return out or (condition,)


def rows_by_key(blob: dict[str, Any], group: str = "family_gene") -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in blob.get("groups", {}).get(group, {}).get("condition_metrics", []):
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def metric_delta(row: dict[str, Any], anchor: dict[str, Any], metric: str) -> float | None:
    a = finite(anchor.get(metric))
    b = finite(row.get(metric))
    if a is None or b is None:
        return None
    return b - a


def read_feature_tsv(path: Path, count_col: str) -> dict[str, float]:
    vals: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, dialect="excel-tab"):
            gene = str(row.get("gene") or "").upper()
            val = finite(row.get(count_col)) or 0.0
            if gene:
                vals[gene] = val
    return vals


def read_omnipath(path: Path) -> dict[str, dict[str, float]]:
    vals: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect="excel-tab")
        for row in reader:
            gene = str(row.get("gene") or "").upper()
            if not gene:
                continue
            vals[gene] = {k: finite(v) or 0.0 for k, v in row.items() if k != "gene"}
    return vals


def load_s0() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with S0_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect="excel-tab")
        for row in reader:
            key = (str(row.get("dataset") or ""), str(row.get("condition") or ""))
            if key[0] and key[1]:
                out[key] = row
    return out


def auc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for ps in pos:
        for ns in neg:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / (len(pos) * len(neg))


def bootstrap_ci(values: list[float], seed: int = 20260625, n_boot: int = 1000) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    stats = []
    n = len(values)
    for _ in range(n_boot):
        stats.append(mean(values[rng.randrange(n)] for _ in range(n)))
    stats.sort()
    low = stats[int(0.025 * (n_boot - 1))]
    high = stats[int(0.975 * (n_boot - 1))]
    p_le_zero = (1 + sum(v <= 0.0 for v in stats)) / (1 + len(stats))
    return low, high, p_le_zero


def build_rows() -> list[dict[str, Any]]:
    s0 = load_s0()
    target_rows = {
        (str(r["dataset"]), str(r["condition"])): r
        for r in load_json(TARGET_JSON).get("rows", [])
        if r.get("activity_status") == "ok"
    }
    goa = read_feature_tsv(GOA, "n_go_terms")
    reactome = read_feature_tsv(REACTOME, "n_reactome_pathways")
    corum = read_feature_tsv(CORUM, "n_complexes")
    omnipath = read_omnipath(OMNIPATH)

    per_key: dict[tuple[str, str], dict[str, Any]] = {}
    for seed in (42, 43, 44):
        pdir = POSTHOC_ROOT / f"xverse_truecell_nested_budget128_tailstable_seed{seed}_6000/posthoc_eval_internal"
        anchor = load_json(pdir / "condition_family_eval_anchor_internal_ode20.json")
        cand = load_json(pdir / "condition_family_eval_candidate_internal_ode20.json")
        arows = rows_by_key(anchor)
        crows = rows_by_key(cand)
        for key in sorted(set(arows) & set(crows)):
            pp = metric_delta(crows[key], arows[key], "pearson_pert")
            mmd = metric_delta(crows[key], arows[key], "test_mmd_clamped")
            if pp is None or mmd is None:
                continue
            item = per_key.setdefault(key, {"pp": {}, "mmd": {}})
            item["pp"][seed] = pp
            item["mmd"][seed] = mmd

    rows: list[dict[str, Any]] = []
    for key, vals in per_key.items():
        if len(vals["pp"]) != 3 or len(vals["mmd"]) != 3:
            continue
        ds, cond = key
        g = [x.upper() for x in genes(cond)]
        meta = s0.get(key, {})
        targ = target_rows.get(key, {})
        pp_vals = [float(vals["pp"][seed]) for seed in (42, 43, 44)]
        mmd_vals = [float(vals["mmd"][seed]) for seed in (42, 43, 44)]

        def avg_feature(table: dict[str, float]) -> float:
            vals2 = [table.get(x, 0.0) for x in g]
            return mean(vals2) if vals2 else 0.0

        op_keys = [
            "tf_out_degree",
            "target_in_degree",
            "tf_activation_out_degree",
            "tf_inhibition_out_degree",
            "target_activation_in_degree",
            "target_inhibition_in_degree",
        ]
        op_features = {}
        for col in op_keys:
            vals2 = [omnipath.get(x, {}).get(col, 0.0) for x in g]
            op_features[col] = mean(vals2) if vals2 else 0.0

        row = {
            "dataset": ds,
            "condition": cond,
            "pp_mean": mean(pp_vals),
            "pp_min_seed": min(pp_vals),
            "mmd_mean": mean(mmd_vals),
            "mmd_max_seed": max(mmd_vals),
            "hard_harm": int(min(pp_vals) < -0.05 or max(mmd_vals) > 0.010),
            "n_genes": len(g),
            "go_terms_mean": avg_feature(goa),
            "reactome_pathways_mean": avg_feature(reactome),
            "corum_complexes_mean": avg_feature(corum),
            "target_expr_mean": finite(targ.get("target_expr_mean")) or 0.0,
            "target_expr_nonzero_fraction": finite(targ.get("target_expr_nonzero_fraction")) or 0.0,
            "n_cells": finite(meta.get("n_cells")) or 0.0,
            "perturbation_type": meta.get("perturbation_type") or "unknown",
            "source_label": meta.get("source_label") or "unknown",
            "cell_background_source": meta.get("cell_background_source") or "unknown",
        }
        row.update(op_features)
        rows.append(row)
    return rows


NUMERIC_FEATURES = [
    "n_genes",
    "go_terms_mean",
    "reactome_pathways_mean",
    "corum_complexes_mean",
    "tf_out_degree",
    "target_in_degree",
    "tf_activation_out_degree",
    "tf_inhibition_out_degree",
    "target_activation_in_degree",
    "target_inhibition_in_degree",
    "target_expr_mean",
    "target_expr_nonzero_fraction",
    "n_cells",
]
CATEGORICAL_FEATURES = ["perturbation_type", "source_label", "cell_background_source"]


def fit_score(train: list[dict[str, Any]], labels_key: str = "hard_harm") -> dict[str, Any]:
    y = [int(r[labels_key]) for r in train]
    weights: dict[str, float] = {}
    stats: dict[str, tuple[float, float]] = {}
    for feat in NUMERIC_FEATURES:
        vals = [float(r[feat]) for r in train]
        mu = mean(vals) if vals else 0.0
        var = mean((v - mu) ** 2 for v in vals) if vals else 0.0
        sd = math.sqrt(var) or 1.0
        pos = [v for v, yy in zip(vals, y) if yy]
        neg = [v for v, yy in zip(vals, y) if not yy]
        weight = ((mean(pos) if pos else mu) - (mean(neg) if neg else mu)) / sd
        # Cap one noisy feature so tiny folds cannot dominate.
        weights[feat] = max(-2.0, min(2.0, weight))
        stats[feat] = (mu, sd)
    cat_rates: dict[str, dict[str, float]] = {}
    global_rate = mean(y) if y else 0.0
    for feat in CATEGORICAL_FEATURES:
        by_level: dict[str, list[int]] = defaultdict(list)
        for row, yy in zip(train, y):
            by_level[str(row[feat])].append(yy)
        cat_rates[feat] = {
            level: (sum(vals) + 1.0) / (len(vals) + 2.0) - global_rate for level, vals in by_level.items()
        }
    return {"weights": weights, "stats": stats, "cat_rates": cat_rates, "global_rate": global_rate}


def score_row(row: dict[str, Any], model: dict[str, Any]) -> float:
    score = float(model["global_rate"])
    for feat, weight in model["weights"].items():
        mu, sd = model["stats"][feat]
        score += float(weight) * ((float(row[feat]) - mu) / sd)
    for feat, rates in model["cat_rates"].items():
        score += float(rates.get(str(row[feat]), 0.0))
    return score


def choose_threshold(train: list[dict[str, Any]], scores: list[float]) -> float | None:
    candidates = sorted(set(scores))
    best: tuple[float, float] | None = None
    for t in candidates:
        sim_pp = [float(r["pp_mean"]) if s <= t else 0.0 for r, s in zip(train, scores)]
        sim_mmd = [float(r["mmd_mean"]) if s <= t else 0.0 for r, s in zip(train, scores)]
        sim_harm = [
            1.0 if (float(r["pp_min_seed"]) < -0.05 or float(r["mmd_max_seed"]) > 0.010) and s <= t else 0.0
            for r, s in zip(train, scores)
        ]
        if not sim_pp:
            continue
        if mean(sim_pp) >= 0.020 and max(sim_mmd) <= 0.004 and mean(sim_harm) <= 0.10:
            coverage = sum(1 for s in scores if s <= t) / len(scores)
            value = mean(sim_pp) + 0.01 * coverage
            if best is None or value > best[0]:
                best = (value, t)
    return None if best is None else best[1]


def run_lodo(rows: list[dict[str, Any]], *, shuffle_labels: bool = False, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    labels = [int(r["hard_harm"]) for r in rows]
    if shuffle_labels:
        labels = labels[:]
        rng.shuffle(labels)
    rows_with_labels = [dict(r, train_label=lab) for r, lab in zip(rows, labels)]
    scores_by_idx: dict[int, float] = {}
    thresholds: dict[str, float | None] = {}
    enabled = [False] * len(rows)
    for ds in sorted({str(r["dataset"]) for r in rows}):
        train_idx = [i for i, r in enumerate(rows_with_labels) if str(r["dataset"]) != ds]
        test_idx = [i for i, r in enumerate(rows_with_labels) if str(r["dataset"]) == ds]
        train = [rows_with_labels[i] for i in train_idx]
        model = fit_score(train, labels_key="train_label")
        train_scores = [score_row(r, model) for r in train]
        t = choose_threshold(train, train_scores)
        thresholds[ds] = t
        for i in test_idx:
            s = score_row(rows_with_labels[i], model)
            scores_by_idx[i] = s
            enabled[i] = bool(t is not None and s <= t)
    ordered_scores = [scores_by_idx[i] for i in range(len(rows))]
    observed_labels = [int(r["hard_harm"]) for r in rows]
    sim_pp = [float(r["pp_mean"]) if en else 0.0 for r, en in zip(rows, enabled)]
    sim_mmd = [float(r["mmd_mean"]) if en else 0.0 for r, en in zip(rows, enabled)]
    sim_harm = [
        1.0 if (float(r["pp_min_seed"]) < -0.05 or float(r["mmd_max_seed"]) > 0.010) and en else 0.0
        for r, en in zip(rows, enabled)
    ]
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row, pp in zip(rows, sim_pp):
        by_ds[str(row["dataset"])].append(pp)
    ci_low, ci_high, p_le_zero = bootstrap_ci(sim_pp)
    return {
        "auc": auc(observed_labels, ordered_scores),
        "enabled_fraction": sum(enabled) / len(enabled) if enabled else 0.0,
        "sim_pp_mean": mean(sim_pp) if sim_pp else None,
        "sim_pp_ci95": [ci_low, ci_high],
        "sim_p_le_zero": p_le_zero,
        "sim_hard_harm_fraction": mean(sim_harm) if sim_harm else None,
        "sim_mmd_max": max(sim_mmd) if sim_mmd else None,
        "worst_dataset_pp": min((mean(v) for v in by_ds.values()), default=None),
        "thresholds_missing": sum(1 for v in thresholds.values() if v is None),
        "scores": ordered_scores,
        "enabled": enabled,
    }


def main() -> None:
    rows = build_rows()
    primary = run_lodo(rows, shuffle_labels=False, seed=20260625)
    shuffled = [run_lodo(rows, shuffle_labels=True, seed=20260625 + i) for i in range(100)]
    shuffled_auc = [x["auc"] for x in shuffled if x["auc"] is not None]
    shuffled_pp = [x["sim_pp_mean"] for x in shuffled if x["sim_pp_mean"] is not None]
    auc_p = None
    pp_margin = None
    if primary["auc"] is not None and shuffled_auc:
        auc_p = (1 + sum(float(x) >= float(primary["auc"]) for x in shuffled_auc)) / (1 + len(shuffled_auc))
    if primary["sim_pp_mean"] is not None and shuffled_pp:
        pp_margin = float(primary["sim_pp_mean"]) - mean(float(x) for x in shuffled_pp)

    canonical = load_json(CANONICAL_ANCHOR)
    footprint = {
        group: len(canonical.get("groups", {}).get(group, {}).get("selected_conditions", []))
        for group in ("test_single", "family_gene")
    }

    reasons: list[str] = []
    if len(rows) < 100:
        reasons.append("too_few_trainonly_rows")
    if primary["auc"] is None or float(primary["auc"]) < 0.70:
        reasons.append("lodo_auc_below_0p70")
    if auc_p is None or auc_p > 0.05:
        reasons.append("auc_not_significant_vs_shuffled_labels")
    if primary["sim_pp_mean"] is None or float(primary["sim_pp_mean"]) < 0.020:
        reasons.append("retained_mean_pp_below_0p020")
    if primary["sim_pp_ci95"][0] <= 0.0:
        reasons.append("retained_pp_ci_lower_not_positive")
    if primary["sim_hard_harm_fraction"] is None or float(primary["sim_hard_harm_fraction"]) > 0.10:
        reasons.append("retained_hard_harm_fraction_above_0p10")
    if primary["worst_dataset_pp"] is None or float(primary["worst_dataset_pp"]) < -0.005:
        reasons.append("worst_dataset_pp_below_minus_0p005")
    if primary["sim_mmd_max"] is None or float(primary["sim_mmd_max"]) > 0.004:
        reasons.append("mmd_max_above_0p004")
    if pp_margin is None or pp_margin < 0.010:
        reasons.append("real_control_margin_below_0p010")
    if footprint["test_single"] < 25 or footprint["family_gene"] < 25:
        reasons.append("canonical_metadata_footprint_too_small")

    status = "multiprior_tailrisk_mask_pass_cpu_review_next_no_gpu" if not reasons else "multiprior_tailrisk_mask_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "train_only_internal_deltas": True,
            "canonical_metadata_footprint_only": True,
            "canonical_performance_used": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
        },
        "summary": {
            "n_rows": len(rows),
            "n_hard_harm": sum(int(r["hard_harm"]) for r in rows),
            "primary": {k: v for k, v in primary.items() if k not in {"scores", "enabled"}},
            "shuffled_auc_mean": mean(shuffled_auc) if shuffled_auc else None,
            "shuffled_pp_mean": mean(float(x) for x in shuffled_pp) if shuffled_pp else None,
            "auc_p_vs_shuffled": auc_p,
            "pp_margin_vs_shuffled": pp_margin,
            "canonical_metadata_footprint": footprint,
        },
        "reasons": reasons,
        "next_action": (
            "external review and implementation/launcher gate before any bounded GPU"
            if not reasons
            else "do not launch multi-prior tail-risk mask GPU; richer prior features do not pass train-only tail gate"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "condition",
            "pp_mean",
            "pp_min_seed",
            "mmd_mean",
            "mmd_max_seed",
            "hard_harm",
            "enabled",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, enabled in zip(rows, primary["enabled"]):
            writer.writerow({**{k: row[k] for k in fieldnames if k != "enabled"}, "enabled": enabled})

    lines = [
        "# LatentFM Multi-Prior Tail-Risk Mask Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only train-only/internal LODO gate.",
        "- Uses frozen GOA/Reactome/CORUM/OmniPath, S0 provenance, and target-activity features.",
        "- Canonical is used only for metadata footprint, not performance.",
        "- No canonical multi, Track C query, training, inference, or GPU.",
        "",
        "## Summary",
        "",
        f"- rows: `{payload['summary']['n_rows']}`",
        f"- hard-harm rows: `{payload['summary']['n_hard_harm']}`",
        f"- LODO AUROC: `{primary['auc']}`",
        f"- AUROC p vs shuffled labels: `{auc_p}`",
        f"- enabled fraction: `{primary['enabled_fraction']}`",
        f"- simulated pp mean: `{primary['sim_pp_mean']}`",
        f"- simulated pp CI95: `{primary['sim_pp_ci95']}`",
        f"- simulated hard-harm fraction: `{primary['sim_hard_harm_fraction']}`",
        f"- simulated MMD max: `{primary['sim_mmd_max']}`",
        f"- worst dataset pp: `{primary['worst_dataset_pp']}`",
        f"- pp margin vs shuffled-label control: `{pp_margin}`",
        f"- canonical metadata footprint: `{footprint}`",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- next action: `{payload['next_action']}`",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- row CSV: `{OUT_CSV}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
