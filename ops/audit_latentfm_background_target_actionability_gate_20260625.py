#!/usr/bin/env python3
"""CPU-only background-conditional target-actionability gate."""

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
TARGET_JSON = ROOT / "reports/latentfm_scaling_target_activity_gate_20260624.json"
S0_TSV = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
CANONICAL_ANCHOR = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_canonical/condition_family_eval_anchor_ode20_canonical.json"

GOA = ROOT / "dataset/external_priors/goa_human_20260519/goa_human_gene_terms.tsv"
REACTOME = ROOT / "dataset/external_priors/reactome_pathways_current_20260623/reactome_gene_pathways.tsv"
CORUM = ROOT / "dataset/external_priors/corum_complexes_20260624/corum_human_gene_complexes.tsv"
OMNIPATH = ROOT / "dataset/external_priors/omnipath_tf_20260623/omnipath_tf_target_gene_features.tsv"

OUT_JSON = ROOT / "reports/latentfm_background_target_actionability_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_BACKGROUND_TARGET_ACTIONABILITY_GATE_20260625.md"
OUT_CSV = ROOT / "reports/latentfm_background_target_actionability_rows_20260625.csv"


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
    out = tuple(tok.strip().upper() for tok in condition.replace(",", "+").split("+") if tok.strip())
    return out or (condition.upper(),)


def read_count(path: Path, count_col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, dialect="excel-tab"):
            gene = str(row.get("gene") or "").upper()
            if gene:
                out[gene] = finite(row.get(count_col)) or 0.0
    return out


def read_omnipath(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, dialect="excel-tab"):
            gene = str(row.get("gene") or "").upper()
            if not gene:
                continue
            vals = [
                finite(row.get("tf_out_degree")) or 0.0,
                finite(row.get("target_in_degree")) or 0.0,
                finite(row.get("tf_activation_out_degree")) or 0.0,
                finite(row.get("tf_inhibition_out_degree")) or 0.0,
                finite(row.get("target_activation_in_degree")) or 0.0,
                finite(row.get("target_inhibition_in_degree")) or 0.0,
            ]
            out[gene] = sum(vals)
    return out


def load_s0() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with S0_TSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, dialect="excel-tab"):
            ds = str(row.get("dataset") or "")
            cond = str(row.get("condition") or "")
            if ds and cond:
                out[(ds, cond)] = row
    return out


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mu = mean(values)
    sd = math.sqrt(mean((x - mu) ** 2 for x in values)) or 1.0
    return [(x - mu) / sd for x in values]


def bootstrap_ci(values: list[float], seed: int = 20260625, n_boot: int = 1000) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    low = stats[int(0.025 * (n_boot - 1))]
    high = stats[int(0.975 * (n_boot - 1))]
    p_le_zero = (1 + sum(x <= 0.0 for x in stats)) / (1 + len(stats))
    return low, high, p_le_zero


def build_rows() -> list[dict[str, Any]]:
    target = load_json(TARGET_JSON)
    s0 = load_s0()
    goa = read_count(GOA, "n_go_terms")
    reactome = read_count(REACTOME, "n_reactome_pathways")
    corum = read_count(CORUM, "n_complexes")
    omnipath = read_omnipath(OMNIPATH)
    rows = []
    for row in target.get("rows", []):
        if row.get("activity_status") != "ok":
            continue
        ds = str(row["dataset"])
        cond = str(row["condition"])
        meta = s0.get((ds, cond), {})
        g = genes(cond)

        def avg(table: dict[str, float]) -> float:
            vals = [table.get(x, 0.0) for x in g]
            return mean(vals) if vals else 0.0

        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "pp_delta": float(row["pp_delta_cap120_minus_cap30"]),
                "mmd_delta": float(row["mmd_delta_cap120_minus_cap30"]),
                "target_expr_mean": finite(row.get("target_expr_mean")) or 0.0,
                "target_expr_nonzero_fraction": finite(row.get("target_expr_nonzero_fraction")) or 0.0,
                "prior_go": avg(goa),
                "prior_reactome": avg(reactome),
                "prior_corum": avg(corum),
                "prior_omnipath": avg(omnipath),
                "cell_background_source": meta.get("cell_background_source") or "unknown",
                "perturbation_type": meta.get("perturbation_type") or "unknown",
                "source_label": meta.get("source_label") or "unknown",
            }
        )
    # Fixed predeclared actionability score: target expression + nonzero +
    # frozen external-prior support. No labels are used to fit this score.
    cols = [
        "target_expr_mean",
        "target_expr_nonzero_fraction",
        "prior_go",
        "prior_reactome",
        "prior_corum",
        "prior_omnipath",
    ]
    zcols = {col: zscore([float(r[col]) for r in rows]) for col in cols}
    for i, row in enumerate(rows):
        row["actionability_score"] = sum(zcols[col][i] for col in cols) / len(cols)
    return rows


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    pp = [float(r["pp_delta"]) for r in rows]
    mmd = [float(r["mmd_delta"]) for r in rows]
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(float(row["pp_delta"]))
    ci_low, ci_high, p_le_zero = bootstrap_ci(pp)
    return {
        "n": len(rows),
        "pp_mean": mean(pp),
        "pp_ci95": [ci_low, ci_high],
        "p_le_zero": p_le_zero,
        "pp_min": min(pp),
        "dataset_min_pp": min(mean(v) for v in by_ds.values()),
        "negative_dataset_tail_count": sum(1 for v in by_ds.values() if mean(v) < -0.020),
        "hard_harm_frac": mean([1.0 if x < -0.020 else 0.0 for x in pp]),
        "mmd_mean": mean(mmd),
        "mmd_max": max(mmd),
    }


def main() -> None:
    rows = build_rows()
    if not rows:
        raise SystemExit("no rows")
    scores = sorted(float(r["actionability_score"]) for r in rows)
    threshold = scores[int(0.75 * (len(scores) - 1))]
    high = [r for r in rows if float(r["actionability_score"]) >= threshold]
    low = [r for r in rows if float(r["actionability_score"]) < threshold]
    high_summary = summarize_subset(high)
    low_summary = summarize_subset(low)

    rng = random.Random(20260625)
    shuffle_means = []
    for _ in range(1000):
        labels = [float(r["actionability_score"]) for r in rows]
        rng.shuffle(labels)
        shigh = [r for r, score in zip(rows, labels) if score >= threshold]
        shuffle_means.append(summarize_subset(shigh)["pp_mean"])
    shuffle_mean = mean(shuffle_means)
    shuffle_p95 = sorted(shuffle_means)[int(0.95 * (len(shuffle_means) - 1))]
    observed_margin = float(high_summary["pp_mean"]) - float(low_summary["pp_mean"])
    shuffle_margin = float(high_summary["pp_mean"]) - shuffle_mean
    shuffle_p = (1 + sum(x >= float(high_summary["pp_mean"]) for x in shuffle_means)) / (1 + len(shuffle_means))

    canonical = load_json(CANONICAL_ANCHOR)
    footprint = {
        group: len(canonical.get("groups", {}).get(group, {}).get("selected_conditions", []))
        for group in ("test_single", "family_gene")
    }

    reasons: list[str] = []
    if high_summary["n"] < 25:
        reasons.append("too_few_high_actionability_rows")
    if high_summary["pp_mean"] < 0.025:
        reasons.append("high_actionability_pp_below_0p025")
    if observed_margin < 0.015:
        reasons.append("high_vs_low_margin_below_0p015")
    if high_summary["pp_ci95"][0] <= 0.0:
        reasons.append("high_actionability_ci_lower_not_positive")
    if high_summary["dataset_min_pp"] < -0.010:
        reasons.append("dataset_min_below_minus_0p010")
    if high_summary["hard_harm_frac"] > 0.20:
        reasons.append("hard_harm_frac_above_0p20")
    if high_summary["mmd_max"] > 0.001:
        reasons.append("mmd_max_above_0p001")
    if high_summary["negative_dataset_tail_count"] > 0:
        reasons.append("negative_dataset_tail_present")
    if shuffle_margin < 0.015 or shuffle_p > 0.05:
        reasons.append("shuffled_actionability_control_not_separated")
    if footprint["test_single"] < 25 or footprint["family_gene"] < 25:
        reasons.append("canonical_metadata_footprint_too_small")

    status = "background_target_actionability_pass_cpu_review_next_no_gpu" if not reasons else "background_target_actionability_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "train_only_internal_target_activity_rows": True,
            "canonical_metadata_footprint_only": True,
            "canonical_performance_used": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
        },
        "threshold_q75": threshold,
        "high_actionability": high_summary,
        "low_actionability": low_summary,
        "controls": {
            "shuffle_mean": shuffle_mean,
            "shuffle_p95": shuffle_p95,
            "high_minus_low_margin": observed_margin,
            "high_minus_shuffle_mean_margin": shuffle_margin,
            "shuffle_p": shuffle_p,
        },
        "canonical_metadata_footprint": footprint,
        "reasons": reasons,
        "next_action": (
            "external review and implementation/launcher gate before any bounded GPU"
            if not reasons
            else "do not launch background-target-actionability GPU; target actionability repeats target-observability failure pattern"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "condition",
            "pp_delta",
            "mmd_delta",
            "actionability_score",
            "target_expr_mean",
            "target_expr_nonzero_fraction",
            "prior_go",
            "prior_reactome",
            "prior_corum",
            "prior_omnipath",
            "high_actionability",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in fieldnames if k != "high_actionability"}, "high_actionability": float(row["actionability_score"]) >= threshold})

    lines = [
        "# LatentFM Background-Conditional Target-Actionability Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only target-activity/actionability gate over completed train-only rows.",
        "- Uses frozen GOA/Reactome/CORUM/OmniPath and S0 provenance.",
        "- Canonical is used only for metadata footprint, not performance.",
        "- No canonical multi, Track C query, training, inference, or GPU.",
        "",
        "## Summary",
        "",
        f"- high-actionability rows: `{high_summary['n']}`",
        f"- high pp mean: `{high_summary['pp_mean']}`",
        f"- high pp CI95: `{high_summary['pp_ci95']}`",
        f"- high dataset min pp: `{high_summary['dataset_min_pp']}`",
        f"- high hard-harm fraction: `{high_summary['hard_harm_frac']}`",
        f"- high MMD max: `{high_summary['mmd_max']}`",
        f"- low pp mean: `{low_summary['pp_mean']}`",
        f"- high-vs-low margin: `{observed_margin}`",
        f"- high-vs-shuffle mean margin: `{shuffle_margin}`",
        f"- shuffle p: `{shuffle_p}`",
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
