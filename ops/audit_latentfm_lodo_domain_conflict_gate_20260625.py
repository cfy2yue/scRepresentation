#!/usr/bin/env python3
"""Train-only LODO/domain-conflict feasibility gate for true-cell scaling."""

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
CANONICAL_ANCHOR = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_canonical/condition_family_eval_anchor_ode20_canonical.json"

OUT_JSON = ROOT / "reports/latentfm_lodo_domain_conflict_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_LODO_DOMAIN_CONFLICT_GATE_20260625.md"
OUT_CSV = ROOT / "reports/latentfm_lodo_domain_conflict_rows_20260625.csv"


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


def load_s0() -> dict[tuple[str, str], dict[str, str]]:
    meta: dict[tuple[str, str], dict[str, str]] = {}
    with S0_TSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect="excel-tab")
        for row in reader:
            ds = row.get("dataset") or ""
            cond = row.get("condition") or ""
            if ds and cond:
                meta[(ds, cond)] = row
    return meta


def bootstrap_ci(values: list[float], *, n_boot: int = 1000, seed: int = 20260625) -> tuple[float, float, float]:
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    stats: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        stats.append(mean(values[rng.randrange(n)] for _ in range(n)))
    stats.sort()
    low = stats[int(0.025 * (n_boot - 1))]
    high = stats[int(0.975 * (n_boot - 1))]
    p_le_zero = (1 + sum(x <= 0.0 for x in stats)) / (1 + len(stats))
    return low, high, p_le_zero


def main() -> None:
    meta = load_s0()
    per_key: dict[tuple[str, str], dict[str, Any]] = {}
    for seed in (42, 43, 44):
        pdir = POSTHOC_ROOT / f"xverse_truecell_nested_budget128_tailstable_seed{seed}_6000/posthoc_eval_internal"
        anchor_p = pdir / "condition_family_eval_anchor_internal_ode20.json"
        cand_p = pdir / "condition_family_eval_candidate_internal_ode20.json"
        anchor = load_json(anchor_p)
        cand = load_json(cand_p)
        arows = rows_by_key(anchor)
        crows = rows_by_key(cand)
        for key in sorted(set(arows) & set(crows)):
            pp = metric_delta(crows[key], arows[key], "pearson_pert")
            mmd = metric_delta(crows[key], arows[key], "test_mmd_clamped")
            if pp is None or mmd is None:
                continue
            item = per_key.setdefault(
                key,
                {
                    "dataset": key[0],
                    "condition": key[1],
                    "pp_by_seed": {},
                    "mmd_by_seed": {},
                },
            )
            item["pp_by_seed"][seed] = pp
            item["mmd_by_seed"][seed] = mmd

    rows: list[dict[str, Any]] = []
    for key, item in per_key.items():
        if len(item["pp_by_seed"]) < 3 or len(item["mmd_by_seed"]) < 3:
            continue
        m = meta.get(key, {})
        pp_vals = [float(item["pp_by_seed"][seed]) for seed in (42, 43, 44)]
        mmd_vals = [float(item["mmd_by_seed"][seed]) for seed in (42, 43, 44)]
        rows.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "pp_mean": mean(pp_vals),
                "pp_min_seed": min(pp_vals),
                "mmd_mean": mean(mmd_vals),
                "mmd_max_seed": max(mmd_vals),
                "hard_harm": min(pp_vals) < -0.05 or max(mmd_vals) > 0.010,
                "dataset_domain": key[0],
                "source_label": m.get("source_label") or "unknown",
                "perturbation_type": m.get("perturbation_type") or "unknown",
                "cell_background_source": m.get("cell_background_source") or "unknown",
            }
        )

    domain_summaries: dict[str, list[dict[str, Any]]] = {}
    for domain in ("dataset_domain", "source_label", "perturbation_type", "cell_background_source"):
        by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_level[str(row[domain])].append(row)
        out = []
        for level, level_rows in sorted(by_level.items()):
            pp_vals = [float(r["pp_mean"]) for r in level_rows]
            mmd_vals = [float(r["mmd_mean"]) for r in level_rows]
            out.append(
                {
                    "level": level,
                    "n": len(level_rows),
                    "pp_mean": mean(pp_vals),
                    "mmd_mean": mean(mmd_vals),
                    "hard_harm_frac": mean([1.0 if r["hard_harm"] else 0.0 for r in level_rows]),
                }
            )
        domain_summaries[domain] = out

    pp_values = [float(r["pp_mean"]) for r in rows]
    mmd_values = [float(r["mmd_mean"]) for r in rows]
    hard_harm_frac = mean([1.0 if r["hard_harm"] else 0.0 for r in rows]) if rows else float("nan")
    ci_low, ci_high, p_le_zero = bootstrap_ci(pp_values)

    domain_gate: dict[str, Any] = {}
    for domain, summaries in domain_summaries.items():
        eligible = [s for s in summaries if int(s["n"]) >= 3]
        worst_pp = min((float(s["pp_mean"]) for s in eligible), default=float("nan"))
        worst_mmd = max((float(s["mmd_mean"]) for s in eligible), default=float("nan"))
        severe_levels = [s for s in eligible if float(s["pp_mean"]) < -0.010 or float(s["mmd_mean"]) > 0.0005]
        domain_gate[domain] = {
            "n_levels": len(summaries),
            "n_eligible_levels": len(eligible),
            "worst_pp_mean": worst_pp,
            "worst_mmd_mean": worst_mmd,
            "severe_or_mmd_harm_levels": severe_levels,
        }

    # Shuffled control: domain labels should matter if this is a real domain-conflict signal.
    rng = random.Random(20260625)
    shuffled_worst: dict[str, list[float]] = {d: [] for d in domain_summaries}
    for domain in domain_summaries:
        labels = [str(r[domain]) for r in rows]
        for _ in range(500):
            labels2 = labels[:]
            rng.shuffle(labels2)
            by_level: dict[str, list[float]] = defaultdict(list)
            for row, label in zip(rows, labels2):
                by_level[label].append(float(row["pp_mean"]))
            eligible = [vals for vals in by_level.values() if len(vals) >= 3]
            if eligible:
                shuffled_worst[domain].append(min(mean(vals) for vals in eligible))
    shuffle_p: dict[str, float | None] = {}
    for domain, values in shuffled_worst.items():
        observed = domain_gate[domain]["worst_pp_mean"]
        if not values or not math.isfinite(observed):
            shuffle_p[domain] = None
        else:
            # Low p would mean real domains are safer than random groupings.
            shuffle_p[domain] = (1 + sum(x >= observed for x in values)) / (1 + len(values))

    canonical = load_json(CANONICAL_ANCHOR)
    footprint = {
        group: len(canonical.get("groups", {}).get(group, {}).get("selected_conditions", []))
        for group in ("test_single", "family_gene")
    }

    reasons: list[str] = []
    if len(rows) < 100:
        reasons.append("too_few_rows")
    if mean(pp_values) < 0.010:
        reasons.append("overall_pp_mean_below_0p010")
    if ci_low <= 0.0:
        reasons.append("bootstrap_ci_low_not_positive")
    if max(mmd_values) > 0.010:
        reasons.append("row_mmd_tail_above_0p010")
    if hard_harm_frac > 0.10:
        reasons.append("hard_harm_fraction_above_0p10")
    for domain, item in domain_gate.items():
        if item["n_eligible_levels"] < 3:
            reasons.append(f"{domain}_too_few_eligible_levels")
        if item["worst_pp_mean"] < -0.010:
            reasons.append(f"{domain}_worst_pp_below_minus_0p010")
        if item["worst_mmd_mean"] > 0.0005:
            reasons.append(f"{domain}_worst_mmd_above_0p0005")
        if item["severe_or_mmd_harm_levels"]:
            reasons.append(f"{domain}_has_domain_conflict_levels")
        if shuffle_p.get(domain) is None or float(shuffle_p[domain]) > 0.05:
            reasons.append(f"{domain}_does_not_beat_shuffled_control")
    if footprint["test_single"] < 25 or footprint["family_gene"] < 25:
        reasons.append("canonical_metadata_footprint_too_small")

    pass_gate = not reasons
    status = "lodo_domain_conflict_pass_cpu_review_next_no_gpu" if pass_gate else "lodo_domain_conflict_fail_no_gpu"
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
            "overall_pp_mean": mean(pp_values) if pp_values else None,
            "overall_pp_ci95": [ci_low, ci_high],
            "overall_p_le_zero": p_le_zero,
            "overall_mmd_mean": mean(mmd_values) if mmd_values else None,
            "row_mmd_max": max(mmd_values) if mmd_values else None,
            "hard_harm_fraction": hard_harm_frac,
            "canonical_metadata_footprint": footprint,
        },
        "domain_gate": domain_gate,
        "shuffle_p": shuffle_p,
        "reasons": sorted(set(reasons)),
        "next_action": (
            "external review and implementation/launcher gate before any bounded GPU"
            if pass_gate
            else "do not launch LODO/domain-conflict GPU; current true-cell gains are not domain-consensus safe"
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
            "source_label",
            "perturbation_type",
            "cell_background_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    lines = [
        "# LatentFM LODO Domain-Conflict Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only train-only/internal row-delta gate for true-cell budget128 6k.",
        "- Canonical data are used only for metadata footprint, not performance.",
        "- No canonical multi, Track C query, training, inference, or GPU.",
        "",
        "## Overall",
        "",
        f"- rows: `{len(rows)}`",
        f"- overall pp mean: `{payload['summary']['overall_pp_mean']}`",
        f"- pp bootstrap CI95: `{payload['summary']['overall_pp_ci95']}`",
        f"- p(pp<=0): `{p_le_zero}`",
        f"- overall MMD mean: `{payload['summary']['overall_mmd_mean']}`",
        f"- row MMD max: `{payload['summary']['row_mmd_max']}`",
        f"- hard-harm fraction: `{hard_harm_frac}`",
        f"- canonical metadata footprint: `{footprint}`",
        "",
        "## Domain Summary",
        "",
        "| domain | levels | eligible | worst pp | worst MMD | shuffled p | conflict levels |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, item in domain_gate.items():
        lines.append(
            f"| `{domain}` | {item['n_levels']} | {item['n_eligible_levels']} | {item['worst_pp_mean']:+.6f} | {item['worst_mmd_mean']:+.6f} | {shuffle_p.get(domain)} | {len(item['severe_or_mmd_harm_levels'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{sorted(set(reasons))}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- row CSV: `{OUT_CSV}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
