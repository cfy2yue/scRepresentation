#!/usr/bin/env python3
"""CPU gate for an orthogonalized soft-archetype router.

The previous conditional router found useful pockets but failed because the
router did not cleanly separate true archetype signal from shuffled-state
controls.  This gate keeps the same train-only/internal proxy setting and tests
a stricter rule: use archetype predictions for a dataset only when train rows
show that archetype beats both dataset_mean and shuffled-state predictions, and
the shuffled-state predictor itself does not beat dataset_mean.

This is CPU-only.  It does not read canonical test, canonical multi, held-out
query, or active GPU outputs.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
sys.path.insert(0, str(OPS))

from audit_latentfm_soft_archetype_conditional_router_cpu_gate_20260623 import (  # noqa: E402
    CANDIDATE,
    DEFAULT_DATA_DIR,
    DEFAULT_GENE_CACHE,
    DEFAULT_PERT_MEANS,
    DEFAULT_SPLIT,
    SHUFFLED,
    fmt,
    load_gene_embeddings,
    load_json,
    score_rows_with_soft_features,
)
from audit_latentfm_xverse_background_state_residual_consensus_gate_20260622 import (  # noqa: E402
    GROUPS,
    collect_rows,
    paired_bootstrap,
)


OUT_JSON = ROOT / "reports/latentfm_soft_archetype_orthogonal_router_cpu_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_ORTHOGONAL_ROUTER_CPU_GATE_20260623.md"
ROUTER = "orthogonal_archetype_router"
SHUFFLED_ROUTER = "orthogonal_shuffled_router"
PRIMARY_RULE = "train_candidate_ge_002_candidate_minus_shuffled_ge_002_shuffled_le_000"


def dataset_margins(train_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        grouped[str(row["dataset"])].append(row)
    out = {}
    for ds, rows in grouped.items():
        out[ds] = {
            "n_train": float(len(rows)),
            "candidate_vs_dataset_mean": mean(float(r[CANDIDATE]) - float(r["dataset_mean"]) for r in rows),
            "candidate_vs_shuffled": mean(float(r[CANDIDATE]) - float(r[SHUFFLED]) for r in rows),
            "shuffled_vs_dataset_mean": mean(float(r[SHUFFLED]) - float(r["dataset_mean"]) for r in rows),
            "candidate_vs_gene_only": mean(float(r[CANDIDATE]) - float(r["gene_only_ridge"]) for r in rows),
        }
    return out


def should_use_archetype(stats: dict[str, float], rule: str) -> bool:
    if rule == PRIMARY_RULE:
        return (
            stats.get("n_train", 0.0) >= 12
            and stats.get("candidate_vs_dataset_mean", -999.0) >= 0.02
            and stats.get("candidate_vs_shuffled", -999.0) >= 0.02
            and stats.get("shuffled_vs_dataset_mean", 999.0) <= 0.00
        )
    if rule == "train_candidate_positive_candidate_minus_shuffled_positive_shuffled_le_000":
        return (
            stats.get("n_train", 0.0) >= 12
            and stats.get("candidate_vs_dataset_mean", -999.0) > 0.00
            and stats.get("candidate_vs_shuffled", -999.0) > 0.00
            and stats.get("shuffled_vs_dataset_mean", 999.0) <= 0.00
        )
    if rule == "train_candidate_ge_002_candidate_minus_shuffled_ge_001_shuffled_le_001":
        return (
            stats.get("n_train", 0.0) >= 12
            and stats.get("candidate_vs_dataset_mean", -999.0) >= 0.02
            and stats.get("candidate_vs_shuffled", -999.0) >= 0.01
            and stats.get("shuffled_vs_dataset_mean", 999.0) <= 0.01
        )
    raise ValueError(rule)


def apply_rule(rows: list[dict[str, Any]], margins: dict[str, dict[str, float]], rule: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ds = str(row["dataset"])
        stats = margins.get(ds, {})
        use = should_use_archetype(stats, rule)
        item = dict(row)
        item[ROUTER] = item[CANDIDATE] if use else item["dataset_mean"]
        use_shuffled = bool(
            stats.get("n_train", 0.0) >= 12
            and stats.get("shuffled_vs_dataset_mean", -999.0) >= 0.02
        )
        item[SHUFFLED_ROUTER] = item[SHUFFLED] if use_shuffled else item["dataset_mean"]
        item["router_used_archetype"] = bool(use)
        item["rule"] = rule
        out.append(item)
    return out


def dataset_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["group"]), str(row["dataset"]))].append(row)
    out = []
    for (group, ds), items in sorted(grouped.items()):
        deltas = [float(r[ROUTER]) - float(r["dataset_mean"]) for r in items]
        out.append(
            {
                "group": group,
                "dataset": ds,
                "n": len(items),
                "delta_vs_dataset_mean": mean(deltas),
                "negative_fraction": sum(x < 0.0 for x in deltas) / max(len(deltas), 1),
                "used_fraction": sum(bool(r["router_used_archetype"]) for r in items) / max(len(items), 1),
            }
        )
    return out


def summarize_rule(rows: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    paired = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        for baseline in ("dataset_mean", "gene_only_ridge", CANDIDATE, SHUFFLED_ROUTER):
            row = paired_bootstrap(group_rows, ROUTER, baseline, n_boot=2000, seed=5300 + len(paired))
            row["group"] = group
            row["baseline"] = baseline
            paired.append(row)
    effects = dataset_effects(rows)
    return {
        "rule": rule,
        "coverage_fraction": sum(bool(r["router_used_archetype"]) for r in rows) / max(len(rows), 1),
        "paired_deltas": paired,
        "dataset_effects": effects,
        "worst_dataset_effects": sorted(effects, key=lambda r: float(r["delta_vs_dataset_mean"]))[:10],
        "best_dataset_effects": sorted(effects, key=lambda r: float(r["delta_vs_dataset_mean"]), reverse=True)[:10],
    }


def decide(primary: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if float(primary["coverage_fraction"]) < 0.05:
        reasons.append("primary_rule_coverage_below_0p05")
    paired = {(r["group"], r["baseline"]): r for r in primary["paired_deltas"]}
    for group in GROUPS:
        ds = paired.get((group, "dataset_mean")) or {}
        if ds.get("status") != "ok" or float(ds.get("delta_mean") or -999.0) < 0.02:
            reasons.append(f"{group}_dataset_mean_delta_below_0p02")
        if float(ds.get("p_harm") if ds.get("p_harm") is not None else 1.0) > 0.20:
            reasons.append(f"{group}_dataset_mean_p_harm_above_0p20")
        if ds.get("leave_one_min") is None or float(ds["leave_one_min"]) < -0.02:
            reasons.append(f"{group}_leave_one_dataset_below_minus_0p02")
        gene = paired.get((group, "gene_only_ridge")) or {}
        if gene.get("status") != "ok" or float(gene.get("delta_mean") or -999.0) < 0.0:
            reasons.append(f"{group}_does_not_beat_gene_only")
        shuf = paired.get((group, SHUFFLED_ROUTER)) or {}
        if shuf.get("status") != "ok" or float(shuf.get("delta_mean") or -999.0) < 0.01:
            reasons.append(f"{group}_shuffled_control_not_separated")
    for row in primary["dataset_effects"]:
        if float(row["used_fraction"]) > 0.0 and (
            float(row["delta_vs_dataset_mean"]) < -0.02 or float(row["negative_fraction"]) > 0.50
        ):
            reasons.append(f"used_dataset_harm_{row['group']}_{row['dataset']}")
    return {
        "status": "soft_archetype_orthogonal_router_cpu_gate_pass_authorize_one_capped_smoke"
        if not reasons
        else "soft_archetype_orthogonal_router_cpu_gate_fail_no_gpu",
        "gpu_authorization": "one_capped_smoke" if not reasons else "none",
        "primary_rule": PRIMARY_RULE,
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Soft-Archetype Orthogonal Router CPU Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"GPU authorization: `{payload['decision']['gpu_authorization']}`",
        "",
        "## Hypothesis",
        "",
        "Soft archetype signal is useful only where train-only margins show signal beyond shuffled-state controls.",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- primary rule: `{payload['decision']['primary_rule']}`",
        f"- train rows: `{payload['n_train_rows']}`",
        f"- validation rows: `{payload['n_val_rows']}`",
        "",
        "## Rule Summary",
        "",
        "| rule | coverage | primary status/reasons |",
        "|---|---:|---|",
    ]
    for rule in payload["rules"]:
        status = payload["decision"]["status"] if rule["rule"] == payload["decision"]["primary_rule"] else "diagnostic_only"
        reasons = ";".join(payload["decision"]["reasons"]) if rule["rule"] == payload["decision"]["primary_rule"] else "not eligible"
        lines.append(f"| {rule['rule']} | {fmt(rule['coverage_fraction'])} | {status}: {reasons or 'none'} |")

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
        "## Worst Used Dataset Effects",
        "",
        "| group | dataset | n | used frac | delta vs dataset_mean | neg frac |",
        "|---|---|---:|---:|---:|---:|",
    ]
    used = [r for r in primary["dataset_effects"] if float(r["used_fraction"]) > 0.0]
    for row in sorted(used, key=lambda r: float(r["delta_vs_dataset_mean"]))[:12]:
        lines.append(
            f"| {row['group']} | {row['dataset']} | {row['n']} | {fmt(row['used_fraction'])} | "
            f"{fmt(row['delta_vs_dataset_mean'])} | {fmt(row['negative_fraction'])} |"
        )
    if not used:
        lines.append("| none | none | 0 | NA | NA | NA |")
    lines += [
        "",
        "## Decision Reasons",
        "",
    ]
    reasons = payload["decision"].get("reasons") or []
    lines.extend(f"- `{reason}`" for reason in reasons) if reasons else lines.append("- none")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Only the primary rule can authorize GPU; relaxed rules are diagnostics.",
        "- Failing keeps archetype CPU-only and preserves the negative evidence.",
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
    margins = dataset_margins(train_scored)
    rules = []
    for rule in (
        PRIMARY_RULE,
        "train_candidate_positive_candidate_minus_shuffled_positive_shuffled_le_000",
        "train_candidate_ge_002_candidate_minus_shuffled_ge_001_shuffled_le_001",
    ):
        rules.append(summarize_rule(apply_rule(val_scored, margins, rule), rule))
    primary = next(r for r in rules if r["rule"] == PRIMARY_RULE)
    payload = {
        "data_dir": str(data_dir),
        "split_file": str(DEFAULT_SPLIT),
        "pert_means_file": str(DEFAULT_PERT_MEANS),
        "leakage_status": "trainonly_internal_proxy_no_canonical_no_multi_no_query_no_active_gpu_artifacts",
        "candidate": CANDIDATE,
        "shuffled_control": SHUFFLED,
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "dataset_margin_summary": {
            "n_datasets": len(margins),
            "primary_rule_datasets": sorted(ds for ds, stats in margins.items() if should_use_archetype(stats, PRIMARY_RULE)),
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
