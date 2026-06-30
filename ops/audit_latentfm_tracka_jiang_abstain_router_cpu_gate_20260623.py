#!/usr/bin/env python3
"""CPU gate for a distinct Track A Jiang/cytokine abstain router.

This uses only the frozen scFoundation train-only internal proxy rows. It does
not read canonical outputs, canonical multi, Track C query, or active runs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_latentfm_tracka_scf_guarded_fallback_cpu_gate_20260623 import (
    GROUPS,
    paired_bootstrap,
)


ROOT = Path("/data/cyx/1030/scLatent")
CPU_JSON = ROOT / "reports/latentfm_crosslatent_scfoundation_gene_reliability_router_gate_20260622.json"
LOWCOUNT_JSON = ROOT / "reports/latentfm_tracka_scf_jiang_lowcount_mask_cpu_gate_20260623.json"
DATASET_NEG_JSON = ROOT / "reports/latentfm_tracka_scf_guarded_fallback_cpu_gate_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_tracka_jiang_abstain_router_cpu_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_JIANG_ABSTAIN_ROUTER_CPU_GATE_20260623.md"

BASE = "shrink_k2"
FOCUS = {"Jiang_IFNG", "Jiang_TNFA"}
CYTOKINE = {"Jiang_IFNB", "Jiang_IFNG", "Jiang_TGFB", "Jiang_TNFA"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("group") == group]


def dataset_margin(rows: list[dict[str, Any]], model: str = BASE) -> dict[str, float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(float(row[model]) - float(row["dataset_mean"]))
    return {ds: float(np.mean(vals)) for ds, vals in by_ds.items()}


def apply_rule(rows: list[dict[str, Any]], rule: str, margins: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        ds = str(row["dataset"])
        count = int(row.get("gene_train_count") or 0)
        margin = float(margins.get(ds, 0.0))
        fallback = False
        if rule == "focus_margin_or_lowcount":
            fallback = ds in FOCUS and (margin < 0.0 or count <= 1)
        elif rule == "cytokine_margin_negative":
            fallback = ds in CYTOKINE and margin < 0.0
        elif rule == "cytokine_margin_or_lowcount":
            fallback = ds in CYTOKINE and (margin < 0.0 or count <= 1)
        elif rule == "focus_strict_margin_negative":
            fallback = ds in FOCUS and margin < 0.0
        else:
            raise ValueError(rule)
        item[rule] = float(row["dataset_mean"] if fallback else row[BASE])
        item[f"{rule}_fallback_used"] = fallback
        out.append(item)
    return out


def policy_delta_by_dataset(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(float(row[candidate]) - float(row[baseline]))
    return {ds: float(np.mean(vals)) for ds, vals in sorted(by_ds.items())}


def best_failed_policy_summary(path: Path, group: str, baseline: str = "dataset_mean") -> float:
    payload = load(path)
    group_summary = next(row for row in payload["group_summaries"] if row["group"] == group)
    paired = group_summary["paired_deltas"]
    row = next(item for item in paired if item["candidate"] == "guarded_fallback" and item["baseline"] == baseline)
    return float(row["delta_mean"])


def evaluate_rule(rows: list[dict[str, Any]], rule: str, margins: dict[str, float], group: str, seed: int) -> dict[str, Any]:
    scored = apply_rule(rows, rule, margins)
    paired = [
        paired_bootstrap(scored, rule, baseline, n_boot=2000, seed=seed + i)
        for i, baseline in enumerate(("dataset_mean", "gene_raw_mean", "global_mean", BASE))
    ]
    by_dataset_vs_dataset = policy_delta_by_dataset(scored, rule, "dataset_mean")
    return {
        "group": group,
        "rule": rule,
        "fallback_counts_by_dataset": dict(
            sorted(
                {
                    ds: sum(1 for row in scored if str(row["dataset"]) == ds and row[f"{rule}_fallback_used"])
                    for ds in sorted({str(row["dataset"]) for row in scored})
                }.items()
            )
        ),
        "paired_deltas": paired,
        "by_dataset_vs_dataset_mean": by_dataset_vs_dataset,
    }


def rule_delta(result: dict[str, Any], baseline: str) -> float:
    row = next(item for item in result["paired_deltas"] if item["baseline"] == baseline)
    return float(row["delta_mean"])


def rule_p_harm(result: dict[str, Any], baseline: str) -> float:
    row = next(item for item in result["paired_deltas"] if item["baseline"] == baseline)
    return float(row["p_harm"])


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    selected_by_group = {}
    for group in GROUPS:
        group_results = [row for row in results if row["group"] == group]
        selected = max(group_results, key=lambda row: rule_delta(row, "dataset_mean"))
        selected_by_group[group] = selected["rule"]
        lowcount_delta = best_failed_policy_summary(LOWCOUNT_JSON, group)
        dataset_neg_delta = best_failed_policy_summary(DATASET_NEG_JSON, group)
        selected_delta = rule_delta(selected, "dataset_mean")
        if selected_delta < 0.02:
            reasons.append(f"{group}_dataset_mean_delta_below_0p02")
        if selected_delta < lowcount_delta + 0.002:
            reasons.append(f"{group}_does_not_beat_lowcount_by_0p002")
        if selected_delta < dataset_neg_delta + 0.002:
            reasons.append(f"{group}_does_not_beat_dataset_negative_by_0p002")
        if rule_p_harm(selected, "dataset_mean") > 0.20:
            reasons.append(f"{group}_dataset_mean_harm_risk")
        focus_deltas = selected["by_dataset_vs_dataset_mean"]
        for ds in sorted(FOCUS):
            if float(focus_deltas.get(ds, 0.0)) < -0.01:
                reasons.append(f"{group}_{ds}_delta_below_minus_0p01")
        non_focus = [v for ds, v in focus_deltas.items() if ds not in FOCUS]
        if min(non_focus) < -0.02:
            reasons.append(f"{group}_non_focus_dataset_below_minus_0p02")
    status = "tracka_jiang_abstain_router_cpu_gate_pass_code_gate_next" if not reasons else "tracka_jiang_abstain_router_cpu_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "none",
        "selected_rules_by_group": selected_by_group,
        "reasons": reasons,
    }


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):+.6f}"


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A Jiang/Cytokine Abstain Router CPU Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Scope",
        "",
        "This gate reads only the frozen scFoundation train-only internal proxy rows.",
        "It does not read canonical outputs, canonical multi, Track C query, or active run artifacts.",
        "",
        "## Rule Results",
        "",
        "| group | rule | delta vs dataset_mean | p harm | delta vs shrink_k2 | IFNG delta | TNFA delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        delta_dataset = rule_delta(row, "dataset_mean")
        p_harm = rule_p_harm(row, "dataset_mean")
        delta_base = rule_delta(row, BASE)
        by_ds = row["by_dataset_vs_dataset_mean"]
        lines.append(
            f"| {row['group']} | {row['rule']} | {fmt(delta_dataset)} | {fmt(p_harm)} | "
            f"{fmt(delta_base)} | {fmt(by_ds.get('Jiang_IFNG'))} | {fmt(by_ds.get('Jiang_TNFA'))} |"
        )
    lines += [
        "",
        "## Failed Policy Baselines",
        "",
        "| group | lowcount delta | dataset-negative delta |",
        "|---|---:|---:|",
    ]
    for group in GROUPS:
        lines.append(
            f"| {group} | {fmt(best_failed_policy_summary(LOWCOUNT_JSON, group))} | "
            f"{fmt(best_failed_policy_summary(DATASET_NEG_JSON, group))} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Decision",
        "",
        "Passing would only authorize a code/provenance gate, not immediate GPU.",
        "Failure means Track A Jiang rescue needs a materially different router or should stay closed.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    cpu = load(CPU_JSON)
    rows = cpu["val_condition_rows"]
    rules = [
        "focus_margin_or_lowcount",
        "cytokine_margin_negative",
        "cytokine_margin_or_lowcount",
        "focus_strict_margin_negative",
    ]
    results = []
    for gi, group in enumerate(GROUPS):
        rows_g = group_rows(rows, group)
        margins = dataset_margin(rows_g)
        for ri, rule in enumerate(rules):
            results.append(evaluate_rule(rows_g, rule, margins, group, seed=42 + gi * 100 + ri * 10))
    payload = {
        "cpu_gate_json": str(CPU_JSON),
        "failed_policy_jsons": {
            "lowcount": str(LOWCOUNT_JSON),
            "dataset_negative": str(DATASET_NEG_JSON),
        },
        "leakage_status": "trainonly_internal_proxy_no_canonical_no_multi_no_query_no_active_run",
        "candidate_rules": rules,
        "results": results,
    }
    payload["decision"] = decide(results)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
