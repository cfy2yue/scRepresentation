#!/usr/bin/env python3
"""CPU-only Track C Wessels absorbable-subset gate.

This diagnostic asks whether the corrected mc256 memory-transfer smokes contain
a support-only, train-time-identifiable Wessels subset that actually absorbed
the support teacher.  It reads only support-trainselect posthoc-derived
condition rows and train-only memory readout rows.  It does not read held-out
query or canonical outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_CONDITION_CSV = ROOT / "reports/latentfm_trackc_memory_transfer_bottleneck_conditions_20260622.csv"
DEFAULT_READOUT = ROOT / "reports/latentfm_trackc_trainonly_memory_readout_gate_20260622.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_wessels_absorbable_subset_gate_20260622.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_WESSELS_ABSORBABLE_SUBSET_GATE_20260622.md"
OUT_CSV = ROOT / "reports/latentfm_trackc_wessels_absorbable_subset_rules_20260622.csv"
SELECTED_MEMORY = "memory_jaccard_k3_same_ds_min0.25"
BOOT_N = 2000
SEED = 20260622


def as_float(value: Any) -> float | None:
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return float(value)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def load_condition_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_readout_features(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in payload.get("condition_rows") or []:
        out[(str(row["dataset"]), str(row["condition"]))] = row
    return out


def aggregate_conditions(rows: list[dict[str, Any]], readout: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["condition"]), str(row["finetune_trainable_scope"]))].append(row)

    out: list[dict[str, Any]] = []
    for (dataset, condition, scope), items in sorted(grouped.items()):
        deltas = [as_float(row.get("delta_pp")) for row in items]
        deltas = [x for x in deltas if x is not None]
        route_gaps = [as_float(row.get("route_gap_from_anchor_pp")) for row in items]
        route_gaps = [x for x in route_gaps if x is not None]
        memory_gaps = [as_float(row.get("memory_gap_from_anchor_pp")) for row in items]
        memory_gaps = [x for x in memory_gaps if x is not None]
        feat = readout.get((dataset, condition), {})
        support_route = as_float(feat.get("support_selected_route"))
        memory_pp = as_float(feat.get(SELECTED_MEMORY))
        dataset_multi = as_float(feat.get("dataset_multi_mean"))
        global_multi = as_float(feat.get("global_multi_mean"))
        additive_sum = as_float(feat.get("additive_single_sum"))
        dataset_single = as_float(feat.get("dataset_single_mean"))
        memory_advantage = None
        if memory_pp is not None and support_route is not None:
            memory_advantage = memory_pp - support_route
        route_vs_additive = None
        if support_route is not None and additive_sum is not None:
            route_vs_additive = support_route - additive_sum
        out.append(
            {
                "dataset": dataset,
                "condition": condition,
                "scope": scope,
                "n_runs": len(items),
                "mean_delta_pp": mean(deltas) if deltas else None,
                "max_delta_pp": max(deltas) if deltas else None,
                "min_delta_pp": min(deltas) if deltas else None,
                "mean_route_gap": mean(route_gaps) if route_gaps else None,
                "mean_memory_gap": mean(memory_gaps) if memory_gaps else None,
                "weighted_route_gap_closure": (
                    sum(deltas) / sum(route_gaps)
                    if deltas and route_gaps and abs(sum(route_gaps)) > 1e-12
                    else None
                ),
                "weighted_memory_gap_closure": (
                    sum(deltas) / sum(memory_gaps)
                    if deltas and memory_gaps and abs(sum(memory_gaps)) > 1e-12
                    else None
                ),
                "support_selected_route": support_route,
                f"{SELECTED_MEMORY}": memory_pp,
                "dataset_multi_mean": dataset_multi,
                "global_multi_mean": global_multi,
                "additive_single_sum": additive_sum,
                "dataset_single_mean": dataset_single,
                "memory_advantage": memory_advantage,
                "route_vs_additive": route_vs_additive,
                "genes": items[0].get("genes"),
            }
        )
    return out


def bootstrap_harm(values: list[float], *, seed: int) -> dict[str, Any]:
    if not values:
        return {"p_harm": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    boot = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, len(vals), size=len(vals))
        boot.append(float(vals[idx].mean()))
    arr = np.asarray(boot, dtype=float)
    return {
        "p_harm": float(np.mean(arr < 0.0)),
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
    }


def rule_metrics(rows: list[dict[str, Any]], rule_name: str, selected: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    deltas = [as_float(row.get("mean_delta_pp")) for row in selected]
    deltas = [x for x in deltas if x is not None]
    route_gaps = [as_float(row.get("mean_route_gap")) for row in selected]
    route_gaps = [x for x in route_gaps if x is not None and x > 0]
    boot = bootstrap_harm(deltas, seed=seed)
    return {
        "rule": rule_name,
        "n_conditions": len(selected),
        "mean_delta_pp": mean(deltas) if deltas else None,
        "min_condition_delta_pp": min(deltas) if deltas else None,
        "max_condition_delta_pp": max(deltas) if deltas else None,
        "weighted_route_gap_closure": (
            sum(deltas) / sum(route_gaps)
            if deltas and route_gaps and abs(sum(route_gaps)) > 1e-12
            else None
        ),
        "bootstrap_p_harm": boot["p_harm"],
        "bootstrap_ci95": boot["ci95"],
        "conditions": [str(row["condition"]) for row in selected],
    }


def candidate_rules(rows: list[dict[str, Any]], *, seed_base: int) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    scopes = sorted({str(row["scope"]) for row in rows})
    features = [
        "support_selected_route",
        SELECTED_MEMORY,
        "memory_advantage",
        "dataset_multi_mean",
        "global_multi_mean",
        "additive_single_sum",
        "dataset_single_mean",
        "route_vs_additive",
        "mean_route_gap",
        "mean_memory_gap",
    ]
    for scope in scopes:
        scope_rows = [row for row in rows if row["scope"] == scope]
        rules.append(rule_metrics(scope_rows, f"{scope}:all_wessels", scope_rows, seed=seed_base + len(rules)))
        sorted_by_delta = sorted(
            scope_rows,
            key=lambda row: as_float(row.get("mean_delta_pp")) if as_float(row.get("mean_delta_pp")) is not None else -999,
            reverse=True,
        )
        for k in (3, 4, 5):
            rules.append(
                rule_metrics(
                    scope_rows,
                    f"{scope}:oracle_top{k}_by_absorbed_delta",
                    sorted_by_delta[:k],
                    seed=seed_base + len(rules),
                )
            )
        for feature in features:
            vals = [as_float(row.get(feature)) for row in scope_rows]
            finite_vals = sorted(v for v in vals if v is not None)
            if len(finite_vals) < 3:
                continue
            thresholds = sorted(
                set(
                    [
                        finite_vals[len(finite_vals) // 4],
                        finite_vals[len(finite_vals) // 2],
                        finite_vals[(3 * len(finite_vals)) // 4],
                        0.0,
                    ]
                )
            )
            for threshold in thresholds:
                for op in ("ge", "le"):
                    if op == "ge":
                        selected = [row for row in scope_rows if (as_float(row.get(feature)) is not None and as_float(row.get(feature)) >= threshold)]
                        rule_name = f"{scope}:{feature}>={threshold:.6g}"
                    else:
                        selected = [row for row in scope_rows if (as_float(row.get(feature)) is not None and as_float(row.get(feature)) <= threshold)]
                        rule_name = f"{scope}:{feature}<={threshold:.6g}"
                    if len(selected) >= 3:
                        rules.append(rule_metrics(scope_rows, rule_name, selected, seed=seed_base + len(rules)))
    return rules


def passes_rule(rule: dict[str, Any]) -> bool:
    return (
        int(rule["n_conditions"]) >= 3
        and (as_float(rule.get("mean_delta_pp")) or -999) >= 0.02
        and (as_float(rule.get("weighted_route_gap_closure")) or -999) >= 0.05
        and (as_float(rule.get("min_condition_delta_pp")) or -999) >= -0.005
        and (as_float(rule.get("bootstrap_p_harm")) if rule.get("bootstrap_p_harm") is not None else 1.0) <= 0.20
    )


def decide(rules: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [rule for rule in rules if passes_rule(rule)]
    best = sorted(
        rules,
        key=lambda row: (
            as_float(row.get("mean_delta_pp")) if as_float(row.get("mean_delta_pp")) is not None else -999,
            as_float(row.get("weighted_route_gap_closure")) if as_float(row.get("weighted_route_gap_closure")) is not None else -999,
        ),
        reverse=True,
    )[0]
    oracle = [rule for rule in rules if "oracle_top3_by_absorbed_delta" in rule["rule"]]
    best_oracle_top3 = max((as_float(row.get("mean_delta_pp")) or -999 for row in oracle), default=None)
    reasons = []
    if not passing:
        reasons.append("no_trainonly_rule_meets_wessels_subset_absorption_gate")
    if best_oracle_top3 is None or best_oracle_top3 < 0.02:
        reasons.append("oracle_top3_absorbed_delta_below_0p02")
    status = "wessels_absorbable_subset_gate_pass" if passing else "wessels_absorbable_subset_gate_fail"
    return {
        "status": status,
        "reasons": reasons,
        "n_passing_rules": len(passing),
        "best_rule": best,
        "best_oracle_top3_mean_delta": best_oracle_top3,
    }


def write_rules_csv(rules: list[dict[str, Any]], path: Path) -> None:
    keys = [
        "rule",
        "n_conditions",
        "mean_delta_pp",
        "min_condition_delta_pp",
        "max_condition_delta_pp",
        "weighted_route_gap_closure",
        "bootstrap_p_harm",
        "bootstrap_ci95",
        "conditions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rules:
            out = dict(row)
            out["bootstrap_ci95"] = json.dumps(out.get("bootstrap_ci95"))
            out["conditions"] = ";".join(out.get("conditions") or [])
            writer.writerow({key: out.get(key) for key in keys})


def write_md(decision: dict[str, Any], rules: list[dict[str, Any]], path: Path) -> None:
    top = sorted(
        rules,
        key=lambda row: as_float(row.get("mean_delta_pp")) if as_float(row.get("mean_delta_pp")) is not None else -999,
        reverse=True,
    )[:16]
    lines = [
        "# Track C Wessels Absorbable-Subset CPU Gate",
        "",
        "CPU-only diagnostic over corrected mc256 memory-transfer condition rows.",
        "Inputs are support-trainselect posthoc-derived condition rows and train-only memory readout rows.",
        "Held-out query and canonical outputs are not read.",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Gate",
        "",
        "A candidate Wessels subset rule must satisfy all of:",
        "",
        "- at least 3 Wessels support conditions;",
        "- mean absorbed pp delta `>= +0.02`;",
        "- weighted route-gap closure `>= +0.05`;",
        "- no selected condition mean delta `< -0.005`;",
        "- bootstrap harm probability `<= 0.20`.",
        "",
        "Rules use only train-only/readout features. Oracle top-k rows are shown as an upper bound and are not deployable.",
        "",
        "## Decision Evidence",
        "",
        f"- passing rules: `{decision['n_passing_rules']}`",
        f"- best rule: `{decision['best_rule']['rule']}`",
        f"- best rule mean delta: `{fmt(as_float(decision['best_rule'].get('mean_delta_pp')))}`",
        f"- best rule route-gap closure: `{fmt(as_float(decision['best_rule'].get('weighted_route_gap_closure')))}`",
        f"- best oracle-top3 mean delta: `{fmt(as_float(decision.get('best_oracle_top3_mean_delta')))}`",
    ]
    if decision["reasons"]:
        lines.append(f"- reasons: `{';'.join(decision['reasons'])}`")
    lines.extend(
        [
            "",
            "## Top Rules By Mean Delta",
            "",
            "| rule | n | mean delta | min delta | route closure | p_harm | conditions |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in top:
        p_harm = as_float(row.get("bootstrap_p_harm"))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['rule']}`",
                    str(row["n_conditions"]),
                    fmt(as_float(row.get("mean_delta_pp"))),
                    fmt(as_float(row.get("min_condition_delta_pp"))),
                    fmt(as_float(row.get("weighted_route_gap_closure"))),
                    "NA" if p_harm is None else f"{p_harm:.4f}",
                    ", ".join(row.get("conditions") or []),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is not a query evaluation and cannot promote a Track C model.",
            "- Failure closes Wessels subset/focused absorption as a GPU unlock route for the completed memory-transfer family.",
            "- A future Track C GPU branch would need a genuinely different mechanism, not another endpoint/replay/memory-dose sweep.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-csv", default=str(DEFAULT_CONDITION_CSV))
    parser.add_argument("--readout-json", default=str(DEFAULT_READOUT))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--out-rules-csv", default=str(OUT_CSV))
    args = parser.parse_args()

    raw_rows = load_condition_rows(Path(args.condition_csv))
    readout = load_readout_features(Path(args.readout_json))
    aggregate = aggregate_conditions(raw_rows, readout)
    wessels = [row for row in aggregate if row["dataset"] == "Wessels"]
    rules = candidate_rules(wessels, seed_base=SEED)
    decision = decide(rules)
    payload = {
        "status": decision["status"],
        "decision": decision,
        "n_raw_rows": len(raw_rows),
        "n_condition_scope_rows": len(aggregate),
        "n_wessels_condition_scope_rows": len(wessels),
        "inputs": {
            "condition_csv": str(args.condition_csv),
            "readout_json": str(args.readout_json),
        },
        "heldout_query_used": False,
        "canonical_outputs_used": False,
        "rules": rules,
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rules_csv(rules, Path(args.out_rules_csv))
    write_md(decision, rules, Path(args.out_md))
    print(json.dumps({"status": decision["status"], "n_rules": len(rules), "out_md": args.out_md}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
