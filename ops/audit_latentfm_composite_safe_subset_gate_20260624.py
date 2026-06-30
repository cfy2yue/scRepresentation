#!/usr/bin/env python3
"""CPU gate: can support + signed-neighborhood features define a safe subset?"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import audit_latentfm_control_state_support_gate_20260624 as support_gate
import audit_latentfm_signed_neighborhood_consistency_gate_20260624 as signed_gate


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_composite_safe_subset_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_COMPOSITE_SAFE_SUBSET_GATE_20260624.md"
GROUPS = support_gate.GROUPS
SEED = 42
BOOT_N = 1000
SUPPORT_MAIN_ROWS_CACHE: list[Any] | None = None
SIGNED_ROWS_CACHE: dict[str, list[Any]] = {}

CORE_SUPPORT_FEATURES = (
    "support_ds_pct_gt_to_ctrl_nn_median",
    "support_ds_z_gt_to_ctrl_nn_median",
    "support_ds_pct_gt_to_ctrl_nn_p90",
    "support_coverage_frac",
    "support_ds_pct_ctrl_effective_rank",
)
CORE_SIGNED_FEATURES = (
    "signed_anchor_delta_cos_consensus",
    "signed_candidate_delta_cos_consensus",
    "signed_update_cos_consensus",
    "signed_update_projection",
    "signed_agreement",
)


@dataclass(frozen=True)
class Row:
    group: str
    dataset: str
    condition: str
    features: dict[str, float]
    delta_pp: float
    delta_mmd: float


@dataclass(frozen=True)
class Clause:
    feature: str
    op: str
    threshold: float


@dataclass(frozen=True)
class Rule:
    name: str
    clauses: tuple[Clause, ...]
    alpha_true: float
    alpha_false: float


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(SEED)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(len(arr)) for _ in arr]
        means.append(float(np.mean(arr[idx])))
    means_arr = np.asarray(means, dtype=np.float64)
    return float(np.quantile(means_arr, 0.025)), float(np.quantile(means_arr, 0.975)), float(np.mean(means_arr < 0.0))


def prefixed_features(obj: Any, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in obj.features.items() if np.isfinite(float(v))}


def support_rows(control: str) -> list[Any]:
    global SUPPORT_MAIN_ROWS_CACHE
    if SUPPORT_MAIN_ROWS_CACHE is None:
        features, _meta = support_gate.compute_features()
        SUPPORT_MAIN_ROWS_CACHE = [r for r in support_gate.metric_rows(features) if r.run == "cap120"]
    transform = control if control in {"shuffled", "inverted", "control_permuted"} else "main"
    return support_gate.transform(SUPPORT_MAIN_ROWS_CACHE, transform)


def signed_rows(control: str) -> list[Any]:
    signed_control = control if control in {"gene_shuffle", "sign_inverted", "feature_shuffle"} else "main"
    if signed_control not in SIGNED_ROWS_CACHE:
        SIGNED_ROWS_CACHE[signed_control] = signed_gate.build_rows(signed_control)
    return SIGNED_ROWS_CACHE[signed_control]


def combine_rows(control: str) -> list[Row]:
    support_control = {
        "main": "main",
        "support_shuffled": "shuffled",
        "support_inverted": "inverted",
        "support_permuted": "control_permuted",
        "all_feature_shuffle": "shuffled",
    }.get(control, "main")
    signed_control = {
        "main": "main",
        "signed_gene_shuffle": "gene_shuffle",
        "signed_sign_inverted": "sign_inverted",
        "signed_feature_shuffle": "feature_shuffle",
        "all_feature_shuffle": "feature_shuffle",
    }.get(control, "main")
    s_rows = support_rows(support_control)
    n_rows = signed_rows(signed_control)
    n_map = {(r.group, r.dataset, r.condition): r for r in n_rows}
    out = []
    for s in s_rows:
        key = (s.group, s.dataset, s.condition)
        n = n_map.get(key)
        if n is None:
            continue
        feats = {}
        feats.update(prefixed_features(s, "support"))
        feats.update(prefixed_features(n, "signed"))
        out.append(Row(s.group, s.dataset, s.condition, feats, s.delta_pp, s.delta_mmd))
    return out


def rule_hit(row: Row, rule: Rule) -> bool:
    if not rule.clauses:
        return True
    for clause in rule.clauses:
        value = row.features.get(clause.feature, float("nan"))
        if not np.isfinite(value):
            return False
        if clause.op == "<=" and not value <= clause.threshold:
            return False
        if clause.op == ">=" and not value >= clause.threshold:
            return False
    return True


def alpha_for(row: Row, rule: Rule) -> float:
    return rule.alpha_true if rule_hit(row, rule) else rule.alpha_false


def apply_rule(rows: list[Row], rule: Rule) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        alpha = alpha_for(row, rule)
        out.append(
            {
                "dataset": row.dataset,
                "delta_pp": float(alpha * row.delta_pp),
                "delta_mmd": float(alpha * row.delta_mmd),
                "alpha": float(alpha),
            }
        )
    return out


def summarize(applied: list[dict[str, Any]], *, with_bootstrap: bool = True) -> dict[str, float]:
    if not applied:
        return {k: float("nan") for k in ("mean_pp_delta", "ci95_low", "ci95_high", "bootstrap_p_harm", "condition_p_harm", "dataset_min_pp_delta", "mean_mmd_delta", "mean_alpha")} | {"n": 0}
    vals = np.asarray([float(r["delta_pp"]) for r in applied], dtype=np.float64)
    lo, hi, p_harm = bootstrap([float(v) for v in vals]) if with_bootstrap else (float("nan"), float("nan"), float(np.mean(vals < 0.0)))
    by_ds: dict[str, list[float]] = {}
    for row in applied:
        by_ds.setdefault(str(row["dataset"]), []).append(float(row["delta_pp"]))
    return {
        "n": len(applied),
        "mean_pp_delta": float(np.mean(vals)),
        "ci95_low": lo,
        "ci95_high": hi,
        "bootstrap_p_harm": p_harm,
        "condition_p_harm": float(np.mean(vals < 0.0)),
        "dataset_min_pp_delta": float(min(sum(v) / len(v) for v in by_ds.values())),
        "mean_mmd_delta": float(np.mean([float(r["delta_mmd"]) for r in applied])),
        "mean_alpha": float(np.mean([float(r["alpha"]) for r in applied])),
    }


def quantile_thresholds(rows: list[Row], feature: str) -> list[float]:
    vals = np.asarray([row.features.get(feature, float("nan")) for row in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 12 or float(np.max(vals) - np.min(vals)) <= 1e-12:
        return []
    return [float(np.quantile(vals, q)) for q in (0.25, 0.5, 0.75)]


def make_single_rules(rows: list[Row]) -> list[Rule]:
    rules = [
        Rule("noop", tuple(), 0.0, 0.0),
        Rule("all_candidate", tuple(), 1.0, 1.0),
    ]
    core = set(CORE_SUPPORT_FEATURES) | set(CORE_SIGNED_FEATURES)
    features = sorted(k for k in {k for row in rows for k in row.features} if k in core)
    for feature in features:
        for threshold in quantile_thresholds(rows, feature):
            for op in ("<=", ">="):
                for alpha_true, alpha_false in ((1.0, 0.0), (0.0, 1.0), (0.75, 0.0)):
                    name = f"{feature}_{op}_{threshold:.5g}_a{alpha_true:.2f}_{alpha_false:.2f}"
                    rules.append(Rule(name, (Clause(feature, op, threshold),), alpha_true, alpha_false))
    return rules


def make_pair_rules(rows: list[Row]) -> list[Rule]:
    rules = []
    support_features = [f for f in CORE_SUPPORT_FEATURES if any(f in row.features for row in rows)]
    signed_features = [f for f in CORE_SIGNED_FEATURES if any(f in row.features for row in rows)]
    for sf in support_features:
        for nf in signed_features:
            st_vals = quantile_thresholds(rows, sf)
            nt_vals = quantile_thresholds(rows, nf)
            if not st_vals or not nt_vals:
                continue
            st = st_vals[1]
            nt = nt_vals[1]
            for sop in ("<=", ">="):
                for nop in ("<=", ">="):
                    name = f"{sf}_{sop}_{st:.4g}__{nf}_{nop}_{nt:.4g}_a1.00_0.00"
                    rules.append(Rule(name, (Clause(sf, sop, st), Clause(nf, nop, nt)), 1.0, 0.0))
    return rules


def candidate_rules(rows: list[Row]) -> list[Rule]:
    return make_single_rules(rows) + make_pair_rules(rows)


def score(summary: dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        summary["mean_pp_delta"] - 5.0 * max(0.0, summary["mean_mmd_delta"]),
        summary["dataset_min_pp_delta"],
        -summary["bootstrap_p_harm"],
        -summary["condition_p_harm"],
        -abs(summary["mean_alpha"] - 0.5),
    )


def select_rule(train_rows: list[Row]) -> tuple[Rule, dict[str, float]]:
    best_rule = Rule("noop", tuple(), 0.0, 0.0)
    best_summary = summarize(apply_rule(train_rows, best_rule), with_bootstrap=False)
    best_score = score(best_summary)
    for rule in candidate_rules(train_rows):
        s = summarize(apply_rule(train_rows, rule), with_bootstrap=False)
        sc = score(s)
        if sc > best_score:
            best_rule, best_summary, best_score = rule, s, sc
    return best_rule, best_summary


def nested_lodo(rows: list[Row], control: str) -> dict[str, Any]:
    applied_all = []
    folds = []
    for heldout in sorted({r.dataset for r in rows}):
        train = [r for r in rows if r.dataset != heldout]
        test = [r for r in rows if r.dataset == heldout]
        if len(train) < 12 or not test:
            continue
        rule, train_summary = select_rule(train)
        applied = apply_rule(test, rule)
        test_summary = summarize(applied, with_bootstrap=False)
        applied_all.extend(applied)
        folds.append(
            {
                "heldout_dataset": heldout,
                "rule": rule.name,
                "train_mean_pp_delta": train_summary["mean_pp_delta"],
                "test_mean_pp_delta": test_summary["mean_pp_delta"],
                "test_mean_alpha": test_summary["mean_alpha"],
            }
        )
    top_rules: dict[str, int] = {}
    for fold in folds:
        top_rules[fold["rule"]] = top_rules.get(fold["rule"], 0) + 1
    return {
        "control": control,
        "summary": summarize(applied_all),
        "folds": folds,
        "top_rules": sorted(top_rules.items(), key=lambda kv: (-kv[1], kv[0]))[:8],
    }


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(r["group"], r["control"]): r for r in results}
    cross = by_key[(GROUPS[0], "main")]["summary"]
    family = by_key[(GROUPS[1], "main")]["summary"]
    reasons = []
    if cross["mean_pp_delta"] < 0.010:
        reasons.append("cross_pp_delta_below_0.010")
    if family["mean_pp_delta"] < 0.010:
        reasons.append("family_pp_delta_below_0.010")
    if cross["dataset_min_pp_delta"] < -0.020:
        reasons.append("cross_dataset_min_below_minus_0.020")
    if family["dataset_min_pp_delta"] < -0.020:
        reasons.append("family_dataset_min_below_minus_0.020")
    if family["mean_mmd_delta"] > 0.0005:
        reasons.append("family_mmd_worse_above_0.0005")
    for control in ("support_shuffled", "support_inverted", "support_permuted", "signed_gene_shuffle", "signed_sign_inverted", "signed_feature_shuffle", "all_feature_shuffle"):
        c = by_key[(GROUPS[0], control)]["summary"]
        if c["mean_pp_delta"] >= 0.005:
            reasons.append(f"{control}_cross_not_collapsed")
    passed = not reasons
    return {
        "status": "composite_safe_subset_gate_pass_gpu_smoke_authorized" if passed else "composite_safe_subset_gate_fail_no_gpu",
        "gpu_authorized": passed,
        "reasons": reasons,
        "cross_mean_pp_delta": cross["mean_pp_delta"],
        "family_mean_pp_delta": family["mean_pp_delta"],
        "cross_dataset_min": cross["dataset_min_pp_delta"],
        "family_dataset_min": family["dataset_min_pp_delta"],
    }


def render_md(payload: dict[str, Any]) -> str:
    d = payload["decision"]
    lines = [
        "# LatentFM Composite Safe-Subset Gate",
        "",
        f"Status: `{d['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only nested leave-one-dataset-out gate.",
        "- Combines train-only control-state support/coverage features with train-only signed perturbation-neighborhood features.",
        "- Uses completed cap120/anchor internal condition metrics only.",
        "- Does not read canonical outcomes, canonical multi, Track C query, active logs, new GPU artifacts, or use GPU.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{d['gpu_authorized']}`",
        f"- reasons: `{d['reasons']}`",
        f"- cross pp delta: `{d['cross_mean_pp_delta']:.6f}`",
        f"- family pp delta: `{d['family_mean_pp_delta']:.6f}`",
        f"- cross dataset-min: `{d['cross_dataset_min']:.6f}`",
        f"- family dataset-min: `{d['family_dataset_min']:.6f}`",
        "",
        "## Nested LODO Summaries",
        "",
        "| group | control | n | mean pp delta | 95% CI | p_harm | dataset min | mean MMD delta | mean alpha | top rules |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        s = row["summary"]
        top = "; ".join(f"{name}:{count}" for name, count in row["top_rules"])
        lines.append(
            f"| `{row['group']}` | `{row['control']}` | {s['n']} | {s['mean_pp_delta']:.6f} | [{s['ci95_low']:.6f}, {s['ci95_high']:.6f}] | {s['bootstrap_p_harm']:.3f} | {s['dataset_min_pp_delta']:.6f} | {s['mean_mmd_delta']:.6f} | {s['mean_alpha']:.3f} | {top} |"
        )
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`", ""])
    return "\n".join(lines)


def main() -> None:
    results = []
    controls = (
        "main",
        "support_shuffled",
        "support_inverted",
        "support_permuted",
        "signed_gene_shuffle",
        "signed_sign_inverted",
        "signed_feature_shuffle",
        "all_feature_shuffle",
    )
    for control in controls:
        rows = combine_rows(control)
        for group in GROUPS:
            result = nested_lodo([r for r in rows if r.group == group], control)
            result["group"] = group
            results.append(result)
    payload = {
        "boundary": {
            "support_script": str(ROOT / "ops/audit_latentfm_control_state_support_gate_20260624.py"),
            "signed_script": str(ROOT / "ops/audit_latentfm_signed_neighborhood_consistency_gate_20260624.py"),
            "target_run": "cap120",
            "groups": GROUPS,
            "seed": SEED,
        },
        "results": results,
        "decision": decide(results),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(OUT_MD)


if __name__ == "__main__":
    main()
