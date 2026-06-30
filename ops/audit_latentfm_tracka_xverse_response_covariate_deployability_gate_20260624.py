#!/usr/bin/env python3
"""Deployability/stability gate for the xverse response-covariate router.

The previous no-harm router had strong average internal-val signal but depended
on residual-forensics covariates, several of which require the held-out target
residual. This CPU gate asks whether the signal survives with deployable
train-time covariates only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import audit_latentfm_tracka_xverse_response_covariate_router_gate_20260624 as base


OUT_JSON = ROOT / "reports/latentfm_tracka_xverse_response_covariate_deployability_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_XVERSE_RESPONSE_COVARIATE_DEPLOYABILITY_GATE_20260624.md"
NOHARM_THRESHOLDS = (-0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "full_forensics_noharm_grid",
        "deployable": False,
        "features": (
            "gene_target_cosine",
            "dataset_target_cosine",
            "target_residual_norm",
            "gene_dataset_cosine",
            "gene_minus_dataset_score",
            "gene_pred_norm",
            "dataset_pred_norm",
            "global_pred_norm",
            "gene_train_count",
        ),
        "note": "Original near-pass feature family; includes target-derived and outcome-score covariates.",
    },
    {
        "name": "no_target_but_score_covariate",
        "deployable": False,
        "features": (
            "gene_dataset_cosine",
            "gene_minus_dataset_score",
            "gene_pred_norm",
            "dataset_pred_norm",
            "global_pred_norm",
            "gene_train_count",
        ),
        "note": "Removes explicit target-residual geometry but keeps gene_minus_dataset_score, which is still outcome-score-derived.",
    },
    {
        "name": "deployable_norms_count_geometry",
        "deployable": True,
        "features": (
            "gene_dataset_cosine",
            "gene_pred_norm",
            "dataset_pred_norm",
            "global_pred_norm",
            "gene_train_count",
        ),
        "note": "Uses only component geometry/count covariates that can be computed before seeing the held-out target residual.",
    },
    {
        "name": "deployable_norms_count",
        "deployable": True,
        "features": (
            "gene_pred_norm",
            "dataset_pred_norm",
            "global_pred_norm",
            "gene_train_count",
        ),
        "note": "Deployable covariates without gene-dataset cosine.",
    },
    {
        "name": "deployable_count_only",
        "deployable": True,
        "features": ("gene_train_count",),
        "note": "Closed-family count-only control.",
    },
)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def result_metrics(result: dict[str, Any]) -> dict[str, Any]:
    vs_gene = base.paired_row(result, "gene_raw_mean")
    vs_anchor = base.paired_row(result, "anchor_pearson_pert")
    vs_shuf = base.paired_row(result, "shuffled_router_anchor_or_gene")
    return {
        "use_anchor_fraction": float(result["use_anchor_fraction"]),
        "delta_vs_gene": float(vs_gene["delta_mean"]),
        "p_harm_vs_gene": float(vs_gene["p_harm"]),
        "dataset_min_vs_gene": float(vs_gene["dataset_min"]),
        "delta_vs_anchor": float(vs_anchor["delta_mean"]),
        "delta_vs_shuffled": float(vs_shuf["delta_mean"]),
    }


def evaluate_variant(rows: list[dict[str, Any]], variant: dict[str, Any]) -> dict[str, Any]:
    base.FEATURES = tuple(variant["features"])
    base.THRESHOLDS = NOHARM_THRESHOLDS
    results = [base.evaluate_group(rows, group) for group in base.GROUPS]
    group_metrics = {result["group"]: result_metrics(result) for result in results}
    reasons = []
    for group, metrics in group_metrics.items():
        if metrics["use_anchor_fraction"] < 0.05:
            reasons.append(f"{variant['name']}:{group}:uses_anchor_too_rarely")
        if metrics["delta_vs_gene"] < 0.02:
            reasons.append(f"{variant['name']}:{group}:delta_vs_gene_below_0p02")
        if metrics["p_harm_vs_gene"] > 0.20:
            reasons.append(f"{variant['name']}:{group}:p_harm_vs_gene_above_0p20")
        if metrics["dataset_min_vs_gene"] < -0.02:
            reasons.append(f"{variant['name']}:{group}:dataset_min_vs_gene_below_minus_0p02")
        if metrics["delta_vs_anchor"] < -0.005:
            reasons.append(f"{variant['name']}:{group}:material_loss_vs_anchor")
        if metrics["delta_vs_shuffled"] < 0.01:
            reasons.append(f"{variant['name']}:{group}:shuffled_not_beaten_by_0p01")
    return {
        "name": variant["name"],
        "deployable": bool(variant["deployable"]),
        "features": list(variant["features"]),
        "note": variant["note"],
        "group_metrics": group_metrics,
        "reasons": reasons,
        "passes_gate": not reasons,
    }


def decide(variant_results: list[dict[str, Any]]) -> dict[str, Any]:
    deployable_passes = [v["name"] for v in variant_results if v["deployable"] and v["passes_gate"]]
    diagnostic_passes = [v["name"] for v in variant_results if (not v["deployable"]) and v["passes_gate"]]
    if deployable_passes:
        status = "tracka_xverse_response_covariate_deployability_gate_pass_code_gate_next_no_gpu"
        action = "design_default_off_train_time_router_code_gate"
        reasons: list[str] = []
    else:
        status = "tracka_xverse_response_covariate_deployability_gate_fail_no_gpu"
        action = "do_not_gpu_launch_response_covariate_router"
        reasons = ["no_deployable_variant_passed_gate"]
        if diagnostic_passes:
            reasons.append("only_non_deployable_diagnostic_variant_passed")
    return {
        "status": status,
        "gpu_authorization": "none",
        "action": action,
        "deployable_passes": deployable_passes,
        "diagnostic_passes": diagnostic_passes,
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A xverse Response-Covariate Deployability Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses only xverse train-only/internal proxy residual-forensics rows.",
        "- Does not read canonical Track A outcomes, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- Tests whether the prior near-pass router remains valid after removing target-derived/non-deployable covariates.",
        "- Nested leave-one-dataset-out selection is inherited from the original router gate.",
        "",
        "## Gate Rule",
        "",
        "At least one deployable variant must pass both internal groups with:",
        "",
        "- use-anchor fraction `>= 0.05`;",
        "- delta vs `gene_raw_mean` `>= +0.02`;",
        "- paired bootstrap harm probability `<= 0.20`;",
        "- dataset-min delta vs `gene_raw_mean` `>= -0.02`;",
        "- delta vs anchor `>= -0.005`;",
        "- delta vs shuffled router `>= +0.01`.",
        "",
        "## Results",
        "",
        "| variant | deployable | group | use anchor | delta vs gene | p harm | dataset min | delta vs anchor | delta vs shuffled | pass |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in payload["variants"]:
        for group, metrics in variant["group_metrics"].items():
            lines.append(
                f"| `{variant['name']}` | {str(variant['deployable']).lower()} | {group} | "
                f"{metrics['use_anchor_fraction']:.3f} | {fmt(metrics['delta_vs_gene'])} | "
                f"{fmt(metrics['p_harm_vs_gene'])} | {fmt(metrics['dataset_min_vs_gene'])} | "
                f"{fmt(metrics['delta_vs_anchor'])} | {fmt(metrics['delta_vs_shuffled'])} | "
                f"{str(variant['passes_gate']).lower()} |"
            )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in payload["decision"].get("reasons", [])] or ["- none"])
    lines.extend(["", "## Variant Notes", ""])
    for variant in payload["variants"]:
        lines.append(f"- `{variant['name']}`: {variant['note']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    source = base.load_json(base.XVERSE_ROWS)
    rows = source["condition_rows"]
    variant_results = [evaluate_variant(rows, variant) for variant in VARIANTS]
    decision = decide(variant_results)
    payload = {
        "status": decision["status"],
        "timestamp": "2026-06-24 03:30 CST",
        "inputs": {"xverse_residual_forensics": str(base.XVERSE_ROWS)},
        "boundary": {
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
            "nested_lodo": True,
        },
        "thresholds": NOHARM_THRESHOLDS,
        "variants": variant_results,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
