#!/usr/bin/env python3
"""No-harm constrained xverse response-covariate router CPU gate."""

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


OUT_JSON = ROOT / "reports/latentfm_tracka_xverse_response_covariate_noharm_router_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_XVERSE_RESPONSE_COVARIATE_NOHARM_ROUTER_GATE_20260624.md"
NOHARM_THRESHOLDS = (-0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    for result in results:
        group = result["group"]
        vs_gene = base.paired_row(result, "gene_raw_mean")
        vs_anchor = base.paired_row(result, "anchor_pearson_pert")
        vs_shuf = base.paired_row(result, "shuffled_router_anchor_or_gene")
        if float(result["use_anchor_fraction"]) < 0.05:
            reasons.append(f"{group}_uses_anchor_too_rarely")
        if float(vs_gene["delta_mean"]) < 0.02:
            reasons.append(f"{group}_delta_vs_gene_below_0p02")
        if float(vs_gene["p_harm"]) > 0.20:
            reasons.append(f"{group}_harm_vs_gene_above_0p20")
        if float(vs_gene["dataset_min"]) < -0.02:
            reasons.append(f"{group}_dataset_min_vs_gene_below_minus_0p02")
        if float(vs_anchor["delta_mean"]) < -0.005:
            reasons.append(f"{group}_material_loss_vs_anchor")
        if float(vs_shuf["delta_mean"]) < 0.01:
            reasons.append(f"{group}_shuffled_router_not_beaten_by_0p01")
    status = (
        "tracka_xverse_response_covariate_noharm_router_gate_pass_code_gate_next_no_gpu"
        if not reasons
        else "tracka_xverse_response_covariate_noharm_router_gate_fail_no_gpu"
    )
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "code_gate_only_if_pass_else_none",
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A xverse Response-Covariate No-Harm Router Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses only xverse internal proxy residual-forensics rows.",
        "- Nested leave-one-dataset-out: each held-out dataset uses ridge alpha and threshold selected only from other datasets.",
        "- This rerun expands the threshold grid to permit high-confidence/no-harm abstention.",
        "- Does not read canonical Track A outcomes, canonical multi, held-out query, active logs, or GPU artifacts.",
        "",
        "## Results",
        "",
        "| group | use anchor | delta vs gene | p harm | dataset min | delta vs anchor | delta vs shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        vs_gene = base.paired_row(result, "gene_raw_mean")
        vs_anchor = base.paired_row(result, "anchor_pearson_pert")
        vs_shuf = base.paired_row(result, "shuffled_router_anchor_or_gene")
        lines.append(
            f"| {result['group']} | {result['use_anchor_fraction']:.3f} | "
            f"{fmt(vs_gene['delta_mean'])} | {fmt(vs_gene['p_harm'])} | "
            f"{fmt(vs_gene['dataset_min'])} | {fmt(vs_anchor['delta_mean'])} | "
            f"{fmt(vs_shuf['delta_mean'])} |"
        )
    lines.extend(["", "## Gate Reasons", ""])
    lines.extend([f"- `{r}`" for r in payload["decision"].get("reasons", [])] or ["- none"])
    lines.extend(["", "## Threshold Grid", "", ", ".join(f"`{x}`" for x in NOHARM_THRESHOLDS), ""])
    return "\n".join(lines)


def main() -> int:
    base.THRESHOLDS = NOHARM_THRESHOLDS
    source = base.load_json(base.XVERSE_ROWS)
    rows = source["condition_rows"]
    results = [base.evaluate_group(rows, group) for group in base.GROUPS]
    decision = decide(results)
    payload = {
        "status": decision["status"],
        "timestamp": "2026-06-24 00:40 CST",
        "inputs": {"xverse_residual_forensics": str(base.XVERSE_ROWS)},
        "boundary": {
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
            "nested_lodo": True,
        },
        "features": base.FEATURES,
        "alphas": base.ALPHAS,
        "thresholds": NOHARM_THRESHOLDS,
        "results": results,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
