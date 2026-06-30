#!/usr/bin/env python3
"""CPU-only gate for matched dataset-breadth scaling continuation.

This audit formalizes whether the existing matched-budget breadth matrix
authorizes another scaling GPU smoke. It is intentionally conservative: if the
few/mid/many breadth arms already failed train-only internal gates, it closes
dataset-breadth GPU continuation from current evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path("/data/cyx/1030/scLatent")
MATRIX_JSON = ROOT / "reports/latentfm_scaling_protocol_matrix_decision_20260624.json"
CANONICAL_JSON = ROOT / "reports/latentfm_scaling_protocol_canonical_noharm_decision_20260624.json"
META_JSON = ROOT / "reports/latentfm_scaling_metainfo_inventory_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_matched_dataset_breadth_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_MATCHED_DATASET_BREADTH_GATE_20260624.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decide(matrix: dict[str, Any], canonical: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    rows = matrix.get("rows") or []
    breadth_rows = [r for r in rows if str(r.get("arm", "")).startswith("breadth_")]
    breadth_failures = []
    for row in breadth_rows:
        metrics = row.get("metrics") or {}
        reasons = []
        if float(metrics.get("cross_pp_delta_vs_anchor", 0.0)) < 0.010:
            reasons.append("cross_pp_delta_vs_anchor_lt_0p010")
        if float(metrics.get("family_gene_pp_delta_vs_anchor", 0.0)) < -0.005:
            reasons.append("family_gene_pp_hard_harm")
        if reasons:
            breadth_failures.append({"arm": row.get("arm"), "reasons": reasons})
    matrix_decision = matrix.get("decision") or {}
    many_minus_few = matrix_decision.get(
        "many_shallow_minus_few_deep_cross_candidate_pp",
        matrix_decision.get("many_minus_few_cross_candidate_pp"),
    )
    cap60_canonical_status = (canonical.get("decision") or canonical).get("status", canonical.get("status"))
    reasons = []
    if len(breadth_rows) < 3:
        reasons.append("missing_three_breadth_arms")
    if breadth_failures:
        reasons.append("matched_breadth_arms_failed_internal_gate")
    if many_minus_few is not None and float(many_minus_few) < 0.003:
        reasons.append("many_shallow_not_better_than_few_deep")
    if str(cap60_canonical_status) == "canonical_noharm_fail":
        reasons.append("condition_count_midpoint_internal_pass_failed_canonical_noharm")
    status = "matched_dataset_breadth_gate_pass_gpu_protocol_design_next"
    if reasons:
        status = "matched_dataset_breadth_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorized": False,
        "n_datasets_inventory": (meta.get("summary") or {}).get("n_datasets", meta.get("datasets")),
        "breadth_arms": [r.get("arm") for r in breadth_rows],
        "breadth_failures": breadth_failures,
        "matrix_passed_arms": matrix_decision.get("passed", []),
        "many_shallow_minus_few_cross_candidate_pp": many_minus_few,
        "cap60_canonical_status": cap60_canonical_status,
        "reasons": reasons,
        "next_action": (
            "Do not launch matched dataset-breadth GPU. Reopen scaling only with a materially new protocol, e.g. source-verified background/type strata or a new no-harm stabilizer gate."
            if reasons
            else "Design exactly one bounded matched-breadth GPU smoke with fresh RUN_STATUS."
        ),
    }


def render_md(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    rows = payload["matrix_rows"]
    lines = [
        "# LatentFM Matched Dataset-Breadth Gate",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of existing matched-budget scaling matrix, canonical no-harm veto, and local metainfo inventory.",
        "- Does not train, launch GPU, read Track C query, or use canonical multi for selection.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{decision['gpu_authorized']}`",
        f"- breadth arms: `{decision['breadth_arms']}`",
        f"- matrix passed arms: `{decision['matrix_passed_arms']}`",
        f"- many-shallow minus few-deep cross candidate pp: `{decision['many_shallow_minus_few_cross_candidate_pp']}`",
        f"- cap60 canonical status: `{decision['cap60_canonical_status']}`",
        "",
        "Reasons:",
    ]
    if decision["reasons"]:
        lines.extend([f"- `{r}`" for r in decision["reasons"]])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Matrix Rows",
            "",
            "| arm | cross pp delta | family pp delta | family MMD delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        m = row.get("metrics") or {}
        lines.append(
            f"| `{row.get('arm')}` | {float(m.get('cross_pp_delta_vs_anchor', 0.0)):+.6f} | "
            f"{float(m.get('family_gene_pp_delta_vs_anchor', 0.0)):+.6f} | "
            f"{float(m.get('family_gene_mmd_delta_vs_anchor', 0.0)):+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            decision["next_action"],
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    matrix = load_json(MATRIX_JSON)
    canonical = load_json(CANONICAL_JSON)
    meta = load_json(META_JSON)
    payload = {
        "boundary": {
            "matrix_json": str(MATRIX_JSON),
            "canonical_noharm_json": str(CANONICAL_JSON),
            "metainfo_inventory_json": str(META_JSON),
            "read_trackc_query": False,
            "canonical_multi_selection": False,
            "launched_gpu": False,
        },
        "decision": decide(matrix, canonical, meta),
        "matrix_rows": matrix.get("rows") or [],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
