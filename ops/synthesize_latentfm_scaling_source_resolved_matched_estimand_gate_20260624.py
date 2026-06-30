#!/usr/bin/env python3
"""CPU-only source-resolved matched estimand gate for scaling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
PROVENANCE = REPORTS / "latentfm_scaling_provenance_estimand_matrix_gate_20260624.json"
MIXED = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"
SOURCE = REPORTS / "latentfm_scaling_source_verified_background_type_strata_gate_20260624.json"
CONFOUND = REPORTS / "latentfm_scaling_matched_background_type_confound_gate_20260624.json"
TARGET = REPORTS / "latentfm_scaling_target_gene_coverage_protocol_gate_20260624.json"
BREADTH = REPORTS / "latentfm_matched_dataset_breadth_gate_20260624.json"
TAIL = REPORTS / "latentfm_scaling_provenance_tail_sentinel_gate_20260624.json"
OUT_JSON = REPORTS / "latentfm_scaling_source_resolved_matched_estimand_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_SOURCE_RESOLVED_MATCHED_ESTIMAND_GATE_20260624.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    provenance = load(PROVENANCE)
    mixed = load(MIXED)
    source = load(SOURCE)
    confound = load(CONFOUND)
    target = load(TARGET)
    breadth = load(BREADTH)
    tail = load(TAIL)

    crossing = confound.get("crossing") or {}
    background_type_nmi = crossing.get("normalized_mi_background_type")
    if background_type_nmi is None:
        background_type_nmi = crossing.get("nmi_background_type")
    mixed_summary = mixed.get("summary") or {}
    breadth_decision = breadth.get("decision") or {}
    reasons = []

    if source.get("status") != "source_verified_strata_gate_pass_gpu_next":
        reasons.append("source_verified_background_type_strata_gate_failed")
    if confound.get("status") != "matched_background_type_confound_gate_pass_gpu_next":
        reasons.append("background_type_metadata_effect_confounded")
    if float(background_type_nmi or 1.0) > 0.35:
        reasons.append("background_type_nmi_too_high")
    if float(mixed_summary.get("dataset_min_pp_delta") or -1.0) < -0.020:
        reasons.append("condition_count_dataset_tail_below_minus_0p020")
    if float((mixed_summary.get("bootstrap_dataset_mean_pp_ci") or [-1.0, 1.0])[0]) <= 0.0:
        reasons.append("condition_count_bootstrap_ci_lower_not_positive")
    if target.get("status") != "target_gene_coverage_gate_pass_gpu_next":
        reasons.append("target_gene_coverage_gate_failed")
    if breadth_decision.get("gpu_authorized") is not True:
        reasons.append("matched_dataset_breadth_gate_failed")
    if not tail.get("pass_rules"):
        reasons.append("no_provenance_tail_sentinel_rule_passed")

    candidate_axes = [
        {
            "axis": "condition_count",
            "status": mixed.get("status"),
            "pp": mixed_summary.get("condition_weighted_pp_delta"),
            "dataset_min": mixed_summary.get("dataset_min_pp_delta"),
            "bootstrap_ci": mixed_summary.get("bootstrap_dataset_mean_pp_ci"),
            "gpu_ready": False,
        },
        {
            "axis": "background_type",
            "status": confound.get("status"),
            "nmi": background_type_nmi,
            "negative_backgrounds": (source.get("background_summary") or {}).get("negative_backgrounds")
            or mixed.get("negative_backgrounds"),
            "negative_types": (source.get("perturbation_type_summary") or {}).get("negative_types")
            or mixed.get("negative_perturbation_types"),
            "gpu_ready": False,
        },
        {
            "axis": "target_gene_coverage",
            "status": target.get("status"),
            "gpu_ready": False,
        },
        {
            "axis": "dataset_breadth",
            "status": breadth_decision.get("status") or breadth.get("status"),
            "many_minus_few_cross_pp": breadth_decision.get("many_minus_few_cross_candidate_pp"),
            "gpu_ready": False,
        },
        {
            "axis": "provenance_tail_sentinel",
            "status": tail.get("status"),
            "pass_rules": tail.get("pass_rules"),
            "gpu_ready": False,
        },
    ]
    status = "scaling_source_resolved_matched_estimand_fail_no_gpu"
    gpu_authorized = False
    if not reasons:
        status = "scaling_source_resolved_matched_estimand_pass_external_review_next"

    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_only": True,
            "reads_completed_train_only_scaling_summaries": True,
            "canonical_single_family_context_only": True,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "provenance": str(PROVENANCE),
            "mixed_effect_lodo": str(MIXED),
            "source_strata": str(SOURCE),
            "confound": str(CONFOUND),
            "target_coverage": str(TARGET),
            "dataset_breadth": str(BREADTH),
            "tail_sentinel": str(TAIL),
        },
        "summary": {
            "source_quality_counts": (provenance.get("summary") or {}).get("source_quality_counts"),
            "background_type_nmi": background_type_nmi,
            "condition_count_pp": mixed_summary.get("condition_weighted_pp_delta"),
            "condition_count_dataset_min": mixed_summary.get("dataset_min_pp_delta"),
            "candidate_axes": candidate_axes,
        },
        "reasons": reasons,
        "decision": {
            "gpu_next_action": "none" if reasons else "external review before one bounded matched-estimand smoke",
            "scaling_claim_scope": "diagnostic_only" if reasons else "matched_estimand_candidate",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling Source-Resolved Matched Estimand Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed train-only scaling/provenance gates.",
        "- Frozen canonical single/family evidence is context only; canonical multi and Track C query are not read.",
        "- Does not train, infer, launch GPU, or select a checkpoint.",
        "",
        "## Summary",
        "",
        f"- background/type NMI: `{fmt(background_type_nmi)}`",
        f"- condition-count pp / dataset min: `{fmt(mixed_summary.get('condition_weighted_pp_delta'))}` / `{fmt(mixed_summary.get('dataset_min_pp_delta'))}`",
        f"- source strata status: `{source.get('status')}`",
        f"- matched confound status: `{confound.get('status')}`",
        f"- target coverage status: `{target.get('status')}`",
        f"- dataset breadth GPU authorized: `{breadth_decision.get('gpu_authorized')}`",
        f"- tail sentinel pass rules: `{tail.get('pass_rules')}`",
        "",
        "## Candidate Axes",
        "",
        "| axis | status | key metric | GPU ready |",
        "|---|---|---:|---:|",
    ]
    for row in candidate_axes:
        key = row.get("pp", row.get("nmi", row.get("many_minus_few_cross_pp", row.get("pass_rules"))))
        lines.append(f"| `{row['axis']}` | `{row.get('status')}` | `{key}` | `{row['gpu_ready']}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            "- scaling remains diagnostic-only until a matched/nested estimand passes tail, bootstrap, and control-collapse gates.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": gpu_authorized}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
