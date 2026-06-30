#!/usr/bin/env python3
"""CPU gate for controlled composition / scaling split candidates.

This gate is intentionally conservative. It asks whether the current repo
already contains a non-duplicate, leakage-safe composition/scaling split that
can authorize a bounded LatentFM GPU smoke after the support-set closures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
BIFLOW = ROOT / "dataset/biFlow_data"
OUT_JSON = REPORTS / "latentfm_controlled_composition_scaling_split_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_CONTROLLED_COMPOSITION_SCALING_SPLIT_GATE_20260627.md"


INPUTS = {
    "scaling_ready_md": REPORTS / "LATENTFM_SCALING_LAW_READY_EVIDENCE_TABLE_20260626.md",
    "scaling_ready_json": REPORTS / "latentfm_scaling_law_ready_evidence_table_20260626.json",
    "scaling_lockdown_json": REPORTS / "latentfm_scaling_lockdown_and_mainline_use_20260626.json",
    "training_norm_json": REPORTS / "latentfm_training_data_normalization_closure_20260624.json",
    "current_inventory_json": REPORTS / "latentfm_current_gpu_candidate_inventory_20260625.json",
    "pathway_gate_json": REPORTS / "latentfm_modality_pathway_sampling_gate_20260624.json",
    "pathway_smoke_json": REPORTS / "latentfm_modality_pathway_sampling_smoke_decision_20260624.json",
    "pathway_randomcount_json": REPORTS / "latentfm_modality_pathway_randomcount_control_smoke_decision_20260624.json",
    "pathway_mmd_json": REPORTS / "latentfm_modality_pathway_mmd_preservation_smoke_decision_20260624.json",
    "sciplex_dose_specific_json": REPORTS / "latentfm_sciplex_dose_specific_outcome_gate_20260627.json",
    "axis_law_readiness_csv": REPORTS / "scaling_law_ready_evidence_table_20260626/axis_law_readiness.csv",
    "missing_experiment_matrix_csv": REPORTS / "scaling_law_ready_evidence_table_20260626/missing_experiment_matrix.csv",
    "closed_training_routes_csv": REPORTS / "scaling_lockdown_and_mainline_use_20260626/closed_training_routes.csv",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def status(obj: dict[str, Any]) -> str:
    decision = obj.get("decision")
    if isinstance(decision, dict) and decision.get("status"):
        return str(decision["status"])
    return str(obj.get("status") or "")


def split_inventory() -> dict[str, Any]:
    patterns = {
        "pathway": "xverse_modality_pathway_sampling_splits_20260624/*.json",
        "scaling_count": "xverse_scaling_splits_v2_20260624/*.json",
        "scaling_protocol": "xverse_scaling_protocol_splits_20260624/*.json",
        "true_cell": "xverse_true_cell_count_scaling_splits_20260624/*.json",
        "true_cell_nested": "xverse_true_cell_count_scaling_nested_splits_20260624/*.json",
        "allmod_doseaware": "xverse_true_cell_count_allmodality_doseaware_splits_20260625/*.json",
        "chemical_v2_ack": "xverse_chemical_unseen_scaffold_v2_splits_20260625/*.json",
        "sciplex_dose_specific": "sciplex_dose_specific_splits_20260627/*.json",
    }
    out: dict[str, Any] = {}
    for family, pattern in patterns.items():
        paths = sorted(BIFLOW.glob(pattern))
        out[family] = [str(path) for path in paths]
    return out


def main() -> int:
    missing_inputs = [name for name, path in INPUTS.items() if not path.is_file()]
    ready = load_json(INPUTS["scaling_ready_json"])
    lockdown = load_json(INPUTS["scaling_lockdown_json"])
    training_norm = load_json(INPUTS["training_norm_json"])
    inventory = load_json(INPUTS["current_inventory_json"])
    pathway_gate = load_json(INPUTS["pathway_gate_json"])
    pathway_smoke = load_json(INPUTS["pathway_smoke_json"])
    pathway_random = load_json(INPUTS["pathway_randomcount_json"])
    pathway_mmd = load_json(INPUTS["pathway_mmd_json"])
    sciplex_dose = load_json(INPUTS["sciplex_dose_specific_json"])
    axes = read_csv(INPUTS["axis_law_readiness_csv"])
    missing_matrix = read_csv(INPUTS["missing_experiment_matrix_csv"])
    closed_routes = read_csv(INPUTS["closed_training_routes_csv"])
    splits = split_inventory()

    reasons: list[str] = []
    if missing_inputs:
        reasons.append("missing_required_inputs")

    if status(ready) != "scaling_law_ready_evidence_table_no_immediate_gpu":
        reasons.append("scaling_ready_status_not_current_no_gpu")
    if status(lockdown) != "scaling_lockdown_no_immediate_gpu":
        reasons.append("scaling_lockdown_status_not_current_no_gpu")
    if status(training_norm) != "training_data_normalization_closure_no_gpu":
        reasons.append("training_norm_closure_status_not_current_no_gpu")
    if status(inventory) != "latentfm_current_gpu_candidate_inventory_no_immediate_gpu":
        reasons.append("current_inventory_not_no_immediate_gpu")

    pathway_statuses = {
        "pathway_design": status(pathway_gate),
        "pathway_quota_smoke": status(pathway_smoke),
        "pathway_randomcount_control": status(pathway_random),
        "pathway_mmd_preservation": status(pathway_mmd),
    }
    if pathway_statuses["pathway_design"] == "modality_pathway_sampling_gate_pass_candidate_design_no_immediate_gpu":
        # The design pass is consumed by later smoke decisions.
        if pathway_statuses["pathway_quota_smoke"] == "internal_fail":
            reasons.append("pathway_quota_candidate_consumed_by_internal_fail")
        if pathway_statuses["pathway_randomcount_control"] == "internal_fail":
            reasons.append("pathway_randomcount_control_consumed_by_mmd_harm")
        if pathway_statuses["pathway_mmd_preservation"] == "internal_fail":
            reasons.append("pathway_mmd_preservation_consumed_by_internal_fail")

    gpu_axes = [row for row in axes if str(row.get("gpu_authorized", "")).lower() == "true"]
    if gpu_axes:
        reasons.append("unexpected_gpu_authorized_axis_present")
    ready_now_rows = [row for row in missing_matrix if str(row.get("gpu_ready_now", "")).lower() == "true"]
    if ready_now_rows:
        reasons.append("unexpected_missing_matrix_gpu_ready_now")

    closed_route_names = {row.get("route", "") for row in closed_routes}
    duplicate_split_families = {
        "pathway": "closed_by_pathway_quota_randomcount_mmdpreserve_smokes",
        "scaling_count": "closed_by_condition_count_tail_and_noharm_gates",
        "scaling_protocol": "closed_by_scaling_protocol_canonical_noharm",
        "true_cell": "closed_by_truecell_nonnoop_tail_meta_and_canonical_noharm",
        "true_cell_nested": "closed_by_truecell_nested_noharm_routes",
        "allmod_doseaware": "closed_by_allmod_family_tradeoff",
        "sciplex_dose_specific": "closed_by_sciplex_dose_specific_outcome_gate",
    }
    available_non_ack_split_families = [
        family
        for family, paths in splits.items()
        if paths and family not in duplicate_split_families and family != "chemical_v2_ack"
    ]
    if available_non_ack_split_families:
        reasons.append("unclassified_non_ack_split_family_requires_manual_audit")
    else:
        reasons.append("no_unconsumed_non_ack_composition_split_candidate")

    status_out = "controlled_composition_scaling_split_gate_fail_no_gpu"
    payload = {
        "status": status_out,
        "gpu_authorized": False,
        "reasons": reasons,
        "boundary": {
            "cpu_report_only": True,
            "heldout_trackc_query_used": False,
            "canonical_multi_selection_used": False,
            "training_or_inference_used": False,
            "gpu_used": False,
        },
        "inputs": {name: str(path) for name, path in INPUTS.items()},
        "missing_inputs": missing_inputs,
        "pathway_statuses": pathway_statuses,
        "sciplex_dose_specific_status": status(sciplex_dose),
        "axis_summary": {
            "n_axes": len(axes),
            "gpu_authorized_axes": [row.get("axis") for row in gpu_axes],
            "ready_now_missing_matrix_rows": ready_now_rows,
        },
        "split_inventory": splits,
        "duplicate_split_families": duplicate_split_families,
        "closed_route_names": sorted(x for x in closed_route_names if x),
        "decision": {
            "next_action": "Do not launch a non-ACK composition/scaling GPU run from existing splits. Continue with new external condition-level artifact gates, a materially new tail-protection CPU gate, or exact user ACK for chemical V2.",
            "chemical_v2_ack_route": {
                "requires_exact_ack": True,
                "ack_env": "LATENTFM_CHEM_V2_FIXEDSTEP_ACK=launch_v2_fixedstep_controls_after_protocol_review",
                "launch_packet": str(REPORTS / "LATENTFM_CHEMICAL_V2_ACK_LAUNCH_PACKET_20260626.md"),
            },
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Controlled Composition / Scaling Split Gate 2026-06-27",
        "",
        f"Status: `{status_out}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only gate over completed scaling/composition evidence.",
        "- No training, inference, canonical multi selection, Track C query, or GPU.",
        "",
        "## Result",
        "",
        "Existing controlled-composition split families are consumed by later failed gates:",
        "",
        "| family | count | disposition |",
        "|---|---:|---|",
    ]
    for family, paths in sorted(splits.items()):
        disposition = duplicate_split_families.get(family)
        if family == "chemical_v2_ack":
            disposition = "ACK-gated chemical V2 protocol only"
        if disposition is None:
            disposition = "unclassified; manual audit required" if paths else "none"
        lines.append(f"| `{family}` | {len(paths)} | {disposition} |")
    lines.extend(
        [
            "",
            "## Pathway Follow-Up Status",
            "",
            f"- design gate: `{pathway_statuses['pathway_design']}`",
            f"- pathway quota smoke: `{pathway_statuses['pathway_quota_smoke']}`",
            f"- random-count control: `{pathway_statuses['pathway_randomcount_control']}`",
            f"- MMD-preservation smoke: `{pathway_statuses['pathway_mmd_preservation']}`",
            f"- SciPlex dose-specific outcome gate: `{status(sciplex_dose)}`",
            "",
            "## Reasons",
            "",
        ]
    )
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No non-ACK composition/scaling GPU smoke is legal from the current split inventory.",
            "The next non-ACK route must first supply a genuinely new CPU artifact/gate, not a",
            "renamed pathway/quota/visit/type/cap variant.",
            "",
            "Chemical V2 remains the only prepared immediate GPU route, but it requires exact user ACK:",
            "`LATENTFM_CHEM_V2_FIXEDSTEP_ACK=launch_v2_fixedstep_controls_after_protocol_review`.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status_out, "reasons": reasons, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
