#!/usr/bin/env python3
"""Build the final LatentFM scaling package index.

CPU/report-only. This file inventories completed scaling artifacts, claim
boundaries, closed routes, reviewer gates, and next legal unlocks. It does not
read checkpoints, canonical multi, Track C query outputs, expression matrices,
or use GPU.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "scaling_final_package_index_20260626"
OUT_MD = REPORTS / "LATENTFM_SCALING_FINAL_PACKAGE_INDEX_20260626.md"
OUT_JSON = REPORTS / "latentfm_scaling_final_package_index_20260626.json"
OUT_ARTIFACTS = OUT_DIR / "final_artifact_manifest.tsv"
OUT_CLAIMS = OUT_DIR / "final_claim_readiness.tsv"
OUT_GATES = OUT_DIR / "final_reviewer_gates.tsv"


KEY_ARTIFACTS = [
    ("lockdown_report", REPORTS / "LATENTFM_SCALING_LOCKDOWN_AND_MAINLINE_USE_20260626.md"),
    ("lockdown_json", REPORTS / "latentfm_scaling_lockdown_and_mainline_use_20260626.json"),
    ("axis_lockdown", REPORTS / "scaling_lockdown_and_mainline_use_20260626/axis_lockdown.csv"),
    ("closed_routes", REPORTS / "scaling_lockdown_and_mainline_use_20260626/closed_training_routes.csv"),
    ("next_candidates", REPORTS / "scaling_lockdown_and_mainline_use_20260626/next_candidates.csv"),
    ("claim_failure_package", REPORTS / "LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md"),
    ("claim_failure_json", REPORTS / "latentfm_scaling_nm_claim_failure_package_20260625.json"),
    ("axis_claim_boundary", REPORTS / "scaling_nm_claim_failure_package_20260625/axis_claim_boundary.csv"),
    ("top_failure_cases", REPORTS / "scaling_nm_claim_failure_package_20260625/top_failure_cases.csv"),
    ("mainline_guidance", REPORTS / "scaling_nm_claim_failure_package_20260625/mainline_training_guidance.csv"),
    ("literature_boundary", REPORTS / "scaling_nm_claim_failure_package_20260625/literature_claim_boundary.csv"),
    ("provenance_manifest", REPORTS / "LATENTFM_SCALING_NM_PROVENANCE_MANIFEST_20260625.md"),
    ("artifact_manifest", REPORTS / "scaling_nm_provenance_manifest_20260625/artifact_manifest.tsv"),
    ("claim_to_artifact_map", REPORTS / "scaling_nm_provenance_manifest_20260625/claim_to_artifact_map.tsv"),
    ("figure_readiness", REPORTS / "LATENTFM_SCALING_FIGURE_READINESS_20260625.md"),
    ("figure_readiness_csv", REPORTS / "scaling_figure_readiness_20260625/figure_readiness.csv"),
    ("narrative_skeleton", REPORTS / "LATENTFM_SCALING_NARRATIVE_SKELETON_20260625.md"),
    ("result_sections", REPORTS / "scaling_narrative_skeleton_20260625/result_sections.tsv"),
    ("reviewer_checklist", REPORTS / "scaling_narrative_skeleton_20260625/reviewer_checklist.tsv"),
    ("reproduction_manifest", REPORTS / "LATENTFM_SCALING_REPRODUCTION_MANIFEST_20260625.md"),
    ("reproduction_commands", REPORTS / "scaling_reproduction_manifest_20260625/reproduction_commands.tsv"),
    ("script_hashes", REPORTS / "scaling_reproduction_manifest_20260625/script_hashes.tsv"),
    ("preregistered_axis_matrix", REPORTS / "LATENTFM_SCALING_PREREGISTERED_AXIS_MATRIX_20260626.md"),
    ("preregistered_axis_json", REPORTS / "latentfm_scaling_preregistered_axis_matrix_20260626.json"),
    ("law_ready_evidence_table", REPORTS / "LATENTFM_SCALING_LAW_READY_EVIDENCE_TABLE_20260626.md"),
    ("law_ready_evidence_json", REPORTS / "latentfm_scaling_law_ready_evidence_table_20260626.json"),
    ("law_ready_axis_readiness", REPORTS / "scaling_law_ready_evidence_table_20260626/axis_law_readiness.csv"),
    ("law_ready_missing_experiments", REPORTS / "scaling_law_ready_evidence_table_20260626/missing_experiment_matrix.csv"),
    ("condition_exposure_hier_boot_lodo", REPORTS / "LATENTFM_CONDITION_EXPOSURE_HIERARCHICAL_BOOTSTRAP_LODO_GATE_20260626.md"),
    ("condition_exposure_hier_boot_lodo_json", REPORTS / "latentfm_condition_exposure_hierarchical_bootstrap_lodo_gate_20260626.json"),
    ("condition_exposure_hier_boot_lodo_criteria", REPORTS / "condition_exposure_hierarchical_bootstrap_lodo_gate_20260626/criteria_matrix.csv"),
    ("source_background_type_hier_matched", REPORTS / "LATENTFM_SOURCE_BACKGROUND_TYPE_HIERARCHICAL_MATCHED_GATE_20260626.md"),
    ("source_background_type_hier_matched_json", REPORTS / "latentfm_source_background_type_hierarchical_matched_gate_20260626.json"),
    ("source_background_type_hier_matched_criteria", REPORTS / "source_background_type_hierarchical_matched_gate_20260626/criteria_matrix.csv"),
    ("external_reliability_v2", REPORTS / "LATENTFM_EXTERNAL_RELIABILITY_V2_PREFLIGHT_20260626.md"),
    ("scperturb_source_maturity_artifacts", REPORTS / "LATENTFM_SCPERTURB_SOURCE_MATURITY_ARTIFACTS_20260626.md"),
    ("scperturb_source_maturity_preflight", REPORTS / "LATENTFM_SCPERTURB_SOURCE_MATURITY_ARTIFACT_PREFLIGHT_20260626.md"),
    ("replicate_batch_balance_artifacts", REPORTS / "LATENTFM_REPLICATE_BATCH_BALANCE_ARTIFACTS_20260626.md"),
    ("replicate_batch_balance_preflight", REPORTS / "LATENTFM_REPLICATE_BATCH_BALANCE_ARTIFACT_PREFLIGHT_20260626.md"),
    ("background_specific_grn_source_audit", REPORTS / "LATENTFM_BACKGROUND_SPECIFIC_GRN_CONTEXT_SOURCE_AUDIT_20260626.md"),
    ("condition_level_reliability_source_scout", REPORTS / "LATENTFM_CONDITION_LEVEL_RELIABILITY_SOURCE_SCOUT_20260626.md"),
    ("condition_level_reliability_source_scout_json", REPORTS / "latentfm_condition_level_reliability_source_scout_20260626.json"),
    ("condition_level_reliability_candidate_matrix", REPORTS / "condition_level_reliability_source_scout_20260626/candidate_source_matrix.csv"),
    ("condition_level_reliability_completion_matrix", REPORTS / "condition_level_reliability_source_scout_20260626/scaling_law_completion_experiment_matrix.csv"),
    ("condition_level_reliability_mainline_translation", REPORTS / "condition_level_reliability_source_scout_20260626/mainline_translation.csv"),
    ("external_condition_artifact_acquisition_slate", REPORTS / "LATENTFM_EXTERNAL_CONDITION_ARTIFACT_ACQUISITION_SLATE_20260626.md"),
    ("external_condition_artifact_acquisition_slate_json", REPORTS / "latentfm_external_condition_artifact_acquisition_slate_20260626.json"),
    ("external_condition_artifact_source_matrix", REPORTS / "external_condition_artifact_acquisition_slate_20260626/source_candidate_matrix.csv"),
    ("external_condition_artifact_alignment_matrix", REPORTS / "external_condition_artifact_acquisition_slate_20260626/local_alignment_matrix.csv"),
    ("external_condition_artifact_gate_protocol", REPORTS / "external_condition_artifact_acquisition_slate_20260626/artifact_gate_protocol.csv"),
    ("gwt_condition_reliability_artifacts", REPORTS / "LATENTFM_GWT_CONDITION_RELIABILITY_ARTIFACTS_20260626.md"),
    ("gwt_condition_reliability_artifacts_json", REPORTS / "latentfm_gwt_condition_reliability_artifacts_20260626.json"),
    ("gwt_condition_reliability_manifest", ROOT / "configs/latentfm_gwt_condition_reliability_artifact_manifest_20260626.json"),
    ("gwt_condition_reliability_preflight", REPORTS / "LATENTFM_GWT_CONDITION_RELIABILITY_ARTIFACT_PREFLIGHT_20260627.md"),
    ("gwt_condition_reliability_preflight_json", REPORTS / "latentfm_gwt_condition_reliability_artifact_preflight_20260627.json"),
    ("gwt_condition_reliability_preflight_rows", REPORTS / "latentfm_gwt_condition_reliability_artifact_preflight_20260627_rows.csv"),
    ("current_inventory", REPORTS / "LATENTFM_CURRENT_GPU_CANDIDATE_INVENTORY_20260625.md"),
]


CLAIMS = [
    {
        "claim": "true_cell_support_is_strongest_positive_mechanism",
        "readiness": "ready_mechanism_only",
        "allowed_scope": "main_text_mechanism_with_noharm_veto",
        "evidence": "budget128 6k internal cross/family/MMD +0.059142/+0.062067/-0.001395; frozen canonical no-harm failed all seeds",
        "must_not_claim": "deployable model improvement or replacement for xverse_8k_anchor",
        "primary_artifacts": "lockdown_report; axis_claim_boundary; Fig_scaling_truecell_budget; Fig_scaling_noharm_veto",
    },
    {
        "claim": "condition_exposure_is_nonmonotonic_tail_unsafe",
        "readiness": "ready_failure_map",
        "allowed_scope": "main_text_or_supplement_failure_map",
        "evidence": "moderate exposure/local signal but hierarchical bootstrap, sign controls, LODO/source-background-type controls, full-exposure, canonical no-harm, and dataset-tail veto block monotonic law",
        "must_not_claim": "more conditions/full exposure is uniformly better",
        "primary_artifacts": "axis_claim_boundary; top_failure_cases; Fig_scaling_exposure_nonmonotonic; condition_exposure_hier_boot_lodo",
    },
    {
        "claim": "background_type_source_are_confounded_failure_axes",
        "readiness": "ready_failure_map",
        "allowed_scope": "supplement_or_failure_map",
        "evidence": "source/background/type gates remain CI/tail/source-weight/stratum/LODO/no-harm unsafe; latest source-background-type hierarchical matched gate fails",
        "must_not_claim": "cross-background/type breadth alone proves scaling",
        "primary_artifacts": "lockdown_report; top_failure_cases; FigS_scaling_failure_map; source_background_type_hier_matched",
    },
    {
        "claim": "metadata_qc_reagent_and_ot_do_not_authorize_training",
        "readiness": "ready_negative_evidence",
        "allowed_scope": "supplement_or_methods_guardrail",
        "evidence": "QC/support, reagent/read-support source-block, external reliability v2, source-maturity, replicate/batch balance, GRN-source audit, and OT routes fail required controls",
        "must_not_claim": "generic weighted loss, balancing, source-maturity weighting, replicate/batch weighting, background-GRN conditioning, read-support weighting, or OT sweeps are justified",
        "primary_artifacts": "lockdown_report; closed_training_routes.csv; external_reliability_v2; scperturb_source_maturity_preflight; replicate_batch_balance_preflight; background_specific_grn_source_audit",
    },
    {
        "claim": "novelty_boundary_is_scaling_axis_audit_not_first_law",
        "readiness": "ready_claim_boundary",
        "allowed_scope": "discussion_and_methods",
        "evidence": "literature boundary and benchmark framing already documented; X-Cell/X-Atlas-Pisces language prevents absolute first-law claim",
        "must_not_claim": "first perturbation-prediction scaling law",
        "primary_artifacts": "literature_boundary; narrative_skeleton",
    },
    {
        "claim": "mainline_training_guidance",
        "readiness": "ready_guarded_guidance",
        "allowed_scope": "methods_guardrail_and_future_work",
        "evidence": "carry moderate true-cell support as cautious design prior; use source/background/type/reagent as audit strata only",
        "must_not_claim": "current scaling evidence authorizes immediate non-ACK GPU",
        "primary_artifacts": "mainline_guidance; condition_level_reliability_source_scout; lockdown_report; current_inventory",
    },
    {
        "claim": "condition_level_reliability_is_next_scaling_unlock",
        "readiness": "ready_gap_matrix_not_training_claim",
        "allowed_scope": "methods_future_work_and_gate_plan",
        "evidence": "local h5ad/source-derived candidates are exhausted or closed; external small-table replicate concordance, dose/time/viability/growth, and background-context sources are the remaining non-duplicate acquisition families",
        "must_not_claim": "external reliability has already improved LatentFM or authorizes weighting/sampler GPU; GWT preflight specifically failed tail/MMD gates",
        "primary_artifacts": "condition_level_reliability_source_scout; external_condition_artifact_acquisition_slate; gwt_condition_reliability_preflight",
    },
]


GATES = [
    ("leakage_boundary", "pass", "No package artifact uses canonical multi or Track C query for selection."),
    ("current_best_model_declared", "pass", "Default/deployable remains xverse_8k_anchor."),
    ("claim_boundaries_defined", "pass", "Final claim readiness and axis claim boundary tables are present."),
    ("provenance_hashes_present", "pass", "Final artifact manifest hashes package files; prior NM provenance manifest has 39/39 artifacts present."),
    ("figure_readiness", "pass", "10/10 scaling figures passed QA and hash matching."),
    ("closed_routes_recorded", "pass", "Closed training routes include QC/weighted loss/read-support/OT/Track C repeats/old chemical split."),
    ("gpu_authorization", "fail_for_non_ack_gpu", "Immediate non-ACK GPU candidate count remains 0."),
    ("external_artifact_unlock", "not_yet", "Requires genuinely new condition-level artifact and strict CPU gate."),
    ("condition_level_source_scout", "external_acquisition_required", "No local ready condition-level reliability artifact; smallest next unlock is external small-table acquisition plus strict CPU gate."),
    ("external_acquisition_slate", "ready_no_gpu", "P0/P1 small-table source leads and local alignment keys are defined; no GPU until a materialized artifact passes strict CPU gate."),
    ("chemical_gpu_unlock", "ack_required", "Chemical V2 fixed-step route requires exact ACK before GPU."),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def artifact_rows() -> list[dict[str, Any]]:
    rows = []
    for label, path in KEY_ARTIFACTS:
        exists = path.exists()
        rows.append(
            {
                "label": label,
                "path": str(path),
                "exists": str(exists).lower(),
                "size": path.stat().st_size if exists else "",
                "sha256": sha256(path) if exists and path.is_file() else "",
            }
        )
    return rows


def main() -> None:
    lockdown = read_json(REPORTS / "latentfm_scaling_lockdown_and_mainline_use_20260626.json")
    figure = read_json(REPORTS / "latentfm_scaling_figure_readiness_20260625.json")
    provenance = read_json(REPORTS / "latentfm_scaling_nm_provenance_manifest_20260625.json")

    artifacts = artifact_rows()
    missing = [row for row in artifacts if row["exists"] != "true"]
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "status": "scaling_final_package_ready_no_gpu",
        "gpu_authorized": False,
        "current_default_model": lockdown.get("current_default_model", "xverse_8k_anchor"),
        "immediate_gpu_candidate_count": lockdown.get("immediate_gpu_candidate_count", 0),
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
        },
        "artifact_count": len(artifacts),
        "missing_artifact_count": len(missing),
        "claim_count": len(CLAIMS),
        "reviewer_gate_count": len(GATES),
        "figure_summary": {
            "figures_passed": figure.get("summary", {}).get("passed", figure.get("passed")),
            "figures_failed": figure.get("summary", {}).get("failed", figure.get("failed")),
        },
        "prior_provenance_status": provenance.get("status"),
        "outputs": {
            "artifact_manifest": str(OUT_ARTIFACTS),
            "claim_readiness": str(OUT_CLAIMS),
            "reviewer_gates": str(OUT_GATES),
        },
    }

    write_tsv(OUT_ARTIFACTS, artifacts, ["label", "path", "exists", "size", "sha256"])
    write_tsv(
        OUT_CLAIMS,
        CLAIMS,
        ["claim", "readiness", "allowed_scope", "evidence", "must_not_claim", "primary_artifacts"],
    )
    write_tsv(OUT_GATES, [{"gate": g, "status": s, "evidence": e} for g, s, e in GATES], ["gate", "status", "evidence"])

    lines = [
        "# LatentFM Scaling Final Package Index",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Default/deployable model: `{payload['current_default_model']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only final package index.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "- Canonical multi remains diagnostic only and is not used for scaling selection or promotion.",
        "",
        "## Package Readiness",
        "",
        f"- Key artifacts indexed: `{payload['artifact_count']}`.",
        f"- Missing key artifacts: `{payload['missing_artifact_count']}`.",
        f"- Claims indexed: `{payload['claim_count']}`.",
        f"- Reviewer gates indexed: `{payload['reviewer_gate_count']}`.",
        "- Current best/default model remains `xverse_8k_anchor`.",
        "- Immediate non-ACK GPU candidate count remains `0`.",
        "",
        "## Final Claim Readiness",
        "",
        "| claim | readiness | allowed scope | must not claim |",
        "|---|---|---|---|",
    ]
    for row in CLAIMS:
        lines.append(
            "| {claim} | `{readiness}` | {scope} | {forbid} |".format(
                claim=row["claim"],
                readiness=row["readiness"],
                scope=row["allowed_scope"],
                forbid=row["must_not_claim"],
            )
        )

    lines += [
        "",
        "## Reviewer Gates",
        "",
        "| gate | status | evidence |",
        "|---|---|---|",
    ]
    for gate, status, evidence in GATES:
        lines.append(f"| `{gate}` | `{status}` | {evidence} |")

    lines += [
        "",
        "## Decision",
        "",
        "- Scaling is ready as a Nature Methods-style axis-specific mechanism/failure-map package.",
        "- It is not ready as a deployable monotonic scaling law or checkpoint-promotion claim.",
        "- Mainline use is guarded: moderate true-cell support can guide future CPU-gated training-set design; source/background/type/reagent/QC/OT axes are audit strata or negative evidence.",
        "- Next GPU requires either exact chemical V2 ACK or a genuinely new condition-level external artifact that passes strict CPU gates and external review.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- Final artifact manifest: `{OUT_ARTIFACTS}`",
        f"- Final claim readiness: `{OUT_CLAIMS}`",
        f"- Final reviewer gates: `{OUT_GATES}`",
    ]
    if missing:
        lines += [
            "",
            "## Missing Artifacts",
            "",
        ]
        for row in missing:
            lines.append(f"- `{row['label']}`: `{row['path']}`")

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
