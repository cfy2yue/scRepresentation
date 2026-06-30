#!/usr/bin/env python3
"""Build a locked scaling-axis decision report.

This is a CPU-only synthesis over completed reports. It does not inspect
checkpoints, canonical multi, Track C query outputs, or any expression matrix.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

AXIS_JSON = REPORTS / "latentfm_scaling_preregistered_axis_matrix_20260626.json"
SLATE_JSON = REPORTS / "latentfm_next_action_slate_20260626.json"
RELIABILITY_JSON = REPORTS / "latentfm_external_reliability_v2_preflight_20260626.json"
CLAIM_JSON = REPORTS / "latentfm_scaling_nm_claim_failure_package_20260625.json"
CATALOG_JSON = REPORTS / "latentfm_scperturb_catalog_source_preflight_20260626.json"
NOTEBOOK_JSON = REPORTS / "latentfm_scperturb_notebook_metadata_preflight_20260626.json"
SOURCE_MATURITY_JSON = REPORTS / "latentfm_scperturb_source_maturity_artifact_preflight_20260626.json"
REPLICATE_BATCH_JSON = REPORTS / "latentfm_replicate_batch_balance_artifact_preflight_20260626.json"
GRN_CONTEXT_JSON = REPORTS / "latentfm_background_specific_grn_context_source_audit_20260626.json"

OUT_MD = REPORTS / "LATENTFM_SCALING_LOCKDOWN_AND_MAINLINE_USE_20260626.md"
OUT_JSON = REPORTS / "latentfm_scaling_lockdown_and_mainline_use_20260626.json"
OUT_DIR = REPORTS / "scaling_lockdown_and_mainline_use_20260626"
OUT_AXIS_CSV = OUT_DIR / "axis_lockdown.csv"
OUT_NEXT_CSV = OUT_DIR / "next_candidates.csv"
OUT_CLOSED_CSV = OUT_DIR / "closed_training_routes.csv"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def claim_tier(axis: dict) -> str:
    status = axis.get("claim_status", "")
    current = axis.get("current_status", "")
    if "not_promoted" in status or "mechanism_positive" in current:
        return "mechanism_positive_no_promotion"
    if "failed" in current or "negative" in status or "failure" in status:
        return "failure_map_or_negative"
    if "hint" in status or "hint" in current:
        return "hint_only"
    if "ack" in current or axis.get("axis") == "chemical_scaffold":
        return "ack_gated_protocol_route"
    if axis.get("axis") == "noharm_transfer_calibration":
        return "veto_framework"
    return "diagnostic_only"


def manuscript_use(tier: str) -> str:
    return {
        "mechanism_positive_no_promotion": "main_text_mechanism_with_noharm_veto",
        "failure_map_or_negative": "failure_map_or_supplement",
        "hint_only": "supplement_hint_only",
        "ack_gated_protocol_route": "separate_protocol_branch",
        "veto_framework": "methods_and_decision_framework",
        "diagnostic_only": "supplement_diagnostic",
    }.get(tier, "supplement_diagnostic")


def mainline_permission(axis: dict, tier: str) -> str:
    axis_name = axis.get("axis", "")
    if axis_name == "true_cell_per_condition_support":
        return "use moderate support as training-set design prior only after non-noop tail/no-harm CPU gate"
    if axis_name == "reagent_read_support":
        return "audit stratum only; no reliability weighting from current source-block-failed evidence"
    if axis_name in {"background_source_breadth", "perturbation_type_breadth", "condition_exposure_count"}:
        return "avoid naive broadening; require matched controls and dataset-tail pass before training changes"
    if axis_name == "chemical_scaffold":
        return "GPU only through explicit ACK-gated V2 fixed-step route"
    if axis_name == "ot_minibatch_pairs":
        return "keep default off for claims and do not run OT sweeps without new pair-quality gate"
    if tier == "hint_only":
        return "use as localization covariate only"
    return "no training change from current evidence"


def build_axis_rows(axis_data: dict) -> list[dict]:
    rows = []
    for row in axis_data.get("rows", []):
        tier = claim_tier(row)
        rows.append(
            {
                "axis": row.get("axis"),
                "lockdown_tier": tier,
                "current_status": row.get("current_status"),
                "claim_status": row.get("claim_status"),
                "manuscript_use": manuscript_use(tier),
                "mainline_permission": mainline_permission(row, tier),
                "gpu_authorized": str(bool(row.get("gpu_authorized", False))).lower(),
                "promotion_gate": row.get("promotion_gate"),
                "fail_close": row.get("fail_close"),
                "evidence": row.get("current_evidence"),
                "source_reports": row.get("source_reports"),
            }
        )
    return rows


def build_next_rows(slate: dict) -> list[dict]:
    rows = []
    for action in slate.get("actions", []):
        rows.append(
            {
                "priority": action.get("priority"),
                "candidate": action.get("name"),
                "type": action.get("type"),
                "gpu_now": str(bool(action.get("gpu_authorized_now", False))).lower(),
                "hypothesis": action.get("hypothesis"),
                "next_gate": action.get("next_gate"),
                "promotion_gate": action.get("promotion_gate"),
                "stop_rule": action.get("fail_close"),
            }
        )
    rows.append(
        {
            "priority": "1a",
            "candidate": "external_reliability_artifact_acquisition_v3",
            "type": "cpu_source_acquisition_or_preflight",
            "gpu_now": "false",
            "hypothesis": "Replicate concordance, dose/time/viability, or source-maturity metadata may explain reliable train conditions better than consumed read/UMI/QC proxies.",
            "next_gate": "materialize a condition-level dataset,condition,artifact_value table with >=2-3 datasets, overlap, within-dataset variation, bootstrap lower > 0, dataset min >= -0.02, MMD <= 0.001, and shuffle/LODO pass",
            "promotion_gate": "external review before any clipped sampler, weighted loss, or staged-training GPU smoke",
            "stop_rule": "if it reuses read/UMI/QC/assignment/multiplicity proxies, is one-source-only, fails MMD/tail/shuffle, or lacks condition-level overlap, keep as failure-map only",
        }
    )
    rows.append(
        {
            "priority": "R",
            "candidate": "matched_axis_failure_map_lockdown",
            "type": "reporting_and_reproducibility",
            "gpu_now": "false",
            "hypothesis": "A pre-registered axis matrix with explicit negative evidence is a valid Nature Methods-level scaling audit even without a promoted checkpoint.",
            "next_gate": "each axis has claim tier, matched-control status, bootstrap/LODO/tail/MMD/no-harm state, provenance, allowed wording, and failure cases",
            "promotion_gate": "not a promotion route",
            "stop_rule": "do not chase monotonic-law wording if no axis passes the promotion gate",
        }
    )
    return rows


def summarize_artifact_failures(label: str, data: dict) -> list[dict]:
    rows = []
    for artifact in data.get("artifacts", []):
        rows.append(
            {
                "route": f"{label}_{artifact.get('artifact')}",
                "reason": (
                    f"preflight {artifact.get('status')}; pp mean {artifact.get('pp_proxy_mean')}; "
                    f"dataset min {artifact.get('dataset_min_pp_proxy')}; MMD max {artifact.get('mmd_proxy_max')}; "
                    f"reasons {','.join(artifact.get('reasons', []))}"
                ),
                "reopen_condition": "do not mutate this proxy; acquire a genuinely new condition-level artifact family and rerun strict CPU gates",
            }
        )
    return rows


def build_closed_rows(reliability: dict, source_maturity: dict, replicate_batch: dict, grn_context: dict) -> list[dict]:
    closed = [
        {
            "route": "generic_qc_filtering_weighted_loss_hard_balancing",
            "reason": "QC/support reliability gates failed bootstrap/shuffle/tail controls.",
            "reopen_condition": "genuinely new external artifact family passes strict preflight and controls",
        },
        {
            "route": "reagent_read_support_weighted_sampler_or_staged_training",
            "reason": "MMD-safe residual signal exists but source-block/LODO gate failed within-dataset shuffle p=0.099900.",
            "reopen_condition": "new independent reliability artifact or stricter source-block signal beats shuffle/null",
        },
        {
            "route": "true_cell_exact_condition_s0_noop_repairs",
            "reason": "Canonical footprint/no-harm gates failed or produced no-op routes.",
            "reopen_condition": "non-noop tail-protection CPU policy with nonzero canonical footprint and frozen no-harm pass",
        },
        {
            "route": "ot_pairmode_cost_sweeps",
            "reason": "OT is wired but pair-quality and OT variants failed to unlock no-harm.",
            "reopen_condition": "new pair-quality feature predicts failed tails and defines bounded non-duplicate route",
        },
        {
            "route": "source_maturity_or_publication_year_weighting",
            "reason": "scPerturb source-maturity artifacts have limited coverage, no useful within-dataset variation, negative pp mean, and MMD/tail failures.",
            "reopen_condition": "real condition-level maturity/dose/time/viability artifacts pass overlap, within-dataset variation, bootstrap, tail, MMD, shuffle, and LODO gates",
        },
        {
            "route": "replicate_or_batch_balance_weighting",
            "reason": "replicate/batch balance artifacts are sparse and have negative pp mean with tail/MMD failures.",
            "reopen_condition": "independent replicate concordance or batch-quality artifacts show positive within-dataset signal and no-harm-safe tails",
        },
        {
            "route": "background_specific_grn_conditioning_from_existing_omnipath",
            "reason": "existing OmniPath/CollecTRI/DoRothEA files are gene-level or TF-target pair-level, not background-specific condition-level GRN context.",
            "reopen_condition": grn_context.get("decision", {}).get("reopen_condition", "genuinely background-specific GRN source passes strict CPU gate"),
        },
        {
            "route": "trackc_routed_distill_support_context_repeats",
            "reason": "Closed before held-out query; support/canonical gates failed.",
            "reopen_condition": "materially new support-only mechanism on safe trainselect split",
        },
        {
            "route": "chemical_old_same_split_seed42",
            "reason": "Same-split seed controls were unstable; V2 fixed-step protocol supersedes it.",
            "reopen_condition": "exact ACK for V2 fixed-step real/control matrix",
        },
    ]
    closed.extend(summarize_artifact_failures("existing_metadata", reliability))
    closed.extend(summarize_artifact_failures("source_maturity", source_maturity))
    closed.extend(summarize_artifact_failures("replicate_batch", replicate_batch))
    return closed


def source_lead_summary(catalog: dict, notebooks: dict, source_maturity: dict, replicate_batch: dict, grn_context: dict) -> dict:
    return {
        "catalog_status": catalog.get("status"),
        "matched_local_datasets": catalog.get("matched_local_dataset_count"),
        "matched_catalog_rows": catalog.get("row_count"),
        "reagent_route_rows": catalog.get("reagent_candidate_rows"),
        "maturity_route_rows": catalog.get("maturity_candidate_rows"),
        "notebook_status": notebooks.get("status"),
        "notebooks_scanned": notebooks.get("notebook_count"),
        "maturity_route_notebooks": notebooks.get("route_support", {}).get("maturity_route_notebooks"),
        "viability_route_notebooks": notebooks.get("route_support", {}).get("viability_route_notebooks"),
        "source_maturity_status": source_maturity.get("status"),
        "source_maturity_artifacts": len(source_maturity.get("artifacts", [])),
        "replicate_batch_status": replicate_batch.get("status"),
        "replicate_batch_artifacts": len(replicate_batch.get("artifacts", [])),
        "grn_context_status": grn_context.get("status"),
        "grn_context_action": grn_context.get("decision", {}).get("action"),
    }


def main() -> None:
    axis_data = read_json(AXIS_JSON)
    slate = read_json(SLATE_JSON)
    reliability = read_json(RELIABILITY_JSON)
    claim = read_json(CLAIM_JSON)
    catalog = read_json(CATALOG_JSON)
    notebooks = read_json(NOTEBOOK_JSON)
    source_maturity = read_json(SOURCE_MATURITY_JSON)
    replicate_batch = read_json(REPLICATE_BATCH_JSON)
    grn_context = read_json(GRN_CONTEXT_JSON)

    axis_rows = build_axis_rows(axis_data)
    next_rows = build_next_rows(slate)
    closed_rows = build_closed_rows(reliability, source_maturity, replicate_batch, grn_context)
    source_summary = source_lead_summary(catalog, notebooks, source_maturity, replicate_batch, grn_context)

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M CST"),
        "status": "scaling_lockdown_no_immediate_gpu",
        "current_default_model": axis_data.get("current_default_model", "xverse_8k_anchor"),
        "gpu_authorized": False,
        "immediate_gpu_candidate_count": slate.get("immediate_gpu_candidate_count", 0),
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
        },
        "axis_rows": axis_rows,
        "next_candidates": next_rows,
        "closed_training_routes": closed_rows,
        "source_lead_summary": source_summary,
        "claim_package_status": claim.get("status"),
        "outputs": {
            "axis_lockdown": str(OUT_AXIS_CSV),
            "next_candidates": str(OUT_NEXT_CSV),
            "closed_training_routes": str(OUT_CLOSED_CSV),
        },
    }

    write_csv(
        OUT_AXIS_CSV,
        axis_rows,
        [
            "axis",
            "lockdown_tier",
            "current_status",
            "claim_status",
            "manuscript_use",
            "mainline_permission",
            "gpu_authorized",
            "promotion_gate",
            "fail_close",
            "evidence",
            "source_reports",
        ],
    )
    write_csv(
        OUT_NEXT_CSV,
        next_rows,
        ["priority", "candidate", "type", "gpu_now", "hypothesis", "next_gate", "promotion_gate", "stop_rule"],
    )
    write_csv(OUT_CLOSED_CSV, closed_rows, ["route", "reason", "reopen_condition"])

    lines = [
        "# LatentFM Scaling Lockdown And Mainline Use",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Default/deployable model: `{payload['current_default_model']}`",
        "",
        "Immediate GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed scaling, source-artifact, and inventory reports.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "- Canonical single/family evidence is treated only as frozen no-harm veto context where prior reports already used it.",
        "",
        "## Lockdown Summary",
        "",
        "- Scaling is locked as an axis-specific mechanism/failure-map package, not a deployable monotonic scaling law.",
        "- Current default remains `xverse_8k_anchor`; no scaling-derived checkpoint is promoted.",
        "- Existing read/UMI/QC/assignment/multiplicity proxies cannot be repackaged into weighted sampler/loss GPU routes.",
        "- A new GPU route requires either exact chemical V2 ACK or a genuinely new external artifact family that passes strict CPU controls.",
        "",
        "## Axis Lockdown",
        "",
        "| axis | tier | manuscript use | mainline permission | GPU |",
        "|---|---|---|---|---|",
    ]
    for row in axis_rows:
        lines.append(
            "| {axis} | {tier} | {use} | {perm} | `{gpu}` |".format(
                axis=row["axis"],
                tier=row["lockdown_tier"],
                use=row["manuscript_use"],
                perm=row["mainline_permission"].replace("|", "/"),
                gpu=row["gpu_authorized"],
            )
        )

    lines += [
        "",
        "## Source Lead Reality Check",
        "",
        f"- scPerturb catalog status: `{source_summary.get('catalog_status')}`.",
        f"- Matched local datasets/rows: `{source_summary.get('matched_local_datasets')}` / `{source_summary.get('matched_catalog_rows')}`.",
        f"- Maturity/time candidate rows: `{source_summary.get('maturity_route_rows')}`; notebook maturity leads: `{source_summary.get('maturity_route_notebooks')}`.",
        f"- Notebook viability leads: `{source_summary.get('viability_route_notebooks')}`.",
        f"- Source-maturity preflight: `{source_summary.get('source_maturity_status')}` across `{source_summary.get('source_maturity_artifacts')}` artifacts.",
        f"- Replicate/batch balance preflight: `{source_summary.get('replicate_batch_status')}` across `{source_summary.get('replicate_batch_artifacts')}` artifacts.",
        f"- Background-specific GRN context audit: `{source_summary.get('grn_context_status')}`; action `{source_summary.get('grn_context_action')}`.",
        "- These are source leads, not condition-level artifacts; no GPU is authorized from catalog/notebook metadata alone.",
        "",
        "## Next Candidates",
        "",
        "| priority | candidate | type | GPU now | gate |",
        "|---:|---|---|---|---|",
    ]
    for row in next_rows:
        lines.append(
            "| {priority} | `{candidate}` | `{type}` | `{gpu}` | {gate} |".format(
                priority=row["priority"],
                candidate=row["candidate"],
                type=row["type"],
                gpu=row["gpu_now"],
                gate=str(row["next_gate"]).replace("|", "/"),
            )
        )

    lines += [
        "",
        "## Closed Training Routes",
        "",
        "| route | reason | reopen condition |",
        "|---|---|---|",
    ]
    for row in closed_rows:
        lines.append(
            "| `{route}` | {reason} | {reopen} |".format(
                route=row["route"],
                reason=str(row["reason"]).replace("|", "/"),
                reopen=str(row["reopen_condition"]).replace("|", "/"),
            )
        )

    lines += [
        "",
        "## Decision",
        "",
        "- Treat scaling as report-ready for mechanism/failure-map claims after final narrative polish.",
        "- For mainline training, only carry forward moderate true-cell support as a cautious design prior and source/background/type/reagent fields as audit strata.",
        "- Do not run generic balancing, generic weighted loss, OT sweeps, old allmod/type broadening, or read-support weighting from current evidence.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- Axis CSV: `{OUT_AXIS_CSV}`",
        f"- Next candidates CSV: `{OUT_NEXT_CSV}`",
        f"- Closed routes CSV: `{OUT_CLOSED_CSV}`",
    ]

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
