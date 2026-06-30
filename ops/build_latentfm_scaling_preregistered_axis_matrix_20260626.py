#!/usr/bin/env python3
"""Build the current pre-registered LatentFM scaling axis matrix.

CPU-only synthesis. This script does not read checkpoints, canonical multi,
Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

EVIDENCE_JSON = REPORTS / "latentfm_scaling_evidence_table_20260625.json"
REAGENT_SIGNAL_JSON = REPORTS / "latentfm_reagent_read_support_combined_signal_gate_20260626.json"
REAGENT_PREFLIGHT_JSON = REPORTS / "latentfm_reagent_read_support_combined_preflight_20260626.json"
REAGENT_SOURCE_BLOCK_JSON = REPORTS / "latentfm_reagent_read_support_source_block_lodo_gate_20260626.json"
INVENTORY_JSON = REPORTS / "latentfm_current_gpu_candidate_inventory_20260625.json"

OUT_JSON = REPORTS / "latentfm_scaling_preregistered_axis_matrix_20260626.json"
OUT_CSV = REPORTS / "latentfm_scaling_preregistered_axis_matrix_20260626.csv"
OUT_MD = REPORTS / "LATENTFM_SCALING_PREREGISTERED_AXIS_MATRIX_20260626.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_rows(axis: str, evidence: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in evidence.get("rows", []) if r.get("axis") == axis]


def first_row(axis: str, evidence: dict[str, Any], estimand: str | None = None) -> dict[str, Any]:
    rows = evidence_rows(axis, evidence)
    if estimand is not None:
        rows = [r for r in rows if r.get("estimand") == estimand]
    return rows[0] if rows else {}


def reagent_summary(name: str, signal: dict[str, Any]) -> dict[str, Any]:
    for row in signal.get("summaries", []):
        if row.get("artifact") == name:
            return row
    return {}


def status_from_inventory(branch: str, inventory: dict[str, Any]) -> dict[str, Any]:
    for row in inventory.get("inventory", []):
        if row.get("branch") == branch:
            return row
    for row in inventory.get("branches", []):
        if row.get("branch") == branch:
            return row
    for row in inventory.get("rows", []):
        if row.get("branch") == branch:
            return row
    return {}


def add_axis(
    rows: list[dict[str, Any]],
    *,
    axis: str,
    hypothesis: str,
    current_status: str,
    current_evidence: str,
    claim_status: str,
    selection_boundary: str,
    promotion_gate: str,
    fail_close: str,
    mainline_use: str,
    gpu_authorized: bool,
    source_reports: list[str],
) -> None:
    rows.append(
        {
            "axis": axis,
            "hypothesis": hypothesis,
            "current_status": current_status,
            "current_evidence": current_evidence,
            "claim_status": claim_status,
            "selection_boundary": selection_boundary,
            "promotion_gate": promotion_gate,
            "fail_close": fail_close,
            "mainline_use": mainline_use,
            "gpu_authorized": gpu_authorized,
            "source_reports": ";".join(source_reports),
        }
    )


def build_rows() -> list[dict[str, Any]]:
    evidence = load_json(EVIDENCE_JSON)
    reagent_signal = load_json(REAGENT_SIGNAL_JSON)
    reagent_preflight = load_json(REAGENT_PREFLIGHT_JSON)
    reagent_source_block = load_json(REAGENT_SOURCE_BLOCK_JSON)
    inventory = load_json(INVENTORY_JSON)

    rows: list[dict[str, Any]] = []
    tc128 = first_row("true_cell_count", evidence, "nested_6k_budget128")
    tc64 = first_row("true_cell_count", evidence, "nested_6k_budget64")
    noharm = evidence_rows("canonical_noharm", evidence)
    condition_count = evidence_rows("condition_count_or_breadth", evidence)
    source_rows = evidence_rows("background_type_source", evidence)
    target = first_row("target_observability", evidence)
    chemical = evidence_rows("chemical_holdout", evidence)
    noharm_transfer = first_row("noharm_transfer", evidence)
    read_support = reagent_summary("external_reagent_read_or_guide_support", reagent_signal)
    umi_support = reagent_summary("external_reagent_mean_umi_count", reagent_signal)
    reagent_branch = status_from_inventory("reagent_read_support_source_artifacts", inventory)

    add_axis(
        rows,
        axis="true_cell_per_condition_support",
        hypothesis="More true cells per condition can improve internal generalization only if tail/no-harm transfer is protected.",
        current_status="mechanism_positive_but_noharm_failed",
        current_evidence=(
            f"budget128 6k cross/family/neg_tails {tc128.get('primary_metric')}/"
            f"{tc128.get('secondary_metric')}/{tc128.get('tail_metric')}; "
            f"canonical no-harm rows={len(noharm)} all veto promotion"
        ),
        claim_status="mechanism_only_not_promoted",
        selection_boundary="train-only/internal for route discovery; frozen canonical single/family only as no-harm veto; no canonical multi/Track C query",
        promotion_gate="new non-noop tail-protected route with internal cross/family >= +0.02, MMD no-harm, dataset min >= -0.02, bootstrap lower > 0, nonzero canonical footprint, and frozen no-harm pass",
        fail_close="close if footprint is zero, tail remains unsafe, or route duplicates exact-condition/S0/uncertainty fallback",
        mainline_use="support-aware staged training is plausible only after a fresh CPU footprint/no-harm gate",
        gpu_authorized=False,
        source_reports=[
            "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_STABILITY_6K_DECISION_20260625.md",
            "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md",
            "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET64_TAIL_STABILITY_6K_DECISION_20260625.md",
        ],
    )

    add_axis(
        rows,
        axis="condition_exposure_count",
        hypothesis="Condition exposure has a moderate-support optimum rather than a monotonic more-is-better law.",
        current_status="diagnostic_negative_tail",
        current_evidence=f"condition/exposure rows={len(condition_count)}; cap120-cap30 mean positive but CI/tails fail in later gates",
        claim_status="nonmonotonic_mechanism_or_negative_evidence",
        selection_boundary="train-only matched row/dataset statistics; canonical only as frozen veto after route freeze",
        promotion_gate="matched count/source/background controls, row and dataset bootstrap lower > 0, LODO pass, no dataset-tail veto",
        fail_close="close if best subset remains dataset-tail unsafe or only beats count/source confounded controls",
        mainline_use="avoid blind full exposure; use moderate support and tail-audited sampling only",
        gpu_authorized=False,
        source_reports=[
            "reports/LATENTFM_CONDITION_COUNT_TAIL_SAFE_SUBSET_GATE_20260625.md",
            "reports/LATENTFM_SCALING_MIXED_EFFECT_LODO_CONDITION_COUNT_GATE_20260624.md",
        ],
    )

    add_axis(
        rows,
        axis="background_source_breadth",
        hypothesis="Cross-background/source breadth may help only if source and background confounding are controlled.",
        current_status="not_supported_currently",
        current_evidence=f"source/background rows={len(source_rows)}; current best policies remain CI/tail/source-weight unsafe",
        claim_status="failure_map",
        selection_boundary="source-verified train-only strata with LODO; no canonical multi/Track C query",
        promotion_gate="matched source/background/type strata, source-count controls, max dataset weight bounded, no dataset-tail veto",
        fail_close="close if effect is carried by one dataset/background or source-count control explains the gain",
        mainline_use="use background/source as audit strata before global normalization or balancing",
        gpu_authorized=False,
        source_reports=[
            "reports/LATENTFM_SOURCE_VERIFIED_BACKGROUND_TYPE_V2_GATE_20260625.md",
            "reports/LATENTFM_SCALING_SOURCE_RESOLVED_ESTIMAND_V2_GATE_20260625.md",
        ],
    )

    add_axis(
        rows,
        axis="perturbation_type_breadth",
        hypothesis="Gene/drug/type breadth requires type-specific mechanisms; naive all-modality mixing is not enough.",
        current_status="failed_family_tradeoff",
        current_evidence="allmod dose-aware 0/4 pass; family-stratified policies 0/56 pass; gene/drug tradeoff persists",
        claim_status="negative_evidence",
        selection_boundary="train-only family_gene/family_drug/test_all; no canonical multi/Track C query",
        promotion_gate="simultaneous all/gene/drug pp improvement, MMD no-harm, family hard-harm veto pass, shuffle/count controls pass",
        fail_close="close simple allmod replays or type routers that are explained by imbalance",
        mainline_use="do not merge modalities blindly; require family-aware controls first",
        gpu_authorized=False,
        source_reports=[
            "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_SMOKE_DECISION_20260625.md",
            "reports/LATENTFM_ALLMODALITY_FAMILY_STRATIFIED_PROTOCOL_GATE_20260625.md",
        ],
    )

    add_axis(
        rows,
        axis="target_observability",
        hypothesis="Observable/highly expressed targets may be easier, but target activity alone is tail-risky.",
        current_status="hint_only_tail_unsafe",
        current_evidence=f"target row primary={target.get('primary_metric')} secondary={target.get('secondary_metric')} tail={target.get('tail_metric')}",
        claim_status="weak_mechanism_clue_only",
        selection_boundary="train-only target/activity strata; no canonical multi/Track C query",
        promotion_gate="predeclared target strata pass permutation/control, dataset-tail, and hard-harm gates",
        fail_close="close if high-observability gains remain concentrated in unsafe tails",
        mainline_use="target observability may be an audit covariate, not a sampler/loss rule yet",
        gpu_authorized=False,
        source_reports=["reports/LATENTFM_TARGET_OBSERVABILITY_V2_GATE_20260625.md"],
    )

    add_axis(
        rows,
        axis="reagent_read_support",
        hypothesis="External reagent/read/UMI support can identify reliable train conditions and guide reliability-aware sampling/loss.",
        current_status=str(reagent_branch.get("state") or reagent_signal.get("status")),
        current_evidence=(
            "read z-spearman={}; read high-low={}; UMI z-spearman={}; UMI high-low={}; preflight status={}; source-block status={}".format(
                read_support.get("within_dataset_spearman_z_vs_pp"),
                read_support.get("dataset_high_minus_low_pp_mean"),
                umi_support.get("within_dataset_spearman_z_vs_pp"),
                umi_support.get("dataset_high_minus_low_pp_mean"),
                reagent_preflight.get("status"),
                reagent_source_block.get("status"),
            )
        ),
        claim_status="mechanism_clue_confound_gate_failed_no_gpu",
        selection_boundary="external train-condition artifacts plus train-only/internal outcome rows; no checkpoints, canonical multi, or Track C query",
        promotion_gate=">=2-3 datasets with overlap/variation, positive within-dataset signal, source/count/shuffle/LODO pass, no dataset-tail or MMD veto, external review before GPU",
        fail_close="close if still single-source, source/count fragile, or MMD unsafe after Frangieh/Dixit extraction",
        mainline_use="only after gate pass: clipped reliability-aware sampling, weighted loss, or staged training smoke",
        gpu_authorized=False,
        source_reports=[
            "reports/LATENTFM_REAGENT_READ_SUPPORT_COMBINED_SIGNAL_GATE_20260626.md",
            "reports/LATENTFM_REAGENT_READ_SUPPORT_COMBINED_PREFLIGHT_20260626.md",
            "reports/LATENTFM_REAGENT_READ_SUPPORT_SOURCE_HANDOFF_20260626.md",
        ],
    )

    add_axis(
        rows,
        axis="qc_local_obs_artifacts",
        hypothesis="Local QC/support metadata might identify reliable conditions, but broad QC fields are confounded.",
        current_status="failed_tail_mmd",
        current_evidence="local h5ad obs preflight had broad overlap but dataset min and MMD veto failed",
        claim_status="negative_evidence",
        selection_boundary="train-only metadata/outcome overlap only",
        promotion_gate="fresh external artifact with provenance, variation, bootstrap, shuffle, LODO, and MMD/tail pass",
        fail_close="do not launch generic QC filtering, hard balancing, or QC weighted loss from consumed obs columns",
        mainline_use="QC stays an audit stratum unless a new external artifact passes",
        gpu_authorized=False,
        source_reports=[
            "reports/LATENTFM_LOCAL_H5AD_OBS_ARTIFACT_PREFLIGHT_20260626.md",
            "reports/LATENTFM_NEW_TRAINONLY_ARTIFACT_OVERLAP_GATE_20260626.md",
        ],
    )

    add_axis(
        rows,
        axis="ot_minibatch_pairs",
        hypothesis="Better OT/minibatch pairing should improve condition transport if pair quality explains failed tails.",
        current_status="closed_default_off_for_claims",
        current_evidence="OT is non-noop, but pair-quality/failure-tail correlation and OT variants fail to unlock no-harm",
        claim_status="negative_evidence",
        selection_boundary="train-only pair-quality and existing internal metrics; no canonical multi/Track C query",
        promotion_gate="new pair-quality feature must correlate with failed tails and define a bounded non-duplicate GPU route",
        fail_close="no assignment/hungarian/random/no-OT sweeps without a new CPU explanatory signal",
        mainline_use="keep current default; do not spend GPU on OT mode sweeps",
        gpu_authorized=False,
        source_reports=[
            "reports/LATENTFM_OT_PAIR_QUALITY_GATE_20260625.md",
            "reports/LATENTFM_OT_PAIR_QUALITY_FAILURE_CORRELATION_GATE_20260625.md",
            "reports/LATENTFM_OT_PAIRING_EFFECTIVENESS_AUDIT_20260625.md",
        ],
    )

    add_axis(
        rows,
        axis="chemical_scaffold",
        hypothesis="Chemical/scaffold scaling may require independent scaffold splits and descriptor controls.",
        current_status="protocol_safe_ack_required",
        current_evidence=f"chemical rows={len(chemical)}; V2 fixed-step launcher is protocol-safe but exact ACK-gated",
        claim_status="not_non_ack_scaling_candidate",
        selection_boundary="V2 trainselect/held-out scaffold controls only after explicit protocol ACK",
        promotion_gate="real descriptor seeds replicate against shuffled/random controls with no family-drug hard harm",
        fail_close="do not use old same-split chemical pass; seed controls were weak and consumed",
        mainline_use="separate ACK-gated chemical branch, not current non-ACK scaling route",
        gpu_authorized=False,
        source_reports=[
            "reports/LATENTFM_CHEMICAL_V2_ACK_LAUNCH_PACKET_20260626.md",
            "reports/LATENTFM_CHEMICAL_V2_FIXEDSTEP_LAUNCHER_PROTOCOL_AUDIT_20260625.md",
            "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_V2_EXTERNAL_AUDIT_LORENTZ_20260625.md",
        ],
    )

    add_axis(
        rows,
        axis="noharm_transfer_calibration",
        hypothesis="Internal scaling gains are useful only if a frozen route transfers without canonical single/family harm.",
        current_status="veto_framework_active",
        current_evidence=f"noharm surrogate row primary={noharm_transfer.get('primary_metric')} secondary={noharm_transfer.get('secondary_metric')}",
        claim_status="required_veto_not_positive_axis",
        selection_boundary="canonical single/family used only after route freeze; canonical multi diagnostic only; Track C query excluded",
        promotion_gate="frozen route passes cross-background and family-gene no-harm with MMD veto clear",
        fail_close="any route failing frozen no-harm remains mechanism-only, not deployable",
        mainline_use="separate insight generation from checkpoint promotion",
        gpu_authorized=False,
        source_reports=[
            "reports/LATENTFM_SCALING_NOHARM_TRANSFER_CALIBRATION_GATE_20260624.md",
            "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md",
        ],
    )

    return rows


def render_md(payload: dict[str, Any]) -> str:
    reagent_row = next((row for row in payload["rows"] if row.get("axis") == "reagent_read_support"), {})
    reagent_state = str(reagent_row.get("current_status") or "")
    if "source_complete_positive_signal_but_confound_gate_failed" in reagent_state:
        reagent_decision = (
            "- The reagent/read-support route has completed source extraction and is closed for GPU from current evidence: "
            "positive MMD-safe signal exists, but source-block/LODO confound gate failed."
        )
    else:
        reagent_decision = (
            "- The open source-dependent route is `reagent_read_support`; rerun the post-source pipeline once Frangieh/Dixit source files complete."
        )
    lines = [
        "# LatentFM Scaling Pre-Registered Axis Matrix",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"Default/deployable model: `{payload['current_default_model']}`",
        "",
        f"Immediate GPU authorized: `{payload['immediate_gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of already completed gates and current source-artifact gate state.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C held-out query, or use GPU.",
        "- This is the registered decision matrix for scaling-axis completion; source reports remain authoritative.",
        "",
        "## Axis Matrix",
        "",
        "| axis | status | claim | evidence | promotion gate | mainline use | GPU |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| `{axis}` | `{status}` | `{claim}` | {evidence} | {gate} | {use} | `{gpu}` |".format(
                axis=row["axis"],
                status=row["current_status"],
                claim=row["claim_status"],
                evidence=row["current_evidence"],
                gate=row["promotion_gate"],
                use=row["mainline_use"],
                gpu=row["gpu_authorized"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No scaling axis currently authorizes non-ACK GPU.",
            reagent_decision,
            "- If a route passes its gate, trigger external review before a bounded GPU smoke.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- CSV: `{OUT_CSV}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = build_rows()
    payload = {
        "status": "scaling_preregistered_axis_matrix_no_immediate_gpu",
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
        },
        "inputs": {
            "evidence_json": str(EVIDENCE_JSON),
            "reagent_signal_json": str(REAGENT_SIGNAL_JSON),
            "reagent_preflight_json": str(REAGENT_PREFLIGHT_JSON),
            "inventory_json": str(INVENTORY_JSON),
        },
        "current_default_model": "xverse_8k_anchor",
        "immediate_gpu_authorized": any(bool(r["gpu_authorized"]) for r in rows),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(OUT_JSON), "out_csv": str(OUT_CSV), "out_md": str(OUT_MD)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
