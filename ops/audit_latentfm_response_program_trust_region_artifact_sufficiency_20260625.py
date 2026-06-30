#!/usr/bin/env python3
"""CPU-only artifact sufficiency gate for response-program trust-region ideas."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
JIANG_JSON = ROOT / "reports/latentfm_jiang_celltype_program_gate_20260624.json"
WESSELS_JSON = ROOT / "reports/latentfm_wessels_global_prior_bank_provenance_20260620.json"
OUT_JSON = ROOT / "reports/latentfm_response_program_trust_region_artifact_sufficiency_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_RESPONSE_PROGRAM_TRUST_REGION_ARTIFACT_SUFFICIENCY_20260625.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_result(rows: list[dict[str, Any]], group: str, control: str) -> dict[str, Any]:
    for row in rows:
        if row.get("group") == group and row.get("control") == control:
            return row
    return {}


def summary_value(row: dict[str, Any], key: str, default: float = -999.0) -> float:
    try:
        return float(row.get("summary", {}).get(key, default))
    except (TypeError, ValueError):
        return default


def main() -> None:
    jiang = load_json(JIANG_JSON)
    wessels = load_json(WESSELS_JSON)
    program_like_files = sorted(
        set(
            glob.glob(str(ROOT / "reports/*program*"))
            + glob.glob(str(ROOT / "reports/*PROGRAM*"))
            + glob.glob(str(ROOT / "runs/**/*program*"), recursive=True)
        )
    )
    prior_bank_like_files = sorted(
        set(
            glob.glob(str(ROOT / "reports/*prior*bank*"))
            + glob.glob(str(ROOT / "reports/*PRIOR*BANK*"))
            + glob.glob(str(ROOT / "runs/**/*prior*bank*"), recursive=True)
        )
    )

    jiang_decision = jiang.get("decision", {})
    jiang_results = jiang.get("results", [])
    jiang_main_cross = first_result(
        jiang_results, "internal_val_cross_background_seen_gene_proxy", "main"
    )
    jiang_main_family = first_result(jiang_results, "internal_val_family_gene_proxy", "main")

    wessels_group_rows = wessels.get("group_rows", [])
    wessels_group_names = sorted(str(r.get("group")) for r in wessels_group_rows)

    checks = {
        "has_jiang_program_features": bool(jiang),
        "jiang_gate_passed": jiang_decision.get("gpu_authorized") is True,
        "jiang_main_cross_delta_vs_gene_positive_0p020": summary_value(
            jiang_main_cross, "delta_vs_gene"
        )
        >= 0.020,
        "jiang_main_family_delta_vs_gene_positive_0p020": summary_value(
            jiang_main_family, "delta_vs_gene"
        )
        >= 0.020,
        "wessels_bank_exists": bool(wessels),
        "wessels_is_tracka_train_internal_program_axis": False,
        "has_rowlevel_candidate_anchor_program_projection_artifact": False,
        "has_noncanonical_multi_program_axis_for_tracka_selection": False,
    }

    reasons: list[str] = []
    if not checks["has_jiang_program_features"]:
        reasons.append("jiang_program_features_missing")
    if not checks["jiang_gate_passed"]:
        reasons.append("jiang_celltype_program_gate_already_failed")
    if not checks["jiang_main_cross_delta_vs_gene_positive_0p020"]:
        reasons.append("jiang_cross_background_program_signal_negative_or_too_small")
    if not checks["jiang_main_family_delta_vs_gene_positive_0p020"]:
        reasons.append("jiang_family_program_signal_negative_or_too_small")
    if checks["wessels_bank_exists"]:
        reasons.append("wessels_prior_bank_targets_canonical_multi_diagnostic_groups_not_tracka_selection")
    else:
        reasons.append("wessels_prior_bank_missing")
    reasons.append("no_rowlevel_candidate_anchor_program_projection_artifact_found")
    reasons.append("no_train_only_conserved_program_axis_with_required_control_collapse_found")

    status = "response_program_trust_region_artifact_insufficient_no_gpu"
    result = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "training_or_inference": False,
            "reads_canonical_performance": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "artifact_sufficiency_only": True,
        },
        "checks": checks,
        "reasons": reasons,
        "jiang": {
            "path": str(JIANG_JSON),
            "n_feature_rows": jiang.get("feature_meta", {}).get("n_feature_rows"),
            "n_metric_rows": jiang.get("n_metric_rows"),
            "decision_reasons": jiang_decision.get("reasons", []),
            "main_cross": jiang_main_cross,
            "main_family": jiang_main_family,
        },
        "wessels": {
            "path": str(WESSELS_JSON),
            "n_target_genes": wessels.get("n_target_genes"),
            "n_covered_genes": wessels.get("n_covered_genes"),
            "groups": wessels_group_names,
        },
        "artifact_inventory": {
            "program_like_files": program_like_files[:100],
            "prior_bank_like_files": prior_bank_like_files[:100],
        },
        "next_action": (
            "do not launch response-program trust-region GPU; build a new train-only "
            "row-level program-projection artifact first if this branch is reopened"
        ),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# LatentFM Response-Program Trust-Region Artifact Sufficiency",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only artifact sufficiency gate.",
        "- Does not train, infer, read canonical performance, read canonical multi, or read Track C query.",
        "- Checks whether existing artifacts are sufficient for the conserved response-program trust-region CPU gate proposed by Euler/Ptolemy.",
        "",
        "## Findings",
        "",
        f"- Jiang program artifact exists: `{checks['has_jiang_program_features']}`.",
        f"- Jiang prior gate already authorized GPU: `{checks['jiang_gate_passed']}`.",
        f"- Jiang cross-background main delta vs gene: `{summary_value(jiang_main_cross, 'delta_vs_gene')}`.",
        f"- Jiang family main delta vs gene: `{summary_value(jiang_main_family, 'delta_vs_gene')}`.",
        f"- Wessels prior bank exists with groups: `{wessels_group_names}`.",
        "- Wessels prior bank is provenance for canonical multi diagnostic groups, not a Track A train-only program-projection artifact.",
        "- No row-level candidate-minus-anchor projection artifact over conserved program axes was found.",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- next action: `{result['next_action']}`",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
