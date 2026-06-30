#!/usr/bin/env python3
"""Track C external support-reliability source inventory gate.

This is a CPU/source gate only. It does not rerun support-set modeling. The
purpose is to decide whether any materially new external support reliability
artifact exists after the closed support-set, reagent/QC, and bulk-difficulty
branches.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_trackc_external_support_reliability_source_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_EXTERNAL_SUPPORT_RELIABILITY_SOURCE_GATE_20260627.md"

SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def status_from_md(path: Path) -> str:
    text = read_text(path)
    match = re.search(r"^Status:\s*`?([^`\n]+)`?", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def gpu_from_md(path: Path) -> bool | None:
    text = read_text(path).lower()
    if "gpu authorized: `true`" in text or "gpu authorized: true" in text:
        return True
    if "gpu authorized: `false`" in text or "gpu authorized: false" in text:
        return False
    return None


def report_rec(
    name: str,
    md: str,
    js: str | None,
    source_type: str,
    trackc_scope: str,
    consumed_or_closed: bool,
    material_new_external: bool,
    reason: str,
) -> dict[str, Any]:
    md_path = REPORTS / md
    json_path = REPORTS / js if js else None
    payload = read_json(json_path) if json_path else {}
    return {
        "name": name,
        "md": str(md_path),
        "json": str(json_path) if json_path else "",
        "exists": md_path.is_file(),
        "status": payload.get("status") or status_from_md(md_path),
        "gpu_authorized": payload.get("gpu_authorized", gpu_from_md(md_path)),
        "source_type": source_type,
        "trackc_scope": trackc_scope,
        "consumed_or_closed": consumed_or_closed,
        "material_new_external": material_new_external,
        "reason": reason,
    }


def main() -> int:
    rows = [
        report_rec(
            "trackc_support_set_source_plumbing",
            "LATENTFM_TRACKC_SUPPORT_SET_SOURCE_PLUMBING_20260627.md",
            "latentfm_trackc_support_set_source_plumbing_20260627.json",
            "internal_condition_mean_support_bank",
            "safe_trainselect_support_val_plumbing_only",
            True,
            False,
            "launcher plumbing only; not an external reliability source and later support-set smokes failed",
        ),
        report_rec(
            "trackc_query_conditioned_support_token",
            "LATENTFM_TRACKC_QUERY_CONDITIONED_SUPPORT_TOKEN_GATE_20260627.md",
            "latentfm_trackc_query_conditioned_support_token_gate_20260627.json",
            "internal_condition_mean_support_policy",
            "safe_trainselect_support_val_gate",
            True,
            False,
            "near-signal but Wessels/no-harm and p_harm gate failed; not material new external source",
        ),
        report_rec(
            "trackc_support_set_policy_sweep",
            "LATENTFM_TRACKC_SUPPORT_SET_POLICY_SWEEP_20260627.md",
            "latentfm_trackc_support_set_policy_sweep_20260627.json",
            "internal_condition_mean_support_policy",
            "safe_trainselect_support_val_gate",
            True,
            False,
            "aggregation-policy route already failed support-val no-harm",
        ),
        report_rec(
            "trackc_support_set_deepset_sketch",
            "LATENTFM_TRACKC_SUPPORT_SET_DEEPSET_SKETCH_GATE_20260627.md",
            "latentfm_trackc_support_set_deepset_sketch_gate_20260627.json",
            "internal_condition_mean_set_encoder_sketch",
            "safe_trainselect_support_val_gate",
            True,
            False,
            "materially different sketch was tested but Wessels/no-harm failed",
        ),
        report_rec(
            "trackc_row_reliability_v2_artifact",
            "LATENTFM_TRACKC_ROW_RELIABILITY_V2_ARTIFACT_GATE_20260624.md",
            "latentfm_trackc_row_reliability_v2_artifact_gate_20260624.json",
            "internal_trainmulti_cv_reliability_missing_rows",
            "safe_trainselect_support_val_gate",
            True,
            False,
            "row-level train_multi reliability artifact absent; old negative-row abstention consumed",
        ),
        report_rec(
            "external_reliability_v2_preflight",
            "LATENTFM_EXTERNAL_RELIABILITY_V2_PREFLIGHT_20260626.md",
            "latentfm_external_reliability_v2_preflight_20260626.json",
            "external_qc_or_assignment_support",
            "train_only_proxy_preflight",
            True,
            False,
            "all source files failed preflight; pass candidates empty",
        ),
        report_rec(
            "gwt_condition_reliability_artifact_preflight",
            "LATENTFM_GWT_CONDITION_RELIABILITY_ARTIFACT_PREFLIGHT_20260627.md",
            "latentfm_gwt_condition_reliability_artifact_preflight_20260627.json",
            "external_gene_level_reliability",
            "train_only_proxy_preflight",
            True,
            False,
            "all four artifacts failed dataset-tail and MMD veto",
        ),
        report_rec(
            "reagent_read_support_source_block_lodo",
            "LATENTFM_REAGENT_READ_SUPPORT_SOURCE_BLOCK_LODO_GATE_20260626.md",
            "latentfm_reagent_read_support_source_block_lodo_gate_20260626.json",
            "external_reagent_read_guide_support",
            "train_only_proxy_preflight",
            True,
            False,
            "source-block/LODO confound gate failed; QC/reagent support duplicate",
        ),
        report_rec(
            "adamson_author_guide_support_preview",
            "LATENTFM_ADAMSON_AUTHOR_GUIDE_SUPPORT_PREVIEW_GATE_20260627.md",
            "latentfm_adamson_author_guide_support_preview_gate_20260627.json",
            "external_single_source_guide_read_coverage",
            "test_single_diagnostic_only",
            True,
            False,
            "single-source guide/read/UMI/coverage preview only; explicitly not GPU-authorizing",
        ),
        report_rec(
            "norman_replogle_replicate_concordance",
            "LATENTFM_NORMAN_REPLOGLE_REPLICATE_CONCORDANCE_GATE_20260627.md",
            "latentfm_norman_replogle_replicate_concordance_gate_20260627.json",
            "external_schema_source_absent",
            "source_schema_only",
            True,
            False,
            "no true replicate-concordance or reproducibility column found",
        ),
    ]

    material_new = [
        row
        for row in rows
        if row["exists"]
        and row["material_new_external"]
        and not row["consumed_or_closed"]
        and row["gpu_authorized"] is True
    ]
    missing = [row for row in rows if not row["exists"]]
    reasons = [
        "no_material_new_external_trackc_support_reliability_artifact_found",
        "existing_support_set_sources_are_internal_condition_mean_routes_already_closed",
        "external_reliability_sources_are_qc_reagent_read_or_bulk_difficulty_routes_already_failed",
        "no_source_survives_dataset_tail_mmd_shuffle_or_source_block_controls",
        "safe_trainselect_boundary_has_no_new_support_reliability_signal",
        "no_gpu_from_source_inventory_gate",
    ]
    status = "trackc_external_support_reliability_source_gate_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "safe_trainselect_split": str(SAFE_SPLIT),
        "safe_trainselect_split_exists": SAFE_SPLIT.is_file(),
        "boundary": {
            "cpu_source_inventory_only": True,
            "training": False,
            "inference": False,
            "gpu": False,
            "canonical_multi_selection_used": False,
            "trackc_heldout_query_used": False,
            "full_v2_query_split_used": False,
        },
        "audited_routes": rows,
        "missing_reports": missing,
        "material_new_external_trackc_support_reliability_candidates": material_new,
        "reopen_criteria": [
            "condition-level external support/reliability/concordance signal, not just read/UMI/coverage/QC",
            "maps to safe trainselect support boundary without held-out query/full-v2 selection",
            "passes dataset-tail, MMD, shuffle, and source-block controls",
            "provides a predeclared no-harm adapter/router hypothesis and stop rule",
        ],
        "reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Track C External Support-Reliability Source Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/source inventory gate only.",
        "- No training, inference, GPU, canonical multi selection, full-v2 query split, or held-out Track C query.",
        f"- Safe trainselect split checked: `{SAFE_SPLIT}` (`exists={SAFE_SPLIT.is_file()}`).",
        "",
        "## Audited Routes",
        "",
        "| route | status | source type | Track C scope | closed/consumed | reason |",
        "|---|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| `{name}` | `{status}` | `{source_type}` | `{trackc_scope}` | `{closed}` | {reason} |".format(
                name=row["name"],
                status=row["status"] or "missing",
                source_type=row["source_type"],
                trackc_scope=row["trackc_scope"],
                closed=row["consumed_or_closed"],
                reason=row["reason"],
            )
        )
    lines += [
        "",
        "## Decision",
        "",
        "No GPU is authorized. I found no materially new external Track C support-reliability source beyond routes that are internal condition-mean support policies, QC/reagent/read/coverage metadata, bulk-difficulty sources, or already failed/consumed gates.",
        "",
        "## Reasons",
        "",
    ]
    lines += [f"- `{reason}`" for reason in reasons]
    lines += [
        "",
        "## Reopen Criteria",
        "",
    ]
    lines += [f"- {item}" for item in payload["reopen_criteria"]]
    lines += [
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "routes": len(rows), "missing": len(missing)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
