#!/usr/bin/env python3
"""Build a law-ready evidence table for LatentFM scaling.

CPU/report-only synthesis. This script inventories the current preregistered
axis matrix, unified matched/LODO gate, and post-Aristotle lockdown reports into
a reviewer-facing table of what is law-ready, what is mechanism/failure-map
ready, and what remains missing before any scaling-law or training claim.

It does not read checkpoints, canonical multi, Track C query outputs, expression
matrices, or launch training/inference/GPU work.
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
OUT_DIR = REPORTS / "scaling_law_ready_evidence_table_20260626"
OUT_MD = REPORTS / "LATENTFM_SCALING_LAW_READY_EVIDENCE_TABLE_20260626.md"
OUT_JSON = REPORTS / "latentfm_scaling_law_ready_evidence_table_20260626.json"
OUT_AXIS = OUT_DIR / "axis_law_readiness.csv"
OUT_CRITERIA = OUT_DIR / "axis_criteria_matrix.csv"
OUT_MISSING = OUT_DIR / "missing_experiment_matrix.csv"
OUT_INPUTS = OUT_DIR / "input_manifest.tsv"


INPUTS = [
    REPORTS / "latentfm_scaling_preregistered_axis_matrix_20260626.json",
    REPORTS / "latentfm_scaling_unified_matched_axis_lodo_gate_20260626.json",
    REPORTS / "latentfm_scaling_lockdown_and_mainline_use_20260626.json",
    REPORTS / "latentfm_scaling_law_completion_and_mainline_translation_20260626.json",
    REPORTS / "latentfm_truecell_scaling_count_tail_completion_gate_20260625.json",
    REPORTS / "latentfm_truecell_nonnoop_tail_protection_meta_gate_20260626.json",
    REPORTS / "latentfm_condition_exposure_row_bootstrap_gate_20260625.json",
    REPORTS / "latentfm_condition_exposure_hierarchical_bootstrap_lodo_gate_20260626.json",
    REPORTS / "latentfm_scaling_nested_condition_exposure_v2_gate_20260625.json",
    REPORTS / "latentfm_scaling_source_resolved_estimand_v2_gate_20260625.json",
    REPORTS / "latentfm_source_background_type_hierarchical_matched_gate_20260626.json",
    REPORTS / "latentfm_target_observability_residual_v3_gate_20260626.json",
    REPORTS / "latentfm_reagent_read_support_source_block_lodo_gate_20260626.json",
    REPORTS / "latentfm_scperturb_source_maturity_artifact_preflight_20260626.json",
    REPORTS / "latentfm_replicate_batch_balance_artifact_preflight_20260626.json",
    REPORTS / "latentfm_background_specific_grn_context_source_audit_20260626.json",
]


AXIS_ALIASES = {
    "background_source_breadth": "background_source_type_breadth",
    "target_observability": "target_observability_actionability",
    "reagent_read_support": "external_reagent_read_support",
    "perturbation_type_breadth": "perturbation_type_allmodality",
    "chemical_scaffold": "chemical_scaffold_semantics",
}


CRITERIA = [
    "pre_registered_estimand",
    "matched_or_lodo_control",
    "condition_or_row_bootstrap_ci",
    "dataset_bootstrap_or_lodo",
    "shuffle_or_negative_control",
    "dataset_tail_safety",
    "mmd_or_noharm_safety",
    "frozen_canonical_noharm_when_training_relevant",
    "artifact_provenance",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], *, delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def clean(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def axis_lookup(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {row.get(key, ""): row for row in rows}


def status_bucket(prereg: dict[str, Any], unified: dict[str, Any]) -> str:
    claim = prereg.get("claim_status", "")
    current = prereg.get("current_status", "")
    gate = unified.get("gate_status", "")
    if "chemical" in prereg.get("axis", ""):
        return "ack_gated_protocol_not_current_law"
    if unified.get("gpu_authorized") is True or gate == "pass_gpu_after_external_review":
        return "law_or_training_gate_passed"
    if "mechanism" in claim or "mechanism_positive" in current:
        return "mechanism_ready_no_promotion"
    if "negative" in claim or "failure" in claim or "failed" in current or "not_supported" in current:
        return "failure_map_ready"
    if "hint" in claim or "hint" in current:
        return "hint_only"
    return "diagnostic_or_guardrail"


def criterion_status(axis: str, criterion: str, prereg: dict[str, Any], unified: dict[str, Any]) -> tuple[str, str]:
    text = " ".join(
        clean(prereg.get(k, ""))
        for k in ["hypothesis", "selection_boundary", "promotion_gate", "current_evidence", "source_reports"]
    )
    utext = " ".join(clean(unified.get(k, "")) for k in ["matched_or_lodo_control", "tail_or_noharm", "evidence", "gate_status"])
    all_text = f"{text} {utext}".lower()
    gate = str(unified.get("gate_status", "")).lower()
    current = str(prereg.get("current_status", "")).lower()
    claim = str(prereg.get("claim_status", "")).lower()

    if criterion == "pre_registered_estimand":
        ok = all(prereg.get(k) for k in ["hypothesis", "selection_boundary", "promotion_gate", "fail_close"])
        return ("present", "axis has hypothesis, boundary, promotion gate, and fail-close") if ok else ("missing", "missing preregistration fields")
    if criterion == "matched_or_lodo_control":
        if "lodo" in all_text or "matched" in all_text or "control" in all_text:
            return ("present_but_failed" if "fail" in gate or "tail" in current else "present", clean(unified.get("matched_or_lodo_control", "")))
        return ("missing", "no matched/LODO control recorded")
    if criterion == "condition_or_row_bootstrap_ci":
        if axis == "true_cell_per_condition_support":
            return ("present_but_not_law_sufficient", "condition bootstrap lower bounds exist for 6k budget128, but monotonic law and no-harm fail")
        if "bootstrap" in all_text or "ci" in all_text:
            return ("present_but_failed", clean(unified.get("matched_or_lodo_control", "")))
        return ("missing", "no bootstrap/CI evidence recorded for this axis")
    if criterion == "dataset_bootstrap_or_lodo":
        if "dataset_bootstrap" in all_text or "lodo" in all_text or "dataset" in all_text:
            return ("present_but_failed" if ("tail" in all_text or "fail" in gate or "not_supported" in current) else "present", clean(unified.get("matched_or_lodo_control", "")))
        return ("missing", "no dataset bootstrap/LODO evidence recorded")
    if criterion == "shuffle_or_negative_control":
        if "shuffle" in all_text or "random" in all_text or "control" in all_text or "ack" in current:
            return ("present_but_failed" if ("fail" in gate or "not_supported" in current or "negative" in claim) else "present", clean(prereg.get("promotion_gate", "")))
        return ("missing", "negative/shuffle control not explicit")
    if criterion == "dataset_tail_safety":
        if "tail" in all_text or "dataset_min" in all_text or "hard_harm" in all_text:
            if any(token in all_text for token in ["unsafe", "negative tails", "tail_below", "failed", "noharm failed", "hard-harm gates fail"]):
                return ("failed", clean(unified.get("tail_or_noharm", "")))
            return ("present", clean(unified.get("tail_or_noharm", "")))
        return ("missing", "dataset-tail safety not quantified")
    if criterion == "mmd_or_noharm_safety":
        if "mmd" in all_text or "no-harm" in all_text or "noharm" in all_text:
            if any(token in all_text for token in ["failed", "unsafe", "veto", "mmd_max_above", "canonical no-harm failed"]):
                return ("failed", clean(unified.get("tail_or_noharm", "")))
            return ("present", clean(unified.get("tail_or_noharm", "")))
        return ("not_applicable_or_missing", "not explicit for this axis")
    if criterion == "frozen_canonical_noharm_when_training_relevant":
        training_relevant = axis in {
            "true_cell_per_condition_support",
            "condition_exposure_count",
            "background_source_breadth",
            "perturbation_type_breadth",
            "target_observability",
            "reagent_read_support",
            "qc_local_obs_artifacts",
            "chemical_scaffold",
        }
        if not training_relevant:
            return ("not_applicable", "axis is a guardrail or default-off route")
        if "canonical" in all_text or "no-harm" in all_text or "noharm" in all_text:
            return ("failed" if ("failed" in all_text or "veto" in all_text) else "present", clean(unified.get("tail_or_noharm", "")))
        return ("missing", "frozen canonical no-harm evidence not explicit")
    if criterion == "artifact_provenance":
        reports = [p for p in clean(prereg.get("source_reports", "")).split(";") if p]
        missing = [p for p in reports if not (ROOT / p).exists()]
        if missing:
            return ("incomplete", "missing source reports: " + ",".join(missing))
        return ("present", f"{len(reports)} source report(s) linked")
    return ("missing", "unknown criterion")


def missing_experiment(axis: str, bucket: str, prereg: dict[str, Any]) -> dict[str, Any]:
    if axis == "true_cell_per_condition_support":
        return {
            "axis": axis,
            "priority": "P0",
            "resource_class": "CPU first; GPU only after pass",
            "missing_experiment": "non-noop tail-protection gate with frozen canonical no-harm",
            "gate": "nonzero canonical footprint, internal cross/family >= +0.02, MMD <= +0.001, dataset min >= -0.02, bootstrap lower > 0, frozen no-harm pass",
            "gpu_ready_now": False,
        }
    if axis == "condition_exposure_count":
        return {
            "axis": axis,
            "priority": "P0",
            "resource_class": "CPU/report then possible GPU matrix",
            "missing_experiment": "pre-registered moderate-vs-full exposure estimand with hierarchical bootstrap and LODO",
            "gate": "row and dataset bootstrap lower > 0, LODO pass, leave-background/type mins >= 0, no dataset tail < -0.02",
            "gpu_ready_now": False,
        }
    if axis == "background_source_breadth":
        return {
            "axis": axis,
            "priority": "P1",
            "resource_class": "CPU gate before GPU",
            "missing_experiment": "source/background/type matched estimand with dataset-ID and source-count controls",
            "gate": "bootstrap lower > 0, dataset min >= -0.02, max dataset weight bounded, no source/background/type stratum negative",
            "gpu_ready_now": False,
        }
    if axis == "reagent_read_support":
        return {
            "axis": axis,
            "priority": "P1",
            "resource_class": "CPU source acquisition/preflight",
            "missing_experiment": "new condition-level reliability artifact distinct from read/UMI/QC/source/guide-count proxies",
            "gate": ">=3 datasets or strong LODO, >=20 overlap, within-dataset variation, shuffle p <= 0.01, MMD <= +0.001, dataset min >= -0.02",
            "gpu_ready_now": False,
        }
    if axis == "chemical_scaffold":
        return {
            "axis": axis,
            "priority": "ACK",
            "resource_class": "GPU only after exact user ACK",
            "missing_experiment": "chemical V2 fixed-step real Morgan512 seed43/44 followed by shuffled/random controls",
            "gate": "exact ACK, fresh resource audit, real descriptor seeds replicate and beat controls without family-drug hard harm",
            "gpu_ready_now": False,
        }
    return {
        "axis": axis,
        "priority": "P2" if bucket == "failure_map_ready" else "P1",
        "resource_class": "CPU gate before any GPU",
        "missing_experiment": clean(prereg.get("promotion_gate", "new matched, tail-safe, provenance-complete gate")),
        "gate": clean(prereg.get("fail_close", "")),
        "gpu_ready_now": False,
    }


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    prereg = read_json(REPORTS / "latentfm_scaling_preregistered_axis_matrix_20260626.json")
    unified = read_json(REPORTS / "latentfm_scaling_unified_matched_axis_lodo_gate_20260626.json")
    lockdown = read_json(REPORTS / "latentfm_scaling_lockdown_and_mainline_use_20260626.json")

    prereg_rows = prereg.get("rows", [])
    unified_by_axis = axis_lookup(unified.get("rows", []), "axis")

    axis_rows: list[dict[str, Any]] = []
    criteria_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for row in prereg_rows:
        axis = row.get("axis", "")
        unified_axis = AXIS_ALIASES.get(axis, axis)
        urow = unified_by_axis.get(unified_axis, {})
        bucket = status_bucket(row, urow)
        crit_statuses = {}
        failed_or_missing = []
        for criterion in CRITERIA:
            cstatus, evidence = criterion_status(axis, criterion, row, urow)
            crit_statuses[criterion] = cstatus
            if cstatus in {"missing", "failed", "incomplete", "present_but_failed", "present_but_not_law_sufficient"}:
                failed_or_missing.append(f"{criterion}:{cstatus}")
            criteria_rows.append(
                {
                    "axis": axis,
                    "criterion": criterion,
                    "status": cstatus,
                    "evidence": evidence,
                }
            )
        law_ready = not failed_or_missing and bucket == "law_or_training_gate_passed"
        manuscript_ready = bucket in {"mechanism_ready_no_promotion", "failure_map_ready", "hint_only", "diagnostic_or_guardrail", "ack_gated_protocol_not_current_law"}
        axis_rows.append(
            {
                "axis": axis,
                "claim_status": row.get("claim_status"),
                "current_status": row.get("current_status"),
                "law_readiness": "law_ready" if law_ready else "not_law_ready",
                "manuscript_readiness": "ready_as_mechanism_or_failure_map" if manuscript_ready else "not_ready",
                "bucket": bucket,
                "gpu_authorized": str(bool(row.get("gpu_authorized") or urow.get("gpu_authorized"))).lower(),
                "failed_or_missing_criteria": ";".join(failed_or_missing),
                "current_evidence": clean(row.get("current_evidence", "")),
                "selection_boundary": clean(row.get("selection_boundary", "")),
                "promotion_gate": clean(row.get("promotion_gate", "")),
                "mainline_use": clean(row.get("mainline_use", "")),
            }
        )
        missing_rows.append(missing_experiment(axis, bucket, row))

    input_rows = []
    for path in INPUTS:
        exists = path.exists()
        input_rows.append(
            {
                "path": str(path),
                "exists": str(exists).lower(),
                "size": path.stat().st_size if exists else "",
                "sha256": sha256(path) if exists and path.is_file() else "",
            }
        )

    write_csv(
        OUT_AXIS,
        axis_rows,
        [
            "axis",
            "claim_status",
            "current_status",
            "law_readiness",
            "manuscript_readiness",
            "bucket",
            "gpu_authorized",
            "failed_or_missing_criteria",
            "current_evidence",
            "selection_boundary",
            "promotion_gate",
            "mainline_use",
        ],
    )
    write_csv(OUT_CRITERIA, criteria_rows, ["axis", "criterion", "status", "evidence"])
    write_csv(OUT_MISSING, missing_rows, ["axis", "priority", "resource_class", "missing_experiment", "gate", "gpu_ready_now"])
    write_csv(OUT_INPUTS, input_rows, ["path", "exists", "size", "sha256"], delimiter="\t")

    law_ready_count = sum(row["law_readiness"] == "law_ready" for row in axis_rows)
    gpu_count = sum(row["gpu_authorized"] == "true" for row in axis_rows)
    missing_input_count = sum(row["exists"] != "true" for row in input_rows)
    payload = {
        "timestamp": timestamp,
        "status": "scaling_law_ready_evidence_table_no_immediate_gpu",
        "default_model": "xverse_8k_anchor",
        "axis_count": len(axis_rows),
        "law_ready_axis_count": law_ready_count,
        "gpu_authorized_axis_count": gpu_count,
        "missing_input_count": missing_input_count,
        "gpu_authorized": False,
        "immediate_gpu_candidate_count": 0,
        "boundary": {
            "cpu_only": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "trains_or_infers": False,
            "uses_gpu": False,
        },
        "lockdown_status": lockdown.get("status"),
        "outputs": {
            "axis_law_readiness": str(OUT_AXIS),
            "criteria_matrix": str(OUT_CRITERIA),
            "missing_experiment_matrix": str(OUT_MISSING),
            "input_manifest": str(OUT_INPUTS),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Scaling Law-Ready Evidence Table",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        "Status: `scaling_law_ready_evidence_table_no_immediate_gpu`",
        "",
        "Default/deployable model: `xverse_8k_anchor`",
        "",
        "Immediate non-ACK GPU candidate count: `0`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis over completed scaling and artifact-gate reports.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "- Canonical single/family evidence is used only as frozen no-harm veto context from completed reports.",
        "",
        "## Summary",
        "",
        f"- Axes audited: `{len(axis_rows)}`.",
        f"- Law-ready axes: `{law_ready_count}`.",
        f"- GPU-authorized axes: `{gpu_count}`.",
        f"- Missing inputs: `{missing_input_count}`.",
        "- Scaling is currently manuscript-ready as a mechanism/failure-map package, not as a systematic deployable scaling law.",
        "",
        "## Axis Readiness",
        "",
        "| axis | bucket | law readiness | manuscript readiness | failed/missing criteria |",
        "|---|---|---|---|---|",
    ]
    for row in axis_rows:
        lines.append(
            "| {axis} | `{bucket}` | `{law}` | `{manuscript}` | {criteria} |".format(
                axis=row["axis"],
                bucket=row["bucket"],
                law=row["law_readiness"],
                manuscript=row["manuscript_readiness"],
                criteria=clean(row["failed_or_missing_criteria"] or "none"),
            )
        )

    lines += [
        "",
        "## Minimal Missing Experiment Matrix",
        "",
        "| priority | axis | resource | missing experiment | GPU ready now |",
        "|---|---|---|---|---|",
    ]
    for row in missing_rows:
        lines.append(
            "| `{priority}` | `{axis}` | {resource} | {missing} | `{gpu}` |".format(
                priority=row["priority"],
                axis=row["axis"],
                resource=clean(row["resource_class"]),
                missing=clean(row["missing_experiment"]),
                gpu=str(row["gpu_ready_now"]).lower(),
            )
        )

    lines += [
        "",
        "## Decision",
        "",
        "- Keep `xverse_8k_anchor` as default/deployable.",
        "- Treat true-cell/per-condition support as a cautious training-set prior only after a new non-noop tail/no-harm CPU gate.",
        "- Treat source/background/type, source-maturity, replicate/batch, GRN, reagent/read-support, target/dependency/constraint, and OT as audit/failure-map strata unless a new gate passes.",
        "- Do not launch generic non-ACK scaling GPU from the current evidence table.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- Axis law readiness: `{OUT_AXIS}`",
        f"- Criteria matrix: `{OUT_CRITERIA}`",
        f"- Missing experiment matrix: `{OUT_MISSING}`",
        f"- Input manifest: `{OUT_INPUTS}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
