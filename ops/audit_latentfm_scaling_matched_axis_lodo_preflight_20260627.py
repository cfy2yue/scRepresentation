#!/usr/bin/env python3
"""Scaling matched-axis LODO preflight for LatentFM.

CPU/report-only synthesis of completed scaling evidence. This does not rerun
training or promote any scaling claim; it turns the current evidence table and
axis-criteria matrix into a strict pass/fail gate for Nature Methods-level
scaling readiness and mainline usefulness.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
EVIDENCE = REPORTS / "latentfm_scaling_evidence_table_20260625.csv"
CRITERIA = REPORTS / "scaling_law_ready_evidence_table_20260626/axis_criteria_matrix.csv"
UNIFIED = REPORTS / "latentfm_scaling_unified_matched_axis_lodo_gate_20260626.csv"
OUT_DIR = REPORTS / "scaling_matched_axis_lodo_preflight_20260627"
OUT_ROWS = OUT_DIR / "scaling_matched_axis_lodo_preflight_rows.csv"
OUT_JSON = REPORTS / "latentfm_scaling_matched_axis_lodo_preflight_20260627.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_MATCHED_AXIS_LODO_PREFLIGHT_20260627.md"

REQUIRED_CRITERIA = {
    "pre_registered_estimand",
    "matched_or_lodo_control",
    "condition_or_row_bootstrap_ci",
    "dataset_bootstrap_or_lodo",
    "shuffle_or_negative_control",
    "dataset_tail_safety",
    "mmd_or_noharm_safety",
    "artifact_provenance",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def bad_status(status: str) -> bool:
    text = status.lower()
    return any(token in text for token in ["missing", "failed", "incomplete", "not_applicable_or_missing", "present_but_failed", "present_but_not_law_sufficient"])


def main() -> None:
    evidence_rows = read_csv(EVIDENCE)
    criteria_rows = read_csv(CRITERIA)
    unified_rows = read_csv(UNIFIED)
    by_axis_criteria: dict[str, dict[str, str]] = {}
    crit_evidence: dict[tuple[str, str], str] = {}
    for row in criteria_rows:
        axis = row.get("axis", "")
        criterion = row.get("criterion", "")
        status = row.get("status", "")
        by_axis_criteria.setdefault(axis, {})[criterion] = status
        crit_evidence[(axis, criterion)] = row.get("evidence", "")
    by_axis_unified = {row.get("axis", ""): row for row in unified_rows}

    rows: list[dict[str, Any]] = []
    axes = sorted({row.get("axis", "") for row in evidence_rows} | set(by_axis_criteria) | set(by_axis_unified))
    for axis in axes:
        if not axis:
            continue
        crits = by_axis_criteria.get(axis, {})
        missing_or_failed = sorted(
            criterion
            for criterion in REQUIRED_CRITERIA
            if criterion not in crits or bad_status(crits.get(criterion, "missing"))
        )
        noharm = crits.get("frozen_canonical_noharm_when_training_relevant", "")
        unified = by_axis_unified.get(axis, {})
        gate_status = unified.get("gate_status", "")
        tail_noharm = unified.get("tail_or_noharm", unified.get("tail/no-harm blocker", ""))
        evidence_text = unified.get("evidence", "")
        immediate_gpu = "pass" in gate_status.lower() and not missing_or_failed and "failed" not in tail_noharm.lower()
        row = {
            "axis": axis,
            "gate_status": gate_status,
            "evidence": evidence_text,
            "matched_or_lodo_control": unified.get("matched_or_lodo_control", ""),
            "tail_noharm": tail_noharm,
            "required_failed_or_missing_count": len(missing_or_failed),
            "required_failed_or_missing": ";".join(missing_or_failed),
            "frozen_canonical_noharm_status": noharm,
            "scaling_claim_ready": False,
            "mechanism_or_failure_map_ready": bool(evidence_text or gate_status),
            "immediate_gpu_authorized": immediate_gpu,
        }
        # The current project policy requires no GPU from this preflight alone.
        row["immediate_gpu_authorized"] = False
        rows.append(row)

    law_ready = [row for row in rows if row["scaling_claim_ready"]]
    mechanism_axes = [row for row in rows if row["mechanism_or_failure_map_ready"]]
    best_axis = None
    for candidate in ("true_cell_per_condition_support", "condition_exposure_count", "background_source_type_breadth"):
        match = next((row for row in rows if row["axis"] == candidate), None)
        if match:
            best_axis = match
            break
    status = "scaling_matched_axis_lodo_preflight_fail_no_gpu"
    if law_ready:
        status = "scaling_matched_axis_lodo_preflight_pass_review_only_no_gpu"

    write_csv(
        OUT_ROWS,
        rows,
        [
            "axis",
            "gate_status",
            "evidence",
            "matched_or_lodo_control",
            "tail_noharm",
            "required_failed_or_missing_count",
            "required_failed_or_missing",
            "frozen_canonical_noharm_status",
            "scaling_claim_ready",
            "mechanism_or_failure_map_ready",
            "immediate_gpu_authorized",
        ],
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "gpu_authorized": False,
        "inputs": [str(EVIDENCE), str(CRITERIA), str(UNIFIED)],
        "axes": rows,
        "law_ready_axes": [row["axis"] for row in law_ready],
        "mechanism_or_failure_map_axes": [row["axis"] for row in mechanism_axes],
        "best_current_axis": best_axis,
        "outputs": {"rows": str(OUT_ROWS), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    for row in rows:
        lines.append(
            "| {axis} | {gate_status} | {failed} | `{noharm}` | {ready} |".format(
                axis=row["axis"],
                gate_status=row["gate_status"],
                failed=row["required_failed_or_missing_count"],
                noharm=row["frozen_canonical_noharm_status"],
                ready=row["scaling_claim_ready"],
            )
        )
    best_axis_text = "None"
    if best_axis:
        best_axis_text = f"`{best_axis['axis']}` with gate `{best_axis['gate_status']}` and blockers `{best_axis['required_failed_or_missing']}`"
    md = f"""# LatentFM Scaling Matched-Axis LODO Preflight 2026-06-27

Timestamp: `{payload['timestamp']}`

Status: `{status}`

GPU authorized: `False`

## Boundary

- CPU/report-only synthesis of completed scaling evidence.
- No training, inference, checkpoint selection, canonical multi selection,
  Track C query, or GPU.
- This preflight can identify mechanism/failure-map axes, but cannot alone
  authorize GPU.

## Axis Readiness

| axis | gate status | required failed/missing | frozen canonical no-harm | scaling claim ready |
|---|---|---:|---|---:|
{chr(10).join(lines)}

## Decision

- law-ready axes: `{[row['axis'] for row in law_ready]}`
- mechanism/failure-map-ready axes: `{[row['axis'] for row in mechanism_axes]}`
- best current axis for narrative/mainline inspiration: {best_axis_text}

The scaling branch remains scientifically useful as a mechanism/failure map.
It is not yet a Nature Methods-level scaling law claim or a GPU launcher because
matched controls, bootstrap/LODO, tail safety, and frozen canonical no-harm do
not all pass on any axis.

## Outputs

- JSON: `{OUT_JSON}`
- rows: `{OUT_ROWS}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
