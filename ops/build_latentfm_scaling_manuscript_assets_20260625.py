#!/usr/bin/env python3
"""Build manuscript-facing LatentFM scaling/failure-map assets.

CPU-only. Converts the scaling/failure-map package and evidence table into
flat CSV/TSV tables for figures, supplements, and handoff. It does not read
checkpoints, canonical multi, held-out Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
PACKAGE_JSON = REPORTS / "latentfm_scaling_failure_map_package_20260625.json"
EVIDENCE_JSON = REPORTS / "latentfm_scaling_evidence_table_20260625.json"
OUT_AXIS_CSV = REPORTS / "latentfm_scaling_axis_claim_matrix_20260625.csv"
OUT_NEG_CSV = REPORTS / "latentfm_scaling_negative_evidence_table_20260625.csv"
OUT_FIG_TSV = REPORTS / "latentfm_scaling_figure_manifest_20260625.tsv"
OUT_JSON = REPORTS / "latentfm_scaling_manuscript_assets_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_MANUSCRIPT_ASSETS_20260625.md"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str], *, dialect: str = "excel") -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, dialect=dialect)
        writer.writeheader()
        writer.writerows(rows)


def axis_rows(package: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in package.get("axes", []):
        axis = item.get("axis", "")
        claim = item.get("claim", "")
        rows.append(
            {
                "axis": axis,
                "claim_level": claim,
                "support": item.get("support", ""),
                "boundary": item.get("boundary", ""),
                "next_gate": item.get("next_gate", ""),
                "manuscript_use": "main_text" if axis in {"true_cell_budget", "condition_count_exposure"} else "supplement_or_failure_map",
                "promotion_allowed": "false",
            }
        )
    return rows


def negative_rows(package: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in evidence.get("rows", []):
        status = str(row.get("status", "")).lower()
        scope = str(row.get("claim_scope", "")).lower()
        decision = str(row.get("decision", "")).lower()
        if any(token in status + scope + decision for token in ["fail", "negative", "veto", "closed", "no_gpu", "no-gpu"]):
            rows.append(
                {
                    "axis": row.get("axis", ""),
                    "estimand": row.get("estimand", ""),
                    "status": row.get("status", ""),
                    "primary_metric": row.get("primary_metric", ""),
                    "secondary_metric": row.get("secondary_metric", ""),
                    "ci95": row.get("ci95", ""),
                    "tail_or_control_metric": row.get("tail_metric", "") or row.get("control_signal", ""),
                    "claim_scope": row.get("claim_scope", ""),
                    "source_report": row.get("source_report", ""),
                    "decision": row.get("decision", ""),
                }
            )
    # Add explicit package-level vetoes that are not all present as rows in the
    # evidence table.
    for item in package.get("axes", []):
        boundary = str(item.get("boundary", "")).lower()
        if any(token in boundary for token in ["failed", "veto", "not", "block", "failure"]):
            rows.append(
                {
                    "axis": item.get("axis", ""),
                    "estimand": "package_axis_boundary",
                    "status": "package_boundary",
                    "primary_metric": item.get("support", ""),
                    "secondary_metric": "",
                    "ci95": "",
                    "tail_or_control_metric": item.get("boundary", ""),
                    "claim_scope": "boundary_or_negative_evidence",
                    "source_report": str(PACKAGE_JSON),
                    "decision": item.get("next_gate", ""),
                }
            )
    return rows


def figure_rows(package: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = package.get("figure_candidates", [])
    mapping = [
        {
            "figure_id": "FigS_scaling_S0_provenance",
            "title": "S0 provenance coverage",
            "source": "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv",
            "panel_type": "bar/table",
            "claim_boundary": "design/provenance only",
        },
        {
            "figure_id": "Fig_scaling_truecell_budget",
            "title": "True-cell budget mechanism signal",
            "source": "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_TAIL_STABILITY_6K_DECISION_20260625.md",
            "panel_type": "curve/point-range",
            "claim_boundary": "mechanism only; no deployable promotion",
        },
        {
            "figure_id": "Fig_scaling_exposure_nonmonotonic",
            "title": "Condition exposure non-monotonicity",
            "source": "reports/LATENTFM_SCALING_NESTED_CONDITION_EXPOSURE_V2_GATE_20260625.md",
            "panel_type": "line/failure annotation",
            "claim_boundary": "moderate exposure local signal only",
        },
        {
            "figure_id": "Fig_scaling_noharm_veto",
            "title": "Frozen canonical no-harm veto",
            "source": "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md",
            "panel_type": "seed table",
            "claim_boundary": "veto context; no canonical multi selection",
        },
        {
            "figure_id": "FigS_scaling_failure_map",
            "title": "Scaling failure-map heat table",
            "source": "reports/LATENTFM_SCALING_FAILURE_MAP_PACKAGE_20260625.md",
            "panel_type": "heat table",
            "claim_boundary": "negative evidence and future-gate map",
        },
    ]
    for idx, item in enumerate(mapping):
        item["package_candidate_text"] = candidates[idx] if idx < len(candidates) else ""
        item["status"] = "ready_from_existing_reports"
    return mapping


def main() -> int:
    package = load_json(PACKAGE_JSON)
    evidence = load_json(EVIDENCE_JSON)

    axes = axis_rows(package)
    negatives = negative_rows(package, evidence)
    figures = figure_rows(package)

    write_csv(
        OUT_AXIS_CSV,
        axes,
        ["axis", "claim_level", "support", "boundary", "next_gate", "manuscript_use", "promotion_allowed"],
    )
    write_csv(
        OUT_NEG_CSV,
        negatives,
        [
            "axis",
            "estimand",
            "status",
            "primary_metric",
            "secondary_metric",
            "ci95",
            "tail_or_control_metric",
            "claim_scope",
            "source_report",
            "decision",
        ],
    )
    write_csv(
        OUT_FIG_TSV,
        figures,
        ["figure_id", "title", "source", "panel_type", "claim_boundary", "package_candidate_text", "status"],
        dialect="excel-tab",
    )

    payload = {
        "status": "scaling_manuscript_assets_ready_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "outputs": {
            "axis_claim_matrix_csv": str(OUT_AXIS_CSV),
            "negative_evidence_csv": str(OUT_NEG_CSV),
            "figure_manifest_tsv": str(OUT_FIG_TSV),
            "json": str(OUT_JSON),
            "md": str(OUT_MD),
        },
        "counts": {
            "axis_rows": len(axes),
            "negative_rows": len(negatives),
            "figure_rows": len(figures),
        },
        "claim_boundary": package.get("claim_boundary", {}),
        "default_model": package.get("default_model"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Scaling Manuscript Assets",
        "",
        "Status: `scaling_manuscript_assets_ready_no_gpu`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only table/manifest generation from completed reports.",
        "- Does not read checkpoints, canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Outputs",
        "",
        f"- Axis claim matrix: `{OUT_AXIS_CSV}`",
        f"- Negative evidence table: `{OUT_NEG_CSV}`",
        f"- Figure manifest: `{OUT_FIG_TSV}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Counts",
        "",
        f"- axis rows: `{len(axes)}`",
        f"- negative evidence rows: `{len(negatives)}`",
        f"- figure rows: `{len(figures)}`",
        "",
        "## Claim Boundary",
        "",
        f"- allowed: {payload['claim_boundary'].get('allowed')}",
        f"- not allowed: {payload['claim_boundary'].get('not_allowed')}",
        f"- default model: `{payload['default_model']}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
