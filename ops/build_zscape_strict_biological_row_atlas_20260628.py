#!/usr/bin/env python3
"""Build a row-level ZSCAPE biological evidence atlas.

CPU/report-only integration of already completed audits. It is designed to
separate exploratory signals, strict controls, trajectory diagnostics, HVG
information concentration, and fixed-cell robustness status.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/zscape_strict_biological_row_atlas_20260628"
EXPLORATORY_ROWS = ROOT / "runs/zscape_expression_ot_continuity_gate_20260628/zscape_expression_ot_continuity_gate_20260628_081745/outputs/zscape_expression_ot_row_results.csv"
STRICT_ROWS = ROOT / "runs/zscape_expression_ot_strict_controls_gate_20260628/zscape_expression_ot_strict_controls_gate_20260628_082748/outputs/zscape_expression_ot_strict_primary_rows.csv"
TRAJ_ROWS = ROOT / "runs/zscape_expression_trajectory_time_gate_20260628/zscape_expression_trajectory_time_gate_20260628_084025/outputs/zscape_expression_trajectory_time_perturb_alignment.csv"
BIOINFO_ROWS = ROOT / "reports/zscape_biological_information_axis_20260628/zscape_bioinformation_row_metrics.csv"
HVG_ROWS = ROOT / "reports/zscape_hvg_fullgene_information_axis_20260628/zscape_hvg_response_energy_rows.csv"
FIXEDCELL_ROWS = ROOT / "runs/zscape_bioinformation_fixedcell_robustness_gate_20260628/zscape_bioinformation_fixedcell_robustness_gate_20260628_112326/outputs/zscape_bioinformation_fixedcell_row_results.csv"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def by_row_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["row_id"]: row for row in rows if row.get("row_id")}


def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes", "pass"}


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def fixed_status(row_id: str, fixed: dict[str, dict[str, str]]) -> str:
    row = fixed.get(row_id)
    if row is None:
        return "pending"
    for key in ("fixedcell_row_gate", "strict_row_gate", "row_gate", "gate"):
        if key in row:
            return "pass" if truthy(row[key]) else "fail"
    return row.get("status", "present_unknown")


def claim_guardrail(row: dict[str, Any]) -> str:
    role = row.get("audit_role", "")
    lineage = row.get("lineage", "")
    strict_gate = truthy(row.get("strict_row_gate", ""))
    fixed = row.get("fixedcell_status", "pending")
    if role == "primary_mechanism_test" and lineage == "periderm":
        if strict_gate and fixed == "pass":
            return "periderm_strict_fixedcell_candidate_needs_placebo"
        if strict_gate and fixed == "pending":
            return "periderm_partial_pending_fixedcell"
        return "periderm_not_strict_positive"
    if role == "primary_mechanism_test" and lineage == "mature fast muscle":
        return "mature_fast_muscle_strict_claim_closed"
    if role == "primary_mechanism_test":
        return "primary_claim_not_open"
    return "diagnostic_or_control_only"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exploratory = by_row_id(read_csv(EXPLORATORY_ROWS))
    strict = by_row_id(read_csv(STRICT_ROWS))
    traj = by_row_id(read_csv(TRAJ_ROWS))
    bioinfo = by_row_id(read_csv(BIOINFO_ROWS))
    hvg = by_row_id(read_csv(HVG_ROWS))
    fixed = by_row_id(read_csv(FIXEDCELL_ROWS))

    row_ids = sorted(set(exploratory) | set(strict) | set(traj) | set(bioinfo) | set(hvg))
    rows: list[dict[str, Any]] = []
    for row_id in row_ids:
        base = exploratory.get(row_id) or strict.get(row_id) or bioinfo.get(row_id) or hvg.get(row_id) or {}
        srow = strict.get(row_id, {})
        trow = traj.get(row_id, {})
        brow = bioinfo.get(row_id, {})
        hrow = hvg.get(row_id, {})
        row: dict[str, Any] = {
            "row_id": row_id,
            "audit_role": base.get("audit_role", srow.get("audit_role", brow.get("audit_role", ""))),
            "lineage": base.get("cell_type_broad", srow.get("cell_type_broad", brow.get("lineage", hrow.get("lineage", "")))),
            "target": base.get("gene_target", srow.get("gene_target", brow.get("target", hrow.get("target", "")))),
            "timepoint": base.get("timepoint", srow.get("timepoint", brow.get("timepoint", hrow.get("timepoint", "")))),
            "exploratory_ot_gate": base.get("row_expression_ot_gate", ""),
            "exploratory_effect_ratio": brow.get("effect_ratio_vs_max_null_p95", ""),
            "exploratory_subtype_jsd": base.get("subtype_jsd", brow.get("exploratory_subtype_jsd", "")),
            "strict_row_gate": srow.get("strict_row_gate", ""),
            "strict_effect_ratio": srow.get("effect_ratio_vs_max_null_p95", ""),
            "matched_subtype_jsd": srow.get("matched_subtype_jsd", ""),
            "expression_library_smd": srow.get("expression_library_smd", ""),
            "trajectory_alignment_gate": trow.get("alignment_gate", ""),
            "trajectory_cosine_margin": trow.get("cosine_margin_vs_wrong_lineage", ""),
            "hvg1000_response_energy_share": hrow.get("hvg1000_response_energy_share", ""),
            "hvg2000_response_energy_share": hrow.get("hvg2000_response_energy_share", ""),
            "hvg4000_response_energy_share": hrow.get("hvg4000_response_energy_share", ""),
            "fixedcell_status": fixed_status(row_id, fixed),
        }
        row["claim_guardrail"] = claim_guardrail(row)
        rows.append(row)

    fields = [
        "row_id",
        "audit_role",
        "lineage",
        "target",
        "timepoint",
        "exploratory_ot_gate",
        "exploratory_effect_ratio",
        "exploratory_subtype_jsd",
        "strict_row_gate",
        "strict_effect_ratio",
        "matched_subtype_jsd",
        "expression_library_smd",
        "trajectory_alignment_gate",
        "trajectory_cosine_margin",
        "hvg1000_response_energy_share",
        "hvg2000_response_energy_share",
        "hvg4000_response_energy_share",
        "fixedcell_status",
        "claim_guardrail",
    ]
    csv_path = OUT_DIR / "zscape_strict_biological_row_atlas.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["claim_guardrail"]] = counts.get(row["claim_guardrail"], 0) + 1

    json_path = OUT_DIR / "zscape_strict_biological_row_atlas_20260628.json"
    json_path.write_text(
        json.dumps(
            {
                "status": "zscape_strict_biological_row_atlas_ready_no_gpu",
                "gpu_authorized": False,
                "n_rows": len(rows),
                "claim_guardrail_counts": counts,
                "fixedcell_rows_available": bool(fixed),
                "atlas_csv": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    primary = [r for r in rows if r["audit_role"] == "primary_mechanism_test"]
    md_path = OUT_DIR / "LATENTFM_ZSCAPE_STRICT_BIOLOGICAL_ROW_ATLAS_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Strict Biological Row Atlas",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        "Status: `zscape_strict_biological_row_atlas_ready_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only merge of completed ZSCAPE audits and the HVG/full-gene preflight.",
        "- Does not train, infer, read scFM embeddings, read canonical multi, or read Track C query.",
        "- The atlas is a claim-control artifact, not a model-selection artifact.",
        "",
        "## Claim Guardrail Counts",
        "",
        "| guardrail | rows |",
        "|---|---:|",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"| `{key}` | {value} |")
    lines.extend(
        [
            "",
            "## Primary Rows",
            "",
            "| lineage | target | time | exploratory ratio | strict gate | strict ratio | trajectory | HVG2k response | fixed-cell | guardrail |",
            "|---|---|---:|---:|---|---:|---|---:|---|---|",
        ]
    )
    for row in primary:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["lineage"],
                    row["target"],
                    row["timepoint"],
                    f"{fnum(row, 'exploratory_effect_ratio'):.3f}",
                    str(row["strict_row_gate"]),
                    f"{fnum(row, 'strict_effect_ratio'):.3f}",
                    str(row["trajectory_alignment_gate"]),
                    f"{fnum(row, 'hvg2000_response_energy_share'):.3f}",
                    row["fixedcell_status"],
                    f"`{row['claim_guardrail']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Mature fast muscle remains closed as a strict positive claim even where exploratory response/HVG signal is strong.",
            "- Periderm remains only a partial, fixed-cell-pending biological-information branch.",
            "- HVG concentration is evidence about candidate information axes, not a proof that LatentFM should use a particular gene budget.",
            "- Fixed-cell `pending` means no placebo or model-enabling conclusion can be drawn yet.",
            "",
            "## Outputs",
            "",
            f"- atlas CSV: `{csv_path}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
