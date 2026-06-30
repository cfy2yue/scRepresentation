#!/usr/bin/env python3
"""CPU-only feasibility table for ZSCAPE-inspired flow-matching constraints."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/zscape_flow_constraint_feasibility_20260628"
ROW_ATLAS = ROOT / "reports/zscape_strict_biological_row_atlas_20260628/zscape_strict_biological_row_atlas.csv"
STRICT_DIAG = ROOT / "runs/zscape_expression_ot_strict_controls_gate_20260628/zscape_expression_ot_strict_controls_gate_20260628_082748/outputs/zscape_expression_ot_strict_diagnostics.csv"
EXPR_LATENT = ROOT / "reports/zscape_expression_latent_biology_preflight_20260628/zscape_latent_alignment_rows.csv"
EXPR_DE_SUMMARY = ROOT / "reports/zscape_expression_latent_biology_preflight_20260628/zscape_expression_de_row_summary.csv"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes", "pass"}


def fnum(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def by_row_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["row_id"]: row for row in rows if row.get("row_id")}


def strict_diag_features(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        row_id = row.get("row_id", "")
        diag = row.get("diagnostic", "")
        if not row_id or not diag:
            continue
        out[row_id][diag] = fnum(row, "ot")
    return out


def latent_features(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("row_id", "")
        space = row.get("latent_space", "")
        if not row_id or not space:
            continue
        prefix = "svd" if space == "log1p_hvg_svd" else "umap3d"
        obj = out.setdefault(row_id, {})
        obj[f"{prefix}_tangent_cosine"] = fnum(row, "cosine_to_lineage_time_vector")
        obj[f"{prefix}_wrong_lineage_max"] = fnum(row, "max_cosine_to_wrong_lineage_time_vector")
        obj[f"{prefix}_tangent_margin"] = fnum(row, "cosine_margin_vs_wrong_lineage")
        obj[f"{prefix}_alignment_gate"] = truthy(row.get("alignment_gate", ""))
    return out


def claim_class(row: dict[str, Any]) -> str:
    lineage = row.get("lineage", "")
    strict = truthy(row.get("strict_row_gate", ""))
    traj = truthy(row.get("trajectory_alignment_gate", ""))
    fixed = row.get("fixedcell_status", "pending")
    if lineage == "periderm" and strict and traj and fixed == "pass":
        return "candidate_constraint_after_placebo"
    if lineage == "periderm" and strict and traj:
        return "best_candidate_pending_fixedcell_placebo"
    if lineage == "periderm" and strict:
        return "strict_only_no_tangent_candidate"
    if lineage == "mature fast muscle":
        return "negative_control_confounded_exploratory"
    if strict:
        return "strict_nonperiderm_diagnostic"
    return "diagnostic_or_unsupported"


def recommended_constraints(row: dict[str, Any]) -> str:
    klass = claim_class(row)
    if klass == "candidate_constraint_after_placebo":
        return "temporal_tangent;wrong_lineage_contrast;state_preservation;sparse_reliability_weight"
    if klass == "best_candidate_pending_fixedcell_placebo":
        return "pending_temporal_tangent;pending_wrong_lineage_contrast;state_preservation_diagnostic"
    if klass == "strict_only_no_tangent_candidate":
        return "state_preservation_diagnostic;no_temporal_tangent"
    if klass == "negative_control_confounded_exploratory":
        return "negative_control_only;do_not_train_positive_weight"
    return "diagnostic_only"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas = read_csv(ROW_ATLAS)
    de = by_row_id(read_csv(EXPR_DE_SUMMARY))
    diag = strict_diag_features(read_csv(STRICT_DIAG))
    latent = latent_features(read_csv(EXPR_LATENT))
    rows: list[dict[str, Any]] = []
    for row in atlas:
        row_id = row["row_id"]
        drow = de.get(row_id, {})
        diag_row = diag.get(row_id, {})
        lrow = latent.get(row_id, {})
        strict_ot = fnum(row, "strict_effect_ratio")
        wrong_time = diag_row.get("wrong_time_control", "")
        wrong_lineage = diag_row.get("wrong_lineage_control", "")
        out: dict[str, Any] = {
            "row_id": row_id,
            "audit_role": row.get("audit_role", ""),
            "lineage": row.get("lineage", ""),
            "target": row.get("target", ""),
            "timepoint": row.get("timepoint", ""),
            "strict_row_gate": row.get("strict_row_gate", ""),
            "trajectory_alignment_gate": row.get("trajectory_alignment_gate", ""),
            "formal_tangent_margin": row.get("trajectory_cosine_margin", ""),
            "hvg2000_response_energy_share": row.get("hvg2000_response_energy_share", ""),
            "response_l2": drow.get("response_energy_l2", ""),
            "top_up_genes": drow.get("top_up_genes", ""),
            "top_down_genes": drow.get("top_down_genes", ""),
            "fixedcell_status": row.get("fixedcell_status", "pending"),
            "wrong_time_control_ot": wrong_time,
            "wrong_lineage_control_ot": wrong_lineage,
            "strict_effect_ratio": row.get("strict_effect_ratio", ""),
            "svd_tangent_margin": lrow.get("svd_tangent_margin", ""),
            "svd_alignment_gate": lrow.get("svd_alignment_gate", ""),
            "umap3d_tangent_margin": lrow.get("umap3d_tangent_margin", ""),
            "umap3d_alignment_gate": lrow.get("umap3d_alignment_gate", ""),
        }
        if wrong_time != "" and strict_ot > 0:
            out["wrong_time_minus_strict_ratio"] = float(wrong_time) / strict_ot
        else:
            out["wrong_time_minus_strict_ratio"] = ""
        if wrong_lineage != "" and strict_ot > 0:
            out["wrong_lineage_minus_strict_ratio"] = float(wrong_lineage) / strict_ot
        else:
            out["wrong_lineage_minus_strict_ratio"] = ""
        out["constraint_feasibility_class"] = claim_class(out)
        out["recommended_constraint_use"] = recommended_constraints(out)
        rows.append(out)

    fields = [
        "row_id",
        "audit_role",
        "lineage",
        "target",
        "timepoint",
        "strict_row_gate",
        "trajectory_alignment_gate",
        "formal_tangent_margin",
        "hvg2000_response_energy_share",
        "response_l2",
        "top_up_genes",
        "top_down_genes",
        "fixedcell_status",
        "wrong_time_control_ot",
        "wrong_lineage_control_ot",
        "strict_effect_ratio",
        "wrong_time_minus_strict_ratio",
        "wrong_lineage_minus_strict_ratio",
        "svd_tangent_margin",
        "svd_alignment_gate",
        "umap3d_tangent_margin",
        "umap3d_alignment_gate",
        "constraint_feasibility_class",
        "recommended_constraint_use",
    ]
    csv_path = OUT_DIR / "zscape_flow_constraint_feasibility_rows.csv"
    write_csv(csv_path, rows, fields)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["constraint_feasibility_class"])
        counts[key] = counts.get(key, 0) + 1
    json_path = OUT_DIR / "zscape_flow_constraint_feasibility_20260628.json"
    status = "zscape_flow_constraint_feasibility_ready_no_gpu"
    json_path.write_text(
        json.dumps(
            {
                "status": status,
                "gpu_authorized": False,
                "class_counts": counts,
                "rows_csv": str(csv_path),
                "hard_gate": [
                    "fixed-cell periderm pass",
                    "periderm wrong-target/wrong-time placebo pass",
                    "frozen design review",
                    "dual-baseline/no-harm CPU gate before GPU",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    md_path = OUT_DIR / "LATENTFM_ZSCAPE_FLOW_CONSTRAINT_FEASIBILITY_20260628.md"
    primary = [r for r in rows if r["audit_role"] == "primary_mechanism_test"]
    lines = [
        "# LatentFM ZSCAPE Flow-Constraint Feasibility",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only integration of strict controls, trajectory alignment, HVG/full-gene, expression DE, and latent-proxy diagnostics.",
        "- Does not train, infer, extract scFM embeddings, read canonical multi, or read Track C query.",
        "- This report can nominate constraints for design review only; it cannot authorize GPU.",
        "",
        "## Class Counts",
        "",
        "| class | rows |",
        "|---|---:|",
    ]
    for key, val in sorted(counts.items()):
        lines.append(f"| `{key}` | {val} |")
    lines.extend(
        [
            "",
            "## Primary Rows",
            "",
            "| lineage | target | time | strict | trajectory | HVG2k | response L2 | fixed-cell | class | recommended use |",
            "|---|---|---:|---|---|---:|---:|---|---|---|",
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
                    str(row["strict_row_gate"]),
                    str(row["trajectory_alignment_gate"]),
                    f"{fnum(row, 'hvg2000_response_energy_share'):.3f}",
                    f"{fnum(row, 'response_l2'):.3f}",
                    row["fixedcell_status"],
                    f"`{row['constraint_feasibility_class']}`",
                    row["recommended_constraint_use"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Best current model-constraint candidates are strict-positive and trajectory-aligned periderm rows, pending fixed-cell and placebo gates.",
            "- Mature fast muscle is explicitly retained as a negative/confounded control, not a positive training signal.",
            "- SVD/UMAP latent proxies are diagnostics only; scFM latent-space claims require a separate embedding extraction protocol.",
            "- OT topology and wrong-lineage/time contrasts may define information axes or constraints, but old OT minibatch pairmode remains closed as a generic training route.",
            "",
            "## Fail-Close Rules",
            "",
            "- Fixed-cell fail: close ZSCAPE as model-enabling.",
            "- Placebo fail: close ZSCAPE as model-enabling.",
            "- Fewer than two supported periderm rows after placebo: no temporal tangent GPU route.",
            "- Any route lacking dual-baseline/no-harm CPU gate: no GPU.",
            "",
            "## Outputs",
            "",
            f"- rows: `{csv_path}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
