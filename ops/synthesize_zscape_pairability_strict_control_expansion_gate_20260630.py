#!/usr/bin/env python3
"""Feasibility gate for extending strict controls to high-pairability ZSCAPE rows."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "zscape_pairability_strict_control_expansion_gate_20260630"

INPUTS = {
    "atlas_rows": REPORTS
    / "zscape_dynamic_pairability_atlas_20260630"
    / "zscape_dynamic_pairability_atlas_rows_20260630.csv",
    "subset_manifest": REPORTS
    / "zscape_expression_subset_ot_manifest_20260628"
    / "zscape_expression_subset_ot_manifest.csv",
    "structural_gate": REPORTS
    / "zscape_structural_dynamic_scaling_x_20260630"
    / "zscape_structural_dynamic_scaling_x_20260630.json",
}


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def row_id(lineage: str, target: str, timepoint: Any) -> str:
    def clean(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

    t = finite_float(timepoint)
    time = "na" if t is None else f"{t:.1f}".replace(".", "p") + "h"
    return f"{clean(lineage)}__{clean(target)}__{time}"


def control_ids(df: pd.DataFrame, row: pd.Series, mask: pd.Series) -> list[str]:
    cols = ["cell_type_broad", "gene_target", "timepoint"]
    out = df.loc[mask, cols].drop_duplicates().copy()
    out["row_id"] = [row_id(a, b, c) for a, b, c in out[cols].itertuples(index=False, name=None)]
    return sorted(out["row_id"].astype(str).tolist())


def build_rows(atlas: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    work = atlas.copy()
    manifest = manifest.copy()
    manifest["row_id"] = [
        row_id(a, b, c)
        for a, b, c in manifest[["cell_type_broad", "gene_target", "timepoint"]].itertuples(
            index=False, name=None
        )
    ]
    manifest["timepoint"] = pd.to_numeric(manifest["timepoint"], errors="coerce")
    work["timepoint"] = pd.to_numeric(work["timepoint"], errors="coerce")
    work["within_state_pairability_score"] = pd.to_numeric(
        work["within_state_pairability_score"], errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        lineage = str(row.get("lineage"))
        target = str(row.get("target"))
        time = finite_float(row.get("timepoint"))
        same_lineage = manifest["cell_type_broad"].astype(str).eq(lineage)
        same_target = manifest["gene_target"].astype(str).eq(target)
        same_time = manifest["timepoint"].eq(time)
        wrong_time = same_lineage & same_target & ~same_time
        wrong_target = same_lineage & same_time & ~same_target
        wrong_lineage_same_time = ~same_lineage & same_time
        wrong_lineage_same_target = ~same_lineage & same_target
        families = {
            "wrong_time": control_ids(manifest, row, wrong_time),
            "wrong_target": control_ids(manifest, row, wrong_target),
            "wrong_lineage_same_time": control_ids(manifest, row, wrong_lineage_same_time),
            "wrong_lineage_same_target": control_ids(manifest, row, wrong_lineage_same_target),
        }
        n_families = sum(1 for ids in families.values() if ids)
        atlas_only = str(row.get("pairability_class")) == "atlas_row_no_strict_context_yet"
        high_pairability = (finite_float(row.get("within_state_pairability_score")) or -999) >= 0.70
        ready = atlas_only and high_pairability and n_families >= 2
        rows.append(
            {
                "row_id": row.get("row_id"),
                "lineage": lineage,
                "target": target,
                "timepoint": time,
                "pairability_class": row.get("pairability_class"),
                "within_state_pairability_score": row.get("within_state_pairability_score"),
                "centroid_response_norm": row.get("centroid_response_norm"),
                "atlas_only": atlas_only,
                "high_pairability": high_pairability,
                "control_family_count": n_families,
                "wrong_time_n": len(families["wrong_time"]),
                "wrong_target_n": len(families["wrong_target"]),
                "wrong_lineage_same_time_n": len(families["wrong_lineage_same_time"]),
                "wrong_lineage_same_target_n": len(families["wrong_lineage_same_target"]),
                "wrong_time_rows": ";".join(families["wrong_time"]),
                "wrong_target_rows": ";".join(families["wrong_target"]),
                "wrong_lineage_same_time_rows": ";".join(families["wrong_lineage_same_time"]),
                "wrong_lineage_same_target_rows": ";".join(families["wrong_lineage_same_target"]),
                "strict_expansion_candidate": ready,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strict_expansion_candidate", "within_state_pairability_score"],
        ascending=[False, False],
    )


def decide(rows: pd.DataFrame) -> dict[str, Any]:
    cand = rows[rows["strict_expansion_candidate"]].copy()
    lineage_n = int(cand["lineage"].nunique()) if not cand.empty else 0
    target_n = int(cand["target"].nunique()) if not cand.empty else 0
    if len(cand) >= 6 and lineage_n >= 2:
        status = "zscape_pairability_strict_control_expansion_ready_cpu_only"
        next_action = (
            "launch a CPU-only strict-control expansion for the highest-pairability "
            "atlas-only rows; no GPU or model loss"
        )
    elif len(cand) > 0:
        status = "zscape_pairability_strict_control_expansion_partial_design_only"
        next_action = (
            "review candidates manually or expand manifest before strict-control OT; "
            "do not train"
        )
    else:
        status = "zscape_pairability_strict_control_expansion_blocked"
        next_action = "do not extend strict controls from current 25-row manifest"
    return {
        "status": status,
        "gpu_authorized_next": False,
        "candidate_rows": int(len(cand)),
        "candidate_lineages": lineage_n,
        "candidate_targets": target_n,
        "candidate_row_ids": cand["row_id"].astype(str).head(20).tolist(),
        "reasons": [
            "uses_existing_25row_manifest_only",
            "requires_high_pairability_atlas_only_rows",
            "requires_at_least_two_control_families_before_expensive_ot",
        ],
        "next_action": next_action,
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def markdown_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if df.empty:
        return "_None._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(n).iterrows():
        vals = []
        for col in cols:
            val = row.get(col)
            if isinstance(val, float):
                vals.append(fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(rows: pd.DataFrame, decision: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "zscape_pairability_strict_control_expansion_rows_20260630.csv"
    json_path = OUT_DIR / "zscape_pairability_strict_control_expansion_gate_20260630.json"
    md_path = OUT_DIR / "LATENTFM_ZSCAPE_PAIRABILITY_STRICT_CONTROL_EXPANSION_GATE_20260630.md"
    rows.to_csv(rows_path, index=False)
    payload = {
        "boundary": {
            "reads_completed_reports_only": True,
            "new_ot_pairing": False,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "outputs": {"rows": str(rows_path), "markdown_report": str(md_path)},
        "decision": decision,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    cols = [
        "row_id",
        "lineage",
        "target",
        "timepoint",
        "within_state_pairability_score",
        "control_family_count",
        "wrong_time_n",
        "wrong_target_n",
        "wrong_lineage_same_time_n",
        "wrong_lineage_same_target_n",
        "strict_expansion_candidate",
    ]
    text = f"""# ZSCAPE Pairability Strict-Control Expansion Gate

## Boundary

- CPU/report-only feasibility gate over completed atlas and manifest files.
- No new OT pairing, training, inference, GPU, checkpoint selection, canonical multi selection, or Track C query access.

## Decision

- Status: `{decision['status']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- Candidate rows: `{decision['candidate_rows']}`
- Candidate lineages: `{decision['candidate_lineages']}`
- Candidate targets: `{decision['candidate_targets']}`
- Candidate row ids: `{', '.join(decision['candidate_row_ids']) or 'none'}`

Reasons:

{chr(10).join(f'- {r}' for r in decision['reasons'])}

## Candidate Table

{markdown_table(rows, cols)}

## Interpretation

This gate only checks whether the existing 25-row ZSCAPE manifest can support a
next strict-control OT expansion. A row is a candidate only if it is atlas-only,
has high pairability, and has at least two available control families among
wrong-time, wrong-target, wrong-lineage-same-time, and wrong-lineage-same-target.

## Outputs

- Rows: `{rows_path}`
- JSON: `{json_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    for name, path in INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    atlas = pd.read_csv(INPUTS["atlas_rows"])
    manifest = pd.read_csv(INPUTS["subset_manifest"])
    rows = build_rows(atlas, manifest)
    decision = decide(rows)
    write_outputs(rows, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
