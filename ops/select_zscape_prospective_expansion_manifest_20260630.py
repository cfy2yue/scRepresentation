#!/usr/bin/env python3
"""Select a capped prospective ZSCAPE expansion manifest from metadata-ready rows."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
DEFAULT_CANDIDATES = (
    REPORTS
    / "zscape_prospective_manifest_expansion_gate_20260630"
    / "zscape_prospective_manifest_expansion_candidates_20260630.csv"
)
OUT_DIR = REPORTS / "zscape_prospective_expansion_manifest_20260630"
CONNECTIVE_LINEAGE = "connective tissue-meninges-dermal FB"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def choose_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.candidates)
    required = {"lineage", "target", "timepoint", "prospective_candidate", "strong_metadata_ready"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Candidate file missing columns: {missing}")
    df = df[df["prospective_candidate"].map(truthy)].copy()
    if args.strong_only:
        df = df[df["strong_metadata_ready"].map(truthy)].copy()
    df = df[df["lineage"].astype(str) != CONNECTIVE_LINEAGE].copy()
    if args.exclude_periderm:
        df = df[df["lineage"].astype(str) != "periderm"].copy()
    for col in [
        "control_family_count",
        "perturb_cells",
        "perturb_embryos",
        "same_lineage_time_control_cells",
        "same_lineage_time_control_embryos",
        "perturb_subtypes",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(
        [
            "strong_metadata_ready",
            "control_family_count",
            "perturb_embryos",
            "perturb_cells",
            "same_lineage_time_control_cells",
            "row_id",
        ],
        ascending=[False, False, False, False, False, True],
    )
    selected: list[pd.Series] = []
    per_lineage: dict[str, int] = {}
    lineage_target_seen: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        lineage = str(row["lineage"])
        target = str(row["target"])
        if per_lineage.get(lineage, 0) >= args.max_per_lineage:
            continue
        if (lineage, target) in lineage_target_seen:
            continue
        selected.append(row)
        per_lineage[lineage] = per_lineage.get(lineage, 0) + 1
        lineage_target_seen.add((lineage, target))
        if len(selected) >= args.max_rows:
            break
    out = pd.DataFrame(selected)
    if out.empty:
        return out
    out = out.copy()
    out["cell_type_broad"] = out["lineage"]
    out["gene_target"] = out["target"]
    out["selection_rank"] = range(len(out))
    out["selection_reason"] = (
        "prospective_second_lineage_strong_metadata_ready_capped_by_lineage_and_target"
    )
    return out


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None._"
    cols = [
        "selection_rank",
        "row_id",
        "lineage",
        "target",
        "timepoint",
        "perturb_cells",
        "perturb_embryos",
        "same_lineage_time_control_cells",
        "control_family_count",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row.get(col)
            vals.append(fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(selected: pd.DataFrame, args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "zscape_prospective_expansion_manifest_20260630.csv"
    rows_path = args.out_dir / "zscape_prospective_expansion_selected_rows_20260630.csv"
    json_path = args.out_dir / "zscape_prospective_expansion_manifest_20260630.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_PROSPECTIVE_EXPANSION_MANIFEST_20260630.md"
    manifest_cols = ["cell_type_broad", "gene_target", "timepoint"]
    selected[manifest_cols].to_csv(manifest_path, index=False)
    selected.to_csv(rows_path, index=False)
    decision = {
        "status": "zscape_prospective_expansion_manifest_ready_cpu_only"
        if len(selected) >= args.min_rows
        else "zscape_prospective_expansion_manifest_underpowered_no_launch",
        "gpu_authorized_next": False,
        "selected_rows": int(len(selected)),
        "selected_lineages": int(selected["lineage"].nunique()) if not selected.empty else 0,
        "selected_targets": int(selected["target"].nunique()) if not selected.empty else 0,
        "max_rows": args.max_rows,
        "max_per_lineage": args.max_per_lineage,
        "exclude_periderm": args.exclude_periderm,
        "strong_only": args.strong_only,
        "next_action": (
            "build a metadata cell manifest, extract the capped raw-count submatrix, "
            "and run CPU-only OT pairability atlas"
            if len(selected) >= args.min_rows
            else "do not launch expanded expression extraction"
        ),
    }
    payload = {
        "timestamp_cst": now_cst(),
        "boundary": {
            "reads_metadata_gate_only": True,
            "reads_expression_matrix": False,
            "new_ot_pairing": False,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {"candidates": str(args.candidates)},
        "outputs": {
            "manifest": str(manifest_path),
            "selected_rows": str(rows_path),
            "markdown_report": str(md_path),
        },
        "decision": decision,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = f"""# ZSCAPE Prospective Expansion Manifest

## Boundary

- CPU/report-only selection from metadata-ready prospective candidate rows.
- Does not read expression matrices, compute OT pairs, train, infer, select checkpoints, use canonical multi, access Track C query, or use GPU.

## Decision

- Status: `{decision['status']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- Selected rows: `{decision['selected_rows']}`
- Selected lineages: `{decision['selected_lineages']}`
- Selected targets: `{decision['selected_targets']}`
- Max rows: `{args.max_rows}`
- Max per lineage: `{args.max_per_lineage}`
- Exclude periderm: `{args.exclude_periderm}`
- Strong only: `{args.strong_only}`
- Next action: {decision['next_action']}

## Selection Rule

Rows must be prospective metadata-ready rows from the full ZSCAPE metadata gate,
new to the current atlas, outside the already passing connective-tissue lineage,
and capped by lineage plus lineage-target to avoid one background dominating
the second-lineage search. This is a discovery manifest only; all OT and
strict-control scoring occurs later under frozen rules.

## Selected Rows

{md_table(selected)}

## Outputs

- Manifest: `{manifest_path}`
- Selected rows: `{rows_path}`
- JSON: `{json_path}`
"""
    md_path.write_text(text, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=48)
    parser.add_argument("--min-rows", type=int, default=24)
    parser.add_argument("--max-per-lineage", type=int, default=4)
    parser.add_argument("--strong-only", action="store_true", default=True)
    parser.add_argument("--include-weak", action="store_false", dest="strong_only")
    parser.add_argument("--exclude-periderm", action="store_true")
    args = parser.parse_args()
    selected = choose_rows(args)
    write_outputs(selected, args)


if __name__ == "__main__":
    main()
