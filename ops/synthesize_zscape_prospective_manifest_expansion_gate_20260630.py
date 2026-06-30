#!/usr/bin/env python3
"""Metadata-only gate for prospective ZSCAPE manifest expansion.

This preflight asks whether the full ZSCAPE metadata contains enough new,
control-rich perturbation rows to justify a later CPU OT/strict-control atlas
expansion. It intentionally does not read expression matrices, compute OT,
train, infer, select checkpoints, use canonical multi, or access Track C query.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
DEFAULT_META = ROOT / "dataset/external/zscape_20260628/GSE202639_zperturb_full_cell_metadata.csv.gz"
DEFAULT_ATLAS_ROWS = (
    REPORTS
    / "zscape_dynamic_pairability_atlas_20260630"
    / "zscape_dynamic_pairability_atlas_rows_20260630.csv"
)
DEFAULT_STRICT_ROWS = (
    REPORTS
    / "zscape_pairability_strict_control_expansion_20260630"
    / "zscape_pairability_strict_control_expansion_rows_20260630.csv"
)
OUT_DIR = REPORTS / "zscape_prospective_manifest_expansion_gate_20260630"

CONNECTIVE_LINEAGE = "connective tissue-meninges-dermal FB"
USECOLS = [
    "gene_target",
    "timepoint",
    "cell_type_broad",
    "cell_type_sub",
    "tissue",
    "germ_layer",
    "embryo",
    "sample",
    "n.umi",
    "num_genes_expressed",
]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def clean_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def row_id(lineage: Any, target: Any, timepoint: Any) -> str:
    t = finite_float(timepoint)
    time = "na" if t is None else f"{t:.1f}".replace(".", "p") + "h"
    return f"{clean_token(lineage)}__{clean_token(target)}__{time}"


def is_control_target(target: Any) -> bool:
    return str(target).strip().lower().startswith("ctrl")


def load_existing_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c == "row_id")
        ids.update(df["row_id"].dropna().astype(str))
    return ids


def top_values(series: pd.Series, n: int = 4) -> str:
    vals = series.dropna().astype(str)
    if vals.empty:
        return ""
    return ";".join(f"{k}:{int(v)}" for k, v in vals.value_counts().head(n).items())


def summarize_conditions(meta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(meta_path, usecols=USECOLS, dtype=str, low_memory=False)
    for col in ["gene_target", "timepoint", "cell_type_broad"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[(df["gene_target"] != "") & (df["cell_type_broad"] != "") & (df["timepoint"] != "")]
    df["timepoint_num"] = pd.to_numeric(df["timepoint"], errors="coerce")
    df = df[df["timepoint_num"].notna()].copy()
    df["is_control"] = df["gene_target"].map(is_control_target)
    df["n_umi_num"] = pd.to_numeric(df["n.umi"], errors="coerce")
    df["num_genes_num"] = pd.to_numeric(df["num_genes_expressed"], errors="coerce")
    grouped = df.groupby(
        ["cell_type_broad", "gene_target", "timepoint_num", "is_control"],
        dropna=False,
        sort=False,
    )
    cond = grouped.agg(
        cells=("gene_target", "size"),
        embryos=("embryo", "nunique"),
        samples=("sample", "nunique"),
        subtypes=("cell_type_sub", "nunique"),
        tissues=("tissue", "nunique"),
        germ_layers=("germ_layer", "nunique"),
        mean_n_umi=("n_umi_num", "mean"),
        mean_num_genes=("num_genes_num", "mean"),
    ).reset_index()
    top_sub = grouped["cell_type_sub"].apply(top_values).reset_index(name="top_subtypes")
    cond = cond.merge(
        top_sub,
        on=["cell_type_broad", "gene_target", "timepoint_num", "is_control"],
        how="left",
    )
    cond["row_id"] = [
        row_id(a, b, c)
        for a, b, c in cond[["cell_type_broad", "gene_target", "timepoint_num"]].itertuples(
            index=False, name=None
        )
    ]
    return cond


def count_controls(cond: pd.DataFrame, row: pd.Series) -> dict[str, Any]:
    perturb = cond[~cond["is_control"]].copy()
    controls = cond[cond["is_control"]].copy()
    lineage = str(row["cell_type_broad"])
    target = str(row["gene_target"])
    time = float(row["timepoint_num"])

    same_lineage_time_control = controls[
        controls["cell_type_broad"].eq(lineage) & controls["timepoint_num"].eq(time)
    ]
    same_lineage_any_control = controls[controls["cell_type_broad"].eq(lineage)]
    wrong_time = perturb[
        perturb["cell_type_broad"].eq(lineage)
        & perturb["gene_target"].eq(target)
        & ~perturb["timepoint_num"].eq(time)
    ]
    wrong_target = perturb[
        perturb["cell_type_broad"].eq(lineage)
        & perturb["timepoint_num"].eq(time)
        & ~perturb["gene_target"].eq(target)
    ]
    wrong_lineage_same_time = perturb[
        ~perturb["cell_type_broad"].eq(lineage) & perturb["timepoint_num"].eq(time)
    ]
    wrong_lineage_same_target = perturb[
        ~perturb["cell_type_broad"].eq(lineage) & perturb["gene_target"].eq(target)
    ]

    family_counts = {
        "same_lineage_time_control_n": int(len(same_lineage_time_control)),
        "same_lineage_any_control_n": int(len(same_lineage_any_control)),
        "wrong_time_n": int(len(wrong_time)),
        "wrong_target_n": int(len(wrong_target)),
        "wrong_lineage_same_time_n": int(len(wrong_lineage_same_time)),
        "wrong_lineage_same_target_n": int(len(wrong_lineage_same_target)),
    }
    control_cells = int(same_lineage_time_control["cells"].sum()) if not same_lineage_time_control.empty else 0
    control_embryos = (
        int(same_lineage_time_control["embryos"].max()) if not same_lineage_time_control.empty else 0
    )
    control_family_count = sum(
        1
        for key in [
            "wrong_time_n",
            "wrong_target_n",
            "wrong_lineage_same_time_n",
            "wrong_lineage_same_target_n",
        ]
        if family_counts[key] > 0
    )
    return {
        **family_counts,
        "same_lineage_time_control_cells": control_cells,
        "same_lineage_time_control_embryos": control_embryos,
        "control_family_count": control_family_count,
        "wrong_time_rows": ";".join(wrong_time["row_id"].astype(str).head(12)),
        "wrong_target_rows": ";".join(wrong_target["row_id"].astype(str).head(12)),
        "wrong_lineage_same_time_rows": ";".join(
            wrong_lineage_same_time["row_id"].astype(str).head(12)
        ),
        "wrong_lineage_same_target_rows": ";".join(
            wrong_lineage_same_target["row_id"].astype(str).head(12)
        ),
    }


def build_candidate_rows(cond: pd.DataFrame, existing_ids: set[str]) -> pd.DataFrame:
    perturb = cond[~cond["is_control"]].copy()
    rows: list[dict[str, Any]] = []
    for _, row in perturb.iterrows():
        controls = count_controls(cond, row)
        new_to_current_atlas = str(row["row_id"]) not in existing_ids
        base_ready = (
            int(row["cells"]) >= 256
            and int(row["embryos"]) >= 4
            and controls["same_lineage_time_control_cells"] >= 512
            and controls["same_lineage_time_control_embryos"] >= 4
            and controls["control_family_count"] >= 2
        )
        strong_ready = (
            int(row["cells"]) >= 512
            and controls["same_lineage_time_control_cells"] >= 1024
            and controls["control_family_count"] >= 3
        )
        prospective_candidate = bool(base_ready and new_to_current_atlas)
        non_connective_candidate = bool(
            prospective_candidate and str(row["cell_type_broad"]) != CONNECTIVE_LINEAGE
        )
        rows.append(
            {
                "row_id": row["row_id"],
                "lineage": row["cell_type_broad"],
                "target": row["gene_target"],
                "timepoint": row["timepoint_num"],
                "perturb_cells": int(row["cells"]),
                "perturb_embryos": int(row["embryos"]),
                "perturb_samples": int(row["samples"]),
                "perturb_subtypes": int(row["subtypes"]),
                "top_perturb_subtypes": row.get("top_subtypes", ""),
                "mean_n_umi": finite_float(row.get("mean_n_umi")),
                "mean_num_genes": finite_float(row.get("mean_num_genes")),
                "new_to_current_atlas": new_to_current_atlas,
                "metadata_ready": base_ready,
                "strong_metadata_ready": strong_ready,
                "prospective_candidate": prospective_candidate,
                "non_connective_candidate": non_connective_candidate,
                **controls,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        [
            "prospective_candidate",
            "non_connective_candidate",
            "strong_metadata_ready",
            "control_family_count",
            "perturb_cells",
            "same_lineage_time_control_cells",
        ],
        ascending=[False, False, False, False, False, False],
    )


def decide(rows: pd.DataFrame) -> dict[str, Any]:
    cand = rows[rows["prospective_candidate"]].copy()
    non_conn = cand[cand["lineage"] != CONNECTIVE_LINEAGE].copy()
    strong = cand[cand["strong_metadata_ready"]].copy()
    status = "zscape_prospective_manifest_expansion_blocked"
    next_action = "do not launch expanded OT from metadata alone"
    if len(non_conn) >= 20 and non_conn["lineage"].nunique() >= 3 and len(strong) >= 10:
        status = "zscape_prospective_manifest_expansion_ready_cpu_ot_only"
        next_action = (
            "launch a capped CPU-only OT atlas expansion over top non-connective "
            "metadata-ready rows, then apply the same strict matched-null controls"
        )
    elif len(non_conn) >= 8 and non_conn["lineage"].nunique() >= 2:
        status = "zscape_prospective_manifest_expansion_partial_design_review"
        next_action = (
            "manually review/cap non-connective candidates before any CPU OT expansion; "
            "do not train"
        )
    return {
        "status": status,
        "gpu_authorized_next": False,
        "candidate_rows": int(len(cand)),
        "candidate_lineages": int(cand["lineage"].nunique()) if not cand.empty else 0,
        "non_connective_candidate_rows": int(len(non_conn)),
        "non_connective_candidate_lineages": int(non_conn["lineage"].nunique()) if not non_conn.empty else 0,
        "strong_candidate_rows": int(len(strong)),
        "top_candidate_row_ids": cand["row_id"].astype(str).head(40).tolist(),
        "top_non_connective_row_ids": non_conn["row_id"].astype(str).head(40).tolist(),
        "closed_do_not_rerun_blindly": [
            "current_failed_basal_retinal_25row_atlas_candidates",
            "ZSCAPE-loss/model-constraint GPU route before strict specificity/no-harm",
            "periderm noto/smo model-positive route before specificity repair",
        ],
        "next_action": next_action,
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def md_table(df: pd.DataFrame, n: int = 30) -> str:
    if df.empty:
        return "_None._"
    cols = [
        "row_id",
        "lineage",
        "target",
        "timepoint",
        "perturb_cells",
        "perturb_embryos",
        "same_lineage_time_control_cells",
        "control_family_count",
        "wrong_time_n",
        "wrong_target_n",
        "wrong_lineage_same_time_n",
        "wrong_lineage_same_target_n",
        "strong_metadata_ready",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(n).iterrows():
        vals: list[str] = []
        for col in cols:
            val = row.get(col)
            vals.append(fmt(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(rows: pd.DataFrame, decision: dict[str, Any], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "zscape_prospective_manifest_expansion_rows_20260630.csv"
    candidate_path = args.out_dir / "zscape_prospective_manifest_expansion_candidates_20260630.csv"
    json_path = args.out_dir / "zscape_prospective_manifest_expansion_gate_20260630.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_PROSPECTIVE_MANIFEST_EXPANSION_GATE_20260630.md"
    rows.to_csv(rows_path, index=False)
    rows[rows["prospective_candidate"]].to_csv(candidate_path, index=False)
    payload = {
        "timestamp_cst": now_cst(),
        "boundary": {
            "metadata_only": True,
            "reads_expression_matrix": False,
            "new_ot_pairing": False,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {
            "metadata": str(args.metadata),
            "existing_atlas_rows": str(args.atlas_rows),
            "existing_strict_rows": str(args.strict_rows),
        },
        "decision": decision,
        "outputs": {
            "rows": str(rows_path),
            "candidates": str(candidate_path),
            "markdown_report": str(md_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    cand = rows[rows["prospective_candidate"]].copy()
    non_conn = cand[cand["lineage"] != CONNECTIVE_LINEAGE].copy()
    text = f"""# ZSCAPE Prospective Manifest Expansion Gate

## Boundary

- Metadata-only preflight over full ZSCAPE zperturb cell metadata.
- Does not read expression matrices, compute OT pairs, train, infer, select checkpoints, use canonical multi, access Track C query, or use GPU.

## Decision

- Status: `{decision['status']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- Candidate rows: `{decision['candidate_rows']}`
- Candidate lineages: `{decision['candidate_lineages']}`
- Non-connective candidate rows: `{decision['non_connective_candidate_rows']}`
- Non-connective candidate lineages: `{decision['non_connective_candidate_lineages']}`
- Strong candidate rows: `{decision['strong_candidate_rows']}`
- Next action: {decision['next_action']}

## Candidate Definition

A row is metadata-ready only if it is new to the current atlas, has at least
`256` perturb cells, at least `4` perturb embryos, at least `512` same-lineage
same-time control cells, at least `4` same-lineage same-time control embryos,
and at least two available strict-control families among wrong-time,
wrong-target, wrong-lineage-same-time, and wrong-lineage-same-target.

This gate only authorizes a later CPU-only OT/strict-control design if the
metadata coverage is broad enough. It never authorizes GPU or a model constraint.

## Top Non-Connective Candidates

{md_table(non_conn)}

## Closed Branches Not To Rerun Blindly

{chr(10).join(f'- `{x}`' for x in decision['closed_do_not_rerun_blindly'])}

## Outputs

- Rows: `{rows_path}`
- Candidates: `{candidate_path}`
- JSON: `{json_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--atlas-rows", type=Path, default=DEFAULT_ATLAS_ROWS)
    parser.add_argument("--strict-rows", type=Path, default=DEFAULT_STRICT_ROWS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    existing_ids = load_existing_ids([args.atlas_rows, args.strict_rows])
    cond = summarize_conditions(args.metadata)
    rows = build_candidate_rows(cond, existing_ids)
    decision = decide(rows)
    write_outputs(rows, decision, args)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
