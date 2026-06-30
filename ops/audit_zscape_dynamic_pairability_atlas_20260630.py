#!/usr/bin/env python3
"""CPU-only ZSCAPE dynamic pairability atlas over a selected manifest.

This script extends the earlier 10-row dynamic-response gate into a broader
descriptive atlas. It computes expression-space OT pseudo-pairs and structural
pairability metrics for all selected ZSCAPE rows, but it does not train,
infer, select checkpoints, or authorize GPU use.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "12")

import numpy as np
import pandas as pd

from audit_zscape_ot_dynamic_response_gate_20260628 import (
    DEFAULT_CELL_INDEX,
    DEFAULT_COUNTS,
    DEFAULT_MATCHED_MANIFEST,
    DEFAULT_SNAPSHOT_ROWS,
    DEFAULT_STRICT_DIAG,
    DEFAULT_STRICT_ROWS,
    finite_float,
    fmt,
    merge_existing_evidence,
    prepare_manifest,
    summarize_ot_row,
    truthy,
)


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports/zscape_dynamic_pairability_atlas_20260630"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def ensure_output_dir(path: Path, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in path.iterdir() if p.name != ".DS_Store"]
    if existing and not force:
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {path}")


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_pairability_scores(row_df: pd.DataFrame) -> pd.DataFrame:
    out = row_df.copy()
    for col in [
        "composition_norm_fraction_of_centroid",
        "within_substate_residual_fraction_of_centroid",
        "same_substate_pair_fraction",
        "substate_jsd",
        "mean_pair_displacement_cosine_to_centroid",
        "centroid_response_norm",
        "mean_pair_expression_distance",
        "wrong_time_margin_ot",
        "wrong_lineage_margin_ot",
        "trajectory_cosine",
    ]:
        if col in out.columns:
            out[col] = safe_numeric(out[col])
    comp = out["composition_norm_fraction_of_centroid"].fillna(1.5).clip(0, 1.5)
    within = out["within_substate_residual_fraction_of_centroid"].fillna(0).clip(0, 1.5)
    same_sub = out.get("same_substate_pair_fraction", pd.Series(0, index=out.index)).fillna(0).clip(0, 1)
    jsd = out.get("substate_jsd", pd.Series(1, index=out.index)).fillna(1).clip(0, 1)
    pair_cos = out.get("mean_pair_displacement_cosine_to_centroid", pd.Series(0, index=out.index)).fillna(0).clip(-1, 1)
    out["within_state_pairability_score"] = (
        0.45 * within.clip(0, 1)
        + 0.25 * same_sub
        + 0.15 * pair_cos.clip(lower=0)
        - 0.35 * comp
        - 0.20 * jsd
    )
    out["magnitude_pairability_ratio"] = out["within_state_pairability_score"] / np.log1p(
        out["centroid_response_norm"].fillna(0).clip(lower=0)
    ).replace(0, np.nan)
    if "strict_row_gate" in out.columns:
        out["has_strict_control_context"] = out["strict_row_gate"].notna()
    else:
        out["has_strict_control_context"] = False
    out["state_preserved_context"] = out.get("state_preserved_by_threshold", False).fillna(False).map(truthy)
    out["dynamic_response_context"] = out.get("dynamic_response_gate", False).fillna(False).map(truthy)
    out["pairability_class"] = "descriptive_pairability_only"
    out.loc[
        out["dynamic_response_context"],
        "pairability_class",
    ] = "strict_dynamic_positive_specificity_unready"
    out.loc[
        ~out["has_strict_control_context"],
        "pairability_class",
    ] = "atlas_row_no_strict_context_yet"
    out.loc[
        out["has_strict_control_context"]
        & (
            (out["composition_norm_fraction_of_centroid"] > 0.25)
            | (out.get("wrong_time_margin_ot", pd.Series(np.nan, index=out.index)) < 0)
        ),
        "pairability_class",
    ] = "magnitude_or_time_confounded"
    out.loc[
        out["has_strict_control_context"] & out["dynamic_response_context"],
        "pairability_class",
    ] = "strict_dynamic_positive_specificity_unready"
    return out


def summarize(row_df: pd.DataFrame, embed_meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    by_role = row_df.groupby("audit_role").size().to_dict() if "audit_role" in row_df.columns else {}
    by_lineage = row_df.groupby("lineage").size().to_dict() if "lineage" in row_df.columns else {}
    dynamic_rows = row_df.loc[row_df["dynamic_response_context"], "row_id"].astype(str).tolist()
    strict_context = row_df[row_df["has_strict_control_context"]]
    atlas_only = row_df[~row_df["has_strict_control_context"]]
    corr = None
    if len(row_df.dropna(subset=["centroid_response_norm", "within_state_pairability_score"])) >= 4:
        corr = finite_float(
            row_df["centroid_response_norm"].rank().corr(row_df["within_state_pairability_score"].rank())
        )
    return {
        "status": "zscape_dynamic_pairability_atlas_complete_no_gpu",
        "gpu_authorized_next": False,
        "evaluated_rows": int(len(row_df)),
        "audit_role_counts": {str(k): int(v) for k, v in by_role.items()},
        "lineage_counts": {str(k): int(v) for k, v in by_lineage.items()},
        "strict_context_rows": int(len(strict_context)),
        "atlas_only_rows_without_strict_controls": int(len(atlas_only)),
        "dynamic_context_rows": dynamic_rows,
        "response_norm_vs_pairability_spearman": corr,
        "top_pairability_rows": row_df.sort_values("within_state_pairability_score", ascending=False)[
            ["row_id", "lineage", "target", "timepoint", "within_state_pairability_score", "pairability_class"]
        ]
        .head(10)
        .to_dict(orient="records"),
        "preprocessing": embed_meta,
        "resource_plan": {
            "runtime_classification": "Long task / unknown runtime, detached tmux",
            "threads": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            },
            "ot_cells_per_row": args.ot_cells,
            "n_hvg": args.n_hvg,
            "n_pca": args.n_pca,
        },
        "decision": (
            "descriptor_atlas_only_no_model_constraint; atlas rows lacking strict wrong-time/"
            "wrong-lineage controls cannot authorize training"
        ),
    }


def markdown_table(df: pd.DataFrame, cols: list[str], n: int = 30) -> str:
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


def write_report(row_df: pd.DataFrame, pair_df: pd.DataFrame, summary: dict[str, Any], args: argparse.Namespace) -> None:
    row_path = args.out_dir / "zscape_dynamic_pairability_atlas_rows_20260630.csv"
    pair_path = args.out_dir / "zscape_dynamic_pairability_atlas_pseudo_pairs_20260630.csv"
    json_path = args.out_dir / "zscape_dynamic_pairability_atlas_20260630.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_DYNAMIC_PAIRABILITY_ATLAS_20260630.md"
    row_df.to_csv(row_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    payload = {
        "timestamp_cst": now_cst(),
        "boundary": {
            "cpu_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "new_latent_embedding_extraction": False,
            "gpu_authorized_next": False,
        },
        "inputs": {
            "counts_npz": str(args.counts_npz),
            "cell_index": str(args.cell_index),
            "matched_manifest": str(args.matched_manifest),
            "snapshot_rows": str(args.snapshot_rows),
            "strict_rows": str(args.strict_rows),
            "strict_diag": str(args.strict_diag),
        },
        "outputs": {
            "rows": str(row_path),
            "pseudo_pairs": str(pair_path),
            "markdown_report": str(md_path),
        },
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    cols = [
        "row_id",
        "audit_role",
        "lineage",
        "target",
        "timepoint",
        "n_pseudo_pairs",
        "centroid_response_norm",
        "composition_norm_fraction_of_centroid",
        "within_substate_residual_fraction_of_centroid",
        "same_substate_pair_fraction",
        "substate_jsd",
        "within_state_pairability_score",
        "pairability_class",
    ]
    text = f"""# ZSCAPE Dynamic Pairability Atlas

## Boundary

- CPU-only atlas over the already selected ZSCAPE expression manifest.
- Raw selected counts are normalized in the imported gate code using the existing policy: control-only HVG/SVD after size normalization and one `log1p`.
- OT pairs are snapshot pseudo-pairs, not true lineage pairs.
- No training, inference, GPU, checkpoint selection, canonical multi selection, or Track C query access.

## Decision

- Status: `{summary['status']}`
- GPU authorized next: `{summary['gpu_authorized_next']}`
- Evaluated rows: `{summary['evaluated_rows']}`
- Strict-context rows: `{summary['strict_context_rows']}`
- Atlas-only rows without strict controls: `{summary['atlas_only_rows_without_strict_controls']}`
- Dynamic-context rows: `{', '.join(summary['dynamic_context_rows']) or 'none'}`
- Response norm vs pairability Spearman: `{fmt(summary['response_norm_vs_pairability_spearman'])}`

This atlas expands descriptive pairability coverage, but atlas-only rows cannot
authorize a model constraint until they receive wrong-time, wrong-lineage,
wrong-target, abundance/variance, and no-harm controls.

## Rows

{markdown_table(row_df.sort_values('within_state_pairability_score', ascending=False), cols)}

## Outputs

- Rows: `{row_path}`
- Pseudo-pairs: `{pair_path}`
- JSON: `{json_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=DEFAULT_MATCHED_MANIFEST)
    parser.add_argument("--snapshot-rows", type=Path, default=DEFAULT_SNAPSHOT_ROWS)
    parser.add_argument("--strict-rows", type=Path, default=DEFAULT_STRICT_ROWS)
    parser.add_argument("--strict-diag", type=Path, default=DEFAULT_STRICT_DIAG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hvg", type=int, default=1500)
    parser.add_argument("--n-pca", type=int, default=24)
    parser.add_argument("--ot-cells", type=int, default=64)
    parser.add_argument("--min-cells", type=int, default=40)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_output_dir(args.out_dir, args.force)
    manifest, emb, embed_meta = prepare_manifest(args)
    row_results: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for row_id, group in manifest.groupby("row_id", sort=True):
        row, pairs = summarize_ot_row(row_id, group, emb, args)
        row_results.append(row)
        pair_results.extend(pairs)
    row_df = merge_existing_evidence(pd.DataFrame(row_results), args)
    row_df = build_pairability_scores(row_df)
    pair_df = pd.DataFrame(pair_results)
    summary = summarize(row_df, embed_meta, args)
    write_report(row_df, pair_df, summary, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
