#!/usr/bin/env python3
"""CPU-only strict-control expansion for high-pairability ZSCAPE atlas rows."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.sparse as sp

from audit_zscape_expression_ot_strict_controls_20260628 import (
    control_only_embed,
    make_cell_level_manifest,
    read_cell_index,
    summarize_primary_row,
)


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_COUNTS = (
    ROOT
    / "runs/zscape_raw_counts_cell_manifest_extraction_20260628"
    / "zscape_raw_counts_cell_manifest_extraction_20260628_074523"
    / "outputs/zscape_manifest_selected_counts_csc.npz"
)
DEFAULT_CELL_INDEX = DEFAULT_COUNTS.parent / "zscape_manifest_selected_expression_cell_index.csv"
DEFAULT_MATCHED_MANIFEST = DEFAULT_COUNTS.parent / "zscape_expression_selected_cell_ids_matched.csv"
DEFAULT_CANDIDATES = (
    ROOT
    / "reports/zscape_pairability_strict_control_expansion_gate_20260630"
    / "zscape_pairability_strict_control_expansion_rows_20260630.csv"
)
DEFAULT_OUT = ROOT / "reports/zscape_pairability_strict_control_expansion_20260630"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def ensure_output_dir(path: Path, force: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [p for p in path.iterdir() if p.name != ".DS_Store"]
    if existing and not force:
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {path}")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def prepare_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, Any, dict[str, Any], pd.DataFrame]:
    counts = sp.load_npz(args.counts_npz)
    cell_index = read_cell_index(args.cell_index)
    manifest = pd.read_csv(args.matched_manifest)
    manifest["expression_col_index"] = manifest["cell"].map(cell_index)
    manifest = manifest.dropna(subset=["expression_col_index"]).copy()
    manifest["expression_col_index"] = manifest["expression_col_index"].astype(int)
    manifest = manifest.set_index("expression_col_index", drop=False)
    cell_manifest = make_cell_level_manifest(manifest, counts.shape[1])
    emb, libraries, embed_meta = control_only_embed(counts, cell_manifest, args.n_hvg, args.n_pca, args.seed)
    manifest["expression_library"] = libraries[manifest.index.to_numpy(dtype=int)]
    manifest["log_library"] = pd.Series(manifest["expression_library"], index=manifest.index).map(
        lambda x: __import__("numpy").log1p(float(x))
    )
    for col in ["n_umi", "num_genes_expressed", "manifest_timepoint"]:
        manifest[col] = pd.to_numeric(manifest[col], errors="coerce")
    candidates = pd.read_csv(args.candidates)
    candidates = candidates[candidates["strict_expansion_candidate"].map(truthy)].copy()
    candidates["within_state_pairability_score"] = pd.to_numeric(
        candidates["within_state_pairability_score"], errors="coerce"
    )
    candidates = candidates.sort_values("within_state_pairability_score", ascending=False).head(args.max_rows)
    return manifest, emb, embed_meta, candidates


def run_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    manifest, emb, embed_meta, candidates = prepare_inputs(args)
    row_results: list[dict[str, Any]] = []
    diag_results: list[dict[str, Any]] = []
    missing: list[str] = []
    for row_id in candidates["row_id"].astype(str):
        group = manifest[manifest["row_id"].astype(str) == row_id]
        if group.empty:
            missing.append(row_id)
            continue
        result, diagnostics = summarize_primary_row(
            row_id,
            group,
            manifest,
            emb,
            args.seed,
            args.ot_cells,
            args.null_repeats,
            args.min_effect_ratio,
            args.max_subtype_jsd,
            args.max_library_abs_smd,
        )
        row_results.append(result)
        diag_results.extend(diagnostics)
    row_df = pd.DataFrame(row_results)
    diag_df = pd.DataFrame(diag_results)
    summary = {
        "status": "zscape_pairability_strict_control_expansion_complete_no_gpu",
        "gpu_authorized_next": False,
        "candidate_rows_requested": int(len(candidates)),
        "candidate_rows_evaluated": int(len(row_df)),
        "missing_candidate_rows": missing,
        "strict_rows_passing": int(row_df.get("strict_row_gate", pd.Series(dtype=bool)).fillna(False).sum())
        if not row_df.empty
        else 0,
        "strict_passing_row_ids": row_df.loc[
            row_df.get("strict_row_gate", pd.Series(dtype=bool)).fillna(False), "row_id"
        ].astype(str).tolist()
        if not row_df.empty
        else [],
        "lineage_pass_counts": {
            str(k): int(v)
            for k, v in row_df[row_df.get("strict_row_gate", pd.Series(dtype=bool)).fillna(False)]
            .groupby("cell_type_broad")
            .size()
            .to_dict()
            .items()
        }
        if not row_df.empty
        else {},
        "embedding": embed_meta,
        "filters": {
            "max_rows": args.max_rows,
            "ot_cells": args.ot_cells,
            "null_repeats": args.null_repeats,
            "min_effect_ratio": args.min_effect_ratio,
            "max_subtype_jsd": args.max_subtype_jsd,
            "max_library_abs_smd": args.max_library_abs_smd,
            "n_hvg": args.n_hvg,
            "n_pca": args.n_pca,
        },
        "resource": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }
    pass_lineages = len(summary["lineage_pass_counts"])
    if summary["strict_rows_passing"] >= args.min_pass_rows and pass_lineages >= args.min_pass_lineages:
        summary["status"] = "zscape_pairability_strict_control_expansion_pass_design_review_only"
        summary["next_action"] = (
            "design a stricter full-control atlas with wrong-target/time/lineage and "
            "heldout controls; still no GPU model route"
        )
    else:
        summary["next_action"] = (
            "do not use high-pairability atlas rows as model positives; treat them as "
            "descriptive candidates or expand controls"
        )
    return row_df, diag_df, summary, candidates


def fmt(value: Any, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{x:.{digits}f}" if x == x else "NA"


def write_outputs(args: argparse.Namespace, row_df: pd.DataFrame, diag_df: pd.DataFrame, summary: dict[str, Any], candidates: pd.DataFrame) -> None:
    row_path = args.out_dir / "zscape_pairability_strict_control_expansion_rows_20260630.csv"
    diag_path = args.out_dir / "zscape_pairability_strict_control_expansion_diagnostics_20260630.csv"
    cand_path = args.out_dir / "zscape_pairability_strict_control_expansion_candidates_20260630.csv"
    json_path = args.out_dir / "zscape_pairability_strict_control_expansion_20260630.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_PAIRABILITY_STRICT_CONTROL_EXPANSION_20260630.md"
    row_df.to_csv(row_path, index=False)
    diag_df.to_csv(diag_path, index=False)
    candidates.to_csv(cand_path, index=False)
    payload = {
        "timestamp_cst": now_cst(),
        "boundary": {
            "cpu_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "gpu_authorized_next": False,
        },
        "inputs": {
            "counts_npz": str(args.counts_npz),
            "cell_index": str(args.cell_index),
            "matched_manifest": str(args.matched_manifest),
            "candidates": str(args.candidates),
        },
        "outputs": {
            "rows": str(row_path),
            "diagnostics": str(diag_path),
            "candidates": str(cand_path),
            "markdown_report": str(md_path),
        },
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# ZSCAPE Pairability Strict-Control Expansion",
        "",
        f"Created: `{payload['timestamp_cst']}`",
        "",
        f"Status: `{summary['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only strict-control expansion over high-pairability atlas-only ZSCAPE rows.",
        "- Reuses control-only HVG/SVD and matched-null OT logic from the original strict-control gate.",
        "- OT pairs remain snapshot pseudo-pairs, not true lineage pairs.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU use.",
        "",
        "## Summary",
        "",
        f"- candidate rows requested: `{summary['candidate_rows_requested']}`",
        f"- candidate rows evaluated: `{summary['candidate_rows_evaluated']}`",
        f"- strict rows passing: `{summary['strict_rows_passing']}`",
        f"- strict passing rows: `{', '.join(summary['strict_passing_row_ids']) or 'none'}`",
        f"- lineage pass counts: `{summary['lineage_pass_counts']}`",
        f"- next action: {summary['next_action']}",
        "",
        "## Row Results",
        "",
        "| row_id | lineage | target | time | obs OT | cc p95 | label p95 | ratio | p_cc | p_label | subtype JSD | lib SMD | gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if row_df.empty:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |  |  |")
    else:
        for _, row in row_df.sort_values("row_id").iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("row_id", "")),
                        str(row.get("cell_type_broad", "")),
                        str(row.get("gene_target", "")),
                        fmt(row.get("timepoint"), 1),
                        fmt(row.get("observed_strict_ot")),
                        fmt(row.get("cc_null_p95")),
                        fmt(row.get("label_null_p95")),
                        fmt(row.get("effect_ratio_vs_max_null_p95")),
                        fmt(row.get("p_observed_le_matched_cc_null")),
                        fmt(row.get("p_observed_le_matched_label_null")),
                        fmt(row.get("matched_subtype_jsd")),
                        fmt(row.get("expression_library_smd")),
                        str(row.get("strict_row_gate", "")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Rows: `{row_path}`",
            f"- Diagnostics: `{diag_path}`",
            f"- Candidates: `{cand_path}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, required=True)
    parser.add_argument("--cell-index", type=Path, required=True)
    parser.add_argument("--matched-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hvg", type=int, default=1500)
    parser.add_argument("--n-pca", type=int, default=24)
    parser.add_argument("--ot-cells", type=int, default=64)
    parser.add_argument("--null-repeats", type=int, default=100)
    parser.add_argument("--max-rows", type=int, required=True)
    parser.add_argument("--min-effect-ratio", type=float, default=1.05)
    parser.add_argument("--max-subtype-jsd", type=float, default=0.10)
    parser.add_argument("--max-library-abs-smd", type=float, default=0.35)
    parser.add_argument("--min-pass-rows", type=int, default=4)
    parser.add_argument("--min-pass-lineages", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_output_dir(args.out_dir, args.force)
    row_df, diag_df, summary, candidates = run_rows(args)
    write_outputs(args, row_df, diag_df, summary, candidates)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
