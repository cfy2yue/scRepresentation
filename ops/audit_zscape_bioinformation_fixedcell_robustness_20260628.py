#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from audit_zscape_expression_ot_strict_controls_20260628 import (
    control_only_embed,
    make_cell_level_manifest,
    read_cell_index,
    summarize_primary_row,
)


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def load_manifest(counts: sp.spmatrix, cell_index_path: Path, matched_manifest_path: Path, args: argparse.Namespace):
    cell_index = read_cell_index(cell_index_path)
    manifest = pd.read_csv(matched_manifest_path)
    manifest["expression_col_index"] = manifest["cell"].map(cell_index)
    manifest = manifest.dropna(subset=["expression_col_index"]).copy()
    manifest["expression_col_index"] = manifest["expression_col_index"].astype(int)
    manifest = manifest.set_index("expression_col_index", drop=False)
    cell_manifest = make_cell_level_manifest(manifest, counts.shape[1])
    emb, libraries, embed_meta = control_only_embed(counts, cell_manifest, args.n_hvg, args.n_pca, args.seed)
    manifest["expression_library"] = libraries[manifest.index.to_numpy(dtype=int)]
    manifest["log_library"] = np.log1p(pd.to_numeric(manifest["expression_library"], errors="coerce").fillna(0))
    for col in ["n_umi", "num_genes_expressed", "manifest_timepoint", "timepoint"]:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors="coerce")
    return manifest, emb, embed_meta


def evaluate_unique_rows(
    manifest: pd.DataFrame,
    emb: np.ndarray,
    row_ids: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row_id in sorted(set(row_ids)):
        group = manifest[manifest["row_id"].astype(str) == row_id]
        if len(group) == 0:
            results.append({"row_id": row_id, "status": "missing_from_manifest", "strict_row_gate": False})
            continue
        result, diag = summarize_primary_row(
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
        results.append(result)
        diagnostics.extend(diag)
    return pd.DataFrame(results), pd.DataFrame(diagnostics)


def subset_summary(spec: dict[str, Any], row_df: pd.DataFrame) -> dict[str, Any]:
    subset_rows = row_df[row_df["row_id"].astype(str).isin(spec["row_ids"])].copy()
    gate = subset_rows.get("strict_row_gate", pd.Series(dtype=bool)).fillna(False).astype(bool)
    ratio = pd.to_numeric(subset_rows.get("effect_ratio_vs_max_null_p95", pd.Series(dtype=float)), errors="coerce")
    jsd = pd.to_numeric(subset_rows.get("matched_subtype_jsd", pd.Series(dtype=float)), errors="coerce")
    lib_smd = pd.to_numeric(subset_rows.get("expression_library_smd", pd.Series(dtype=float)), errors="coerce").abs()
    ok = subset_rows["status"].eq("ok") if "status" in subset_rows else pd.Series([False] * len(subset_rows))
    lineage_pass = {}
    if "cell_type_broad" in subset_rows:
        for lineage, group in subset_rows.groupby("cell_type_broad"):
            lineage_pass[str(lineage)] = int(group.get("strict_row_gate", pd.Series(dtype=bool)).fillna(False).sum())
    return {
        "subset": spec["name"],
        "purpose": spec["purpose"],
        "n_spec_rows": int(len(spec["row_ids"])),
        "n_evaluated_rows": int(len(subset_rows)),
        "n_ok_rows": int(ok.sum()) if len(ok) else 0,
        "n_pass_rows": int(gate.sum()) if len(gate) else 0,
        "pass_fraction": float(gate.mean()) if len(gate) else float("nan"),
        "mean_effect_ratio": float(ratio.mean()) if ratio.notna().any() else float("nan"),
        "median_effect_ratio": float(ratio.median()) if ratio.notna().any() else float("nan"),
        "max_matched_subtype_jsd": float(jsd.max()) if jsd.notna().any() else float("nan"),
        "max_abs_library_smd": float(lib_smd.max()) if lib_smd.notna().any() else float("nan"),
        "lineage_pass_counts": json.dumps(lineage_pass, sort_keys=True),
    }


def decide(
    summary_df: pd.DataFrame,
    min_primary_pass: int,
    min_pass_margin: float,
    min_ratio_margin: float,
    decision_mode: str,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    def get(name: str) -> pd.Series | None:
        rows = summary_df[summary_df["subset"] == name]
        if rows.empty:
            reasons.append(f"missing_subset:{name}")
            return None
        return rows.iloc[0]

    periderm = get("primary_clean_periderm")
    retinal = get("secondary_retinal_or_demoted_low_signal")
    basal = get("secondary_response_control_basal")
    if decision_mode == "periderm_partial":
        if any(x is None for x in [periderm, retinal, basal]):
            return "zscape_bioinformation_fixedcell_periderm_partial_fail_no_gpu", reasons

        if int(periderm["n_pass_rows"]) < min_primary_pass:
            reasons.append("periderm_clean_subset_does_not_retain_min_pass_count")
        low_pass_frac = max(float(retinal["pass_fraction"]), float(basal["pass_fraction"]))
        periderm_pass_frac = float(periderm["pass_fraction"])
        if periderm_pass_frac - low_pass_frac < min_pass_margin:
            reasons.append("periderm_pass_fraction_not_above_low_controls")
        low_ratio = max(float(retinal["mean_effect_ratio"]), float(basal["mean_effect_ratio"]))
        periderm_ratio = float(periderm["mean_effect_ratio"])
        if periderm_ratio - low_ratio < min_ratio_margin:
            reasons.append("periderm_mean_ratio_not_above_low_controls")
        status = (
            "zscape_bioinformation_fixedcell_periderm_partial_pass_no_gpu"
            if not reasons
            else "zscape_bioinformation_fixedcell_periderm_partial_or_fail_no_gpu"
        )
        return status, reasons

    muscle = get("primary_high_effect_muscle")
    if any(x is None for x in [muscle, periderm, retinal, basal]):
        return "zscape_bioinformation_fixedcell_robustness_fail_no_gpu", reasons

    high_pass = int(muscle["n_pass_rows"]) >= min_primary_pass and int(periderm["n_pass_rows"]) >= min_primary_pass
    if not high_pass:
        reasons.append("primary_high_information_subsets_do_not_retain_min_pass_count")

    low_pass_frac = max(float(retinal["pass_fraction"]), float(basal["pass_fraction"]))
    high_pass_frac = min(float(muscle["pass_fraction"]), float(periderm["pass_fraction"]))
    if high_pass_frac - low_pass_frac < min_pass_margin:
        reasons.append("high_information_pass_fraction_not_above_low_controls")

    low_ratio = max(float(retinal["mean_effect_ratio"]), float(basal["mean_effect_ratio"]))
    high_ratio = min(float(muscle["mean_effect_ratio"]), float(periderm["mean_effect_ratio"]))
    if high_ratio - low_ratio < min_ratio_margin:
        reasons.append("high_information_mean_ratio_not_above_low_controls")

    status = (
        "zscape_bioinformation_fixedcell_robustness_pass_no_gpu"
        if not reasons
        else "zscape_bioinformation_fixedcell_robustness_partial_or_fail_no_gpu"
    )
    return status, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, required=True)
    parser.add_argument("--cell-index", type=Path, required=True)
    parser.add_argument("--matched-manifest", type=Path, required=True)
    parser.add_argument("--subset-spec-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pca", type=int, default=32)
    parser.add_argument("--ot-cells", type=int, default=96)
    parser.add_argument("--null-repeats", type=int, default=500)
    parser.add_argument("--min-effect-ratio", type=float, default=1.05)
    parser.add_argument("--max-subtype-jsd", type=float, default=0.10)
    parser.add_argument("--max-library-abs-smd", type=float, default=0.35)
    parser.add_argument("--min-primary-pass", type=int, default=3)
    parser.add_argument("--min-pass-margin", type=float, default=0.40)
    parser.add_argument("--min-ratio-margin", type=float, default=0.05)
    parser.add_argument("--decision-mode", choices=["all_primary", "periderm_partial"], default="all_primary")
    parser.add_argument(
        "--subset-names",
        default="",
        help="Optional comma-separated subset names to evaluate from the frozen spec.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.subset_spec_json.read_text(encoding="utf-8"))
    subsets = spec["subsets"]
    if args.subset_names.strip():
        wanted = {name.strip() for name in args.subset_names.split(",") if name.strip()}
        subsets = [subset for subset in subsets if subset["name"] in wanted]
        missing = sorted(wanted - {subset["name"] for subset in subsets})
        if missing:
            raise ValueError(f"Requested subset names not found in spec: {missing}")
    row_ids = sorted({rid for subset in subsets for rid in subset["row_ids"]})

    counts = sp.load_npz(args.counts_npz)
    manifest, emb, embed_meta = load_manifest(counts, args.cell_index, args.matched_manifest, args)
    row_df, diag_df = evaluate_unique_rows(manifest, emb, row_ids, args)
    summary_df = pd.DataFrame([subset_summary(item, row_df) for item in subsets])
    status, reasons = decide(
        summary_df,
        args.min_primary_pass,
        args.min_pass_margin,
        args.min_ratio_margin,
        args.decision_mode,
    )

    row_csv = args.out_dir / "zscape_bioinformation_fixedcell_row_results.csv"
    diag_csv = args.out_dir / "zscape_bioinformation_fixedcell_diagnostics.csv"
    summary_csv = args.out_dir / "zscape_bioinformation_fixedcell_subset_summary.csv"
    row_df.to_csv(row_csv, index=False)
    diag_df.to_csv(diag_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    json_path = args.out_dir / "zscape_bioinformation_fixedcell_robustness_20260628.json"
    output = {
        "timestamp_utc": utc_now(),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "subset_spec_json": str(args.subset_spec_json),
        "counts_npz": str(args.counts_npz),
        "matched_manifest": str(args.matched_manifest),
        "embed_meta": embed_meta,
        "ot_cells": args.ot_cells,
        "null_repeats": args.null_repeats,
        "decision_mode": args.decision_mode,
        "subset_names": args.subset_names,
        "row_csv": str(row_csv),
        "diagnostics_csv": str(diag_csv),
        "summary_csv": str(summary_csv),
    }
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    md_path = args.out_dir / "LATENTFM_ZSCAPE_BIOINFORMATION_FIXEDCELL_ROBUSTNESS_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Bioinformation Fixed-Cell Robustness Gate",
        "",
        f"Timestamp: `{utc_now()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only robustness gate over frozen subset specs.",
        "- Uses control-only HVG/SVD and strict row-level matching/null logic.",
        "- Does not train, infer, run scFM embeddings, read canonical multi, or read Track C query.",
        "- Passing this gate can only support a later bounded LatentFM design review.",
        "",
        "## Gate Parameters",
        "",
        f"- fixed OT cells per row: `{args.ot_cells}`",
        f"- null repeats: `{args.null_repeats}`",
        f"- min primary pass count per high-I subset: `{args.min_primary_pass}`",
        f"- min high-vs-low pass fraction margin: `{args.min_pass_margin}`",
        f"- min high-vs-low ratio margin: `{args.min_ratio_margin}`",
        f"- decision mode: `{args.decision_mode}`",
        f"- subset names: `{args.subset_names or 'all'}`",
        "",
        "## Subset Summary",
        "",
        "| subset | rows | pass | pass frac | mean ratio | max JSD | max abs lib SMD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['subset']} | {int(row['n_evaluated_rows'])} | {int(row['n_pass_rows'])} | "
            f"{float(row['pass_fraction']):.3f} | {float(row['mean_effect_ratio']):.3f} | "
            f"{float(row['max_matched_subtype_jsd']):.3f} | {float(row['max_abs_library_smd']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Reasons",
            "",
            *(f"- {reason}" for reason in reasons),
            "",
            "## Output Files",
            "",
            f"- row results: `{row_csv}`",
            f"- subset summary: `{summary_csv}`",
            f"- diagnostics: `{diag_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(status)
    return 0 if status.endswith("pass_no_gpu") else 2


if __name__ == "__main__":
    raise SystemExit(main())
