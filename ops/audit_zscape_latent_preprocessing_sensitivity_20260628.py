#!/usr/bin/env python3
"""ZSCAPE latent preprocessing sensitivity gate.

CPU-only audit for whether ZSCAPE expression-latent conclusions depend on
simple QC filtering, HVG budget, or applying log1p after size-factor
normalization.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

import audit_zscape_expression_latent_biology_preflight_20260628 as base


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_OUT = ROOT / "reports/zscape_latent_preprocessing_sensitivity_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    base.write_csv(path, rows, list(rows[0].keys()))


def sizenorm_counts(counts: sparse.csc_matrix, keep_cols: np.ndarray) -> sparse.csc_matrix:
    sub = counts[:, keep_cols].astype(np.float32, copy=True)
    lib = np.asarray(sub.sum(axis=0)).ravel().astype(np.float64)
    positive = lib[lib > 0]
    scale_base = float(np.median(positive)) if positive.size else 1.0
    scale = scale_base / np.maximum(lib, 1.0)
    indptr = sub.indptr
    for col in range(sub.shape[1]):
        start, end = int(indptr[col]), int(indptr[col + 1])
        if end > start:
            sub.data[start:end] *= scale[col]
    return sub


def response_vectors(latent: np.ndarray, groups: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for row_id, group in groups.items():
        if not group["perturb"] or not group["control"]:
            continue
        vec = base.centroid(latent, group["perturb"]) - base.centroid(latent, group["control"])
        out[row_id] = vec
    return out


def pairwise_signature(vectors: dict[str, np.ndarray], row_ids: list[str]) -> np.ndarray:
    vals: list[float] = []
    for i, a in enumerate(row_ids):
        va = vectors.get(a)
        if va is None:
            continue
        for b in row_ids[i + 1 :]:
            vb = vectors.get(b)
            if vb is None:
                continue
            vals.append(base.cosine(va, vb))
    return np.asarray(vals, dtype=float)


def signature_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 3:
        return float("nan")
    return base.pearson(a[:n], b[:n])


def variant_matrix(
    counts: sparse.csc_matrix,
    keep_old: np.ndarray,
    use_log1p: bool,
) -> sparse.csc_matrix:
    if use_log1p:
        return base.lognorm_counts(counts, keep_old)
    return sizenorm_counts(counts, keep_old)


def run_variant(
    counts: sparse.csc_matrix,
    manifest: list[dict[str, str]],
    qc_flags: np.ndarray,
    variant: dict[str, Any],
    reference: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    keep_mask = qc_flags if variant["apply_qc"] else np.ones(len(manifest), dtype=bool)
    keep_old = np.where(keep_mask)[0]
    old_to_new = {int(old): int(new) for new, old in enumerate(keep_old)}
    groups = base.build_row_groups(manifest, old_to_new)
    matrix = variant_matrix(counts, keep_old, bool(variant["use_log1p"]))
    _mean, var, _det = base.gene_moments(matrix, list(range(matrix.shape[1])))
    hvg_rank = np.argsort(-var, kind="mergesort")
    latent = base.fit_svd_latent(matrix, hvg_rank, int(variant["n_hvg"]), int(args.latent_dim))
    vectors = response_vectors(latent, groups)
    alignment = base.latent_alignment_rows(latent, groups, str(variant["label"]))

    primary_ids = sorted(
        row_id for row_id, group in groups.items() if group.get("audit_role") == "primary_mechanism_test"
    )
    all_ids = sorted(vectors)
    periderm_ids = [row_id for row_id in primary_ids if groups[row_id].get("lineage") == "periderm"]
    muscle_ids = [row_id for row_id in primary_ids if groups[row_id].get("lineage") == "mature fast muscle"]

    primary_sig = pairwise_signature(vectors, primary_ids)
    all_sig = pairwise_signature(vectors, all_ids)
    ref_primary_corr = float("nan")
    ref_all_corr = float("nan")
    if reference is not None:
        ref_primary_corr = signature_corr(primary_sig, reference["primary_signature"])
        ref_all_corr = signature_corr(all_sig, reference["all_signature"])

    align_by_id = {row["row_id"]: row for row in alignment}
    primary_align = [align_by_id[row_id] for row_id in primary_ids if row_id in align_by_id]
    periderm_align = [align_by_id[row_id] for row_id in periderm_ids if row_id in align_by_id]
    muscle_align = [align_by_id[row_id] for row_id in muscle_ids if row_id in align_by_id]

    def gate_frac(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return float("nan")
        return float(np.mean([bool(row.get("alignment_gate")) for row in rows]))

    focus_rows = [
        "periderm__noto__24p0h",
        "periderm__smo__24p0h",
        "periderm__tbx16_tbx16l__24p0h",
        "mature_fast_muscle__tbx16_tbx16l__24p0h",
    ]
    focus = {}
    for row_id in focus_rows:
        row = align_by_id.get(row_id)
        if row is None:
            continue
        focus[f"{row_id}.temporal_cosine"] = float(row["cosine_to_lineage_time_vector"])
        focus[f"{row_id}.margin"] = float(row["cosine_margin_vs_wrong_lineage"])
        focus[f"{row_id}.gate"] = bool(row["alignment_gate"])

    summary = {
        "variant": variant["label"],
        "apply_qc": bool(variant["apply_qc"]),
        "use_log1p": bool(variant["use_log1p"]),
        "n_hvg": int(variant["n_hvg"]),
        "n_cells": int(len(keep_old)),
        "n_primary_rows": int(len(primary_ids)),
        "primary_alignment_gate_fraction": gate_frac(primary_align),
        "periderm_alignment_gate_fraction": gate_frac(periderm_align),
        "muscle_alignment_gate_fraction": gate_frac(muscle_align),
        "median_primary_response_norm": float(
            np.median([float(row["response_norm"]) for row in primary_align])
        )
        if primary_align
        else float("nan"),
        "signature_corr_vs_ref_primary": ref_primary_corr,
        "signature_corr_vs_ref_all": ref_all_corr,
        **focus,
    }
    reference_payload = {
        "primary_signature": primary_sig,
        "all_signature": all_sig,
        "summary": summary,
    }
    return summary, alignment, reference_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=base.DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=base.DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-umi", type=float, default=100.0)
    parser.add_argument("--min-genes", type=float, default=100.0)
    parser.add_argument("--latent-dim", type=int, default=32)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = base.load_csc(args.counts_npz)
    manifest = base.index_manifest(args.cell_index, args.matched_manifest, counts.shape[1])
    qc_flags = np.asarray([base.qc_pass(row, args.min_umi, args.min_genes) for row in manifest], dtype=bool)

    variants = [
        {"label": "ref_noqc_log1p_hvg2000", "apply_qc": False, "use_log1p": True, "n_hvg": 2000},
        {"label": "noqc_log1p_hvg1000", "apply_qc": False, "use_log1p": True, "n_hvg": 1000},
        {"label": "noqc_log1p_hvg4000", "apply_qc": False, "use_log1p": True, "n_hvg": 4000},
        {"label": "noqc_log1p_hvg8000", "apply_qc": False, "use_log1p": True, "n_hvg": 8000},
        {"label": "qc_log1p_hvg2000", "apply_qc": True, "use_log1p": True, "n_hvg": 2000},
        {"label": "noqc_sizenorm_no_log1p_hvg2000", "apply_qc": False, "use_log1p": False, "n_hvg": 2000},
        {"label": "noqc_sizenorm_no_log1p_hvg8000", "apply_qc": False, "use_log1p": False, "n_hvg": 8000},
    ]

    summaries: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    for variant in variants:
        summary, alignment, payload = run_variant(counts, manifest, qc_flags, variant, reference, args)
        if reference is None:
            reference = payload
            summary["signature_corr_vs_ref_primary"] = 1.0
            summary["signature_corr_vs_ref_all"] = 1.0
        summaries.append(summary)
        for row in alignment:
            alignments.append({"variant": variant["label"], **row})

    summary_by_variant = {row["variant"]: row for row in summaries}
    qc_corr = float(summary_by_variant["qc_log1p_hvg2000"]["signature_corr_vs_ref_primary"])
    hvg_corr_min = float(
        min(
            summary_by_variant["noqc_log1p_hvg1000"]["signature_corr_vs_ref_primary"],
            summary_by_variant["noqc_log1p_hvg4000"]["signature_corr_vs_ref_primary"],
            summary_by_variant["noqc_log1p_hvg8000"]["signature_corr_vs_ref_primary"],
        )
    )
    no_log_corr = float(summary_by_variant["noqc_sizenorm_no_log1p_hvg2000"]["signature_corr_vs_ref_primary"])
    qc_stable = qc_corr >= 0.99
    hvg_stable = hvg_corr_min >= 0.85
    log1p_required = no_log_corr < 0.85
    status = (
        "zscape_latent_preprocessing_sensitivity_pass_no_gpu"
        if qc_stable and hvg_stable and log1p_required
        else "zscape_latent_preprocessing_sensitivity_partial_no_gpu"
    )

    summary_csv = args.out_dir / "zscape_latent_preprocessing_sensitivity_summary.csv"
    alignment_csv = args.out_dir / "zscape_latent_preprocessing_sensitivity_alignment_rows.csv"
    json_path = args.out_dir / "zscape_latent_preprocessing_sensitivity_20260628.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_LATENT_PREPROCESSING_SENSITIVITY_20260628.md"

    write_csv(summary_csv, summaries)
    write_csv(alignment_csv, alignments)

    result = {
        "timestamp_cst": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "boundary": "CPU-only ZSCAPE latent preprocessing sensitivity; no training/inference/scFM extraction/canonical multi/Track C query.",
        "qc_rule": {"min_umi": args.min_umi, "min_genes": args.min_genes},
        "n_cells_before_qc": int(len(manifest)),
        "n_cells_after_qc": int(qc_flags.sum()),
        "latent_dim": int(args.latent_dim),
        "qc_stable": qc_stable,
        "hvg_budget_stable": hvg_stable,
        "log1p_required_for_claims": log1p_required,
        "qc_signature_corr_primary": qc_corr,
        "min_hvg_signature_corr_primary": hvg_corr_min,
        "no_log1p_signature_corr_primary": no_log_corr,
        "outputs": {"summary_csv": str(summary_csv), "alignment_csv": str(alignment_csv)},
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Latent Preprocessing Sensitivity",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only sensitivity analysis over the validated ZSCAPE selected-cell raw counts.",
        "- No LatentFM training, no inference, no true scFM embedding extraction, no canonical multi, and no Track C query.",
        "- The reference latent is control-free row response geometry in `size-factor + exactly one log1p + HVG2000 + SVD32`.",
        "",
        "## Gate Summary",
        "",
        f"- QC cells: `{len(manifest)} -> {int(qc_flags.sum())}` under `n_umi >= {args.min_umi:g}` and `num_genes_expressed >= {args.min_genes:g}`.",
        f"- QC stability primary signature correlation: `{qc_corr:.4f}`; stable: `{qc_stable}`.",
        f"- minimum HVG-budget primary signature correlation across 1k/4k/8k: `{hvg_corr_min:.4f}`; stable: `{hvg_stable}`.",
        f"- no-log1p primary signature correlation at HVG2000: `{no_log_corr:.4f}`; log1p required for claims: `{log1p_required}`.",
        "",
        "## Variant Table",
        "",
        "| variant | QC | log1p | HVG | primary gate frac | periderm gate frac | muscle gate frac | corr vs ref primary | corr vs ref all |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["apply_qc"]),
                    str(row["use_log1p"]),
                    str(row["n_hvg"]),
                    f"{float(row['primary_alignment_gate_fraction']):.3f}",
                    f"{float(row['periderm_alignment_gate_fraction']):.3f}",
                    f"{float(row['muscle_alignment_gate_fraction']):.3f}",
                    f"{float(row['signature_corr_vs_ref_primary']):.4f}",
                    f"{float(row['signature_corr_vs_ref_all']):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- QC filtering at this threshold removes almost no cells; if the QC gate is stable, low-UMI/low-gene cells are not driving the current latent conclusions.",
            "- HVG-budget stability supports using information-content/HVG concentration as a scaling variable, while still separating it from a model-performance claim.",
            "- If no-log1p changes the row-response geometry, downstream ZSCAPE expression and latent-proxy analyses should keep exactly one log1p after size-factor normalization.",
            "- True scFM latent analysis remains blocked until a Danio-compatible checkpoint or frozen orthology route is available.",
            "",
            "## Outputs",
            "",
            f"- summary: `{summary_csv}`",
            f"- alignment rows: `{alignment_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
