#!/usr/bin/env python3
"""CPU-only ZSCAPE HVG/full-gene information-axis preflight.

This asks whether biological and perturbation-response structure is concentrated
in a small HVG-like gene set or remains meaningfully distributed across the full
gene space. It is a scaling-axis design audit, not a model result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_COUNTS = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_counts_csc.npz"
DEFAULT_CELL_INDEX = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_expression_cell_index.csv"
DEFAULT_MANIFEST = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_expression_selected_cell_ids_matched.csv"
DEFAULT_GENES = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_gene_names.txt"
DEFAULT_OUT = ROOT / "reports/zscape_hvg_fullgene_information_axis_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_csc(path: Path) -> sparse.csc_matrix:
    obj = np.load(path)
    shape = tuple(int(x) for x in obj["shape"])
    return sparse.csc_matrix((obj["data"], obj["indices"], obj["indptr"]), shape=shape)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"
    return str(value)


def lognorm_counts(counts: sparse.csc_matrix) -> sparse.csc_matrix:
    lib = np.asarray(counts.sum(axis=0)).ravel().astype(np.float64)
    positive = lib[lib > 0]
    scale_base = float(np.median(positive)) if positive.size else 1.0
    scale = scale_base / np.maximum(lib, 1.0)
    out = counts.astype(np.float32, copy=True)
    indptr = out.indptr
    for col in range(out.shape[1]):
        start, end = int(indptr[col]), int(indptr[col + 1])
        if end > start:
            out.data[start:end] *= scale[col]
    np.log1p(out.data, out=out.data)
    return out


def gene_moments(matrix: sparse.csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    n = float(matrix.shape[1])
    sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float64)
    sq = matrix.copy()
    sq.data = sq.data.astype(np.float64, copy=False) ** 2
    sumsq = np.asarray(sq.sum(axis=1)).ravel().astype(np.float64)
    mean = sums / max(n, 1.0)
    var = np.maximum(sumsq / max(n, 1.0) - mean * mean, 0.0)
    return mean, var


def index_manifest(cell_index_path: Path, manifest_path: Path, n_cells: int) -> list[dict[str, str]]:
    manifest_by_cell = {row["cell"]: row for row in read_csv(manifest_path)}
    rows: list[dict[str, str]] = []
    for row in read_csv(cell_index_path):
        col = int(row["expression_col_index"])
        if col >= n_cells:
            raise ValueError(f"cell index {col} exceeds count matrix columns {n_cells}")
        meta = manifest_by_cell.get(row["cell"])
        if meta is None:
            raise ValueError(f"cell {row['cell']} missing from matched manifest")
        rows.append({**meta, "expression_col_index": str(col)})
    rows.sort(key=lambda r: int(r["expression_col_index"]))
    if len(rows) != n_cells:
        raise ValueError(f"manifest/index has {len(rows)} cells but matrix has {n_cells}")
    return rows


def group_between_fraction(
    matrix: sparse.csc_matrix,
    labels: list[str],
    total_mean: np.ndarray,
    total_var: np.ndarray,
    selected: np.ndarray,
) -> float:
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label = str(label or "unknown")
        by_label[label].append(idx)
    if len(by_label) < 2:
        return 0.0
    between = np.zeros(matrix.shape[0], dtype=np.float64)
    n_total = float(matrix.shape[1])
    for cols in by_label.values():
        if not cols:
            continue
        sums = np.asarray(matrix[:, cols].sum(axis=1)).ravel().astype(np.float64)
        mu = sums / float(len(cols))
        between += float(len(cols)) * (mu - total_mean) ** 2
    between_var = between / max(n_total, 1.0)
    den = float(total_var[selected].sum())
    return float(between_var[selected].sum() / den) if den > 1e-12 else 0.0


def mean_for_cols(matrix: sparse.csc_matrix, cols: list[int]) -> np.ndarray:
    if not cols:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    return np.asarray(matrix[:, cols].mean(axis=1)).ravel().astype(np.float64)


def response_energy_rows(
    matrix: sparse.csc_matrix,
    manifest_rows: list[dict[str, str]],
    ranked_genes: np.ndarray,
    topks: list[int],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    row_meta: dict[str, dict[str, str]] = {}
    for idx, meta in enumerate(manifest_rows):
        row_id = meta.get("row_id", "")
        role = meta.get("selection_role", "")
        grouped[row_id][role].append(idx)
        row_meta.setdefault(row_id, meta)
    out: list[dict[str, Any]] = []
    for row_id, role_cols in sorted(grouped.items()):
        perturb = role_cols.get("perturb", [])
        control = role_cols.get("control", [])
        if not perturb or not control:
            continue
        diff = mean_for_cols(matrix, perturb) - mean_for_cols(matrix, control)
        energy = diff * diff
        total = float(energy.sum())
        meta = row_meta[row_id]
        row: dict[str, Any] = {
            "row_id": row_id,
            "audit_role": meta.get("audit_role", ""),
            "lineage": meta.get("manifest_cell_type_broad", meta.get("cell_type_broad", "")),
            "target": meta.get("manifest_gene_target", meta.get("gene_target", "")),
            "timepoint": meta.get("manifest_timepoint", meta.get("timepoint", "")),
            "n_perturb_cells": len(perturb),
            "n_control_cells": len(control),
            "response_energy_total": total,
        }
        csum = np.cumsum(energy[ranked_genes])
        for k in topks:
            kk = min(k, len(ranked_genes))
            row[f"hvg{k}_response_energy_share"] = float(csum[kk - 1] / total) if total > 1e-12 and kk > 0 else 0.0
        out.append(row)
    return out


def summarize_response(rows: list[dict[str, Any]], topks: list[int]) -> list[dict[str, Any]]:
    groups = {
        "all_rows": rows,
        "primary_rows": [r for r in rows if r["audit_role"] == "primary_mechanism_test"],
        "primary_periderm": [
            r for r in rows if r["audit_role"] == "primary_mechanism_test" and r["lineage"] == "periderm"
        ],
        "primary_mature_fast_muscle": [
            r for r in rows if r["audit_role"] == "primary_mechanism_test" and r["lineage"] == "mature fast muscle"
        ],
        "secondary_or_control_rows": [r for r in rows if r["audit_role"] != "primary_mechanism_test"],
    }
    out: list[dict[str, Any]] = []
    for name, part in groups.items():
        if not part:
            continue
        row: dict[str, Any] = {"subset": name, "n_rows": len(part)}
        for k in topks:
            vals = [float(r[f"hvg{k}_response_energy_share"]) for r in part]
            row[f"hvg{k}_response_energy_share_mean"] = float(np.mean(vals))
            row[f"hvg{k}_response_energy_share_min"] = float(np.min(vals))
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gene-names", type=Path, default=DEFAULT_GENES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = load_csc(args.counts_npz)
    manifest = index_manifest(args.cell_index, args.matched_manifest, counts.shape[1])
    genes = [line.strip() for line in args.gene_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(genes) != counts.shape[0]:
        raise ValueError(f"gene file has {len(genes)} genes but matrix has {counts.shape[0]}")

    lognorm = lognorm_counts(counts)
    detection = np.asarray(counts.getnnz(axis=1)).ravel().astype(np.float64) / float(counts.shape[1])
    raw_total = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    log_mean, log_var = gene_moments(lognorm)
    ranked = np.argsort(-log_var, kind="mergesort")
    topks = sorted({k for k in (500, 1000, 2000, 4000, 8000, 16000, counts.shape[0]) if 0 < k <= counts.shape[0]})

    label_specs = {
        "cell_type_broad": [r.get("cell_type_broad", "") for r in manifest],
        "cell_type_sub": [r.get("cell_type_sub", "") for r in manifest],
        "target": [r.get("manifest_gene_target", r.get("gene_target", "")) for r in manifest],
        "timepoint": [r.get("manifest_timepoint", r.get("timepoint", "")) for r in manifest],
        "row_id": [r.get("row_id", "") for r in manifest],
        "selection_role": [r.get("selection_role", "") for r in manifest],
    }

    curve_rows: list[dict[str, Any]] = []
    total_var_sum = float(log_var.sum())
    for k in topks:
        selected = ranked[:k]
        row: dict[str, Any] = {
            "top_genes": k,
            "gene_fraction": float(k / counts.shape[0]),
            "lognorm_variance_share": float(log_var[selected].sum() / total_var_sum) if total_var_sum > 1e-12 else 0.0,
            "raw_count_share": float(raw_total[selected].sum() / raw_total.sum()) if raw_total.sum() > 0 else 0.0,
            "mean_detection_fraction": float(detection[selected].mean()),
            "median_detection_fraction": float(np.median(detection[selected])),
        }
        for name, labels in label_specs.items():
            row[f"{name}_between_fraction"] = group_between_fraction(lognorm, labels, log_mean, log_var, selected)
        curve_rows.append(row)

    response_rows = response_energy_rows(lognorm, manifest, ranked, topks)
    response_summary = summarize_response(response_rows, topks)
    summary_by_subset = {r["subset"]: r for r in response_summary}
    for row in curve_rows:
        k = int(row["top_genes"])
        for subset, srow in summary_by_subset.items():
            row[f"{subset}_response_energy_share_mean"] = srow.get(f"hvg{k}_response_energy_share_mean", "")
            row[f"{subset}_response_energy_share_min"] = srow.get(f"hvg{k}_response_energy_share_min", "")

    top_gene_rows = [
        {
            "rank": rank + 1,
            "gene_id": genes[idx],
            "lognorm_mean": float(log_mean[idx]),
            "lognorm_variance": float(log_var[idx]),
            "detection_fraction": float(detection[idx]),
            "raw_total_count": float(raw_total[idx]),
        }
        for rank, idx in enumerate(ranked[:2000])
    ]

    curve_csv = args.out_dir / "zscape_hvg_fullgene_information_curve.csv"
    response_csv = args.out_dir / "zscape_hvg_response_energy_rows.csv"
    response_summary_csv = args.out_dir / "zscape_hvg_response_energy_summary.csv"
    top_genes_csv = args.out_dir / "zscape_hvg_top_gene_metrics.csv"
    json_path = args.out_dir / "zscape_hvg_fullgene_information_axis_20260628.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_HVG_FULLGENE_INFORMATION_AXIS_20260628.md"

    curve_fields = list(curve_rows[0])
    write_csv(curve_csv, curve_rows, curve_fields)
    response_fields = list(response_rows[0]) if response_rows else ["row_id"]
    write_csv(response_csv, response_rows, response_fields)
    response_summary_fields = list(response_summary[0]) if response_summary else ["subset"]
    write_csv(response_summary_csv, response_summary, response_summary_fields)
    write_csv(top_genes_csv, top_gene_rows, list(top_gene_rows[0]))

    status = "zscape_hvg_fullgene_information_preflight_no_gpu"
    result = {
        "status": status,
        "gpu_authorized": False,
        "counts_npz": str(args.counts_npz),
        "matched_manifest": str(args.matched_manifest),
        "n_genes": int(counts.shape[0]),
        "n_cells": int(counts.shape[1]),
        "topks": topks,
        "curve_csv": str(curve_csv),
        "response_rows_csv": str(response_csv),
        "response_summary_csv": str(response_summary_csv),
        "top_genes_csv": str(top_genes_csv),
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    selected_rows = [r for r in curve_rows if int(r["top_genes"]) in {1000, 2000, 4000, 8000, counts.shape[0]}]
    lines = [
        "# LatentFM ZSCAPE HVG/Full-Gene Information Axis Preflight",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only preflight over the validated ZSCAPE selected-cell raw-count matrix.",
        "- Does not train, infer, read scFM embeddings, read canonical multi, or read Track C query.",
        "- This tests candidate scaling x-variables; it is not a scaling-law proof and not a LatentFM promotion gate.",
        "",
        "## Question",
        "",
        "The working hypothesis is that downstream perturbation scaling should use",
        "information content rather than raw dataset size. This preflight asks whether",
        "ZSCAPE biological/state and perturbation-response information is captured by",
        "a compact HVG-like subspace or whether full-gene dimensionality contributes",
        "substantial additional signal.",
        "",
        "## Inputs",
        "",
        f"- counts: `{args.counts_npz}`",
        f"- matched manifest: `{args.matched_manifest}`",
        f"- genes: `{args.gene_names}`",
        f"- matrix: `{counts.shape[0]}` genes x `{counts.shape[1]}` cells",
        "",
        "## HVG/Full-Gene Curve",
        "",
        "| top genes | gene frac | log-var share | raw-count share | cell type broad | subtype | target | time | row | primary response mean | periderm response mean | muscle response mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["top_genes"]),
                    fmt(row["gene_fraction"]),
                    fmt(row["lognorm_variance_share"]),
                    fmt(row["raw_count_share"]),
                    fmt(row["cell_type_broad_between_fraction"]),
                    fmt(row["cell_type_sub_between_fraction"]),
                    fmt(row["target_between_fraction"]),
                    fmt(row["timepoint_between_fraction"]),
                    fmt(row["row_id_between_fraction"]),
                    fmt(row.get("primary_rows_response_energy_share_mean", "")),
                    fmt(row.get("primary_periderm_response_energy_share_mean", "")),
                    fmt(row.get("primary_mature_fast_muscle_response_energy_share_mean", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- HVG concentration can support an information-axis design, not a model claim.",
            "- Full-gene residual signal would motivate a later full-gene/longer-token scaling test only after leakage-safe split design and dual-baseline no-harm rules.",
            "- OT/cluster metrics and HVG/full-gene metrics are separate x-variable candidates; a failure of OT as a training pair mode does not close OT-derived information metrics.",
            "- No GPU launch is authorized from this preflight alone.",
            "",
            "## Outputs",
            "",
            f"- curve: `{curve_csv}`",
            f"- response rows: `{response_csv}`",
            f"- response summary: `{response_summary_csv}`",
            f"- top genes: `{top_genes_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
