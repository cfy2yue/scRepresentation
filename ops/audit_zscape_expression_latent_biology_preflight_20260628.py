#!/usr/bin/env python3
"""ZSCAPE expression/latent biology preflight.

CPU-only analysis of selected ZSCAPE raw counts:
- QC-reporting and optional QC filtering;
- size-factor log1p expression space differential response;
- top-gene lists suitable for later zebrafish pathway enrichment;
- control-only SVD/PCA latent and metadata UMAP3D latent response alignment.

This is a biological-analysis preflight, not a model-training or claim gate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_COUNTS = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_counts_csc.npz"
DEFAULT_CELL_INDEX = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_expression_cell_index.csv"
DEFAULT_MANIFEST = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_expression_selected_cell_ids_matched.csv"
DEFAULT_GENES = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs/zscape_manifest_selected_gene_names.txt"
DEFAULT_GENE_META = ROOT / "dataset/external/zscape_20260628/GSE202639_zperturb_full_gene_metadata.csv.gz"
DEFAULT_OUT = ROOT / "reports/zscape_expression_latent_biology_preflight_20260628"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_csc(path: Path) -> sparse.csc_matrix:
    obj = np.load(path)
    shape = tuple(int(x) for x in obj["shape"])
    return sparse.csc_matrix((obj["data"], obj["indices"], obj["indptr"]), shape=shape)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_gene_metadata(path: Path) -> dict[str, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict[str, str] = {}
    with opener(path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gid = str(row.get("id", "")).strip()
            symbol = str(row.get("gene_short_name", "")).strip()
            if gid:
                out[gid] = symbol or gid
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def index_manifest(cell_index_path: Path, manifest_path: Path, n_cells: int) -> list[dict[str, str]]:
    manifest_rows = read_csv(manifest_path)
    # The selected manifest can contain repeated cells in different row contexts.
    # Preserve order through the extracted expression column index.
    by_cell_queue: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        by_cell_queue[row["cell"]].append(row)
    rows: list[dict[str, str]] = []
    for row in read_csv(cell_index_path):
        col = int(row["expression_col_index"])
        if col >= n_cells:
            raise ValueError(f"column {col} outside matrix width {n_cells}")
        queue = by_cell_queue.get(row["cell"], [])
        if queue:
            meta = queue.pop(0)
        else:
            matches = [r for r in manifest_rows if r["cell"] == row["cell"]]
            if not matches:
                raise ValueError(f"cell {row['cell']} missing from manifest")
            meta = matches[0]
        rows.append({**meta, "expression_col_index": str(col)})
    rows.sort(key=lambda r: int(r["expression_col_index"]))
    if len(rows) != n_cells:
        raise ValueError(f"indexed manifest has {len(rows)} rows but matrix has {n_cells} cells")
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def qc_pass(row: dict[str, str], min_umi: float, min_genes: float) -> bool:
    return as_float(row, "n_umi") >= min_umi and as_float(row, "num_genes_expressed") >= min_genes


def lognorm_counts(counts: sparse.csc_matrix, keep_cols: np.ndarray) -> sparse.csc_matrix:
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
    np.log1p(sub.data, out=sub.data)
    return sub


def gene_moments(matrix: sparse.csc_matrix, cols: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not cols:
        z = np.zeros(matrix.shape[0], dtype=np.float64)
        return z, z, z
    sub = matrix[:, cols]
    n = float(sub.shape[1])
    sums = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
    sq = sub.copy()
    sq.data = sq.data.astype(np.float64, copy=False) ** 2
    sumsq = np.asarray(sq.sum(axis=1)).ravel().astype(np.float64)
    mean = sums / n
    var = np.maximum(sumsq / n - mean * mean, 0.0)
    det = np.asarray(sub.getnnz(axis=1)).ravel().astype(np.float64) / n
    return mean, var, det


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def build_row_groups(manifest: list[dict[str, str]], old_to_new: dict[int, int]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for old_idx, meta in enumerate(manifest):
        new_idx = old_to_new.get(old_idx)
        if new_idx is None:
            continue
        row_id = meta["row_id"]
        obj = groups.setdefault(
            row_id,
            {
                "row_id": row_id,
                "audit_role": meta.get("audit_role", ""),
                "lineage": meta.get("manifest_cell_type_broad") or meta.get("cell_type_broad", ""),
                "target": meta.get("manifest_gene_target") or meta.get("gene_target", ""),
                "timepoint": meta.get("manifest_timepoint") or meta.get("timepoint", ""),
                "perturb": [],
                "control": [],
            },
        )
        role = meta.get("selection_role", "")
        if role in ("perturb", "control"):
            obj[role].append(new_idx)
    return groups


def top_de_for_row(
    matrix: sparse.csc_matrix,
    group: dict[str, Any],
    gene_ids: list[str],
    gene_symbols: list[str],
    top_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[str]]:
    pcols = group["perturb"]
    ccols = group["control"]
    pm, pv, pd = gene_moments(matrix, pcols)
    cm, cv, cd = gene_moments(matrix, ccols)
    diff = pm - cm
    se = np.sqrt(pv / max(len(pcols), 1) + cv / max(len(ccols), 1) + 1e-8)
    z = diff / se
    abs_order = np.argsort(-np.abs(z), kind="mergesort")[:top_n]
    up_order = np.argsort(-z, kind="mergesort")[:top_n]
    down_order = np.argsort(z, kind="mergesort")[:top_n]

    rows = []
    for rank, idx in enumerate(abs_order, start=1):
        rows.append(
            {
                "row_id": group["row_id"],
                "rank_abs_z": rank,
                "gene_id": gene_ids[idx],
                "gene_symbol": gene_symbols[idx],
                "lognorm_mean_perturb": float(pm[idx]),
                "lognorm_mean_control": float(cm[idx]),
                "lognorm_diff": float(diff[idx]),
                "welch_z_proxy": float(z[idx]),
                "detect_perturb": float(pd[idx]),
                "detect_control": float(cd[idx]),
                "detect_diff": float(pd[idx] - cd[idx]),
            }
        )
    up_symbols = [gene_symbols[i] for i in up_order if gene_symbols[i]]
    down_symbols = [gene_symbols[i] for i in down_order if gene_symbols[i]]
    energy_total = float(np.sum(diff * diff))
    summary = {
        "row_id": group["row_id"],
        "audit_role": group["audit_role"],
        "lineage": group["lineage"],
        "target": group["target"],
        "timepoint": group["timepoint"],
        "n_perturb": len(pcols),
        "n_control": len(ccols),
        "response_energy_l2": energy_total,
        "top_up_genes": ";".join(up_symbols[:12]),
        "top_down_genes": ";".join(down_symbols[:12]),
        "top_abs_genes": ";".join([gene_symbols[i] for i in abs_order[:12] if gene_symbols[i]]),
        "target_symbol_in_top_abs50": str(group["target"]).lower() in {s.lower() for s in [gene_symbols[i] for i in abs_order[:50]]},
    }
    return summary, rows, up_symbols, down_symbols


def fit_svd_latent(matrix: sparse.csc_matrix, hvg_rank: np.ndarray, n_hvg: int, n_components: int) -> np.ndarray:
    genes = hvg_rank[: min(n_hvg, len(hvg_rank))]
    x = matrix[genes, :].T.tocsr()
    n_components = min(n_components, max(2, min(x.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=20260628)
    latent = svd.fit_transform(x)
    # Variance-scale columns for cosine stability.
    latent = latent - latent.mean(axis=0, keepdims=True)
    sd = latent.std(axis=0, keepdims=True)
    latent = latent / np.maximum(sd, 1e-6)
    return latent.astype(np.float32)


def centroid(mat: np.ndarray, cols: list[int]) -> np.ndarray:
    if not cols:
        return np.zeros(mat.shape[1], dtype=np.float64)
    return mat[np.asarray(cols, dtype=int)].mean(axis=0).astype(np.float64)


def latent_alignment_rows(
    latent: np.ndarray,
    groups: dict[str, dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    controls_by_lineage_time: dict[tuple[str, float], list[int]] = defaultdict(list)
    for group in groups.values():
        try:
            t = float(group["timepoint"])
        except ValueError:
            continue
        controls_by_lineage_time[(str(group["lineage"]), t)].extend(group["control"])
    temporal_vecs: dict[str, list[tuple[float, float, np.ndarray]]] = defaultdict(list)
    by_lineage: dict[str, list[float]] = defaultdict(list)
    for lineage, t in controls_by_lineage_time:
        by_lineage[lineage].append(t)
    for lineage, times in by_lineage.items():
        ordered = sorted(set(times))
        for a, b in zip(ordered, ordered[1:]):
            va = centroid(latent, controls_by_lineage_time[(lineage, a)])
            vb = centroid(latent, controls_by_lineage_time[(lineage, b)])
            temporal_vecs[lineage].append((a, b, vb - va))

    out: list[dict[str, Any]] = []
    for group in groups.values():
        response = centroid(latent, group["perturb"]) - centroid(latent, group["control"])
        lineage = str(group["lineage"])
        choices = temporal_vecs.get(lineage, [])
        best_same = 0.0
        best_pair = ""
        if choices:
            vals = [(cosine(response, vec), a, b) for a, b, vec in choices]
            best_same, a, b = max(vals, key=lambda x: x[0])
            best_pair = f"{a:g}->{b:g}"
        wrong = []
        for other_lineage, other_choices in temporal_vecs.items():
            if other_lineage == lineage:
                continue
            wrong.extend(cosine(response, vec) for _a, _b, vec in other_choices)
        wrong_max = max(wrong) if wrong else 0.0
        out.append(
            {
                "row_id": group["row_id"],
                "latent_space": label,
                "lineage": lineage,
                "target": group["target"],
                "timepoint": group["timepoint"],
                "temporal_pair": best_pair,
                "cosine_to_lineage_time_vector": best_same,
                "max_cosine_to_wrong_lineage_time_vector": wrong_max,
                "cosine_margin_vs_wrong_lineage": best_same - wrong_max,
                "response_norm": float(np.linalg.norm(response)),
                "alignment_gate": best_same >= 0.25 and best_same - wrong_max >= 0.10,
            }
        )
    return out


def umap_latent_from_manifest(manifest: list[dict[str, str]], keep_cols: np.ndarray) -> np.ndarray:
    rows = [manifest[int(i)] for i in keep_cols]
    arr = np.asarray(
        [
            [as_float(row, "umap3d_1"), as_float(row, "umap3d_2"), as_float(row, "umap3d_3")]
            for row in rows
        ],
        dtype=np.float32,
    )
    arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    arr = np.nan_to_num(arr)
    sd = arr.std(axis=0, keepdims=True)
    return arr / np.maximum(sd, 1e-6)


def write_gene_lists(out_dir: Path, row_id: str, up: list[str], down: list[str], n: int) -> None:
    safe = safe_name(row_id)
    gene_dir = out_dir / "gprofiler_input_gene_lists"
    gene_dir.mkdir(parents=True, exist_ok=True)
    (gene_dir / f"{safe}.top{n}.up.txt").write_text("\n".join(up[:n]) + "\n", encoding="utf-8")
    (gene_dir / f"{safe}.top{n}.down.txt").write_text("\n".join(down[:n]) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-npz", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--cell-index", type=Path, default=DEFAULT_CELL_INDEX)
    parser.add_argument("--matched-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gene-names", type=Path, default=DEFAULT_GENES)
    parser.add_argument("--gene-metadata", type=Path, default=DEFAULT_GENE_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-umi", type=float, default=100.0)
    parser.add_argument("--min-genes", type=float, default=100.0)
    parser.add_argument("--apply-qc", action="store_true")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--latent-hvg", type=int, default=2000)
    parser.add_argument("--latent-dim", type=int, default=32)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = load_csc(args.counts_npz)
    manifest = index_manifest(args.cell_index, args.matched_manifest, counts.shape[1])
    gene_ids = [line.strip() for line in args.gene_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    symbol_map = read_gene_metadata(args.gene_metadata)
    gene_symbols = [symbol_map.get(gid, gid) for gid in gene_ids]
    if len(gene_ids) != counts.shape[0]:
        raise ValueError(f"gene list has {len(gene_ids)} rows but matrix has {counts.shape[0]}")

    qc_flags = np.asarray([qc_pass(row, args.min_umi, args.min_genes) for row in manifest], dtype=bool)
    keep_old = np.where(qc_flags if args.apply_qc else np.ones(len(manifest), dtype=bool))[0]
    old_to_new = {int(old): int(new) for new, old in enumerate(keep_old)}
    filtered_manifest = [manifest[int(i)] for i in keep_old]
    lognorm = lognorm_counts(counts, keep_old)
    all_mean, all_var, _det = gene_moments(lognorm, list(range(lognorm.shape[1])))
    hvg_rank = np.argsort(-all_var, kind="mergesort")
    groups = build_row_groups(manifest, old_to_new)

    row_summaries: list[dict[str, Any]] = []
    de_rows: list[dict[str, Any]] = []
    for group in groups.values():
        if not group["perturb"] or not group["control"]:
            continue
        summary, rows, up, down = top_de_for_row(lognorm, group, gene_ids, gene_symbols, args.top_n)
        row_summaries.append(summary)
        de_rows.extend(rows)
        write_gene_lists(args.out_dir, group["row_id"], up, down, args.top_n)

    svd_latent = fit_svd_latent(lognorm, hvg_rank, args.latent_hvg, args.latent_dim)
    umap_latent = umap_latent_from_manifest(manifest, keep_old)
    latent_rows = latent_alignment_rows(svd_latent, groups, "log1p_hvg_svd")
    latent_rows.extend(latent_alignment_rows(umap_latent, groups, "zscape_metadata_umap3d"))

    qc_by_row: list[dict[str, Any]] = []
    grouped_old: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(manifest):
        grouped_old[row["row_id"]].append(idx)
    for row_id, idxs in sorted(grouped_old.items()):
        vals_umi = [as_float(manifest[i], "n_umi") for i in idxs]
        vals_gene = [as_float(manifest[i], "num_genes_expressed") for i in idxs]
        role = manifest[idxs[0]].get("audit_role", "")
        lineage = manifest[idxs[0]].get("manifest_cell_type_broad", manifest[idxs[0]].get("cell_type_broad", ""))
        qc_by_row.append(
            {
                "row_id": row_id,
                "audit_role": role,
                "lineage": lineage,
                "n_cells_before_qc": len(idxs),
                "n_cells_after_qc_rule": int(qc_flags[idxs].sum()),
                "qc_fail_fraction": float(1.0 - qc_flags[idxs].mean()),
                "median_n_umi": float(np.nanmedian(vals_umi)),
                "median_num_genes_expressed": float(np.nanmedian(vals_gene)),
            }
        )

    summary_csv = args.out_dir / "zscape_expression_de_row_summary.csv"
    de_csv = args.out_dir / "zscape_expression_de_top_genes.csv"
    latent_csv = args.out_dir / "zscape_latent_alignment_rows.csv"
    qc_csv = args.out_dir / "zscape_qc_row_summary.csv"
    json_path = args.out_dir / "zscape_expression_latent_biology_preflight_20260628.json"
    md_path = args.out_dir / "LATENTFM_ZSCAPE_EXPRESSION_LATENT_BIOLOGY_PREFLIGHT_20260628.md"

    write_csv(summary_csv, row_summaries, list(row_summaries[0]))
    write_csv(de_csv, de_rows, list(de_rows[0]))
    write_csv(latent_csv, latent_rows, list(latent_rows[0]))
    write_csv(qc_csv, qc_by_row, list(qc_by_row[0]))

    primary = [r for r in row_summaries if r["audit_role"] == "primary_mechanism_test"]
    periderm = [r for r in primary if r["lineage"] == "periderm"]
    latent_primary = [r for r in latent_rows if any(p["row_id"] == r["row_id"] for p in primary)]

    result = {
        "status": "zscape_expression_latent_biology_preflight_no_gpu",
        "gpu_authorized": False,
        "qc_filter_applied": bool(args.apply_qc),
        "qc_rule": {"min_umi": args.min_umi, "min_genes": args.min_genes},
        "n_cells_before_qc": len(manifest),
        "n_cells_used": len(filtered_manifest),
        "n_genes": len(gene_ids),
        "log1p_policy": "size-factor normalize to median selected-cell library then log1p exactly once",
        "formal_pathway_enrichment_status": "not_run_gene_lists_written",
        "scfm_latent_status": "not_available_in_current_zscape_artifacts",
        "cpu_latent_proxies": ["log1p_hvg_svd", "zscape_metadata_umap3d"],
        "outputs": {
            "row_summary_csv": str(summary_csv),
            "top_gene_csv": str(de_csv),
            "latent_alignment_csv": str(latent_csv),
            "qc_csv": str(qc_csv),
            "gprofiler_gene_lists_dir": str(args.out_dir / "gprofiler_input_gene_lists"),
        },
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Expression/Latent Biology Preflight",
        "",
        f"Timestamp: `{now_cst()}`",
        "",
        "Status: `zscape_expression_latent_biology_preflight_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only analysis of validated selected ZSCAPE raw counts.",
        "- No training, no scFM embedding extraction, no canonical multi, and no Track C query.",
        "- Expression is size-factor normalized to the median selected-cell library and `log1p` transformed exactly once.",
        "- QC is reported by default; filtering is applied only if `--apply-qc` is set.",
        "",
        "## QC And Log1p Policy",
        "",
        f"- QC rule audited: `n_umi >= {args.min_umi:g}` and `num_genes_expressed >= {args.min_genes:g}`.",
        f"- QC filtering applied in this run: `{bool(args.apply_qc)}`.",
        f"- Cells before QC: `{len(manifest)}`; cells used: `{len(filtered_manifest)}`.",
        "- Formal pathway enrichment is not run here; ranked gene lists are written for a frozen zebrafish enrichment step.",
        "",
        "## Primary Expression Responses",
        "",
        "| lineage | target | time | n perturb | n control | response L2 | top up genes | top down genes | target in top abs50 |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in primary:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["lineage"],
                    row["target"],
                    str(row["timepoint"]),
                    str(row["n_perturb"]),
                    str(row["n_control"]),
                    fmt(row["response_energy_l2"]),
                    row["top_up_genes"],
                    row["top_down_genes"],
                    str(row["target_symbol_in_top_abs50"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Primary Latent Alignment",
            "",
            "| space | lineage | target | time | pair | cosine | wrong max | margin | gate |",
            "|---|---|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in latent_primary:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["latent_space"],
                    row["lineage"],
                    row["target"],
                    str(row["timepoint"]),
                    row["temporal_pair"],
                    fmt(row["cosine_to_lineage_time_vector"]),
                    fmt(row["max_cosine_to_wrong_lineage_time_vector"]),
                    fmt(row["cosine_margin_vs_wrong_lineage"]),
                    str(row["alignment_gate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Formal Enrichment Next Step",
            "",
            "The current gene identifiers are ZSCAPE zebrafish Ensembl IDs with",
            "`gene_short_name` symbols from the ZSCAPE gene metadata. Formal pathway",
            "analysis should run in a frozen enrichment environment against a recorded",
            "Danio rerio database snapshot. This report writes top-gene lists but does",
            "not treat symbol-level enrichment as evidence yet.",
            "",
            "Recommended frozen enrichment step:",
            "",
            "```bash",
            "python -m venv /data/cyx/1030/scLatent/.venvs/zscape_bio_20260628",
            "source /data/cyx/1030/scLatent/.venvs/zscape_bio_20260628/bin/activate",
            "pip install gprofiler-official==1.0.0 gseapy==1.1.9",
            "```",
            "",
            "Before using any enrichment result, record database version/date, mapping",
            "rate, query size, background universe, and whether terms survive FDR.",
            "",
            "## Outputs",
            "",
            f"- row summary: `{summary_csv}`",
            f"- top genes: `{de_csv}`",
            f"- latent alignment: `{latent_csv}`",
            f"- QC summary: `{qc_csv}`",
            f"- gene lists: `{args.out_dir / 'gprofiler_input_gene_lists'}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
