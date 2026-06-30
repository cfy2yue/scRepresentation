#!/usr/bin/env python3
"""Prepare a species-safe UCE Danio latent smoke input for ZSCAPE.

This is a CPU-only gate. It verifies that local UCE assets are compatible with
ZSCAPE zebrafish gene symbols, freezes the expression preprocessing contract,
and writes a small 128-cell h5ad for a later detached GPU embedding smoke.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_EXTRACTION = ROOT / "runs/zscape_raw_counts_cell_manifest_extraction_20260628/zscape_raw_counts_cell_manifest_extraction_20260628_074523/outputs"
DEFAULT_GENE_META = ROOT / "dataset/external/zscape_20260628/GSE202639_zperturb_full_gene_metadata.csv.gz"
DEFAULT_UCE = ROOT / "scFM_pretrained/uce/model_files"
DEFAULT_OUT = ROOT / "reports/zscape_uce_danio_latent_gate_20260628"
PRIMARY_ROWS = ["periderm__noto__24p0h", "periderm__smo__24p0h"]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def read_gene_meta(path: Path) -> dict[str, dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict[str, dict[str, str]] = {}
    with opener(path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gid = str(row.get("id", "")).strip()
            if gid:
                out[gid] = {k: str(v) for k, v in row.items()}
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_offsets(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    return dict(obj) if isinstance(obj, dict) else {}


def choose_ranked_unique(sub: pd.DataFrame, used_cells: set[str], cells_per_role: int) -> pd.DataFrame:
    sub = sub.sort_values(["selection_rank", "cell"])
    unique_sub = sub[~sub["cell"].astype(str).isin(used_cells)]
    if len(unique_sub) < cells_per_role:
        raise RuntimeError(f"not enough unique cells: {len(unique_sub)} < {cells_per_role}")
    return unique_sub.head(cells_per_role).copy()


def choose_embryo_roundrobin(sub: pd.DataFrame, used_cells: set[str], cells_per_role: int) -> pd.DataFrame:
    sub = sub.sort_values(["embryo", "selection_rank", "cell"])
    sub = sub[~sub["cell"].astype(str).isin(used_cells)].copy()
    if len(sub) < cells_per_role:
        raise RuntimeError(f"not enough unique cells: {len(sub)} < {cells_per_role}")
    groups = {
        str(embryo): group.sort_values(["selection_rank", "cell"]).reset_index(drop=True)
        for embryo, group in sub.groupby("embryo", sort=True)
    }
    chosen = []
    pos = {embryo: 0 for embryo in groups}
    while len(chosen) < cells_per_role:
        progressed = False
        for embryo in sorted(groups):
            group = groups[embryo]
            j = pos[embryo]
            if j >= len(group):
                continue
            chosen.append(group.iloc[j])
            pos[embryo] = j + 1
            progressed = True
            if len(chosen) >= cells_per_role:
                break
        if not progressed:
            break
    if len(chosen) < cells_per_role:
        raise RuntimeError(f"round-robin exhausted at {len(chosen)} < {cells_per_role}")
    return pd.DataFrame(chosen).reset_index(drop=True)


def select_smoke_cells(manifest: pd.DataFrame, cells_per_role: int, strategy: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    used_cells: set[str] = set()
    for row_id in PRIMARY_ROWS:
        for role in ["control", "perturb"]:
            sub = manifest[(manifest["row_id"] == row_id) & (manifest["selection_role"] == role)]
            if strategy == "rank":
                chosen = choose_ranked_unique(sub, used_cells, cells_per_role)
            elif strategy == "embryo_roundrobin":
                chosen = choose_embryo_roundrobin(sub, used_cells, cells_per_role)
            else:
                raise ValueError(f"unknown selection strategy: {strategy}")
            used_cells.update(chosen["cell"].astype(str))
            pieces.append(chosen)
    out = pd.concat(pieces, ignore_index=True)
    return out.reset_index(drop=True)


def build_gene_symbol_matrix(
    counts_genes_by_cells: sp.spmatrix,
    gene_ids: list[str],
    gene_meta: dict[str, dict[str, str]],
    selected_cols: np.ndarray,
    uce_vocab_upper: set[str],
    chrom_upper: set[str],
) -> tuple[sp.csr_matrix, pd.DataFrame, dict[str, Any]]:
    x_cells_by_genes = counts_genes_by_cells[:, selected_cols].T.tocsr()

    symbol_upper_by_gene: list[str] = []
    symbol_original: dict[str, str] = {}
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    missing_symbol = 0
    for gid in gene_ids:
        sym = gene_meta.get(gid, {}).get("gene_short_name", "").strip()
        if not sym:
            missing_symbol += 1
            sym = gid
        sup = sym.upper()
        symbol_upper_by_gene.append(sup)
        symbol_original.setdefault(sup, sym)
        grouped_ids[sup].append(gid)

    symbols = sorted(grouped_ids)
    symbol_index = {sym: i for i, sym in enumerate(symbols)}
    row_idx = np.arange(len(gene_ids), dtype=np.int64)
    col_idx = np.array([symbol_index[s] for s in symbol_upper_by_gene], dtype=np.int64)
    agg = sp.csr_matrix(
        (np.ones(len(gene_ids), dtype=np.float32), (row_idx, col_idx)),
        shape=(len(gene_ids), len(symbols)),
    )
    x_symbol = (x_cells_by_genes @ agg).tocsr()

    var_rows = []
    for sym in symbols:
        ids = grouped_ids[sym]
        meta = gene_meta.get(ids[0], {})
        var_rows.append(
            {
                "symbol_upper": sym,
                "gene_short_name": symbol_original[sym],
                "ensembl_ids": ";".join(ids[:20]) + (";..." if len(ids) > 20 else ""),
                "n_ensdarg_collapsed": len(ids),
                "chromosome_first": meta.get("chromosome", ""),
                "uce_valid": sym in uce_vocab_upper,
                "uce_chrom_known": sym in chrom_upper,
            }
        )
    var = pd.DataFrame(var_rows, index=[symbol_original[s] for s in symbols])
    if var.index.has_duplicates:
        # anndata requires unique var_names. Keep UCE matching via symbol_upper.
        var.index = pd.Index(symbols, name="symbol_upper")
    summary = {
        "input_gene_ids": len(gene_ids),
        "missing_symbol_count": missing_symbol,
        "unique_symbols": len(symbols),
        "collapsed_duplicate_symbol_groups": int(sum(len(v) > 1 for v in grouped_ids.values())),
        "max_ids_per_symbol": int(max(len(v) for v in grouped_ids.values())),
        "uce_valid_symbols": int(sum(s in uce_vocab_upper for s in symbols)),
        "uce_chrom_known_symbols": int(sum(s in chrom_upper for s in symbols)),
    }
    return x_symbol, var, summary


def normalize_log1p(x_counts: sp.csr_matrix, target_sum: float) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    lib = np.asarray(x_counts.sum(axis=1)).ravel().astype(np.float64)
    scale = np.divide(target_sum, lib, out=np.zeros_like(lib), where=lib > 0)
    x_norm = x_counts.astype(np.float32).multiply(scale[:, None]).tocsr()
    x_norm.data = np.log1p(x_norm.data)
    return x_norm.astype(np.float32), lib, scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path, default=DEFAULT_EXTRACTION)
    parser.add_argument("--gene-meta", type=Path, default=DEFAULT_GENE_META)
    parser.add_argument("--uce-root", type=Path, default=DEFAULT_UCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cells-per-role", type=int, default=32)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--selection-strategy", choices=["rank", "embryo_roundrobin"], default="rank")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = args.extraction_dir / "zscape_manifest_selected_counts_csc.npz"
    genes_path = args.extraction_dir / "zscape_manifest_selected_gene_names.txt"
    manifest_path = args.extraction_dir / "zscape_expression_selected_cell_ids_matched.csv"
    cell_index_path = args.extraction_dir / "zscape_manifest_selected_expression_cell_index.csv"

    model_path = args.uce_root / "33layer_model.torch"
    token_path = args.uce_root / "all_tokens.torch"
    offset_path = args.uce_root / "species_offsets.pkl"
    chrom_path = args.uce_root / "species_chrom.csv"
    pe_path = args.uce_root / "protein_embeddings/Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt"

    gene_ids = read_lines(genes_path)
    gene_meta = read_gene_meta(args.gene_meta)
    manifest = pd.read_csv(manifest_path)
    cell_index = pd.read_csv(cell_index_path)
    smoke_manifest = select_smoke_cells(manifest, args.cells_per_role, args.selection_strategy)
    smoke_manifest = smoke_manifest.merge(cell_index, on="cell", how="left", validate="one_to_one")
    if smoke_manifest["expression_col_index"].isna().any():
        raise RuntimeError("selected smoke cells are missing expression_col_index")

    offsets = load_offsets(offset_path)
    pe_dict = torch.load(str(pe_path), map_location="cpu", weights_only=False) if pe_path.is_file() else {}
    pe_vocab_upper = {str(k).upper() for k in getattr(pe_dict, "keys", lambda: [])()}
    del pe_dict
    chrom_df = pd.read_csv(chrom_path) if chrom_path.is_file() else pd.DataFrame()
    z_chrom = chrom_df[chrom_df["species"].astype(str) == "zebrafish"] if "species" in chrom_df.columns else pd.DataFrame()
    chrom_upper = {str(x).upper() for x in z_chrom.get("gene_symbol", [])}

    z_symbols_upper = {
        gene_meta.get(g, {}).get("gene_short_name", "").strip().upper()
        for g in gene_ids
        if gene_meta.get(g, {}).get("gene_short_name", "").strip()
    }
    coverage = {
        "zscape_unique_symbols": len(z_symbols_upper),
        "uce_pe_vocab_symbols": len(pe_vocab_upper),
        "uce_pe_overlap_symbols": len(z_symbols_upper & pe_vocab_upper),
        "uce_pe_overlap_fraction": round(len(z_symbols_upper & pe_vocab_upper) / max(1, len(z_symbols_upper)), 6),
        "uce_chrom_symbols": len(chrom_upper),
        "uce_chrom_overlap_symbols": len(z_symbols_upper & chrom_upper),
        "uce_chrom_overlap_fraction": round(len(z_symbols_upper & chrom_upper) / max(1, len(z_symbols_upper)), 6),
    }

    counts = sp.load_npz(counts_path)
    selected_cols = smoke_manifest["expression_col_index"].astype(int).to_numpy()
    x_counts, var, matrix_summary = build_gene_symbol_matrix(
        counts, gene_ids, gene_meta, selected_cols, pe_vocab_upper, chrom_upper
    )
    x_log, lib_size, size_factor = normalize_log1p(x_counts, args.target_sum)

    obs = smoke_manifest.copy()
    obs.index = pd.Index(obs["cell"].astype(str), name="cell")
    obs["perturbation_label"] = np.where(obs["selection_role"].eq("control"), "control", obs["gene_target"].astype(str))
    obs["uce_perturbation_forced"] = False
    obs["raw_library_size_from_matrix"] = lib_size
    obs["size_factor_to_1e4"] = size_factor

    adata = ad.AnnData(X=x_log, obs=obs, var=var)
    adata.layers["counts"] = x_counts.astype(np.float32)
    adata.uns["zscape_uce_preprocessing"] = {
        "source_counts_orientation": "genes_by_cells",
        "gene_symbol_collapse": "sum ENSDARG rows with the same gene_short_name upper-case symbol",
        "normalization": f"per-cell size factor to {args.target_sum:g}, then exactly one log1p",
        "qc_policy": "cells inherited from prior selected manifest; n_umi and num_genes_expressed retained in obs for sensitivity",
        "perturbation_forcing": "disabled for this expression-state latent smoke; no obs['perturbation'] and no obsm['pert_var_idx']",
        "primary_rows": PRIMARY_ROWS,
        "selection_strategy": args.selection_strategy,
    }
    h5ad_path = args.out_dir / "zscape_uce_danio_128cell_smoke_input.h5ad"
    adata.write_h5ad(h5ad_path, compression="gzip")

    selected_summary_rows = []
    for (row_id, role), sub in smoke_manifest.groupby(["row_id", "selection_role"], sort=True):
        selected_summary_rows.append(
            {
                "row_id": row_id,
                "selection_role": role,
                "n_cells": len(sub),
                "median_n_umi": float(np.median(sub["n_umi"].astype(float))),
                "median_num_genes_expressed": float(np.median(sub["num_genes_expressed"].astype(float))),
                "n_embryos": int(sub["embryo"].nunique()),
            }
        )
    write_csv(
        args.out_dir / "zscape_uce_danio_smoke_cell_summary.csv",
        selected_summary_rows,
        ["row_id", "selection_role", "n_cells", "median_n_umi", "median_num_genes_expressed", "n_embryos"],
    )
    smoke_manifest.to_csv(args.out_dir / "zscape_uce_danio_smoke_cells.csv", index=False)

    asset_status = {
        "model_exists": model_path.is_file(),
        "token_file_exists": token_path.is_file(),
        "offset_file_exists": offset_path.is_file(),
        "chrom_file_exists": chrom_path.is_file(),
        "danio_pe_exists": pe_path.is_file(),
        "zebrafish_offset_present": "zebrafish" in offsets,
    }
    pass_gate = (
        all(asset_status.values())
        and coverage["uce_pe_overlap_symbols"] >= 15000
        and coverage["uce_chrom_overlap_symbols"] >= 15000
        and int(matrix_summary["uce_valid_symbols"]) >= 15000
        and adata.n_obs == 4 * args.cells_per_role
        and not adata.var_names.has_duplicates
    )
    status = "zscape_uce_danio_latent_gate_pass_prepare_gpu_smoke" if pass_gate else "zscape_uce_danio_latent_gate_blocked_no_gpu"

    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized_next_step": bool(pass_gate),
        "asset_status": asset_status,
        "coverage": coverage,
        "matrix_summary": matrix_summary,
        "smoke_h5ad": str(h5ad_path),
        "selected_cell_summary": selected_summary_rows,
        "selection_strategy": args.selection_strategy,
        "qc_log1p_policy": adata.uns["zscape_uce_preprocessing"],
    }
    (args.out_dir / "zscape_uce_danio_latent_gate_20260628.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    top_roles = Counter(smoke_manifest["row_id"] + "/" + smoke_manifest["selection_role"]).most_common()
    report = args.out_dir / "LATENTFM_ZSCAPE_UCE_DANIO_LATENT_GATE_20260628.md"
    lines = [
        "# LatentFM ZSCAPE UCE Danio Latent Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized next step: `{bool(pass_gate)}`",
        "",
        "## Boundary",
        "",
        "- CPU-only species/resource/preprocessing gate for UCE Danio embeddings.",
        "- Does not train, select LatentFM checkpoints, read canonical multi for selection, or read Track C query.",
        "- A pass only authorizes one bounded ZSCAPE UCE embedding smoke on the frozen 128-cell h5ad.",
        "",
        "## UCE Asset Status",
        "",
        "| asset | present |",
        "|---|---:|",
    ]
    for key, val in asset_status.items():
        lines.append(f"| {key} | `{val}` |")
    lines.extend(
        [
            "",
            "## Danio Coverage",
            "",
            f"- ZSCAPE unique symbols: `{coverage['zscape_unique_symbols']}`",
            f"- UCE PE overlap: `{coverage['uce_pe_overlap_symbols']}` (`{coverage['uce_pe_overlap_fraction']}`)",
            f"- UCE chromosome overlap: `{coverage['uce_chrom_overlap_symbols']}` (`{coverage['uce_chrom_overlap_fraction']}`)",
            f"- unique symbols after ENSDARG collapse: `{matrix_summary['unique_symbols']}`",
            f"- collapsed duplicate-symbol groups: `{matrix_summary['collapsed_duplicate_symbol_groups']}`",
            f"- UCE-valid symbols in smoke matrix: `{matrix_summary['uce_valid_symbols']}`",
            "",
            "## Frozen Smoke Input",
            "",
            f"- h5ad: `{h5ad_path}`",
            f"- shape: `{adata.n_obs} cells x {adata.n_vars} unique gene symbols`",
            f"- selection strategy: `{args.selection_strategy}`",
            "- rows: `periderm__noto__24p0h` and `periderm__smo__24p0h`, 32 control + 32 perturb cells each",
            "- preprocessing: raw counts -> per-cell size factor to 1e4 -> exactly one `log1p`",
            "- counts retained: `layers['counts']`; ENSDARG provenance retained in `var['ensembl_ids']`",
            "- perturbation forcing disabled so latent reflects expression state rather than inserted target tokens",
            "",
            "## Cell QC Summary",
            "",
            "| row/role | cells |",
            "|---|---:|",
        ]
    )
    for key, val in top_roles:
        lines.append(f"| {key} | {val} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if pass_gate:
        lines.extend(
            [
                "- UCE is the first local species-safe candidate for a true ZSCAPE latent embedding smoke.",
                "- Launch exactly one detached 128-cell UCE GPU embedding smoke, then run latent continuity/posthoc before any larger extraction.",
            ]
        )
    else:
        lines.extend(
            [
                "- Do not launch ZSCAPE UCE GPU embedding yet.",
                "- Fix missing UCE assets or coverage/provenance issues first.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{args.out_dir / 'zscape_uce_danio_latent_gate_20260628.json'}`",
            f"- cell summary: `{args.out_dir / 'zscape_uce_danio_smoke_cell_summary.csv'}`",
            f"- cell manifest: `{args.out_dir / 'zscape_uce_danio_smoke_cells.csv'}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "smoke_h5ad": str(h5ad_path)}, indent=2))


if __name__ == "__main__":
    main()
