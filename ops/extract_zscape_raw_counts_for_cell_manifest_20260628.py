#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path("/data/cyx/1030/scLatent")
RDATA_DEPS = ROOT / "software/python_deps/rdata_20260628_nodeps"
if RDATA_DEPS.exists():
    sys.path.insert(1, str(RDATA_DEPS))

try:
    import rdata  # type: ignore
except Exception as exc:  # pragma: no cover - runtime env gate
    raise SystemExit(
        "Missing rdata reader. Expected local dependency at "
        f"{RDATA_DEPS}. Original error: {exc!r}"
    )


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def sha256sum(path: Path) -> str:
    result = subprocess.run(["sha256sum", str(path)], check=True, capture_output=True, text=True)
    return result.stdout.strip().split()[0]


def maybe_outer_gzip_handle(path: Path):
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rb")
    return path.open("rb")


def read_rds(path: Path) -> Any:
    # ZSCAPE raw_counts is an outer .gz containing a saveRDS-compressed object.
    # Feeding the outer-decompressed stream lets rdata handle the inner RDS.
    with maybe_outer_gzip_handle(path) as handle:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            obj = rdata.read_rds(handle)
    return obj, [str(w.message) for w in caught]


def as_strings(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, pd.Index):
        return [str(x) for x in values.tolist()]
    if isinstance(values, np.ndarray):
        return [str(x) for x in values.tolist()]
    if isinstance(values, (list, tuple)):
        return [str(x) for x in values]
    return [str(values)]


def object_summary(obj: Any) -> dict[str, Any]:
    attrs = {}
    for name in ["class", "Dim", "i", "p", "x", "Dimnames"]:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if name in {"i", "p", "x"}:
                attrs[name] = {"type": type(value).__name__, "len": int(len(value))}
            elif name == "Dimnames":
                attrs[name] = {"type": type(value).__name__}
            else:
                try:
                    attrs[name] = np.asarray(value).tolist()
                except Exception:
                    attrs[name] = str(value)
    return {"python_type": type(obj).__name__, "slots": attrs}


def pick_matrix_object(obj: Any) -> Any:
    if hasattr(obj, "i") and hasattr(obj, "p") and hasattr(obj, "x") and hasattr(obj, "Dim"):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            if hasattr(value, "i") and hasattr(value, "p") and hasattr(value, "x") and hasattr(value, "Dim"):
                return value
    raise ValueError(f"Could not find dgCMatrix-like object in RDS object of type {type(obj).__name__}")


def dimnames_from_object(obj: Any) -> tuple[list[str], list[str]]:
    dimnames = getattr(obj, "Dimnames", None)
    if isinstance(dimnames, (list, tuple)) and len(dimnames) >= 2:
        return as_strings(dimnames[0]), as_strings(dimnames[1])
    if isinstance(dimnames, dict):
        values = list(dimnames.values())
        if len(values) >= 2:
            return as_strings(values[0]), as_strings(values[1])
    raise ValueError("Missing or unsupported Dimnames in raw-count matrix")


def sparse_from_dgc(obj: Any) -> tuple[sp.csc_matrix, list[str], list[str]]:
    dims = np.asarray(getattr(obj, "Dim"), dtype=np.int64)
    if dims.size != 2:
        raise ValueError(f"Expected two matrix dimensions, got {dims}")
    i = np.asarray(getattr(obj, "i"), dtype=np.int32)
    p = np.asarray(getattr(obj, "p"), dtype=np.int64)
    x = np.asarray(getattr(obj, "x"), dtype=np.float32)
    matrix = sp.csc_matrix((x, i, p), shape=(int(dims[0]), int(dims[1])))
    row_names, col_names = dimnames_from_object(obj)
    return matrix, row_names, col_names


def read_cell_manifest(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    if "cell" not in df.columns:
        raise ValueError(f"cell manifest lacks `cell` column: {path}")
    ordered = list(OrderedDict((str(cell), None) for cell in df["cell"].astype(str)).keys())
    return df, ordered


def write_list(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-counts", type=Path, required=True)
    parser.add_argument("--cell-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-cell-match-frac", type=float, default=0.99)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.out_dir / "LATENTFM_ZSCAPE_RAW_COUNTS_CELL_MANIFEST_EXTRACTION_20260628.md"
    json_path = args.out_dir / "zscape_raw_counts_cell_manifest_extraction_20260628.json"

    if not args.raw_counts.exists():
        payload = {
            "timestamp_utc": utc_now(),
            "status": "zscape_raw_counts_extract_waiting_for_raw_counts_no_gpu",
            "gpu_authorized": False,
            "raw_counts": str(args.raw_counts),
            "reason": "raw counts file not present",
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(
            "# LatentFM ZSCAPE Raw Counts Cell-Manifest Extraction\n\n"
            f"Status: `{payload['status']}`\n\n"
            "Raw counts file is not present yet. Do not run expression OT.\n",
            encoding="utf-8",
        )
        print(md_path)
        print(json_path)
        print(payload["status"])
        return 2

    cell_manifest, ordered_cells = read_cell_manifest(args.cell_manifest)
    raw_sha = sha256sum(args.raw_counts)
    obj, read_warnings = read_rds(args.raw_counts)
    matrix_obj = pick_matrix_object(obj)
    preflight = object_summary(matrix_obj)
    matrix, row_names, col_names = sparse_from_dgc(matrix_obj)

    col_index = {name: idx for idx, name in enumerate(col_names)}
    row_index = {name: idx for idx, name in enumerate(row_names)}
    matched_cols = [cell for cell in ordered_cells if cell in col_index]
    matched_rows = [cell for cell in ordered_cells if cell in row_index]

    orientation = "genes_by_cells"
    if len(matched_cols) < len(matched_rows):
        matrix = matrix.T.tocsc()
        row_names, col_names = col_names, row_names
        col_index = {name: idx for idx, name in enumerate(col_names)}
        matched_cols = [cell for cell in ordered_cells if cell in col_index]
        orientation = "cells_by_genes_transposed_to_genes_by_cells"

    match_frac = len(matched_cols) / max(1, len(ordered_cells))
    status = (
        "zscape_raw_counts_cell_manifest_extraction_pass_no_gpu"
        if match_frac >= args.min_cell_match_frac
        else "zscape_raw_counts_cell_manifest_extraction_fail_no_gpu"
    )

    selected_col_indices = np.array([col_index[cell] for cell in matched_cols], dtype=np.int64)
    selected = matrix[:, selected_col_indices].tocsc()
    counts_path = args.out_dir / "zscape_manifest_selected_counts_csc.npz"
    gene_path = args.out_dir / "zscape_manifest_selected_gene_names.txt"
    cell_index_path = args.out_dir / "zscape_manifest_selected_expression_cell_index.csv"
    sp.save_npz(counts_path, selected)
    write_list(gene_path, row_names)

    cell_index_rows = [{"cell": cell, "expression_col_index": i} for i, cell in enumerate(matched_cols)]
    with cell_index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell", "expression_col_index"])
        writer.writeheader()
        writer.writerows(cell_index_rows)

    manifest_matched = cell_manifest[cell_manifest["cell"].astype(str).isin(set(matched_cols))].copy()
    manifest_matched_path = args.out_dir / "zscape_expression_selected_cell_ids_matched.csv"
    manifest_matched.to_csv(manifest_matched_path, index=False)

    payload = {
        "timestamp_utc": utc_now(),
        "status": status,
        "gpu_authorized": False,
        "raw_counts": str(args.raw_counts),
        "raw_counts_sha256": raw_sha,
        "cell_manifest": str(args.cell_manifest),
        "read_warnings": read_warnings,
        "r_object_preflight": preflight,
        "orientation": orientation,
        "matrix_shape_genes_by_cells": [int(selected.shape[0]), int(selected.shape[1])],
        "matrix_nnz": int(selected.nnz),
        "requested_unique_cells": len(ordered_cells),
        "matched_unique_cells": len(matched_cols),
        "cell_match_frac": match_frac,
        "outputs": {
            "counts_npz": str(counts_path),
            "gene_names": str(gene_path),
            "cell_index": str(cell_index_path),
            "matched_manifest": str(manifest_matched_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Raw Counts Cell-Manifest Extraction",
        "",
        f"Timestamp: `{payload['timestamp_utc']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only expression extraction from the downloaded ZPERTURB raw-count source.",
        "- Extracts only cells selected by the audited cell-ID manifest.",
        "- Does not train, infer, embed with scFM, read canonical multi, or read Track C query.",
        "",
        "## Gate Summary",
        "",
        f"- requested unique cells: `{len(ordered_cells)}`",
        f"- matched unique cells: `{len(matched_cols)}`",
        f"- cell match fraction: `{match_frac:.6f}`",
        f"- selected count matrix shape genes x cells: `{selected.shape[0]} x {selected.shape[1]}`",
        f"- selected matrix nnz: `{selected.nnz}`",
        f"- orientation: `{orientation}`",
        "",
        "## Decision",
        "",
        (
            "Proceed to CPU expression continuity/OT validation."
            if status.endswith("pass_no_gpu")
            else "Do not proceed to OT; fix cell-ID mapping/provenance first."
        ),
        "This extraction still does not authorize GPU training.",
        "",
        "## Output Files",
        "",
        f"- counts: `{counts_path}`",
        f"- genes: `{gene_path}`",
        f"- cell index: `{cell_index_path}`",
        f"- matched manifest: `{manifest_matched_path}`",
        f"- JSON: `{json_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(json_path)
    print(status)
    return 0 if status.endswith("pass_no_gpu") else 2


if __name__ == "__main__":
    raise SystemExit(main())
