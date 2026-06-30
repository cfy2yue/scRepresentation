#!/usr/bin/env python3
"""Audit h5ad input eligibility for TranscriptFormer and NicheFormer.

The benchmark h5ads keep expression in ``X`` as log1p-normalized values. The
new count-required adapters therefore only use explicit count sources
(``raw.X`` or named count layers) and never infer counts with ``expm1(X)``.
This script is read-only and inspects h5ad structure through h5py so it does
not materialize expression matrices.

Writes:
  output/benchmark_inventory/new_model_input_eligibility.csv
  output/benchmark_inventory/new_model_input_eligibility.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import h5py

BENCH_ROOT = Path(__file__).resolve().parents[1]
SCFM_ROOT = BENCH_ROOT.parent

sys.path.insert(0, str(SCFM_ROOT / "fm"))
import paths

GENE_ID_CANDIDATES = ("ensembl_id", "ensemblid", "Ensembl_ID", "ENSEMBL", "gene_id", "feature_id", "gene_ids")
COUNT_LAYER_CANDIDATES = ("counts", "raw_counts", "count")
TRANSCRIPTFORMER_AUX_DEFAULTS = ("assay",)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _attr_list(group: h5py.Group, attr: str) -> list[str]:
    value = group.attrs.get(attr, [])
    return [_decode(x) for x in list(value)]


def _matrix_shape(node: h5py.Group | h5py.Dataset) -> tuple[int, int] | None:
    if isinstance(node, h5py.Dataset):
        if node.shape is not None and len(node.shape) == 2:
            return int(node.shape[0]), int(node.shape[1])
        return None
    shape = node.attrs.get("shape")
    if shape is not None and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    data = node.get("data")
    return (int(data.shape[0]), 1) if isinstance(data, h5py.Dataset) else None


def _var_names(f: h5py.File) -> list[str]:
    var = f["var"]
    index_key = _decode(var.attrs.get("_index", "_index"))
    if index_key in var:
        return _read_string_array(var[index_key])
    if "_index" in var:
        return _read_string_array(var["_index"])
    return []


def _read_string_array(node: h5py.Group | h5py.Dataset) -> list[str]:
    if isinstance(node, h5py.Dataset):
        return [_decode(x) for x in node[:]]
    if "categories" in node and "codes" in node:
        cats = [_decode(x) for x in node["categories"][:]]
        codes = node["codes"][:]
        return [cats[int(c)] if int(c) >= 0 else "" for c in codes]
    return []


def _gene_values(f: h5py.File) -> tuple[str, list[str]]:
    var = f["var"]
    cols = set(_attr_list(var, "column-order"))
    for candidate in GENE_ID_CANDIDATES:
        if candidate in cols and candidate in var:
            vals = _read_string_array(var[candidate])
            if vals:
                return candidate, vals
    names = _var_names(f)
    if names and all(x.startswith(("ENS", "ENSG", "ENSMUSG")) for x in names[: min(50, len(names))]):
        return "var_names", names
    return "", names


def _count_source(f: h5py.File) -> str:
    if "raw" in f and isinstance(f["raw"], h5py.Group) and "X" in f["raw"]:
        return "raw.X"
    layers = f.get("layers")
    if isinstance(layers, h5py.Group):
        for candidate in COUNT_LAYER_CANDIDATES:
            if candidate in layers:
                return f"layers[{candidate!r}]"
    return ""


def _has_log1p_uns(f: h5py.File) -> bool:
    return "uns" in f and isinstance(f["uns"], h5py.Group) and "log1p" in f["uns"]


def _category(path: Path) -> str:
    s = str(path)
    if "chempert" in s or "chemicalpert" in s or path.name.startswith("sciplex3_"):
        return "chempert"
    if "genepert" in s:
        return "genepert"
    if "atlas" in s or path.name.startswith("TS_") or path.stem in {"Blood", "BoneMarrow", "Heart", "Lung", "Skin", "LymphNode"}:
        return "atlas"
    return "other"


def _source_group(path: Path, dataset_root: Path) -> str:
    try:
        rel = path.relative_to(dataset_root)
    except ValueError:
        return path.parent.name
    if len(rel.parts) >= 2:
        return "/".join(rel.parts[:-1])
    return path.parent.name


def _mean_genes(mean_h5ad: Path) -> set[str]:
    if not mean_h5ad.is_file():
        return set()
    with h5py.File(mean_h5ad, "r") as f:
        return set(_var_names(f))


def _audit_file(path: Path, dataset_root: Path, nicheformer_genes: set[str]) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        shape = _matrix_shape(f["X"]) if "X" in f else None
        layers = sorted(list(f.get("layers", {}).keys())) if isinstance(f.get("layers"), h5py.Group) else []
        obs_cols = set(_attr_list(f["obs"], "column-order")) if "obs" in f else set()
        var_cols = _attr_list(f["var"], "column-order") if "var" in f else []
        gene_col, genes = _gene_values(f)
        count_source = _count_source(f)
        niche_overlap = len(nicheformer_genes.intersection(genes)) if nicheformer_genes and genes else 0
        missing_aux = [c for c in TRANSCRIPTFORMER_AUX_DEFAULTS if c not in obs_cols]
        n_obs, n_vars = shape if shape is not None else ("", "")
        transcriptformer_ready = bool(count_source and gene_col)
        nicheformer_ready = bool(count_source and niche_overlap >= 1000)
        return {
            "dataset_id": path.stem,
            "category": _category(path),
            "source_group": _source_group(path, dataset_root),
            "path": str(path),
            "n_obs": n_obs,
            "n_vars": n_vars,
            "x_marked_log1p": _has_log1p_uns(f),
            "layers": ";".join(layers),
            "count_source": count_source,
            "gene_id_source": gene_col,
            "var_columns_with_gene_ids": ";".join([c for c in var_cols if c in GENE_ID_CANDIDATES]),
            "missing_transcriptformer_aux_obs": ";".join(missing_aux),
            "nicheformer_gene_overlap": niche_overlap,
            "transcriptformer_formal_ready": transcriptformer_ready,
            "nicheformer_formal_ready": nicheformer_ready,
            "reason_if_not_ready": _reason(count_source, gene_col, niche_overlap, bool(nicheformer_genes)),
        }


def _reason(count_source: str, gene_col: str, niche_overlap: int, have_niche_mean: bool) -> str:
    reasons: list[str] = []
    if not count_source:
        reasons.append("missing explicit raw/count source")
    if not gene_col:
        reasons.append("missing Ensembl gene id column/var_names")
    if not have_niche_mean:
        reasons.append("missing NicheFormer model mean h5ad")
    elif niche_overlap < 1000:
        reasons.append(f"NicheFormer overlap <1000 ({niche_overlap})")
    return "; ".join(reasons)


def _discover_h5ads(dataset_root: Path) -> list[Path]:
    # Accept either the canonical dataset parent (/.../dataset) or the
    # scFM_data directory itself. This keeps the CLI aligned with SCFM_DATA_ROOT.
    if (dataset_root / "staging").is_dir():
        scfm_data = dataset_root
        dataset_parent = dataset_root.parent
    else:
        scfm_data = dataset_root / "scFM_data"
        dataset_parent = dataset_root
    roots = [
        scfm_data / "staging",
        dataset_parent / "raw" / "atlas_TS",
        dataset_parent / "raw" / "chemicalpert_bench",
        dataset_parent / "raw" / "genepert_bench",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.h5ad")))
    return sorted(dict.fromkeys(files))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=paths.data_root().parent)
    ap.add_argument("--out-dir", type=Path, default=paths.output_root() / "benchmark_inventory")
    ap.add_argument("--nicheformer-mean-h5ad", type=Path, default=paths.third_party_root() / "nicheformer/data/model_means/model.h5ad")
    args = ap.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    niche_genes = _mean_genes(args.nicheformer_mean_h5ad.expanduser().resolve())

    rows = [_audit_file(p, dataset_root, niche_genes) for p in _discover_h5ads(dataset_root)]
    csv_path = out_dir / "new_model_input_eligibility.csv"
    md_path = out_dir / "new_model_input_eligibility.md"
    cols = [
        "dataset_id",
        "category",
        "source_group",
        "n_obs",
        "n_vars",
        "x_marked_log1p",
        "layers",
        "count_source",
        "gene_id_source",
        "var_columns_with_gene_ids",
        "missing_transcriptformer_aux_obs",
        "nicheformer_gene_overlap",
        "transcriptformer_formal_ready",
        "nicheformer_formal_ready",
        "reason_if_not_ready",
        "path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    _write_markdown(md_path, rows, dataset_root, args.nicheformer_mean_h5ad)
    print(json.dumps({"csv": str(csv_path), "md": str(md_path), "n_rows": len(rows)}, ensure_ascii=False))
    return 0


def _write_markdown(md_path: Path, rows: list[dict[str, Any]], dataset_root: Path, niche_mean: Path) -> None:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["source_group"]), []).append(row)
    tf_ready = sum(bool(r["transcriptformer_formal_ready"]) for r in rows)
    nf_ready = sum(bool(r["nicheformer_formal_ready"]) for r in rows)
    lines = [
        "# New Model Input Eligibility",
        "",
        f"Dataset root: `{dataset_root}`",
        f"NicheFormer mean h5ad: `{niche_mean}`",
        "",
        "Benchmark `X` is treated as log1p-normalized. TranscriptFormer and NicheFormer are considered formal-ready only when an explicit raw/count source is present (`raw.X` or a count layer such as `layers['counts']`). No `expm1(X)` count recovery is counted as ready.",
        "",
        f"- Total h5ad files audited: {len(rows)}",
        f"- TranscriptFormer formal-ready files: {tf_ready}",
        f"- NicheFormer formal-ready files: {nf_ready}",
        "",
        "## By Source Group",
        "",
    ]
    for group in sorted(by_group):
        sub = by_group[group]
        n = len(sub)
        lines.append(f"### {group}")
        lines.append(f"- files: {n}")
        lines.append(f"- TranscriptFormer ready: {sum(bool(r['transcriptformer_formal_ready']) for r in sub)}")
        lines.append(f"- NicheFormer ready: {sum(bool(r['nicheformer_formal_ready']) for r in sub)}")
        missing = sorted({str(r["reason_if_not_ready"]) for r in sub if str(r["reason_if_not_ready"])})
        if missing:
            lines.append(f"- common blockers: {' | '.join(missing[:5])}")
        lines.append("")
    lines.extend(["## Full table", "", "See `new_model_input_eligibility.csv`."])
    md_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
