#!/usr/bin/env python3
"""Audit condition-source alignment against xverse train-single residual geometry.

This uses canonical train single-gene conditions only. It compares whether
pretrained gene embedding cosine similarities align with xverse response
residual cosine similarities. It is CPU-only and does not use held-out GT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_SCGPT = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
DEFAULT_CELLNAVI = ROOT / "pretrainckpt/genepert_cache/cellnavi_embed_gene"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_condition_source_alignment_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CONDITION_SOURCE_ALIGNMENT_20260621.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def stable_subset(items: list[str], k: int, key: str) -> list[str]:
    if k <= 0 or len(items) <= k:
        return list(items)
    return sorted(items, key=lambda x: hashlib.sha256(f"{key}|{x}".encode()).hexdigest())[:k]


def condition_mean(handle: h5py.File, group: str, idx: int, max_cells: int) -> np.ndarray | None:
    offsets = np.asarray(handle[f"{group}/offsets"])
    start, end = int(offsets[idx]), int(offsets[idx + 1])
    if end <= start:
        return None
    if max_cells > 0 and end - start > max_cells:
        end = start + max_cells
    return np.asarray(handle[f"{group}/emb"][start:end], dtype=np.float32).mean(axis=0)


def collect_residuals(
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    *,
    max_genes_per_dataset: int,
    max_cells: int,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for ds, obj in sorted(split.items()):
        path = data_dir / f"{ds}.h5"
        if not path.is_file():
            continue
        train = [str(x) for x in obj.get("train", [])]
        singles = []
        for cond in train:
            meta = (metadata.get(ds) or {}).get(cond) or {}
            genes = [str(g) for g in meta.get("genes") or []]
            if len(genes) == 1:
                singles.append((cond, genes[0]))
        chosen_conds = stable_subset([c for c, _ in singles], max_genes_per_dataset, f"align|{ds}")
        gene_by_cond = {c: g for c, g in singles}
        ds_out: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            conditions = decode(np.asarray(handle["conditions"]))
            by_cond = {c: i for i, c in enumerate(conditions)}
            for cond in chosen_conds:
                idx = by_cond.get(cond)
                if idx is None:
                    continue
                ctrl = condition_mean(handle, "ctrl", idx, max_cells)
                gt = condition_mean(handle, "gt", idx, max_cells)
                if ctrl is None or gt is None:
                    continue
                ds_out[gene_by_cond[cond]] = (gt - ctrl).astype(np.float32)
        if ds_out:
            out[str(ds)] = ds_out
    return out


def load_gene_embeddings(cache_dir: Path) -> tuple[dict[str, int], np.ndarray]:
    index: dict[str, int] = {}
    with (cache_dir / "gene_index.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene, idx = parts[0], parts[1]
            if gene in {"PAD", "UNK", ""}:
                continue
            try:
                index[gene] = int(idx)
            except ValueError:
                continue
    emb = np.load(cache_dir / "gene_embeddings.npy")
    emb = emb.astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    return index, emb


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # Average exact ties.
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return pearson(rankdata(x), rankdata(y))


def pairwise_alignment(
    genes: list[str],
    residuals: dict[str, np.ndarray],
    index: dict[str, int],
    emb: np.ndarray,
    max_pairs: int,
    key: str,
) -> dict[str, Any]:
    genes = [g for g in genes if g in residuals and g in index]
    pairs = [(genes[i], genes[j]) for i in range(len(genes)) for j in range(i + 1, len(genes))]
    pair_labels = [f"{a}|{b}" for a, b in pairs]
    chosen_labels = set(stable_subset(pair_labels, max_pairs, key)) if max_pairs > 0 else set(pair_labels)
    x_resp = []
    x_emb = []
    for a, b in pairs:
        if f"{a}|{b}" not in chosen_labels:
            continue
        ra = residuals[a]
        rb = residuals[b]
        resp = float(np.dot(ra, rb) / max(float(np.linalg.norm(ra) * np.linalg.norm(rb)), 1e-8))
        ge = float(np.dot(emb[index[a]], emb[index[b]]))
        x_resp.append(resp)
        x_emb.append(ge)
    arr_resp = np.asarray(x_resp, dtype=np.float64)
    arr_emb = np.asarray(x_emb, dtype=np.float64)
    return {
        "n_genes": len(genes),
        "n_pairs": int(len(arr_resp)),
        "pearson": pearson(arr_resp, arr_emb),
        "spearman": spearman(arr_resp, arr_emb),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--scgpt-cache", type=Path, default=DEFAULT_SCGPT)
    parser.add_argument("--cellnavi-cache", type=Path, default=DEFAULT_CELLNAVI)
    parser.add_argument("--max-genes-per-dataset", type=int, default=256)
    parser.add_argument("--max-cells-per-condition", type=int, default=512)
    parser.add_argument("--max-pairs-per-dataset", type=int, default=20000)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    split = load_json(args.split_file)
    metadata = load_json(args.data_dir / "condition_metadata.json")
    residuals = collect_residuals(
        args.data_dir,
        split,
        metadata,
        max_genes_per_dataset=args.max_genes_per_dataset,
        max_cells=args.max_cells_per_condition,
    )
    sc_idx, sc_emb = load_gene_embeddings(args.scgpt_cache)
    cn_idx, cn_emb = load_gene_embeddings(args.cellnavi_cache)

    rows = []
    for ds, ds_resid in sorted(residuals.items()):
        genes = sorted(ds_resid)
        if len(genes) < 8:
            continue
        sc = pairwise_alignment(genes, ds_resid, sc_idx, sc_emb, args.max_pairs_per_dataset, f"scgpt|{ds}")
        cn = pairwise_alignment(genes, ds_resid, cn_idx, cn_emb, args.max_pairs_per_dataset, f"cellnavi|{ds}")
        rows.append(
            {
                "dataset": ds,
                "n_train_single_genes": len(genes),
                "scgpt": sc,
                "cellnavi": cn,
                "spearman_delta_cellnavi_minus_scgpt": (
                    None
                    if sc.get("spearman") is None or cn.get("spearman") is None
                    else float(cn["spearman"] - sc["spearman"])
                ),
            }
        )

    valid_sc = [r["scgpt"]["spearman"] for r in rows if r["scgpt"].get("spearman") is not None]
    valid_cn = [r["cellnavi"]["spearman"] for r in rows if r["cellnavi"].get("spearman") is not None]
    deltas = [r["spearman_delta_cellnavi_minus_scgpt"] for r in rows if r["spearman_delta_cellnavi_minus_scgpt"] is not None]
    summary = {
        "n_datasets": len(rows),
        "median_scgpt_spearman": float(np.median(valid_sc)) if valid_sc else None,
        "median_cellnavi_spearman": float(np.median(valid_cn)) if valid_cn else None,
        "median_delta_cellnavi_minus_scgpt": float(np.median(deltas)) if deltas else None,
        "cellnavi_better_fraction": float(np.mean(np.asarray(deltas) > 0)) if deltas else None,
    }
    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "scgpt_cache": str(args.scgpt_cache),
        "cellnavi_cache": str(args.cellnavi_cache),
        "max_genes_per_dataset": int(args.max_genes_per_dataset),
        "max_cells_per_condition": int(args.max_cells_per_condition),
        "max_pairs_per_dataset": int(args.max_pairs_per_dataset),
        "leakage_status": "train_single_residuals_only_no_test_gt_no_posthoc_no_pert_means",
        "summary": summary,
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# LatentFM xverse Condition-Source Alignment Audit 2026-06-21",
        "",
        "This CPU audit compares pretrained gene embedding pair similarities against xverse train-single residual similarities.",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        "",
        "## Summary",
        "",
        f"- datasets: `{summary['n_datasets']}`",
        f"- median scGPT Spearman: `{summary['median_scgpt_spearman']}`",
        f"- median CellNavi Spearman: `{summary['median_cellnavi_spearman']}`",
        f"- median CellNavi - scGPT Spearman: `{summary['median_delta_cellnavi_minus_scgpt']}`",
        f"- CellNavi better fraction: `{summary['cellnavi_better_fraction']}`",
        "",
        "## Dataset Table",
        "",
        "| dataset | n genes | scGPT Spearman | CellNavi Spearman | CellNavi - scGPT |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {n} | {sc} | {cn} | {delta} |".format(
                dataset=row["dataset"],
                n=row["n_train_single_genes"],
                sc=row["scgpt"].get("spearman"),
                cn=row["cellnavi"].get("spearman"),
                delta=row["spearman_delta_cellnavi_minus_scgpt"],
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If neither source aligns with train-single xverse residual geometry, do not start a condition-source GPU smoke.",
        "- If one source is consistently better, use it as a CPU-gated hypothesis for a small condition-adapter smoke.",
        "",
    ])
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
