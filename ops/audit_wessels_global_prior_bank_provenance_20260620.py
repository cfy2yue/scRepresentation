#!/usr/bin/env python3
"""Reconstruct targeted global-prior bank provenance for Wessels held-out genes.

This is CPU-only and intended to document the 2026-06-20 Wessels global-prior
diagnostic.  It mirrors the training bank's deterministic capped mean rule for
canonical train single-gene conditions, but limits reads to Wessels held-out
component genes so it remains lightweight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


WESSELS_GROUPS = ("test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")
BAD_TOKENS = {
    "CONTROL",
    "CTRL",
    "NON-TARGETING",
    "NONTARGETING",
    "POS",
    "TSS",
    "KLANN",
    "MOSAIC",
    "INTERGENIC",
}


def stable_int_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def parse_genes(condition: str) -> list[str]:
    genes = []
    for token in re.split(r"\+", str(condition)):
        gene = token.strip().upper()
        if not gene or gene in BAD_TOKENS or gene.startswith("CONTROL"):
            continue
        if re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", gene):
            genes.append(gene)
    return sorted(set(genes))


def load_condition_metadata(data_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = data_dir / "condition_metadata.json"
    if not path.is_file():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def genes_for_condition(
    *,
    metadata: dict[str, dict[str, dict[str, Any]]],
    dataset: str,
    condition: str,
) -> list[str]:
    entry = metadata.get(dataset, {}).get(condition, {})
    raw = entry.get("genes") if isinstance(entry, dict) else None
    if isinstance(raw, list):
        genes = sorted({str(g).strip().upper() for g in raw if str(g).strip()})
        return genes
    if isinstance(raw, str) and raw.strip():
        return parse_genes(raw)
    return parse_genes(condition)


class H5Means:
    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        self.handle = h5py.File(h5_path, "r")
        self.conditions = self.handle["conditions"].asstr()[:].tolist()
        self.cond2idx = {cond: idx for idx, cond in enumerate(self.conditions)}
        self.ctrl_key = "ctrl" if "ctrl/offsets" in self.handle else "ir"
        self.ctrl_offsets = self.handle[f"{self.ctrl_key}/offsets"][:]
        self.gt_offsets = self.handle["gt/offsets"][:]

    def close(self) -> None:
        self.handle.close()

    def _mean_rows(self, key: str, start: int, end: int, *, max_cells: int, seed: int) -> np.ndarray:
        n = int(end - start)
        if n <= 0:
            raise ValueError("empty condition slice")
        if max_cells > 0 and n > max_cells:
            rng = np.random.RandomState(int(seed))
            rel = np.sort(rng.choice(n, size=int(max_cells), replace=False))
            rows = self.handle[key][rel + int(start)]
        else:
            rows = self.handle[key][int(start):int(end)]
        return np.asarray(rows, dtype=np.float32).mean(axis=0)

    def capped_delta(self, condition: str, *, max_cells: int, seed_base: int) -> tuple[np.ndarray, int, int]:
        idx = self.cond2idx[condition]
        cs, ce = int(self.ctrl_offsets[idx]), int(self.ctrl_offsets[idx + 1])
        gs, ge = int(self.gt_offsets[idx]), int(self.gt_offsets[idx + 1])
        ctrl_mean = self._mean_rows(
            f"{self.ctrl_key}/emb",
            cs,
            ce,
            max_cells=max_cells,
            seed=seed_base,
        )
        gt_mean = self._mean_rows("gt/emb", gs, ge, max_cells=max_cells, seed=seed_base + 17)
        return (gt_mean - ctrl_mean).astype(np.float32), int(ce - cs), int(ge - gs)


def get_handle(cache: dict[str, H5Means], data_dir: Path, dataset: str) -> H5Means:
    if dataset not in cache:
        cache[dataset] = H5Means(data_dir / f"{dataset}.h5")
    return cache[dataset]


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Wessels Global Prior Bank Provenance",
        "",
        "This CPU-only audit reconstructs the train-only global gene-response prior bank for Wessels held-out component genes.",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- max_cells: `{payload['max_cells']}`",
        f"- target genes: `{payload['n_target_genes']}`",
        "",
        "## Group Coverage",
        "",
        "| group | conditions | unique component genes | covered genes | full-coverage conditions |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["group_rows"]:
        lines.append(
            f"| `{row['group']}` | {row['n_conditions']} | {row['n_unique_genes']} | "
            f"{row['n_covered_genes']} | {row['n_full_coverage_conditions']} |"
        )
    lines.extend(
        [
            "",
            "## Gene Provenance",
            "",
            "| gene | train records | source datasets | delta norm | mean ctrl cells | mean gt cells |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in payload["gene_rows"]:
        lines.append(
            f"| `{row['gene']}` | {row['n_train_records']} | `{row['source_datasets']}` | "
            f"{fmt(row['gene_mean_delta_norm'])} | {fmt(row['mean_n_ctrl'])} | {fmt(row['mean_n_gt'])} |"
        )
    lines.extend(
        [
            "",
            "Only canonical train single-gene conditions are used for source records. Held-out Wessels multi-condition ground truth is not read for this provenance audit.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/cyx/1030/dataset/latentfm_full/scfoundation"))
    parser.add_argument("--split-file", type=Path, default=Path("/data/cyx/1030/dataset/biFlow_data/split_seed42.json"))
    parser.add_argument("--max-cells", type=int, default=512)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-genes-csv", type=Path, required=True)
    parser.add_argument("--out-groups-csv", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    metadata = load_condition_metadata(args.data_dir)
    wessels = split.get("Wessels", {})
    target_genes = {
        gene
        for group in WESSELS_GROUPS
        for cond in wessels.get(group, [])
        for gene in genes_for_condition(metadata=metadata, dataset="Wessels", condition=cond)
    }
    handles: dict[str, H5Means] = {}
    records_by_gene: dict[str, list[dict[str, Any]]] = defaultdict(list)

    try:
        for dataset, groups in split.items():
            for condition in groups.get("train", []):
                genes = genes_for_condition(metadata=metadata, dataset=dataset, condition=condition)
                if len(genes) != 1 or genes[0] not in target_genes:
                    continue
                handle = get_handle(handles, args.data_dir, dataset)
                if condition not in handle.cond2idx:
                    continue
                seed_base = stable_int_hash(f"condition_prior:{dataset}:{condition}")
                delta, n_ctrl, n_gt = handle.capped_delta(
                    condition,
                    max_cells=int(args.max_cells),
                    seed_base=seed_base,
                )
                records_by_gene[genes[0]].append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "n_ctrl": n_ctrl,
                        "n_gt": n_gt,
                        "delta_norm": float(np.linalg.norm(delta)),
                        "delta": delta,
                    }
                )
    finally:
        for handle in handles.values():
            handle.close()

    gene_rows: list[dict[str, Any]] = []
    for gene in sorted(target_genes):
        records = records_by_gene.get(gene, [])
        if records:
            mean_delta = np.mean([row["delta"] for row in records], axis=0).astype(np.float32)
            ds_counts: dict[str, int] = defaultdict(int)
            for row in records:
                ds_counts[row["dataset"]] += 1
            source_datasets = ";".join(f"{ds}:{count}" for ds, count in sorted(ds_counts.items()))
            mean_n_ctrl = float(np.mean([row["n_ctrl"] for row in records]))
            mean_n_gt = float(np.mean([row["n_gt"] for row in records]))
            gene_mean_delta_norm = float(np.linalg.norm(mean_delta))
        else:
            source_datasets = ""
            mean_n_ctrl = None
            mean_n_gt = None
            gene_mean_delta_norm = None
        gene_rows.append(
            {
                "gene": gene,
                "n_train_records": len(records),
                "source_datasets": source_datasets,
                "gene_mean_delta_norm": gene_mean_delta_norm,
                "mean_n_ctrl": mean_n_ctrl,
                "mean_n_gt": mean_n_gt,
            }
        )

    covered = {row["gene"] for row in gene_rows if int(row["n_train_records"]) > 0}
    group_rows: list[dict[str, Any]] = []
    for group in WESSELS_GROUPS:
        conds = list(wessels.get(group, []))
        group_genes = {
            gene
            for cond in conds
            for gene in genes_for_condition(metadata=metadata, dataset="Wessels", condition=cond)
        }
        full = 0
        for cond in conds:
            genes = genes_for_condition(metadata=metadata, dataset="Wessels", condition=cond)
            if genes and all(gene in covered for gene in genes):
                full += 1
        group_rows.append(
            {
                "group": group,
                "n_conditions": len(conds),
                "n_unique_genes": len(group_genes),
                "n_covered_genes": len(group_genes & covered),
                "n_full_coverage_conditions": full,
            }
        )

    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "max_cells": int(args.max_cells),
        "n_target_genes": len(target_genes),
        "n_covered_genes": len(covered),
        "gene_rows": gene_rows,
        "group_rows": group_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(args.out_md, payload)
    write_csv(args.out_genes_csv, gene_rows)
    write_csv(args.out_groups_csv, group_rows)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
