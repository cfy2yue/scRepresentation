#!/usr/bin/env python3
"""CPU-only Wessels cross-dataset gene-response prior diagnostic.

This script estimates simple additive gene-response priors from canonical train
conditions and evaluates whether they can explain Wessels multi-condition
held-out groups.  It is diagnostic only: all train-prior rows use canonical
train conditions, while the optional Wessels test-single oracle is explicitly
marked as leaky/non-claim evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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
WESSELS_GROUPS = ("test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")


def parse_genes(condition: str) -> list[str]:
    genes = []
    for token in re.split(r"\+", str(condition)):
        gene = token.strip().upper()
        if not gene or gene in BAD_TOKENS or gene.startswith("CONTROL"):
            continue
        if re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", gene):
            genes.append(gene)
    return sorted(set(genes))


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

    def mean_delta(self, condition: str) -> tuple[np.ndarray, int, int]:
        idx = self.cond2idx[condition]
        cs, ce = int(self.ctrl_offsets[idx]), int(self.ctrl_offsets[idx + 1])
        gs, ge = int(self.gt_offsets[idx]), int(self.gt_offsets[idx + 1])
        ctrl = self.handle[f"{self.ctrl_key}/emb"][cs:ce]
        gt = self.handle["gt/emb"][gs:ge]
        return np.asarray(gt.mean(axis=0) - ctrl.mean(axis=0), dtype=np.float32), int(ce - cs), int(ge - gs)


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size == 0 or b.size == 0:
        return None
    aa = a.astype(np.float64) - float(np.mean(a))
    bb = b.astype(np.float64) - float(np.mean(b))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return None
    return float(np.dot(aa, bb) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(clean)) if clean else None


def load_split(path: Path) -> dict[str, dict[str, list[str]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_handle(cache: dict[str, H5Means], data_dir: Path, dataset: str) -> H5Means:
    if dataset not in cache:
        cache[dataset] = H5Means(data_dir / f"{dataset}.h5")
    return cache[dataset]


def build_gene_priors(
    *,
    data_dir: Path,
    split: dict[str, dict[str, list[str]]],
    source: str,
    target_genes: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    handles: dict[str, H5Means] = {}
    vectors_by_gene: dict[str, list[np.ndarray]] = defaultdict(list)
    meta_by_gene: dict[str, dict[str, Any]] = defaultdict(lambda: {"n_conditions": 0, "datasets": defaultdict(int)})
    for dataset, groups in split.items():
        for condition in groups.get("train", []):
            genes = parse_genes(condition)
            if len(genes) != 1:
                continue
            if genes[0] not in target_genes:
                continue
            if source == "hematopoietic" and not (
                "K562" in dataset or "jurket" in dataset.lower() or dataset == "Wessels"
            ):
                continue
            handle = get_handle(handles, data_dir, dataset)
            if condition not in handle.cond2idx:
                continue
            delta, _, _ = handle.mean_delta(condition)
            gene = genes[0]
            vectors_by_gene[gene].append(delta)
            meta_by_gene[gene]["n_conditions"] += 1
            meta_by_gene[gene]["datasets"][dataset] += 1
    for handle in handles.values():
        handle.close()
    priors = {gene: np.mean(vectors, axis=0).astype(np.float32) for gene, vectors in vectors_by_gene.items()}
    meta = {
        gene: {
            "n_conditions": info["n_conditions"],
            "datasets": dict(sorted(info["datasets"].items())),
        }
        for gene, info in meta_by_gene.items()
    }
    return priors, meta


def build_wessels_test_single_oracle(
    *,
    data_dir: Path,
    wessels_split: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    handle = H5Means(data_dir / "Wessels.h5")
    priors = {}
    meta = {}
    for condition in wessels_split.get("test_single", []):
        genes = parse_genes(condition)
        if len(genes) != 1 or condition not in handle.cond2idx:
            continue
        delta, n_ctrl, n_gt = handle.mean_delta(condition)
        gene = genes[0]
        priors[gene] = delta
        meta[gene] = {"n_conditions": 1, "datasets": {"Wessels_test_single": 1}, "n_ctrl": n_ctrl, "n_gt": n_gt}
    handle.close()
    return priors, meta


def evaluate_prior(
    *,
    data_dir: Path,
    wessels_split: dict[str, list[str]],
    prior_name: str,
    priors: dict[str, np.ndarray],
    prior_meta: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    handle = H5Means(data_dir / "Wessels.h5")
    condition_rows = []
    for group in WESSELS_GROUPS:
        for condition in wessels_split.get(group, []):
            genes = parse_genes(condition)
            available = [gene for gene in genes if gene in priors]
            missing = [gene for gene in genes if gene not in priors]
            target, n_ctrl, n_gt = handle.mean_delta(condition)
            pred = None
            if available and len(available) == len(genes):
                pred = np.sum([priors[gene] for gene in available], axis=0).astype(np.float32)
            row: dict[str, Any] = {
                "prior": prior_name,
                "group": group,
                "dataset": "Wessels",
                "condition": condition,
                "genes": "+".join(genes),
                "n_genes": len(genes),
                "coverage_n": len(available),
                "coverage_fraction": len(available) / max(len(genes), 1),
                "missing_genes": "+".join(missing),
                "prior_train_condition_count_sum": sum(int(prior_meta.get(gene, {}).get("n_conditions", 0)) for gene in available),
                "n_ctrl": n_ctrl,
                "n_gt": n_gt,
                "target_norm": float(np.linalg.norm(target)),
                "pred_norm": None if pred is None else float(np.linalg.norm(pred)),
                "pearson": None if pred is None else pearson(pred, target),
                "cosine": None if pred is None else cosine(pred, target),
                "norm_ratio": None if pred is None or np.linalg.norm(target) <= 1e-12 else float(np.linalg.norm(pred) / np.linalg.norm(target)),
            }
            condition_rows.append(row)
    handle.close()

    group_rows = []
    for group in WESSELS_GROUPS:
        rows = [row for row in condition_rows if row["group"] == group]
        covered = [row for row in rows if row["coverage_fraction"] >= 1.0 and row["pearson"] is not None]
        group_rows.append(
            {
                "prior": prior_name,
                "group": group,
                "n_conditions": len(rows),
                "n_full_coverage": len(covered),
                "full_coverage_fraction": len(covered) / max(len(rows), 1),
                "mean_pearson": mean_or_none([row["pearson"] for row in covered]),
                "mean_cosine": mean_or_none([row["cosine"] for row in covered]),
                "mean_norm_ratio": mean_or_none([row["norm_ratio"] for row in covered]),
                "median_train_condition_count_sum": None
                if not covered
                else float(np.median([row["prior_train_condition_count_sum"] for row in covered])),
            }
        )
    return condition_rows, group_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
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


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Wessels Cross-Dataset Gene-Response Prior Diagnostic",
        "",
        "This CPU-only diagnostic evaluates additive gene-response priors against Wessels multi-condition held-out groups.",
        "",
        "Train-prior variants use canonical train single-gene conditions only. `wessels_test_single_oracle` is leaky and is included only as an upper-bound diagnostic.",
        "",
        "## Group Summary",
        "",
        "| prior | group | n | full coverage | mean Pearson | mean cosine | mean pred/target norm | median train cond count |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["group_rows"]:
        lines.append(
            f"| `{row['prior']}` | `{row['group']}` | {row['n_conditions']} | "
            f"{row['n_full_coverage']}/{row['n_conditions']} | {fmt(row['mean_pearson'])} | "
            f"{fmt(row['mean_cosine'])} | {fmt(row['mean_norm_ratio'])} | "
            f"{fmt(row['median_train_condition_count_sum'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- If train-only priors have weak/negative unseen2 Pearson with good coverage, condition-cache swaps and sampling alone are unlikely to solve Wessels.",
            "- If the leaky Wessels test-single oracle is much stronger than train-only priors, the missing ingredient is Wessels-context single-gene response supervision or a context-transfer model.",
            "- This analysis does not use held-out multi GT for training; it uses it only for evaluation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/cyx/1030/dataset/latentfm_full/scfoundation"))
    parser.add_argument("--split-file", type=Path, default=Path("/data/cyx/1030/dataset/biFlow_data/split_seed42.json"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-conditions-csv", type=Path, required=True)
    parser.add_argument("--out-groups-csv", type=Path, required=True)
    args = parser.parse_args()

    split = load_split(args.split_file)
    target_genes = {
        gene
        for group in WESSELS_GROUPS
        for condition in split["Wessels"].get(group, [])
        for gene in parse_genes(condition)
    }
    priors_all, meta_all = build_gene_priors(
        data_dir=args.data_dir,
        split=split,
        source="all_train_single",
        target_genes=target_genes,
    )
    priors_heme, meta_heme = build_gene_priors(
        data_dir=args.data_dir,
        split=split,
        source="hematopoietic",
        target_genes=target_genes,
    )
    priors_oracle, meta_oracle = build_wessels_test_single_oracle(
        data_dir=args.data_dir,
        wessels_split=split["Wessels"],
    )
    prior_sets = [
        ("train_single_all_datasets", priors_all, meta_all),
        ("train_single_hematopoietic", priors_heme, meta_heme),
        ("wessels_test_single_oracle", priors_oracle, meta_oracle),
    ]
    condition_rows = []
    group_rows = []
    for name, priors, meta in prior_sets:
        c_rows, g_rows = evaluate_prior(
            data_dir=args.data_dir,
            wessels_split=split["Wessels"],
            prior_name=name,
            priors=priors,
            prior_meta=meta,
        )
        condition_rows.extend(c_rows)
        group_rows.extend(g_rows)
    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "target_genes": sorted(target_genes),
        "prior_sizes": {name: len(priors) for name, priors, _ in prior_sets},
        "group_rows": group_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(args.out_md, payload)
    write_csv(args.out_conditions_csv, condition_rows)
    write_csv(args.out_groups_csv, group_rows)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "out_conditions_csv": str(args.out_conditions_csv),
                "out_groups_csv": str(args.out_groups_csv),
                "n_condition_rows": len(condition_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
