#!/usr/bin/env python3
"""CPU-only feasibility audit for train-only interaction residual priors."""

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
        self.handle = h5py.File(h5_path, "r")
        self.conditions = self.handle["conditions"].asstr()[:].tolist()
        self.cond2idx = {cond: idx for idx, cond in enumerate(self.conditions)}
        self.ctrl_key = "ctrl" if "ctrl/offsets" in self.handle else "ir"
        self.ctrl_offsets = self.handle[f"{self.ctrl_key}/offsets"][:]
        self.gt_offsets = self.handle["gt/offsets"][:]

    def close(self) -> None:
        self.handle.close()

    def mean_delta(self, condition: str) -> np.ndarray:
        idx = self.cond2idx[condition]
        cs, ce = int(self.ctrl_offsets[idx]), int(self.ctrl_offsets[idx + 1])
        gs, ge = int(self.gt_offsets[idx]), int(self.gt_offsets[idx + 1])
        ctrl = self.handle[f"{self.ctrl_key}/emb"][cs:ce]
        gt = self.handle["gt/emb"][gs:ge]
        return np.asarray(gt.mean(axis=0) - ctrl.mean(axis=0), dtype=np.float32)


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
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


def get_handle(cache: dict[str, H5Means], data_dir: Path, dataset: str) -> H5Means:
    if dataset not in cache:
        cache[dataset] = H5Means(data_dir / f"{dataset}.h5")
    return cache[dataset]


def build_single_priors(
    *,
    data_dir: Path,
    split: dict[str, dict[str, list[str]]],
    target_genes: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    handles: dict[str, H5Means] = {}
    vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = defaultdict(lambda: {"n_conditions": 0, "datasets": defaultdict(int)})
    for dataset, groups in split.items():
        for condition in groups.get("train", []):
            genes = parse_genes(condition)
            if len(genes) != 1:
                continue
            if genes[0] not in target_genes:
                continue
            handle = get_handle(handles, data_dir, dataset)
            if condition not in handle.cond2idx:
                continue
            gene = genes[0]
            vectors[gene].append(handle.mean_delta(condition))
            meta[gene]["n_conditions"] += 1
            meta[gene]["datasets"][dataset] += 1
    for handle in handles.values():
        handle.close()
    return {gene: np.mean(vals, axis=0).astype(np.float32) for gene, vals in vectors.items()}, {
        gene: {"n_conditions": info["n_conditions"], "datasets": dict(info["datasets"])}
        for gene, info in meta.items()
    }


def build_train_residuals(
    *,
    data_dir: Path,
    split: dict[str, dict[str, list[str]]],
    single_priors: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    handles: dict[str, H5Means] = {}
    rows: list[dict[str, Any]] = []
    residuals_by_n: dict[int, list[np.ndarray]] = defaultdict(list)
    for dataset, groups in split.items():
        for condition in groups.get("train", []):
            genes = parse_genes(condition)
            if len(genes) < 2:
                continue
            missing = [gene for gene in genes if gene not in single_priors]
            handle = get_handle(handles, data_dir, dataset)
            if condition not in handle.cond2idx:
                continue
            target = handle.mean_delta(condition)
            additive = None if missing else np.sum([single_priors[gene] for gene in genes], axis=0).astype(np.float32)
            residual = None if additive is None else (target - additive).astype(np.float32)
            row = {
                "dataset": dataset,
                "condition": condition,
                "genes": "+".join(genes),
                "n_genes": len(genes),
                "missing_single_prior_genes": "+".join(missing),
                "has_full_single_prior": not missing,
                "additive_pearson": None if additive is None else pearson(additive, target),
                "additive_cosine": None if additive is None else cosine(additive, target),
                "target_norm": float(np.linalg.norm(target)),
                "additive_norm": None if additive is None else float(np.linalg.norm(additive)),
                "residual_norm": None if residual is None else float(np.linalg.norm(residual)),
            }
            rows.append(row)
            if residual is not None:
                residuals_by_n[len(genes)].append(residual)
    for handle in handles.values():
        handle.close()
    residual_prior = {n: np.mean(vals, axis=0).astype(np.float32) for n, vals in residuals_by_n.items()}
    return rows, residual_prior


def evaluate_wessels(
    *,
    data_dir: Path,
    wessels_split: dict[str, list[str]],
    single_priors: dict[str, np.ndarray],
    residual_prior: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    handle = H5Means(data_dir / "Wessels.h5")
    rows: list[dict[str, Any]] = []
    for group in WESSELS_GROUPS:
        for condition in wessels_split.get(group, []):
            genes = parse_genes(condition)
            missing = [gene for gene in genes if gene not in single_priors]
            target = handle.mean_delta(condition)
            additive = None if missing else np.sum([single_priors[gene] for gene in genes], axis=0).astype(np.float32)
            residual = residual_prior.get(len(genes))
            interaction = None if additive is None or residual is None else (additive + residual).astype(np.float32)
            rows.append(
                {
                    "group": group,
                    "condition": condition,
                    "genes": "+".join(genes),
                    "n_genes": len(genes),
                    "missing_single_prior_genes": "+".join(missing),
                    "has_residual_prior_for_n": residual is not None,
                    "additive_pearson": None if additive is None else pearson(additive, target),
                    "interaction_pearson": None if interaction is None else pearson(interaction, target),
                    "additive_cosine": None if additive is None else cosine(additive, target),
                    "interaction_cosine": None if interaction is None else cosine(interaction, target),
                    "target_norm": float(np.linalg.norm(target)),
                    "additive_norm": None if additive is None else float(np.linalg.norm(additive)),
                    "interaction_norm": None if interaction is None else float(np.linalg.norm(interaction)),
                }
            )
    handle.close()
    group_rows: list[dict[str, Any]] = []
    for group in WESSELS_GROUPS:
        subset = [row for row in rows if row["group"] == group]
        group_rows.append(
            {
                "group": group,
                "n_conditions": len(subset),
                "n_with_interaction_prior": sum(1 for row in subset if row["interaction_pearson"] is not None),
                "mean_additive_pearson": mean_or_none([row["additive_pearson"] for row in subset]),
                "mean_interaction_pearson": mean_or_none([row["interaction_pearson"] for row in subset]),
                "mean_interaction_delta": None
                if mean_or_none([row["interaction_pearson"] for row in subset]) is None
                or mean_or_none([row["additive_pearson"] for row in subset]) is None
                else mean_or_none([row["interaction_pearson"] for row in subset])
                - mean_or_none([row["additive_pearson"] for row in subset]),
                "mean_additive_cosine": mean_or_none([row["additive_cosine"] for row in subset]),
                "mean_interaction_cosine": mean_or_none([row["interaction_cosine"] for row in subset]),
            }
        )
    return rows, group_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
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
        "# Wessels Interaction Residual Feasibility Diagnostic",
        "",
        "This CPU-only diagnostic asks whether canonical train has any multi-condition interaction residual signal that can be used without leakage.",
        "",
        "## Train Split Counts",
        "",
        f"Train single conditions: `{payload['train_counts']['single']}`",
        f"Train multi conditions: `{payload['train_counts']['multi']}`",
        f"Train multi with full single-prior coverage: `{payload['train_counts']['multi_full_single_prior']}`",
        "",
        "## Wessels Held-Out Multi Evaluation",
        "",
        "| group | n | n with residual prior | additive Pearson | interaction Pearson | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["wessels_group_rows"]:
        lines.append(
            f"| `{row['group']}` | {row['n_conditions']} | {row['n_with_interaction_prior']} | "
            f"{fmt(row['mean_additive_pearson'])} | {fmt(row['mean_interaction_pearson'])} | "
            f"{fmt(row['mean_interaction_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Status: `{payload['decision']['status']}`",
            "",
            f"Next action: `{payload['decision']['next_action']}`",
            "",
            f"Reason: {payload['decision']['reason']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/cyx/1030/dataset/latentfm_full/scfoundation"))
    parser.add_argument("--split-file", type=Path, default=Path("/data/cyx/1030/dataset/biFlow_data/split_seed42.json"))
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_interaction_residual_feasibility_20260620.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_WESSELS_INTERACTION_RESIDUAL_FEASIBILITY_20260620.md"),
    )
    parser.add_argument(
        "--out-train-csv",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_interaction_residual_train_conditions_20260620.csv"),
    )
    parser.add_argument(
        "--out-wessels-csv",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_wessels_interaction_residual_wessels_conditions_20260620.csv"),
    )
    args = parser.parse_args()

    split = json.loads(args.split_file.read_text(encoding="utf-8"))
    train_counts = {"single": 0, "multi": 0, "multi_full_single_prior": 0}
    for groups in split.values():
        for condition in groups.get("train", []):
            genes = parse_genes(condition)
            if len(genes) == 1:
                train_counts["single"] += 1
            elif len(genes) > 1:
                train_counts["multi"] += 1
    if train_counts["multi"] == 0:
        payload = {
            "data_dir": str(args.data_dir),
            "split_file": str(args.split_file),
            "train_counts": {**train_counts, "multi_full_single_prior": 0},
            "single_prior_gene_count": 0,
            "single_prior_meta": {},
            "residual_prior_by_n_genes": {},
            "wessels_group_rows": [],
            "decision": {
                "status": "no_train_multi_supervision",
                "next_action": "prioritize_residual_preprocessing_or_unsupervised_combo_architecture_diagnostic",
                "reason": "canonical train split contains no multi-condition records, so supervised interaction residual learning is not identifiable",
            },
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_csv(args.out_train_csv, [])
        write_csv(args.out_wessels_csv, [])
        write_md(args.out_md, payload)
        print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "decision": payload["decision"]}, indent=2))
        return 0

    target_genes = {
        gene
        for groups in split.values()
        for condition in groups.get("train", [])
        for gene in parse_genes(condition)
        if len(parse_genes(condition)) > 1
    }
    target_genes.update(
        gene
        for group in WESSELS_GROUPS
        for condition in split["Wessels"].get(group, [])
        for gene in parse_genes(condition)
    )
    single_priors, single_meta = build_single_priors(data_dir=args.data_dir, split=split, target_genes=target_genes)
    train_rows, residual_prior = build_train_residuals(data_dir=args.data_dir, split=split, single_priors=single_priors)
    train_counts["multi_full_single_prior"] = sum(1 for row in train_rows if row["has_full_single_prior"])
    wessels_rows, wessels_group_rows = evaluate_wessels(
        data_dir=args.data_dir,
        wessels_split=split["Wessels"],
        single_priors=single_priors,
        residual_prior=residual_prior,
    )
    unseen2 = next(row for row in wessels_group_rows if row["group"] == "test_multi_unseen2")
    unseen2_delta = unseen2.get("mean_interaction_delta")
    has_supervision = train_counts["multi_full_single_prior"] > 0
    if not has_supervision:
        decision = {
            "status": "no_train_multi_supervision",
            "next_action": "prioritize_residual_preprocessing_or_unsupervised_combo_architecture_diagnostic",
            "reason": "canonical train split contains no multi-condition residual records with full single-prior coverage",
        }
    elif unseen2_delta is not None and unseen2_delta >= 0.05:
        decision = {
            "status": "interaction_prior_signal",
            "next_action": "design_train_only_interaction_residual_gpu_probe",
            "reason": "train-only residual prior improves Wessels unseen2 additive prior by at least +0.05",
        }
    else:
        decision = {
            "status": "interaction_prior_no_signal",
            "next_action": "prioritize_residual_preprocessing_or_unsupervised_combo_architecture_diagnostic",
            "reason": "train-only residual prior does not improve Wessels unseen2 enough for GPU launch",
        }
    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "train_counts": train_counts,
        "single_prior_gene_count": len(single_priors),
        "single_prior_meta": single_meta,
        "residual_prior_by_n_genes": {str(k): {"n_records": sum(1 for row in train_rows if row["n_genes"] == k and row["has_full_single_prior"])} for k in residual_prior},
        "wessels_group_rows": wessels_group_rows,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.out_train_csv, train_rows)
    write_csv(args.out_wessels_csv, wessels_rows)
    write_md(args.out_md, payload)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
