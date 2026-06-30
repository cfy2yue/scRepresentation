#!/usr/bin/env python3
"""Materialize train-only condition/residual information metrics for LatentFM.

CPU-only. This reads split JSONs, xVERSE H5 latent bundles, and condition
metadata. It does not train, infer, read canonical multi, read Track C query, or
select checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_INFO_CSV = ROOT / "reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_COND_META = DEFAULT_DATA_DIR / "condition_metadata.json"
DEFAULT_OUT_DIR = ROOT / "reports/trainonly_condition_residual_information_20260628"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRAINONLY_CONDITION_RESIDUAL_INFORMATION_20260628.md"


@dataclass(frozen=True)
class ConditionVectors:
    dataset: str
    condition: str
    ctrl_mean: np.ndarray
    gt_mean: np.ndarray
    n_ctrl: int
    n_gt: int

    @property
    def residual(self) -> np.ndarray:
        return self.gt_mean - self.ctrl_mean


def set_low_thread_env() -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ.setdefault(key, "1")


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def decode_conditions(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode())
        else:
            out.append(str(value))
    return out


def read_split_rows(info_csv: Path, max_splits: int) -> list[dict[str, str]]:
    with info_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if row.get("split_file")]
    if max_splits > 0:
        rows = rows[:max_splits]
    return rows


def collect_needed_conditions(split_rows: list[dict[str, str]]) -> dict[str, set[str]]:
    needed: dict[str, set[str]] = {}
    for row in split_rows:
        split_path = ROOT / row["split_file"]
        split = load_json(split_path)
        for dataset, groups in split.items():
            train_conditions = groups.get("train", [])
            if not train_conditions:
                continue
            needed.setdefault(dataset, set()).update(str(c) for c in train_conditions)
    return needed


def mean_for_offsets(emb: h5py.Dataset, offsets: np.ndarray, idx: int) -> tuple[np.ndarray, int]:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    n = end - start
    if n <= 0:
        return np.full((emb.shape[1],), np.nan, dtype=np.float32), 0
    return np.asarray(emb[start:end], dtype=np.float32).mean(axis=0), n


def load_condition_vector_cache(
    data_dir: Path,
    needed: dict[str, set[str]],
) -> tuple[dict[str, dict[str, ConditionVectors]], list[dict[str, Any]]]:
    cache: dict[str, dict[str, ConditionVectors]] = {}
    missing_rows: list[dict[str, Any]] = []
    for dataset in sorted(needed):
        h5_path = data_dir / f"{dataset}.h5"
        if not h5_path.exists():
            missing_rows.append({"dataset": dataset, "condition": "", "reason": "missing_h5"})
            continue
        cache[dataset] = {}
        with h5py.File(h5_path, "r") as h5:
            h5_conditions = decode_conditions(h5["conditions"][:])
            index = {condition: idx for idx, condition in enumerate(h5_conditions)}
            ctrl_emb = h5["ctrl/emb"]
            gt_emb = h5["gt/emb"]
            ctrl_offsets = np.asarray(h5["ctrl/offsets"])
            gt_offsets = np.asarray(h5["gt/offsets"])
            for condition in sorted(needed[dataset]):
                idx = index.get(condition)
                if idx is None:
                    missing_rows.append({"dataset": dataset, "condition": condition, "reason": "missing_condition"})
                    continue
                ctrl_mean, n_ctrl = mean_for_offsets(ctrl_emb, ctrl_offsets, idx)
                gt_mean, n_gt = mean_for_offsets(gt_emb, gt_offsets, idx)
                if n_ctrl <= 0 or n_gt <= 0:
                    missing_rows.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "reason": f"empty_slice_ctrl{n_ctrl}_gt{n_gt}",
                        }
                    )
                    continue
                cache[dataset][condition] = ConditionVectors(dataset, condition, ctrl_mean, gt_mean, n_ctrl, n_gt)
    return cache, missing_rows


def entropy(values: list[str]) -> dict[str, float]:
    if not values:
        return {"n_unique": 0.0, "entropy_norm": 0.0, "effective_count": 0.0, "max_share": 0.0}
    counts = Counter(values)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    n_unique = len(counts)
    return {
        "n_unique": float(n_unique),
        "entropy_norm": ent / math.log(n_unique) if n_unique > 1 else 0.0,
        "effective_count": math.exp(ent) if total else 0.0,
        "max_share": max(probs) if probs else 0.0,
    }


def effective_rank(matrix: np.ndarray) -> dict[str, float]:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"effective_rank": 0.0, "rank_entropy_norm": 0.0}
    x = matrix.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(x, compute_uv=False)
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.size == 0:
        return {"effective_rank": 0.0, "rank_entropy_norm": 0.0}
    probs = singular_values / singular_values.sum()
    ent = -float(np.sum(probs * np.log(probs)))
    return {
        "effective_rank": math.exp(ent),
        "rank_entropy_norm": ent / math.log(len(probs)) if len(probs) > 1 else 0.0,
    }


def pairwise_stats(matrix: np.ndarray, max_points: int, seed: int) -> dict[str, float]:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {
            "pairwise_l2_mean": 0.0,
            "pairwise_l2_median": 0.0,
            "pairwise_cosine_distance_mean": 0.0,
            "sampled_points": float(matrix.shape[0]) if matrix.ndim == 2 else 0.0,
        }
    x = matrix.astype(np.float64, copy=False)
    n = x.shape[0]
    if max_points > 0 and n > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        x = x[idx]
    diffs = x[:, None, :] - x[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    tri = np.triu_indices_from(dists, k=1)
    pair_l2 = dists[tri]
    norms = np.linalg.norm(x, axis=1)
    denom = np.maximum(norms[:, None] * norms[None, :], 1e-12)
    cosine = (x @ x.T) / denom
    cosine_dist = 1.0 - cosine[tri]
    return {
        "pairwise_l2_mean": float(pair_l2.mean()) if pair_l2.size else 0.0,
        "pairwise_l2_median": float(np.median(pair_l2)) if pair_l2.size else 0.0,
        "pairwise_cosine_distance_mean": float(cosine_dist.mean()) if cosine_dist.size else 0.0,
        "sampled_points": float(x.shape[0]),
    }


def vendi_rbf(matrix: np.ndarray, max_points: int, seed: int) -> dict[str, float]:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"vendi_rbf_effective_count": 0.0, "vendi_rbf_sigma": 0.0, "vendi_sampled_points": 0.0}
    x = matrix.astype(np.float64, copy=False)
    n = x.shape[0]
    if max_points > 0 and n > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        x = x[idx]
    diffs = x[:, None, :] - x[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    tri = d2[np.triu_indices_from(d2, k=1)]
    positive = tri[tri > 1e-12]
    if positive.size == 0:
        return {"vendi_rbf_effective_count": 1.0, "vendi_rbf_sigma": 0.0, "vendi_sampled_points": float(x.shape[0])}
    sigma2 = float(np.median(positive))
    kernel = np.exp(-d2 / max(2.0 * sigma2, 1e-12))
    kernel = (kernel + kernel.T) / 2.0
    kernel /= max(float(np.trace(kernel)), 1e-12)
    eigvals = np.linalg.eigvalsh(kernel)
    eigvals = eigvals[eigvals > 1e-12]
    if eigvals.size == 0:
        eff = 0.0
    else:
        eigvals = eigvals / eigvals.sum()
        ent = -float(np.sum(eigvals * np.log(eigvals)))
        eff = math.exp(ent)
    return {
        "vendi_rbf_effective_count": eff,
        "vendi_rbf_sigma": math.sqrt(sigma2),
        "vendi_sampled_points": float(x.shape[0]),
    }


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_split(
    split_row: dict[str, str],
    cache: dict[str, dict[str, ConditionVectors]],
    cond_meta: dict[str, dict[str, dict[str, Any]]],
    max_pairwise_points: int,
    seed: int,
) -> dict[str, Any]:
    split_path = ROOT / split_row["split_file"]
    split = load_json(split_path)
    residuals: list[np.ndarray] = []
    ctrl_centers: list[np.ndarray] = []
    gt_centers: list[np.ndarray] = []
    response_norms: list[float] = []
    n_ctrl_cells: list[int] = []
    n_gt_cells: list[int] = []
    dataset_labels: list[str] = []
    perturbation_labels: list[str] = []
    gene_labels: list[str] = []
    missing = 0

    for dataset, groups in split.items():
        for condition in groups.get("train", []):
            vectors = cache.get(dataset, {}).get(str(condition))
            if vectors is None:
                missing += 1
                continue
            residual = vectors.residual
            residuals.append(residual)
            ctrl_centers.append(vectors.ctrl_mean)
            gt_centers.append(vectors.gt_mean)
            response_norms.append(float(np.linalg.norm(residual)))
            n_ctrl_cells.append(vectors.n_ctrl)
            n_gt_cells.append(vectors.n_gt)
            dataset_labels.append(dataset)
            meta = cond_meta.get(dataset, {}).get(str(condition), {})
            perturbation_labels.append(str(meta.get("perturbation_type_raw", "unknown")))
            for gene in meta.get("genes", []):
                gene_labels.append(str(gene).upper())

    if residuals:
        residual_matrix = np.vstack(residuals)
        ctrl_matrix = np.vstack(ctrl_centers)
        gt_matrix = np.vstack(gt_centers)
    else:
        residual_matrix = np.zeros((0, 0), dtype=np.float32)
        ctrl_matrix = np.zeros((0, 0), dtype=np.float32)
        gt_matrix = np.zeros((0, 0), dtype=np.float32)

    residual_erank = effective_rank(residual_matrix)
    ctrl_erank = effective_rank(ctrl_matrix)
    gt_erank = effective_rank(gt_matrix)
    residual_pairwise = pairwise_stats(residual_matrix, max_pairwise_points, seed)
    residual_vendi = vendi_rbf(residual_matrix, max_pairwise_points, seed)
    ctrl_pairwise = pairwise_stats(ctrl_matrix, max_pairwise_points, seed)
    gt_pairwise = pairwise_stats(gt_matrix, max_pairwise_points, seed)
    ds_entropy = entropy(dataset_labels)
    ptype_entropy = entropy(perturbation_labels)
    gene_entropy = entropy(gene_labels)

    row: dict[str, Any] = {
        "split_file": split_row["split_file"],
        "split_name": split_row["split_name"],
        "n_train_conditions_declared": int(safe_float(split_row.get("n_train_conditions", 0))),
        "n_train_conditions_with_vectors": len(residuals),
        "n_missing_vectors": missing,
        "n_dataset_labels": int(ds_entropy["n_unique"]),
        "dataset_effective_count_condition_vectors": ds_entropy["effective_count"],
        "dataset_entropy_norm_condition_vectors": ds_entropy["entropy_norm"],
        "n_perturbation_types": int(ptype_entropy["n_unique"]),
        "perturbation_type_effective_count_condition_vectors": ptype_entropy["effective_count"],
        "n_target_genes_condition_vectors": int(gene_entropy["n_unique"]),
        "target_gene_effective_count_condition_vectors": gene_entropy["effective_count"],
        "response_norm_mean": float(np.mean(response_norms)) if response_norms else 0.0,
        "response_norm_median": float(np.median(response_norms)) if response_norms else 0.0,
        "response_norm_cv": float(np.std(response_norms) / max(np.mean(response_norms), 1e-12)) if response_norms else 0.0,
        "ctrl_cells_mean": float(np.mean(n_ctrl_cells)) if n_ctrl_cells else 0.0,
        "gt_cells_mean": float(np.mean(n_gt_cells)) if n_gt_cells else 0.0,
        "residual_effective_rank": residual_erank["effective_rank"],
        "residual_rank_entropy_norm": residual_erank["rank_entropy_norm"],
        "ctrl_center_effective_rank": ctrl_erank["effective_rank"],
        "ctrl_center_rank_entropy_norm": ctrl_erank["rank_entropy_norm"],
        "gt_center_effective_rank": gt_erank["effective_rank"],
        "gt_center_rank_entropy_norm": gt_erank["rank_entropy_norm"],
        "residual_pairwise_l2_mean": residual_pairwise["pairwise_l2_mean"],
        "residual_pairwise_l2_median": residual_pairwise["pairwise_l2_median"],
        "residual_pairwise_cosine_distance_mean": residual_pairwise["pairwise_cosine_distance_mean"],
        "residual_pairwise_sampled_points": residual_pairwise["sampled_points"],
        "residual_vendi_rbf_effective_count": residual_vendi["vendi_rbf_effective_count"],
        "residual_vendi_rbf_sigma": residual_vendi["vendi_rbf_sigma"],
        "residual_vendi_sampled_points": residual_vendi["vendi_sampled_points"],
        "ctrl_pairwise_l2_mean": ctrl_pairwise["pairwise_l2_mean"],
        "gt_pairwise_l2_mean": gt_pairwise["pairwise_l2_mean"],
    }
    for key in [
        "dataset_effective_count",
        "background_effective_count",
        "perturbation_type_effective_count",
        "target_gene_effective_count",
        "dataset_mean_effective_rank",
        "dataset_mean_pairwise_l2",
    ]:
        if key in split_row:
            row[f"preflight_{key}"] = safe_float(split_row.get(key))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def build_report(
    out_md: Path,
    rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    out_csv: Path,
    out_json: Path,
    meta: dict[str, Any],
) -> None:
    top_vendi = sorted(rows, key=lambda r: float(r["residual_vendi_rbf_effective_count"]), reverse=True)[:8]
    top_rank = sorted(rows, key=lambda r: float(r["residual_effective_rank"]), reverse=True)[:8]
    top_response = sorted(rows, key=lambda r: float(r["response_norm_mean"]), reverse=True)[:8]

    lines = [
        "# LatentFM Train-Only Condition Residual Information",
        "",
        f"Timestamp: `{meta['timestamp']}`",
        "",
        "Status: `condition_residual_information_materialized_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only over train-condition xVERSE H5 latent bundles and split JSONs.",
        "- Uses only `train` conditions from each split.",
        "- Does not train, infer, use canonical multi, use Track C held-out query, or select checkpoints.",
        "- Geometry is measured in existing xVERSE latent bundle space; it is an information-axis diagnostic, not a model-promotion result.",
        "",
        "## Summary",
        "",
        f"- Split rows processed: `{len(rows)}`.",
        f"- Datasets with requested vectors: `{meta['datasets_requested']}`.",
        f"- Condition vectors materialized: `{meta['condition_vectors_materialized']}`.",
        f"- Missing/empty requested vectors: `{len(missing_rows)}`.",
        f"- Pairwise/Vendi sampling cap: `{meta['max_pairwise_points']}` conditions per split.",
        "",
        "## Highest Residual Vendi Arms",
        "",
        "| split | train vectors | residual Vendi | residual eff. rank | response norm mean | dataset eff. count | ptype eff. count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_vendi:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                row["split_name"],
                row["n_train_conditions_with_vectors"],
                fmt(row["residual_vendi_rbf_effective_count"]),
                fmt(row["residual_effective_rank"]),
                fmt(row["response_norm_mean"]),
                fmt(row["dataset_effective_count_condition_vectors"]),
                fmt(row["perturbation_type_effective_count_condition_vectors"]),
            )
        )

    lines.extend(
        [
            "",
            "## Highest Residual Effective-Rank Arms",
            "",
            "| split | train vectors | residual eff. rank | residual rank entropy | residual pairwise L2 | residual Vendi |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_rank:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                row["split_name"],
                row["n_train_conditions_with_vectors"],
                fmt(row["residual_effective_rank"]),
                fmt(row["residual_rank_entropy_norm"]),
                fmt(row["residual_pairwise_l2_mean"]),
                fmt(row["residual_vendi_rbf_effective_count"]),
            )
        )

    lines.extend(
        [
            "",
            "## Largest Mean Response-Norm Arms",
            "",
            "| split | train vectors | response norm mean | response norm CV | residual Vendi | perturbation type eff. count |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_response:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                row["split_name"],
                row["n_train_conditions_with_vectors"],
                fmt(row["response_norm_mean"]),
                fmt(row["response_norm_cv"]),
                fmt(row["residual_vendi_rbf_effective_count"]),
                fmt(row["perturbation_type_effective_count_condition_vectors"]),
            )
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Condition/residual-level train-only geometry is now available as a downstream-scaling candidate x-axis. The immediate gate is not GPU training; it is to join these metrics against frozen downstream outcome rows and identify matched split pairs where raw condition count is similar but residual information differs.",
            "",
            "Promotion to GPU should require a predeclared matched-pair hypothesis, leakage-safe split boundary, and Track A no-harm/promotion rule. This report only nominates measurable axes.",
            "",
            "## Outputs",
            "",
            f"- CSV: `{out_csv}`",
            f"- JSON: `{out_json}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-csv", type=Path, default=DEFAULT_INFO_CSV)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--condition-metadata", type=Path, default=DEFAULT_COND_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--max-splits", type=int, default=0)
    parser.add_argument("--max-pairwise-points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    set_low_thread_env()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "trainonly_condition_residual_information_rows.csv"
    out_missing = args.out_dir / "trainonly_condition_residual_missing_vectors.csv"
    out_json = args.out_dir / "trainonly_condition_residual_information_20260628.json"

    split_rows = read_split_rows(args.info_csv, args.max_splits)
    needed = collect_needed_conditions(split_rows)
    cache, missing_rows = load_condition_vector_cache(args.data_dir, needed)
    cond_meta = load_json(args.condition_metadata)
    rows = [
        summarize_split(row, cache, cond_meta, args.max_pairwise_points, args.seed + idx)
        for idx, row in enumerate(split_rows)
    ]

    write_csv(out_csv, rows)
    write_csv(out_missing, missing_rows)
    meta = {
        "timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "info_csv": str(args.info_csv),
        "data_dir": str(args.data_dir),
        "condition_metadata": str(args.condition_metadata),
        "max_splits": args.max_splits,
        "max_pairwise_points": args.max_pairwise_points,
        "seed": args.seed,
        "datasets_requested": len(needed),
        "condition_vectors_materialized": sum(len(v) for v in cache.values()),
        "missing_rows": len(missing_rows),
        "status": "condition_residual_information_materialized_no_gpu",
        "gpu_authorized": False,
    }
    write_json(out_json, {"meta": meta, "rows": rows, "missing_rows": missing_rows})
    build_report(args.out_md, rows, missing_rows, out_csv, out_json, meta)


if __name__ == "__main__":
    main()
