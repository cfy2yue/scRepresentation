#!/usr/bin/env python3
"""CPU-only preflight for downstream perturbation information-scaling metrics.

This script only reads split JSONs, train-only condition metadata, and existing
mean-artifact NPZ files. It does not train, infer, read canonical multi, read
Track C held-out query, or use GPU.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports" / "downstream_information_scaling_preflight_20260628"
OUT_MD = ROOT / "reports" / "LATENTFM_DOWNSTREAM_INFORMATION_SCALING_PREFLIGHT_20260628.md"
OUT_JSON = ROOT / "reports" / "latentfm_downstream_information_scaling_preflight_20260628.json"
OUT_CSV = OUT_DIR / "split_information_metrics.csv"

SPLIT_GLOBS = [
    "dataset/biFlow_data/xverse_scaling_splits_20260624/*.json",
    "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/*.json",
    "dataset/biFlow_data/xverse_scaling_protocol_splits_20260624/*.json",
    "dataset/biFlow_data/xverse_true_cell_count_scaling_splits_20260624/*.json",
    "dataset/biFlow_data/xverse_true_cell_count_scaling_nested_splits_20260624/*.json",
    "dataset/biFlow_data/xverse_modality_pathway_sampling_splits_20260624/*.json",
]

ARTIFACT_GLOBS = [
    "runs/latentfm_xverse_scaling_splits_20260624/artifacts/*pert_means.npz",
    "runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/*pert_means.npz",
    "runs/latentfm_scaling_protocol_splits_20260624/artifacts/*pert_means.npz",
    "runs/latentfm_true_cell_count_scaling_capped_h5_20260624/artifacts/**/pert_means.npz",
    "runs/latentfm_true_cell_count_scaling_nested_capped_h5_20260624/artifacts/**/pert_means.npz",
    "runs/latentfm_modality_pathway_sampling_artifacts_20260624/artifacts/*pert_means.npz",
    "runs/latentfm_modality_pathway_mmd_preservation_artifacts_20260624/artifacts/*pert_means.npz",
]

STOP_TOKENS = {
    "split",
    "seed42",
    "seed43",
    "seed44",
    "xverse",
    "trainonly",
    "scaling",
    "latentfm",
    "artifacts",
    "pert",
    "means",
    "json",
    "npz",
}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def entropy_metrics(values: list[str]) -> dict[str, float]:
    if not values:
        return {
            "n_unique": 0.0,
            "entropy": 0.0,
            "entropy_norm": 0.0,
            "effective_count": 0.0,
            "max_share": 0.0,
        }
    counts = Counter(values)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    n_unique = len(counts)
    return {
        "n_unique": float(n_unique),
        "entropy": entropy,
        "entropy_norm": entropy / math.log(n_unique) if n_unique > 1 else 0.0,
        "effective_count": math.exp(entropy) if total else 0.0,
        "max_share": max(probs) if probs else 0.0,
    }


def effective_rank(matrix: np.ndarray) -> dict[str, float]:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return {"effective_rank": 0.0, "rank_entropy_norm": 0.0, "mean_pairwise_l2": 0.0}
    x = matrix.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        singular_values = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"effective_rank": float("nan"), "rank_entropy_norm": float("nan"), "mean_pairwise_l2": float("nan")}
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.size == 0:
        eff = 0.0
        norm = 0.0
    else:
        probs = singular_values / singular_values.sum()
        ent = -float(np.sum(probs * np.log(probs)))
        eff = math.exp(ent)
        norm = ent / math.log(len(probs)) if len(probs) > 1 else 0.0
    diffs = x[:, None, :] - x[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    tri = dists[np.triu_indices_from(dists, k=1)]
    return {
        "effective_rank": eff,
        "rank_entropy_norm": norm,
        "mean_pairwise_l2": float(tri.mean()) if tri.size else 0.0,
    }


def tokenize(path: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", path.lower()))
    return {t for t in tokens if len(t) > 2 and t not in STOP_TOKENS}


def best_artifact_for_split(split_path: Path, artifact_paths: list[Path]) -> Path | None:
    base = re.sub(r"^split_seed\d+_", "", split_path.stem)
    for artifact_path in artifact_paths:
        if base in artifact_path.name:
            return artifact_path
    split_tokens = tokenize(split_path.stem)
    best: tuple[int, Path | None] = (0, None)
    for artifact_path in artifact_paths:
        artifact_tokens = tokenize(str(artifact_path.relative_to(ROOT)))
        score = len(split_tokens & artifact_tokens)
        if score > best[0]:
            best = (score, artifact_path)
    return best[1] if best[0] >= 3 else None


def dataset_meta() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for rel in [
        "dataset/raw/genepert_bench/metainfo.json",
        "dataset/raw/chemicalpert_DE5000/metainfo.json",
    ]:
        path = ROOT / rel
        if not path.exists():
            continue
        for item in load_json(path):
            ds = item.get("dataset")
            if not ds:
                continue
            rows[ds] = {
                "cell_line": str(item.get("cell_line", "unknown")),
                "perturbation_type": str(item.get("perturbation_type", "unknown")),
            }
    return rows


def condition_meta() -> dict[str, dict[str, dict[str, Any]]]:
    path = ROOT / "dataset/latentfm_full/scfoundation/condition_metadata.json"
    return load_json(path)


def split_metrics(
    split_path: Path,
    cond_meta: dict[str, dict[str, dict[str, Any]]],
    ds_meta: dict[str, dict[str, str]],
    artifact_path: Path | None,
) -> dict[str, Any]:
    split = load_json(split_path)
    dataset_labels: list[str] = []
    background_labels: list[str] = []
    type_labels: list[str] = []
    target_gene_labels: list[str] = []
    condition_keys: list[str] = []
    missing_condition_metadata = 0
    drug_conditions = 0
    gene_conditions = 0

    for ds, groups in split.items():
        train_conditions = groups.get("train", [])
        for cond in train_conditions:
            dataset_labels.append(ds)
            background_labels.append(ds_meta.get(ds, {}).get("cell_line", "unknown"))
            meta = cond_meta.get(ds, {}).get(cond)
            if meta is None:
                missing_condition_metadata += 1
                type_labels.append(ds_meta.get(ds, {}).get("perturbation_type", "unknown"))
                condition_keys.append(f"{ds}::{cond}")
                continue
            ptype = str(meta.get("perturbation_type_raw") or ds_meta.get(ds, {}).get("perturbation_type", "unknown"))
            type_labels.append(ptype)
            genes = [str(g).upper() for g in meta.get("genes", []) if str(g)]
            if ptype.lower() == "drug":
                drug_conditions += 1
            elif genes:
                gene_conditions += 1
            for gene in genes:
                target_gene_labels.append(gene)
            condition_keys.append(f"{ds}::{cond}")

    dataset_ent = entropy_metrics(dataset_labels)
    background_ent = entropy_metrics(background_labels)
    type_ent = entropy_metrics(type_labels)
    target_ent = entropy_metrics(target_gene_labels)

    row: dict[str, Any] = {
        "split_file": str(split_path.relative_to(ROOT)),
        "split_name": split_path.stem,
        "n_train_conditions": len(condition_keys),
        "n_dataset_labels": int(dataset_ent["n_unique"]),
        "dataset_entropy_norm": dataset_ent["entropy_norm"],
        "dataset_effective_count": dataset_ent["effective_count"],
        "max_dataset_share": dataset_ent["max_share"],
        "n_background_labels": int(background_ent["n_unique"]),
        "background_entropy_norm": background_ent["entropy_norm"],
        "background_effective_count": background_ent["effective_count"],
        "max_background_share": background_ent["max_share"],
        "n_perturbation_types": int(type_ent["n_unique"]),
        "perturbation_type_entropy_norm": type_ent["entropy_norm"],
        "perturbation_type_effective_count": type_ent["effective_count"],
        "max_perturbation_type_share": type_ent["max_share"],
        "n_target_genes": int(target_ent["n_unique"]),
        "target_gene_entropy_norm": target_ent["entropy_norm"],
        "target_gene_effective_count": target_ent["effective_count"],
        "drug_condition_fraction": drug_conditions / len(condition_keys) if condition_keys else 0.0,
        "gene_condition_fraction": gene_conditions / len(condition_keys) if condition_keys else 0.0,
        "missing_condition_metadata": missing_condition_metadata,
        "artifact_path": str(artifact_path.relative_to(ROOT)) if artifact_path else "",
        "geometry_level": "dataset_mean" if artifact_path else "",
        "dataset_mean_effective_rank": "",
        "dataset_mean_rank_entropy_norm": "",
        "dataset_mean_pairwise_l2": "",
    }

    if artifact_path:
        try:
            artifact = np.load(artifact_path, allow_pickle=True)
            vectors = []
            for ds in split:
                if ds in artifact.files and len(split[ds].get("train", [])) > 0:
                    vectors.append(np.asarray(artifact[ds]).reshape(-1))
            if len(vectors) >= 2:
                geom = effective_rank(np.vstack(vectors))
                row["dataset_mean_effective_rank"] = geom["effective_rank"]
                row["dataset_mean_rank_entropy_norm"] = geom["rank_entropy_norm"]
                row["dataset_mean_pairwise_l2"] = geom["mean_pairwise_l2"]
        except Exception as exc:  # noqa: BLE001 - report artifact issues, do not fail the preflight.
            row["artifact_error"] = repr(exc)

    return row


def fmt(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = sorted({Path(p) for pattern in SPLIT_GLOBS for p in glob.glob(str(ROOT / pattern))})
    artifacts = sorted({Path(p) for pattern in ARTIFACT_GLOBS for p in glob.glob(str(ROOT / pattern), recursive=True)})
    cond_meta = condition_meta()
    ds_meta = dataset_meta()

    rows = []
    for split_path in splits:
        artifact_path = best_artifact_for_split(split_path, artifacts)
        rows.append(split_metrics(split_path, cond_meta, ds_meta, artifact_path))

    fieldnames = [
        "split_file",
        "split_name",
        "n_train_conditions",
        "n_dataset_labels",
        "dataset_entropy_norm",
        "dataset_effective_count",
        "max_dataset_share",
        "n_background_labels",
        "background_entropy_norm",
        "background_effective_count",
        "max_background_share",
        "n_perturbation_types",
        "perturbation_type_entropy_norm",
        "perturbation_type_effective_count",
        "max_perturbation_type_share",
        "n_target_genes",
        "target_gene_entropy_norm",
        "target_gene_effective_count",
        "drug_condition_fraction",
        "gene_condition_fraction",
        "missing_condition_metadata",
        "artifact_path",
        "geometry_level",
        "dataset_mean_effective_rank",
        "dataset_mean_rank_entropy_norm",
        "dataset_mean_pairwise_l2",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    rows_with_artifact = [r for r in rows if r.get("artifact_path")]
    geometry_ready = [r for r in rows if r.get("dataset_mean_effective_rank") != ""]
    missing_total = sum(int(r["missing_condition_metadata"]) for r in rows)
    top_by_conditions = sorted(rows, key=lambda r: int(r["n_train_conditions"]), reverse=True)[:8]
    top_by_dataset_entropy = sorted(rows, key=lambda r: float(r["dataset_effective_count"]), reverse=True)[:8]

    payload = {
        "status": "downstream_information_scaling_preflight_ready_no_gpu",
        "gpu_authorized": False,
        "n_split_files": len(rows),
        "n_artifact_files_seen": len(artifacts),
        "n_splits_with_artifact_match": len(rows_with_artifact),
        "n_splits_with_dataset_mean_geometry": len(geometry_ready),
        "missing_condition_metadata_total": missing_total,
        "csv": str(OUT_CSV),
        "roadmap": str(ROOT / "reports/LATENTFM_INFORMATION_SCALING_DYNAMIC_TRAJECTORY_ROADMAP_20260628.md"),
    }
    with OUT_JSON.open("w") as f:
        json.dump({"summary": payload, "rows": rows}, f, indent=2)

    lines = [
        "# LatentFM Downstream Information Scaling Preflight",
        "",
        "Timestamp: `2026-06-28 04:18 CST`",
        "",
        "Status: `downstream_information_scaling_preflight_ready_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over train-only split files, condition metadata, and existing train-only mean artifacts.",
        "- Does not train, infer, read canonical multi, read Track C held-out query, select checkpoints, or use GPU.",
        "- Dataset-mean geometry is only a preflight proxy. It is not yet the required condition/residual-level Vendi/effective-rank metric.",
        "",
        "## Summary",
        "",
        f"- Split files audited: `{payload['n_split_files']}`.",
        f"- Candidate artifact files seen: `{payload['n_artifact_files_seen']}`.",
        f"- Split files with heuristic mean-artifact match: `{payload['n_splits_with_artifact_match']}`.",
        f"- Split files with dataset-mean geometry available: `{payload['n_splits_with_dataset_mean_geometry']}`.",
        f"- Missing condition metadata entries across audited splits: `{payload['missing_condition_metadata_total']}`.",
        "",
        "## Largest Train-Support Arms",
        "",
        "| split | train conditions | datasets | max dataset share | perturbation type eff. count | target genes | artifact |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in top_by_conditions:
        lines.append(
            "| `{split}` | {n} | {ds} | {share} | {ptype} | {genes} | `{artifact}` |".format(
                split=row["split_name"],
                n=row["n_train_conditions"],
                ds=row["n_dataset_labels"],
                share=fmt(row["max_dataset_share"]),
                ptype=fmt(row["perturbation_type_effective_count"]),
                genes=row["n_target_genes"],
                artifact=os.path.basename(row["artifact_path"]) if row["artifact_path"] else "",
            )
        )
    lines.extend(
        [
            "",
            "## Highest Dataset-Entropy Arms",
            "",
            "| split | train conditions | dataset eff. count | background eff. count | max dataset share | dataset-mean eff. rank |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_by_dataset_entropy:
        lines.append(
            "| `{split}` | {n} | {deff} | {beff} | {share} | {rank} |".format(
                split=row["split_name"],
                n=row["n_train_conditions"],
                deff=fmt(row["dataset_effective_count"]),
                beff=fmt(row["background_effective_count"]),
                share=fmt(row["max_dataset_share"]),
                rank=fmt(row["dataset_mean_effective_rank"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This preflight confirms that split-level composition metrics are immediately computable from current assets. The next gate should add condition/residual-level geometry where available and perform matched/LODO association against frozen Track A/source-control/scaling outcome tables.",
            "",
            "No GPU is authorized by this preflight. A GPU smoke requires a follow-up association gate that nominates an equal-cell-count/different-info or matched-entropy split and predeclares dual-baseline pp/MMD/tail promotion rules.",
            "",
            "## Outputs",
            "",
            f"- CSV: `{OUT_CSV}`",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
