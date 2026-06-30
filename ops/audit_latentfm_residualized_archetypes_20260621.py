#!/usr/bin/env python3
"""CPU-only audit for dataset-residualized control-latent archetypes.

This is a no-leakage proof-of-concept for biology/state archetypes.  It fits
clusters from source/control latent cells only.  Ground-truth perturbation
embeddings, perturbed means, posthoc predictions, and held-out response
residuals are intentionally unused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


FOCUS_DATASETS = ("Wessels", "NormanWeissman2019_filtered", "GasperiniShendure2019_lowMOI")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def pick_indices(n: int, k: int, *, key: str) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(stable_seed(key))
    return np.sort(rng.choice(n, size=k, replace=False).astype(np.int64))


def pick_slices(n: int, k: int, *, key: str, chunks: int = 8) -> list[slice]:
    if n <= k:
        return [slice(0, n)]
    chunks = max(1, min(chunks, k))
    chunk_size = max(1, k // chunks)
    rng = np.random.default_rng(stable_seed(key))
    starts = rng.choice(max(1, n - chunk_size), size=chunks, replace=False)
    slices = [slice(int(start), int(min(start + chunk_size, n))) for start in sorted(starts)]
    total = sum(s.stop - s.start for s in slices)
    if total < k:
        last_stop = slices[-1].stop
        extra_stop = min(n, last_stop + (k - total))
        slices[-1] = slice(slices[-1].start, extra_stop)
    return slices


def read_sampled_rows(dataset: h5py.Dataset, k: int, *, key: str) -> np.ndarray:
    parts = [np.asarray(dataset[s], dtype=np.float32) for s in pick_slices(int(dataset.shape[0]), k, key=key)]
    arr = np.concatenate(parts, axis=0)
    if arr.shape[0] > k:
        arr = arr[:k]
    return arr


def decode(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values:
        out.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return out


def normalized_entropy_from_counts(counts: Counter[Any], k: int) -> float:
    total = sum(counts.values())
    if total <= 0 or k <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            ent -= p * math.log(p)
    return float(ent / math.log(k))


def sample_controls(
    data_dir: Path,
    manifest: dict[str, Any],
    max_cells_per_dataset: int,
    residualization: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    per_dataset: dict[str, Any] = {}
    for dataset in sorted(manifest.get("datasets", {})):
        path = data_dir / f"{dataset}.h5"
        with h5py.File(path, "r") as handle:
            emb = handle["ctrl/emb"]
            arr = read_sampled_rows(
                emb,
                max_cells_per_dataset,
                key=f"resid_arch|{dataset}|{max_cells_per_dataset}",
            )
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        if residualization == "raw":
            z = arr
        elif residualization == "dataset_center":
            z = arr - mean
        elif residualization == "dataset_center_scale":
            z = (arr - mean) / np.maximum(std, 1e-3)
        else:
            raise ValueError(f"unknown residualization: {residualization}")
        arrays.append(z.astype(np.float32))
        labels.extend([dataset] * z.shape[0])
        per_dataset[dataset] = {
            "sampled_controls": int(z.shape[0]),
            "ctrl_emb_shape": [int(arr.shape[0]), int(arr.shape[1])],
            "mean_norm": float(np.linalg.norm(mean)),
            "median_feature_std": float(np.median(std)),
        }
    return np.concatenate(arrays, axis=0), np.asarray(labels), per_dataset


def fit_representation(x: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    xs = scaler.fit_transform(x)
    n_components = min(n_components, xs.shape[0] - 1, xs.shape[1])
    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(xs)
    return z.astype(np.float32), scaler, pca


def cluster_dataset_rows(labels: np.ndarray, datasets: np.ndarray) -> tuple[float, dict[str, Any]]:
    rows: dict[str, Any] = {}
    dominant_total = 0
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        counts = Counter(datasets[mask].tolist())
        dominant, n_dom = counts.most_common(1)[0]
        dominant_total += n_dom
        rows[str(cluster)] = {
            "size": int(mask.sum()),
            "dominant_dataset": dominant,
            "dominant_fraction": float(n_dom / mask.sum()),
            "dataset_entropy": normalized_entropy_from_counts(counts, len(set(datasets.tolist()))),
            "n_datasets_with_cells": int(len(counts)),
            "top_datasets": [[str(k), int(v)] for k, v in counts.most_common(6)],
        }
    return float(dominant_total / len(labels)), rows


def focus_rows(labels: np.ndarray, datasets: np.ndarray, k: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in FOCUS_DATASETS:
        mask = datasets == dataset
        counts = Counter(labels[mask].tolist())
        total = int(mask.sum())
        max_fraction = (max(counts.values()) / total) if total else None
        out[dataset] = {
            "n_sampled_controls": total,
            "n_clusters_used": int(len(counts)),
            "normalized_entropy": normalized_entropy_from_counts(counts, k),
            "max_cluster_fraction": max_fraction,
            "cluster_counts": {str(key): int(value) for key, value in sorted(counts.items())},
        }
    return out


def condition_archetype_coverage(
    data_dir: Path,
    split: dict[str, Any],
    scaler: StandardScaler,
    pca: PCA,
    kmeans: MiniBatchKMeans,
    k: int,
    residualization: str,
    max_conditions_per_dataset: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for dataset in FOCUS_DATASETS:
        path = data_dir / f"{dataset}.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            conditions = decode(np.asarray(handle["conditions"]))
            offsets = np.asarray(handle["ctrl/offsets"])
            ctrl = handle["ctrl/emb"]
            # Residualize condition means with dataset-level source cells only.
            ds_sample = read_sampled_rows(
                ctrl,
                min(int(ctrl.shape[0]), 4096),
                key=f"cond_resid|{dataset}",
            )
            ds_mean = ds_sample.mean(axis=0, keepdims=True)
            ds_std = ds_sample.std(axis=0, keepdims=True)
            selected = conditions[:max_conditions_per_dataset]
            labels: list[int] = []
            split_tags: list[str] = []
            split_obj = split.get(dataset, {})
            split_by_cond = {}
            for tag in ("train", "test", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"):
                for cond in split_obj.get(tag, []):
                    split_by_cond[str(cond)] = tag
            for cond in selected:
                idx = conditions.index(cond)
                c0, c1 = int(offsets[idx]), int(offsets[idx + 1])
                if c1 <= c0:
                    continue
                mean = np.asarray(ctrl[c0:c1], dtype=np.float32).mean(axis=0, keepdims=True)
                if residualization == "raw":
                    x = mean
                elif residualization == "dataset_center":
                    x = mean - ds_mean
                elif residualization == "dataset_center_scale":
                    x = (mean - ds_mean) / np.maximum(ds_std, 1e-3)
                else:
                    raise ValueError(f"unknown residualization: {residualization}")
                lab = int(kmeans.predict(pca.transform(scaler.transform(x)))[0])
                labels.append(lab)
                split_tags.append(split_by_cond.get(cond, "unknown"))
        by_split: dict[str, Any] = {}
        for tag in sorted(set(split_tags)):
            vals = [lab for lab, split_tag in zip(labels, split_tags) if split_tag == tag]
            counts = Counter(vals)
            by_split[tag] = {
                "n_conditions": int(len(vals)),
                "n_clusters_used": int(len(counts)),
                "normalized_entropy": normalized_entropy_from_counts(counts, k),
                "cluster_counts": {str(key): int(value) for key, value in sorted(counts.items())},
            }
        rows[dataset] = by_split
    return rows


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Residualized Control Archetype CPU Audit",
        "",
        "Status: `{}`".format(payload["overall_status"]),
        "",
        "This CPU-only gate tests whether control/source latent cells contain",
        "stable cross-dataset archetypes after removing dataset-level location",
        "effects. It does not use perturbed GT, `pert_means.npz`, posthoc",
        "predictions, or held-out multi response residuals.",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- max_cells_per_dataset: `{payload['max_cells_per_dataset']}`",
        f"- max_conditions_per_focus_dataset: `{payload['max_conditions_per_focus_dataset']}`",
        f"- pca_components: `{payload['pca_components']}`",
        f"- residualizations: `{payload['residualizations']}`",
        f"- forbidden_inputs_used: `{payload['forbidden_inputs_used']}`",
        "",
        "## Gate Summary",
        "",
        "| residualization | K | min cluster | max cluster | dataset purity | dataset NMI | seed ARI | focus entropy min | focus max-frac max | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        lines.append(
            "| {resid} | {k} | {mn} | {mx} | {pur:.4f} | {nmi:.4f} | {ari:.4f} | {fent:.4f} | {ffrac:.4f} | {gate} |".format(
                resid=row["residualization"],
                k=row["k"],
                mn=row["cluster_size_min"],
                mx=row["cluster_size_max"],
                pur=row["dataset_purity"],
                nmi=row["dataset_nmi"],
                ari=row["seed_ari_mean"],
                fent=row["focus_entropy_min"],
                ffrac=row["focus_max_cluster_fraction_max"],
                gate=row["gate_status"],
            )
        )
    lines += ["", "## Best Candidate", ""]
    best = payload.get("best_candidate")
    if best:
        lines += [
            f"- residualization: `{best['residualization']}`",
            f"- K: `{best['k']}`",
            f"- gate_status: `{best['gate_status']}`",
            f"- rationale: {best['rationale']}",
            "",
        ]
    else:
        lines += ["No candidate passed the predeclared CPU gate.", ""]
    lines += ["## Focus Dataset Coverage", ""]
    for row in payload["results"]:
        lines.append(f"### {row['residualization']} K={row['k']}")
        for dataset, obj in row["focus_distribution"].items():
            lines.append(
                f"- `{dataset}`: clusters={obj['n_clusters_used']}, "
                f"entropy={obj['normalized_entropy']:.4f}, "
                f"max_fraction={obj['max_cluster_fraction']:.4f}, "
                f"counts={obj['cluster_counts']}"
            )
        lines.append("")
    lines += [
        "## Decision Rule",
        "",
        "A GPU archetype-conditioned adapter can be considered only if at least",
        "one row is `candidate`: stable seed ARI, non-tiny clusters, low dataset",
        "NMI/purity, and each focus dataset uses multiple archetypes rather than",
        "collapsing to a dataset-specific label. Even then it remains a capped",
        "mechanism smoke, not a promotion candidate.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/cyx/1030/dataset/latentfm_full/scfoundation"))
    parser.add_argument("--split-file", type=Path, default=Path("/data/cyx/1030/dataset/biFlow_data/split_seed42.json"))
    parser.add_argument("--max-cells-per-dataset", type=int, default=512)
    parser.add_argument("--max-conditions-per-focus-dataset", type=int, default=256)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--kmeans-n-init", type=int, default=5)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument(
        "--residualizations",
        nargs="+",
        default=["dataset_center", "dataset_center_scale"],
        choices=["raw", "dataset_center", "dataset_center_scale"],
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/latentfm_residualized_archetype_cpu_audit_20260621.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_RESIDUALIZED_ARCHETYPE_CPU_AUDIT_20260621.md"),
    )
    args = parser.parse_args()

    manifest = load_json(args.data_dir / "manifest.json")
    split = load_json(args.split_file)
    results: list[dict[str, Any]] = []

    for residualization in args.residualizations:
        x, datasets, per_dataset = sample_controls(
            args.data_dir,
            manifest,
            args.max_cells_per_dataset,
            residualization,
        )
        z, scaler, pca = fit_representation(x, args.pca_components, seed=args.seeds[0])
        for k in args.ks:
            label_sets: list[np.ndarray] = []
            models: list[MiniBatchKMeans] = []
            for seed in args.seeds:
                km = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=seed,
                    batch_size=2048,
                    n_init=args.kmeans_n_init,
                    max_iter=args.kmeans_max_iter,
                )
                labels = km.fit_predict(z)
                label_sets.append(labels)
                models.append(km)
            labels = label_sets[0]
            sizes = Counter(labels.tolist())
            aris = [
                adjusted_rand_score(label_sets[i], label_sets[j])
                for i in range(len(label_sets))
                for j in range(i + 1, len(label_sets))
            ]
            dataset_purity, cluster_rows = cluster_dataset_rows(labels, datasets)
            dataset_nmi = float(normalized_mutual_info_score(datasets, labels))
            focus_distribution = focus_rows(labels, datasets, k)
            focus_entropy_min = min(obj["normalized_entropy"] for obj in focus_distribution.values())
            focus_max_fraction_max = max(obj["max_cluster_fraction"] for obj in focus_distribution.values())
            min_size = min(sizes.values())
            max_size = max(sizes.values())
            tiny_fraction = min_size / len(labels)
            seed_ari_mean = float(np.mean(aris)) if aris else 1.0
            condition_coverage = condition_archetype_coverage(
                args.data_dir,
                split,
                scaler,
                pca,
                models[0],
                k,
                residualization,
                args.max_conditions_per_focus_dataset,
            )

            reasons: list[str] = []
            if tiny_fraction < 0.01:
                reasons.append("tiny_cluster")
            if seed_ari_mean < 0.50:
                reasons.append("unstable_across_seeds")
            if dataset_purity >= 0.65 or dataset_nmi >= 0.45:
                reasons.append("dataset_proxy_risk")
            if focus_entropy_min < 0.35 or focus_max_fraction_max > 0.80:
                reasons.append("focus_dataset_collapse")
            gate_status = "candidate" if not reasons else "reject_" + ",".join(reasons)
            results.append(
                {
                    "residualization": residualization,
                    "k": int(k),
                    "n_sampled_control_cells": int(z.shape[0]),
                    "cluster_size_min": int(min_size),
                    "cluster_size_max": int(max_size),
                    "cluster_size_fraction_min": float(tiny_fraction),
                    "dataset_purity": float(dataset_purity),
                    "dataset_nmi": dataset_nmi,
                    "seed_ari_mean": seed_ari_mean,
                    "focus_entropy_min": float(focus_entropy_min),
                    "focus_max_cluster_fraction_max": float(focus_max_fraction_max),
                    "gate_status": gate_status,
                    "reject_reasons": reasons,
                    "cluster_rows": cluster_rows,
                    "focus_distribution": focus_distribution,
                    "focus_condition_coverage": condition_coverage,
                    "per_dataset_sampling": per_dataset,
                }
            )

    candidates = [row for row in results if row["gate_status"] == "candidate"]
    best_candidate = None
    if candidates:
        best = sorted(
            candidates,
            key=lambda row: (
                -row["seed_ari_mean"],
                row["dataset_nmi"],
                -row["focus_entropy_min"],
                row["focus_max_cluster_fraction_max"],
            ),
        )[0]
        best_candidate = {
            "residualization": best["residualization"],
            "k": best["k"],
            "gate_status": best["gate_status"],
            "rationale": "passes stability, non-collapse, and dataset-proxy CPU thresholds",
        }
    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "max_cells_per_dataset": args.max_cells_per_dataset,
        "max_conditions_per_focus_dataset": args.max_conditions_per_focus_dataset,
        "pca_components": args.pca_components,
        "ks": args.ks,
        "seeds": args.seeds,
        "kmeans_n_init": args.kmeans_n_init,
        "kmeans_max_iter": args.kmeans_max_iter,
        "residualizations": args.residualizations,
        "forbidden_inputs_used": {
            "gt_emb": False,
            "pert_means_npz": False,
            "posthoc_predictions": False,
            "heldout_multi_gt": False,
            "test_response_residuals": False,
        },
        "overall_status": "candidate_found" if best_candidate else "reject_all_cpu_gate",
        "best_candidate": best_candidate,
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "out_json": str(args.out_json), "status": payload["overall_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
