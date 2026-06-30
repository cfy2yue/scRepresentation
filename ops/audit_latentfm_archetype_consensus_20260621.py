#!/usr/bin/env python3
"""Consensus stability audit for residualized control-latent archetypes.

This CPU-only proof tests whether archetype labels are stable across KMeans
seeds/subsamples on a fixed held-in control/source evaluation panel.  It uses
only ``ctrl/emb`` and never reads perturbed GT, perturbed means, or posthoc
predictions.
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


FOCUS = ("Wessels", "NormanWeissman2019_filtered", "GasperiniShendure2019_lowMOI")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


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
        slices[-1] = slice(slices[-1].start, min(n, slices[-1].stop + (k - total)))
    return slices


def read_sample(dataset: h5py.Dataset, k: int, *, key: str) -> np.ndarray:
    parts = [np.asarray(dataset[s], dtype=np.float32) for s in pick_slices(int(dataset.shape[0]), k, key=key)]
    arr = np.concatenate(parts, axis=0)
    return arr[:k]


def normalized_entropy(counts: Counter[Any], k: int) -> float:
    total = sum(counts.values())
    if total <= 0 or k <= 1:
        return 0.0
    ent = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            ent -= p * math.log(p)
    return float(ent / math.log(k))


def residualize(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    return ((arr - mean) / np.maximum(std, 1e-3)).astype(np.float32)


def load_panel(
    data_dir: Path,
    datasets: list[str],
    cells_per_dataset: int,
    *,
    key_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    for dataset in datasets:
        with h5py.File(data_dir / f"{dataset}.h5", "r") as handle:
            arr = read_sample(handle["ctrl/emb"], cells_per_dataset, key=f"{key_prefix}|{dataset}|{cells_per_dataset}")
        arrays.append(residualize(arr))
        labels.extend([dataset] * arrays[-1].shape[0])
    return np.concatenate(arrays, axis=0), np.asarray(labels)


def focus_distribution(labels: np.ndarray, datasets: np.ndarray, k: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in FOCUS:
        mask = datasets == dataset
        counts = Counter(labels[mask].tolist())
        total = int(mask.sum())
        out[dataset] = {
            "n_eval_controls": total,
            "n_clusters_used": int(len(counts)),
            "normalized_entropy": normalized_entropy(counts, k),
            "max_cluster_fraction": (float(max(counts.values()) / total) if total else None),
            "cluster_counts": {str(key): int(value) for key, value in sorted(counts.items())},
        }
    return out


def run_k(
    train_x: np.ndarray,
    eval_x: np.ndarray,
    eval_datasets: np.ndarray,
    *,
    k: int,
    pca_components: int,
    seeds: list[int],
    n_init: int,
    max_iter: int,
) -> dict[str, Any]:
    label_sets: list[np.ndarray] = []
    for seed in seeds:
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_scaled = scaler.fit_transform(train_x)
        eval_scaled = scaler.transform(eval_x)
        pca = PCA(
            n_components=min(pca_components, train_scaled.shape[0] - 1, train_scaled.shape[1]),
            whiten=True,
            svd_solver="randomized",
            random_state=seed,
        )
        train_z = pca.fit_transform(train_scaled)
        eval_z = pca.transform(eval_scaled)
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=2048,
            n_init=n_init,
            max_iter=max_iter,
        )
        km.fit(train_z)
        label_sets.append(km.predict(eval_z))
    aris = [
        float(adjusted_rand_score(label_sets[i], label_sets[j]))
        for i in range(len(label_sets))
        for j in range(i + 1, len(label_sets))
    ]
    primary = label_sets[0]
    sizes = Counter(primary.tolist())
    focus = focus_distribution(primary, eval_datasets, k)
    focus_max = max(v["max_cluster_fraction"] or 0.0 for v in focus.values())
    focus_min_clusters = min(v["n_clusters_used"] for v in focus.values())
    dataset_nmi = float(normalized_mutual_info_score(eval_datasets, primary))
    dataset_purity_num = 0
    for cluster in sorted(sizes):
        mask = primary == cluster
        dataset_purity_num += Counter(eval_datasets[mask].tolist()).most_common(1)[0][1]
    dataset_purity = float(dataset_purity_num / len(primary))
    median_ari = float(np.median(aris)) if aris else 1.0
    p10_ari = float(np.quantile(aris, 0.10)) if aris else 1.0
    reasons: list[str] = []
    if median_ari < 0.55:
        reasons.append("median_ari_low")
    if p10_ari < 0.35:
        reasons.append("p10_ari_low")
    if dataset_nmi > 0.25:
        reasons.append("dataset_nmi_high")
    if focus_max > 0.35:
        reasons.append("focus_max_fraction_high")
    if focus_min_clusters < 4:
        reasons.append("focus_cluster_count_low")
    return {
        "k": int(k),
        "median_ari": median_ari,
        "p10_ari": p10_ari,
        "ari_values": aris,
        "dataset_nmi": dataset_nmi,
        "dataset_purity": dataset_purity,
        "cluster_size_min": int(min(sizes.values())),
        "cluster_size_max": int(max(sizes.values())),
        "focus_distribution": focus,
        "gate_status": "candidate" if not reasons else "reject_" + ",".join(reasons),
        "reject_reasons": reasons,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Archetype Consensus CPU Audit",
        "",
        f"Status: `{payload['overall_status']}`",
        "",
        "Controls/source only; no GT, perturbed means, held-out response residuals, or posthoc predictions.",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- train_cells_per_dataset: `{payload['train_cells_per_dataset']}`",
        f"- eval_cells_per_dataset: `{payload['eval_cells_per_dataset']}`",
        f"- pca_components: `{payload['pca_components']}`",
        f"- seeds: `{payload['seeds']}`",
        "",
        "## Summary",
        "",
        "| K | median ARI | p10 ARI | dataset NMI | dataset purity | min cluster | max cluster | gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        lines.append(
            "| {k} | {median:.4f} | {p10:.4f} | {nmi:.4f} | {purity:.4f} | {mn} | {mx} | {gate} |".format(
                k=row["k"],
                median=row["median_ari"],
                p10=row["p10_ari"],
                nmi=row["dataset_nmi"],
                purity=row["dataset_purity"],
                mn=row["cluster_size_min"],
                mx=row["cluster_size_max"],
                gate=row["gate_status"],
            )
        )
    lines += ["", "## Focus Coverage", ""]
    for row in payload["results"]:
        lines.append(f"### K={row['k']}")
        for dataset, obj in row["focus_distribution"].items():
            lines.append(
                f"- `{dataset}`: clusters={obj['n_clusters_used']}, "
                f"entropy={obj['normalized_entropy']:.4f}, "
                f"max_fraction={obj['max_cluster_fraction']:.4f}, "
                f"counts={obj['cluster_counts']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/cyx/1030/dataset/latentfm_full/scfoundation"))
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23, 29, 31])
    parser.add_argument("--train-cells-per-dataset", type=int, default=256)
    parser.add_argument("--eval-cells-per-dataset", type=int, default=128)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--kmeans-n-init", type=int, default=1)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--out-json", type=Path, default=Path("/data/cyx/1030/scLatent/reports/latentfm_archetype_consensus_cpu_audit_20260621.json"))
    parser.add_argument("--out-md", type=Path, default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_ARCHETYPE_CONSENSUS_CPU_AUDIT_20260621.md"))
    args = parser.parse_args()

    manifest = load_json(args.data_dir / "manifest.json")
    datasets = sorted(manifest.get("datasets", {}))
    train_x, _ = load_panel(args.data_dir, datasets, args.train_cells_per_dataset, key_prefix="arch_consensus_train")
    eval_x, eval_datasets = load_panel(args.data_dir, datasets, args.eval_cells_per_dataset, key_prefix="arch_consensus_eval")
    results = [
        run_k(
            train_x,
            eval_x,
            eval_datasets,
            k=k,
            pca_components=args.pca_components,
            seeds=args.seeds,
            n_init=args.kmeans_n_init,
            max_iter=args.kmeans_max_iter,
        )
        for k in args.ks
    ]
    payload = {
        "data_dir": str(args.data_dir),
        "train_cells_per_dataset": args.train_cells_per_dataset,
        "eval_cells_per_dataset": args.eval_cells_per_dataset,
        "pca_components": args.pca_components,
        "ks": args.ks,
        "seeds": args.seeds,
        "forbidden_inputs_used": {
            "gt_emb": False,
            "pert_means_npz": False,
            "posthoc_predictions": False,
            "heldout_multi_gt": False,
            "test_response_residuals": False,
        },
        "overall_status": "candidate_found" if any(r["gate_status"] == "candidate" for r in results) else "reject_all_cpu_gate",
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "status": payload["overall_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
