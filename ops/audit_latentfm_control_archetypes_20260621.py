#!/usr/bin/env python3
"""CPU-only leakage audit for control-latent archetypes.

This script intentionally uses only ``ctrl/emb`` for archetype fitting.  It may
optionally use canonical train GT only for a train-only residual diagnostic; no
test GT or posthoc residuals are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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


def pick_indices(n: int, k: int, *, key: str) -> np.ndarray:
    if n <= k:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(stable_seed(key))
    return np.sort(rng.choice(n, size=k, replace=False).astype(np.int64))


def decode_conditions(values: np.ndarray) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def sample_controls(data_dir: Path, manifest: dict[str, Any], max_cells_per_dataset: int) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    labels: list[str] = []
    for ds in sorted(manifest.get("datasets", {})):
        path = data_dir / f"{ds}.h5"
        with h5py.File(path, "r") as handle:
            emb = handle["ctrl/emb"]
            idx = pick_indices(int(emb.shape[0]), max_cells_per_dataset, key=f"ctrl|{ds}|{max_cells_per_dataset}")
            arr = np.asarray(emb[idx], dtype=np.float32)
        arrays.append(arr)
        labels.extend([ds] * arr.shape[0])
    return np.concatenate(arrays, axis=0), labels


def fit_representation(x: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    xs = scaler.fit_transform(x)
    pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(xs)
    return z.astype(np.float32), scaler, pca


def cluster_purity(labels: np.ndarray, dataset_ids: np.ndarray) -> tuple[float, dict[str, Any]]:
    total = len(labels)
    rows: dict[str, Any] = {}
    weighted = 0
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        counts = Counter(dataset_ids[mask].tolist())
        dominant, n_dom = counts.most_common(1)[0]
        weighted += n_dom
        rows[str(cluster)] = {
            "size": int(mask.sum()),
            "dominant_dataset": dominant,
            "dominant_fraction": float(n_dom / mask.sum()),
            "top_datasets": counts.most_common(5),
        }
    return float(weighted / total), rows


def focus_distribution(labels: np.ndarray, dataset_ids: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ds in FOCUS:
        mask = dataset_ids == ds
        counts = Counter(labels[mask].tolist())
        total = int(mask.sum())
        out[ds] = {
            "n_sampled_controls": total,
            "cluster_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "max_cluster_fraction": (max(counts.values()) / total if total else None),
        }
    return out


def condition_means_for_train(
    data_dir: Path,
    split: dict[str, Any],
    max_train_conditions_per_dataset: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ctrl_means: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    ds_labels: list[str] = []
    for ds in sorted(split):
        train = [str(x) for x in split.get(ds, {}).get("train", [])]
        if not train:
            continue
        selected = sorted(train, key=lambda x: hashlib.sha256(f"trainresid|{ds}|{x}".encode()).hexdigest())[
            :max_train_conditions_per_dataset
        ]
        selected_set = set(selected)
        path = data_dir / f"{ds}.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            conditions = decode_conditions(np.asarray(handle["conditions"]))
            offsets = np.asarray(handle["ctrl/offsets"])
            gt_offsets = np.asarray(handle["gt/offsets"])
            ctrl = handle["ctrl/emb"]
            gt = handle["gt/emb"]
            by_cond = {cond: i for i, cond in enumerate(conditions)}
            for cond in selected:
                i = by_cond.get(cond)
                if i is None:
                    continue
                c0, c1 = int(offsets[i]), int(offsets[i + 1])
                g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
                if c1 <= c0 or g1 <= g0:
                    continue
                ctrl_mean = np.asarray(ctrl[c0:c1], dtype=np.float32).mean(axis=0)
                gt_mean = np.asarray(gt[g0:g1], dtype=np.float32).mean(axis=0)
                ctrl_means.append(ctrl_mean)
                residuals.append(gt_mean - ctrl_mean)
                ds_labels.append(ds)
    if not ctrl_means:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), []
    return np.stack(ctrl_means).astype(np.float32), np.stack(residuals).astype(np.float32), ds_labels


def residual_r2(
    ctrl_means: np.ndarray,
    residuals: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    kmeans: MiniBatchKMeans,
) -> float | None:
    if ctrl_means.size == 0 or residuals.shape[0] < 3:
        return None
    z = pca.transform(scaler.transform(ctrl_means))
    labels = kmeans.predict(z)
    global_mean = residuals.mean(axis=0, keepdims=True)
    sst = float(np.square(residuals - global_mean).sum())
    if sst <= 0:
        return None
    pred = np.zeros_like(residuals)
    for cluster in sorted(set(labels.tolist())):
        mask = labels == cluster
        pred[mask] = residuals[mask].mean(axis=0, keepdims=True)
    sse = float(np.square(residuals - pred).sum())
    return float(1.0 - sse / sst)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Control Archetype CPU Audit",
        "",
        "This is a CPU-only leakage audit. Archetypes are fit from control/source latent cells only.",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- max_cells_per_dataset: `{payload['max_cells_per_dataset']}`",
        f"- pca_components: `{payload['pca_components']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        "",
        "## K Summary",
        "",
        "| K | min cluster | max cluster | dataset purity | dataset NMI | seed ARI mean | train-only residual R2 | gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["k_results"]:
        lines.append(
            "| {k} | {mn} | {mx} | {pur:.4f} | {nmi:.4f} | {ari} | {r2} | {gate} |".format(
                k=row["k"],
                mn=row["cluster_size_min"],
                mx=row["cluster_size_max"],
                pur=row["dataset_purity"],
                nmi=row["dataset_nmi"],
                ari=("NA" if row["seed_ari_mean"] is None else f"{row['seed_ari_mean']:.4f}"),
                r2=("NA" if row["train_only_residual_r2"] is None else f"{row['train_only_residual_r2']:.4f}"),
                gate=row["gate_status"],
            )
        )
    lines += [
        "",
        "## Focus Dataset Coverage",
        "",
    ]
    for row in payload["k_results"]:
        lines.append(f"### K={row['k']}")
        for ds, obj in row["focus_distribution"].items():
            lines.append(
                f"- `{ds}`: sampled_controls={obj['n_sampled_controls']}, "
                f"max_cluster_fraction={obj['max_cluster_fraction']}, "
                f"cluster_counts={obj['cluster_counts']}"
            )
        lines.append("")
    lines += [
        "## Decision Rule",
        "",
        "This branch can proceed to code implementation only if a K has stable, non-tiny clusters and is not simply a dataset label.",
        "If all K values are high-purity dataset clusters or unstable, record negative evidence and do not launch `scf_prior010_inject_archcond_k12_4k`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/cyx/1030/dataset/latentfm_full/scfoundation"))
    parser.add_argument("--split-file", type=Path, default=Path("/data/cyx/1030/dataset/biFlow_data/split_seed42.json"))
    parser.add_argument("--max-cells-per-dataset", type=int, default=512)
    parser.add_argument("--max-train-conditions-per-dataset", type=int, default=64)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--out-json", type=Path, default=Path("/data/cyx/1030/scLatent/reports/latentfm_control_archetype_cpu_audit_20260621.json"))
    parser.add_argument("--out-md", type=Path, default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_CONTROL_ARCHETYPE_CPU_AUDIT_20260621.md"))
    args = parser.parse_args()

    manifest = load_json(args.data_dir / "manifest.json")
    split = load_json(args.split_file)
    x, ds_labels = sample_controls(args.data_dir, manifest, args.max_cells_per_dataset)
    dataset_ids = np.asarray(ds_labels)
    z, scaler, pca = fit_representation(x, args.pca_components, seed=args.seeds[0])
    train_ctrl, train_resid, train_ds = condition_means_for_train(
        args.data_dir, split, args.max_train_conditions_per_dataset
    )

    k_results = []
    for k in args.ks:
        seed_labels: list[np.ndarray] = []
        seed_models: list[MiniBatchKMeans] = []
        for seed in args.seeds:
            km = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed,
                batch_size=2048,
                n_init=5,
                max_iter=200,
            )
            labels = km.fit_predict(z)
            seed_labels.append(labels)
            seed_models.append(km)
        labels = seed_labels[0]
        sizes = Counter(labels.tolist())
        purity, cluster_rows = cluster_purity(labels, dataset_ids)
        nmi = float(normalized_mutual_info_score(dataset_ids, labels))
        aris = []
        for i in range(len(seed_labels)):
            for j in range(i + 1, len(seed_labels)):
                aris.append(float(adjusted_rand_score(seed_labels[i], seed_labels[j])))
        r2 = residual_r2(train_ctrl, train_resid, scaler, pca, seed_models[0])
        min_size = min(sizes.values())
        max_size = max(sizes.values())
        tiny_fraction = min_size / len(labels)
        dataset_like = purity >= 0.85 or nmi >= 0.65
        unstable = (sum(aris) / len(aris)) < 0.50 if aris else False
        tiny = tiny_fraction < 0.01
        gate_status = "candidate"
        reasons = []
        if dataset_like:
            reasons.append("dataset_like_clusters")
        if unstable:
            reasons.append("unstable_across_seeds")
        if tiny:
            reasons.append("tiny_cluster")
        if reasons:
            gate_status = "reject_" + ",".join(reasons)
        k_results.append(
            {
                "k": int(k),
                "cluster_size_min": int(min_size),
                "cluster_size_max": int(max_size),
                "cluster_size_fraction_min": float(tiny_fraction),
                "dataset_purity": float(purity),
                "dataset_nmi": nmi,
                "seed_ari_mean": (float(sum(aris) / len(aris)) if aris else None),
                "train_only_residual_r2": r2,
                "gate_status": gate_status,
                "cluster_rows": cluster_rows,
                "focus_distribution": focus_distribution(labels, dataset_ids),
            }
        )

    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "max_cells_per_dataset": args.max_cells_per_dataset,
        "max_train_conditions_per_dataset": args.max_train_conditions_per_dataset,
        "pca_components": args.pca_components,
        "ks": args.ks,
        "seeds": args.seeds,
        "n_sampled_control_cells": int(x.shape[0]),
        "n_train_residual_conditions_sampled": int(train_resid.shape[0]),
        "leakage_status": "pass_controls_only_fit_train_gt_only_diagnostic",
        "forbidden_inputs_used": {
            "test_gt": False,
            "pert_means_npz": False,
            "posthoc_test_residual": False,
            "heldout_multi_gt": False,
        },
        "k_results": k_results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
