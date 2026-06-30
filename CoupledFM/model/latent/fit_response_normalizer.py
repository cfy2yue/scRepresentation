#!/usr/bin/env python3
"""Fit train-only response normalization artifacts for LatentFM.

The fitted artifact is safe to use for training-time auxiliary losses only when
``fit_scope=train_only`` and the split hash matches the active training split.
It never reads test conditions, posthoc predictions, or pert_means.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.decomposition import PCA

from model.latent.response_normalizer import sha256_file


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def decode_conditions(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def sample_mean(
    arr: h5py.Dataset,
    start: int,
    end: int,
    *,
    max_cells: int,
    key: str,
) -> np.ndarray:
    n = int(end - start)
    if n <= 0:
        raise ValueError("empty condition slice")
    if max_cells > 0 and n > max_cells:
        rng = np.random.default_rng(stable_seed(key))
        rel = np.sort(rng.choice(n, size=int(max_cells), replace=False))
        block = arr[start + rel]
    else:
        block = arr[start:end]
    return np.asarray(block, dtype=np.float32).mean(axis=0)


def collect_train_residuals(
    *,
    data_dir: Path,
    split: dict[str, Any],
    max_train_conditions_per_dataset: int,
    max_cells_per_condition: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    residuals: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for ds_name in sorted(split):
        train = [str(x) for x in split.get(ds_name, {}).get("train", [])]
        if not train:
            continue
        train = sorted(train, key=lambda c: hashlib.sha256(f"fit_response|{ds_name}|{c}".encode()).hexdigest())
        if max_train_conditions_per_dataset > 0:
            train = train[: int(max_train_conditions_per_dataset)]
        path = data_dir / f"{ds_name}.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            conditions = decode_conditions(np.asarray(handle["conditions"]))
            by_cond = {cond: i for i, cond in enumerate(conditions)}
            ctrl = handle["ctrl/emb"]
            gt = handle["gt/emb"]
            ctrl_offsets = np.asarray(handle["ctrl/offsets"])
            gt_offsets = np.asarray(handle["gt/offsets"])
            for cond in train:
                i = by_cond.get(cond)
                if i is None:
                    continue
                c0, c1 = int(ctrl_offsets[i]), int(ctrl_offsets[i + 1])
                g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
                if c1 <= c0 or g1 <= g0:
                    continue
                ctrl_mean = sample_mean(
                    ctrl,
                    c0,
                    c1,
                    max_cells=max_cells_per_condition,
                    key=f"ctrl|{ds_name}|{cond}|{max_cells_per_condition}",
                )
                gt_mean = sample_mean(
                    gt,
                    g0,
                    g1,
                    max_cells=max_cells_per_condition,
                    key=f"gt|{ds_name}|{cond}|{max_cells_per_condition}",
                )
                resid = (gt_mean - ctrl_mean).astype(np.float32)
                residuals.append(resid)
                rows.append(
                    {
                        "dataset": ds_name,
                        "condition": cond,
                        "residual_norm": float(np.linalg.norm(resid)),
                    }
                )
    if not residuals:
        raise RuntimeError("No train residuals collected for response normalizer")
    return np.stack(residuals).astype(np.float32), rows


def robust_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    med = float(np.median(arr))
    return med if med > 1e-8 else 1.0


def fit_artifact(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir.expanduser().resolve()
    split_file = args.split_file.expanduser().resolve()
    manifest = load_json(data_dir / args.manifest)
    split = load_json(split_file)
    residuals, rows = collect_train_residuals(
        data_dir=data_dir,
        split=split,
        max_train_conditions_per_dataset=args.max_train_conditions_per_dataset,
        max_cells_per_condition=args.max_cells_per_condition,
    )

    emb_dim = int(residuals.shape[1])
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(float(row["residual_norm"]))
    global_median = robust_median([float(r["residual_norm"]) for r in rows])
    dataset_median_norms = {ds: robust_median(vals) for ds, vals in sorted(by_ds.items())}
    dataset_scale_factors = {
        ds: max(float(med / global_median), 1e-6)
        for ds, med in dataset_median_norms.items()
    }

    scaled = residuals.copy()
    for i, row in enumerate(rows):
        scaled[i] = scaled[i] / float(dataset_scale_factors.get(str(row["dataset"]), 1.0))

    n_components = min(int(args.pca_components), scaled.shape[0] - 1, scaled.shape[1])
    if n_components <= 0:
        raise RuntimeError("not enough residuals for PCA")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.seed)
    pca.fit(scaled)
    pc_std = np.sqrt(np.maximum(pca.explained_variance_.astype(np.float32), 1e-8))
    median_pc_std = float(np.median(pc_std)) if pc_std.size else 1.0
    pca_scales = np.maximum(pc_std / max(median_pc_std, 1e-8), 1e-3).astype(np.float32)

    train_counts = defaultdict(int)
    for row in rows:
        train_counts[str(row["dataset"])] += 1
    metadata = {
        "artifact_type": "latentfm_response_normalizer",
        "fit_scope": "train_only",
        "mode_recommended": str(args.mode),
        "data_dir": str(data_dir),
        "manifest": str(data_dir / args.manifest),
        "manifest_emb_dim": int(manifest.get("emb_dim", emb_dim)),
        "split_file": str(split_file),
        "split_sha256": sha256_file(split_file),
        "emb_dim": emb_dim,
        "max_train_conditions_per_dataset": int(args.max_train_conditions_per_dataset),
        "max_cells_per_condition": int(args.max_cells_per_condition),
        "n_train_residuals": int(residuals.shape[0]),
        "global_median_norm": float(global_median),
        "dataset_median_norms": dataset_median_norms,
        "dataset_train_residual_counts": dict(sorted(train_counts.items())),
        "pca_components": int(n_components),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "pca_cumulative_ev": {
            str(k): float(np.sum(pca.explained_variance_ratio_[: min(k, n_components)]))
            for k in (4, 8, 16, 32, 64)
            if k <= n_components
        },
        "forbidden_inputs_used": {
            "test_gt": False,
            "heldout_multi_gt": False,
            "posthoc_predictions": False,
            "pert_means_npz": False,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        dataset_scale_factors_json=np.asarray(json.dumps(dataset_scale_factors, ensure_ascii=False)),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_scales=pca_scales.astype(np.float32),
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--mode", default="dataset_scale_pca", choices=["dataset_scale", "pca_subspace", "dataset_scale_pca"])
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--max-train-conditions-per-dataset", type=int, default=256)
    parser.add_argument("--max-cells-per-condition", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = fit_artifact(args)
    print(json.dumps({"out": str(args.out), "metadata": metadata}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
