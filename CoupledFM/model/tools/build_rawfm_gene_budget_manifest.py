#!/usr/bin/env python3
"""Build train-only raw-gene budget manifests for RawFM observable-budget probes.

The output schema is intentionally simple:

{
  "version": 1,
  "label": "response_topk_k2000",
  "source": {...},
  "datasets": {
    "dataset_name": {"keep_indices": [0, 5, ...], "n_genes": 5000}
  }
}

Indices are local to each dataset handle's ``gene_ids_valid`` / in-vocab raw
expression columns.  They are therefore safe for mixed-dataset raw-expression
runs without assuming a global gene-order intersection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from model.data.vocab import GeneVocab
from model.data.dataset import _DatasetHandle
from model.paths import gene_name_path, nichenet_node2idx_path, biflow_dir as default_biflow_dir
from model.utils.data.biflow_paths import resolve_biflow_control_gt_h5ad
from model.utils.data.split import load_split_json


def _topk(score: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), int(score.shape[0])))
    idx = np.argpartition(-score, kth=k - 1)[:k]
    idx = idx[np.argsort(-score[idx])]
    return idx.astype(int)


def _random_k(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    k = max(1, min(int(k), int(n)))
    return np.sort(rng.choice(n, size=k, replace=False)).astype(int)


def _abundance_matched_random(
    response_keep: np.ndarray,
    abundance_score: np.ndarray,
    rng: np.random.Generator,
    bins: int,
) -> np.ndarray:
    n = int(abundance_score.shape[0])
    if n <= len(response_keep):
        return np.arange(n, dtype=int)
    q = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.quantile(abundance_score, q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_id = np.digitize(abundance_score, edges[1:-1], right=True)
    chosen: list[int] = []
    response_bins = bin_id[np.asarray(response_keep, dtype=int)]
    response_set = set(map(int, response_keep))
    for b in range(len(edges) - 1):
        need = int(np.sum(response_bins == b))
        if need <= 0:
            continue
        pool = np.where(bin_id == b)[0]
        non_response_pool = np.asarray([int(i) for i in pool if int(i) not in response_set], dtype=int)
        use_pool = non_response_pool if len(non_response_pool) >= need else pool
        take = rng.choice(use_pool, size=min(need, len(use_pool)), replace=False)
        chosen.extend(map(int, take))
    if len(chosen) < len(response_keep):
        remaining = np.asarray([i for i in range(n) if i not in set(chosen)], dtype=int)
        extra = rng.choice(remaining, size=len(response_keep) - len(chosen), replace=False)
        chosen.extend(map(int, extra))
    return np.asarray(sorted(chosen[: len(response_keep)]), dtype=int)


def _quantile_bins(score: np.ndarray, bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.quantile(np.asarray(score, dtype=float), q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(score, edges[1:-1], right=True).astype(int)


def _multi_confound_matched_random(
    selected_keep: np.ndarray,
    confound_scores: list[np.ndarray],
    rng: np.random.Generator,
    bins: int,
) -> np.ndarray:
    n = int(confound_scores[0].shape[0])
    if n <= len(selected_keep):
        return np.arange(n, dtype=int)
    bin_arrays = [_quantile_bins(score, bins) for score in confound_scores]
    selected_set = set(map(int, selected_keep))
    selected_keys = [tuple(int(arr[i]) for arr in bin_arrays) for i in selected_keep]
    all_keys: dict[tuple[int, ...], list[int]] = {}
    for i in range(n):
        all_keys.setdefault(tuple(int(arr[i]) for arr in bin_arrays), []).append(i)
    chosen: list[int] = []
    used: set[int] = set()
    for key in sorted(set(selected_keys)):
        need = int(sum(k == key for k in selected_keys))
        pool = [i for i in all_keys.get(key, []) if i not in selected_set and i not in used]
        if len(pool) < need:
            pool = [i for i in all_keys.get(key, []) if i not in used]
        if not pool:
            continue
        take = rng.choice(np.asarray(pool, dtype=int), size=min(need, len(pool)), replace=False)
        chosen.extend(map(int, take))
        used.update(map(int, take))
    if len(chosen) < len(selected_keep):
        remaining = np.asarray([i for i in range(n) if i not in used and i not in selected_set], dtype=int)
        if remaining.size < len(selected_keep) - len(chosen):
            remaining = np.asarray([i for i in range(n) if i not in used], dtype=int)
        extra = rng.choice(remaining, size=len(selected_keep) - len(chosen), replace=False)
        chosen.extend(map(int, extra))
    return np.asarray(sorted(chosen[: len(selected_keep)]), dtype=int)


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not math.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd


def _residualize_score(score: np.ndarray, confounds: list[np.ndarray]) -> np.ndarray:
    y = _standardize(np.log1p(np.maximum(score, 0.0)))
    cols = [np.ones_like(y)]
    for c in confounds:
        cols.append(_standardize(np.log1p(np.maximum(c, 0.0))))
    design = np.stack(cols, axis=1)
    mask = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    resid = np.zeros_like(y, dtype=np.float64)
    if int(mask.sum()) <= design.shape[1]:
        return resid.astype(np.float32)
    beta, *_ = np.linalg.lstsq(design[mask], y[mask], rcond=None)
    resid[mask] = y[mask] - design[mask] @ beta
    return resid.astype(np.float32)


def _iter_datasets(split: dict, requested: Iterable[str] | None) -> list[str]:
    names = sorted(split.keys())
    if requested:
        allowed = set(requested)
        names = [x for x in names if x in allowed]
    return names


def _condition_response_score(handle: _DatasetHandle, train_conds: list[str]) -> np.ndarray:
    ctrl = handle.ctrl_mean_gene()
    scores = []
    for cond in train_conds:
        if cond not in handle.gt_cond2idx or cond not in handle.pert_cond2idx:
            continue
        gt = handle.compute_gt_mean_gene_cond(cond)
        scores.append(np.abs(gt - ctrl))
    if not scores:
        return np.zeros_like(ctrl, dtype=np.float32)
    return np.mean(np.stack(scores, axis=0), axis=0).astype(np.float32)


def _condition_response_matrix(
    handle: _DatasetHandle,
    train_conds: list[str],
) -> np.ndarray:
    ctrl = handle.ctrl_mean_gene()
    rows = []
    for cond in train_conds:
        if cond not in handle.gt_cond2idx or cond not in handle.pert_cond2idx:
            continue
        gt = handle.compute_gt_mean_gene_cond(cond)
        rows.append(np.abs(gt - ctrl))
    if not rows:
        return np.zeros((0, ctrl.shape[0]), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def _control_gene_moments(handle: _DatasetHandle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = handle.X_ctrl[:, handle.in_vocab].astype(np.float32, copy=False)
    mean = x.mean(axis=0).astype(np.float32)
    var = x.var(axis=0).astype(np.float32)
    det = (x > 0).mean(axis=0).astype(np.float32)
    return mean, var, det


def _write_manifest(path: Path, label: str, source: dict, datasets: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "version": 1,
        "label": label,
        "source": source,
        "datasets": datasets,
    }
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build RawFM gene-budget/control manifests")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--biflow-dir", default=str(default_biflow_dir()))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--latent-backbone", default="stack")
    ap.add_argument("--abundance-bins", type=int, default=20)
    args = ap.parse_args()

    split_path = Path(args.split_file).expanduser()
    split = load_split_json(split_path)
    vocab = GeneVocab(str(gene_name_path()), str(nichenet_node2idx_path()))
    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.out_dir).expanduser()

    manifests: Dict[str, Dict[str, dict]] = {
        "response_topk": {},
        "response_abundance_residual_topk": {},
        "condition_diversity_topk": {},
        "abundance_topk": {},
        "abundance_matched_random": {},
        "residual_abundance_matched_random": {},
        "residual_confound_matched_random": {},
        "random_gene_set": {},
    }
    source = {
        "split_file": str(split_path),
        "biflow_dir": str(Path(args.biflow_dir).expanduser()),
        "k": int(args.k),
        "seed": int(args.seed),
        "train_only": True,
    }

    for ds_name in _iter_datasets(split, args.datasets):
        train_conds = list((split.get(ds_name) or {}).get("train", []))
        if not train_conds:
            continue
        pair = resolve_biflow_control_gt_h5ad(
            args.biflow_dir,
            ds_name,
            latent_backbone=str(args.latent_backbone),
        )
        if pair is None:
            continue
        cc_p, gt_p = pair
        handle = _DatasetHandle(ds_name, str(cc_p), str(gt_p), vocab)
        try:
            n_genes = len(handle.gene_ids_valid)
            response_matrix = _condition_response_matrix(handle, train_conds)
            if response_matrix.shape[0] == 0:
                response_score = np.zeros(n_genes, dtype=np.float32)
                diversity_score = np.zeros(n_genes, dtype=np.float32)
            else:
                response_score = response_matrix.mean(axis=0).astype(np.float32)
                diversity_score = response_matrix.std(axis=0).astype(np.float32)
            abundance_score, variance_score, detection_score = _control_gene_moments(handle)
            residual_response_score = _residualize_score(
                response_score,
                [abundance_score, variance_score, detection_score],
            )
            response_keep = _topk(response_score, args.k)
            residual_keep = _topk(residual_response_score, args.k)
            diversity_keep = _topk(diversity_score, args.k)
            abundance_keep = _topk(abundance_score, args.k)
            matched_keep = _abundance_matched_random(
                response_keep,
                abundance_score,
                rng,
                bins=int(args.abundance_bins),
            )
            residual_matched_keep = _abundance_matched_random(
                residual_keep,
                abundance_score,
                rng,
                bins=int(args.abundance_bins),
            )
            residual_confound_keep = _multi_confound_matched_random(
                residual_keep,
                [abundance_score, variance_score, detection_score],
                rng,
                bins=max(3, min(8, int(args.abundance_bins))),
            )
            random_keep = _random_k(n_genes, args.k, rng)
            manifests["response_topk"][ds_name] = {
                "keep_indices": response_keep.tolist(),
                "n_genes": int(n_genes),
                "n_train_conditions": len(train_conds),
            }
            manifests["response_abundance_residual_topk"][ds_name] = {
                "keep_indices": residual_keep.tolist(),
                "n_genes": int(n_genes),
                "n_train_conditions": len(train_conds),
                "score": "log1p_mean_abs_response_residualized_by_control_abundance_variance_detection",
            }
            manifests["condition_diversity_topk"][ds_name] = {
                "keep_indices": diversity_keep.tolist(),
                "n_genes": int(n_genes),
                "n_train_conditions": len(train_conds),
                "score": "std_abs_response_across_train_conditions",
            }
            manifests["abundance_topk"][ds_name] = {
                "keep_indices": abundance_keep.tolist(),
                "n_genes": int(n_genes),
                "n_train_conditions": len(train_conds),
            }
            manifests["abundance_matched_random"][ds_name] = {
                "keep_indices": matched_keep.tolist(),
                "n_genes": int(n_genes),
                "n_train_conditions": len(train_conds),
            }
            manifests["residual_abundance_matched_random"][ds_name] = {
                "keep_indices": residual_matched_keep.tolist(),
                "n_genes": int(n_genes),
                "n_train_conditions": len(train_conds),
                "matched_to": "response_abundance_residual_topk",
            }
            manifests["residual_confound_matched_random"][ds_name] = {
                "keep_indices": residual_confound_keep.tolist(),
                "n_genes": int(n_genes),
                "n_train_conditions": len(train_conds),
                "matched_to": "response_abundance_residual_topk",
                "confounds": ["control_abundance", "control_variance", "control_detection"],
            }
            manifests["random_gene_set"][ds_name] = {
                "keep_indices": random_keep.tolist(),
                "n_genes": int(n_genes),
                "n_train_conditions": len(train_conds),
            }
        finally:
            handle.close()

    for label, datasets in manifests.items():
        _write_manifest(
            out_dir / f"{label}_k{int(args.k)}_seed{int(args.seed)}.json",
            f"{label}_k{int(args.k)}_seed{int(args.seed)}",
            source,
            datasets,
        )


if __name__ == "__main__":
    main()
