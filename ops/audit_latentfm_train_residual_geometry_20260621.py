#!/usr/bin/env python3
"""Train-only residual geometry audit for LatentFM normalization/subspace ideas.

This script reads only canonical train conditions. It may use train GT to form
condition-level residuals for diagnostics, but it never reads test conditions,
test GT, posthoc predictions, or pert_means.npz.
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
from sklearn.decomposition import PCA


FOCUS = ("Wessels", "NormanWeissman2019_filtered", "GasperiniShendure2019_lowMOI")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_conditions(values: np.ndarray) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def stable_selected(items: list[str], k: int, key: str) -> list[str]:
    ordered = sorted(items, key=lambda x: hashlib.sha256(f"{key}|{x}".encode()).hexdigest())
    return ordered[:k] if k > 0 else ordered


def fallback_nperts(condition: str) -> int:
    for sep in ("+", "|", ",", ";", "&", "/"):
        if sep in condition:
            return len([x for x in condition.split(sep) if x.strip()])
    return 1


def nperts_from_metadata(metadata: dict[str, Any], ds: str, condition: str) -> int:
    obj = metadata.get(ds, {}).get(condition)
    if isinstance(obj, dict):
        genes = obj.get("genes")
        if isinstance(genes, list) and genes:
            return len(genes)
    return fallback_nperts(condition)


def condition_mean(handle: h5py.File, group: str, i: int) -> np.ndarray | None:
    offsets = np.asarray(handle[f"{group}/offsets"])
    start, end = int(offsets[i]), int(offsets[i + 1])
    if end <= start:
        return None
    return np.asarray(handle[f"{group}/emb"][start:end], dtype=np.float32).mean(axis=0)


def collect_train_residuals(
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    max_train_conditions_per_dataset: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    residuals: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for ds in sorted(split):
        train = [str(x) for x in split.get(ds, {}).get("train", [])]
        if not train:
            continue
        selected = stable_selected(train, max_train_conditions_per_dataset, key=f"train_residual_geometry|{ds}")
        path = data_dir / f"{ds}.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            conditions = decode_conditions(np.asarray(handle["conditions"]))
            by_cond = {cond: i for i, cond in enumerate(conditions)}
            for cond in selected:
                i = by_cond.get(cond)
                if i is None:
                    continue
                ctrl_mean = condition_mean(handle, "ctrl", i)
                gt_mean = condition_mean(handle, "gt", i)
                if ctrl_mean is None or gt_mean is None:
                    continue
                resid = (gt_mean - ctrl_mean).astype(np.float32)
                residuals.append(resid)
                rows.append(
                    {
                        "dataset": ds,
                        "condition": cond,
                        "nperts": int(nperts_from_metadata(metadata, ds, cond)),
                        "residual_norm": float(np.linalg.norm(resid)),
                        "ctrl_norm": float(np.linalg.norm(ctrl_mean)),
                        "gt_norm": float(np.linalg.norm(gt_mean)),
                    }
                )
    if not residuals:
        raise RuntimeError("No train residuals collected")
    return np.stack(residuals).astype(np.float32), rows


def robust_summary(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def grouped_summaries(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(float(row["residual_norm"]))
    return {name: robust_summary(vals) for name, vals in sorted(groups.items())}


def variance_explained_by_groups(x: np.ndarray, labels: list[str]) -> float | None:
    if x.shape[0] < 3:
        return None
    global_mean = x.mean(axis=0, keepdims=True)
    sst = float(np.square(x - global_mean).sum())
    if sst <= 0:
        return None
    pred = np.zeros_like(x)
    labels_arr = np.asarray(labels)
    for label in sorted(set(labels)):
        mask = labels_arr == label
        pred[mask] = x[mask].mean(axis=0, keepdims=True)
    sse = float(np.square(x - pred).sum())
    return float(1.0 - sse / sst)


def pca_summary(x: np.ndarray, max_components: int) -> dict[str, Any]:
    n_components = min(max_components, x.shape[0] - 1, x.shape[1])
    if n_components < 2:
        return {"n_components": n_components, "cum": {}}
    centered = x - x.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=17)
    pca.fit(centered)
    evr = pca.explained_variance_ratio_
    cum: dict[str, float] = {}
    for k in (4, 8, 16, 32, 64):
        if k <= n_components:
            cum[str(k)] = float(evr[:k].sum())
    return {
        "n_components": int(n_components),
        "cum": cum,
        "top10": [float(x) for x in evr[:10]],
    }


def decision(payload: dict[str, Any]) -> dict[str, Any]:
    ds_medians = [
        row["median"]
        for row in payload["dataset_norm_summary"].values()
        if int(row.get("n", 0)) >= 8 and row.get("median", 0) > 0
    ]
    ds_ratio = (max(ds_medians) / min(ds_medians)) if len(ds_medians) >= 2 else None
    dim = payload["per_dim_std_summary"]
    per_dim_ratio = dim.get("p90_p10_ratio")
    pca_cum = payload["pca_summary"]["cum"]
    cum32 = pca_cum.get("32")
    cum64 = pca_cum.get("64")
    dataset_r2 = payload.get("dataset_residual_r2")

    candidates: list[str] = []
    reasons: list[str] = []
    if ds_ratio is not None and ds_ratio >= 1.8:
        candidates.append("dataset_residual_scale_normalization")
        reasons.append(f"dataset median residual norm ratio {ds_ratio:.2f} >= 1.8")
    if per_dim_ratio is not None and per_dim_ratio >= 4.0:
        candidates.append("diagonal_residual_std_normalization")
        reasons.append(f"per-dimension residual std p90/p10 ratio {per_dim_ratio:.2f} >= 4.0")
    if cum32 is not None and cum32 >= 0.45:
        candidates.append("response_pca_subspace_loss_or_whitening")
        reasons.append(f"top32 residual PCs explain {cum32:.3f} >= 0.45")
    elif cum64 is not None and cum64 >= 0.55:
        candidates.append("response_pca_subspace_loss_or_whitening")
        reasons.append(f"top64 residual PCs explain {cum64:.3f} >= 0.55")
    if dataset_r2 is not None and dataset_r2 >= 0.20:
        reasons.append(f"dataset means explain {dataset_r2:.3f} residual variance")

    status = "candidate_cpu_signal" if candidates else "no_strong_cpu_signal"
    return {
        "status": status,
        "candidates": sorted(set(candidates)),
        "reasons": reasons,
        "dataset_norm_median_ratio": ds_ratio,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Train Residual Geometry Audit",
        "",
        "This is a CPU-only train-set diagnostic for response normalization/subspace ideas.",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- max_train_conditions_per_dataset: `{payload['max_train_conditions_per_dataset']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- collected train residuals: `{payload['n_train_residuals']}`",
        "",
        "## Decision",
        "",
        f"- status: `{payload['decision']['status']}`",
        f"- candidates: `{payload['decision']['candidates']}`",
    ]
    for reason in payload["decision"]["reasons"]:
        lines.append(f"- reason: {reason}")
    lines += [
        "",
        "## Global Geometry",
        "",
        f"- dataset residual R2: `{payload['dataset_residual_r2']}`",
        f"- nperts residual R2: `{payload['nperts_residual_r2']}`",
        f"- per-dimension std p90/p10 ratio: `{payload['per_dim_std_summary']['p90_p10_ratio']}`",
        f"- PCA cumulative EV: `{payload['pca_summary']['cum']}`",
        "",
        "## Focus Dataset Norms",
        "",
        "| dataset | n | median residual norm | p10 | p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds in FOCUS:
        row = payload["dataset_norm_summary"].get(ds)
        if not row:
            continue
        lines.append(
            f"| {ds} | {int(row['n'])} | {row['median']:.6f} | {row['p10']:.6f} | {row['p90']:.6f} |"
        )
    lines += [
        "",
        "## Largest Dataset Norm Differences",
        "",
        "| dataset | n | median residual norm | p10 | p90 |",
        "|---|---:|---:|---:|---:|",
    ]
    sortable = [
        (ds, row)
        for ds, row in payload["dataset_norm_summary"].items()
        if int(row.get("n", 0)) >= 8 and row.get("median") is not None
    ]
    sortable.sort(key=lambda x: x[1]["median"])
    for ds, row in (sortable[:5] + sortable[-5:]):
        lines.append(
            f"| {ds} | {int(row['n'])} | {row['median']:.6f} | {row['p10']:.6f} | {row['p90']:.6f} |"
        )
    lines += [
        "",
        "## N-Perturbation Norms",
        "",
        "| nperts | n | median residual norm | p10 | p90 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for nperts, row in payload["nperts_norm_summary"].items():
        lines.append(
            f"| {nperts} | {int(row['n'])} | {row['median']:.6f} | {row['p10']:.6f} | {row['p90']:.6f} |"
        )
    lines += [
        "",
        "## Guardrail",
        "",
        "This report is not a promotion result. It can only justify preparing a small gated GPU smoke; it cannot use held-out multi GT or posthoc predictions.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/cyx/1030/dataset/latentfm_full/scfoundation"))
    parser.add_argument("--split-file", type=Path, default=Path("/data/cyx/1030/dataset/biFlow_data/split_seed42.json"))
    parser.add_argument("--max-train-conditions-per-dataset", type=int, default=256)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--out-json", type=Path, default=Path("/data/cyx/1030/scLatent/reports/latentfm_train_residual_geometry_audit_20260621.json"))
    parser.add_argument("--out-md", type=Path, default=Path("/data/cyx/1030/scLatent/reports/LATENTFM_TRAIN_RESIDUAL_GEOMETRY_AUDIT_20260621.md"))
    args = parser.parse_args()

    split = load_json(args.split_file)
    metadata = load_json(args.data_dir / "condition_metadata.json")
    residuals, rows = collect_train_residuals(
        args.data_dir,
        split,
        metadata,
        max_train_conditions_per_dataset=args.max_train_conditions_per_dataset,
    )
    labels_ds = [row["dataset"] for row in rows]
    labels_nperts = [str(row["nperts"]) for row in rows]
    per_dim_std = residuals.std(axis=0)
    nz_std = per_dim_std[per_dim_std > 1e-8]
    p10 = float(np.percentile(nz_std, 10)) if nz_std.size else None
    p90 = float(np.percentile(nz_std, 90)) if nz_std.size else None
    payload: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "max_train_conditions_per_dataset": args.max_train_conditions_per_dataset,
        "pca_components": args.pca_components,
        "n_train_residuals": int(residuals.shape[0]),
        "emb_dim": int(residuals.shape[1]),
        "leakage_status": "pass_train_conditions_only_no_test_gt_no_pert_means_no_posthoc",
        "forbidden_inputs_used": {
            "test_gt": False,
            "pert_means_npz": False,
            "posthoc_predictions": False,
            "heldout_multi_gt": False,
        },
        "dataset_norm_summary": grouped_summaries(rows, "dataset"),
        "nperts_norm_summary": grouped_summaries(rows, "nperts"),
        "dataset_residual_r2": variance_explained_by_groups(residuals, labels_ds),
        "nperts_residual_r2": variance_explained_by_groups(residuals, labels_nperts),
        "per_dim_std_summary": {
            "n_nonzero": int(nz_std.size),
            "p10": p10,
            "p50": float(np.percentile(nz_std, 50)) if nz_std.size else None,
            "p90": p90,
            "p90_p10_ratio": (float(p90 / p10) if p10 and p10 > 0 and p90 is not None else None),
        },
        "pca_summary": pca_summary(residuals, args.pca_components),
        "nperts_counts": {str(k): int(v) for k, v in Counter(labels_nperts).items()},
    }
    payload["decision"] = decision(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
