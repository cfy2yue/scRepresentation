#!/usr/bin/env python3
"""One-shot Track C query evaluation for a support-selected routed baseline.

Route selection was fixed from support_val diagnostics before this script reads
query_multi:

* NormanWeissman2019_filtered -> additive_single_sum
* Wessels -> dataset_multi_mean

This script evaluates that frozen route on held-out query_multi once. It is a
CPU diagnostic baseline, not a GPU model-promotion run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from model.latent.fm_ot import median_sigmas, mmd2_biased, mmd2_unbiased


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_support_selected_routed_baseline_query_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SELECTED_ROUTED_BASELINE_QUERY_20260622.md"

FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")
ROUTE = {
    "NormanWeissman2019_filtered": "additive_single_sum",
    "Wessels": "dataset_multi_mean",
}
MODELS = (
    "support_selected_route",
    "dataset_multi_mean",
    "global_multi_mean",
    "additive_single_mean",
    "additive_single_sum",
    "dataset_single_mean",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def condition_cells(handle: h5py.File, group: str, idx: int, max_cells: int) -> np.ndarray | None:
    offsets = np.asarray(handle[f"{group}/offsets"])
    start, end = int(offsets[idx]), int(offsets[idx + 1])
    if end <= start:
        return None
    if max_cells > 0 and end - start > max_cells:
        end = start + max_cells
    return np.asarray(handle[f"{group}/emb"][start:end], dtype=np.float32)


def condition_mean(handle: h5py.File, group: str, idx: int, max_cells: int) -> np.ndarray | None:
    cells = condition_cells(handle, group, idx, max_cells)
    return None if cells is None else cells.mean(axis=0)


def arrays_for_condition(handle: h5py.File, by_cond: dict[str, int], cond: str, max_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    idx = by_cond.get(cond)
    if idx is None:
        return None
    ctrl_cells = condition_cells(handle, "ctrl", idx, max_cells)
    gt_cells = condition_cells(handle, "gt", idx, max_cells)
    if ctrl_cells is None or gt_cells is None:
        return None
    ctrl = ctrl_cells.mean(axis=0).astype(np.float32)
    gt = gt_cells.mean(axis=0).astype(np.float32)
    return ctrl_cells.astype(np.float32), gt_cells.astype(np.float32), ctrl, gt, (gt - ctrl).astype(np.float32)


def residual_for_condition(handle: h5py.File, by_cond: dict[str, int], cond: str, max_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    vals = arrays_for_condition(handle, by_cond, cond, max_cells)
    if vals is None:
        return None
    _ctrl_cells, _gt_cells, ctrl, gt, residual = vals
    return ctrl.astype(np.float32), gt.astype(np.float32), (gt - ctrl).astype(np.float32)


def genes_for(metadata: dict[str, Any], ds: str, cond: str) -> list[str]:
    meta = (metadata.get(ds) or {}).get(cond) or {}
    genes = [str(g).strip() for g in meta.get("genes") or [] if str(g).strip()]
    raw = str(meta.get("perturbation_type_raw") or "").lower()
    if "drug" in raw or "compound" in raw or "chemical" in raw:
        return []
    return genes


def collect_role_rows(data_dir: Path, split: dict[str, Any], metadata: dict[str, Any], role: str, *, max_cells: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ds in FOCUS_DATASETS:
        obj = split.get(ds) or {}
        path = data_dir / f"{ds}.h5"
        if not path.is_file():
            continue
        conds = obj.get(role) or obj.get("test_multi" if role == "query_multi" else role) or []
        with h5py.File(path, "r") as handle:
            by_cond = {c: i for i, c in enumerate(decode(np.asarray(handle["conditions"])))}
            for cond in conds:
                cond = str(cond)
                genes = genes_for(metadata, ds, cond)
                if len(genes) < 2:
                    continue
                vals = arrays_for_condition(handle, by_cond, cond, max_cells)
                if vals is None:
                    continue
                ctrl_cells, gt_cells, ctrl, gt, residual = vals
                rows.append(
                    {
                        "dataset": ds,
                        "condition": cond,
                        "role": role,
                        "genes": genes,
                        "nperts": len(genes),
                        "ctrl": ctrl,
                        "gt": gt,
                        "ctrl_cells": ctrl_cells,
                        "gt_cells": gt_cells,
                        "residual": residual,
                        "query_stratum": query_stratum(obj, cond),
                    }
                )
    return rows


def query_stratum(obj: dict[str, Any], cond: str) -> str:
    for key in ("query_multi_seen", "query_multi_unseen1", "query_multi_unseen2"):
        if cond in set(map(str, obj.get(key) or [])):
            return key
    return "query_multi"


def train_single_components(data_dir: Path, split: dict[str, Any], metadata: dict[str, Any], *, max_cells: int) -> dict[str, Any]:
    by_gene: dict[str, list[np.ndarray]] = defaultdict(list)
    by_ds: dict[str, list[np.ndarray]] = defaultdict(list)
    for ds in FOCUS_DATASETS:
        obj = split.get(ds) or {}
        path = data_dir / f"{ds}.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            by_cond = {c: i for i, c in enumerate(decode(np.asarray(handle["conditions"])))}
            for cond in obj.get("train_single") or obj.get("train") or []:
                cond = str(cond)
                genes = genes_for(metadata, ds, cond)
                if len(genes) != 1:
                    continue
                vals = residual_for_condition(handle, by_cond, cond, max_cells)
                if vals is None:
                    continue
                _ctrl, _gt, residual = vals
                by_gene[genes[0]].append(residual)
                by_ds[ds].append(residual)
    all_res = [x for vals in by_gene.values() for x in vals]
    global_single = np.mean(np.vstack(all_res), axis=0).astype(np.float32)
    return {
        "gene_raw_mean": {g: np.mean(np.vstack(v), axis=0).astype(np.float32) for g, v in by_gene.items()},
        "dataset_single_mean": {d: np.mean(np.vstack(v), axis=0).astype(np.float32) for d, v in by_ds.items()},
        "global_single_mean": global_single,
    }


def train_multi_components(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in train_rows:
        by_ds[str(row["dataset"])].append(np.asarray(row["residual"], dtype=np.float32))
    global_multi = np.mean(np.vstack([r["residual"] for r in train_rows]), axis=0).astype(np.float32)
    return {
        "dataset_multi_mean": {d: np.mean(np.vstack(v), axis=0).astype(np.float32) for d, v in by_ds.items()},
        "global_multi_mean": global_multi,
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 3 or x.size != y.size:
        return None
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def score(row: dict[str, Any], pred_residual: np.ndarray, pert_means: dict[str, np.ndarray]) -> float | None:
    pert = pert_means.get(str(row["dataset"]))
    pred_endpoint = np.asarray(row["ctrl"], dtype=np.float32) + np.asarray(pred_residual, dtype=np.float32)
    gt_endpoint = np.asarray(row["gt"], dtype=np.float32)
    if pert is None:
        return pearson(pred_residual, row["residual"])
    return pearson(pred_endpoint - pert, gt_endpoint - pert)


def mmd_scores(row: dict[str, Any], pred_residual: np.ndarray) -> dict[str, float]:
    pred_cells = np.asarray(row["ctrl_cells"], dtype=np.float32) + np.asarray(pred_residual, dtype=np.float32)[None, :]
    gt_cells = np.asarray(row["gt_cells"], dtype=np.float32)
    pred_t = torch.from_numpy(pred_cells).float()
    gt_t = torch.from_numpy(gt_cells).float()
    sigmas, dyy = median_sigmas(gt_t, return_D2=True)
    raw = float(mmd2_unbiased(pred_t, gt_t, sigmas, Dyy=dyy).item())
    biased = float(mmd2_biased(pred_t, gt_t, sigmas, Dyy=dyy).item())
    return {"test_mmd": raw, "test_mmd_biased": biased, "test_mmd_clamped": max(raw, 0.0)}


def predict_baselines(row: dict[str, Any], single: dict[str, Any], multi: dict[str, Any]) -> dict[str, np.ndarray]:
    ds = str(row["dataset"])
    genes = list(row["genes"])
    gene_terms = [single["gene_raw_mean"][g] for g in genes if g in single["gene_raw_mean"]]
    if gene_terms:
        additive_mean = np.mean(np.vstack(gene_terms), axis=0).astype(np.float32)
        additive_sum = np.sum(np.vstack(gene_terms), axis=0).astype(np.float32)
    else:
        additive_mean = single["global_single_mean"]
        additive_sum = single["global_single_mean"]
    preds = {
        "dataset_multi_mean": multi["dataset_multi_mean"].get(ds, multi["global_multi_mean"]),
        "global_multi_mean": multi["global_multi_mean"],
        "additive_single_mean": additive_mean,
        "additive_single_sum": additive_sum,
        "dataset_single_mean": single["dataset_single_mean"].get(ds, single["global_single_mean"]),
    }
    preds["support_selected_route"] = preds[ROUTE[ds]]
    return preds


def evaluate(query_rows: list[dict[str, Any]], single: dict[str, Any], multi: dict[str, Any], pert_means: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    out = []
    for row in query_rows:
        preds = predict_baselines(row, single, multi)
        scored = {
            "dataset": row["dataset"],
            "condition": row["condition"],
            "query_stratum": row["query_stratum"],
            "nperts": row["nperts"],
        }
        for name, pred in preds.items():
            scored[name] = score(row, pred, pert_means)
            ms = mmd_scores(row, pred)
            for key, value in ms.items():
                scored[f"{name}__{key}"] = value
        out.append(scored)
    return out


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = row.get(key)
        if val is not None:
            by_ds[str(row["dataset"])].append(float(val))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


def group_scores(rows: list[dict[str, Any]], models: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    groups = ["query_multi"] + sorted({str(r["query_stratum"]) for r in rows})
    for group in groups:
        sub = rows if group == "query_multi" else [r for r in rows if r["query_stratum"] == group]
        if not sub:
            continue
        for model in models:
            out.append({"group": group, "model": model, "mean": equal_dataset_mean(sub, model), "n_conditions": len(sub)})
    return out


def group_mmd_scores(rows: list[dict[str, Any]], models: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    groups = ["query_multi"] + sorted({str(r["query_stratum"]) for r in rows})
    for group in groups:
        sub = rows if group == "query_multi" else [r for r in rows if r["query_stratum"] == group]
        if not sub:
            continue
        for model in models:
            out.append(
                {
                    "group": group,
                    "model": model,
                    "mean_mmd_clamped": equal_dataset_mean(sub, f"{model}__test_mmd_clamped"),
                    "mean_mmd_biased": equal_dataset_mean(sub, f"{model}__test_mmd_biased"),
                    "n_conditions": len(sub),
                }
            )
    return out


def dataset_breakdown(rows: list[dict[str, Any]], models: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for ds in sorted({str(r["dataset"]) for r in rows}):
        sub = [r for r in rows if str(r["dataset"]) == ds]
        item: dict[str, Any] = {"dataset": ds, "n_conditions": len(sub)}
        for model in models:
            vals = [float(r[model]) for r in sub if r.get(model) is not None]
            item[model] = None if not vals else float(np.mean(vals))
        out.append(item)
    return out


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    diffs_by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        a = row.get(candidate)
        b = row.get(baseline)
        if a is not None and b is not None:
            diffs_by_ds[str(row["dataset"])].append(float(a) - float(b))
    datasets = sorted(ds for ds, vals in diffs_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline}
    point = float(np.mean([np.mean(diffs_by_ds[d]) for d in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for ds in sample_ds:
            vals = np.asarray(diffs_by_ds[str(ds)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(means)))
    arr = np.asarray(boot)
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "n_conditions": int(sum(len(diffs_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
    }


def paired_bootstrap_mmd(rows: list[dict[str, Any]], candidate: str, baseline: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    diffs_by_ds: dict[str, list[float]] = defaultdict(list)
    ck = f"{candidate}__test_mmd_clamped"
    bk = f"{baseline}__test_mmd_clamped"
    for row in rows:
        a = row.get(ck)
        b = row.get(bk)
        if a is not None and b is not None:
            diffs_by_ds[str(row["dataset"])].append(float(a) - float(b))
    datasets = sorted(ds for ds, vals in diffs_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline, "metric": "test_mmd_clamped"}
    point = float(np.mean([np.mean(diffs_by_ds[d]) for d in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        means = []
        for ds in sample_ds:
            vals = np.asarray(diffs_by_ds[str(ds)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(means)))
    arr = np.asarray(boot)
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "metric": "test_mmd_clamped",
        "n_conditions": int(sum(len(diffs_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr < 0.0)),
        "p_harm": float(np.mean(arr > 0.0)),
    }


def decide(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["baseline"]: r for r in deltas if r.get("candidate") == "support_selected_route"}
    reasons = []
    for baseline in ("dataset_multi_mean", "additive_single_sum", "additive_single_mean"):
        row = by.get(baseline) or {}
        if row.get("status") != "ok":
            reasons.append(f"{baseline}_comparison_missing")
            continue
        if baseline == "additive_single_mean":
            if not (float(row.get("delta_mean") or 0.0) >= 0.02 or float((row.get("ci95") or [0.0])[0]) > 0.0):
                reasons.append("route_not_materially_better_than_additive_single_mean")
        elif float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.20:
            reasons.append(f"{baseline}_harm_risk")
    status = "query_diagnostic_route_supported" if not reasons else "query_diagnostic_route_not_supported"
    return {"status": status, "action": "do_not_reuse_query_for_selection", "reasons": reasons}


def decide_with_mmd(pp_deltas: list[dict[str, Any]], mmd_deltas: list[dict[str, Any]]) -> dict[str, Any]:
    dec = decide(pp_deltas)
    reasons = list(dec.get("reasons") or [])
    by = {r["baseline"]: r for r in mmd_deltas if r.get("candidate") == "support_selected_route"}
    for baseline in ("dataset_multi_mean", "additive_single_sum", "additive_single_mean"):
        row = by.get(baseline) or {}
        if row.get("status") != "ok":
            reasons.append(f"{baseline}_mmd_comparison_missing")
        elif float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.80:
            reasons.append(f"{baseline}_mmd_hard_harm")
    status = "query_diagnostic_route_supported_with_mmd_no_hard_harm" if not reasons else "query_diagnostic_route_not_supported_with_mmd"
    return {"status": status, "action": "do_not_reuse_query_for_selection", "reasons": reasons}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Track C Support-Selected Routed Baseline Query Diagnostic",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- split_file: `{payload['split_file']}`",
        f"- data_dir: `{payload['data_dir']}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- route: `{payload['route']}`",
        f"- train_multi rows: `{payload['n_train_multi_rows']}`",
        f"- query_multi rows evaluated once: `{payload['n_query_multi_rows']}`",
        "",
        "## Query Scores",
        "",
        "| group | model | n cond | equal-dataset pp |",
        "|---|---|---:|---:|",
    ]
    for row in payload["group_scores"]:
        lines.append(f"| {row['group']} | `{row['model']}` | {row['n_conditions']} | {fmt(row['mean'])} |")
    lines += [
        "",
        "## Query MMD Scores",
        "",
        "| group | model | n cond | MMD clamped | MMD biased |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["group_mmd_scores"]:
        lines.append(
            f"| {row['group']} | `{row['model']}` | {row['n_conditions']} | "
            f"{fmt(row.get('mean_mmd_clamped'))} | {fmt(row.get('mean_mmd_biased'))} |"
        )
    lines += [
        "",
        "## Dataset Breakdown",
        "",
        "| dataset | n cond | route pp | dataset_multi pp | additive_mean pp | additive_sum pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('support_selected_route'))} | "
            f"{fmt(row.get('dataset_multi_mean'))} | {fmt(row.get('additive_single_mean'))} | "
            f"{fmt(row.get('additive_single_sum'))} |"
        )
    lines += [
        "",
        "## Paired Deltas",
        "",
        "| candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | status |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {row['candidate']} | {row['baseline']} | {row.get('n_conditions', 0)} | "
            f"{row.get('n_datasets', 0)} | {fmt(row.get('delta_mean'))} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row.get('p_improve'))} | "
            f"{fmt(row.get('p_harm'))} | {row.get('status')} |"
        )
    lines += [
        "",
        "## Paired MMD Deltas",
        "",
        "| candidate | baseline | n cond | n ds | MMD delta | 95% CI | p improve | p harm | status |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["paired_mmd_deltas"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {row['candidate']} | {row['baseline']} | {row.get('n_conditions', 0)} | "
            f"{row.get('n_datasets', 0)} | {fmt(row.get('delta_mean'))} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(row.get('p_improve'))} | "
            f"{fmt(row.get('p_harm'))} | {row.get('status')} |"
        )
    lines += ["", "## Decision Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Usage Rule",
        "",
        "- This is the one-shot Track C query diagnostic for this support-selected route.",
        "- Do not tune route/model choices on this query result.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    split = load_json(args.split_file)
    manifest = load_json(data_dir / "manifest.json")
    metadata = load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}
    train_multi = collect_role_rows(data_dir, split, metadata, "train_multi", max_cells=args.max_cells_per_condition)
    query_multi = collect_role_rows(data_dir, split, metadata, "query_multi", max_cells=args.max_cells_per_condition)
    single = train_single_components(data_dir, split, metadata, max_cells=args.max_cells_per_condition)
    multi = train_multi_components(train_multi)
    eval_rows = evaluate(query_multi, single, multi, pert_means)
    deltas = [
        paired_bootstrap(eval_rows, "support_selected_route", baseline, n_boot=args.n_boot, seed=args.seed + i)
        for i, baseline in enumerate(m for m in MODELS if m != "support_selected_route")
    ]
    mmd_deltas = [
        paired_bootstrap_mmd(eval_rows, "support_selected_route", baseline, n_boot=args.n_boot, seed=args.seed + 100 + i)
        for i, baseline in enumerate(m for m in MODELS if m != "support_selected_route")
    ]
    payload = {
        "data_dir": str(data_dir),
        "split_file": str(args.split_file),
        "pert_means_file": str(args.pert_means_file),
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "route": ROUTE,
        "leakage_status": "route_fixed_from_support_val_then_query_multi_evaluated_once_no_query_selection_no_canonical_test_no_posthoc",
        "n_train_multi_rows": len(train_multi),
        "n_query_multi_rows": len(query_multi),
        "group_scores": group_scores(eval_rows, MODELS),
        "group_mmd_scores": group_mmd_scores(eval_rows, MODELS),
        "dataset_breakdown": dataset_breakdown(eval_rows, MODELS),
        "paired_deltas": deltas,
        "paired_mmd_deltas": mmd_deltas,
        "decision": decide_with_mmd(deltas, mmd_deltas),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md), "n_query_multi_rows": len(query_multi)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
