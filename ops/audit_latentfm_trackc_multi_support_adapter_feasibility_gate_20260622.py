#!/usr/bin/env python3
"""Track C support-only feasibility gate for true-multi adaptation.

This CPU gate uses only `train_multi` and `support_val_multi` from the
multi-support v2 split. It never evaluates `query_multi` and never reads
canonical test/posthoc outcomes. The goal is to decide whether a tiny true-multi
adapter GPU smoke is even justified.
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


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_GENE_CACHE = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_multi_support_adapter_feasibility_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_MULTI_SUPPORT_ADAPTER_FEASIBILITY_GATE_20260622.md"
FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def condition_mean(handle: h5py.File, group: str, idx: int, max_cells: int) -> np.ndarray | None:
    offsets = np.asarray(handle[f"{group}/offsets"])
    start, end = int(offsets[idx]), int(offsets[idx + 1])
    if end <= start:
        return None
    if max_cells > 0 and end - start > max_cells:
        end = start + max_cells
    return np.asarray(handle[f"{group}/emb"][start:end], dtype=np.float32).mean(axis=0)


def residual_for_condition(handle: h5py.File, by_cond: dict[str, int], cond: str, max_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    idx = by_cond.get(cond)
    if idx is None:
        return None
    ctrl = condition_mean(handle, "ctrl", idx, max_cells)
    gt = condition_mean(handle, "gt", idx, max_cells)
    if ctrl is None or gt is None:
        return None
    return ctrl.astype(np.float32), gt.astype(np.float32), (gt - ctrl).astype(np.float32)


def genes_for(metadata: dict[str, Any], ds: str, cond: str) -> list[str]:
    meta = (metadata.get(ds) or {}).get(cond) or {}
    genes = [str(g).strip() for g in meta.get("genes") or [] if str(g).strip()]
    raw = str(meta.get("perturbation_type_raw") or "").lower()
    if "drug" in raw or "compound" in raw or "chemical" in raw:
        return []
    return genes


def normalize_gene(gene: str) -> str:
    return str(gene).strip().upper()


def load_gene_embeddings(cache_dir: Path) -> tuple[dict[str, int], np.ndarray, int]:
    emb = np.load(str(cache_dir / "gene_embeddings.npy"), mmap_mode="r")
    mapping: dict[str, int] = {}
    for line in (cache_dir / "gene_index.tsv").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].lower() in {"symbol", "gene", "gene_symbol"}:
            continue
        if len(parts) < 2:
            parts = line.split(None, 1)
        if len(parts) >= 2:
            mapping[normalize_gene(parts[0])] = int(parts[-1].strip().split()[0])
    unk_index = int(load_json(cache_dir / "manifest.json").get("unk_index", 1))
    return mapping, np.asarray(emb), unk_index


def gene_vec(gene: str, mapping: dict[str, int], emb: np.ndarray, unk_index: int) -> np.ndarray:
    idx = int(mapping.get(normalize_gene(gene), unk_index))
    return np.asarray(emb[idx], dtype=np.float32)


def collect_rows(
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    mapping: dict[str, int],
    emb: np.ndarray,
    unk_index: int,
    *,
    max_cells: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for ds in FOCUS_DATASETS:
        obj = split.get(ds) or {}
        path = data_dir / f"{ds}.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as handle:
            by_cond = {c: i for i, c in enumerate(decode(np.asarray(handle["conditions"])))}
            for role, conds, dest in (
                ("train_multi", obj.get("train_multi") or [], train_rows),
                ("support_val_multi", obj.get("support_val_multi") or [], val_rows),
            ):
                for cond in conds:
                    cond = str(cond)
                    genes = genes_for(metadata, ds, cond)
                    if len(genes) < 2:
                        continue
                    vals = residual_for_condition(handle, by_cond, cond, max_cells)
                    if vals is None:
                        continue
                    ctrl, gt, residual = vals
                    gvecs = np.vstack([gene_vec(g, mapping, emb, unk_index) for g in genes]).astype(np.float32)
                    dest.append(
                        {
                            "dataset": ds,
                            "condition": cond,
                            "role": role,
                            "genes": genes,
                            "nperts": len(genes),
                            "gene_sum": gvecs.sum(axis=0),
                            "gene_mean": gvecs.mean(axis=0),
                            "gene_pair_mean": pairwise_hadamard_mean(gvecs),
                            "ctrl": ctrl,
                            "gt": gt,
                            "residual": residual,
                        }
                    )
    return train_rows, val_rows


def pairwise_hadamard_mean(gvecs: np.ndarray) -> np.ndarray:
    if gvecs.shape[0] < 2:
        return np.zeros(gvecs.shape[1], dtype=np.float32)
    vals = []
    for i in range(gvecs.shape[0]):
        for j in range(i + 1, gvecs.shape[0]):
            vals.append(gvecs[i] * gvecs[j])
    return np.mean(np.stack(vals), axis=0).astype(np.float32)


def fit_pca(x: np.ndarray, k: int) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    n = min(int(k), vt.shape[0])
    return {"mean": mean.astype(np.float32), "components": vt[:n].astype(np.float32)}


def transform_pca(x: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    comps = np.asarray(pca["components"], dtype=np.float32)
    return ((np.asarray(x, dtype=np.float32) - pca["mean"]) @ comps.T).astype(np.float32)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    xs, mean, std = standardize_fit(np.asarray(x, dtype=np.float32))
    x_aug = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float32), xs], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ np.asarray(y, dtype=np.float64))
    return {"coef": coef.astype(np.float32), "mean": mean, "std": std}


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = standardize_apply(np.asarray(x, dtype=np.float32), model["mean"], model["std"])
    x_aug = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float32), xs], axis=1)
    return (x_aug @ model["coef"]).astype(np.float32)


def make_features(rows: list[dict[str, Any]], pcas: dict[str, dict[str, np.ndarray]] | None = None, *, pcs: int) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    mats = {
        "sum": np.vstack([r["gene_sum"] for r in rows]).astype(np.float32),
        "mean": np.vstack([r["gene_mean"] for r in rows]).astype(np.float32),
        "pair": np.vstack([r["gene_pair_mean"] for r in rows]).astype(np.float32),
        "ctrl": np.vstack([r["ctrl"] for r in rows]).astype(np.float32),
    }
    if pcas is None:
        pcas = {k: fit_pca(v, pcs) for k, v in mats.items()}
    parts = [transform_pca(mats[k], pcas[k]) for k in ("sum", "mean", "pair", "ctrl")]
    nperts = np.asarray([[float(r["nperts"])] for r in rows], dtype=np.float32)
    return np.concatenate(parts + [nperts], axis=1), pcas


def single_gene_components(data_dir: Path, split: dict[str, Any], metadata: dict[str, Any], *, max_cells: int) -> dict[str, Any]:
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
    global_mean = np.mean(np.vstack(all_res), axis=0).astype(np.float32)
    return {
        "gene_raw_mean": {g: np.mean(np.vstack(v), axis=0).astype(np.float32) for g, v in by_gene.items()},
        "dataset_single_mean": {d: np.mean(np.vstack(v), axis=0).astype(np.float32) for d, v in by_ds.items()},
        "global_single_mean": global_mean,
    }


def multi_baselines(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def evaluate(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], single: dict[str, Any], pert_means: dict[str, np.ndarray], *, pcs: int, alpha: float) -> list[dict[str, Any]]:
    x_train, pcas = make_features(train_rows, pcs=pcs)
    x_val, _ = make_features(val_rows, pcas, pcs=pcs)
    y_train = np.vstack([r["residual"] for r in train_rows]).astype(np.float32)
    ridge = fit_ridge(x_train, y_train, alpha)
    pred_ridge = predict_ridge(ridge, x_val)
    multi = multi_baselines(train_rows)
    out = []
    for i, row in enumerate(val_rows):
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
            "multi_support_ridge": pred_ridge[i],
            "dataset_multi_mean": multi["dataset_multi_mean"].get(ds, multi["global_multi_mean"]),
            "global_multi_mean": multi["global_multi_mean"],
            "additive_single_mean": additive_mean,
            "additive_single_sum": additive_sum,
            "dataset_single_mean": single["dataset_single_mean"].get(ds, single["global_single_mean"]),
        }
        scored = {"dataset": ds, "condition": row["condition"], "nperts": row["nperts"], "group": "support_val_multi"}
        for name, pred in preds.items():
            scored[name] = score(row, pred, pert_means)
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


def dataset_breakdown(rows: list[dict[str, Any]], models: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for ds in sorted({str(r["dataset"]) for r in rows}):
        ds_rows = [r for r in rows if str(r["dataset"]) == ds]
        item: dict[str, Any] = {"dataset": ds, "n_conditions": len(ds_rows)}
        for model in models:
            vals = [float(r[model]) for r in ds_rows if r.get(model) is not None]
            item[model] = None if not vals else float(np.mean(vals))
        cand = item.get("multi_support_ridge")
        for baseline in models:
            if baseline == "multi_support_ridge":
                continue
            base = item.get(baseline)
            item[f"delta_vs_{baseline}"] = None if cand is None or base is None else float(cand) - float(base)
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


def decide(deltas: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["baseline"]: r for r in deltas if r.get("candidate") == "multi_support_ridge"}
    reasons = []
    material = by.get("additive_single_mean") or {}
    if material.get("status") != "ok" or not (
        float(material.get("delta_mean") or 0.0) >= 0.02 or float((material.get("ci95") or [0.0])[0]) > 0.0
    ):
        reasons.append("support_val_multi_not_materially_better_than_additive_single_mean")
    for baseline in ("dataset_multi_mean", "global_multi_mean", "additive_single_sum"):
        row = by.get(baseline) or {}
        if row.get("status") != "ok":
            reasons.append(f"{baseline}_comparison_missing")
        elif float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.20:
            reasons.append(f"{baseline}_harm_risk")
    status = "cpu_gate_pass_launch_one_trackc_adapter_smoke" if not reasons else "cpu_gate_fail_do_not_launch_gpu"
    return {
        "status": status,
        "action": "launch_one_trackc_pairwise_adapter_smoke" if not reasons else "keep_trackc_support_cpu_only",
        "reasons": reasons,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Track C Multi-Support Adapter Feasibility Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- split_file: `{payload['split_file']}`",
        f"- data_dir: `{payload['data_dir']}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- gene_cache_dir: `{payload['gene_cache_dir']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- train_multi rows: `{payload['n_train_multi_rows']}`",
        f"- support_val_multi rows: `{payload['n_support_val_multi_rows']}`",
        "",
        "## Absolute Scores",
        "",
        "| model | equal-dataset support-val pp proxy |",
        "|---|---:|",
    ]
    for row in payload["absolute_scores"]:
        lines.append(f"| `{row['model']}` | {fmt(row['mean'])} |")
    lines += [
        "",
        "## Dataset Breakdown",
        "",
        "| dataset | n cond | ridge pp | dataset_multi pp | additive_mean pp | additive_sum pp | delta vs dataset_multi | delta vs additive_mean | delta vs additive_sum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('multi_support_ridge'))} | "
            f"{fmt(row.get('dataset_multi_mean'))} | {fmt(row.get('additive_single_mean'))} | "
            f"{fmt(row.get('additive_single_sum'))} | {fmt(row.get('delta_vs_dataset_multi_mean'))} | "
            f"{fmt(row.get('delta_vs_additive_single_mean'))} | {fmt(row.get('delta_vs_additive_single_sum'))} |"
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
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Usage Rule",
        "",
        "- This gate uses only Track C support train/val. It does not evaluate query multi.",
        "- Passing would only justify one adapter-only GPU smoke; final multi query remains held out.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--gene-cache-dir", type=Path, default=DEFAULT_GENE_CACHE)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--pcs", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    split = load_json(args.split_file)
    manifest = load_json(data_dir / "manifest.json")
    metadata = load_json(Path(manifest["condition_metadata_file"]))
    mapping, emb, unk_index = load_gene_embeddings(args.gene_cache_dir)
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}
    train_rows, val_rows = collect_rows(
        data_dir,
        split,
        metadata,
        mapping,
        emb,
        unk_index,
        max_cells=args.max_cells_per_condition,
    )
    single = single_gene_components(data_dir, split, metadata, max_cells=args.max_cells_per_condition)
    eval_rows = evaluate(train_rows, val_rows, single, pert_means, pcs=args.pcs, alpha=args.ridge_alpha)
    models = (
        "multi_support_ridge",
        "dataset_multi_mean",
        "global_multi_mean",
        "additive_single_mean",
        "additive_single_sum",
        "dataset_single_mean",
    )
    absolute = [{"model": m, "mean": equal_dataset_mean(eval_rows, m)} for m in models]
    by_dataset = dataset_breakdown(eval_rows, models)
    deltas = []
    for baseline in models:
        if baseline == "multi_support_ridge":
            continue
        deltas.append(paired_bootstrap(eval_rows, "multi_support_ridge", baseline, n_boot=args.n_boot, seed=args.seed + len(deltas)))
    payload = {
        "data_dir": str(data_dir),
        "split_file": str(args.split_file),
        "pert_means_file": str(args.pert_means_file),
        "gene_cache_dir": str(args.gene_cache_dir),
        "max_cells_per_condition": args.max_cells_per_condition,
        "pcs": args.pcs,
        "ridge_alpha": args.ridge_alpha,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "leakage_status": "trackc_train_multi_to_support_val_multi_only_no_query_multi_no_canonical_test_no_posthoc",
        "n_train_multi_rows": len(train_rows),
        "n_support_val_multi_rows": len(val_rows),
        "absolute_scores": absolute,
        "dataset_breakdown": by_dataset,
        "paired_deltas": deltas,
        "decision": decide(deltas),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md), "n_train_multi_rows": len(train_rows), "n_support_val_multi_rows": len(val_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
