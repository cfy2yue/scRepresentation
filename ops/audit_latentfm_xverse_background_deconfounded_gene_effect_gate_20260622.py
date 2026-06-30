#!/usr/bin/env python3
"""CPU gate for background-deconfounded gene-effect residuals.

This tests whether a simple train-only two-way decomposition,

    residual ~= dataset/background mean + perturbation gene effect,

contains Track A pp-direction signal beyond a dataset-only residual control.
It uses the xverse train-only v2 proxy split and never reads canonical test
outcomes, posthoc predictions, or held-out multi GT.
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
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_background_deconfounded_gene_effect_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_BACKGROUND_DECONFOUNDED_GENE_EFFECT_GATE_20260622.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def stable_subset(items: list[str], k: int, key: str) -> list[str]:
    if k <= 0 or len(items) <= k:
        return list(items)
    return sorted(items, key=lambda x: hashlib.sha256(f"{key}|{x}".encode()).hexdigest())[:k]


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


def single_gene(metadata: dict[str, Any], ds: str, cond: str) -> str | None:
    meta = (metadata.get(ds) or {}).get(cond) or {}
    genes = [str(g) for g in meta.get("genes") or []]
    if len(genes) != 1:
        return None
    raw = str(meta.get("perturbation_type_raw") or "").lower()
    if "drug" in raw or "compound" in raw or "chemical" in raw:
        return None
    return genes[0]


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


def collect_rows(
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    *,
    max_train_per_dataset: int,
    max_cells: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for ds, obj in sorted(split.items()):
        path = data_dir / f"{ds}.h5"
        if not path.is_file():
            continue
        train_single = []
        for cond in obj.get("train") or []:
            cond = str(cond)
            gene = single_gene(metadata, ds, cond)
            if gene is not None:
                train_single.append((cond, gene))
        chosen_train = set(stable_subset([c for c, _ in train_single], max_train_per_dataset, f"p13train|{ds}"))
        with h5py.File(path, "r") as handle:
            by_cond = {c: i for i, c in enumerate(decode(np.asarray(handle["conditions"])))}
            for cond, gene in train_single:
                if cond not in chosen_train:
                    continue
                vals = residual_for_condition(handle, by_cond, cond, max_cells)
                if vals is None:
                    continue
                ctrl, gt, residual = vals
                train_rows.append({"dataset": ds, "condition": cond, "gene": gene, "ctrl": ctrl, "gt": gt, "residual": residual})
            for group in GROUPS:
                for cond in obj.get(group) or []:
                    cond = str(cond)
                    gene = single_gene(metadata, ds, cond)
                    if gene is None:
                        continue
                    vals = residual_for_condition(handle, by_cond, cond, max_cells)
                    if vals is None:
                        continue
                    ctrl, gt, residual = vals
                    val_rows.append(
                        {
                            "dataset": ds,
                            "condition": cond,
                            "gene": gene,
                            "group": group,
                            "ctrl": ctrl,
                            "gt": gt,
                            "residual": residual,
                        }
                    )
    return train_rows, val_rows


def build_components(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in train_rows:
        by_ds[str(row["dataset"])].append(np.asarray(row["residual"], dtype=np.float32))
    ds_mean = {ds: np.mean(np.vstack(vals), axis=0).astype(np.float32) for ds, vals in by_ds.items()}
    global_mean = np.mean(np.vstack([r["residual"] for r in train_rows]), axis=0).astype(np.float32)

    gene_raw: dict[str, list[np.ndarray]] = defaultdict(list)
    gene_effect: dict[str, list[np.ndarray]] = defaultdict(list)
    gene_ds: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        ds = str(row["dataset"])
        gene = str(row["gene"])
        residual = np.asarray(row["residual"], dtype=np.float32)
        gene_raw[gene].append(residual)
        gene_effect[gene].append((residual - ds_mean[ds]).astype(np.float32))
        gene_ds[gene].add(ds)
    return {
        "dataset_mean": ds_mean,
        "global_mean": global_mean,
        "gene_raw_mean": {g: np.mean(np.vstack(vals), axis=0).astype(np.float32) for g, vals in gene_raw.items()},
        "gene_effect_mean": {g: np.mean(np.vstack(vals), axis=0).astype(np.float32) for g, vals in gene_effect.items()},
        "gene_dataset_count": {g: len(v) for g, v in gene_ds.items()},
    }


def score(row: dict[str, Any], pred_residual: np.ndarray, pert_means: dict[str, np.ndarray]) -> float | None:
    pert = pert_means.get(str(row["dataset"]))
    pred_endpoint = np.asarray(row["ctrl"], dtype=np.float32) + np.asarray(pred_residual, dtype=np.float32)
    gt_endpoint = np.asarray(row["gt"], dtype=np.float32)
    if pert is None:
        return pearson(pred_residual, row["residual"])
    return pearson(pred_endpoint - pert, gt_endpoint - pert)


def evaluate(val_rows: list[dict[str, Any]], components: dict[str, Any], pert_means: dict[str, np.ndarray], *, seed: int) -> list[dict[str, Any]]:
    out = []
    rng = np.random.default_rng(seed)
    genes = sorted(components["gene_effect_mean"])
    shuffled_map = {g: components["gene_effect_mean"][genes[i]] for i, g in enumerate(rng.permutation(genes))}
    for row in val_rows:
        ds = str(row["dataset"])
        gene = str(row["gene"])
        ds_mean = components["dataset_mean"].get(ds, components["global_mean"])
        global_mean = components["global_mean"]
        gene_raw = components["gene_raw_mean"].get(gene, global_mean)
        gene_effect = components["gene_effect_mean"].get(gene)
        shuffled_effect = shuffled_map.get(gene)
        preds = {
            "dataset_mean": ds_mean,
            "global_mean": global_mean,
            "gene_raw_mean": gene_raw,
            "dataset_plus_gene_effect": ds_mean if gene_effect is None else ds_mean + gene_effect,
            "dataset_plus_shuffled_gene_effect": ds_mean if shuffled_effect is None else ds_mean + shuffled_effect,
        }
        scored = {
            "dataset": ds,
            "condition": row["condition"],
            "gene": gene,
            "group": row["group"],
            "gene_dataset_count": components["gene_dataset_count"].get(gene, 0),
        }
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


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    diffs_by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        a = row.get(candidate)
        b = row.get(baseline)
        if a is not None and b is not None:
            diffs_by_ds[str(row["dataset"])].append(float(a) - float(b))
    datasets = sorted(ds for ds, vals in diffs_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing"}
    point = float(np.mean([np.mean(diffs_by_ds[ds]) for ds in datasets]))
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
    leave = {}
    for ds in datasets:
        rest = [d for d in datasets if d != ds]
        if rest:
            leave[ds] = float(np.mean([np.mean(diffs_by_ds[d]) for d in rest]))
    return {
        "status": "ok",
        "n_conditions": int(sum(len(diffs_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "leave_one_min": min(leave.values()) if leave else None,
        "leave_one_max": max(leave.values()) if leave else None,
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
        "# LatentFM xverse Background-Deconfounded Gene-Effect Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- train rows: `{payload['n_train_rows']}`",
        f"- validation rows: `{payload['n_val_rows']}`",
        "",
        "## Absolute Scores",
        "",
        "| group | model | equal-dataset pp proxy |",
        "|---|---|---:|",
    ]
    for row in payload["absolute_scores"]:
        lines.append(f"| {row['group']} | {row['model']} | {fmt(row['mean'])} |")
    lines += [
        "",
        "## Paired Deltas",
        "",
        "| group | candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | leave-one min | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {row['group']} | {row['candidate']} | {row['baseline']} | "
            f"{row.get('n_conditions', 0)} | {row.get('n_datasets', 0)} | "
            f"{fmt(row.get('delta_mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row.get('p_improve'))} | {fmt(row.get('p_harm'))} | "
            f"{fmt(row.get('leave_one_min'))} | {row.get('status')} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Interpretation",
        "",
        "- Passing would justify one tiny background-orthogonal condition adapter smoke.",
        "- Failing means the simple train-only two-way dataset+gene decomposition is diagnostic only.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--max-train-per-dataset", type=int, default=0)
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
    train_rows, val_rows = collect_rows(
        data_dir,
        split,
        metadata,
        max_train_per_dataset=args.max_train_per_dataset,
        max_cells=args.max_cells_per_condition,
    )
    components = build_components(train_rows)
    eval_rows = evaluate(val_rows, components, pert_means, seed=args.seed)
    models = [
        "dataset_plus_gene_effect",
        "dataset_mean",
        "gene_raw_mean",
        "global_mean",
        "dataset_plus_shuffled_gene_effect",
    ]
    absolute_scores = []
    paired_deltas = []
    for group in GROUPS:
        rows = [r for r in eval_rows if r["group"] == group]
        for model in models:
            absolute_scores.append({"group": group, "model": model, "mean": equal_dataset_mean(rows, model)})
        for baseline in ("dataset_mean", "gene_raw_mean", "global_mean", "dataset_plus_shuffled_gene_effect"):
            row = paired_bootstrap(
                rows,
                "dataset_plus_gene_effect",
                baseline,
                n_boot=args.n_boot,
                seed=args.seed + len(paired_deltas),
            )
            row.update({"group": group, "candidate": "dataset_plus_gene_effect", "baseline": baseline})
            paired_deltas.append(row)

    by = {(r["group"], r["baseline"]): r for r in paired_deltas}
    cross_ds = by[(GROUPS[0], "dataset_mean")]
    fam_ds = by[(GROUPS[1], "dataset_mean")]
    cross_shuf = by[(GROUPS[0], "dataset_plus_shuffled_gene_effect")]
    fam_shuf = by[(GROUPS[1], "dataset_plus_shuffled_gene_effect")]
    reasons = []
    if cross_ds.get("status") != "ok" or not (
        float(cross_ds.get("p_improve") or 0.0) >= 0.90 or float((cross_ds.get("ci95") or [0.0])[0]) > 0.0
    ):
        reasons.append("cross_background_gene_effect_not_better_than_dataset_mean")
    if fam_ds.get("status") != "ok" or float(fam_ds.get("p_harm") if fam_ds.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("family_gene_effect_vs_dataset_mean_harm_risk")
    if cross_shuf.get("status") != "ok" or float(cross_shuf.get("delta_mean") or 0.0) <= 0.0:
        reasons.append("cross_background_not_better_than_shuffled_gene_effect")
    if fam_shuf.get("status") != "ok" or float(fam_shuf.get("delta_mean") or 0.0) <= 0.0:
        reasons.append("family_not_better_than_shuffled_gene_effect")
    if cross_ds.get("leave_one_min") is None or float(cross_ds["leave_one_min"]) <= 0.0:
        reasons.append("cross_background_leave_one_dataset_flips_or_nonpositive")

    status = "cpu_gate_pass_launch_one_background_orthogonal_smoke" if not reasons else "cpu_gate_fail_do_not_launch_gpu"
    action = "launch_one_background_orthogonal_adapter_smoke" if not reasons else "keep_background_deconfounding_cpu_only"
    payload = {
        "data_dir": str(data_dir),
        "split_file": str(args.split_file),
        "pert_means_file": str(args.pert_means_file),
        "max_train_per_dataset": args.max_train_per_dataset,
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "leakage_status": "train_only_v2_train_single_residuals_to_internal_proxy_no_canonical_test_no_posthoc_no_heldout_multi",
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "absolute_scores": absolute_scores,
        "paired_deltas": paired_deltas,
        "decision": {"status": status, "action": action, "reasons": reasons},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(args.out_md), "n_train_rows": len(train_rows), "n_val_rows": len(val_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
