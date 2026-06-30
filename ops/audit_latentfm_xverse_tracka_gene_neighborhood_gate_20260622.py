#!/usr/bin/env python3
"""CPU-only Track A gate for gene-neighborhood residual baselines.

This tests whether a stronger nonparametric residual predictor can beat the
train-only gene/dataset baselines before any new GPU training. It uses only the
xverse train/internal-val split and explicitly avoids canonical test, canonical
multi, and Track C query artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
BASE_GATE_SCRIPT = ROOT / "ops/audit_latentfm_xverse_gene_reliability_router_gate_20260622.py"
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_tracka_gene_neighborhood_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_TRACKA_GENE_NEIGHBORHOOD_GATE_20260622.md"
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
K_VALUES = (3, 5, 10, 20)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("gene_reliability_gate", BASE_GATE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {BASE_GATE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else np.zeros_like(arr)


def build_gene_library(rows: list[dict[str, Any]], base: Any) -> dict[str, Any]:
    sums = base.build_sums(rows)
    genes = sorted(sums["by_gene_sum"])
    vectors = []
    counts = []
    for gene in genes:
        count = int(sums["by_gene_count"][gene])
        vectors.append(np.asarray(sums["by_gene_sum"][gene], dtype=np.float32) / float(count))
        counts.append(count)
    matrix = np.stack(vectors).astype(np.float32)
    unit_matrix = np.stack([unit(v) for v in matrix]).astype(np.float32)
    return {
        "sums": sums,
        "genes": genes,
        "gene_to_idx": {gene: i for i, gene in enumerate(genes)},
        "matrix": matrix,
        "unit_matrix": unit_matrix,
        "counts": np.asarray(counts, dtype=np.int32),
    }


def component_for_row(lib: dict[str, Any], row: dict[str, Any], base: Any, *, exclude_row: bool) -> dict[str, Any]:
    dataset_mean, gene_mean, global_mean, gene_count = base.component_means(
        lib["sums"], row, exclude_row=exclude_row
    )
    gene = str(row["gene"])
    current = gene_mean
    return {
        "dataset_mean": dataset_mean,
        "gene_mean": gene_mean,
        "global_mean": global_mean,
        "gene_count": gene_count,
        "gene": gene,
        "current_unit": unit(current),
    }


def neighbor_mean(lib: dict[str, Any], row_state: dict[str, Any], k: int) -> np.ndarray:
    sims = np.asarray(lib["unit_matrix"] @ row_state["current_unit"], dtype=np.float32)
    idx = lib["gene_to_idx"].get(row_state["gene"])
    if idx is not None:
        sims[idx] = -np.inf
    valid = np.flatnonzero(np.isfinite(sims))
    if valid.size == 0:
        return np.asarray(row_state["gene_mean"], dtype=np.float32)
    take = valid[np.argsort(sims[valid])[-min(k, valid.size):]]
    weights = np.maximum(sims[take], 0.0).astype(np.float32)
    if float(weights.sum()) <= 1e-8:
        return np.mean(lib["matrix"][take], axis=0).astype(np.float32)
    return np.average(lib["matrix"][take], axis=0, weights=weights).astype(np.float32)


def predict(model: str, lib: dict[str, Any], row_state: dict[str, Any]) -> np.ndarray:
    if model == "dataset_mean":
        return row_state["dataset_mean"]
    if model == "gene_raw_mean":
        return row_state["gene_mean"]
    if model == "global_mean":
        return row_state["global_mean"]
    if model.startswith("knn_gene_k"):
        k = int(model.removeprefix("knn_gene_k"))
        return neighbor_mean(lib, row_state, k)
    if model.startswith("knn_blend_k"):
        k = int(model.removeprefix("knn_blend_k"))
        neigh = neighbor_mean(lib, row_state, k)
        return (0.5 * row_state["gene_mean"] + 0.5 * neigh).astype(np.float32)
    if model.startswith("knn_dataset_shift_k"):
        k = int(model.removeprefix("knn_dataset_shift_k"))
        neigh = neighbor_mean(lib, row_state, k)
        return (row_state["dataset_mean"] + (neigh - row_state["global_mean"])).astype(np.float32)
    raise ValueError(model)


def model_names() -> list[str]:
    names = ["dataset_mean", "gene_raw_mean", "global_mean"]
    names += [f"knn_gene_k{k}" for k in K_VALUES]
    names += [f"knn_blend_k{k}" for k in K_VALUES]
    names += [f"knn_dataset_shift_k{k}" for k in K_VALUES]
    return names


def evaluate(rows: list[dict[str, Any]], lib: dict[str, Any], base: Any, pert_means: dict[str, np.ndarray], models: list[str], *, exclude_row: bool) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        state = component_for_row(lib, row, base, exclude_row=exclude_row)
        scored = {
            "dataset": row["dataset"],
            "condition": row["condition"],
            "gene": row["gene"],
            "group": row.get("group", "train_leave_one"),
            "gene_train_count": state["gene_count"],
        }
        for model in models:
            scored[model] = base.score(row, predict(model, lib, state), pert_means)
        out.append(scored)
    return out


def group_rows(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("group") == group]


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
    point = float(np.mean([np.mean(diffs_by_ds[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        boot.append(float(np.mean([
            np.mean(rng.choice(diffs_by_ds[str(ds)], size=len(diffs_by_ds[str(ds)]), replace=True))
            for ds in sample_ds
        ])))
    arr = np.asarray(boot, dtype=np.float64)
    by_dataset = {ds: float(np.mean(vals)) for ds, vals in diffs_by_ds.items()}
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "n_conditions": int(sum(len(diffs_by_ds[ds]) for ds in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "by_dataset": by_dataset,
    }


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    selected = payload["selected_model"]
    reasons = []
    deltas = {
        (row["group"], row["baseline"]): row
        for row in payload["paired_deltas"]
        if row.get("candidate") == selected and row.get("status") == "ok"
    }
    for group in GROUPS:
        for baseline in ("gene_raw_mean", "dataset_mean"):
            row = deltas.get((group, baseline), {})
            if row.get("status") != "ok":
                reasons.append(f"{group}_{baseline}_comparison_missing")
                continue
            if float(row.get("delta_mean") or 0.0) < 0.02:
                reasons.append(f"{group}_selected_not_0p02_better_than_{baseline}")
            if float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.20:
                reasons.append(f"{group}_harm_risk_vs_{baseline}")
            if any(float(v) < -0.02 for v in (row.get("by_dataset") or {}).values()):
                reasons.append(f"{group}_dataset_level_material_harm_vs_{baseline}")
    status = "cpu_gate_pass_neighborhood_baseline_candidate" if not reasons else "cpu_gate_fail_close_neighborhood_baseline"
    action = (
        "perform_mechanism_review_before_any_gpu_training"
        if not reasons
        else "do_not_launch_gpu_from_gene_neighborhood_baseline"
    )
    return {"status": status, "action": action, "reasons": reasons}


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Track A Gene-Neighborhood Baseline Gate",
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
        f"- selected model: `{payload['selected_model']}`",
        f"- train rows: `{payload['n_train_rows']}`",
        f"- validation rows: `{payload['n_val_rows']}`",
        "- canonical test, canonical multi, and Track C query are not used.",
        "",
        "## Train Leave-One Scores",
        "",
        "| model | equal-dataset pp |",
        "|---|---:|",
    ]
    for row in payload["train_leave_one_scores"]:
        lines.append(f"| `{row['model']}` | {fmt(row['score'])} |")
    lines += [
        "",
        "## Internal Validation Scores",
        "",
        "| group | model | equal-dataset pp |",
        "|---|---|---:|",
    ]
    for row in payload["absolute_scores"]:
        lines.append(f"| {row['group']} | `{row['model']}` | {fmt(row['score'])} |")
    lines += [
        "",
        "## Paired Deltas",
        "",
        "| group | candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {row['group']} | `{row['candidate']}` | `{row['baseline']}` | "
            f"{row.get('n_conditions', 0)} | {row.get('n_datasets', 0)} | "
            f"{fmt(row.get('delta_mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row.get('p_improve'))} | {fmt(row.get('p_harm'))} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Interpretation",
        "",
        "- Passing this CPU gate would not authorize training by itself.",
        "- Failure closes this nonparametric gene-neighborhood baseline branch.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--max-train-per-dataset", type=int, default=768)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    base = load_base_module()
    data_dir = args.data_dir.resolve()
    split = base.load_json(args.split_file)
    manifest = base.load_json(data_dir / "manifest.json")
    metadata_file, metadata_source = base.resolve_condition_metadata_file(data_dir, manifest)
    metadata = base.load_json(metadata_file)
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}
    train_rows, val_rows = base.collect_rows(
        data_dir,
        split,
        metadata,
        max_train_per_dataset=args.max_train_per_dataset,
        max_cells=args.max_cells_per_condition,
    )
    lib = build_gene_library(train_rows, base)
    models = model_names()
    train_eval = evaluate(train_rows, lib, base, pert_means, models, exclude_row=True)
    nontrivial = [m for m in models if m not in {"dataset_mean", "gene_raw_mean", "global_mean"}]
    train_scores = [{"model": m, "score": base.equal_dataset_mean(train_eval, m)} for m in models]
    selected = max(nontrivial, key=lambda m: float(base.equal_dataset_mean(train_eval, m) or -1e9))
    eval_models = ["dataset_mean", "gene_raw_mean", "global_mean", selected]
    val_eval = evaluate(val_rows, lib, base, pert_means, eval_models, exclude_row=False)
    absolute = []
    paired = []
    for group in GROUPS:
        rows = group_rows(val_eval, group)
        for model in eval_models:
            absolute.append({"group": group, "model": model, "score": base.equal_dataset_mean(rows, model)})
        for baseline in ("gene_raw_mean", "dataset_mean", "global_mean"):
            paired.append(
                {
                    "group": group,
                    **paired_bootstrap(rows, selected, baseline, n_boot=args.n_boot, seed=args.seed + len(paired)),
                }
            )
    payload = {
        "data_dir": str(data_dir),
        "split_file": str(args.split_file),
        "condition_metadata_file": str(metadata_file),
        "condition_metadata_source": metadata_source,
        "pert_means_file": str(args.pert_means_file),
        "max_train_per_dataset": args.max_train_per_dataset,
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "leakage_status": "train_only_v2_gene_neighborhood_no_canonical_no_query",
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "selected_model": selected,
        "train_leave_one_scores": train_scores,
        "absolute_scores": absolute,
        "paired_deltas": paired,
        "val_condition_rows": val_eval,
    }
    payload["decision"] = decide(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "selected_model": selected, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
