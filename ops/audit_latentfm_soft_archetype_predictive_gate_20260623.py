#!/usr/bin/env python3
"""CPU gate for soft residualized state archetypes.

This is a no-leakage Track A internal-proxy gate.  It fits soft state
prototypes from control/source latent embeddings only, then asks whether those
state memberships plus gene embeddings predict train-only internal validation
response residuals better than gene and dataset baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
sys.path.insert(0, str(OPS))

from audit_latentfm_xverse_background_state_residual_consensus_gate_20260622 import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_GENE_CACHE,
    DEFAULT_PERT_MEANS,
    DEFAULT_SPLIT,
    GROUPS,
    build_baselines,
    collect_rows,
    equal_dataset_mean,
    fit_pca,
    fit_ridge,
    features,
    load_gene_embeddings,
    load_json,
    paired_bootstrap,
    predict_ridge,
    score,
    transform_pca,
)


DEFAULT_OUT_JSON = ROOT / "reports/latentfm_soft_archetype_predictive_gate_20260623.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_PREDICTIVE_GATE_20260623.md"
MODELS = (
    "soft_archetype_gene_interact_ridge",
    "soft_archetype_gene_ridge",
    "soft_archetype_only_ridge",
    "gene_only_ridge",
    "soft_archetype_gene_shuffled_ridge",
    "dataset_mean",
    "gene_raw_mean",
    "global_mean",
)
FOCUS_HINTS = ("wessels", "norman", "jiang", "gasperini")


def dataset_stats(rows: list[dict[str, Any]], pca: dict[str, np.ndarray]) -> dict[str, Any]:
    by_ds: dict[str, list[np.ndarray]] = defaultdict(list)
    all_ctrl = []
    for row in rows:
        ctrl_pc = transform_pca(np.asarray(row["ctrl"], dtype=np.float32)[None, :], pca)[0]
        by_ds[str(row["dataset"])].append(ctrl_pc)
        all_ctrl.append(ctrl_pc)
    global_arr = np.vstack(all_ctrl).astype(np.float32)
    stats = {
        "global_mean": global_arr.mean(axis=0).astype(np.float32),
        "global_std": np.maximum(global_arr.std(axis=0), 1e-6).astype(np.float32),
        "by_dataset": {},
    }
    for ds, vals in by_ds.items():
        arr = np.vstack(vals).astype(np.float32)
        stats["by_dataset"][ds] = {
            "mean": arr.mean(axis=0).astype(np.float32),
            "std": np.maximum(arr.std(axis=0), 1e-6).astype(np.float32),
        }
    return stats


def residualized_ctrl(rows: list[dict[str, Any]], pca: dict[str, np.ndarray], stats: dict[str, Any]) -> np.ndarray:
    ctrl = np.vstack([r["ctrl"] for r in rows]).astype(np.float32)
    pcs = transform_pca(ctrl, pca)
    out = []
    for row, vec in zip(rows, pcs):
        ds_stats = stats["by_dataset"].get(str(row["dataset"]))
        mean = ds_stats["mean"] if ds_stats is not None else stats["global_mean"]
        std = ds_stats["std"] if ds_stats is not None else stats["global_std"]
        out.append((vec - mean) / std)
    return np.vstack(out).astype(np.float32)


def soft_assign(x: np.ndarray, centroids: np.ndarray, temperature: float | None = None) -> np.ndarray:
    d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    if temperature is None:
        temperature = float(np.median(d2))
    temperature = max(float(temperature), 1e-6)
    logits = -d2 / temperature
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights.astype(np.float32)


def normalized_entropy(weights: np.ndarray) -> np.ndarray:
    k = max(int(weights.shape[1]), 2)
    ent = -(weights * np.log(np.maximum(weights, 1e-12))).sum(axis=1) / math.log(k)
    return ent.astype(np.float32)


def simple_kmeans(x: np.ndarray, k: int, seed: int, max_iter: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n < k:
        raise ValueError(f"not enough rows for k={k}: n={n}")
    centroids = x[rng.choice(n, size=k, replace=False)].astype(np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = x[mask].mean(axis=0)
            else:
                centroids[c] = x[int(rng.integers(0, n))]
    return centroids.astype(np.float32)


def normalized_mutual_info(labels: list[str], clusters: np.ndarray) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    label_ids = {label: i for i, label in enumerate(sorted(set(labels)))}
    cluster_ids = {int(c): i for i, c in enumerate(sorted(set(int(x) for x in clusters)))}
    table = np.zeros((len(label_ids), len(cluster_ids)), dtype=np.float64)
    for label, cluster in zip(labels, clusters):
        table[label_ids[str(label)], cluster_ids[int(cluster)]] += 1.0
    pxy = table / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    denom = math.sqrt(max(hx * hy, 1e-12))
    return float(mi / denom)


def purity(labels: list[str], clusters: np.ndarray) -> float:
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for label, cluster in zip(labels, clusters):
        by_cluster[int(cluster)].append(str(label))
    total = len(labels)
    if total == 0:
        return 0.0
    return float(sum(max(v.count(x) for x in set(v)) for v in by_cluster.values()) / total)


def soft_kernel_stability(assignments: list[np.ndarray], max_items: int, seed: int) -> dict[str, Any]:
    n = assignments[0].shape[0]
    if n > max_items:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_items, replace=False))
    else:
        idx = np.arange(n)
    kernels = []
    tri = np.triu_indices(len(idx), k=1)
    for weights in assignments:
        k = weights[idx] @ weights[idx].T
        vec = k[tri]
        vec = vec - vec.mean()
        denom = float(np.linalg.norm(vec))
        kernels.append(vec / denom if denom > 1e-12 else vec)
    vals = []
    for i in range(len(kernels)):
        for j in range(i + 1, len(kernels)):
            vals.append(float(np.dot(kernels[i], kernels[j])))
    return {
        "pairwise_soft_kernel_cosine_mean": float(np.mean(vals)) if vals else None,
        "pairwise_soft_kernel_cosine_min": float(np.min(vals)) if vals else None,
        "n_items": int(len(idx)),
    }


def focus_coverage(rows: list[dict[str, Any]], weights: np.ndarray) -> dict[str, Any]:
    out = {}
    datasets = sorted({str(r["dataset"]) for r in rows})
    for hint in FOCUS_HINTS:
        matched = [ds for ds in datasets if hint in ds.lower()]
        if not matched:
            continue
        mask = np.asarray([str(r["dataset"]) in matched for r in rows], dtype=bool)
        mean_w = weights[mask].mean(axis=0)
        out[hint] = {
            "datasets": matched,
            "n_rows": int(mask.sum()),
            "entropy": float(normalized_entropy(mean_w[None, :])[0]),
            "max_soft_mass": float(mean_w.max()),
        }
    return out


def fit_one_spec(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    pert_means: dict[str, np.ndarray],
    *,
    k: int,
    bg_pcs: int,
    gene_pcs: int,
    interaction_dim: int,
    ridge_alpha: float,
    kmeans_seeds: list[int],
    max_stability_items: int,
    seed: int,
) -> dict[str, Any]:
    train_ctrl = np.vstack([r["ctrl"] for r in train_rows]).astype(np.float32)
    train_gene = np.vstack([r["gene_emb"] for r in train_rows]).astype(np.float32)
    val_gene = np.vstack([r["gene_emb"] for r in val_rows]).astype(np.float32)
    train_y = np.vstack([r["residual"] for r in train_rows]).astype(np.float32)

    bg_pca = fit_pca(train_ctrl, bg_pcs)
    gene_pca = fit_pca(train_gene, gene_pcs)
    stats = dataset_stats(train_rows, bg_pca)
    train_state = residualized_ctrl(train_rows, bg_pca, stats)
    val_state = residualized_ctrl(val_rows, bg_pca, stats)
    train_gene_p = transform_pca(train_gene, gene_pca)
    val_gene_p = transform_pca(val_gene, gene_pca)

    assignments = []
    centroids_by_seed = []
    for kseed in kmeans_seeds:
        centroids = simple_kmeans(train_state, k=k, seed=int(kseed), max_iter=120)
        train_w = soft_assign(train_state, centroids)
        assignments.append(train_w)
        centroids_by_seed.append(centroids)

    centroids0 = centroids_by_seed[0]
    train_w = assignments[0]
    val_w = soft_assign(val_state, centroids0)
    stability = soft_kernel_stability(assignments, max_stability_items, seed)

    hard = np.argmax(train_w, axis=1)
    ds_labels = [str(r["dataset"]) for r in train_rows]
    proxy = {
        "dataset_nmi": normalized_mutual_info(ds_labels, hard),
        "dataset_ami": None,
        "dataset_purity": purity(ds_labels, hard),
        "train_entropy_mean": float(np.mean(normalized_entropy(train_w))),
        "train_max_soft_mass_mean": float(np.mean(train_w.max(axis=1))),
        "focus_coverage": focus_coverage(train_rows + val_rows, np.vstack([train_w, val_w])),
    }

    rng = np.random.default_rng(seed + k)
    perm = rng.permutation(train_w.shape[0])
    train_w_shuf = train_w[perm]
    val_w_shuf = val_w[rng.permutation(val_w.shape[0])]

    fit_specs = {
        "soft_archetype_gene_interact_ridge": ("background_gene_interact", train_w, train_gene_p, val_w, val_gene_p),
        "soft_archetype_gene_ridge": ("background_gene", train_w, train_gene_p, val_w, val_gene_p),
        "soft_archetype_only_ridge": ("background_only", train_w, train_gene_p, val_w, val_gene_p),
        "gene_only_ridge": ("gene_only", train_w, train_gene_p, val_w, val_gene_p),
        "soft_archetype_gene_shuffled_ridge": ("background_gene_interact", train_w_shuf, train_gene_p, val_w_shuf, val_gene_p),
    }
    pred_by_model: dict[str, np.ndarray] = {}
    for name, (mode, tr_state, tr_gene, va_state, va_gene) in fit_specs.items():
        x_train = features(tr_state, tr_gene, mode, interaction_dim)
        x_val = features(va_state, va_gene, mode, interaction_dim)
        model = fit_ridge(x_train, train_y, ridge_alpha)
        pred_by_model[name] = predict_ridge(model, x_val)

    baselines = build_baselines(train_rows)
    eval_rows = []
    for i, row in enumerate(val_rows):
        ds = str(row["dataset"])
        gene = str(row["gene"])
        scored = {
            "dataset": ds,
            "condition": row["condition"],
            "gene": gene,
            "group": row["group"],
        }
        baseline_preds = {
            "dataset_mean": baselines["dataset_mean"].get(ds, baselines["global_mean"]),
            "gene_raw_mean": baselines["gene_raw_mean"].get(gene, baselines["global_mean"]),
            "global_mean": baselines["global_mean"],
        }
        for name, preds in pred_by_model.items():
            scored[name] = score(row, preds[i], pert_means)
        for name, pred in baseline_preds.items():
            scored[name] = score(row, pred, pert_means)
        eval_rows.append(scored)
    return {
        "k": int(k),
        "stability": stability,
        "dataset_proxy": proxy,
        "eval_rows": eval_rows,
    }


def summarize_spec(spec: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    rows = spec["eval_rows"]
    absolute_scores = []
    paired = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        for model in MODELS:
            absolute_scores.append({"group": group, "model": model, "mean": equal_dataset_mean(group_rows, model)})
        for baseline in ("dataset_mean", "gene_raw_mean", "gene_only_ridge", "soft_archetype_gene_shuffled_ridge"):
            delta = paired_bootstrap(
                group_rows,
                "soft_archetype_gene_interact_ridge",
                baseline,
                n_boot=n_boot,
                seed=seed + int(spec["k"]) + len(paired),
            )
            delta["group"] = group
            paired.append(delta)
    return {
        "k": int(spec["k"]),
        "stability": spec["stability"],
        "dataset_proxy": spec["dataset_proxy"],
        "absolute_scores": absolute_scores,
        "paired_deltas": paired,
    }


def decide(specs: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_by_k = {int(s["k"]): s for s in specs}
    ranked = sorted(
        specs,
        key=lambda s: (
            float(s["stability"].get("pairwise_soft_kernel_cosine_mean") or -999.0),
            -float(s["dataset_proxy"].get("dataset_nmi") or 999.0),
        ),
        reverse=True,
    )
    chosen = ranked[0] if ranked else {}
    reasons = []
    if not chosen:
        return {"status": "soft_archetype_cpu_gate_fail", "gpu_authorization": "none", "reasons": ["no_specs"], "chosen_k": None}
    stability_mean = float(chosen["stability"].get("pairwise_soft_kernel_cosine_mean") or 0.0)
    stability_min = float(chosen["stability"].get("pairwise_soft_kernel_cosine_min") or 0.0)
    proxy = chosen["dataset_proxy"]
    if stability_mean < 0.70 or stability_min < 0.60:
        reasons.append("soft_assignment_stability_below_gate")
    if float(proxy.get("dataset_nmi") or 1.0) > 0.45 or float(proxy.get("dataset_purity") or 1.0) > 0.65:
        reasons.append("soft_archetype_still_dataset_proxy_like")
    focus = proxy.get("focus_coverage") or {}
    for name, row in focus.items():
        if float(row.get("entropy") or 0.0) < 0.50 or float(row.get("max_soft_mass") or 1.0) > 0.45:
            reasons.append(f"focus_coverage_gate_fail_{name}")
    paired = {(r["group"], r["baseline"]): r for r in chosen.get("paired_deltas") or []}

    def supported(row: dict[str, Any], min_delta: float = 0.02) -> bool:
        if row.get("status") != "ok":
            return False
        ci = row.get("ci95") or [0.0, 0.0]
        return float(row.get("delta_mean") or 0.0) >= min_delta or float(ci[0]) > 0.0

    for group in GROUPS:
        for baseline in ("dataset_mean", "gene_raw_mean"):
            row = paired.get((group, baseline)) or {}
            if not supported(row):
                reasons.append(f"{group}_not_better_than_{baseline}")
        shuf = paired.get((group, "soft_archetype_gene_shuffled_ridge")) or {}
        if not supported(shuf, min_delta=0.0):
            reasons.append(f"{group}_not_better_than_shuffled_state")
        for baseline in ("dataset_mean", "gene_raw_mean", "soft_archetype_gene_shuffled_ridge"):
            row = paired.get((group, baseline)) or {}
            if float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.20:
                reasons.append(f"{group}_{baseline}_harm_risk")
            if row.get("leave_one_min") is None or float(row["leave_one_min"]) < -0.02:
                reasons.append(f"{group}_{baseline}_leave_one_dataset_below_minus_002")
    status = "soft_archetype_cpu_gate_pass_authorize_one_capped_smoke" if not reasons else "soft_archetype_cpu_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "one_capped_smoke" if not reasons else "none",
        "chosen_k": int(chosen["k"]),
        "reasons": reasons,
        "candidate_specs": sorted(candidate_by_k),
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
        "# LatentFM Soft Archetype Predictive Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"GPU authorization: `{payload['decision']['gpu_authorization']}`",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- gene_cache_dir: `{payload['gene_cache_dir']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- train rows: `{payload['n_train_rows']}`",
        f"- validation rows: `{payload['n_val_rows']}`",
        f"- k values: `{payload['k_values']}`",
        f"- bg_pcs/gene_pcs/interaction_dim/ridge_alpha: `{payload['bg_pcs']}/{payload['gene_pcs']}/{payload['interaction_dim']}/{payload['ridge_alpha']}`",
        "",
        "## Spec Diagnostics",
        "",
        "| K | stability mean | stability min | dataset NMI | dataset AMI | purity | entropy mean | max mass mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in payload["spec_summaries"]:
        proxy = spec["dataset_proxy"]
        stab = spec["stability"]
        lines.append(
            f"| {spec['k']} | {fmt(stab.get('pairwise_soft_kernel_cosine_mean'))} | "
            f"{fmt(stab.get('pairwise_soft_kernel_cosine_min'))} | "
            f"{fmt(proxy.get('dataset_nmi'))} | {fmt(proxy.get('dataset_ami'))} | "
            f"{fmt(proxy.get('dataset_purity'))} | {fmt(proxy.get('train_entropy_mean'))} | "
            f"{fmt(proxy.get('train_max_soft_mass_mean'))} |"
        )
    lines += [
        "",
        "## Chosen Spec Paired Deltas",
        "",
        "| group | candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | better frac | median ds delta | leave-one min | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    chosen_k = payload["decision"].get("chosen_k")
    chosen = next((s for s in payload["spec_summaries"] if s["k"] == chosen_k), None)
    if chosen:
        for row in chosen["paired_deltas"]:
            ci = row.get("ci95") or [None, None]
            lines.append(
                f"| {row['group']} | {row['candidate']} | {row['baseline']} | "
                f"{row.get('n_conditions', 0)} | {row.get('n_datasets', 0)} | "
                f"{fmt(row.get('delta_mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
                f"{fmt(row.get('p_improve'))} | {fmt(row.get('p_harm'))} | "
                f"{fmt(row.get('better_fraction'))} | {fmt(row.get('median_dataset_delta'))} | "
                f"{fmt(row.get('leave_one_min'))} | {row.get('status')} |"
            )
        lines += ["", "## Chosen Focus Coverage", ""]
        focus = chosen["dataset_proxy"].get("focus_coverage") or {}
        if focus:
            lines += ["| focus | datasets | n rows | entropy | max soft mass |", "|---|---|---:|---:|---:|"]
            for name, row in focus.items():
                lines.append(
                    f"| {name} | `{', '.join(row.get('datasets') or [])}` | {row.get('n_rows')} | "
                    f"{fmt(row.get('entropy'))} | {fmt(row.get('max_soft_mass'))} |"
                )
        else:
            lines.append("- no focus datasets matched by name")
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Interpretation",
        "",
        "- This gate uses only train-only/internal proxy data and control/source state features.",
        "- Passing would authorize only one capped archetype-conditioned smoke.",
        "- Failing keeps archetype as CPU-only negative evidence.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--gene-cache-dir", type=Path, default=DEFAULT_GENE_CACHE)
    parser.add_argument("--max-train-per-dataset", type=int, default=160)
    parser.add_argument("--max-cells-per-condition", type=int, default=128)
    parser.add_argument("--k-values", type=str, default="8,12,16")
    parser.add_argument("--kmeans-seeds", type=str, default="42,43,44")
    parser.add_argument("--bg-pcs", type=int, default=24)
    parser.add_argument("--gene-pcs", type=int, default=16)
    parser.add_argument("--interaction-dim", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--max-stability-items", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    kmeans_seeds = [int(x) for x in args.kmeans_seeds.split(",") if x.strip()]
    data_dir = args.data_dir.resolve()
    split = load_json(args.split_file)
    manifest = load_json(data_dir / "manifest.json")
    metadata = load_json(Path(manifest["condition_metadata_file"]))
    gene_mapping, gene_emb, unk_index = load_gene_embeddings(args.gene_cache_dir)
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}
    train_rows, val_rows = collect_rows(
        data_dir,
        split,
        metadata,
        gene_mapping,
        gene_emb,
        unk_index,
        max_train_per_dataset=args.max_train_per_dataset,
        max_cells=args.max_cells_per_condition,
    )
    specs = []
    for k in k_values:
        raw = fit_one_spec(
            train_rows,
            val_rows,
            pert_means,
            k=k,
            bg_pcs=args.bg_pcs,
            gene_pcs=args.gene_pcs,
            interaction_dim=args.interaction_dim,
            ridge_alpha=args.ridge_alpha,
            kmeans_seeds=kmeans_seeds,
            max_stability_items=args.max_stability_items,
            seed=args.seed,
        )
        specs.append(summarize_spec(raw, args.n_boot, args.seed))
    payload = {
        "data_dir": str(data_dir),
        "split_file": str(args.split_file),
        "pert_means_file": str(args.pert_means_file),
        "gene_cache_dir": str(args.gene_cache_dir),
        "leakage_status": "trainonly_internal_proxy_no_canonical_no_query_no_gt_for_archetype_fit",
        "n_train_rows": int(len(train_rows)),
        "n_val_rows": int(len(val_rows)),
        "k_values": k_values,
        "kmeans_seeds": kmeans_seeds,
        "bg_pcs": int(args.bg_pcs),
        "gene_pcs": int(args.gene_pcs),
        "interaction_dim": int(args.interaction_dim),
        "ridge_alpha": float(args.ridge_alpha),
        "max_train_per_dataset": int(args.max_train_per_dataset),
        "max_cells_per_condition": int(args.max_cells_per_condition),
        "spec_summaries": specs,
    }
    payload["decision"] = decide(specs)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
