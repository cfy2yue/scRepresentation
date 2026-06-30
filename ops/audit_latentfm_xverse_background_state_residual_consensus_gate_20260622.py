#!/usr/bin/env python3
"""CPU gate for deployable background-state residual consensus.

This tests whether control/source latent state plus perturbation gene embeddings
contain train-only Track A pp-direction signal beyond dataset and gene-mean
controls. It uses only the xverse train-only v2 split and internal proxy groups;
it does not read canonical test outcomes, posthoc predictions, or held-out multi
GT.
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
DEFAULT_GENE_CACHE = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_background_state_residual_consensus_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_BACKGROUND_STATE_RESIDUAL_CONSENSUS_GATE_20260622.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
MODELS = (
    "background_gene_interact_ridge",
    "background_gene_ridge",
    "background_only_ridge",
    "gene_only_ridge",
    "background_gene_shuffled_ridge",
    "dataset_mean",
    "gene_raw_mean",
    "global_mean",
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
    genes = [str(g).strip() for g in meta.get("genes") or [] if str(g).strip()]
    if len(genes) != 1:
        return None
    raw = str(meta.get("perturbation_type_raw") or "").lower()
    if "drug" in raw or "compound" in raw or "chemical" in raw:
        return None
    return genes[0]


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


def gene_vector(gene: str, mapping: dict[str, int], emb: np.ndarray, unk_index: int) -> np.ndarray:
    idx = int(mapping.get(normalize_gene(gene), unk_index))
    return np.asarray(emb[idx], dtype=np.float32)


def collect_rows(
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    gene_mapping: dict[str, int],
    gene_emb: np.ndarray,
    unk_index: int,
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
        chosen_train = set(stable_subset([c for c, _ in train_single], max_train_per_dataset, f"bgstate|train|{ds}"))
        with h5py.File(path, "r") as handle:
            by_cond = {c: i for i, c in enumerate(decode(np.asarray(handle["conditions"])))}
            for cond, gene in train_single:
                if cond not in chosen_train:
                    continue
                vals = residual_for_condition(handle, by_cond, cond, max_cells)
                if vals is None:
                    continue
                ctrl, gt, residual = vals
                train_rows.append(
                    {
                        "dataset": ds,
                        "condition": cond,
                        "gene": gene,
                        "gene_emb": gene_vector(gene, gene_mapping, gene_emb, unk_index),
                        "ctrl": ctrl,
                        "gt": gt,
                        "residual": residual,
                    }
                )
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
                            "gene_emb": gene_vector(gene, gene_mapping, gene_emb, unk_index),
                            "group": group,
                            "ctrl": ctrl,
                            "gt": gt,
                            "residual": residual,
                        }
                    )
    return train_rows, val_rows


def fit_pca(x: np.ndarray, k: int) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    if k <= 0:
        return {"mean": mean.astype(np.float32), "components": np.zeros((0, x.shape[1]), dtype=np.float32)}
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    n_comp = min(int(k), vt.shape[0])
    return {"mean": mean.astype(np.float32), "components": vt[:n_comp].astype(np.float32)}


def transform_pca(x: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    comps = np.asarray(pca["components"], dtype=np.float32)
    if comps.size == 0:
        return np.zeros((len(x), 0), dtype=np.float32)
    return ((np.asarray(x, dtype=np.float32) - pca["mean"]) @ comps.T).astype(np.float32)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def features(bg: np.ndarray, gene: np.ndarray, mode: str, interaction_dim: int) -> np.ndarray:
    if mode == "background_only":
        return bg
    if mode == "gene_only":
        return gene
    if mode == "background_gene":
        return np.concatenate([bg, gene], axis=1)
    if mode == "background_gene_interact":
        n = min(int(interaction_dim), bg.shape[1], gene.shape[1])
        inter = bg[:, :n] * gene[:, :n]
        return np.concatenate([bg, gene, inter], axis=1)
    raise ValueError(mode)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    xs, mean, std = standardize_fit(np.asarray(x, dtype=np.float32))
    x_aug = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float32), xs], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    xtx = x_aug.T @ x_aug
    xty = x_aug.T @ np.asarray(y, dtype=np.float64)
    coef = np.linalg.solve(xtx + reg, xty)
    return {"coef": coef.astype(np.float32), "mean": mean, "std": std}


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = standardize_apply(np.asarray(x, dtype=np.float32), model["mean"], model["std"])
    x_aug = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float32), xs], axis=1)
    return (x_aug @ model["coef"]).astype(np.float32)


def build_baselines(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[np.ndarray]] = defaultdict(list)
    by_gene: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in train_rows:
        by_ds[str(row["dataset"])].append(np.asarray(row["residual"], dtype=np.float32))
        by_gene[str(row["gene"])].append(np.asarray(row["residual"], dtype=np.float32))
    return {
        "dataset_mean": {k: np.mean(np.vstack(v), axis=0).astype(np.float32) for k, v in by_ds.items()},
        "gene_raw_mean": {k: np.mean(np.vstack(v), axis=0).astype(np.float32) for k, v in by_gene.items()},
        "global_mean": np.mean(np.vstack([r["residual"] for r in train_rows]), axis=0).astype(np.float32),
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


def evaluate(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    pert_means: dict[str, np.ndarray],
    *,
    bg_pcs: int,
    gene_pcs: int,
    interaction_dim: int,
    ridge_alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    train_ctrl = np.vstack([r["ctrl"] for r in train_rows]).astype(np.float32)
    val_ctrl = np.vstack([r["ctrl"] for r in val_rows]).astype(np.float32)
    train_gene = np.vstack([r["gene_emb"] for r in train_rows]).astype(np.float32)
    val_gene = np.vstack([r["gene_emb"] for r in val_rows]).astype(np.float32)
    train_y = np.vstack([r["residual"] for r in train_rows]).astype(np.float32)

    bg_pca = fit_pca(train_ctrl, bg_pcs)
    gene_pca = fit_pca(train_gene, gene_pcs)
    train_bg = transform_pca(train_ctrl, bg_pca)
    val_bg = transform_pca(val_ctrl, bg_pca)
    train_gene_p = transform_pca(train_gene, gene_pca)
    val_gene_p = transform_pca(val_gene, gene_pca)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_gene_p.shape[0])
    train_gene_shuf = train_gene_p[perm]
    val_genes = sorted({str(r["gene"]) for r in val_rows})
    shuf_gene_map = {g: val_genes[i] for i, g in enumerate(rng.permutation(val_genes))}
    val_gene_by_name = {str(r["gene"]): val_gene_p[i] for i, r in enumerate(val_rows)}
    val_gene_shuf = np.vstack([val_gene_by_name.get(shuf_gene_map[str(r["gene"])], val_gene_p[i]) for i, r in enumerate(val_rows)]).astype(np.float32)

    fit_specs = {
        "background_gene_interact_ridge": ("background_gene_interact", train_bg, train_gene_p, val_bg, val_gene_p),
        "background_gene_ridge": ("background_gene", train_bg, train_gene_p, val_bg, val_gene_p),
        "background_only_ridge": ("background_only", train_bg, train_gene_p, val_bg, val_gene_p),
        "gene_only_ridge": ("gene_only", train_bg, train_gene_p, val_bg, val_gene_p),
        "background_gene_shuffled_ridge": ("background_gene_interact", train_bg, train_gene_shuf, val_bg, val_gene_shuf),
    }
    pred_by_model: dict[str, np.ndarray] = {}
    for name, (mode, tr_bg, tr_gene, va_bg, va_gene) in fit_specs.items():
        x_train = features(tr_bg, tr_gene, mode, interaction_dim)
        x_val = features(va_bg, va_gene, mode, interaction_dim)
        model = fit_ridge(x_train, train_y, ridge_alpha)
        pred_by_model[name] = predict_ridge(model, x_val)

    baselines = build_baselines(train_rows)
    out = []
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
    better = 0
    total = 0
    for row in rows:
        a = row.get(candidate)
        b = row.get(baseline)
        if a is not None and b is not None:
            diff = float(a) - float(b)
            diffs_by_ds[str(row["dataset"])].append(diff)
            better += int(diff > 0.0)
            total += 1
    datasets = sorted(ds for ds, vals in diffs_by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline}
    ds_means = [float(np.mean(diffs_by_ds[ds])) for ds in datasets]
    point = float(np.mean(ds_means))
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
        "candidate": candidate,
        "baseline": baseline,
        "n_conditions": int(sum(len(diffs_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "better_fraction": None if total == 0 else float(better / total),
        "median_dataset_delta": float(np.median(ds_means)),
        "leave_one_min": min(leave.values()) if leave else None,
    }


def decide(paired: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = "background_gene_interact_ridge"
    by = {(r["group"], r["baseline"]): r for r in paired if r.get("candidate") == candidate}
    reasons = []
    cross_ds = by.get((GROUPS[0], "dataset_mean")) or {}
    cross_gene = by.get((GROUPS[0], "gene_raw_mean")) or {}
    cross_shuf = by.get((GROUPS[0], "background_gene_shuffled_ridge")) or {}
    fam_ds = by.get((GROUPS[1], "dataset_mean")) or {}
    fam_gene = by.get((GROUPS[1], "gene_raw_mean")) or {}
    fam_shuf = by.get((GROUPS[1], "background_gene_shuffled_ridge")) or {}

    def supported(row: dict[str, Any], min_delta: float = 0.02) -> bool:
        if row.get("status") != "ok":
            return False
        ci = row.get("ci95") or [0.0, 0.0]
        return float(row.get("delta_mean") or 0.0) >= min_delta or float(ci[0]) > 0.0

    if not supported(cross_ds):
        reasons.append("cross_background_not_materially_better_than_dataset_mean")
    if not supported(cross_gene):
        reasons.append("cross_background_not_materially_better_than_gene_raw_mean")
    if not supported(cross_shuf, min_delta=0.0):
        reasons.append("cross_background_not_better_than_shuffled_gene_control")
    if float(cross_ds.get("better_fraction") or 0.0) < 0.65:
        reasons.append("cross_background_condition_better_fraction_lt_065")
    if float(cross_ds.get("median_dataset_delta") or 0.0) <= 0.0:
        reasons.append("cross_background_median_dataset_delta_nonpositive")
    if cross_ds.get("leave_one_min") is None or float(cross_ds["leave_one_min"]) <= 0.0:
        reasons.append("cross_background_leave_one_dataset_flips_or_nonpositive")
    for name, row in (("family_vs_dataset_mean", fam_ds), ("family_vs_gene_raw_mean", fam_gene), ("family_vs_shuffled_gene", fam_shuf)):
        if row.get("status") != "ok":
            reasons.append(f"{name}_missing")
        elif float(row.get("p_harm") if row.get("p_harm") is not None else 1.0) > 0.20:
            reasons.append(f"{name}_harm_risk")
    status = "cpu_gate_pass_launch_one_background_film_adapter_smoke" if not reasons else "cpu_gate_fail_do_not_launch_gpu"
    return {
        "status": status,
        "action": "launch_one_background_film_adapter_smoke" if not reasons else "keep_background_state_consensus_cpu_only",
        "candidate": candidate,
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
        "# LatentFM xverse Background-State Residual Consensus Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
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
        f"- bg_pcs/gene_pcs/interaction_dim/ridge_alpha: `{payload['bg_pcs']}/{payload['gene_pcs']}/{payload['interaction_dim']}/{payload['ridge_alpha']}`",
        "",
        "## Absolute Scores",
        "",
        "| group | model | equal-dataset pp proxy |",
        "|---|---|---:|",
    ]
    for row in payload["absolute_scores"]:
        lines.append(f"| {row['group']} | `{row['model']}` | {fmt(row['mean'])} |")
    lines += [
        "",
        "## Paired Deltas",
        "",
        "| group | candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | better frac | median ds delta | leave-one min | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| {row['group']} | {row['candidate']} | {row['baseline']} | "
            f"{row.get('n_conditions', 0)} | {row.get('n_datasets', 0)} | "
            f"{fmt(row.get('delta_mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row.get('p_improve'))} | {fmt(row.get('p_harm'))} | "
            f"{fmt(row.get('better_fraction'))} | {fmt(row.get('median_dataset_delta'))} | "
            f"{fmt(row.get('leave_one_min'))} | {row.get('status')} |"
        )
    lines += ["", "## Gate Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Interpretation",
        "",
        "- Passing would justify one tiny background/FiLM-style adapter smoke only.",
        "- Failing means this deployable background-state residual consensus is diagnostic only.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    parser.add_argument("--gene-cache-dir", type=Path, default=DEFAULT_GENE_CACHE)
    parser.add_argument("--max-train-per-dataset", type=int, default=0)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--bg-pcs", type=int, default=16)
    parser.add_argument("--gene-pcs", type=int, default=16)
    parser.add_argument("--interaction-dim", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

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
    eval_rows = evaluate(
        train_rows,
        val_rows,
        pert_means,
        bg_pcs=args.bg_pcs,
        gene_pcs=args.gene_pcs,
        interaction_dim=args.interaction_dim,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
    )
    absolute_scores = []
    paired_deltas = []
    for group in GROUPS:
        rows = [r for r in eval_rows if r["group"] == group]
        for model in MODELS:
            absolute_scores.append({"group": group, "model": model, "mean": equal_dataset_mean(rows, model)})
        for candidate in ("background_gene_interact_ridge", "background_gene_ridge"):
            for baseline in ("dataset_mean", "gene_raw_mean", "global_mean", "background_only_ridge", "gene_only_ridge", "background_gene_shuffled_ridge"):
                row = paired_bootstrap(
                    rows,
                    candidate,
                    baseline,
                    n_boot=args.n_boot,
                    seed=args.seed + len(paired_deltas),
                )
                row.update({"group": group, "candidate": candidate, "baseline": baseline})
                paired_deltas.append(row)

    payload = {
        "data_dir": str(data_dir),
        "split_file": str(args.split_file),
        "pert_means_file": str(args.pert_means_file),
        "gene_cache_dir": str(args.gene_cache_dir),
        "max_train_per_dataset": args.max_train_per_dataset,
        "max_cells_per_condition": args.max_cells_per_condition,
        "bg_pcs": args.bg_pcs,
        "gene_pcs": args.gene_pcs,
        "interaction_dim": args.interaction_dim,
        "ridge_alpha": args.ridge_alpha,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "leakage_status": "train_only_v2_train_single_residuals_to_internal_proxy_no_canonical_test_no_posthoc_no_heldout_multi",
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "absolute_scores": absolute_scores,
        "paired_deltas": paired_deltas,
        "decision": decide(paired_deltas),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md), "n_train_rows": len(train_rows), "n_val_rows": len(val_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
