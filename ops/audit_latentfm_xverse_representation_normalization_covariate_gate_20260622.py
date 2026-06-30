#!/usr/bin/env python3
"""CPU gate for xverse response normalization / covariate hypotheses.

This audit is train-only/internal-val. It does not train LatentFM and does not
use canonical test metrics for model selection. The question is whether a
deployable residual predictor built from train-only single perturbations and
gene embeddings benefits from response-space normalization when scored back in
the raw endpoint pp frame on train-only proxy groups.
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
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model.latent.response_normalizer import ResponseNormalizer


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
DEFAULT_PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DEFAULT_NORMALIZER = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    "xverse_trainonly_crossbgval_v2_dataset_scale_pca32.npz"
)
DEFAULT_GENE_CACHE = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_representation_normalization_covariate_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_REPRESENTATION_NORMALIZATION_COVARIATE_GATE_20260622.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
MODES = ("raw_ridge", "dataset_scale_ridge", "pca_subspace_ridge", "dataset_scale_pca_ridge")


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    av = a - float(np.mean(a))
    bv = b - float(np.mean(b))
    den = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(av, bv) / den)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def sample_mean(arr: h5py.Dataset, start: int, end: int, *, max_cells: int, key: str) -> np.ndarray:
    n = int(end - start)
    if n <= 0:
        raise ValueError("empty slice")
    if max_cells > 0 and n > max_cells:
        rng = np.random.default_rng(stable_int(key))
        rel = np.sort(rng.choice(n, size=int(max_cells), replace=False))
        block = arr[start + rel]
    else:
        block = arr[start:end]
    return np.asarray(block, dtype=np.float32).mean(axis=0)


def load_gene_embeddings(cache_dir: Path) -> tuple[dict[str, int], np.ndarray]:
    index: dict[str, int] = {}
    with (cache_dir / "gene_index.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene = parts[0].strip()
            if gene in {"", "PAD", "UNK", "gene_symbol"}:
                continue
            try:
                index[gene] = int(parts[1])
            except ValueError:
                continue
    emb = np.load(cache_dir / "gene_embeddings.npy").astype(np.float32)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    return index, emb


def feature_for_genes(genes: list[str], gene_index: dict[str, int], gene_emb: np.ndarray) -> np.ndarray | None:
    ids = [gene_index[g] for g in genes if g in gene_index]
    if not ids:
        return None
    vals = gene_emb[ids]
    mean = vals.mean(axis=0)
    if len(ids) >= 2:
        pair = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair.append(float(np.dot(vals[i], vals[j])))
        pair_mean = float(np.mean(pair)) if pair else 0.0
    else:
        pair_mean = 0.0
    extra = np.asarray(
        [
            float(len(genes)),
            float(len(ids)) / max(float(len(genes)), 1.0),
            pair_mean,
        ],
        dtype=np.float32,
    )
    return np.concatenate([mean.astype(np.float32), extra], axis=0)


def condition_means(data_dir: Path, ds: str, cond: str, *, max_cells: int) -> tuple[np.ndarray, np.ndarray] | None:
    path = data_dir / f"{ds}.h5"
    if not path.is_file():
        return None
    with h5py.File(path, "r") as handle:
        conditions = decode(np.asarray(handle["conditions"]))
        by_cond = {c: i for i, c in enumerate(conditions)}
        idx = by_cond.get(cond)
        if idx is None:
            return None
        ctrl_offsets = np.asarray(handle["ctrl/offsets"])
        gt_offsets = np.asarray(handle["gt/offsets"])
        c0, c1 = int(ctrl_offsets[idx]), int(ctrl_offsets[idx + 1])
        g0, g1 = int(gt_offsets[idx]), int(gt_offsets[idx + 1])
        if c1 <= c0 or g1 <= g0:
            return None
        ctrl = sample_mean(
            handle["ctrl/emb"],
            c0,
            c1,
            max_cells=max_cells,
            key=f"ctrl|{ds}|{cond}|{max_cells}",
        )
        gt = sample_mean(
            handle["gt/emb"],
            g0,
            g1,
            max_cells=max_cells,
            key=f"gt|{ds}|{cond}|{max_cells}",
        )
    return ctrl, gt


def collect_rows(
    *,
    data_dir: Path,
    split: dict[str, Any],
    metadata: dict[str, Any],
    gene_index: dict[str, int],
    gene_emb: np.ndarray,
    max_cells: int,
    max_train_conditions_per_dataset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for ds, parts in sorted(split.items()):
        train = [str(c) for c in parts.get("train", [])]
        train = sorted(train, key=lambda c: hashlib.sha1(f"train|{ds}|{c}".encode()).hexdigest())
        if max_train_conditions_per_dataset > 0:
            train = train[: int(max_train_conditions_per_dataset)]
        for group, conds in [("train", train)] + [(g, [str(c) for c in parts.get(g, [])]) for g in GROUPS]:
            for cond in conds:
                meta = (metadata.get(ds) or {}).get(cond) or {}
                genes = [str(g) for g in meta.get("genes") or []]
                if not genes:
                    continue
                feat = feature_for_genes(genes, gene_index, gene_emb)
                if feat is None:
                    continue
                means = condition_means(data_dir, str(ds), str(cond), max_cells=max_cells)
                if means is None:
                    continue
                ctrl, gt = means
                row = {
                    "dataset": str(ds),
                    "condition": str(cond),
                    "group": group,
                    "genes": genes,
                    "feature": feat.astype(np.float32),
                    "ctrl_mean": ctrl.astype(np.float32),
                    "gt_mean": gt.astype(np.float32),
                    "residual": (gt - ctrl).astype(np.float32),
                }
                if group == "train" and len(genes) == 1:
                    train_rows.append(row)
                elif group != "train":
                    val_rows.append(row)
    return train_rows, val_rows


def transform_rows(rows: list[dict[str, Any]], normalizers: dict[str, ResponseNormalizer], mode: str) -> np.ndarray:
    ys = []
    for row in rows:
        y = torch.as_tensor(row["residual"], dtype=torch.float32)
        if mode == "raw_ridge":
            out = y
        elif mode == "dataset_scale_ridge":
            out = normalizers["dataset_scale"].transform_delta(str(row["dataset"]), y)
        elif mode == "pca_subspace_ridge":
            out = normalizers["pca_subspace"].transform_delta(str(row["dataset"]), y)
        elif mode == "dataset_scale_pca_ridge":
            out = normalizers["dataset_scale_pca"].transform_delta(str(row["dataset"]), y)
        else:
            raise ValueError(mode)
        ys.append(out.cpu().numpy().astype(np.float32))
    return np.stack(ys)


def inverse_pred(pred: np.ndarray, row: dict[str, Any], normalizers: dict[str, ResponseNormalizer], mode: str) -> np.ndarray:
    y = torch.as_tensor(pred, dtype=torch.float32)
    if mode == "raw_ridge":
        out = y
    elif mode == "dataset_scale_ridge":
        out = normalizers["dataset_scale"].inverse_delta(str(row["dataset"]), y)
    elif mode == "pca_subspace_ridge":
        out = normalizers["pca_subspace"].inverse_delta(str(row["dataset"]), y)
    elif mode == "dataset_scale_pca_ridge":
        out = normalizers["dataset_scale_pca"].inverse_delta(str(row["dataset"]), y)
    else:
        raise ValueError(mode)
    return out.cpu().numpy().astype(np.float32)


def score_pred(row: dict[str, Any], pred_resid: np.ndarray, pert_means: dict[str, np.ndarray]) -> dict[str, float]:
    pred_endpoint = row["ctrl_mean"] + pred_resid
    gt_endpoint = row["gt_mean"]
    pert_mean = pert_means.get(str(row["dataset"]))
    return {
        "direct_pearson": pearson(pred_endpoint, gt_endpoint),
        "residual_pearson": pearson(pred_resid, row["residual"]),
        "pearson_pert_proxy": float("nan") if pert_mean is None else pearson(pred_endpoint - pert_mean, gt_endpoint - pert_mean),
    }


def fit_predict_lodo(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    *,
    normalizers: dict[str, ResponseNormalizer],
    pert_means: dict[str, np.ndarray],
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    out = []
    datasets = sorted({r["dataset"] for r in val_rows})
    global_mean = np.mean(np.stack([r["residual"] for r in train_rows]), axis=0)
    for heldout_ds in datasets:
        train = [r for r in train_rows if r["dataset"] != heldout_ds]
        vals = [r for r in val_rows if r["dataset"] == heldout_ds]
        if len(train) < 8 or not vals:
            continue
        x_train = np.stack([r["feature"] for r in train]).astype(np.float32)
        x_val = np.stack([r["feature"] for r in vals]).astype(np.float32)
        for mode in MODES:
            y_train = transform_rows(train, normalizers, mode)
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), random_state=int(seed)))
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            for row, p in zip(vals, pred):
                pred_resid = inverse_pred(p, row, normalizers, mode)
                score = score_pred(row, pred_resid, pert_means)
                out.append(
                    {
                        "mode": mode,
                        "dataset": row["dataset"],
                        "condition": row["condition"],
                        "group": row["group"],
                        "pearson_pert_proxy": score["pearson_pert_proxy"],
                        "direct_pearson": score["direct_pearson"],
                        "residual_pearson": score["residual_pearson"],
                    }
                )
            rng = np.random.default_rng(stable_int(f"shuffle|{heldout_ds}|{mode}|{seed}"))
            shuffled = y_train[rng.permutation(len(y_train))]
            shuf_model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), random_state=int(seed)))
            shuf_model.fit(x_train, shuffled)
            shuf_pred = shuf_model.predict(x_val)
            for row, p in zip(vals, shuf_pred):
                pred_resid = inverse_pred(p, row, normalizers, mode)
                score = score_pred(row, pred_resid, pert_means)
                out.append(
                    {
                        "mode": f"{mode}_shuffled_target_control",
                        "dataset": row["dataset"],
                        "condition": row["condition"],
                        "group": row["group"],
                        "pearson_pert_proxy": score["pearson_pert_proxy"],
                        "direct_pearson": score["direct_pearson"],
                        "residual_pearson": score["residual_pearson"],
                    }
                )
        for row in vals:
            score = score_pred(row, global_mean, pert_means)
            out.append(
                {
                    "mode": "global_mean_residual_control",
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "group": row["group"],
                    "pearson_pert_proxy": score["pearson_pert_proxy"],
                    "direct_pearson": score["direct_pearson"],
                    "residual_pearson": score["residual_pearson"],
                }
            )
    return out


def dataset_equal(rows: list[dict[str, Any]], *, group: str, mode: str, metric: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["group"] == group and row["mode"] == mode:
            val = row.get(metric)
            if val is not None and np.isfinite(float(val)):
                by_ds[str(row["dataset"])].append(float(val))
    vals = [float(np.mean(v)) for v in by_ds.values() if v]
    return None if not vals else float(np.mean(vals))


def bootstrap_delta(
    rows: list[dict[str, Any]],
    *,
    group: str,
    candidate: str,
    baseline: str,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    paired: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["group"] != group or row["mode"] not in {candidate, baseline}:
            continue
        val = row.get(metric)
        if val is None or not np.isfinite(float(val)):
            continue
        paired[str(row["dataset"])][str(row["mode"])].append(float(val))
    by_ds = {}
    for ds, modes in paired.items():
        if candidate in modes and baseline in modes and modes[candidate] and modes[baseline]:
            by_ds[ds] = float(np.mean(modes[candidate]) - np.mean(modes[baseline]))
    datasets = sorted(by_ds)
    observed = None if not datasets else float(np.mean([by_ds[d] for d in datasets]))
    if not datasets:
        return {"group": group, "candidate": candidate, "baseline": baseline, "metric": metric, "n_datasets": 0}
    rng = np.random.default_rng(seed)
    samples = []
    vals = np.asarray([by_ds[d] for d in datasets], dtype=np.float64)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(vals), size=len(vals))
        samples.append(float(np.mean(vals[idx])))
    arr = np.asarray(samples, dtype=np.float64)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return {
        "group": group,
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "n_datasets": len(datasets),
        "delta": observed,
        "ci95": [float(lo), float(hi)],
        "p_improve": float(np.mean(arr > 0.0)),
        "p_harm": float(np.mean(arr < 0.0)),
        "per_dataset_delta": by_ds,
    }


def leave_one_delta(rows: list[dict[str, Any]], *, group: str, candidate: str, baseline: str, metric: str) -> dict[str, Any]:
    boot = bootstrap_delta(rows, group=group, candidate=candidate, baseline=baseline, metric=metric, n_boot=100, seed=1)
    datasets = sorted((boot.get("per_dataset_delta") or {}).keys())
    leaves = {}
    for drop in datasets:
        vals = [v for ds, v in (boot.get("per_dataset_delta") or {}).items() if ds != drop]
        if vals:
            leaves[drop] = float(np.mean(vals))
    return {
        "group": group,
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "observed": boot.get("delta"),
        "leave_one_min": None if not leaves else float(min(leaves.values())),
        "leave_one_max": None if not leaves else float(max(leaves.values())),
        "sign_consistent_positive": bool(leaves) and all(v > 0.0 for v in leaves.values()),
        "leave_one_by_dataset": leaves,
    }


def summarize(rows: list[dict[str, Any]], n_boot: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    deltas = []
    loo = []
    all_modes = sorted({r["mode"] for r in rows})
    for group in GROUPS:
        for mode in all_modes:
            summaries.append(
                {
                    "group": group,
                    "mode": mode,
                    "n_conditions": sum(1 for r in rows if r["group"] == group and r["mode"] == mode),
                    "n_datasets": len({r["dataset"] for r in rows if r["group"] == group and r["mode"] == mode}),
                    "pearson_pert_proxy_dataset_equal": dataset_equal(rows, group=group, mode=mode, metric="pearson_pert_proxy"),
                    "residual_pearson_dataset_equal": dataset_equal(rows, group=group, mode=mode, metric="residual_pearson"),
                }
            )
        for mode in MODES:
            for baseline in ("raw_ridge", f"{mode}_shuffled_target_control", "global_mean_residual_control"):
                if mode == baseline:
                    continue
                deltas.append(
                    bootstrap_delta(
                        rows,
                        group=group,
                        candidate=mode,
                        baseline=baseline,
                        metric="pearson_pert_proxy",
                        n_boot=n_boot,
                        seed=stable_int(f"{seed}|{group}|{mode}|{baseline}"),
                    )
                )
            loo.append(
                leave_one_delta(
                    rows,
                    group=group,
                    candidate=mode,
                    baseline="raw_ridge",
                    metric="pearson_pert_proxy",
                )
            )
    return summaries, deltas, loo


def assess(deltas: list[dict[str, Any]], loo: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(d["group"], d["candidate"], d["baseline"]): d for d in deltas}
    loo_by = {(d["group"], d["candidate"], d["baseline"]): d for d in loo}
    decisions = []
    for mode in MODES:
        if mode == "raw_ridge":
            continue
        cross = by.get((GROUPS[0], mode, "raw_ridge")) or {}
        fam = by.get((GROUPS[1], mode, "raw_ridge")) or {}
        shuf_cross = by.get((GROUPS[0], mode, f"{mode}_shuffled_target_control")) or {}
        mean_cross = by.get((GROUPS[0], mode, "global_mean_residual_control")) or {}
        cross_loo = loo_by.get((GROUPS[0], mode, "raw_ridge")) or {}
        reasons = []
        if not cross or cross.get("delta") is None:
            reasons.append("missing_cross_background_delta")
        else:
            ci = cross.get("ci95") or [None, None]
            if float(cross.get("p_improve", 0.0)) < 0.90 and not (ci[0] is not None and float(ci[0]) > 0.0):
                reasons.append("cross_background_pp_not_supported")
        if not fam or fam.get("delta") is None:
            reasons.append("missing_family_delta")
        elif float(fam.get("p_harm", 1.0)) > 0.20:
            reasons.append("family_pp_harm_risk")
        if shuf_cross and shuf_cross.get("delta") is not None and float(shuf_cross["delta"]) <= 0:
            reasons.append("not_better_than_shuffled_target_control")
        if mean_cross and mean_cross.get("delta") is not None and float(mean_cross["delta"]) <= 0:
            reasons.append("not_better_than_global_mean_control")
        if cross_loo and not bool(cross_loo.get("sign_consistent_positive")):
            reasons.append("leave_one_dataset_not_stable")
        status = "cpu_gate_pass_candidate" if not reasons else "cpu_gate_fail_or_diagnostic"
        decisions.append(
            {
                "mode": mode,
                "status": status,
                "reasons": reasons,
                "cross_background_delta": cross,
                "family_delta": fam,
                "cross_background_vs_shuffle": shuf_cross,
                "cross_background_vs_global_mean": mean_cross,
                "cross_background_leave_one": cross_loo,
            }
        )
    overall = "cpu_gate_pass" if any(d["status"] == "cpu_gate_pass_candidate" for d in decisions) else "cpu_gate_fail_or_diagnostic"
    return {"overall_status": overall, "decisions": decisions}


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        if not np.isfinite(float(v)):
            return "NA"
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Representation-Normalization / Covariate CPU Gate",
        "",
        f"Status: `{payload['decision']['overall_status']}`",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- pert_means_file: `{payload['pert_means_file']}`",
        f"- response_normalizer: `{payload['response_normalizer']}`",
        f"- gene_cache: `{payload['gene_cache']}`",
        f"- train rows: `{payload['n_train_rows']}`",
        f"- val rows: `{payload['n_val_rows']}`",
        "- forbidden inputs: canonical test outcomes, held-out multi GT, and posthoc predictions are not used.",
        "",
        "## Mode Summary",
        "",
        "| group | mode | n cond | n ds | pp proxy | residual corr |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["summaries"]:
        lines.append(
            "| {group} | `{mode}` | {n_conditions} | {n_datasets} | {pp} | {resid} |".format(
                group=row["group"],
                mode=row["mode"],
                n_conditions=row["n_conditions"],
                n_datasets=row["n_datasets"],
                pp=fmt(row["pearson_pert_proxy_dataset_equal"]),
                resid=fmt(row["residual_pearson_dataset_equal"]),
            )
        )
    lines += [
        "",
        "## Gate Decisions",
        "",
        "| mode | status | reasons | crossbg delta vs raw | crossbg CI | crossbg p_improve | family delta vs raw | family p_harm |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for dec in payload["decision"]["decisions"]:
        c = dec.get("cross_background_delta") or {}
        f = dec.get("family_delta") or {}
        ci = c.get("ci95") or []
        lines.append(
            "| {mode} | `{status}` | {reasons} | {cd} | {ci} | {pi} | {fd} | {fh} |".format(
                mode=dec["mode"],
                status=dec["status"],
                reasons=", ".join(dec["reasons"]) or "-",
                cd=fmt(c.get("delta")),
                ci="NA" if len(ci) != 2 else f"[{fmt(ci[0])}, {fmt(ci[1])}]",
                pi=fmt(c.get("p_improve")),
                fd=fmt(f.get("delta")),
                fh=fmt(f.get("p_harm")),
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- A pass would allow one small xverse warm-start GPU smoke with the winning normalization/covariate idea.",
        "- A fail means normalization/covariate information is diagnostic only; do not launch GPU from this evidence alone.",
        "- This audit scores predictions after inverse normalization in raw endpoint pp coordinates, so MMD-only geometry gains cannot pass by themselves.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--pert-means-file", type=Path, default=DEFAULT_PERT_MEANS)
    ap.add_argument("--response-normalizer", type=Path, default=DEFAULT_NORMALIZER)
    ap.add_argument("--gene-cache", type=Path, default=DEFAULT_GENE_CACHE)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--max-train-conditions-per-dataset", type=int, default=256)
    ap.add_argument("--max-cells-per-condition", type=int, default=512)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    split_path = args.split_file.expanduser().resolve()
    normalizer_path = args.response_normalizer.expanduser().resolve()
    manifest = load_json(data_dir / "manifest.json")
    metadata = load_json(Path(manifest["condition_metadata_file"]))
    split = load_json(split_path)
    pert_means = {k: v.astype(np.float32) for k, v in np.load(str(args.pert_means_file.expanduser().resolve())).items()}
    gene_index, gene_emb = load_gene_embeddings(args.gene_cache.expanduser().resolve())

    normalizers = {
        mode: ResponseNormalizer.from_npz(
            normalizer_path,
            mode=mode,
            strict_split_file=split_path,
            strict_emb_dim=int(manifest["emb_dim"]),
        )
        for mode in ("dataset_scale", "pca_subspace", "dataset_scale_pca")
    }

    train_rows, val_rows = collect_rows(
        data_dir=data_dir,
        split=split,
        metadata=metadata,
        gene_index=gene_index,
        gene_emb=gene_emb,
        max_cells=int(args.max_cells_per_condition),
        max_train_conditions_per_dataset=int(args.max_train_conditions_per_dataset),
    )
    rows = fit_predict_lodo(
        train_rows,
        val_rows,
        normalizers=normalizers,
        pert_means=pert_means,
        alpha=float(args.ridge_alpha),
        seed=int(args.seed),
    )
    summaries, deltas, loo = summarize(rows, int(args.bootstrap), int(args.seed))
    decision = assess(deltas, loo)
    payload = {
        "status": "complete",
        "decision": decision,
        "data_dir": str(data_dir),
        "split_file": str(split_path),
        "pert_means_file": str(args.pert_means_file.expanduser().resolve()),
        "response_normalizer": str(normalizer_path),
        "gene_cache": str(args.gene_cache.expanduser().resolve()),
        "n_train_rows": len(train_rows),
        "n_val_rows": len(val_rows),
        "modes": list(MODES),
        "groups": list(GROUPS),
        "summaries": summaries,
        "deltas": deltas,
        "leave_one": loo,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": decision["overall_status"], "out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
