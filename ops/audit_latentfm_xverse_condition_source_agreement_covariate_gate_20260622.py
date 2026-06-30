#!/usr/bin/env python3
"""CPU gate for xverse condition-source agreement/confidence covariates.

This audit uses only the train split from the train-only v2 proxy split. It fits
simple multi-output ridge residual predictors from deployable gene-source
features and evaluates them on the split's internal single/background proxy
groups. It is a gate for whether a tiny condition-source confidence/router GPU
smoke is justified; it is not a canonical test result.
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
DEFAULT_SCGPT = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
DEFAULT_CELLNAVI = ROOT / "pretrainckpt/genepert_cache/cellnavi_embed_gene"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_condition_source_agreement_covariate_gate_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CONDITION_SOURCE_AGREEMENT_COVARIATE_GATE_20260622.md"

PROXY_GROUPS = (
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


def residual_for_condition(handle: h5py.File, by_cond: dict[str, int], cond: str, max_cells: int) -> np.ndarray | None:
    idx = by_cond.get(cond)
    if idx is None:
        return None
    ctrl = condition_mean(handle, "ctrl", idx, max_cells)
    gt = condition_mean(handle, "gt", idx, max_cells)
    if ctrl is None or gt is None:
        return None
    return (gt - ctrl).astype(np.float32)


def single_gene(metadata: dict[str, Any], ds: str, cond: str) -> str | None:
    meta = (metadata.get(ds) or {}).get(cond) or {}
    genes = [str(g) for g in meta.get("genes") or []]
    if len(genes) != 1:
        return None
    raw = str(meta.get("perturbation_type_raw") or "").lower()
    if "drug" in raw or "compound" in raw or "chemical" in raw:
        return None
    return genes[0]


def load_gene_embeddings(cache_dir: Path) -> tuple[dict[str, int], np.ndarray]:
    index: dict[str, int] = {}
    with (cache_dir / "gene_index.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene, idx = parts[0], parts[1]
            if gene in {"PAD", "UNK", ""}:
                continue
            try:
                index[gene] = int(idx)
            except ValueError:
                continue
    emb = np.load(cache_dir / "gene_embeddings.npy").astype(np.float32)
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    return index, emb / norms


def zero_feature(dim: int) -> np.ndarray:
    return np.zeros((dim,), dtype=np.float32)


def gene_feature(
    gene: str,
    sc_idx: dict[str, int],
    sc_emb: np.ndarray,
    cn_idx: dict[str, int],
    cn_emb: np.ndarray,
) -> dict[str, Any]:
    sc_hit = gene in sc_idx
    cn_hit = gene in cn_idx
    sc = sc_emb[sc_idx[gene]] if sc_hit else zero_feature(sc_emb.shape[1])
    cn = cn_emb[cn_idx[gene]] if cn_hit else zero_feature(cn_emb.shape[1])
    return {
        "scgpt": sc.astype(np.float32),
        "cellnavi": cn.astype(np.float32),
        "agreement": np.asarray(
            [
                float(sc_hit),
                float(cn_hit),
                float(sc_hit and cn_hit),
                float(np.linalg.norm(sc)),
                float(np.linalg.norm(cn)),
            ],
            dtype=np.float32,
        ),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or y.size != x.size:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    y_mean = y.mean(axis=0, keepdims=True)
    xs = (x - x_mean) / x_std
    yc = y - y_mean
    xtx = xs.T @ xs
    xtx.flat[:: xtx.shape[0] + 1] += alpha
    w = np.linalg.solve(xtx, xs.T @ yc)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "w": w}


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, dtype=np.float64) - model["x_mean"]) / model["x_std"]
    return (xs @ model["w"] + model["y_mean"]).astype(np.float32)


def build_rows(
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
        chosen_train = set(stable_subset([c for c, _ in train_single], max_train_per_dataset, f"p11train|{ds}"))
        with h5py.File(path, "r") as handle:
            conditions = decode(np.asarray(handle["conditions"]))
            by_cond = {c: i for i, c in enumerate(conditions)}
            for cond, gene in train_single:
                if cond not in chosen_train:
                    continue
                residual = residual_for_condition(handle, by_cond, cond, max_cells)
                if residual is None:
                    continue
                train_rows.append({"dataset": ds, "condition": cond, "gene": gene, "residual": residual})
            for group in PROXY_GROUPS:
                for cond in obj.get(group) or []:
                    cond = str(cond)
                    gene = single_gene(metadata, ds, cond)
                    if gene is None:
                        continue
                    residual = residual_for_condition(handle, by_cond, cond, max_cells)
                    if residual is None:
                        continue
                    val_rows.append({"dataset": ds, "condition": cond, "gene": gene, "group": group, "residual": residual})
    return train_rows, val_rows


def feature_matrix(
    rows: list[dict[str, Any]],
    mode: str,
    sc_idx: dict[str, int],
    sc_emb: np.ndarray,
    cn_idx: dict[str, int],
    cn_emb: np.ndarray,
) -> np.ndarray:
    feats = []
    for row in rows:
        f = gene_feature(str(row["gene"]), sc_idx, sc_emb, cn_idx, cn_emb)
        if mode == "scgpt":
            feats.append(f["scgpt"])
        elif mode == "cellnavi":
            feats.append(f["cellnavi"])
        elif mode == "agreement":
            feats.append(np.concatenate([f["scgpt"], f["cellnavi"], f["agreement"]]).astype(np.float32))
        else:
            raise ValueError(f"unknown feature mode: {mode}")
    return np.vstack(feats).astype(np.float32)


def mean_by_dataset(train_rows: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    by_ds: dict[str, list[np.ndarray]] = defaultdict(list)
    all_resid = []
    for row in train_rows:
        r = np.asarray(row["residual"], dtype=np.float32)
        by_ds[str(row["dataset"])].append(r)
        all_resid.append(r)
    global_mean = np.mean(np.vstack(all_resid), axis=0).astype(np.float32)
    ds_mean = {ds: np.mean(np.vstack(vals), axis=0).astype(np.float32) for ds, vals in by_ds.items()}
    return ds_mean, global_mean


def equal_dataset_mean(values: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in values:
        value = row.get(key)
        if value is not None:
            by_ds[str(row["dataset"])].append(float(value))
    if not by_ds:
        return None
    return float(np.mean([np.mean(v) for v in by_ds.values() if v]))


def paired_bootstrap(values: list[dict[str, Any]], candidate: str, baseline: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    diffs_by_ds: dict[str, list[float]] = defaultdict(list)
    for row in values:
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
        ds_means = []
        for ds in sample_ds:
            vals = np.asarray(diffs_by_ds[str(ds)], dtype=np.float64)
            ds_means.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
        boot.append(float(np.mean(ds_means)))
    boot_arr = np.asarray(boot, dtype=np.float64)
    leave_one = {}
    for ds in datasets:
        rest = [d for d in datasets if d != ds]
        if rest:
            leave_one[ds] = float(np.mean([np.mean(diffs_by_ds[d]) for d in rest]))
    return {
        "status": "ok",
        "n_conditions": int(sum(len(diffs_by_ds[d]) for d in datasets)),
        "n_datasets": int(len(datasets)),
        "delta_mean": point,
        "ci95": [float(np.quantile(boot_arr, 0.025)), float(np.quantile(boot_arr, 0.975))],
        "p_improve": float(np.mean(boot_arr > 0.0)),
        "p_harm": float(np.mean(boot_arr < 0.0)),
        "leave_one_min": min(leave_one.values()) if leave_one else None,
        "leave_one_max": max(leave_one.values()) if leave_one else None,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Condition-Source Agreement Covariate Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
        f"- data_dir: `{payload['data_dir']}`",
        f"- split_file: `{payload['split_file']}`",
        f"- scGPT cache: `{payload['scgpt_cache']}`",
        f"- CellNavi cache: `{payload['cellnavi_cache']}`",
        f"- leakage status: `{payload['leakage_status']}`",
        f"- train rows: `{payload['n_train_rows']}`",
        f"- validation rows: `{payload['n_val_rows']}`",
        "",
        "## Absolute Proxy Scores",
        "",
        "| group | model | equal-dataset residual pp proxy |",
        "|---|---|---:|",
    ]
    for row in payload["absolute_scores"]:
        lines.append(f"| {row['group']} | {row['model']} | {row['mean']:.6f} |")
    lines += [
        "",
        "## Paired Deltas",
        "",
        "| group | candidate | baseline | n cond | n ds | delta | 95% CI | p improve | p harm | leave-one min | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["paired_deltas"]:
        ci = row.get("ci95") or [None, None]
        def fmt(x: Any) -> str:
            return "NA" if x is None else f"{float(x):+.6f}"
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
        "- This is a train-only internal proxy audit, not a canonical test result.",
        "- Passing this gate would only justify one small condition-source confidence/router GPU smoke.",
        "- Failing it keeps CellNavi/source-agreement work CPU-only for the current Track A queue.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--scgpt-cache", type=Path, default=DEFAULT_SCGPT)
    parser.add_argument("--cellnavi-cache", type=Path, default=DEFAULT_CELLNAVI)
    parser.add_argument("--max-train-per-dataset", type=int, default=512)
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    split = load_json(args.split_file)
    metadata = load_json(args.data_dir / "condition_metadata.json")
    train_rows, val_rows = build_rows(
        args.data_dir,
        split,
        metadata,
        max_train_per_dataset=args.max_train_per_dataset,
        max_cells=args.max_cells_per_condition,
    )
    if not train_rows or not val_rows:
        raise RuntimeError("missing train or validation rows")

    sc_idx, sc_emb = load_gene_embeddings(args.scgpt_cache)
    cn_idx, cn_emb = load_gene_embeddings(args.cellnavi_cache)
    y_train = np.vstack([row["residual"] for row in train_rows]).astype(np.float32)
    x_sc = feature_matrix(train_rows, "scgpt", sc_idx, sc_emb, cn_idx, cn_emb)
    x_cn = feature_matrix(train_rows, "cellnavi", sc_idx, sc_emb, cn_idx, cn_emb)
    x_ag = feature_matrix(train_rows, "agreement", sc_idx, sc_emb, cn_idx, cn_emb)
    rng = np.random.default_rng(args.seed)
    x_ag_shuf = x_ag[rng.permutation(len(x_ag))]

    models = {
        "scgpt_ridge": fit_ridge(x_sc, y_train, args.ridge_alpha),
        "cellnavi_ridge": fit_ridge(x_cn, y_train, args.ridge_alpha),
        "agreement_ridge": fit_ridge(x_ag, y_train, args.ridge_alpha),
        "shuffled_agreement_ridge": fit_ridge(x_ag_shuf, y_train, args.ridge_alpha),
    }
    ds_mean, global_mean = mean_by_dataset(train_rows)

    x_val = {
        "scgpt_ridge": feature_matrix(val_rows, "scgpt", sc_idx, sc_emb, cn_idx, cn_emb),
        "cellnavi_ridge": feature_matrix(val_rows, "cellnavi", sc_idx, sc_emb, cn_idx, cn_emb),
        "agreement_ridge": feature_matrix(val_rows, "agreement", sc_idx, sc_emb, cn_idx, cn_emb),
        "shuffled_agreement_ridge": feature_matrix(val_rows, "agreement", sc_idx, sc_emb, cn_idx, cn_emb),
    }
    preds = {name: predict_ridge(model, x_val[name]) for name, model in models.items()}

    eval_rows: list[dict[str, Any]] = []
    for i, row in enumerate(val_rows):
        true = np.asarray(row["residual"], dtype=np.float32)
        out = {
            "dataset": row["dataset"],
            "condition": row["condition"],
            "gene": row["gene"],
            "group": row["group"],
        }
        for name, arr in preds.items():
            out[name] = pearson(arr[i], true)
        out["dataset_mean"] = pearson(ds_mean.get(str(row["dataset"]), global_mean), true)
        out["global_mean"] = pearson(global_mean, true)
        eval_rows.append(out)

    model_names = [
        "agreement_ridge",
        "scgpt_ridge",
        "cellnavi_ridge",
        "shuffled_agreement_ridge",
        "dataset_mean",
        "global_mean",
    ]
    absolute_scores = []
    paired_deltas = []
    for group in PROXY_GROUPS:
        group_rows = [r for r in eval_rows if r["group"] == group]
        for name in model_names:
            absolute_scores.append({"group": group, "model": name, "mean": equal_dataset_mean(group_rows, name)})
        for baseline in ("scgpt_ridge", "cellnavi_ridge", "shuffled_agreement_ridge", "dataset_mean"):
            delta = paired_bootstrap(
                group_rows,
                "agreement_ridge",
                baseline,
                n_boot=args.n_boot,
                seed=args.seed + len(paired_deltas),
            )
            delta.update({"group": group, "candidate": "agreement_ridge", "baseline": baseline})
            paired_deltas.append(delta)

    by_key = {(r["group"], r["baseline"]): r for r in paired_deltas}
    cross_sc = by_key[(PROXY_GROUPS[0], "scgpt_ridge")]
    cross_shuf = by_key[(PROXY_GROUPS[0], "shuffled_agreement_ridge")]
    cross_ds = by_key[(PROXY_GROUPS[0], "dataset_mean")]
    fam_sc = by_key[(PROXY_GROUPS[1], "scgpt_ridge")]
    fam_shuf = by_key[(PROXY_GROUPS[1], "shuffled_agreement_ridge")]
    fam_ds = by_key[(PROXY_GROUPS[1], "dataset_mean")]

    reasons = []
    if cross_sc.get("status") != "ok" or not (
        float(cross_sc.get("p_improve") or 0.0) >= 0.90 or float((cross_sc.get("ci95") or [0.0])[0]) > 0.0
    ):
        reasons.append("cross_background_vs_scgpt_not_supported")
    if fam_sc.get("status") != "ok" or float(fam_sc.get("p_harm") if fam_sc.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("family_vs_scgpt_harm_risk")
    for label, row in (
        ("cross_background_vs_shuffled", cross_shuf),
        ("cross_background_vs_dataset_mean", cross_ds),
        ("family_vs_shuffled", fam_shuf),
        ("family_vs_dataset_mean", fam_ds),
    ):
        if row.get("status") != "ok" or float(row.get("delta_mean") or 0.0) <= 0.0:
            reasons.append(f"{label}_not_positive")
    if cross_sc.get("leave_one_min") is None or float(cross_sc["leave_one_min"]) <= 0.0:
        reasons.append("cross_background_leave_one_dataset_flips_or_nonpositive")

    status = "cpu_gate_pass_launch_one_tiny_gpu_smoke" if not reasons else "cpu_gate_fail_do_not_launch_gpu"
    action = "launch_one_condition_source_confidence_smoke" if not reasons else "keep_condition_source_cpu_only"

    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "scgpt_cache": str(args.scgpt_cache),
        "cellnavi_cache": str(args.cellnavi_cache),
        "max_train_per_dataset": args.max_train_per_dataset,
        "max_cells_per_condition": args.max_cells_per_condition,
        "ridge_alpha": args.ridge_alpha,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "leakage_status": "train_only_v2_train_single_residuals_to_internal_proxy_no_canonical_test_no_posthoc_no_pert_means",
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
