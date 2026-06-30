#!/usr/bin/env python3
"""CPU gate for a Track A background-conditioned bilinear operator.

This is a leakage-safe feasibility gate, not model training. It asks whether a
simple gene x background interaction feature can predict train/internal
cross-background repair opportunity better than gene-only, background-only,
additive, and shuffled-background controls.

No training loop, dataset inference, checkpoint selection, canonical multi,
Track C query, held-out exact-row selection, or GPU is used.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/data/cyx/1030/scLatent")
REPO = ROOT / "CoupledFM"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.latent.config import Config  # noqa: E402
from model.utils.embeddings.gene_cache import GeneEmbeddingCache  # noqa: E402


ANCHOR_CKPT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
FORENSICS = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
PERT_MEANS = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    / "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz"
)
DATASET_PCA = (
    ROOT
    / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
    / "xverse_trainonly_crossbgval_v2_dataset_scale_pca32.npz"
)
OUT_JSON = ROOT / "reports/latentfm_tracka_background_bilinear_operator_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_BACKGROUND_BILINEAR_OPERATOR_GATE_20260627.md"


def _cfg_from_checkpoint(path: Path) -> Config:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    raw = ckpt.get("config") or {}
    fields = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in fields})


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with FORENSICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") != "internal_val_cross_background_seen_gene_proxy":
                continue
            anchor = fnum(row.get("anchor_pearson_pert"))
            if anchor is None:
                continue
            candidates = [
                fnum(row.get("gene_raw_mean")),
                fnum(row.get("dataset_mean")),
                fnum(row.get("global_mean")),
                fnum(row.get("shrink_k8")),
            ]
            if any(v is None for v in candidates):
                continue
            rows.append(
                {
                    "dataset": str(row.get("dataset")),
                    "condition": str(row.get("condition")),
                    "gene": str(row.get("gene", "")).strip().upper(),
                    "anchor": float(anchor),
                    "oracle_delta": float(max([anchor] + [float(v) for v in candidates]) - anchor),
                    "safe_scalar": np.asarray(
                        [
                            math.log1p(float(row.get("gene_train_count") or 0.0)),
                            float(row.get("gene_pred_norm") or 0.0),
                            float(row.get("dataset_pred_norm") or 0.0),
                            float(row.get("global_pred_norm") or 0.0),
                            float(row.get("gene_dataset_cosine") or 0.0),
                        ],
                        dtype=float,
                    ),
                }
            )
    return rows


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def pca_project_unsupervised(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comp = vt[:dim].T
    z = xc @ comp
    sd = z.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return z / sd


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    mu, sd = standardize_fit(x_train)
    xt = standardize_apply(x_train, mu, sd)
    xv = standardize_apply(x_test, mu, sd)
    xt = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
    xv = np.concatenate([np.ones((xv.shape[0], 1)), xv], axis=1)
    reg = np.eye(xt.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xt.T @ xt + reg, xt.T @ y_train)
    return xv @ beta


def bootstrap(vals: np.ndarray, seed: int, n_boot: int = 5000) -> dict[str, float]:
    if vals.size == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "p_gt0": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    return {
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "p_gt0": float(np.mean(boot > 0)),
    }


def build_feature_blocks(rows: list[dict[str, Any]], cfg: Config) -> dict[str, np.ndarray]:
    cache = GeneEmbeddingCache(Path(cfg.pert_gene_emb_cache_dir))
    emb = cache.embedding_numpy(copy=False)
    gene_raw = []
    for row in rows:
        idx = int(cache.lookup(row["gene"]))
        if idx in {int(cache.pad_index), int(cache.unk_index)}:
            gene_raw.append(np.zeros((cache.embed_dim,), dtype=float))
        else:
            gene_raw.append(np.asarray(emb[idx], dtype=float))
    gene = pca_project_unsupervised(np.vstack(gene_raw), dim=min(8, len(rows) - 1))

    z = np.load(DATASET_PCA, allow_pickle=True)
    pert = np.load(PERT_MEANS, allow_pickle=True)
    factors = json.loads(str(z["dataset_scale_factors_json"]))
    pca_mean = np.asarray(z["pca_mean"], dtype=float)
    pca_components = np.asarray(z["pca_components"], dtype=float)
    pca_scales = np.asarray(z["pca_scales"], dtype=float)
    bg_rows = []
    for row in rows:
        ds = row["dataset"]
        vec = np.asarray(pert[ds], dtype=float)
        coords = ((vec - pca_mean) @ pca_components[:8].T) / np.maximum(pca_scales[:8], 1e-6)
        bg_rows.append(np.concatenate([[math.log(float(factors[ds]))], coords]))
    bg = np.vstack(bg_rows)
    scalar = np.vstack([row["safe_scalar"] for row in rows])
    bilinear = np.einsum("bi,bj->bij", gene, bg).reshape(len(rows), -1)
    return {
        "intercept_only": np.zeros((len(rows), 1), dtype=float),
        "gene_only": np.concatenate([scalar, gene], axis=1),
        "background_only": np.concatenate([scalar, bg], axis=1),
        "additive": np.concatenate([scalar, gene, bg], axis=1),
        "bilinear": np.concatenate([scalar, gene, bg, bilinear], axis=1),
    }


def lodo_predictions(rows: list[dict[str, Any]], features: dict[str, np.ndarray], y: np.ndarray) -> dict[str, dict[str, Any]]:
    datasets = sorted({row["dataset"] for row in rows})
    out: dict[str, dict[str, Any]] = {}
    for name, x in features.items():
        pred = np.zeros_like(y, dtype=float)
        for ds in datasets:
            test = np.asarray([i for i, row in enumerate(rows) if row["dataset"] == ds], dtype=int)
            train = np.asarray([i for i, row in enumerate(rows) if row["dataset"] != ds], dtype=int)
            if name == "intercept_only":
                pred[test] = float(y[train].mean())
            else:
                pred[test] = ridge_predict(x[train], y[train], x[test], alpha=10.0)
        corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1e-9 and np.std(y) > 1e-9 else 0.0
        mse = float(np.mean((pred - y) ** 2))
        top = pred >= float(np.quantile(pred, 0.50))
        selected_mean = float(y[top].mean()) if np.any(top) else 0.0
        selected_gain = selected_mean - float(y.mean())
        ds_means = {}
        for ds in datasets:
            vals = y[[i for i, row in enumerate(rows) if row["dataset"] == ds and top[i]]]
            ds_means[ds] = float(vals.mean()) if vals.size else 0.0
        out[name] = {
            "corr": corr,
            "mse": mse,
            "selected_fraction": float(np.mean(top)),
            "selected_mean_delta": selected_mean,
            "selected_gain_vs_all": selected_gain,
            "selected_dataset_min": float(min(ds_means.values())) if ds_means else 0.0,
            "selected_dataset_means": ds_means,
            "predictions": pred.tolist(),
        }
    return out


def shuffled_background_control(
    rows: list[dict[str, Any]],
    base_features: dict[str, np.ndarray],
    y: np.ndarray,
    n_perm: int = 2000,
) -> dict[str, float]:
    rng = np.random.default_rng(20260627)
    additive = base_features["additive"]
    bilinear = base_features["bilinear"]
    add_dim = additive.shape[1]
    gains = []
    corrs = []
    for _ in range(n_perm):
        perm = rng.permutation(len(rows))
        x = np.concatenate([additive, bilinear[perm, add_dim:]], axis=1)
        pred_rows = lodo_predictions(rows, {"shuffle_bilinear": x}, y)["shuffle_bilinear"]
        gains.append(float(pred_rows["selected_gain_vs_all"]))
        corrs.append(float(pred_rows["corr"]))
    return {
        "shuffle_gain_mean": float(np.mean(gains)),
        "shuffle_gain_p_ge_actual": float(np.mean(np.asarray(gains) >= float(lodo_predictions(rows, {"bilinear": bilinear}, y)["bilinear"]["selected_gain_vs_all"]))),
        "shuffle_corr_mean": float(np.mean(corrs)),
        "shuffle_corr_p_ge_actual": float(np.mean(np.asarray(corrs) >= float(lodo_predictions(rows, {"bilinear": bilinear}, y)["bilinear"]["corr"]))),
    }


def main() -> None:
    cfg = _cfg_from_checkpoint(ANCHOR_CKPT)
    rows = load_rows()
    y = np.asarray([row["oracle_delta"] for row in rows], dtype=float)
    features = build_feature_blocks(rows, cfg)
    metrics = lodo_predictions(rows, features, y)
    bilinear_gain_minus_additive = (
        float(metrics["bilinear"]["selected_gain_vs_all"])
        - float(metrics["additive"]["selected_gain_vs_all"])
    )
    bilinear_corr_minus_additive = float(metrics["bilinear"]["corr"]) - float(metrics["additive"]["corr"])
    shuffle = shuffled_background_control(rows, features, y)
    ds_deltas = np.asarray(
        [
            metrics["bilinear"]["selected_dataset_means"][ds]
            - metrics["additive"]["selected_dataset_means"][ds]
            for ds in sorted(metrics["bilinear"]["selected_dataset_means"])
        ],
        dtype=float,
    )
    bs = bootstrap(ds_deltas, seed=20260627)

    reasons: list[str] = []
    if bilinear_gain_minus_additive < 0.01:
        reasons.append("bilinear_selected_gain_minus_additive_lt_0p01")
    if bs["ci_low"] <= 0:
        reasons.append("dataset_bootstrap_bilinear_minus_additive_ci_low_not_above_0")
    if bilinear_corr_minus_additive < 0.05:
        reasons.append("bilinear_corr_minus_additive_lt_0p05")
    if shuffle["shuffle_gain_p_ge_actual"] > 0.01 or shuffle["shuffle_corr_p_ge_actual"] > 0.01:
        reasons.append("shuffled_background_control_not_beaten")
    if metrics["bilinear"]["selected_dataset_min"] < -0.02:
        reasons.append("bilinear_selected_dataset_min_below_minus_0p02")
    reasons.append("real_trainonly_mmd_noharm_not_run_no_gpu")

    status = "tracka_background_bilinear_operator_gate_fail_no_gpu"
    if not any(r != "real_trainonly_mmd_noharm_not_run_no_gpu" for r in reasons):
        status = "tracka_background_bilinear_operator_gate_pass_needs_mmd_noharm_no_gpu"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "training_loop": False,
            "dataset_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "heldout_exact_rows_used_for_selection": False,
        },
        "inputs": {
            "anchor_checkpoint_for_config": str(ANCHOR_CKPT),
            "residual_forensics": str(FORENSICS),
            "pert_means": str(PERT_MEANS),
            "dataset_pca": str(DATASET_PCA),
        },
        "n_rows": len(rows),
        "n_datasets": len({row["dataset"] for row in rows}),
        "target_oracle_delta_mean": float(y.mean()),
        "target_oracle_delta_positive_fraction": float(np.mean(y > 0)),
        "feature_dims": {k: int(v.shape[1]) for k, v in features.items()},
        "metrics": {k: {kk: vv for kk, vv in val.items() if kk != "predictions"} for k, val in metrics.items()},
        "bilinear_gain_minus_additive": bilinear_gain_minus_additive,
        "bilinear_corr_minus_additive": bilinear_corr_minus_additive,
        "dataset_bootstrap_bilinear_minus_additive": bs,
        "shuffle_control": shuffle,
        "decision_reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Background-Conditioned Bilinear Operator Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only LODO feasibility gate over train/internal cross-background proxy rows. No training loop, dataset inference, checkpoint selection, canonical multi selection, Track C query, held-out exact-row selection, or GPU.",
        "",
        "## Summary",
        "",
        f"- Rows/datasets: `{len(rows)}` / `{len({row['dataset'] for row in rows})}`",
        f"- Target oracle delta mean: `{float(y.mean()):+.6f}`",
        f"- Bilinear selected-gain minus additive: `{bilinear_gain_minus_additive:+.6f}`",
        f"- Bilinear corr minus additive: `{bilinear_corr_minus_additive:+.6f}`",
        f"- Dataset bootstrap CI for bilinear-additive selected delta: `[{bs['ci_low']:+.6f}, {bs['ci_high']:+.6f}]`",
        f"- Shuffled-background gain p>=actual: `{shuffle['shuffle_gain_p_ge_actual']:.4f}`",
        f"- Shuffled-background corr p>=actual: `{shuffle['shuffle_corr_p_ge_actual']:.4f}`",
        "",
        "## Model Family Comparison",
        "",
        "| model | corr | mse | selected_gain_vs_all | selected_dataset_min |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("intercept_only", "gene_only", "background_only", "additive", "bilinear"):
        m = metrics[name]
        lines.append(
            f"| `{name}` | {float(m['corr']):+.6f} | {float(m['mse']):.6f} | "
            f"{float(m['selected_gain_vs_all']):+.6f} | {float(m['selected_dataset_min']):+.6f} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This gate does not authorize GPU. A pass would only allow a stricter train-only MMD/no-harm gate and then a bounded smoke; a fail closes the current bilinear operator as an immediate route.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
