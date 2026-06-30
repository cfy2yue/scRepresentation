#!/usr/bin/env python3
"""CPU gate for a source/control-anchored residual predictor.

This is a mean-space feasibility gate, not a model checkpoint. It predicts
`GT_mean - ctrl_mean` from gene embeddings using train-only internal LODO
splits and evaluates whether `ctrl + predicted_residual` beats both the frozen
xverse anchor and source/control baselines.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
IN_DIR = ROOT / "reports/latentfm_xverse_8k_seed_ensemble_internal_means_20260627"
SEED_FILES = {
    "seed42": IN_DIR / "seed42_internal_split_group_means_evalseed42.json",
    "seed43": IN_DIR / "seed43_internal_split_group_means_evalseed42.json",
}
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
GENE_EMB = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene/gene_embeddings.npy"
GENE_INDEX = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene/gene_index.tsv"
OUT_DIR = ROOT / "reports/source_anchored_residual_cpu_gate_20260628"
OUT_JSON = ROOT / "reports/latentfm_source_anchored_residual_cpu_gate_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_SOURCE_ANCHORED_RESIDUAL_CPU_GATE_20260628.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def pp(endpoint: np.ndarray, gt: np.ndarray, pert: np.ndarray) -> float:
    return pearson(endpoint - pert, gt - pert)


def load_gene_embeddings() -> tuple[np.ndarray, dict[str, int]]:
    emb = np.load(GENE_EMB, mmap_mode="r")
    idx: dict[str, int] = {}
    with GENE_INDEX.open(encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                idx[parts[0].upper()] = int(parts[1])
    return np.asarray(emb, dtype=np.float32), idx


def load_condition_genes() -> dict[tuple[str, str], list[str]]:
    obj = json.loads(COND_META.read_text(encoding="utf-8"))
    out = {}
    for ds, ds_obj in obj.items():
        for cond, rec in ds_obj.items():
            genes = [str(g).upper() for g in rec.get("genes", []) if str(g).strip()]
            out[(str(ds), str(cond))] = genes
    return out


def gene_feature(genes: list[str], emb: np.ndarray, idx: dict[str, int]) -> np.ndarray:
    vecs = [emb[idx[g]] for g in genes if g in idx]
    if not vecs:
        return np.zeros(int(emb.shape[1]), dtype=np.float32)
    return np.mean(np.stack(vecs).astype(np.float32), axis=0)


def load_rows(path: Path, seed: str, emb: np.ndarray, idx: dict[str, int], genes_by_key: dict[tuple[str, str], list[str]]) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for group in GROUPS:
        for row in obj.get("groups", {}).get(group, {}).get("condition_metrics", []):
            ds = str(row["dataset"])
            cond = str(row["condition"])
            ctrl = np.asarray(row["ctrl_mean"], dtype=np.float32)
            gt = np.asarray(row["gt_mean"], dtype=np.float32)
            pred = np.asarray(row["pred_mean"], dtype=np.float32)
            pert = np.asarray(row["pert_mean"], dtype=np.float32)
            feat = gene_feature(genes_by_key.get((ds, cond), [cond]), emb, idx)
            anchor_pp = pp(pred, gt, pert)
            ctrl_pp = pp(ctrl, gt, pert)
            rows.append(
                {
                    "seed": seed,
                    "group": group,
                    "dataset": ds,
                    "condition": cond,
                    "feature": feat,
                    "target_resid": gt - ctrl,
                    "ctrl": ctrl,
                    "gt": gt,
                    "pert": pert,
                    "anchor_pp": float(anchor_pp),
                    "ctrl_pp": float(ctrl_pp),
                }
            )
    return rows


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {}
    for row in rows:
        seen.setdefault((row["dataset"], row["condition"]), row)
    return list(seen.values())


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-6] = 1.0
    xs = (x - mu) / sd
    xb = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float32), xs.astype(np.float32)], axis=1)
    reg = float(alpha) * np.eye(xb.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    norms = np.linalg.norm(y, axis=1)
    clip_norm = float(np.quantile(norms, 0.9)) if norms.size else 1.0
    return {"mu": mu, "sd": sd, "beta": beta.astype(np.float32), "clip_norm": np.asarray([clip_norm], dtype=np.float32)}


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    xs = (x - model["mu"]) / model["sd"]
    xb = np.concatenate([np.ones((xs.shape[0], 1), dtype=np.float32), xs.astype(np.float32)], axis=1)
    pred = xb @ model["beta"]
    clip = float(model["clip_norm"][0])
    norms = np.linalg.norm(pred, axis=1)
    scale = np.minimum(1.0, clip / np.maximum(norms, 1e-8))
    return pred * scale[:, None]


def score_predictions(eval_rows: list[dict[str, Any]], pred_resid: np.ndarray, mode: str) -> list[dict[str, Any]]:
    out = []
    for row, resid in zip(eval_rows, pred_resid):
        endpoint = row["ctrl"] + resid
        cand_pp = pp(endpoint, row["gt"], row["pert"])
        out.append(
            {
                "seed": row["seed"],
                "group": row["group"],
                "dataset": row["dataset"],
                "condition": row["condition"],
                "mode": mode,
                "anchor_pp": row["anchor_pp"],
                "ctrl_pp": row["ctrl_pp"],
                "candidate_pp": float(cand_pp),
                "dominance_pp": float(cand_pp - max(row["anchor_pp"], row["ctrl_pp"])),
            }
        )
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["group"], row["mode"])].append(row)
    out = {}
    for (group, mode), part in by_group.items():
        by_ds = defaultdict(list)
        for row in part:
            by_ds[row["dataset"]].append(float(row["dominance_pp"]))
        ds_vals = [float(np.mean(v)) for v in by_ds.values()]
        rng = np.random.default_rng(20260628)
        if ds_vals:
            arr = np.asarray(ds_vals, dtype=float)
            idx = rng.integers(0, arr.size, size=(5000, arr.size))
            means = arr[idx].mean(axis=1)
            ci_low = float(np.quantile(means, 0.025))
            ci_high = float(np.quantile(means, 0.975))
        else:
            ci_low = ci_high = 0.0
        out[f"{group}|{mode}"] = {
            "group": group,
            "mode": mode,
            "n": len(part),
            "n_datasets": len(by_ds),
            "candidate_pp": float(np.mean([r["candidate_pp"] for r in part])) if part else 0.0,
            "anchor_pp": float(np.mean([r["anchor_pp"] for r in part])) if part else 0.0,
            "ctrl_pp": float(np.mean([r["ctrl_pp"] for r in part])) if part else 0.0,
            "dominance_pp": float(np.mean(ds_vals)) if ds_vals else 0.0,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "dataset_min": float(min(ds_vals)) if ds_vals else 0.0,
            "row_positive_fraction": float(np.mean([r["dominance_pp"] > 0 for r in part])) if part else 0.0,
        }
    return out


def choose_alpha(train_rows: list[dict[str, Any]]) -> float:
    datasets = sorted({r["dataset"] for r in train_rows})
    uniq = unique_rows(train_rows)
    best_alpha = ALPHAS[0]
    best_score = -1e9
    for alpha in ALPHAS:
        preds = []
        for heldout in datasets:
            fit_rows = [r for r in uniq if r["dataset"] != heldout]
            val_rows = [r for r in train_rows if r["dataset"] == heldout]
            if len(fit_rows) < 8 or not val_rows:
                continue
            model = fit_ridge(
                np.stack([r["feature"] for r in fit_rows]),
                np.stack([r["target_resid"] for r in fit_rows]),
                alpha,
            )
            pred = predict(model, np.stack([r["feature"] for r in val_rows]))
            preds.extend(score_predictions(val_rows, pred, "inner"))
        summ = summarize(preds)
        vals = [v["dominance_pp"] - max(0.0, -v["dataset_min"]) for v in summ.values()]
        score = float(np.mean(vals)) if vals else -1e9
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return float(best_alpha)


def lodo(seed: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = sorted({r["dataset"] for r in rows})
    uniq_all = unique_rows(rows)
    predictions = []
    selected = []
    for heldout in datasets:
        train_rows = [r for r in rows if r["dataset"] != heldout]
        fit_rows = [r for r in uniq_all if r["dataset"] != heldout]
        val_rows = [r for r in rows if r["dataset"] == heldout]
        if len(fit_rows) < 8 or not val_rows:
            continue
        alpha = choose_alpha(train_rows)
        model = fit_ridge(
            np.stack([r["feature"] for r in fit_rows]),
            np.stack([r["target_resid"] for r in fit_rows]),
            alpha,
        )
        x_val = np.stack([r["feature"] for r in val_rows])
        pred = predict(model, x_val)
        predictions.extend(score_predictions(val_rows, pred, "ridge_residual"))

        rng = np.random.default_rng(20260628 + len(heldout))
        shuffled_fit = list(fit_rows)
        y = np.stack([r["target_resid"] for r in shuffled_fit])
        y = y[rng.permutation(y.shape[0])]
        shuf_model = fit_ridge(np.stack([r["feature"] for r in shuffled_fit]), y, alpha)
        shuf_pred = predict(shuf_model, x_val)
        predictions.extend(score_predictions(val_rows, shuf_pred, "gene_shuffle_control"))
        predictions.extend(score_predictions(val_rows, -pred, "residual_signflip_control"))
        selected.append({"seed": seed, "heldout_dataset": heldout, "alpha": alpha, "n_train_unique": len(fit_rows), "n_val": len(val_rows)})
    return predictions, selected


def main() -> None:
    emb, idx = load_gene_embeddings()
    genes = load_condition_genes()
    seed_results = {}
    all_pred_rows = []
    all_selected = []
    for seed, path in SEED_FILES.items():
        rows = load_rows(path, seed, emb, idx, genes)
        preds, selected = lodo(seed, rows)
        seed_results[seed] = {"summary": summarize(preds), "selected": selected}
        all_pred_rows.extend(preds)
        all_selected.extend(selected)

    reasons = []
    for seed, obj in seed_results.items():
        for group in GROUPS:
            key = f"{group}|ridge_residual"
            s = obj["summary"].get(key)
            shuf = obj["summary"].get(f"{group}|gene_shuffle_control", {})
            if not s:
                reasons.append(f"{seed}_{group}_missing")
                continue
            if s["dominance_pp"] < 0.015:
                reasons.append(f"{seed}_{group}_dominance_lt_0p015")
            if s["ci_low"] <= 0:
                reasons.append(f"{seed}_{group}_ci_low_not_gt0")
            if s["dataset_min"] < -0.005:
                reasons.append(f"{seed}_{group}_dataset_min_lt_minus0p005")
            if shuf and shuf["dominance_pp"] >= s["dominance_pp"] - 0.003:
                reasons.append(f"{seed}_{group}_shuffle_too_close")
    reasons.append("real_mmd_not_computed_cpu_mean_gate_only")
    status = "source_anchored_residual_cpu_gate_fail_no_gpu"
    if reasons == ["real_mmd_not_computed_cpu_mean_gate_only"]:
        status = "source_anchored_residual_cpu_gate_pass_needs_mmd_no_gpu"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_csv = OUT_DIR / "source_anchored_residual_lodo_rows.csv"
    with rows_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = ["seed", "group", "dataset", "condition", "mode", "anchor_pp", "ctrl_pp", "candidate_pp", "dominance_pp"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in all_pred_rows])
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_mean_space_only": True,
            "trainonly_internal_lodo": True,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "explicit_tracka_rows_used_for_selection": False,
            "real_mmd_computed": False,
        },
        "seed_results": seed_results,
        "selected": all_selected,
        "decision_reasons": sorted(set(reasons)),
        "outputs": {"rows_csv": str(rows_csv)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Source-Anchored Residual CPU Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only mean-space feasibility gate. It fits a ridge predictor for `GT_mean - ctrl_mean` from scGPT gene embeddings using train-only internal leave-one-dataset-out rows. Explicit Track A rows, canonical multi, and Track C query are not used for selection.",
        "",
        "## Results",
        "",
        "| seed | group | mode | n | candidate pp | anchor pp | ctrl pp | dominance | CI low | dataset min | positive rows |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, obj in sorted(seed_results.items()):
        for group in GROUPS:
            for mode in ("ridge_residual", "gene_shuffle_control", "residual_signflip_control"):
                s = obj["summary"].get(f"{group}|{mode}")
                if not s:
                    continue
                lines.append(
                    f"| `{seed}` | `{group}` | `{mode}` | {s['n']} | {s['candidate_pp']:+.6f} | "
                    f"{s['anchor_pp']:+.6f} | {s['ctrl_pp']:+.6f} | {s['dominance_pp']:+.6f} | "
                    f"{s['ci_low']:+.6f} | {s['dataset_min']:+.6f} | {s['row_positive_fraction']:.3f} |"
                )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in sorted(set(reasons)))
    lines.extend(["", "## Decision", "", "Do not launch GPU from this source-anchored residual route unless a future gate beats both anchor and source/control and computes real MMD/no-harm. This CPU gate is mean-space only.", "", "## Outputs", "", f"- JSON: `{OUT_JSON}`", f"- rows: `{rows_csv}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
