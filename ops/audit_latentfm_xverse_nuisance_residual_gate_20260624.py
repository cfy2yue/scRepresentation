#!/usr/bin/env python3
"""CPU gate for nuisance-invariant residual training.

Reads train-only/internal condition-mean artifacts prepared by
``latentfm_xverse_nuisance_condition_means_20260624``. It does not train, read
canonical split outputs, or inspect Track C query files.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
IN_DIR = ROOT / "reports/latentfm_xverse_nuisance_condition_means_20260624"
OUT_JSON = ROOT / "reports/latentfm_xverse_nuisance_residual_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_NUISANCE_RESIDUAL_GATE_20260624.md"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _as_vec(row: dict[str, Any], key: str) -> np.ndarray:
    val = row.get(key)
    if val is None:
        raise ValueError(f"missing vector key={key!r} for {row.get('dataset')}::{row.get('condition')}")
    return np.asarray(val, dtype=np.float32)


def _align_rows(anchor_group: dict[str, Any], cap120_group: dict[str, Any]) -> list[dict[str, Any]]:
    anchor = {
        (str(r["dataset"]), str(r["condition"])): r
        for r in anchor_group.get("condition_metrics", [])
    }
    rows: list[dict[str, Any]] = []
    for cr in cap120_group.get("condition_metrics", []):
        key = (str(cr["dataset"]), str(cr["condition"]))
        ar = anchor.get(key)
        if ar is None:
            continue
        cap_pred = _as_vec(cr, "pred_mean")
        anc_pred = _as_vec(ar, "pred_mean")
        gt = _as_vec(cr, "gt_mean")
        cap_err = cap_pred - gt
        anc_err = anc_pred - gt
        shift = cap_err - anc_err
        rows.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "mmd_delta": float(cr["test_mmd_clamped"]) - float(ar["test_mmd_clamped"]),
                "pp_delta": float(cr["pearson_pert"]) - float(ar["pearson_pert"]),
                "shift_norm": float(np.linalg.norm(shift)),
                "cap_error_norm": float(np.linalg.norm(cap_err)),
                "anchor_error_norm": float(np.linalg.norm(anc_err)),
                "shift": shift.astype(np.float32),
            }
        )
    return rows


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def _cosine_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def _loo_centroid_accuracy(x: np.ndarray, labels: list[str], *, min_class_n: int = 3) -> dict[str, Any]:
    counts = Counter(labels)
    keep = [i for i, lab in enumerate(labels) if counts[lab] >= min_class_n]
    if len(keep) < 10:
        return {"accuracy": float("nan"), "n": len(keep), "n_classes": 0}
    xk = _cosine_normalize(_standardize(x[keep]))
    labs = [labels[i] for i in keep]
    classes = sorted(set(labs))
    correct = 0
    used = 0
    for idx, lab in enumerate(labs):
        scores: dict[str, float] = {}
        for cls in classes:
            cls_idx = [j for j, y in enumerate(labs) if y == cls and j != idx]
            if not cls_idx:
                continue
            centroid = xk[cls_idx].mean(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
            scores[cls] = float(xk[idx] @ centroid)
        if not scores:
            continue
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        correct += int(pred == lab)
        used += 1
    return {
        "accuracy": correct / max(used, 1),
        "n": used,
        "n_classes": len(classes),
        "class_counts": dict(sorted(Counter(labs).items())),
    }


def _shuffle_accuracy(x: np.ndarray, labels: list[str], *, seed: int = 42, n_perm: int = 200) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    vals: list[float] = []
    labs = np.asarray(labels, dtype=object)
    for _ in range(n_perm):
        shuffled = labs.copy()
        rng.shuffle(shuffled)
        acc = _loo_centroid_accuracy(x, [str(v) for v in shuffled]).get("accuracy")
        if acc is not None and math.isfinite(float(acc)):
            vals.append(float(acc))
    if not vals:
        return {"mean": float("nan"), "p95": float("nan"), "std": float("nan")}
    return {
        "mean": float(statistics.fmean(vals)),
        "p95": float(np.quantile(vals, 0.95)),
        "std": float(statistics.pstdev(vals)),
    }


def _rankdata(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum() * (y * y).sum())) + 1e-12
    return float((x * y).sum() / den)


def _spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_rankdata(a), _rankdata(b))


def _by_dataset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[str(row["dataset"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for ds, rs in sorted(by.items()):
        out[ds] = {
            "n": len(rs),
            "mmd_delta_mean": float(statistics.fmean(float(r["mmd_delta"]) for r in rs)),
            "pp_delta_mean": float(statistics.fmean(float(r["pp_delta"]) for r in rs)),
            "shift_norm_mean": float(statistics.fmean(float(r["shift_norm"]) for r in rs)),
            "cap_error_norm_mean": float(statistics.fmean(float(r["cap_error_norm"]) for r in rs)),
            "anchor_error_norm_mean": float(statistics.fmean(float(r["anchor_error_norm"]) for r in rs)),
        }
    return out


def _gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.stack([np.asarray(r["shift"], dtype=np.float32) for r in rows], axis=0)
    labels = [str(r["dataset"]) for r in rows]
    acc = _loo_centroid_accuracy(x, labels)
    shuf = _shuffle_accuracy(x, labels)
    by_ds = _by_dataset(rows)
    ds_rows = [v for v in by_ds.values() if int(v["n"]) >= 3]
    mmd = [float(v["mmd_delta_mean"]) for v in ds_rows]
    norms = [float(v["shift_norm_mean"]) for v in ds_rows]
    rho = _spearman(norms, mmd)
    top_mmd = {
        ds for ds, v in sorted(by_ds.items(), key=lambda kv: float(kv[1]["mmd_delta_mean"]), reverse=True)[:5]
    }
    top_norm = {
        ds for ds, v in sorted(by_ds.items(), key=lambda kv: float(kv[1]["shift_norm_mean"]), reverse=True)[:5]
    }
    overlap = len(top_mmd & top_norm)
    reasons: list[str] = []
    if len(rows) < 100:
        reasons.append("too_few_condition_rows")
    acc_val = float(acc.get("accuracy", float("nan")))
    if not math.isfinite(acc_val):
        reasons.append("dataset_predictability_not_estimable")
    else:
        threshold = max(0.25, float(shuf.get("p95", 0.0)) + 0.05)
        if acc_val < threshold:
            reasons.append("residual_shift_dataset_predictability_weak")
    if not math.isfinite(rho) or rho < 0.30:
        reasons.append("residual_shift_not_correlated_with_mmd_harm")
    if overlap < 2:
        reasons.append("top_mmd_harm_not_aligned_with_top_residual_shift")
    status = "nuisance_residual_gate_pass_no_gpu" if not reasons else "nuisance_residual_gate_fail_no_gpu"
    return {
        "status": status,
        "action": (
            "design_conservative_nuisance_invariant_smoke"
            if status.endswith("pass_no_gpu")
            else "do_not_launch_nuisance_invariant_smoke"
        ),
        "reasons": reasons,
        "dataset_predictability": acc,
        "shuffle_predictability": shuf,
        "mmd_shift_spearman": rho,
        "top5_mmd_harm_overlap_top5_shift_norm": overlap,
        "top5_mmd_harm": sorted(top_mmd),
        "top5_shift_norm": sorted(top_norm),
    }


def main() -> int:
    manifest = _load(IN_DIR / "manifest.json")
    if manifest.get("status") != "condition_means_ready_for_nuisance_gate":
        raise RuntimeError(f"condition means manifest is not ready: {manifest.get('status')!r}")
    anchor = _load(IN_DIR / "condition_family_eval_anchor_internal_means_ode20.json")
    cap120 = _load(IN_DIR / "condition_family_eval_cap120_internal_means_ode20.json")
    rows = _align_rows(anchor["groups"]["family_gene"], cap120["groups"]["family_gene"])
    by_ds = _by_dataset(rows)
    decision = _gate(rows)
    payload = {
        "decision": decision,
        "boundary": {
            "source": str(IN_DIR),
            "group": "family_gene",
            "canonical_or_query_used": False,
        },
        "n_rows": len(rows),
        "by_dataset": by_ds,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    top = sorted(by_ds.items(), key=lambda kv: float(kv[1]["mmd_delta_mean"]), reverse=True)[:10]
    lines = [
        "# LatentFM xverse Nuisance Residual Gate",
        "",
        "## Boundary",
        "",
        "- Reads train-only/internal condition-mean artifacts only.",
        "- Does not read canonical split, Track C query, or active training logs.",
        "- Tests whether cap120-vs-anchor residual shifts carry dataset nuisance signal aligned with MMD harm.",
        "",
        "## Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Key Metrics",
        "",
        f"- rows: `{len(rows)}`",
        f"- LOO dataset centroid accuracy: `{decision['dataset_predictability'].get('accuracy')}`",
        f"- shuffled p95 accuracy: `{decision['shuffle_predictability'].get('p95')}`",
        f"- MMD vs residual-shift Spearman: `{decision['mmd_shift_spearman']}`",
        f"- top5 MMD/shift overlap: `{decision['top5_mmd_harm_overlap_top5_shift_norm']}`",
        "",
        "## Top MMD-Harm Datasets",
        "",
        "| dataset | n | pp delta | MMD delta | shift norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, row in top:
        lines.append(
            f"| {ds} | {int(row['n'])} | {float(row['pp_delta_mean']):+.6f} | "
            f"{float(row['mmd_delta_mean']):+.6f} | {float(row['shift_norm_mean']):.6f} |"
        )
    if decision["reasons"]:
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend(f"- `{r}`" for r in decision["reasons"])
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "action": decision["action"], "out_md": str(OUT_MD)}, indent=2))
    return 0 if decision["status"].endswith("pass_no_gpu") else 1


if __name__ == "__main__":
    raise SystemExit(main())
