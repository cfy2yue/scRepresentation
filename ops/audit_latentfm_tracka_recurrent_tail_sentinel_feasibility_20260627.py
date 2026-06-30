#!/usr/bin/env python3
"""CPU feasibility gate for a Track A recurrent-tail sentinel.

The sentinel is only allowed to use train-only/internal validation rows and
non-target-derived features. Exact recurrent tails remain post-freeze audit
targets, not labels for selection.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
FORENSICS = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
EXACT_TAIL = ROOT / "reports/latentfm_tracka_recurrent_tail_gate_20260627.json"
OUT_JSON = ROOT / "reports/latentfm_tracka_recurrent_tail_sentinel_feasibility_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_RECURRENT_TAIL_SENTINEL_FEASIBILITY_20260627.md"


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with FORENSICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") not in {
                "internal_val_cross_background_seen_gene_proxy",
                "internal_val_family_gene_proxy",
            }:
                continue
            pp = fnum(row.get("anchor_pearson_pert"))
            mmd = fnum(row.get("anchor_mmd_clamped"))
            if pp is None or mmd is None:
                continue
            rows.append(
                {
                    "group": row["group"],
                    "dataset": str(row["dataset"]),
                    "condition": str(row["condition"]),
                    "gene": str(row["gene"]).strip().upper(),
                    "pp": float(pp),
                    "mmd": float(mmd),
                    "features": np.asarray(
                        [
                            np.log1p(float(row.get("gene_train_count") or 0.0)),
                            float(row.get("gene_pred_norm") or 0.0),
                            float(row.get("dataset_pred_norm") or 0.0),
                            float(row.get("global_pred_norm") or 0.0),
                            float(row.get("gene_dataset_cosine") or 0.0),
                        ],
                        dtype=float,
                    ),
                    "count_only": np.asarray([np.log1p(float(row.get("gene_train_count") or 0.0))], dtype=float),
                }
            )
    return rows


def standardize(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd


def ridge_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    xt, xv = standardize(x_train, x_test)
    xt = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
    xv = np.concatenate([np.ones((xv.shape[0], 1)), xv], axis=1)
    reg = np.eye(xt.shape[1]) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xt.T @ xt + reg, xt.T @ y_train)
    return xv @ beta


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos = s[y > 0.5]
    neg = s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def lodo(rows: list[dict[str, Any]], y: np.ndarray, feature_key: str) -> dict[str, Any]:
    datasets = sorted({r["dataset"] for r in rows})
    scores = np.zeros(len(rows), dtype=float)
    for ds in datasets:
        test = np.asarray([i for i, r in enumerate(rows) if r["dataset"] == ds], dtype=int)
        train = np.asarray([i for i, r in enumerate(rows) if r["dataset"] != ds], dtype=int)
        x_train = np.vstack([rows[i][feature_key] for i in train])
        x_test = np.vstack([rows[i][feature_key] for i in test])
        scores[test] = ridge_score(x_train, y[train], x_test, alpha=1.0)
    top = scores >= np.quantile(scores, 0.80)
    prevalence = float(y.mean())
    top_rate = float(y[top].mean()) if np.any(top) else 0.0
    by_ds = defaultdict(list)
    for i, r in enumerate(rows):
        if top[i]:
            by_ds[r["dataset"]].append(float(y[i]))
    return {
        "auroc": auroc(y, scores),
        "prevalence": prevalence,
        "top20_tail_rate": top_rate,
        "top20_enrichment": top_rate - prevalence,
        "top20_fraction": float(np.mean(top)),
        "top20_dataset_count": len(by_ds),
        "top20_dataset_min_tail_rate": float(min(np.mean(v) for v in by_ds.values())) if by_ds else 0.0,
        "scores": scores.tolist(),
    }


def shuffle_p(rows: list[dict[str, Any]], y: np.ndarray, actual_auroc: float, actual_enrichment: float) -> dict[str, float]:
    rng = np.random.default_rng(20260627)
    aucs = []
    enrich = []
    for _ in range(1000):
        yp = rng.permutation(y)
        m = lodo(rows, yp, "features")
        aucs.append(m["auroc"])
        enrich.append(m["top20_enrichment"])
    return {
        "auroc_p_ge_actual": float(np.mean(np.asarray(aucs) >= actual_auroc)),
        "enrichment_p_ge_actual": float(np.mean(np.asarray(enrich) >= actual_enrichment)),
        "shuffle_auroc_mean": float(np.mean(aucs)),
        "shuffle_enrichment_mean": float(np.mean(enrich)),
    }


def main() -> None:
    rows = load_rows()
    y = np.asarray([(r["pp"] < 0.05) or (r["mmd"] > 0.05) for r in rows], dtype=float)
    deployable = lodo(rows, y, "features")
    count_only = lodo(rows, y, "count_only")
    shuf = shuffle_p(rows, y, deployable["auroc"], deployable["top20_enrichment"])
    exact = json.loads(EXACT_TAIL.read_text(encoding="utf-8")) if EXACT_TAIL.is_file() else {}

    reasons: list[str] = []
    if deployable["auroc"] < 0.65:
        reasons.append("deployable_internal_sentinel_auroc_lt_0p65")
    if deployable["top20_enrichment"] < 0.15:
        reasons.append("top20_tail_enrichment_lt_0p15")
    if deployable["auroc"] - count_only["auroc"] < 0.05:
        reasons.append("deployable_features_do_not_beat_count_only_by_0p05")
    if shuf["auroc_p_ge_actual"] > 0.01 or shuf["enrichment_p_ge_actual"] > 0.01:
        reasons.append("shuffle_control_not_beaten")
    if deployable["top20_dataset_count"] < 3:
        reasons.append("top20_risk_set_covers_lt_3_datasets")
    reasons.append("sentinel_only_no_model_footprint_no_gpu")

    status = "tracka_recurrent_tail_sentinel_feasibility_fail_no_gpu"
    if not any(r != "sentinel_only_no_model_footprint_no_gpu" for r in reasons):
        status = "tracka_recurrent_tail_sentinel_feasibility_pass_no_gpu_infrastructure_only"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "trainonly_internal_rows_only": True,
            "exact_recurrent_tails_used_as_labels": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
        },
        "inputs": {
            "forensics": str(FORENSICS),
            "exact_tail_context": str(EXACT_TAIL),
        },
        "n_rows": len(rows),
        "n_datasets": len({r["dataset"] for r in rows}),
        "label_definition": "internal surrogate hard tail: anchor_pearson_pert < 0.05 OR anchor_mmd_clamped > 0.05",
        "deployable_features": deployable | {"scores": "<omitted>"},
        "count_only_control": count_only | {"scores": "<omitted>"},
        "shuffle_control": shuf,
        "exact_tail_context_status": exact.get("status", "missing"),
        "decision_reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Recurrent-Tail Sentinel Feasibility",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU/report-only over train-only/internal residual-forensics rows. Exact recurrent tails are context only, not labels for selection. No train/infer/checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Metrics",
        "",
        "| model | AUROC | prevalence | top20 tail rate | top20 enrichment | top20 datasets |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `deployable_features` | {deployable['auroc']:.6f} | {deployable['prevalence']:.6f} | {deployable['top20_tail_rate']:.6f} | {deployable['top20_enrichment']:.6f} | {deployable['top20_dataset_count']} |",
        f"| `count_only` | {count_only['auroc']:.6f} | {count_only['prevalence']:.6f} | {count_only['top20_tail_rate']:.6f} | {count_only['top20_enrichment']:.6f} | {count_only['top20_dataset_count']} |",
        "",
        "## Shuffle Control",
        "",
        f"- AUROC p>=actual: `{shuf['auroc_p_ge_actual']:.4f}`",
        f"- enrichment p>=actual: `{shuf['enrichment_p_ge_actual']:.4f}`",
        "",
        "## Decision Reasons",
        "",
    ]
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
