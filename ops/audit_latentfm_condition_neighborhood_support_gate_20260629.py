#!/usr/bin/env python3
"""Condition-level response-neighborhood support gate.

CPU/report-only. This is the next ZSCAPE-inspired translation attempt after
split-level proxies failed: use train-only residual vectors to ask whether
conditions have cross-dataset/local-neighborhood support that could support a
future matched high/low information split.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

for _key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_key, "4")

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from materialize_latentfm_trainonly_condition_residual_information_20260628 import (  # noqa: E402
    collect_needed_conditions,
    load_condition_vector_cache,
    load_json,
)


PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
PARENT_SPLIT_NAME = "split_seed42_xverse_trainonly_scaling_cap120_all_v2"
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
COND_META = DATA_DIR / "condition_metadata.json"
STATE_CONTEXT = ROOT / "reports/state_context_condition_matrix_20260628/state_context_condition_matrix.csv"
EXACT_CONDITION_ROWS = ROOT / "reports/exact_response_information_parent_train_complete_20260628/exact_response_information_condition_rows.csv"
OUT_DIR = ROOT / "reports/condition_neighborhood_support_gate_20260629"
OUT_MD = OUT_DIR / "LATENTFM_CONDITION_NEIGHBORHOOD_SUPPORT_GATE_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_condition_neighborhood_support_gate_20260629.json"
OUT_ROWS = OUT_DIR / "condition_neighborhood_support_rows.csv"
OUT_PAIRS = OUT_DIR / "condition_neighborhood_support_matched_pairs.csv"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def effective_count(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    probs = [count / total for count in counts.values()]
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    return float(math.exp(ent))


def target_key(meta: dict[str, Any], condition: str) -> str:
    genes = [str(g).upper() for g in meta.get("genes", []) if str(g)]
    if genes:
        return "gene:" + "|".join(sorted(genes))
    chem = str(meta.get("chem_obs_value") or condition).strip().lower()
    return "chem:" + chem


def load_state_context() -> pd.DataFrame:
    if not STATE_CONTEXT.exists():
        return pd.DataFrame()
    df = pd.read_csv(STATE_CONTEXT)
    cols = [
        "dataset",
        "condition",
        "split_role",
        "n_state_keys",
        "n_context_keys",
        "max_state_unique",
        "max_state_entropy",
        "max_context_unique",
        "has_state_context_signal",
    ]
    return df[[col for col in cols if col in df.columns]].drop_duplicates(["dataset", "condition"], keep="first")


def load_exact_set() -> set[tuple[str, str]]:
    if not EXACT_CONDITION_ROWS.exists():
        return set()
    df = pd.read_csv(EXACT_CONDITION_ROWS)
    return set(zip(df["dataset"].astype(str), df["condition"].astype(str)))


def collect_parent_vectors() -> tuple[pd.DataFrame, np.ndarray]:
    split_row = {
        "split_file": str(PARENT_SPLIT.relative_to(ROOT)),
        "split_name": PARENT_SPLIT_NAME,
        "n_train_conditions": "",
    }
    needed = collect_needed_conditions([split_row])
    cache, missing = load_condition_vector_cache(DATA_DIR, needed)
    if missing:
        raise RuntimeError(f"missing parent train vectors: {len(missing)}")
    cond_meta = load_json(COND_META)
    split = load_json(PARENT_SPLIT)
    exact_set = load_exact_set()
    state_context = load_state_context()
    state_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if not state_context.empty:
        for _, row in state_context.iterrows():
            state_lookup[(str(row["dataset"]), str(row["condition"]))] = row.to_dict()

    rows: list[dict[str, Any]] = []
    residuals: list[np.ndarray] = []
    for dataset, groups in split.items():
        for condition in groups.get("train", []):
            condition_s = str(condition)
            vectors = cache.get(dataset, {}).get(condition_s)
            if vectors is None:
                continue
            meta = cond_meta.get(dataset, {}).get(condition_s, {})
            state = state_lookup.get((dataset, condition_s), {})
            residual = vectors.residual.astype(np.float64)
            residuals.append(residual)
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition_s,
                    "perturbation_type_raw": str(meta.get("perturbation_type_raw", "unknown")),
                    "target_key": target_key(meta, condition_s),
                    "n_ctrl": int(vectors.n_ctrl),
                    "n_gt": int(vectors.n_gt),
                    "response_norm": float(np.linalg.norm(residual)),
                    "exact_response_available": (dataset, condition_s) in exact_set,
                    "n_state_keys": int(float(state.get("n_state_keys", 0) or 0)),
                    "n_context_keys": int(float(state.get("n_context_keys", 0) or 0)),
                    "max_state_entropy": float(state.get("max_state_entropy", 0.0) or 0.0),
                    "has_state_context_signal": str(state.get("has_state_context_signal", "")).lower() in {"true", "1", "yes"},
                }
            )
    if not residuals:
        raise RuntimeError("no residuals collected for parent train split")
    return pd.DataFrame(rows), np.vstack(residuals)


def add_neighbor_metrics(rows: pd.DataFrame, residuals: np.ndarray, top_k: int = 20) -> pd.DataFrame:
    x = residuals.astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    z = (x - mean) / std
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    zn = z / np.maximum(norms, 1e-12)
    cosine = zn @ zn.T
    np.fill_diagonal(cosine, -np.inf)
    n = cosine.shape[0]
    k = min(top_k, max(1, n - 1))
    top_idx_unsorted = np.argpartition(-cosine, kth=k - 1, axis=1)[:, :k]
    top_cos_unsorted = np.take_along_axis(cosine, top_idx_unsorted, axis=1)
    order = np.argsort(-top_cos_unsorted, axis=1)
    top_idx = np.take_along_axis(top_idx_unsorted, order, axis=1)
    top_cos = np.take_along_axis(top_cos_unsorted, order, axis=1)

    datasets = rows["dataset"].astype(str).to_numpy()
    ptypes = rows["perturbation_type_raw"].astype(str).to_numpy()
    targets = rows["target_key"].astype(str).to_numpy()
    all_same_target_cross = []
    metrics: list[dict[str, Any]] = []
    for i in range(n):
        neigh = top_idx[i]
        neigh_cos = top_cos[i]
        cross_mask = datasets[neigh] != datasets[i]
        same_dataset_frac = float(1.0 - np.mean(cross_mask)) if len(neigh) else 1.0
        cross_idx = neigh[cross_mask]
        cross_cos = neigh_cos[cross_mask]
        same_target_cross_mask = (targets != targets[i]) * False
        total_same_target_cross = int(((targets == targets[i]) & (datasets != datasets[i])).sum())
        all_same_target_cross.append(total_same_target_cross)
        same_target_top = int(((targets[neigh] == targets[i]) & cross_mask).sum())
        same_ptype_top = int(((ptypes[neigh] == ptypes[i]) & cross_mask).sum())
        top5_cross = cross_cos[:5] if len(cross_cos) >= 5 else cross_cos
        support_score = (
            float(len(cross_idx)) / max(float(k), 1.0)
            + max(float(cross_cos[0]), 0.0) if len(cross_cos) else 0.0
        )
        support_score += math.log1p(same_target_top) * 0.25
        support_score -= same_dataset_frac * 0.50
        metrics.append(
            {
                "topk": int(k),
                "same_dataset_fraction_top20": same_dataset_frac,
                "cross_dataset_neighbor_count_top20": int(len(cross_idx)),
                "cross_dataset_effective_count_top20": effective_count(list(datasets[cross_idx])),
                "same_ptype_cross_dataset_neighbor_count_top20": same_ptype_top,
                "same_target_cross_dataset_neighbor_count_top20": same_target_top,
                "same_target_cross_dataset_total": total_same_target_cross,
                "best_cross_dataset_cosine": float(cross_cos[0]) if len(cross_cos) else float("nan"),
                "mean_top5_cross_dataset_cosine": float(np.mean(top5_cross)) if len(top5_cross) else float("nan"),
                "mean_top20_cross_dataset_cosine": float(np.mean(cross_cos)) if len(cross_cos) else float("nan"),
                "neighbor_support_score": support_score,
                "nearest_neighbor_dataset": str(datasets[neigh[0]]) if len(neigh) else "",
                "nearest_neighbor_condition": str(rows.iloc[int(neigh[0])]["condition"]) if len(neigh) else "",
                "nearest_neighbor_cosine": float(neigh_cos[0]) if len(neigh_cos) else float("nan"),
            }
        )
    out = pd.concat([rows.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    out["support_quantile"] = out["neighbor_support_score"].rank(method="average", pct=True)
    out["support_class"] = "middle"
    out.loc[out["support_quantile"] >= 0.75, "support_class"] = "high"
    out.loc[out["support_quantile"] <= 0.25, "support_class"] = "low"
    return out


def build_matched_pairs(rows: pd.DataFrame, max_pairs: int = 50000) -> pd.DataFrame:
    high = rows[rows["support_class"].eq("high")].copy()
    low = rows[rows["support_class"].eq("low")].copy()
    if high.empty or low.empty:
        return pd.DataFrame()
    high["response_bin"] = pd.qcut(high["response_norm"].rank(method="first"), q=4, labels=False, duplicates="drop")
    low["response_bin"] = pd.qcut(low["response_norm"].rank(method="first"), q=4, labels=False, duplicates="drop")
    pair_rows: list[dict[str, Any]] = []
    for _, h in high.iterrows():
        candidates = low[
            low["perturbation_type_raw"].astype(str).eq(str(h["perturbation_type_raw"]))
            & low["response_bin"].eq(h["response_bin"])
            & low["dataset"].astype(str).ne(str(h["dataset"]))
        ].copy()
        if candidates.empty:
            continue
        candidates["log_cell_diff"] = (
            np.log1p(pd.to_numeric(candidates["n_gt"], errors="coerce"))
            - math.log1p(float(h["n_gt"]))
        ).abs()
        candidates = candidates[candidates["log_cell_diff"] <= 1.0]
        if candidates.empty:
            continue
        candidates["score_gap"] = float(h["neighbor_support_score"]) - pd.to_numeric(candidates["neighbor_support_score"], errors="coerce")
        candidates = candidates[candidates["score_gap"] > 0]
        candidates = candidates.sort_values(["score_gap", "log_cell_diff"], ascending=[False, True]).head(20)
        for _, l in candidates.iterrows():
            pair_rows.append(
                {
                    "high_dataset": h["dataset"],
                    "high_condition": h["condition"],
                    "low_dataset": l["dataset"],
                    "low_condition": l["condition"],
                    "perturbation_type_raw": h["perturbation_type_raw"],
                    "response_bin": int(h["response_bin"]) if pd.notna(h["response_bin"]) else -1,
                    "high_support_score": float(h["neighbor_support_score"]),
                    "low_support_score": float(l["neighbor_support_score"]),
                    "score_gap": float(h["neighbor_support_score"] - l["neighbor_support_score"]),
                    "high_cross_neighbors": int(h["cross_dataset_neighbor_count_top20"]),
                    "low_cross_neighbors": int(l["cross_dataset_neighbor_count_top20"]),
                    "high_same_dataset_fraction": float(h["same_dataset_fraction_top20"]),
                    "low_same_dataset_fraction": float(l["same_dataset_fraction_top20"]),
                    "log_cell_diff": float(l["log_cell_diff"]),
                }
            )
            if len(pair_rows) >= max_pairs:
                return pd.DataFrame(pair_rows)
    return pd.DataFrame(pair_rows)


def decide(rows: pd.DataFrame, pairs: pd.DataFrame) -> tuple[str, list[str], dict[str, Any]]:
    high = rows[rows["support_class"].eq("high")]
    low = rows[rows["support_class"].eq("low")]
    pair_datasets: set[str] = set()
    pair_ptypes: set[str] = set()
    if not pairs.empty:
        pair_datasets.update(pairs["high_dataset"].astype(str))
        pair_datasets.update(pairs["low_dataset"].astype(str))
        pair_ptypes.update(pairs["perturbation_type_raw"].astype(str))
    summary = {
        "n_parent_conditions": int(len(rows)),
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "n_matched_pairs": int(len(pairs)),
        "n_pair_datasets": int(len(pair_datasets)),
        "n_pair_perturbation_types": int(len(pair_ptypes)),
        "high_median_cross_neighbors": float(high["cross_dataset_neighbor_count_top20"].median()) if not high.empty else 0.0,
        "high_median_same_dataset_fraction": float(high["same_dataset_fraction_top20"].median()) if not high.empty else 1.0,
        "median_score_gap": float(pairs["score_gap"].median()) if not pairs.empty else 0.0,
    }
    reasons: list[str] = []
    if summary["n_parent_conditions"] < 1000:
        reasons.append("parent_condition_count_below_1000")
    if summary["n_high"] < 150 or summary["n_low"] < 150:
        reasons.append("high_low_support_bins_below_150_each")
    if summary["n_matched_pairs"] < 300:
        reasons.append("matched_pairs_below_300")
    if summary["n_pair_datasets"] < 15:
        reasons.append("matched_pair_datasets_below_15")
    if summary["n_pair_perturbation_types"] < 2:
        reasons.append("matched_pair_perturbation_types_below_2")
    if summary["high_median_cross_neighbors"] < 5:
        reasons.append("high_support_cross_neighbors_below_5")
    if summary["high_median_same_dataset_fraction"] > 0.80:
        reasons.append("high_support_dataset_dominated")
    if summary["median_score_gap"] < 0.50:
        reasons.append("median_score_gap_below_0p50")
    status = (
        "condition_neighborhood_support_gate_pass_prepare_split_audit_no_gpu"
        if not reasons
        else "condition_neighborhood_support_gate_no_matched_split_no_gpu"
    )
    return status, reasons, summary


def write_report(rows: pd.DataFrame, pairs: pd.DataFrame, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    top_high = rows.sort_values("neighbor_support_score", ascending=False).head(10)
    top_low = rows.sort_values("neighbor_support_score", ascending=True).head(10)
    ds_summary = (
        rows.groupby("dataset")
        .agg(
            n=("condition", "size"),
            median_support=("neighbor_support_score", "median"),
            median_cross_neighbors=("cross_dataset_neighbor_count_top20", "median"),
            median_same_dataset_fraction=("same_dataset_fraction_top20", "median"),
        )
        .reset_index()
        .sort_values("median_support", ascending=False)
        .head(12)
    )
    lines = [
        "# LatentFM Condition Neighborhood Support Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only over parent train split residual vectors in existing xVERSE latent bundles.",
        "* Uses only train conditions from `split_seed42_xverse_trainonly_scaling_cap120_all_v2`.",
        "* No training, inference, new OT pairing, GPU, canonical multi, Track C query, or checkpoint selection.",
        "* This is a feasibility gate for a future prospective high/low support split, not a promotion result.",
        "",
        "## Gate Summary",
        "",
        f"* Parent train conditions: `{summary['n_parent_conditions']}`.",
        f"* High/low support bins: `{summary['n_high']}` / `{summary['n_low']}`.",
        f"* Matched high-low pairs: `{summary['n_matched_pairs']}` across `{summary['n_pair_datasets']}` datasets and `{summary['n_pair_perturbation_types']}` perturbation types.",
        f"* High median cross-dataset top20 neighbors: `{fmt(summary['high_median_cross_neighbors'])}`.",
        f"* High median same-dataset top20 fraction: `{fmt(summary['high_median_same_dataset_fraction'])}`.",
        f"* Median matched-pair score gap: `{fmt(summary['median_score_gap'])}`.",
        f"* Blockers/reasons: `{'; '.join(payload['reasons']) if payload['reasons'] else 'none'}`.",
        "",
        "## Interpretation",
        "",
        "* This gate is the closest current LatentFM analogue to the ZSCAPE vector insight: it measures whether a condition has cross-dataset residual-neighborhood support rather than just raw cells or exact same-gene coverage.",
        "* Passing this gate does not authorize training directly; it authorizes drafting a leakage-safe prospective split and getting external audit.",
        "* Failing this gate means the current train residual space is too dataset-dominated or under-matched for a clean high/low support scaling experiment.",
        "",
        "## Top High-Support Conditions",
        "",
        "| dataset | condition | ptype | score | cross top20 | same-dataset frac | best cross cosine | exact | state entropy |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in top_high.iterrows():
        lines.append(
            f"| `{row['dataset']}` | `{row['condition']}` | `{row['perturbation_type_raw']}` | "
            f"`{fmt(row['neighbor_support_score'])}` | `{int(row['cross_dataset_neighbor_count_top20'])}` | "
            f"`{fmt(row['same_dataset_fraction_top20'])}` | `{fmt(row['best_cross_dataset_cosine'])}` | "
            f"`{bool(row['exact_response_available'])}` | `{fmt(row['max_state_entropy'])}` |"
        )
    lines.extend(
        [
            "",
            "## Top Low-Support Conditions",
            "",
            "| dataset | condition | ptype | score | cross top20 | same-dataset frac | best cross cosine | exact | state entropy |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in top_low.iterrows():
        lines.append(
            f"| `{row['dataset']}` | `{row['condition']}` | `{row['perturbation_type_raw']}` | "
            f"`{fmt(row['neighbor_support_score'])}` | `{int(row['cross_dataset_neighbor_count_top20'])}` | "
            f"`{fmt(row['same_dataset_fraction_top20'])}` | `{fmt(row['best_cross_dataset_cosine'])}` | "
            f"`{bool(row['exact_response_available'])}` | `{fmt(row['max_state_entropy'])}` |"
        )
    lines.extend(
        [
            "",
            "## Dataset Support Snapshot",
            "",
            "| dataset | n | median score | median cross top20 | median same-dataset frac |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in ds_summary.iterrows():
        lines.append(
            f"| `{row['dataset']}` | `{int(row['n'])}` | `{fmt(row['median_support'])}` | "
            f"`{fmt(row['median_cross_neighbors'])}` | `{fmt(row['median_same_dataset_fraction'])}` |"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
        ]
    )
    if payload["status"].endswith("prepare_split_audit_no_gpu"):
        lines.append(
            "* Draft a prospective high/low support split from the matched pairs, then ask an external subagent to audit leakage, confounding, and expected Track A gate before any GPU launch."
        )
    else:
        lines.append(
            "* Do not draft a GPU split from this exact support-score definition. Either refine the condition-neighborhood definition with stronger source/background controls or pivot to another biology/scaling axis."
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"* Condition rows: `{OUT_ROWS}`",
            f"* Matched pairs: `{OUT_PAIRS}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_rows, residuals = collect_parent_vectors()
    rows = add_neighbor_metrics(base_rows, residuals)
    pairs = build_matched_pairs(rows)
    status, reasons, summary = decide(rows, pairs)
    rows.to_csv(OUT_ROWS, index=False)
    pairs.to_csv(OUT_PAIRS, index=False)
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized_next": False,
        "reasons": reasons,
        "summary": summary,
        "inputs": {
            "parent_split": str(PARENT_SPLIT),
            "data_dir": str(DATA_DIR),
            "condition_metadata": str(COND_META),
            "state_context": str(STATE_CONTEXT),
            "exact_condition_rows": str(EXACT_CONDITION_ROWS),
        },
        "outputs": {
            "report": str(OUT_MD),
            "json": str(OUT_JSON),
            "condition_rows": str(OUT_ROWS),
            "matched_pairs": str(OUT_PAIRS),
        },
        "boundary": "cpu_report_only_parent_train_residual_vectors_no_training_no_inference_no_new_ot_no_gpu_no_canonical_multi_no_trackc_query",
    }
    write_json(OUT_JSON, payload)
    write_report(rows, pairs, payload)
    print(json.dumps({"status": status, "summary": summary, "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
