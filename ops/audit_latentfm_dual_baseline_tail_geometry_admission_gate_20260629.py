#!/usr/bin/env python3
"""Train-only dual-baseline tail-geometry admission gate.

CPU/report-only. This gate asks whether source/control-vs-GT tail geometry in
the parent training split defines a balanced high/low condition axis that could
be used for later training-set design. It does not use canonical Track A rows,
canonical multi rows, Track C query rows, checkpoints, training, inference, or
GPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from audit_latentfm_condition_neighborhood_response_residualized_support_gate_20260629 import (  # noqa: E402
    PARENT_SPLIT,
    auc_discriminability,
    load_json,
    load_rows,
    pair_direction_fraction,
    per_ptype_claims,
    side_dataset_fraction,
    smd,
    split_from_conditions,
    state_bin,
    target_availability_bin,
    write_json,
)


DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
OUT_DIR = ROOT / "reports/dual_baseline_tail_geometry_admission_gate_20260629"
SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_dual_baseline_tail_geometry_splits_20260629"
OUT_MD = OUT_DIR / "LATENTFM_DUAL_BASELINE_TAIL_GEOMETRY_ADMISSION_GATE_20260629.md"
OUT_JSON = OUT_DIR / "latentfm_dual_baseline_tail_geometry_admission_gate_20260629.json"
OUT_FEATURES = OUT_DIR / "dual_baseline_tail_geometry_feature_rows.csv"
OUT_CONFIGS = OUT_DIR / "dual_baseline_tail_geometry_config_rows.csv"
OUT_SELECTED = OUT_DIR / "dual_baseline_tail_geometry_selected_pairs.csv"
OUT_BALANCE = OUT_DIR / "dual_baseline_tail_geometry_balance.csv"
OUT_CONTROLS = OUT_DIR / "dual_baseline_tail_geometry_negative_controls.csv"

CONFIGS = [
    {
        "name": f"q{int(q * 100)}_resp{resp}_cell{cell}_supp{supp}_ds{int(ds)}",
        "quantile": q,
        "response_log_caliper": resp,
        "cell_log_caliper": cell,
        "cross_count_caliper": supp,
        "include_dataset_dummies": ds,
        "max_pairs": 320,
        "max_per_side_dataset": 50,
        "max_per_dataset_pair": 30,
    }
    for q in [0.20, 0.25, 0.30]
    for resp in [0.35, 0.50, 0.75]
    for cell in [0.50, 0.75]
    for supp in [2, 4, 8]
    for ds in [False, True]
]

SIGMAS = (0.25, 0.5, 1.0, 2.0)


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def set_thread_env(threads: int) -> None:
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ.setdefault(key, str(max(1, int(threads))))


def parent_train_keys() -> set[str]:
    split = load_json(PARENT_SPLIT)
    keys: set[str] = set()
    for dataset, groups in split.items():
        for condition in groups.get("train", []):
            keys.add(f"{dataset}||{condition}")
    return keys


def decode_conditions(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values:
        out.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return out


def sample_block(emb: h5py.Dataset, start: int, end: int, cap: int, seed_text: str) -> np.ndarray:
    n = int(end - start)
    if n <= 0:
        return np.empty((0, int(emb.shape[1])), dtype=np.float32)
    if cap > 0 and n > cap:
        rng = np.random.default_rng(stable_seed(seed_text))
        rel = np.sort(rng.choice(n, size=cap, replace=False))
        return np.asarray(emb[start + rel], dtype=np.float32)
    return np.asarray(emb[start:end], dtype=np.float32)


def sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    x_norm = np.sum(x64 * x64, axis=1, keepdims=True)
    y_norm = np.sum(y64 * y64, axis=1, keepdims=True).T
    d2 = x_norm + y_norm - 2.0 * (x64 @ y64.T)
    return np.maximum(d2, 0.0)


def rbf_mmd_biased(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    dxx = sqdist(x, x)
    dyy = sqdist(y, y)
    dxy = sqdist(x, y)
    vals: list[float] = []
    for sigma in SIGMAS:
        gamma = 1.0 / (2.0 * sigma * sigma)
        vals.append(float(np.exp(-gamma * dxx).mean() + np.exp(-gamma * dyy).mean() - 2.0 * np.exp(-gamma * dxy).mean()))
    return float(max(np.mean(vals), 0.0))


def condition_tail_features(dataset: str, condition: str, h5: h5py.File, index: dict[str, int], cap: int) -> dict[str, Any] | None:
    idx = index.get(condition)
    if idx is None:
        return None
    ctrl_key = "ctrl" if "ctrl/emb" in h5 else "ir"
    ctrl_offsets = np.asarray(h5[f"{ctrl_key}/offsets"])
    gt_offsets = np.asarray(h5["gt/offsets"])
    c0, c1 = int(ctrl_offsets[idx]), int(ctrl_offsets[idx + 1])
    g0, g1 = int(gt_offsets[idx]), int(gt_offsets[idx + 1])
    ctrl = sample_block(h5[f"{ctrl_key}/emb"], c0, c1, cap, f"ctrl|{dataset}|{condition}|{cap}")
    gt = sample_block(h5["gt/emb"], g0, g1, cap, f"gt|{dataset}|{condition}|{cap}")
    if ctrl.shape[0] == 0 or gt.shape[0] == 0:
        return None

    ctrl_mean = ctrl.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    delta = gt_mean - ctrl_mean
    response_norm = float(np.linalg.norm(delta))
    ctrl_var = np.var(ctrl, axis=0)
    gt_var = np.var(gt, axis=0)
    ctrl_var_mean = float(np.mean(ctrl_var))
    gt_var_mean = float(np.mean(gt_var))
    sem_proxy = float(math.sqrt(ctrl_var_mean / max(1, len(ctrl)) + gt_var_mean / max(1, len(gt))))
    ctrl_dist = np.linalg.norm(ctrl - ctrl_mean[None, :], axis=1)
    gt_dist_to_ctrl = np.linalg.norm(gt - ctrl_mean[None, :], axis=1)
    ctrl_tail95 = float(np.quantile(ctrl_dist, 0.95))
    gt_tail95 = float(np.quantile(gt_dist_to_ctrl, 0.95))
    ctrl_tail99 = float(np.quantile(ctrl_dist, 0.99))
    gt_tail99 = float(np.quantile(gt_dist_to_ctrl, 0.99))
    tail95_shift = float(gt_tail95 - ctrl_tail95)
    tail99_shift = float(gt_tail99 - ctrl_tail99)
    abs_var_shift = float(abs(gt_var_mean - ctrl_var_mean))
    return {
        "key": f"{dataset}||{condition}",
        "dataset": dataset,
        "condition": condition,
        "mmd_cap": int(cap),
        "n_ctrl_actual": int(c1 - c0),
        "n_gt_actual": int(g1 - g0),
        "n_ctrl_sampled": int(ctrl.shape[0]),
        "n_gt_sampled": int(gt.shape[0]),
        "ctrl_gt_mmd_proxy": rbf_mmd_biased(ctrl, gt),
        "sample_response_norm": response_norm,
        "sem_proxy": sem_proxy,
        "ctrl_var_mean": ctrl_var_mean,
        "gt_var_mean": gt_var_mean,
        "log_var_ratio": float(math.log((gt_var_mean + 1e-12) / (ctrl_var_mean + 1e-12))),
        "abs_var_shift": abs_var_shift,
        "ctrl_tail95": ctrl_tail95,
        "gt_tail95_to_ctrl": gt_tail95,
        "tail95_shift": tail95_shift,
        "ctrl_tail99": ctrl_tail99,
        "gt_tail99_to_ctrl": gt_tail99,
        "tail99_shift": tail99_shift,
        "tail_risk_ratio": float(max(tail95_shift, 0.0) / (response_norm + sem_proxy + 1e-8)),
    }


def materialize_features(rows: pd.DataFrame, data_dir: Path, cap: int, max_conditions: int) -> pd.DataFrame:
    needed = rows[["dataset", "condition", "key"]].drop_duplicates().copy()
    if max_conditions > 0:
        needed = needed.head(max_conditions).copy()
    feature_rows: list[dict[str, Any]] = []
    for dataset, part in needed.groupby("dataset", sort=True):
        h5_path = data_dir / f"{dataset}.h5"
        if not h5_path.is_file():
            continue
        with h5py.File(h5_path, "r") as h5:
            index = {condition: idx for idx, condition in enumerate(decode_conditions(np.asarray(h5["conditions"])))}
            for condition in sorted(part["condition"].astype(str).unique()):
                feats = condition_tail_features(str(dataset), str(condition), h5, index, cap)
                if feats is not None:
                    feature_rows.append(feats)
    return pd.DataFrame(feature_rows)


def add_tail_score(features: pd.DataFrame) -> pd.DataFrame:
    work = features.copy()
    for col in ["ctrl_gt_mmd_proxy", "tail95_shift", "tail99_shift", "abs_var_shift", "tail_risk_ratio"]:
        values = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work[f"{col}_pct"] = values.rank(pct=True, method="average")
    pct_cols = [
        "ctrl_gt_mmd_proxy_pct",
        "tail95_shift_pct",
        "tail99_shift_pct",
        "abs_var_shift_pct",
        "tail_risk_ratio_pct",
    ]
    work["tail_geometry_score"] = work[pct_cols].mean(axis=1)
    return work


def residualize_tail(rows: pd.DataFrame, include_dataset_dummies: bool) -> np.ndarray:
    columns: list[np.ndarray] = [
        np.ones(len(rows), dtype=float),
        rows["log_response_norm"].to_numpy(dtype=float),
        rows["log_n_gt"].to_numpy(dtype=float),
        rows["log_n_ctrl"].to_numpy(dtype=float),
        rows["exact_bool"].astype(float).to_numpy(dtype=float),
        pd.to_numeric(rows["max_state_entropy"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(rows["same_target_cross_dataset_total"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        np.log1p(pd.to_numeric(rows["cross_dataset_neighbor_count_top20"], errors="coerce").fillna(0.0)).to_numpy(dtype=float),
        pd.to_numeric(rows["cross_dataset_effective_count_top20"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(rows["same_ptype_cross_dataset_neighbor_count_top20"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        pd.to_numeric(rows["same_dataset_fraction_top20"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
        pd.to_numeric(rows["mean_top5_cross_dataset_cosine"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    ]
    for ptype in sorted(rows["perturbation_type_raw"].astype(str).unique())[1:]:
        columns.append((rows["perturbation_type_raw"].astype(str).to_numpy() == ptype).astype(float))
    if include_dataset_dummies:
        for dataset in sorted(rows["dataset"].astype(str).unique())[1:]:
            columns.append((rows["dataset"].astype(str).to_numpy() == dataset).astype(float))
    x = np.column_stack(columns).astype(float)
    y = pd.to_numeric(rows["tail_geometry_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def prepare_rows(support_rows: pd.DataFrame, features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    feats = add_tail_score(features)
    cols_to_drop = [c for c in support_rows.columns if c in feats.columns and c not in {"key", "dataset", "condition"}]
    work = support_rows.drop(columns=cols_to_drop).merge(feats, on=["key", "dataset", "condition"], how="inner")
    work = work[np.isfinite(pd.to_numeric(work["tail_geometry_score"], errors="coerce"))].copy()
    work["tail_resid_score"] = residualize_tail(work, bool(config["include_dataset_dummies"]))
    return work


def build_candidates(rows: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = prepare_rows(rows, rows.attrs["features"], config)
    high_thr = float(work["tail_resid_score"].quantile(1.0 - float(config["quantile"])))
    low_thr = float(work["tail_resid_score"].quantile(float(config["quantile"])))
    high = work[work["tail_resid_score"] >= high_thr].copy()
    low = work[work["tail_resid_score"] <= low_thr].copy()
    candidate_rows: list[dict[str, Any]] = []
    for _, h in high.iterrows():
        sub = low[
            low["perturbation_type_raw"].astype(str).eq(str(h["perturbation_type_raw"]))
            & low["exact_bool"].eq(bool(h["exact_bool"]))
            & low["state_bin"].astype(str).eq(str(h["state_bin"]))
            & low["target_availability_bin"].astype(str).eq(str(h["target_availability_bin"]))
            & low["dataset"].astype(str).ne(str(h["dataset"]))
        ].copy()
        if sub.empty:
            continue
        sub["response_log_abs_diff"] = (sub["log_response_norm"] - float(h["log_response_norm"])).abs()
        sub["n_gt_log_abs_diff"] = (sub["log_n_gt"] - float(h["log_n_gt"])).abs()
        sub["n_ctrl_log_abs_diff"] = (sub["log_n_ctrl"] - float(h["log_n_ctrl"])).abs()
        sub["cross_count_abs_diff"] = (
            pd.to_numeric(sub["cross_dataset_neighbor_count_top20"], errors="coerce")
            - float(h["cross_dataset_neighbor_count_top20"])
        ).abs()
        sub = sub[
            (sub["response_log_abs_diff"] <= float(config["response_log_caliper"]))
            & (sub["n_gt_log_abs_diff"] <= float(config["cell_log_caliper"]))
            & (sub["n_ctrl_log_abs_diff"] <= float(config["cell_log_caliper"]))
            & (sub["cross_count_abs_diff"] <= float(config["cross_count_caliper"]))
        ].copy()
        if sub.empty:
            continue
        for _, l in sub.iterrows():
            resid_gap = float(h["tail_resid_score"] - l["tail_resid_score"])
            raw_gap = float(h["tail_geometry_score"] - l["tail_geometry_score"])
            if resid_gap <= 0:
                continue
            candidate_rows.append(
                {
                    "high_key": h["key"],
                    "low_key": l["key"],
                    "high_dataset": h["dataset"],
                    "low_dataset": l["dataset"],
                    "high_condition": h["condition"],
                    "low_condition": l["condition"],
                    "perturbation_type_raw": h["perturbation_type_raw"],
                    "high_tail_geometry_score": float(h["tail_geometry_score"]),
                    "low_tail_geometry_score": float(l["tail_geometry_score"]),
                    "tail_geometry_gap": raw_gap,
                    "high_tail_resid_score": float(h["tail_resid_score"]),
                    "low_tail_resid_score": float(l["tail_resid_score"]),
                    "residual_score_gap": resid_gap,
                    "high_ctrl_gt_mmd_proxy": float(h["ctrl_gt_mmd_proxy"]),
                    "low_ctrl_gt_mmd_proxy": float(l["ctrl_gt_mmd_proxy"]),
                    "high_tail95_shift": float(h["tail95_shift"]),
                    "low_tail95_shift": float(l["tail95_shift"]),
                    "high_tail99_shift": float(h["tail99_shift"]),
                    "low_tail99_shift": float(l["tail99_shift"]),
                    "high_abs_var_shift": float(h["abs_var_shift"]),
                    "low_abs_var_shift": float(l["abs_var_shift"]),
                    "high_tail_risk_ratio": float(h["tail_risk_ratio"]),
                    "low_tail_risk_ratio": float(l["tail_risk_ratio"]),
                    "high_directional_alignment": float(h["mean_top5_cross_dataset_cosine"]),
                    "low_directional_alignment": float(l["mean_top5_cross_dataset_cosine"]),
                    "high_cross_count": int(h["cross_dataset_neighbor_count_top20"]),
                    "low_cross_count": int(l["cross_dataset_neighbor_count_top20"]),
                    "high_cross_effective": float(h["cross_dataset_effective_count_top20"]),
                    "low_cross_effective": float(l["cross_dataset_effective_count_top20"]),
                    "high_raw_support_score": float(h["neighbor_support_score"]),
                    "low_raw_support_score": float(l["neighbor_support_score"]),
                    "high_response_norm": float(h["response_norm"]),
                    "low_response_norm": float(l["response_norm"]),
                    "high_n_gt": int(h["n_gt"]),
                    "low_n_gt": int(l["n_gt"]),
                    "high_n_ctrl": int(h["n_ctrl"]),
                    "low_n_ctrl": int(l["n_ctrl"]),
                    "high_max_state_entropy": float(h["max_state_entropy"]),
                    "low_max_state_entropy": float(l["max_state_entropy"]),
                    "high_same_target_cross_dataset_total": float(h["same_target_cross_dataset_total"]),
                    "low_same_target_cross_dataset_total": float(l["same_target_cross_dataset_total"]),
                    "high_exact_bool": bool(h["exact_bool"]),
                    "low_exact_bool": bool(l["exact_bool"]),
                    "response_log_abs_diff": float(l["response_log_abs_diff"]),
                    "n_gt_log_abs_diff": float(l["n_gt_log_abs_diff"]),
                    "n_ctrl_log_abs_diff": float(l["n_ctrl_log_abs_diff"]),
                    "cross_count_abs_diff": float(l["cross_count_abs_diff"]),
                    "dataset_pair": f"{h['dataset']}->{l['dataset']}",
                }
            )
    return pd.DataFrame(candidate_rows), work


def greedy_select(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    work = candidates.copy()
    work["objective"] = (
        pd.to_numeric(work["residual_score_gap"], errors="coerce")
        + 0.5 * pd.to_numeric(work["tail_geometry_gap"], errors="coerce")
        - 0.25 * pd.to_numeric(work["response_log_abs_diff"], errors="coerce")
        - 0.10 * pd.to_numeric(work["n_gt_log_abs_diff"], errors="coerce")
        - 0.10 * pd.to_numeric(work["n_ctrl_log_abs_diff"], errors="coerce")
        - 0.03 * pd.to_numeric(work["cross_count_abs_diff"], errors="coerce")
    )
    high_used: set[str] = set()
    low_used: set[str] = set()
    high_ds: Counter[str] = Counter()
    low_ds: Counter[str] = Counter()
    pair_dir: Counter[str] = Counter()
    selected: list[pd.Series] = []
    for _, row in work.sort_values(["objective", "residual_score_gap"], ascending=[False, False]).iterrows():
        high_key = str(row["high_key"])
        low_key = str(row["low_key"])
        hds = str(row["high_dataset"])
        lds = str(row["low_dataset"])
        direction = str(row["dataset_pair"])
        if high_key in high_used or low_key in low_used:
            continue
        if high_ds[hds] >= int(config["max_per_side_dataset"]):
            continue
        if low_ds[lds] >= int(config["max_per_side_dataset"]):
            continue
        if pair_dir[direction] >= int(config["max_per_dataset_pair"]):
            continue
        high_used.add(high_key)
        low_used.add(low_key)
        high_ds[hds] += 1
        low_ds[lds] += 1
        pair_dir[direction] += 1
        selected.append(row)
        if len(selected) >= int(config["max_pairs"]):
            break
    return pd.DataFrame(selected)


def balance_rows(selected: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("tail_geometry_score", "high_tail_geometry_score", "low_tail_geometry_score"),
        ("tail_resid_score", "high_tail_resid_score", "low_tail_resid_score"),
        ("ctrl_gt_mmd_proxy", "high_ctrl_gt_mmd_proxy", "low_ctrl_gt_mmd_proxy"),
        ("tail95_shift", "high_tail95_shift", "low_tail95_shift"),
        ("tail99_shift", "high_tail99_shift", "low_tail99_shift"),
        ("abs_var_shift", "high_abs_var_shift", "low_abs_var_shift"),
        ("tail_risk_ratio", "high_tail_risk_ratio", "low_tail_risk_ratio"),
        ("directional_alignment", "high_directional_alignment", "low_directional_alignment"),
        ("cross_count", "high_cross_count", "low_cross_count"),
        ("cross_effective", "high_cross_effective", "low_cross_effective"),
        ("raw_support_score", "high_raw_support_score", "low_raw_support_score"),
        ("response_norm", "high_response_norm", "low_response_norm"),
        ("n_gt", "high_n_gt", "low_n_gt"),
        ("n_ctrl", "high_n_ctrl", "low_n_ctrl"),
        ("max_state_entropy", "high_max_state_entropy", "low_max_state_entropy"),
        ("same_target_cross_dataset_total", "high_same_target_cross_dataset_total", "low_same_target_cross_dataset_total"),
        ("exact_bool", "high_exact_bool", "low_exact_bool"),
    ]
    rows: list[dict[str, Any]] = []
    for name, hcol, lcol in specs:
        h = selected[hcol].astype(float) if selected[hcol].dtype == bool else pd.to_numeric(selected[hcol], errors="coerce")
        l = selected[lcol].astype(float) if selected[lcol].dtype == bool else pd.to_numeric(selected[lcol], errors="coerce")
        rows.append(
            {
                "feature": name,
                "high_mean": float(h.mean()),
                "low_mean": float(l.mean()),
                "high_median": float(h.median()),
                "low_median": float(l.median()),
                "smd_high_minus_low": smd(h, l),
                "auc_discriminability": auc_discriminability(h, l),
            }
        )
    return pd.DataFrame(rows)


def negative_controls(selected: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    thresholds = {
        "directional_alignment": 0.70,
        "cross_count": 0.65,
        "cross_effective": 0.65,
        "raw_support_score": 0.70,
        "response_norm": 0.65,
        "n_gt": 0.65,
        "n_ctrl": 0.65,
        "max_state_entropy": 0.65,
        "same_target_cross_dataset_total": 0.65,
        "exact_bool": 0.65,
    }
    for feature, threshold in thresholds.items():
        hit = balance[balance["feature"].eq(feature)]
        if hit.empty:
            continue
        auc = float(hit.iloc[0]["auc_discriminability"])
        rows.append(
            {
                "control": f"{feature}_only_discriminability",
                "value": auc,
                "risk": bool(auc > threshold),
                "notes": f"max(AUC,1-AUC); threshold <= {threshold:.2f}",
            }
        )
    top_pair_frac = pair_direction_fraction(selected)
    high_ds_frac = side_dataset_fraction(selected, "high")
    low_ds_frac = side_dataset_fraction(selected, "low")
    crispri_frac = float((selected["perturbation_type_raw"].astype(str) == "CRISPRi").mean()) if not selected.empty else 0.0
    rows.extend(
        [
            {"control": "top_dataset_pair_direction_fraction", "value": top_pair_frac, "risk": bool(top_pair_frac > 0.15), "notes": "threshold <=0.15"},
            {"control": "top_high_side_dataset_fraction", "value": high_ds_frac, "risk": bool(high_ds_frac > 0.25), "notes": "threshold <=0.25"},
            {"control": "top_low_side_dataset_fraction", "value": low_ds_frac, "risk": bool(low_ds_frac > 0.25), "notes": "threshold <=0.25"},
            {"control": "crispri_pair_fraction", "value": crispri_frac, "risk": bool(crispri_frac > 0.80), "notes": "CRISPRi-only explanation risk"},
        ]
    )
    return pd.DataFrame(rows)


def assess_config(rows: pd.DataFrame, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates, scored_rows = build_candidates(rows, config)
    selected = greedy_select(candidates, config)
    if selected.empty:
        balance = pd.DataFrame()
        controls = pd.DataFrame()
        claims = pd.DataFrame()
    else:
        balance = balance_rows(selected)
        controls = negative_controls(selected, balance)
        claims = per_ptype_claims(selected)
    cov_features = [
        "directional_alignment",
        "cross_count",
        "cross_effective",
        "raw_support_score",
        "response_norm",
        "n_gt",
        "n_ctrl",
        "max_state_entropy",
        "same_target_cross_dataset_total",
        "exact_bool",
    ]
    max_cov_smd = (
        float(balance[balance["feature"].isin(cov_features)]["smd_high_minus_low"].abs().max())
        if not balance.empty
        else float("nan")
    )
    def auc_for(feature: str) -> float:
        if balance.empty:
            return float("nan")
        hit = balance.loc[balance["feature"].eq(feature), "auc_discriminability"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    summary = {
        **config,
        "n_scored_conditions": int(len(scored_rows)),
        "n_candidates": int(len(candidates)),
        "n_pairs_unique": int(len(selected)),
        "n_high_conditions": int(selected["high_key"].nunique()) if not selected.empty else 0,
        "n_low_conditions": int(selected["low_key"].nunique()) if not selected.empty else 0,
        "n_datasets_total": int(len(set(selected["high_dataset"].astype(str)) | set(selected["low_dataset"].astype(str)))) if not selected.empty else 0,
        "n_high_datasets": int(selected["high_dataset"].nunique()) if not selected.empty else 0,
        "n_low_datasets": int(selected["low_dataset"].nunique()) if not selected.empty else 0,
        "n_perturbation_types": int(selected["perturbation_type_raw"].nunique()) if not selected.empty else 0,
        "n_claim_ready_ptypes": int(claims["claim_ready"].sum()) if not claims.empty else 0,
        "median_tail_geometry_gap": float(selected["tail_geometry_gap"].median()) if not selected.empty else 0.0,
        "median_residual_score_gap": float(selected["residual_score_gap"].median()) if not selected.empty else 0.0,
        "median_mmd_proxy_gap": float((selected["high_ctrl_gt_mmd_proxy"] - selected["low_ctrl_gt_mmd_proxy"]).median()) if not selected.empty else 0.0,
        "max_abs_covariate_smd": max_cov_smd,
        "directional_auc": auc_for("directional_alignment"),
        "raw_support_auc": auc_for("raw_support_score"),
        "response_norm_auc": auc_for("response_norm"),
        "top_dataset_pair_direction_fraction": pair_direction_fraction(selected),
        "top_high_dataset_fraction": side_dataset_fraction(selected, "high"),
        "top_low_dataset_fraction": side_dataset_fraction(selected, "low"),
        "crispri_pair_fraction": float((selected["perturbation_type_raw"].astype(str) == "CRISPRi").mean()) if not selected.empty else 0.0,
        "any_negative_control_risk": bool(controls["risk"].astype(bool).any()) if not controls.empty else True,
    }
    reasons: list[str] = []
    if summary["n_pairs_unique"] < 250:
        reasons.append("pairs_below_250")
    if summary["n_high_conditions"] < 100 or summary["n_low_conditions"] < 100:
        reasons.append("unique_high_low_below_100")
    if summary["n_datasets_total"] < 12:
        reasons.append("datasets_total_below_12")
    if summary["n_high_datasets"] < 6 or summary["n_low_datasets"] < 6:
        reasons.append("side_datasets_below_6")
    if summary["n_perturbation_types"] < 3:
        reasons.append("perturbation_types_below_3")
    if summary["n_claim_ready_ptypes"] < 2:
        reasons.append("claim_ready_ptypes_below_2")
    if summary["median_tail_geometry_gap"] < 0.15:
        reasons.append("tail_geometry_gap_below_0p15")
    if summary["median_residual_score_gap"] < 0.10:
        reasons.append("residual_score_gap_below_0p10")
    if math.isfinite(summary["max_abs_covariate_smd"]) and summary["max_abs_covariate_smd"] > 0.35:
        reasons.append("max_covariate_smd_above_0p35")
    if math.isfinite(summary["directional_auc"]) and summary["directional_auc"] > 0.70:
        reasons.append("directional_auc_above_0p70")
    if math.isfinite(summary["raw_support_auc"]) and summary["raw_support_auc"] > 0.70:
        reasons.append("raw_support_auc_above_0p70")
    if math.isfinite(summary["response_norm_auc"]) and summary["response_norm_auc"] > 0.65:
        reasons.append("response_norm_auc_above_0p65")
    if summary["top_dataset_pair_direction_fraction"] > 0.15:
        reasons.append("dataset_pair_direction_above_15pct")
    if summary["top_high_dataset_fraction"] > 0.25 or summary["top_low_dataset_fraction"] > 0.25:
        reasons.append("side_dataset_fraction_above_25pct")
    if summary["crispri_pair_fraction"] > 0.80:
        reasons.append("crispri_fraction_above_80pct")
    if summary["any_negative_control_risk"]:
        reasons.append("negative_control_risk_present")
    summary["reasons"] = ";".join(reasons)
    summary["strict_pass"] = not reasons
    return summary, selected, balance, controls


def choose_best(configs: pd.DataFrame) -> pd.Series:
    work = configs.copy()
    work["score"] = (
        1000 * work["strict_pass"].astype(bool).astype(int)
        + work["n_pairs_unique"]
        + 40 * work["n_claim_ready_ptypes"]
        + 80 * work["median_tail_geometry_gap"]
        + 60 * work["median_residual_score_gap"]
        - 70 * work["max_abs_covariate_smd"].fillna(9)
        - 40 * work["raw_support_auc"].fillna(1)
        - 35 * work["response_norm_auc"].fillna(1)
        - 30 * work["directional_auc"].fillna(1)
        - 20 * work["top_dataset_pair_direction_fraction"].fillna(1)
        - 10 * work["crispri_pair_fraction"].fillna(1)
    )
    return work.sort_values("score", ascending=False).iloc[0]


def write_report(
    payload: dict[str, Any],
    config_rows: pd.DataFrame,
    selected: pd.DataFrame,
    balance: pd.DataFrame,
    controls: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    top = config_rows.sort_values(["strict_pass", "n_pairs_unique"], ascending=[False, False]).head(14)
    lines = [
        "# Dual-Baseline Tail-Geometry Admission Gate",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only over the parent train split.",
        "* Tail geometry uses train-only source/control-vs-GT latent samples: fixed-sigma RBF MMD proxy, tail95/tail99 shifts, variance shift, and a tail-risk ratio.",
        "* The score is residualized against response norm, cell counts, exact coverage, state entropy, same-target availability, cross-dataset support, direction alignment, perturbation type, and optional dataset dummies before high/low matching.",
        "* No canonical Track A rows, canonical multi rows, Track C query rows, checkpoints, training, inference, or GPU are used.",
        "",
        "## Decision",
        "",
        f"* Selected config: `{payload['selected_config_name']}`.",
        f"* Reasons: `{payload['reasons'] if payload['reasons'] else 'none'}`.",
        f"* Scored conditions: `{payload['summary']['n_scored_conditions']}`.",
        f"* Unique pairs: `{payload['summary']['n_pairs_unique']}`; datasets total/high/low: `{payload['summary']['n_datasets_total']}/{payload['summary']['n_high_datasets']}/{payload['summary']['n_low_datasets']}`.",
        f"* Median tail score gap: `{fmt(payload['summary']['median_tail_geometry_gap'])}`; residual-score gap: `{fmt(payload['summary']['median_residual_score_gap'])}`; MMD-proxy gap: `{fmt(payload['summary']['median_mmd_proxy_gap'])}`.",
        f"* Max covariate SMD: `{fmt(payload['summary']['max_abs_covariate_smd'])}`; support AUC: `{fmt(payload['summary']['raw_support_auc'])}`; response AUC: `{fmt(payload['summary']['response_norm_auc'])}`; direction AUC: `{fmt(payload['summary']['directional_auc'])}`.",
        "",
        "## Config Sweep",
        "",
        "| config | q | resp cal | cell cal | support cal | dataset residualized | scored | candidates | pairs | datasets | ptypes | claim ptypes | tail gap | resid gap | cov SMD | support AUC | response AUC | direction AUC | pair-dir frac | strict | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| `{row['name']}` | `{fmt(row['quantile'], 2)}` | `{fmt(row['response_log_caliper'], 2)}` | `{fmt(row['cell_log_caliper'], 2)}` | `{fmt(row['cross_count_caliper'], 0)}` | `{bool(row['include_dataset_dummies'])}` | "
            f"`{int(row['n_scored_conditions'])}` | `{int(row['n_candidates'])}` | `{int(row['n_pairs_unique'])}` | `{int(row['n_datasets_total'])}` | `{int(row['n_perturbation_types'])}` | `{int(row['n_claim_ready_ptypes'])}` | "
            f"`{fmt(row['median_tail_geometry_gap'])}` | `{fmt(row['median_residual_score_gap'])}` | `{fmt(row['max_abs_covariate_smd'])}` | `{fmt(row['raw_support_auc'])}` | `{fmt(row['response_norm_auc'])}` | `{fmt(row['directional_auc'])}` | `{fmt(row['top_dataset_pair_direction_fraction'])}` | `{bool(row['strict_pass'])}` | `{row['reasons']}` |"
        )
    lines.extend(["", "## Selected Balance", "", "| feature | high mean | low mean | high median | low median | SMD | AUC |", "|---|---:|---:|---:|---:|---:|---:|"])
    for _, row in balance.iterrows():
        lines.append(
            f"| `{row['feature']}` | `{fmt(row['high_mean'])}` | `{fmt(row['low_mean'])}` | `{fmt(row['high_median'])}` | `{fmt(row['low_median'])}` | `{fmt(row['smd_high_minus_low'])}` | `{fmt(row['auc_discriminability'])}` |"
        )
    lines.extend(["", "## Negative Controls", "", "| control | value | risk | notes |", "|---|---:|---:|---|"])
    for _, row in controls.iterrows():
        lines.append(f"| `{row['control']}` | `{fmt(row['value'])}` | `{bool(row['risk'])}` | {row['notes']} |")
    lines.extend(["", "## Per-Perturbation-Type Claim Readiness", "", "| ptype | pairs | high uniq | low uniq | high datasets | low datasets | claim ready |", "|---|---:|---:|---:|---:|---:|---:|"])
    for _, row in claims.iterrows():
        lines.append(
            f"| `{row['perturbation_type_raw']}` | `{int(row['n_pairs'])}` | `{int(row['n_high_conditions'])}` | `{int(row['n_low_conditions'])}` | `{int(row['n_high_datasets'])}` | `{int(row['n_low_datasets'])}` | `{bool(row['claim_ready'])}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    if payload["status"].endswith("pass_external_audit_no_gpu"):
        lines.append("* The tail-geometry axis passes this train-only balance gate. It still needs external audit/nulls before any GPU training-set intervention.")
    else:
        lines.append("* The tail-geometry axis does not pass the current train-only balance/control gate. Do not launch a GPU training-set intervention from this axis.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"* Feature rows: `{OUT_FEATURES}`",
            f"* Config rows: `{OUT_CONFIGS}`",
            f"* Selected pairs: `{OUT_SELECTED}`",
            f"* Balance: `{OUT_BALANCE}`",
            f"* Negative controls: `{OUT_CONTROLS}`",
            f"* High split: `{payload['outputs'].get('high_split', '')}`",
            f"* Low split: `{payload['outputs'].get('low_split', '')}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUT_DIR, SPLIT_DIR, OUT_MD, OUT_JSON, OUT_FEATURES, OUT_CONFIGS, OUT_SELECTED, OUT_BALANCE, OUT_CONTROLS

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--mmd-cap", type=int, default=64)
    parser.add_argument("--max-conditions", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    set_thread_env(args.threads)
    OUT_DIR = args.out_dir
    SPLIT_DIR = args.split_dir
    OUT_MD = OUT_DIR / "LATENTFM_DUAL_BASELINE_TAIL_GEOMETRY_ADMISSION_GATE_20260629.md"
    OUT_JSON = OUT_DIR / "latentfm_dual_baseline_tail_geometry_admission_gate_20260629.json"
    OUT_FEATURES = OUT_DIR / "dual_baseline_tail_geometry_feature_rows.csv"
    OUT_CONFIGS = OUT_DIR / "dual_baseline_tail_geometry_config_rows.csv"
    OUT_SELECTED = OUT_DIR / "dual_baseline_tail_geometry_selected_pairs.csv"
    OUT_BALANCE = OUT_DIR / "dual_baseline_tail_geometry_balance.csv"
    OUT_CONTROLS = OUT_DIR / "dual_baseline_tail_geometry_negative_controls.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    support_rows = load_rows()
    allowed = parent_train_keys()
    support_rows = support_rows[support_rows["key"].astype(str).isin(allowed)].copy()
    features = materialize_features(support_rows, args.data_dir, int(args.mmd_cap), int(args.max_conditions))
    features = add_tail_score(features)
    features.to_csv(OUT_FEATURES, index=False)
    support_rows.attrs["features"] = features

    config_summaries: list[dict[str, Any]] = []
    selected_by_config: dict[str, pd.DataFrame] = {}
    balance_by_config: dict[str, pd.DataFrame] = {}
    controls_by_config: dict[str, pd.DataFrame] = {}
    for config in CONFIGS:
        summary, selected, balance, controls = assess_config(support_rows, config)
        config_summaries.append(summary)
        selected_by_config[str(config["name"])] = selected
        balance_by_config[str(config["name"])] = balance
        controls_by_config[str(config["name"])] = controls

    config_rows = pd.DataFrame(config_summaries)
    best = choose_best(config_rows)
    best_name = str(best["name"])
    selected = selected_by_config[best_name]
    balance = balance_by_config[best_name]
    controls = controls_by_config[best_name]
    claims = per_ptype_claims(selected) if not selected.empty else pd.DataFrame()

    config_rows.to_csv(OUT_CONFIGS, index=False)
    selected.to_csv(OUT_SELECTED, index=False)
    balance.to_csv(OUT_BALANCE, index=False)
    controls.to_csv(OUT_CONTROLS, index=False)

    outputs: dict[str, str] = {}
    if not selected.empty:
        parent = load_json(PARENT_SPLIT)
        high_split = split_from_conditions(parent, selected, "high_dataset", "high_condition")
        low_split = split_from_conditions(parent, selected, "low_dataset", "low_condition")
        high_path = SPLIT_DIR / f"split_seed42_xverse_tail_geometry_high_{best_name}.json"
        low_path = SPLIT_DIR / f"split_seed42_xverse_tail_geometry_low_{best_name}.json"
        write_json(high_path, high_split)
        write_json(low_path, low_split)
        outputs["high_split"] = str(high_path)
        outputs["low_split"] = str(low_path)

    status = (
        "dual_baseline_tail_geometry_admission_pass_external_audit_no_gpu"
        if bool(best["strict_pass"])
        else "dual_baseline_tail_geometry_admission_fail_no_gpu"
    )
    payload = {
        "created_at": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "selected_config_name": best_name,
        "summary": best.drop(labels=["score"], errors="ignore").to_dict(),
        "reasons": str(best["reasons"]),
        "inputs": {
            "parent_split": str(PARENT_SPLIT),
            "support_rows": "/data/cyx/1030/scLatent/reports/condition_neighborhood_support_gate_20260629/condition_neighborhood_support_rows.csv",
            "data_dir": str(args.data_dir),
        },
        "boundary": {
            "train_only_parent_split": True,
            "mmd_proxy_cap": int(args.mmd_cap),
            "fixed_rbf_sigmas": list(SIGMAS),
            "no_canonical_tracka_rows": True,
            "no_canonical_multi": True,
            "no_trackc_query": True,
            "no_checkpoint_selection": True,
            "no_training_or_inference": True,
        },
        "outputs": {
            "features": str(OUT_FEATURES),
            "config_rows": str(OUT_CONFIGS),
            "selected_pairs": str(OUT_SELECTED),
            "balance": str(OUT_BALANCE),
            "negative_controls": str(OUT_CONTROLS),
            **outputs,
            "report": str(OUT_MD),
        },
    }
    write_json(OUT_JSON, payload)
    write_report(payload, config_rows, selected, balance, controls, claims)


if __name__ == "__main__":
    main()
