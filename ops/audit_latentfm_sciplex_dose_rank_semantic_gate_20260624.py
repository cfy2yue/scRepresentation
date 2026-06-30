#!/usr/bin/env python3
"""CPU train-only gate for SciPlex dose-rank semantic signal.

This uses xverse per-cell latent embeddings and SciPlex obs metadata to test
whether log-dose adds predictive signal beyond background+drug intercepts.
It does not train LatentFM, launch GPU, read canonical multi, or read Track C
query artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
EMB_ROOT = ROOT / "scFM_output/embeddings/xverse"
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
CAP120_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
OUT_JSON = ROOT / "reports/latentfm_sciplex_dose_rank_semantic_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCIPLEX_DOSE_RANK_SEMANTIC_GATE_20260624.md"

CHEMICAL_DATASETS = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")
DOSE_VALUES = (0.001, 0.01, 0.1, 1.0)
ALPHA = 10.0
SEED = 20260624


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 0:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    xtr = (train_x - mu) / sd
    xte = (test_x - mu) / sd
    xtx = xtr.T @ xtr
    coef = np.linalg.solve(xtx + float(alpha) * np.eye(xtx.shape[0]), xtr.T @ train_y)
    return xte @ coef


def one_hot(value: str, universe: list[str]) -> np.ndarray:
    arr = np.zeros((len(universe),), dtype=np.float64)
    try:
        arr[universe.index(value)] = 1.0
    except ValueError:
        pass
    return arr


def load_rows() -> list[dict[str, Any]]:
    base = load_json(BASE_SPLIT)
    cap120 = load_json(CAP120_SPLIT)
    rows: list[dict[str, Any]] = []
    for ds in CHEMICAL_DATASETS:
        train_drugs = set(str(x) for x in (cap120.get(ds) or {}).get("train") or [])
        canonical_ref = set(str(x) for x in (base.get(ds) or {}).get("canonical_test_reference") or [])
        train_drugs -= canonical_ref
        obs_path = EMB_ROOT / ds / "raw/obs.parquet"
        latent_path = EMB_ROOT / ds / "raw/latent.npy"
        if not obs_path.exists() or not latent_path.exists():
            raise FileNotFoundError(f"missing xverse raw embedding artifacts for {ds}")
        obs = pd.read_parquet(obs_path)
        latent = np.load(latent_path, mmap_mode="r")
        if len(obs) != latent.shape[0]:
            raise ValueError(f"{ds}: obs rows {len(obs)} != latent rows {latent.shape[0]}")

        condition = obs["condition"].astype(str)
        control = obs["control"].astype(str)
        ctrl_mask = (control == "1") | (condition == "control")
        if int(ctrl_mask.sum()) <= 0:
            raise ValueError(f"{ds}: no control rows")
        ctrl_mean = np.asarray(latent[np.flatnonzero(ctrl_mask.to_numpy())], dtype=np.float64).mean(axis=0)

        pert = obs["perturbation"].astype(str)
        dose = obs["dose"].astype(float)
        keep = (~ctrl_mask) & pert.isin(train_drugs) & dose.isin(DOSE_VALUES)
        frame = pd.DataFrame(
            {
                "idx": np.arange(len(obs), dtype=np.int64),
                "drug": pert,
                "dose": dose,
            }
        ).loc[keep.to_numpy()]
        for (drug, dose_value), group in frame.groupby(["drug", "dose"], sort=True):
            idx = group["idx"].to_numpy(dtype=np.int64)
            if len(idx) < 3:
                continue
            mean = np.asarray(latent[idx], dtype=np.float64).mean(axis=0)
            rows.append(
                {
                    "dataset": ds,
                    "drug": str(drug),
                    "dose": float(dose_value),
                    "n_cells": int(len(idx)),
                    "delta": (mean - ctrl_mean).astype(np.float64),
                }
            )
    return rows


def build_x(
    rows: list[dict[str, Any]],
    drugs: list[str],
    *,
    use_dose: bool,
    dose_mode: str = "actual",
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_doses = np.asarray([math.log10(float(row["dose"])) for row in rows], dtype=np.float64)
    if dose_mode == "shuffle":
        log_doses = rng.permutation(log_doses)
    elif dose_mode == "random":
        log_doses = rng.standard_normal(log_doses.shape)
    elif dose_mode != "actual":
        raise ValueError(f"unknown dose mode {dose_mode}")

    xs = []
    for i, row in enumerate(rows):
        parts = [one_hot(str(row["dataset"]), list(CHEMICAL_DATASETS)), one_hot(str(row["drug"]), drugs)]
        if use_dose:
            parts.append(np.asarray([log_doses[i]], dtype=np.float64))
        xs.append(np.concatenate(parts))
    return np.vstack(xs)


def cv_eval(rows: list[dict[str, Any]], *, use_dose: bool, dose_mode: str) -> dict[str, Any]:
    drugs = sorted({str(row["drug"]) for row in rows})
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_rows: list[dict[str, Any]] = []
    for held_dose in DOSE_VALUES:
        train_rows = [row for row in rows if abs(float(row["dose"]) - held_dose) > 1e-12]
        test_rows = [row for row in rows if abs(float(row["dose"]) - held_dose) <= 1e-12]
        if not train_rows or not test_rows:
            continue
        train_x = build_x(train_rows, drugs, use_dose=use_dose, dose_mode=dose_mode, seed=SEED + int(round(held_dose * 10000)))
        test_x = build_x(test_rows, drugs, use_dose=use_dose, dose_mode=("actual" if dose_mode == "actual" else dose_mode), seed=SEED + 100 + int(round(held_dose * 10000)))
        train_y = np.vstack([row["delta"] for row in train_rows])
        test_y = np.vstack([row["delta"] for row in test_rows])
        pred = ridge_predict(train_x, train_y, test_x)
        all_true.extend([x for x in test_y])
        all_pred.extend([x for x in pred])
        all_rows.extend(test_rows)
    per = [pearson(p, y) for p, y in zip(all_pred, all_true)]
    by_ds: dict[str, list[float]] = {ds: [] for ds in CHEMICAL_DATASETS}
    by_dose: dict[str, list[float]] = {str(d): [] for d in DOSE_VALUES}
    for row, val in zip(all_rows, per):
        if math.isfinite(val):
            by_ds[str(row["dataset"])].append(float(val))
            by_dose[str(float(row["dose"]))].append(float(val))
    ds_mean = {ds: (float(np.mean(vals)) if vals else None) for ds, vals in by_ds.items()}
    dose_mean = {d: (float(np.mean(vals)) if vals else None) for d, vals in by_dose.items()}
    finite = [float(x) for x in per if math.isfinite(x)]
    return {
        "mean_pp_proxy": float(np.mean(finite)) if finite else None,
        "n_eval": len(finite),
        "dataset_means": ds_mean,
        "dataset_min": min((v for v in ds_mean.values() if v is not None), default=None),
        "dose_means": dose_mean,
        "dose_min": min((v for v in dose_mean.values() if v is not None), default=None),
        "negative_dataset_tails_lt_minus_0p02": sum(1 for v in ds_mean.values() if v is not None and v < -0.02),
    }


def diff(model: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    ds_inc = {}
    for ds, val in (model.get("dataset_means") or {}).items():
        b = (baseline.get("dataset_means") or {}).get(ds)
        ds_inc[ds] = None if val is None or b is None else float(val) - float(b)
    finite = [v for v in ds_inc.values() if v is not None]
    return {
        "mean_increment": None
        if model.get("mean_pp_proxy") is None or baseline.get("mean_pp_proxy") is None
        else float(model["mean_pp_proxy"]) - float(baseline["mean_pp_proxy"]),
        "dataset_increment_means": ds_inc,
        "dataset_increment_min": min(finite, default=None),
    }


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def main() -> int:
    rows = load_rows()
    baseline = cv_eval(rows, use_dose=False, dose_mode="actual")
    actual = cv_eval(rows, use_dose=True, dose_mode="actual")
    shuffled = cv_eval(rows, use_dose=True, dose_mode="shuffle")
    random = cv_eval(rows, use_dose=True, dose_mode="random")
    actual_inc = diff(actual, baseline)
    shuffled_inc = diff(shuffled, baseline)
    random_inc = diff(random, baseline)
    control_inc = max(shuffled_inc["mean_increment"] or 0.0, random_inc["mean_increment"] or 0.0)

    reasons = []
    if len(rows) < 900:
        reasons.append("too_few_dose_rows")
    if actual_inc["mean_increment"] is None or actual_inc["mean_increment"] < 0.020:
        reasons.append("actual_dose_increment_lt_0p020")
    if actual_inc["dataset_increment_min"] is None or actual_inc["dataset_increment_min"] < 0.0:
        reasons.append("dataset_increment_min_below_0")
    if control_inc >= 0.003:
        reasons.append("dose_shuffle_or_random_control_not_collapsed_below_0p003")
    if actual["dataset_min"] is None or actual["dataset_min"] < -0.020:
        reasons.append("actual_dataset_tail_below_minus_0p020")
    status = "sciplex_dose_rank_semantic_gate_fail_no_gpu" if reasons else "sciplex_dose_rank_semantic_gate_pass_external_review_next"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "train_only_sciplex_cap120_drugs": True,
            "canonical_reference_excluded": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "latentfm_training_or_inference": False,
            "gpu": False,
            "validation": "leave_dose_rank_out_cv_on_xverse_per_cell_latent_deltas",
        },
        "inputs": {
            "emb_root": str(EMB_ROOT),
            "base_split": str(BASE_SPLIT),
            "cap120_split": str(CAP120_SPLIT),
        },
        "summary": {
            "dose_rows": len(rows),
            "datasets": list(CHEMICAL_DATASETS),
            "unique_drugs": len({row["drug"] for row in rows}),
            "dose_values": list(DOSE_VALUES),
            "baseline_background_plus_drug_intercept": baseline,
            "actual_logdose": actual,
            "dose_shuffled": shuffled,
            "dose_random": random,
            "increments_vs_intercept": {
                "actual_logdose": actual_inc,
                "dose_shuffled": shuffled_inc,
                "dose_random": random_inc,
            },
        },
        "reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM SciPlex Dose-Rank Semantic Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only train-only gate on xverse per-cell latent deltas.",
        "- Uses cap120 train SciPlex drugs and excludes canonical reference drugs.",
        "- Validation is leave-dose-rank-out over four SciPlex doses.",
        "- Does not train LatentFM, run inference, launch GPU, read canonical multi, or read Track C query.",
        "",
        "## Summary",
        "",
        f"- dose rows: `{len(rows)}`",
        f"- unique drugs: `{payload['summary']['unique_drugs']}`",
        f"- dose values: `{list(DOSE_VALUES)}`",
        "",
        "| model | mean pp proxy | dataset min | increment vs intercept | increment dataset min |",
        "|---|---:|---:|---:|---:|",
        f"| `background_plus_drug_intercept` | {fmt(baseline['mean_pp_proxy'])} | {fmt(baseline['dataset_min'])} | NA | NA |",
        f"| `actual_logdose` | {fmt(actual['mean_pp_proxy'])} | {fmt(actual['dataset_min'])} | {fmt(actual_inc['mean_increment'])} | {fmt(actual_inc['dataset_increment_min'])} |",
        f"| `dose_shuffled` | {fmt(shuffled['mean_pp_proxy'])} | {fmt(shuffled['dataset_min'])} | {fmt(shuffled_inc['mean_increment'])} | {fmt(shuffled_inc['dataset_increment_min'])} |",
        f"| `dose_random` | {fmt(random['mean_pp_proxy'])} | {fmt(random['dataset_min'])} | {fmt(random_inc['mean_increment'])} | {fmt(random_inc['dataset_increment_min'])} |",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        "- GPU authorized: `False`",
        "- If this fails, the current SciPlex molecule/dose semantics branch should remain a provenance/negative-control branch until a genuinely new train-only gate is proposed.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "reasons": reasons}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
