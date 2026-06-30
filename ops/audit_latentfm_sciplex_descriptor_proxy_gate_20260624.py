#!/usr/bin/env python3
"""CPU descriptor proxy gate for SciPlex chemical scaling."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_chemical_holdout_splits_20260624/split_seed42_xverse_scaling_cap120_chemical_holdout_v1.json"
CACHE = ROOT / "dataset/drug_cache/sciplex_smiles_morgan2048_20260624"
OUT_JSON = ROOT / "reports/latentfm_sciplex_descriptor_proxy_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCIPLEX_DESCRIPTOR_PROXY_GATE_20260624.md"

CHEMICAL_DATASETS = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")
ALPHA = 10.0
SEED = 20260624


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_index_map(h5: h5py.File) -> dict[str, int]:
    out = {}
    for idx, cond in enumerate(h5["conditions"][:]):
        out[cond.decode("utf-8") if isinstance(cond, bytes) else str(cond)] = int(idx)
    return out


def mean_for(h5: h5py.File, group: str, idx: int) -> np.ndarray:
    lo = int(h5[group]["offsets"][idx])
    hi = int(h5[group]["offsets"][idx + 1])
    if hi <= lo:
        raise ValueError(f"empty {group} rows for index {idx}")
    return np.asarray(h5[group]["emb"][lo:hi], dtype=np.float64).mean(axis=0)


def load_deltas(split: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    hold_rows: list[dict[str, Any]] = []
    for ds in CHEMICAL_DATASETS:
        with h5py.File(DATA_DIR / f"{ds}.h5", "r") as h5:
            cmap = condition_index_map(h5)
            for mode, out_rows in (("train", train_rows), ("family_drug_trainonly_holdout", hold_rows)):
                for cond in split[ds].get(mode) or []:
                    idx = cmap.get(str(cond))
                    if idx is None:
                        continue
                    ctrl = mean_for(h5, "ctrl", idx)
                    gt = mean_for(h5, "gt", idx)
                    out_rows.append({"dataset": ds, "drug": str(cond), "delta": (gt - ctrl).astype(np.float64)})
    return train_rows, hold_rows


def load_descriptors() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    idx = load_json(CACHE / "drug_index.json")
    emb = np.load(CACHE / "drug_embeddings.npy")
    desc = {drug: np.asarray(emb[int(row)], dtype=np.float64) for drug, row in idx.items() if not drug.startswith("<")}
    scaffolds: dict[str, str] = {}
    with (CACHE / "drug_metadata.tsv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            scaffolds[str(row["drug"])] = str(row.get("scaffold") or "")
    return desc, scaffolds


def background_features(ds: str) -> np.ndarray:
    return np.asarray([1.0 if ds == x else 0.0 for x in CHEMICAL_DATASETS], dtype=np.float64)


def build_xy(rows: list[dict[str, Any]], desc: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs = []
    ys = []
    kept = []
    for row in rows:
        d = desc.get(row["drug"])
        if d is None:
            continue
        xs.append(np.concatenate([d, background_features(row["dataset"])]))
        ys.append(row["delta"])
        kept.append(row)
    return np.vstack(xs), np.vstack(ys), kept


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (train_x - mu) / sd, (test_x - mu) / sd


def kernel_ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    xtr, xte = standardize(train_x, test_x)
    k = xtr @ xtr.T
    coef = np.linalg.solve(k + float(alpha) * np.eye(k.shape[0]), train_y)
    return xte @ xtr.T @ coef


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den <= 0:
        return float("nan")
    return float(np.dot(aa, bb) / den)


def eval_pred(pred: np.ndarray, test_y: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    per = [pearson(p, y) for p, y in zip(pred, test_y)]
    by_ds: dict[str, list[float]] = {ds: [] for ds in CHEMICAL_DATASETS}
    for row, val in zip(rows, per):
        if math.isfinite(val):
            by_ds[row["dataset"]].append(float(val))
    ds_mean = {ds: (sum(vals) / len(vals) if vals else None) for ds, vals in by_ds.items()}
    finite = [x for x in per if math.isfinite(x)]
    return {
        "mean_pp_proxy": sum(finite) / len(finite) if finite else None,
        "n": len(finite),
        "dataset_means": ds_mean,
        "dataset_min": min((v for v in ds_mean.values() if v is not None), default=None),
        "negative_dataset_tails_lt_minus_0p02": sum(1 for v in ds_mean.values() if v is not None and v < -0.02),
    }


def shuffled_desc(desc: dict[str, np.ndarray], *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = sorted(desc)
    vals = [desc[k] for k in keys]
    perm = rng.permutation(len(vals))
    return {k: vals[int(perm[i])] for i, k in enumerate(keys)}


def random_desc(desc: dict[str, np.ndarray], *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(v.shape).astype(np.float64) for k, v in desc.items()}


def run_model(train_rows: list[dict[str, Any]], hold_rows: list[dict[str, Any]], desc: dict[str, np.ndarray]) -> dict[str, Any]:
    train_x, train_y, train_kept = build_xy(train_rows, desc)
    test_x, test_y, test_kept = build_xy(hold_rows, desc)
    pred = kernel_ridge_predict(train_x, train_y, test_x, ALPHA)
    out = eval_pred(pred, test_y, test_kept)
    out["n_train"] = len(train_kept)
    out["n_holdout"] = len(test_kept)
    return out


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def main() -> int:
    split = load_json(SPLIT)
    train_rows, hold_rows = load_deltas(split)
    desc, scaffolds = load_descriptors()
    actual = run_model(train_rows, hold_rows, desc)
    shuffled = run_model(train_rows, hold_rows, shuffled_desc(desc, seed=SEED + 1))
    random = run_model(train_rows, hold_rows, random_desc(desc, seed=SEED + 2))
    hold_scaffolds = {scaffolds.get(row["drug"], "") for row in hold_rows if scaffolds.get(row["drug"], "")}
    train_scaffolds = {scaffolds.get(row["drug"], "") for row in train_rows if scaffolds.get(row["drug"], "")}
    unseen_scaffolds = hold_scaffolds - train_scaffolds
    reasons = []
    if actual["n_holdout"] < 45:
        reasons.append("too_few_holdout_rows")
    if len(hold_scaffolds) < 30:
        reasons.append("too_few_holdout_scaffolds")
    if (actual["mean_pp_proxy"] or -1.0) < 0.020:
        reasons.append("actual_morgan_proxy_pp_lt_0p020")
    if (actual["mean_pp_proxy"] or -1.0) - max(shuffled["mean_pp_proxy"] or 0.0, random["mean_pp_proxy"] or 0.0) < 0.010:
        reasons.append("actual_not_0p010_above_controls")
    if actual["dataset_min"] is None or actual["dataset_min"] < -0.020:
        reasons.append("dataset_tail_below_minus_0p020")
    if max(shuffled["mean_pp_proxy"] or 0.0, random["mean_pp_proxy"] or 0.0) >= 0.003:
        reasons.append("negative_controls_do_not_collapse_below_0p003")
    status = "sciplex_descriptor_proxy_gate_fail_no_gpu" if reasons else "sciplex_descriptor_proxy_gate_pass_external_review_next"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "train_only_sciplex_parent": True,
            "canonical_reference_excluded": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "split": str(SPLIT),
            "cache": str(CACHE),
        },
        "summary": {
            "train_rows": len(train_rows),
            "holdout_rows": len(hold_rows),
            "holdout_scaffolds": len(hold_scaffolds),
            "unseen_holdout_scaffolds": len(unseen_scaffolds),
            "actual_morgan": actual,
            "shuffled_descriptor": shuffled,
            "random_descriptor": random,
        },
        "reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM SciPlex Descriptor Proxy Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only descriptor proxy on train-only SciPlex condition means.",
        "- Canonical reference drugs are excluded by the holdout split.",
        "- Does not train, infer, launch GPU, read canonical multi, or read Track C query.",
        "",
        "## Summary",
        "",
        f"- train rows: `{len(train_rows)}`",
        f"- holdout rows: `{len(hold_rows)}`",
        f"- holdout scaffolds: `{len(hold_scaffolds)}`",
        f"- unseen holdout scaffolds: `{len(unseen_scaffolds)}`",
        "",
        "| model | mean pp proxy | dataset min | negative tails |",
        "|---|---:|---:|---:|",
        f"| `actual_morgan` | {fmt(actual['mean_pp_proxy'])} | {fmt(actual['dataset_min'])} | {actual['negative_dataset_tails_lt_minus_0p02']} |",
        f"| `shuffled_descriptor` | {fmt(shuffled['mean_pp_proxy'])} | {fmt(shuffled['dataset_min'])} | {shuffled['negative_dataset_tails_lt_minus_0p02']} |",
        f"| `random_descriptor` | {fmt(random['mean_pp_proxy'])} | {fmt(random['dataset_min'])} | {random['negative_dataset_tails_lt_minus_0p02']} |",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        "- GPU authorized: `False`",
        "- A pass would require external review before any descriptor-cache training smoke.",
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
