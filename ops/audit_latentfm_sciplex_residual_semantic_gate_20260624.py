#!/usr/bin/env python3
"""CPU residual semantic gate for SciPlex pathway/target/scaffold signals.

This tests whether curated SciPlex semantic annotations add signal over the
background-only latent-delta proxy on the train-only chemical holdout. It does
not use Morgan descriptors, train LatentFM, run inference, read canonical multi,
or read Track C query artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_chemical_holdout_splits_20260624/split_seed42_xverse_scaling_cap120_chemical_holdout_v1.json"
CACHE = ROOT / "dataset/drug_cache/sciplex_smiles_morgan2048_20260624"
OUT_JSON = ROOT / "reports/latentfm_sciplex_residual_semantic_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCIPLEX_RESIDUAL_SEMANTIC_GATE_20260624.md"

CHEMICAL_DATASETS = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")
ALPHA = 10.0
SEED = 20260624


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_index_map(h5: h5py.File) -> dict[str, int]:
    return {
        (cond.decode("utf-8") if isinstance(cond, bytes) else str(cond)): int(i)
        for i, cond in enumerate(h5["conditions"][:])
    }


def mean_for(h5: h5py.File, group: str, idx: int) -> np.ndarray:
    lo = int(h5[group]["offsets"][idx])
    hi = int(h5[group]["offsets"][idx + 1])
    if hi <= lo:
        raise ValueError(f"empty rows for {group} idx={idx}")
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


def split_terms(value: str) -> list[str]:
    out = []
    for part in str(value or "").replace(";", ",").split(","):
        item = part.strip()
        if item:
            out.append(item)
    return out


def load_metadata() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    with (CACHE / "drug_metadata.tsv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            drug = str(row["drug"])
            meta[drug] = {
                "pathways": split_terms(row.get("pathways") or ""),
                "targets": split_terms(row.get("targets") or ""),
                "scaffold": str(row.get("scaffold") or ""),
                "bit_count": float(row.get("bit_count") or 0.0),
            }
    return meta


def background_features(ds: str) -> np.ndarray:
    return np.asarray([1.0 if ds == x else 0.0 for x in CHEMICAL_DATASETS], dtype=np.float64)


def vocab_from(meta: dict[str, dict[str, Any]], train_drugs: set[str]) -> dict[str, list[str]]:
    pathways = sorted({x for drug in train_drugs for x in meta.get(drug, {}).get("pathways", [])})
    targets = sorted({x for drug in train_drugs for x in meta.get(drug, {}).get("targets", [])})
    scaffolds = sorted({str(meta.get(drug, {}).get("scaffold") or "") for drug in train_drugs if meta.get(drug, {}).get("scaffold")})
    return {"pathways": pathways, "targets": targets, "scaffolds": scaffolds}


def multihot(items: list[str], vocab: list[str]) -> np.ndarray:
    out = np.zeros((len(vocab),), dtype=np.float64)
    pos = {v: i for i, v in enumerate(vocab)}
    for item in items:
        idx = pos.get(item)
        if idx is not None:
            out[idx] = 1.0
    return out


def semantic_features(drug: str, meta: dict[str, dict[str, Any]], vocab: dict[str, list[str]]) -> np.ndarray:
    row = meta.get(drug) or {}
    scaffold = str(row.get("scaffold") or "")
    return np.concatenate(
        [
            multihot(list(row.get("pathways") or []), vocab["pathways"]),
            multihot(list(row.get("targets") or []), vocab["targets"]),
            multihot([scaffold] if scaffold else [], vocab["scaffolds"]),
            np.asarray([float(row.get("bit_count") or 0.0)], dtype=np.float64),
        ]
    )


def build_x(
    rows: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    vocab: dict[str, list[str]],
    *,
    use_semantics: bool,
    semantic_mode: str = "actual",
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sem_by_drug = {row["drug"]: semantic_features(row["drug"], meta, vocab) for row in rows}
    if semantic_mode == "shuffle":
        rng = np.random.default_rng(seed)
        drugs = sorted(sem_by_drug)
        vals = [sem_by_drug[d] for d in drugs]
        perm = rng.permutation(len(vals))
        sem_by_drug = {d: vals[int(perm[i])] for i, d in enumerate(drugs)}
    elif semantic_mode == "random":
        rng = np.random.default_rng(seed)
        sem_by_drug = {d: rng.standard_normal(v.shape).astype(np.float64) for d, v in sem_by_drug.items()}
    elif semantic_mode != "actual":
        raise ValueError(f"unknown semantic mode {semantic_mode}")

    xs = []
    ys = []
    kept = []
    for row in rows:
        parts = [background_features(row["dataset"])]
        if use_semantics:
            parts.append(sem_by_drug[row["drug"]])
        xs.append(np.concatenate(parts))
        ys.append(row["delta"])
        kept.append(row)
    return np.vstack(xs), np.vstack(ys), kept


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (train_x - mu) / sd, (test_x - mu) / sd


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    xtr, xte = standardize(train_x, test_x)
    k = xtr @ xtr.T
    coef = np.linalg.solve(k + ALPHA * np.eye(k.shape[0]), train_y)
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
    by_ds = {ds: [] for ds in CHEMICAL_DATASETS}
    for row, val in zip(rows, per):
        if np.isfinite(val):
            by_ds[row["dataset"]].append(float(val))
    ds_mean = {ds: (float(np.mean(vals)) if vals else None) for ds, vals in by_ds.items()}
    finite = [x for x in per if np.isfinite(x)]
    return {
        "mean_pp_proxy": float(np.mean(finite)) if finite else None,
        "n": len(finite),
        "dataset_means": ds_mean,
        "dataset_min": min((v for v in ds_mean.values() if v is not None), default=None),
        "negative_dataset_tails_lt_minus_0p02": sum(1 for v in ds_mean.values() if v is not None and v < -0.02),
    }


def run_model(
    train_rows: list[dict[str, Any]],
    hold_rows: list[dict[str, Any]],
    meta: dict[str, dict[str, Any]],
    vocab: dict[str, list[str]],
    *,
    use_semantics: bool,
    semantic_mode: str,
) -> dict[str, Any]:
    train_x, train_y, train_kept = build_x(train_rows, meta, vocab, use_semantics=use_semantics, semantic_mode=semantic_mode, seed=SEED + 1)
    test_x, test_y, test_kept = build_x(hold_rows, meta, vocab, use_semantics=use_semantics, semantic_mode=semantic_mode, seed=SEED + 2)
    pred = ridge_predict(train_x, train_y, test_x)
    out = eval_pred(pred, test_y, test_kept)
    out["n_train"] = len(train_kept)
    out["n_holdout"] = len(test_kept)
    return out


def delta(model: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    by_ds = {}
    for ds, val in (model.get("dataset_means") or {}).items():
        b = (baseline.get("dataset_means") or {}).get(ds)
        by_ds[ds] = None if val is None or b is None else float(val) - float(b)
    finite = [v for v in by_ds.values() if v is not None]
    return {
        "mean_increment": None
        if model.get("mean_pp_proxy") is None or baseline.get("mean_pp_proxy") is None
        else float(model["mean_pp_proxy"]) - float(baseline["mean_pp_proxy"]),
        "dataset_increment_means": by_ds,
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
    split = load_json(SPLIT)
    train_rows, hold_rows = load_deltas(split)
    meta = load_metadata()
    train_drugs = {row["drug"] for row in train_rows}
    hold_drugs = {row["drug"] for row in hold_rows}
    vocab = vocab_from(meta, train_drugs)
    baseline = run_model(train_rows, hold_rows, meta, vocab, use_semantics=False, semantic_mode="actual")
    actual = run_model(train_rows, hold_rows, meta, vocab, use_semantics=True, semantic_mode="actual")
    shuffled = run_model(train_rows, hold_rows, meta, vocab, use_semantics=True, semantic_mode="shuffle")
    random = run_model(train_rows, hold_rows, meta, vocab, use_semantics=True, semantic_mode="random")
    actual_inc = delta(actual, baseline)
    shuffled_inc = delta(shuffled, baseline)
    random_inc = delta(random, baseline)
    control_inc = max(shuffled_inc["mean_increment"] or 0.0, random_inc["mean_increment"] or 0.0)

    reasons: list[str] = []
    if len(hold_rows) < 45:
        reasons.append("too_few_holdout_rows")
    if actual_inc["mean_increment"] is None or actual_inc["mean_increment"] < 0.020:
        reasons.append("actual_semantic_increment_lt_0p020")
    if actual_inc["dataset_increment_min"] is None or actual_inc["dataset_increment_min"] < -0.020:
        reasons.append("actual_semantic_dataset_increment_tail_below_minus_0p020")
    if actual_inc["mean_increment"] is None or actual_inc["mean_increment"] - control_inc < 0.010:
        reasons.append("actual_semantic_not_0p010_above_controls")
    if control_inc >= 0.003:
        reasons.append("semantic_shuffle_or_random_control_not_collapsed_below_0p003")
    if actual["dataset_min"] is None or actual["dataset_min"] < -0.020:
        reasons.append("actual_semantic_dataset_tail_below_minus_0p020")

    status = "sciplex_residual_semantic_gate_fail_no_gpu" if reasons else "sciplex_residual_semantic_gate_pass_external_review_next"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "train_only_sciplex_parent": True,
            "canonical_reference_excluded": True,
            "uses_morgan_descriptor": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "split": str(SPLIT),
            "metadata": str(CACHE / "drug_metadata.tsv"),
        },
        "summary": {
            "train_rows": len(train_rows),
            "holdout_rows": len(hold_rows),
            "train_drugs": len(train_drugs),
            "holdout_drugs": len(hold_drugs),
            "pathway_vocab": len(vocab["pathways"]),
            "target_vocab": len(vocab["targets"]),
            "scaffold_vocab": len(vocab["scaffolds"]),
            "background_only": baseline,
            "actual_pathway_target_scaffold": actual,
            "semantic_shuffled": shuffled,
            "semantic_random": random,
            "increments_vs_background_only": {
                "actual_pathway_target_scaffold": actual_inc,
                "semantic_shuffled": shuffled_inc,
                "semantic_random": random_inc,
            },
        },
        "reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM SciPlex Residual Semantic Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only train-only chemical holdout proxy.",
        "- Tests pathway/target/scaffold metadata over a background-only baseline.",
        "- Does not use Morgan descriptors, train, infer, launch GPU, read canonical multi, or read Track C query.",
        "",
        "## Summary",
        "",
        f"- train rows: `{len(train_rows)}`",
        f"- holdout rows: `{len(hold_rows)}`",
        f"- train/holdout drugs: `{len(train_drugs)}` / `{len(hold_drugs)}`",
        f"- pathway/target/scaffold vocab: `{len(vocab['pathways'])}` / `{len(vocab['targets'])}` / `{len(vocab['scaffolds'])}`",
        "",
        "| model | mean pp proxy | dataset min | increment vs background | increment dataset min |",
        "|---|---:|---:|---:|---:|",
        f"| `background_only` | {fmt(baseline['mean_pp_proxy'])} | {fmt(baseline['dataset_min'])} | NA | NA |",
        f"| `actual_pathway_target_scaffold` | {fmt(actual['mean_pp_proxy'])} | {fmt(actual['dataset_min'])} | {fmt(actual_inc['mean_increment'])} | {fmt(actual_inc['dataset_increment_min'])} |",
        f"| `semantic_shuffled` | {fmt(shuffled['mean_pp_proxy'])} | {fmt(shuffled['dataset_min'])} | {fmt(shuffled_inc['mean_increment'])} | {fmt(shuffled_inc['dataset_increment_min'])} |",
        f"| `semantic_random` | {fmt(random['mean_pp_proxy'])} | {fmt(random['dataset_min'])} | {fmt(random_inc['mean_increment'])} | {fmt(random_inc['dataset_increment_min'])} |",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        "- GPU authorized: `False`",
        "- If this fails, current SciPlex semantic metadata remains provenance/negative-control evidence, not a training candidate.",
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
