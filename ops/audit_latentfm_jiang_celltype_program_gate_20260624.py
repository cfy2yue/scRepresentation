#!/usr/bin/env python3
"""CPU-only Jiang cell-type response-program prior gate.

This gate tests whether Jiang perturbation response programs stratified by
cell_type can safely route anchor vs gene baselines on train-only/internal
proxy rows. It is query-blind and does not read canonical outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
BIFLOW = ROOT / "dataset/biFlow_data"
SPLIT_FILE = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
FORENSICS_JSON = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_20260622.json"
OUT_JSON = ROOT / "reports/latentfm_jiang_celltype_program_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_JIANG_CELLTYPE_PROGRAM_GATE_20260624.md"

JIANG_DATASETS = (
    "Jiang_IFNB",
    "Jiang_IFNG",
    "Jiang_INS",
    "Jiang_TGFB",
    "Jiang_TNFA",
)
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
FEATURES = (
    "program_mean_norm",
    "program_max_norm",
    "program_std_norm",
    "program_cv_norm",
    "program_dominant_share",
    "program_mean_pair_cos",
    "program_min_pair_cos",
    "program_n_celltypes",
    "program_total_cells",
)
MAX_CELLS_PER_TYPE = 128
BOOT_N = 2000
SEED = 20260624


@dataclass(frozen=True)
class Row:
    group: str
    dataset: str
    condition: str
    features: dict[str, float]
    anchor_pp: float
    gene_pp: float


@dataclass(frozen=True)
class Rule:
    name: str
    feature: str | None
    op: str
    threshold: float
    use_anchor_true: bool


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def sample_indices(indices: np.ndarray, key: str) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= MAX_CELLS_PER_TYPE:
        return indices
    rng = np.random.default_rng(stable_seed(key))
    return np.sort(rng.choice(indices, size=MAX_CELLS_PER_TYPE, replace=False))


def mean_emb(a: ad.AnnData, indices: np.ndarray, key: str) -> np.ndarray:
    idx = sample_indices(indices, key)
    return np.asarray(a.obsm["emb"][idx], dtype=np.float64).mean(axis=0)


def pair_cosines(vectors: list[np.ndarray]) -> list[float]:
    out = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            if den > 1e-12:
                out.append(float(np.dot(a, b) / den))
    return out


def compute_dataset_features(dataset: str, needed_conditions: set[str]) -> dict[tuple[str, str], dict[str, float]]:
    gt_path = BIFLOW / "gt_stack" / f"{dataset}.h5ad"
    ctrl_path = BIFLOW / "control_stack" / f"{dataset}.h5ad"
    gt = ad.read_h5ad(gt_path, backed="r")
    ctrl = ad.read_h5ad(ctrl_path, backed="r")
    try:
        gt_obs = gt.obs[["perturbation", "cell_type"]].copy()
        ctrl_obs = ctrl.obs[["cell_type"]].copy()
        gt_obs["perturbation"] = gt_obs["perturbation"].astype(str)
        gt_obs["cell_type"] = gt_obs["cell_type"].astype(str)
        ctrl_obs["cell_type"] = ctrl_obs["cell_type"].astype(str)

        ctrl_by_type: dict[str, np.ndarray] = {}
        for cell_type, sub in ctrl_obs.groupby("cell_type", observed=True):
            idx = sub.index.to_numpy()
            pos = ctrl_obs.index.get_indexer(idx)
            pos = pos[pos >= 0]
            if pos.size:
                ctrl_by_type[str(cell_type)] = mean_emb(ctrl, pos, f"ctrl|{dataset}|{cell_type}")

        out: dict[tuple[str, str], dict[str, float]] = {}
        for cond in sorted(needed_conditions):
            sub_cond = gt_obs.index[gt_obs["perturbation"] == str(cond)]
            if len(sub_cond) == 0:
                continue
            vectors = []
            counts = []
            for cell_type, sub in gt_obs.loc[sub_cond].groupby("cell_type", observed=True):
                ct = str(cell_type)
                if ct not in ctrl_by_type:
                    continue
                pos = gt_obs.index.get_indexer(sub.index)
                pos = pos[pos >= 0]
                if pos.size == 0:
                    continue
                gt_mean = mean_emb(gt, pos, f"gt|{dataset}|{cond}|{ct}")
                vectors.append(gt_mean - ctrl_by_type[ct])
                counts.append(int(pos.size))
            if not vectors:
                continue
            norms = np.asarray([float(np.linalg.norm(v)) for v in vectors], dtype=np.float64)
            total = float(sum(counts))
            cos = pair_cosines(vectors)
            out[(dataset, cond)] = {
                "program_mean_norm": float(np.mean(norms)),
                "program_max_norm": float(np.max(norms)),
                "program_std_norm": float(np.std(norms)),
                "program_cv_norm": float(np.std(norms) / (np.mean(norms) + 1e-8)),
                "program_dominant_share": float(np.max(counts) / max(1.0, total)),
                "program_mean_pair_cos": float(np.mean(cos)) if cos else 1.0,
                "program_min_pair_cos": float(np.min(cos)) if cos else 1.0,
                "program_n_celltypes": float(len(vectors)),
                "program_total_cells": total,
            }
        return out
    finally:
        gt.file.close()
        ctrl.file.close()


def compute_features() -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    split = read_json(SPLIT_FILE)
    needed: dict[str, set[str]] = {}
    for ds in JIANG_DATASETS:
        groups = split.get(ds) or {}
        conds = set(str(c) for c in groups.get("train", []))
        for group in GROUPS:
            conds.update(str(c) for c in groups.get(group, []))
        needed[ds] = conds
    features: dict[tuple[str, str], dict[str, float]] = {}
    for ds, conds in needed.items():
        features.update(compute_dataset_features(ds, conds))
    return features, {
        "n_feature_rows": len(features),
        "n_datasets": len({k[0] for k in features}),
        "feature_boundary": "Jiang gt/control h5ad obsm['emb'] response programs grouped by perturbation and cell_type; no canonical/query/model artifacts",
    }


def metric_rows(features: dict[tuple[str, str], dict[str, float]]) -> list[Row]:
    obj = read_json(FORENSICS_JSON)
    rows = []
    for raw in obj["condition_rows"]:
        key = (str(raw["dataset"]), str(raw["condition"]))
        if key not in features or key[0] not in JIANG_DATASETS:
            continue
        rows.append(Row(
            group=str(raw["group"]),
            dataset=key[0],
            condition=key[1],
            features=features[key],
            anchor_pp=float(raw["anchor_pearson_pert"]),
            gene_pp=float(raw["gene_raw_mean"]),
        ))
    return rows


def alpha_for(row: Row, rule: Rule) -> bool:
    if rule.name == "all_gene":
        return False
    if rule.name == "all_anchor":
        return True
    assert rule.feature is not None
    val = row.features.get(rule.feature, float("nan"))
    if not np.isfinite(val):
        return False
    hit = val <= rule.threshold if rule.op == "<=" else val >= rule.threshold
    return bool(rule.use_anchor_true if hit else (not rule.use_anchor_true))


def apply_rule(rows: list[Row], rule: Rule) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        use_anchor = alpha_for(row, rule)
        pred = row.anchor_pp if use_anchor else row.gene_pp
        out.append({
            "dataset": row.dataset,
            "condition": row.condition,
            "delta_vs_gene": float(pred - row.gene_pp),
            "delta_vs_anchor": float(pred - row.anchor_pp),
            "use_anchor": bool(use_anchor),
        })
    return out


def summarize(applied: list[dict[str, Any]]) -> dict[str, float]:
    if not applied:
        return {"n": 0, "delta_vs_gene": float("nan"), "dataset_min": float("nan"), "harm_frac": float("nan"), "use_anchor_fraction": 0.0}
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in applied:
        by_ds[row["dataset"]].append(float(row["delta_vs_gene"]))
    ds_vals = [float(np.mean(v)) for v in by_ds.values()]
    return {
        "n": len(applied),
        "delta_vs_gene": float(np.mean(ds_vals)),
        "dataset_min": float(min(ds_vals)),
        "harm_frac": float(np.mean([v < 0.0 for v in ds_vals])),
        "use_anchor_fraction": float(np.mean([row["use_anchor"] for row in applied])),
        "delta_vs_anchor": float(np.mean([row["delta_vs_anchor"] for row in applied])),
    }


def candidate_rules(rows: list[Row], features: tuple[str, ...]) -> list[Rule]:
    rules = [Rule("all_gene", None, ">=", 0.0, False), Rule("all_anchor", None, ">=", 0.0, True)]
    for feat in features:
        vals = np.asarray([r.features.get(feat, float("nan")) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size < 8 or float(np.max(vals) - np.min(vals)) <= 1e-12:
            continue
        for q in (0.2, 0.35, 0.5, 0.65, 0.8):
            thr = float(np.quantile(vals, q))
            for op in ("<=", ">="):
                for use_anchor_true in (True, False):
                    rules.append(Rule(f"{feat}_{op}_{thr:.5g}_anchor{int(use_anchor_true)}", feat, op, thr, use_anchor_true))
    return rules


def score(summary: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        0.05 <= summary["use_anchor_fraction"] <= 0.50,
        summary["dataset_min"],
        summary["delta_vs_gene"],
        -abs(summary["use_anchor_fraction"] - 0.20),
    )


def select_rule(rows: list[Row], features: tuple[str, ...]) -> Rule:
    best_rule = None
    best_score = None
    for rule in candidate_rules(rows, features):
        s = summarize(apply_rule(rows, rule))
        sc = score(s)
        if best_score is None or sc > best_score:
            best_rule, best_score = rule, sc
    assert best_rule is not None
    return best_rule


def transform(rows: list[Row], control: str) -> list[Row]:
    if control == "main":
        return rows
    if control == "broad_cytokine":
        return rows
    rng = random.Random(SEED + 733)
    feat_names = sorted({k for row in rows for k in row.features})
    shuf = {}
    for feat in feat_names:
        vals = [row.features[feat] for row in rows]
        rng.shuffle(vals)
        shuf[feat] = vals
    out = []
    for i, row in enumerate(rows):
        feats = {feat: shuf[feat][i] for feat in feat_names}
        out.append(Row(row.group, row.dataset, row.condition, feats, row.anchor_pp, row.gene_pp))
    return out


def nested_lodo(rows: list[Row], control: str) -> dict[str, Any]:
    work = transform(rows, control)
    applied_all = []
    folds = []
    for ds in sorted({r.dataset for r in work}):
        train = [r for r in work if r.dataset != ds]
        test = [r for r in work if r.dataset == ds]
        if not train or not test:
            continue
        if control == "broad_cytokine":
            rule = Rule("all_anchor", None, ">=", 0.0, True)
        else:
            rule = select_rule(train, FEATURES)
        applied = apply_rule(test, rule)
        applied_all.extend(applied)
        folds.append({"heldout_dataset": ds, "rule": rule.name, "test": summarize(applied)})
    return {"control": control, "summary": summarize(applied_all), "folds": folds}


def paired_bootstrap(applied: list[dict[str, Any]]) -> tuple[float, float, float]:
    if not applied:
        return float("nan"), float("nan"), float("nan")
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in applied:
        by_ds[row["dataset"]].append(float(row["delta_vs_gene"]))
    keys = sorted(by_ds)
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(BOOT_N):
        sample = rng.choice(keys, size=len(keys), replace=True)
        vals.append(float(np.mean([np.mean(rng.choice(by_ds[str(ds)], size=len(by_ds[str(ds)]), replace=True)) for ds in sample])))
    arr = np.asarray(vals)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(np.mean(arr < 0.0))


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    by_key = {(r["group"], r["control"]): r["summary"] for r in results}
    for group in GROUPS:
        main = by_key.get((group, "main"), {})
        shuf = by_key.get((group, "shuffled"), {})
        broad = by_key.get((group, "broad_cytokine"), {})
        if not main:
            reasons.append(f"{group}_missing_main")
            continue
        if float(main["delta_vs_gene"]) < 0.020:
            reasons.append(f"{group}_jiang_delta_vs_gene_below_0p020")
        if float(main["dataset_min"]) < -0.020:
            reasons.append(f"{group}_dataset_min_below_minus_0p02")
        if float(main["harm_frac"]) > 0.20:
            reasons.append(f"{group}_harm_frac_above_0p20")
        if shuf and float(main["delta_vs_gene"]) < float(shuf["delta_vs_gene"]) + 0.010:
            reasons.append(f"{group}_shuffled_not_beaten_by_0p010")
        if broad and float(main["delta_vs_gene"]) < float(broad["delta_vs_gene"]) + 0.010:
            reasons.append(f"{group}_broad_cytokine_control_not_beaten_by_0p010")
    return {
        "status": "jiang_celltype_program_gate_pass_gpu_smoke_authorized" if not reasons else "jiang_celltype_program_gate_fail_no_gpu",
        "gpu_authorized": not reasons,
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Jiang Cell-Type Response Program Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only nested leave-one-Jiang-dataset-out gate.",
        "- Reads Jiang gt/control h5ad `.obs[['perturbation','cell_type']]` and `.obsm['emb']` only.",
        "- Uses residual-forensics internal proxy rows for anchor-vs-gene scoring.",
        "- Does not read canonical outcomes, canonical multi, Track C query, active logs, or GPU artifacts.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{payload['decision']['gpu_authorized']}`",
        f"- reasons: `{payload['decision']['reasons']}`",
        f"- feature rows: `{payload['feature_meta']['n_feature_rows']}`",
        f"- metric rows: `{payload['n_metric_rows']}`",
        "",
        "## Results",
        "",
        "| group | control | n | use anchor | delta vs gene | dataset min | harm frac | delta vs anchor |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]:
        s = row["summary"]
        lines.append(f"| `{row['group']}` | `{row['control']}` | {s['n']} | {s['use_anchor_fraction']:.3f} | {s['delta_vs_gene']:+.6f} | {s['dataset_min']:+.6f} | {s['harm_frac']:.3f} | {s['delta_vs_anchor']:+.6f} |")
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`", ""])
    return "\n".join(lines)


def main() -> int:
    features, meta = compute_features()
    rows = metric_rows(features)
    results = []
    for group in GROUPS:
        subset = [r for r in rows if r.group == group]
        for control in ("main", "shuffled", "broad_cytokine"):
            r = nested_lodo(subset, control)
            r["group"] = group
            results.append(r)
    payload = {
        "boundary": {
            "split_file": str(SPLIT_FILE),
            "forensics_json": str(FORENSICS_JSON),
            "datasets": JIANG_DATASETS,
            "features": FEATURES,
            "max_cells_per_type": MAX_CELLS_PER_TYPE,
        },
        "feature_meta": meta,
        "n_metric_rows": len(rows),
        "results": results,
        "decision": decide(results),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
