#!/usr/bin/env python3
"""CPU gate for train-only bootstrap perturbation-target noise."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SPLIT_FILE = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
METADATA_FILE = DATA_DIR / "condition_metadata.json"
MEAN_DIR = ROOT / "reports/latentfm_xverse_nuisance_condition_means_20260624"
ANCHOR_MEANS = MEAN_DIR / "split_group_eval_anchor_internal_means_ode20.json"
CAP120_MEANS = MEAN_DIR / "split_group_eval_cap120_internal_means_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_bootstrap_target_noise_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_BOOTSTRAP_TARGET_NOISE_GATE_20260624.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
FEATURES = (
    "support_n",
    "support_dataset_n",
    "mean_boot_cos",
    "min_boot_cos",
    "mean_norm_cv",
    "max_norm_cv",
    "mean_delta_norm",
)
MAX_CELLS_PER_CONDITION = 128
BOOT_REPS = 32
SEED = 42
BOOT_N = 1000


@dataclass(frozen=True)
class TrainNoise:
    dataset: str
    condition: str
    genes: tuple[str, ...]
    features: dict[str, float]


@dataclass(frozen=True)
class EvalRow:
    group: str
    dataset: str
    condition: str
    genes: tuple[str, ...]
    delta_pp: float
    delta_mmd: float


@dataclass(frozen=True)
class FeatureRow:
    group: str
    dataset: str
    condition: str
    features: dict[str, float]
    delta_pp: float
    delta_mmd: float


@dataclass(frozen=True)
class Rule:
    name: str
    feature: str | None
    op: str
    threshold: float
    alpha_true: float
    alpha_false: float


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(key: str) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def decode(values: np.ndarray) -> list[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def sample_slice(arr: h5py.Dataset, lo: int, hi: int, *, key: str) -> np.ndarray:
    n = int(hi - lo)
    if n <= 0:
        raise ValueError(f"empty slice for {key}")
    if n > MAX_CELLS_PER_CONDITION:
        rng = np.random.default_rng(stable_seed(key))
        idx = np.sort(rng.choice(n, size=MAX_CELLS_PER_CONDITION, replace=False))
        return np.asarray(arr[lo + idx], dtype=np.float64)
    return np.asarray(arr[lo:hi], dtype=np.float64)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def metadata_genes(metadata: dict[str, dict[str, dict[str, Any]]], ds: str, cond: str) -> tuple[str, ...]:
    genes = metadata.get(ds, {}).get(cond, {}).get("genes") or []
    return tuple(sorted(str(g).upper() for g in genes if str(g)))


def bootstrap_noise_features(ctrl_arr: np.ndarray, gt_arr: np.ndarray, seed: int) -> dict[str, float]:
    full_delta = np.mean(gt_arr, axis=0) - np.mean(ctrl_arr, axis=0)
    rng = np.random.default_rng(seed)
    cosines = []
    norms = []
    for _ in range(BOOT_REPS):
        cidx = rng.integers(0, ctrl_arr.shape[0], size=ctrl_arr.shape[0])
        gidx = rng.integers(0, gt_arr.shape[0], size=gt_arr.shape[0])
        delta = np.mean(gt_arr[gidx], axis=0) - np.mean(ctrl_arr[cidx], axis=0)
        cos = cosine(delta, full_delta)
        if np.isfinite(cos):
            cosines.append(cos)
        norms.append(float(np.linalg.norm(delta)))
    norm_arr = np.asarray(norms, dtype=np.float64)
    return {
        "boot_cos_mean": float(np.mean(cosines)) if cosines else 0.0,
        "boot_cos_min": float(np.min(cosines)) if cosines else 0.0,
        "norm_cv": float(np.std(norm_arr) / max(float(np.mean(norm_arr)), 1e-12)),
        "delta_norm": float(np.linalg.norm(full_delta)),
    }


def train_noise_rows(split: dict[str, Any], metadata: dict[str, dict[str, dict[str, Any]]]) -> list[TrainNoise]:
    rows: list[TrainNoise] = []
    for ds, groups in split.items():
        h5_path = DATA_DIR / f"{ds}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as h5:
            cidx = {c: i for i, c in enumerate(decode(np.asarray(h5["conditions"])))}
            ctrl = h5["ctrl/emb"] if "ctrl/emb" in h5 else h5["ir/emb"]
            gt = h5["gt/emb"]
            ctrl_offsets = np.asarray(h5["ctrl/offsets"] if "ctrl/offsets" in h5 else h5["ir/offsets"])
            gt_offsets = np.asarray(h5["gt/offsets"])
            for cond in groups.get("train", []):
                cond = str(cond)
                genes = metadata_genes(metadata, ds, cond)
                if not genes or cond not in cidx:
                    continue
                i = cidx[cond]
                c0, c1 = int(ctrl_offsets[i]), int(ctrl_offsets[i + 1])
                g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
                ctrl_arr = sample_slice(ctrl, c0, c1, key=f"ctrl|{ds}|{cond}")
                gt_arr = sample_slice(gt, g0, g1, key=f"gt|{ds}|{cond}")
                feats = bootstrap_noise_features(ctrl_arr, gt_arr, stable_seed(f"boot|{ds}|{cond}"))
                rows.append(TrainNoise(ds, cond, genes, feats))
    return rows


def mean_rows(path: Path, group: str) -> dict[tuple[str, str], dict[str, Any]]:
    obj = load_json(path)
    return {
        (str(r["dataset"]), str(r["condition"])): r
        for r in obj["groups"][group]["condition_metrics"]
    }


def eval_rows(metadata: dict[str, dict[str, dict[str, Any]]]) -> list[EvalRow]:
    rows = []
    for group in GROUPS:
        anchor = mean_rows(ANCHOR_MEANS, group)
        cap = mean_rows(CAP120_MEANS, group)
        for key in sorted(set(anchor) & set(cap)):
            genes = metadata_genes(metadata, key[0], key[1])
            if not genes:
                continue
            rows.append(
                EvalRow(
                    group=group,
                    dataset=key[0],
                    condition=key[1],
                    genes=genes,
                    delta_pp=float(cap[key]["pearson_pert"] - anchor[key]["pearson_pert"]),
                    delta_mmd=float(cap[key]["test_mmd_clamped"] - anchor[key]["test_mmd_clamped"]),
                )
            )
    return rows


def aggregate_features(noise_rows: list[TrainNoise], eval_row: EvalRow, heldout_dataset: str) -> dict[str, float]:
    genes = set(eval_row.genes)
    support = [r for r in noise_rows if r.dataset != heldout_dataset and genes.intersection(r.genes)]
    if not support:
        return {f: 0.0 for f in FEATURES}
    cos = [r.features["boot_cos_mean"] for r in support]
    min_cos = [r.features["boot_cos_min"] for r in support]
    cv = [r.features["norm_cv"] for r in support]
    norms = [r.features["delta_norm"] for r in support]
    return {
        "support_n": float(len(support)),
        "support_dataset_n": float(len({r.dataset for r in support})),
        "mean_boot_cos": float(np.mean(cos)),
        "min_boot_cos": float(np.min(min_cos)),
        "mean_norm_cv": float(np.mean(cv)),
        "max_norm_cv": float(np.max(cv)),
        "mean_delta_norm": float(np.mean(norms)),
    }


def feature_rows(noise_rows: list[TrainNoise], evals: list[EvalRow], control: str) -> list[FeatureRow]:
    work_noise = noise_rows
    if control == "gene_shuffle":
        rng = random.Random(SEED + 91)
        genes = [r.genes for r in work_noise]
        rng.shuffle(genes)
        work_noise = [TrainNoise(r.dataset, r.condition, genes[i], r.features) for i, r in enumerate(work_noise)]
    elif control == "feature_shuffle":
        rng = random.Random(SEED + 92)
        feature_names = sorted({k for r in work_noise for k in r.features})
        shuffled: dict[str, list[float]] = {}
        for feat in feature_names:
            vals = [r.features[feat] for r in work_noise]
            rng.shuffle(vals)
            shuffled[feat] = vals
        work_noise = [
            TrainNoise(r.dataset, r.condition, r.genes, {feat: shuffled[feat][i] for feat in feature_names})
            for i, r in enumerate(work_noise)
        ]
    elif control == "inverted_noise":
        work_noise = [
            TrainNoise(
                r.dataset,
                r.condition,
                r.genes,
                {
                    "boot_cos_mean": -r.features["boot_cos_mean"],
                    "boot_cos_min": -r.features["boot_cos_min"],
                    "norm_cv": -r.features["norm_cv"],
                    "delta_norm": -r.features["delta_norm"],
                },
            )
            for r in work_noise
        ]
    out = []
    for row in evals:
        feats = aggregate_features(work_noise, row, row.dataset)
        out.append(FeatureRow(row.group, row.dataset, row.condition, feats, row.delta_pp, row.delta_mmd))
    return out


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(SEED)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(len(arr)) for _ in arr]
        means.append(float(np.mean(arr[idx])))
    means_arr = np.asarray(means, dtype=np.float64)
    return float(np.quantile(means_arr, 0.025)), float(np.quantile(means_arr, 0.975)), float(np.mean(means_arr < 0.0))


def alpha_for(row: FeatureRow, rule: Rule) -> float:
    if rule.name == "noop":
        return 0.0
    if rule.name == "all_candidate":
        return 1.0
    assert rule.feature is not None
    value = row.features.get(rule.feature, float("nan"))
    if not np.isfinite(value):
        return 0.0
    hit = value <= rule.threshold if rule.op == "<=" else value >= rule.threshold
    return rule.alpha_true if hit else rule.alpha_false


def apply_rule(rows: list[FeatureRow], rule: Rule) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        alpha = alpha_for(row, rule)
        out.append({"dataset": row.dataset, "delta_pp": alpha * row.delta_pp, "delta_mmd": alpha * row.delta_mmd, "alpha": alpha})
    return out


def summarize(applied: list[dict[str, Any]], *, with_bootstrap: bool = True) -> dict[str, float]:
    if not applied:
        return {k: float("nan") for k in ("mean_pp_delta", "ci95_low", "ci95_high", "bootstrap_p_harm", "condition_p_harm", "dataset_min_pp_delta", "mean_mmd_delta", "mean_alpha")} | {"n": 0}
    vals = np.asarray([float(r["delta_pp"]) for r in applied], dtype=np.float64)
    lo, hi, p_harm = bootstrap([float(v) for v in vals]) if with_bootstrap else (float("nan"), float("nan"), float(np.mean(vals < 0.0)))
    by_ds: dict[str, list[float]] = {}
    for row in applied:
        by_ds.setdefault(str(row["dataset"]), []).append(float(row["delta_pp"]))
    return {
        "n": len(applied),
        "mean_pp_delta": float(np.mean(vals)),
        "ci95_low": lo,
        "ci95_high": hi,
        "bootstrap_p_harm": p_harm,
        "condition_p_harm": float(np.mean(vals < 0.0)),
        "dataset_min_pp_delta": float(min(sum(v) / len(v) for v in by_ds.values())),
        "mean_mmd_delta": float(np.mean([float(r["delta_mmd"]) for r in applied])),
        "mean_alpha": float(np.mean([float(r["alpha"]) for r in applied])),
    }


def candidate_rules(rows: list[FeatureRow]) -> list[Rule]:
    rules = [Rule("noop", None, ">=", 0.0, 0.0, 0.0), Rule("all_candidate", None, ">=", 0.0, 1.0, 1.0)]
    for feat in FEATURES:
        vals = np.asarray([r.features.get(feat, float("nan")) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size < 12 or float(np.max(vals) - np.min(vals)) <= 1e-12:
            continue
        for q in (0.25, 0.5, 0.75):
            threshold = float(np.quantile(vals, q))
            for op in ("<=", ">="):
                for alpha_true, alpha_false in ((1.0, 0.0), (0.0, 1.0), (0.75, 0.0)):
                    rules.append(Rule(f"{feat}_{op}_{threshold:.5g}_a{alpha_true:.2f}_{alpha_false:.2f}", feat, op, threshold, alpha_true, alpha_false))
    return rules


def score(summary: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        summary["mean_pp_delta"] - 5.0 * max(0.0, summary["mean_mmd_delta"]),
        summary["dataset_min_pp_delta"],
        -summary["bootstrap_p_harm"],
        -summary["condition_p_harm"],
    )


def select_rule(train_rows: list[FeatureRow]) -> tuple[Rule, dict[str, float]]:
    best_rule = Rule("noop", None, ">=", 0.0, 0.0, 0.0)
    best_summary = summarize(apply_rule(train_rows, best_rule), with_bootstrap=False)
    best_score = score(best_summary)
    for rule in candidate_rules(train_rows):
        s = summarize(apply_rule(train_rows, rule), with_bootstrap=False)
        sc = score(s)
        if sc > best_score:
            best_rule, best_summary, best_score = rule, s, sc
    return best_rule, best_summary


def nested_lodo(rows: list[FeatureRow], control: str) -> dict[str, Any]:
    applied_all = []
    folds = []
    for heldout in sorted({r.dataset for r in rows}):
        train = [r for r in rows if r.dataset != heldout]
        test = [r for r in rows if r.dataset == heldout]
        if len(train) < 12 or not test:
            continue
        rule, train_summary = select_rule(train)
        applied = apply_rule(test, rule)
        test_summary = summarize(applied, with_bootstrap=False)
        applied_all.extend(applied)
        folds.append({"heldout_dataset": heldout, "rule": rule.name, "train_mean_pp_delta": train_summary["mean_pp_delta"], "test_mean_pp_delta": test_summary["mean_pp_delta"], "test_mean_alpha": test_summary["mean_alpha"]})
    top_rules: dict[str, int] = {}
    for fold in folds:
        top_rules[fold["rule"]] = top_rules.get(fold["rule"], 0) + 1
    return {"control": control, "summary": summarize(applied_all), "folds": folds, "top_rules": sorted(top_rules.items(), key=lambda kv: (-kv[1], kv[0]))[:8]}


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(r["group"], r["control"]): r for r in results}
    cross = by_key[(GROUPS[0], "main")]["summary"]
    family = by_key[(GROUPS[1], "main")]["summary"]
    reasons = []
    if cross["mean_pp_delta"] < 0.010:
        reasons.append("cross_pp_delta_below_0.010")
    if family["mean_pp_delta"] < 0.010:
        reasons.append("family_pp_delta_below_0.010")
    if cross["dataset_min_pp_delta"] < -0.020:
        reasons.append("cross_dataset_min_below_minus_0.020")
    if family["dataset_min_pp_delta"] < -0.020:
        reasons.append("family_dataset_min_below_minus_0.020")
    if family["mean_mmd_delta"] > 0.0005:
        reasons.append("family_mmd_worse_above_0.0005")
    for control in ("gene_shuffle", "feature_shuffle", "inverted_noise"):
        c = by_key[(GROUPS[0], control)]["summary"]
        if c["mean_pp_delta"] >= 0.005:
            reasons.append(f"{control}_cross_not_collapsed")
    passed = not reasons
    return {
        "status": "bootstrap_target_noise_gate_pass_gpu_smoke_authorized" if passed else "bootstrap_target_noise_gate_fail_no_gpu",
        "gpu_authorized": passed,
        "reasons": reasons,
        "cross_mean_pp_delta": cross["mean_pp_delta"],
        "family_mean_pp_delta": family["mean_pp_delta"],
        "cross_dataset_min": cross["dataset_min_pp_delta"],
        "family_dataset_min": family["dataset_min_pp_delta"],
    }


def render_md(payload: dict[str, Any]) -> str:
    d = payload["decision"]
    lines = [
        "# LatentFM Bootstrap Target-Noise Gate",
        "",
        f"Status: `{d['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only nested leave-one-dataset-out gate.",
        "- Bootstrap noise features are computed only from train split H5 conditions.",
        "- Internal validation rows receive same-gene train-neighborhood noise aggregates excluding the eval dataset.",
        "- Uses completed cap120/anchor internal condition means only.",
        "- Does not read canonical outcomes, canonical multi, Track C query, active logs, new GPU artifacts, or use GPU.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{d['gpu_authorized']}`",
        f"- reasons: `{d['reasons']}`",
        f"- cross pp delta: `{d['cross_mean_pp_delta']:.6f}`",
        f"- family pp delta: `{d['family_mean_pp_delta']:.6f}`",
        f"- cross dataset-min: `{d['cross_dataset_min']:.6f}`",
        f"- family dataset-min: `{d['family_dataset_min']:.6f}`",
        f"- train noise rows: `{payload['n_train_noise_rows']}`",
        "",
        "## Nested LODO Summaries",
        "",
        "| group | control | n | mean pp delta | 95% CI | p_harm | dataset min | mean MMD delta | mean alpha | top rules |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        s = row["summary"]
        top = "; ".join(f"{name}:{count}" for name, count in row["top_rules"])
        lines.append(f"| `{row['group']}` | `{row['control']}` | {s['n']} | {s['mean_pp_delta']:.6f} | [{s['ci95_low']:.6f}, {s['ci95_high']:.6f}] | {s['bootstrap_p_harm']:.3f} | {s['dataset_min_pp_delta']:.6f} | {s['mean_mmd_delta']:.6f} | {s['mean_alpha']:.3f} | {top} |")
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`", ""])
    return "\n".join(lines)


def main() -> None:
    split = load_json(SPLIT_FILE)
    metadata = load_json(METADATA_FILE)
    noise = train_noise_rows(split, metadata)
    evals = eval_rows(metadata)
    results = []
    for control in ("main", "gene_shuffle", "feature_shuffle", "inverted_noise"):
        rows = feature_rows(noise, evals, control)
        for group in GROUPS:
            result = nested_lodo([r for r in rows if r.group == group], control)
            result["group"] = group
            results.append(result)
    payload = {
        "boundary": {
            "split_file": str(SPLIT_FILE),
            "metadata_file": str(METADATA_FILE),
            "anchor_means": str(ANCHOR_MEANS),
            "cap120_means": str(CAP120_MEANS),
            "max_cells_per_condition": MAX_CELLS_PER_CONDITION,
            "boot_reps": BOOT_REPS,
            "seed": SEED,
        },
        "n_train_noise_rows": len(noise),
        "results": results,
        "decision": decide(results),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(OUT_MD)


if __name__ == "__main__":
    main()
