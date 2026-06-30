#!/usr/bin/env python3
"""CPU gate for signed perturbation-neighborhood consistency."""

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
OUT_JSON = ROOT / "reports/latentfm_signed_neighborhood_consistency_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SIGNED_NEIGHBORHOOD_CONSISTENCY_GATE_20260624.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
FEATURES = (
    "support_n",
    "support_dataset_n",
    "agreement",
    "consensus_norm",
    "update_cos_consensus",
    "update_projection",
    "anchor_delta_cos_consensus",
    "candidate_delta_cos_consensus",
)
ALIGNMENT_FEATURES = (
    "update_cos_consensus",
    "update_projection",
    "anchor_delta_cos_consensus",
    "candidate_delta_cos_consensus",
)
MAX_CELLS_PER_CONDITION = 256
SEED = 42
BOOT_N = 1000


@dataclass(frozen=True)
class EvalRow:
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


def unit(x: np.ndarray) -> np.ndarray:
    den = float(np.linalg.norm(x))
    if den <= 1e-12:
        return np.zeros_like(x)
    return x / den


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def metadata_genes(metadata: dict[str, dict[str, dict[str, Any]]], ds: str, cond: str) -> tuple[str, ...]:
    genes = metadata.get(ds, {}).get(cond, {}).get("genes") or []
    return tuple(sorted(str(g).upper() for g in genes if str(g)))


def train_deltas(split: dict[str, Any], metadata: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
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
                if cond not in cidx:
                    continue
                genes = metadata_genes(metadata, ds, cond)
                if not genes:
                    continue
                i = cidx[cond]
                c0, c1 = int(ctrl_offsets[i]), int(ctrl_offsets[i + 1])
                g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
                ctrl_arr = sample_slice(ctrl, c0, c1, key=f"ctrl|{ds}|{cond}")
                gt_arr = sample_slice(gt, g0, g1, key=f"gt|{ds}|{cond}")
                delta = np.mean(gt_arr, axis=0) - np.mean(ctrl_arr, axis=0)
                rows.append({"dataset": ds, "condition": cond, "genes": genes, "delta": delta, "unit": unit(delta)})
    return rows


def mean_rows(path: Path, group: str) -> dict[tuple[str, str], dict[str, Any]]:
    obj = load_json(path)
    return {
        (str(r["dataset"]), str(r["condition"])): r
        for r in obj["groups"][group]["condition_metrics"]
    }


def neighborhood_features(
    train_rows: list[dict[str, Any]],
    metadata: dict[str, dict[str, dict[str, Any]]],
    heldout_dataset: str,
    ds: str,
    cond: str,
    anchor_row: dict[str, Any],
    cap_row: dict[str, Any],
) -> dict[str, float]:
    genes = set(metadata_genes(metadata, ds, cond))
    support = [r for r in train_rows if r["dataset"] != heldout_dataset and genes.intersection(r["genes"])]
    if not support:
        return {f: 0.0 for f in FEATURES}
    units = np.stack([r["unit"] for r in support], axis=0)
    deltas = np.stack([r["delta"] for r in support], axis=0)
    consensus_unit = np.mean(units, axis=0)
    agreement = float(np.linalg.norm(consensus_unit))
    consensus = np.mean(deltas, axis=0)
    update = np.asarray(cap_row["pred_mean"], dtype=np.float64) - np.asarray(anchor_row["pred_mean"], dtype=np.float64)
    ctrl = np.asarray(anchor_row["ctrl_mean"], dtype=np.float64)
    anchor_delta = np.asarray(anchor_row["pred_mean"], dtype=np.float64) - ctrl
    candidate_delta = np.asarray(cap_row["pred_mean"], dtype=np.float64) - ctrl
    cu = unit(consensus_unit)
    return {
        "support_n": float(len(support)),
        "support_dataset_n": float(len({r["dataset"] for r in support})),
        "agreement": agreement,
        "consensus_norm": float(np.linalg.norm(consensus)),
        "update_cos_consensus": cosine(update, cu),
        "update_projection": float(np.dot(update, cu) / max(np.linalg.norm(update), 1e-12)),
        "anchor_delta_cos_consensus": cosine(anchor_delta, cu),
        "candidate_delta_cos_consensus": cosine(candidate_delta, cu),
    }


def build_rows(control: str = "main") -> list[EvalRow]:
    split = load_json(SPLIT_FILE)
    metadata = load_json(METADATA_FILE)
    tr = train_deltas(split, metadata)
    if control == "gene_shuffle":
        rng = random.Random(SEED + 501)
        genes = [r["genes"] for r in tr]
        rng.shuffle(genes)
        tr = [dict(r, genes=genes[i]) for i, r in enumerate(tr)]
    if control == "sign_inverted":
        tr = [dict(r, delta=-r["delta"], unit=-r["unit"]) for r in tr]
    rows = []
    for group in GROUPS:
        anchor = mean_rows(ANCHOR_MEANS, group)
        cap = mean_rows(CAP120_MEANS, group)
        for key in sorted(set(anchor) & set(cap)):
            ds, cond = key
            feats = neighborhood_features(tr, metadata, ds, ds, cond, anchor[key], cap[key])
            if control == "feature_shuffle":
                pass
            rows.append(
                EvalRow(
                    group=group,
                    dataset=ds,
                    condition=cond,
                    features=feats,
                    delta_pp=float(cap[key]["pearson_pert"] - anchor[key]["pearson_pert"]),
                    delta_mmd=float(cap[key]["test_mmd_clamped"] - anchor[key]["test_mmd_clamped"]),
                )
            )
    if control == "feature_shuffle":
        rng = random.Random(SEED + 777)
        feat_names = sorted({k for row in rows for k in row.features})
        shuffled: dict[str, list[float]] = {}
        for feat in feat_names:
            vals = [row.features.get(feat, float("nan")) for row in rows]
            rng.shuffle(vals)
            shuffled[feat] = vals
        rows = [
            EvalRow(row.group, row.dataset, row.condition, {f: shuffled[f][i] for f in feat_names}, row.delta_pp, row.delta_mmd)
            for i, row in enumerate(rows)
        ]
    return rows


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    rng = random.Random(SEED)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(len(arr)) for _ in arr]
        means.append(float(np.mean(arr[idx])))
    means_arr = np.asarray(means, dtype=np.float64)
    return float(np.quantile(means_arr, 0.025)), float(np.quantile(means_arr, 0.975)), float(np.mean(means_arr < 0.0))


def alpha_for(row: EvalRow, rule: Rule) -> float:
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


def apply_rule(rows: list[EvalRow], rule: Rule) -> list[dict[str, Any]]:
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


def candidate_rules(rows: list[EvalRow]) -> list[Rule]:
    rules = [Rule("noop", None, ">=", 0.0, 0.0, 0.0), Rule("all_candidate", None, ">=", 0.0, 1.0, 1.0)]
    for feat in FEATURES:
        vals = np.asarray([row.features.get(feat, float("nan")) for row in rows], dtype=np.float64)
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
    return (summary["mean_pp_delta"] - 5.0 * max(0.0, summary["mean_mmd_delta"]), -summary["bootstrap_p_harm"], summary["dataset_min_pp_delta"], -summary["condition_p_harm"])


def select_rule(train_rows: list[EvalRow]) -> tuple[Rule, dict[str, float]]:
    best_rule = Rule("noop", None, ">=", 0.0, 0.0, 0.0)
    best_summary = summarize(apply_rule(train_rows, best_rule), with_bootstrap=False)
    best_score = score(best_summary)
    for rule in candidate_rules(train_rows):
        s = summarize(apply_rule(train_rows, rule), with_bootstrap=False)
        sc = score(s)
        if sc > best_score:
            best_rule, best_summary, best_score = rule, s, sc
    return best_rule, best_summary


def nested_lodo(rows: list[EvalRow], control: str) -> dict[str, Any]:
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
    for control in ("gene_shuffle", "sign_inverted", "feature_shuffle"):
        c = by_key[(GROUPS[0], control)]["summary"]
        if c["mean_pp_delta"] >= 0.005:
            reasons.append(f"{control}_cross_not_collapsed")
    passed = not reasons
    return {
        "status": "signed_neighborhood_consistency_gate_pass_gpu_smoke_authorized" if passed else "signed_neighborhood_consistency_gate_fail_no_gpu",
        "gpu_authorized": passed,
        "reasons": reasons,
        "cross_mean_pp_delta": cross["mean_pp_delta"],
        "family_mean_pp_delta": family["mean_pp_delta"],
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Signed Neighborhood Consistency Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only nested leave-one-dataset-out gate.",
        "- Uses train-only H5 condition deltas, condition metadata, and completed cap120/anchor internal condition means.",
        "- Does not read canonical outcomes, canonical multi, Track C query, active logs, new GPU artifacts, or use GPU.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{payload['decision']['gpu_authorized']}`",
        f"- reasons: `{payload['decision']['reasons']}`",
        f"- cross pp delta: `{payload['decision']['cross_mean_pp_delta']:.6f}`",
        f"- family pp delta: `{payload['decision']['family_mean_pp_delta']:.6f}`",
        f"- train delta rows: `{payload['n_train_delta_rows']}`",
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
    n_train = len(train_deltas(split, metadata))
    results = []
    for control in ("main", "gene_shuffle", "sign_inverted", "feature_shuffle"):
        rows = build_rows(control)
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
            "seed": SEED,
        },
        "n_train_delta_rows": n_train,
        "results": results,
        "decision": decide(results),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(OUT_MD)


if __name__ == "__main__":
    main()
