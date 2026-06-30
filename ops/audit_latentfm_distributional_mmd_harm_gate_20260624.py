#!/usr/bin/env python3
"""CPU-only train-only distributional MMD-harm safety gate.

This gate tests whether condition-level distributional proxies can route the
completed general-exposure candidate back to anchor on high-MMD-risk rows while
preserving internal Pearson gains. It reads only train-only/internal artifacts.
"""

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
SPLIT_FILE = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_general_exposure_cap_v2.json"
RUN_DIR = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_general_exposure_cap_v2_3k_seed42/posthoc_eval_internal"
ANCHOR_JSON = RUN_DIR / "split_group_eval_anchor_internal_ode20.json"
CAND_JSON = RUN_DIR / "split_group_eval_candidate_internal_ode20.json"
FAILURE_JSON = ROOT / "reports/latentfm_xverse_general_exposure_failure_cases_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_distributional_mmd_harm_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_DISTRIBUTIONAL_MMD_HARM_GATE_20260624.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
RAW_FEATURES = (
    "log_n_gt",
    "log_n_ctrl",
    "response_norm",
    "sem_proxy",
    "snr_proxy",
    "ctrl_var_mean",
    "gt_var_mean",
    "log_var_ratio",
    "abs_var_shift",
    "ctrl_eff_rank_diag",
    "gt_eff_rank_diag",
    "eff_rank_shift",
    "ctrl_tail95",
    "gt_tail95",
    "tail95_shift",
    "tail99_shift",
    "tail_risk_ratio",
)
COUNT_FEATURES = {"log_n_gt", "log_n_ctrl", "ds_z_log_n_gt", "ds_z_log_n_ctrl", "ds_pct_log_n_gt", "ds_pct_log_n_ctrl"}
MAX_CELLS_PER_CONDITION = 384
SEED = 42
BOOT_N = 1000


@dataclass(frozen=True)
class MetricRow:
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


def decode_conditions(values: np.ndarray) -> list[str]:
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


def diag_effective_rank(var: np.ndarray) -> float:
    v = np.asarray(var, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return 0.0
    p = v / v.sum()
    return float(math.exp(-float(np.sum(p * np.log(p + 1e-12)))))


def condition_features(h5: h5py.File, ds: str, cond: str, cidx: dict[str, int]) -> dict[str, float] | None:
    if cond not in cidx:
        return None
    i = cidx[cond]
    ctrl_key = "ctrl" if "ctrl/emb" in h5 else "ir"
    ctrl = h5[f"{ctrl_key}/emb"]
    gt = h5["gt/emb"]
    ctrl_offsets = np.asarray(h5[f"{ctrl_key}/offsets"])
    gt_offsets = np.asarray(h5["gt/offsets"])
    c0, c1 = int(ctrl_offsets[i]), int(ctrl_offsets[i + 1])
    g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
    ctrl_arr = sample_slice(ctrl, c0, c1, key=f"ctrl|{ds}|{cond}")
    gt_arr = sample_slice(gt, g0, g1, key=f"gt|{ds}|{cond}")

    ctrl_mean = ctrl_arr.mean(axis=0)
    gt_mean = gt_arr.mean(axis=0)
    delta = gt_mean - ctrl_mean
    response_norm = float(np.linalg.norm(delta))
    ctrl_var = np.var(ctrl_arr, axis=0)
    gt_var = np.var(gt_arr, axis=0)
    ctrl_var_mean = float(np.mean(ctrl_var))
    gt_var_mean = float(np.mean(gt_var))
    sem = math.sqrt(ctrl_var_mean / max(1, len(ctrl_arr)) + gt_var_mean / max(1, len(gt_arr)))
    ctrl_dist = np.linalg.norm(ctrl_arr - ctrl_mean[None, :], axis=1)
    gt_dist = np.linalg.norm(gt_arr - ctrl_mean[None, :], axis=1)
    ctrl_tail95 = float(np.quantile(ctrl_dist, 0.95))
    gt_tail95 = float(np.quantile(gt_dist, 0.95))
    tail99_shift = float(np.quantile(gt_dist, 0.99) - np.quantile(ctrl_dist, 0.99))
    tail95_shift = float(gt_tail95 - ctrl_tail95)
    return {
        "n_ctrl": float(c1 - c0),
        "n_gt": float(g1 - g0),
        "log_n_ctrl": float(math.log1p(c1 - c0)),
        "log_n_gt": float(math.log1p(g1 - g0)),
        "response_norm": response_norm,
        "sem_proxy": float(sem),
        "snr_proxy": float(response_norm / (sem + 1e-8)),
        "ctrl_var_mean": ctrl_var_mean,
        "gt_var_mean": gt_var_mean,
        "log_var_ratio": float(math.log((gt_var_mean + 1e-12) / (ctrl_var_mean + 1e-12))),
        "abs_var_shift": float(abs(gt_var_mean - ctrl_var_mean)),
        "ctrl_eff_rank_diag": diag_effective_rank(ctrl_var),
        "gt_eff_rank_diag": diag_effective_rank(gt_var),
        "eff_rank_shift": float(diag_effective_rank(gt_var) - diag_effective_rank(ctrl_var)),
        "ctrl_tail95": ctrl_tail95,
        "gt_tail95": gt_tail95,
        "tail95_shift": tail95_shift,
        "tail99_shift": tail99_shift,
        "tail_risk_ratio": float(tail95_shift / (response_norm + sem + 1e-8)),
    }


def median_mad(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad if mad > 1e-12 else 1.0


def percentile(value: float, values: list[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(np.mean(arr <= value))


def compute_features() -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    split = load_json(SPLIT_FILE)
    needed: dict[str, set[str]] = {}
    train_by_ds: dict[str, set[str]] = {}
    for ds, groups in split.items():
        s = set(str(c) for c in groups.get("train", []))
        for group in GROUPS:
            s.update(str(c) for c in groups.get(group, []))
        needed[ds] = s
        train_by_ds[ds] = set(str(c) for c in groups.get("train", []))

    raw: dict[tuple[str, str], dict[str, float]] = {}
    for ds, conds in needed.items():
        h5_path = DATA_DIR / f"{ds}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as h5:
            cidx = {c: i for i, c in enumerate(decode_conditions(np.asarray(h5["conditions"])))}
            for cond in sorted(conds):
                feats = condition_features(h5, ds, cond, cidx)
                if feats is not None:
                    raw[(ds, cond)] = feats

    train_values: dict[str, dict[str, list[float]]] = {}
    normalizers: dict[str, dict[str, tuple[float, float]]] = {}
    for ds, conds in train_by_ds.items():
        rows = [raw[(ds, c)] for c in conds if (ds, c) in raw]
        train_values[ds] = {}
        normalizers[ds] = {}
        for feat in RAW_FEATURES:
            vals = [float(r[feat]) for r in rows]
            train_values[ds][feat] = vals
            normalizers[ds][feat] = median_mad(vals)

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, feats in raw.items():
        ds, _ = key
        enriched = dict(feats)
        for feat in RAW_FEATURES:
            med, mad = normalizers.get(ds, {}).get(feat, (0.0, 1.0))
            enriched[f"ds_z_{feat}"] = float((float(feats[feat]) - med) / mad)
            enriched[f"ds_pct_{feat}"] = percentile(float(feats[feat]), train_values.get(ds, {}).get(feat, []))
        out[key] = enriched

    return out, {
        "n_feature_rows": len(out),
        "n_datasets": len({k[0] for k in out}),
        "feature_boundary": "raw distributional features computed for train/internal-val split conditions; dataset normalizers fit on train conditions only",
        "max_cells_per_condition": MAX_CELLS_PER_CONDITION,
    }


def condition_metric_map(path: Path, group: str) -> dict[tuple[str, str], dict[str, float]]:
    obj = load_json(path)
    metrics = obj["groups"][group]["condition_metrics"]
    return {
        (str(row["dataset"]), str(row["condition"])): {
            "pearson_pert": float(row["pearson_pert"]),
            "test_mmd_clamped": float(row["test_mmd_clamped"]),
        }
        for row in metrics
    }


def metric_rows(features: dict[tuple[str, str], dict[str, float]]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for group in GROUPS:
        anchor = condition_metric_map(ANCHOR_JSON, group)
        cand = condition_metric_map(CAND_JSON, group)
        for key in sorted(set(anchor) & set(cand) & set(features)):
            rows.append(MetricRow(
                group=group,
                dataset=key[0],
                condition=key[1],
                features=features[key],
                delta_pp=float(cand[key]["pearson_pert"] - anchor[key]["pearson_pert"]),
                delta_mmd=float(cand[key]["test_mmd_clamped"] - anchor[key]["test_mmd_clamped"]),
            ))
    return rows


def bootstrap(values: list[float]) -> tuple[float, float, float]:
    rng = random.Random(SEED)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(len(values)) for _ in values]
        means.append(float(np.mean(arr[idx])))
    q = np.asarray(means)
    return float(np.quantile(q, 0.025)), float(np.quantile(q, 0.975)), float(np.mean(q < 0.0))


def alpha_for(row: MetricRow, rule: Rule) -> float:
    if rule.name == "noop":
        return 0.0
    if rule.name == "all_candidate":
        return 1.0
    assert rule.feature is not None
    val = row.features.get(rule.feature, float("nan"))
    if not np.isfinite(val):
        return 0.0
    hit = val <= rule.threshold if rule.op == "<=" else val >= rule.threshold
    return rule.alpha_true if hit else rule.alpha_false


def apply_rule(rows: list[MetricRow], rule: Rule) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        a = alpha_for(row, rule)
        out.append({
            "dataset": row.dataset,
            "condition": row.condition,
            "delta_pp": float(a * row.delta_pp),
            "delta_mmd": float(a * row.delta_mmd),
            "alpha": float(a),
            "raw_delta_mmd": float(row.delta_mmd),
        })
    return out


def summarize(rows: list[dict[str, Any]], *, do_bootstrap: bool = True) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mean_pp_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "bootstrap_p_harm": float("nan"), "dataset_min_pp_delta": float("nan"), "mean_mmd_delta": float("nan"), "mean_alpha": float("nan"), "top_mmd_harm_block_rate": float("nan")}
    pp = np.asarray([r["delta_pp"] for r in rows], dtype=np.float64)
    mmd = np.asarray([r["delta_mmd"] for r in rows], dtype=np.float64)
    if do_bootstrap:
        lo, hi, p_harm = bootstrap([float(x) for x in pp])
    else:
        lo, hi, p_harm = float("nan"), float("nan"), float(np.mean(pp < 0))
    by_ds: dict[str, list[float]] = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], []).append(float(row["delta_pp"]))
    raw_sorted = sorted(rows, key=lambda r: float(r["raw_delta_mmd"]), reverse=True)
    top = raw_sorted[: max(1, min(10, len(raw_sorted)))]
    return {
        "n": len(rows),
        "mean_pp_delta": float(np.mean(pp)),
        "ci95_low": lo,
        "ci95_high": hi,
        "bootstrap_p_harm": p_harm,
        "condition_p_harm": float(np.mean(pp < 0)),
        "dataset_min_pp_delta": float(min(np.mean(v) for v in by_ds.values())),
        "mean_mmd_delta": float(np.mean(mmd)),
        "mean_alpha": float(np.mean([r["alpha"] for r in rows])),
        "top_mmd_harm_block_rate": float(np.mean([float(r["alpha"]) < 0.5 for r in top])),
    }


def candidate_rules(rows: list[MetricRow], features: list[str]) -> list[Rule]:
    rules = [Rule("noop", None, ">=", 0, 0, 0), Rule("all_candidate", None, ">=", 0, 1, 1)]
    for feat in features:
        vals = np.asarray([r.features.get(feat, float("nan")) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size < 12 or float(np.max(vals) - np.min(vals)) <= 1e-12:
            continue
        for q in (0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
            thr = float(np.quantile(vals, q))
            for op in ("<=", ">="):
                for at, af in ((1.0, 0.0), (0.0, 1.0), (0.75, 0.0), (0.0, 0.75), (0.5, 0.0), (1.0, 0.25), (0.25, 1.0)):
                    rules.append(Rule(f"{feat}_{op}_{thr:.5g}_a{at:.2f}_{af:.2f}", feat, op, thr, at, af))
    return rules


def transform(rows: list[MetricRow], control: str) -> list[MetricRow]:
    if control == "main":
        return rows
    feature_names = sorted({k for r in rows for k in r.features})
    shuffled: dict[str, list[float]] = {}
    if control == "shuffled":
        rng = random.Random(SEED + 991)
        for feat in feature_names:
            vals = [float(r.features.get(feat, float("nan"))) for r in rows]
            rng.shuffle(vals)
            shuffled[feat] = vals
    out = []
    for i, row in enumerate(rows):
        feats = dict(row.features)
        if control == "inverted":
            feats = {k: -float(v) for k, v in feats.items()}
        elif control == "shuffled":
            feats = {k: shuffled[k][i] for k in feature_names}
        out.append(MetricRow(row.group, row.dataset, row.condition, feats, row.delta_pp, row.delta_mmd))
    return out


def feature_names(rows: list[MetricRow], control: str) -> list[str]:
    feats = sorted({k for r in rows for k, v in r.features.items() if np.isfinite(v)})
    if control == "count_only":
        return [f for f in feats if f in COUNT_FEATURES]
    if control == "noncount":
        return [f for f in feats if f not in COUNT_FEATURES and f not in {"n_ctrl", "n_gt"}]
    return feats


def score(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    pos_mmd = max(0.0, float(summary["mean_mmd_delta"]))
    return (
        float(summary["mean_pp_delta"]) - 12.0 * pos_mmd + 0.002 * float(summary["top_mmd_harm_block_rate"]),
        -float(summary["bootstrap_p_harm"]),
        float(summary["dataset_min_pp_delta"]),
        -float(summary["condition_p_harm"]),
    )


def select_rule(rows: list[MetricRow], feats: list[str]) -> tuple[Rule, dict[str, Any]]:
    best_rule: Rule | None = None
    best_summary: dict[str, Any] | None = None
    best_score: tuple[float, float, float, float] | None = None
    for rule in candidate_rules(rows, feats):
        s = summarize(apply_rule(rows, rule), do_bootstrap=False)
        sc = score(s)
        if best_score is None or sc > best_score:
            best_rule, best_summary, best_score = rule, s, sc
    assert best_rule is not None and best_summary is not None
    return best_rule, best_summary


def nested_lodo(rows: list[MetricRow], control: str) -> dict[str, Any]:
    work = transform(rows, control if control in {"shuffled", "inverted"} else "main")
    feats = feature_names(work, control)
    applied_all: list[dict[str, Any]] = []
    folds = []
    for ds in sorted({r.dataset for r in work}):
        train = [r for r in work if r.dataset != ds]
        test = [r for r in work if r.dataset == ds]
        if len(train) < 12 or not test or not feats:
            continue
        rule, tr_sum = select_rule(train, feats)
        applied = apply_rule(test, rule)
        te_sum = summarize(applied, do_bootstrap=False)
        applied_all.extend(applied)
        folds.append({"heldout_dataset": ds, "rule": rule.name, "train_score_pp": tr_sum["mean_pp_delta"], "test_pp": te_sum["mean_pp_delta"], "test_mmd": te_sum["mean_mmd_delta"], "test_alpha": te_sum["mean_alpha"]})
    s = summarize(applied_all)
    counts: dict[str, int] = {}
    for f in folds:
        counts[f["rule"]] = counts.get(f["rule"], 0) + 1
    return {"control": control, "features": feats, "summary": s, "folds": folds, "top_rules": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]}


def correlations(rows: list[MetricRow]) -> list[dict[str, Any]]:
    try:
        from scipy.stats import spearmanr
    except Exception:
        return []
    out = []
    feats = feature_names(rows, "noncount")
    y = np.asarray([r.delta_mmd for r in rows], dtype=np.float64)
    for feat in feats:
        x = np.asarray([r.features.get(feat, float("nan")) for r in rows], dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 12:
            rho, p = spearmanr(x[mask], y[mask])
            out.append({"feature": feat, "n": int(mask.sum()), "spearman_mmd": float(rho), "p": float(p)})
    return sorted(out, key=lambda r: -abs(r["spearman_mmd"]))[:12]


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(r["group"], r["control"]): r["summary"] for r in results}
    cross = by_key.get((GROUPS[0], "main"), {})
    family = by_key.get((GROUPS[1], "main"), {})
    reasons = []
    if not cross or not family:
        reasons.append("missing_main_summary")
    else:
        if float(cross["mean_pp_delta"]) < 0.010:
            reasons.append("cross_pp_delta_below_0.010")
        if float(cross["dataset_min_pp_delta"]) < -0.020:
            reasons.append("cross_dataset_min_below_minus_0.020")
        if float(family["mean_pp_delta"]) < 0.0:
            reasons.append("family_pp_harmed")
        if float(family["mean_mmd_delta"]) > 0.0:
            reasons.append("family_mmd_delta_above_0")
        if float(family["top_mmd_harm_block_rate"]) < 0.50:
            reasons.append("top_mmd_harm_recall_below_0.50")
    for control in ("shuffled", "inverted", "count_only"):
        c = by_key.get((GROUPS[0], control), {})
        f = by_key.get((GROUPS[1], control), {})
        if c and cross and float(c["mean_pp_delta"]) >= float(cross["mean_pp_delta"]) - 0.002:
            reasons.append(f"{control}_cross_matches_main")
        if f and family and float(f["mean_mmd_delta"]) <= float(family["mean_mmd_delta"]) + 0.0005:
            reasons.append(f"{control}_mmd_not_worse_than_main")
    return {
        "status": "distributional_mmd_harm_gate_pass_gpu_smoke_authorized" if not reasons else "distributional_mmd_harm_gate_fail_no_gpu",
        "gpu_authorized": not reasons,
        "reasons": reasons,
        "cross_mean_pp_delta": None if not cross else cross["mean_pp_delta"],
        "family_mean_pp_delta": None if not family else family["mean_pp_delta"],
        "family_mean_mmd_delta": None if not family else family["mean_mmd_delta"],
    }


def render_md(payload: dict[str, Any]) -> str:
    d = payload["decision"]
    lines = [
        "# LatentFM Distributional MMD-Harm Safety Gate",
        "",
        f"Status: `{d['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only nested leave-one-dataset-out gate.",
        "- Reads train-only split H5 embeddings and completed general-exposure internal posthoc JSONs.",
        "- Does not read canonical outcomes, canonical multi, Track C query, active logs, or new GPU artifacts.",
        f"- Feature provenance: {payload['feature_meta']['feature_boundary']}.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{d['gpu_authorized']}`",
        f"- reasons: `{d['reasons']}`",
        f"- cross pp delta: `{d['cross_mean_pp_delta']}`",
        f"- family pp delta: `{d['family_mean_pp_delta']}`",
        f"- family MMD delta: `{d['family_mean_mmd_delta']}`",
        "",
        "## Nested LODO Summaries",
        "",
        "| group | control | n | mean pp delta | 95% CI | p_harm | dataset min | mean MMD delta | mean alpha | top MMD harm block | top rules |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload["results"]:
        s = r["summary"]
        top = "; ".join(f"{name}:{count}" for name, count in r["top_rules"])
        lines.append(f"| `{r['group']}` | `{r['control']}` | {s['n']} | {s['mean_pp_delta']:.6f} | [{s['ci95_low']:.6f}, {s['ci95_high']:.6f}] | {s['bootstrap_p_harm']:.3f} | {s['dataset_min_pp_delta']:.6f} | {s['mean_mmd_delta']:.6f} | {s['mean_alpha']:.3f} | {s['top_mmd_harm_block_rate']:.3f} | {top} |")
    lines.extend(["", "## Top Feature Correlations With MMD Delta", "", "| group | feature | n | Spearman vs delta MMD | p |", "|---|---|---:|---:|---:|"])
    for group, rows in payload["feature_correlations"].items():
        for row in rows:
            lines.append(f"| `{group}` | `{row['feature']}` | {row['n']} | {row['spearman_mmd']:+.4f} | {row['p']:.4f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A pass here would authorize at most one capped GPU smoke that explicitly targets distributional no-harm. A fail closes this MMD-risk routing idea under current train-only evidence.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    features, meta = compute_features()
    rows = metric_rows(features)
    results = []
    for group in GROUPS:
        subset = [r for r in rows if r.group == group]
        for control in ("main", "shuffled", "inverted", "count_only", "noncount"):
            res = nested_lodo(subset, control)
            res["group"] = group
            results.append(res)
    payload = {
        "boundary": {
            "split_file": str(SPLIT_FILE),
            "anchor_json": str(ANCHOR_JSON),
            "candidate_json": str(CAND_JSON),
            "failure_json": str(FAILURE_JSON),
            "groups": GROUPS,
            "seed": SEED,
        },
        "feature_meta": meta,
        "n_metric_rows": len(rows),
        "feature_correlations": {g: correlations([r for r in rows if r.group == g]) for g in GROUPS},
        "results": results,
        "decision": decide(results),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(OUT_MD)


if __name__ == "__main__":
    main()
