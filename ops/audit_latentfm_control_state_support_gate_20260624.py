#!/usr/bin/env python3
"""CPU gate for train-only control-state support/coverage signal."""

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
OUT_JSON = ROOT / "reports/latentfm_control_state_support_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_CONTROL_STATE_SUPPORT_GATE_20260624.md"

RUNS = {
    "cap120": {
        "anchor": ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal/split_group_eval_anchor_internal_ode20.json",
        "candidate": ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json",
    },
    "cap60_protocol": {
        "anchor": ROOT / "runs/latentfm_scaling_protocol_matrix_20260624/xverse_scaling_protocol_cap60_primary19_3k_seed42/posthoc_eval_internal/split_group_eval_anchor_internal_ode20.json",
        "candidate": ROOT / "runs/latentfm_scaling_protocol_matrix_20260624/xverse_scaling_protocol_cap60_primary19_3k_seed42/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json",
    },
}
GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)
RAW_FEATURES = (
    "ctrl_knn5_median",
    "ctrl_knn5_p90",
    "gt_to_ctrl_nn_median",
    "gt_to_ctrl_nn_p90",
    "coverage_frac",
    "gt_ctrl_centroid_distance",
    "ctrl_effective_rank",
    "gt_effective_rank",
    "rank_ratio",
    "ctrl_anisotropy",
    "gt_anisotropy",
)
COUNT_FEATURES = ("log_n_gt", "log_n_ctrl", "ds_z_log_n_gt", "ds_z_log_n_ctrl", "ds_pct_log_n_gt", "ds_pct_log_n_ctrl")
MAX_CELLS_PER_CONDITION = 128
KNN_K = 5
SEED = 42
BOOT_N = 1000


@dataclass(frozen=True)
class MetricRow:
    run: str
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


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    d2 = np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2)


def effective_rank_and_anisotropy(x: np.ndarray) -> tuple[float, float]:
    xc = x - np.mean(x, axis=0, keepdims=True)
    if x.shape[0] < 3:
        return 1.0, 1.0
    gram = (xc @ xc.T) / max(1, x.shape[0] - 1)
    eig = np.linalg.eigvalsh(gram)
    eig = np.maximum(eig, 0.0)
    total = float(np.sum(eig))
    if total <= 1e-12:
        return 1.0, 1.0
    p = eig / total
    p = p[p > 1e-12]
    eff_rank = float(math.exp(-np.sum(p * np.log(p))))
    anisotropy = float(np.max(eig) / (total / max(1, len(eig))))
    return eff_rank, anisotropy


def condition_features(h5: h5py.File, ds: str, condition: str, cidx: dict[str, int]) -> dict[str, float] | None:
    if condition not in cidx:
        return None
    i = cidx[condition]
    ctrl = h5["ctrl/emb"] if "ctrl/emb" in h5 else h5["ir/emb"]
    gt = h5["gt/emb"]
    ctrl_offsets = np.asarray(h5["ctrl/offsets"] if "ctrl/offsets" in h5 else h5["ir/offsets"])
    gt_offsets = np.asarray(h5["gt/offsets"])
    c0, c1 = int(ctrl_offsets[i]), int(ctrl_offsets[i + 1])
    g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
    ctrl_arr = sample_slice(ctrl, c0, c1, key=f"ctrl|{ds}|{condition}")
    gt_arr = sample_slice(gt, g0, g1, key=f"gt|{ds}|{condition}")

    cdist = pairwise_dist(ctrl_arr, ctrl_arr)
    np.fill_diagonal(cdist, np.inf)
    k = min(KNN_K, max(1, ctrl_arr.shape[0] - 1))
    ctrl_knn = np.partition(cdist, kth=k - 1, axis=1)[:, k - 1]
    support_radius = float(np.median(ctrl_knn))

    gdist = pairwise_dist(gt_arr, ctrl_arr)
    gt_nn = np.min(gdist, axis=1)
    ctrl_eff, ctrl_aniso = effective_rank_and_anisotropy(ctrl_arr)
    gt_eff, gt_aniso = effective_rank_and_anisotropy(gt_arr)
    centroid_distance = float(np.linalg.norm(np.mean(gt_arr, axis=0) - np.mean(ctrl_arr, axis=0)))
    return {
        "n_ctrl": float(c1 - c0),
        "n_gt": float(g1 - g0),
        "log_n_ctrl": float(math.log1p(c1 - c0)),
        "log_n_gt": float(math.log1p(g1 - g0)),
        "ctrl_knn5_median": support_radius,
        "ctrl_knn5_p90": float(np.quantile(ctrl_knn, 0.90)),
        "gt_to_ctrl_nn_median": float(np.median(gt_nn)),
        "gt_to_ctrl_nn_p90": float(np.quantile(gt_nn, 0.90)),
        "coverage_frac": float(np.mean(gt_nn <= support_radius)),
        "gt_ctrl_centroid_distance": centroid_distance,
        "ctrl_effective_rank": ctrl_eff,
        "gt_effective_rank": gt_eff,
        "rank_ratio": float(gt_eff / max(ctrl_eff, 1e-8)),
        "ctrl_anisotropy": ctrl_aniso,
        "gt_anisotropy": gt_aniso,
    }


def median_mad(values: list[float]) -> tuple[float, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return 0.0, 1.0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, mad if mad > 1e-12 else 1.0


def percentile(value: float, values: list[float]) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(np.mean(vals <= value))


def compute_features() -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    split = load_json(SPLIT_FILE)
    needed: dict[str, set[str]] = {}
    train_conditions: dict[str, set[str]] = {}
    for ds, groups in split.items():
        conds = set(str(c) for c in groups.get("train", []))
        for group in GROUPS:
            conds.update(str(c) for c in groups.get(group, []))
        needed[ds] = conds
        train_conditions[ds] = set(str(c) for c in groups.get("train", []))

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
    norm: dict[str, dict[str, tuple[float, float]]] = {}
    all_raw_features = RAW_FEATURES + ("log_n_gt", "log_n_ctrl")
    for ds, conds in train_conditions.items():
        ds_rows = [raw[(ds, c)] for c in conds if (ds, c) in raw]
        train_values[ds] = {}
        norm[ds] = {}
        for feat in all_raw_features:
            vals = [float(r[feat]) for r in ds_rows]
            train_values[ds][feat] = vals
            norm[ds][feat] = median_mad(vals)

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, feats in raw.items():
        ds, _cond = key
        enriched = dict(feats)
        for feat in all_raw_features:
            med, mad = norm.get(ds, {}).get(feat, (0.0, 1.0))
            enriched[f"ds_z_{feat}"] = float((float(feats[feat]) - med) / mad)
            enriched[f"ds_pct_{feat}"] = percentile(float(feats[feat]), train_values.get(ds, {}).get(feat, []))
        out[key] = enriched
    return out, {
        "n_feature_rows": len(out),
        "n_datasets": len({k[0] for k in out}),
        "max_cells_per_condition": MAX_CELLS_PER_CONDITION,
        "feature_boundary": "raw support features computed on train/internal-val split conditions; dataset normalizers fit on train conditions only",
    }


def metric_map(path: Path, group: str) -> dict[tuple[str, str], dict[str, float]]:
    obj = load_json(path)
    return {
        (str(r["dataset"]), str(r["condition"])): {
            "pearson_pert": float(r["pearson_pert"]),
            "test_mmd_clamped": float(r["test_mmd_clamped"]),
        }
        for r in obj["groups"][group]["condition_metrics"]
    }


def metric_rows(features: dict[tuple[str, str], dict[str, float]]) -> list[MetricRow]:
    rows = []
    for run, paths in RUNS.items():
        for group in GROUPS:
            anchor = metric_map(paths["anchor"], group)
            cand = metric_map(paths["candidate"], group)
            for key in sorted(set(anchor) & set(cand) & set(features)):
                rows.append(
                    MetricRow(
                        run=run,
                        group=group,
                        dataset=key[0],
                        condition=key[1],
                        features=features[key],
                        delta_pp=float(cand[key]["pearson_pert"] - anchor[key]["pearson_pert"]),
                        delta_mmd=float(cand[key]["test_mmd_clamped"] - anchor[key]["test_mmd_clamped"]),
                    )
                )
    return rows


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


def alpha_for(row: MetricRow, rule: Rule) -> float:
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


def apply_rule(rows: list[MetricRow], rule: Rule) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.dataset,
            "delta_pp": alpha_for(row, rule) * row.delta_pp,
            "delta_mmd": alpha_for(row, rule) * row.delta_mmd,
            "alpha": alpha_for(row, rule),
        }
        for row in rows
    ]


def summarize(applied: list[dict[str, Any]], *, with_bootstrap: bool = True) -> dict[str, float]:
    if not applied:
        return {k: float("nan") for k in ("mean_pp_delta", "ci95_low", "ci95_high", "bootstrap_p_harm", "condition_p_harm", "dataset_min_pp_delta", "mean_mmd_delta", "mean_alpha")} | {"n": 0}
    vals = np.asarray([float(r["delta_pp"]) for r in applied], dtype=np.float64)
    if with_bootstrap:
        lo, hi, p_harm = bootstrap([float(v) for v in vals])
    else:
        lo, hi, p_harm = float("nan"), float("nan"), float(np.mean(vals < 0.0))
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


def candidate_rules(rows: list[MetricRow], features: list[str]) -> list[Rule]:
    rules = [Rule("noop", None, ">=", 0.0, 0.0, 0.0), Rule("all_candidate", None, ">=", 0.0, 1.0, 1.0)]
    for feat in features:
        vals = np.asarray([row.features.get(feat, float("nan")) for row in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size < 12 or float(np.max(vals) - np.min(vals)) <= 1e-12:
            continue
        for q in (0.25, 0.50, 0.75):
            threshold = float(np.quantile(vals, q))
            for op in ("<=", ">="):
                for alpha_true, alpha_false in ((1.0, 0.0), (0.0, 1.0), (0.75, 0.0)):
                    rules.append(Rule(f"{feat}_{op}_{threshold:.5g}_a{alpha_true:.2f}_{alpha_false:.2f}", feat, op, threshold, alpha_true, alpha_false))
    return rules


def score(summary: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        summary["mean_pp_delta"] - 5.0 * max(0.0, summary["mean_mmd_delta"]),
        -summary["bootstrap_p_harm"],
        summary["dataset_min_pp_delta"],
        -summary["condition_p_harm"],
    )


def transform(rows: list[MetricRow], control: str) -> list[MetricRow]:
    if control == "main":
        return rows
    features = sorted({k for row in rows for k in row.features})
    rng = random.Random(SEED + 313)
    shuffled = {}
    if control in {"shuffled", "control_permuted"}:
        for feat in features:
            vals = [row.features.get(feat, float("nan")) for row in rows]
            rng.shuffle(vals)
            shuffled[feat] = vals
    out = []
    for i, row in enumerate(rows):
        feats = dict(row.features)
        if control == "inverted":
            feats = {k: -float(v) for k, v in feats.items()}
        elif control in {"shuffled", "control_permuted"}:
            feats = {k: shuffled[k][i] for k in features}
        out.append(MetricRow(row.run, row.group, row.dataset, row.condition, feats, row.delta_pp, row.delta_mmd))
    return out


def feature_set(control: str, rows: list[MetricRow]) -> list[str]:
    features = sorted({k for row in rows for k, v in row.features.items() if np.isfinite(v)})
    count = [f for f in features if f in COUNT_FEATURES]
    if control == "count_only":
        return count
    if control == "support_only":
        return [f for f in features if f not in COUNT_FEATURES and f not in {"n_ctrl", "n_gt"} and "response_norm" not in f]
    return features


def select_rule(train_rows: list[MetricRow], features: list[str]) -> tuple[Rule, dict[str, float]]:
    best_rule = Rule("noop", None, ">=", 0.0, 0.0, 0.0)
    best_summary = summarize(apply_rule(train_rows, best_rule), with_bootstrap=False)
    best_score = score(best_summary)
    for rule in candidate_rules(train_rows, features):
        s = summarize(apply_rule(train_rows, rule), with_bootstrap=False)
        sc = score(s)
        if sc > best_score:
            best_rule, best_summary, best_score = rule, s, sc
    return best_rule, best_summary


def nested_lodo(rows: list[MetricRow], control: str) -> dict[str, Any]:
    work = transform(rows, control if control in {"shuffled", "inverted", "control_permuted"} else "main")
    features = feature_set(control, work)
    applied_all = []
    folds = []
    for heldout in sorted({r.dataset for r in work}):
        train = [r for r in work if r.dataset != heldout]
        test = [r for r in work if r.dataset == heldout]
        if len(train) < 12 or not test or not features:
            continue
        rule, train_summary = select_rule(train, features)
        applied = apply_rule(test, rule)
        test_summary = summarize(applied, with_bootstrap=False)
        applied_all.extend(applied)
        folds.append(
            {
                "heldout_dataset": heldout,
                "rule": rule.name,
                "train_mean_pp_delta": train_summary["mean_pp_delta"],
                "test_mean_pp_delta": test_summary["mean_pp_delta"],
                "test_mean_alpha": test_summary["mean_alpha"],
            }
        )
    top_rules: dict[str, int] = {}
    for fold in folds:
        top_rules[fold["rule"]] = top_rules.get(fold["rule"], 0) + 1
    return {
        "control": control,
        "features": features,
        "summary": summarize(applied_all),
        "folds": folds,
        "top_rules": sorted(top_rules.items(), key=lambda kv: (-kv[1], kv[0]))[:8],
    }


def decide(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(r["run"], r["group"], r["control"]): r for r in results}
    run_decisions = []
    for run in sorted({r["run"] for r in results}):
        cross = by_key[(run, GROUPS[0], "main")]["summary"]
        family = by_key[(run, GROUPS[1], "main")]["summary"]
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
        for control in ("shuffled", "inverted", "control_permuted"):
            c = by_key[(run, GROUPS[0], control)]["summary"]
            if c["mean_pp_delta"] >= 0.005:
                reasons.append(f"{control}_cross_not_collapsed")
        count = by_key[(run, GROUPS[0], "count_only")]["summary"]
        main = by_key[(run, GROUPS[0], "main")]["summary"]
        if count["mean_pp_delta"] >= main["mean_pp_delta"] - 0.002:
            reasons.append("count_only_matches_main_signal")
        run_decisions.append(
            {
                "run": run,
                "passed": not reasons,
                "reasons": reasons,
                "cross_mean_pp_delta": cross["mean_pp_delta"],
                "family_mean_pp_delta": family["mean_pp_delta"],
            }
        )
    passed = [r["run"] for r in run_decisions if r["passed"]]
    return {
        "status": "control_state_support_gate_pass_gpu_smoke_authorized" if passed else "control_state_support_gate_fail_no_gpu",
        "gpu_authorized": bool(passed),
        "passed_runs": passed,
        "run_decisions": run_decisions,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Control-State Support Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only nested leave-one-dataset-out gate.",
        "- Reads train-only split H5 embeddings and completed train-only internal posthoc JSONs.",
        "- Does not read canonical outcomes, canonical multi, Track C query, active logs, or new GPU artifacts.",
        f"- Feature provenance: {payload['feature_meta']['feature_boundary']}.",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{payload['decision']['gpu_authorized']}`",
        f"- passed runs: `{payload['decision']['passed_runs']}`",
        f"- feature rows: `{payload['feature_meta']['n_feature_rows']}`",
        f"- metric rows: `{payload['n_metric_rows']}`",
        "",
        "## Run Decisions",
        "",
        "| run | passed | cross pp delta | family pp delta | reasons |",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload["decision"]["run_decisions"]:
        reasons = ", ".join(row["reasons"]) if row["reasons"] else "none"
        lines.append(f"| `{row['run']}` | `{row['passed']}` | {row['cross_mean_pp_delta']:.6f} | {row['family_mean_pp_delta']:.6f} | {reasons} |")
    lines.extend(
        [
            "",
            "## Nested LODO Summaries",
            "",
            "| run | group | control | n | mean pp delta | 95% CI | p_harm | dataset min | mean MMD delta | mean alpha | top rules |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["results"]:
        s = row["summary"]
        top = "; ".join(f"{name}:{count}" for name, count in row["top_rules"])
        lines.append(
            f"| `{row['run']}` | `{row['group']}` | `{row['control']}` | {s['n']} | {s['mean_pp_delta']:.6f} | [{s['ci95_low']:.6f}, {s['ci95_high']:.6f}] | {s['bootstrap_p_harm']:.3f} | {s['dataset_min_pp_delta']:.6f} | {s['mean_mmd_delta']:.6f} | {s['mean_alpha']:.3f} | {top} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A pass here would only authorize one capped GPU smoke. A fail keeps support/coverage as diagnostic evidence only.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    features, feature_meta = compute_features()
    rows = metric_rows(features)
    results = []
    for run in sorted({r.run for r in rows}):
        for group in GROUPS:
            subset = [r for r in rows if r.run == run and r.group == group]
            for control in ("main", "support_only", "shuffled", "inverted", "control_permuted", "count_only"):
                result = nested_lodo(subset, control)
                result["run"] = run
                result["group"] = group
                results.append(result)
    payload = {
        "boundary": {
            "split_file": str(SPLIT_FILE),
            "runs": {k: {kk: str(vv) for kk, vv in val.items()} for k, val in RUNS.items()},
            "groups": GROUPS,
            "max_cells_per_condition": MAX_CELLS_PER_CONDITION,
            "seed": SEED,
        },
        "feature_meta": feature_meta,
        "n_metric_rows": len(rows),
        "results": results,
        "decision": decide(results),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(OUT_MD)


if __name__ == "__main__":
    main()
