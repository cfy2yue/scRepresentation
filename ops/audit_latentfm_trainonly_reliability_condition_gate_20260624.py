#!/usr/bin/env python3
"""Strict train-only condition-level reliability gate.

This CPU-only gate asks whether condition reliability features, computed from
the train-only split H5 embeddings, predict where existing xverse candidate
updates help internal validation conditions. It does not read canonical test
outcomes, Track C query outcomes, active logs, or new GPU artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import h5py
import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
SPLIT_FILE = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
OUT_JSON = ROOT / "reports/latentfm_trainonly_reliability_condition_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRAINONLY_RELIABILITY_CONDITION_GATE_20260624.md"

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
    "log_n_gt",
    "log_n_ctrl",
    "response_norm",
    "mean_var",
    "sem_proxy",
    "snr_proxy",
)
COUNT_FEATURES = ("log_n_gt", "log_n_ctrl", "ds_z_log_n_gt", "ds_z_log_n_ctrl", "ds_pct_log_n_gt", "ds_pct_log_n_ctrl")
MAX_CELLS_PER_CONDITION = 512
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
    out = []
    for value in values:
        out.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return out


def sample_slice(arr: h5py.Dataset, lo: int, hi: int, *, max_cells: int, key: str) -> np.ndarray:
    n = int(hi - lo)
    if n <= 0:
        raise ValueError(f"empty slice for {key}")
    if max_cells > 0 and n > max_cells:
        rng = np.random.default_rng(stable_seed(key))
        idx = np.sort(rng.choice(n, size=max_cells, replace=False))
        return np.asarray(arr[lo + idx], dtype=np.float64)
    return np.asarray(arr[lo:hi], dtype=np.float64)


def reliability_for_condition(
    h5: h5py.File,
    ds: str,
    condition: str,
    cidx: dict[str, int],
) -> dict[str, float] | None:
    if condition not in cidx:
        return None
    i = cidx[condition]
    ctrl = h5["ctrl/emb"] if "ctrl/emb" in h5 else h5["ir/emb"]
    gt = h5["gt/emb"]
    ctrl_offsets = np.asarray(h5["ctrl/offsets"] if "ctrl/offsets" in h5 else h5["ir/offsets"])
    gt_offsets = np.asarray(h5["gt/offsets"])
    c0, c1 = int(ctrl_offsets[i]), int(ctrl_offsets[i + 1])
    g0, g1 = int(gt_offsets[i]), int(gt_offsets[i + 1])
    ctrl_arr = sample_slice(ctrl, c0, c1, max_cells=MAX_CELLS_PER_CONDITION, key=f"ctrl|{ds}|{condition}")
    gt_arr = sample_slice(gt, g0, g1, max_cells=MAX_CELLS_PER_CONDITION, key=f"gt|{ds}|{condition}")
    ctrl_mean = ctrl_arr.mean(axis=0)
    gt_mean = gt_arr.mean(axis=0)
    delta = gt_mean - ctrl_mean
    ctrl_var = float(np.mean(np.var(ctrl_arr, axis=0)))
    gt_var = float(np.mean(np.var(gt_arr, axis=0)))
    sem = math.sqrt(ctrl_var / max(1, len(ctrl_arr)) + gt_var / max(1, len(gt_arr)))
    norm = float(np.linalg.norm(delta))
    return {
        "n_ctrl": float(c1 - c0),
        "n_gt": float(g1 - g0),
        "log_n_ctrl": float(math.log1p(c1 - c0)),
        "log_n_gt": float(math.log1p(g1 - g0)),
        "response_norm": norm,
        "mean_var": float((ctrl_var + gt_var) / 2.0),
        "sem_proxy": sem,
        "snr_proxy": float(norm / (sem + 1e-8)),
    }


def median_mad(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad if mad > 1e-12 else 1.0


def percentile_vs_train(value: float, train_values: list[float]) -> float:
    vals = np.asarray([v for v in train_values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(np.mean(vals <= value))


def compute_reliability_features() -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, Any]]:
    split = load_json(SPLIT_FILE)
    needed: dict[str, set[str]] = {}
    train_conditions: dict[str, set[str]] = {}
    for ds, groups in split.items():
        ds_needed: set[str] = set()
        for key in ("train",) + GROUPS:
            ds_needed.update(str(c) for c in groups.get(key, []))
        needed[ds] = ds_needed
        train_conditions[ds] = {str(c) for c in groups.get("train", [])}

    raw: dict[tuple[str, str], dict[str, float]] = {}
    for ds, conds in needed.items():
        h5_path = DATA_DIR / f"{ds}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as h5:
            cidx = {c: i for i, c in enumerate(decode_conditions(np.asarray(h5["conditions"])))}
            for cond in sorted(conds):
                feats = reliability_for_condition(h5, ds, cond, cidx)
                if feats is not None:
                    raw[(ds, cond)] = feats

    normalizers: dict[str, dict[str, tuple[float, float]]] = {}
    train_values: dict[str, dict[str, list[float]]] = {}
    for ds, conds in train_conditions.items():
        ds_rows = [raw[(ds, c)] for c in conds if (ds, c) in raw]
        normalizers[ds] = {}
        train_values[ds] = {}
        for feat in RAW_FEATURES:
            vals = [float(r[feat]) for r in ds_rows]
            normalizers[ds][feat] = median_mad(vals)
            train_values[ds][feat] = vals

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, feats in raw.items():
        ds, _cond = key
        enriched = dict(feats)
        for feat in RAW_FEATURES:
            med, mad = normalizers.get(ds, {}).get(feat, (0.0, 1.0))
            enriched[f"ds_z_{feat}"] = float((float(feats[feat]) - med) / mad)
            enriched[f"ds_pct_{feat}"] = percentile_vs_train(float(feats[feat]), train_values.get(ds, {}).get(feat, []))
        out[key] = enriched

    meta = {
        "n_feature_rows": len(out),
        "n_datasets": len({k[0] for k in out}),
        "feature_boundary": "raw features computed for train/internal-val split conditions; dataset normalizers fit on train conditions only",
    }
    return out, meta


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
    for run_name, paths in RUNS.items():
        for group in GROUPS:
            anchor = condition_metric_map(paths["anchor"], group)
            cand = condition_metric_map(paths["candidate"], group)
            for key in sorted(set(anchor) & set(cand) & set(features)):
                rows.append(
                    MetricRow(
                        run=run_name,
                        group=group,
                        dataset=key[0],
                        condition=key[1],
                        features=features[key],
                        delta_pp=float(cand[key]["pearson_pert"] - anchor[key]["pearson_pert"]),
                        delta_mmd=float(cand[key]["test_mmd_clamped"] - anchor[key]["test_mmd_clamped"]),
                    )
                )
    return rows


def bootstrap_ci_and_harm(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(SEED)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(len(values)) for _ in values]
        means.append(float(np.mean(arr[idx])))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float(np.mean(np.asarray(means) < 0.0))


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
    out = []
    for row in rows:
        alpha = alpha_for(row, rule)
        out.append(
            {
                "run": row.run,
                "group": row.group,
                "dataset": row.dataset,
                "condition": row.condition,
                "delta_pp": float(alpha * row.delta_pp),
                "delta_mmd": float(alpha * row.delta_mmd),
                "alpha": float(alpha),
            }
        )
    return out


def summarize_applied(rows: list[dict[str, Any]], *, with_bootstrap: bool = True) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean_pp_delta": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "bootstrap_p_harm": float("nan"),
            "condition_p_harm": float("nan"),
            "dataset_min_pp_delta": float("nan"),
            "mean_mmd_delta": float("nan"),
            "mean_alpha": float("nan"),
        }
    vals = [float(r["delta_pp"]) for r in rows]
    vals_arr = np.asarray(vals, dtype=np.float64)
    if with_bootstrap:
        lo, hi, p_harm = bootstrap_ci_and_harm(vals)
    else:
        lo, hi = float("nan"), float("nan")
        p_harm = float(np.mean(vals_arr < 0.0))
    by_ds: dict[str, list[float]] = {}
    for row in rows:
        by_ds.setdefault(str(row["dataset"]), []).append(float(row["delta_pp"]))
    return {
        "n": len(rows),
        "mean_pp_delta": float(np.mean(vals_arr)),
        "ci95_low": lo,
        "ci95_high": hi,
        "bootstrap_p_harm": p_harm,
        "condition_p_harm": float(np.mean(vals_arr < 0.0)),
        "dataset_min_pp_delta": float(min(sum(v) / len(v) for v in by_ds.values())),
        "mean_mmd_delta": float(np.mean([float(r["delta_mmd"]) for r in rows])),
        "mean_alpha": float(np.mean([float(r["alpha"]) for r in rows])),
    }


def score_summary(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    mmd_penalty = max(0.0, float(summary["mean_mmd_delta"]))
    return (
        float(summary["mean_pp_delta"]) - 5.0 * mmd_penalty,
        -float(summary["bootstrap_p_harm"]),
        float(summary["dataset_min_pp_delta"]),
        -float(summary["condition_p_harm"]),
    )


def candidate_rules(rows: list[MetricRow], features: list[str]) -> list[Rule]:
    rules = [
        Rule("noop", None, ">=", 0.0, 0.0, 0.0),
        Rule("all_candidate", None, ">=", 0.0, 1.0, 1.0),
    ]
    for feat in features:
        vals = np.asarray([r.features.get(feat, float("nan")) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size < 12 or float(np.max(vals) - np.min(vals)) <= 1e-12:
            continue
        for q in (0.2, 0.35, 0.5, 0.65, 0.8):
            threshold = float(np.quantile(vals, q))
            for op in ("<=", ">="):
                for alpha_true, alpha_false in ((1.0, 0.0), (0.0, 1.0), (0.75, 0.0), (0.0, 0.75), (0.5, 0.0)):
                    name = f"{feat}_{op}_{threshold:.5g}_a{alpha_true:.2f}_{alpha_false:.2f}"
                    rules.append(Rule(name, feat, op, threshold, alpha_true, alpha_false))
    return rules


def transform_rows(rows: list[MetricRow], control: str) -> list[MetricRow]:
    if control == "main":
        return rows
    transformed = []
    feature_names = sorted({k for r in rows for k in r.features})
    shuffled_values: dict[str, list[float]] = {}
    if control == "shuffled":
        rng = random.Random(SEED + 171)
        for feat in feature_names:
            vals = [float(r.features.get(feat, float("nan"))) for r in rows]
            rng.shuffle(vals)
            shuffled_values[feat] = vals
    for i, row in enumerate(rows):
        feats = dict(row.features)
        if control == "inverted":
            feats = {k: -float(v) for k, v in feats.items()}
        elif control == "shuffled":
            feats = {k: shuffled_values[k][i] for k in feature_names}
        transformed.append(
            MetricRow(
                run=row.run,
                group=row.group,
                dataset=row.dataset,
                condition=row.condition,
                features=feats,
                delta_pp=row.delta_pp,
                delta_mmd=row.delta_mmd,
            )
        )
    return transformed


def feature_set(control: str, rows: list[MetricRow]) -> list[str]:
    all_features = sorted({k for row in rows for k in row.features if np.isfinite(row.features[k])})
    count_features = [f for f in all_features if f in COUNT_FEATURES]
    if control == "count_only":
        return count_features
    if control == "noncount":
        return [f for f in all_features if f not in COUNT_FEATURES and f not in {"n_ctrl", "n_gt"}]
    return all_features


def select_rule(train_rows: list[MetricRow], features: list[str]) -> tuple[Rule, dict[str, Any]]:
    best_rule: Rule | None = None
    best_summary: dict[str, Any] | None = None
    best_score: tuple[float, float, float, float] | None = None
    for rule in candidate_rules(train_rows, features):
        summary = summarize_applied(apply_rule(train_rows, rule), with_bootstrap=False)
        score = score_summary(summary)
        if best_score is None or score > best_score:
            best_rule = rule
            best_summary = summary
            best_score = score
    assert best_rule is not None and best_summary is not None
    return best_rule, best_summary


def nested_lodo(rows: list[MetricRow], control: str) -> dict[str, Any]:
    work_rows = transform_rows(rows, control if control in {"shuffled", "inverted"} else "main")
    features = feature_set(control, work_rows)
    applied_all: list[dict[str, Any]] = []
    fold_rows = []
    datasets = sorted({r.dataset for r in work_rows})
    for heldout in datasets:
        train_rows = [r for r in work_rows if r.dataset != heldout]
        test_rows = [r for r in work_rows if r.dataset == heldout]
        if len(train_rows) < 12 or not test_rows or not features:
            continue
        rule, train_summary = select_rule(train_rows, features)
        applied = apply_rule(test_rows, rule)
        test_summary = summarize_applied(applied, with_bootstrap=False)
        applied_all.extend(applied)
        fold_rows.append(
            {
                "heldout_dataset": heldout,
                "rule": rule.name,
                "train_mean_pp_delta": train_summary["mean_pp_delta"],
                "test_mean_pp_delta": test_summary["mean_pp_delta"],
                "test_mean_mmd_delta": test_summary["mean_mmd_delta"],
                "test_mean_alpha": test_summary["mean_alpha"],
            }
        )
    summary = summarize_applied(applied_all)
    rules: dict[str, int] = {}
    for row in fold_rows:
        rules[row["rule"]] = rules.get(row["rule"], 0) + 1
    return {
        "control": control,
        "features": features,
        "summary": summary,
        "folds": fold_rows,
        "top_rules": sorted(rules.items(), key=lambda kv: (-kv[1], kv[0]))[:8],
    }


def pass_reasons(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(r["run"], r["group"], r["control"]): r for r in results}
    decisions = []
    for run in sorted({r["run"] for r in results}):
        cross = by_key.get((run, GROUPS[0], "main"), {}).get("summary", {})
        family = by_key.get((run, GROUPS[1], "main"), {}).get("summary", {})
        reasons = []
        if not cross or not family:
            reasons.append("missing_main_group_summary")
        else:
            if float(cross["mean_pp_delta"]) < 0.010:
                reasons.append("cross_pp_delta_below_0.010")
            if float(cross["dataset_min_pp_delta"]) < -0.020:
                reasons.append("cross_dataset_min_below_minus_0.020")
            if float(cross["bootstrap_p_harm"]) > 0.35:
                reasons.append("cross_bootstrap_p_harm_above_0.35")
            if float(family["mean_pp_delta"]) < 0.000:
                reasons.append("family_pp_harmed")
            if float(family["dataset_min_pp_delta"]) < -0.020:
                reasons.append("family_dataset_min_below_minus_0.020")
            if float(family["mean_mmd_delta"]) > 0.0005:
                reasons.append("family_mmd_worse_above_0.0005")
        for control in ("shuffled", "inverted"):
            csum = by_key.get((run, GROUPS[0], control), {}).get("summary", {})
            if csum and float(csum["mean_pp_delta"]) >= 0.005:
                reasons.append(f"{control}_control_cross_not_collapsed")
        count = by_key.get((run, GROUPS[0], "count_only"), {}).get("summary", {})
        main = by_key.get((run, GROUPS[0], "main"), {}).get("summary", {})
        if count and main and float(count["mean_pp_delta"]) >= float(main["mean_pp_delta"]) - 0.002:
            reasons.append("count_only_matches_main_signal")
        decisions.append(
            {
                "run": run,
                "passed": not reasons,
                "reasons": reasons,
                "cross_mean_pp_delta": None if not cross else cross["mean_pp_delta"],
                "family_mean_pp_delta": None if not family else family["mean_pp_delta"],
            }
        )
    passed = [d for d in decisions if d["passed"]]
    return {
        "status": "trainonly_reliability_condition_gate_pass_gpu_smoke_authorized" if passed else "trainonly_reliability_condition_gate_fail_no_gpu",
        "gpu_authorized": bool(passed),
        "passed_runs": [d["run"] for d in passed],
        "run_decisions": decisions,
    }


def render_md(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# LatentFM Train-Only Reliability Condition Gate",
        "",
        f"Status: `{decision['status']}`",
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
        f"- GPU authorized: `{decision['gpu_authorized']}`",
        f"- passed runs: `{decision['passed_runs']}`",
        f"- feature rows: `{payload['feature_meta']['n_feature_rows']}`",
        f"- metric rows: `{payload['n_metric_rows']}`",
        "",
        "## Run Decisions",
        "",
        "| run | passed | cross pp delta | family pp delta | reasons |",
        "|---|---:|---:|---:|---|",
    ]
    for row in decision["run_decisions"]:
        reasons = ", ".join(row["reasons"]) if row["reasons"] else "none"
        cross = row["cross_mean_pp_delta"]
        family = row["family_mean_pp_delta"]
        lines.append(
            f"| `{row['run']}` | `{row['passed']}` | {cross if cross is not None else 'NA'} | {family if family is not None else 'NA'} | {reasons} |"
        )
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
            "A pass here would only authorize one capped GPU smoke. A fail means the current reliability signal is not strong enough to justify reliability-weighted training.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    features, feature_meta = compute_reliability_features()
    rows = metric_rows(features)
    results = []
    for run in sorted({r.run for r in rows}):
        for group in GROUPS:
            subset = [r for r in rows if r.run == run and r.group == group]
            for control in ("main", "shuffled", "inverted", "count_only", "noncount"):
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
        "decision": pass_reasons(results),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))
    print(OUT_MD)


if __name__ == "__main__":
    main()
