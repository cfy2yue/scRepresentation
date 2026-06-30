#!/usr/bin/env python3
"""Exact-coverage propensity residual matched-design gate.

CPU/report-only. This is a stricter follow-up to the cross-dataset exact
coverage feasibility audit. It estimates the probability that a parent-train
condition has exact raw-expression coverage from source, perturbation type,
gene-count bin, cell counts, and response magnitude. It then asks whether the
remaining high-vs-low residual exact-coverage contrast can form a balanced
matched split.

No training, inference, GPU, canonical multi, Track C query, or checkpoint
selection is performed. A pass would authorize only external review and an
axis-specific pair-shuffle/null gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

for _key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_key, "4")

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from materialize_latentfm_trainonly_condition_residual_information_20260628 import (  # noqa: E402
    ConditionVectors,
    collect_needed_conditions,
    load_condition_vector_cache,
    load_json,
)


PARENT_SPLIT = (
    ROOT
    / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624"
    / "split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
)
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
OBS_ROWS = (
    ROOT
    / "reports/exact_analog_observability_matched_feasibility_20260629"
    / "condition_observability_rows.csv"
)
OUT_DIR = ROOT / "reports/exact_coverage_propensity_residual_match_gate_20260629"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def source_family(dataset: str) -> str:
    if dataset.startswith("sciplex3_"):
        return "sciplex3"
    if dataset.startswith("Jiang_"):
        return "Jiang"
    if dataset.startswith("Nadig_"):
        return "Nadig"
    if dataset.startswith("Replogle"):
        return "Replogle"
    if dataset.startswith("Tian"):
        return "Tian"
    return dataset


def perturbation_type(meta: dict[str, Any]) -> str:
    return str(meta.get("perturbation_type_raw") or "unknown")


def gene_count_bin(meta: dict[str, Any]) -> str:
    genes = meta.get("genes", [])
    if not isinstance(genes, list):
        return "unknown"
    n = len([g for g in genes if str(g).strip()])
    if n <= 1:
        return "single"
    if n == 2:
        return "double"
    return "multi3plus"


def js_divergence(left: Counter[str], right: Counter[str]) -> float:
    keys = sorted(set(left) | set(right))
    lt = float(sum(left.values()))
    rt = float(sum(right.values()))
    if lt <= 0 or rt <= 0:
        return float("nan")
    p = np.asarray([left.get(k, 0) / lt for k in keys], dtype=float)
    q = np.asarray([right.get(k, 0) / rt for k in keys], dtype=float)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def smd(high: list[float], low: list[float]) -> float:
    h = np.asarray([float(x) for x in high if math.isfinite(float(x))], dtype=float)
    l = np.asarray([float(x) for x in low if math.isfinite(float(x))], dtype=float)
    if h.size < 2 or l.size < 2:
        return float("nan")
    pooled = math.sqrt((float(np.var(h)) + float(np.var(l))) / 2.0)
    if pooled <= 1e-12:
        return 0.0
    return float((float(np.mean(h)) - float(np.mean(l))) / pooled)


def max_share(values: list[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return max(counts.values()) / total if total else 0.0


def load_train_rows(parent_split: Path, cond_meta_path: Path, obs_rows_path: Path) -> list[dict[str, Any]]:
    parent = load_json(parent_split)
    cond_meta = load_json(cond_meta_path)
    obs_lookup = {(row["dataset"], row["condition"]): row for row in read_csv(obs_rows_path)}
    split_rows = [{"split_file": str(parent_split.relative_to(ROOT)), "split_name": parent_split.stem}]
    cache, missing = load_condition_vector_cache(
        ROOT / "dataset/latentfm_full/xverse",
        collect_needed_conditions(split_rows),
    )
    if missing:
        print(f"warning: missing vectors for {len(missing)} parent-train conditions", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for dataset, groups in parent.items():
        for condition in groups.get("train", []):
            condition_s = str(condition)
            vectors: ConditionVectors | None = cache.get(dataset, {}).get(condition_s)
            if vectors is None:
                continue
            meta = cond_meta.get(dataset, {}).get(condition_s, {})
            obs = obs_lookup.get((dataset, condition_s), {})
            response_norm = float(np.linalg.norm(vectors.residual))
            rows.append(
                {
                    "dataset": dataset,
                    "source_family": source_family(dataset),
                    "condition": condition_s,
                    "perturbation_type": perturbation_type(meta),
                    "gene_count_bin": gene_count_bin(meta),
                    "exact_train_covered": truthy(obs.get("exact_train_covered", False)),
                    "analog_support_dataset_count": float(obs.get("analog_support_dataset_count") or 0.0),
                    "response_norm": response_norm,
                    "log_response_norm": math.log1p(response_norm),
                    "n_ctrl": int(vectors.n_ctrl),
                    "n_gt": int(vectors.n_gt),
                    "log_n_ctrl": math.log1p(max(int(vectors.n_ctrl), 0)),
                    "log_n_gt": math.log1p(max(int(vectors.n_gt), 0)),
                }
            )
    return rows


def fit_propensity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categorical = ["dataset", "source_family", "perturbation_type", "gene_count_bin"]
    numeric = ["log_n_ctrl", "log_n_gt", "log_response_norm", "analog_support_dataset_count"]
    x_cat = [[row[col] for col in categorical] for row in rows]
    x_num = [[float(row[col]) for col in numeric] for row in rows]
    # ColumnTransformer expects a rectangular mixed object array.
    x = np.asarray([cat + num for cat, num in zip(x_cat, x_num)], dtype=object)
    y = np.asarray([1 if row["exact_train_covered"] else 0 for row in rows], dtype=int)
    transformer = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(range(len(categorical)))),
            ("num", StandardScaler(), list(range(len(categorical), len(categorical) + len(numeric)))),
        ]
    )
    model = Pipeline(
        [
            ("prep", transformer),
            (
                "logreg",
                LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )
    model.fit(x, y)
    prop = model.predict_proba(x)[:, 1]
    auc = float(roc_auc_score(y, prop))
    for row, p in zip(rows, prop):
        p_clamped = min(max(float(p), 1e-4), 1.0 - 1e-4)
        row["coverage_propensity"] = p_clamped
        row["coverage_logit"] = math.log(p_clamped / (1.0 - p_clamped))
        row["coverage_residual"] = (1.0 if row["exact_train_covered"] else 0.0) - p_clamped
    return {"auc": auc, "positive_rate": float(y.mean())}


def standardize(rows: list[dict[str, Any]], cols: list[str]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for col in cols:
        vals = np.asarray([float(row[col]) for row in rows], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        out[col] = (mean, std if std > 1e-12 else 1.0)
    return out


def pair_cost(
    high: list[dict[str, Any]],
    low: list[dict[str, Any]],
    stats: dict[str, tuple[float, float]],
    cols: list[str],
    source_penalty: float,
) -> np.ndarray:
    out = np.zeros((len(high), len(low)), dtype=np.float64)
    for i, h in enumerate(high):
        for j, l in enumerate(low):
            cost = 0.0
            for col in cols:
                mean, std = stats[col]
                hz = (float(h[col]) - mean) / std
                lz = (float(l[col]) - mean) / std
                cost += (hz - lz) ** 2
            if h["source_family"] != l["source_family"]:
                cost += source_penalty
            out[i, j] = cost
    return out


def make_pairs(rows: list[dict[str, Any]], policy: dict[str, Any], stats: dict[str, tuple[float, float]]) -> list[dict[str, Any]]:
    high = [
        row
        for row in rows
        if row["exact_train_covered"]
        and policy["min_propensity"] <= row["coverage_propensity"] <= policy["max_propensity"]
        and row["coverage_residual"] >= policy["high_residual_min"]
    ]
    low = [
        row
        for row in rows
        if (not row["exact_train_covered"])
        and policy["min_propensity"] <= row["coverage_propensity"] <= policy["max_propensity"]
        and row["coverage_residual"] <= policy["low_residual_max"]
    ]
    hard_cols = policy["hard_cols"]
    high_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    low_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in high:
        high_groups[tuple(row[col] for col in hard_cols)].append(row)
    for row in low:
        low_groups[tuple(row[col] for col in hard_cols)].append(row)

    pair_rows: list[dict[str, Any]] = []
    cost_cols = ["coverage_logit", "log_n_ctrl", "log_n_gt", "log_response_norm", "analog_support_dataset_count"]
    for key in sorted(set(high_groups) & set(low_groups)):
        hg = high_groups[key]
        lg = low_groups[key]
        if not hg or not lg:
            continue
        costs = pair_cost(hg, lg, stats, cost_cols, float(policy["source_penalty"]))
        r_ind, c_ind = linear_sum_assignment(costs)
        for r, c in zip(r_ind, c_ind):
            h = hg[int(r)]
            l = lg[int(c)]
            pair_rows.append(
                {
                    "policy": policy["name"],
                    "hard_key": "|".join(str(x) for x in key),
                    "cost": float(costs[int(r), int(c)]),
                    "high_dataset": h["dataset"],
                    "low_dataset": l["dataset"],
                    "high_source_family": h["source_family"],
                    "low_source_family": l["source_family"],
                    "high_condition": h["condition"],
                    "low_condition": l["condition"],
                    "perturbation_type": h["perturbation_type"],
                    "gene_count_bin": h["gene_count_bin"],
                    "high_propensity": h["coverage_propensity"],
                    "low_propensity": l["coverage_propensity"],
                    "high_residual": h["coverage_residual"],
                    "low_residual": l["coverage_residual"],
                    "residual_gap": h["coverage_residual"] - l["coverage_residual"],
                    "high_response_norm": h["response_norm"],
                    "low_response_norm": l["response_norm"],
                    "high_n_ctrl": h["n_ctrl"],
                    "low_n_ctrl": l["n_ctrl"],
                    "high_n_gt": h["n_gt"],
                    "low_n_gt": l["n_gt"],
                    "high_analog_support_dataset_count": h["analog_support_dataset_count"],
                    "low_analog_support_dataset_count": l["analog_support_dataset_count"],
                }
            )
    pair_rows.sort(key=lambda row: float(row["cost"]))
    max_side = int(policy["max_per_side_dataset"])
    max_pair = int(policy["max_per_dataset_pair"])
    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    for pair in pair_rows:
        hd = pair["high_dataset"]
        ld = pair["low_dataset"]
        if high_counts[hd] >= max_side or low_counts[ld] >= max_side:
            continue
        if pair_counts[(hd, ld)] >= max_pair:
            continue
        selected.append(pair)
        high_counts[hd] += 1
        low_counts[ld] += 1
        pair_counts[(hd, ld)] += 1
    return selected


def summarize(policy: dict[str, Any], pairs: list[dict[str, Any]], prop_auc: float) -> dict[str, Any]:
    high_ds = [pair["high_dataset"] for pair in pairs]
    low_ds = [pair["low_dataset"] for pair in pairs]
    high_src = [pair["high_source_family"] for pair in pairs]
    low_src = [pair["low_source_family"] for pair in pairs]
    smds = {
        "smd_propensity": smd([pair["high_propensity"] for pair in pairs], [pair["low_propensity"] for pair in pairs]),
        "smd_log_n_ctrl": smd([math.log1p(pair["high_n_ctrl"]) for pair in pairs], [math.log1p(pair["low_n_ctrl"]) for pair in pairs]),
        "smd_log_n_gt": smd([math.log1p(pair["high_n_gt"]) for pair in pairs], [math.log1p(pair["low_n_gt"]) for pair in pairs]),
        "smd_response_norm": smd([pair["high_response_norm"] for pair in pairs], [pair["low_response_norm"] for pair in pairs]),
        "smd_analog_support": smd(
            [pair["high_analog_support_dataset_count"] for pair in pairs],
            [pair["low_analog_support_dataset_count"] for pair in pairs],
        ),
    }
    max_abs_smd = max([abs(v) for v in smds.values() if math.isfinite(v)] or [float("nan")])
    dataset_js = js_divergence(Counter(high_ds), Counter(low_ds))
    source_js = js_divergence(Counter(high_src), Counter(low_src))
    top_pair_fraction = max_share([f"{pair['high_dataset']}->{pair['low_dataset']}" for pair in pairs])
    reasons: list[str] = []
    if prop_auc > 0.90:
        reasons.append("coverage_propensity_auc_gt_0p90_poor_overlap")
    if len(pairs) < 250:
        reasons.append("pairs_below_250")
    if len(set(high_ds) | set(low_ds)) < 12:
        reasons.append("datasets_below_12")
    if max_share(high_ds) > 0.25:
        reasons.append("high_dataset_max_share_gt_0p25")
    if max_share(low_ds) > 0.25:
        reasons.append("low_dataset_max_share_gt_0p25")
    if source_js > 0.25:
        reasons.append("source_js_gt_0p25")
    if dataset_js > 0.35:
        reasons.append("dataset_js_gt_0p35")
    if top_pair_fraction > 0.15:
        reasons.append("top_dataset_pair_fraction_gt_0p15")
    if max_abs_smd > 0.25:
        reasons.append("max_abs_covariate_smd_gt_0p25")
    return {
        "policy": policy["name"],
        "n_pairs": len(pairs),
        "n_total_datasets": len(set(high_ds) | set(low_ds)),
        "high_dataset_max_share": max_share(high_ds),
        "low_dataset_max_share": max_share(low_ds),
        "dataset_js_divergence": dataset_js,
        "source_js_divergence": source_js,
        "top_dataset_pair_fraction": top_pair_fraction,
        "median_residual_gap": float(np.median([pair["residual_gap"] for pair in pairs])) if pairs else float("nan"),
        "mean_high_propensity": float(np.mean([pair["high_propensity"] for pair in pairs])) if pairs else float("nan"),
        "mean_low_propensity": float(np.mean([pair["low_propensity"] for pair in pairs])) if pairs else float("nan"),
        **smds,
        "max_abs_covariate_smd": max_abs_smd,
        "reasons": ";".join(reasons),
        "feasibility_pass": not reasons,
    }


def split_from_pairs(parent: dict[str, Any], pairs: list[dict[str, Any]], side: str) -> dict[str, Any]:
    out = json.loads(json.dumps(parent))
    selected: dict[str, set[str]] = defaultdict(set)
    d_col = "high_dataset" if side == "high" else "low_dataset"
    c_col = "high_condition" if side == "high" else "low_condition"
    for pair in pairs:
        selected[pair[d_col]].add(pair[c_col])
    for dataset in out:
        out[dataset]["train"] = sorted(selected.get(dataset, set()))
    return out


def write_report(out_dir: Path, payload: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# LatentFM Exact-Coverage Propensity Residual Match Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only gate on the parent train split.",
        "- Coverage propensity is estimated from dataset/source, perturbation type, gene-count bin, cell counts, response norm, and analog support.",
        "- Matched high/low rows compare positive versus negative exact-coverage residuals inside common support.",
        "- No training, inference, checkpoint selection, canonical multi, or Track C query use.",
        "",
        "## Propensity Model",
        "",
        f"- parent train rows: `{payload['n_train_rows']}`",
        f"- exact covered / uncovered: `{payload['n_exact_high']}` / `{payload['n_uncovered_low']}`",
        f"- propensity AUC: `{fmt(payload['propensity_auc'])}`",
        f"- positive rate: `{fmt(payload['positive_rate'])}`",
        "",
        "## Matched Designs",
        "",
        "| policy | pairs | datasets | high max ds | low max ds | source JS | prop SMD | max SMD | median residual gap | pass | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['policy']}` | {row['n_pairs']} | {row['n_total_datasets']} | "
            f"{fmt(row['high_dataset_max_share'])} | {fmt(row['low_dataset_max_share'])} | "
            f"{fmt(row['source_js_divergence'])} | {fmt(row['smd_propensity'])} | "
            f"{fmt(row['max_abs_covariate_smd'])} | {fmt(row['median_residual_gap'])} | "
            f"`{row['feasibility_pass']}` | {row['reasons']} |"
        )
    best = payload.get("best_policy") or {}
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Best policy: `{best.get('policy', '')}` with `{best.get('n_pairs', 0)}` pairs.",
            f"- Decision: `{payload['decision']}`.",
            "- A pass would still require external audit and pair-shuffle/null calibration before GPU.",
            "- A fail keeps exact coverage as a descriptor/covariate rather than a launchable split.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{payload['json_path']}`",
            f"- Summary CSV: `{payload['summary_csv']}`",
            f"- Condition rows: `{payload['condition_rows_csv']}`",
            f"- Best pairs: `{payload['best_pairs_csv']}`",
            f"- Draft high split: `{payload['draft_high_split']}`",
            f"- Draft low split: `{payload['draft_low_split']}`",
            "",
        ]
    )
    (out_dir / "LATENTFM_EXACT_COVERAGE_PROPENSITY_RESIDUAL_MATCH_GATE_20260629.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-split", type=Path, default=PARENT_SPLIT)
    parser.add_argument("--condition-metadata", type=Path, default=COND_META)
    parser.add_argument("--observability-rows", type=Path, default=OBS_ROWS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_train_rows(args.parent_split, args.condition_metadata, args.observability_rows)
    prop = fit_propensity(rows)
    stats = standardize(
        rows,
        [
            "coverage_logit",
            "log_n_ctrl",
            "log_n_gt",
            "log_response_norm",
            "analog_support_dataset_count",
        ],
    )
    policies = [
        {
            "name": "common_ptype_gene_resid025_cap60",
            "hard_cols": ["perturbation_type", "gene_count_bin"],
            "min_propensity": 0.10,
            "max_propensity": 0.90,
            "high_residual_min": 0.25,
            "low_residual_max": -0.25,
            "source_penalty": 0.2,
            "max_per_side_dataset": 60,
            "max_per_dataset_pair": 25,
        },
        {
            "name": "common_ptype_gene_resid015_cap60",
            "hard_cols": ["perturbation_type", "gene_count_bin"],
            "min_propensity": 0.10,
            "max_propensity": 0.90,
            "high_residual_min": 0.15,
            "low_residual_max": -0.15,
            "source_penalty": 0.2,
            "max_per_side_dataset": 60,
            "max_per_dataset_pair": 25,
        },
        {
            "name": "wide_ptype_gene_resid015_cap80",
            "hard_cols": ["perturbation_type", "gene_count_bin"],
            "min_propensity": 0.05,
            "max_propensity": 0.95,
            "high_residual_min": 0.15,
            "low_residual_max": -0.15,
            "source_penalty": 0.2,
            "max_per_side_dataset": 80,
            "max_per_dataset_pair": 30,
        },
        {
            "name": "common_ptype_resid015_cap60",
            "hard_cols": ["perturbation_type"],
            "min_propensity": 0.10,
            "max_propensity": 0.90,
            "high_residual_min": 0.15,
            "low_residual_max": -0.15,
            "source_penalty": 0.2,
            "max_per_side_dataset": 60,
            "max_per_dataset_pair": 25,
        },
    ]
    all_pairs: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for policy in policies:
        pairs = make_pairs(rows, policy, stats)
        all_pairs[policy["name"]] = pairs
        summary_rows.append(summarize(policy, pairs, float(prop["auc"])))
    summary_rows.sort(
        key=lambda row: (
            bool(row["feasibility_pass"]),
            row["n_pairs"],
            -row["max_abs_covariate_smd"],
            -row["source_js_divergence"],
        ),
        reverse=True,
    )
    best = summary_rows[0] if summary_rows else {}
    best_pairs = all_pairs.get(str(best.get("policy", "")), [])
    parent = load_json(args.parent_split)

    condition_rows_csv = args.out_dir / "exact_coverage_propensity_condition_rows.csv"
    summary_csv = args.out_dir / "exact_coverage_propensity_match_summary.csv"
    best_pairs_csv = args.out_dir / "best_exact_coverage_propensity_residual_pairs.csv"
    high_split = args.out_dir / "draft_split_seed42_xverse_exact_coverage_propensity_high_from_cap120_all_v2.json"
    low_split = args.out_dir / "draft_split_seed42_xverse_exact_coverage_propensity_low_from_cap120_all_v2.json"
    json_path = args.out_dir / "latentfm_exact_coverage_propensity_residual_match_gate_20260629.json"

    write_csv(condition_rows_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_csv(best_pairs_csv, best_pairs)
    write_json(high_split, split_from_pairs(parent, best_pairs, "high"))
    write_json(low_split, split_from_pairs(parent, best_pairs, "low"))

    pass_any = any(bool(row["feasibility_pass"]) for row in summary_rows)
    status = (
        "exact_coverage_propensity_residual_match_pass_external_audit_only_no_gpu"
        if pass_any
        else "exact_coverage_propensity_residual_match_fail_no_gpu"
    )
    decision = (
        "external_audit_then_pairshuffle_null_before_gpu"
        if pass_any
        else "exact_coverage_descriptor_only_or_new_axis_needed"
    )
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "decision": decision,
        "parent_split": str(args.parent_split),
        "n_train_rows": len(rows),
        "n_exact_high": int(sum(1 for row in rows if row["exact_train_covered"])),
        "n_uncovered_low": int(sum(1 for row in rows if not row["exact_train_covered"])),
        "propensity_auc": prop["auc"],
        "positive_rate": prop["positive_rate"],
        "best_policy": best,
        "summary_rows": summary_rows,
        "json_path": str(json_path),
        "summary_csv": str(summary_csv),
        "condition_rows_csv": str(condition_rows_csv),
        "best_pairs_csv": str(best_pairs_csv),
        "draft_high_split": str(high_split),
        "draft_low_split": str(low_split),
    }
    write_json(json_path, payload)
    write_report(args.out_dir, payload, summary_rows)
    print(json.dumps({"status": status, "best_policy": best.get("policy"), "n_pairs": best.get("n_pairs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
