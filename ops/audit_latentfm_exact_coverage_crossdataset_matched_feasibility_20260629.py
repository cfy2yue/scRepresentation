#!/usr/bin/env python3
"""Cross-dataset exact-coverage matched-design feasibility audit.

CPU/report-only. This asks whether exact raw-expression train coverage can form
a larger matched high/low design if we relax same-dataset pairing but keep
perturbation type and gene-count strata fixed and audit dataset imbalance.

The output is a launch-design feasibility artifact only. It does not authorize
GPU training, checkpoint selection, canonical multi use, or Track C query use.
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
OUT_DIR = ROOT / "reports/exact_coverage_crossdataset_matched_feasibility_20260629"
SEED = 42


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


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def perturbation_type(meta: dict[str, Any]) -> str:
    return str(meta.get("perturbation_type_raw") or "unknown")


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
    h = np.asarray([x for x in high if math.isfinite(float(x))], dtype=float)
    l = np.asarray([x for x in low if math.isfinite(float(x))], dtype=float)
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


def load_train_rows(
    parent_split: Path,
    cond_meta_path: Path,
    obs_rows_path: Path,
) -> list[dict[str, Any]]:
    parent = load_json(parent_split)
    cond_meta = load_json(cond_meta_path)
    obs_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(obs_rows_path):
        obs_lookup[(row["dataset"], row["condition"])] = row

    split_rows = [{"split_file": str(parent_split.relative_to(ROOT)), "split_name": parent_split.stem}]
    cache, missing = load_condition_vector_cache(
        ROOT / "dataset/latentfm_full/xverse",
        collect_needed_conditions(split_rows),
    )
    if missing:
        print(f"warning: missing vectors for {len(missing)} conditions", file=sys.stderr)

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


def standardize_columns(rows: list[dict[str, Any]], cols: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in cols:
        vals = np.asarray([float(row[col]) for row in rows], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        stats[col] = (mean, std if std > 1e-12 else 1.0)
    return stats


def cost_matrix(
    high: list[dict[str, Any]],
    low: list[dict[str, Any]],
    stats: dict[str, tuple[float, float]],
    cols: list[str],
    dataset_penalty: float,
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
            if h["dataset"] != l["dataset"]:
                cost += dataset_penalty
            if h["source_family"] != l["source_family"]:
                cost += source_penalty
            out[i, j] = cost
    return out


def make_pairs_for_policy(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    stats: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    hard_cols = policy["hard_cols"]
    high_all = [row for row in rows if row["exact_train_covered"]]
    low_all = [row for row in rows if not row["exact_train_covered"]]
    high_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    low_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in high_all:
        high_groups[tuple(row[col] for col in hard_cols)].append(row)
    for row in low_all:
        low_groups[tuple(row[col] for col in hard_cols)].append(row)

    raw_pairs: list[dict[str, Any]] = []
    cost_cols = ["log_response_norm", "log_n_ctrl", "log_n_gt", "analog_support_dataset_count"]
    for key in sorted(set(high_groups) & set(low_groups)):
        high = high_groups[key]
        low = low_groups[key]
        costs = cost_matrix(
            high,
            low,
            stats,
            cost_cols,
            float(policy.get("dataset_penalty", 0.0)),
            float(policy.get("source_penalty", 0.0)),
        )
        r_ind, c_ind = linear_sum_assignment(costs)
        for r, c in zip(r_ind, c_ind):
            h = high[int(r)]
            l = low[int(c)]
            raw_pairs.append(
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

    raw_pairs.sort(key=lambda row: float(row["cost"]))
    max_side = int(policy.get("max_per_side_dataset") or 0)
    max_pair_dir = int(policy.get("max_per_dataset_pair") or 0)
    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    selected: list[dict[str, Any]] = []
    for pair in raw_pairs:
        hd = pair["high_dataset"]
        ld = pair["low_dataset"]
        if max_side and high_counts[hd] >= max_side:
            continue
        if max_side and low_counts[ld] >= max_side:
            continue
        if max_pair_dir and pair_counts[(hd, ld)] >= max_pair_dir:
            continue
        high_counts[hd] += 1
        low_counts[ld] += 1
        pair_counts[(hd, ld)] += 1
        selected.append(pair)
    return selected


def summarize_pairs(policy: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    features = [
        ("response_norm", "high_response_norm", "low_response_norm"),
        ("log_n_ctrl", "high_n_ctrl", "low_n_ctrl"),
        ("log_n_gt", "high_n_gt", "low_n_gt"),
        (
            "analog_support_dataset_count",
            "high_analog_support_dataset_count",
            "low_analog_support_dataset_count",
        ),
    ]
    smds: dict[str, float] = {}
    for name, h_col, l_col in features:
        if name.startswith("log_n"):
            high = [math.log1p(float(pair[h_col])) for pair in pairs]
            low = [math.log1p(float(pair[l_col])) for pair in pairs]
        else:
            high = [float(pair[h_col]) for pair in pairs]
            low = [float(pair[l_col]) for pair in pairs]
        smds[f"smd_{name}"] = smd(high, low)

    high_datasets = [pair["high_dataset"] for pair in pairs]
    low_datasets = [pair["low_dataset"] for pair in pairs]
    high_sources = [pair["high_source_family"] for pair in pairs]
    low_sources = [pair["low_source_family"] for pair in pairs]
    ptypes = Counter(pair["perturbation_type"] for pair in pairs)
    gene_bins = Counter(pair["gene_count_bin"] for pair in pairs)
    dataset_js = js_divergence(Counter(high_datasets), Counter(low_datasets))
    source_js = js_divergence(Counter(high_sources), Counter(low_sources))
    top_pair_fraction = max_share([f"{pair['high_dataset']}->{pair['low_dataset']}" for pair in pairs])
    max_abs_smd = max([abs(v) for v in smds.values() if math.isfinite(v)] or [float("nan")])

    reasons: list[str] = []
    if len(pairs) < 300:
        reasons.append("pairs_below_300")
    if len(set(high_datasets) | set(low_datasets)) < 12:
        reasons.append("datasets_below_12")
    if max_share(high_datasets) > 0.25:
        reasons.append("high_dataset_max_share_gt_0p25")
    if max_share(low_datasets) > 0.25:
        reasons.append("low_dataset_max_share_gt_0p25")
    if dataset_js > 0.35:
        reasons.append("dataset_js_gt_0p35")
    if source_js > 0.25:
        reasons.append("source_js_gt_0p25")
    if top_pair_fraction > 0.15:
        reasons.append("top_dataset_pair_fraction_gt_0p15")
    if max_abs_smd > 0.25:
        reasons.append("max_abs_covariate_smd_gt_0p25")
    if len(ptypes) < 2:
        reasons.append("perturbation_types_below_2")

    return {
        "policy": policy["name"],
        "n_pairs": len(pairs),
        "n_high_datasets": len(set(high_datasets)),
        "n_low_datasets": len(set(low_datasets)),
        "n_total_datasets": len(set(high_datasets) | set(low_datasets)),
        "high_dataset_max_share": max_share(high_datasets),
        "low_dataset_max_share": max_share(low_datasets),
        "high_source_max_share": max_share(high_sources),
        "low_source_max_share": max_share(low_sources),
        "dataset_js_divergence": dataset_js,
        "source_js_divergence": source_js,
        "top_dataset_pair_fraction": top_pair_fraction,
        "perturbation_type_counts": dict(sorted(ptypes.items())),
        "gene_count_bin_counts": dict(sorted(gene_bins.items())),
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
        "# LatentFM Exact-Coverage Cross-Dataset Matched Feasibility",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only feasibility audit over the parent train split.",
        "- High means exact raw-expression train coverage; low means no exact train coverage.",
        "- Matching may cross datasets, but hard-matches perturbation type and gene-count bin unless noted by policy.",
        "- No model training, inference, checkpoint selection, canonical multi, or Track C query use.",
        "- A pass would authorize only external review plus an axis-specific pair-shuffle/null design, not GPU training.",
        "",
        "## Summary",
        "",
        "| policy | pairs | datasets | high max ds | low max ds | dataset JS | source JS | top pair | max SMD | pass | reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['policy']}` | {row['n_pairs']} | {row['n_total_datasets']} | "
            f"{fmt(row['high_dataset_max_share'])} | {fmt(row['low_dataset_max_share'])} | "
            f"{fmt(row['dataset_js_divergence'])} | {fmt(row['source_js_divergence'])} | "
            f"{fmt(row['top_dataset_pair_fraction'])} | {fmt(row['max_abs_covariate_smd'])} | "
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
            "- If no policy passes, exact coverage remains a scaling descriptor and manuscript covariate, not a launch split.",
            "- If a policy passes, next required gate is pair-shuffle/null calibration using the same matching policy and then external review.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{payload['json_path']}`",
            f"- Summary CSV: `{payload['summary_csv']}`",
            f"- Best pairs CSV: `{payload['best_pairs_csv']}`",
            f"- Draft high split: `{payload['draft_high_split']}`",
            f"- Draft low split: `{payload['draft_low_split']}`",
            "",
        ]
    )
    (out_dir / "LATENTFM_EXACT_COVERAGE_CROSSDATASET_MATCHED_FEASIBILITY_20260629.md").write_text(
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
    stats = standardize_columns(
        rows,
        ["log_response_norm", "log_n_ctrl", "log_n_gt", "analog_support_dataset_count"],
    )
    policies = [
        {
            "name": "cross_ptype_gene_cap80",
            "hard_cols": ["perturbation_type", "gene_count_bin"],
            "dataset_penalty": 0.0,
            "source_penalty": 0.2,
            "max_per_side_dataset": 80,
            "max_per_dataset_pair": 30,
        },
        {
            "name": "cross_ptype_gene_cap60",
            "hard_cols": ["perturbation_type", "gene_count_bin"],
            "dataset_penalty": 0.0,
            "source_penalty": 0.2,
            "max_per_side_dataset": 60,
            "max_per_dataset_pair": 25,
        },
        {
            "name": "cross_ptype_only_cap80",
            "hard_cols": ["perturbation_type"],
            "dataset_penalty": 0.0,
            "source_penalty": 0.2,
            "max_per_side_dataset": 80,
            "max_per_dataset_pair": 30,
        },
        {
            "name": "cross_ptype_gene_same_source_bonus",
            "hard_cols": ["perturbation_type", "gene_count_bin"],
            "dataset_penalty": 0.0,
            "source_penalty": -0.2,
            "max_per_side_dataset": 80,
            "max_per_dataset_pair": 30,
        },
    ]
    all_pairs: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for policy in policies:
        pairs = make_pairs_for_policy(rows, policy, stats)
        all_pairs[policy["name"]] = pairs
        summary_rows.append(summarize_pairs(policy, pairs))
    summary_rows.sort(key=lambda row: (bool(row["feasibility_pass"]), row["n_pairs"], -row["max_abs_covariate_smd"]), reverse=True)
    best = summary_rows[0] if summary_rows else {}
    best_pairs = all_pairs.get(best.get("policy", ""), [])

    parent = load_json(args.parent_split)
    best_pairs_csv = args.out_dir / "best_crossdataset_matched_pairs.csv"
    summary_csv = args.out_dir / "crossdataset_matched_design_summary.csv"
    high_split = args.out_dir / "draft_split_seed42_xverse_exact_coverage_crossdataset_high_from_cap120_all_v2.json"
    low_split = args.out_dir / "draft_split_seed42_xverse_exact_coverage_crossdataset_low_from_cap120_all_v2.json"
    write_csv(best_pairs_csv, best_pairs)
    write_csv(summary_csv, summary_rows)
    write_json(high_split, split_from_pairs(parent, best_pairs, "high"))
    write_json(low_split, split_from_pairs(parent, best_pairs, "low"))

    pass_any = any(bool(row["feasibility_pass"]) for row in summary_rows)
    status = (
        "exact_coverage_crossdataset_matched_feasibility_pass_external_audit_only_no_gpu"
        if pass_any
        else "exact_coverage_crossdataset_matched_feasibility_fail_no_gpu"
    )
    decision = (
        "external_audit_then_pairshuffle_null_before_gpu"
        if pass_any
        else "keep_exact_coverage_as_descriptor_or_redesign_matching"
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
        "best_policy": best,
        "summary_rows": summary_rows,
        "json_path": str(args.out_dir / "latentfm_exact_coverage_crossdataset_matched_feasibility_20260629.json"),
        "summary_csv": str(summary_csv),
        "best_pairs_csv": str(best_pairs_csv),
        "draft_high_split": str(high_split),
        "draft_low_split": str(low_split),
    }
    write_json(Path(payload["json_path"]), payload)
    write_report(args.out_dir, payload, summary_rows)
    print(json.dumps({"status": status, "best_policy": best.get("policy"), "n_pairs": best.get("n_pairs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
