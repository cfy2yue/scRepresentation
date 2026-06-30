#!/usr/bin/env python3
"""CPU gate for target/gene-coverage scaling evidence.

This audit reads existing train-only scaling posthoc artifacts and split JSONs.
It does not train, launch GPU work, read canonical metrics for selection, read
canonical multi, or touch Track C query artifacts.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624"
OUT_JSON = ROOT / "reports/latentfm_scaling_target_gene_coverage_protocol_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_TARGET_GENE_COVERAGE_PROTOCOL_GATE_20260624.md"

RUNS = {
    "cap30_all": RUN_ROOT / "xverse_scaling_cap30_all_3k_seed42",
    "cap120_all": RUN_ROOT / "xverse_scaling_cap120_all_3k_seed42",
    "gene_cap120_allbg": RUN_ROOT / "xverse_scaling_gene_cap120_allbg_3k_seed42",
    "gene_cap120_k562bg": RUN_ROOT / "xverse_scaling_gene_cap120_k562bg_3k_seed42",
    "full_trainonly": RUN_ROOT / "xverse_scaling_full_trainonly_3k_seed42",
}

GROUP = "internal_val_cross_background_seen_gene_proxy"
FAMILY_GROUP = "internal_val_family_gene_proxy"
SEED = 42


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def genes_for_condition(cond: str) -> set[str]:
    return {part.strip() for part in str(cond).split("+") if part.strip()}


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float:
    return corr(rank(xs), rank(ys))


def permutation_p_abs(xs: list[float], ys: list[float], *, n_perm: int = 2000, seed: int = SEED) -> float:
    obs = abs(spearman(xs, ys))
    if not math.isfinite(obs):
        return 1.0
    rng = random.Random(seed)
    y = list(ys)
    hit = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        val = abs(spearman(xs, y))
        if math.isfinite(val) and val >= obs:
            hit += 1
    return (hit + 1.0) / (n_perm + 1.0)


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def condition_rows(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def load_run(run_dir: Path) -> dict[str, Any]:
    eval_dir = run_dir / "posthoc_eval_internal"
    cand = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
    anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
    return {"candidate": cand, "anchor": anchor, "split": load_json(Path(cand["split_file"]))}


def train_gene_stats(split: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"datasets": set(), "conditions": 0})
    for ds, groups in split.items():
        for cond in groups.get("train") or []:
            genes = genes_for_condition(str(cond))
            for gene in genes:
                stats[gene]["datasets"].add(str(ds))
                stats[gene]["conditions"] += 1
    return stats


def coverage_for(stats: dict[str, dict[str, Any]], cond: str) -> dict[str, float]:
    genes = sorted(genes_for_condition(cond))
    if not genes:
        return {"gene_dataset_count_min": 0.0, "gene_dataset_count_sum": 0.0, "gene_condition_count_sum": 0.0}
    ds_counts = [len(stats.get(g, {}).get("datasets", set())) for g in genes]
    cond_counts = [int(stats.get(g, {}).get("conditions", 0)) for g in genes]
    return {
        "gene_dataset_count_min": float(min(ds_counts)),
        "gene_dataset_count_sum": float(sum(ds_counts)),
        "gene_condition_count_sum": float(sum(cond_counts)),
    }


def paired_rows(runs: dict[str, dict[str, Any]], arm_a: str, arm_b: str, group: str) -> list[dict[str, Any]]:
    a_c = condition_rows(runs[arm_a]["candidate"], group)
    b_c = condition_rows(runs[arm_b]["candidate"], group)
    a_a = condition_rows(runs[arm_a]["anchor"], group)
    b_a = condition_rows(runs[arm_b]["anchor"], group)
    stats_a = train_gene_stats(runs[arm_a]["split"])
    stats_b = train_gene_stats(runs[arm_b]["split"])
    rows = []
    for key in sorted(set(a_c) & set(b_c) & set(a_a) & set(b_a)):
        ds, cond = key
        cov_a = coverage_for(stats_a, cond)
        cov_b = coverage_for(stats_b, cond)
        vals = {
            "a_c_pp": a_c[key].get("pearson_pert"),
            "a_a_pp": a_a[key].get("pearson_pert"),
            "b_c_pp": b_c[key].get("pearson_pert"),
            "b_a_pp": b_a[key].get("pearson_pert"),
            "a_c_mmd": a_c[key].get("test_mmd_clamped"),
            "a_a_mmd": a_a[key].get("test_mmd_clamped"),
            "b_c_mmd": b_c[key].get("test_mmd_clamped"),
            "b_a_mmd": b_a[key].get("test_mmd_clamped"),
        }
        if any(v is None for v in vals.values()):
            continue
        row = {
            "dataset": ds,
            "condition": cond,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "pp_delta_a_vs_anchor": float(vals["a_c_pp"]) - float(vals["a_a_pp"]),
            "pp_delta_b_vs_anchor": float(vals["b_c_pp"]) - float(vals["b_a_pp"]),
            "mmd_delta_a_vs_anchor": float(vals["a_c_mmd"]) - float(vals["a_a_mmd"]),
            "mmd_delta_b_vs_anchor": float(vals["b_c_mmd"]) - float(vals["b_a_mmd"]),
            "coverage_a": cov_a,
            "coverage_b": cov_b,
        }
        row["pp_delta_b_minus_a"] = row["pp_delta_b_vs_anchor"] - row["pp_delta_a_vs_anchor"]
        row["mmd_delta_b_minus_a"] = row["mmd_delta_b_vs_anchor"] - row["mmd_delta_a_vs_anchor"]
        row["coverage_gain_dataset_count"] = cov_b["gene_dataset_count_sum"] - cov_a["gene_dataset_count_sum"]
        row["coverage_gain_condition_count"] = cov_b["gene_condition_count_sum"] - cov_a["gene_condition_count_sum"]
        row["coverage_b_dataset_count"] = cov_b["gene_dataset_count_sum"]
        row["coverage_b_condition_count"] = cov_b["gene_condition_count_sum"]
        rows.append(row)
    return rows


def bucket_label(value: float) -> str:
    if value <= 0:
        return "0"
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    return "8+"


def summarize_pair(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pp = [r["pp_delta_b_minus_a"] for r in rows]
    mmd = [r["mmd_delta_b_minus_a"] for r in rows]
    cov_gain = [r["coverage_gain_dataset_count"] for r in rows]
    cov_abs = [r["coverage_b_dataset_count"] for r in rows]
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[bucket_label(row["coverage_b_dataset_count"])].append(row)
        by_dataset[row["dataset"]].append(row)
    dataset_means = {
        ds: {
            "n": len(ds_rows),
            "pp_delta_mean": mean([r["pp_delta_b_minus_a"] for r in ds_rows]),
            "mmd_delta_mean": mean([r["mmd_delta_b_minus_a"] for r in ds_rows]),
        }
        for ds, ds_rows in sorted(by_dataset.items())
    }
    return {
        "n_conditions": len(rows),
        "pp_delta_mean": mean(pp),
        "mmd_delta_mean": mean(mmd),
        "spearman_pp_vs_coverage_gain_dataset_count": spearman(cov_gain, pp),
        "spearman_pp_vs_coverage_gain_dataset_count_perm_p_abs": permutation_p_abs(cov_gain, pp, seed=SEED + 1),
        "spearman_pp_vs_absolute_coverage_dataset_count": spearman(cov_abs, pp),
        "spearman_pp_vs_absolute_coverage_dataset_count_perm_p_abs": permutation_p_abs(cov_abs, pp, seed=SEED + 2),
        "spearman_mmd_vs_coverage_gain_dataset_count": spearman(cov_gain, mmd),
        "bucket_rows": {
            bucket: {
                "n": len(bucket_rows),
                "pp_delta_mean": mean([r["pp_delta_b_minus_a"] for r in bucket_rows]),
                "mmd_delta_mean": mean([r["mmd_delta_b_minus_a"] for r in bucket_rows]),
            }
            for bucket, bucket_rows in sorted(by_bucket.items())
        },
        "dataset_means": dataset_means,
        "min_dataset_pp_delta": min(
            (float(v["pp_delta_mean"]) for v in dataset_means.values() if v["pp_delta_mean"] is not None),
            default=None,
        ),
    }


def decide(cap120_vs_cap30: dict[str, Any], full_vs_cap120: dict[str, Any]) -> tuple[str, list[str]]:
    reasons = []
    if int(cap120_vs_cap30.get("n_conditions") or 0) < 100:
        reasons.append("cap120_vs_cap30_n_lt_100")
    if float(cap120_vs_cap30.get("pp_delta_mean") or -999.0) < 0.001:
        reasons.append("cap120_vs_cap30_mean_pp_gain_lt_0p001")
    rho = float(cap120_vs_cap30.get("spearman_pp_vs_coverage_gain_dataset_count") or 0.0)
    pval = float(cap120_vs_cap30.get("spearman_pp_vs_coverage_gain_dataset_count_perm_p_abs") or 1.0)
    if rho < 0.25 or pval > 0.05:
        reasons.append("coverage_gain_not_significantly_predictive_of_pp_gain")
    if float(cap120_vs_cap30.get("mmd_delta_mean") or 0.0) > 0.002:
        reasons.append("cap120_vs_cap30_mmd_harm")
    if float(cap120_vs_cap30.get("min_dataset_pp_delta") or -999.0) < -0.02:
        reasons.append("cap120_vs_cap30_dataset_tail_harm")
    if float(full_vs_cap120.get("pp_delta_mean") or -999.0) < -0.005:
        reasons.append("full_vs_cap120_negative_nonmonotonic")
    status = "target_gene_coverage_gate_pass_one_split_builder_next" if not reasons else "target_gene_coverage_gate_fail_no_gpu"
    return status, reasons


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM Scaling Target/Gene Coverage Protocol Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only audit of existing train-only scaling posthoc and split artifacts.",
        "- Does not train, launch GPU, read canonical multi, or read Track C query.",
        "- Canonical metrics are not used for this gate.",
        "",
        "## Primary Pairwise Tests",
        "",
        "| comparison | n | mean pp delta | mean MMD delta | rho pp~coverage gain | perm p | min dataset pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["comparisons"].items():
        lines.append(
            f"| `{name}` | {row['n_conditions']} | {fmt(row['pp_delta_mean'])} | "
            f"{fmt(row['mmd_delta_mean'])} | {fmt(row['spearman_pp_vs_coverage_gain_dataset_count'])} | "
            f"{fmt(row['spearman_pp_vs_coverage_gain_dataset_count_perm_p_abs'])} | "
            f"{fmt(row['min_dataset_pp_delta'])} |"
        )
    lines.extend(["", "## Coverage Buckets: cap120 minus cap30", "", "| bucket | n | pp delta | MMD delta |", "|---|---:|---:|---:|"])
    for bucket, row in payload["comparisons"]["cap120_minus_cap30"]["bucket_rows"].items():
        lines.append(f"| `{bucket}` | {row['n']} | {fmt(row['pp_delta_mean'])} | {fmt(row['mmd_delta_mean'])} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{payload['reasons']}`",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            "",
            "Interpretation:",
            "",
            "- A pass would only authorize building a new target-coverage split protocol; it would not authorize canonical no-harm or promotion.",
            "- A fail means target/gene coverage should be reported as a diagnostic/figure axis from existing artifacts, not expanded to GPU now.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    runs = {name: load_run(path) for name, path in RUNS.items()}
    comparisons = {
        "cap120_minus_cap30": summarize_pair(paired_rows(runs, "cap30_all", "cap120_all", GROUP)),
        "full_minus_cap120": summarize_pair(paired_rows(runs, "cap120_all", "full_trainonly", GROUP)),
        "gene_allbg_minus_cap120": summarize_pair(paired_rows(runs, "cap120_all", "gene_cap120_allbg", GROUP)),
        "k562bg_minus_gene_allbg": summarize_pair(paired_rows(runs, "gene_cap120_allbg", "gene_cap120_k562bg", GROUP)),
        "cap120_minus_cap30_family_proxy": summarize_pair(paired_rows(runs, "cap30_all", "cap120_all", FAMILY_GROUP)),
    }
    status, reasons = decide(comparisons["cap120_minus_cap30"], comparisons["full_minus_cap120"])
    payload = {
        "status": status,
        "gpu_authorized": status.endswith("_next"),
        "reasons": reasons,
        "boundary": {
            "run_root": str(RUN_ROOT),
            "group": GROUP,
            "family_group": FAMILY_GROUP,
            "reads_train_only_posthoc": True,
            "reads_split_json": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "launches_gpu": False,
        },
        "comparisons": comparisons,
        "next_action": (
            "build a target/gene-coverage split protocol with shuffled-target controls"
            if status.endswith("_next")
            else "use existing artifacts for a diagnostic scaling-law figure; do not launch target/gene-coverage GPU"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_md(payload)
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": payload["gpu_authorized"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
