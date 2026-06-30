#!/usr/bin/env python3
"""Build a deployable Track A benchmark and failure taxonomy for xverse_8k_anchor.

This is a CPU/report-only script. It reads frozen evaluation JSON files from the
existing seed42 anchor and seed43 replicate, writes provenance manifests and
summary CSV/JSON/Markdown, and does not train, infer, read Track C query, or use
canonical multi for checkpoint selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports" / "tracka_deployable_benchmark_failure_taxonomy_20260627"
MD_OUT = ROOT / "reports" / "LATENTFM_TRACKA_DEPLOYABLE_BENCHMARK_FAILURE_TAXONOMY_20260627.md"
JSON_OUT = ROOT / "reports" / "latentfm_tracka_deployable_benchmark_failure_taxonomy_20260627.json"

ANCHOR_ROOT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval"
)
REPLICATE_ROOT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
    / "xverse_comp006_endpoint5_8k_seed43_fulleval"
)

INPUTS = {
    "seed42_iid": ANCHOR_ROOT / "iid_eval_results.json",
    "seed42_split": ANCHOR_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed42_family": ANCHOR_ROOT
    / "posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43_iid": REPLICATE_ROOT / "iid_eval_results.json",
    "seed43_split": REPLICATE_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43_family": REPLICATE_ROOT
    / "posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
}

METRICS = [
    "test_mse",
    "test_mae",
    "test_mmd",
    "test_mmd_clamped",
    "direct_pearson",
    "pearson_ctrl",
    "pearson_pert",
]

PRIMARY_GROUPS = {
    "iid_all",
    "test_all",
    "test",
    "test_single",
    "family_gene",
}

MULTI_DIAGNOSTIC = {
    "test_multi",
    "test_multi_seen",
    "test_multi_unseen1",
    "test_multi_unseen2",
    "structure_multi",
}

REQUESTED_TRACKA = {
    "simple_single_unseen": {
        "status": "not_available_in_existing_anchor_json",
        "proxy": "",
        "note": "No explicit simple-single-unseen group in the available frozen xverse_8k JSON.",
    },
    "cross_background_seen_gene": {
        "status": "not_available_in_existing_anchor_json",
        "proxy": "",
        "note": "No explicit cross_background_seen_gene group in the available frozen xverse_8k JSON.",
    },
    "all_test_single": {
        "status": "proxy_available_as_test_single",
        "proxy": "test_single",
        "note": "Use only as a proxy label; do not rename it to all_test_single without a matching evaluator.",
    },
    "family_gene": {
        "status": "available",
        "proxy": "family_gene",
        "note": "Available in the condition-family posthoc JSON.",
    },
    "canonical_multi": {
        "status": "diagnostic_zero_selection_weight",
        "proxy": "test_multi*",
        "note": "Canonical multi groups are zero-shot composition diagnostics only, never Track A checkpoint selection.",
    },
}


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(label: str, offset: int = 0) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + offset) % (2**32)


def finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(out):
        return out
    return None


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_mean(values: list[float], seed: int, n_boot: int = 2000) -> dict:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    samples = []
    n = len(vals)
    for _ in range(n_boot):
        samples.append(sum(rng.choice(vals) for _ in range(n)) / n)
    samples.sort()
    return {
        "n": n,
        "mean": mean(vals),
        "ci_low": quantile(samples, 0.025),
        "ci_high": quantile(samples, 0.975),
    }


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "NA"
    return f"{val:+.{digits}f}"


def group_role(source: str, group: str) -> str:
    if group in MULTI_DIAGNOSTIC:
        return "diagnostic_zero_selection_weight"
    if group == "family_drug":
        return "drug_family_diagnostic"
    if group in PRIMARY_GROUPS:
        return "tracka_primary_or_noharm_context"
    if group == "structure_single":
        return "single_structure_diagnostic"
    return "diagnostic"


def all_group_records(payloads: dict[str, dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    summaries: list[dict] = []
    condition_rows: dict[str, list[dict]] = {}
    for key, payload in payloads.items():
        seed, source = key.split("_", 1)
        if source == "iid":
            groups = {"iid_all": payload}
        else:
            groups = payload.get("groups", {})
        for group, data in groups.items():
            rows = data.get("condition_metrics") or []
            row_key = f"{seed}:{source}:{group}"
            condition_rows[row_key] = rows
            pp_values = [finite_float(r.get("pearson_pert")) for r in rows]
            mmd_values = [finite_float(r.get("test_mmd_clamped")) for r in rows]
            pp_boot = bootstrap_mean([v for v in pp_values if v is not None], seed=stable_seed(row_key, 0))
            mmd_boot = bootstrap_mean([v for v in mmd_values if v is not None], seed=stable_seed(row_key, 17))
            rec = {
                "seed": seed,
                "source": source,
                "group": group,
                "role": group_role(source, group),
                "selection_weight": 0 if group in MULTI_DIAGNOSTIC else 1,
                "n_conds": data.get("n_conds"),
                "n_rows": len(rows),
                "checkpoint": payload.get("checkpoint", ""),
                "checkpoint_step": payload.get("checkpoint_step", ""),
                "used_ema": payload.get("used_ema", ""),
                "pp_condition_boot_mean": pp_boot["mean"],
                "pp_condition_boot_ci_low": pp_boot["ci_low"],
                "pp_condition_boot_ci_high": pp_boot["ci_high"],
                "mmd_condition_boot_mean": mmd_boot["mean"],
                "mmd_condition_boot_ci_low": mmd_boot["ci_low"],
                "mmd_condition_boot_ci_high": mmd_boot["ci_high"],
            }
            for metric in METRICS:
                rec[f"reported_{metric}"] = data.get(metric)
            summaries.append(rec)
    return summaries, condition_rows


def dataset_breakdown(condition_rows: dict[str, list[dict]]) -> list[dict]:
    selected_keys = [
        k
        for k in condition_rows
        if k.endswith(":test_all")
        or k.endswith(":test")
        or k.endswith(":test_single")
        or k.endswith(":family_gene")
        or k.endswith(":family_drug")
        or k.endswith(":test_multi")
    ]
    out = []
    for key in selected_keys:
        seed, source, group = key.split(":", 2)
        by_ds: dict[str, list[dict]] = defaultdict(list)
        for row in condition_rows[key]:
            by_ds[str(row.get("dataset", "NA"))].append(row)
        for dataset, rows in sorted(by_ds.items()):
            pp = [finite_float(r.get("pearson_pert")) for r in rows]
            pp = [v for v in pp if v is not None]
            mmd = [finite_float(r.get("test_mmd_clamped")) for r in rows]
            mmd = [v for v in mmd if v is not None]
            sorted_by_pp = sorted(rows, key=lambda r: finite_float(r.get("pearson_pert")) or -999)
            worst_conditions = ";".join(str(r.get("condition", "")) for r in sorted_by_pp[:5])
            out.append(
                {
                    "seed": seed,
                    "source": source,
                    "group": group,
                    "role": group_role(source, group),
                    "dataset": dataset,
                    "n_rows": len(rows),
                    "mean_pearson_pert": mean(pp) if pp else None,
                    "min_pearson_pert": min(pp) if pp else None,
                    "frac_pp_below_0": sum(1 for v in pp if v < 0) / len(pp) if pp else None,
                    "mean_mmd_clamped": mean(mmd) if mmd else None,
                    "max_mmd_clamped": max(mmd) if mmd else None,
                    "worst_conditions_by_pp": worst_conditions,
                }
            )
    return out


def failure_tags(group: str, row: dict) -> str:
    tags = []
    pp = finite_float(row.get("pearson_pert"))
    mmd = finite_float(row.get("test_mmd_clamped"))
    if group in MULTI_DIAGNOSTIC:
        tags.append("multi_diagnostic_zero_weight")
    if group == "family_drug":
        tags.append("drug_family_diagnostic")
    if pp is not None and pp < 0:
        tags.append("negative_pp")
    if pp is not None and pp < 0.05:
        tags.append("low_pp_lt_0p05")
    if mmd is not None and mmd > 0.05:
        tags.append("high_mmd_gt_0p05")
    if not tags:
        tags.append("watch")
    return ";".join(tags)


def failure_cases(condition_rows: dict[str, list[dict]]) -> list[dict]:
    wanted_suffixes = [":test_all", ":test", ":test_single", ":family_gene", ":family_drug", ":test_multi"]
    rows_out = []
    for key, rows in condition_rows.items():
        if not any(key.endswith(s) for s in wanted_suffixes):
            continue
        seed, source, group = key.split(":", 2)
        if seed != "seed42":
            continue
        for row in rows:
            rows_out.append(
                {
                    "seed": seed,
                    "source": source,
                    "group": group,
                    "role": group_role(source, group),
                    "selection_weight": 0 if group in MULTI_DIAGNOSTIC else 1,
                    "dataset": row.get("dataset", ""),
                    "condition": row.get("condition", ""),
                    "pearson_pert": finite_float(row.get("pearson_pert")),
                    "pearson_ctrl": finite_float(row.get("pearson_ctrl")),
                    "direct_pearson": finite_float(row.get("direct_pearson")),
                    "test_mmd_clamped": finite_float(row.get("test_mmd_clamped")),
                    "n_src_eval": row.get("n_src_eval", ""),
                    "n_gt_eval": row.get("n_gt_eval", ""),
                    "failure_tags": failure_tags(group, row),
                }
            )
    rows_out.sort(
        key=lambda r: (
            r["pearson_pert"] if r["pearson_pert"] is not None else 999,
            -(r["test_mmd_clamped"] if r["test_mmd_clamped"] is not None else -999),
        )
    )
    return rows_out[:100]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def replicate_summary(group_rows: list[dict]) -> list[dict]:
    by_group: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in group_rows:
        if row["seed"] not in {"seed42", "seed43"}:
            continue
        by_group[(row["source"], row["group"])][row["seed"]] = row
    out = []
    for (source, group), rows in sorted(by_group.items()):
        if "seed42" not in rows or "seed43" not in rows:
            continue
        a, b = rows["seed42"], rows["seed43"]
        out.append(
            {
                "source": source,
                "group": group,
                "role": group_role(source, group),
                "selection_weight": 0 if group in MULTI_DIAGNOSTIC else 1,
                "seed42_reported_pearson_pert": a.get("reported_pearson_pert"),
                "seed43_reported_pearson_pert": b.get("reported_pearson_pert"),
                "delta_seed43_minus_seed42_pearson_pert": finite_float(b.get("reported_pearson_pert"))
                - finite_float(a.get("reported_pearson_pert")),
                "seed42_reported_mmd_clamped": a.get("reported_test_mmd_clamped"),
                "seed43_reported_mmd_clamped": b.get("reported_test_mmd_clamped"),
                "delta_seed43_minus_seed42_mmd_clamped": finite_float(b.get("reported_test_mmd_clamped"))
                - finite_float(a.get("reported_test_mmd_clamped")),
            }
        )
    return out


def md_table(rows: list[dict], fields: list[str], max_rows: int = 20) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows[:max_rows]:
        vals = []
        for field in fields:
            val = row.get(field)
            vals.append(fmt(val) if isinstance(val, (float, int)) or val is None else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = {k: load_json(v) for k, v in INPUTS.items()}
    manifest = [
        {
            "input_key": key,
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else "",
        }
        for key, path in INPUTS.items()
    ]
    groups, condition_rows = all_group_records(payloads)
    datasets = dataset_breakdown(condition_rows)
    failures = failure_cases(condition_rows)
    replicate = replicate_summary(groups)

    group_csv = OUT_DIR / "group_summary.csv"
    dataset_csv = OUT_DIR / "dataset_breakdown.csv"
    failure_csv = OUT_DIR / "failure_cases.csv"
    replicate_csv = OUT_DIR / "seed_replicate_summary.csv"
    manifest_tsv = OUT_DIR / "input_manifest.tsv"
    write_csv(group_csv, groups)
    write_csv(dataset_csv, datasets)
    write_csv(failure_csv, failures)
    write_csv(replicate_csv, replicate)
    with manifest_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_key", "path", "exists", "sha256"], delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)

    primary_seed42 = [
        r
        for r in groups
        if r["seed"] == "seed42"
        and r["source"] in {"iid", "family", "split"}
        and r["group"] in {"iid_all", "test_all", "test", "test_single", "family_gene", "family_drug"}
    ]
    primary_seed42.sort(key=lambda r: (r["source"], r["group"]))

    worst_primary = [
        r
        for r in failures
        if r["role"] in {"tracka_primary_or_noharm_context", "drug_family_diagnostic"}
    ][:20]

    result = {
        "status": "tracka_deployable_benchmark_failure_taxonomy_ready_no_gpu",
        "gpu_authorized": False,
        "default_model": "xverse_8k_anchor",
        "boundary": {
            "cpu_report_only": True,
            "no_training": True,
            "no_inference": True,
            "no_trackc_query": True,
            "canonical_multi_selection_weight": 0,
            "canonical_multi_role": "diagnostic_only",
        },
        "requested_tracka_metric_availability": REQUESTED_TRACKA,
        "outputs": {
            "group_summary_csv": str(group_csv),
            "dataset_breakdown_csv": str(dataset_csv),
            "failure_cases_csv": str(failure_csv),
            "seed_replicate_summary_csv": str(replicate_csv),
            "input_manifest_tsv": str(manifest_tsv),
            "markdown": str(MD_OUT),
            "json": str(JSON_OUT),
        },
        "group_summary": groups,
        "dataset_breakdown": datasets,
        "failure_cases": failures,
        "seed_replicate_summary": replicate,
        "input_manifest": manifest,
    }
    JSON_OUT.write_text(json.dumps(result, indent=2, sort_keys=True))

    md = []
    md.append("# LatentFM Track A Deployable Benchmark + Failure Taxonomy")
    md.append("")
    md.append("Timestamp: `2026-06-27 CST`")
    md.append("")
    md.append("Status: `tracka_deployable_benchmark_failure_taxonomy_ready_no_gpu`")
    md.append("")
    md.append("GPU authorized: `False`")
    md.append("")
    md.append("Default/deployable model: `xverse_8k_anchor`")
    md.append("")
    md.append("## Boundary")
    md.append("")
    md.append("- CPU/report-only over frozen xverse_8k seed42 anchor and seed43 replicate eval JSON.")
    md.append("- Does not train, infer, read Track C query, recut splits, or select checkpoints.")
    md.append("- Canonical `test_multi*` groups are diagnostic only with `selection_weight=0`.")
    md.append("- Condition-bootstrap CIs are row-level descriptive CIs, not paired model-delta CIs.")
    md.append("")
    md.append("## Requested Track A Metric Availability")
    md.append("")
    md.append("| requested metric | status | proxy | note |")
    md.append("|---|---|---|---|")
    for metric, rec in REQUESTED_TRACKA.items():
        md.append(f"| `{metric}` | `{rec['status']}` | `{rec['proxy']}` | {rec['note']} |")
    md.append("")
    md.append("## Seed42 Deployable/Diagnostic Summary")
    md.append("")
    md.append(
        md_table(
            primary_seed42,
            [
                "source",
                "group",
                "role",
                "n_conds",
                "reported_pearson_pert",
                "pp_condition_boot_ci_low",
                "pp_condition_boot_ci_high",
                "reported_test_mmd_clamped",
            ],
            max_rows=30,
        )
    )
    md.append("")
    md.append("## Seed Replicate Check")
    md.append("")
    md.append(
        md_table(
            replicate,
            [
                "source",
                "group",
                "role",
                "seed42_reported_pearson_pert",
                "seed43_reported_pearson_pert",
                "delta_seed43_minus_seed42_pearson_pert",
                "delta_seed43_minus_seed42_mmd_clamped",
            ],
            max_rows=40,
        )
    )
    md.append("")
    md.append("## Worst Seed42 Failure Cases")
    md.append("")
    md.append(
        md_table(
            worst_primary,
            [
                "source",
                "group",
                "dataset",
                "condition",
                "pearson_pert",
                "test_mmd_clamped",
                "failure_tags",
            ],
            max_rows=20,
        )
    )
    md.append("")
    md.append("## Outputs")
    md.append("")
    for label, path in result["outputs"].items():
        md.append(f"- {label}: `{path}`")
    md.append("")
    md.append("## Decision")
    md.append("")
    md.append(
        "- This report fills the deployable benchmark/failure-taxonomy gap but does not by itself authorize GPU."
    )
    md.append(
        "- Next valid GPU branch must target a failure cluster from this report and first pass a train-only/query-blind CPU gate."
    )
    md.append(
        "- Missing requested Track A metrics should be generated by a matching evaluator if needed; do not substitute canonical multi or Track C query."
    )
    MD_OUT.write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
