#!/usr/bin/env python3
"""Build an exact CPU-only Track A simple-single-unseen evaluator.

This report does not train or infer. It reuses existing frozen xverse 8k
uncapped posthoc condition rows and canonical split_seed42.json to define:

* simple_single_unseen: canonical test_single, exactly one non-drug gene target,
  target absent from all canonical train single-gene rows.
* cross_background_seen_gene_exact: canonical test_single, exactly one non-drug
  gene target, target absent from this dataset's train split but present in at
  least one other dataset/background train split.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
SEED42_FAMILY = ROOT / (
    "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/"
    "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
SEED43_FAMILY = ROOT / (
    "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
    "xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/"
    "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)

OUT_DIR = ROOT / "reports/tracka_simple_single_unseen_exact_20260627"
OUT_ROWS = OUT_DIR / "condition_rows.csv"
OUT_DATASETS = OUT_DIR / "dataset_summary.csv"
OUT_WORST = OUT_DIR / "worst_conditions.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_simple_single_unseen_exact_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_SIMPLE_SINGLE_UNSEEN_EXACT_20260627.md"

GROUPS = (
    "test_single_gene_exact",
    "simple_single_unseen",
    "cross_background_seen_gene_exact",
    "local_seen_gene",
)
METRICS = ("pearson_pert", "test_mmd_clamped")
BOOTSTRAP_SEED_OFFSET = {"pearson_pert": 101, "test_mmd_clamped": 202}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def pert_type(entry: dict[str, Any]) -> str:
    raw = clean(entry.get("perturbation_type_raw", entry.get("perturbation_type")))
    low = raw.lower()
    if low in {"crispri", "knockdown", "kd"}:
        return "CRISPRi"
    if low in {"crispra", "activation", "overexpression"}:
        return "CRISPRa"
    if low in {"crisprko", "ko", "knockout"}:
        return "CRISPRko"
    if low == "cas13":
        return "Cas13"
    if low in {"drug", "chemical", "compound", "small molecule", "small-molecule"}:
        return "drug"
    return raw or "unknown"


def genes(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("genes")
    if not isinstance(raw, list):
        return []
    return [str(g).strip() for g in raw if str(g).strip()]


def is_drug(entry: dict[str, Any], ds: str) -> bool:
    if pert_type(entry).lower() == "drug":
        return True
    dsl = ds.lower()
    return any(tok in dsl for tok in ("sciplex", "drug", "chemical", "chempert"))


def single_gene(entry: dict[str, Any], ds: str) -> str | None:
    gs = genes(entry)
    if len(gs) != 1 or is_drug(entry, ds):
        return None
    return gs[0]


def build_train_gene_maps(
    split: dict[str, dict[str, list[str]]],
    meta: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    train_by_ds: dict[str, set[str]] = defaultdict(set)
    datasets_by_gene: dict[str, set[str]] = defaultdict(set)
    for ds, groups in split.items():
        ds_meta = meta.get(ds, {})
        for cond in groups.get("train", []):
            gene = single_gene(ds_meta.get(str(cond), {}), ds)
            if not gene:
                continue
            train_by_ds[ds].add(gene)
            datasets_by_gene[gene].add(ds)
    return train_by_ds, datasets_by_gene


def load_condition_metrics(path: Path, seed: str) -> dict[tuple[str, str], dict[str, Any]]:
    payload = load_json(path)
    rows = payload["groups"]["test_single"].get("condition_metrics", [])
    out = {}
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("condition")))
        rec = dict(row)
        rec["seed"] = seed
        out[key] = rec
    return out


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def bootstrap_ci(values: list[float], *, seed: int, n_boot: int = 5000) -> list[float | None]:
    if not values:
        return [None, None]
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return [float(lo), float(hi)]


def summarize(rows: list[dict[str, Any]], *, seed_int: int) -> dict[str, Any]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
        ds_means = {
            ds: mean([float(r[metric]) for r in ds_rows if r.get(metric) is not None])
            for ds, ds_rows in by_ds.items()
        }
        ds_vals = [v for v in ds_means.values() if v is not None]
        metrics[metric] = {
            "mean": mean(vals),
            "condition_bootstrap_ci95": bootstrap_ci(vals, seed=seed_int + BOOTSTRAP_SEED_OFFSET[metric]),
            "dataset_mean_min": float(min(ds_vals)) if ds_vals else None,
            "dataset_mean_max": float(max(ds_vals)) if ds_vals else None,
        }
    return {
        "n_conditions": len(rows),
        "n_datasets": len(by_ds),
        "datasets": sorted(by_ds),
        "metrics": metrics,
    }


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = load_json(SPLIT)
    meta = load_json(META)
    train_by_ds, datasets_by_gene = build_train_gene_maps(split, meta)
    global_train_genes = set(datasets_by_gene)
    seed_metrics = {
        "seed42": load_condition_metrics(SEED42_FAMILY, "seed42"),
        "seed43": load_condition_metrics(SEED43_FAMILY, "seed43"),
    }

    all_rows: list[dict[str, Any]] = []
    skipped = defaultdict(int)
    for ds, groups in split.items():
        ds_meta = meta.get(ds, {})
        for cond in groups.get("test_single", []):
            gene = single_gene(ds_meta.get(str(cond), {}), ds)
            if not gene:
                skipped["non_gene_or_not_exactly_one_gene"] += 1
                continue
            local_seen = gene in train_by_ds.get(ds, set())
            global_seen = gene in global_train_genes
            other_seen = bool(datasets_by_gene.get(gene, set()) - {ds})
            group_labels = ["test_single_gene_exact"]
            if not global_seen:
                group_labels.append("simple_single_unseen")
            if not local_seen and other_seen:
                group_labels.append("cross_background_seen_gene_exact")
            if local_seen:
                group_labels.append("local_seen_gene")
            for seed, metrics_by_key in seed_metrics.items():
                metric_row = metrics_by_key.get((ds, str(cond)))
                if not metric_row:
                    skipped[f"missing_metric_{seed}"] += 1
                    continue
                for group in group_labels:
                    all_rows.append(
                        {
                            "seed": seed,
                            "group": group,
                            "dataset": ds,
                            "condition": str(cond),
                            "gene": gene,
                            "perturbation_type": pert_type(ds_meta.get(str(cond), {})),
                            "local_train_seen": local_seen,
                            "global_train_seen": global_seen,
                            "other_train_background_seen": other_seen,
                            "n_train_datasets_for_gene": len(datasets_by_gene.get(gene, set())),
                            "pearson_pert": metric_row.get("pearson_pert"),
                            "test_mmd_clamped": metric_row.get("test_mmd_clamped"),
                            "n_src_eval": metric_row.get("n_src_eval"),
                            "n_gt_eval": metric_row.get("n_gt_eval"),
                        }
                    )

    fields = [
        "seed",
        "group",
        "dataset",
        "condition",
        "gene",
        "perturbation_type",
        "local_train_seen",
        "global_train_seen",
        "other_train_background_seen",
        "n_train_datasets_for_gene",
        "pearson_pert",
        "test_mmd_clamped",
        "n_src_eval",
        "n_gt_eval",
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summaries: dict[str, dict[str, Any]] = {}
    dataset_rows: list[dict[str, Any]] = []
    for seed in seed_metrics:
        summaries[seed] = {}
        for group in GROUPS:
            rows = [r for r in all_rows if r["seed"] == seed and r["group"] == group]
            summaries[seed][group] = summarize(rows, seed_int=42 if seed == "seed42" else 43)
            by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_ds[row["dataset"]].append(row)
            for ds, ds_rows in sorted(by_ds.items()):
                dataset_rows.append(
                    {
                        "seed": seed,
                        "group": group,
                        "dataset": ds,
                        "n_conditions": len(ds_rows),
                        "pearson_pert_mean": mean([float(r["pearson_pert"]) for r in ds_rows]),
                        "test_mmd_clamped_mean": mean([float(r["test_mmd_clamped"]) for r in ds_rows]),
                    }
                )

    with OUT_DATASETS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "group", "dataset", "n_conditions", "pearson_pert_mean", "test_mmd_clamped_mean"],
        )
        writer.writeheader()
        writer.writerows(dataset_rows)

    worst_rows: list[dict[str, Any]] = []
    for seed in seed_metrics:
        for group in GROUPS:
            rows = [r for r in all_rows if r["seed"] == seed and r["group"] == group]
            rows = sorted(rows, key=lambda r: float(r["pearson_pert"]))
            for rank, row in enumerate(rows[:20], start=1):
                rec = {k: row[k] for k in fields}
                rec["worst_rank_by_pearson_pert"] = rank
                worst_rows.append(rec)
    with OUT_WORST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["worst_rank_by_pearson_pert"] + fields)
        writer.writeheader()
        writer.writerows(worst_rows)

    stability: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        s42 = summaries["seed42"][group]
        s43 = summaries["seed43"][group]
        stability[group] = {
            "n_conditions_seed42": s42["n_conditions"],
            "n_conditions_seed43": s43["n_conditions"],
            "pearson_pert_seed43_minus_seed42": (
                s43["metrics"]["pearson_pert"]["mean"] - s42["metrics"]["pearson_pert"]["mean"]
                if s42["metrics"]["pearson_pert"]["mean"] is not None
                and s43["metrics"]["pearson_pert"]["mean"] is not None
                else None
            ),
            "mmd_seed43_minus_seed42": (
                s43["metrics"]["test_mmd_clamped"]["mean"] - s42["metrics"]["test_mmd_clamped"]["mean"]
                if s42["metrics"]["test_mmd_clamped"]["mean"] is not None
                and s43["metrics"]["test_mmd_clamped"]["mean"] is not None
                else None
            ),
        }

    payload = {
        "status": "tracka_simple_single_unseen_exact_ready_no_gpu",
        "gpu_authorized": False,
        "default_model": "xverse_8k_anchor",
        "boundary": {
            "cpu_report_only": True,
            "no_training": True,
            "no_inference": True,
            "canonical_split_not_recut": str(SPLIT),
            "canonical_multi_selection_weight": 0,
            "trackc_query_read": False,
        },
        "definitions": {
            "simple_single_unseen": "canonical test_single, exactly one non-drug gene target, gene absent from all canonical train single-gene rows",
            "cross_background_seen_gene_exact": "canonical test_single, exactly one non-drug gene target, gene absent from same dataset train but present in at least one other dataset train",
            "local_seen_gene": "canonical test_single, exactly one non-drug gene target, gene present in same dataset train",
        },
        "inputs": {
            "split": str(SPLIT),
            "condition_metadata": str(META),
            "seed42_family_eval": str(SEED42_FAMILY),
            "seed43_family_eval": str(SEED43_FAMILY),
        },
        "skipped": dict(skipped),
        "summaries": summaries,
        "seed_replicate_stability": stability,
        "outputs": {
            "rows": str(OUT_ROWS),
            "dataset_summary": str(OUT_DATASETS),
            "worst_conditions": str(OUT_WORST),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track A Simple Single Unseen Exact Evaluator",
        "",
        "Status: `tracka_simple_single_unseen_exact_ready_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "Default/deployable model: `xverse_8k_anchor`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over existing frozen uncapped xverse 8k posthoc rows.",
        "- Uses canonical `split_seed42.json`; does not recut splits.",
        "- Does not train, infer, select checkpoints, read Track C query, or use canonical multi for selection.",
        "",
        "## Definitions",
        "",
        "- `simple_single_unseen`: canonical `test_single`, exactly one non-drug gene target, gene absent from all canonical train single-gene rows.",
        "- `cross_background_seen_gene_exact`: canonical `test_single`, exactly one non-drug gene target, gene absent from same dataset train but present in at least one other dataset train.",
        "- `local_seen_gene`: canonical `test_single`, exactly one non-drug gene target, gene present in same dataset train.",
        "",
        "## Summary",
        "",
        "| seed | group | n cond | n datasets | pp mean | pp CI95 | dataset min pp | MMD mean | dataset max MMD |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for seed in ("seed42", "seed43"):
        for group in GROUPS:
            s = summaries[seed][group]
            pp = s["metrics"]["pearson_pert"]
            mmd = s["metrics"]["test_mmd_clamped"]
            ci = pp["condition_bootstrap_ci95"]
            lines.append(
                f"| `{seed}` | `{group}` | {s['n_conditions']} | {s['n_datasets']} | "
                f"{fmt(pp['mean'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(pp['dataset_mean_min'])} | "
                f"{fmt(mmd['mean'])} | {fmt(mmd['dataset_mean_max'])} |"
            )
    lines += [
        "",
        "## Seed Replicate Stability",
        "",
        "| group | pp seed43-seed42 | MMD seed43-seed42 |",
        "|---|---:|---:|",
    ]
    for group, row in stability.items():
        lines.append(
            f"| `{group}` | {fmt(row['pearson_pert_seed43_minus_seed42'])} | {fmt(row['mmd_seed43_minus_seed42'])} |"
        )
    lines += [
        "",
        "## Worst Conditions",
        "",
        "| seed | group | rank | dataset | condition | gene | pp | MMD |",
        "|---|---|---:|---|---|---|---:|---:|",
    ]
    for row in worst_rows:
        if row["worst_rank_by_pearson_pert"] > 5:
            continue
        if row["group"] not in {"simple_single_unseen", "cross_background_seen_gene_exact"}:
            continue
        lines.append(
            f"| `{row['seed']}` | `{row['group']}` | {row['worst_rank_by_pearson_pert']} | "
            f"`{row['dataset']}` | `{row['condition']}` | `{row['gene']}` | "
            f"{fmt(row['pearson_pert'])} | {fmt(row['test_mmd_clamped'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "- This fills the exact `simple_single_unseen` evaluator/provenance gap for the current default model.",
        "- It is descriptive benchmark evidence, not paired candidate-improvement evidence.",
        "- No GPU is authorized by this evaluator.",
        "",
        "## Outputs",
        "",
        f"- Rows: `{OUT_ROWS}`",
        f"- Dataset summary: `{OUT_DATASETS}`",
        f"- Worst conditions: `{OUT_WORST}`",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
