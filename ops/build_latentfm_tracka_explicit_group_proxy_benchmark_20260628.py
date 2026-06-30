#!/usr/bin/env python3
"""Build explicit Track A proxy groups from frozen xverse_8k eval rows.

CPU/report-only. This script only relabels already-completed condition metrics
from the frozen xverse_8k seed42 anchor and seed43 replicate. It does not train,
infer, select checkpoints, read canonical multi for selection, read Track C
query, or use GPU.
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
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "tracka_explicit_group_proxy_benchmark_20260628"
OUT_MD = REPORTS / "LATENTFM_TRACKA_EXPLICIT_GROUP_PROXY_BENCHMARK_20260628.md"
OUT_JSON = REPORTS / "latentfm_tracka_explicit_group_proxy_benchmark_20260628.json"

SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
GENE_META = ROOT / "dataset/raw/genepert_bench/metainfo.json"

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
    "seed42_split": ANCHOR_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed42_family": ANCHOR_ROOT
    / "posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43_split": REPLICATE_ROOT
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
    "seed43_family": REPLICATE_ROOT
    / "posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**32)


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def bootstrap_mean(values: list[float], *, label: str, n_boot: int = 2000) -> dict[str, Any]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(stable_seed(label))
    draws = []
    n = len(vals)
    for _ in range(n_boot):
        draws.append(sum(rng.choice(vals) for _ in range(n)) / n)
    draws.sort()
    return {
        "n": n,
        "mean": mean(vals),
        "ci_low": quantile(draws, 0.025),
        "ci_high": quantile(draws, 0.975),
    }


def clean_cell(value: str) -> str:
    return str(value or "").strip()


def cell_background_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for row in load_json(GENE_META):
        ds = str(row.get("dataset", "")).strip()
        if ds:
            out[ds] = clean_cell(row.get("cell_line", ""))
    for ds in ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7"):
        out[ds] = ds.rsplit("_", 1)[-1]
    return out


def genes_for(cond_meta: dict[str, Any], ds: str, cond: str) -> list[str]:
    entry = ((cond_meta.get(ds) or {}).get(cond) or {})
    genes = entry.get("genes")
    if isinstance(genes, list):
        return [str(g).strip() for g in genes if str(g).strip()]
    return []


def train_gene_backgrounds(
    split: dict[str, dict[str, list[str]]],
    cond_meta: dict[str, Any],
    backgrounds: dict[str, str],
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for ds, groups in split.items():
        bg = backgrounds.get(ds, "")
        for cond in groups.get("train", []) or []:
            genes = genes_for(cond_meta, ds, str(cond))
            if len(genes) == 1:
                out[genes[0]].add(bg)
    return out


def condition_row_index(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics") or []
    return {(str(r.get("dataset")), str(r.get("condition"))): r for r in rows}


def build_group_rows(seed: str, split_payload: dict[str, Any], family_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    split = load_json(SPLIT)
    cond_meta = load_json(COND_META)
    backgrounds = cell_background_map()
    train_gene_bgs = train_gene_backgrounds(split, cond_meta, backgrounds)

    test_single_rows = condition_row_index(split_payload, "test_single")
    family_gene_rows = condition_row_index(family_payload, "family_gene")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["all_test_single_proxy"].extend(test_single_rows.values())
    groups["family_gene"].extend(family_gene_rows.values())

    for (ds, cond), row in test_single_rows.items():
        genes = genes_for(cond_meta, ds, cond)
        if len(genes) != 1:
            continue
        gene = genes[0]
        this_bg = backgrounds.get(ds, "")
        seen_bgs = train_gene_bgs.get(gene, set())
        if not seen_bgs:
            groups["simple_single_unseen_global_gene_proxy"].append(row)
        elif any(bg and bg != this_bg for bg in seen_bgs):
            groups["cross_background_seen_gene_proxy"].append(row)

    # Attach provenance fields without mutating source objects.
    out: dict[str, list[dict[str, Any]]] = {}
    for group, rows in groups.items():
        clean_rows = []
        for row in rows:
            ds = str(row.get("dataset", ""))
            cond = str(row.get("condition", ""))
            genes = genes_for(cond_meta, ds, cond)
            gene = genes[0] if len(genes) == 1 else ""
            rec = dict(row)
            rec.update(
                {
                    "seed": seed,
                    "explicit_group": group,
                    "cell_background": backgrounds.get(ds, ""),
                    "single_gene": gene,
                    "train_backgrounds_for_gene": ";".join(sorted(train_gene_bgs.get(gene, set()))) if gene else "",
                }
            )
            clean_rows.append(rec)
        out[group] = clean_rows
    return out


def summarize(seed: str, group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pp = [finite(r.get("pearson_pert")) for r in rows]
    pp = [v for v in pp if v is not None]
    mmd = [finite(r.get("test_mmd_clamped")) for r in rows]
    mmd = [v for v in mmd if v is not None]
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = finite(row.get("pearson_pert"))
        if val is not None:
            by_ds[str(row.get("dataset"))].append(val)
    ds_means = {ds: mean(vals) for ds, vals in by_ds.items() if vals}
    return {
        "seed": seed,
        "group": group,
        "n": len(rows),
        "datasets": len(by_ds),
        "mean_pearson_pert": mean(pp) if pp else None,
        "pp_boot_ci_low": bootstrap_mean(pp, label=f"{seed}:{group}:pp")["ci_low"],
        "pp_boot_ci_high": bootstrap_mean(pp, label=f"{seed}:{group}:pp")["ci_high"],
        "dataset_min_pearson_pert": min(ds_means.values()) if ds_means else None,
        "negative_dataset_tails": sum(1 for v in ds_means.values() if v < 0),
        "mean_mmd_clamped": mean(mmd) if mmd else None,
        "max_mmd_clamped": max(mmd) if mmd else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    payloads = {key: load_json(path) for key, path in INPUTS.items()}
    rows_all: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    group_payload: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for seed in ("seed42", "seed43"):
        groups = build_group_rows(seed, payloads[f"{seed}_split"], payloads[f"{seed}_family"])
        group_payload[seed] = groups
        for group, rows in sorted(groups.items()):
            rows_all.extend(rows)
            summaries.append(summarize(seed, group, rows))

    replicate_rows = []
    by_key = {(r["seed"], r["group"]): r for r in summaries}
    for group in sorted({r["group"] for r in summaries}):
        s42 = by_key.get(("seed42", group), {})
        s43 = by_key.get(("seed43", group), {})
        replicate_rows.append(
            {
                "group": group,
                "seed42_mean_pearson_pert": s42.get("mean_pearson_pert"),
                "seed43_mean_pearson_pert": s43.get("mean_pearson_pert"),
                "seed43_minus_seed42_pp": None
                if s42.get("mean_pearson_pert") is None or s43.get("mean_pearson_pert") is None
                else float(s43["mean_pearson_pert"]) - float(s42["mean_pearson_pert"]),
                "seed42_dataset_min": s42.get("dataset_min_pearson_pert"),
                "seed43_dataset_min": s43.get("dataset_min_pearson_pert"),
            }
        )

    write_csv(
        OUT_DIR / "group_summary.csv",
        summaries,
        [
            "seed",
            "group",
            "n",
            "datasets",
            "mean_pearson_pert",
            "pp_boot_ci_low",
            "pp_boot_ci_high",
            "dataset_min_pearson_pert",
            "negative_dataset_tails",
            "mean_mmd_clamped",
            "max_mmd_clamped",
        ],
    )
    write_csv(
        OUT_DIR / "condition_rows.csv",
        rows_all,
        [
            "seed",
            "explicit_group",
            "dataset",
            "condition",
            "cell_background",
            "single_gene",
            "train_backgrounds_for_gene",
            "pearson_pert",
            "pearson_ctrl",
            "direct_pearson",
            "test_mmd_clamped",
            "n_src_eval",
            "n_gt_eval",
        ],
    )
    write_csv(
        OUT_DIR / "seed_replicate_summary.csv",
        replicate_rows,
        [
            "group",
            "seed42_mean_pearson_pert",
            "seed43_mean_pearson_pert",
            "seed43_minus_seed42_pp",
            "seed42_dataset_min",
            "seed43_dataset_min",
        ],
    )

    status = "tracka_explicit_group_proxy_benchmark_ready_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "uses_existing_frozen_eval_rows": True,
            "training_or_inference": False,
            "canonical_multi_selection": False,
            "trackc_query": False,
            "gpu": False,
        },
        "group_definitions": {
            "all_test_single_proxy": "canonical split_seed42 test_single rows from frozen split-group eval",
            "family_gene": "condition-family eval rows with non-drug gene perturbations",
            "simple_single_unseen_global_gene_proxy": "test_single rows whose single gene is absent from every canonical train split gene condition",
            "cross_background_seen_gene_proxy": "test_single rows whose single gene appears in train in at least one different metainfo cell_line/background string",
        },
        "limitations": [
            "cell_background is dataset-level metainfo, not per-cell background for mixed Jiang/Wessels datasets",
            "explicit proxy groups are relabels of frozen condition metrics, not new inference",
            "cross_background_seen_gene_proxy is conservative and may miss genes present only in mixed-background datasets with ambiguous background strings",
        ],
        "summaries": summaries,
        "replicate_summary": replicate_rows,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "group_summary": str(OUT_DIR / "group_summary.csv"),
            "condition_rows": str(OUT_DIR / "condition_rows.csv"),
            "seed_replicate_summary": str(OUT_DIR / "seed_replicate_summary.csv"),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track A Explicit Group Proxy Benchmark",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only relabeling of frozen xverse_8k seed42 and seed43 condition metrics.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "- Canonical multi remains diagnostic-only and is not used here.",
        "",
        "## Group Definitions",
        "",
        "- `all_test_single_proxy`: canonical `split_seed42.json` `test_single` rows.",
        "- `simple_single_unseen_global_gene_proxy`: `test_single` rows whose single gene is absent from all canonical train gene conditions.",
        "- `cross_background_seen_gene_proxy`: `test_single` rows whose single gene appears in train in at least one different dataset-level cell-line/background string.",
        "- `family_gene`: frozen condition-family gene rows.",
        "",
        "## Summary",
        "",
        "| seed | group | n | datasets | mean pp | pp CI95 | dataset min pp | neg dataset tails | mean MMD | max MMD |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| `{seed}` | `{group}` | {n} | {datasets} | {mean_pp} | [{ci_low}, {ci_high}] | {ds_min} | {neg} | {mmd_mean} | {mmd_max} |".format(
                seed=row["seed"],
                group=row["group"],
                n=row["n"],
                datasets=row["datasets"],
                mean_pp=fmt(row["mean_pearson_pert"]),
                ci_low=fmt(row["pp_boot_ci_low"]),
                ci_high=fmt(row["pp_boot_ci_high"]),
                ds_min=fmt(row["dataset_min_pearson_pert"]),
                neg=row["negative_dataset_tails"],
                mmd_mean=fmt(row["mean_mmd_clamped"]),
                mmd_max=fmt(row["max_mmd_clamped"]),
            )
        )
    lines.extend(
        [
            "",
            "## Seed Replicate Check",
            "",
            "| group | seed42 pp | seed43 pp | seed43-seed42 | seed42 dataset min | seed43 dataset min |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in replicate_rows:
        lines.append(
            "| `{group}` | {s42} | {s43} | {delta} | {min42} | {min43} |".format(
                group=row["group"],
                s42=fmt(row["seed42_mean_pearson_pert"]),
                s43=fmt(row["seed43_mean_pearson_pert"]),
                delta=fmt(row["seed43_minus_seed42_pp"]),
                min42=fmt(row["seed42_dataset_min"]),
                min43=fmt(row["seed43_dataset_min"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- This fills a Track A benchmark-label gap but does not authorize GPU by itself.",
            "- The explicit groups remain proxy labels because cell background is dataset-level metainfo.",
            "- A future GPU branch still needs a train-only/query-blind CPU gate with paired candidate-vs-anchor deltas and no-harm controls.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- group summary: `{OUT_DIR / 'group_summary.csv'}`",
            f"- condition rows: `{OUT_DIR / 'condition_rows.csv'}`",
            f"- seed replicate summary: `{OUT_DIR / 'seed_replicate_summary.csv'}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
