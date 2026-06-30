#!/usr/bin/env python3
"""Draft strict matched high/low exact-coverage split designs.

CPU/report-only. The generated JSONs are draft artifacts for audit, not
launch-ready training splits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
COVERAGE_CSV = ROOT / "runs/latentfm_exact_response_information_coverage_20260628/latentfm_exact_response_information_coverage_20260628_20260628_144814/outputs/exact_response_information_condition_rows.csv"
OUT_DIR = ROOT / "reports/exact_coverage_strict_matched_draft_splits_20260628"
OUT_MD = OUT_DIR / "LATENTFM_EXACT_COVERAGE_STRICT_MATCHED_DRAFT_SPLITS_20260628.md"
OUT_JSON = OUT_DIR / "latentfm_exact_coverage_strict_matched_draft_splits_20260628.json"
OUT_PAIR_CSV = OUT_DIR / "matched_condition_pairs.csv"
OUT_DATASET_CSV = OUT_DIR / "dataset_match_summary.csv"
HIGH_SPLIT = OUT_DIR / "draft_split_seed42_xverse_exactcov_high_matched_from_cap120_all_v2.json"
LOW_SPLIT = OUT_DIR / "draft_split_seed42_xverse_exactcov_low_matched_from_cap120_all_v2.json"
SEED = 42
MAX_PAIRS_PER_DATASET = 96


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def condition_type(meta: dict[str, Any]) -> str:
    return str(meta.get("perturbation_type_raw", "unknown") or "unknown")


def gene_count_bin(meta: dict[str, Any]) -> str:
    genes = meta.get("genes", [])
    if not isinstance(genes, list):
        return "unknown"
    if len(genes) <= 1:
        return "single"
    if len(genes) == 2:
        return "double"
    return "multi3plus"


def main() -> None:
    global COVERAGE_CSV, OUT_DIR, OUT_MD, OUT_JSON, OUT_PAIR_CSV, OUT_DATASET_CSV, HIGH_SPLIT, LOW_SPLIT

    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-csv", type=Path, default=COVERAGE_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    COVERAGE_CSV = args.coverage_csv
    OUT_DIR = args.out_dir
    OUT_MD = OUT_DIR / "LATENTFM_EXACT_COVERAGE_STRICT_MATCHED_DRAFT_SPLITS_20260628.md"
    OUT_JSON = OUT_DIR / "latentfm_exact_coverage_strict_matched_draft_splits_20260628.json"
    OUT_PAIR_CSV = OUT_DIR / "matched_condition_pairs.csv"
    OUT_DATASET_CSV = OUT_DIR / "dataset_match_summary.csv"
    HIGH_SPLIT = OUT_DIR / "draft_split_seed42_xverse_exactcov_high_matched_from_cap120_all_v2.json"
    LOW_SPLIT = OUT_DIR / "draft_split_seed42_xverse_exactcov_low_matched_from_cap120_all_v2.json"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    parent = load_json(PARENT_SPLIT)
    meta = load_json(COND_META)
    covered = {
        (str(row.dataset), str(row.condition))
        for row in pd.read_csv(COVERAGE_CSV).itertuples(index=False)
    }
    high_split = json.loads(json.dumps(parent))
    low_split = json.loads(json.dumps(parent))
    pair_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    for dataset, groups in parent.items():
        strata: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: {"covered": [], "uncovered": []})
        for condition in groups.get("train", []):
            cond_meta = meta.get(dataset, {}).get(str(condition), {})
            key = (condition_type(cond_meta), gene_count_bin(cond_meta))
            bucket = "covered" if (dataset, str(condition)) in covered else "uncovered"
            strata[key][bucket].append(str(condition))
        selected_high: list[str] = []
        selected_low: list[str] = []
        for key, buckets in sorted(strata.items()):
            cov = sorted(buckets["covered"])
            uncov = sorted(buckets["uncovered"])
            rng.shuffle(cov)
            rng.shuffle(uncov)
            n = min(len(cov), len(uncov), MAX_PAIRS_PER_DATASET - len(selected_high))
            if n <= 0:
                continue
            for hi, lo in zip(cov[:n], uncov[:n]):
                selected_high.append(hi)
                selected_low.append(lo)
                pair_rows.append(
                    {
                        "dataset": dataset,
                        "perturbation_type": key[0],
                        "gene_count_bin": key[1],
                        "high_covered_condition": hi,
                        "low_uncovered_condition": lo,
                    }
                )
            if len(selected_high) >= MAX_PAIRS_PER_DATASET:
                break
        high_split[dataset]["train"] = sorted(selected_high)
        low_split[dataset]["train"] = sorted(selected_low)
        dataset_rows.append(
            {
                "dataset": dataset,
                "parent_train": len(groups.get("train", [])),
                "matched_pairs": len(selected_high),
                "high_train": len(selected_high),
                "low_train": len(selected_low),
                "has_matched_pairs": bool(selected_high),
            }
        )

    write_json(HIGH_SPLIT, high_split)
    write_json(LOW_SPLIT, low_split)
    pd.DataFrame(pair_rows).to_csv(OUT_PAIR_CSV, index=False)
    pd.DataFrame(dataset_rows).to_csv(OUT_DATASET_CSV, index=False)
    total_pairs = len(pair_rows)
    datasets_with_pairs = sum(1 for row in dataset_rows if row["matched_pairs"] > 0)
    status = "exact_coverage_strict_matched_draft_partial_no_gpu"
    if total_pairs >= 300 and datasets_with_pairs >= 8:
        status = "exact_coverage_strict_matched_draft_feasible_no_gpu"
    payload = {
        "created_at": now_cst(),
        "status": status,
        "parent_split": str(PARENT_SPLIT),
        "high_split": str(HIGH_SPLIT),
        "low_split": str(LOW_SPLIT),
        "matched_pairs": total_pairs,
        "datasets_with_pairs": datasets_with_pairs,
        "max_pairs_per_dataset": MAX_PAIRS_PER_DATASET,
        "seed": SEED,
        "boundary": "draft_no_gpu_not_launch_ready",
    }
    write_json(OUT_JSON, payload)

    top_dataset_rows = sorted(dataset_rows, key=lambda r: r["matched_pairs"], reverse=True)[:12]
    lines = [
        "# LatentFM Exact-Coverage Strict Matched Draft Splits",
        "",
        f"Created: {payload['created_at']}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU/report-only draft split construction from parent train conditions.",
        "* High split uses exact-covered train conditions; low split uses uncovered train conditions matched within dataset, perturbation type, and gene-count class.",
        "* Test/internal-val/canonical reference groups are copied from the parent split, but these drafts are not launch-ready and do not authorize GPU.",
        "* No train/infer/GPU/canonical multi/Track C query/checkpoint selection.",
        "",
        "## Summary",
        "",
        f"* Parent split: `{PARENT_SPLIT}`",
        f"* Matched condition pairs: `{total_pairs}`.",
        f"* Datasets with matched pairs: `{datasets_with_pairs}`.",
        f"* Max pairs per dataset: `{MAX_PAIRS_PER_DATASET}`.",
        "",
        "## Dataset Match Summary",
        "",
        "| dataset | parent train | matched pairs |",
        "|---|---:|---:|",
    ]
    for row in top_dataset_rows:
        lines.append(f"| {row['dataset']} | {row['parent_train']} | {row['matched_pairs']} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"* Status: `{status}`.",
            "* These drafts are a feasibility artifact only. They need independent leakage audit, provenance review, balanced validation construction, launcher design, dual baseline, and no-harm gates before any GPU.",
            "",
            "## Outputs",
            "",
            f"* High draft split: `{HIGH_SPLIT}`",
            f"* Low draft split: `{LOW_SPLIT}`",
            f"* Matched pairs: `{OUT_PAIR_CSV}`",
            f"* Dataset summary: `{OUT_DATASET_CSV}`",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"status {status}")


if __name__ == "__main__":
    main()
