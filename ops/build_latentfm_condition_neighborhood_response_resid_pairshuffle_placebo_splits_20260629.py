#!/usr/bin/env python3
"""Build pair-shuffle placebo splits for response-residualized support.

CPU/report-only. The placebo preserves the matched condition pairs and all
non-train buckets, then randomly flips high/low labels within each pair. This
destroys the direction of the response-residualized support score while keeping
the paired condition universe fixed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
PAIR_CSV = ROOT / "reports/condition_neighborhood_response_residualized_support_gate_20260629/condition_neighborhood_response_residualized_selected_pairs.csv"
PACKET_JSON = ROOT / "reports/condition_neighborhood_response_residualized_support_gate_20260629/latentfm_condition_neighborhood_response_residualized_support_gate_20260629.json"
OUT_DIR = ROOT / "reports/condition_neighborhood_response_resid_pairshuffle_placebo_splits_20260629"
SPLIT_DIR = ROOT / "dataset/biFlow_data/xverse_condition_neighborhood_support_splits_20260629"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_items(split: dict[str, Any], group: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for dataset, groups in split.items():
        for condition in (groups or {}).get(group, []) or []:
            out.add((str(dataset), str(condition)))
    return out


def nontrain_groups(split: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for groups in split.values():
        for group in groups:
            if str(group) != "train":
                out.add(str(group))
    return out


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {
        "high_dataset",
        "low_dataset",
        "high_condition",
        "low_condition",
        "perturbation_type_raw",
        "high_support_resid_score",
        "low_support_resid_score",
        "residual_score_gap",
    }
    missing = sorted(required - set(rows[0].keys() if rows else []))
    if missing:
        raise ValueError(f"pair CSV missing columns: {missing}")
    return rows


def write_split(parent: dict[str, Any], rows: list[dict[str, Any]], side: str, path: Path) -> None:
    out = copy.deepcopy(parent)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row[f"placebo_{side}_dataset"])].append(str(row[f"placebo_{side}_condition"]))
    for dataset in out:
        out[dataset]["train"] = sorted(set(by_dataset.get(str(dataset), [])))
    write_json(path, out)


def compare_nontrain(parent: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for dataset, groups in parent.items():
        cand = candidate.get(dataset, {})
        for group, values in groups.items():
            if str(group) == "train":
                continue
            if list(values or []) != list(cand.get(group, []) or []):
                reasons.append(f"nontrain changed: {dataset}/{group}")
    return reasons


def validate_split(parent: dict[str, Any], high: dict[str, Any], low: dict[str, Any], n_pairs: int) -> list[str]:
    reasons: list[str] = []
    parent_train = split_items(parent, "train")
    high_train = split_items(high, "train")
    low_train = split_items(low, "train")
    if len(high_train) != n_pairs:
        reasons.append(f"high train count {len(high_train)} != pairs {n_pairs}")
    if len(low_train) != n_pairs:
        reasons.append(f"low train count {len(low_train)} != pairs {n_pairs}")
    if high_train & low_train:
        reasons.append(f"high/low overlap {len(high_train & low_train)}")
    if not high_train <= parent_train:
        reasons.append(f"high outside parent train {len(high_train - parent_train)}")
    if not low_train <= parent_train:
        reasons.append(f"low outside parent train {len(low_train - parent_train)}")
    for group in sorted(nontrain_groups(parent)):
        overlap_high = high_train & split_items(high, group)
        overlap_low = low_train & split_items(low, group)
        if overlap_high:
            reasons.append(f"high train overlaps {group}: {len(overlap_high)}")
        if overlap_low:
            reasons.append(f"low train overlaps {group}: {len(overlap_low)}")
    reasons.extend(compare_nontrain(parent, high))
    reasons.extend(compare_nontrain(parent, low))
    return reasons


def build_seed_rows(pairs: list[dict[str, str]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    true_high_in_placebo_high = 0
    placebo_gaps: list[float] = []
    original_gaps: list[float] = []
    high_dataset_counts: Counter[str] = Counter()
    low_dataset_counts: Counter[str] = Counter()
    ptype_counts: Counter[str] = Counter()
    for pair_id, row in enumerate(pairs):
        keep = rng.random() < 0.5
        hi_score = float(row["high_support_resid_score"])
        lo_score = float(row["low_support_resid_score"])
        original_gaps.append(float(row["residual_score_gap"]))
        ptype_counts[str(row["perturbation_type_raw"])] += 1
        if keep:
            ph_dataset = row["high_dataset"]
            ph_condition = row["high_condition"]
            ph_score = hi_score
            pl_dataset = row["low_dataset"]
            pl_condition = row["low_condition"]
            pl_score = lo_score
            true_high_in_placebo_high += 1
        else:
            ph_dataset = row["low_dataset"]
            ph_condition = row["low_condition"]
            ph_score = lo_score
            pl_dataset = row["high_dataset"]
            pl_condition = row["high_condition"]
            pl_score = hi_score
        high_dataset_counts[str(ph_dataset)] += 1
        low_dataset_counts[str(pl_dataset)] += 1
        placebo_gaps.append(ph_score - pl_score)
        rows.append(
            {
                **row,
                "pair_id": pair_id,
                "seed": seed,
                "swapped": not keep,
                "placebo_high_dataset": ph_dataset,
                "placebo_high_condition": ph_condition,
                "placebo_high_support_resid_score": ph_score,
                "placebo_low_dataset": pl_dataset,
                "placebo_low_condition": pl_condition,
                "placebo_low_support_resid_score": pl_score,
                "placebo_residual_score_gap": ph_score - pl_score,
            }
        )
    original_mean = statistics.fmean(original_gaps) if original_gaps else 0.0
    placebo_mean = statistics.fmean(placebo_gaps) if placebo_gaps else 0.0
    high_total = sum(high_dataset_counts.values()) or 1
    low_total = sum(low_dataset_counts.values()) or 1
    summary = {
        "seed": seed,
        "pairs": len(pairs),
        "true_high_fraction_in_placebo_high": true_high_in_placebo_high / len(pairs) if pairs else 0.0,
        "mean_original_residual_score_gap": original_mean,
        "mean_placebo_residual_score_gap": placebo_mean,
        "abs_placebo_over_original_score_gap": abs(placebo_mean) / abs(original_mean) if original_mean else None,
        "top_high_dataset_fraction": high_dataset_counts.most_common(1)[0][1] / high_total if high_dataset_counts else 0.0,
        "top_low_dataset_fraction": low_dataset_counts.most_common(1)[0][1] / low_total if low_dataset_counts else 0.0,
        "high_datasets": len(high_dataset_counts),
        "low_datasets": len(low_dataset_counts),
        "perturbation_type_counts": dict(ptype_counts),
        "top_high_dataset_counts": dict(high_dataset_counts.most_common(12)),
        "top_low_dataset_counts": dict(low_dataset_counts.most_common(12)),
    }
    return rows, summary


def write_seed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-split", type=Path, default=PARENT_SPLIT)
    parser.add_argument("--pair-csv", type=Path, default=PAIR_CSV)
    parser.add_argument("--packet-json", type=Path, default=PACKET_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--seeds", default="43,44,45,46")
    parser.add_argument("--max-true-high-imbalance", type=float, default=0.10)
    parser.add_argument("--max-axis-ratio", type=float, default=0.10)
    parser.add_argument("--max-top-side-dataset-fraction", type=float, default=0.18)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    parent = load_json(args.parent_split)
    packet = load_json(args.packet_json)
    pairs = read_pairs(args.pair_csv)

    reasons: list[str] = []
    if packet.get("status") != "condition_neighborhood_response_residualized_support_pass_external_audit_no_gpu":
        reasons.append(f"packet status not expected pass: {packet.get('status')}")
    if len(pairs) < 300:
        reasons.append(f"pair count {len(pairs)} < 300")

    seed_summaries: list[dict[str, Any]] = []
    generated: list[dict[str, str]] = []
    for seed in seeds:
        seed_rows, summary = build_seed_rows(pairs, seed)
        high_path = args.split_dir / f"split_seed42_xverse_condition_neighborhood_response_resid_pairshuffle_seed{seed}_high_320pair.json"
        low_path = args.split_dir / f"split_seed42_xverse_condition_neighborhood_response_resid_pairshuffle_seed{seed}_low_320pair.json"
        row_csv = args.out_dir / f"condition_neighborhood_response_resid_pairshuffle_seed{seed}_matched_pairs.csv"
        write_split(parent, seed_rows, "high", high_path)
        write_split(parent, seed_rows, "low", low_path)
        write_seed_csv(row_csv, seed_rows)
        high = load_json(high_path)
        low = load_json(low_path)
        seed_reasons = validate_split(parent, high, low, len(pairs))
        imbalance = abs(float(summary["true_high_fraction_in_placebo_high"]) - 0.5)
        ratio = summary["abs_placebo_over_original_score_gap"]
        if imbalance > args.max_true_high_imbalance:
            seed_reasons.append(f"true-high imbalance {imbalance:.4f} > {args.max_true_high_imbalance:.4f}")
        if ratio is None or float(ratio) > args.max_axis_ratio:
            seed_reasons.append(f"axis ratio {ratio} > {args.max_axis_ratio}")
        if float(summary["top_high_dataset_fraction"]) > args.max_top_side_dataset_fraction:
            seed_reasons.append(f"top high dataset fraction {summary['top_high_dataset_fraction']:.4f} > {args.max_top_side_dataset_fraction:.4f}")
        if float(summary["top_low_dataset_fraction"]) > args.max_top_side_dataset_fraction:
            seed_reasons.append(f"top low dataset fraction {summary['top_low_dataset_fraction']:.4f} > {args.max_top_side_dataset_fraction:.4f}")
        summary["reasons"] = seed_reasons
        summary["status"] = "pass" if not seed_reasons else "fail"
        summary["high_split"] = str(high_path)
        summary["low_split"] = str(low_path)
        summary["pair_csv"] = str(row_csv)
        seed_summaries.append(summary)
        generated.append({"seed": str(seed), "high_split": str(high_path), "low_split": str(low_path), "pair_csv": str(row_csv)})
        reasons.extend(f"seed {seed}: {reason}" for reason in seed_reasons)

    passed_seeds = [row for row in seed_summaries if row["status"] == "pass"]
    status = (
        "condition_neighborhood_response_resid_pairshuffle_placebo_ready_no_gpu"
        if passed_seeds and not reasons
        else "condition_neighborhood_response_resid_pairshuffle_placebo_blocked_no_gpu"
    )

    manifest = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "boundary": "CPU/report-only pair-label-shuffle placebo split package; no train/infer/checkpoint selection.",
        "parent_split": str(args.parent_split),
        "pair_csv": str(args.pair_csv),
        "packet_json": str(args.packet_json),
        "file_hashes": {
            str(args.parent_split): sha256(args.parent_split),
            str(args.pair_csv): sha256(args.pair_csv),
            str(args.packet_json): sha256(args.packet_json),
        },
        "seeds": seed_summaries,
        "generated": generated,
        "reasons": reasons,
        "future_gate": [
            "Use only if the real response-residualized high/low smoke passes its internal gate.",
            "Placebo high must not beat placebo low by the same direction/margin as real high.",
            "Placebo/candidate comparisons must use the same training budget, seed policy, and internal-only selection boundary.",
        ],
    }
    manifest_path = args.out_dir / "latentfm_condition_neighborhood_response_resid_pairshuffle_placebo_splits_20260629.json"
    write_json(manifest_path, manifest)

    report = args.out_dir / "LATENTFM_CONDITION_NEIGHBORHOOD_RESPONSE_RESID_PAIRSHUFFLE_PLACEBO_SPLITS_20260629.md"
    lines = [
        "# LatentFM Condition-Neighborhood Response-Resid Pair-Shuffle Placebo Splits",
        "",
        f"Timestamp: `{manifest['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only split artifact.",
        "- Preserves the matched condition-pair universe and all non-train buckets.",
        "- Randomly flips labels within each pair to destroy the response-residualized support direction.",
        "- Does not train, infer, select checkpoints, use canonical multi, or use Track C query.",
        "",
        "## Seed Summary",
        "",
        "| seed | status | true high in placebo high | mean placebo gap | placebo/original gap | top high ds | top low ds | outputs |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in seed_summaries:
        ratio = row["abs_placebo_over_original_score_gap"]
        ratio_text = "NA" if ratio is None else f"{float(ratio):.4f}"
        lines.append(
            f"| {row['seed']} | `{row['status']}` | {row['true_high_fraction_in_placebo_high']:.3f} | "
            f"{row['mean_placebo_residual_score_gap']:.4f} | {ratio_text} | "
            f"{row['top_high_dataset_fraction']:.3f} | {row['top_low_dataset_fraction']:.3f} | "
            f"`{Path(row['high_split']).name}`, `{Path(row['low_split']).name}` |"
        )
    lines.extend(["", "## Decision", ""])
    if status.endswith("ready_no_gpu"):
        lines.extend(
            [
                "The pair-shuffle placebo package is ready for a future launch if, and only if, the real high/low smoke first passes its internal gate.",
                "A valid mechanism result should show real high > real low, while placebo high should not reproduce that advantage.",
            ]
        )
    else:
        lines.append("The placebo package is blocked and must not be launched until the listed reasons are repaired.")
    if reasons:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(["", "## Outputs", "", f"- manifest: `{manifest_path}`"])
    for item in generated:
        lines.append(f"- seed {item['seed']} high split: `{item['high_split']}`")
        lines.append(f"- seed {item['seed']} low split: `{item['low_split']}`")
        lines.append(f"- seed {item['seed']} pair CSV: `{item['pair_csv']}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "passed_seeds": [x["seed"] for x in passed_seeds], "out": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
