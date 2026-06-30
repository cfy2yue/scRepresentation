#!/usr/bin/env python3
"""Build pair-label-shuffle placebo splits for scaling-v2 condition information.

This is CPU/report-only. It preserves the exact matched condition set and all
non-train buckets, then randomly flips high/low labels within each matched pair.
The placebo should destroy the information-axis direction while preserving
dataset, perturbation-type, gene-count, and condition-count structure.
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
DRAFT_DIR = ROOT / "reports/scaling_v2_condition_information_draft_splits_20260628"
PAIR_CSV = DRAFT_DIR / "condition_information_matched_pairs.csv"
PACKET_JSON = ROOT / "reports/scaling_v2_condition_information_packet_audit_20260628/latentfm_scaling_v2_condition_information_packet_audit_20260628.json"
OUT_DIR = ROOT / "reports/scaling_v2_condition_information_pairshuffle_placebo_splits_20260628"


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
        "axis",
        "dataset",
        "perturbation_type",
        "gene_count_bin",
        "high_condition",
        "low_condition",
        "high_value",
        "low_value",
        "axis_delta",
    }
    missing = sorted(required - set(rows[0].keys() if rows else []))
    if missing:
        raise ValueError(f"pair CSV missing columns: {missing}")
    return rows


def write_split(parent: dict[str, Any], rows: list[dict[str, Any]], side: str, path: Path) -> None:
    out = copy.deepcopy(parent)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    col = f"placebo_{side}_condition"
    for row in rows:
        by_dataset[str(row["dataset"])].append(str(row[col]))
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
    axis_deltas: list[float] = []
    original_deltas: list[float] = []
    dataset_counts = Counter()
    for pair_id, row in enumerate(pairs):
        keep = rng.random() < 0.5
        high_value = float(row["high_value"])
        low_value = float(row["low_value"])
        original_deltas.append(float(row["axis_delta"]))
        if keep:
            placebo_high = row["high_condition"]
            placebo_low = row["low_condition"]
            placebo_high_value = high_value
            placebo_low_value = low_value
            true_high_in_placebo_high += 1
        else:
            placebo_high = row["low_condition"]
            placebo_low = row["high_condition"]
            placebo_high_value = low_value
            placebo_low_value = high_value
        dataset_counts[str(row["dataset"])] += 1
        axis_deltas.append(placebo_high_value - placebo_low_value)
        rows.append(
            {
                **row,
                "pair_id": pair_id,
                "seed": seed,
                "swapped": not keep,
                "placebo_high_condition": placebo_high,
                "placebo_low_condition": placebo_low,
                "placebo_high_value": placebo_high_value,
                "placebo_low_value": placebo_low_value,
                "placebo_axis_delta": placebo_high_value - placebo_low_value,
            }
        )
    original_mean = statistics.fmean(original_deltas) if original_deltas else 0.0
    placebo_mean = statistics.fmean(axis_deltas) if axis_deltas else 0.0
    true_high_fraction = true_high_in_placebo_high / len(pairs) if pairs else 0.0
    summary = {
        "seed": seed,
        "pairs": len(pairs),
        "true_high_fraction_in_placebo_high": true_high_fraction,
        "mean_original_axis_delta": original_mean,
        "mean_placebo_axis_delta": placebo_mean,
        "abs_placebo_over_original_axis_delta": abs(placebo_mean) / abs(original_mean) if original_mean else None,
        "datasets": len(dataset_counts),
        "top_dataset_counts": dict(dataset_counts.most_common(12)),
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
    parser.add_argument("--seeds", default="43,44")
    parser.add_argument("--max-true-high-imbalance", type=float, default=0.10)
    parser.add_argument("--max-axis-ratio", type=float, default=0.10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    parent = load_json(args.parent_split)
    packet = load_json(args.packet_json)
    pairs = read_pairs(args.pair_csv)

    reasons: list[str] = []
    if packet.get("status") != "scaling_v2_condition_information_packet_audit_pass_prepare_gpu_smoke":
        reasons.append(f"packet audit not pass_prepare_gpu_smoke: {packet.get('status')}")
    if len(pairs) < 300:
        reasons.append(f"pair count {len(pairs)} < 300")

    seed_summaries: list[dict[str, Any]] = []
    generated: list[dict[str, str]] = []
    for seed in seeds:
        seed_rows, summary = build_seed_rows(pairs, seed)
        high_path = args.out_dir / f"pairshuffle_seed{seed}_xverse_info_composite_placebo_high_from_cap120_all_v2.json"
        low_path = args.out_dir / f"pairshuffle_seed{seed}_xverse_info_composite_placebo_low_from_cap120_all_v2.json"
        row_csv = args.out_dir / f"pairshuffle_seed{seed}_matched_pairs.csv"
        write_split(parent, seed_rows, "high", high_path)
        write_split(parent, seed_rows, "low", low_path)
        write_seed_csv(row_csv, seed_rows)
        high = load_json(high_path)
        low = load_json(low_path)
        seed_reasons = validate_split(parent, high, low, len(pairs))
        imbalance = abs(float(summary["true_high_fraction_in_placebo_high"]) - 0.5)
        ratio = summary["abs_placebo_over_original_axis_delta"]
        if imbalance > args.max_true_high_imbalance:
            seed_reasons.append(f"true-high imbalance {imbalance:.4f} > {args.max_true_high_imbalance:.4f}")
        if ratio is None or float(ratio) > args.max_axis_ratio:
            seed_reasons.append(f"axis ratio {ratio} > {args.max_axis_ratio}")
        summary["reasons"] = seed_reasons
        summary["status"] = "pass" if not seed_reasons else "fail"
        summary["high_split"] = str(high_path)
        summary["low_split"] = str(low_path)
        summary["pair_csv"] = str(row_csv)
        seed_summaries.append(summary)
        generated.append({"seed": str(seed), "high_split": str(high_path), "low_split": str(low_path), "pair_csv": str(row_csv)})
        reasons.extend(f"seed {seed}: {reason}" for reason in seed_reasons)

    passed_seeds = [row for row in seed_summaries if row["status"] == "pass"]
    status = "scaling_v2_condition_information_pairshuffle_placebo_ready_no_gpu" if passed_seeds and not reasons else "scaling_v2_condition_information_pairshuffle_placebo_blocked_no_gpu"

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
            "Use only if the real high/low smoke passes its internal no-harm gate.",
            "Placebo high must not beat placebo low by the same direction/margin as real high.",
            "Placebo/candidate comparisons must use the same training budget, seed policy, and internal-only selection boundary.",
        ],
    }
    manifest_path = args.out_dir / "latentfm_scaling_v2_condition_information_pairshuffle_placebo_splits_20260628.json"
    write_json(manifest_path, manifest)

    report = args.out_dir / "LATENTFM_SCALING_V2_CONDITION_INFORMATION_PAIRSHUFFLE_PLACEBO_SPLITS_20260628.md"
    lines = [
        "# LatentFM Scaling V2 Condition-Information Pair-Shuffle Placebo Splits",
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
        "- Preserves the exact matched condition set, dataset, perturbation-type, gene-count-bin, and non-train buckets.",
        "- Randomly flips labels within each matched pair to destroy the information-axis direction.",
        "- Does not train, infer, select checkpoints, use canonical multi, or use Track C query.",
        "",
        "## Seed Summary",
        "",
        "| seed | status | true high in placebo high | mean placebo axis delta | placebo/original axis | outputs |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in seed_summaries:
        lines.append(
            f"| {row['seed']} | `{row['status']}` | {row['true_high_fraction_in_placebo_high']:.3f} | "
            f"{row['mean_placebo_axis_delta']:.4f} | {row['abs_placebo_over_original_axis_delta']:.4f} | "
            f"`{Path(row['high_split']).name}`, `{Path(row['low_split']).name}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if status.endswith("ready_no_gpu"):
        lines.extend(
            [
                "The pair-shuffle placebo package is ready for a future launch if, and only if, the real high/low smoke first passes its internal gate.",
                "A placebo run is expected to collapse the high-vs-low advantage; otherwise the condition-information mechanism claim is weak.",
            ]
        )
    else:
        lines.extend(
            [
                "The placebo package is blocked and must not be launched until the listed reasons are repaired.",
            ]
        )
    if reasons:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- manifest: `{manifest_path}`",
        ]
    )
    for item in generated:
        lines.append(f"- seed {item['seed']} high split: `{item['high_split']}`")
        lines.append(f"- seed {item['seed']} low split: `{item['low_split']}`")
        lines.append(f"- seed {item['seed']} pair CSV: `{item['pair_csv']}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "passed_seeds": [x["seed"] for x in passed_seeds], "out": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
