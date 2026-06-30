#!/usr/bin/env python3
"""Build pair-label-shuffle placebo splits for HVG-advantage residual v3.

CPU/report-only. The placebo preserves the selected v3 matched-pair universe
and all non-train buckets, then randomly flips high/low labels within each
pair. This destroys the residual HVG-advantage direction while preserving the
condition set, dataset, perturbation-type, gene-count, and train/eval boundary.
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
PAIR_DIR = ROOT / "reports/hvg_advantage_resid_v3_pair_pool_20260630"
PAIR_CSV = PAIR_DIR / "hvg_advantage_resid_v3_selected_pairs_20260630.csv"
PACKET_JSON = PAIR_DIR / "hvg_advantage_resid_v3_packet_audit_20260630.json"
OUT_DIR = ROOT / "reports/hvg_advantage_resid_v3_pairshuffle_placebo_splits_20260630"
EXPECTED_PACKET_STATUS = "hvg_advantage_resid_v3_pair_pool_pass_prepare_gpu_smoke"


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
        "mode",
        "axis",
        "dataset",
        "perturbation_type",
        "gene_count_bin",
        "high_condition",
        "low_condition",
        "axis_delta",
        "confound_distance",
        "high_axis_value",
        "low_axis_value",
    }
    missing = sorted(required - set(rows[0].keys() if rows else []))
    if missing:
        raise ValueError(f"pair CSV missing columns: {missing}")
    return rows


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


def write_split(parent: dict[str, Any], rows: list[dict[str, Any]], side: str, path: Path) -> None:
    out = copy.deepcopy(parent)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    col = f"placebo_{side}_condition"
    for row in rows:
        by_dataset[str(row["dataset"])].append(str(row[col]))
    for dataset in out:
        out[dataset]["train"] = sorted(set(by_dataset.get(str(dataset), [])))
    write_json(path, out)


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
        high_overlap = high_train & split_items(high, group)
        low_overlap = low_train & split_items(low, group)
        if high_overlap:
            reasons.append(f"high train overlaps {group}: {len(high_overlap)}")
        if low_overlap:
            reasons.append(f"low train overlaps {group}: {len(low_overlap)}")
    reasons.extend(compare_nontrain(parent, high))
    reasons.extend(compare_nontrain(parent, low))
    return reasons


def build_seed_rows(pairs: list[dict[str, str]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    true_high_in_placebo_high = 0
    placebo_deltas: list[float] = []
    original_deltas: list[float] = []
    confound_distances: list[float] = []
    dataset_counts: Counter[str] = Counter()
    ptype_counts: Counter[str] = Counter()

    for pair_id, row in enumerate(pairs):
        keep = rng.random() < 0.5
        high_value = float(row["high_axis_value"])
        low_value = float(row["low_axis_value"])
        original_deltas.append(float(row["axis_delta"]))
        confound_distances.append(float(row["confound_distance"]))
        dataset_counts[str(row["dataset"])] += 1
        ptype_counts[str(row["perturbation_type"])] += 1
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
        placebo_delta = placebo_high_value - placebo_low_value
        placebo_deltas.append(placebo_delta)
        rows.append(
            {
                **row,
                "pair_id": pair_id,
                "seed": seed,
                "swapped": not keep,
                "placebo_high_condition": placebo_high,
                "placebo_low_condition": placebo_low,
                "placebo_high_axis_value": placebo_high_value,
                "placebo_low_axis_value": placebo_low_value,
                "placebo_axis_delta": placebo_delta,
            }
        )

    original_mean = statistics.fmean(original_deltas) if original_deltas else 0.0
    placebo_mean = statistics.fmean(placebo_deltas) if placebo_deltas else 0.0
    total = len(pairs) or 1
    summary = {
        "seed": seed,
        "pairs": len(pairs),
        "true_high_fraction_in_placebo_high": true_high_in_placebo_high / total,
        "mean_original_axis_delta": original_mean,
        "mean_placebo_axis_delta": placebo_mean,
        "abs_placebo_over_original_axis_delta": abs(placebo_mean) / abs(original_mean) if original_mean else None,
        "datasets": len(dataset_counts),
        "top_dataset_fraction": dataset_counts.most_common(1)[0][1] / total if dataset_counts else 0.0,
        "top_dataset_counts": dict(dataset_counts.most_common(12)),
        "top_perturbation_type_counts": dict(ptype_counts.most_common(12)),
        "mean_confound_distance": statistics.fmean(confound_distances) if confound_distances else None,
        "pair_distance_gt_0p5_fraction": sum(x > 0.5 for x in confound_distances) / total,
    }
    return rows, summary


def write_seed_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-split", type=Path, default=PARENT_SPLIT)
    parser.add_argument("--pair-csv", type=Path, default=PAIR_CSV)
    parser.add_argument("--packet-json", type=Path, default=PACKET_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seeds", default="43,44,45,46")
    parser.add_argument("--min-pairs", type=int, default=200)
    parser.add_argument("--max-true-high-imbalance", type=float, default=0.10)
    parser.add_argument("--max-axis-ratio", type=float, default=0.10)
    parser.add_argument("--max-top-dataset-fraction", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"output directory is not empty; use --force to overwrite: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    parent = load_json(args.parent_split)
    packet = load_json(args.packet_json)
    pairs = read_pairs(args.pair_csv)

    reasons: list[str] = []
    if packet.get("status") != EXPECTED_PACKET_STATUS:
        reasons.append(f"packet status not expected pass: {packet.get('status')}")
    if len(pairs) < args.min_pairs:
        reasons.append(f"pair count {len(pairs)} < {args.min_pairs}")

    seed_summaries: list[dict[str, Any]] = []
    generated: list[dict[str, str]] = []
    for seed in seeds:
        seed_rows, summary = build_seed_rows(pairs, seed)
        high_path = args.out_dir / f"split_seed42_xverse_hvg_advantage_resid_v3_pairshuffle_seed{seed}_high_from_cap120_all_v2.json"
        low_path = args.out_dir / f"split_seed42_xverse_hvg_advantage_resid_v3_pairshuffle_seed{seed}_low_from_cap120_all_v2.json"
        pair_path = args.out_dir / f"hvg_advantage_resid_v3_pairshuffle_seed{seed}_matched_pairs.csv"
        write_split(parent, seed_rows, "high", high_path)
        write_split(parent, seed_rows, "low", low_path)
        write_seed_csv(pair_path, seed_rows)
        high = load_json(high_path)
        low = load_json(low_path)

        seed_reasons = validate_split(parent, high, low, len(pairs))
        imbalance = abs(float(summary["true_high_fraction_in_placebo_high"]) - 0.5)
        ratio = summary["abs_placebo_over_original_axis_delta"]
        top_dataset_fraction = float(summary["top_dataset_fraction"])
        if imbalance > args.max_true_high_imbalance:
            seed_reasons.append(f"true-high imbalance {imbalance:.4f} > {args.max_true_high_imbalance:.4f}")
        if ratio is None or float(ratio) > args.max_axis_ratio:
            seed_reasons.append(f"axis ratio {ratio} > {args.max_axis_ratio}")
        if top_dataset_fraction > args.max_top_dataset_fraction:
            seed_reasons.append(f"top dataset fraction {top_dataset_fraction:.4f} > {args.max_top_dataset_fraction:.4f}")
        summary["reasons"] = seed_reasons
        summary["status"] = "pass" if not seed_reasons else "fail"
        summary["high_split"] = str(high_path)
        summary["low_split"] = str(low_path)
        summary["pair_csv"] = str(pair_path)
        seed_summaries.append(summary)
        generated.append({"seed": str(seed), "high_split": str(high_path), "low_split": str(low_path), "pair_csv": str(pair_path)})
        reasons.extend(f"seed {seed}: {reason}" for reason in seed_reasons)

    passed_seeds = [row for row in seed_summaries if row["status"] == "pass"]
    failed_seeds = [row for row in seed_summaries if row["status"] != "pass"]
    if passed_seeds and not failed_seeds and not reasons:
        status = "hvg_advantage_resid_v3_pairshuffle_placebo_ready_no_gpu"
    elif passed_seeds:
        status = "hvg_advantage_resid_v3_pairshuffle_placebo_partial_ready_no_gpu"
    else:
        status = "hvg_advantage_resid_v3_pairshuffle_placebo_blocked_no_gpu"

    manifest = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "boundary": "CPU/report-only pair-label-shuffle placebo splits; no train/infer/checkpoint selection/canonical multi/Track C query.",
        "parent_split": str(args.parent_split),
        "pair_csv": str(args.pair_csv),
        "packet_json": str(args.packet_json),
        "file_hashes": {
            str(args.parent_split): sha256(args.parent_split),
            str(args.pair_csv): sha256(args.pair_csv),
            str(args.packet_json): sha256(args.packet_json),
        },
        "seeds": seed_summaries,
        "passed_seeds": [row["seed"] for row in passed_seeds],
        "failed_seeds": [row["seed"] for row in failed_seeds],
        "generated": generated,
        "reasons": reasons,
        "future_gate": [
            "Use only if the real v3 high/low smoke first passes the internal gate.",
            "A placebo high/low run should collapse the real high-vs-low advantage.",
            "If placebo recovers a comparable gap, demote or close the v3 mechanism.",
            "Any later promotion still requires dual-baseline no-harm; this package does not authorize GPU by itself.",
        ],
    }

    manifest_path = args.out_dir / "latentfm_hvg_advantage_resid_v3_pairshuffle_placebo_splits_20260630.json"
    write_json(manifest_path, manifest)

    report = args.out_dir / "LATENTFM_HVG_ADVANTAGE_RESID_V3_PAIRSHUFFLE_PLACEBO_SPLITS_20260630.md"
    lines = [
        "# LatentFM HVG-Advantage Residual V3 Pair-Shuffle Placebo Splits",
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
        "- Preserves the exact v3 matched condition set and all non-train buckets.",
        "- Randomly flips labels within each matched pair to destroy the HVG-advantage direction.",
        "- Does not train, infer, select checkpoints, use canonical multi, or use Track C query.",
        "",
        "## Seed Summary",
        "",
        "| seed | status | true high in placebo high | mean placebo axis delta | placebo/original axis | top dataset frac | distance >0.5 frac | outputs |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in seed_summaries:
        lines.append(
            f"| {row['seed']} | `{row['status']}` | {row['true_high_fraction_in_placebo_high']:.3f} | "
            f"{fmt(row['mean_placebo_axis_delta'])} | {fmt(row['abs_placebo_over_original_axis_delta'])} | "
            f"{fmt(row['top_dataset_fraction'])} | {fmt(row['pair_distance_gt_0p5_fraction'])} | "
            f"`{Path(row['high_split']).name}`, `{Path(row['low_split']).name}` |"
        )
    lines.extend(["", "## Decision", ""])
    if status.endswith("ready_no_gpu"):
        lines.extend(
            [
                "The pair-shuffle placebo package is ready for a future launch if, and only if, the real v3 high/low smoke first passes its internal gate.",
                "The expected result is collapse of the high-vs-low advantage. A comparable placebo gap would block the mechanism claim.",
            ]
        )
    elif status.endswith("partial_ready_no_gpu"):
        lines.extend(
            [
                "The pair-shuffle placebo package is partially ready. Only the passing seeds should be considered for a future launch, and only if the real v3 high/low smoke first passes its internal gate.",
                "Failed seeds are retained as provenance but should not be trained because their shuffled axis direction did not collapse enough.",
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
