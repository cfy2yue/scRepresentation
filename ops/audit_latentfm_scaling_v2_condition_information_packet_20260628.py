#!/usr/bin/env python3
"""Audit scaling-v2 condition-information draft splits before any GPU launch.

The draft split feasibility report produced 306 high/low matched condition
pairs.  This packet audit checks provenance, split safety, reconstruction, and
actual latent dataloader visibility.  It remains CPU/report-only and never
authorizes GPU by itself.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
PARENT_SPLIT = ROOT / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
DRAFT_DIR = ROOT / "reports/scaling_v2_condition_information_draft_splits_20260628"
HIGH_SPLIT = DRAFT_DIR / "draft_split_seed42_xverse_info_composite_high_from_cap120_all_v2.json"
LOW_SPLIT = DRAFT_DIR / "draft_split_seed42_xverse_info_composite_low_from_cap120_all_v2.json"
PAIR_CSV = DRAFT_DIR / "condition_information_matched_pairs.csv"
TABLE_CSV = DRAFT_DIR / "condition_information_table.csv"
SUMMARY_CSV = DRAFT_DIR / "condition_information_dataset_summary.csv"
DESIGN_SCRIPT = ROOT / "ops/design_latentfm_scaling_v2_condition_information_draft_splits_20260628.py"
COVERAGE_CSV = ROOT / "reports/exact_response_information_combined_coverage_20260628/exact_response_information_condition_rows.csv"
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
OUT_DIR = ROOT / "reports/scaling_v2_condition_information_packet_audit_20260628"


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


def split_items(split: dict[str, Any], group: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for dataset, groups in split.items():
        for condition in (groups or {}).get(group, []) or []:
            out.add((str(dataset), str(condition)))
    return out


def group_counts(split: dict[str, Any]) -> dict[str, int]:
    counts = Counter()
    for groups in split.values():
        for group, values in (groups or {}).items():
            counts[str(group)] += len(values or [])
    return dict(sorted(counts.items()))


def nontrain_groups(split: dict[str, Any]) -> set[str]:
    groups = set()
    for dataset_groups in split.values():
        groups.update(str(g) for g in dataset_groups.keys() if str(g) != "train")
    return groups


def reconstruct_split(parent: dict[str, Any], pairs: pd.DataFrame, side: str) -> dict[str, Any]:
    out = copy.deepcopy(parent)
    col = f"{side}_condition"
    by_dataset: dict[str, set[str]] = defaultdict(set)
    for _, row in pairs.iterrows():
        by_dataset[str(row["dataset"])].add(str(row[col]))
    for dataset in out:
        out[dataset]["train"] = sorted(by_dataset.get(dataset, set()))
    return out


def compare_nontrain(parent: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for dataset, groups in parent.items():
        cand_groups = candidate.get(dataset, {})
        for group, values in groups.items():
            if group == "train":
                continue
            if list(values or []) != list(cand_groups.get(group, []) or []):
                reasons.append(f"nontrain group changed: {dataset}/{group}")
    return reasons


def split_overlap_report(split: dict[str, Any]) -> dict[str, Any]:
    train = split_items(split, "train")
    groups = sorted(nontrain_groups(split))
    overlaps = {}
    for group in groups:
        overlap = sorted(train & split_items(split, group))
        if overlap:
            overlaps[group] = overlap[:20]
    return {
        "train_conditions": len(train),
        "nontrain_groups": groups,
        "overlap_groups": overlaps,
    }


def dataloader_dryrun(split: dict[str, Any], label: str, args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(COUPLED))
    from model.latent.dataset import CrossDatasetFMDataset  # noqa: WPS433

    ds = CrossDatasetFMDataset(
        str(args.data_dir),
        split,
        batch_size=args.batch_size,
        seed=args.seed,
        mode="train",
        min_cells=args.min_cells,
        ds_alpha=1.0,
        scale_noise=0.0,
        min_selected_conditions_per_dataset=0,
        condition_visit_power=1.0,
        condition_visit_cap=0,
        use_pert_condition=False,
        biflow_dir=str(ROOT / "dataset/biFlow_data"),
        silent=True,
    )
    try:
        counts = {dataset: len(conds) for dataset, conds in sorted(ds.ds_conds.items())}
        sizes = {
            dataset: {
                condition: {
                    "n_source": int(size[0]),
                    "n_gt": int(size[1]),
                }
                for condition, size in sorted(ds._cond_sizes[dataset].items())
            }
            for dataset in sorted(ds._cond_sizes)
        }
        payload = {
            "label": label,
            "status": "pass" if ds.total_conditions > 0 else "fail_empty",
            "datasets": len(counts),
            "total_conditions": int(ds.total_conditions),
            "epoch_steps": int(ds.epoch_steps),
            "conditions_by_dataset": counts,
            "condition_sizes_preview": {
                dataset: dict(list(items.items())[:3])
                for dataset, items in sizes.items()
            },
        }
    finally:
        for handle in getattr(ds, "handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-split", type=Path, default=PARENT_SPLIT)
    parser.add_argument("--high-split", type=Path, default=HIGH_SPLIT)
    parser.add_argument("--low-split", type=Path, default=LOW_SPLIT)
    parser.add_argument("--pair-csv", type=Path, default=PAIR_CSV)
    parser.add_argument("--table-csv", type=Path, default=TABLE_CSV)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--coverage-csv", type=Path, default=COVERAGE_CSV)
    parser.add_argument("--condition-metadata", type=Path, default=COND_META)
    parser.add_argument("--design-script", type=Path, default=DESIGN_SCRIPT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-cells", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    required = [
        args.parent_split,
        args.high_split,
        args.low_split,
        args.pair_csv,
        args.table_csv,
        args.summary_csv,
        args.coverage_csv,
        args.condition_metadata,
        args.design_script,
        args.data_dir / "manifest.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    reasons: list[str] = []
    if missing:
        reasons.extend(f"missing required artifact: {p}" for p in missing)

    parent = load_json(args.parent_split)
    high = load_json(args.high_split)
    low = load_json(args.low_split)
    pairs = pd.read_csv(args.pair_csv)
    table = pd.read_csv(args.table_csv)
    summary = pd.read_csv(args.summary_csv)

    file_hashes = {str(path): sha256(path) for path in required if path.exists() and path.is_file()}

    if not {"dataset", "high_condition", "low_condition", "axis", "high_value", "low_value"}.issubset(pairs.columns):
        reasons.append("pair CSV missing required columns")
    if not {"dataset", "condition", "info_composite", "response_energy", "hvg_k80", "matrix_source", "log1p_policy"}.issubset(table.columns):
        reasons.append("condition table missing required provenance/info columns")

    high_train = split_items(high, "train")
    low_train = split_items(low, "train")
    parent_train = split_items(parent, "train")
    pair_high = set(zip(pairs["dataset"].astype(str), pairs["high_condition"].astype(str)))
    pair_low = set(zip(pairs["dataset"].astype(str), pairs["low_condition"].astype(str)))

    if high_train != pair_high:
        reasons.append(f"high split train set != pair high set ({len(high_train)} vs {len(pair_high)})")
    if low_train != pair_low:
        reasons.append(f"low split train set != pair low set ({len(low_train)} vs {len(pair_low)})")
    if high_train & low_train:
        reasons.append(f"high/low train overlap {len(high_train & low_train)}")
    if not high_train <= parent_train:
        reasons.append(f"high train has {len(high_train - parent_train)} conditions outside parent train")
    if not low_train <= parent_train:
        reasons.append(f"low train has {len(low_train - parent_train)} conditions outside parent train")
    if len(high_train) != len(pairs) or len(low_train) != len(pairs):
        reasons.append("duplicate high or low train conditions found relative to pair count")

    recon_high = reconstruct_split(parent, pairs, "high")
    recon_low = reconstruct_split(parent, pairs, "low")
    if recon_high != high:
        reasons.append("high split does not reconstruct exactly from parent + pair CSV")
    if recon_low != low:
        reasons.append("low split does not reconstruct exactly from parent + pair CSV")
    reasons.extend(compare_nontrain(parent, high))
    reasons.extend(compare_nontrain(parent, low))

    high_overlap = split_overlap_report(high)
    low_overlap = split_overlap_report(low)
    if high_overlap["overlap_groups"]:
        reasons.append(f"high train overlaps nontrain groups: {sorted(high_overlap['overlap_groups'])}")
    if low_overlap["overlap_groups"]:
        reasons.append(f"low train overlaps nontrain groups: {sorted(low_overlap['overlap_groups'])}")

    table_keys = set(zip(table["dataset"].astype(str), table["condition"].astype(str)))
    missing_table = (high_train | low_train) - table_keys
    if missing_table:
        reasons.append(f"selected conditions missing condition table rows: {len(missing_table)}")

    pair_axis_ok = bool((pairs["high_value"] > pairs["low_value"]).all())
    if not pair_axis_ok:
        reasons.append("some high_value <= low_value")

    high_response_gt = int((pairs["high_response_energy"] > pairs["low_response_energy"]).sum())
    high_hvg_k80_lt = int((pairs["high_hvg_k80"] < pairs["low_hvg_k80"]).sum())
    zero_train_eval_datasets = []
    for dataset, groups in high.items():
        train_n = len(groups.get("train", []) or [])
        nontrain_n = sum(len(values or []) for key, values in groups.items() if key != "train")
        if train_n == 0 and nontrain_n > 0:
            zero_train_eval_datasets.append(dataset)

    loader_reports = []
    try:
        loader_reports.append(dataloader_dryrun(high, "high", args))
        loader_reports.append(dataloader_dryrun(low, "low", args))
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"dataloader dry-run failed: {type(exc).__name__}: {exc}")
        loader_reports.append({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    for report in loader_reports:
        if report.get("status") != "pass":
            reasons.append(f"dataloader dry-run did not pass for {report.get('label')}: {report.get('status')}")

    dataset_rows = []
    for dataset in sorted(set(parent) | set(high) | set(low)):
        dataset_rows.append(
            {
                "dataset": dataset,
                "parent_train": len((parent.get(dataset) or {}).get("train", []) or []),
                "high_train": len((high.get(dataset) or {}).get("train", []) or []),
                "low_train": len((low.get(dataset) or {}).get("train", []) or []),
                "matched_pairs": int((pairs["dataset"].astype(str) == dataset).sum()),
                "loader_high": next((r.get("conditions_by_dataset", {}).get(dataset, 0) for r in loader_reports if r.get("label") == "high"), 0),
                "loader_low": next((r.get("conditions_by_dataset", {}).get(dataset, 0) for r in loader_reports if r.get("label") == "low"), 0),
                "nontrain_conditions": sum(
                    len(values or [])
                    for key, values in (parent.get(dataset) or {}).items()
                    if key != "train"
                ),
            }
        )
    dataset_csv = args.out_dir / "scaling_v2_condition_information_packet_dataset_rows.csv"
    pd.DataFrame(dataset_rows).to_csv(dataset_csv, index=False)

    status = (
        "scaling_v2_condition_information_packet_audit_pass_prepare_gpu_smoke"
        if not reasons
        and len(pairs) >= 300
        and summary["dataset"].nunique() >= 8
        and all(report.get("status") == "pass" for report in loader_reports)
        else "scaling_v2_condition_information_packet_audit_fail_or_block_no_gpu"
    )

    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "file_hashes": file_hashes,
        "counts": {
            "matched_pairs": int(len(pairs)),
            "datasets_with_pairs": int(summary["dataset"].nunique()) if "dataset" in summary else 0,
            "high_train_conditions": len(high_train),
            "low_train_conditions": len(low_train),
            "parent_train_conditions": len(parent_train),
            "pair_axis_all_high_gt_low": pair_axis_ok,
            "high_response_energy_gt_low_pairs": high_response_gt,
            "high_hvg_k80_lt_low_pairs": high_hvg_k80_lt,
        },
        "split_group_counts": {
            "parent": group_counts(parent),
            "high": group_counts(high),
            "low": group_counts(low),
        },
        "zero_train_eval_datasets": zero_train_eval_datasets,
        "loader_reports": loader_reports,
        "provenance": {
            "axis": "info_composite",
            "axis_definition": "dataset-robust z(log_response_energy) + z(hvg_concentration_80) + 0.5*z(hvg_advantage_80) + 0.25*z(cell_support_log)",
            "condition_table_columns": list(table.columns),
            "source_coverage_csv": str(args.coverage_csv),
            "condition_metadata": str(args.condition_metadata),
            "design_script": str(args.design_script),
            "log1p_policies": sorted(set(map(str, table.get("log1p_policy", [])))),
            "matrix_sources": sorted(set(map(str, table.get("matrix_source", [])))),
        },
        "required_gpu_gate_if_promoted": [
            "bounded high-vs-low matched smoke under identical seed/config/budget",
            "matched label-shuffle or random high/low placebo split",
            "dual baseline against xverse_8k_anchor and source/control",
            "train-only/internal checkpoint rule; no canonical multi or Track C query selection",
            "clustered/dataset bootstrap and explicit zero-train-eval stratum",
        ],
    }
    json_path = args.out_dir / "latentfm_scaling_v2_condition_information_packet_audit_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    md_path = args.out_dir / "LATENTFM_SCALING_V2_CONDITION_INFORMATION_PACKET_AUDIT_20260628.md"
    top_dataset_rows = sorted(dataset_rows, key=lambda r: r["matched_pairs"], reverse=True)[:12]
    lines = [
        "# LatentFM Scaling V2 Condition-Information Packet Audit",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only preflight for the condition-information draft high/low splits.",
        "- No training, inference, checkpoint selection, canonical multi selection, or Track C query use.",
        "- A pass means the packet is coherent enough to design a bounded GPU smoke; it is not itself model evidence.",
        "",
        "## Core Checks",
        "",
        f"- matched pairs: `{len(pairs)}` across `{payload['counts']['datasets_with_pairs']}` datasets.",
        f"- high/low train conditions: `{len(high_train)}` / `{len(low_train)}`.",
        f"- high/low disjoint and parent-train subset: `{not bool(high_train & low_train) and high_train <= parent_train and low_train <= parent_train}`.",
        f"- reconstructed from parent + pair CSV: `{recon_high == high and recon_low == low}`.",
        f"- train/nontrain overlap free: `{not high_overlap['overlap_groups'] and not low_overlap['overlap_groups']}`.",
        f"- high response-energy > low in `{high_response_gt}/{len(pairs)}` pairs.",
        f"- high HVG k80 < low in `{high_hvg_k80_lt}/{len(pairs)}` pairs.",
        "",
        "## Dataloader Dry Run",
        "",
        "| split | status | datasets | total conditions | epoch steps |",
        "|---|---|---:|---:|---:|",
    ]
    for report in loader_reports:
        lines.append(
            f"| {report.get('label', '')} | {report.get('status', '')} | "
            f"{report.get('datasets', 0)} | {report.get('total_conditions', 0)} | {report.get('epoch_steps', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Dataset Balance",
            "",
            "| dataset | parent train | high train | low train | loader high | loader low | matched pairs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['parent_train']} | {row['high_train']} | {row['low_train']} | "
            f"{row['loader_high']} | {row['loader_low']} | {row['matched_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## Zero-Train Eval Datasets",
            "",
            ", ".join(f"`{x}`" for x in zero_train_eval_datasets) if zero_train_eval_datasets else "None.",
            "",
            "## Decision",
            "",
        ]
    )
    if reasons:
        lines.extend(["The packet is blocked for GPU design because:", ""])
        lines.extend(f"- {reason}" for reason in reasons)
    else:
        lines.extend(
            [
                "The packet passes structural/provenance/loadability checks and can be used to design a bounded high-vs-low GPU smoke. GPU remains unauthorized until a separate launcher/no-harm protocol is written with dual baselines and placebo splits.",
            ]
        )
    lines.extend(
        [
            "",
            "## Required If Promoted",
            "",
            "- high-information split must beat matched low-information split under identical seed/config/budget;",
            "- a pair-label shuffle or random matched split must fail/collapse;",
            "- candidate must be compared to `xverse_8k_anchor` and source/control baseline;",
            "- checkpoint/route selection must stay train-only/internal; canonical multi and Track C query remain forbidden;",
            "- report clustered/dataset bootstrap and isolate the zero-train-eval dataset stratum.",
            "",
            "## Outputs",
            "",
            f"- dataset rows: `{dataset_csv}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "reasons": reasons, "out_dir": str(args.out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
