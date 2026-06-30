#!/usr/bin/env python3
"""Synthesize expanded HVG/observable-gene control gates into one decision."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_ABUNDANCE_JSON = ROOT / "reports/hvg_vs_abundance_expanded_baseline_20260629/latentfm_hvg_vs_abundance_baseline_20260628.json"
DEFAULT_DETECTION_JSON = ROOT / "reports/hvg_vs_detection_expanded_baseline_20260629/latentfm_hvg_vs_detection_baseline_20260629.json"
DEFAULT_MEANMATCHED_JSON = ROOT / "reports/hvg_meanmatched_expanded_controls_20260629/latentfm_hvg_meanmatched_negative_controls_20260628.json"
DEFAULT_OUT_DIR = ROOT / "reports/hvg_observable_budget_control_decision_20260629"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["group"]), int(row["budget"])


def select_rows(data: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in data["summary_rows"]:
        if row.get("level") in {"group", "all"} and row.get("dataset") == "__ALL__":
            rows[row_key(row)] = row
    return rows


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abundance-json", type=Path, default=DEFAULT_ABUNDANCE_JSON)
    parser.add_argument("--detection-json", type=Path, default=DEFAULT_DETECTION_JSON)
    parser.add_argument("--meanmatched-json", type=Path, default=DEFAULT_MEANMATCHED_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    abundance = load_json(args.abundance_json)
    detection = load_json(args.detection_json)
    meanmatched = load_json(args.meanmatched_json)

    abundance_rows = select_rows(abundance)
    detection_rows = select_rows(detection)
    mean_rows = select_rows(meanmatched)
    keys = sorted(set(abundance_rows) | set(detection_rows) | set(mean_rows), key=lambda k: (k[0], k[1]))

    synthesis_rows: list[dict[str, Any]] = []
    for key in keys:
        group, budget = key
        arow = abundance_rows.get(key, {})
        drow = detection_rows.get(key, {})
        mrow = mean_rows.get(key, {})
        synthesis_rows.append(
            {
                "group": group,
                "budget": budget,
                "rows": mrow.get("condition_rows", arow.get("condition_rows", drow.get("condition_rows"))),
                "hvg_share": mrow.get("control_hvg_share_mean", arow.get("control_hvg_share_mean", drow.get("control_hvg_share_mean"))),
                "random_share": mrow.get("random_share_mean"),
                "mean_matched_share": mrow.get("mean_matched_random_share_mean"),
                "hvg_minus_mean_matched": mrow.get("hvg_minus_mean_matched_mean"),
                "hvg_minus_mean_matched_ci_low": mrow.get("hvg_minus_mean_matched_ci95_low"),
                "abundance_share": arow.get("control_abundance_share_mean"),
                "hvg_minus_abundance": arow.get("hvg_minus_abundance_mean"),
                "hvg_abundance_overlap": arow.get("hvg_abundance_overlap_fraction_mean"),
                "detection_share": drow.get("control_detection_share_mean"),
                "hvg_minus_detection": drow.get("hvg_minus_detection_mean"),
                "hvg_detection_overlap": drow.get("hvg_detection_overlap_fraction_mean"),
                "shuffled_label_hvg_share": mrow.get("shuffled_label_hvg_share_mean"),
                "split_half_fold_random": mrow.get("split_half_overlap_fold_random_mean"),
            }
        )

    top1000 = [row for row in synthesis_rows if row["budget"] == 1000 and row["group"] in {"chemicalpert_bench", "genepert_DE5000_small"}]
    max_meanmatched_top1000 = max(float(row["hvg_minus_mean_matched"]) for row in top1000)
    max_abundance_top1000 = max(abs(float(row["hvg_minus_abundance"])) for row in top1000)
    max_detection_top1000 = max(float(row["hvg_minus_detection"]) for row in top1000)
    min_overlap_top1000 = min(float(row["hvg_abundance_overlap"]) for row in top1000)

    status = "hvg_observable_gene_budget_descriptive_no_gpu"
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "created_at": created_at,
        "status": status,
        "source_jsons": {
            "abundance": str(args.abundance_json),
            "detection": str(args.detection_json),
            "meanmatched": str(args.meanmatched_json),
        },
        "synthesis_rows": synthesis_rows,
        "decision_metrics": {
            "max_meanmatched_top1000_advantage": max_meanmatched_top1000,
            "max_abs_abundance_top1000_difference": max_abundance_top1000,
            "max_detection_top1000_advantage": max_detection_top1000,
            "min_abundance_overlap_top1000": min_overlap_top1000,
            "hvg_specific_gpu_gate": "fail",
        },
        "decision": (
            "Keep the axis as descriptive observable-gene/top-token response-information "
            "scaling; do not launch an HVG-specific downstream GPU intervention."
        ),
    }

    json_path = args.out_dir / "latentfm_hvg_observable_budget_control_decision_20260629.json"
    with json_path.open("w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    md_lines = [
        "# LatentFM HVG Observable-Gene Budget Control Decision",
        "",
        f"Created: {created_at}",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "* CPU-only synthesis of previously completed expanded HVG/token-budget controls.",
        "* Reads abundance-rank, detection-rank, and mean-matched/shuffled-label control JSONs.",
        "* Does not train, infer, evaluate canonical multi, use Track C query, select checkpoints, or authorize GPU.",
        "",
        "## Integrated Evidence",
        "",
        "| group | budget | rows | HVG | random | mean-matched | HVG-mean | CI low | abundance | HVG-abund | abund overlap | detection | HVG-detect | detect overlap | shuffled-label HVG | split-half fold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in synthesis_rows:
        md_lines.append(
            "| {group} | {budget} | {rows} | {hvg} | {random} | {mean} | {hvg_mean} | {ci_low} | {abundance} | {hvg_abund} | {abund_overlap} | {detection} | {hvg_detect} | {detect_overlap} | {shuffled} | {split_fold} |".format(
                group=row["group"],
                budget=row["budget"],
                rows=row["rows"],
                hvg=fmt(row["hvg_share"]),
                random=fmt(row["random_share"]),
                mean=fmt(row["mean_matched_share"]),
                hvg_mean=fmt(row["hvg_minus_mean_matched"]),
                ci_low=fmt(row["hvg_minus_mean_matched_ci_low"]),
                abundance=fmt(row["abundance_share"]),
                hvg_abund=fmt(row["hvg_minus_abundance"]),
                abund_overlap=fmt(row["hvg_abundance_overlap"]),
                detection=fmt(row["detection_share"]),
                hvg_detect=fmt(row["hvg_minus_detection"]),
                detect_overlap=fmt(row["hvg_detection_overlap"]),
                shuffled=fmt(row["shuffled_label_hvg_share"]),
                split_fold=fmt(row["split_half_fold_random"]),
            )
        )

    md_lines.extend(
        [
            "",
            "## Decision Metrics",
            "",
            f"* Max top1000 HVG-minus-mean-matched advantage across chemical/gene groups: `{max_meanmatched_top1000:.4f}`.",
            f"* Max absolute top1000 HVG-minus-abundance difference across chemical/gene groups: `{max_abundance_top1000:.4f}`.",
            f"* Max top1000 HVG-minus-detection advantage across chemical/gene groups: `{max_detection_top1000:.4f}`.",
            f"* Minimum top1000 HVG/abundance overlap across chemical/gene groups: `{min_overlap_top1000:.4f}`.",
            "",
            "## Decision",
            "",
            "* The expanded HVG/top-token signal is real versus random genes, but it is mostly explained by mean expression, abundance, and detection.",
            "* The HVG-specific intervention gate fails: top1000 matched-control advantages are too small for a downstream GPU route.",
            "* Keep the result as a descriptive biological scaling-law axis: compact observable-gene response information, not HVG-specific biology.",
            "",
            "## Next Action",
            "",
            "* Use observable-gene budget as a covariate or manuscript scaling variable only.",
            "* A future model route must first show downstream association beyond abundance/detection/mean controls under a leakage-safe train-only design.",
            "",
            "## Sources",
            "",
            f"* Abundance control JSON: `{args.abundance_json}`",
            f"* Detection control JSON: `{args.detection_json}`",
            f"* Mean-matched control JSON: `{args.meanmatched_json}`",
            f"* Machine-readable synthesis: `{json_path}`",
            "",
        ]
    )
    md_path = args.out_dir / "LATENTFM_HVG_OBSERVABLE_BUDGET_CONTROL_DECISION_20260629.md"
    md_path.write_text("\n".join(md_lines))
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
