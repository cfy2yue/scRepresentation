#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def is_control(target: str) -> bool:
    return target.lower().startswith("ctrl")


def f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuity-json", required=True)
    parser.add_argument("--response-rows", required=True)
    parser.add_argument("--target-time-broad-coverage", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-perturb-cells", type=int, default=100)
    parser.add_argument("--min-control-cells", type=int, default=500)
    parser.add_argument("--min-perturb-embryos", type=int, default=10)
    parser.add_argument("--min-control-embryos", type=int, default=30)
    parser.add_argument("--max-per-cell-type", type=int, default=5)
    parser.add_argument("--min-lineages", type=int, default=2)
    args = parser.parse_args()

    continuity = json.loads(Path(args.continuity_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = read_csv(Path(args.target_time_broad_coverage))

    selected = set(continuity.get("selected_cell_types", []))
    full_response_rows = read_csv(Path(args.response_rows))
    response_rows = [
        row for row in full_response_rows
        if row.get("coord_space") == "umap3d"
        and row.get("cell_type_broad") in selected
        and f(row, "cells") >= args.min_perturb_cells
    ]

    control_cov: dict[tuple[str, float], dict[str, float]] = defaultdict(lambda: {"cells": 0.0, "n_embryos": 0.0, "n_samples": 0.0})
    perturb_cov: dict[tuple[str, str, float], dict[str, float]] = {}
    for row in coverage_rows:
        cell_type = row.get("cell_type_broad", "")
        if cell_type not in selected:
            continue
        target = row.get("gene_target", "")
        timepoint = f(row, "timepoint")
        key_control = (cell_type, timepoint)
        if is_control(target):
            control_cov[key_control]["cells"] += f(row, "cells")
            control_cov[key_control]["n_embryos"] += f(row, "n_embryos")
            control_cov[key_control]["n_samples"] += f(row, "n_samples")
        else:
            perturb_cov[(cell_type, target, timepoint)] = {
                "cells": f(row, "cells"),
                "n_embryos": f(row, "n_embryos"),
                "n_samples": f(row, "n_samples"),
            }

    eligible = []
    for row in response_rows:
        cell_type = row["cell_type_broad"]
        target = row["gene_target"]
        timepoint = f(row, "timepoint")
        p_cov = perturb_cov.get((cell_type, target, timepoint), {})
        c_cov = control_cov.get((cell_type, timepoint), {})
        if p_cov.get("cells", 0) < args.min_perturb_cells:
            continue
        if p_cov.get("n_embryos", 0) < args.min_perturb_embryos:
            continue
        if c_cov.get("cells", 0) < args.min_control_cells:
            continue
        if c_cov.get("n_embryos", 0) < args.min_control_embryos:
            continue
        eligible.append(
            {
                "cell_type_broad": cell_type,
                "gene_target": target,
                "timepoint": timepoint,
                "metadata_coord_space": row["coord_space"],
                "metadata_response_distance": f(row, "response_distance_to_matched_control"),
                "perturb_cells": int(p_cov["cells"]),
                "perturb_embryos": int(p_cov["n_embryos"]),
                "control_pool": "all_ctrl_targets_same_cell_type_timepoint",
                "control_cells": int(c_cov["cells"]),
                "control_embryos": int(c_cov["n_embryos"]),
                "role": "candidate_expression_subset_ot_pair",
            }
        )

    eligible.sort(key=lambda r: r["metadata_response_distance"], reverse=True)
    counts = Counter()
    manifest_rows = []
    for row in eligible:
        if counts[row["cell_type_broad"]] >= args.max_per_cell_type:
            continue
        manifest_rows.append(row)
        counts[row["cell_type_broad"]] += 1

    selected_lineages = sorted({row["cell_type_broad"] for row in manifest_rows})
    selected_targets = sorted({row["gene_target"] for row in manifest_rows})
    status = "zscape_expression_subset_ot_manifest_ready_no_gpu"
    fail_reasons = []
    if len(selected_lineages) < args.min_lineages:
        fail_reasons.append("too few lineages after min-cell/control filtering")
    if len(manifest_rows) < args.min_lineages * 3:
        fail_reasons.append("too few target-time rows after filtering")
    if fail_reasons:
        status = "zscape_expression_subset_ot_manifest_fail_no_gpu"

    manifest_csv = out_dir / "zscape_expression_subset_ot_manifest.csv"
    with manifest_csv.open("w", newline="") as handle:
        fieldnames = [
            "role",
            "cell_type_broad",
            "gene_target",
            "timepoint",
            "metadata_coord_space",
            "metadata_response_distance",
            "perturb_cells",
            "perturb_embryos",
            "control_pool",
            "control_cells",
            "control_embryos",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    payload = {
        "timestamp_utc": now_utc(),
        "status": status,
        "gpu_authorized": False,
        "expression_download_authorized": False,
        "fail_reasons": fail_reasons,
        "filters": {
            "min_perturb_cells": args.min_perturb_cells,
            "min_control_cells": args.min_control_cells,
            "min_perturb_embryos": args.min_perturb_embryos,
            "min_control_embryos": args.min_control_embryos,
            "max_per_cell_type": args.max_per_cell_type,
        },
        "response_rows_csv": args.response_rows,
        "n_manifest_rows": len(manifest_rows),
        "selected_lineages": selected_lineages,
        "selected_targets": selected_targets,
        "manifest_rows": manifest_rows,
        "outputs": {
            "manifest_csv": str(manifest_csv),
        },
    }
    json_path = out_dir / "zscape_expression_subset_ot_manifest.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = out_dir / "LATENTFM_ZSCAPE_EXPRESSION_SUBSET_OT_MANIFEST_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Expression-Subset OT Manifest",
        "",
        f"Timestamp: `{payload['timestamp_utc']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "Expression download authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only manifest planning from ZSCAPE metadata coverage and coordinate continuity outputs.",
        "- Does not download expression matrices/CDS/raw counts.",
        "- Does not train, infer, embed, use canonical multi, or use Track C query.",
        "",
        "## Filters",
        "",
        f"- min perturb cells: `{args.min_perturb_cells}`",
        f"- min control cells: `{args.min_control_cells}`",
        f"- min perturb embryos: `{args.min_perturb_embryos}`",
        f"- min control embryos: `{args.min_control_embryos}`",
        f"- max rows per cell type: `{args.max_per_cell_type}`",
        "",
        "## Manifest Summary",
        "",
        f"- rows: `{len(manifest_rows)}`",
        f"- selected lineages: `{selected_lineages}`",
        f"- selected targets: `{selected_targets}`",
        "",
        "## Candidate Rows",
        "",
        "| cell_type_broad | gene_target | timepoint | response_dist | perturb_cells | perturb_embryos | control_cells | control_embryos |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifest_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["cell_type_broad"],
                    row["gene_target"],
                    str(row["timepoint"]),
                    f"{row['metadata_response_distance']:.4f}",
                    str(row["perturb_cells"]),
                    str(row["perturb_embryos"]),
                    str(row["control_cells"]),
                    str(row["control_embryos"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Decision", ""])
    if fail_reasons:
        lines.append("Do not proceed to expression subset download.")
        lines.append("")
        lines.append("Reasons:")
        for reason in fail_reasons:
            lines.append(f"- {reason}.")
    else:
        lines.append(
            "This manifest is ready for an external/code audit before any expression subset download or OT computation. It does not itself authorize downloading expression matrices."
        )
    lines.extend(["", "## Output Files", "", f"- manifest CSV: `{manifest_csv}`", f"- JSON: `{json_path}`"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    print(json_path)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
