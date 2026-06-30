#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT_METRICS = ROOT / "reports/downstream_information_scaling_preflight_20260628/split_information_metrics.csv"
ASSOCIATION_ROWS = ROOT / "reports/downstream_information_association_gate_20260628/association_rows.csv"
JOIN_ROWS = ROOT / "reports/downstream_information_association_gate_20260628/information_outcome_join_rows.csv"
METADATA_INVENTORY = ROOT / "reports/biological_information_metadata_preflight_20260628/h5ad_obs_metadata_inventory.csv"
OUT_DIR = ROOT / "reports/bioinfo_split_design_gate_20260628"
REPORT = ROOT / "reports/LATENTFM_BIOINFO_SPLIT_DESIGN_GATE_20260628.md"
JSON_PATH = ROOT / "reports/latentfm_bioinfo_split_design_gate_20260628.json"

BIO_AXES = [
    "n_target_genes",
    "target_gene_entropy_norm",
    "target_gene_effective_count",
]

MIXED_AXES = [
    "n_background_labels",
    "background_entropy_norm",
    "background_effective_count",
    "perturbation_type_entropy_norm",
    "perturbation_type_effective_count",
    "drug_condition_fraction",
    "gene_condition_fraction",
]

CONFOUNDS = [
    "n_train_conditions",
    "n_dataset_labels",
    "dataset_effective_count",
    "background_effective_count",
    "perturbation_type_effective_count",
    "drug_condition_fraction",
    "dataset_mean_pairwise_l2",
]


def now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def stdev(values: list[float]) -> float:
    vals = finite(values)
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var)


def z_abs_diff(a: float, b: float, scale: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    if scale <= 0:
        return 0.0 if a == b else float("inf")
    return abs(a - b) / scale


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_rows = read_rows(SPLIT_METRICS)
    assoc_rows = read_rows(ASSOCIATION_ROWS)
    join_rows = read_rows(JOIN_ROWS)
    metadata_rows = read_rows(METADATA_INVENTORY)

    scales = {}
    for key in BIO_AXES + MIXED_AXES + CONFOUNDS:
        scales[key] = stdev([f(row, key) for row in split_rows])

    pair_rows = []
    for axis in BIO_AXES:
        axis_scale = scales[axis]
        for left, right in itertools.combinations(split_rows, 2):
            axis_delta_z = z_abs_diff(f(left, axis), f(right, axis), axis_scale)
            confound_diffs = {
                key: z_abs_diff(f(left, key), f(right, key), scales[key])
                for key in CONFOUNDS
            }
            finite_confound = [v for v in confound_diffs.values() if math.isfinite(v)]
            max_confound_z = max(finite_confound) if finite_confound else float("inf")
            mean_confound_z = sum(finite_confound) / len(finite_confound) if finite_confound else float("inf")
            candidate = axis_delta_z >= 0.5 and max_confound_z <= 0.1
            pair_rows.append(
                {
                    "axis": axis,
                    "left_split": left["split_name"],
                    "right_split": right["split_name"],
                    "axis_delta_z": axis_delta_z,
                    "max_confound_z": max_confound_z,
                    "mean_confound_z": mean_confound_z,
                    "candidate_equal_cell_different_info": candidate,
                    **{f"{key}_z": value for key, value in confound_diffs.items()},
                }
            )
    pair_rows.sort(
        key=lambda r: (
            r["candidate_equal_cell_different_info"],
            r["axis_delta_z"],
            -r["max_confound_z"],
        ),
        reverse=True,
    )

    assoc_bio = [r for r in assoc_rows if r.get("metric_tag") == "biological"]
    assoc_mixed = [r for r in assoc_rows if r.get("metric_tag") == "mixed"]
    bio_gate_signals = [r for r in assoc_bio if r.get("gate_signal") == "True"]
    mixed_gate_signals = [r for r in assoc_mixed if r.get("gate_signal") == "True"]

    files_with_celltype = [
        r for r in metadata_rows
        if int(float(r.get("cell_type_n_columns") or 0)) > 0
    ]
    files_with_informative_celltype = [
        r for r in metadata_rows
        if int(float(r.get("cell_type_first_unique") or 0)) > 1
    ]
    files_with_subcluster = [
        r for r in metadata_rows
        if int(float(r.get("subcluster_n_columns") or 0)) > 0
    ]
    files_with_pathway = [
        r for r in metadata_rows
        if int(float(r.get("pathway_n_columns") or 0)) > 0
    ]
    files_with_dose_time = [
        r for r in metadata_rows
        if int(float(r.get("dose_time_n_columns") or 0)) > 0
    ]

    candidate_pairs = [r for r in pair_rows if r["candidate_equal_cell_different_info"]]
    candidate_axes = sorted({r["axis"] for r in candidate_pairs})
    status = "bioinfo_split_design_gate_pass_gpu_design_allowed"
    fail_reasons = []
    if len(bio_gate_signals) == 0:
        fail_reasons.append("no biological association predictor passed all controls")
    if len(candidate_axes) < 2:
        fail_reasons.append("fewer than two biological axes have matched equal-cell/different-info split candidates")
    if len(files_with_subcluster) == 0:
        fail_reasons.append("current h5ad inventory has no subcluster/state metadata columns")
    if len(files_with_informative_celltype) < 2:
        fail_reasons.append("too few current h5ad files have informative multi-cell-type labels")
    if fail_reasons:
        status = "bioinfo_split_design_gate_fail_tracka_only_no_gpu"

    pair_csv = OUT_DIR / "matched_split_pair_candidates.csv"
    write_csv(
        pair_csv,
        pair_rows,
        [
            "axis",
            "left_split",
            "right_split",
            "axis_delta_z",
            "max_confound_z",
            "mean_confound_z",
            "candidate_equal_cell_different_info",
            *[f"{key}_z" for key in CONFOUNDS],
        ],
    )

    payload = {
        "timestamp_utc": now_utc(),
        "status": status,
        "gpu_authorized": status.endswith("allowed"),
        "fail_reasons": fail_reasons,
        "n_split_rows": len(split_rows),
        "n_outcome_join_rows": len(join_rows),
        "n_association_rows": len(assoc_rows),
        "n_biological_association_rows": len(assoc_bio),
        "n_biological_gate_signals": len(bio_gate_signals),
        "n_mixed_gate_signals": len(mixed_gate_signals),
        "mixed_gate_signals": mixed_gate_signals,
        "n_candidate_pair_rows": len(candidate_pairs),
        "candidate_axes": candidate_axes,
        "top_pair_rows": pair_rows[:20],
        "metadata_inventory": {
            "n_h5ad_files": len(metadata_rows),
            "files_with_celltype_columns": len(files_with_celltype),
            "files_with_informative_celltype": len(files_with_informative_celltype),
            "files_with_subcluster_columns": len(files_with_subcluster),
            "files_with_pathway_columns": len(files_with_pathway),
            "files_with_dose_time_columns": len(files_with_dose_time),
        },
        "outputs": {
            "pair_candidates_csv": str(pair_csv),
            "report": str(REPORT),
        },
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Bioinfo Split Design Gate",
        "",
        f"Timestamp: `{payload['timestamp_utc']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only split-design gate over completed Track A/scaling artifacts.",
        "- No training, inference, canonical multi, Track C held-out query, checkpoint selection, or GPU.",
        "- Tests whether current assets can support biological, equal-cell/different-information split designs.",
        "",
        "## Gate Summary",
        "",
        f"- split rows audited: `{len(split_rows)}`",
        f"- outcome join rows: `{len(join_rows)}`",
        f"- biological association rows: `{len(assoc_bio)}`",
        f"- biological gate signals: `{len(bio_gate_signals)}`",
        f"- mixed gate signals: `{len(mixed_gate_signals)}`",
        f"- matched biological split-pair candidates: `{len(candidate_pairs)}`",
        f"- candidate biological axes: `{candidate_axes}`",
        "",
        "## Metadata Support",
        "",
        f"- h5ad files audited: `{len(metadata_rows)}`",
        f"- files with any cell-type-like columns: `{len(files_with_celltype)}`",
        f"- files with informative multi-cell-type labels: `{len(files_with_informative_celltype)}`",
        f"- files with subcluster/state columns: `{len(files_with_subcluster)}`",
        f"- files with pathway columns: `{len(files_with_pathway)}`",
        f"- files with dose/time columns: `{len(files_with_dose_time)}`",
        "",
        "## Mixed Signals Not Accepted As Biological Claims",
        "",
        "| outcome | predictor | rho | p | partial |",
        "|---|---|---:|---:|---:|",
    ]
    for row in mixed_gate_signals:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("outcome", ""),
                    row.get("predictor", ""),
                    row.get("spearman_rho", ""),
                    row.get("spearman_perm_p", ""),
                    row.get("partial_corr", ""),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Top Matched Pair Attempts",
            "",
            "| axis | left | right | axis_delta_z | max_confound_z | candidate |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in pair_rows[:10]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["axis"],
                    row["left_split"],
                    row["right_split"],
                    f"{row['axis_delta_z']:.4f}" if math.isfinite(row["axis_delta_z"]) else "nan",
                    f"{row['max_confound_z']:.4f}" if math.isfinite(row["max_confound_z"]) else "nan",
                    str(row["candidate_equal_cell_different_info"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if fail_reasons:
        lines.append("Fail-close Track A-only biological scaling as an immediate GPU route.")
        lines.append("")
        lines.append("Reasons:")
        for reason in fail_reasons:
            lines.append(f"- {reason}.")
        lines.extend(
            [
                "",
                "Proceed with ZSCAPE metadata coverage and later continuity/OT gates for",
                "cell-type/state/trajectory biological axes. Current Track A metrics remain",
                "useful controls and failure-map evidence, but not a Nature Methods-level",
                "biological scaling law.",
            ]
        )
    else:
        lines.append(
            "A bounded GPU split-design smoke may be specified, but only after writing a",
            "dual-baseline promotion/fail-close protocol."
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- pair candidates: `{pair_csv}`",
            f"- JSON: `{JSON_PATH}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)
    print(JSON_PATH)
    print(pair_csv)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
