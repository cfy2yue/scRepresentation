#!/usr/bin/env python3
"""Post-audit CPU gate for nonstatic observable-information axes.

This script only reads completed CPU/GPU-decision artifacts and writes a
synthesis report. It does not train, infer, select checkpoints, read Track C
query, use canonical multi for selection, or use GPU.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports" / "nonstatic_observable_information_gate_20260630"
REPORT = OUT_DIR / "LATENTFM_NONSTATIC_OBSERVABLE_INFORMATION_GATE_20260630.md"
JSON_OUT = OUT_DIR / "nonstatic_observable_information_gate_20260630.json"
AXIS_CSV = OUT_DIR / "nonstatic_observable_axis_admission_20260630.csv"

SUMMARY_CSV = ROOT / "reports/scaling_v2_residualized_condition_axes_gate_20260628/residualized_condition_axis_summary.csv"
PAIRS_CSV = ROOT / "reports/scaling_v2_residualized_condition_axes_gate_20260628/residualized_condition_axis_pairs.csv"
MATCHED_INFO_SUMMARY = ROOT / "reports/scaling_v2_matched_information_gate_20260628/scaling_v2_matched_information_summary.csv"
HIGHLOW_DECISION = ROOT / "reports/scaling_v2_condition_information_highlow_smoke_20260628/LATENTFM_SCALING_V2_CONDITION_INFORMATION_HIGHLOW_DECISION_20260628.md"
POST_AUDIT_SCALING = ROOT / "reports/post_audit_scaling_law_decision_20260630/LATENTFM_POST_AUDIT_SCALING_LAW_DECISION_20260630.md"


STRICT_MIN_PAIRS = 200
STRICT_MIN_DATASETS = 12
STRICT_MAX_CONFOUND_DISTANCE = 0.50
STRICT_MAX_SMD = 0.25
NEAR_MIN_PAIRS = 150
NEAR_MAX_CONFOUND_DISTANCE = 0.50
MAX_DATASET_FRACTION = 0.25
MAX_TYPE_FRACTION = 0.70


@dataclass(frozen=True)
class AxisDecision:
    axis: str
    match_mode: str
    n_pairs: int
    n_datasets: int
    mean_axis_delta: float
    mean_confound_distance: float
    max_abs_confound_smd: float
    max_dataset_fraction: float
    top_dataset: str
    max_perturbation_type_fraction: float
    top_perturbation_type: str
    strict_admission: bool
    near_miss: bool
    status: str
    reasons: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    if value.lower() in {"nan", "none", "na"}:
        return default
    return float(value)


def as_int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def require_inputs() -> None:
    missing = [
        str(path)
        for path in [SUMMARY_CSV, PAIRS_CSV, MATCHED_INFO_SUMMARY, HIGHLOW_DECISION, POST_AUDIT_SCALING]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required input files: " + "; ".join(missing))


def pair_dominance() -> dict[tuple[str, str], dict[str, object]]:
    rows = read_csv(PAIRS_CSV)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["axis"], row["match_mode"])].append(row)

    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, group in grouped.items():
        n = len(group)
        dataset_counts = Counter(row["dataset"] for row in group)
        type_counts = Counter(row["perturbation_type"] for row in group)
        top_dataset, top_dataset_count = dataset_counts.most_common(1)[0]
        top_type, top_type_count = type_counts.most_common(1)[0]
        out[key] = {
            "max_dataset_fraction": top_dataset_count / n if n else 0.0,
            "top_dataset": top_dataset,
            "max_perturbation_type_fraction": top_type_count / n if n else 0.0,
            "top_perturbation_type": top_type,
        }
    return out


def highlow_failure_reasons() -> list[str]:
    text = HIGHLOW_DECISION.read_text(encoding="utf-8", errors="replace")
    reasons = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- reason:"):
            reasons.append(stripped.split(":", 1)[1].strip().strip("`"))
    return reasons


def matched_info_max_pairs_and_signals() -> tuple[int, int]:
    rows = read_csv(MATCHED_INFO_SUMMARY)
    max_pairs = 0
    signals = 0
    for row in rows:
        max_pairs = max(max_pairs, as_int(row.get("n_pairs")))
        if str(row.get("gate_signal", "")).lower() == "true":
            signals += 1
    return max_pairs, signals


def decide_axes() -> list[AxisDecision]:
    summary_rows = read_csv(SUMMARY_CSV)
    dominance = pair_dominance()
    decisions: list[AxisDecision] = []

    for row in summary_rows:
        axis = row["axis"]
        match_mode = row["match_mode"]
        n_pairs = as_int(row.get("n_pairs"))
        n_datasets = as_int(row.get("n_datasets"))
        mean_axis_delta = as_float(row.get("mean_axis_delta"))
        mean_conf = as_float(row.get("mean_confound_distance"))
        max_smd = as_float(row.get("max_abs_confound_smd"))
        dom = dominance.get(
            (axis, match_mode),
            {
                "max_dataset_fraction": 1.0,
                "top_dataset": "NA",
                "max_perturbation_type_fraction": 1.0,
                "top_perturbation_type": "NA",
            },
        )
        max_dataset_frac = float(dom["max_dataset_fraction"])
        max_type_frac = float(dom["max_perturbation_type_fraction"])

        reasons: list[str] = []
        if n_pairs < STRICT_MIN_PAIRS:
            reasons.append(f"pairs_below_{STRICT_MIN_PAIRS}")
        if n_datasets < STRICT_MIN_DATASETS:
            reasons.append(f"datasets_below_{STRICT_MIN_DATASETS}")
        if mean_conf > STRICT_MAX_CONFOUND_DISTANCE:
            reasons.append(f"confound_distance_gt_{STRICT_MAX_CONFOUND_DISTANCE}")
        if max_smd > STRICT_MAX_SMD:
            reasons.append(f"max_smd_gt_{STRICT_MAX_SMD}")
        if max_dataset_frac > MAX_DATASET_FRACTION:
            reasons.append(f"dataset_dominance_gt_{MAX_DATASET_FRACTION}")
        if max_type_frac > MAX_TYPE_FRACTION:
            reasons.append(f"perturbation_type_dominance_gt_{MAX_TYPE_FRACTION}")

        strict = not reasons
        near = (
            not strict
            and n_pairs >= NEAR_MIN_PAIRS
            and n_datasets >= STRICT_MIN_DATASETS
            and mean_conf <= NEAR_MAX_CONFOUND_DISTANCE
            and max_smd <= STRICT_MAX_SMD
            and max_dataset_frac <= MAX_DATASET_FRACTION
        )
        if strict:
            status = "strict_cpu_admission_pass_external_review_before_gpu"
        elif near:
            status = "near_miss_cpu_design_only_no_gpu"
        elif n_pairs >= STRICT_MIN_PAIRS and mean_conf > STRICT_MAX_CONFOUND_DISTANCE:
            status = "count_only_confounded_no_gpu"
        else:
            status = "fail_no_gpu"

        decisions.append(
            AxisDecision(
                axis=axis,
                match_mode=match_mode,
                n_pairs=n_pairs,
                n_datasets=n_datasets,
                mean_axis_delta=mean_axis_delta,
                mean_confound_distance=mean_conf,
                max_abs_confound_smd=max_smd,
                max_dataset_fraction=max_dataset_frac,
                top_dataset=str(dom["top_dataset"]),
                max_perturbation_type_fraction=max_type_frac,
                top_perturbation_type=str(dom["top_perturbation_type"]),
                strict_admission=strict,
                near_miss=near,
                status=status,
                reasons=";".join(reasons) if reasons else "none",
            )
        )
    return decisions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def main() -> None:
    require_inputs()
    if REPORT.exists() or JSON_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing outputs under {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    axis_decisions = decide_axes()
    axis_rows = [asdict(row) for row in axis_decisions]
    strict_passes = [row for row in axis_decisions if row.strict_admission]
    near_misses = [row for row in axis_decisions if row.near_miss]
    max_split_pairs, matched_info_signals = matched_info_max_pairs_and_signals()
    highlow_reasons = highlow_failure_reasons()

    gpu_authorized = bool(strict_passes) and matched_info_signals > 0 and not highlow_reasons
    status = (
        "nonstatic_observable_information_strict_pass_review_needed"
        if gpu_authorized
        else "nonstatic_observable_information_no_gpu_near_miss_only"
        if near_misses
        else "nonstatic_observable_information_all_fail_no_gpu"
    )

    payload = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "status": status,
        "gpu_authorized": gpu_authorized,
        "strict_axis_pass_count": len(strict_passes),
        "near_miss_count": len(near_misses),
        "matched_split_max_pairs": max_split_pairs,
        "matched_split_gate_signals": matched_info_signals,
        "closed_highlow_reasons": highlow_reasons,
        "axis_decisions": axis_rows,
        "decision": {
            "default_model": "xverse_8k_anchor",
            "next_local_cpu_gate": "hvg_advantage_resid_v3_pair_pool_or_external_small_table",
            "gpu_route": "blocked",
        },
    }

    write_csv(AXIS_CSV, axis_rows)
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    near_rows = [asdict(row) for row in near_misses]
    if not near_rows:
        near_text = "No near-miss axes under the relaxed CPU-design-only criteria."
    else:
        near_text = markdown_table(
            near_rows,
            [
                "axis",
                "match_mode",
                "n_pairs",
                "n_datasets",
                "mean_confound_distance",
                "max_abs_confound_smd",
                "max_dataset_fraction",
                "top_dataset",
                "status",
                "reasons",
            ],
        )

    report = f"""# LatentFM Nonstatic Observable-Information Gate 20260630

Created: `{payload["created"]}`

Status: `{status}`

GPU authorized: `{gpu_authorized}`

## Boundary

- CPU/report-only post-audit gate.
- Inputs are completed residualized condition-axis, matched-information, and
  condition-information high/low decision artifacts.
- No training, inference, checkpoint selection, canonical multi selection,
  Track C query, or GPU.

## Bottom Line

- Strict nonstatic observable-information CPU admissions: `{len(strict_passes)}`.
- Near-miss CPU-design-only axes: `{len(near_misses)}`.
- Split-level matched-information gate signals: `{matched_info_signals}`;
  max split-level matched pairs: `{max_split_pairs}`.
- The previous `info_composite` high/low smoke is closed with reasons:
  `{'; '.join(highlow_reasons) if highlow_reasons else 'none'}`.

Therefore no LatentFM GPU route is authorized from current nonstatic
observable-information artifacts.

## Axis Admission Table

{markdown_table(axis_rows, ["axis", "match_mode", "n_pairs", "n_datasets", "mean_confound_distance", "max_abs_confound_smd", "max_dataset_fraction", "top_dataset", "max_perturbation_type_fraction", "top_perturbation_type", "status", "reasons"])}

## Near-Miss Rows

{near_text}

## Interpretation

- `hvg_advantage_resid` is the only meaningful near miss: it is nonstatic and
  reasonably controlled in `moderate` mode, but it has only `156` matched pairs,
  below the strict `200`-pair admission threshold.
- Count-only relaxed rows for `hvg_advantage_resid` and `support_resid` reach
  pair counts only by loosening confound distance, so they are not launch-ready.
- Split-level scaling axes have at most `4` matched pairs and no signal, so they
  are manuscript/failure-map descriptors, not training-selection variables.
- The closed high/low `info_composite` smoke showed high worse than low and
  failed no-harm, so rerunning that family is not a legal GPU refill.

## Next CPU Gate

If continuing locally, the only non-duplicative CPU gate is a v3 pair-pool
attempt centered on `hvg_advantage_resid`:

- input matrix:
  `/data/cyx/1030/scLatent/reports/scaling_v2_residualized_condition_axes_gate_20260628/residualized_condition_axis_matrix.csv`;
- input pairs:
  `/data/cyx/1030/scLatent/reports/scaling_v2_residualized_condition_axes_gate_20260628/residualized_condition_axis_pairs.csv`;
- required pass:
  `>=200` high/low pairs, `>=12` datasets, mean confound distance `<=0.50`,
  max absolute confound SMD `<=0.25`, max dataset fraction `<=0.25`, no single
  perturbation type carrying the result, and a placebo/random matched split
  that collapses;
- fail-close:
  if the extra pairs require relaxed confound distance, source/type dominance,
  or a reuse of the closed `info_composite` split, close local observable-x as
  descriptor-only and require a new external small table.

Passing this CPU gate would still only authorize external review and a bounded
launcher/no-harm protocol, not model promotion.

## Outputs

- JSON: `{JSON_OUT}`
- axis rows: `{AXIS_CSV}`
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": gpu_authorized, "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
