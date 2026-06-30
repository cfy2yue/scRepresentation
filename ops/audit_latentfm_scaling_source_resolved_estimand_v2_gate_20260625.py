#!/usr/bin/env python3
"""Source-resolved matched-estimand v2 gate for LatentFM scaling.

CPU-only. Uses the frozen S0 provenance table plus completed train-only
condition-count LODO summaries to test whether a source-verified, background
/ perturbation-type resolved subset can reopen a non-duplicate GPU scaling
candidate. It does not read checkpoints, canonical multi, Track C query, train,
infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
S0_TSV = REPORTS / "latentfm_scaling_s0_provenance_freeze_20260625.tsv"
MIXED_JSON = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"
OUT_JSON = REPORTS / "latentfm_scaling_source_resolved_estimand_v2_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_SOURCE_RESOLVED_ESTIMAND_V2_GATE_20260625.md"

N_BOOT = 5000
SEED = 20260625


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": True, "_path": str(path)}
    with path.open() as f:
        return json.load(f)


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> str:
    val = as_float(value)
    if val is not None:
        return f"{val:+.6f}"
    return "NA" if value is None else str(value)


def load_s0_by_dataset() -> dict[str, dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    if not S0_TSV.is_file():
        return by_dataset
    accum: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "source_verified_rows": 0,
            "resolved_rows": 0,
            "backgrounds": Counter(),
            "types": Counter(),
            "modalities": Counter(),
        }
    )
    with S0_TSV.open(newline="") as f:
        for row in csv.DictReader(f, dialect="excel-tab"):
            ds = row.get("dataset") or ""
            if not ds:
                continue
            rec = accum[ds]
            rec["rows"] += 1
            if row.get("source_quality") == "source_verified":
                rec["source_verified_rows"] += 1
            if row.get("scaling_claim_inclusion") == "s0_resolved_for_gene_or_nonchemical_axes":
                rec["resolved_rows"] += 1
            if row.get("cell_background_source"):
                rec["backgrounds"][row["cell_background_source"]] += 1
            if row.get("perturbation_type"):
                rec["types"][row["perturbation_type"]] += 1
            if row.get("modality"):
                rec["modalities"][row["modality"]] += 1
    for ds, rec in accum.items():
        by_dataset[ds] = {
            "rows": rec["rows"],
            "source_verified_rows": rec["source_verified_rows"],
            "resolved_rows": rec["resolved_rows"],
            "primary_background": rec["backgrounds"].most_common(1)[0][0] if rec["backgrounds"] else "",
            "primary_type": rec["types"].most_common(1)[0][0] if rec["types"] else "",
            "backgrounds": dict(rec["backgrounds"]),
            "types": dict(rec["types"]),
            "modalities": dict(rec["modalities"]),
        }
    return by_dataset


def weighted(rows: list[dict[str, Any]], key: str) -> float | None:
    num = 0.0
    den = 0.0
    for row in rows:
        val = as_float(row.get(key))
        n = as_float(row.get("n")) or 0.0
        if val is not None and n > 0:
            num += n * val
            den += n
    return None if den <= 0 else num / den


def total_n(rows: list[dict[str, Any]]) -> int:
    return int(sum(int(row.get("n") or 0) for row in rows))


def bootstrap_ci(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rng = random.Random(SEED)
    vals = []
    if not rows:
        return {"n_boot": 0, "ci95": None, "p_le_zero": None}
    for _ in range(N_BOOT):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        val = weighted(sample, key)
        if val is not None:
            vals.append(val)
    vals.sort()
    if not vals:
        return {"n_boot": 0, "ci95": None, "p_le_zero": None}
    return {
        "n_boot": len(vals),
        "ci95": [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]],
        "p_le_zero": sum(1 for v in vals if v <= 0.0) / len(vals),
    }


def group_summary(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    out = []
    for val in sorted({str(row.get(field) or "") for row in rows}):
        group = [row for row in rows if str(row.get(field) or "") == val]
        out.append(
            {
                field: val,
                "datasets": [row["dataset"] for row in group],
                "dataset_count": len(group),
                "n": total_n(group),
                "pp_delta_mean": weighted(group, "pp_delta_mean"),
                "mmd_delta_mean": weighted(group, "mmd_delta_mean"),
            }
        )
    return out


def main() -> int:
    s0_by_dataset = load_s0_by_dataset()
    mixed = load_json(MIXED_JSON)
    merged = []
    for row in mixed.get("dataset_rows", []):
        ds = str(row.get("dataset") or "")
        s0 = s0_by_dataset.get(ds, {})
        rec = dict(row)
        rec.update(
            {
                "s0_rows": s0.get("rows", 0),
                "s0_source_verified_rows": s0.get("source_verified_rows", 0),
                "s0_resolved_rows": s0.get("resolved_rows", 0),
                "s0_primary_background": s0.get("primary_background", ""),
                "s0_primary_type": s0.get("primary_type", ""),
            }
        )
        merged.append(rec)

    primary = [
        row
        for row in merged
        if int(row.get("cap_gain") or 0) > 0
        and int(row.get("s0_source_verified_rows") or 0) > 0
        and int(row.get("s0_resolved_rows") or 0) > 0
    ]
    background_rows = group_summary(primary, "background")
    type_rows = group_summary(primary, "perturbation_type")
    pp_mean = weighted(primary, "pp_delta_mean")
    mmd_mean = weighted(primary, "mmd_delta_mean")
    ci = bootstrap_ci(primary, "pp_delta_mean")
    dataset_min = min((v for v in (as_float(r.get("pp_delta_mean")) for r in primary) if v is not None), default=None)
    negative_tails = sum(1 for r in primary if (as_float(r.get("pp_delta_mean")) or 0.0) < -0.02)
    max_dataset_weight = 0.0
    n_total = total_n(primary)
    if n_total > 0:
        max_dataset_weight = max((int(r.get("n") or 0) / n_total for r in primary), default=0.0)
    min_background = min(
        (v for v in (as_float(r.get("pp_delta_mean")) for r in background_rows) if v is not None),
        default=None,
    )
    min_type = min(
        (v for v in (as_float(r.get("pp_delta_mean")) for r in type_rows) if v is not None),
        default=None,
    )

    reasons: list[str] = []
    if len(primary) < 8:
        reasons.append("too_few_source_verified_resolved_cap_gain_datasets")
    if n_total < 80:
        reasons.append("too_few_condition_rows_after_source_resolved_filter")
    if pp_mean is None or pp_mean < 0.015:
        reasons.append("source_resolved_pp_mean_lt_0p015")
    if not ci.get("ci95") or float(ci["ci95"][0]) <= 0.0:
        reasons.append("bootstrap_ci_lower_not_positive")
    if dataset_min is None or dataset_min < -0.02:
        reasons.append("dataset_tail_below_minus_0p020")
    if negative_tails:
        reasons.append("negative_dataset_tails_present")
    if len(background_rows) < 2:
        reasons.append("background_diversity_lt_2")
    if len(type_rows) < 2:
        reasons.append("perturbation_type_diversity_lt_2")
    if min_background is None or min_background < 0.0:
        reasons.append("background_stratum_min_not_positive")
    if min_type is None or min_type < 0.0:
        reasons.append("type_stratum_min_not_positive")
    if max_dataset_weight > 0.35:
        reasons.append("max_dataset_weight_gt_0p35")
    if mmd_mean is None or mmd_mean > 0.001:
        reasons.append("mmd_mean_gt_0p001")

    status = (
        "scaling_source_resolved_estimand_v2_pass_external_review_before_gpu"
        if not reasons
        else "scaling_source_resolved_estimand_v2_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_s0_provenance": True,
            "reads_completed_trainonly_reports": True,
            "reads_checkpoints": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {"s0_tsv": str(S0_TSV), "mixed_lodo": str(MIXED_JSON)},
        "summary": {
            "merged_datasets": len(merged),
            "primary_datasets": len(primary),
            "primary_n": n_total,
            "pp_delta_mean": pp_mean,
            "mmd_delta_mean": mmd_mean,
            "bootstrap": ci,
            "dataset_min_pp": dataset_min,
            "negative_tails_lt_minus_0p020": negative_tails,
            "background_count": len(background_rows),
            "perturbation_type_count": len(type_rows),
            "min_background_pp": min_background,
            "min_type_pp": min_type,
            "max_dataset_weight": max_dataset_weight,
        },
        "background_rows": background_rows,
        "type_rows": type_rows,
        "primary_rows": primary,
        "reasons": reasons,
        "next_action": (
            "external_review_then_bounded_gpu_candidate"
            if not reasons
            else "do_not_launch_source_resolved_background_type_gpu_from_current_evidence"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Scaling Source-Resolved Estimand V2 Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate using S0 frozen provenance and completed train-only mixed/LODO summaries.",
        "- Does not read checkpoints, canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| merged datasets | `{len(merged)}` |",
        f"| source-resolved cap-gain datasets / rows | `{len(primary)}` / `{n_total}` |",
        f"| pp / MMD mean | `{fmt(pp_mean)}` / `{fmt(mmd_mean)}` |",
        f"| pp CI95 | `{ci.get('ci95')}` |",
        f"| dataset min pp / negative tails | `{fmt(dataset_min)}` / `{negative_tails}` |",
        f"| backgrounds / perturbation types | `{len(background_rows)}` / `{len(type_rows)}` |",
        f"| min background/type pp | `{fmt(min_background)}` / `{fmt(min_type)}` |",
        f"| max dataset weight | `{fmt(max_dataset_weight)}` |",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        f"- next action: `{payload['next_action']}`",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
