#!/usr/bin/env python3
"""Build a CPU-only LatentFM run-outcome mechanism panel.

The panel is intentionally conservative: it extracts comparable run-level
posthoc deltas from completed reports, assigns coarse mechanism families, and
uses already-closed/null-calibrated branches as negative controls. It does not
train, infer, select checkpoints, or read Track C held-out query data.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "latentfm_run_outcome_mechanism_panel_20260630"


JSON_GLOBS = [
    "latentfm_*decision*.json",
    "latentfm_*summary*.json",
    "smoke_*decision*.json",
    "scaling_v2_condition_information_highlow_smoke_20260628/*.json",
    "observable_information_condition_weight_smoke_20260629/*.json",
    "condition_neighborhood_response_resid_highlow_smoke_20260629/*.json",
    "condition_neighborhood_response_resid_placebo_controls_20260629/*.json",
    "condition_neighborhood_response_resid_null_variance_panel_20260629/*.json",
    "tracka_exact_tail_candidate_gate_20260627/*.json",
    "lincs_gse92742_train_gene_candidate_panel_20260627/*.json",
    "lincs_gse92742_train_gene_outcome_eval_20260627/*.json",
]


CLOSED_FAMILIES = {
    "observable_weight",
    "condition_information_highlow",
    "support_neighborhood",
    "support_placebo",
    "tail_geometry",
    "exact_coverage",
    "hvg_budget",
    "zscape_translation",
    "condition_delta_trust",
    "pairmode_assignment",
    "true_cell_count",
}


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def first_float(obj: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in obj:
            x = as_float(obj.get(key))
            if x is not None:
                return x
    return None


def axis_family(path: Path, row_name: str = "") -> str:
    text = f"{path.as_posix()} {row_name}".lower()
    if "condition_neighborhood_response_resid_pairshuffle" in text or "placebo" in text:
        return "support_placebo"
    if "condition_neighborhood" in text or "support_context" in text or "support_only" in text:
        return "support_neighborhood"
    if "observable_information" in text:
        return "observable_weight"
    if "scaling_v2_condition_information" in text:
        return "condition_information_highlow"
    if "scaling_count" in text or "cap120" in text or "cap60" in text or "count_smoke" in text:
        return "condition_count_scaling"
    if "true_cell" in text or "truecell" in text or "cell_count" in text:
        return "true_cell_count"
    if "ot_pairmode" in text or "hungarian" in text or "pairmode" in text:
        return "pairmode_assignment"
    if "trackc" in text or "multi_support" in text:
        return "trackc_multi"
    if "tail" in text:
        return "tail_geometry"
    if "exact" in text or "lincs_gse" in text:
        return "exact_or_train_gene"
    if "condition_delta" in text or "conddelta" in text or "prior_adapter" in text:
        return "condition_delta_trust"
    if "zscape" in text:
        return "zscape_translation"
    if "soft_exposure" in text or "visit_cap" in text or "sampling" in text:
        return "sampling_curriculum"
    if "type_adapter" in text or "strategy" in text:
        return "architecture_strategy"
    return "other"


def flatten_numbers(obj: Any, prefix: str = "", max_depth: int = 4) -> dict[str, float]:
    out: dict[str, float] = {}
    if max_depth < 0:
        return out
    if isinstance(obj, dict):
        for key, val in obj.items():
            new_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(val, (dict, list)):
                out.update(flatten_numbers(val, new_key, max_depth - 1))
            else:
                x = as_float(val)
                if x is not None:
                    out[new_key] = x
    elif isinstance(obj, list):
        for idx, val in enumerate(obj[:20]):
            out.update(flatten_numbers(val, f"{prefix}.{idx}" if prefix else str(idx), max_depth - 1))
    return out


def extract_rows_from_row_list(path: Path, obj: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_rows = obj.get("rows")
    if not isinstance(raw_rows, list):
        return rows
    for idx, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            continue
        cross = first_float(row, ["cross_pp_delta", "crossbg_pp_delta", "test_single_pp_delta"])
        family = first_float(row, ["family_pp_delta", "family_gene_pp_delta", "test_single_pp_delta"])
        mmd = first_float(row, ["family_mmd_delta", "test_mmd_delta", "mmd_delta"])
        if cross is None and family is None and mmd is None:
            continue
        name = str(row.get("run_name") or row.get("arm") or row.get("name") or f"row{idx}")
        rows.append(
            {
                "source_path": str(path),
                "record_type": "row",
                "record_name": name,
                "axis_family": axis_family(path, name),
                "status": str(row.get("status") or obj.get("status") or (obj.get("decision") or {}).get("status") or ""),
                "cross_pp_delta": cross,
                "family_pp_delta": family,
                "family_mmd_delta": mmd,
                "is_placebo_or_null": "placebo" in path.as_posix().lower() or "pairshuffle" in name.lower(),
            }
        )
    return rows


def extract_rows_from_summaries(path: Path, obj: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries = obj.get("summaries")
    if not isinstance(summaries, list):
        return rows
    by_metric: dict[str, dict[str, Any]] = {}
    for item in summaries:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "")
        by_metric[metric] = item
    if not by_metric:
        return rows
    pp = by_metric.get("pearson_pert") or by_metric.get("pp") or {}
    mmd = by_metric.get("test_mmd_clamped") or by_metric.get("mmd") or {}
    pp_delta = as_float(pp.get("delta_mean"))
    mmd_delta = as_float(mmd.get("delta_mean"))
    if pp_delta is None and mmd_delta is None:
        return rows
    rows.append(
        {
            "source_path": str(path),
            "record_type": "summary",
            "record_name": path.stem,
            "axis_family": axis_family(path, path.stem),
            "status": str(obj.get("status") or ""),
            "cross_pp_delta": pp_delta,
            "family_pp_delta": pp_delta,
            "family_mmd_delta": mmd_delta,
            "is_placebo_or_null": False,
        }
    )
    return rows


def extract_rows_from_decision_checks(path: Path, obj: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, dict[str, Any]]] = []
    dec = obj.get("decision")
    if isinstance(dec, dict):
        if isinstance(dec.get("checks"), dict):
            candidates.append(("decision.checks", dec["checks"]))
        if isinstance(dec.get("gate_checks"), dict):
            candidates.append(("decision.gate_checks", dec["gate_checks"]))
    for key in [
        "full_extension_decision",
        "type_balance_extension_decision",
        "jiang_exposure_extension_decision",
        "general_exposure_extension_decision",
        "matrix_summary",
        "summary",
    ]:
        val = obj.get(key)
        if isinstance(val, dict):
            if isinstance(val.get("gate_checks"), dict):
                candidates.append((f"{key}.gate_checks", val["gate_checks"]))
            else:
                candidates.append((key, val))
    for name, checks in candidates:
        nums = flatten_numbers(checks, max_depth=2)
        cross_items = [(k, v) for k, v in nums.items() if "cross" in k.lower() and ("pp" in k.lower() or "pearson" in k.lower())]
        fam_items = [(k, v) for k, v in nums.items() if "family" in k.lower() and ("pp" in k.lower() or "pearson" in k.lower())]
        mmd_items = [(k, v) for k, v in nums.items() if "mmd" in k.lower()]
        if not cross_items and not fam_items and not mmd_items:
            if "mean_pp_delta" in nums or "mean_mmd_delta" in nums:
                cross_items = [("mean_pp_delta", nums["mean_pp_delta"])] if "mean_pp_delta" in nums else []
                fam_items = list(cross_items)
                mmd_items = [("mean_mmd_delta", nums["mean_mmd_delta"])] if "mean_mmd_delta" in nums else []
            else:
                continue
        rows.append(
            {
                "source_path": str(path),
                "record_type": "decision_checks",
                "record_name": name,
                "axis_family": axis_family(path, name),
                "status": str(
                    (checks.get("status") if isinstance(checks, dict) else "")
                    or (dec.get("status") if isinstance(dec, dict) else "")
                    or obj.get("status")
                    or ""
                ),
                "cross_pp_delta": cross_items[0][1] if cross_items else None,
                "family_pp_delta": fam_items[0][1] if fam_items else (cross_items[0][1] if cross_items else None),
                "family_mmd_delta": mmd_items[0][1] if mmd_items else None,
                "is_placebo_or_null": "placebo" in path.as_posix().lower() or "null" in path.as_posix().lower(),
            }
        )
    return rows


def collect_json_paths() -> list[Path]:
    paths: set[Path] = set()
    for glob in JSON_GLOBS:
        paths.update(REPORTS.glob(glob))
    return sorted(paths)


def collect_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for path in collect_json_paths():
        obj = read_json(path)
        if not isinstance(obj, dict):
            continue
        before = len(rows)
        rows.extend(extract_rows_from_row_list(path, obj))
        rows.extend(extract_rows_from_summaries(path, obj))
        rows.extend(extract_rows_from_decision_checks(path, obj))
        coverage.append(
            {
                "source_path": str(path),
                "extracted_rows": len(rows) - before,
                "status": str((obj.get("decision") or {}).get("status") if isinstance(obj.get("decision"), dict) else obj.get("status") or ""),
                "keys": ",".join(sorted(str(k) for k in obj.keys())[:12]),
            }
        )
    return rows, coverage


def is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def annotate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        cross = row.get("cross_pp_delta")
        family = row.get("family_pp_delta")
        mmd = row.get("family_mmd_delta")
        row = dict(row)
        row["closed_family_prior"] = row["axis_family"] in CLOSED_FAMILIES
        status_l = str(row.get("status") or "").lower()
        row["closed_status_prior"] = any(
            token in status_l
            for token in [
                "fail",
                "close",
                "closed",
                "mechanism_only",
                "no_gpu",
                "blocks",
                "harm",
            ]
        )
        row["positive_pp"] = (
            is_numeric(cross)
            and is_numeric(family)
            and float(cross) >= 0.02
            and float(family) >= 0.02
        )
        row["weak_positive_pp"] = (
            is_numeric(cross)
            and is_numeric(family)
            and float(cross) > 0.0
            and float(family) > 0.0
        )
        row["mmd_harm"] = is_numeric(mmd) and float(mmd) > 0.001
        row["candidate_nonduplicate"] = (
            row["positive_pp"]
            and not row["mmd_harm"]
            and not row["closed_family_prior"]
            and not row["closed_status_prior"]
            and not bool(row.get("is_placebo_or_null"))
        )
        out.append(row)
    return out


def median(vals: list[float]) -> float | None:
    return float(statistics.median(vals)) if vals else None


def mean(vals: list[float]) -> float | None:
    return float(statistics.mean(vals)) if vals else None


def family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["axis_family"]), []).append(row)
    out = []
    for fam, vals in sorted(groups.items()):
        crosses = [float(r["cross_pp_delta"]) for r in vals if is_numeric(r.get("cross_pp_delta"))]
        families = [float(r["family_pp_delta"]) for r in vals if is_numeric(r.get("family_pp_delta"))]
        mmds = [float(r["family_mmd_delta"]) for r in vals if is_numeric(r.get("family_mmd_delta"))]
        out.append(
            {
                "axis_family": fam,
                "n_rows": len(vals),
                "closed_family_prior": fam in CLOSED_FAMILIES,
                "n_positive_pp": sum(bool(r["positive_pp"]) for r in vals),
                "n_weak_positive_pp": sum(bool(r["weak_positive_pp"]) for r in vals),
                "n_mmd_harm": sum(bool(r["mmd_harm"]) for r in vals),
                "n_candidate_nonduplicate": sum(bool(r["candidate_nonduplicate"]) for r in vals),
                "cross_mean": mean(crosses),
                "cross_median": median(crosses),
                "cross_max": max(crosses) if crosses else None,
                "family_mean": mean(families),
                "family_median": median(families),
                "family_max": max(families) if families else None,
                "mmd_mean": mean(mmds),
                "mmd_max": max(mmds) if mmds else None,
            }
        )
    return out


def decision(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [r for r in rows if r.get("candidate_nonduplicate")]
    closed_positive = [r for r in rows if r.get("positive_pp") and r.get("closed_family_prior")]
    weak_nonclosed = [
        r
        for r in rows
        if r.get("weak_positive_pp")
        and not r.get("closed_family_prior")
        and not r.get("is_placebo_or_null")
    ]
    reasons: list[str] = []
    if not candidates:
        reasons.append("no_nonclosed_family_has_cross_and_family_pp_ge_0p02_without_mmd_harm")
    if closed_positive:
        reasons.append("positive_signals_exist_but_are_in_closed_or_null_calibrated_families")
    status = "run_outcome_mechanism_panel_gpu_candidate_found" if candidates else "run_outcome_mechanism_panel_no_new_gpu_axis"
    top_families = sorted(
        summaries,
        key=lambda r: (
            int(r.get("n_candidate_nonduplicate") or 0),
            float(r.get("cross_max") or -999),
            float(r.get("family_max") or -999),
        ),
        reverse=True,
    )[:8]
    return {
        "status": status,
        "gpu_authorized_next": bool(candidates),
        "reasons": reasons,
        "n_rows": len(rows),
        "n_candidate_nonduplicate": len(candidates),
        "n_closed_positive": len(closed_positive),
        "n_weak_nonclosed": len(weak_nonclosed),
        "candidate_records": candidates[:10],
        "weak_nonclosed_records": weak_nonclosed[:10],
        "top_family_summaries": top_families,
        "next_action": (
            "design_guarded_gpu_smoke_from_candidate_family"
            if candidates
            else "do_not_launch_gpu_from_retrospective_panel; use weak_nonclosed_rows_for_next_cpu_gate"
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_report(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "latentfm_run_outcome_mechanism_panel_20260630.json"
    rows_path = OUT_DIR / "latentfm_run_outcome_mechanism_rows_20260630.csv"
    fam_path = OUT_DIR / "latentfm_run_outcome_mechanism_family_summary_20260630.csv"
    cov_path = OUT_DIR / "latentfm_run_outcome_mechanism_source_coverage_20260630.csv"
    md_path = OUT_DIR / "LATENTFM_RUN_OUTCOME_MECHANISM_PANEL_20260630.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows_path, payload["rows"])
    write_csv(fam_path, payload["family_summary"])
    write_csv(cov_path, payload["source_coverage"])
    dec = payload["decision"]
    lines = [
        "# LatentFM Run Outcome Mechanism Panel 20260630",
        "",
        "## Boundary",
        "",
        "- CPU-only retrospective synthesis of completed report artifacts.",
        "- No training, inference, active-log polling, checkpoint selection, canonical multi selection, or Track C held-out query access.",
        "- Closed support/tail/HVG/exact/ZSCAPE axes are used as covariates or negative controls, not relaunch candidates.",
        "",
        "## Decision",
        "",
        f"- status: `{dec['status']}`",
        f"- gpu authorized next: `{dec['gpu_authorized_next']}`",
        f"- extracted rows: `{dec['n_rows']}`",
        f"- candidate nonduplicate rows: `{dec['n_candidate_nonduplicate']}`",
        f"- closed positive rows: `{dec['n_closed_positive']}`",
        f"- weak nonclosed rows: `{dec['n_weak_nonclosed']}`",
        f"- reasons: `{', '.join(dec['reasons']) if dec['reasons'] else 'none'}`",
        f"- next action: `{dec['next_action']}`",
        "",
        "## Family Summary",
        "",
        "| family | rows | closed | positive | weak positive | MMD harm | candidates | cross max | family max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        payload["family_summary"],
        key=lambda r: (int(r["n_candidate_nonduplicate"]), float(r.get("cross_max") or -999)),
        reverse=True,
    ):
        lines.append(
            "| {axis_family} | {n_rows} | {closed_family_prior} | {n_positive_pp} | "
            "{n_weak_positive_pp} | {n_mmd_harm} | {n_candidate_nonduplicate} | "
            "{cross_max} | {family_max} |".format(
                axis_family=row["axis_family"],
                n_rows=row["n_rows"],
                closed_family_prior=row["closed_family_prior"],
                n_positive_pp=row["n_positive_pp"],
                n_weak_positive_pp=row["n_weak_positive_pp"],
                n_mmd_harm=row["n_mmd_harm"],
                n_candidate_nonduplicate=row["n_candidate_nonduplicate"],
                cross_max=fmt(row.get("cross_max")),
                family_max=fmt(row.get("family_max")),
            )
        )
    lines.extend(["", "## Weak Nonclosed Rows", ""])
    weak = dec.get("weak_nonclosed_records") or []
    if weak:
        lines.extend(
            [
                "| family | record | cross pp | family pp | family MMD | status |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for row in weak:
            lines.append(
                f"| {row['axis_family']} | `{row['record_name']}` | "
                f"{fmt(row.get('cross_pp_delta'))} | {fmt(row.get('family_pp_delta'))} | "
                f"{fmt(row.get('family_mmd_delta'))} | `{row.get('status','')}` |"
            )
    else:
        lines.append("No weak nonclosed rows were found.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{json_path}`",
            f"- rows: `{rows_path}`",
            f"- family summary: `{fam_path}`",
            f"- source coverage: `{cov_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows, coverage = collect_rows()
    rows = annotate_rows(rows)
    fam = family_summary(rows)
    payload = {
        "boundary": {
            "cpu_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_heldout_query_access": False,
        },
        "closed_families": sorted(CLOSED_FAMILIES),
        "rows": rows,
        "family_summary": fam,
        "source_coverage": coverage,
    }
    payload["decision"] = decision(rows, fam)
    write_report(payload)
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
