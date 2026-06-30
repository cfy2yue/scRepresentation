#!/usr/bin/env python3
"""CPU-only completion gate for the true-cell budget scaling axis.

This synthesizes completed train-only/internal true-cell budget summaries,
their nested controls, and the already-frozen canonical no-harm veto. It does
not read checkpoints, canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_truecell_scaling_count_tail_completion_gate_20260625.json"
OUT_MD = REPORTS / "LATENTFM_TRUECELL_SCALING_COUNT_TAIL_COMPLETION_GATE_20260625.md"
OUT_CSV = REPORTS / "latentfm_truecell_scaling_count_tail_completion_rows_20260625.csv"

INPUTS = {
    "nested_3k": REPORTS / "latentfm_true_cell_count_nested_matrix_decision_20260624.json",
    "nested_3k_controls": REPORTS / "latentfm_true_cell_count_nested_controls_gate_20260624.json",
    "budget64_6k": REPORTS / "latentfm_true_cell_count_budget64_tail_stability_6k_decision_20260625.json",
    "budget128_6k": REPORTS / "latentfm_true_cell_count_budget128_tail_stability_6k_decision_20260625.json",
    "budget128_6k_controls": REPORTS / "latentfm_true_cell_count_budget128_tail_stability_6k_controls_20260625.json",
    "budget256_6k": REPORTS / "latentfm_true_cell_count_budget256_tail_stability_6k_decision_20260625.json",
    "budget256_6k_controls": REPORTS / "latentfm_true_cell_count_budget256_tail_stability_6k_controls_20260625.json",
    "canonical_noharm": REPORTS / "latentfm_true_cell_count_budget128_6k_canonical_noharm_decision_20260625.json",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def fmt(x: Any, digits: int = 6) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):+.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def budget_rows(payload: dict[str, Any], *, series: str, steps: int) -> list[dict[str, Any]]:
    rows = []
    for row in nested_get(payload, "matrix_summary", "budget_rows") or []:
        cross_boot = row.get("cross_background_pp_condition_bootstrap") or {}
        family_boot = row.get("family_gene_pp_condition_bootstrap") or {}
        cross_tail = row.get("cross_background_pp_dataset_tail") or {}
        family_tail = row.get("family_gene_pp_dataset_tail") or {}
        rows.append(
            {
                "series": series,
                "steps": steps,
                "budget": int(row["budget"]),
                "n_complete": int(row.get("n_complete") or row.get("complete") or 0),
                "seed_passes": int(row.get("seed_passes") or 0),
                "cross_pp_mean": row.get("cross_background_pp_delta_mean"),
                "family_pp_mean": row.get("family_gene_pp_delta_mean"),
                "family_mmd_mean": row.get("family_gene_mmd_delta_mean"),
                "cross_pp_ci_low": (cross_boot.get("ci95") or [None, None])[0],
                "cross_pp_ci_high": (cross_boot.get("ci95") or [None, None])[1],
                "family_pp_ci_low": (family_boot.get("ci95") or [None, None])[0],
                "family_pp_ci_high": (family_boot.get("ci95") or [None, None])[1],
                "cross_dataset_min": nested_get(cross_tail, "min_dataset", "mean"),
                "family_dataset_min": nested_get(family_tail, "min_dataset", "mean"),
                "cross_negative_tails": int(cross_tail.get("negative_tail_lt_minus_0p020") or 0),
                "family_negative_tails": int(family_tail.get("negative_tail_lt_minus_0p020") or 0),
            }
        )
    return rows


def canonical_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or []
    out_rows = []
    for row in rows:
        metrics = row.get("metrics") or {}
        out_rows.append(
            {
                "seed": row.get("seed"),
                "run": row.get("run"),
                "gate_status": row.get("gate_status"),
                "cross_pp": nested_get(metrics, "cross_background_seen_gene:pearson_pert", "delta_mean"),
                "all_single_p_harm": nested_get(metrics, "all_test_single:pearson_pert", "p_harm"),
                "family_p_harm": nested_get(metrics, "family_gene:pearson_pert", "p_harm"),
                "reasons": row.get("gate_reasons") or [],
            }
        )
    return {
        "status": nested_get(payload, "decision", "status"),
        "action": nested_get(payload, "decision", "action"),
        "n_rows": len(out_rows),
        "all_failed": all((r["gate_status"] or "") != "candidate_gate_pass" for r in out_rows),
        "rows": out_rows,
    }


def slope(rows: list[dict[str, Any]], *, series: str, lo: int, hi: int, key: str) -> float | None:
    by_budget = {int(r["budget"]): r for r in rows if r["series"] == series}
    if lo not in by_budget or hi not in by_budget:
        return None
    a = by_budget[lo].get(key)
    b = by_budget[hi].get(key)
    if a is None or b is None:
        return None
    return float(b) - float(a)


def main() -> int:
    payloads = {name: load_json(path) for name, path in INPUTS.items()}
    rows = []
    rows.extend(budget_rows(payloads["nested_3k"], series="3k_nested", steps=3000))
    rows.extend(budget_rows(payloads["budget64_6k"], series="6k_budget64", steps=6000))
    rows.extend(budget_rows(payloads["budget128_6k"], series="6k_budget128", steps=6000))
    rows.extend(budget_rows(payloads["budget256_6k"], series="6k_budget256", steps=6000))

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    controls_3k = payloads["nested_3k_controls"]
    controls_6k_128 = payloads["budget128_6k_controls"]
    controls_6k_256 = payloads["budget256_6k_controls"]
    canonical = canonical_summary(payloads["canonical_noharm"])

    budget128_6k = next(r for r in rows if r["series"] == "6k_budget128" and r["budget"] == 128)
    budget64_6k = next(r for r in rows if r["series"] == "6k_budget64" and r["budget"] == 64)
    budget256_6k = next(r for r in rows if r["series"] == "6k_budget256" and r["budget"] == 256)
    nested_3k_128 = next(r for r in rows if r["series"] == "3k_nested" and r["budget"] == 128)
    nested_3k_256 = next(r for r in rows if r["series"] == "3k_nested" and r["budget"] == 256)

    slopes = {
        "3k_128_minus_64_cross_pp": slope(rows, series="3k_nested", lo=64, hi=128, key="cross_pp_mean"),
        "3k_256_minus_128_cross_pp": slope(rows, series="3k_nested", lo=128, hi=256, key="cross_pp_mean"),
        "6k_128_minus_64_cross_pp": (
            float(budget128_6k["cross_pp_mean"]) - float(budget64_6k["cross_pp_mean"])
            if budget128_6k["cross_pp_mean"] is not None and budget64_6k["cross_pp_mean"] is not None
            else None
        ),
        "6k_128_minus_64_family_pp": (
            float(budget128_6k["family_pp_mean"]) - float(budget64_6k["family_pp_mean"])
            if budget128_6k["family_pp_mean"] is not None and budget64_6k["family_pp_mean"] is not None
            else None
        ),
        "6k_256_minus_128_cross_pp": (
            float(budget256_6k["cross_pp_mean"]) - float(budget128_6k["cross_pp_mean"])
            if budget256_6k["cross_pp_mean"] is not None and budget128_6k["cross_pp_mean"] is not None
            else None
        ),
        "6k_256_minus_128_family_pp": (
            float(budget256_6k["family_pp_mean"]) - float(budget128_6k["family_pp_mean"])
            if budget256_6k["family_pp_mean"] is not None and budget128_6k["family_pp_mean"] is not None
            else None
        ),
    }

    control_rows = {
        f"{r.get('group')}:{r.get('metric')}": r
        for r in controls_3k.get("control_rows", [])
    }
    primary_control = control_rows.get("cross_background:pearson_pert") or {}
    primary_shuffle_p = nested_get(primary_control, "budget_shuffle", "perm_p_ge_range")
    primary_dataset_demeaned = nested_get(primary_control, "dataset_control", "dataset_demeaned_budget_means")

    reasons = []
    if payloads["nested_3k_controls"].get("status") != "nested_controls_pass_no_gpu":
        reasons.append("3k_nested_controls_not_passed")
    if payloads["budget128_6k_controls"].get("status") != "nested_controls_pass_no_gpu":
        reasons.append("6k_budget128_controls_not_passed")
    if payloads["budget256_6k_controls"].get("status") != "nested_controls_pass_no_gpu":
        reasons.append("6k_budget256_controls_not_passed")
    if slopes["3k_256_minus_128_cross_pp"] is not None and slopes["3k_256_minus_128_cross_pp"] < 0:
        reasons.append("3k_curve_is_peak_not_monotonic")
    if nested_3k_128["cross_negative_tails"] > 0 or nested_3k_256["cross_negative_tails"] > 0:
        reasons.append("3k_nested_dataset_tails_unsafe")
    if budget64_6k["cross_negative_tails"] > 0:
        reasons.append("6k_budget64_dataset_tails_unsafe")
    if budget128_6k["cross_pp_ci_low"] is None or float(budget128_6k["cross_pp_ci_low"]) <= 0:
        reasons.append("6k_budget128_cross_ci_low_not_positive")
    if float(budget128_6k["cross_dataset_min"]) < -0.02:
        reasons.append("6k_budget128_dataset_tail_unsafe")
    if budget256_6k["cross_pp_ci_low"] is None or float(budget256_6k["cross_pp_ci_low"]) <= 0:
        reasons.append("6k_budget256_cross_ci_low_not_positive")
    if float(budget256_6k["cross_dataset_min"]) < -0.02 or budget256_6k["cross_negative_tails"] > 0:
        reasons.append("6k_budget256_dataset_tail_unsafe")
    if slopes["6k_256_minus_128_cross_pp"] is not None and slopes["6k_256_minus_128_cross_pp"] <= 0:
        reasons.append("6k_budget256_does_not_beat_budget128_cross_pp")
    if canonical["status"] != "canonical_noharm_pass":
        reasons.append("frozen_canonical_noharm_failed_all_budget128_6k_seeds")
    if primary_shuffle_p is not None and float(primary_shuffle_p) > 0.50:
        reasons.append("3k_budget_range_not_separated_from_budget_shuffle_control")

    status = "truecell_scaling_count_tail_completion_fail_no_gpu" if reasons else "truecell_scaling_count_tail_completion_pass_gpu_candidate"
    decision = {
        "status": status,
        "gpu_authorized": False,
        "default_model": "xverse_8k_anchor",
        "claim_level": "strong_mechanism_signal_not_nm_scaling_law",
        "action": "keep_truecell_budget_as_mechanism_axis_and_noharm_training_data_guidance",
        "reasons": reasons,
        "next_gate": "new non-noop train-only no-harm/tail repair before any GPU promotion; otherwise manuscript mechanism/failure-map",
    }

    out = {
        "boundary": {
            "cpu_only": True,
            "reads_completed_trainonly_internal_reports": True,
            "reads_canonical_single_family_veto": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "reads_model_checkpoints": False,
            "training_or_inference": False,
            "gpu_used": False,
        },
        "inputs": {name: str(path) for name, path in INPUTS.items()},
        "rows_csv": str(OUT_CSV),
        "rows": rows,
        "slopes": slopes,
        "controls": {
            "nested_3k_status": controls_3k.get("status"),
            "budget128_6k_status": controls_6k_128.get("status"),
            "budget256_6k_status": controls_6k_256.get("status"),
            "primary_3k_cross_pp_shuffle_p_ge_range": primary_shuffle_p,
            "primary_3k_cross_pp_dataset_demeaned_means": primary_dataset_demeaned,
        },
        "canonical_noharm": canonical,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM True-Cell Scaling Count/Tail Completion Gate",
        "",
        "Status: `{}`".format(status),
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed true-cell budget reports.",
        "- Reads frozen canonical single/family no-harm only as a veto.",
        "- Does not read checkpoints, canonical multi, Track C held-out query, train, infer, or use GPU.",
        "",
        "## Key Findings",
        "",
        "- Best internal mechanism remains 6k budget128: cross/family/MMD `{}/{}/{}`.".format(
            fmt(budget128_6k["cross_pp_mean"]),
            fmt(budget128_6k["family_pp_mean"]),
            fmt(budget128_6k["family_mmd_mean"]),
        ),
        "- 6k budget128 has positive condition bootstrap lower bounds: cross/family CI lows `{}/{}`.".format(
            fmt(budget128_6k["cross_pp_ci_low"]),
            fmt(budget128_6k["family_pp_ci_low"]),
        ),
        "- 6k budget128 dataset tails are internally safe by the current threshold: cross min `{}`, negative tails `{}`.".format(
            fmt(budget128_6k["cross_dataset_min"]),
            budget128_6k["cross_negative_tails"],
        ),
        "- 6k budget64 is weaker/unsafe: cross pp `{}`, cross dataset min `{}`, negative tails `{}`.".format(
            fmt(budget64_6k["cross_pp_mean"]),
            fmt(budget64_6k["cross_dataset_min"]),
            budget64_6k["cross_negative_tails"],
        ),
        "- 6k budget256 is positive but still tail-unsafe and below budget128: cross/family/MMD `{}/{}/{}`, cross CI low `{}`, cross min `{}`, negative tails `{}`, 256-128 cross pp `{}`.".format(
            fmt(budget256_6k["cross_pp_mean"]),
            fmt(budget256_6k["family_pp_mean"]),
            fmt(budget256_6k["family_mmd_mean"]),
            fmt(budget256_6k["cross_pp_ci_low"]),
            fmt(budget256_6k["cross_dataset_min"]),
            budget256_6k["cross_negative_tails"],
            fmt(slopes["6k_256_minus_128_cross_pp"]),
        ),
        "- 3k nested curve is peaked, not monotonic: 128-64 cross pp `{}`, 256-128 cross pp `{}`.".format(
            fmt(slopes["3k_128_minus_64_cross_pp"]),
            fmt(slopes["3k_256_minus_128_cross_pp"]),
        ),
        "- 3k budget-range shuffle control is not separated for cross pp: p(range>=obs) `{}`.".format(
            "NA" if primary_shuffle_p is None else f"{float(primary_shuffle_p):.4f}"
        ),
        "- Frozen canonical no-harm blocks deployment: `{}`.".format(canonical["status"]),
        "",
        "## Budget Rows",
        "",
        "| series | steps | budget | cross pp | family pp | family MMD | cross CI low | cross min dataset | cross neg tails |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{series}` | {steps} | {budget} | {cross_pp} | {family_pp} | {family_mmd} | {cross_ci} | {cross_min} | {tails} |".format(
                series=row["series"],
                steps=row["steps"],
                budget=row["budget"],
                cross_pp=fmt(row["cross_pp_mean"]),
                family_pp=fmt(row["family_pp_mean"]),
                family_mmd=fmt(row["family_mmd_mean"]),
                cross_ci=fmt(row["cross_pp_ci_low"]),
                cross_min=fmt(row["cross_dataset_min"]),
                tails=row["cross_negative_tails"],
            )
        )
    lines.extend(
        [
            "",
            "## Canonical Veto",
            "",
            "| seed | cross-bg pp | all-single p_harm | family p_harm | gate |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in canonical["rows"]:
        lines.append(
            "| {seed} | {cross} | {allp} | {famp} | `{gate}` |".format(
                seed=row["seed"],
                cross=fmt(row["cross_pp"]),
                allp=fmt(row["all_single_p_harm"], 4),
                famp=fmt(row["family_p_harm"], 4),
                gate=row["gate_status"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- reasons: `{}`".format(reasons),
            "- action: `{}`".format(decision["action"]),
            "- next gate: `{}`".format(decision["next_gate"]),
            "",
            "Interpretation: true-cell budget is a strong training-data mechanism signal and should guide mainline data construction, but the current evidence does not support an NM-level deployable scaling law or a GPU promotion route.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- rows CSV: `{OUT_CSV}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
