#!/usr/bin/env python3
"""Condition-count no-harm predictor rescue gate.

This CPU-only gate asks whether the weak count-scaling signal can be promoted
into a new GPU smoke by adding a leakage-safe no-harm predictor. Frozen
canonical no-harm audits are used only as veto/calibration context, not for
tuning a new threshold.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "condition_count_noharm_predictor_rescue_20260630"

INPUTS = {
    "condition_count_rescue_rows": REPORTS
    / "condition_count_scaling_rescue_gate_20260630"
    / "condition_count_scaling_rescue_rows_20260630.csv",
    "condition_count_rescue_json": REPORTS
    / "condition_count_scaling_rescue_gate_20260630"
    / "condition_count_scaling_rescue_gate_20260630.json",
    "failure_localization": REPORTS / "latentfm_scaling_failure_localization_gate_20260624.json",
    "noharm_surrogate": REPORTS / "latentfm_scaling_noharm_surrogate_v2_gate_20260624.json",
    "positive_inventory": REPORTS / "latentfm_global_noharm_positive_class_inventory_20260624.json",
    "positive_calibration": REPORTS / "latentfm_noharm_calibration_positive_controls_gate_20260624.json",
    "clean_scaling_x": REPORTS / "clean_scaling_x_gate_20260628" / "clean_scaling_x_gate_20260628.json",
    "clean_scaling_x_associations": REPORTS
    / "clean_scaling_x_gate_20260628"
    / "clean_scaling_x_split_associations.csv",
    "trainset_strategy_queue": REPORTS
    / "trainset_strategy_queue_20260630"
    / "trainset_strategy_queue_20260630.json",
}

HIGHLOW_REQUIRED_CROSS_PP = 0.0408
HIGHLOW_REQUIRED_FAMILY_PP = 0.0365
MMD_HARD_HARM_DELTA = 0.001


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(obj: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def route_table(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for col in ["cross_pp_delta", "family_pp_delta", "family_mmd_delta"]:
        rows[col] = pd.to_numeric(rows.get(col), errors="coerce")
    output = []
    for _, row in rows.iterrows():
        cross = finite_float(row.get("cross_pp_delta"))
        family = finite_float(row.get("family_pp_delta"))
        mmd = finite_float(row.get("family_mmd_delta"))
        internal_positive = (cross is not None and cross > 0) and (family is not None and family > 0)
        mmd_harm = mmd is not None and mmd > MMD_HARD_HARM_DELTA
        clears_highlow = (
            cross is not None
            and family is not None
            and cross >= HIGHLOW_REQUIRED_CROSS_PP
            and family >= HIGHLOW_REQUIRED_FAMILY_PP
            and not mmd_harm
        )
        vetoed = "canonical_noharm_fail" in str(row.get("keep_or_close")) or mmd_harm
        output.append(
            {
                "source": row.get("source"),
                "route": row.get("route"),
                "status": row.get("status"),
                "cross_pp_delta": cross,
                "family_pp_delta": family,
                "family_mmd_delta": mmd,
                "internal_positive": internal_positive,
                "mmd_hard_harm": mmd_harm,
                "clears_null_calibrated_highlow_bar": clears_highlow,
                "known_veto": vetoed,
                "keep_or_close": row.get("keep_or_close"),
                "blockers": row.get("blockers"),
            }
        )
    out = pd.DataFrame(output)
    return out.sort_values(
        ["clears_null_calibrated_highlow_bar", "internal_positive", "cross_pp_delta"],
        ascending=[False, False, False],
    )


def calibration_summary(inventory: dict[str, Any], calibration: dict[str, Any], surrogate: dict[str, Any]) -> dict[str, Any]:
    inv_rows = inventory.get("rows") or []
    cal_rows = calibration.get("rows") or []
    material_inv = [
        r
        for r in inv_rows
        if str(r.get("class")) == "noharm_positive" and not bool(r.get("trivial_noop"))
    ]
    material_cal = [
        r
        for r in cal_rows
        if str(r.get("calibration_class")) == "material_noharm_positive"
    ]
    trivial = [r for r in inv_rows if bool(r.get("trivial_noop"))]
    harmful = [r for r in cal_rows if str(r.get("calibration_class")) == "harmful_negative_control"]
    return {
        "material_inventory_positive_count": int(len(material_inv)),
        "material_calibration_positive_count": int(len(material_cal)),
        "trivial_noop_positive_count": int(len(trivial)),
        "harmful_negative_controls": int(len(harmful)),
        "surrogate_status": surrogate.get("status"),
        "surrogate_n_canonical_pass": nested_get(surrogate, ["summary", "n_canonical_pass"], 0),
        "surrogate_n_canonical_fail": nested_get(surrogate, ["summary", "n_canonical_fail"], 0),
        "surrogate_n_internal_pass_like": nested_get(surrogate, ["summary", "n_internal_pass_like"], 0),
        "surrogate_n_pp_hard_harm": nested_get(surrogate, ["summary", "n_pp_hard_harm"], 0),
        "closed_family_vetoes": nested_get(surrogate, ["summary", "closed_family_vetoes"], []),
        "unsafe_envelope": surrogate.get("unsafe_envelope") or {},
    }


def clean_axis_summary(clean: dict[str, Any], assoc: pd.DataFrame) -> dict[str, Any]:
    if assoc.empty:
        best = []
    else:
        work = assoc.copy()
        work["spearman_rho"] = pd.to_numeric(work.get("spearman_rho"), errors="coerce")
        best = (
            work.reindex(work["spearman_rho"].abs().sort_values(ascending=False).index)
            .head(6)
            .to_dict(orient="records")
        )
    return {
        "status": clean.get("status"),
        "gpu_authorized_next": bool(clean.get("gpu_authorized_next", False)),
        "gpu_authorized_axes": clean.get("gpu_authorized_axes", []),
        "top_associations": best,
    }


def decide(routes: pd.DataFrame, cal: dict[str, Any], failure: dict[str, Any], clean_summary: dict[str, Any]) -> dict[str, Any]:
    train_positive = routes[
        (routes["internal_positive"] == True) & (routes["mmd_hard_harm"] == False)  # noqa: E712
    ]
    pass_highlow = routes[routes["clears_null_calibrated_highlow_bar"] == True]  # noqa: E712
    material_positive_count = max(
        int(cal.get("material_inventory_positive_count", 0)),
        int(cal.get("material_calibration_positive_count", 0)),
    )
    failure_summary = failure.get("summary") or {}
    reasons: list[str] = []
    if material_positive_count < 3:
        reasons.append("material_noharm_positive_count_lt_3")
    if int(cal.get("surrogate_n_canonical_pass", 0)) == 0:
        reasons.append("zero_nontrivial_canonical_noharm_positive_examples")
    if int(cal.get("surrogate_n_internal_pass_like", 0)) > 0 and int(cal.get("surrogate_n_canonical_fail", 0)) > 0:
        reasons.append("internal_pass_like_examples_are_canonical_failures")
    if pass_highlow.empty:
        reasons.append("no_count_route_clears_null_calibrated_highlow_bar")
    if not routes[routes["mmd_hard_harm"] == True].empty:  # noqa: E712
        reasons.append("best_large_gain_count_variants_have_mmd_hard_harm")
    if not clean_summary.get("gpu_authorized_next"):
        reasons.append("clean_scaling_x_gate_does_not_authorize_gpu")
    if failure_summary.get("n_canonical_pass", 0) == 0:
        reasons.append("failure_localization_has_no_canonical_pass_family")

    gpu_authorized = False
    status = "condition_count_noharm_predictor_rescue_fail_no_gpu"
    next_action = (
        "close count-scaling no-harm predictor rescue as a GPU route; keep count "
        "scaling as weak scaling/failure-map evidence and move to an orthogonal "
        "CPU-gated mechanism such as DepMap tail-risk or wait for ZSCAPE strict controls"
    )
    if material_positive_count >= 3 and not pass_highlow.empty:
        gpu_authorized = True
        status = "condition_count_noharm_predictor_rescue_pass_gpu_allowed_after_audit"
        next_action = "run GPU audit and launch rescued cap120 plus matched placebo"

    return {
        "status": status,
        "gpu_authorized_next": gpu_authorized,
        "reasons": reasons,
        "train_positive_no_mmd_rows": int(len(train_positive)),
        "null_calibrated_pass_rows": int(len(pass_highlow)),
        "material_noharm_positive_count": material_positive_count,
        "highlow_required_cross_pp": HIGHLOW_REQUIRED_CROSS_PP,
        "highlow_required_family_pp": HIGHLOW_REQUIRED_FAMILY_PP,
        "canonical_metrics_use": "frozen_veto_context_only_not_threshold_tuning",
        "next_action": next_action,
    }


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def md_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if df.empty:
        return "_None._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].head(n).iterrows():
        vals = []
        for col in cols:
            value = row.get(col)
            vals.append(fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(
    routes: pd.DataFrame,
    cal: dict[str, Any],
    clean_summary: dict[str, Any],
    decision: dict[str, Any],
    inputs_payload: dict[str, Any],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "condition_count_noharm_predictor_rescue_rows_20260630.csv"
    json_path = OUT_DIR / "condition_count_noharm_predictor_rescue_20260630.json"
    md_path = OUT_DIR / "LATENTFM_CONDITION_COUNT_NOHARM_PREDICTOR_RESCUE_20260630.md"
    routes.to_csv(rows_path, index=False)
    payload = {
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "canonical_metrics_use": "frozen_veto_context_only_not_threshold_tuning",
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "calibration_summary": cal,
        "clean_axis_summary": clean_summary,
        "prior_status": {
            "condition_count_rescue": nested_get(inputs_payload["condition_count"], ["decision", "status"]),
            "failure_localization": inputs_payload["failure"].get("status"),
            "noharm_surrogate": inputs_payload["surrogate"].get("status"),
            "positive_calibration_gpu": inputs_payload["calibration"].get("gpu_authorized"),
        },
        "decision": decision,
        "outputs": {"rows": str(rows_path), "markdown_report": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    text = f"""# LatentFM Condition-Count No-Harm Predictor Rescue 20260630

## Boundary

- CPU/report-only gate over completed count-scaling, no-harm surrogate, positive-class inventory, and clean-scaling-x artifacts.
- No training, inference, active-log polling, checkpoint selection, canonical multi selection, or Track C query access.
- Frozen canonical single/family no-harm audits are used only as immutable veto/calibration context; they are not used to tune a new threshold.

## Decision

- status: `{decision['status']}`
- GPU authorized next: `{decision['gpu_authorized_next']}`
- reasons: `{', '.join(decision['reasons'])}`
- material no-harm positives: `{decision['material_noharm_positive_count']}`
- train-positive/no-MMD count rows: `{decision['train_positive_no_mmd_rows']}`
- null-calibrated pass rows: `{decision['null_calibrated_pass_rows']}`
- next action: `{decision['next_action']}`

## Route Table

{md_table(routes, ['source', 'route', 'cross_pp_delta', 'family_pp_delta', 'family_mmd_delta', 'internal_positive', 'mmd_hard_harm', 'clears_null_calibrated_highlow_bar', 'known_veto', 'keep_or_close'])}

## Calibration Summary

- no-harm surrogate status: `{cal['surrogate_status']}`
- internal pass-like candidates: `{cal['surrogate_n_internal_pass_like']}`
- canonical pass/fail: `{cal['surrogate_n_canonical_pass']}` / `{cal['surrogate_n_canonical_fail']}`
- PP hard-harm examples: `{cal['surrogate_n_pp_hard_harm']}`
- closed family vetoes: `{', '.join(map(str, cal['closed_family_vetoes']))}`
- clean scaling x status: `{clean_summary['status']}`
- clean scaling x GPU authorized: `{clean_summary['gpu_authorized_next']}`

## Interpretation

The count-scaling branch still has a weak internal scaling signal, especially
`cap120_all`, but it cannot be rescued into a GPU route by a no-harm predictor
right now. The predictor has no nontrivial positive class, all pass-like scaling
families have failed canonical no-harm audits, and the larger count/exposure
gains are the same family of routes with known MMD or canonical harm.

This closes another route for simply keeping GPUs busy with count-scaling
variants. Future GPU work needs an orthogonal mechanism that creates new
positive calibration evidence first.

## Artifacts

- JSON: `{json_path}`
- rows: `{rows_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    rows = pd.read_csv(INPUTS["condition_count_rescue_rows"])
    condition_count = load_json(INPUTS["condition_count_rescue_json"])
    failure = load_json(INPUTS["failure_localization"])
    surrogate = load_json(INPUTS["noharm_surrogate"])
    inventory = load_json(INPUTS["positive_inventory"])
    calibration = load_json(INPUTS["positive_calibration"])
    clean = load_json(INPUTS["clean_scaling_x"])
    assoc = (
        pd.read_csv(INPUTS["clean_scaling_x_associations"])
        if INPUTS["clean_scaling_x_associations"].exists()
        else pd.DataFrame()
    )
    queue = load_json(INPUTS["trainset_strategy_queue"])
    routes = route_table(rows)
    cal = calibration_summary(inventory, calibration, surrogate)
    clean_summary = clean_axis_summary(clean, assoc)
    decision = decide(routes, cal, failure, clean_summary)
    write_outputs(
        routes,
        cal,
        clean_summary,
        decision,
        {
            "condition_count": condition_count,
            "failure": failure,
            "surrogate": surrogate,
            "calibration": calibration,
            "queue": queue,
        },
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
