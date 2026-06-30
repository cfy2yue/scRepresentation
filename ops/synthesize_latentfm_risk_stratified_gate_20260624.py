#!/usr/bin/env python3
"""CPU-only risk-stratified gate after corrected risk-conditioned adjudication.

This gate uses completed train-only internal posthoc artifacts only. It exists
to decide whether the tian-norman positive mechanism is safe enough to promote
or whether any future mutation must directly target non-target risk-row tails.
It does not read canonical metrics, canonical multi outputs, or Track C query.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_general_exposure_tian_norman_mmd20_replayall_3k_seed42"
RUN_ROOT = ROOT / "runs/latentfm_risk_conditioned_general_exposure_smoke_20260624" / RUN_NAME
EVAL_ROOT = RUN_ROOT / "posthoc_eval_internal"
SOURCE_JSON = ROOT / "reports/latentfm_risk_conditioned_general_exposure_smoke_decision_20260624.json"
CORRECTED_JSON = ROOT / "reports/latentfm_risk_conditioned_corrected_adjudication_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_risk_stratified_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_RISK_STRATIFIED_GATE_20260624.md"

MMD_HARM_EPS = 0.001
SEVERE_MMD_EPS = 0.005
SEVERE_PP_DROP = -0.02


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_condition(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for row in payload["groups"]["family_gene"].get("condition_metrics", []):
        rows[(str(row["dataset"]), str(row["condition"]))] = row
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def condition_deltas() -> list[dict[str, Any]]:
    anchor = rows_by_condition(load_json(EVAL_ROOT / "condition_family_eval_anchor_internal_ode20.json"))
    cand = rows_by_condition(load_json(EVAL_ROOT / "condition_family_eval_candidate_internal_ode20.json"))
    rows = []
    for key in sorted(set(anchor) & set(cand)):
        ds, cond = key
        arow, crow = anchor[key], cand[key]
        a_mmd = arow.get("test_mmd_clamped", arow.get("test_mmd"))
        c_mmd = crow.get("test_mmd_clamped", crow.get("test_mmd"))
        if a_mmd is None or c_mmd is None:
            continue
        pp_delta = None
        if arow.get("pearson_pert") is not None and crow.get("pearson_pert") is not None:
            pp_delta = float(crow["pearson_pert"]) - float(arow["pearson_pert"])
        rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "mmd_delta": float(c_mmd) - float(a_mmd),
                "pp_delta": pp_delta,
                "candidate_mmd": float(c_mmd),
                "anchor_mmd": float(a_mmd),
                "candidate_pp": crow.get("pearson_pert"),
                "anchor_pp": arow.get("pearson_pert"),
            }
        )
    return rows


def cvar_top(values: list[float], frac: float) -> float | None:
    if not values:
        return None
    k = max(1, math.ceil(len(values) * frac))
    top = sorted(values, reverse=True)[:k]
    return sum(top) / len(top)


def main() -> int:
    source = load_json(SOURCE_JSON)
    corrected = load_json(CORRECTED_JSON)
    risk_datasets = set(source["risk_datasets"])
    target_risk_datasets = {"TianActivation", "NormanWeissman2019_filtered"}
    non_target_risk_datasets = risk_datasets - target_risk_datasets
    source_row = next(row for row in source["rows"] if row["name"] == RUN_NAME)
    source_dataset_rows = {row["dataset"]: row for row in source_row["metrics"]["dataset_rows"]}

    all_condition_rows = condition_deltas()
    risk_condition_rows = [row for row in all_condition_rows if row["dataset"] in risk_datasets]

    dataset_gate_rows = []
    for ds in sorted(risk_datasets):
        ds_rows = [row for row in risk_condition_rows if row["dataset"] == ds]
        mmd_values = [row["mmd_delta"] for row in ds_rows]
        severe_rows = [
            row for row in ds_rows
            if row["mmd_delta"] > SEVERE_MMD_EPS
            or (row["mmd_delta"] > MMD_HARM_EPS and (row["pp_delta"] or 0.0) < SEVERE_PP_DROP)
        ]
        prior = source_dataset_rows.get(ds, {})
        is_target = ds in target_risk_datasets
        fail_reasons = []
        if is_target:
            if (prior.get("mean_mmd_delta") or 0.0) > 0.0:
                fail_reasons.append("target_mean_mmd_not_improved")
            if (prior.get("mean_pp_delta") or -999.0) < 0.0:
                fail_reasons.append("target_mean_pp_not_improved")
            if len(severe_rows) > 1:
                fail_reasons.append("target_severe_tail_rows")
        else:
            if (prior.get("mean_mmd_delta") or 0.0) > MMD_HARM_EPS:
                fail_reasons.append("nontarget_mean_mmd_harm")
            if len(severe_rows) > 1:
                fail_reasons.append("nontarget_severe_tail_rows")
            tail = cvar_top(mmd_values, 0.20)
            if tail is not None and tail > SEVERE_MMD_EPS:
                fail_reasons.append("nontarget_top20_cvar_mmd_harm")
        dataset_gate_rows.append(
            {
                "dataset": ds,
                "is_target_risk_dataset": is_target,
                "n": len(ds_rows),
                "mean_mmd_delta": prior.get("mean_mmd_delta"),
                "mean_pp_delta": prior.get("mean_pp_delta"),
                "mmd_harm_rows": prior.get("mmd_harm_rows"),
                "top20_cvar_mmd_delta": cvar_top(mmd_values, 0.20),
                "severe_rows": len(severe_rows),
                "fail_reasons": fail_reasons,
                "status": "pass" if not fail_reasons else "fail",
            }
        )

    failed_non_target = [
        row for row in dataset_gate_rows
        if not row["is_target_risk_dataset"] and row["status"] == "fail"
    ]
    target_fail = [
        row for row in dataset_gate_rows
        if row["is_target_risk_dataset"] and row["status"] == "fail"
    ]
    status = "risk_stratified_gate_fail_no_gpu" if failed_non_target or target_fail else "risk_stratified_gate_pass_needs_external_review"

    top_severe = [
        row for row in risk_condition_rows
        if row["mmd_delta"] > SEVERE_MMD_EPS
        or (row["mmd_delta"] > MMD_HARM_EPS and (row["pp_delta"] or 0.0) < SEVERE_PP_DROP)
    ]
    top_severe.sort(key=lambda row: (row["mmd_delta"], -(row["pp_delta"] or 0.0)), reverse=True)

    payload = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "train_only_internal_posthoc": True,
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
        },
        "corrected_adjudication_status": corrected["status"],
        "corrected_decision": corrected["decision"],
        "run_name": RUN_NAME,
        "predeclared_gate": {
            "target_risk_datasets": sorted(target_risk_datasets),
            "non_target_risk_datasets": sorted(non_target_risk_datasets),
            "mmd_harm_eps": MMD_HARM_EPS,
            "severe_mmd_eps": SEVERE_MMD_EPS,
            "severe_pp_drop": SEVERE_PP_DROP,
            "target_requirements": [
                "mean_mmd_delta <= 0",
                "mean_pp_delta >= 0",
                "severe_rows <= 1",
            ],
            "non_target_requirements": [
                "mean_mmd_delta <= 0.001",
                "severe_rows <= 1",
                "top20_cvar_mmd_delta <= 0.005",
            ],
        },
        "dataset_gate_rows": dataset_gate_rows,
        "top_severe_rows": top_severe[:25],
        "next_action": {
            "gpu_authorized": False,
            "canonical_authorized": False,
            "recommendation": "wait_for_external_review_or_design_cpu_unit_gate_for_default_off_risk_row_cvar_loss",
            "bounded_hypothesis": (
                "If this branch continues, test a distinct risk-row CVaR/top-k MMD loss that "
                "penalizes worst condition-level MMD rows in the fixed risk dataset set, rather "
                "than changing scalar gamma or replay scope."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Risk-Stratified Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate over completed train-only internal posthoc artifacts.",
        "- No canonical metrics, canonical multi, or Track C query were read.",
        "- Uses corrected adjudication as current branch state: `mutate_not_promote`.",
        "",
        "## Gate Definition",
        "",
        "- Target risk datasets: `TianActivation`, `NormanWeissman2019_filtered`.",
        "- Non-target risk datasets: Nadig/Replogle risk datasets from the predeclared risk set.",
        "- Target pass requires mean MMD improvement, mean Pearson improvement, and at most one severe tail row.",
        "- Non-target pass requires mean MMD delta <= `+0.001`, at most one severe tail row, and top-20% MMD CVaR <= `+0.005`.",
        "",
        "## Dataset Results",
        "",
        "| dataset | target | n | mean MMD | mean pp | harm rows | severe rows | top20 CVaR MMD | status | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in dataset_gate_rows:
        lines.append(
            f"| `{row['dataset']}` | `{row['is_target_risk_dataset']}` | {row['n']} | "
            f"{fmt(row['mean_mmd_delta'])} | {fmt(row['mean_pp_delta'])} | "
            f"{row['mmd_harm_rows']} | {row['severe_rows']} | "
            f"{fmt(row['top20_cvar_mmd_delta'])} | `{row['status']}` | "
            f"`{','.join(row['fail_reasons'])}` |"
        )

    lines.extend(
        [
            "",
            "## Top Severe Rows",
            "",
            "| dataset | condition | MMD delta | pp delta |",
            "|---|---|---:|---:|",
        ]
    )
    for row in top_severe[:15]:
        lines.append(
            f"| `{row['dataset']}` | `{row['condition']}` | {fmt(row['mmd_delta'])} | {fmt(row['pp_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- No GPU or canonical no-harm is authorized by this CPU gate.",
            "- The tian-norman arm remains positive mechanism evidence but fails non-target risk stratification.",
            "- Next distinct hypothesis, if externally approved: default-off risk-row CVaR/top-k MMD loss over the fixed risk dataset set, with CPU/unit validation before any capped 2k train-only smoke.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
