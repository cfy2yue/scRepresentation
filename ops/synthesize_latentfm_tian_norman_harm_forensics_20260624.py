#!/usr/bin/env python3
"""Condition-level forensics for the Tian+Norman risk-conditioned arm.

Reads completed train-only internal posthoc artifacts only. It does not read
canonical metrics, canonical multi outputs, or Track C held-out query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_general_exposure_tian_norman_mmd20_replayall_3k_seed42"
RUN_ROOT = ROOT / "runs/latentfm_risk_conditioned_general_exposure_smoke_20260624" / RUN_NAME
EVAL_ROOT = RUN_ROOT / "posthoc_eval_internal"
DECISION_JSON = ROOT / "reports/latentfm_risk_conditioned_general_exposure_smoke_decision_20260624.json"
CORRECTED_JSON = ROOT / "reports/latentfm_risk_conditioned_corrected_adjudication_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_tian_norman_harm_forensics_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TIAN_NORMAN_HARM_FORENSICS_20260624.md"

MMD_HARM_EPS = 0.001


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def family_rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for row in payload["groups"]["family_gene"].get("condition_metrics", []):
        key = (str(row["dataset"]), str(row["condition"]))
        rows[key] = row
    return rows


def metric_delta(cand: dict[str, Any], anchor: dict[str, Any], key: str) -> float | None:
    if cand.get(key) is None or anchor.get(key) is None:
        return None
    return float(cand[key]) - float(anchor[key])


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def main() -> int:
    decision = load_json(DECISION_JSON)
    corrected = load_json(CORRECTED_JSON) if CORRECTED_JSON.is_file() else {}
    decision_row = next(row for row in decision["rows"] if row["name"] == RUN_NAME)
    risk_datasets = set(decision["risk_datasets"])
    anchor = family_rows(load_json(EVAL_ROOT / "condition_family_eval_anchor_internal_ode20.json"))
    cand = family_rows(load_json(EVAL_ROOT / "condition_family_eval_candidate_internal_ode20.json"))

    condition_rows = []
    for key in sorted(set(anchor) & set(cand)):
        ds, cond = key
        arow = anchor[key]
        crow = cand[key]
        a_mmd = arow.get("test_mmd_clamped", arow.get("test_mmd"))
        c_mmd = crow.get("test_mmd_clamped", crow.get("test_mmd"))
        if a_mmd is None or c_mmd is None:
            continue
        mmd_delta = float(c_mmd) - float(a_mmd)
        pp_delta = metric_delta(crow, arow, "pearson_pert")
        condition_rows.append(
            {
                "dataset": ds,
                "condition": cond,
                "risk_dataset": ds in risk_datasets,
                "mmd_delta": mmd_delta,
                "pp_delta": pp_delta,
                "candidate_mmd": float(c_mmd),
                "anchor_mmd": float(a_mmd),
                "candidate_pp": crow.get("pearson_pert"),
                "anchor_pp": arow.get("pearson_pert"),
                "mmd_harm": mmd_delta > MMD_HARM_EPS,
            }
        )

    risk_harm_rows = [r for r in condition_rows if r["risk_dataset"] and r["mmd_harm"]]
    risk_harm_rows.sort(key=lambda r: (r["mmd_delta"], r["pp_delta"] or -999.0), reverse=True)

    dataset_rows = decision_row["metrics"]["dataset_rows"]
    risk_dataset_rows = [r for r in dataset_rows if r["dataset"] in risk_datasets]
    failing_risk_dataset_rows = [
        r for r in risk_dataset_rows if r["mean_mmd_delta"] > 0.005 or r["mmd_harm_rows"] > 0
    ]

    corrected_reasons = (
        (corrected.get("bug_fix") or {}).get("remaining_tian_norman_fail_reasons")
        or next(item["reasons"] for item in decision["decision"]["failed"] if item["name"] == RUN_NAME)
    )

    payload = {
        "status": "tian_norman_aggregate_positive_but_non_target_risk_gate_fail",
        "boundary": {
            "train_only_internal_posthoc": True,
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "trackc_query_read": False,
        },
        "source_decision_json": str(DECISION_JSON),
        "corrected_adjudication_json": str(CORRECTED_JSON) if corrected else None,
        "corrected_adjudication_status": corrected.get("status"),
        "run_name": RUN_NAME,
        "summary_metrics": decision_row["metrics"],
        "gate_failure_reasons": corrected_reasons,
        "risk_dataset_rows": risk_dataset_rows,
        "failing_risk_dataset_rows": failing_risk_dataset_rows,
        "top_risk_harm_conditions": risk_harm_rows[:25],
        "non_overlapping_preparation": {
            "do_not_launch_now": True,
            "reason": "Peirce audit is pending and current failure is a row-tail conflict, not a scalar gamma/replay question.",
            "proposed_next_bounded_hypothesis": (
                "If Peirce agrees the aggregate-positive tian-norman arm is worth mutating, "
                "the next GPU direction should be a default-off risk-row CVaR/top-k MMD hook "
                "that directly optimizes worst condition-level MMD rows over a predeclared risk "
                "dataset set, starting with CPU/unit validation and a capped 2k smoke only after "
                "external approval. Do not sweep scalar gamma or replay scope."
            ),
            "first_cpu_gate": (
                "Before any launch, build a no-GPU gate that checks whether the top harmed risk "
                "rows are stable, nontrivial in magnitude, and not already paired with Pearson "
                "gains that would make the current hard row-count gate overly conservative."
            ),
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Tian+Norman Harm Forensics",
        "",
        "Status: `tian_norman_aggregate_positive_but_non_target_risk_gate_fail`",
        "",
        "## Boundary",
        "",
        "- Reads completed train-only internal posthoc artifacts only.",
        "- Does not read canonical metrics, canonical multi, or Track C held-out query.",
        "",
        "## Aggregate Signal",
        "",
        "| metric | delta |",
        "|---|---:|",
    ]
    for key in [
        "cross_pp_delta_vs_anchor",
        "family_gene_pp_delta_vs_anchor",
        "family_gene_mmd_delta_vs_anchor",
        "target_dataset_mean_mmd_delta",
        "target_dataset_mean_pp_delta",
        "target_dataset_mmd_harm_rows",
        "risk_dataset_harm_count",
    ]:
        lines.append(f"| `{key}` | {fmt(decision_row['metrics'].get(key))} |")

    lines.extend(
        [
            "",
            "## Risk Dataset Rows",
            "",
            "| dataset | n | mean MMD delta | mean pp delta | MMD harm rows | failing current risk criterion |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    failing_names = {r["dataset"] for r in failing_risk_dataset_rows}
    for row in risk_dataset_rows:
        lines.append(
            f"| `{row['dataset']}` | {row['n']} | {fmt(row['mean_mmd_delta'])} | "
            f"{fmt(row['mean_pp_delta'])} | {row['mmd_harm_rows']} | "
            f"`{row['dataset'] in failing_names}` |"
        )

    lines.extend(
        [
            "",
            "## Top Risk Harm Conditions",
            "",
            "| dataset | condition | MMD delta | pp delta | candidate MMD | anchor MMD |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in risk_harm_rows[:15]:
        lines.append(
            f"| `{row['dataset']}` | `{row['condition']}` | {fmt(row['mmd_delta'])} | "
            f"{fmt(row['pp_delta'])} | {fmt(row['candidate_mmd'])} | {fmt(row['anchor_mmd'])} |"
        )

    lines.extend(
        [
            "",
        "## Preparation Note",
        "",
        "- Do not launch a scalar gamma/replay sweep from this result.",
        "- Corrected adjudication removed the target-harm-row bug; the remaining fail reason is broad non-target risk-dataset harm.",
        "- Proposed bounded hypothesis, if externally approved: default-off risk-row CVaR/top-k MMD over a predeclared risk dataset set, with CPU/unit validation first and only a capped 2k train-only smoke afterward.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
