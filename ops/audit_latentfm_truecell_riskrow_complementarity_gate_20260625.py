#!/usr/bin/env python3
"""CPU-only gate for true-cell / risk-row tail complementarity.

The question is whether the failed risk-row CVaR mechanism protects the
canonical row tails of the true-cell budget128 route strongly enough to justify
a new combined GPU smoke. This script is intentionally conservative: it uses
frozen canonical posthoc artifacts only as a failure-analysis veto/context, not
for selecting a checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
ANCHOR = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_canonical/condition_family_eval_anchor_ode20_canonical.json"
RISK = ROOT / "runs/latentfm_risk_row_cvar_canonical_noharm_20260624/xverse_risk_row_cvar_allrisk_w020_2k_seed42/posthoc_eval_canonical/condition_family_eval_candidate_ode20_canonical.json"
TRUECELL = [
    ROOT / f"runs/latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625/xverse_truecell_nested_budget128_tailstable_seed{seed}_6000/posthoc_eval_canonical/condition_family_eval_candidate_ode20_canonical.json"
    for seed in (42, 43, 44)
]

OUT_JSON = ROOT / "reports/latentfm_truecell_riskrow_complementarity_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUECELL_RISKROW_COMPLEMENTARITY_GATE_20260625.md"
OUT_CSV = ROOT / "reports/latentfm_truecell_riskrow_complementarity_rows_20260625.csv"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def rows_by_key(blob: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = blob.get("groups", {}).get(group, {}).get("condition_metrics", [])
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if ds and cond:
            out[(ds, cond)] = row
    return out


def metric_delta(row: dict[str, Any], anchor: dict[str, Any], metric: str) -> float | None:
    a = finite(anchor.get(metric))
    b = finite(row.get(metric))
    if a is None or b is None:
        return None
    return b - a


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    anchor = load_json(ANCHOR)
    risk = load_json(RISK)
    truecells = [load_json(path) for path in TRUECELL]

    row_records: list[dict[str, Any]] = []
    groups_out: dict[str, Any] = {}
    for group in ("test_single", "family_gene"):
        anchor_rows = rows_by_key(anchor, group)
        risk_rows = rows_by_key(risk, group)
        true_rows = [rows_by_key(blob, group) for blob in truecells]
        common = set(anchor_rows) & set(risk_rows)
        for tr in true_rows:
            common &= set(tr)

        pp_true_mean_values: list[float] = []
        pp_risk_values: list[float] = []
        mmd_true_mean_values: list[float] = []
        mmd_risk_values: list[float] = []
        true_tail_rows = 0
        risk_protect_rows = 0
        shared_harm_rows = 0
        risk_new_harm_rows = 0
        any_true_seed_severe_rows = 0

        for ds, cond in sorted(common):
            arow = anchor_rows[(ds, cond)]
            risk_pp = metric_delta(risk_rows[(ds, cond)], arow, "pearson_pert")
            risk_mmd = metric_delta(risk_rows[(ds, cond)], arow, "test_mmd_clamped")
            true_pps = [metric_delta(tr[(ds, cond)], arow, "pearson_pert") for tr in true_rows]
            true_mmds = [metric_delta(tr[(ds, cond)], arow, "test_mmd_clamped") for tr in true_rows]
            if risk_pp is None or risk_mmd is None or any(x is None for x in true_pps + true_mmds):
                continue
            true_pp_values = [float(x) for x in true_pps if x is not None]
            true_mmd_values = [float(x) for x in true_mmds if x is not None]
            true_pp_mean = mean(true_pp_values)
            true_mmd_mean = mean(true_mmd_values)
            pp_true_mean_values.append(true_pp_mean)
            pp_risk_values.append(float(risk_pp))
            mmd_true_mean_values.append(true_mmd_mean)
            mmd_risk_values.append(float(risk_mmd))

            true_tail = true_pp_mean < -0.05 or min(true_pp_values) < -0.10 or true_mmd_mean > 0.010
            risk_protect = true_tail and float(risk_pp) >= true_pp_mean + 0.02 and float(risk_pp) >= -0.02 and float(risk_mmd) <= true_mmd_mean + 0.002
            shared_harm = true_tail and (float(risk_pp) < -0.02 or float(risk_mmd) > 0.010)
            risk_new_harm = (not true_tail) and (float(risk_pp) < -0.05 or float(risk_mmd) > 0.010)
            any_seed_severe = min(true_pp_values) < -0.10

            true_tail_rows += int(true_tail)
            risk_protect_rows += int(risk_protect)
            shared_harm_rows += int(shared_harm)
            risk_new_harm_rows += int(risk_new_harm)
            any_true_seed_severe_rows += int(any_seed_severe)

            row_records.append(
                {
                    "group": group,
                    "dataset": ds,
                    "condition": cond,
                    "truecell_pp_delta_mean": true_pp_mean,
                    "truecell_mmd_delta_mean": true_mmd_mean,
                    "riskrow_pp_delta": float(risk_pp),
                    "riskrow_mmd_delta": float(risk_mmd),
                    "true_tail": true_tail,
                    "risk_protect": risk_protect,
                    "shared_harm": shared_harm,
                    "risk_new_harm": risk_new_harm,
                    "any_true_seed_severe": any_seed_severe,
                }
            )

        protect_fraction = risk_protect_rows / true_tail_rows if true_tail_rows else 0.0
        shared_fraction = shared_harm_rows / true_tail_rows if true_tail_rows else 0.0
        new_harm_fraction = risk_new_harm_rows / max(1, len(pp_risk_values) - true_tail_rows)
        pass_gate = (
            true_tail_rows >= 20
            and protect_fraction >= 0.60
            and shared_fraction <= 0.20
            and new_harm_fraction <= 0.10
            and summarize(pp_risk_values).get("mean", -1.0) >= -0.005
        )
        groups_out[group] = {
            "common_rows": len(pp_risk_values),
            "true_tail_rows": true_tail_rows,
            "risk_protect_rows": risk_protect_rows,
            "risk_protect_fraction": protect_fraction,
            "shared_harm_rows": shared_harm_rows,
            "shared_harm_fraction": shared_fraction,
            "risk_new_harm_rows": risk_new_harm_rows,
            "risk_new_harm_fraction": new_harm_fraction,
            "any_true_seed_severe_rows": any_true_seed_severe_rows,
            "truecell_pp_delta_mean_summary": summarize(pp_true_mean_values),
            "riskrow_pp_delta_summary": summarize(pp_risk_values),
            "truecell_mmd_delta_mean_summary": summarize(mmd_true_mean_values),
            "riskrow_mmd_delta_summary": summarize(mmd_risk_values),
            "pass_gate": pass_gate,
        }

    reasons: list[str] = []
    if not all(item["pass_gate"] for item in groups_out.values()):
        reasons.append("risk_row_does_not_protect_truecell_canonical_tails_across_required_groups")
    if any(item["shared_harm_fraction"] > 0.20 for item in groups_out.values()):
        reasons.append("risk_row_and_truecell_share_too_many_harm_tail_rows")
    if any(item["risk_new_harm_fraction"] > 0.10 for item in groups_out.values()):
        reasons.append("risk_row_introduces_non_truecell_tail_harm")
    if any(item["riskrow_pp_delta_summary"].get("mean", -1.0) < -0.005 for item in groups_out.values()):
        reasons.append("risk_row_aggregate_row_pp_delta_is_negative")
    if not reasons:
        reasons.append("gate_passed_cpu_only_external_review_required_before_gpu")

    gpu_authorized = False
    status = "truecell_riskrow_complementarity_pass_cpu_review_next_no_gpu" if all(item["pass_gate"] for item in groups_out.values()) else "truecell_riskrow_complementarity_fail_no_gpu"

    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_only": True,
            "reads_frozen_canonical_posthoc_for_failure_analysis": True,
            "uses_canonical_for_checkpoint_selection": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
        },
        "inputs": {
            "anchor": str(ANCHOR),
            "risk_row": str(RISK),
            "truecell": [str(p) for p in TRUECELL],
        },
        "groups": groups_out,
        "reasons": reasons,
        "next_action": (
            "do not launch combined truecell+riskrow GPU; require a genuinely new tail-protection mechanism"
            if status.endswith("fail_no_gpu")
            else "external review of combo mechanism and launcher gate before any bounded GPU"
        ),
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "group",
            "dataset",
            "condition",
            "truecell_pp_delta_mean",
            "truecell_mmd_delta_mean",
            "riskrow_pp_delta",
            "riskrow_mmd_delta",
            "true_tail",
            "risk_protect",
            "shared_harm",
            "risk_new_harm",
            "any_true_seed_severe",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_records)

    lines = [
        "# LatentFM True-Cell / Risk-Row Complementarity Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU-only row-level failure-analysis gate over completed frozen posthoc artifacts.",
        "- Canonical single/family rows are used only as frozen no-harm failure context.",
        "- No canonical multi, Track C query, training, inference, or GPU.",
        "",
        "## Group Summary",
        "",
        "| group | common rows | true tail rows | protected | protect frac | shared harm frac | new harm frac | risk pp mean | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, item in groups_out.items():
        lines.append(
            "| {group} | {common_rows} | {true_tail_rows} | {risk_protect_rows} | {risk_protect_fraction:.3f} | {shared_harm_fraction:.3f} | {risk_new_harm_fraction:.3f} | {risk_pp_mean:+.6f} | `{pass_gate}` |".format(
                group=group,
                common_rows=item["common_rows"],
                true_tail_rows=item["true_tail_rows"],
                risk_protect_rows=item["risk_protect_rows"],
                risk_protect_fraction=item["risk_protect_fraction"],
                shared_harm_fraction=item["shared_harm_fraction"],
                risk_new_harm_fraction=item["risk_new_harm_fraction"],
                risk_pp_mean=item["riskrow_pp_delta_summary"].get("mean", float("nan")),
                pass_gate=item["pass_gate"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- row CSV: `{OUT_CSV}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
