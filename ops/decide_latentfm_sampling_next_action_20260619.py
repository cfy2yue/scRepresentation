#!/usr/bin/env python3
"""Decide the next LatentFM action from the sampling-smoke capped gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_A = "scf_prior010_inject_visitcap8_power05_floor32_4k"
RUN_B_OLD = "scf_prior010_inject_visitcap8_power05_floor32_dsloss05_4k"
RUN_B_CORRECTED = "scf_prior010_inject_visitcap8_power05_floor32_dsloss05_corrected_4k"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_run(gate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run")): row for row in gate.get("runs", []) if row.get("run")}


def _check(row: dict[str, Any], name: str) -> dict[str, Any]:
    checks = row.get("checks") or {}
    item = checks.get(name) or {}
    return item if isinstance(item, dict) else {}


def _status(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing"
    return str(row.get("triage_status") or "unknown")


def _passed(row: dict[str, Any] | None) -> bool:
    return bool(row) and _status(row) == "triage_pass_uncapped_required"


def _delta(row: dict[str, Any] | None, check: str) -> float | None:
    if not row:
        return None
    value = _check(row, check).get("delta")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decide(gate: dict[str, Any]) -> dict[str, Any]:
    rows = _by_run(gate)
    a = rows.get(RUN_A)
    b = rows.get(RUN_B_OLD)
    a_pass = _passed(a)
    b_pass = _passed(b)
    a_unseen2 = _delta(a, "test_multi_unseen2_pp")
    b_unseen2 = _delta(b, "test_multi_unseen2_pp")

    reasons: list[str] = []
    next_action = "wait_or_debug_missing_gate"
    priority = "pending"

    if a_pass:
        next_action = "run_uncapped_full_posthoc_for_valid_sampling_A"
        priority = "high"
        reasons.append(
            "Smoke A passed capped triage and is not affected by the old dataset-loss warmup ambiguity."
        )
        if b_pass:
            reasons.append(
                "Old smoke B also passed, but B was launched before the dataset-loss warmup fix; treat it as non-decisive until corrected rerun."
            )
        elif b_unseen2 is not None:
            reasons.append(
                f"Old smoke B did not pass capped triage; A is still the valid sampling signal. B unseen2 delta={b_unseen2:.6f}."
            )
    elif b_pass:
        next_action = "launch_corrected_dsloss_B_before_interpreting_dataset_loss"
        priority = "high"
        reasons.append(
            "Only old smoke B passed capped triage, but it was launched before ds_loss_warmup_start was fixed."
        )
        reasons.append(
            f"Launch prepared corrected run {RUN_B_CORRECTED} before making a dataset-loss claim."
        )
    elif a is not None or b is not None:
        if a_unseen2 is not None and a_unseen2 > 0:
            next_action = "try_stronger_condition_or_family_balance"
            priority = "medium"
            reasons.append(
                "Smoke A did not pass full capped triage but has positive unseen2 movement; stronger balance is more informative than blind scale-up."
            )
        else:
            next_action = "run_focus_learnability_diagnostic_or_stronger_balance"
            priority = "medium"
            reasons.append(
                "Neither valid smoke passed capped triage; choose stronger balance or Norman/Wessels/Gasperini focus diagnostic based on per-dataset failure pattern."
            )
        if b is not None:
            reasons.append(
                "Do not use old B to reject dataset-equal loss because ds_loss may not have been active in that run."
            )
    else:
        reasons.append("Gate JSON has no recognized sampling-smoke rows.")

    return {
        "baseline": gate.get("baseline"),
        "next_action": next_action,
        "priority": priority,
        "runs": {
            RUN_A: {
                "triage_status": _status(a),
                "unseen2_delta": a_unseen2,
                "overall_pp_delta": _delta(a, "overall_pp_non_regression"),
                "gene_pp_delta": _delta(a, "family_gene_pp_stable"),
                "mmd_ratio": (_check(a or {}, "overall_mmd_ratio").get("ratio") if a else None),
            },
            RUN_B_OLD: {
                "triage_status": _status(b),
                "unseen2_delta": b_unseen2,
                "overall_pp_delta": _delta(b, "overall_pp_non_regression"),
                "gene_pp_delta": _delta(b, "family_gene_pp_stable"),
                "mmd_ratio": (_check(b or {}, "overall_mmd_ratio").get("ratio") if b else None),
                "interpretation": "ambiguous_old_dataset_loss_run",
            },
        },
        "reasons": reasons,
        "prepared_corrected_run": RUN_B_CORRECTED,
        "rules": [
            "capped triage pass is not final promotion",
            "valid promotion requires uncapped full posthoc",
            "old dsloss05 run cannot prove dataset-loss benefit because it predates ds_loss_warmup_start fix",
            "do not launch corrected run unless the gate makes the old B ambiguity decision-blocking",
        ],
    }


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return str(x)


def _write_md(path: Path, payload: dict[str, Any], gate_path: Path) -> None:
    lines = [
        "# LatentFM Sampling-Smoke Next Action Decision",
        "",
        f"Gate JSON: `{gate_path}`",
        "",
        f"Next action: **{payload['next_action']}**",
        f"Priority: **{payload['priority']}**",
        "",
        "## Run Signals",
        "",
        "| run | status | unseen2 delta | overall pp delta | gene pp delta | MMD ratio | note |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for run, row in payload["runs"].items():
        lines.append(
            "| {run} | {status} | {u2} | {opp} | {gene} | {mmd} | {note} |".format(
                run=run,
                status=row.get("triage_status"),
                u2=_fmt(row.get("unseen2_delta")),
                opp=_fmt(row.get("overall_pp_delta")),
                gene=_fmt(row.get("gene_pp_delta")),
                mmd=_fmt(row.get("mmd_ratio")),
                note=row.get("interpretation", ""),
            )
        )
    lines.extend(["", "## Reasons", ""])
    for reason in payload["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(["", "## Rules", ""])
    for rule in payload["rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-json", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    gate = _load(args.gate_json)
    decision = decide(gate)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _write_md(args.out_md, decision, args.gate_json)
    print(json.dumps({"next_action": decision["next_action"], "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
