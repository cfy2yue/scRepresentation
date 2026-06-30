#!/usr/bin/env python3
"""Decide the next LatentFM action from the focus-learnability audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUN_A = "scf_prior010_inject_nwg_focus_4k"
RUN_B = "scf_prior010_inject_nwg_focus_dsloss05_4k"
FOCUS_DATASETS = (
    "NormanWeissman2019_filtered",
    "Wessels",
    "GasperiniShendure2019_lowMOI",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_run(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run")): row for row in audit.get("runs", []) if row.get("run")}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_delta(row: dict[str, Any] | None, group: str, metric: str) -> float | None:
    if not row:
        return None
    item = row.get("groups", {}).get(group, {}).get(metric, {})
    if not isinstance(item, dict):
        return None
    return _float_or_none(item.get("delta"))


def _group_ratio(row: dict[str, Any] | None, group: str) -> float | None:
    if not row:
        return None
    item = row.get("groups", {}).get(group, {}).get("mmd_gate", {})
    if not isinstance(item, dict):
        return None
    return _float_or_none(item.get("ratio"))


def _dataset_delta(row: dict[str, Any] | None, dataset: str) -> float | None:
    if not row:
        return None
    for item in row.get("focus_dataset_unseen2", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("dataset") == dataset:
            return _float_or_none(item.get("unseen2_pp_delta"))
    return None


def _status(row: dict[str, Any] | None) -> str:
    if not row:
        return "missing"
    return str(row.get("status") or "unknown")


def _focus_signal(row: dict[str, Any] | None) -> bool:
    return _status(row) == "focus_learnability_signal"


def _wessels_rescued(row: dict[str, Any] | None) -> bool:
    delta = _dataset_delta(row, "Wessels")
    return delta is not None and delta >= 0.0


def _any_dataset_rescued(row: dict[str, Any] | None) -> bool:
    return any((_dataset_delta(row, ds) or -999.0) >= 0.0 for ds in FOCUS_DATASETS)


def _mmd_or_direct_moved_but_pp_did_not(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    unseen2_pp = _group_delta(row, "test_multi_unseen2", "pearson_pert")
    direct = _group_delta(row, "test_multi_unseen2", "direct_pearson")
    mmd_ratio = _group_ratio(row, "test_multi_unseen2")
    return (
        (unseen2_pp is None or unseen2_pp <= 0.0)
        and (
            (direct is not None and direct > 0.0)
            or (mmd_ratio is not None and mmd_ratio <= 1.0)
        )
    )


def decide(audit: dict[str, Any]) -> dict[str, Any]:
    rows = _by_run(audit)
    a = rows.get(RUN_A)
    b = rows.get(RUN_B)
    candidates = [row for row in (a, b) if row]
    invalid = [row for row in candidates if _status(row) == "invalid_selection_mismatch"]

    reasons: list[str] = []
    priority = "pending"
    next_action = "wait_or_debug_missing_focus_audit"

    if invalid:
        next_action = "rerun_or_reaudit_focus_posthoc_selection_mismatch"
        priority = "high"
        reasons.append(
            "At least one focus comparison has mismatched selected_conditions; do not use the diagnostic until the capped condition set is identical."
        )
    elif not candidates:
        reasons.append("Focus audit has no recognized focus runs.")
    else:
        signal_rows = [row for row in candidates if _focus_signal(row)]
        wessels_signal_rows = [row for row in signal_rows if _wessels_rescued(row)]
        if wessels_signal_rows:
            next_action = "launch_stronger_all_split_balance_4k"
            priority = "high"
            reasons.append(
                "Focus diagnostic rescued unseen2 and Wessels without excessive test MMD regression; exposure/balance is the next primary bottleneck to test on the full split."
            )
        elif signal_rows:
            next_action = "run_dataset_upper_bound_before_all_split_balance"
            priority = "medium"
            reasons.append(
                "Focus diagnostic has aggregate unseen2 signal but Wessels was not rescued; run dataset-specific upper-bound diagnostics before a full all-split long run."
            )
        elif any(_any_dataset_rescued(row) for row in candidates):
            next_action = "run_dataset_upper_bound_diagnostics"
            priority = "medium"
            reasons.append(
                "At least one focus dataset improved but the full focus gate did not pass; isolate Norman/Wessels/Gasperini learnability with single-dataset diagnostics."
            )
        elif any(_mmd_or_direct_moved_but_pp_did_not(row) for row in candidates):
            next_action = "run_norm_target_strata_or_residual_preprocessing_diagnostic"
            priority = "medium"
            reasons.append(
                "Focus MMD/direct movement without pp rescue points to target-norm or residual/preprocessing diagnostics before more sampling runs."
            )
        else:
            next_action = "prioritize_combo_aware_condition_modeling_or_latent_sensitivity"
            priority = "medium"
            reasons.append(
                "Focus did not rescue unseen2; further exposure tuning is unlikely to be decisive without condition-modeling or representation changes."
            )

    run_summaries = {}
    for name, row in ((RUN_A, a), (RUN_B, b)):
        run_summaries[name] = {
            "status": _status(row),
            "test_unseen2_pp_delta": _group_delta(row, "test_multi_unseen2", "pearson_pert"),
            "test_mmd_ratio": _group_ratio(row, "test"),
            "wessels_unseen2_pp_delta": _dataset_delta(row, "Wessels"),
            "norman_unseen2_pp_delta": _dataset_delta(row, "NormanWeissman2019_filtered"),
            "gasperini_unseen2_pp_delta": _dataset_delta(row, "GasperiniShendure2019_lowMOI"),
        }

    return {
        "baseline": audit.get("baseline"),
        "next_action": next_action,
        "priority": priority,
        "runs": run_summaries,
        "reasons": reasons,
        "rules": [
            "focus-only results are learnability diagnostics, not promotion evidence",
            "selected_conditions mismatch invalidates capped focus comparisons",
            "stronger all-split balance is justified only after focus unseen2 rescue, preferably including Wessels",
            "full claims still require all-split condition-uncapped/mmd-capped posthoc and condition-level bootstrap",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _write_md(path: Path, payload: dict[str, Any], audit_path: Path) -> None:
    lines = [
        "# LatentFM Focus Next Action Decision",
        "",
        f"Audit JSON: `{audit_path}`",
        "",
        f"Next action: **{payload['next_action']}**",
        f"Priority: **{payload['priority']}**",
        "",
        "## Run Signals",
        "",
        "| run | status | unseen2 delta | Wessels delta | Norman delta | Gasperini delta | test MMD ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run, row in payload["runs"].items():
        lines.append(
            "| {run} | {status} | {u2} | {w} | {n} | {g} | {mmd} |".format(
                run=f"`{run}`",
                status=row.get("status"),
                u2=_fmt(row.get("test_unseen2_pp_delta")),
                w=_fmt(row.get("wessels_unseen2_pp_delta")),
                n=_fmt(row.get("norman_unseen2_pp_delta")),
                g=_fmt(row.get("gasperini_unseen2_pp_delta")),
                mmd=_fmt(row.get("test_mmd_ratio")),
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
    ap.add_argument("--audit-json", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    audit = _load(args.audit_json)
    decision = decide(audit)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    _write_md(args.out_md, decision, args.audit_json)
    print(json.dumps({"next_action": decision["next_action"], "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
