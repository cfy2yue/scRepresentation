#!/usr/bin/env python3
"""Two-seed branch decision for Track C shared-gene support-set smokes.

This script is intentionally posthoc-only. It reads query-free support-val
decision JSONs produced by ``summarize_latentfm_trackc_support_only_robustness``
for the seed42/seed43 support-set runs and synthesizes a branch-level decision.
It does not read held-out Track C query or canonical multi for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_RUNS = (
    "xverse_trackc_support_set_sharedgene_adapter_2k_seed42",
    "xverse_trackc_support_set_sharedgene_adapter_2k_seed43",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decision_path(run: str) -> Path:
    return ROOT / "reports" / f"latentfm_trackc_support_set_sharedgene_decision_{run}.json"


def key_delta(payload: dict[str, Any], key: str) -> float | None:
    row = (((payload.get("decision") or {}).get("key_rows") or {}).get(key) or {})
    value = row.get("delta_mean")
    return None if value is None else float(value)


def key_harm(payload: dict[str, Any], key: str) -> float | None:
    row = (((payload.get("decision") or {}).get("key_rows") or {}).get(key) or {})
    value = row.get("p_harm")
    return None if value is None else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "reports/latentfm_trackc_support_set_sharedgene_two_seed_decision_20260627.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_SHAREDGENE_TWO_SEED_DECISION_20260627.md",
    )
    args = parser.parse_args()

    inputs: dict[str, Path] = {run: decision_path(run) for run in args.runs}
    missing = {run: str(path) for run, path in inputs.items() if not path.is_file()}
    if missing:
        raise SystemExit(f"missing decision JSON(s): {missing}")

    payloads = {run: load_json(path) for run, path in inputs.items()}
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    pass_runs = 0
    for run, payload in payloads.items():
        boundary = payload.get("boundary") or {}
        if boundary.get("heldout_query_read") or boundary.get("canonical_multi_selection"):
            reasons.append(f"{run}:unsafe_boundary")
        decision = payload.get("decision") or {}
        status = str(decision.get("status") or "")
        if status == "trackc_support_only_robustness_pass_support_gate":
            pass_runs += 1
        else:
            reasons.append(f"{run}:{status or 'missing_status'}")
        row = {
            "run": run,
            "status": status,
            "actual_pp_delta": key_delta(payload, "actual_pp"),
            "actual_pp_p_harm": key_harm(payload, "actual_pp"),
            "actual_mmd_delta": key_delta(payload, "actual_mmd"),
            "family_pp_delta": key_delta(payload, "family_pp"),
            "family_mmd_delta": key_delta(payload, "family_mmd"),
            "zero_pp_delta": key_delta(payload, "zero_pp"),
            "shuffle_pp_delta": key_delta(payload, "shuffle_pp"),
            "absent_pp_delta": key_delta(payload, "absent_pp"),
            "reasons": decision.get("reasons") or [],
        }
        rows.append(row)

    if pass_runs == len(payloads) and not reasons:
        status = "trackc_support_set_sharedgene_two_seed_pass_external_audit_next_no_query"
        action = (
            "run external audit and prepare an uncapped/no-harm support-val protocol; "
            "held-out query remains forbidden until route/checkpoint is frozen"
        )
    elif pass_runs > 0:
        status = "trackc_support_set_sharedgene_mixed_seed_close_or_mutate_no_query"
        action = "inspect seed-specific failures and controls; do not query-evaluate"
    else:
        status = "trackc_support_set_sharedgene_two_seed_fail_close_no_query"
        action = "close this support-set token branch unless a concrete non-duplicate source/control mutation is proposed"

    out = {
        "status": status,
        "action": action,
        "boundary": {
            "heldout_query_read": False,
            "canonical_multi_selection": False,
            "input_decisions": {run: str(path) for run, path in inputs.items()},
        },
        "pass_runs": pass_runs,
        "n_runs": len(payloads),
        "decision_reasons": reasons,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Track C Shared-Gene Support-Set Two-Seed Decision",
        "",
        f"Status: `{status}`",
        f"Action: {action}",
        "",
        "## Boundary",
        "",
        "- Reads only query-free support-val decision JSONs.",
        "- Does not read held-out Track C query or canonical multi for selection.",
        "",
        "## Runs",
        "",
        "| run | status | actual pp | p_harm | actual MMD | family pp | family MMD | zero pp | shuffle pp | absent pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(v: Any) -> str:
            return "NA" if v is None else f"{float(v):+.6f}"
        lines.append(
            f"| `{row['run']}` | `{row['status']}` | {fmt(row['actual_pp_delta'])} | "
            f"{fmt(row['actual_pp_p_harm'])} | {fmt(row['actual_mmd_delta'])} | "
            f"{fmt(row['family_pp_delta'])} | {fmt(row['family_mmd_delta'])} | "
            f"{fmt(row['zero_pp_delta'])} | {fmt(row['shuffle_pp_delta'])} | "
            f"{fmt(row['absent_pp_delta'])} |"
        )
    lines.extend(["", "## Reasons", ""])
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines.extend(["", "## Outputs", "", f"* JSON: `{args.out_json}`"])
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
