#!/usr/bin/env python3
"""Summarize Track C pairwise latest-checkpoint posthoc decisions.

This is a read-only aggregator for the posthoc-only latest checkpoint gate.
It reads existing decision JSONs produced by
``summarize_latentfm_trackc_routed_distill_smoke_20260622.py`` and does not
read logs, tmux state, canonical query, or held-out Track C query outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUNS = [
    "xverse_trackc_noharm_pc_ep050_replay2_all_2k_seed42",
    "xverse_trackc_noharm_pc_ep050_replay4_nongm_2k_seed42",
    "xverse_trackc_noharm_pc_ep050del_replay2_all_2k_seed42",
    "xverse_trackc_noharm_pc_ep100_replay2_all_2k_seed42",
    "xverse_trackc_noharm_pc_ep100del_replay4_all_2k_seed42",
    "xverse_trackc_noharm_pc_ep100del_replay4_nongm_2k_seed42",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row(payload: dict[str, Any], table: str, group: str, metric: str) -> dict[str, Any] | None:
    return (payload.get("tables") or {}).get(table, {}).get(f"{group}:{metric}")


def usable_support_row(payload: dict[str, Any], metric: str) -> dict[str, Any] | None:
    for group in ("test_multi", "test"):
        found = row(payload, "support_split", group, metric)
        if not found:
            continue
        if found.get("status") == "ok" and int(found.get("n_matched_conditions") or 0) > 0:
            return found
    return row(payload, "support_split", "test", metric) or row(payload, "support_split", "test_multi", metric)


def f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: Any) -> str:
    value = f(value)
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def collect(report_dir: Path, runs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name in runs:
        path = report_dir / f"latentfm_trackc_pairwise_latest_decision_{run_name}.json"
        if not path.is_file():
            rows.append(
                {
                    "run": run_name,
                    "decision_path": str(path),
                    "present": False,
                    "status": "missing_decision",
                    "reasons": "missing_decision_json",
                }
            )
            continue
        payload = load_json(path)
        decision = payload.get("decision") or {}
        support_pp = usable_support_row(payload, "pearson_pert")
        support_mmd = usable_support_row(payload, "test_mmd_clamped")
        canon_single_pp = row(payload, "canonical_split", "test_single", "pearson_pert")
        canon_single_mmd = row(payload, "canonical_split", "test_single", "test_mmd_clamped")
        canon_family_pp = row(payload, "canonical_family", "family_gene", "pearson_pert")
        canon_family_mmd = row(payload, "canonical_family", "family_gene", "test_mmd_clamped")
        rows.append(
            {
                "run": run_name,
                "decision_path": str(path),
                "present": True,
                "status": decision.get("status", "missing_status"),
                "action": decision.get("action", ""),
                "reasons": ";".join(str(x) for x in decision.get("reasons") or []),
                "support_pp_delta": f(None if not support_pp else support_pp.get("delta_mean")),
                "support_pp_p_improve": f(None if not support_pp else support_pp.get("p_improvement")),
                "support_mmd_p_harm": f(None if not support_mmd else support_mmd.get("p_harm")),
                "canonical_single_pp_p_harm": f(None if not canon_single_pp else canon_single_pp.get("p_harm")),
                "canonical_single_mmd_p_harm": f(None if not canon_single_mmd else canon_single_mmd.get("p_harm")),
                "canonical_family_pp_p_harm": f(None if not canon_family_pp else canon_family_pp.get("p_harm")),
                "canonical_family_mmd_p_harm": f(None if not canon_family_mmd else canon_family_mmd.get("p_harm")),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for item in rows:
        for key in item:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    counts: dict[str, int] = {}
    for item in rows:
        counts[str(item.get("status"))] = counts.get(str(item.get("status")), 0) + 1
    pass_rows = [
        item
        for item in rows
        if item.get("status") == "trackc_smoke_support_pass_needs_uncapped_noharm_before_query"
    ]
    missing = [item for item in rows if not item.get("present")]
    if missing:
        bottom = "Latest-checkpoint decision set is incomplete; do not act on partial rows."
    elif pass_rows:
        bottom = (
            "At least one latest checkpoint passed the capped support/canonical smoke gate. "
            "This only authorizes protocol review for uncapped canonical no-harm, not query."
        )
    else:
        bottom = (
            "No latest checkpoint passed the capped support/canonical smoke gate. "
            "Close the checkpoint-selection branch and keep E-block scale-up disabled."
        )

    lines = [
        "# Track C Pairwise Latest-Checkpoint Decision Summary",
        "",
        "Read-only summary from existing latest-checkpoint posthoc decision JSONs.",
        "Held-out Track C query outputs are not read.",
        "",
        "## Bottom Line",
        "",
        bottom,
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| run | status | support pp delta | p improve | family pp harm | family mmd harm | reasons |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['run']}`",
                    f"`{item.get('status')}`",
                    fmt(item.get("support_pp_delta")),
                    fmt(item.get("support_pp_p_improve")),
                    fmt(item.get("canonical_family_pp_p_harm")),
                    fmt(item.get("canonical_family_mmd_p_harm")),
                    str(item.get("reasons") or ""),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Gate Consequence",
            "",
            "- A pass here is not a final multi claim and does not authorize held-out query.",
            "- If all rows fail, close the latest-checkpoint selection branch.",
            "- E-block 4k scale-up remains disabled unless a separate CPU/mechanism gate justifies it.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows = collect(args.report_dir, RUNS)
    missing = [item for item in rows if not item.get("present")]
    if missing and not args.allow_missing:
        names = ", ".join(str(item["run"]) for item in missing)
        raise SystemExit(f"missing decision JSONs: {names}")
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    print(json.dumps({"rows": len(rows), "missing": len(missing), "out_md": str(args.out_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
