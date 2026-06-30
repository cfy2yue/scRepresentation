#!/usr/bin/env python3
"""Summarize Track C smoke decisions from manifest rows.

This is a read-only report helper. It reads only manifest JSONL files and the
corresponding smoke decision JSON artifacts under ``reports/``. It does not
inspect tmux, logs, exit-code files, or held-out query outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["_manifest"] = str(path)
        rows.append(row)
    return rows


def metric(payload: dict[str, Any], table: str, key: str, field: str) -> Any:
    return (((payload.get("tables") or {}).get(table) or {}).get(key) or {}).get(field)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def decision_path_for(row: dict[str, Any]) -> Path:
    if row.get("decision_json"):
        return Path(str(row["decision_json"]))
    run = str(row["run_name"])
    return ROOT / "reports" / f"latentfm_trackc_routed_distill_smoke_decision_{run}.json"


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    run = str(row["run_name"])
    decision_path = decision_path_for(row)
    out: dict[str, Any] = {
        "run_name": run,
        "manifest": row.get("_manifest", ""),
        "decision_json": str(decision_path),
        "decision_present": decision_path.is_file(),
        "status": "pending_decision",
        "action": "",
        "reasons": "",
        "support_pp_delta": "",
        "support_pp_p_improvement": "",
        "support_mmd_p_harm": "",
        "canonical_single_pp_p_harm": "",
        "canonical_family_pp_p_harm": "",
        "canonical_family_mmd_p_harm": "",
    }
    for key in (
        "forced_gpu",
        "finetune_trainable_scope",
        "pert_pairwise_mode",
        "endpoint_weight",
        "head_distill_weight",
        "anchor_replay_weight",
        "anchor_replay_filter",
        "hypothesis",
    ):
        if key in row:
            out[key] = row[key]
    if not decision_path.is_file():
        return out

    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    decision = payload.get("decision") or {}
    out.update(
        {
            "status": decision.get("status", ""),
            "action": decision.get("action", ""),
            "reasons": ";".join(str(x) for x in decision.get("reasons") or []),
            "support_pp_delta": metric(payload, "support_split", "test:pearson_pert", "delta_mean"),
            "support_pp_p_improvement": metric(payload, "support_split", "test:pearson_pert", "p_improvement"),
            "support_mmd_p_harm": metric(payload, "support_split", "test:test_mmd_clamped", "p_harm"),
            "canonical_single_pp_p_harm": metric(payload, "canonical_split", "test_single:pearson_pert", "p_harm"),
            "canonical_family_pp_p_harm": metric(payload, "canonical_family", "family_gene:pearson_pert", "p_harm"),
            "canonical_family_mmd_p_harm": metric(
                payload,
                "canonical_family",
                "family_gene:test_mmd_clamped",
                "p_harm",
            ),
        }
    )
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_name",
        "status",
        "action",
        "reasons",
        "support_pp_delta",
        "support_pp_p_improvement",
        "support_mmd_p_harm",
        "canonical_single_pp_p_harm",
        "canonical_family_pp_p_harm",
        "canonical_family_mmd_p_harm",
        "forced_gpu",
        "finetune_trainable_scope",
        "pert_pairwise_mode",
        "endpoint_weight",
        "head_distill_weight",
        "anchor_replay_weight",
        "anchor_replay_filter",
        "decision_present",
        "decision_json",
        "manifest",
        "hypothesis",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row.get("status", ""))] = status_counts.get(str(row.get("status", "")), 0) + 1
    lines = [
        "# Track C Manifest Decision Summary",
        "",
        "Read-only summary from manifest rows and existing smoke decision JSONs.",
        "Held-out query outputs are not read.",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
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
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('run_name', '')}`",
                    f"`{row.get('status', '')}`",
                    fmt(row.get("support_pp_delta")),
                    fmt(row.get("support_pp_p_improvement")),
                    fmt(row.get("canonical_family_pp_p_harm")),
                    fmt(row.get("canonical_family_mmd_p_harm")),
                    str(row.get("reasons", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Manifest JSONL path; repeatable")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for manifest_s in args.manifest:
        rows.extend(load_manifest(Path(manifest_s)))
    summary = [summarize_row(row) for row in rows]
    write_csv(summary, Path(args.out_csv))
    write_md(summary, Path(args.out_md))
    print(json.dumps({"status": "ok", "rows": len(summary), "out_csv": args.out_csv, "out_md": args.out_md}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
