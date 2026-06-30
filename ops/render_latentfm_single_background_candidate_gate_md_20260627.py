#!/usr/bin/env python3
"""Render paired single/background candidate gate JSON to Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--title", type=str, default="LatentFM Single/Background Candidate Gate")
    args = ap.parse_args()

    payload = json.loads(args.gate_json.read_text(encoding="utf-8"))
    gate = payload.get("gate") or {}
    lines = [
        f"# {args.title}",
        "",
        f"Status: `{gate.get('status')}`",
        "",
        "## Boundary",
        "",
        "- Frozen candidate checkpoint comparison against frozen anchor metrics.",
        "- Canonical multi is not used for selection.",
        "- Track C held-out query is not read.",
        "- Gate JSON is the authoritative machine-readable artifact.",
        "",
        "## Decision",
        "",
        f"Reasons: `{gate.get('reasons') or []}`",
        "",
        "Rules:",
        "",
    ]
    for rule in gate.get("rules") or []:
        lines.append(f"- {rule}")
    lines.extend(
        [
            "",
            "## Paired Deltas",
            "",
            "| stratum | metric | n | datasets | delta | CI95 | p_improve | p_harm |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in payload.get("paired_deltas") or []:
        ci = row.get("ci95") or [None, None]
        lines.append(
            f"| `{row.get('stratum')}` | `{row.get('metric')}` | "
            f"`{row.get('n_matched_conditions')}` | `{row.get('n_matched_datasets')}` | "
            f"{fmt(row.get('delta_mean'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
            f"{fmt(row.get('p_improve'))} | {fmt(row.get('p_harm'))} |"
        )
    lines.extend(["", "## JSON", "", f"`{args.gate_json}`", ""])
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "status": gate.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
