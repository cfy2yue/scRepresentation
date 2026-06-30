#!/usr/bin/env python3
"""Summarize low-rank signflip/scaled-output internal diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _alpha_from_dir(path: Path) -> float:
    text = path.name.removeprefix("alpha_")
    text = text.replace("m", "-", 1) if text.startswith("m") else text
    text = text.replace("p", ".")
    return float(text)


def _read_summary(alpha_dir: Path) -> dict[str, Any]:
    path = alpha_dir / "posthoc" / "internal_eval_vs_anchor_summary.json"
    if not path.is_file():
        return {"alpha_dir": str(alpha_dir), "missing": True}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {
        "alpha": _alpha_from_dir(alpha_dir),
        "alpha_dir": str(alpha_dir),
        "status": payload.get("status"),
        "reasons": ";".join(map(str, payload.get("reasons") or [])),
        "report": str(alpha_dir / "posthoc" / "LATENTFM_LOOKAHEAD_TRUST_REGION_INTERNAL_EVAL_DECISION.md"),
    }
    for summary in payload.get("summaries") or []:
        group = str(summary.get("group"))
        prefix = "cross" if "cross_background" in group else "family" if "family" in group else group
        out[f"{prefix}_mean_delta_pp"] = summary.get("mean_delta_pearson_pert")
        out[f"{prefix}_dataset_min_delta_pp"] = summary.get("dataset_min_delta_pearson_pert")
        out[f"{prefix}_ci_low"] = summary.get("dataset_bootstrap_ci_low")
        out[f"{prefix}_mean_delta_mmd"] = summary.get("mean_delta_mmd_clamped")
    return out


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key)
        return default if value is None else float(value)
    except Exception:
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    rows = [_read_summary(path) for path in sorted(run_dir.glob("alpha_*")) if path.is_dir()]
    rows = sorted(rows, key=lambda r: _f(r, "alpha", 999.0))
    pass_rows = [row for row in rows if row.get("status") == "lookahead_trust_region_internal_eval_pass_needs_canonical_noharm"]
    negative_pass = [row for row in pass_rows if _f(row, "alpha", 0.0) < 0.0]
    if negative_pass:
        status = "lowrank_signflip_negative_alpha_internal_pass_needs_canonical_noharm"
        next_action = "launch frozen canonical single/family no-harm for the best negative-alpha checkpoint only"
    elif pass_rows:
        status = "lowrank_signflip_positive_or_neutral_internal_pass_diagnostic_only"
        next_action = "do not promote low-rank v1; inspect whether the pass is merely shrink-to-anchor"
    elif rows and all(not row.get("missing") for row in rows):
        status = "lowrank_signflip_all_alpha_internal_fail_close_family"
        next_action = "close low-rank residual v1 family and pivot to proxy-aligned objective/admission"
    else:
        status = "lowrank_signflip_incomplete"
        next_action = "wait for missing alpha posthoc outputs"

    out_csv = run_dir / "signflip_decision_rows.csv"
    out_json = run_dir / "signflip_decision.json"
    out_md = run_dir / "LATENTFM_LOWRANK_SIGNFLIP_INTERNAL_DECISION.md"
    write_csv(out_csv, rows)
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "status": status,
        "next_action": next_action,
        "run_dir": str(run_dir),
        "rows": rows,
        "pass_alpha_dirs": [row.get("alpha_dir") for row in pass_rows],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Low-Rank Signflip Internal Decision",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "Frozen checkpoint surgery diagnostic only. It scales the failed 5-step low-rank residual output by alpha and evaluates safe internal proxy groups. No training, canonical multi, or Track C query is used.",
        "",
        "## Rows",
        "",
        "| alpha | status | cross mean pp delta | family mean pp delta | report |",
        "|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.get('alpha')}` | `{row.get('status', 'missing')}` | "
            f"`{row.get('cross_mean_delta_pp', '')}` | `{row.get('family_mean_delta_pp', '')}` | "
            f"`{row.get('report', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- next action: {next_action}",
            "",
            "## Outputs",
            "",
            f"- JSON: `{out_json}`",
            f"- CSV: `{out_csv}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(out_md)}, indent=2), flush=True)
    return 0 if status != "lowrank_signflip_incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
