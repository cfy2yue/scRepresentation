#!/usr/bin/env python3
"""Background-interaction closure for SciPlex dose-specific evidence.

CPU/report-only. Reads the completed SciPlex dose-specific outcome gate and
tests whether dose response is robust across backgrounds. No training,
inference, checkpoint selection, canonical multi selection, Track C query, or
GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
IN_JSON = ROOT / "reports/latentfm_sciplex_dose_specific_outcome_gate_20260627.json"
OUT_JSON = ROOT / "reports/latentfm_sciplex_dose_background_interaction_gate_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_SCIPLEX_DOSE_BACKGROUND_INTERACTION_GATE_20260628.md"


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    src = json.loads(IN_JSON.read_text(encoding="utf-8"))
    gate = src.get("gate", {})
    ds_summary = gate.get("dataset_summary", {}) or {}
    background_rows = []
    passing_backgrounds = []
    for ds, row in sorted(ds_summary.items()):
        pp = row.get("pp_mean")
        mmd = row.get("mmd_mean")
        n_pairs = int(row.get("n_pairs", 0) or 0)
        reasons = []
        if n_pairs < 50:
            reasons.append("pairs_below_50")
        if pp is None or float(pp) < 0.030:
            reasons.append("pp_highlow_below_0p030")
        if mmd is None or float(mmd) > 0.001:
            reasons.append("mmd_highlow_above_0p001")
        status = "background_pass" if not reasons else "background_fail"
        if status == "background_pass":
            passing_backgrounds.append(ds)
        background_rows.append(
            {
                "dataset": ds,
                "status": status,
                "n_pairs": n_pairs,
                "pp_high_minus_low": pp,
                "mmd_high_minus_low": mmd,
                "reasons": reasons,
            }
        )

    reasons = []
    if len(background_rows) < 3:
        reasons.append("background_count_below_3")
    if len(passing_backgrounds) < 2:
        reasons.append("passing_backgrounds_below_2")
    pp_boot = gate.get("pp_high_minus_low_bootstrap", {}) or {}
    mmd_boot = gate.get("mmd_high_minus_low_bootstrap", {}) or {}
    if pp_boot.get("ci_low") is None or float(pp_boot["ci_low"]) <= 0:
        reasons.append("global_pp_ci_low_not_above_0")
    if mmd_boot.get("mean") is None or float(mmd_boot["mean"]) > 0.001:
        reasons.append("global_mmd_highlow_above_0p001")
    if "dataset_min_pp_below_minus_0p020" in (gate.get("reasons") or []):
        reasons.append("global_dataset_min_pp_below_minus_0p020")
    status = "sciplex_dose_background_interaction_pass_external_review_only_no_gpu" if not reasons else "sciplex_dose_background_interaction_fail_close_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "input_gate": str(IN_JSON),
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query": False,
            "gpu": False,
        },
        "global_gate_status": gate.get("status"),
        "global_reasons": gate.get("reasons", []),
        "passing_backgrounds": passing_backgrounds,
        "background_rows": background_rows,
        "reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM SciPlex Dose Background Interaction Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over the completed SciPlex dose-specific outcome gate.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Global Dose Gate",
        "",
        f"- source status: `{gate.get('status')}`",
        f"- within-drug pairs: `{gate.get('within_drug_pairs')}`",
        f"- global pp high-low: `{fmt(pp_boot.get('mean'))}` CI `[{fmt(pp_boot.get('ci_low'))}, {fmt(pp_boot.get('ci_high'))}]`",
        f"- global MMD high-low: `{fmt(mmd_boot.get('mean'))}`",
        f"- source reasons: `{gate.get('reasons', [])}`",
        "",
        "## Background Interaction",
        "",
        "| background | status | pairs | pp high-low | MMD high-low | reasons |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in background_rows:
        lines.append(
            f"| `{row['dataset']}` | `{row['status']}` | {row['n_pairs']} | {fmt(row['pp_high_minus_low'])} | {fmt(row['mmd_high_minus_low'])} | `{row['reasons']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- passing backgrounds: `{passing_backgrounds}`",
            f"- decision reasons: `{reasons}`",
            "- The observed positive dose signal is not robust across backgrounds; do not launch SciPlex dose-aware training without a materially new source or exact Chemical V2 ACK.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "passing_backgrounds": passing_backgrounds, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
