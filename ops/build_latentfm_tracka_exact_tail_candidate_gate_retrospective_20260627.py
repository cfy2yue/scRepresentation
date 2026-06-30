#!/usr/bin/env python3
"""Summarize retrospective exact-tail gates for closed Track A candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
GATE_DIR = ROOT / "reports/tracka_exact_tail_candidate_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_tracka_exact_tail_candidate_gate_retrospective_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_EXACT_TAIL_CANDIDATE_GATE_RETROSPECTIVE_20260627.md"

RUNS = [
    ("scfoundation_jiang_lowcount_seed42", "near-miss cross-latent Jiang-lowcount"),
    ("scfoundation_dataset_negative_seed42", "near-duplicate dataset-negative comparator"),
    ("xverse_softvisit_p085_no_cap_seed42", "soft-exposure mechanistic negative"),
    ("xverse_conddelta_seed42", "condition-delta tail-positive/MMD-risk"),
]
KEYS = [
    ("canonical_family_gene", "pearson_pert"),
    ("canonical_family_gene", "test_mmd_clamped"),
    ("exact_simple_single_unseen", "pearson_pert"),
    ("exact_simple_single_unseen", "test_mmd_clamped"),
    ("exact_cross_background_seen_gene", "pearson_pert"),
    ("exact_cross_background_seen_gene", "test_mmd_clamped"),
    ("recurrent_simple_hard_tail", "pearson_pert"),
    ("recurrent_cross_background_hard_tail", "pearson_pert"),
    ("recurrent_cross_background_hard_tail", "test_mmd_clamped"),
]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for name, label in RUNS:
        path = GATE_DIR / f"{name}.json"
        payload = load(path)
        lookup = {(r["group"], r["metric"]): r for r in payload["summaries"]}
        row = {
            "run": name,
            "label": label,
            "status": payload["status"],
            "gpu_authorized": payload["gpu_authorized"],
            "gate_reasons": payload["gate_reasons"],
            "path": str(path),
        }
        for group, metric in KEYS:
            rec = lookup.get((group, metric), {})
            row[f"{group}:{metric}:n"] = rec.get("n_conditions")
            row[f"{group}:{metric}:delta_mean"] = rec.get("delta_mean")
            row[f"{group}:{metric}:p_improve"] = rec.get("p_improve")
            row[f"{group}:{metric}:p_harm"] = rec.get("p_harm")
        rows.append(row)

    payload = {
        "status": "tracka_exact_tail_candidate_gate_retrospective_all_fail_no_gpu",
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "no_training": True,
            "no_inference": True,
            "no_checkpoint_selection": True,
            "canonical_multi_selection_weight": 0,
            "trackc_query_read": False,
        },
        "rows": rows,
        "outputs": {"json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track A Exact-Tail Candidate Gate Retrospective",
        "",
        "Status: `tracka_exact_tail_candidate_gate_retrospective_all_fail_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only retrospective over existing closed-candidate posthoc rows.",
        "- Does not train, infer, select checkpoints, read Track C query, or use canonical multi for selection.",
        "",
        "## Candidate Summary",
        "",
        "| run | status | reasons | exact simple pp | exact cross pp | recurrent simple pp | recurrent cross pp | recurrent cross MMD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        reasons = ", ".join(f"`{r}`" for r in row["gate_reasons"]) or "none"
        lines.append(
            f"| `{row['run']}` | `{row['status']}` | {reasons} | "
            f"{fmt(row.get('exact_simple_single_unseen:pearson_pert:delta_mean'))} | "
            f"{fmt(row.get('exact_cross_background_seen_gene:pearson_pert:delta_mean'))} | "
            f"{fmt(row.get('recurrent_simple_hard_tail:pearson_pert:delta_mean'))} | "
            f"{fmt(row.get('recurrent_cross_background_hard_tail:pearson_pert:delta_mean'))} | "
            f"{fmt(row.get('recurrent_cross_background_hard_tail:test_mmd_clamped:delta_mean'))} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- scFoundation near-miss variants improve exact cross-background means and MMD, but strongly harm recurrent cross-background hard tails.",
        "- xverse condition-delta improves recurrent cross-background hard tails, but fails exact simple-single no-harm and MMD no-harm.",
        "- soft-exposure is MMD-helpful but harms exact cross-background pp and recurrent tails.",
        "- No retrospective candidate satisfies the new exact-tail gate; these are mechanism clues, not GPU authorization.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- Gate directory: `{GATE_DIR}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
