#!/usr/bin/env python3
"""Formal CPU gate for Track C train_multi row-reliability V2.

This consumes the newly materialized train_multi row-level LOO reliability
artifact and decides whether any train-only reliability spec is clean enough to
authorize a GPU follow-up. It is deliberately query-blind and canonical-blind.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
ARTIFACT = REPORTS / "latentfm_trackc_trainmulti_row_reliability_artifact_20260624.json"
ARTIFACT_GATE = REPORTS / "latentfm_trackc_row_reliability_v2_artifact_gate_20260624.json"
OUT_JSON = REPORTS / "latentfm_trackc_row_reliability_v2_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_ROW_RELIABILITY_V2_GATE_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    artifact = load_json(ARTIFACT)
    old_gate = load_json(ARTIFACT_GATE)

    spec_tables = list(artifact.get("spec_tables") or [])
    pass_specs = [row for row in spec_tables if (row.get("gate") or {}).get("pass")]
    best = artifact.get("best_spec") or {}
    best_gate = best.get("gate") or {}
    reason_counts: Counter[str] = Counter()
    for row in spec_tables:
        gate = row.get("gate") or {}
        reason_counts.update(gate.get("reasons") or [])

    reasons: list[str] = []
    if int(artifact.get("n_pass_specs_train_only") or 0) <= 0:
        reasons.append("zero_train_only_pass_specs")
    if int(best_gate.get("enabled_negative_rows") or 0) > 2:
        reasons.append("best_spec_too_many_negative_enabled_rows")
    if float(best_gate.get("enabled_min_pp_delta") or 0.0) < -0.02:
        reasons.append("best_spec_min_enabled_pp_too_negative")
    if float(best_gate.get("pp_delta") or 0.0) < 0.03:
        reasons.append("best_spec_pp_below_material_threshold")
    if float(best_gate.get("p_harm") or 1.0) > 0.20:
        reasons.append("best_spec_p_harm_above_threshold")
    if float(best_gate.get("norman_pp") or 0.0) < -0.01:
        reasons.append("best_spec_norman_tail_below_threshold")
    if float(best_gate.get("wessels_pp") or 0.0) < 0.02:
        reasons.append("best_spec_wessels_below_threshold")

    status = "trackc_row_reliability_v2_fail_no_gpu"
    gpu_authorized = False
    if pass_specs:
        status = "trackc_row_reliability_v2_pass_external_review_next"

    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_only": True,
            "reads_safe_trainselect_split": True,
            "selection_role": "train_multi_leave_one_condition_only",
            "support_val_scoring": False,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "inputs": {
            "artifact": str(ARTIFACT),
            "prior_artifact_sufficiency_gate": str(ARTIFACT_GATE),
        },
        "summary": {
            "n_specs": artifact.get("n_specs"),
            "n_train_multi_rows": artifact.get("n_train_multi_rows"),
            "n_support_val_rows_metadata_only": artifact.get("n_support_val_rows_metadata_only"),
            "n_pass_specs_train_only": artifact.get("n_pass_specs_train_only"),
            "best_spec": best.get("spec"),
            "best_gate": best_gate,
            "reason_counts": dict(reason_counts.most_common()),
            "old_artifact_gate_status": old_gate.get("status"),
        },
        "reasons": reasons,
        "decision": {
            "trackc_query_allowed": False,
            "canonical_noharm_allowed": False,
            "gpu_next_action": "none",
            "next_action": "close Track C row-reliability V2 before GPU; require a materially new train-only tail-safe support route",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top_reasons = ", ".join(f"{name}={count}" for name, count in reason_counts.most_common(6))
    lines = [
        "# LatentFM Track C Row-Reliability V2 Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only formal gate using the safe trainselect train_multi row-level artifact.",
        "- Train-only selection role: leave-one-condition reliability rows from `train_multi`.",
        "- Does not score support_val for selection, read canonical metrics, canonical multi, held-out Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- specs tested: `{artifact.get('n_specs')}`",
        f"- train_multi rows: `{artifact.get('n_train_multi_rows')}`",
        f"- support_val rows counted as metadata only: `{artifact.get('n_support_val_rows_metadata_only')}`",
        f"- train-only pass specs: `{artifact.get('n_pass_specs_train_only')}`",
        f"- best spec: `{best.get('spec')}`",
        f"- best pp delta / p_harm: `{fmt(best_gate.get('pp_delta'))}` / `{fmt(best_gate.get('p_harm'))}`",
        f"- best enabled rows / negative rows / min pp: `{best_gate.get('enabled_rows')}` / `{best_gate.get('enabled_negative_rows')}` / `{fmt(best_gate.get('enabled_min_pp_delta'))}`",
        f"- best Norman / Wessels pp: `{fmt(best_gate.get('norman_pp'))}` / `{fmt(best_gate.get('wessels_pp'))}`",
        f"- top failure reasons: `{top_reasons}`",
        "",
        "## Decision",
        "",
        f"- reasons: `{reasons}`",
        "- GPU authorized: `False`",
        "- canonical no-harm authorized: `False`",
        "- Track C held-out query authorized: `False`",
        "- close row-reliability V2 before GPU; it is not tail-safe even in train-only rows.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": gpu_authorized}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
