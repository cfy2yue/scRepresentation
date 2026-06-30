#!/usr/bin/env python3
"""CPU-only no-harm surrogate v2 gate for LatentFM scaling.

This is a fail-closed surrogate audit. With zero canonical no-harm positives,
the gate cannot authorize a new GPU candidate. It records whether existing
train-only internal features can at least support a high-risk veto for closed
families and what evidence would be required to reopen scaling GPU work.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
TRANSFER = REPORTS / "latentfm_scaling_noharm_transfer_calibration_gate_20260624.json"
FAILURE = REPORTS / "latentfm_scaling_failure_localization_gate_20260624.json"
LODO = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"

OUT_JSON = REPORTS / "latentfm_scaling_noharm_surrogate_v2_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_NOHARM_SURROGATE_V2_GATE_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[idx]


def main() -> int:
    transfer = load_json(TRANSFER)
    failure = load_json(FAILURE)
    lodo = load_json(LODO)
    candidates = list(transfer.get("candidates") or [])

    canonical_pass = [c for c in candidates if c.get("canonical_pass")]
    internal_pass_like = [c for c in candidates if c.get("internal_pass_like")]
    pp_hard = [c for c in candidates if float(c.get("canonical_max_pp_p_harm") or 0.0) >= 0.65]
    canonical_fail = [c for c in candidates if not c.get("canonical_pass")]

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_family[str(c.get("family") or "unknown")].append(c)

    family_rows = []
    closed_family_vetoes = []
    for family, rows in sorted(by_family.items()):
        n = len(rows)
        n_pass = sum(1 for r in rows if r.get("canonical_pass"))
        n_hard = sum(1 for r in rows if float(r.get("canonical_max_pp_p_harm") or 0.0) >= 0.65)
        max_pp_harm = max(float(r.get("canonical_max_pp_p_harm") or 0.0) for r in rows)
        med_internal = median(float(r.get("internal_score") or 0.0) for r in rows)
        row = {
            "family": family,
            "n": n,
            "n_canonical_pass": n_pass,
            "n_pp_hard_harm": n_hard,
            "median_internal_score": med_internal,
            "max_canonical_pp_p_harm": max_pp_harm,
            "veto": n_pass == 0 and (n_hard > 0 or max_pp_harm >= 0.57),
        }
        family_rows.append(row)
        if row["veto"]:
            closed_family_vetoes.append(family)

    internal_scores = [float(c.get("internal_score") or 0.0) for c in candidates]
    pp_harms = [float(c.get("canonical_max_pp_p_harm") or 0.0) for c in candidates]
    family_pp = [float(c.get("internal_family_pp_delta") or 0.0) for c in candidates]
    family_mmd = [float(c.get("internal_family_mmd_delta") or 0.0) for c in candidates]

    # Conservative one-class risk summary. These thresholds are not promotion
    # thresholds; they describe the range in which all observed candidates were
    # unsafe under frozen canonical no-harm.
    unsafe_envelope = {
        "internal_score_min": min(internal_scores) if internal_scores else None,
        "internal_score_median": median(internal_scores) if internal_scores else None,
        "internal_score_q90": quantile(internal_scores, 0.90),
        "internal_family_pp_min": min(family_pp) if family_pp else None,
        "internal_family_pp_median": median(family_pp) if family_pp else None,
        "internal_family_mmd_min": min(family_mmd) if family_mmd else None,
        "internal_family_mmd_max": max(family_mmd) if family_mmd else None,
        "canonical_pp_harm_min": min(pp_harms) if pp_harms else None,
        "canonical_pp_harm_median": median(pp_harms) if pp_harms else None,
    }

    reason_counts = Counter()
    for c in canonical_fail:
        reason_counts.update(map(str, c.get("canonical_gate_reasons") or []))

    reasons = []
    if not canonical_pass:
        reasons.append("zero_canonical_noharm_positive_examples")
    if len(internal_pass_like) == len(candidates) and canonical_fail:
        reasons.append("all_internal_pass_like_examples_are_canonical_failures")
    if len(pp_hard) >= max(1, len(candidates) // 2):
        reasons.append("canonical_pp_hard_harm_common")
    if closed_family_vetoes:
        reasons.append("closed_family_vetoes_available_but_not_promotional")
    if (lodo.get("status") or "").endswith("fail_no_gpu"):
        reasons.append("condition_count_lodo_gate_failed")
    if not canonical_pass:
        reasons.append("leave_family_out_surrogate_unlearnable_without_positive_class")

    status = "scaling_noharm_surrogate_v2_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports_only": True,
            "canonical_metrics_used_as_frozen_veto_context": True,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "summary": {
            "n_candidates": len(candidates),
            "n_internal_pass_like": len(internal_pass_like),
            "n_canonical_pass": len(canonical_pass),
            "n_canonical_fail": len(canonical_fail),
            "n_pp_hard_harm": len(pp_hard),
            "closed_family_vetoes": closed_family_vetoes,
            "failure_localization_status": failure.get("status"),
            "condition_count_lodo_status": lodo.get("status"),
            "spearman_internal_score_vs_pp_harm": (failure.get("summary") or {}).get("spearman_internal_score_vs_pp_harm")
            or (transfer.get("summary") or {}).get("spearman_internal_score_vs_canonical_max_pp_p_harm"),
        },
        "unsafe_envelope": unsafe_envelope,
        "family_rows": family_rows,
        "canonical_reason_counts": dict(reason_counts.most_common()),
        "reasons": reasons,
        "surrogate_use": {
            "can_authorize_gpu": False,
            "can_veto_closed_families": True,
            "veto_rule": "Do not relaunch same-family scaling variants whose train-only internal gains sit inside the observed unsafe envelope unless a new orthogonal no-harm mechanism is added and passes a fresh CPU gate.",
            "promotion_requirements": [
                "at least one query-blind canonical no-harm positive or exact no-op control family to calibrate specificity",
                "leave-family-out validation separating safe from unsafe candidates",
                "tail-aware train-only features not already consumed by closed cap/count/replay/response/risk-row families",
            ],
        },
        "next_action": "do not launch scaling GPU; use this surrogate only as a veto until a new tail-safe mechanism creates positive calibration evidence",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling No-Harm Surrogate V2 Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed reports.",
        "- Frozen canonical single/family metrics are used only as no-harm veto/calibration context.",
        "- Does not read canonical multi, held-out Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- candidates: `{len(candidates)}`",
        f"- internal-pass-like: `{len(internal_pass_like)}`",
        f"- canonical no-harm pass: `{len(canonical_pass)}`",
        f"- canonical failures: `{len(canonical_fail)}`",
        f"- pp hard-harm rows: `{len(pp_hard)}`",
        f"- closed family vetoes: `{closed_family_vetoes}`",
        f"- unsafe internal-score envelope: min/median/q90 `{fmt(unsafe_envelope['internal_score_min'])}` / `{fmt(unsafe_envelope['internal_score_median'])}` / `{fmt(unsafe_envelope['internal_score_q90'])}`",
        f"- canonical pp-harm min/median: `{fmt(unsafe_envelope['canonical_pp_harm_min'])}` / `{fmt(unsafe_envelope['canonical_pp_harm_median'])}`",
        "",
        "## Family Veto Rows",
        "",
        "| family | n | pass | pp hard-harm | median internal score | max pp p_harm | veto |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in family_rows:
        lines.append(
            f"| `{row['family']}` | {row['n']} | {row['n_canonical_pass']} | {row['n_pp_hard_harm']} | "
            f"{fmt(row['median_internal_score'])} | {fmt(row['max_canonical_pp_p_harm'])} | `{row['veto']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            "- surrogate use: veto only, not promotion.",
            "- next action: do not launch scaling GPU from current internal-pass signals; require a new orthogonal tail-safe mechanism or positive no-harm calibration evidence.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
