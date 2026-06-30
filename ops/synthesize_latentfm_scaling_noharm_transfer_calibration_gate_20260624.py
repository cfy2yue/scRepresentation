#!/usr/bin/env python3
"""Retrospective no-harm transfer calibration gate for scaling candidates.

This CPU-only gate asks whether train-only/internal metrics are currently a
reliable surrogate for frozen canonical no-harm. It reads completed internal
and canonical no-harm reports only. Canonical metrics are used as retrospective
veto/calibration evidence, not for choosing a new checkpoint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_scaling_noharm_transfer_calibration_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_NOHARM_TRANSFER_CALIBRATION_GATE_20260624.md"


def load_json(name: str) -> dict[str, Any]:
    path = REPORTS / name
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def rank(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            out[order[k]] = avg
        i = j
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def by_name(payload: dict[str, Any], key: str = "name") -> dict[str, dict[str, Any]]:
    return {str(row.get(key) or row.get("run") or ""): row for row in payload.get("rows") or []}


def metric_delta(row: dict[str, Any], metric: str) -> float | None:
    m = (row.get("metrics") or {}).get(metric) or {}
    value = m.get("delta_mean")
    return float(value) if value is not None else None


def metric_pharm(row: dict[str, Any], metric: str) -> float | None:
    m = (row.get("metrics") or {}).get(metric) or {}
    value = m.get("p_harm")
    return float(value) if value is not None else None


def canonical_pass(row: dict[str, Any]) -> bool:
    status = str(row.get("gate_status") or row.get("status") or "")
    reasons = row.get("gate_reasons") or []
    if "pass" in status and "fail" not in status and not reasons:
        return True
    return False


def canonical_summary(row: dict[str, Any]) -> dict[str, Any]:
    cross = metric_delta(row, "cross_background_seen_gene:pearson_pert")
    all_pp = metric_delta(row, "all_test_single:pearson_pert")
    fam_pp = metric_delta(row, "family_gene:pearson_pert")
    fam_mmd = metric_delta(row, "family_gene:test_mmd_clamped")
    p_harms = [
        metric_pharm(row, "cross_background_seen_gene:pearson_pert"),
        metric_pharm(row, "all_test_single:pearson_pert"),
        metric_pharm(row, "family_gene:pearson_pert"),
        metric_pharm(row, "family_gene:test_mmd_clamped"),
    ]
    p_harms = [v for v in p_harms if v is not None]
    return {
        "canonical_pass": canonical_pass(row),
        "canonical_cross_pp_delta": cross,
        "canonical_all_single_pp_delta": all_pp,
        "canonical_family_pp_delta": fam_pp,
        "canonical_family_mmd_delta": fam_mmd,
        "canonical_max_pp_p_harm": max(
            [
                v
                for v in [
                    metric_pharm(row, "cross_background_seen_gene:pearson_pert"),
                    metric_pharm(row, "all_test_single:pearson_pert"),
                    metric_pharm(row, "family_gene:pearson_pert"),
                ]
                if v is not None
            ],
            default=None,
        ),
        "canonical_max_any_p_harm": max(p_harms, default=None),
        "canonical_gate_status": row.get("gate_status") or row.get("status"),
        "canonical_gate_reasons": row.get("gate_reasons") or [],
    }


def add_candidate(rows: list[dict[str, Any]], *, run: str, family: str, internal: dict[str, Any], canonical: dict[str, Any], evidence: list[str]) -> None:
    cross = internal.get("cross_pp_delta")
    fam = internal.get("family_pp_delta")
    mmd = internal.get("family_mmd_delta")
    internal_pass_like = (
        cross is not None
        and fam is not None
        and mmd is not None
        and float(cross) >= 0.005
        and float(fam) >= 0.005
        and float(mmd) <= 0.001
    )
    score = None
    if cross is not None and fam is not None and mmd is not None:
        score = float(cross) + 0.5 * float(fam) - max(0.0, float(mmd)) * 5.0
    item = {
        "run": run,
        "family": family,
        "internal_cross_pp_delta": cross,
        "internal_family_pp_delta": fam,
        "internal_family_mmd_delta": mmd,
        "internal_pass_like": internal_pass_like,
        "internal_score": score,
        "evidence": evidence,
    }
    item.update(canonical_summary(canonical))
    rows.append(item)


def first_group_delta(payload: dict[str, Any], group: str) -> dict[str, Any]:
    for row in payload.get("split_group_deltas") or []:
        if row.get("group") == group:
            return row
    for row in payload.get("family_group_deltas") or []:
        if row.get("group") == group:
            return row
    return {}


def main() -> int:
    candidates: list[dict[str, Any]] = []

    count = load_json("latentfm_xverse_scaling_count_smokes_decision_20260624.json")
    count_canon = load_json("latentfm_xverse_scaling_canonical_noharm_decision_20260624.json")
    count_rows = by_name(count)
    count_canon_rows = by_name(count_canon, "run")
    for run in [
        "xverse_scaling_cap120_all_3k_seed42",
        "xverse_scaling_gene_cap120_allbg_3k_seed42",
        "xverse_scaling_gene_cap120_k562bg_3k_seed42",
    ]:
        row = count_rows.get(run, {})
        groups = row.get("groups") or {}
        add_candidate(
            candidates,
            run=run,
            family="count_gene_background",
            internal={
                "cross_pp_delta": (groups.get("internal_val_cross_background_seen_gene_proxy") or {}).get("delta_pearson_pert"),
                "family_pp_delta": (groups.get("internal_val_family_gene_proxy") or {}).get("delta_pearson_pert"),
                "family_mmd_delta": (groups.get("internal_val_family_gene_proxy") or {}).get("delta_mmd"),
            },
            canonical=count_canon_rows.get(run, {}),
            evidence=[
                "reports/LATENTFM_XVERSE_SCALING_COUNT_SMOKES_DECISION_20260624.md",
                "reports/LATENTFM_XVERSE_SCALING_CANONICAL_NOHARM_DECISION_20260624.md",
            ],
        )

    protocol = load_json("latentfm_scaling_protocol_matrix_decision_20260624.json")
    protocol_canon = load_json("latentfm_scaling_protocol_canonical_noharm_decision_20260624.json")
    protocol_row = by_name(protocol).get("xverse_scaling_protocol_cap60_primary19_3k_seed42", {})
    add_candidate(
        candidates,
        run="xverse_scaling_protocol_cap60_primary19_3k_seed42",
        family="protocol_cap60",
        internal={
            "cross_pp_delta": (protocol_row.get("metrics") or {}).get("cross_pp_delta_vs_anchor"),
            "family_pp_delta": (protocol_row.get("metrics") or {}).get("family_gene_pp_delta_vs_anchor"),
            "family_mmd_delta": (protocol_row.get("metrics") or {}).get("family_gene_mmd_delta_vs_anchor"),
        },
        canonical=by_name(protocol_canon, "run").get("xverse_scaling_protocol_cap60_primary19_3k_seed42", {}),
        evidence=[
            "reports/LATENTFM_SCALING_PROTOCOL_MATRIX_DECISION_20260624.md",
            "reports/LATENTFM_SCALING_PROTOCOL_CANONICAL_NOHARM_DECISION_20260624.md",
        ],
    )

    high = load_json("latentfm_scaling_highthroughput_smokes_decision_20260624.json")
    high_canon = load_json("latentfm_scaling_highthroughput_canonical_noharm_decision_20260624.json")
    high_rows = by_name(high)
    high_canon_rows = by_name(high_canon, "run")
    for run in ["xverse_scaling_cap60_6k_seed42", "xverse_scaling_cap60_replay05_4k_seed42"]:
        row = high_rows.get(run, {})
        add_candidate(
            candidates,
            run=run,
            family="cap60_step_replay",
            internal={
                "cross_pp_delta": (row.get("metrics") or {}).get("cross_pp_delta_vs_anchor"),
                "family_pp_delta": (row.get("metrics") or {}).get("family_gene_pp_delta_vs_anchor"),
                "family_mmd_delta": (row.get("metrics") or {}).get("family_gene_mmd_delta_vs_anchor"),
            },
            canonical=high_canon_rows.get(run, {}),
            evidence=[
                "reports/LATENTFM_SCALING_HIGH_THROUGHPUT_SMOKES_DECISION_20260624.md",
                "reports/LATENTFM_SCALING_HIGH_THROUGHPUT_CANONICAL_NOHARM_DECISION_20260624.md",
            ],
        )

    response = load_json("latentfm_scaling_cap60_response_repair_decision_20260624.json")
    response_canon = load_json("latentfm_scaling_cap60_response_canonical_noharm_decision_20260624.json")
    response_rows = by_name(response)
    response_canon_rows = by_name(response_canon, "run")
    for run in ["xverse_scaling_cap60_resp010_replay05_4k_seed42", "xverse_scaling_cap60_resp025_replay05_4k_seed42"]:
        row = response_rows.get(run, {})
        add_candidate(
            candidates,
            run=run,
            family="response_normalized",
            internal={
                "cross_pp_delta": (row.get("metrics") or {}).get("cross_pp_delta_vs_anchor"),
                "family_pp_delta": (row.get("metrics") or {}).get("family_gene_pp_delta_vs_anchor"),
                "family_mmd_delta": (row.get("metrics") or {}).get("family_gene_mmd_delta_vs_anchor"),
            },
            canonical=response_canon_rows.get(run, {}),
            evidence=[
                "reports/LATENTFM_SCALING_CAP60_RESPONSE_REPAIR_DECISION_20260624.md",
                "reports/LATENTFM_SCALING_CAP60_RESPONSE_CANONICAL_NOHARM_DECISION_20260624.md",
            ],
        )

    soft = load_json("latentfm_xverse_soft_exposure_smokes_decision_20260624.json")
    soft_canon = load_json("latentfm_xverse_soft_exposure_canonical_noharm_decision_20260624.json")
    soft_rows = by_name(soft, "run")
    soft_run = "xverse_softvisit_p085_no_cap_3k_seed42"
    soft_row = soft_rows.get(soft_run, {})
    add_candidate(
        candidates,
        run=soft_run,
        family="soft_exposure",
        internal={
            "cross_pp_delta": (soft_row.get("metrics") or {}).get("cross_pp_minus_anchor"),
            "family_pp_delta": (soft_row.get("metrics") or {}).get("family_pp_minus_anchor"),
            "family_mmd_delta": (soft_row.get("metrics") or {}).get("family_mmd_minus_anchor"),
        },
        canonical=by_name(soft_canon, "run").get(soft_run, {}),
        evidence=[
            "reports/LATENTFM_XVERSE_SOFT_EXPOSURE_SMOKES_DECISION_20260624.md",
            "reports/LATENTFM_XVERSE_SOFT_EXPOSURE_CANONICAL_NOHARM_DECISION_20260624.md",
        ],
    )

    risk = load_json("latentfm_risk_row_cvar_internal_posthoc_decision_20260624.json")
    risk_canon = load_json("latentfm_risk_row_cvar_canonical_noharm_decision_20260624.json")
    risk_cross = first_group_delta(risk, "internal_val_cross_background_seen_gene_proxy")
    risk_family = first_group_delta(risk, "family_gene")
    add_candidate(
        candidates,
        run="xverse_risk_row_cvar_allrisk_w020_2k_seed42",
        family="risk_row_cvar",
        internal={
            "cross_pp_delta": risk_cross.get("delta_pearson_pert"),
            "family_pp_delta": risk_family.get("delta_pearson_pert"),
            "family_mmd_delta": risk_family.get("delta_test_mmd"),
        },
        canonical=risk_canon,
        evidence=[
            "reports/LATENTFM_RISK_ROW_CVAR_INTERNAL_POSTHOC_DECISION_20260624.md",
            "reports/LATENTFM_RISK_ROW_CVAR_CANONICAL_NOHARM_DECISION_20260624.md",
        ],
    )

    # Drop malformed rows that lack a canonical evaluation.
    candidates = [row for row in candidates if row.get("canonical_gate_status")]
    internal_pass_like = [row for row in candidates if row.get("internal_pass_like")]
    canonical_passes = [row for row in candidates if row.get("canonical_pass")]
    false_positive = [row for row in internal_pass_like if not row.get("canonical_pass")]
    xs = [float(row["internal_score"]) for row in candidates if row.get("internal_score") is not None and row.get("canonical_max_pp_p_harm") is not None]
    ys = [float(row["canonical_max_pp_p_harm"]) for row in candidates if row.get("internal_score") is not None and row.get("canonical_max_pp_p_harm") is not None]
    rho = spearman(xs, ys)

    reasons = []
    if len(candidates) < 8:
        reasons.append("too_few_frozen_canonical_pairs")
    if len(canonical_passes) < 2:
        reasons.append("no_enough_canonical_noharm_positive_examples")
    if len(false_positive) >= 3:
        reasons.append("multiple_internal_pass_like_candidates_failed_canonical_noharm")
    if rho is None:
        reasons.append("internal_score_to_canonical_harm_correlation_not_estimable")
    elif rho > -0.25:
        reasons.append("internal_score_does_not_anticorrelate_with_canonical_harm")
    severe = [
        row
        for row in internal_pass_like
        if (row.get("canonical_cross_pp_delta") is not None and float(row["canonical_cross_pp_delta"]) < -0.005)
        or (row.get("canonical_family_pp_delta") is not None and float(row["canonical_family_pp_delta"]) < -0.005)
    ]
    if severe:
        reasons.append("internal_pass_like_candidates_have_canonical_pp_hard_harm")

    status = (
        "noharm_transfer_calibration_gate_pass_one_protocol_next"
        if not reasons
        else "noharm_transfer_calibration_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": status.endswith("_next"),
        "boundary": {
            "reads_completed_reports_only": True,
            "canonical_metrics_used_as_retrospective_veto_context": True,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "training_or_inference": False,
            "active_logs_read": False,
            "gpu": False,
        },
        "summary": {
            "n_pairs": len(candidates),
            "n_internal_pass_like": len(internal_pass_like),
            "n_canonical_pass": len(canonical_passes),
            "n_internal_pass_like_canonical_fail": len(false_positive),
            "n_internal_pass_like_canonical_hard_harm": len(severe),
            "spearman_internal_score_vs_canonical_max_pp_p_harm": rho,
        },
        "reasons": reasons,
        "candidates": candidates,
        "next_action": (
            "build a seed-matched micro-matrix protocol or a new no-harm surrogate with positive controls"
            if status.endswith("_next")
            else "do not launch scaling GPU; internal metrics are not calibrated to canonical no-harm"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling No-Harm Transfer Calibration Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only retrospective calibration.",
        "- Uses completed frozen canonical no-harm reports only as veto/context, not as new checkpoint selection.",
        "- Does not read canonical multi, Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- frozen internal/canonical pairs: `{len(candidates)}`",
        f"- internal pass-like pairs: `{len(internal_pass_like)}`",
        f"- canonical no-harm passes: `{len(canonical_passes)}`",
        f"- internal pass-like but canonical fail: `{len(false_positive)}`",
        f"- internal pass-like with canonical pp hard harm: `{len(severe)}`",
        f"- Spearman internal score vs canonical max pp p_harm: `{fmt(rho)}`",
        "",
        "## Candidate Matrix",
        "",
        "| run | family | internal cross | internal family pp | internal family MMD | pass-like | canonical cross | canonical family pp | max pp p_harm | canonical pass |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in candidates:
        lines.append(
            f"| `{row['run']}` | `{row['family']}` | {fmt(row['internal_cross_pp_delta'])} | "
            f"{fmt(row['internal_family_pp_delta'])} | {fmt(row['internal_family_mmd_delta'])} | "
            f"`{row['internal_pass_like']}` | {fmt(row['canonical_cross_pp_delta'])} | "
            f"{fmt(row['canonical_family_pp_delta'])} | {fmt(row['canonical_max_pp_p_harm'])} | "
            f"`{row['canonical_pass']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## Interpretation",
            "",
            "- Current train-only internal scaling metrics are useful for mechanism discovery, but not calibrated enough to predict canonical no-harm.",
            "- The absence of canonical no-harm positive examples means a reliable surrogate cannot be learned yet.",
            "- Re-entering GPU for scaling requires a materially new surrogate/protocol, not another internal-pass candidate.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": payload["gpu_authorized"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
