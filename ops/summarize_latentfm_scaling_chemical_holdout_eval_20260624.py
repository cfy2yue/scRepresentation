#!/usr/bin/env python3
"""Summarize train-only chemical holdout eval for scaling diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_scaling_chemical_holdout_eval_20260624"
OUT_JSON = ROOT / "reports/latentfm_scaling_chemical_holdout_eval_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_CHEMICAL_HOLDOUT_EVAL_GATE_20260624.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group(payload: dict[str, Any], name: str = "family_drug") -> dict[str, Any]:
    return (payload.get("groups") or {}).get(name) or {}


def rows(payload: dict[str, Any], name: str = "family_drug") -> list[dict[str, Any]]:
    return list(group(payload, name).get("condition_metrics") or [])


def metric(payload: dict[str, Any], metric_name: str, name: str = "family_drug") -> float | None:
    val = group(payload, name).get(metric_name)
    return None if val is None else float(val)


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def dataset_means(candidate: dict[str, Any], anchor: dict[str, Any]) -> dict[str, dict[str, float]]:
    c_by = {(r["dataset"], r["condition"]): r for r in rows(candidate)}
    a_by = {(r["dataset"], r["condition"]): r for r in rows(anchor)}
    accum: dict[str, list[dict[str, float]]] = {}
    for key in sorted(set(c_by) & set(a_by)):
        ds, _cond = key
        cr = c_by[key]
        ar = a_by[key]
        if cr.get("pearson_pert") is None or ar.get("pearson_pert") is None:
            continue
        accum.setdefault(ds, []).append(
            {
                "pp": float(cr["pearson_pert"]) - float(ar["pearson_pert"]),
                "mmd": float(cr["test_mmd_clamped"]) - float(ar["test_mmd_clamped"]),
            }
        )
    out = {}
    for ds, vals in accum.items():
        out[ds] = {
            "n": len(vals),
            "pp_delta": sum(v["pp"] for v in vals) / len(vals),
            "mmd_delta": sum(v["mmd"] for v in vals) / len(vals),
        }
    return out


def summarize_arm(arm: str) -> dict[str, Any]:
    eval_dir = RUN_ROOT / arm
    anchor = load(eval_dir / "condition_family_eval_anchor_chemical_ode20.json")
    candidate = load(eval_dir / "condition_family_eval_candidate_chemical_ode20.json")
    pp_delta = metric(candidate, "pearson_pert") - metric(anchor, "pearson_pert")
    mmd_delta = metric(candidate, "test_mmd_clamped") - metric(anchor, "test_mmd_clamped")
    ds = dataset_means(candidate, anchor)
    return {
        "arm": arm,
        "n_conditions": len(rows(candidate)),
        "anchor_pp": metric(anchor, "pearson_pert"),
        "candidate_pp": metric(candidate, "pearson_pert"),
        "pp_delta_vs_anchor": pp_delta,
        "anchor_mmd": metric(anchor, "test_mmd_clamped"),
        "candidate_mmd": metric(candidate, "test_mmd_clamped"),
        "mmd_delta_vs_anchor": mmd_delta,
        "dataset_means": ds,
        "dataset_min_pp_delta": min((v["pp_delta"] for v in ds.values()), default=None),
        "negative_dataset_tails_lt_minus_0p02": sum(1 for v in ds.values() if v["pp_delta"] < -0.02),
    }


def main() -> int:
    arms = [summarize_arm("cap30"), summarize_arm("cap120")]
    by = {row["arm"]: row for row in arms}
    cap120_minus_cap30_pp = by["cap120"]["pp_delta_vs_anchor"] - by["cap30"]["pp_delta_vs_anchor"]
    cap120_minus_cap30_mmd = by["cap120"]["mmd_delta_vs_anchor"] - by["cap30"]["mmd_delta_vs_anchor"]
    reasons = []
    if by["cap120"]["n_conditions"] < 90:
        reasons.append("too_few_chemical_holdout_conditions")
    if by["cap120"]["pp_delta_vs_anchor"] < 0.020:
        reasons.append("cap120_chemical_pp_delta_lt_0p020")
    if cap120_minus_cap30_pp < 0.010:
        reasons.append("cap120_minus_cap30_chemical_pp_lt_0p010")
    if by["cap120"]["mmd_delta_vs_anchor"] > 0.001:
        reasons.append("cap120_chemical_mmd_hard_harm")
    if by["cap120"]["dataset_min_pp_delta"] is None or by["cap120"]["dataset_min_pp_delta"] < -0.020:
        reasons.append("cap120_chemical_dataset_tail_below_minus_0p020")
    status = "chemical_holdout_eval_fail_no_gpu" if reasons else "chemical_holdout_eval_pass_external_review_next"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "train_only_chemical_holdout": True,
            "canonical_reference_excluded": True,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training": False,
            "gpu_eval_only": True,
        },
        "arms": arms,
        "cap120_minus_cap30": {
            "pp_delta_vs_anchor_delta": cap120_minus_cap30_pp,
            "mmd_delta_vs_anchor_delta": cap120_minus_cap30_mmd,
        },
        "reasons": reasons,
        "decision": {
            "gpu_training_next": False,
            "next_action": "external review before descriptor-cache smoke" if not reasons else "do not launch descriptor-cache smoke from this gate",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Scaling Chemical Holdout Eval Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- GPU eval-only diagnostic on train-only SciPlex holdout; no training.",
        "- Canonical reference drugs are excluded from the holdout.",
        "- Canonical multi and Track C query are not read.",
        "",
        "## Arms",
        "",
        "| arm | n | pp delta vs anchor | MMD delta vs anchor | dataset min pp | negative tails |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in arms:
        lines.append(
            f"| `{row['arm']}` | {row['n_conditions']} | {fmt(row['pp_delta_vs_anchor'])} | "
            f"{fmt(row['mmd_delta_vs_anchor'])} | {fmt(row['dataset_min_pp_delta'])} | "
            f"{row['negative_dataset_tails_lt_minus_0p02']} |"
        )
    lines.extend(
        [
            "",
            "## Count Effect",
            "",
            f"- cap120 minus cap30 pp-delta-vs-anchor: `{fmt(cap120_minus_cap30_pp)}`",
            f"- cap120 minus cap30 MMD-delta-vs-anchor: `{fmt(cap120_minus_cap30_mmd)}`",
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU training authorized: `False`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "reasons": reasons}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
