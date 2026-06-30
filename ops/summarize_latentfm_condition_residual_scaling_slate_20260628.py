#!/usr/bin/env python3
"""Summarize condition-residual scaling slate seed43 GPU smokes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_condition_residual_scaling_slate_20260628"
OUT_JSON = ROOT / "reports/latentfm_condition_residual_scaling_slate_decision_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_CONDITION_RESIDUAL_SCALING_SLATE_DECISION_20260628.md"
OLD_CURVE = ROOT / "reports/scaling_figure_data_20260625/condition_exposure_curve.csv"

RUNS = [
    {
        "name": "xverse_crscale_resp_gene_k562bg_3k_seed43",
        "arm": "gene_cap120_k562bg",
        "pair": "response_strength_vs_breadth",
    },
    {
        "name": "xverse_crscale_resp_breadth_manyshallow_3k_seed43",
        "arm": "breadth_many_shallow_19ds_cap30_budget480",
        "pair": "response_strength_vs_breadth",
    },
    {
        "name": "xverse_crscale_ptype_gene_allbg_3k_seed43",
        "arm": "gene_cap120_allbg",
        "pair": "perturbation_type_breadth",
    },
    {
        "name": "xverse_crscale_ptype_typebalanced_3k_seed43",
        "arm": "type_balanced_cap120",
        "pair": "perturbation_type_breadth",
    },
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def read_exit(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def metric(payload: dict[str, Any] | None, group: str, key: str) -> float | None:
    if not payload:
        return None
    value = ((payload.get("groups") or {}).get(group) or {}).get(key)
    return None if value is None else float(value)


def delta(candidate: float | None, anchor: float | None) -> float | None:
    if candidate is None or anchor is None:
        return None
    return float(candidate) - float(anchor)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def old_seed42_rows() -> dict[str, dict[str, float]]:
    if not OLD_CURVE.is_file():
        return {}
    out: dict[str, dict[str, float]] = {}
    with OLD_CURVE.open(newline="") as f:
        for row in csv.DictReader(f):
            arm = row.get("arm", "")
            if arm not in {r["arm"] for r in RUNS}:
                continue
            out[arm] = {
                "cross_pp_delta": float(row["cross_pp_delta"]),
                "family_pp_delta": float(row["family_pp_delta"]),
                "family_mmd_delta": float(row["family_mmd_delta"]),
            }
    return out


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in RUNS:
        run_dir = RUN_ROOT / spec["name"]
        eval_dir = run_dir / "posthoc_eval_internal"
        split_anchor = load_json(eval_dir / "split_group_eval_anchor_internal_ode20.json")
        split_cand = load_json(eval_dir / "split_group_eval_candidate_internal_ode20.json")
        fam_anchor = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
        fam_cand = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
        cross_c = metric(split_cand, "internal_val_cross_background_seen_gene_proxy", "pearson_pert")
        cross_a = metric(split_anchor, "internal_val_cross_background_seen_gene_proxy", "pearson_pert")
        family_c = metric(split_cand, "internal_val_family_gene_proxy", "pearson_pert")
        family_a = metric(split_anchor, "internal_val_family_gene_proxy", "pearson_pert")
        family_mmd_c = metric(split_cand, "internal_val_family_gene_proxy", "test_mmd")
        family_mmd_a = metric(split_anchor, "internal_val_family_gene_proxy", "test_mmd")
        fam_gene_c = metric(fam_cand, "family_gene", "pearson_pert")
        fam_gene_a = metric(fam_anchor, "family_gene", "pearson_pert")
        rows.append(
            {
                **spec,
                "exists": run_dir.is_dir(),
                "train_exit": read_exit(run_dir / f"{spec['name']}.EXIT_CODE"),
                "posthoc_exit": read_exit(run_dir / "POSTHOC_EXIT_CODE"),
                "status": "done"
                if read_exit(run_dir / f"{spec['name']}.EXIT_CODE") == 0 and read_exit(run_dir / "POSTHOC_EXIT_CODE") == 0
                else "pending_or_failed",
                "cross_pp_delta": delta(cross_c, cross_a),
                "family_pp_delta": delta(family_c, family_a),
                "family_mmd_delta": delta(family_mmd_c, family_mmd_a),
                "family_gene_pp_delta": delta(fam_gene_c, fam_gene_a),
            }
        )
    return rows


def pair_decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row["status"] != "done" for row in rows):
        return {"status": "pending", "action": "wait_without_polling"}
    by_arm = {row["arm"]: row for row in rows}
    old = old_seed42_rows()
    decisions: dict[str, Any] = {}

    k562 = by_arm["gene_cap120_k562bg"]
    breadth = by_arm["breadth_many_shallow_19ds_cap30_budget480"]
    resp_checks = {
        "seed43_cross_pp_advantage": delta(k562["cross_pp_delta"], breadth["cross_pp_delta"]),
        "seed43_family_pp_advantage": delta(k562["family_pp_delta"], breadth["family_pp_delta"]),
        "seed43_family_mmd_delta_advantage": delta(k562["family_mmd_delta"], breadth["family_mmd_delta"]),
        "seed42_cross_pp_advantage": delta(old.get("gene_cap120_k562bg", {}).get("cross_pp_delta"), old.get("breadth_many_shallow_19ds_cap30_budget480", {}).get("cross_pp_delta")),
        "seed42_family_pp_advantage": delta(old.get("gene_cap120_k562bg", {}).get("family_pp_delta"), old.get("breadth_many_shallow_19ds_cap30_budget480", {}).get("family_pp_delta")),
    }
    resp_reasons = []
    if (resp_checks["seed43_cross_pp_advantage"] or -999) <= 0.005:
        resp_reasons.append("seed43_cross_pp_advantage_too_small")
    if (resp_checks["seed43_family_pp_advantage"] or -999) <= 0.005:
        resp_reasons.append("seed43_family_pp_advantage_too_small")
    if (resp_checks["seed43_family_mmd_delta_advantage"] or 999) > 0.002:
        resp_reasons.append("seed43_mmd_tail_harm_vs_breadth")
    decisions["response_strength_vs_breadth"] = {
        "status": "pass_extend_seed44" if not resp_reasons else "fail_or_mechanism_only",
        "reasons": resp_reasons,
        "checks": resp_checks,
    }

    gene = by_arm["gene_cap120_allbg"]
    typeb = by_arm["type_balanced_cap120"]
    ptype_checks = {
        "seed43_type_cross_pp_minus_gene": delta(typeb["cross_pp_delta"], gene["cross_pp_delta"]),
        "seed43_type_family_pp_minus_gene": delta(typeb["family_pp_delta"], gene["family_pp_delta"]),
        "seed43_type_mmd_delta_minus_gene": delta(typeb["family_mmd_delta"], gene["family_mmd_delta"]),
        "seed42_type_cross_pp_minus_gene": delta(old.get("type_balanced_cap120", {}).get("cross_pp_delta"), old.get("gene_cap120_allbg", {}).get("cross_pp_delta")),
        "seed42_type_family_pp_minus_gene": delta(old.get("type_balanced_cap120", {}).get("family_pp_delta"), old.get("gene_cap120_allbg", {}).get("family_pp_delta")),
    }
    ptype_reasons = []
    if (ptype_checks["seed43_type_cross_pp_minus_gene"] or -999) < -0.005:
        ptype_reasons.append("type_breadth_cross_pp_regression")
    if (ptype_checks["seed43_type_family_pp_minus_gene"] or -999) < -0.002:
        ptype_reasons.append("type_breadth_family_pp_regression")
    if (ptype_checks["seed43_type_mmd_delta_minus_gene"] or 999) > 0.001:
        ptype_reasons.append("type_breadth_mmd_tail_regression")
    decisions["perturbation_type_breadth"] = {
        "status": "pass_extend_seed44" if not ptype_reasons else "fail_close_or_demote_axis",
        "reasons": ptype_reasons,
        "checks": ptype_checks,
    }
    return decisions


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Condition-Residual Scaling Slate Decision",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- Summarizes seed43 exploratory matched-pair GPU smokes only.",
        "- Uses train-only internal validation groups; no canonical multi or Track C query.",
        "- Existing seed42 rows are used only as prior stability context.",
        "",
        "## Rows",
        "",
        "| run | arm | status | cross pp delta | family pp delta | family mmd delta |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['name']}` | `{row['arm']}` | `{row['status']}` | {_fmt(row['cross_pp_delta'])} | {_fmt(row['family_pp_delta'])} | {_fmt(row['family_mmd_delta'])} |"
        )
    lines.extend(["", "## Pair Decisions", ""])
    decision = payload["decision"]
    if decision.get("status") == "pending":
        lines.append("- `pending`: wait without polling.")
    else:
        for name, obj in decision.items():
            lines.append(f"### {name}")
            lines.append(f"- status: `{obj['status']}`")
            if obj.get("reasons"):
                lines.extend(f"- reason: `{reason}`" for reason in obj["reasons"])
            for key, value in (obj.get("checks") or {}).items():
                lines.append(f"- {key}: `{_fmt(value)}`")
            lines.append("")
    lines.extend(["## Outputs", "", f"- JSON: `{OUT_JSON}`"])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_rows()
    decision = pair_decisions(rows)
    status = "pending"
    if isinstance(decision, dict) and decision.get("status") == "pending":
        status = "pending"
    elif all(row["status"] == "done" for row in rows):
        statuses = [obj["status"] for obj in decision.values()]
        status = "slate_pass_extend" if any(s == "pass_extend_seed44" for s in statuses) else "slate_fail_close_or_mechanism_only"
    payload = {"status": status, "rows": rows, "decision": decision}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
