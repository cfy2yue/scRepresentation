#!/usr/bin/env python3
"""CPU-only dataset-veto route gate for Track A scaling candidates.

This is an optimistic train-only/internal diagnostic. It asks whether a simple
dataset-level veto could preserve the internal Pearson gain of a scaling
candidate while avoiding family-gene MMD harm. Because the veto is selected
from internal family metrics, a pass is not GPU authorization; it is only
permission to design a deployable router gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_scaling_count_smokes_20260624"
OUT_JSON = ROOT / "reports/latentfm_xverse_dataset_veto_route_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_DATASET_VETO_ROUTE_GATE_20260624.md"

CANDIDATES = [
    "xverse_scaling_cap120_all_3k_seed42",
    "xverse_scaling_gene_cap120_allbg_3k_seed42",
    "xverse_scaling_general_exposure_cap_v2_3k_seed42",
]


def load_group(run: str, stem: str, group: str) -> dict[str, Any]:
    path = RUN_ROOT / run / "posthoc_eval_internal" / stem
    obj = json.loads(path.read_text(encoding="utf-8"))
    return (obj.get("groups") or {})[group]


def by_key(group: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["dataset"]), str(row["condition"])): row
        for row in group.get("condition_metrics", [])
    }


def per_ds_delta(anchor: dict[str, Any], cand: dict[str, Any]) -> dict[str, dict[str, float]]:
    a_rows = by_key(anchor)
    by_ds: dict[str, list[dict[str, float]]] = {}
    for key, c in by_key(cand).items():
        a = a_rows.get(key)
        if not a:
            continue
        ds = key[0]
        by_ds.setdefault(ds, []).append(
            {
                "delta_mmd": float(c.get("test_mmd", 0.0)) - float(a.get("test_mmd", 0.0)),
                "delta_pp": float(c.get("pearson_pert", 0.0)) - float(a.get("pearson_pert", 0.0)),
            }
        )
    out = {}
    for ds, rows in by_ds.items():
        out[ds] = {
            "n": len(rows),
            "delta_mmd_mean": sum(r["delta_mmd"] for r in rows) / len(rows),
            "delta_pp_mean": sum(r["delta_pp"] for r in rows) / len(rows),
        }
    return out


def routed_group(anchor: dict[str, Any], cand: dict[str, Any], use_candidate_ds: set[str]) -> dict[str, Any]:
    a_rows = by_key(anchor)
    c_rows = by_key(cand)
    pp_vals = []
    mmd_vals = []
    used = {"candidate": 0, "anchor": 0}
    for key, a in a_rows.items():
        row = c_rows.get(key, a) if key[0] in use_candidate_ds else a
        source = "candidate" if key[0] in use_candidate_ds and key in c_rows else "anchor"
        used[source] += 1
        if row.get("pearson_pert") is not None:
            pp_vals.append(float(row["pearson_pert"]))
        if row.get("test_mmd") is not None:
            mmd_vals.append(float(row["test_mmd"]))
    return {
        "pearson_pert": sum(pp_vals) / len(pp_vals) if pp_vals else None,
        "test_mmd": sum(mmd_vals) / len(mmd_vals) if mmd_vals else None,
        "n_conds": len(a_rows),
        "used": used,
        "candidate_fraction": used["candidate"] / max(1, used["candidate"] + used["anchor"]),
    }


def summarize_run(run: str) -> dict[str, Any]:
    anchor_family = load_group(run, "condition_family_eval_anchor_internal_ode20.json", "family_gene")
    cand_family = load_group(run, "condition_family_eval_candidate_internal_ode20.json", "family_gene")
    anchor_cross = load_group(run, "split_group_eval_anchor_internal_ode20.json", "internal_val_cross_background_seen_gene_proxy")
    cand_cross = load_group(run, "split_group_eval_candidate_internal_ode20.json", "internal_val_cross_background_seen_gene_proxy")
    ds = per_ds_delta(anchor_family, cand_family)

    # Optimistic train-only veto: candidate is allowed only on datasets that
    # do not harm family MMD and do not reduce family Pearson on internal rows.
    use_ds = {
        name
        for name, row in ds.items()
        if row["delta_mmd_mean"] <= 1e-3 and row["delta_pp_mean"] >= -2e-3
    }
    route_family = routed_group(anchor_family, cand_family, use_ds)
    route_cross = routed_group(anchor_cross, cand_cross, use_ds)
    anchor_family_mean = {
        "pearson_pert": float(anchor_family["pearson_pert"]),
        "test_mmd": float(anchor_family["test_mmd"]),
    }
    anchor_cross_mean = {
        "pearson_pert": float(anchor_cross["pearson_pert"]),
        "test_mmd": float(anchor_cross["test_mmd"]),
    }
    cand_family_mean = {
        "pearson_pert": float(cand_family["pearson_pert"]),
        "test_mmd": float(cand_family["test_mmd"]),
    }
    cand_cross_mean = {
        "pearson_pert": float(cand_cross["pearson_pert"]),
        "test_mmd": float(cand_cross["test_mmd"]),
    }
    checks = {
        "route_cross_pp_minus_anchor": route_cross["pearson_pert"] - anchor_cross_mean["pearson_pert"],
        "route_family_pp_minus_anchor": route_family["pearson_pert"] - anchor_family_mean["pearson_pert"],
        "route_family_mmd_minus_anchor": route_family["test_mmd"] - anchor_family_mean["test_mmd"],
        "route_candidate_fraction_family": route_family["candidate_fraction"],
        "route_candidate_fraction_cross": route_cross["candidate_fraction"],
        "selected_candidate_datasets": sorted(use_ds),
        "vetoed_datasets": sorted(set(ds) - use_ds),
        "thresholds": {
            "cross_pp_min_delta": 0.01,
            "family_pp_min_delta": 0.0,
            "family_mmd_max_delta": 0.001,
            "candidate_fraction_min": 0.25,
        },
    }
    reasons = []
    if checks["route_cross_pp_minus_anchor"] < 0.01:
        reasons.append("route_cross_pp_below_gate")
    if checks["route_family_pp_minus_anchor"] < 0.0:
        reasons.append("route_family_pp_below_anchor")
    if checks["route_family_mmd_minus_anchor"] > 0.001:
        reasons.append("route_family_mmd_harm")
    if checks["route_candidate_fraction_family"] < 0.25:
        reasons.append("route_uses_too_little_candidate")
    status = "dataset_veto_route_gate_pass_diagnostic_only" if not reasons else "dataset_veto_route_gate_fail_no_gpu"
    return {
        "run": run,
        "status": status,
        "reasons": reasons,
        "checks": checks,
        "anchor": {"family": anchor_family_mean, "cross": anchor_cross_mean},
        "candidate": {"family": cand_family_mean, "cross": cand_cross_mean},
        "routed": {"family": route_family, "cross": route_cross},
        "per_dataset": ds,
    }


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [r for r in rows if r["status"] == "dataset_veto_route_gate_pass_diagnostic_only"]
    if not passed:
        return {
            "status": "dataset_veto_route_gate_fail_no_gpu",
            "action": "do_not_launch_dataset_veto_or_more_hard_capping",
            "passed_runs": [],
        }
    best = max(passed, key=lambda r: r["checks"]["route_cross_pp_minus_anchor"])
    return {
        "status": "dataset_veto_route_gate_pass_diagnostic_only",
        "action": "design_deployable_router_before_any_gpu",
        "passed_runs": [r["run"] for r in passed],
        "best_run": best["run"],
    }


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):+.6f}"
    except Exception:
        return "NA"


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM xverse Dataset-Veto Route Gate",
        "",
        "## Boundary",
        "",
        "- CPU-only diagnostic over train-only internal posthoc artifacts.",
        "- Does not read canonical split, Track C query, active logs, or held-out query artifacts.",
        "- The dataset veto is selected from internal family metrics, so a pass is diagnostic-only and does not authorize GPU.",
        "",
        "## Decision",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Action: `{payload['decision']['action']}`",
        "",
        "## Rows",
        "",
        "| run | status | route cross pp delta | route family pp delta | route family MMD delta | candidate frac | vetoed datasets | reasons |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        c = row["checks"]
        lines.append(
            f"| {row['run']} | {row['status']} | {_fmt(c['route_cross_pp_minus_anchor'])} | "
            f"{_fmt(c['route_family_pp_minus_anchor'])} | {_fmt(c['route_family_mmd_minus_anchor'])} | "
            f"{c['route_candidate_fraction_family']:.3f} | {', '.join(c['vetoed_datasets'][:8])} | "
            f"{', '.join(row['reasons'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If this optimistic gate fails, simple dataset-veto routing is not worth a GPU branch.",
        "- If it passes, the next step is a deployable router gate using prediction-time metadata only, not immediate training.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [summarize_run(run) for run in CANDIDATES]
    payload = {
        "boundary": {
            "no_canonical_or_query": True,
            "diagnostic_only": True,
            "route_uses_internal_family_metrics": True,
        },
        "decision": decide(rows),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_md(payload)
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
