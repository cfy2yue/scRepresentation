#!/usr/bin/env python3
"""CPU-only provenance-tail sentinel simulation for scaling curriculum.

This gate tests whether simple provenance/metainfo sentinel rules can retain
the cap120-vs-cap30 train-only signal while setting high-risk strata back to
cap30-like behavior. It is simulation only: no model training, inference, GPU,
canonical multi, or held-out Track C query.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
LODO = REPORTS / "latentfm_scaling_mixed_effect_lodo_condition_count_gate_20260624.json"
PROVENANCE = REPORTS / "latentfm_scaling_provenance_estimand_matrix_gate_20260624.json"
SOURCE = REPORTS / "latentfm_scaling_source_verified_background_type_strata_gate_20260624.json"

OUT_JSON = REPORTS / "latentfm_scaling_provenance_tail_sentinel_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_SCALING_PROVENANCE_TAIL_SENTINEL_GATE_20260624.md"

SEED = 20260624


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def weighted_mean(rows: list[dict[str, Any]], key: str = "sim_pp_delta") -> float | None:
    total = sum(int(r.get("n") or 0) for r in rows)
    if total <= 0:
        return None
    return sum(float(r.get(key) or 0.0) * int(r.get("n") or 0) for r in rows) / total


def bootstrap_ci(rows: list[dict[str, Any]], key: str = "sim_pp_delta", n_boot: int = 2000) -> dict[str, Any]:
    rng = random.Random(SEED)
    vals = []
    if not rows:
        return {"ci": [None, None], "p_le_zero": None}
    for _ in range(n_boot):
        sample = [rng.choice(rows) for _ in rows]
        vals.append(float(weighted_mean(sample, key=key) or 0.0))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
    return {"ci": [lo, hi], "p_le_zero": sum(1 for v in vals if v <= 0.0) / len(vals)}


def simulate(rows: list[dict[str, Any]], sentinel: Callable[[dict[str, Any]], bool], name: str) -> dict[str, Any]:
    sim_rows = []
    for row in rows:
        is_sentinel = sentinel(row)
        sim = dict(row)
        sim["sentinel"] = bool(is_sentinel)
        # Sentinel means this stratum would stay at cap30-like behavior for the
        # next design, so the cap120-minus-cap30 simulated delta is zero.
        sim["sim_pp_delta"] = 0.0 if is_sentinel else float(row.get("pp_delta_mean") or 0.0)
        sim["sim_mmd_delta"] = 0.0 if is_sentinel else float(row.get("mmd_delta_mean") or 0.0)
        sim_rows.append(sim)
    n_sentinel = sum(1 for r in sim_rows if r["sentinel"])
    pp_mean = weighted_mean(sim_rows, "sim_pp_delta")
    mmd_mean = weighted_mean(sim_rows, "sim_mmd_delta")
    min_pp = min((float(r["sim_pp_delta"]) for r in sim_rows), default=None)
    neg_tails = [r for r in sim_rows if float(r["sim_pp_delta"]) < -0.02]
    boot = bootstrap_ci(sim_rows, "sim_pp_delta")
    return {
        "name": name,
        "n_sentinel_datasets": n_sentinel,
        "sentinel_datasets": [r["dataset"] for r in sim_rows if r["sentinel"]],
        "pp_mean": pp_mean,
        "mmd_mean": mmd_mean,
        "min_pp": min_pp,
        "negative_tail_count": len(neg_tails),
        "bootstrap": boot,
        "rows": sim_rows,
    }


def random_controls(rows: list[dict[str, Any]], n_sentinel: int, n_perm: int = 2000) -> dict[str, Any]:
    rng = random.Random(SEED + n_sentinel)
    names = [r["dataset"] for r in rows]
    pp_vals = []
    min_vals = []
    neg_counts = []
    for _ in range(n_perm):
        chosen = set(rng.sample(names, min(n_sentinel, len(names))))
        sim_rows = []
        for row in rows:
            sim = dict(row)
            sim["sim_pp_delta"] = 0.0 if row["dataset"] in chosen else float(row.get("pp_delta_mean") or 0.0)
            sim_rows.append(sim)
        pp_vals.append(float(weighted_mean(sim_rows, "sim_pp_delta") or 0.0))
        min_vals.append(min(float(r["sim_pp_delta"]) for r in sim_rows))
        neg_counts.append(sum(1 for r in sim_rows if float(r["sim_pp_delta"]) < -0.02))
    return {
        "n_perm": n_perm,
        "pp_mean_median": median(pp_vals),
        "pp_mean_q90": sorted(pp_vals)[int(0.90 * len(pp_vals))],
        "min_pp_median": median(min_vals),
        "negative_tail_count_median": median(neg_counts),
    }


def main() -> int:
    lodo = load_json(LODO)
    provenance = load_json(PROVENANCE)
    source = load_json(SOURCE)
    prov_rows = {row["dataset"]: row for row in provenance.get("dataset_rows", [])}

    rows = []
    for row in lodo.get("dataset_rows", []):
        ds = row["dataset"]
        prov = prov_rows.get(ds, {})
        merged = dict(row)
        merged.update(
            {
                "source_quality": prov.get("source_quality", row.get("source_quality", "")),
                "obs_bg_available": bool(prov.get("obs_cell_background_columns")),
                "obs_dose_available": bool(prov.get("obs_dose_columns")),
                "buckets": prov.get("buckets", ""),
                "n_multi_conditions_selected": int(prov.get("n_multi_conditions_selected") or 0),
            }
        )
        rows.append(merged)

    neg_bgs = {
        k for k, v in (source.get("background_summary") or {}).items() if float(v.get("pp_delta_mean") or 0.0) < -0.02
    }
    neg_types = {
        k for k, v in (source.get("perturbation_type_summary") or {}).items() if float(v.get("pp_delta_mean") or 0.0) < -0.02
    }

    rules: list[tuple[str, Callable[[dict[str, Any]], bool], str]] = [
        (
            "source_verified_only",
            lambda r: r.get("source_quality") != "source_verified",
            "sentinel non-source-verified datasets",
        ),
        (
            "obs_background_required",
            lambda r: not bool(r.get("obs_bg_available")),
            "sentinel datasets without obs-level cell-background column",
        ),
        (
            "negative_source_background",
            lambda r: str(r.get("background")) in neg_bgs,
            "sentinel train-only negative source-background strata",
        ),
        (
            "negative_source_type",
            lambda r: str(r.get("perturbation_type")) in neg_types,
            "sentinel train-only negative perturbation-type strata",
        ),
        (
            "negative_background_or_type",
            lambda r: str(r.get("background")) in neg_bgs or str(r.get("perturbation_type")) in neg_types,
            "sentinel union of negative source-background and type strata",
        ),
        (
            "known_no_cap_gain_or_mixed_source",
            lambda r: int(r.get("cap_gain") or 0) <= 0 or str(r.get("buckets")) == "multiple+single",
            "sentinel datasets with no cap gain or mixed single/multi bucket provenance",
        ),
    ]

    simulations = []
    for name, fn, desc in rules:
        sim = simulate(rows, fn, name)
        ctrl = random_controls(rows, int(sim["n_sentinel_datasets"]))
        sim["description"] = desc
        sim["random_control"] = ctrl
        sim["control_collapse"] = (
            float(sim["pp_mean"] or -999.0) >= float(ctrl["pp_mean_q90"] or 999.0) + 0.001
            and float(sim["min_pp"] or -999.0) >= -0.02
            and int(sim["negative_tail_count"]) == 0
        )
        sim["pass_gate"] = (
            float(sim["pp_mean"] or -999.0) >= 0.010
            and float(sim["min_pp"] or -999.0) >= -0.020
            and float(sim["mmd_mean"] or 999.0) <= 0.001
            and (
                float((sim["bootstrap"].get("ci") or [None, None])[0] or -999.0) > 0.0
                or float(sim["bootstrap"].get("p_le_zero") or 1.0) <= 0.20
            )
            and bool(sim["control_collapse"])
        )
        simulations.append(sim)

    pass_rules = [s for s in simulations if s["pass_gate"]]
    reasons = []
    if not pass_rules:
        reasons.append("no_provenance_tail_sentinel_rule_passes_all_gates")
    if all(not s["control_collapse"] for s in simulations):
        reasons.append("shuffled_controls_do_not_collapse_or_candidate_not_distinct")
    if all(float(s["pp_mean"] or -999.0) < 0.010 for s in simulations):
        reasons.append("sentinel_rules_do_not_retain_pp_ge_0p010")
    if any(float(s["min_pp"] or 0.0) < -0.020 for s in simulations):
        reasons.append("at_least_one_rule_retains_dataset_tail_harm")
    reasons.append("simulation_only_no_training_done")

    status = "scaling_provenance_tail_sentinel_gate_fail_no_gpu"
    if pass_rules:
        status = "scaling_provenance_tail_sentinel_gate_pass_launcher_design_next"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "simulation_only": True,
            "reads_train_only_completed_reports": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "negative_source_backgrounds": sorted(neg_bgs),
        "negative_source_types": sorted(neg_types),
        "simulations": [
            {k: v for k, v in sim.items() if k != "rows"} for sim in simulations
        ],
        "pass_rules": [s["name"] for s in pass_rules],
        "reasons": reasons,
        "next_action": (
            "design one bounded train-only scaling curriculum launcher"
            if pass_rules
            else "do not launch scaling curriculum GPU; no sentinel rule passes tail and control gates"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Scaling Provenance-Tail Sentinel Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only simulation over completed train-only scaling summaries.",
        "- Does not read canonical metrics, canonical multi, held-out Track C query, active logs, train, infer, or use GPU.",
        "",
        "## Rules",
        "",
        "| rule | sentinel datasets | pp mean | MMD mean | min pp | neg tails | boot CI | p<=0 | control collapse | pass |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for sim in simulations:
        ci = sim["bootstrap"].get("ci") or [None, None]
        lines.append(
            f"| `{sim['name']}` | {sim['n_sentinel_datasets']} | {fmt(sim['pp_mean'])} | "
            f"{fmt(sim['mmd_mean'])} | {fmt(sim['min_pp'])} | {sim['negative_tail_count']} | "
            f"[{fmt(ci[0])}, {fmt(ci[1])}] | {sim['bootstrap'].get('p_le_zero')} | "
            f"`{sim['control_collapse']}` | `{sim['pass_gate']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- negative source backgrounds: `{sorted(neg_bgs)}`",
            f"- negative source types: `{sorted(neg_types)}`",
            f"- pass rules: `{[s['name'] for s in pass_rules]}`",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            f"- next action: `{payload['next_action']}`",
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
