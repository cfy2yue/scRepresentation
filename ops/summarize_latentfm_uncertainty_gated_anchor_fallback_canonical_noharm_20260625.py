#!/usr/bin/env python3
"""Frozen canonical no-harm summary for uncertainty-gated anchor fallback.

Selection is recomputed only from train-only/internal budget128 6k metrics.
Canonical metrics are used only as a frozen no-harm veto.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
INTERNAL_RUN_ROOT = ROOT / "runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625"
CANON_RUN_ROOT = ROOT / "runs/latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625"
GATE_JSON = ROOT / "reports/latentfm_uncertainty_gated_anchor_fallback_gate_20260625.json"
ANCHOR_SPLIT_JSON = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_canonical/split_group_eval_anchor_ode20_canonical.json"
ANCHOR_FAMILY_JSON = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_canonical/condition_family_eval_anchor_ode20_canonical.json"
OUT_JSON = ROOT / "reports/latentfm_uncertainty_gated_anchor_fallback_canonical_noharm_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_UNCERTAINTY_GATED_ANCHOR_FALLBACK_CANONICAL_NOHARM_20260625.md"
SEEDS = (42, 43, 44)
RUN_TMPL = "xverse_truecell_nested_budget128_tailstable_seed{seed}_6000"


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def key(row: dict) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def load_internal_group(seed: int, kind: str, group: str) -> list[dict]:
    fn = "split_group_eval" if group.startswith("internal_val_") else "condition_family_eval"
    path = INTERNAL_RUN_ROOT / RUN_TMPL.format(seed=seed) / "posthoc_eval_internal" / f"{fn}_{kind}_internal_ode20.json"
    return load_json(path)["groups"][group]["condition_metrics"]


def internal_enabled(group: str, policy: dict[str, float]) -> set[tuple[str, str]]:
    records: dict[tuple[str, str], dict[str, list[float]]] = {}
    for seed in SEEDS:
        cand = {key(r): r for r in load_internal_group(seed, "candidate", group)}
        anch = {key(r): r for r in load_internal_group(seed, "anchor", group)}
        for k in sorted(set(cand) & set(anch)):
            c, a = cand[k], anch[k]
            if not (finite(c.get("pearson_pert")) and finite(a.get("pearson_pert"))):
                continue
            if not (finite(c.get("test_mmd")) and finite(a.get("test_mmd"))):
                continue
            rec = records.setdefault(k, {"pp": [], "mmd": []})
            rec["pp"].append(float(c["pearson_pert"]) - float(a["pearson_pert"]))
            rec["mmd"].append(float(c["test_mmd"]) - float(a["test_mmd"]))
    enabled = set()
    for k, rec in records.items():
        if len(rec["pp"]) != len(SEEDS):
            continue
        pp_mean = statistics.mean(rec["pp"])
        pp_sd = statistics.pstdev(rec["pp"])
        pp_lcb = pp_mean - pp_sd
        mmd_mean = statistics.mean(rec["mmd"])
        mmd_max = max(rec["mmd"])
        if (
            pp_lcb >= float(policy["min_pp_lcb"])
            and pp_sd <= float(policy["max_pp_sd"])
            and mmd_mean <= float(policy["max_mmd_mean"])
            and mmd_max <= float(policy["max_mmd_max"])
        ):
            enabled.add(k)
    return enabled


def group_rows(payload: dict, group: str) -> dict[tuple[str, str], dict]:
    return {key(r): r for r in payload["groups"][group].get("condition_metrics", [])}


def dataset_equal_mean(rows: list[dict], field: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if finite(r.get(field)):
            by_ds[str(r["dataset"])].append(float(r[field]))
    vals = [statistics.mean(vs) for vs in by_ds.values() if vs]
    return statistics.mean(vals) if vals else None


def summarize_rows(rows: list[dict]) -> dict:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_ds[str(r["dataset"])].append(float(r["route_pp_delta"]))
    ds_rows = [
        {"dataset": ds, "mean": statistics.mean(vs), "n": len(vs)}
        for ds, vs in sorted(by_ds.items())
    ]
    enabled = [r for r in rows if r["enabled"]]
    return {
        "n_conditions": len(rows),
        "n_enabled": len(enabled),
        "enabled_fraction": len(enabled) / len(rows) if rows else 0.0,
        "pp_dataset_equal_delta": dataset_equal_mean(rows, "route_pp_delta"),
        "mmd_dataset_equal_delta": dataset_equal_mean(rows, "route_mmd_delta"),
        "dataset_rows": ds_rows,
        "dataset_min_pp_delta": min((r["mean"] for r in ds_rows), default=None),
        "negative_dataset_tails_lt_minus_0p010": sum(1 for r in ds_rows if r["mean"] < -0.010),
        "condition_hard_harm_frac_lt_minus_0p020": sum(1 for r in rows if r["route_pp_delta"] < -0.020) / len(rows) if rows else 0.0,
        "enabled_conditions": [
            {"dataset": r["dataset"], "condition": r["condition"], "pp_delta": r["route_pp_delta"], "mmd_delta": r["route_mmd_delta"]}
            for r in rows
            if r["enabled"]
        ],
    }


def route_seed(seed: int, enabled_split: set[tuple[str, str]], enabled_family: set[tuple[str, str]]) -> dict:
    cand_split = load_json(CANON_RUN_ROOT / RUN_TMPL.format(seed=seed) / "posthoc_eval_canonical/split_group_eval_candidate_ode20_canonical.json")
    cand_family = load_json(CANON_RUN_ROOT / RUN_TMPL.format(seed=seed) / "posthoc_eval_canonical/condition_family_eval_candidate_ode20_canonical.json")
    anch_split = load_json(ANCHOR_SPLIT_JSON)
    anch_family = load_json(ANCHOR_FAMILY_JSON)
    specs = {
        "test_single": (group_rows(cand_split, "test_single"), group_rows(anch_split, "test_single"), enabled_split),
        "family_gene": (group_rows(cand_family, "family_gene"), group_rows(anch_family, "family_gene"), enabled_family),
    }
    out = {}
    for group, (cand, anch, enabled) in specs.items():
        rows = []
        for k in sorted(set(cand) & set(anch)):
            c, a = cand[k], anch[k]
            if not (finite(c.get("pearson_pert")) and finite(a.get("pearson_pert"))):
                continue
            if not (finite(c.get("test_mmd_clamped")) and finite(a.get("test_mmd_clamped"))):
                continue
            use_candidate = k in enabled
            pp_delta = float(c["pearson_pert"]) - float(a["pearson_pert"]) if use_candidate else 0.0
            mmd_delta = float(c["test_mmd_clamped"]) - float(a["test_mmd_clamped"]) if use_candidate else 0.0
            rows.append(
                {
                    "dataset": k[0],
                    "condition": k[1],
                    "enabled": use_candidate,
                    "route_pp_delta": pp_delta,
                    "route_mmd_delta": mmd_delta,
                }
            )
        summary = summarize_rows(rows)
        summary["pass_noharm"] = bool(
            summary["n_enabled"] >= 1
            and summary["pp_dataset_equal_delta"] is not None
            and summary["pp_dataset_equal_delta"] >= -0.001
            and summary["mmd_dataset_equal_delta"] is not None
            and summary["mmd_dataset_equal_delta"] <= 0.001
            and summary["dataset_min_pp_delta"] is not None
            and summary["dataset_min_pp_delta"] > -0.010
            and summary["negative_dataset_tails_lt_minus_0p010"] == 0
            and summary["condition_hard_harm_frac_lt_minus_0p020"] <= 0.05
        )
        out[group] = summary
    out["seed_pass"] = bool(out["test_single"]["pass_noharm"] and out["family_gene"]["pass_noharm"])
    return out


def main() -> int:
    gate = load_json(GATE_JSON)
    split_policy = gate["groups"]["cross_background"]["best"]["policy"]
    family_policy = gate["groups"]["family_gene"]["best"]["policy"]
    enabled_split = internal_enabled("internal_val_cross_background_seen_gene_proxy", split_policy)
    enabled_family = internal_enabled("family_gene", family_policy)
    rows = []
    for seed in SEEDS:
        rows.append({"seed": seed, **route_seed(seed, enabled_split, enabled_family)})
    all_pass = all(r["seed_pass"] for r in rows)
    total_enabled_canonical = sum(
        r[group]["n_enabled"] for r in rows for group in ("test_single", "family_gene")
    )
    if all_pass:
        status = "uncertainty_gated_anchor_fallback_canonical_noharm_pass_route_freeze_next"
    elif total_enabled_canonical == 0:
        status = "uncertainty_gated_anchor_fallback_canonical_noharm_exact_noop_no_promotion"
    else:
        status = "uncertainty_gated_anchor_fallback_canonical_noharm_fail_close"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "route_freeze_authorized": all_pass,
        "boundary": {
            "selection_source": str(GATE_JSON),
            "canonical_use": "frozen no-harm veto only",
            "canonical_multi_used": False,
            "trackc_query_used": False,
        },
        "enabled_counts": {
            "test_single_policy_enabled_from_internal_cross_background": len(enabled_split),
            "family_gene_policy_enabled_from_internal_family_gene": len(enabled_family),
            "canonical_enabled_total_across_seed_groups": total_enabled_canonical,
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# LatentFM Uncertainty-Gated Anchor Fallback Canonical No-Harm",
        "",
        f"Status: `{status}`",
        f"Route-freeze authorized: `{all_pass}`",
        "",
        "## Boundary",
        "",
        f"- Selection source: `{GATE_JSON}`.",
        "- Enabled condition sets are recomputed from train-only/internal metrics only.",
        "- Canonical `test_single` and `family_gene` are used only as frozen no-harm veto.",
        "- Canonical multi and Track C held-out query are not read.",
        "",
        "## Rows",
        "",
        "| seed | group | enabled / n | pp delta | MMD delta | dataset min pp | hard-harm frac | pass |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        for group in ("test_single", "family_gene"):
            g = row[group]
            lines.append(
                "| {seed} | `{group}` | {en}/{n} | {pp:+.6f} | {mmd:+.6f} | {mn:+.6f} | {hh:.3f} | `{ps}` |".format(
                    seed=row["seed"],
                    group=group,
                    en=g["n_enabled"],
                    n=g["n_conditions"],
                    pp=g["pp_dataset_equal_delta"] or 0.0,
                    mmd=g["mmd_dataset_equal_delta"] or 0.0,
                    mn=g["dataset_min_pp_delta"] or 0.0,
                    hh=g["condition_hard_harm_frac_lt_minus_0p020"],
                    ps=g["pass_noharm"],
                )
            )
    lines += ["", "## Decision", ""]
    if all_pass:
        lines += [
            "- The frozen routed no-harm veto passes for all seeds.",
            "- Next action: write a route-freeze artifact, then external audit before any deployable wording or optional implementation smoke.",
            "- This still is not canonical multi success and does not touch Track C query.",
        ]
    elif total_enabled_canonical == 0:
        lines += [
            "- The frozen route is an exact no-op on canonical `test_single` and `family_gene`: no canonical conditions overlap the internal enabled set.",
            "- This is no-harm by construction, but it does not demonstrate a deployable improvement over `xverse_8k_anchor`.",
            "- Keep the internal pass as mechanism/headroom evidence; do not launch GPU or promote this route.",
        ]
    else:
        lines += [
            "- The frozen routed no-harm veto fails. Close this fallback as a deployable route.",
            "- Keep the internal pass as mechanism evidence only.",
        ]
    lines += ["", "## JSON", "", f"`{OUT_JSON}`", ""]
    OUT_MD.write_text("\n".join(lines))
    print(OUT_MD)
    print(OUT_JSON)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
