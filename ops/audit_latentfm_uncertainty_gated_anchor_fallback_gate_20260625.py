#!/usr/bin/env python3
"""CPU gate for uncertainty-gated anchor fallback on true-cell budget128 6k.

This script uses only train-only/internal posthoc condition metrics.  It does
not read canonical multi, Track C query, or canonical no-harm outputs.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625"
REPORT_MD = ROOT / "reports/LATENTFM_UNCERTAINTY_GATED_ANCHOR_FALLBACK_GATE_20260625.md"
REPORT_JSON = ROOT / "reports/latentfm_uncertainty_gated_anchor_fallback_gate_20260625.json"
SEEDS = (42, 43, 44)
RUN_TMPL = "xverse_truecell_nested_budget128_tailstable_seed{seed}_6000"
GROUPS = {
    "cross_background": "internal_val_cross_background_seen_gene_proxy",
    "family_gene": "family_gene",
}


def load_group(seed: int, kind: str, group: str) -> list[dict]:
    path = (
        RUN_ROOT
        / RUN_TMPL.format(seed=seed)
        / "posthoc_eval_internal"
        / f"{'split_group' if group.startswith('internal_val_') else 'condition_family'}_eval_{kind}_internal_ode20.json"
    )
    with path.open() as f:
        data = json.load(f)
    rows = data["groups"][group]["condition_metrics"]
    return rows


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def key(row: dict) -> tuple[str, str]:
    return (str(row["dataset"]), str(row["condition"]))


def build_records(group: str) -> dict[tuple[str, str], dict]:
    by_key: dict[tuple[str, str], dict] = {}
    for seed in SEEDS:
        cand = {key(r): r for r in load_group(seed, "candidate", group)}
        anch = {key(r): r for r in load_group(seed, "anchor", group)}
        for k in sorted(set(cand) & set(anch)):
            c = cand[k]
            a = anch[k]
            if not finite(c.get("pearson_pert")) or not finite(a.get("pearson_pert")):
                continue
            if not finite(c.get("test_mmd")) or not finite(a.get("test_mmd")):
                continue
            rec = by_key.setdefault(
                k,
                {
                    "dataset": k[0],
                    "condition": k[1],
                    "pp_deltas": [],
                    "mmd_deltas": [],
                },
            )
            rec["pp_deltas"].append(float(c["pearson_pert"]) - float(a["pearson_pert"]))
            rec["mmd_deltas"].append(float(c["test_mmd"]) - float(a["test_mmd"]))
    return {k: v for k, v in by_key.items() if len(v["pp_deltas"]) == len(SEEDS)}


def summarize_records(records: dict[tuple[str, str], dict]) -> dict[tuple[str, str], dict]:
    out = {}
    for k, rec in records.items():
        pp = rec["pp_deltas"]
        mmd = rec["mmd_deltas"]
        pp_sd = statistics.pstdev(pp) if len(pp) > 1 else 0.0
        mmd_sd = statistics.pstdev(mmd) if len(mmd) > 1 else 0.0
        out[k] = {
            **rec,
            "pp_mean": statistics.mean(pp),
            "pp_min": min(pp),
            "pp_sd": pp_sd,
            "pp_lcb": statistics.mean(pp) - 1.0 * pp_sd,
            "mmd_mean": statistics.mean(mmd),
            "mmd_max": max(mmd),
            "mmd_sd": mmd_sd,
        }
    return out


def dataset_equal_mean(rows: list[dict], field: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get(field)
        if finite(v):
            by_ds[r["dataset"]].append(float(v))
    vals = [statistics.mean(vs) for vs in by_ds.values() if vs]
    return statistics.mean(vals) if vals else None


def dataset_rows(rows: list[dict], field: str) -> list[dict]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if finite(r.get(field)):
            by_ds[r["dataset"]].append(float(r[field]))
    return [
        {"dataset": ds, "mean": statistics.mean(vs), "n": len(vs)}
        for ds, vs in sorted(by_ds.items())
        if vs
    ]


def apply_policy(
    records: dict[tuple[str, str], dict],
    min_lcb: float,
    max_sd: float,
    max_mmd_mean: float,
    max_mmd_max: float,
    force_keys: set[tuple[str, str]] | None = None,
) -> dict:
    rows = []
    for k, rec in records.items():
        enabled = (
            rec["pp_lcb"] >= min_lcb
            and rec["pp_sd"] <= max_sd
            and rec["mmd_mean"] <= max_mmd_mean
            and rec["mmd_max"] <= max_mmd_max
        )
        if force_keys is not None:
            enabled = k in force_keys
        route_pp_delta = rec["pp_mean"] if enabled else 0.0
        route_mmd_delta = rec["mmd_mean"] if enabled else 0.0
        rows.append(
            {
                **rec,
                "enabled": enabled,
                "route_pp_delta": route_pp_delta,
                "route_mmd_delta": route_mmd_delta,
                "hard_harm": route_pp_delta < -0.02,
            }
        )
    pp_mean = dataset_equal_mean(rows, "route_pp_delta")
    mmd_mean = dataset_equal_mean(rows, "route_mmd_delta")
    ds_rows = dataset_rows(rows, "route_pp_delta")
    enabled = [r for r in rows if r["enabled"]]
    enabled_by_ds: dict[str, int] = defaultdict(int)
    for r in enabled:
        enabled_by_ds[r["dataset"]] += 1
    max_dataset_weight = max(enabled_by_ds.values()) / len(enabled) if enabled else 0.0
    return {
        "n_conditions": len(rows),
        "n_enabled": len(enabled),
        "enabled_fraction": len(enabled) / len(rows) if rows else 0.0,
        "pp_dataset_equal_mean": pp_mean,
        "mmd_dataset_equal_mean": mmd_mean,
        "dataset_rows": ds_rows,
        "dataset_min_pp": min((r["mean"] for r in ds_rows), default=None),
        "negative_dataset_tails_lt_minus_0p010": sum(1 for r in ds_rows if r["mean"] < -0.010),
        "hard_harm_fraction": sum(1 for r in rows if r["hard_harm"]) / len(rows) if rows else 0.0,
        "max_enabled_dataset_weight": max_dataset_weight,
        "enabled_preview": [
            {
                "dataset": r["dataset"],
                "condition": r["condition"],
                "pp_mean": r["pp_mean"],
                "pp_lcb": r["pp_lcb"],
                "pp_sd": r["pp_sd"],
                "mmd_mean": r["mmd_mean"],
            }
            for r in sorted(enabled, key=lambda x: x["pp_lcb"], reverse=True)[:20]
        ],
    }


def shuffle_control(records: dict[tuple[str, str], dict], n_enabled: int, actual_pp: float, n: int = 1000) -> dict:
    rng = random.Random(20260625)
    keys = list(records)
    vals = []
    if not keys or n_enabled <= 0:
        return {"n": 0, "mean": None, "p95": None, "p_ge_actual": None}
    for _ in range(n):
        selected = set(rng.sample(keys, min(n_enabled, len(keys))))
        res = apply_policy(records, 0.0, 999.0, 999.0, 999.0, force_keys=selected)
        vals.append(res["pp_dataset_equal_mean"] or 0.0)
    vals.sort()
    return {
        "n": n,
        "mean": statistics.mean(vals),
        "p95": vals[int(0.95 * (len(vals) - 1))],
        "p_ge_actual": sum(1 for v in vals if v >= actual_pp) / len(vals),
    }


def main() -> int:
    group_results = {}
    policies = []
    for min_lcb in (0.0, 0.005, 0.010, 0.020):
        for max_sd in (0.05, 0.10, 0.20):
            for max_mmd_mean in (0.0005, 0.001, 0.005):
                policies.append((min_lcb, max_sd, max_mmd_mean, 0.010))

    for label, group in GROUPS.items():
        records = summarize_records(build_records(group))
        candidates = []
        for min_lcb, max_sd, max_mmd_mean, max_mmd_max in policies:
            res = apply_policy(records, min_lcb, max_sd, max_mmd_mean, max_mmd_max)
            res["policy"] = {
                "min_pp_lcb": min_lcb,
                "max_pp_sd": max_sd,
                "max_mmd_mean": max_mmd_mean,
                "max_mmd_max": max_mmd_max,
            }
            pp = res["pp_dataset_equal_mean"]
            # Predeclared strict pass-like criteria.
            res["passes"] = bool(
                res["n_enabled"] >= 40
                and pp is not None
                and pp >= 0.015
                and (res["mmd_dataset_equal_mean"] is not None and res["mmd_dataset_equal_mean"] <= 0.0005)
                and (res["dataset_min_pp"] is not None and res["dataset_min_pp"] > -0.010)
                and res["hard_harm_fraction"] <= 0.25
                and res["max_enabled_dataset_weight"] <= 0.35
            )
            candidates.append(res)
        candidates.sort(
            key=lambda r: (
                r["passes"],
                r["pp_dataset_equal_mean"] if r["pp_dataset_equal_mean"] is not None else -999,
                r["n_enabled"],
            ),
            reverse=True,
        )
        best = candidates[0]
        best["shuffle_control"] = shuffle_control(
            records, best["n_enabled"], best["pp_dataset_equal_mean"] or 0.0
        )
        best["passes_with_control"] = bool(
            best["passes"]
            and best["shuffle_control"]["p_ge_actual"] is not None
            and best["shuffle_control"]["p_ge_actual"] <= 0.05
            and best["pp_dataset_equal_mean"] > (best["shuffle_control"]["p95"] or 0.0)
        )
        group_results[label] = {
            "n_aligned_conditions": len(records),
            "best": best,
            "top5": candidates[:5],
        }

    overall_pass = all(v["best"]["passes_with_control"] for v in group_results.values())
    status = (
        "uncertainty_gated_anchor_fallback_pass_gpu_candidate"
        if overall_pass
        else "uncertainty_gated_anchor_fallback_fail_no_gpu"
    )
    result = {
        "status": status,
        "gpu_authorized": overall_pass,
        "boundary": {
            "uses_only": "train-only/internal condition_metrics from budget128 6k seeds 42/43/44",
            "does_not_use": ["canonical_multi", "TrackC_query", "canonical_noharm_for_selection"],
        },
        "groups": group_results,
    }
    REPORT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LatentFM Uncertainty-Gated Anchor Fallback Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{overall_pass}`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate using existing budget128 6k train-only/internal condition metrics.",
        "- Seeds: `42/43/44`.",
        "- Does not read canonical multi, Track C held-out query, or use canonical no-harm for selection.",
        "- Simulated route: enable candidate only when seed-level pp LCB is positive/stable and MMD is safe; otherwise fallback to anchor.",
        "",
        "## Results",
        "",
        "| group | aligned conditions | enabled | pp mean | MMD mean | dataset min pp | hard-harm frac | max enabled dataset weight | shuffle p95 | p(shuffle>=actual) | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, payload in group_results.items():
        b = payload["best"]
        sc = b["shuffle_control"]
        lines.append(
            "| {label} | {n} | {en} | {pp:+.6f} | {mmd:+.6f} | {mn:+.6f} | {hh:.3f} | {mw:.3f} | {p95:+.6f} | {pge:.3f} | `{ps}` |".format(
                label=label,
                n=payload["n_aligned_conditions"],
                en=b["n_enabled"],
                pp=b["pp_dataset_equal_mean"] or 0.0,
                mmd=b["mmd_dataset_equal_mean"] or 0.0,
                mn=b["dataset_min_pp"] or 0.0,
                hh=b["hard_harm_fraction"],
                mw=b["max_enabled_dataset_weight"],
                p95=sc["p95"] or 0.0,
                pge=sc["p_ge_actual"] if sc["p_ge_actual"] is not None else 1.0,
                ps=b["passes_with_control"],
            )
        )
    lines += [
        "",
        "## Decision",
        "",
    ]
    if overall_pass:
        lines += [
            "- The gate passes. This authorizes designing exactly one bounded GPU smoke that implements the same fallback policy.",
            "- Promotion still requires a fresh RUN_STATUS, resource audit, train-only gate, route freeze, and frozen canonical single/family no-harm.",
        ]
    else:
        lines += [
            "- The gate fails. Do not launch a GPU fallback route from this policy.",
            "- The current seed-disagreement/LCB signal is insufficient to turn the true-cell budget128 6k mechanism into a tail-safe route.",
            "- Scaling remains a mechanism/failure map until another tail-protection gate passes.",
        ]
    lines += [
        "",
        "## JSON",
        "",
        f"`{REPORT_JSON}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))
    print(REPORT_MD)
    print(REPORT_JSON)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
