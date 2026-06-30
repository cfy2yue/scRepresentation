#!/usr/bin/env python3
"""Stratum-level non-noop tail-protection gate for true-cell budget128 6k.

Exact condition-level uncertainty gating was canonical-noop.  This CPU-only
gate derives broader dataset/type/source strata from train-only/internal
multi-seed metrics, freezes them, then uses canonical single/family only as a
non-noop + no-harm veto.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
INTERNAL_RUN_ROOT = ROOT / "runs/latentfm_true_cell_count_budget128_tail_stability_6k_20260625"
CANON_RUN_ROOT = ROOT / "runs/latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625"
ANCHOR_SPLIT_JSON = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_canonical/split_group_eval_anchor_ode20_canonical.json"
ANCHOR_FAMILY_JSON = ROOT / "runs/latentfm_xverse_scaling_canonical_noharm_20260624/xverse_scaling_cap120_all_3k_seed42/posthoc_eval_canonical/condition_family_eval_anchor_ode20_canonical.json"
S0_TSV = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
OUT_JSON = ROOT / "reports/latentfm_truecell_stratum_tail_protection_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUECELL_STRATUM_TAIL_PROTECTION_GATE_20260625.md"

SEEDS = (42, 43, 44)
RUN_TMPL = "xverse_truecell_nested_budget128_tailstable_seed{seed}_6000"
INTERNAL_GROUPS = {
    "test_single": "internal_val_cross_background_seen_gene_proxy",
    "family_gene": "family_gene",
}
FEATURES = ("dataset", "perturbation_type", "cell_background_source", "source_label")


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def load_s0() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    with S0_TSV.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            out[(row["dataset"], row["condition"])] = row
    return out


def meta_feature(meta: dict[str, str], dataset: str, feature: str) -> str:
    if feature == "dataset":
        return dataset
    return meta.get(feature) or "UNKNOWN"


def load_internal_group(seed: int, kind: str, group: str) -> list[dict]:
    fn = "split_group_eval" if group.startswith("internal_val_") else "condition_family_eval"
    path = INTERNAL_RUN_ROOT / RUN_TMPL.format(seed=seed) / "posthoc_eval_internal" / f"{fn}_{kind}_internal_ode20.json"
    return load_json(path)["groups"][group]["condition_metrics"]


def internal_records(group: str) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for seed in SEEDS:
        cand = {key(r): r for r in load_internal_group(seed, "candidate", group)}
        anch = {key(r): r for r in load_internal_group(seed, "anchor", group)}
        for k in sorted(set(cand) & set(anch)):
            c, a = cand[k], anch[k]
            if not (finite(c.get("pearson_pert")) and finite(a.get("pearson_pert"))):
                continue
            if not (finite(c.get("test_mmd")) and finite(a.get("test_mmd"))):
                continue
            rec = records.setdefault(k, {"dataset": k[0], "condition": k[1], "pp": [], "mmd": []})
            rec["pp"].append(float(c["pearson_pert"]) - float(a["pearson_pert"]))
            rec["mmd"].append(float(c["test_mmd"]) - float(a["test_mmd"]))
    return {k: v for k, v in records.items() if len(v["pp"]) == len(SEEDS)}


def summarize_values(records: list[dict[str, Any]]) -> dict[str, Any]:
    pp_means = [statistics.mean(r["pp"]) for r in records]
    mmd_means = [statistics.mean(r["mmd"]) for r in records]
    pp_seed_means = [statistics.mean([r["pp"][i] for r in records]) for i in range(len(SEEDS))]
    pp_sd = statistics.pstdev(pp_seed_means) if len(pp_seed_means) > 1 else 0.0
    by_ds: dict[str, list[float]] = defaultdict(list)
    for rec, val in zip(records, pp_means):
        by_ds[rec["dataset"]].append(val)
    ds_rows = [{"dataset": ds, "mean": statistics.mean(vals), "n": len(vals)} for ds, vals in sorted(by_ds.items())]
    return {
        "n": len(records),
        "n_datasets": len(by_ds),
        "pp_mean": statistics.mean(pp_means),
        "pp_seed_lcb": statistics.mean(pp_seed_means) - pp_sd,
        "pp_min_condition": min(pp_means),
        "mmd_mean": statistics.mean(mmd_means),
        "mmd_max_condition": max(mmd_means),
        "dataset_min_pp": min((r["mean"] for r in ds_rows), default=0.0),
        "condition_hard_harm_frac_lt_minus_0p010": sum(v < -0.010 for v in pp_means) / len(pp_means),
        "dataset_rows": ds_rows,
    }


def derive_enabled_features(group: str, s0: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    records = internal_records(group)
    candidates = []
    for feature in FEATURES:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for k, rec in records.items():
            meta = s0.get(k, {})
            buckets[meta_feature(meta, rec["dataset"], feature)].append(rec)
        for value, vals in buckets.items():
            if value == "UNKNOWN" or len(vals) < 8:
                continue
            summary = summarize_values(vals)
            reasons = []
            if summary["n"] < 8:
                reasons.append("n_lt_8")
            if feature != "dataset" and summary["n_datasets"] < 2:
                reasons.append("n_datasets_lt_2")
            if summary["pp_seed_lcb"] < 0.015:
                reasons.append("pp_seed_lcb_lt_0p015")
            if summary["dataset_min_pp"] < -0.005:
                reasons.append("dataset_min_pp_lt_minus_0p005")
            if summary["condition_hard_harm_frac_lt_minus_0p010"] > 0.10:
                reasons.append("hard_harm_frac_gt_0p10")
            if summary["mmd_mean"] > 0.0005:
                reasons.append("mmd_mean_gt_0p0005")
            if summary["mmd_max_condition"] > 0.005:
                reasons.append("mmd_max_gt_0p005")
            candidates.append(
                {
                    "feature": feature,
                    "value": value,
                    "summary": summary,
                    "passes": not reasons,
                    "reasons": reasons,
                }
            )
    passing = [c for c in candidates if c["passes"]]
    return {
        "group": group,
        "enabled_features": [(c["feature"], c["value"]) for c in passing],
        "passing": passing,
        "top_candidates": sorted(
            candidates,
            key=lambda x: (x["passes"], x["summary"]["pp_seed_lcb"], x["summary"]["n"]),
            reverse=True,
        )[:20],
    }


def group_rows(payload: dict, group: str) -> dict[tuple[str, str], dict]:
    return {key(r): r for r in payload["groups"][group].get("condition_metrics", [])}


def enabled_by_features(k: tuple[str, str], enabled: set[tuple[str, str]], s0: dict[tuple[str, str], dict[str, str]]) -> bool:
    dataset, _ = k
    meta = s0.get(k, {})
    for feature, value in enabled:
        if meta_feature(meta, dataset, feature) == value:
            return True
    return False


def dataset_equal_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if finite(r.get(field)):
            by_ds[str(r["dataset"])].append(float(r[field]))
    vals = [statistics.mean(vs) for vs in by_ds.values() if vs]
    return statistics.mean(vals) if vals else None


def summarize_route_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_ds[str(r["dataset"])].append(float(r["route_pp_delta"]))
    ds_rows = [{"dataset": ds, "mean": statistics.mean(vals), "n": len(vals)} for ds, vals in sorted(by_ds.items())]
    enabled = [r for r in rows if r["enabled"]]
    return {
        "n_conditions": len(rows),
        "n_enabled": len(enabled),
        "enabled_fraction": len(enabled) / len(rows) if rows else 0.0,
        "pp_dataset_equal_delta": dataset_equal_mean(rows, "route_pp_delta"),
        "mmd_dataset_equal_delta": dataset_equal_mean(rows, "route_mmd_delta"),
        "dataset_min_pp_delta": min((r["mean"] for r in ds_rows), default=None),
        "negative_dataset_tails_lt_minus_0p010": sum(1 for r in ds_rows if r["mean"] < -0.010),
        "condition_hard_harm_frac_lt_minus_0p020": sum(1 for r in rows if r["route_pp_delta"] < -0.020) / len(rows) if rows else 0.0,
        "dataset_rows": ds_rows,
    }


def route_seed(seed: int, enabled_split: set[tuple[str, str]], enabled_family: set[tuple[str, str]], s0: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
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
            use_candidate = enabled_by_features(k, enabled, s0)
            rows.append(
                {
                    "dataset": k[0],
                    "condition": k[1],
                    "enabled": use_candidate,
                    "route_pp_delta": float(c["pearson_pert"]) - float(a["pearson_pert"]) if use_candidate else 0.0,
                    "route_mmd_delta": float(c["test_mmd_clamped"]) - float(a["test_mmd_clamped"]) if use_candidate else 0.0,
                }
            )
        summary = summarize_route_rows(rows)
        summary["pass_noharm"] = bool(
            summary["n_enabled"] >= 25
            and summary["enabled_fraction"] >= 0.05
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
    s0 = load_s0()
    derived = {
        route_group: derive_enabled_features(internal_group, s0)
        for route_group, internal_group in INTERNAL_GROUPS.items()
    }
    enabled_split = set(tuple(x) for x in derived["test_single"]["enabled_features"])
    enabled_family = set(tuple(x) for x in derived["family_gene"]["enabled_features"])
    rows = [route_seed(seed, enabled_split, enabled_family, s0) | {"seed": seed} for seed in SEEDS]
    all_pass = all(r["seed_pass"] for r in rows)
    total_enabled = sum(r[g]["n_enabled"] for r in rows for g in ("test_single", "family_gene"))
    if all_pass:
        status = "truecell_stratum_tail_protection_pass_gpu_candidate"
    elif total_enabled == 0:
        status = "truecell_stratum_tail_protection_exact_noop_fail_no_gpu"
    else:
        status = "truecell_stratum_tail_protection_noharm_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": all_pass,
        "route_freeze_authorized": all_pass,
        "derived": derived,
        "enabled_features": {
            "test_single": sorted(list(enabled_split)),
            "family_gene": sorted(list(enabled_family)),
        },
        "canonical_rows": rows,
        "boundary": {
            "selection_source": "train-only/internal budget128 6k multi-seed condition metrics plus S0 provenance strata",
            "canonical_use": "frozen non-noop and no-harm veto only",
            "canonical_multi_used": False,
            "trackc_query_used": False,
            "gpu_used": False,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM True-Cell Stratum Tail-Protection Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{all_pass}`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate using train-only/internal budget128 6k metrics and S0 provenance strata.",
        "- Canonical `test_single`/`family_gene` are used only after route freeze as non-noop/no-harm veto.",
        "- No canonical multi, Track C query, training, inference, or GPU is used.",
        "",
        "## Enabled Features",
        "",
        f"- test_single route features: `{sorted(list(enabled_split))}`",
        f"- family_gene route features: `{sorted(list(enabled_family))}`",
        "",
        "## Canonical Veto Rows",
        "",
        "| seed | group | enabled / n | pp delta | MMD delta | dataset min pp | hard-harm frac | pass |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        for group in ("test_single", "family_gene"):
            s = row[group]
            lines.append(
                "| {seed} | `{group}` | {en}/{n} | {pp:+.6f} | {mmd:+.6f} | {mn:+.6f} | {hh:.3f} | `{ps}` |".format(
                    seed=row["seed"],
                    group=group,
                    en=s["n_enabled"],
                    n=s["n_conditions"],
                    pp=float(s["pp_dataset_equal_delta"] or 0.0),
                    mmd=float(s["mmd_dataset_equal_delta"] or 0.0),
                    mn=float(s["dataset_min_pp_delta"] or 0.0),
                    hh=float(s["condition_hard_harm_frac_lt_minus_0p020"]),
                    ps=s["pass_noharm"],
                )
            )
    lines += [
        "",
        "## Top Train-Only Strata",
        "",
        "| route group | feature | value | n | pp LCB | dataset min | hard-harm frac | pass | reasons |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for route_group, info in derived.items():
        for cand in info["top_candidates"][:10]:
            s = cand["summary"]
            lines.append(
                f"| `{route_group}` | `{cand['feature']}` | `{cand['value']}` | {s['n']} | {s['pp_seed_lcb']:+.6f} | {s['dataset_min_pp']:+.6f} | {s['condition_hard_harm_frac_lt_minus_0p010']:.3f} | `{cand['passes']}` | {', '.join(cand['reasons']) or 'none'} |"
            )
    lines += [
        "",
        "## Decision",
        "",
        f"- total canonical enabled rows across seeds/groups: `{total_enabled}`",
        "- A broader stratum route only matters if it is non-noop and passes frozen canonical no-harm.",
        "- If this fails, keep true-cell budget128 6k as mechanism evidence and require a genuinely new tail-protection mechanism.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
