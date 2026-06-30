#!/usr/bin/env python3
"""Evaluate a candidate against anchor on exact Track A tail strata.

CPU/report-only. The inputs must already be posthoc JSONs with per-condition
``condition_metrics`` rows, typically from ``eval_condition_families``. This
script does not run model inference. It projects existing rows onto the exact
``simple_single_unseen``, exact cross-background, and seed-recurrent hard-tail
sets produced on 2026-06-27, then computes paired anchor-vs-candidate deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
EXACT_ROWS = ROOT / "reports/tracka_simple_single_unseen_exact_20260627/condition_rows.csv"
RECURRENT_ROWS = ROOT / "reports/tracka_recurrent_tail_gate_20260627/recurrent_tail_rows.csv"

DEFAULT_GROUPS = (
    "canonical_test_single",
    "canonical_family_gene",
    "exact_simple_single_unseen",
    "exact_cross_background_seen_gene",
    "recurrent_simple_hard_tail",
    "recurrent_cross_background_hard_tail",
)
METRICS = ("pearson_pert", "test_mmd_clamped")
LOWER_IS_BETTER = {"test_mmd_clamped"}
BOOTSTRAP_SEED_OFFSET = {"pearson_pert": 1009, "test_mmd_clamped": 2003}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):+.6f}"
    except Exception:
        return str(v)


def group_rows(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = payload.get("groups", {}).get(group, {}).get("condition_metrics", [])
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        ds = str(row.get("dataset", ""))
        cond = str(row.get("condition", ""))
        if ds and cond:
            out[(ds, cond)] = row
    return out


def load_exact_sets() -> dict[str, set[tuple[str, str]]]:
    sets = {
        "exact_simple_single_unseen": set(),
        "exact_cross_background_seen_gene": set(),
    }
    with EXACT_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # The exact row file contains seed42 and seed43 duplicates; row keys are
            # canonical condition identities, so a set removes seed duplication.
            key = (row["dataset"], row["condition"])
            if row["group"] == "simple_single_unseen":
                sets["exact_simple_single_unseen"].add(key)
            elif row["group"] == "cross_background_seen_gene_exact":
                sets["exact_cross_background_seen_gene"].add(key)
    with RECURRENT_ROWS.open(newline="", encoding="utf-8") as handle:
        simple_hard = set()
        cross_hard = set()
        for row in csv.DictReader(handle):
            if str(row.get("recurrent_hard_tail")).lower() != "true":
                continue
            key = (row["dataset"], row["condition"])
            if row["group"] == "simple_single_unseen":
                simple_hard.add(key)
            elif row["group"] == "cross_background_seen_gene_exact":
                cross_hard.add(key)
        sets["recurrent_simple_hard_tail"] = simple_hard
        sets["recurrent_cross_background_hard_tail"] = cross_hard
    return sets


def bootstrap(vals: list[float], *, seed: int, n_boot: int) -> dict[str, float | None]:
    if not vals:
        return {"ci_low": None, "ci_high": None, "p_gt0": None, "p_lt0": None}
    arr = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_gt0": float(np.mean(boots > 0.0)),
        "p_lt0": float(np.mean(boots < 0.0)),
    }


def paired_summary(
    *,
    group: str,
    keys: set[tuple[str, str]],
    anchor_rows: dict[tuple[str, str], dict[str, Any]],
    candidate_rows: dict[tuple[str, str], dict[str, Any]],
    n_boot: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired: list[dict[str, Any]] = []
    for key in sorted(keys):
        a = anchor_rows.get(key)
        c = candidate_rows.get(key)
        if not a or not c:
            continue
        row = {"group": group, "dataset": key[0], "condition": key[1]}
        ok = True
        for metric in METRICS:
            av = fnum(a.get(metric))
            cv = fnum(c.get(metric))
            if av is None or cv is None:
                ok = False
                break
            row[f"anchor_{metric}"] = av
            row[f"candidate_{metric}"] = cv
            row[f"delta_{metric}"] = cv - av
        if ok:
            paired.append(row)

    summaries: list[dict[str, Any]] = []
    ds_count = len({r["dataset"] for r in paired})
    for metric in METRICS:
        vals = [float(r[f"delta_{metric}"]) for r in paired]
        bs = bootstrap(vals, seed=seed + BOOTSTRAP_SEED_OFFSET[metric], n_boot=n_boot)
        if metric in LOWER_IS_BETTER:
            p_improve = bs["p_lt0"]
            p_harm = bs["p_gt0"]
        else:
            p_improve = bs["p_gt0"]
            p_harm = bs["p_lt0"]
        summaries.append(
            {
                "group": group,
                "metric": metric,
                "n_conditions": len(paired),
                "n_datasets": ds_count,
                "delta_mean": float(np.mean(vals)) if vals else None,
                "ci_low": bs["ci_low"],
                "ci_high": bs["ci_high"],
                "p_improve": p_improve,
                "p_harm": p_harm,
                "higher_is_better": metric not in LOWER_IS_BETTER,
            }
        )
    return paired, summaries


def status_from_summaries(summaries: list[dict[str, Any]]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    lookup = {(r["group"], r["metric"]): r for r in summaries}
    for group in ("exact_simple_single_unseen", "exact_cross_background_seen_gene"):
        pp = lookup.get((group, "pearson_pert"))
        mmd = lookup.get((group, "test_mmd_clamped"))
        if not pp or pp["n_conditions"] == 0:
            reasons.append(f"{group}_missing_pp_rows")
        elif (pp["delta_mean"] or 0.0) < 0.0 or (pp["p_harm"] or 0.0) > 0.35:
            reasons.append(f"{group}_pp_noharm_fail")
        if not mmd or mmd["n_conditions"] == 0:
            reasons.append(f"{group}_missing_mmd_rows")
        elif (mmd["delta_mean"] or 0.0) > 0.001 or (mmd["p_harm"] or 0.0) > 0.80:
            reasons.append(f"{group}_mmd_noharm_fail")
    tail_pp = lookup.get(("recurrent_cross_background_hard_tail", "pearson_pert"))
    if tail_pp and tail_pp["n_conditions"] > 0:
        if (tail_pp["delta_mean"] or 0.0) < 0.01 or (tail_pp["p_improve"] or 0.0) < 0.75:
            reasons.append("recurrent_cross_tail_material_gain_fail")
    else:
        reasons.append("recurrent_cross_tail_missing_rows")
    status = "candidate_exact_tail_gate_pass_gpu_candidate" if not reasons else "candidate_exact_tail_gate_fail_no_gpu"
    return status, reasons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-json", type=Path, required=True)
    ap.add_argument("--candidate-json", type=Path, required=True)
    ap.add_argument("--out-prefix", type=str, required=True)
    ap.add_argument("--title", type=str, default="LatentFM Track A Exact Tail Candidate Gate")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    anchor = load_json(args.anchor_json)
    candidate = load_json(args.candidate_json)
    exact_sets = load_exact_sets()

    anchor_test_single = group_rows(anchor, "test_single")
    candidate_test_single = group_rows(candidate, "test_single")
    anchor_family_gene = group_rows(anchor, "family_gene")
    candidate_family_gene = group_rows(candidate, "family_gene")
    anchor_exact_source = "test_single" if anchor_test_single else "family_gene"
    candidate_exact_source = "test_single" if candidate_test_single else "family_gene"
    anchor_exact_rows = anchor_test_single if anchor_test_single else anchor_family_gene
    candidate_exact_rows = candidate_test_single if candidate_test_single else candidate_family_gene

    groups: dict[str, tuple[set[tuple[str, str]], dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]] = {
        "canonical_test_single": (set(anchor_test_single) & set(candidate_test_single), anchor_test_single, candidate_test_single),
        "canonical_family_gene": (set(anchor_family_gene) & set(candidate_family_gene), anchor_family_gene, candidate_family_gene),
        "exact_simple_single_unseen": (exact_sets["exact_simple_single_unseen"], anchor_exact_rows, candidate_exact_rows),
        "exact_cross_background_seen_gene": (exact_sets["exact_cross_background_seen_gene"], anchor_exact_rows, candidate_exact_rows),
        "recurrent_simple_hard_tail": (exact_sets["recurrent_simple_hard_tail"], anchor_exact_rows, candidate_exact_rows),
        "recurrent_cross_background_hard_tail": (exact_sets["recurrent_cross_background_hard_tail"], anchor_exact_rows, candidate_exact_rows),
    }

    all_paired: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for group in DEFAULT_GROUPS:
        keys, arows, crows = groups[group]
        paired, summary = paired_summary(
            group=group,
            keys=keys,
            anchor_rows=arows,
            candidate_rows=crows,
            n_boot=int(args.n_boot),
            seed=int(args.seed),
        )
        all_paired.extend(paired)
        summaries.extend(summary)

    status, reasons = status_from_summaries(summaries)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    out_md = out_prefix.with_suffix(".md")
    out_rows = out_prefix.with_name(out_prefix.name + "_paired_rows.csv")

    fields = [
        "group",
        "dataset",
        "condition",
        "anchor_pearson_pert",
        "candidate_pearson_pert",
        "delta_pearson_pert",
        "anchor_test_mmd_clamped",
        "candidate_test_mmd_clamped",
        "delta_test_mmd_clamped",
    ]
    with out_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_paired)

    payload = {
        "status": status,
        "gpu_authorized": status.endswith("_pass_gpu_candidate"),
        "gate_reasons": reasons,
        "boundary": {
            "cpu_report_only": True,
            "no_training": True,
            "no_inference": True,
            "no_checkpoint_selection": True,
            "canonical_multi_selection_weight": 0,
            "trackc_query_read": False,
        },
        "inputs": {
            "anchor_json": str(args.anchor_json),
            "candidate_json": str(args.candidate_json),
            "exact_rows": str(EXACT_ROWS),
            "recurrent_rows": str(RECURRENT_ROWS),
            "anchor_exact_row_source": anchor_exact_source,
            "candidate_exact_row_source": candidate_exact_source,
        },
        "summaries": summaries,
        "outputs": {"json": str(out_json), "markdown": str(out_md), "paired_rows": str(out_rows)},
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        f"# {args.title}",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only over existing posthoc condition rows.",
        f"- Exact-tail rows are read from anchor `{anchor_exact_source}` and candidate `{candidate_exact_source}` groups.",
        "- No training, inference, checkpoint selection, canonical multi selection, or Track C query.",
        "",
        "## Gate Summary",
        "",
        "| group | metric | n cond | n ds | delta mean | CI95 | p improve | p harm |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row['group']}` | `{row['metric']}` | {row['n_conditions']} | {row['n_datasets']} | "
            f"{fmt(row['delta_mean'])} | [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | "
            f"{fmt(row['p_improve'])} | {fmt(row['p_harm'])} |"
        )
    lines += [
        "",
        "## Gate Reasons",
        "",
    ]
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines += [
        "",
        "## Outputs",
        "",
        f"- JSON: `{out_json}`",
        f"- Paired rows: `{out_rows}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
