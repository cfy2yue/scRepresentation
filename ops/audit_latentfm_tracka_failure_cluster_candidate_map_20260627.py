#!/usr/bin/env python3
"""CPU-only Track A failure-cluster candidate map.

This diagnostic projects existing exact-tail candidate paired rows onto the
pre-reported Track A failure clusters. It does not train, run inference, select
checkpoints, read canonical multi for selection, or read Track C query.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
GATE_DIR = ROOT / "reports/tracka_exact_tail_candidate_gate_20260627"
FAILURE_CASES = ROOT / "reports/tracka_deployable_benchmark_failure_taxonomy_20260627/failure_cases.csv"
OUT_DIR = ROOT / "reports/tracka_failure_cluster_candidate_map_20260627"
OUT_JSON = ROOT / "reports/latentfm_tracka_failure_cluster_candidate_map_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_FAILURE_CLUSTER_CANDIDATE_MAP_20260627.md"


def fnum(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def cluster_for(dataset: str, condition: str) -> str:
    ds = str(dataset)
    if ds == "Adamson":
        return "adamson_stress_translation_like"
    if ds == "GasperiniShendure2019_lowMOI":
        return "gasperini_lowmoi"
    if ds.startswith("Jiang_"):
        return "jiang_cytokine_single_gene"
    if ds == "Replogle_RPE1essential":
        return "replogle_rpe1_essential_crispr"
    if ds.startswith("Nadig_"):
        return "nadig_cellline_single_gene"
    if ds == "Wessels":
        return "wessels_combinatorial_diagnostic"
    return "other"


def load_failure_keys() -> dict[str, set[tuple[str, str]]]:
    clusters: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with FAILURE_CASES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("seed") != "seed42":
                continue
            if row.get("group") not in {"test_single", "family_gene", "test_all", "test"}:
                continue
            try:
                pp = float(row.get("pearson_pert", "nan"))
            except ValueError:
                continue
            if pp > 0.05:
                continue
            ds = str(row.get("dataset", ""))
            cond = str(row.get("condition", ""))
            if not ds or not cond:
                continue
            clusters[cluster_for(ds, cond)].add((ds, cond))
    clusters.pop("other", None)
    return dict(clusters)


def summarize(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def read_candidate(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") not in {
                "canonical_test_single",
                "canonical_family_gene",
                "exact_simple_single_unseen",
                "exact_cross_background_seen_gene",
                "recurrent_simple_hard_tail",
                "recurrent_cross_background_hard_tail",
            }:
                continue
            pp = fnum(row.get("delta_pearson_pert"))
            mmd = fnum(row.get("delta_test_mmd_clamped"))
            if pp is None or mmd is None:
                continue
            row["delta_pearson_pert_f"] = pp
            row["delta_test_mmd_clamped_f"] = mmd
            out.append(row)
    return out


def main() -> None:
    failure_clusters = load_failure_keys()
    candidate_paths = sorted(p for p in GATE_DIR.glob("*_paired_rows.csv") if not p.name.startswith("anchor_selfcheck"))
    candidate_rows = {p.name.removesuffix("_paired_rows.csv"): read_candidate(p) for p in candidate_paths}

    rows_out: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for candidate, rows in candidate_rows.items():
        by_key_group: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            by_key_group[(row["dataset"], row["condition"], row["group"])] = row

        for cluster, keys in sorted(failure_clusters.items()):
            for group in ("canonical_test_single", "canonical_family_gene", "exact_cross_background_seen_gene", "recurrent_cross_background_hard_tail"):
                matched = []
                for ds, cond in keys:
                    row = by_key_group.get((ds, cond, group))
                    if row is not None:
                        matched.append(row)
                pp_vals = [float(r["delta_pearson_pert_f"]) for r in matched]
                mmd_vals = [float(r["delta_test_mmd_clamped_f"]) for r in matched]
                pp_summary = summarize(pp_vals)
                mmd_summary = summarize(mmd_vals)
                material_gain = (
                    pp_summary["n"] > 0
                    and float(pp_summary["mean"]) >= 0.05
                    and float(mmd_summary["mean"]) <= 0.001
                )
                summaries.append(
                    {
                        "candidate": candidate,
                        "cluster": cluster,
                        "group": group,
                        "n": pp_summary["n"],
                        "pp_mean": pp_summary["mean"],
                        "pp_median": pp_summary["median"],
                        "pp_min": pp_summary["min"],
                        "pp_max": pp_summary["max"],
                        "mmd_mean": mmd_summary["mean"],
                        "mmd_max": mmd_summary["max"],
                        "material_cluster_gain": material_gain,
                    }
                )
                rows_out.extend(
                    {
                        "candidate": candidate,
                        "cluster": cluster,
                        "group": group,
                        "dataset": r["dataset"],
                        "condition": r["condition"],
                        "delta_pearson_pert": float(r["delta_pearson_pert_f"]),
                        "delta_test_mmd_clamped": float(r["delta_test_mmd_clamped_f"]),
                    }
                    for r in matched
                )

    # A candidate would be GPU-interesting only if it has at least one material
    # exact/recurrent cluster gain and does not harm canonical exact groups on
    # average. This report is still diagnostic; it cannot authorize GPU alone.
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_candidate[row["candidate"]].append(row)
    decisions = []
    for candidate, ss in sorted(by_candidate.items()):
        material = [r for r in ss if r["material_cluster_gain"] and r["group"] in {"exact_cross_background_seen_gene", "recurrent_cross_background_hard_tail"}]
        canonical = [r for r in ss if r["group"] in {"canonical_test_single", "canonical_family_gene"} and r["n"]]
        canonical_pp_min = min((float(r["pp_mean"]) for r in canonical), default=0.0)
        canonical_mmd_max = max((float(r["mmd_mean"]) for r in canonical), default=0.0)
        reasons = []
        if not material:
            reasons.append("no_material_exact_or_recurrent_cluster_gain")
        if canonical_pp_min < -0.01:
            reasons.append("canonical_failure_cluster_pp_harm")
        if canonical_mmd_max > 0.001:
            reasons.append("canonical_failure_cluster_mmd_harm")
        decisions.append(
            {
                "candidate": candidate,
                "status": "tracka_failure_cluster_candidate_map_fail_no_gpu" if reasons else "tracka_failure_cluster_candidate_map_needs_trainonly_gate",
                "material_cluster_hits": len(material),
                "canonical_cluster_pp_min": canonical_pp_min,
                "canonical_cluster_mmd_max": canonical_mmd_max,
                "reasons": reasons,
            }
        )

    status = "tracka_failure_cluster_candidate_map_all_fail_no_gpu"
    if any(d["status"].endswith("needs_trainonly_gate") for d in decisions):
        status = "tracka_failure_cluster_candidate_map_has_candidate_needs_trainonly_gate_no_gpu"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "cluster_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()) if rows_out else ["candidate"])
        writer.writeheader()
        writer.writerows(rows_out)
    with (OUT_DIR / "cluster_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()) if summaries else ["candidate"])
        writer.writeheader()
        writer.writerows(summaries)
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "training": False,
            "inference": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
        },
        "failure_cluster_sizes": {k: len(v) for k, v in sorted(failure_clusters.items())},
        "n_candidates": len(candidate_rows),
        "decisions": decisions,
        "outputs": {
            "cluster_rows_csv": str(OUT_DIR / "cluster_rows.csv"),
            "cluster_summary_csv": str(OUT_DIR / "cluster_summary.csv"),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top = sorted(
        summaries,
        key=lambda r: (bool(r["material_cluster_gain"]), float(r["pp_mean"] or -999), -float(r["mmd_mean"] or 999)),
        reverse=True,
    )[:20]
    lines = [
        "# Track A Failure-Cluster Candidate Map",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU/report-only over existing exact-tail paired rows. No training, inference, checkpoint selection, canonical multi selection, or Track C query.",
        "",
        "## Failure Cluster Sizes",
        "",
    ]
    lines.extend(f"- `{k}`: `{len(v)}` conditions" for k, v in sorted(failure_clusters.items()))
    lines.extend(["", "## Candidate Decisions", "", "| candidate | status | material hits | canonical pp min | canonical MMD max | reasons |", "|---|---|---:|---:|---:|---|"])
    for d in decisions:
        lines.append(
            f"| `{d['candidate']}` | `{d['status']}` | {d['material_cluster_hits']} | "
            f"{d['canonical_cluster_pp_min']:+.6f} | {d['canonical_cluster_mmd_max']:+.6f} | "
            f"`{';'.join(d['reasons']) or 'none'}` |"
        )
    lines.extend(["", "## Top Cluster Rows", "", "| candidate | cluster | group | n | pp mean | MMD mean | material |", "|---|---|---|---:|---:|---:|---:|"])
    for r in top:
        lines.append(
            f"| `{r['candidate']}` | `{r['cluster']}` | `{r['group']}` | {r['n']} | "
            f"{float(r['pp_mean'] or 0):+.6f} | {float(r['mmd_mean'] or 0):+.6f} | `{r['material_cluster_gain']}` |"
        )
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`", f"- rows: `{OUT_DIR / 'cluster_rows.csv'}`", f"- summary: `{OUT_DIR / 'cluster_summary.csv'}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
