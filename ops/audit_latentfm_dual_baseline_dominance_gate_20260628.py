#!/usr/bin/env python3
"""Dual-baseline dominance gate for old Track A candidate rows.

Retrospective CPU-only gate: candidates must beat both frozen anchor and
source/control baselines on overlapping explicit Track A proxy rows.
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
CAND_DIR = ROOT / "reports/tracka_exact_tail_candidate_gate_20260627"
BASELINE_ROWS = ROOT / "reports/tracka_control_baseline_synthesis_20260628/joined_rows.csv"
OUT_DIR = ROOT / "reports/dual_baseline_dominance_gate_20260628"
OUT_JSON = ROOT / "reports/latentfm_dual_baseline_dominance_gate_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_DUAL_BASELINE_DOMINANCE_GATE_20260628.md"

GROUP_MAP = {
    "canonical_test_single": "all_test_single_proxy",
    "canonical_family_gene": "family_gene",
    "exact_cross_background_seen_gene": "cross_background_seen_gene_proxy",
    "exact_simple_single_unseen": "simple_single_unseen_global_gene_proxy",
}
TARGET_GROUPS = (
    "all_test_single_proxy",
    "cross_background_seen_gene_proxy",
    "family_gene",
    "simple_single_unseen_global_gene_proxy",
)


def infer_seed(name: str) -> str:
    if "seed43" in name:
        return "seed43"
    return "seed42"


def load_baselines() -> dict[tuple[str, str, str, str], dict[str, Any]]:
    out = {}
    with BASELINE_ROWS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["seed"], row["explicit_group"], row["dataset"], row["condition"])
            out[key] = {
                "anchor_pp": float(row["anchor_pp"]),
                "ctrl_pp": float(row["ctrl_pp"]),
                "anchor_mmd": float(row["anchor_mmd"]),
                "ctrl_mmd": float(row["ctrl_mmd"]),
            }
    return out


def bootstrap(vals: list[float], *, seed: int = 20260628) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(5000, arr.size))
    means = arr[idx].mean(axis=1)
    return {"ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975))}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds_dom: dict[str, list[float]] = defaultdict(list)
    by_ds_mmd: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_ds_dom[str(row["dataset"])].append(float(row["dominance_pp"]))
        by_ds_mmd[str(row["dataset"])].append(float(row["max_mmd_harm"]))
    ds_dom = [float(np.mean(v)) for v in by_ds_dom.values()]
    ds_mmd = [float(np.mean(v)) for v in by_ds_mmd.values()]
    return {
        "n": len(rows),
        "n_datasets": len(by_ds_dom),
        "candidate_pp": float(np.mean([r["candidate_pp"] for r in rows])) if rows else 0.0,
        "anchor_pp": float(np.mean([r["anchor_pp"] for r in rows])) if rows else 0.0,
        "ctrl_pp": float(np.mean([r["ctrl_pp"] for r in rows])) if rows else 0.0,
        "dominance_pp": float(np.mean(ds_dom)) if ds_dom else 0.0,
        "dominance_pp_ci": bootstrap(ds_dom),
        "dataset_min_dominance_pp": float(min(ds_dom)) if ds_dom else 0.0,
        "candidate_mmd": float(np.mean([r["candidate_mmd"] for r in rows])) if rows else 0.0,
        "max_mmd_harm": float(np.mean(ds_mmd)) if ds_mmd else 0.0,
        "dataset_max_mmd_harm": float(max(ds_mmd)) if ds_mmd else 0.0,
        "rows_beating_both_pp_fraction": float(np.mean([r["dominance_pp"] > 0.0 for r in rows])) if rows else 0.0,
    }


def evaluate_candidate(path: Path, baselines: dict[tuple[str, str, str, str], dict[str, Any]]) -> dict[str, Any]:
    name = path.name.replace("_paired_rows.csv", "")
    seed = infer_seed(name)
    matched: list[dict[str, Any]] = []
    missing = 0
    skipped_unmapped = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            explicit_group = GROUP_MAP.get(row["group"])
            if explicit_group is None:
                skipped_unmapped += 1
                continue
            key = (seed, explicit_group, row["dataset"], row["condition"])
            base = baselines.get(key)
            if base is None:
                missing += 1
                continue
            candidate_pp = float(row["candidate_pearson_pert"])
            candidate_mmd = float(row["candidate_test_mmd_clamped"])
            dominance = candidate_pp - max(float(base["anchor_pp"]), float(base["ctrl_pp"]))
            harm_anchor = candidate_mmd - float(base["anchor_mmd"])
            harm_ctrl = candidate_mmd - float(base["ctrl_mmd"])
            matched.append(
                {
                    "candidate": name,
                    "seed": seed,
                    "explicit_group": explicit_group,
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "candidate_pp": candidate_pp,
                    "candidate_mmd": candidate_mmd,
                    "anchor_pp": float(base["anchor_pp"]),
                    "ctrl_pp": float(base["ctrl_pp"]),
                    "anchor_mmd": float(base["anchor_mmd"]),
                    "ctrl_mmd": float(base["ctrl_mmd"]),
                    "dominance_pp": dominance,
                    "mmd_harm_vs_anchor": harm_anchor,
                    "mmd_harm_vs_ctrl": harm_ctrl,
                    "max_mmd_harm": max(harm_anchor, harm_ctrl),
                }
            )
    group_summary = {}
    reasons = []
    for group in TARGET_GROUPS:
        part = [r for r in matched if r["explicit_group"] == group]
        if not part:
            reasons.append(f"{name}_{group}_missing")
            continue
        s = summarize(part)
        group_summary[group] = s
        if s["dominance_pp"] < 0.015:
            reasons.append(f"{name}_{group}_dominance_pp_lt_0p015")
        if s["dominance_pp_ci"]["ci_low"] <= 0.0:
            reasons.append(f"{name}_{group}_dominance_ci_low_not_gt0")
        if s["dataset_min_dominance_pp"] < -0.005:
            reasons.append(f"{name}_{group}_dataset_min_dominance_lt_minus0p005")
        if s["max_mmd_harm"] > 0.001:
            reasons.append(f"{name}_{group}_mean_mmd_harm_gt_0p001")
        if s["dataset_max_mmd_harm"] > 0.01:
            reasons.append(f"{name}_{group}_dataset_mmd_harm_gt_0p01")
    status = "dual_baseline_dominance_candidate_pass_needs_external_review" if not reasons else "dual_baseline_dominance_candidate_fail"
    return {
        "candidate": name,
        "seed": seed,
        "status": status,
        "matched_rows": len(matched),
        "missing_baseline_rows": missing,
        "skipped_unmapped_rows": skipped_unmapped,
        "group_summary": group_summary,
        "decision_reasons": reasons,
        "matched": matched,
    }


def main() -> None:
    baselines = load_baselines()
    results = []
    all_rows = []
    for fname in sorted(glob.glob(str(CAND_DIR / "*_paired_rows.csv"))):
        result = evaluate_candidate(Path(fname), baselines)
        results.append({k: v for k, v in result.items() if k != "matched"})
        all_rows.extend(result["matched"])
    pass_candidates = [r["candidate"] for r in results if r["status"].endswith("_pass_needs_external_review")]
    status = "dual_baseline_dominance_gate_all_fail_no_gpu" if not pass_candidates else "dual_baseline_dominance_gate_has_pass_needs_review"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_csv = OUT_DIR / "dual_baseline_matched_rows.csv"
    summary_csv = OUT_DIR / "dual_baseline_candidate_summary.csv"
    with rows_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "candidate",
            "seed",
            "explicit_group",
            "dataset",
            "condition",
            "candidate_pp",
            "anchor_pp",
            "ctrl_pp",
            "dominance_pp",
            "candidate_mmd",
            "anchor_mmd",
            "ctrl_mmd",
            "max_mmd_harm",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in fields} for r in all_rows])
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "candidate",
            "seed",
            "status",
            "group",
            "n",
            "dominance_pp",
            "ci_low",
            "dataset_min_dominance_pp",
            "max_mmd_harm",
            "dataset_max_mmd_harm",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for group, s in result["group_summary"].items():
                writer.writerow(
                    {
                        "candidate": result["candidate"],
                        "seed": result["seed"],
                        "status": result["status"],
                        "group": group,
                        "n": s["n"],
                        "dominance_pp": s["dominance_pp"],
                        "ci_low": s["dominance_pp_ci"]["ci_low"],
                        "dataset_min_dominance_pp": s["dataset_min_dominance_pp"],
                        "max_mmd_harm": s["max_mmd_harm"],
                        "dataset_max_mmd_harm": s["dataset_max_mmd_harm"],
                    }
                )

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "retrospective_cpu_only": True,
            "candidate_dir": str(CAND_DIR),
            "baseline_rows": str(BASELINE_ROWS),
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "selection_weight": 0,
        },
        "pass_candidates": pass_candidates,
        "candidate_results": results,
        "outputs": {"matched_rows_csv": str(rows_csv), "summary_csv": str(summary_csv)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Dual-Baseline Dominance Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only retrospective over existing Track A candidate paired rows. Rows are evaluated only when they can be matched to frozen explicit Track A proxy rows with source/control baseline MMD. No training, inference, checkpoint selection, canonical multi selection, or Track C query is used.",
        "",
        "## Candidate Summary",
        "",
        "| candidate | seed | group | n | dominance pp | CI low | dataset min | MMD harm | dataset max MMD harm | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        for group in TARGET_GROUPS:
            s = result["group_summary"].get(group)
            if not s:
                continue
            lines.append(
                f"| `{result['candidate']}` | `{result['seed']}` | `{group}` | {s['n']} | "
                f"{s['dominance_pp']:+.6f} | {s['dominance_pp_ci']['ci_low']:+.6f} | "
                f"{s['dataset_min_dominance_pp']:+.6f} | {s['max_mmd_harm']:+.6f} | "
                f"{s['dataset_max_mmd_harm']:+.6f} | `{result['status']}` |"
            )
    lines.extend(["", "## Decision", ""])
    if pass_candidates:
        lines.append("At least one retrospective candidate passes the dual-baseline screen and requires external review before any GPU follow-up.")
    else:
        lines.append("No existing retrospective candidate beats both anchor and source/control under the dual-baseline pp/MMD/tail gate. Do not revive these candidates without fresh paired rows or a materially new model hypothesis.")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`", f"- summary CSV: `{summary_csv}`", f"- matched rows: `{rows_csv}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
