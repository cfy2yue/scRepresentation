#!/usr/bin/env python3
"""CPU gate for Track A same-gene cross-background residual transport.

This is a no-training diagnostic over train-only/internal residual-forensics
rows. It tests whether using same-gene cross-background residual proxies
(``gene_raw_mean`` and related train-only baselines) has enough broad,
non-oracle signal to justify a future adapter/smoke.

It does not train, infer, select checkpoints, read canonical multi for
selection, read Track C query, or use GPU. Candidate MMD is unavailable for
these scalar proxy rows, so this gate cannot directly authorize GPU.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
FORENSICS = ROOT / "reports/latentfm_xverse_tracka_residual_forensics_conditions_20260622.csv"
EXACT_CROSS = ROOT / "reports/tracka_cross_background_seen_gene_exact_20260627/cross_background_seen_gene_rows.csv"
FAILURES = ROOT / "reports/tracka_deployable_benchmark_failure_taxonomy_20260627/failure_cases.csv"
OUT_DIR = ROOT / "reports/tracka_samegene_transport_cpu_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_tracka_samegene_transport_cpu_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_SAMEGENE_TRANSPORT_CPU_GATE_20260627.md"

GROUP = "internal_val_cross_background_seen_gene_proxy"
CANDIDATES = {
    "samegene_gene_raw_mean": "gene_raw_mean",
    "global_gene_mean_control": "global_mean",
    "shrink_k8_control": "shrink_k8",
    "dataset_mean_control": "dataset_mean",
}


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def read_forensics() -> list[dict[str, Any]]:
    rows = []
    with FORENSICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("group") != GROUP:
                continue
            anchor = fnum(row.get("anchor_pearson_pert"))
            if anchor is None:
                continue
            out = {
                "dataset": str(row.get("dataset", "")),
                "condition": str(row.get("condition", "")),
                "gene": str(row.get("gene", "")),
                "gene_train_count": int(float(row.get("gene_train_count", 0) or 0)),
                "anchor_pearson_pert": anchor,
                "anchor_mmd_clamped": fnum(row.get("anchor_mmd_clamped")),
            }
            ok = True
            for name, col in CANDIDATES.items():
                val = fnum(row.get(col))
                if val is None:
                    ok = False
                    break
                out[name] = val
                out[f"{name}_delta"] = val - anchor
            if ok:
                rows.append(out)
    return rows


def read_exact_cross_footprint() -> dict[str, Any]:
    rows = []
    with EXACT_CROSS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("seed") != "seed42":
                continue
            rows.append(row)
    return {
        "n_seed42_rows": len(rows),
        "n_datasets": len({r.get("dataset") for r in rows}),
        "n_genes": len({r.get("gene") for r in rows}),
        "path": str(EXACT_CROSS),
    }


def read_failure_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {(r["dataset"], r["condition"]) for r in rows}
    fail_keys = set()
    with FAILURES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("seed") != "seed42":
                continue
            if row.get("group") not in {"test", "test_single", "family_gene", "test_all"}:
                continue
            pp = fnum(row.get("pearson_pert"))
            if pp is None or pp > 0.05:
                continue
            fail_keys.add((str(row.get("dataset", "")), str(row.get("condition", ""))))
    return {
        "failure_keys": len(fail_keys),
        "overlap_with_internal_proxy_rows": len(keys & fail_keys),
    }


def hierarchical_bootstrap(vals_by_dataset: dict[str, list[float]], *, seed: int, n_boot: int = 5000) -> dict[str, float | None]:
    datasets = sorted(k for k, v in vals_by_dataset.items() if v)
    if not datasets:
        return {"ci_low": None, "ci_high": None, "p_gt0": None, "p_lt0": None}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(datasets, size=len(datasets), replace=True)
        sampled = []
        for ds in chosen:
            arr = np.asarray(vals_by_dataset[ds], dtype=float)
            idx = rng.integers(0, len(arr), size=len(arr))
            sampled.extend(arr[idx].tolist())
        boots.append(float(np.mean(sampled)))
    arr = np.asarray(boots, dtype=float)
    return {
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_gt0": float(np.mean(arr > 0.0)),
        "p_lt0": float(np.mean(arr < 0.0)),
    }


def shuffled_control(rows: list[dict[str, Any]], candidate: str, *, seed: int, n_perm: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    actual = float(np.mean([float(row[f"{candidate}_delta"]) for row in rows]))
    means = []
    for _ in range(n_perm):
        vals = []
        for ds_rows in by_ds.values():
            cand_vals = np.asarray([float(row[candidate]) for row in ds_rows], dtype=float)
            anchors = np.asarray([float(row["anchor_pearson_pert"]) for row in ds_rows], dtype=float)
            shuffled = rng.permutation(cand_vals)
            vals.extend((shuffled - anchors).tolist())
        means.append(float(np.mean(vals)))
    arr = np.asarray(means, dtype=float)
    return {
        "actual_mean": actual,
        "shuffle_mean": float(arr.mean()),
        "shuffle_p_ge_actual": float(np.mean(arr >= actual)),
        "shuffle_q95": float(np.quantile(arr, 0.95)),
        "mean_statistic_is_permutation_invariant": True,
    }


def summarize_candidate(rows: list[dict[str, Any]], candidate: str, idx: int) -> dict[str, Any]:
    deltas = [float(row[f"{candidate}_delta"]) for row in rows]
    by_ds: dict[str, list[float]] = defaultdict(list)
    by_count: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = float(row[f"{candidate}_delta"])
        by_ds[row["dataset"]].append(val)
        count_bucket = "count0" if int(row["gene_train_count"]) == 0 else "count_ge1"
        by_count[count_bucket].append(val)
    ds_summary = [
        {
            "dataset": ds,
            "n": len(vals),
            "delta_mean": float(np.mean(vals)),
            "delta_min": float(np.min(vals)),
        }
        for ds, vals in sorted(by_ds.items())
    ]
    bs = hierarchical_bootstrap(by_ds, seed=20260627 + idx * 17)
    shuf = shuffled_control(rows, candidate, seed=20260627 + idx * 101)
    reasons = []
    mean = float(np.mean(deltas)) if deltas else 0.0
    ds_min = min((row["delta_mean"] for row in ds_summary), default=0.0)
    if mean < 0.01:
        reasons.append("mean_delta_lt_0p01")
    if bs["ci_low"] is None or float(bs["ci_low"]) <= 0.0:
        reasons.append("hierarchical_ci_low_not_above_0")
    if ds_min < -0.02:
        reasons.append("dataset_min_below_minus_0p02")
    # The primary statistic here is a mean of candidate-anchor deltas. Any
    # permutation of scalar candidate values preserves the global sum, so this
    # shuffle is provenance only rather than a valid pass/fail criterion.
    reasons.append("candidate_mmd_unavailable_no_gpu")
    return {
        "candidate": candidate,
        "status": "samegene_transport_proxy_fail_no_gpu" if reasons else "samegene_transport_proxy_needs_mmd_gate_no_gpu",
        "n": len(deltas),
        "n_datasets": len(by_ds),
        "delta_mean": mean,
        "delta_median": float(np.median(deltas)) if deltas else None,
        "dataset_min": ds_min,
        "hierarchical_bootstrap": bs,
        "shuffle_control": shuf,
        "count_bucket_mean": {k: float(np.mean(v)) for k, v in sorted(by_count.items())},
        "dataset_summary": ds_summary,
        "reasons": reasons,
    }


def main() -> None:
    rows = read_forensics()
    reports = [summarize_candidate(rows, name, idx) for idx, name in enumerate(CANDIDATES)]
    ranked = sorted(reports, key=lambda r: (float(r["delta_mean"]), -float(abs(r["dataset_min"]))), reverse=True)
    status = "tracka_samegene_transport_cpu_gate_fail_close_no_gpu"
    if any(r["status"].endswith("needs_mmd_gate_no_gpu") for r in reports):
        status = "tracka_samegene_transport_cpu_gate_needs_mmd_no_gpu"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "samegene_proxy_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["dataset"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    ds_path = OUT_DIR / "samegene_dataset_summary.csv"
    flat_ds = []
    for report in reports:
        for row in report["dataset_summary"]:
            flat_ds.append({"candidate": report["candidate"], **row})
    with ds_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_ds[0].keys()) if flat_ds else ["candidate"])
        writer.writeheader()
        writer.writerows(flat_ds)

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "training": False,
            "inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "candidate_mmd_available": False,
        },
        "inputs": {
            "forensics": str(FORENSICS),
            "exact_cross_footprint": read_exact_cross_footprint(),
            "failure_overlap": read_failure_overlap(rows),
        },
        "candidate_reports": reports,
        "ranked_candidates": [r["candidate"] for r in ranked],
        "outputs": {
            "rows_csv": str(rows_path),
            "dataset_summary_csv": str(ds_path),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Same-Gene Transport CPU Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU/report-only over train-only/internal residual-forensics rows. This tests same-gene cross-background proxy signal and controls only; candidate MMD is unavailable, so this gate cannot directly authorize GPU.",
        "",
        "## Candidate Summary",
        "",
        "| candidate | status | n | datasets | delta mean | CI95 | dataset min | shuffle note | reasons |",
        "|---|---|---:|---:|---:|---|---:|---|---|",
    ]
    for report in ranked:
        bs = report["hierarchical_bootstrap"]
        shuf = report["shuffle_control"]
        lines.append(
            f"| `{report['candidate']}` | `{report['status']}` | {report['n']} | {report['n_datasets']} | "
            f"{report['delta_mean']:+.6f} | [{bs['ci_low']}, {bs['ci_high']}] | "
            f"{report['dataset_min']:+.6f} | `mean-invariant` | "
            f"`{';'.join(report['reasons']) or 'none'}` |"
        )
    best = ranked[0] if ranked else None
    lines.extend(["", "## Best Candidate Dataset Tail", ""])
    if best is not None:
        lines.append(f"Best proxy by mean delta: `{best['candidate']}`.")
        lines.append("")
        lines.append("| dataset | n | delta mean | delta min |")
        lines.append("|---|---:|---:|---:|")
        for row in sorted(best["dataset_summary"], key=lambda r: float(r["delta_mean"]))[:12]:
            lines.append(
                f"| `{row['dataset']}` | {row['n']} | {row['delta_mean']:+.6f} | {row['delta_min']:+.6f} |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Close same-gene residual transport as an immediate GPU route. Reopen only if a materially new implementation supplies candidate MMD/no-harm evidence and beats within-dataset shuffled-gene controls on a train-only gate.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- rows: `{rows_path}`",
            f"- dataset summary: `{ds_path}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
