#!/usr/bin/env python3
"""Effect-stratified Track A benchmark-control table.

The goal is evaluator clarity, not model selection. It stratifies frozen
explicit Track A proxy rows by source/control-vs-GT effect proxies and reports
anchor vs control behavior within each stratum.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
JOINED = ROOT / "reports/tracka_control_baseline_synthesis_20260628/joined_rows.csv"
EXPLICIT = ROOT / "reports/tracka_explicit_group_proxy_benchmark_20260628/condition_rows.csv"
OUT_DIR = ROOT / "reports/tracka_control_effect_stratified_benchmark_20260628"
OUT_JSON = ROOT / "reports/latentfm_tracka_control_effect_stratified_benchmark_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_CONTROL_EFFECT_STRATIFIED_BENCHMARK_20260628.md"

GROUP_ORDER = (
    "all_test_single_proxy",
    "cross_background_seen_gene_proxy",
    "family_gene",
    "simple_single_unseen_global_gene_proxy",
)


def load_direct_pearson(path: Path) -> dict[tuple[str, str, str, str], float]:
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["seed"], row["explicit_group"], row["dataset"], row["condition"])] = float(row["direct_pearson"])
    return out


def load_rows(path: Path) -> list[dict[str, Any]]:
    direct_by_key = load_direct_pearson(EXPLICIT)
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row2: dict[str, Any] = dict(row)
            for key in ("anchor_pp", "ctrl_pp", "ctrl_minus_anchor_pp", "anchor_mmd", "ctrl_mmd", "ctrl_minus_anchor_mmd"):
                row2[key] = float(row2[key])
            direct = direct_by_key[(row2["seed"], row2["explicit_group"], row2["dataset"], row2["condition"])]
            row2["direct_pearson"] = float(direct)
            row2["direct_effect"] = float(1.0 - direct)
            row2["ctrl_better_pp"] = str(row2["ctrl_better_pp"]).lower() == "true"
            row2["ctrl_mmd_nonharm"] = str(row2["ctrl_mmd_nonharm"]).lower() == "true"
            rows.append(row2)
    return rows


def bootstrap(vals: list[float], *, seed: int = 20260628) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(5000, arr.size))
    means = arr[idx].mean(axis=1)
    return {"ci_low": float(np.quantile(means, 0.025)), "ci_high": float(np.quantile(means, 0.975))}


def assign_quantile_strata(rows: list[dict[str, Any]], key: str) -> dict[int, str]:
    vals = np.asarray([float(r[key]) for r in rows], dtype=float)
    q25, q33, q67, q75 = np.quantile(vals, [0.25, 1 / 3, 2 / 3, 0.75])
    labels: dict[int, str] = {}
    for i, row in enumerate(rows):
        v = float(row[key])
        if v <= q25:
            edge = "bottom25"
        elif v >= q75:
            edge = "top25"
        elif v <= q33:
            edge = "low_mid"
        elif v >= q67:
            edge = "high_mid"
        else:
            edge = "middle"
        tertile = "low" if v <= q33 else "high" if v >= q67 else "mid"
        labels[i] = f"{tertile}|{edge}"
    return labels


def summarize(part: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds_pp: dict[str, list[float]] = defaultdict(list)
    by_ds_mmd: dict[str, list[float]] = defaultdict(list)
    for row in part:
        by_ds_pp[str(row["dataset"])].append(float(row["ctrl_minus_anchor_pp"]))
        by_ds_mmd[str(row["dataset"])].append(float(row["ctrl_minus_anchor_mmd"]))
    ds_pp = [float(np.mean(v)) for v in by_ds_pp.values()]
    ds_mmd = [float(np.mean(v)) for v in by_ds_mmd.values()]
    return {
        "n": len(part),
        "n_datasets": len(by_ds_pp),
        "anchor_pp": float(np.mean([r["anchor_pp"] for r in part])) if part else 0.0,
        "ctrl_pp": float(np.mean([r["ctrl_pp"] for r in part])) if part else 0.0,
        "ctrl_minus_anchor_pp": float(np.mean(ds_pp)) if ds_pp else 0.0,
        "ctrl_minus_anchor_pp_ci": bootstrap(ds_pp),
        "anchor_mmd": float(np.mean([r["anchor_mmd"] for r in part])) if part else 0.0,
        "ctrl_mmd": float(np.mean([r["ctrl_mmd"] for r in part])) if part else 0.0,
        "ctrl_minus_anchor_mmd": float(np.mean(ds_mmd)) if ds_mmd else 0.0,
        "ctrl_minus_anchor_mmd_ci": bootstrap(ds_mmd),
        "ctrl_better_fraction": float(np.mean([r["ctrl_better_pp"] for r in part])) if part else 0.0,
        "ctrl_better_mmd_nonharm_fraction": float(np.mean([r["ctrl_better_pp"] and r["ctrl_mmd_nonharm"] for r in part])) if part else 0.0,
        "anchor_negative_fraction": float(np.mean([r["anchor_pp"] < 0.0 for r in part])) if part else 0.0,
    }


def main() -> None:
    rows = load_rows(JOINED)
    strata_specs = {
        "ctrl_mmd": "source/control-vs-GT distribution effect",
        "direct_effect": "1 - direct_pearson; higher means lower anchor/GT mean-vector similarity",
        "anchor_mmd": "anchor-vs-GT distribution effect",
    }
    labels_by_axis = {axis: assign_quantile_strata(rows, axis) for axis in strata_specs}

    summaries: dict[str, Any] = {}
    flat_rows = []
    for axis in strata_specs:
        summaries[axis] = {}
        labels = labels_by_axis[axis]
        for seed in sorted({r["seed"] for r in rows}):
            summaries[axis][seed] = {}
            for group in GROUP_ORDER:
                summaries[axis][seed][group] = {}
                base = [(i, r) for i, r in enumerate(rows) if r["seed"] == seed and r["explicit_group"] == group]
                for stratum in ("low", "mid", "high", "bottom25", "top25"):
                    if stratum in {"low", "mid", "high"}:
                        part = [r for i, r in base if labels[i].split("|")[0] == stratum]
                    else:
                        part = [r for i, r in base if labels[i].split("|")[1] == stratum]
                    if not part:
                        continue
                    s = summarize(part)
                    summaries[axis][seed][group][stratum] = s
                    flat_rows.append(
                        {
                            "axis": axis,
                            "axis_description": strata_specs[axis],
                            "seed": seed,
                            "explicit_group": group,
                            "stratum": stratum,
                            **s,
                        }
                    )

    status = "tracka_control_effect_stratified_benchmark_ready_no_gpu"
    high_effect_findings = []
    for axis in ("ctrl_mmd", "direct_effect", "anchor_mmd"):
        for seed, groups in summaries[axis].items():
            for group in ("all_test_single_proxy", "cross_background_seen_gene_proxy", "family_gene"):
                high = groups.get(group, {}).get("high", {})
                if high and high["ctrl_minus_anchor_pp"] > 0 and high["ctrl_minus_anchor_mmd"] <= 0.003:
                    high_effect_findings.append(f"{axis}_{seed}_{group}_control_still_dominates_high_effect")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "effect_stratified_group_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "axis",
            "axis_description",
            "seed",
            "explicit_group",
            "stratum",
            "n",
            "n_datasets",
            "anchor_pp",
            "ctrl_pp",
            "ctrl_minus_anchor_pp",
            "ctrl_minus_anchor_mmd",
            "ctrl_better_fraction",
            "ctrl_better_mmd_nonharm_fraction",
            "anchor_negative_fraction",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "report_only": True,
            "joined_rows": str(JOINED),
            "selection_weight": 0,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
        },
        "strata_axes": strata_specs,
        "summaries": summaries,
        "high_effect_findings": high_effect_findings,
        "outputs": {"summary_csv": str(csv_path)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Control-Effect Stratified Benchmark",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "Report-only stratification of frozen explicit Track A proxy rows. Rows are stratified by exact cap2048 `ctrl_mmd`, `direct_effect = 1 - direct_pearson`, and `anchor_mmd` effect axes. No model, training, threshold selection, canonical multi selection, or Track C query is used.",
        "",
        "## High-Effect Strata",
        "",
        "| axis | seed | group | stratum | n | anchor pp | ctrl pp | ctrl-anchor pp | ctrl-anchor MMD | ctrl better | ctrl better + MMD nonharm |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for axis in ("ctrl_mmd", "direct_effect", "anchor_mmd"):
        for seed, groups in sorted(summaries[axis].items()):
            for group in GROUP_ORDER:
                for stratum in ("high", "top25"):
                    s = groups.get(group, {}).get(stratum)
                    if not s:
                        continue
                    lines.append(
                        f"| `{axis}` | `{seed}` | `{group}` | `{stratum}` | {s['n']} | "
                        f"{s['anchor_pp']:+.6f} | {s['ctrl_pp']:+.6f} | {s['ctrl_minus_anchor_pp']:+.6f} | "
                        f"{s['ctrl_minus_anchor_mmd']:+.6f} | {s['ctrl_better_fraction']:.3f} | "
                        f"{s['ctrl_better_mmd_nonharm_fraction']:.3f} |"
                    )
    lines.extend(["", "## Decision", ""])
    if high_effect_findings:
        lines.append("Control/source remains competitive in high-effect strata, so Track A proxy reporting must include source/control as a mandatory comparator. This does not authorize GPU.")
    else:
        lines.append("Control/source dominance is mainly low-effect; high-effect strata may be usable for stricter benchmark reporting. This still does not authorize GPU.")
    lines.extend(["", "## Outputs", "", f"- JSON: `{OUT_JSON}`", f"- CSV: `{csv_path}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
