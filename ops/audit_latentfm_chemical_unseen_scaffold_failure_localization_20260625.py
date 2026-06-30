#!/usr/bin/env python3
"""CPU-only localization for failed chemical unseen-scaffold seed controls."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_chemical_unseen_drug_scaffold_smokes_20260625"
DRUG_META = ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625/drug_metadata.tsv"
OUT_JSON = ROOT / "reports/latentfm_chemical_unseen_scaffold_failure_localization_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_FAILURE_LOCALIZATION_20260625.md"
SEEDS = (42, 43, 44)
SCIPLEX = {"sciplex3_A549": "A549", "sciplex3_K562": "K562", "sciplex3_MCF7": "MCF7"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def drug_meta() -> dict[str, dict[str, str]]:
    with DRUG_META.open(newline="") as handle:
        return {r["drug"]: r for r in csv.DictReader(handle, delimiter="\t")}


def family_path(seed: int, kind: str) -> Path:
    rd = RUN_ROOT / f"xverse_chemical_unseen_scaffold_morgan512_2500_seed{seed}"
    return rd / "posthoc_eval_internal" / f"condition_family_eval_{kind}_internal_ode20.json"


def rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return list(((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or [])


def scaffold_bin(scaffold: str, scaffold_counts: Counter[str]) -> str:
    n = scaffold_counts[scaffold]
    if n <= 1:
        return "singleton_scaffold"
    if n <= 2:
        return "two_drug_scaffold"
    return "multi_drug_scaffold"


def collect_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = drug_meta()
    scaffold_counts = Counter(r["scaffold"] for r in meta.values())
    out: list[dict[str, Any]] = []
    for seed in SEEDS:
        anchor = load_json(family_path(seed, "anchor"))
        candidate = load_json(family_path(seed, "candidate"))
        a_by = {(r["dataset"], r["condition"]): r for r in rows(anchor, "family_drug")}
        c_by = {(r["dataset"], r["condition"]): r for r in rows(candidate, "family_drug")}
        for key in sorted(set(a_by) & set(c_by)):
            ds, cond = key
            if ds not in SCIPLEX or cond not in meta:
                continue
            ar = a_by[key]
            cr = c_by[key]
            m = meta[cond]
            out.append(
                {
                    "seed": seed,
                    "dataset": ds,
                    "background": SCIPLEX[ds],
                    "drug": cond,
                    "scaffold": m["scaffold"],
                    "scaffold_bin": scaffold_bin(m["scaffold"], scaffold_counts),
                    "pathway": m["pathways"] or "unknown",
                    "target": m["targets"] or "unknown",
                    "delta_pp": float(cr["pearson_pert"]) - float(ar["pearson_pert"]),
                    "delta_mmd": float(cr.get("test_mmd_clamped", cr.get("test_mmd", 0.0))) - float(ar.get("test_mmd_clamped", ar.get("test_mmd", 0.0))),
                    "anchor_pp": float(ar["pearson_pert"]),
                    "candidate_pp": float(cr["pearson_pert"]),
                    "n_gt_eval": int(cr.get("n_gt_eval") or 0),
                }
            )
    return out, {"drug_metadata": str(DRUG_META), "n_drugs": len(meta), "n_scaffolds": len(scaffold_counts)}


def summarize_axis(rows_: list[dict[str, Any]], axis: str, *, min_per_seed: int = 5) -> list[dict[str, Any]]:
    by_value_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows_:
        by_value_seed[(str(row[axis]), int(row["seed"]))].append(row)
    values = sorted({str(row[axis]) for row in rows_})
    out = []
    for value in values:
        seed_rows = []
        for seed in SEEDS:
            vals = by_value_seed.get((value, seed), [])
            if not vals:
                continue
            pp = [v["delta_pp"] for v in vals]
            mmd = [v["delta_mmd"] for v in vals]
            seed_rows.append(
                {
                    "seed": seed,
                    "n": len(vals),
                    "mean_delta_pp": sum(pp) / len(pp),
                    "median_delta_pp": statistics.median(pp),
                    "mean_delta_mmd": sum(mmd) / len(mmd),
                    "min_delta_pp": min(pp),
                }
            )
        eligible = [r for r in seed_rows if r["n"] >= min_per_seed]
        pass_like = [
            r
            for r in eligible
            if r["mean_delta_pp"] >= 0.005 and r["mean_delta_mmd"] <= 0.00025 and r["min_delta_pp"] > -0.05
        ]
        med = statistics.median([r["mean_delta_pp"] for r in eligible]) if eligible else None
        out.append(
            {
                "axis": axis,
                "value": value,
                "eligible_seeds": len(eligible),
                "pass_like_seeds": len(pass_like),
                "median_seed_mean_delta_pp": med,
                "seed_rows": seed_rows,
                "status": "stable_hint" if len(pass_like) >= 2 and med is not None and med >= 0.008 else "no_stable_hint",
            }
        )
    return sorted(out, key=lambda r: (r["status"] != "stable_hint", -(r["median_seed_mean_delta_pp"] or -999), r["value"]))


def main() -> int:
    row_data, meta_summary = collect_rows()
    axes = {
        "background": summarize_axis(row_data, "background", min_per_seed=10),
        "pathway": summarize_axis(row_data, "pathway", min_per_seed=3),
        "target": summarize_axis(row_data, "target", min_per_seed=3),
        "scaffold_bin": summarize_axis(row_data, "scaffold_bin", min_per_seed=5),
    }
    stable_hints = [r for rows_ in axes.values() for r in rows_ if r["status"] == "stable_hint"]
    status = "chemical_unseen_scaffold_failure_localization_stable_hint_cpu_next_no_gpu" if stable_hints else "chemical_unseen_scaffold_failure_localization_fail_close_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "task": "CPU-only failure localization",
            "uses_training": False,
            "uses_model_outputs": True,
            "model_outputs_scope": "completed train-only/internal chemical unseen-scaffold posthoc only",
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
        },
        "metadata": meta_summary,
        "n_condition_seed_rows": len(row_data),
        "axes": axes,
        "stable_hints": stable_hints[:20],
        "next_action": (
            "run CPU negative-control/protocol gate for stable hints before any mutation GPU"
            if stable_hints
            else "close same-split chemical unseen-scaffold as seed-unstable; do not launch more chemical GPU without a new non-posthoc hypothesis"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Chemical Unseen-Scaffold Failure Localization",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only localization over completed train-only/internal posthoc outputs.",
        "- No training, canonical multi, or Track C query.",
        "- This report cannot authorize GPU by itself.",
        "",
        "## Summary",
        "",
        f"- condition-seed rows: `{len(row_data)}`",
        f"- stable hints: `{len(stable_hints)}`",
        "",
    ]
    for axis, axis_rows in axes.items():
        lines += [
            f"## Axis: {axis}",
            "",
            "| value | status | eligible seeds | pass-like seeds | median seed mean pp |",
            "|---|---|---:|---:|---:|",
        ]
        for row in axis_rows[:15]:
            med = row["median_seed_mean_delta_pp"]
            lines.append(
                f"| `{row['value']}` | `{row['status']}` | {row['eligible_seeds']} | {row['pass_like_seeds']} | "
                f"{'NA' if med is None else f'{med:+.6f}'} |"
            )
        lines.append("")
    lines += [
        "## Decision",
        "",
        "- GPU authorized: `False`",
        f"- next action: {payload['next_action']}",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
