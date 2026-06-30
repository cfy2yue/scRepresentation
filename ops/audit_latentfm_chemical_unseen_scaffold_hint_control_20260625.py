#!/usr/bin/env python3
"""Permutation control for posthoc chemical unseen-scaffold localization hints."""

from __future__ import annotations

import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
LOCALIZER = ROOT / "ops/audit_latentfm_chemical_unseen_scaffold_failure_localization_20260625.py"
LOC_JSON = ROOT / "reports/latentfm_chemical_unseen_scaffold_failure_localization_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_chemical_unseen_scaffold_hint_control_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_HINT_CONTROL_20260625.md"
N_PERM = 1000


def load_localizer():
    spec = importlib.util.spec_from_file_location("chem_scaffold_localizer", LOCALIZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {LOCALIZER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def max_stable_score(axis_rows: list[dict[str, Any]]) -> float:
    vals = [
        float(r["median_seed_mean_delta_pp"])
        for r in axis_rows
        if r.get("status") == "stable_hint" and r.get("median_seed_mean_delta_pp") is not None
    ]
    return max(vals) if vals else float("-inf")


def shuffle_axis(rows: list[dict[str, Any]], axis: str, rng: random.Random) -> list[dict[str, Any]]:
    drugs = sorted({r["drug"] for r in rows})
    labels = []
    label_by_drug = {}
    for drug in drugs:
        label = next(r[axis] for r in rows if r["drug"] == drug)
        labels.append(label)
    shuffled = labels[:]
    rng.shuffle(shuffled)
    for drug, label in zip(drugs, shuffled):
        label_by_drug[drug] = label
    out = []
    for row in rows:
        new = dict(row)
        new[axis] = label_by_drug[row["drug"]]
        out.append(new)
    return out


def main() -> int:
    mod = load_localizer()
    rows, _meta = mod.collect_rows()
    loc = load_json(LOC_JSON)
    rng = random.Random(20260625)
    axis_results = {}
    for axis, min_per_seed in [("pathway", 3), ("target", 3)]:
        actual_rows = loc["axes"][axis]
        actual_score = max_stable_score(actual_rows)
        perm_scores = []
        for _ in range(N_PERM):
            shuffled_rows = shuffle_axis(rows, axis, rng)
            perm_axis = mod.summarize_axis(shuffled_rows, axis, min_per_seed=min_per_seed)
            perm_scores.append(max_stable_score(perm_axis))
        finite = [x for x in perm_scores if x != float("-inf")]
        ge = sum(1 for x in perm_scores if x >= actual_score)
        p_value = (ge + 1) / (N_PERM + 1)
        p95 = sorted(finite)[int(0.95 * (len(finite) - 1))] if finite else None
        axis_results[axis] = {
            "actual_max_stable_median_seed_mean_delta_pp": actual_score if actual_score != float("-inf") else None,
            "permutation_p95_max_stable_score": p95,
            "permutation_p_value": p_value,
            "n_permutations": N_PERM,
            "actual_stable_hints": [r for r in actual_rows if r.get("status") == "stable_hint"],
        }
    passed_axes = [
        axis
        for axis, row in axis_results.items()
        if row["actual_max_stable_median_seed_mean_delta_pp"] is not None
        and row["permutation_p_value"] <= 0.05
        and (
            row["permutation_p95_max_stable_score"] is None
            or row["actual_max_stable_median_seed_mean_delta_pp"] > row["permutation_p95_max_stable_score"]
        )
    ]
    status = "chemical_unseen_scaffold_hint_control_pass_protocol_next_no_gpu" if passed_axes else "chemical_unseen_scaffold_hint_control_fail_close_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "task": "CPU-only posthoc hint label-shuffle control",
            "uses_completed_internal_posthoc": True,
            "uses_training": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
        },
        "axis_results": axis_results,
        "passed_axes": passed_axes,
        "next_action": (
            "external review and independent-split/fixed-step negative-control protocol before any mutation GPU"
            if passed_axes
            else "close chemical unseen-scaffold mutation from current hints"
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Chemical Unseen-Scaffold Hint Control",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only label-shuffle control for posthoc pathway/target hints.",
        "- No training, canonical multi, or Track C query.",
        "- Cannot authorize GPU by itself.",
        "",
        "| axis | actual max stable pp | perm p95 | p-value | pass |",
        "|---|---:|---:|---:|---|",
    ]
    for axis, row in axis_results.items():
        actual = row["actual_max_stable_median_seed_mean_delta_pp"]
        p95 = row["permutation_p95_max_stable_score"]
        lines.append(
            f"| `{axis}` | {'NA' if actual is None else f'{actual:+.6f}'} | "
            f"{'NA' if p95 is None else f'{p95:+.6f}'} | {row['permutation_p_value']:.4f} | `{axis in passed_axes}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- passed axes: `{passed_axes}`",
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
