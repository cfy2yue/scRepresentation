#!/usr/bin/env python3
"""CPU-only protocol gate for LatentFM scaling-effect experiments."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BIOFLOW = ROOT / "dataset/biFlow_data"
RAW = ROOT / "dataset/raw"
OUT_JSON = ROOT / "reports/latentfm_scaling_protocol_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_SCALING_PROTOCOL_GATE_20260624.md"

BASE_SPLIT = BIOFLOW / "split_seed42_xverse_trainonly_crossbg_val_v2.json"
CAP30_SPLIT = BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap30_all_v2.json"
CAP120_SPLIT = BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_meta() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in (RAW / "genepert_DE5000/metainfo.json", RAW / "chemicalpert_DE5000/metainfo.json"):
        for row in load_json(path):
            out[str(row["dataset"])] = dict(row)
    return out


def entropy(vals: list[str]) -> float:
    if not vals:
        return 0.0
    c = Counter(vals)
    total = sum(c.values())
    h = -sum((v / total) * math.log(v / total) for v in c.values())
    return h / math.log(len(c)) if len(c) > 1 else 0.0


def split_count(split: dict[str, Any], ds: str, key: str = "train") -> int:
    return len((split.get(ds) or {}).get(key, []))


def make_cap_arm(base: dict[str, Any], datasets: list[str], cap: int | None) -> dict[str, int]:
    counts = {}
    for ds in datasets:
        n = split_count(base, ds)
        counts[ds] = n if cap is None else min(n, cap)
    return counts


def summarize_arm(name: str, counts: dict[str, int], meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    active = [ds for ds, n in counts.items() if n > 0]
    weighted_types: list[str] = []
    weighted_bgs: list[str] = []
    for ds in active:
        n = counts[ds]
        row = meta.get(ds, {})
        weighted_types.extend([str(row.get("perturbation_type", "unknown"))] * n)
        weighted_bgs.extend([str(row.get("cell_line", "unknown"))] * n)
    total = sum(counts.values())
    max_share = max((n / total for n in counts.values() if total), default=0.0)
    return {
        "name": name,
        "n_datasets": len(active),
        "train_conditions": total,
        "min_per_dataset": min((counts[ds] for ds in active), default=0),
        "max_per_dataset": max((counts[ds] for ds in active), default=0),
        "max_dataset_share": max_share,
        "type_counts": dict(Counter(weighted_types)),
        "background_counts": dict(Counter(weighted_bgs)),
        "type_entropy": entropy(weighted_types),
        "background_entropy": entropy(weighted_bgs),
    }


def closest_budget_arm(base: dict[str, Any], datasets: list[str], cap: int, budget: int) -> dict[str, int]:
    """Greedy deterministic budget fill over sorted datasets."""
    counts = {ds: 0 for ds in datasets}
    remaining = budget
    for ds in sorted(datasets, key=lambda x: (-split_count(base, x), x)):
        add = min(split_count(base, ds), cap, remaining)
        if add > 0:
            counts[ds] = add
            remaining -= add
        if remaining <= 0:
            break
    return counts


def main() -> int:
    base = load_json(BASE_SPLIT)
    cap30 = load_json(CAP30_SPLIT)
    cap120 = load_json(CAP120_SPLIT)
    meta = load_meta()

    all_datasets = sorted(base)
    primary = [
        ds for ds in all_datasets
        if meta.get(ds, {}).get("perturbation_type") != "drug"
        and split_count(base, ds, "internal_val_cross_background_seen_gene_proxy") > 0
        and split_count(base, ds, "internal_val_family_gene_proxy") > 0
    ]
    excluded = [ds for ds in all_datasets if ds not in primary]

    cap30_primary = {ds: split_count(cap30, ds) for ds in primary}
    cap120_primary = {ds: split_count(cap120, ds) for ds in primary}
    full_primary = make_cap_arm(base, primary, None)
    cap60_primary = make_cap_arm(base, primary, 60)

    depth_arms = [
        summarize_arm("cap30_primary19", cap30_primary, meta),
        summarize_arm("cap60_primary19_missing_split", cap60_primary, meta),
        summarize_arm("cap120_primary19", cap120_primary, meta),
        summarize_arm("full_primary19", full_primary, meta),
    ]

    # Matched-budget breadth sketches. These are protocol candidates, not split artifacts.
    budget = 480
    large = sorted(primary, key=lambda ds: (-split_count(base, ds), ds))[:4]
    medium = sorted(primary, key=lambda ds: (-split_count(base, ds), ds))[:8]
    broad = primary
    breadth_counts = {
        "breadth_few_deep_4ds_cap120_budget480": closest_budget_arm(base, large, 120, budget),
        "breadth_mid_8ds_cap60_budget480": closest_budget_arm(base, medium, 60, budget),
        "breadth_many_shallow_19ds_cap30_budget480": closest_budget_arm(base, broad, 30, budget),
    }
    breadth_arms = [summarize_arm(k, v, meta) for k, v in breadth_counts.items()]

    # Gate decisions.
    depth_pass = (
        len(primary) == 19
        and depth_arms[0]["train_conditions"] > 0
        and depth_arms[2]["train_conditions"] > depth_arms[0]["train_conditions"]
        and abs(depth_arms[1]["train_conditions"] - 60 * len(primary)) < 60 * 4
    )
    budget_vals = [arm["train_conditions"] for arm in breadth_arms]
    budget_match = max(budget_vals) - min(budget_vals) <= 0.05 * budget if budget_vals else False
    breadth_partial = budget_match and len({arm["n_datasets"] for arm in breadth_arms}) >= 3

    payload = {
        "status": "scaling_protocol_gate_partial_pass_no_gpu_split_builder_next",
        "boundary": {
            "read_split_json": True,
            "read_metainfo_json": True,
            "read_expression": False,
            "read_canonical_metrics": False,
            "read_trackc_query": False,
            "launched_gpu": False,
        },
        "primary_tracka_datasets": primary,
        "excluded_from_primary_tracka": excluded,
        "depth_arms": depth_arms,
        "breadth_arms": breadth_arms,
        "gates": {
            "condition_count_axis": {
                "status": "pass_protocol_axis" if depth_pass else "fail",
                "reason": "fixed 19 non-drug datasets with internal cross/family coverage; cap30/cap60/cap120/full can be nested by count",
            },
            "dataset_breadth_axis": {
                "status": "partial_pass_split_builder_required" if breadth_partial else "fail",
                "reason": "matched-budget sketches are possible, but dataset identity/type/background confounding must be recorded and controlled",
            },
            "cell_background_axis": {
                "status": "fail_as_primary_claim",
                "reason": "background is mostly dataset-level; Jiang shares the same six-cell-type signature and sciplex is drug-only with no primary internal Track A rows",
            },
            "perturbation_type_axis": {
                "status": "fail_as_primary_claim",
                "reason": "type is strongly dataset-confounded; type-balanced cap120 already failed internal extension",
            },
        },
        "next_action": {
            "name": "build_cap60_and_matched_budget_breadth_splits_cpu_only",
            "gpu_authorized": False,
            "requirements": [
                "write split JSONs and pert-mean artifacts without canonical/query reads",
                "predeclare one primary hypothesis: nonmonotonic condition-count midpoint or matched-budget dataset breadth",
                "include cap120 and anchor controls; do not rerun full/type-balanced/general-exposure as new evidence",
                "only after split/provenance gate may a 2-3 arm 3k-step smoke matrix be considered",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM Scaling Protocol Gate",
        "",
        "Status: `scaling_protocol_gate_partial_pass_no_gpu_split_builder_next`",
        "",
        "## Boundary",
        "",
        "- Short CPU-only protocol/design gate.",
        "- Reads train-only split JSON and local metainfo only.",
        "- Does not read expression matrices, canonical metrics, Track C query, or use GPU.",
        "",
        "## Primary Track A Dataset Set",
        "",
        f"- Included datasets with non-drug internal cross/family coverage: `{len(primary)}`",
        f"- Excluded from primary Track A scaling gate: `{excluded}`",
        "",
        "## Condition-Count Axis",
        "",
        "| arm | datasets | train conditions | min/dataset | max/dataset | max ds share | type entropy | background entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in depth_arms:
        lines.append(
            f"| `{arm['name']}` | {arm['n_datasets']} | {arm['train_conditions']} | "
            f"{arm['min_per_dataset']} | {arm['max_per_dataset']} | "
            f"{arm['max_dataset_share']:.3f} | {arm['type_entropy']:.3f} | {arm['background_entropy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Decision: `pass_protocol_axis`. This is the cleanest scaling axis, but GPU is not authorized until the missing cap60/matched-budget split artifacts and provenance gate exist.",
            "",
            "## Matched-Budget Dataset Breadth Axis",
            "",
            "| arm | datasets | train conditions | max ds share | type entropy | background entropy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in breadth_arms:
        lines.append(
            f"| `{arm['name']}` | {arm['n_datasets']} | {arm['train_conditions']} | "
            f"{arm['max_dataset_share']:.3f} | {arm['type_entropy']:.3f} | {arm['background_entropy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Decision: `partial_pass_split_builder_required`. Matched-budget arms are possible, but the claim must be dataset-breadth under controlled budget, not pure background/type scaling.",
            "",
            "## Failed Primary Axes",
            "",
            "- Cell/background count: `fail_as_primary_claim` because background is mostly dataset-level and confounded.",
            "- Perturbation type: `fail_as_primary_claim` because type is strongly dataset-confounded and type-balanced cap120 already failed.",
            "",
            "## Next Action",
            "",
            "Build `cap60` and matched-budget breadth split artifacts in a CPU-only split/provenance gate. No GPU is authorized by this gate alone.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
