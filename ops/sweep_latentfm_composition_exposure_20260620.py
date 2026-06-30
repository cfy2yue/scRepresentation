#!/usr/bin/env python3
"""CPU-only sweep of LatentFM training exposure recipes.

This is a planning audit, not a model metric.  It reuses the same approximate
epoch-accounting logic as ``analyze_latentfm_training_composition.py`` and
scores sampler recipes by whether they increase exposure for datasets with
single-dataset upper-bound signal while avoiding extra emphasis on Wessels,
which failed multiple Wessels-specific rescue diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_latentfm_training_composition import build_rows  # noqa: E402


DATA_DIR = Path("/data/cyx/1030/dataset/latentfm_full/scfoundation")
SPLIT_FILE = Path("/data/cyx/1030/dataset/biFlow_data/split_seed42.json")

RESCUABLE_DATASETS = {
    "NormanWeissman2019_filtered": "upper_bound_unseen2_signal",
    "GasperiniShendure2019_lowMOI": "upper_bound_unseen2_signal",
}
NONRESCUABLE_DATASETS = {
    "Wessels": "no_upper_bound_unseen2_rescue",
}
FOCUS = tuple(RESCUABLE_DATASETS) + tuple(NONRESCUABLE_DATASETS)


def row_by_dataset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["dataset"]): row for row in rows}


def f(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def summarize_recipe(
    *,
    data_dir: Path,
    split_file: Path,
    ds_alpha: float,
    visit_power: float,
    visit_cap: int,
    min_selected_conditions: int,
    batch_size: int,
    min_cells: int,
    label: str,
    baseline_focus: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows, summary = build_rows(
        data_dir=data_dir,
        split_path=split_file,
        ds_alpha=ds_alpha,
        batch_size=batch_size,
        min_cells=min_cells,
        visit_power=visit_power,
        visit_cap=visit_cap,
        min_selected_conditions=min_selected_conditions,
    )
    by_ds = row_by_dataset(rows)
    focus = {ds: by_ds[ds] for ds in FOCUS if ds in by_ds}
    norman_share = f(focus["NormanWeissman2019_filtered"], "epoch_step_share")
    gasperini_share = f(focus["GasperiniShendure2019_lowMOI"], "epoch_step_share")
    wessels_share = f(focus["Wessels"], "epoch_step_share")
    rescuable_share = norman_share + gasperini_share

    if baseline_focus:
        norman_ratio = norman_share / max(f(baseline_focus["NormanWeissman2019_filtered"], "epoch_step_share"), 1e-12)
        gasperini_ratio = gasperini_share / max(f(baseline_focus["GasperiniShendure2019_lowMOI"], "epoch_step_share"), 1e-12)
        wessels_ratio = wessels_share / max(f(baseline_focus["Wessels"], "epoch_step_share"), 1e-12)
    else:
        norman_ratio = gasperini_ratio = wessels_ratio = 1.0

    # Planning score: reward rescuable exposure and penalize Wessels emphasis
    # plus Replogle dominance.  It is not a statistical model-performance score.
    replogle = by_ds.get("ReplogleWeissman2022_K562_gwps", {})
    replogle_share = f(replogle, "epoch_step_share")
    score = (
        3.0 * min(norman_ratio, 4.0)
        + 2.0 * min(gasperini_ratio, 8.0)
        - 2.5 * max(wessels_ratio - 1.0, 0.0)
        - 2.0 * max(replogle_share - 0.20, 0.0)
    )

    gate = {
        "norman_share_ge_0p03": norman_share >= 0.03,
        "gasperini_share_ge_0p003": gasperini_share >= 0.003,
        "wessels_share_le_0p005": wessels_share <= 0.005,
        "replogle_share_le_0p20": replogle_share <= 0.20,
    }
    gate["all"] = all(gate.values())
    return {
        "label": label,
        "ds_alpha": ds_alpha,
        "visit_power": visit_power,
        "visit_cap": visit_cap,
        "min_selected_conditions": min_selected_conditions,
        "batch_size": batch_size,
        "min_cells": min_cells,
        "epoch_steps_est": summary["epoch_steps_est"],
        "selected_conditions_per_epoch": summary["selected_conditions_per_epoch"],
        "replogle_gwps_share": replogle_share,
        "norman_share": norman_share,
        "gasperini_share": gasperini_share,
        "wessels_share": wessels_share,
        "rescuable_share": rescuable_share,
        "norman_ratio_vs_baseline": norman_ratio,
        "gasperini_ratio_vs_baseline": gasperini_ratio,
        "wessels_ratio_vs_baseline": wessels_ratio,
        "planning_score": score,
        "gate": gate,
        "focus_rows": {ds: focus[ds] for ds in focus},
    }


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, bool):
        return "pass" if x else "fail"
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    candidates = payload["candidates"]
    top = sorted(candidates, key=lambda x: float(x["planning_score"]), reverse=True)[:12]
    gated = [x for x in candidates if x["gate"]["all"]]
    baseline = payload["baseline"]
    failed = payload["known_failed_recipe"]

    lines = [
        "# LatentFM Composition Exposure Sweep",
        "",
        "CPU-only planning audit. No training, inference, GPU access, or metric recomputation was run.",
        "",
        "## Reference Evidence",
        "",
        "- Norman and Gasperini have single-dataset upper-bound unseen2 signal.",
        "- Wessels has no upper-bound unseen2 rescue and failed CellNavi/global/context-prior diagnostics.",
        "- Therefore a useful sampler should increase Norman/Gasperini exposure without further emphasizing Wessels.",
        "",
        "## Baselines",
        "",
        "| recipe | Norman share | Gasperini share | Wessels share | Replogle GWPS share | epoch steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in (baseline, failed):
        lines.append(
            f"| `{row['label']}` | {fmt(row['norman_share'])} | {fmt(row['gasperini_share'])} | "
            f"{fmt(row['wessels_share'])} | {fmt(row['replogle_gwps_share'])} | {fmt(row['epoch_steps_est'])} |"
        )
    lines += [
        "",
        "## Top Planning Scores",
        "",
        "| recipe | score | gate | ds_alpha | power | cap | floor | Norman share | Gasperini share | Wessels share | Replogle share | epoch steps |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| `{row['label']}` | {fmt(row['planning_score'])} | {fmt(row['gate']['all'])} | "
            f"{row['ds_alpha']} | {row['visit_power']} | {row['visit_cap']} | "
            f"{row['min_selected_conditions']} | {fmt(row['norman_share'])} | "
            f"{fmt(row['gasperini_share'])} | {fmt(row['wessels_share'])} | "
            f"{fmt(row['replogle_gwps_share'])} | {fmt(row['epoch_steps_est'])} |"
        )
    lines += [
        "",
        "## Gate Result",
        "",
        f"- gated recipes: `{len(gated)}` / `{len(candidates)}`",
    ]
    if gated:
        best = sorted(gated, key=lambda x: float(x["planning_score"]), reverse=True)[0]
        lines += [
            f"- best gated recipe: `{best['label']}`",
            f"- suggested only if paired with a condition-strata rationale: `ds_alpha={best['ds_alpha']}`, "
            f"`condition_visit_power={best['visit_power']}`, `condition_visit_cap={best['visit_cap']}`, "
            f"`min_selected_conditions_per_dataset={best['min_selected_conditions']}`",
        ]
    else:
        lines.append("- no global sampler recipe satisfies the CPU gate.")
    lines += [
        "",
        "## Decision",
        "",
        "- This sweep alone does not justify GPU. It only identifies exposure-feasible recipes.",
        "- Because the already-tested visitcap/power/floor recipe failed the metric gate, any new GPU recipe must add a new non-leaky condition-strata or prior-correction mechanism, not just another sampler knob.",
        "- Wessels should be reported separately and treated as a condition/composition limitation, not optimized away by increasing its sampling share.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-cells", type=int, default=32)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    baseline = summarize_recipe(
        data_dir=args.data_dir,
        split_file=args.split_file,
        ds_alpha=0.7,
        visit_power=1.0,
        visit_cap=0,
        min_selected_conditions=0,
        batch_size=args.batch_size,
        min_cells=args.min_cells,
        label="baseline_sampler_estimate",
    )
    baseline_focus = baseline["focus_rows"]
    failed_recipe = summarize_recipe(
        data_dir=args.data_dir,
        split_file=args.split_file,
        ds_alpha=0.7,
        visit_power=0.5,
        visit_cap=8,
        min_selected_conditions=32,
        batch_size=args.batch_size,
        min_cells=args.min_cells,
        label="tested_visitcap8_power05_floor32",
        baseline_focus=baseline_focus,
    )

    candidates: list[dict[str, Any]] = []
    for ds_alpha, visit_power, visit_cap, floor in itertools.product(
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.25, 0.35, 0.5, 0.65],
        [4, 6, 8, 12, 16],
        [0, 16, 32, 64, 96, 128],
    ):
        label = f"alpha{ds_alpha:g}_power{visit_power:g}_cap{visit_cap}_floor{floor}"
        candidates.append(
            summarize_recipe(
                data_dir=args.data_dir,
                split_file=args.split_file,
                ds_alpha=ds_alpha,
                visit_power=visit_power,
                visit_cap=visit_cap,
                min_selected_conditions=floor,
                batch_size=args.batch_size,
                min_cells=args.min_cells,
                label=label,
                baseline_focus=baseline_focus,
            )
        )

    payload = {
        "data_dir": str(args.data_dir),
        "split_file": str(args.split_file),
        "reference_interpretation": {
            "NormanWeissman2019_filtered": RESCUABLE_DATASETS["NormanWeissman2019_filtered"],
            "GasperiniShendure2019_lowMOI": RESCUABLE_DATASETS["GasperiniShendure2019_lowMOI"],
            "Wessels": NONRESCUABLE_DATASETS["Wessels"],
        },
        "baseline": baseline,
        "known_failed_recipe": failed_recipe,
        "candidates": candidates,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    fieldnames = [
        "label",
        "planning_score",
        "ds_alpha",
        "visit_power",
        "visit_cap",
        "min_selected_conditions",
        "epoch_steps_est",
        "selected_conditions_per_epoch",
        "replogle_gwps_share",
        "norman_share",
        "gasperini_share",
        "wessels_share",
        "rescuable_share",
        "norman_ratio_vs_baseline",
        "gasperini_ratio_vs_baseline",
        "wessels_ratio_vs_baseline",
        "gate_all",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(candidates, key=lambda x: float(x["planning_score"]), reverse=True):
            out = {k: row.get(k) for k in fieldnames if k != "gate_all"}
            out["gate_all"] = row["gate"]["all"]
            writer.writerow(out)
    write_markdown(args.out_md, payload)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
