#!/usr/bin/env python
"""Audit LatentFM train/test composition and epoch sampling weights.

This script is CPU-only and read-only.  It mirrors the current
``CrossDatasetFMDataset`` epoch accounting closely enough for experiment
planning: train conditions are selected with ``ceil(n_conditions ** ds_alpha)``
per dataset, and each selected condition contributes
``ceil(n_gt_cells / batch_size)`` visits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py


GROUP_KEYS = (
    "test",
    "test_single",
    "test_multi",
    "test_multi_seen",
    "test_multi_unseen1",
    "test_multi_unseen2",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"expected object JSON at {path}")
    return obj


def _condition_sizes(h5_path: Path) -> dict[str, tuple[int, int]]:
    with h5py.File(h5_path, "r") as f:
        conditions = f["conditions"].asstr()[:].tolist()
        if "ctrl/offsets" in f:
            src_offsets = f["ctrl/offsets"][:]
        else:
            src_offsets = f["ir/offsets"][:]
        gt_offsets = f["gt/offsets"][:]
        return {
            str(cond): (
                int(src_offsets[i + 1] - src_offsets[i]),
                int(gt_offsets[i + 1] - gt_offsets[i]),
            )
            for i, cond in enumerate(conditions)
        }


def _metadata_family(entry: dict[str, Any], ds_name: str) -> str:
    typ = str(entry.get("perturbation_type_raw") or entry.get("perturbation_type") or "").lower()
    if typ in {"drug", "chemical", "compound", "small molecule", "small-molecule"}:
        return "drug"
    if entry.get("chem_obs_value") or entry.get("chem_source"):
        return "drug"
    if any(tok in ds_name.lower() for tok in ("sciplex", "chempert", "chemical", "drug")):
        return "drug"
    return "gene"


def _genes_count(entry: dict[str, Any], cond: str) -> int:
    genes = entry.get("genes")
    if isinstance(genes, list):
        return len([g for g in genes if str(g).strip()])
    return max(1, len([x for x in str(cond).split("+") if x.strip()]))


def _n_eff(n: int, ds_alpha: float, min_selected: int) -> int:
    if n <= 0:
        return 0
    if ds_alpha >= 1.0:
        base = n
    else:
        base = int(math.ceil(n**ds_alpha))
    return max(1, min(n, max(base, int(min_selected))))


def _visits(n_gt: int, batch_size: int, visit_power: float, visit_cap: int) -> float:
    raw = max(1, math.ceil(int(n_gt) / int(batch_size)))
    if visit_power != 1.0:
        raw = max(1.0, float(raw) ** float(visit_power))
    if visit_cap > 0:
        raw = min(float(visit_cap), float(raw))
    return float(raw)


def build_rows(
    *,
    data_dir: Path,
    split_path: Path,
    ds_alpha: float,
    batch_size: int,
    min_cells: int,
    visit_power: float,
    visit_cap: int,
    min_selected_conditions: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _load_json(data_dir / "manifest.json")
    split = _load_json(split_path)
    metadata = _load_json(data_dir / "condition_metadata.json")

    rows: list[dict[str, Any]] = []
    total_epoch_steps = 0.0
    total_train_conditions = 0
    total_selected_conditions = 0
    total_train_gt_cells = 0

    for ds_name in sorted(manifest.get("datasets", {})):
        h5_path = data_dir / f"{ds_name}.h5"
        if not h5_path.is_file():
            continue
        sizes = _condition_sizes(h5_path)
        sp = split.get(ds_name, {})
        train_raw = [str(c) for c in sp.get("train", []) if str(c) in sizes]
        train_valid = [
            c for c in train_raw
            if sizes[c][0] >= int(min_cells) and sizes[c][1] >= int(min_cells)
        ]
        selected = _n_eff(len(train_valid), ds_alpha, min_selected_conditions)
        visit_values = [_visits(sizes[c][1], batch_size, visit_power, visit_cap) for c in train_valid]
        avg_visits = sum(visit_values) / max(1, len(visit_values))
        epoch_steps = float(selected) * float(avg_visits)
        train_gt_cells = sum(sizes[c][1] for c in train_valid)

        meta_ds = metadata.get(ds_name, {})
        train_gene = 0
        train_drug = 0
        train_single = 0
        train_multi = 0
        for cond in train_valid:
            entry = meta_ds.get(cond, {})
            if _metadata_family(entry, ds_name) == "drug":
                train_drug += 1
            else:
                train_gene += 1
            if _genes_count(entry, cond) > 1 or "+" in cond:
                train_multi += 1
            else:
                train_single += 1

        row: dict[str, Any] = {
            "dataset": ds_name,
            "train_conditions": len(train_valid),
            "train_single": train_single,
            "train_multi": train_multi,
            "train_gene": train_gene,
            "train_drug": train_drug,
            "selected_conditions_per_epoch": selected,
            "mean_visits_per_selected_condition": round(avg_visits, 4),
            "epoch_steps_est": round(epoch_steps, 4),
            "train_gt_cells": train_gt_cells,
            "mean_gt_cells_per_train_condition": round(train_gt_cells / max(1, len(train_valid)), 2),
        }
        for key in GROUP_KEYS:
            row[key] = len([c for c in sp.get(key, []) if str(c) in sizes])
        rows.append(row)
        total_epoch_steps += epoch_steps
        total_train_conditions += len(train_valid)
        total_selected_conditions += selected
        total_train_gt_cells += train_gt_cells

    for row in rows:
        row["epoch_step_share"] = round(float(row["epoch_steps_est"]) / max(total_epoch_steps, 1.0), 6)

    summary = {
        "data_dir": str(data_dir),
        "split_path": str(split_path),
        "ds_alpha": ds_alpha,
        "batch_size": batch_size,
        "min_cells": min_cells,
        "visit_power": visit_power,
        "visit_cap": visit_cap,
        "min_selected_conditions": min_selected_conditions,
        "datasets": len(rows),
        "train_conditions": total_train_conditions,
        "selected_conditions_per_epoch": total_selected_conditions,
        "epoch_steps_est": round(total_epoch_steps, 4),
        "train_gt_cells": total_train_gt_cells,
    }
    return rows, summary


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    top_by_share = sorted(rows, key=lambda r: float(r["epoch_step_share"]), reverse=True)[:8]
    multi_relevant = [
        r for r in rows
        if int(r.get("test_multi", 0)) > 0 or r["dataset"] in {"NormanWeissman2019_filtered", "Wessels"}
    ]
    lines = [
        "# LatentFM Training Composition Audit",
        "",
        f"Generated from `{summary['data_dir']}` and `{summary['split_path']}`.",
        "",
        "## Sampling Parameters",
        "",
        f"- `ds_alpha`: {summary['ds_alpha']}",
        f"- `batch_size`: {summary['batch_size']}",
        f"- `min_cells`: {summary['min_cells']}",
        f"- `visit_power`: {summary['visit_power']}",
        f"- `visit_cap`: {summary['visit_cap']}",
        f"- `min_selected_conditions`: {summary['min_selected_conditions']}",
        "",
        "## Totals",
        "",
        f"- datasets: {summary['datasets']}",
        f"- train conditions: {summary['train_conditions']}",
        f"- selected conditions per epoch: {summary['selected_conditions_per_epoch']}",
        f"- estimated microsteps per epoch: {summary['epoch_steps_est']}",
        f"- train GT cells: {summary['train_gt_cells']}",
        "",
        "## Largest Estimated Epoch Contributors",
        "",
        "| dataset | train conds | selected | mean visits | epoch steps | share | test multi | unseen2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in top_by_share:
        lines.append(
            f"| `{r['dataset']}` | {r['train_conditions']} | "
            f"{r['selected_conditions_per_epoch']} | {r['mean_visits_per_selected_condition']} | "
            f"{r['epoch_steps_est']} | {r['epoch_step_share']:.3f} | "
            f"{r['test_multi']} | {r['test_multi_unseen2']} |"
        )
    lines += [
        "",
        "## Multi-Perturbation-Relevant Datasets",
        "",
        "| dataset | train conds | train single | selected | epoch steps | share | test multi seen | unseen1 | unseen2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in multi_relevant:
        lines.append(
            f"| `{r['dataset']}` | {r['train_conditions']} | {r['train_single']} | "
            f"{r['selected_conditions_per_epoch']} | {r['epoch_steps_est']} | "
            f"{r['epoch_step_share']:.3f} | {r['test_multi_seen']} | "
            f"{r['test_multi_unseen1']} | {r['test_multi_unseen2']} |"
        )
    lines += [
        "",
        "## Interpretation Hooks",
        "",
        "- This is an estimate of training exposure, not a metric result.",
        "- Under the current strict split, train multi-condition count should remain zero for the formal zero-shot composition task.",
        "- Useful follow-up comparisons are changes in exposure distribution at fixed split, fixed checkpoint init, and fixed posthoc gates.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data/cyx/1030/dataset/latentfm_full/scfoundation")
    ap.add_argument("--split-file", default="/data/cyx/1030/dataset/biFlow_data/split_seed42.json")
    ap.add_argument("--ds-alpha", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--min-cells", type=int, default=32)
    ap.add_argument("--visit-power", type=float, default=1.0)
    ap.add_argument("--visit-cap", type=int, default=0)
    ap.add_argument("--min-selected-conditions", type=int, default=0)
    ap.add_argument("--out-prefix", default="/data/cyx/1030/scLatent/reports/latentfm_training_composition_audit_20260619")
    args = ap.parse_args()

    rows, summary = build_rows(
        data_dir=Path(args.data_dir),
        split_path=Path(args.split_file),
        ds_alpha=args.ds_alpha,
        batch_size=args.batch_size,
        min_cells=args.min_cells,
        visit_power=args.visit_power,
        visit_cap=args.visit_cap,
        min_selected_conditions=args.min_selected_conditions,
    )
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(md_path, rows, summary)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
