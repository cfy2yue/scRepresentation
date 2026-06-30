#!/usr/bin/env python3
"""CPU-only paired condition strata for LatentFM focus diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


GROUPS = ("test", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")
METRICS = ("pearson_pert", "pearson_ctrl", "direct_pearson", "test_mmd_clamped")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _condition_rows(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    for row in payload.get("groups", {}).get(group, {}).get("condition_metrics", []) or []:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or "")
        if not ds or not cond:
            continue
        rows[(ds, cond)] = row
    return rows


def _quantile_edges(values: list[float], bins: int = 3) -> list[float]:
    vals = sorted(values)
    if not vals:
        return []
    edges = []
    for i in range(1, bins):
        idx = round(i * (len(vals) - 1) / bins)
        edges.append(vals[idx])
    return edges


def _bin(value: float | None, edges: list[float], labels: tuple[str, ...]) -> str:
    if value is None:
        return "missing"
    for i, edge in enumerate(edges):
        if value <= edge:
            return labels[i]
    return labels[min(len(edges), len(labels) - 1)]


def _summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, vals in sorted(grouped.items()):
        item = {k: v for k, v in zip(keys, key)}
        item["n"] = len(vals)
        for metric in METRICS:
            deltas = [v[f"delta_{metric}"] for v in vals if v.get(f"delta_{metric}") is not None]
            base_vals = [v[f"baseline_{metric}"] for v in vals if v.get(f"baseline_{metric}") is not None]
            run_vals = [v[f"run_{metric}"] for v in vals if v.get(f"run_{metric}") is not None]
            item[f"mean_delta_{metric}"] = mean(deltas) if deltas else None
            item[f"median_delta_{metric}"] = median(deltas) if deltas else None
            item[f"mean_baseline_{metric}"] = mean(base_vals) if base_vals else None
            item[f"mean_run_{metric}"] = mean(run_vals) if run_vals else None
        out.append(item)
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def build_rows(baseline: dict[str, Any], run: dict[str, Any], run_name: str) -> list[dict[str, Any]]:
    out = []
    for group in GROUPS:
        base_rows = _condition_rows(baseline, group)
        run_rows = _condition_rows(run, group)
        for key in sorted(set(base_rows) & set(run_rows)):
            brow = base_rows[key]
            rrow = run_rows[key]
            item: dict[str, Any] = {
                "run": run_name,
                "group": group,
                "dataset": key[0],
                "condition": key[1],
                "n_src_eval": _float(brow.get("n_src_eval")),
                "n_gt_eval": _float(brow.get("n_gt_eval")),
            }
            for metric in METRICS:
                bval = _float(brow.get(metric))
                rval = _float(rrow.get(metric))
                item[f"baseline_{metric}"] = bval
                item[f"run_{metric}"] = rval
                item[f"delta_{metric}"] = None if bval is None or rval is None else rval - bval
            out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM Focus Condition Strata",
        "",
        "CPU-only paired condition analysis from stablecaps focus posthoc JSONs.",
        "",
        "## Dataset x Group",
        "",
        "| run | dataset | group | n | mean delta pp | mean delta pc | mean delta MMD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["by_dataset_group"]:
        lines.append(
            f"| `{row['run']}` | {row['dataset']} | `{row['group']}` | {row['n']} | "
            f"{_fmt(row.get('mean_delta_pearson_pert'))} | "
            f"{_fmt(row.get('mean_delta_pearson_ctrl'))} | "
            f"{_fmt(row.get('mean_delta_test_mmd_clamped'))} |"
        )
    lines.extend(
        [
            "",
            "## Baseline PP Bins",
            "",
            "| run | group | baseline pp bin | n | mean baseline pp | mean run pp | mean delta pp | mean delta MMD |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["by_group_pp_bin"]:
        lines.append(
            f"| `{row['run']}` | `{row['group']}` | `{row['baseline_pp_bin']}` | {row['n']} | "
            f"{_fmt(row.get('mean_baseline_pearson_pert'))} | "
            f"{_fmt(row.get('mean_run_pearson_pert'))} | "
            f"{_fmt(row.get('mean_delta_pearson_pert'))} | "
            f"{_fmt(row.get('mean_delta_test_mmd_clamped'))} |"
        )
    lines.extend(
        [
            "",
            "## Cell-Count Bins",
            "",
            "| run | group | eval-cell bin | n | mean delta pp | mean delta pc | mean delta MMD |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["by_group_cell_bin"]:
        lines.append(
            f"| `{row['run']}` | `{row['group']}` | `{row['cell_bin']}` | {row['n']} | "
            f"{_fmt(row.get('mean_delta_pearson_pert'))} | "
            f"{_fmt(row.get('mean_delta_pearson_ctrl'))} | "
            f"{_fmt(row.get('mean_delta_test_mmd_clamped'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive pp deltas concentrated in seen/unseen1 but negative deltas in unseen2 support a composition-specific failure rather than a broad fit failure.",
            "- If bad deltas concentrate in low-cell or high-baseline-MMD bins, later preprocessing/strata diagnostics should be prioritized.",
            "- This capped analysis is diagnostic only; it is not a formal confidence interval or promotion gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-conditions-csv", type=Path, required=True)
    parser.add_argument("--out-summary-csv", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load(args.baseline_json)
    rows: list[dict[str, Any]] = []
    for run_name, run_json in args.run:
        rows.extend(build_rows(baseline, _load(Path(run_json)), run_name))
    pp_edges = _quantile_edges([r["baseline_pearson_pert"] for r in rows if r.get("baseline_pearson_pert") is not None])
    mmd_edges = _quantile_edges([r["baseline_test_mmd_clamped"] for r in rows if r.get("baseline_test_mmd_clamped") is not None])
    cell_edges = _quantile_edges([r["n_gt_eval"] for r in rows if r.get("n_gt_eval") is not None])
    for row in rows:
        row["baseline_pp_bin"] = _bin(row.get("baseline_pearson_pert"), pp_edges, ("low", "mid", "high"))
        row["baseline_mmd_bin"] = _bin(row.get("baseline_test_mmd_clamped"), mmd_edges, ("low", "mid", "high"))
        row["cell_bin"] = _bin(row.get("n_gt_eval"), cell_edges, ("low", "mid", "high"))

    by_dataset_group = _summarize(rows, ("run", "dataset", "group"))
    by_group_pp_bin = _summarize(rows, ("run", "group", "baseline_pp_bin"))
    by_group_cell_bin = _summarize(rows, ("run", "group", "cell_bin"))
    by_group_mmd_bin = _summarize(rows, ("run", "group", "baseline_mmd_bin"))
    payload = {
        "baseline_json": str(args.baseline_json),
        "n_paired_condition_rows": len(rows),
        "baseline_pp_edges": pp_edges,
        "baseline_mmd_edges": mmd_edges,
        "cell_edges": cell_edges,
        "by_dataset_group": by_dataset_group,
        "by_group_pp_bin": by_group_pp_bin,
        "by_group_cell_bin": by_group_cell_bin,
        "by_group_mmd_bin": by_group_mmd_bin,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(args.out_md, payload)
    write_csv(args.out_conditions_csv, rows)
    summary_rows = []
    for section, vals in (
        ("dataset_group", by_dataset_group),
        ("group_pp_bin", by_group_pp_bin),
        ("group_cell_bin", by_group_cell_bin),
        ("group_mmd_bin", by_group_mmd_bin),
    ):
        for row in vals:
            item = {"section": section}
            item.update(row)
            summary_rows.append(item)
    write_csv(args.out_summary_csv, summary_rows)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "out_conditions_csv": str(args.out_conditions_csv),
                "out_summary_csv": str(args.out_summary_csv),
                "n_paired_condition_rows": len(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
