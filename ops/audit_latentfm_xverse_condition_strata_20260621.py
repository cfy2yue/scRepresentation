#!/usr/bin/env python3
"""Condition-strata audit for the xverse LatentFM stage result."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_JSON = Path(
    "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/"
    "split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_FAMILY_JSON = Path(
    "/data/cyx/1030/scLatent/CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    "xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/"
    "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_OUT_JSON = Path("/data/cyx/1030/scLatent/reports/latentfm_xverse_condition_strata_audit_20260621.json")
DEFAULT_OUT_MD = Path("/data/cyx/1030/scLatent/reports/LATENTFM_XVERSE_CONDITION_STRATA_AUDIT_20260621.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if val != val or val in {float("inf"), float("-inf")}:
            return None
        return val
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rows_for(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    rows = (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics") or []
    return [r for r in rows if isinstance(r, dict) and r.get("dataset") and r.get("condition")]


def dataset_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(row)
    out = []
    for ds, ds_rows in sorted(by_ds.items()):
        pp = [v for r in ds_rows if (v := fnum(r.get("pearson_pert"))) is not None]
        pc = [v for r in ds_rows if (v := fnum(r.get("pearson_ctrl"))) is not None]
        mmd = [v for r in ds_rows if (v := fnum(r.get("test_mmd_clamped"))) is not None]
        out.append(
            {
                "dataset": ds,
                "n_conditions": len(ds_rows),
                "mean_pearson_pert": mean(pp),
                "mean_pearson_ctrl": mean(pc),
                "mean_mmd_clamped": mean(mmd),
                "frac_pp_positive": (sum(v > 0.0 for v in pp) / len(pp)) if pp else None,
                "frac_mmd_gt_0p05": (sum(v > 0.05 for v in mmd) / len(mmd)) if mmd else None,
            }
        )
    return out


def group_summary(payload: dict[str, Any], group: str) -> dict[str, Any]:
    rows = rows_for(payload, group)
    pp = [v for r in rows if (v := fnum(r.get("pearson_pert"))) is not None]
    pc = [v for r in rows if (v := fnum(r.get("pearson_ctrl"))) is not None]
    mmd = [v for r in rows if (v := fnum(r.get("test_mmd_clamped"))) is not None]
    ds_rows = dataset_summary(rows)
    return {
        "group": group,
        "n_conditions": len(rows),
        "n_datasets": len({str(r["dataset"]) for r in rows}),
        "mean_pearson_pert_condition": mean(pp),
        "mean_pearson_ctrl_condition": mean(pc),
        "mean_mmd_clamped_condition": mean(mmd),
        "frac_pp_positive": (sum(v > 0.0 for v in pp) / len(pp)) if pp else None,
        "frac_mmd_gt_0p05": (sum(v > 0.05 for v in mmd) / len(mmd)) if mmd else None,
        "datasets_by_low_pp": sorted(ds_rows, key=lambda r: r["mean_pearson_pert"] if r["mean_pearson_pert"] is not None else 999.0),
        "datasets_by_high_pp": sorted(ds_rows, key=lambda r: r["mean_pearson_pert"] if r["mean_pearson_pert"] is not None else -999.0, reverse=True),
        "worst_conditions_by_pp": sorted(
            [
                {
                    "dataset": str(r["dataset"]),
                    "condition": str(r["condition"]),
                    "pearson_pert": fnum(r.get("pearson_pert")),
                    "pearson_ctrl": fnum(r.get("pearson_ctrl")),
                    "mmd_clamped": fnum(r.get("test_mmd_clamped")),
                }
                for r in rows
                if fnum(r.get("pearson_pert")) is not None
            ],
            key=lambda r: r["pearson_pert"],
        )[:20],
        "best_conditions_by_pp": sorted(
            [
                {
                    "dataset": str(r["dataset"]),
                    "condition": str(r["condition"]),
                    "pearson_pert": fnum(r.get("pearson_pert")),
                    "pearson_ctrl": fnum(r.get("pearson_ctrl")),
                    "mmd_clamped": fnum(r.get("test_mmd_clamped")),
                }
                for r in rows
                if fnum(r.get("pearson_pert")) is not None
            ],
            key=lambda r: r["pearson_pert"],
            reverse=True,
        )[:20],
        "highest_mmd_conditions": sorted(
            [
                {
                    "dataset": str(r["dataset"]),
                    "condition": str(r["condition"]),
                    "pearson_pert": fnum(r.get("pearson_pert")),
                    "pearson_ctrl": fnum(r.get("pearson_ctrl")),
                    "mmd_clamped": fnum(r.get("test_mmd_clamped")),
                }
                for r in rows
                if fnum(r.get("test_mmd_clamped")) is not None
            ],
            key=lambda r: r["mmd_clamped"],
            reverse=True,
        )[:20],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Condition-Strata Audit 2026-06-21",
        "",
        f"Split JSON: `{payload['split_json']}`",
        f"Family JSON: `{payload['family_json']}`",
        "",
        "Note: table means below are unweighted condition-level descriptive means.",
        "Use the CI reports for equal-dataset aggregate claims.",
        "",
        "## Group Summary",
        "",
        "| group | n conds | n datasets | mean pp | mean pc | mean MMD | pp positive frac | MMD >0.05 frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in payload["groups"]:
        lines.append(
            "| {group} | {n} | {nds} | {pp} | {pc} | {mmd} | {fpp} | {fmmd} |".format(
                group=group["group"],
                n=group["n_conditions"],
                nds=group["n_datasets"],
                pp=fmt(group["mean_pearson_pert_condition"]),
                pc=fmt(group["mean_pearson_ctrl_condition"]),
                mmd=fmt(group["mean_mmd_clamped_condition"]),
                fpp=fmt(group["frac_pp_positive"]),
                fmmd=fmt(group["frac_mmd_gt_0p05"]),
            )
        )
    lines += ["", "## Multi / Unseen2 Dataset Focus", ""]
    for group_name in ["test_multi", "test_multi_unseen2", "structure_multi"]:
        group = next(g for g in payload["groups"] if g["group"] == group_name)
        lines += [
            f"### {group_name}",
            "",
            "| dataset | n | mean pp | mean pc | mean MMD | pp positive frac | MMD >0.05 frac |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in group["datasets_by_low_pp"]:
            lines.append(
                "| {dataset} | {n} | {pp} | {pc} | {mmd} | {fpp} | {fmmd} |".format(
                    dataset=row["dataset"],
                    n=row["n_conditions"],
                    pp=fmt(row["mean_pearson_pert"]),
                    pc=fmt(row["mean_pearson_ctrl"]),
                    mmd=fmt(row["mean_mmd_clamped"]),
                    fpp=fmt(row["frac_pp_positive"]),
                    fmmd=fmt(row["frac_mmd_gt_0p05"]),
                )
            )
        lines.append("")
    lines += [
        "## Top Multi Failure Cases",
        "",
        "### test_multi_unseen2 worst pp",
        "",
        "| dataset | condition | pp | pc | MMD |",
        "|---|---|---:|---:|---:|",
    ]
    unseen2 = next(g for g in payload["groups"] if g["group"] == "test_multi_unseen2")
    for row in unseen2["worst_conditions_by_pp"][:12]:
        lines.append(
            "| {dataset} | {condition} | {pp} | {pc} | {mmd} |".format(
                dataset=row["dataset"],
                condition=row["condition"],
                pp=fmt(row["pearson_pert"]),
                pc=fmt(row["pearson_ctrl"]),
                mmd=fmt(row["mmd_clamped"]),
            )
        )
    lines += [
        "",
        "### test_multi_unseen2 highest MMD",
        "",
        "| dataset | condition | pp | pc | MMD |",
        "|---|---|---:|---:|---:|",
    ]
    for row in unseen2["highest_mmd_conditions"][:12]:
        lines.append(
            "| {dataset} | {condition} | {pp} | {pc} | {mmd} |".format(
                dataset=row["dataset"],
                condition=row["condition"],
                pp=fmt(row["pearson_pert"]),
                pc=fmt(row["pearson_ctrl"]),
                mmd=fmt(row["mmd_clamped"]),
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- xverse 8k has a positive aggregate/family signal, but pp sign balance is only modest at the condition level.",
        "- The main failure mode is multi-composition, especially unseen2 Wessels/Norman conditions with high MMD burden.",
        "- Next experiments should target xverse same-latent composition repair or no-harm response normalization, not closed scFoundation prior/replay variants.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--family-json", type=Path, default=DEFAULT_FAMILY_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    split_payload = load_json(args.split_json)
    family_payload = load_json(args.family_json)
    groups = []
    for group in ["test", "test_single", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"]:
        groups.append(group_summary(split_payload, group))
    for group in ["family_gene", "family_drug", "structure_single", "structure_multi"]:
        groups.append(group_summary(family_payload, group))

    out = {
        "split_json": str(args.split_json),
        "family_json": str(args.family_json),
        "groups": groups,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(out), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
