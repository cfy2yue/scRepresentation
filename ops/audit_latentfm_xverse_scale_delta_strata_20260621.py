#!/usr/bin/env python3
"""Audit condition-level xverse 8k-vs-2k uncapped scale deltas."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_2K_SPLIT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_smoke_20260620/xverse_comp006_endpoint5_2k_smoke/"
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_8K_SPLIT = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/xverse_comp006_endpoint5_8k_seed42_fulleval/"
    / "posthoc_eval_uncapped_20260621/split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_scale_delta_strata_audit_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SCALE_DELTA_STRATA_AUDIT_20260621.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        val = float(value)
        if val != val:
            return None
        return val
    except (TypeError, ValueError):
        return None


def table(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics") or []
    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get("dataset") and row.get("condition"):
            out[(str(row["dataset"]), str(row["condition"]))] = row
    return out


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def classify(pp_delta: float | None, mmd_delta: float | None) -> str:
    pp_good = pp_delta is not None and pp_delta > 0.0
    mmd_good = mmd_delta is not None and mmd_delta < 0.0
    if pp_good and mmd_good:
        return "pp_and_mmd_improved"
    if pp_good:
        return "pp_only"
    if mmd_good:
        return "mmd_only"
    return "not_improved"


def paired_rows(base: dict[str, Any], cand: dict[str, Any], group: str) -> list[dict[str, Any]]:
    btab = table(base, group)
    ctab = table(cand, group)
    rows = []
    for key in sorted(set(btab) & set(ctab)):
        b = btab[key]
        c = ctab[key]
        pp2 = fnum(b.get("pearson_pert"))
        pp8 = fnum(c.get("pearson_pert"))
        pc2 = fnum(b.get("pearson_ctrl"))
        pc8 = fnum(c.get("pearson_ctrl"))
        mmd2 = fnum(b.get("test_mmd_clamped"))
        mmd8 = fnum(c.get("test_mmd_clamped"))
        pp_delta = None if pp2 is None or pp8 is None else pp8 - pp2
        pc_delta = None if pc2 is None or pc8 is None else pc8 - pc2
        mmd_delta = None if mmd2 is None or mmd8 is None else mmd8 - mmd2
        rows.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "group": group,
                "pp_2k": pp2,
                "pp_8k": pp8,
                "pp_delta": pp_delta,
                "pc_delta": pc_delta,
                "mmd_2k": mmd2,
                "mmd_8k": mmd8,
                "mmd_delta": mmd_delta,
                "category": classify(pp_delta, mmd_delta),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    cat_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        cat_counts[str(row["category"])] += 1
    ds_summary = []
    for ds, ds_rows in sorted(by_ds.items()):
        ds_summary.append(
            {
                "dataset": ds,
                "n": len(ds_rows),
                "mean_pp_delta": mean([r["pp_delta"] for r in ds_rows if r["pp_delta"] is not None]),
                "mean_pc_delta": mean([r["pc_delta"] for r in ds_rows if r["pc_delta"] is not None]),
                "mean_mmd_delta": mean([r["mmd_delta"] for r in ds_rows if r["mmd_delta"] is not None]),
                "pp_improve_frac": sum((r["pp_delta"] or 0.0) > 0.0 for r in ds_rows) / len(ds_rows),
                "mmd_improve_frac": sum((r["mmd_delta"] or 0.0) < 0.0 for r in ds_rows) / len(ds_rows),
            }
        )
    return {
        "group": group,
        "n": len(rows),
        "n_datasets": len(by_ds),
        "mean_pp_delta": mean([r["pp_delta"] for r in rows if r["pp_delta"] is not None]),
        "mean_pc_delta": mean([r["pc_delta"] for r in rows if r["pc_delta"] is not None]),
        "mean_mmd_delta": mean([r["mmd_delta"] for r in rows if r["mmd_delta"] is not None]),
        "category_counts": dict(sorted(cat_counts.items())),
        "dataset_summary": ds_summary,
        "worst_pp_delta": sorted(rows, key=lambda r: r["pp_delta"] if r["pp_delta"] is not None else 999.0)[:20],
        "best_pp_delta": sorted(rows, key=lambda r: r["pp_delta"] if r["pp_delta"] is not None else -999.0, reverse=True)[:20],
        "mmd_only_high_residual": [
            r
            for r in sorted(
                rows,
                key=lambda r: (r["mmd_delta"] if r["mmd_delta"] is not None else 999.0),
            )
            if r["category"] == "mmd_only"
        ][:20],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Scale-Delta Strata Audit 2026-06-21",
        "",
        f"Baseline 2k split JSON: `{payload['baseline_split_json']}`",
        f"Candidate 8k split JSON: `{payload['candidate_split_json']}`",
        "",
        "Deltas are condition-level `8k - 2k`. Positive pp/pc deltas are better; negative MMD deltas are better.",
        "",
        "## Group Summary",
        "",
        "| group | n | n datasets | mean pp delta | mean pc delta | mean MMD delta | categories |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for group in payload["groups"]:
        cats = ", ".join(f"{k}:{v}" for k, v in group["category_counts"].items())
        lines.append(
            "| {group} | {n} | {nds} | {pp} | {pc} | {mmd} | {cats} |".format(
                group=group["group"],
                n=group["n"],
                nds=group["n_datasets"],
                pp=fmt(group["mean_pp_delta"]),
                pc=fmt(group["mean_pc_delta"]),
                mmd=fmt(group["mean_mmd_delta"]),
                cats=cats,
            )
        )
    for name in ["test_multi", "test_multi_unseen2"]:
        group = next(g for g in payload["groups"] if g["group"] == name)
        lines += [
            "",
            f"## {name} Dataset Summary",
            "",
            "| dataset | n | mean pp delta | mean pc delta | mean MMD delta | pp improve frac | MMD improve frac |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in group["dataset_summary"]:
            lines.append(
                "| {dataset} | {n} | {pp} | {pc} | {mmd} | {fpp:.3f} | {fmmd:.3f} |".format(
                    dataset=row["dataset"],
                    n=row["n"],
                    pp=fmt(row["mean_pp_delta"]),
                    pc=fmt(row["mean_pc_delta"]),
                    mmd=fmt(row["mean_mmd_delta"]),
                    fpp=float(row["pp_improve_frac"]),
                    fmmd=float(row["mmd_improve_frac"]),
                )
            )
    unseen2 = next(g for g in payload["groups"] if g["group"] == "test_multi_unseen2")
    lines += [
        "",
        "## test_multi_unseen2 Worst pp Deltas",
        "",
        "| dataset | condition | pp 2k | pp 8k | pp delta | MMD delta | category |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in unseen2["worst_pp_delta"][:12]:
        lines.append(
            "| {dataset} | {condition} | {pp2} | {pp8} | {ppd} | {mmdd} | {cat} |".format(
                dataset=row["dataset"],
                condition=row["condition"],
                pp2=fmt(row["pp_2k"]),
                pp8=fmt(row["pp_8k"]),
                ppd=fmt(row["pp_delta"]),
                mmdd=fmt(row["mmd_delta"]),
                cat=row["category"],
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Use paired bootstrap reports for inferential claims; this audit is descriptive condition-level triage.",
        "- If many unseen2 conditions are `mmd_only`, the next repair should target response direction/composition rather than only geometry.",
        "- If a dataset or gene-pattern stratum has consistent pp gains, it can motivate a deployable route or focused biological analysis.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-split-json", type=Path, default=DEFAULT_2K_SPLIT)
    parser.add_argument("--candidate-split-json", type=Path, default=DEFAULT_8K_SPLIT)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    base = load_json(args.baseline_split_json)
    cand = load_json(args.candidate_split_json)
    groups = []
    for group in ["test", "test_single", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2"]:
        rows = paired_rows(base, cand, group)
        groups.append(summarize(rows, group))
    out = {
        "baseline_split_json": str(args.baseline_split_json),
        "candidate_split_json": str(args.candidate_split_json),
        "groups": groups,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(out), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
