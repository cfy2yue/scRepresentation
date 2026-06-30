#!/usr/bin/env python3
"""Audit condition-level deltas for xverse response-repair capped smokes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_xverse_response_repair_smoke_20260621/xverse_response_pca32_aux025_replay1_4k"
DEFAULT_BASE_SPLIT = RUN_ROOT / "posthoc_eval_stablecaps/split_group_eval_anchor_ode20_stablecaps.json"
DEFAULT_CAND_SPLIT = RUN_ROOT / "posthoc_eval_stablecaps/split_group_eval_candidate_ode20_stablecaps.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_response_repair_aux025_delta_audit_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_RESPONSE_REPAIR_AUX025_DELTA_AUDIT_20260621.md"


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


def rows_by_key(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    rows = (payload.get("groups", {}).get(group, {}) or {}).get("condition_metrics") or []
    return {
        (str(row.get("dataset")), str(row.get("condition"))): row
        for row in rows
        if isinstance(row, dict) and row.get("dataset") and row.get("condition")
    }


def paired_rows(base: dict[str, Any], cand: dict[str, Any], group: str) -> list[dict[str, Any]]:
    btab = rows_by_key(base, group)
    ctab = rows_by_key(cand, group)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(btab) & set(ctab)):
        b = btab[key]
        c = ctab[key]
        pp_b = fnum(b.get("pearson_pert"))
        pp_c = fnum(c.get("pearson_pert"))
        pc_b = fnum(b.get("pearson_ctrl"))
        pc_c = fnum(c.get("pearson_ctrl"))
        mmd_b = fnum(b.get("test_mmd_clamped"))
        mmd_c = fnum(c.get("test_mmd_clamped"))
        pp_delta = None if pp_b is None or pp_c is None else pp_c - pp_b
        pc_delta = None if pc_b is None or pc_c is None else pc_c - pc_b
        mmd_delta = None if mmd_b is None or mmd_c is None else mmd_c - mmd_b
        rows.append(
            {
                "group": group,
                "dataset": key[0],
                "condition": key[1],
                "anchor_pp": pp_b,
                "candidate_pp": pp_c,
                "pp_delta": pp_delta,
                "pc_delta": pc_delta,
                "anchor_mmd": mmd_b,
                "candidate_mmd": mmd_c,
                "mmd_delta": mmd_delta,
                "category": classify(pp_delta, mmd_delta),
            }
        )
    return rows


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


def mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cats: dict[str, int] = defaultdict(int)
    for row in rows:
        by_ds[str(row["dataset"])].append(row)
        cats[str(row["category"])] += 1
    ds_rows = []
    for ds, vals in sorted(by_ds.items()):
        n = len(vals)
        ds_rows.append(
            {
                "dataset": ds,
                "n": n,
                "mean_pp_delta": mean([r["pp_delta"] for r in vals]),
                "mean_pc_delta": mean([r["pc_delta"] for r in vals]),
                "mean_mmd_delta": mean([r["mmd_delta"] for r in vals]),
                "pp_improve_frac": sum((r.get("pp_delta") or 0.0) > 0.0 for r in vals) / n,
                "mmd_improve_frac": sum((r.get("mmd_delta") or 0.0) < 0.0 for r in vals) / n,
                "pp_positive_frac_delta": (
                    sum((r.get("candidate_pp") or 0.0) > 0.0 for r in vals)
                    - sum((r.get("anchor_pp") or 0.0) > 0.0 for r in vals)
                )
                / n,
                "mmd_gt_005_frac_delta": (
                    sum((r.get("candidate_mmd") or 0.0) > 0.05 for r in vals)
                    - sum((r.get("anchor_mmd") or 0.0) > 0.05 for r in vals)
                )
                / n,
            }
        )
    return {
        "n": len(rows),
        "n_datasets": len(by_ds),
        "mean_pp_delta": mean([r["pp_delta"] for r in rows]),
        "mean_pc_delta": mean([r["pc_delta"] for r in rows]),
        "mean_mmd_delta": mean([r["mmd_delta"] for r in rows]),
        "pp_positive_frac_delta": (
            sum((r.get("candidate_pp") or 0.0) > 0.0 for r in rows)
            - sum((r.get("anchor_pp") or 0.0) > 0.0 for r in rows)
        )
        / len(rows)
        if rows
        else None,
        "mmd_gt_005_frac_delta": (
            sum((r.get("candidate_mmd") or 0.0) > 0.05 for r in rows)
            - sum((r.get("anchor_mmd") or 0.0) > 0.05 for r in rows)
        )
        / len(rows)
        if rows
        else None,
        "category_counts": dict(sorted(cats.items())),
        "dataset_summary": ds_rows,
        "worst_pp_delta": sorted(rows, key=lambda r: r["pp_delta"] if r["pp_delta"] is not None else 999.0)[:12],
        "best_pp_delta": sorted(rows, key=lambda r: r["pp_delta"] if r["pp_delta"] is not None else -999.0, reverse=True)[:12],
        "worst_mmd_delta": sorted(rows, key=lambda r: r["mmd_delta"] if r["mmd_delta"] is not None else -999.0, reverse=True)[:12],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.4f}"
    return str(value)


def render_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    return out


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Response Repair aux0.25 Delta Audit 2026-06-21",
        "",
        f"Baseline split JSON: `{payload['baseline_split_json']}`",
        f"Candidate split JSON: `{payload['candidate_split_json']}`",
        "",
        "This capped/stablecaps audit is descriptive. Use paired bootstrap for inferential claims.",
        "",
        "## Group Summary",
        "",
        "| group | n | n datasets | mean pp delta | mean pc delta | mean MMD delta | pp-positive frac delta | MMD>0.05 frac delta | categories |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group in payload["groups"]:
        cats = ", ".join(f"{k}:{v}" for k, v in group["summary"]["category_counts"].items())
        s = group["summary"]
        lines.append(
            "| {group} | {n} | {nds} | {pp} | {pc} | {mmd} | {ppf} | {mmdf} | {cats} |".format(
                group=group["group"],
                n=s["n"],
                nds=s["n_datasets"],
                pp=fmt(s["mean_pp_delta"]),
                pc=fmt(s["mean_pc_delta"]),
                mmd=fmt(s["mean_mmd_delta"]),
                ppf=fmt(s["pp_positive_frac_delta"]),
                mmdf=fmt(s["mmd_gt_005_frac_delta"]),
                cats=cats,
            )
        )
    for group_name in ("test_multi", "test_multi_unseen2"):
        group = next((g for g in payload["groups"] if g["group"] == group_name), None)
        if not group:
            continue
        lines.extend([
            "",
            f"## {group_name} Dataset Summary",
            "",
            "| dataset | n | mean pp delta | mean pc delta | mean MMD delta | pp improve frac | MMD improve frac | pp-positive frac delta | MMD>0.05 frac delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in group["summary"]["dataset_summary"]:
            lines.append(
                "| {dataset} | {n} | {pp} | {pc} | {mmd} | {fpp:.3f} | {fmmd:.3f} | {ppf} | {mmdf} |".format(
                    dataset=row["dataset"],
                    n=row["n"],
                    pp=fmt(row["mean_pp_delta"]),
                    pc=fmt(row["mean_pc_delta"]),
                    mmd=fmt(row["mean_mmd_delta"]),
                    fpp=float(row["pp_improve_frac"]),
                    fmmd=float(row["mmd_improve_frac"]),
                    ppf=fmt(row["pp_positive_frac_delta"]),
                    mmdf=fmt(row["mmd_gt_005_frac_delta"]),
                )
            )
    unseen = next((g for g in payload["groups"] if g["group"] == "test_multi_unseen2"), None)
    if unseen:
        for title, key in (
            ("Best Unseen2 pp Deltas", "best_pp_delta"),
            ("Worst Unseen2 pp Deltas", "worst_pp_delta"),
            ("Worst Unseen2 MMD Deltas", "worst_mmd_delta"),
        ):
            lines.extend(["", f"## {title}", ""])
            lines.extend(
                render_table(
                    unseen["summary"][key],
                    ["dataset", "condition", "anchor_pp", "candidate_pp", "pp_delta", "anchor_mmd", "candidate_mmd", "mmd_delta", "category"],
                )
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- aux0.25 response repair has a real unseen2 pp direction signal but does not pass the full fraction/no-harm gate.",
        "- The next step should diagnose deployable strata/covariates rather than increasing response weight blindly.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-split-json", type=Path, default=DEFAULT_BASE_SPLIT)
    parser.add_argument("--candidate-split-json", type=Path, default=DEFAULT_CAND_SPLIT)
    parser.add_argument("--groups", nargs="+", default=["test", "test_single", "test_multi", "test_multi_unseen2"])
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    base = load_json(args.baseline_split_json)
    cand = load_json(args.candidate_split_json)
    group_payloads = []
    for group in args.groups:
        rows = paired_rows(base, cand, group)
        group_payloads.append({"group": group, "rows": rows, "summary": summarize(rows)})
    payload = {
        "baseline_split_json": str(args.baseline_split_json),
        "candidate_split_json": str(args.candidate_split_json),
        "groups": group_payloads,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
