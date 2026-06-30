#!/usr/bin/env python3
"""Condition-level delta audit for the xverse condition-prior adapter smoke."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_prior_adapter_global_genemean_w005_add002_replay1_4k"
RUN_ROOT = ROOT / "runs/latentfm_xverse_condition_prior_adapter_smoke_20260621" / RUN_NAME
DEFAULT_MANIFEST = RUN_ROOT / "posthoc_manifest.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_condition_prior_adapter_delta_audit_20260621.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_CONDITION_PRIOR_ADAPTER_DELTA_AUDIT_20260621.md"


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
    except Exception:
        return None


def condition_table(path: Path, group: str) -> dict[tuple[str, str], dict[str, Any]]:
    payload = load_json(path)
    rows = ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            out[(str(row.get("dataset")), str(row.get("condition")))] = row
    return out


def paired_rows(base_path: Path, cand_path: Path, group: str) -> list[dict[str, Any]]:
    base = condition_table(base_path, group)
    cand = condition_table(cand_path, group)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(base) & set(cand)):
        b = base[key]
        c = cand[key]
        row = {
            "dataset": key[0],
            "condition": key[1],
            "group": group,
            "base_pp": fnum(b.get("pearson_pert")),
            "cand_pp": fnum(c.get("pearson_pert")),
            "base_pc": fnum(b.get("pearson_ctrl")),
            "cand_pc": fnum(c.get("pearson_ctrl")),
            "base_mmd": fnum(b.get("test_mmd_clamped")),
            "cand_mmd": fnum(c.get("test_mmd_clamped")),
        }
        for metric in ("pp", "pc", "mmd"):
            bval = row.get(f"base_{metric}")
            cval = row.get(f"cand_{metric}")
            row[f"delta_{metric}"] = None if bval is None or cval is None else cval - bval
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    out = []
    for name, vals in sorted(groups.items()):
        item: dict[str, Any] = {"group": name, "n": len(vals)}
        for metric in ("delta_pp", "delta_pc", "delta_mmd", "base_pp", "cand_pp", "base_mmd", "cand_mmd"):
            xs = [float(v[metric]) for v in vals if fnum(v.get(metric)) is not None]
            item[f"mean_{metric}"] = float(np.mean(xs)) if xs else None
            item[f"median_{metric}"] = float(np.median(xs)) if xs else None
        item["pp_improve_frac"] = sum((fnum(v.get("delta_pp")) or 0.0) > 0 for v in vals) / max(len(vals), 1)
        item["mmd_improve_frac"] = sum((fnum(v.get("delta_mmd")) or 0.0) < 0 for v in vals) / max(len(vals), 1)
        out.append(item)
    return out


def fmt(value: Any) -> str:
    val = fnum(value)
    if val is None:
        return "NA"
    return f"{val:+.5f}"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Condition-Prior Adapter Delta Audit 2026-06-21",
        "",
        "This audit compares capped stable posthoc condition metrics for the frozen-anchor xverse condition-prior adapter against the xverse 8k anchor.",
        "",
        "## Inputs",
        "",
        f"- baseline_split_json: `{payload['baseline_split_json']}`",
        f"- run_split_json: `{payload['run_split_json']}`",
        f"- baseline_family_json: `{payload['baseline_family_json']}`",
        f"- run_family_json: `{payload['run_family_json']}`",
        "",
        "## Dataset Summary: test_multi_unseen2",
        "",
        "| dataset | n | mean delta pp | median delta pp | pp improve frac | mean delta MMD | mmd improve frac | base pp | cand pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["unseen2_by_dataset"]:
        lines.append(
            "| {group} | {n} | {dpp} | {mdpp} | {pif} | {dmmd} | {mif} | {bpp} | {cpp} |".format(
                group=row["group"],
                n=row["n"],
                dpp=fmt(row.get("mean_delta_pp")),
                mdpp=fmt(row.get("median_delta_pp")),
                pif=fmt(row.get("pp_improve_frac")),
                dmmd=fmt(row.get("mean_delta_mmd")),
                mif=fmt(row.get("mmd_improve_frac")),
                bpp=fmt(row.get("mean_base_pp")),
                cpp=fmt(row.get("mean_cand_pp")),
            )
        )
    lines.extend([
        "",
        "## Dataset Summary: test_multi",
        "",
        "| dataset | n | mean delta pp | median delta pp | pp improve frac | mean delta MMD | mmd improve frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["multi_by_dataset"]:
        lines.append(
            "| {group} | {n} | {dpp} | {mdpp} | {pif} | {dmmd} | {mif} |".format(
                group=row["group"],
                n=row["n"],
                dpp=fmt(row.get("mean_delta_pp")),
                mdpp=fmt(row.get("median_delta_pp")),
                pif=fmt(row.get("pp_improve_frac")),
                dmmd=fmt(row.get("mean_delta_mmd")),
                mif=fmt(row.get("mmd_improve_frac")),
            )
        )
    lines.extend([
        "",
        "## Worst test_multi_unseen2 pp Deltas",
        "",
        "| dataset | condition | base pp | cand pp | delta pp | base MMD | cand MMD | delta MMD |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["worst_unseen2_pp"]:
        lines.append(
            "| {dataset} | {condition} | {bpp} | {cpp} | {dpp} | {bmmd} | {cmmd} | {dmmd} |".format(
                dataset=row["dataset"],
                condition=row["condition"],
                bpp=fmt(row.get("base_pp")),
                cpp=fmt(row.get("cand_pp")),
                dpp=fmt(row.get("delta_pp")),
                bmmd=fmt(row.get("base_mmd")),
                cmmd=fmt(row.get("cand_mmd")),
                dmmd=fmt(row.get("delta_mmd")),
            )
        )
    lines.extend([
        "",
        "## Best test_multi_unseen2 pp Deltas",
        "",
        "| dataset | condition | base pp | cand pp | delta pp | base MMD | cand MMD | delta MMD |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["best_unseen2_pp"]:
        lines.append(
            "| {dataset} | {condition} | {bpp} | {cpp} | {dpp} | {bmmd} | {cmmd} | {dmmd} |".format(
                dataset=row["dataset"],
                condition=row["condition"],
                bpp=fmt(row.get("base_pp")),
                cpp=fmt(row.get("cand_pp")),
                dpp=fmt(row.get("delta_pp")),
                bmmd=fmt(row.get("base_mmd")),
                cmmd=fmt(row.get("cand_mmd")),
                dmmd=fmt(row.get("delta_mmd")),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Condition-prior adapter deltas should be treated as capped diagnostic evidence only.",
        "- A useful follow-up would need to explain why MMD improves while response-direction pp worsens on multi/unseen2.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    row = (manifest.get("launched_runs") or [manifest])[0]
    base_split = Path(row["baseline_split_json"])
    cand_split = Path(row["run_split_json"])
    base_family = Path(row["baseline_family_json"])
    cand_family = Path(row["run_family_json"])
    unseen2 = paired_rows(base_split, cand_split, "test_multi_unseen2")
    multi = paired_rows(base_split, cand_split, "test_multi")
    payload = {
        "manifest": str(args.manifest),
        "baseline_split_json": str(base_split),
        "run_split_json": str(cand_split),
        "baseline_family_json": str(base_family),
        "run_family_json": str(cand_family),
        "unseen2_by_dataset": summarize(unseen2, "dataset"),
        "multi_by_dataset": summarize(multi, "dataset"),
        "worst_unseen2_pp": sorted(
            [r for r in unseen2 if fnum(r.get("delta_pp")) is not None],
            key=lambda r: float(r["delta_pp"]),
        )[:16],
        "best_unseen2_pp": sorted(
            [r for r in unseen2 if fnum(r.get("delta_pp")) is not None],
            key=lambda r: float(r["delta_pp"]),
            reverse=True,
        )[:16],
        "rows": {"test_multi_unseen2": unseen2, "test_multi": multi},
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
