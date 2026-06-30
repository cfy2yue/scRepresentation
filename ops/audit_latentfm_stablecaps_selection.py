#!/usr/bin/env python3
"""Audit split/family posthoc selected-condition consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_GROUP_ALIASES = {
    "test": "test_all",
    "test_single": "test_single",
    "test_multi": "test_multi",
    "test_multi_seen": "test_multi_seen",
    "test_multi_unseen1": "test_multi_unseen1",
    "test_multi_unseen2": "test_multi_unseen2",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected(payload: dict[str, Any], group: str) -> set[tuple[str, str]]:
    g = payload.get("groups", {}).get(group, {})
    rows = g.get("selected_conditions", [])
    out = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset", ""))
        cond = str(row.get("condition", ""))
        if ds and cond:
            out.add((ds, cond))
    return out


def _metric_summary(payload: dict[str, Any], group: str) -> dict[str, Any]:
    g = payload.get("groups", {}).get(group, {})
    return {
        "skipped": g.get("skipped"),
        "n_requested": g.get("n_requested"),
        "n_available_conditions": g.get("n_available_conditions"),
        "n_conds": g.get("n_conds"),
        "test_mmd": g.get("test_mmd"),
        "direct_pearson": g.get("direct_pearson"),
        "pearson_ctrl": g.get("pearson_ctrl"),
        "pearson_pert": g.get("pearson_pert"),
        "eval_caps": g.get("eval_caps"),
    }


def _write_md(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# LatentFM Stable-Caps Selection Audit",
        "",
        f"Split JSON: `{audit['split_json']}`",
        f"Family JSON: `{audit['family_json']}`",
        "",
        f"Overall status: **{audit['status']}**",
        "",
        "## Group Comparisons",
        "",
        "| split group | family group | split n | family n | equal | only split | only family | split pp | family pp |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in audit["comparisons"]:
        lines.append(
            "| {split_group} | {family_group} | {split_n} | {family_n} | {equal} | "
            "{only_split_n} | {only_family_n} | {split_pp:.6f} | {family_pp:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit checks whether split-group and family-group posthoc evaluators "
            "used the same selected `(dataset, condition)` pairs under capped eval. "
            "Promotion decisions should use uncapped full posthoc, but capped smoke "
            "comparisons are only acceptable when same-named groups select identical "
            "condition subsets.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--family-json", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    split = _load(args.split_json)
    family = _load(args.family_json)
    comparisons = []
    all_equal = True
    for split_group, family_group in _GROUP_ALIASES.items():
        s = _selected(split, split_group)
        f = _selected(family, family_group)
        only_s = sorted(s - f)
        only_f = sorted(f - s)
        sm = _metric_summary(split, split_group)
        fm = _metric_summary(family, family_group)
        both_empty = not s and not f
        both_skipped_or_empty = both_empty and (
            bool(sm.get("skipped"))
            or bool(fm.get("skipped"))
            or (sm.get("n_requested") in (0, None) and fm.get("n_requested") in (0, None))
        )
        equal = not only_s and not only_f and (bool(s or f) or both_skipped_or_empty)
        all_equal = all_equal and equal
        comparisons.append(
            {
                "split_group": split_group,
                "family_group": family_group,
                "split_n": len(s),
                "family_n": len(f),
                "equal": equal,
                "only_split_n": len(only_s),
                "only_family_n": len(only_f),
                "only_split_examples": only_s[:10],
                "only_family_examples": only_f[:10],
                "split_metrics": sm,
                "family_metrics": fm,
                "split_pp": float(sm.get("pearson_pert") or float("nan")),
                "family_pp": float(fm.get("pearson_pert") or float("nan")),
            }
        )

    audit = {
        "status": "pass" if all_equal else "fail",
        "split_json": str(args.split_json),
        "family_json": str(args.family_json),
        "comparisons": comparisons,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _write_md(args.out_md, audit)
    print(json.dumps({"status": audit["status"], "out_json": str(args.out_json)}, indent=2))
    return 0 if all_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
