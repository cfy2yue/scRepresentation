#!/usr/bin/env python3
"""Audit multi-perturbation visibility for LatentFM split files.

The canonical split keeps multi-gene perturbations in test for zero-shot
composition evaluation. This tool makes that contract explicit for a converted
LatentFM bundle and reports whether training contains exact multi-condition
supervision, whether test combinations have all/some/no single components in
train, and which datasets dominate each bucket.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from model.utils.data.split import classify_multi_perturbation_tests, is_multi_pert, pert_components


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _filter_manifest_conditions(split_conds: Iterable[str], manifest_conds: set[str]) -> List[str]:
    return [str(c) for c in split_conds if str(c) in manifest_conds]


def audit_composition_split(manifest: Mapping[str, Any], split: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a serializable composition audit for one manifest/split pair."""
    datasets: Dict[str, Any] = {}
    totals = {
        "datasets": 0,
        "train_single": 0,
        "train_multi": 0,
        "test_single": 0,
        "test_multi": 0,
        "multi_seen": 0,
        "multi_unseen1": 0,
        "multi_unseen2": 0,
        "test_multi_with_exact_train_leak": 0,
    }

    for ds_name, ds_meta in sorted(dict(manifest.get("datasets", {})).items()):
        allowed = set(map(str, ds_meta.get("conditions", [])))
        sp = dict(split.get(ds_name, {}))
        train = _filter_manifest_conditions(sp.get("train", []), allowed)
        test = _filter_manifest_conditions(sp.get("test", []), allowed)
        train_single = [c for c in train if not is_multi_pert(c)]
        train_multi = [c for c in train if is_multi_pert(c)]
        test_single = [c for c in test if not is_multi_pert(c)]
        test_multi = [c for c in test if is_multi_pert(c)]
        groups = classify_multi_perturbation_tests(test_multi, train_single)
        exact_leak = sorted(set(test_multi) & set(train_multi))
        component_vocab = set(train_single)
        unseen_component_counts = [
            sum(1 for comp in pert_components(c) if comp not in component_vocab)
            for c in test_multi
        ]

        ds_row = {
            "manifest_conditions": len(allowed),
            "train_single": len(train_single),
            "train_multi": len(train_multi),
            "test_single": len(test_single),
            "test_multi": len(test_multi),
            "multi_seen": len(groups["seen"]),
            "multi_unseen1": len(groups["unseen1"]),
            "multi_unseen2": len(groups["unseen2"]),
            "test_multi_with_exact_train_leak": len(exact_leak),
            "max_unseen_components": max(unseen_component_counts) if unseen_component_counts else 0,
            "examples": {
                "seen": groups["seen"][:5],
                "unseen1": groups["unseen1"][:5],
                "unseen2": groups["unseen2"][:5],
                "exact_leak": exact_leak[:5],
            },
        }
        datasets[ds_name] = ds_row

        totals["datasets"] += 1
        for key in (
            "train_single",
            "train_multi",
            "test_single",
            "test_multi",
            "multi_seen",
            "multi_unseen1",
            "multi_unseen2",
            "test_multi_with_exact_train_leak",
        ):
            totals[key] += int(ds_row[key])

    totals["has_exact_multi_train_supervision"] = totals["train_multi"] > 0
    totals["has_exact_test_multi_leak"] = totals["test_multi_with_exact_train_leak"] > 0
    totals["test_multi_seen_fraction"] = (
        totals["multi_seen"] / totals["test_multi"] if totals["test_multi"] else 0.0
    )
    return {"totals": totals, "datasets": datasets}


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = dict(report.get("totals", {}))
    lines = [
        "# LatentFM Composition Split Audit",
        "",
        "## Totals",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for key in (
        "datasets",
        "train_single",
        "train_multi",
        "test_single",
        "test_multi",
        "multi_seen",
        "multi_unseen1",
        "multi_unseen2",
        "test_multi_with_exact_train_leak",
        "test_multi_seen_fraction",
    ):
        value = totals.get(key, 0)
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if totals.get("train_multi", 0) == 0:
        lines.append(
            "Training contains no exact multi-perturbation conditions; multi-condition evaluation is exact-combination zero-shot."
        )
    else:
        lines.append(
            "Training contains exact multi-perturbation conditions; inspect `test_multi_with_exact_train_leak` before calling this zero-shot."
        )
    if totals.get("test_multi_with_exact_train_leak", 0) == 0:
        lines.append("No exact test multi-condition is also present in train.")
    else:
        lines.append("WARNING: at least one exact test multi-condition is also present in train.")

    lines += [
        "",
        "## Per-Dataset",
        "",
        "| Dataset | train single | train multi | test single | test multi | seen | unseen1 | unseen2 | leak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, row in sorted(dict(report.get("datasets", {})).items()):
        lines.append(
            "| {ds} | {train_single} | {train_multi} | {test_single} | {test_multi} | "
            "{multi_seen} | {multi_unseen1} | {multi_unseen2} | {leak} |".format(
                ds=ds,
                train_single=row["train_single"],
                train_multi=row["train_multi"],
                test_single=row["test_single"],
                test_multi=row["test_multi"],
                multi_seen=row["multi_seen"],
                multi_unseen1=row["multi_unseen1"],
                multi_unseen2=row["multi_unseen2"],
                leak=row["test_multi_with_exact_train_leak"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="LatentFM bundle directory containing manifest.json")
    ap.add_argument("--split-file", required=True, help="Canonical split JSON")
    ap.add_argument("--out-json", default="", help="Optional JSON report path")
    ap.add_argument("--out-md", default="", help="Optional Markdown report path")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest_path = data_dir / "manifest.json"
    split_path = Path(args.split_file).expanduser().resolve()
    manifest = _load_json(manifest_path)
    split = _load_json(split_path)
    report = audit_composition_split(manifest, split)

    if args.out_json:
        out = Path(args.out_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out_md:
        out = Path(args.out_md).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report), encoding="utf-8")
    if not args.out_json and not args.out_md:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
