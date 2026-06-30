#!/usr/bin/env python3
"""CPU-only audit for LatentFM focus-learnability stablecaps outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GROUPS = ("test", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")
FOCUS_DATASETS = (
    "NormanWeissman2019_filtered",
    "Wessels",
    "GasperiniShendure2019_lowMOI",
)
MMD_GATE_KEYS = (
    ("test_mmd_clamped", "test_mmd_clamped"),
    ("test_mmd_biased", "test_mmd_biased"),
    ("test_mmd", "test_mmd"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group(payload: dict[str, Any], group: str) -> dict[str, Any]:
    item = payload.get("groups", {}).get(group, {})
    return item if isinstance(item, dict) else {}


def _selected_keys(payload: dict[str, Any], group: str) -> list[str]:
    keys = []
    for row in _group(payload, group).get("selected_conditions", []) or []:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or row.get("combo_id") or row.get("perturbation") or "")
        if ds and cond:
            keys.append(f"{ds}\t{cond}")
    return sorted(keys)


def _common_mmd_metric(base_group: dict[str, Any], run_group: dict[str, Any]) -> tuple[str, str]:
    for public_key, payload_key in MMD_GATE_KEYS:
        if base_group.get(payload_key) is not None and run_group.get(payload_key) is not None:
            return public_key, payload_key
    return "test_mmd", "test_mmd"


def _metric_delta(
    base_group: dict[str, Any],
    run_group: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    base = _float_or_none(base_group.get(metric))
    run = _float_or_none(run_group.get(metric))
    return {
        "baseline": base,
        "value": run,
        "delta": None if base is None or run is None else run - base,
    }


def _per_dataset_metric(group_payload: dict[str, Any], dataset: str, metric_map: str) -> float | None:
    values = group_payload.get(metric_map, {}) or {}
    if not isinstance(values, dict):
        return None
    return _float_or_none(values.get(dataset))


def audit(
    baseline_name: str,
    baseline_json: Path,
    runs: list[tuple[str, Path]],
) -> dict[str, Any]:
    baseline = _load(baseline_json)
    out: dict[str, Any] = {
        "baseline": baseline_name,
        "baseline_json": str(baseline_json),
        "runs": [],
        "interpretation_rules": [
            "focus_learnability_signal requires test_multi_unseen2 pearson_pert delta > 0 and MMD ratio <= 1.15",
            "selection mismatch invalidates focus comparison for that run",
            "focus-only results are diagnostic and not promotion evidence",
        ],
    }
    for run_name, run_json in runs:
        run = _load(run_json)
        group_rows = {}
        selection_mismatches = []
        for group in GROUPS:
            base_g = _group(baseline, group)
            run_g = _group(run, group)
            base_keys = _selected_keys(baseline, group)
            run_keys = _selected_keys(run, group)
            if base_keys != run_keys:
                base_set = set(base_keys)
                run_set = set(run_keys)
                selection_mismatches.append(
                    {
                        "group": group,
                        "baseline_n": len(base_keys),
                        "run_n": len(run_keys),
                        "baseline_only_n": len(base_set - run_set),
                        "run_only_n": len(run_set - base_set),
                        "baseline_only_examples": sorted(base_set - run_set)[:5],
                        "run_only_examples": sorted(run_set - base_set)[:5],
                    }
                )
            mmd_public_key, mmd_payload_key = _common_mmd_metric(base_g, run_g)
            base_mmd = _float_or_none(base_g.get(mmd_payload_key))
            run_mmd = _float_or_none(run_g.get(mmd_payload_key))
            group_rows[group] = {
                "n_conds": run_g.get("n_conds"),
                "pearson_pert": _metric_delta(base_g, run_g, "pearson_pert"),
                "pearson_ctrl": _metric_delta(base_g, run_g, "pearson_ctrl"),
                "direct_pearson": _metric_delta(base_g, run_g, "direct_pearson"),
                "mmd_gate_metric": mmd_public_key,
                "mmd_gate": {
                    "baseline": base_mmd,
                    "value": run_mmd,
                    "ratio": None if base_mmd is None or run_mmd is None else run_mmd / max(base_mmd, 1e-12),
                },
            }

        unseen2 = group_rows.get("test_multi_unseen2", {})
        unseen2_pp_delta = (unseen2.get("pearson_pert") or {}).get("delta")
        test_mmd_ratio = (group_rows.get("test", {}).get("mmd_gate") or {}).get("ratio")
        selection_ok = not selection_mismatches
        focus_signal = (
            selection_ok
            and unseen2_pp_delta is not None
            and unseen2_pp_delta > 0.0
            and test_mmd_ratio is not None
            and test_mmd_ratio <= 1.15
        )

        dataset_rows = []
        base_unseen2_g = _group(baseline, "test_multi_unseen2")
        run_unseen2_g = _group(run, "test_multi_unseen2")
        for dataset in FOCUS_DATASETS:
            base_pp = _per_dataset_metric(base_unseen2_g, dataset, "per_ds_p_pert")
            run_pp = _per_dataset_metric(run_unseen2_g, dataset, "per_ds_p_pert")
            base_mmd = _per_dataset_metric(base_unseen2_g, dataset, "per_ds_mmd")
            run_mmd = _per_dataset_metric(run_unseen2_g, dataset, "per_ds_mmd")
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "unseen2_pp_baseline": base_pp,
                    "unseen2_pp": run_pp,
                    "unseen2_pp_delta": None if base_pp is None or run_pp is None else run_pp - base_pp,
                    "unseen2_mmd_baseline": base_mmd,
                    "unseen2_mmd": run_mmd,
                    "unseen2_mmd_ratio": None if base_mmd is None or run_mmd is None else run_mmd / max(base_mmd, 1e-12),
                }
            )

        out["runs"].append(
            {
                "run": run_name,
                "run_json": str(run_json),
                "status": (
                    "invalid_selection_mismatch"
                    if not selection_ok
                    else ("focus_learnability_signal" if focus_signal else "no_focus_unseen2_rescue")
                ),
                "selection_mismatches": selection_mismatches,
                "groups": group_rows,
                "focus_dataset_unseen2": dataset_rows,
            }
        )
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM Focus Learnability Gate Audit",
        "",
        f"Baseline: `{payload['baseline']}`",
        f"Baseline JSON: `{payload['baseline_json']}`",
        "",
        "This audit is CPU-only and verifies cross-run selected-condition identity before interpreting the focus diagnostic.",
        "",
        "## Run Summary",
        "",
        "| run | status | selected mismatch groups | unseen2 pp delta | test MMD ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        groups = run["groups"]
        lines.append(
            "| {run} | {status} | {n_mismatch} | {pp_delta} | {mmd_ratio} |".format(
                run=f"`{run['run']}`",
                status=run["status"],
                n_mismatch=len(run["selection_mismatches"]),
                pp_delta=_fmt((groups.get("test_multi_unseen2", {}).get("pearson_pert") or {}).get("delta")),
                mmd_ratio=_fmt((groups.get("test", {}).get("mmd_gate") or {}).get("ratio")),
            )
        )

    lines.extend(
        [
            "",
            "## Focus Dataset Unseen2",
            "",
            "| run | dataset | pp baseline | pp | delta pp | MMD baseline | MMD | MMD ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in payload["runs"]:
        for row in run["focus_dataset_unseen2"]:
            lines.append(
                "| {run} | {dataset} | {bpp} | {pp} | {dpp} | {bmmd} | {mmd} | {ratio} |".format(
                    run=f"`{run['run']}`",
                    dataset=row["dataset"],
                    bpp=_fmt(row["unseen2_pp_baseline"]),
                    pp=_fmt(row["unseen2_pp"]),
                    dpp=_fmt(row["unseen2_pp_delta"]),
                    bmmd=_fmt(row["unseen2_mmd_baseline"]),
                    mmd=_fmt(row["unseen2_mmd"]),
                    ratio=_fmt(row["unseen2_mmd_ratio"]),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A focus signal supports trying a stronger all-split balance run. "
            "No focus unseen2 rescue supports single-dataset upper-bound, norm/target strata, or condition-modeling diagnostics before more all-split training.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-name", required=True)
    ap.add_argument("--baseline-json", type=Path, required=True)
    ap.add_argument("--run", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    payload = audit(
        args.baseline_name,
        args.baseline_json,
        [(name, Path(path)) for name, path in args.run],
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(args.out_md, payload)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
