#!/usr/bin/env python3
"""Summarize LatentFM single-dataset upper-bound diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = (
    ("norman", "NormanWeissman2019_filtered", "scf_prior010_upperbound_norman_4k"),
    ("wessels", "Wessels", "scf_prior010_upperbound_wessels_4k"),
    ("gasperini", "GasperiniShendure2019_lowMOI", "scf_prior010_upperbound_gasperini_4k"),
)
GROUPS = ("test", "test_multi", "test_multi_seen", "test_multi_unseen1", "test_multi_unseen2")
MMD_KEYS = ("test_mmd_clamped", "test_mmd_biased", "test_mmd")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group(payload: dict[str, Any], group: str) -> dict[str, Any]:
    item = payload.get("groups", {}).get(group, {})
    return item if isinstance(item, dict) else {}


def _selected_keys(payload: dict[str, Any], group: str) -> list[str]:
    keys: list[str] = []
    for row in _group(payload, group).get("selected_conditions", []) or []:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or row.get("combo_id") or row.get("perturbation") or "")
        if ds and cond:
            keys.append(f"{ds}\t{cond}")
    return sorted(keys)


def _common_mmd_key(base: dict[str, Any], run: dict[str, Any]) -> str:
    for key in MMD_KEYS:
        if base.get(key) is not None and run.get(key) is not None:
            return key
    return "test_mmd"


def _metric_delta(base: dict[str, Any], run: dict[str, Any], metric: str) -> dict[str, Any]:
    bval = _float(base.get(metric))
    rval = _float(run.get(metric))
    return {"baseline": bval, "value": rval, "delta": None if bval is None or rval is None else rval - bval}


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def summarize(root: Path, baseline_dir: Path, out_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "baseline": "scf_prior010_inject_e2_4k",
        "baseline_dir": str(baseline_dir),
        "out_root": str(out_root),
        "datasets": [],
        "interpretation_rules": [
            "single-dataset upper-bound diagnostics are not promotion evidence",
            "selected_conditions mismatch invalidates capped comparisons",
            "unseen2 rescue means positive test_multi_unseen2 pp delta with test MMD ratio <= 1.15",
            "no upper-bound rescue supports norm/target strata or combo-aware condition modeling before more all-split scale-up",
        ],
    }
    for key, dataset, run_name in DATASETS:
        base_json = baseline_dir / f"posthoc_eval_upperbound_{key}" / "split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
        run_json = out_root / run_name / "posthoc_eval_upperbound" / "split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json"
        base = _load(base_json)
        run = _load(run_json)
        groups: dict[str, Any] = {}
        mismatches = []
        for group in GROUPS:
            bg = _group(base, group)
            rg = _group(run, group)
            bkeys = _selected_keys(base, group)
            rkeys = _selected_keys(run, group)
            if bkeys != rkeys:
                bset = set(bkeys)
                rset = set(rkeys)
                mismatches.append(
                    {
                        "group": group,
                        "baseline_n": len(bkeys),
                        "run_n": len(rkeys),
                        "baseline_only_examples": sorted(bset - rset)[:5],
                        "run_only_examples": sorted(rset - bset)[:5],
                    }
                )
            mmd_key = _common_mmd_key(bg, rg)
            bmmd = _float(bg.get(mmd_key))
            rmmd = _float(rg.get(mmd_key))
            groups[group] = {
                "n_conds": rg.get("n_conds"),
                "pearson_pert": _metric_delta(bg, rg, "pearson_pert"),
                "pearson_ctrl": _metric_delta(bg, rg, "pearson_ctrl"),
                "direct_pearson": _metric_delta(bg, rg, "direct_pearson"),
                "mmd_gate_metric": mmd_key,
                "mmd_gate": {
                    "baseline": bmmd,
                    "value": rmmd,
                    "ratio": None if bmmd is None or rmmd is None else rmmd / max(bmmd, 1e-12),
                },
            }
        unseen2 = groups.get("test_multi_unseen2", {})
        unseen2_delta = (unseen2.get("pearson_pert") or {}).get("delta")
        test_ratio = (groups.get("test", {}).get("mmd_gate") or {}).get("ratio")
        signal = (
            not mismatches
            and unseen2_delta is not None
            and unseen2_delta > 0.0
            and test_ratio is not None
            and test_ratio <= 1.15
        )
        payload["datasets"].append(
            {
                "dataset_key": key,
                "dataset": dataset,
                "run": run_name,
                "baseline_json": str(base_json),
                "run_json": str(run_json),
                "status": "invalid_selection_mismatch" if mismatches else ("upper_bound_unseen2_signal" if signal else "no_upper_bound_unseen2_rescue"),
                "selection_mismatches": mismatches,
                "groups": groups,
            }
        )
    return payload


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    fields = [
        "dataset",
        "run",
        "status",
        "group",
        "n_conds",
        "pp_baseline",
        "pp",
        "pp_delta",
        "pc_baseline",
        "pc",
        "pc_delta",
        "mmd_metric",
        "mmd_baseline",
        "mmd",
        "mmd_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ds in payload["datasets"]:
            for group, row in ds["groups"].items():
                pp = row.get("pearson_pert") or {}
                pc = row.get("pearson_ctrl") or {}
                mmd = row.get("mmd_gate") or {}
                writer.writerow(
                    {
                        "dataset": ds["dataset"],
                        "run": ds["run"],
                        "status": ds["status"],
                        "group": group,
                        "n_conds": row.get("n_conds"),
                        "pp_baseline": pp.get("baseline"),
                        "pp": pp.get("value"),
                        "pp_delta": pp.get("delta"),
                        "pc_baseline": pc.get("baseline"),
                        "pc": pc.get("value"),
                        "pc_delta": pc.get("delta"),
                        "mmd_metric": row.get("mmd_gate_metric"),
                        "mmd_baseline": mmd.get("baseline"),
                        "mmd": mmd.get("value"),
                        "mmd_ratio": mmd.get("ratio"),
                    }
                )


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM Dataset Upper-Bound Stable-Caps Summary",
        "",
        "Baseline: `scf_prior010_inject_e2_4k`, re-evaluated on each single-dataset split.",
        "",
        "These diagnostics ask whether Norman, Wessels, and Gasperini become learnable when each dataset has all training exposure by itself. They are not promotion evidence.",
        "",
        "## Dataset Summary",
        "",
        "| dataset | run | status | selected mismatch groups | unseen2 pp delta | test pp delta | test MMD ratio |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for ds in payload["datasets"]:
        groups = ds["groups"]
        lines.append(
            "| {dataset} | `{run}` | {status} | {mismatch} | {u2} | {test_pp} | {test_mmd} |".format(
                dataset=ds["dataset"],
                run=ds["run"],
                status=ds["status"],
                mismatch=len(ds["selection_mismatches"]),
                u2=_fmt((groups.get("test_multi_unseen2", {}).get("pearson_pert") or {}).get("delta")),
                test_pp=_fmt((groups.get("test", {}).get("pearson_pert") or {}).get("delta")),
                test_mmd=_fmt((groups.get("test", {}).get("mmd_gate") or {}).get("ratio")),
            )
        )
    lines.extend(
        [
            "",
            "## Group Details",
            "",
            "| dataset | group | n | pp baseline | pp | delta pp | pc baseline | pc | delta pc | MMD metric | MMD baseline | MMD | MMD ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for ds in payload["datasets"]:
        for group, row in ds["groups"].items():
            pp = row.get("pearson_pert") or {}
            pc = row.get("pearson_ctrl") or {}
            mmd = row.get("mmd_gate") or {}
            lines.append(
                "| {dataset} | `{group}` | {n} | {bpp} | {pp} | {dpp} | {bpc} | {pc} | {dpc} | `{metric}` | {bmmd} | {mmd} | {ratio} |".format(
                    dataset=ds["dataset"],
                    group=group,
                    n=row.get("n_conds"),
                    bpp=_fmt(pp.get("baseline")),
                    pp=_fmt(pp.get("value")),
                    dpp=_fmt(pp.get("delta")),
                    bpc=_fmt(pc.get("baseline")),
                    pc=_fmt(pc.get("value")),
                    dpc=_fmt(pc.get("delta")),
                    metric=row.get("mmd_gate_metric"),
                    bmmd=_fmt(mmd.get("baseline")),
                    mmd=_fmt(mmd.get("value")),
                    ratio=_fmt(mmd.get("ratio")),
                )
            )
    lines.extend(["", "## Rules", ""])
    lines.extend([f"- {rule}" for rule in payload["interpretation_rules"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/cyx/1030/scLatent"))
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    payload = summarize(args.root, args.baseline_dir, args.out_root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(args.out_md, payload)
    write_csv(args.out_csv, payload)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "out_csv": str(args.out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
