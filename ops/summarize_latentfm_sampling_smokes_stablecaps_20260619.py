#!/usr/bin/env python3
"""Summarize stable-caps posthoc metrics for LatentFM sampling smokes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


_GROUP_SOURCES = {
    "test": ("split", "test"),
    "test_single": ("split", "test_single"),
    "test_multi": ("split", "test_multi"),
    "test_multi_seen": ("split", "test_multi_seen"),
    "test_multi_unseen1": ("split", "test_multi_unseen1"),
    "test_multi_unseen2": ("split", "test_multi_unseen2"),
    "family_gene": ("family", "family_gene"),
    "family_drug": ("family", "family_drug"),
}

_FOCUS_DATASETS = {
    "GasperiniShendure2019_lowMOI",
    "NormanWeissman2019_filtered",
    "Wessels",
}

_MMD_GATE_SOURCE_KEYS = ("test_mmd_clamped", "test_mmd_biased", "test_mmd")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], group: str) -> dict[str, Any]:
    g = payload.get("groups", {}).get(group, {})
    mmd_gate_metric = next(
        (key for key in _MMD_GATE_SOURCE_KEYS if g.get(key) is not None),
        "test_mmd",
    )
    return {
        "n_requested": g.get("n_requested"),
        "n_conds": g.get("n_conds"),
        "mmd": g.get("test_mmd"),
        "mmd_biased": g.get("test_mmd_biased"),
        "mmd_clamped": g.get("test_mmd_clamped"),
        "mmd_gate": g.get(mmd_gate_metric),
        "mmd_gate_metric": mmd_gate_metric,
        "dp": g.get("direct_pearson"),
        "pc": g.get("pearson_ctrl"),
        "pp": g.get("pearson_pert"),
    }


def _paths(run_dir: Path) -> tuple[Path, Path]:
    posthoc = run_dir / "posthoc_eval"
    return (
        posthoc / "split_group_eval_best_ode20_mse1024_mmd1024_stablecaps.json",
        posthoc / "condition_family_eval_best_ode20_mse1024_mmd1024_stablecaps.json",
    )


def _row(run_name: str, run_dir: Path, group: str) -> dict[str, Any]:
    split_path, family_path = _paths(run_dir)
    split = _load(split_path)
    family = _load(family_path)
    source, source_group = _GROUP_SOURCES[group]
    payload = split if source == "split" else family
    metric = _metric(payload, source_group)
    return {
        "run": run_name,
        "group": group,
        "n_requested": metric["n_requested"],
        "n_conds": metric["n_conds"],
        "mmd": metric["mmd"],
        "mmd_biased": metric["mmd_biased"],
        "mmd_clamped": metric["mmd_clamped"],
        "mmd_gate": metric["mmd_gate"],
        "mmd_gate_metric": metric["mmd_gate_metric"],
        "dp": metric["dp"],
        "pc": metric["pc"],
        "pp": metric["pp"],
        "selected_condition_keys": _selected_condition_keys(payload, source_group),
    }


def _selected_dataset_counts(payload: dict[str, Any], group: str) -> dict[str, int]:
    g = payload.get("groups", {}).get(group, {})
    counts: dict[str, int] = {}
    for row in g.get("selected_conditions", []) or []:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset") or "")
        if ds:
            counts[ds] = counts.get(ds, 0) + 1
    return counts


def _selected_condition_keys(payload: dict[str, Any], group: str) -> list[str]:
    g = payload.get("groups", {}).get(group, {})
    keys = []
    for row in g.get("selected_conditions", []) or []:
        if not isinstance(row, dict):
            continue
        ds = str(row.get("dataset") or "")
        cond = str(row.get("condition") or row.get("combo_id") or row.get("perturbation") or "")
        if ds and cond:
            keys.append(f"{ds}\t{cond}")
    return sorted(keys)


def _dataset_rows(run_name: str, run_dir: Path, group: str) -> list[dict[str, Any]]:
    split_path, family_path = _paths(run_dir)
    split = _load(split_path)
    family = _load(family_path)
    source, source_group = _GROUP_SOURCES[group]
    payload = split if source == "split" else family
    g = payload.get("groups", {}).get(source_group, {})
    selected_counts = _selected_dataset_counts(payload, source_group)
    datasets = sorted(
        set(g.get("per_ds_mmd", {}) or {})
        | set(g.get("per_ds_direct", {}) or {})
        | set(g.get("per_ds_p_ctrl", {}) or {})
        | set(g.get("per_ds_p_pert", {}) or {})
        | set(selected_counts)
    )
    out = []
    for ds in datasets:
        out.append(
            {
                "run": run_name,
                "group": group,
                "dataset": ds,
                "n_selected_conditions": selected_counts.get(ds, 0),
                "mmd": (g.get("per_ds_mmd", {}) or {}).get(ds),
                "dp": (g.get("per_ds_direct", {}) or {}).get(ds),
                "pc": (g.get("per_ds_p_ctrl", {}) or {}).get(ds),
                "pp": (g.get("per_ds_p_pert", {}) or {}).get(ds),
            }
        )
    return out


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return str(x)


def _float_or_none(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _common_mmd_gate_metric(
    baseline_row: dict[str, Any],
    run_row: dict[str, Any],
) -> tuple[str, str]:
    for source_key, row_key in (
        ("test_mmd_clamped", "mmd_clamped"),
        ("test_mmd_biased", "mmd_biased"),
        ("test_mmd", "mmd"),
    ):
        if baseline_row.get(row_key) is not None and run_row.get(row_key) is not None:
            return source_key, row_key
    return "test_mmd", "mmd"


def _gate_status(
    rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    baseline_name: str,
) -> list[dict[str, Any]]:
    by_key = {(r["run"], r["group"]): r for r in rows}
    by_dataset = {
        (r["run"], r["group"], r["dataset"]): r
        for r in dataset_rows
    }
    runs = []
    for r in rows:
        run = str(r["run"])
        if run != baseline_name and run not in runs:
            runs.append(run)

    out: list[dict[str, Any]] = []
    for run in runs:
        checks: dict[str, dict[str, Any]] = {}

        base_test = by_key.get((baseline_name, "test"), {})
        run_test = by_key.get((run, "test"), {})
        base_gene = by_key.get((baseline_name, "family_gene"), {})
        run_gene = by_key.get((run, "family_gene"), {})
        base_unseen2 = by_key.get((baseline_name, "test_multi_unseen2"), {})
        run_unseen2 = by_key.get((run, "test_multi_unseen2"), {})
        base_wessels = by_dataset.get((baseline_name, "test_multi_unseen2", "Wessels"), {})
        run_wessels = by_dataset.get((run, "test_multi_unseen2", "Wessels"), {})

        selection_mismatches = []
        for group in ("test", "test_multi_unseen2", "family_gene"):
            base_keys = by_key.get((baseline_name, group), {}).get("selected_condition_keys")
            run_keys = by_key.get((run, group), {}).get("selected_condition_keys")
            if base_keys is None or run_keys is None:
                selection_mismatches.append(
                    {"group": group, "reason": "missing_selected_conditions"}
                )
                continue
            if base_keys != run_keys:
                base_set = set(base_keys)
                run_set = set(run_keys)
                selection_mismatches.append(
                    {
                        "group": group,
                        "reason": "selected_conditions_mismatch",
                        "baseline_n": len(base_keys),
                        "run_n": len(run_keys),
                        "baseline_only_n": len(base_set - run_set),
                        "run_only_n": len(run_set - base_set),
                        "baseline_only_examples": sorted(base_set - run_set)[:5],
                        "run_only_examples": sorted(run_set - base_set)[:5],
                    }
                )
        checks["selected_conditions_match_baseline"] = {
            "pass": not selection_mismatches,
            "mismatches": selection_mismatches,
            "rule": "baseline and candidate selected_conditions must match for gate groups",
        }

        test_pp = _float_or_none(run_test.get("pp"))
        base_test_pp = _float_or_none(base_test.get("pp"))
        mmd_source_key, mmd_row_key = _common_mmd_gate_metric(base_test, run_test)
        test_mmd = _float_or_none(run_test.get(mmd_row_key))
        base_test_mmd = _float_or_none(base_test.get(mmd_row_key))
        gene_pp = _float_or_none(run_gene.get("pp"))
        base_gene_pp = _float_or_none(base_gene.get("pp"))
        unseen2_pp = _float_or_none(run_unseen2.get("pp"))
        base_unseen2_pp = _float_or_none(base_unseen2.get("pp"))
        wessels_pp = _float_or_none(run_wessels.get("pp"))
        base_wessels_pp = _float_or_none(base_wessels.get("pp"))

        overall_delta = None
        if test_pp is not None and base_test_pp is not None:
            overall_delta = test_pp - base_test_pp
        checks["overall_pp_non_regression"] = {
            "pass": overall_delta is not None and overall_delta >= 0.0,
            "value": test_pp,
            "baseline": base_test_pp,
            "delta": overall_delta,
            "rule": "test pp delta >= 0",
        }

        gene_delta = None
        if gene_pp is not None and base_gene_pp is not None:
            gene_delta = gene_pp - base_gene_pp
        checks["family_gene_pp_stable"] = {
            "pass": gene_delta is not None and gene_delta >= -0.01,
            "value": gene_pp,
            "baseline": base_gene_pp,
            "delta": gene_delta,
            "rule": "family_gene pp delta >= -0.01",
        }

        mmd_ratio = None
        if test_mmd is not None and base_test_mmd is not None:
            mmd_ratio = test_mmd / max(base_test_mmd, 1e-12)
        checks["overall_mmd_ratio"] = {
            "pass": mmd_ratio is not None and mmd_ratio <= 1.15,
            "value": test_mmd,
            "baseline": base_test_mmd,
            "ratio": mmd_ratio,
            "mmd_gate_metric": mmd_source_key,
            "rule": "test MMD gate-metric ratio <= 1.15",
        }

        unseen2_delta = None
        if unseen2_pp is not None and base_unseen2_pp is not None:
            unseen2_delta = unseen2_pp - base_unseen2_pp
        unseen2_turns_positive = (
            unseen2_pp is not None
            and unseen2_pp > 0.0
            and (base_unseen2_pp is None or base_unseen2_pp <= 0.0)
        )
        checks["test_multi_unseen2_pp"] = {
            "pass": (
                unseen2_delta is not None
                and (unseen2_delta >= 0.02 or unseen2_turns_positive)
            ),
            "value": unseen2_pp,
            "baseline": base_unseen2_pp,
            "delta": unseen2_delta,
            "turns_positive": unseen2_turns_positive,
            "rule": "test_multi_unseen2 pp delta >= +0.02 or value turns positive",
        }

        wessels_delta = None
        if wessels_pp is not None and base_wessels_pp is not None:
            wessels_delta = wessels_pp - base_wessels_pp
        checks["wessels_unseen2_reported"] = {
            "pass": bool(run_wessels),
            "value": wessels_pp,
            "baseline": base_wessels_pp,
            "delta": wessels_delta,
            "n_selected_conditions": run_wessels.get("n_selected_conditions"),
            "rule": "Wessels test_multi_unseen2 row must be present and separately reported",
        }

        selection_ok = bool(checks["selected_conditions_match_baseline"]["pass"])
        triage_pass = selection_ok and all(bool(v.get("pass")) for v in checks.values())
        out.append(
            {
                "run": run,
                "triage_status": (
                    "triage_pass_uncapped_required"
                    if triage_pass
                    else (
                        "invalid_selection_mismatch"
                        if not selection_ok
                        else "triage_fail_or_diagnostic"
                    )
                ),
                "capped_results_only": True,
                "requires_uncapped_full_posthoc_for_promotion": True,
                "mmd_gate_metric_preference": list(_MMD_GATE_SOURCE_KEYS),
                "checks": checks,
            }
        )
    return out


def _write_md(
    path: Path,
    rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    baseline_name: str,
) -> None:
    by_key = {(r["run"], r["group"]): r for r in rows}
    runs = []
    for r in rows:
        if r["run"] not in runs:
            runs.append(r["run"])
    groups = list(_GROUP_SOURCES)
    lines = [
        "# LatentFM Sampling Smokes Stable-Caps Summary",
        "",
        f"Baseline: `{baseline_name}`",
        "",
        "| run | group | n | pp | delta_pp | raw MMD | MMD_gate_ratio | pc | dp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        for group in groups:
            r = by_key[(run, group)]
            b = by_key.get((baseline_name, group), {})
            pp = r.get("pp")
            mmd = r.get("mmd")
            bpp = b.get("pp")
            _, mmd_row_key = _common_mmd_gate_metric(b, r)
            mmd_gate = r.get(mmd_row_key)
            bmmd_gate = b.get(mmd_row_key)
            delta_pp = None
            mmd_ratio = None
            try:
                delta_pp = float(pp) - float(bpp)
            except (TypeError, ValueError):
                pass
            try:
                mmd_ratio = float(mmd_gate) / max(float(bmmd_gate), 1e-12)
            except (TypeError, ValueError):
                pass
            lines.append(
                f"| `{run}` | `{group}` | {r.get('n_conds')} | {_fmt(pp)} | "
                f"{_fmt(delta_pp)} | {_fmt(mmd)} | {_fmt(mmd_ratio)} | "
                f"{_fmt(r.get('pc'))} | {_fmt(r.get('dp'))} |"
            )
    lines.extend(
        [
            "",
            "## Wessels Unseen2",
            "",
            "Wessels `test_multi_unseen2` is a required separate gate because overall "
            "multi-composition improvements can be dominated by Norman or Gasperini.",
            "",
            "| run | n | pp | delta_pp | MMD | MMD_ratio | pc | dp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    w_rows = [
        r
        for r in dataset_rows
        if r["group"] == "test_multi_unseen2" and r["dataset"] == "Wessels"
    ]
    w_by_run = {r["run"]: r for r in w_rows}
    w_baseline = w_by_run.get(baseline_name, {})
    for run in runs:
        r = w_by_run.get(run)
        if not r:
            lines.append(f"| `{run}` | 0 | NA | NA | NA | NA | NA | NA |")
            continue
        pp = r.get("pp")
        mmd = r.get("mmd")
        bpp = w_baseline.get("pp")
        bmmd = w_baseline.get("mmd")
        delta_pp = None
        mmd_ratio = None
        try:
            delta_pp = float(pp) - float(bpp)
        except (TypeError, ValueError):
            pass
        try:
            mmd_ratio = float(mmd) / max(float(bmmd), 1e-12)
        except (TypeError, ValueError):
            pass
        lines.append(
            f"| `{run}` | {r.get('n_selected_conditions')} | {_fmt(pp)} | "
            f"{_fmt(delta_pp)} | {_fmt(mmd)} | {_fmt(mmd_ratio)} | "
            f"{_fmt(r.get('pc'))} | {_fmt(r.get('dp'))} |"
        )

    lines.extend(
        [
            "",
            "## Focus Dataset Diagnostics",
            "",
            "| run | group | dataset | n | pp | MMD | pc | dp |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in dataset_rows:
        if r["dataset"] not in _FOCUS_DATASETS:
            continue
        if not str(r["group"]).startswith("test_multi"):
            continue
        lines.append(
            f"| `{r['run']}` | `{r['group']}` | `{r['dataset']}` | "
            f"{r.get('n_selected_conditions')} | {_fmt(r.get('pp'))} | "
            f"{_fmt(r.get('mmd'))} | {_fmt(r.get('pc'))} | {_fmt(r.get('dp'))} |"
        )

    lines.extend(
        [
            "",
            "## Triage Gate",
            "",
            "These are capped stable-caps triage decisions. A passing row is not a "
            "final promotion; it only means the setting should proceed to uncapped "
            "full posthoc.",
            "",
            "| run | triage status | selected conditions | unseen2 pp | overall pp | family gene pp | MMD ratio | Wessels row |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for item in gate_rows:
        checks = item["checks"]
        lines.append(
            f"| `{item['run']}` | `{item['triage_status']}` | "
            f"{'pass' if checks['selected_conditions_match_baseline']['pass'] else 'fail'} | "
            f"{'pass' if checks['test_multi_unseen2_pp']['pass'] else 'fail'} | "
            f"{'pass' if checks['overall_pp_non_regression']['pass'] else 'fail'} | "
            f"{'pass' if checks['family_gene_pp_stable']['pass'] else 'fail'} | "
            f"{'pass' if checks['overall_mmd_ratio']['pass'] else 'fail'} | "
            f"{'present' if checks['wessels_unseen2_reported']['pass'] else 'missing'} |"
        )

    lines.extend(
        [
            "",
            "## Gate",
            "",
            "Promotion requires `test_multi_unseen2` pp to improve by at least +0.02 "
            "or turn positive, overall pp to avoid regression, gene-family pp to stay "
            "stable, MMD ratio <= 1.15, and Wessels unseen2 to be reported separately. "
            "The MMD gate ratio uses clamped MMD when available, then biased MMD, "
            "then raw MMD as a legacy fallback. The gate is invalid if baseline and "
            "candidate selected_conditions differ for the gate groups. "
            "Capped results are triage only; any promoted setting needs uncapped posthoc.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-name", required=True)
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--run", nargs=2, action="append", metavar=("NAME", "DIR"), required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    specs = [(args.baseline_name, args.baseline_dir)] + [
        (name, Path(path)) for name, path in args.run
    ]
    rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for name, run_dir in specs:
        for group in _GROUP_SOURCES:
            rows.append(_row(name, Path(run_dir), group))
            dataset_rows.extend(_dataset_rows(name, Path(run_dir), group))
    gate_rows = _gate_status(rows, dataset_rows, args.baseline_name)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "group",
                "n_requested",
                "n_conds",
                "mmd",
                "mmd_biased",
                "mmd_clamped",
                "mmd_gate",
                "mmd_gate_metric",
                "dp",
                "pc",
                "pp",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    dataset_csv = args.out_csv.with_name(
        f"{args.out_csv.stem}_per_dataset{args.out_csv.suffix}"
    )
    with dataset_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "group",
                "dataset",
                "n_selected_conditions",
                "mmd",
                "dp",
                "pc",
                "pp",
            ],
        )
        writer.writeheader()
        writer.writerows(dataset_rows)
    gate_json = args.out_csv.with_name(
        f"{args.out_csv.stem}_gate{'.json'}"
    )
    gate_json.write_text(json.dumps({"baseline": args.baseline_name, "runs": gate_rows}, indent=2), encoding="utf-8")
    _write_md(args.out_md, rows, dataset_rows, gate_rows, args.baseline_name)
    print(
        json.dumps(
            {
                "out_csv": str(args.out_csv),
                "out_dataset_csv": str(dataset_csv),
                "out_gate_json": str(gate_json),
                "out_md": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
