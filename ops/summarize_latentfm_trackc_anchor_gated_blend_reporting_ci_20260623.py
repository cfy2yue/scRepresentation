#!/usr/bin/env python3
"""Reporting-grade CI and failure-case summary for the frozen Track C blend.

This script is read-only over already-frozen artifacts.  It must not be used to
select alpha, checkpoints, gates, thresholds, or future branches.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_SUPPORT_JSON = (
    ROOT
    / "runs/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623"
    / "xverse_support_film_retry1_anchor_gated_blend_posthoc_ode20_retry1"
    / "posthoc_eval/support_trainselect_support_val_multi_blend_ode20.json"
)
DEFAULT_CANONICAL_SINGLE_JSON = (
    ROOT
    / "runs/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623"
    / "xverse_support_film_retry1_anchor_gated_blend_posthoc_ode20_retry1"
    / "posthoc_eval/canonical_test_single_blend_ode20.json"
)
DEFAULT_CANONICAL_FAMILY_JSON = (
    ROOT
    / "runs/latentfm_trackc_anchor_gated_support_teacher_blend_posthoc_20260623"
    / "xverse_support_film_retry1_anchor_gated_blend_posthoc_ode20_retry1"
    / "posthoc_eval/canonical_family_gene_blend_ode20.json"
)
DEFAULT_QUERY_JSON = (
    ROOT
    / "runs/latentfm_trackc_anchor_gated_blend_query_once_20260623_retry1"
    / "eval/anchor_gated_blend_query_once_ode20.json"
)
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_anchor_gated_blend_reporting_ci_20260623.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ANCHOR_GATED_BLEND_REPORTING_CI_20260623.md"

GROUP_ORDER = (
    ("support_val_multi", "support_trainselect", "support_val_multi"),
    ("canonical_test_single", "canonical_noharm", "test_single"),
    ("canonical_family_gene", "canonical_noharm", "family_gene"),
    ("query_all", "heldout_query_once", "heldout_query_multi_final_only"),
    ("query_seen", "heldout_query_once", "heldout_query_multi_seen_final_only"),
    ("query_unseen1", "heldout_query_once", "heldout_query_multi_unseen1_final_only"),
    ("query_unseen2", "heldout_query_once", "heldout_query_multi_unseen2_final_only"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def fmt(value: Any) -> str:
    value = finite(value)
    return "NA" if value is None else f"{value:+.6f}"


def rows_for(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def equal_dataset_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row.get(key))
        if value is not None:
            by_ds[str(row.get("dataset"))].append(value)
    vals = [mean(v) for v in by_ds.values() if v]
    return float(mean(vals)) if vals else None


def bootstrap_equal_dataset(
    rows: list[dict[str, Any]],
    key: str,
    *,
    n_boot: int,
    seed: int,
    pp_harm_threshold: float = -0.02,
    pp_improve_threshold: float = 0.02,
    mmd_harm_threshold: float = 0.005,
    mmd_improve_threshold: float = -0.005,
) -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row.get(key))
        if value is not None:
            by_ds[str(row.get("dataset"))].append(value)
    observed = equal_dataset_mean(rows, key)
    if observed is None or not by_ds:
        return {"observed": None, "ci_low": None, "ci_high": None, "n_datasets": 0}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(n_boot)):
        ds_vals = []
        for vals in by_ds.values():
            arr = np.asarray(vals, dtype=np.float64)
            ds_vals.append(float(np.mean(rng.choice(arr, size=arr.size, replace=True))))
        samples.append(float(np.mean(ds_vals)))
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "observed": float(observed),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_positive": float(np.mean(arr > 0.0)),
        "p_harm_pp": float(np.mean(arr < pp_harm_threshold)),
        "p_improve_pp": float(np.mean(arr > pp_improve_threshold)),
        "p_harm_mmd": float(np.mean(arr > mmd_harm_threshold)),
        "p_improve_mmd": float(np.mean(arr < mmd_improve_threshold)),
        "n_datasets": int(len(by_ds)),
        "n_rows": int(sum(len(v) for v in by_ds.values())),
    }


def dataset_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[str(row.get("dataset"))].append(row)
    out = []
    for ds, items in sorted(by_ds.items()):
        pp_vals = [finite(r.get("blend_delta_vs_anchor_pearson_pert")) for r in items]
        mmd_vals = [finite(r.get("blend_delta_vs_anchor_test_mmd_clamped")) for r in items]
        pp = [v for v in pp_vals if v is not None]
        mmd = [v for v in mmd_vals if v is not None]
        out.append(
            {
                "dataset": ds,
                "n_rows": len(items),
                "pp_delta_mean": float(mean(pp)) if pp else None,
                "pp_negative_fraction": float(sum(v < 0.0 for v in pp) / len(pp)) if pp else None,
                "mmd_delta_mean": float(mean(mmd)) if mmd else None,
                "mmd_harm_fraction": float(sum(v > 0.005 for v in mmd) / len(mmd)) if mmd else None,
            }
        )
    return out


def row_brief(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "dataset",
        "condition",
        "anchor_pearson_pert",
        "blend_pearson_pert",
        "blend_delta_vs_anchor_pearson_pert",
        "anchor_test_mmd_clamped",
        "blend_test_mmd_clamped",
        "blend_delta_vs_anchor_test_mmd_clamped",
    ):
        value = row.get(key)
        out[key] = str(value) if key in {"dataset", "condition"} else finite(value)
    return out


def worst_rows(rows: list[dict[str, Any]], key: str, *, reverse: bool, n: int) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        value = finite(row.get(key))
        if value is not None:
            scored.append((value, str(row.get("dataset")), str(row.get("condition")), row))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=reverse)
    return [row_brief(row) for _, _, _, row in scored[: int(n)]]


def summarize_group(label: str, rows: list[dict[str, Any]], *, n_boot: int, seed_base: int) -> dict[str, Any]:
    return {
        "label": label,
        "n_rows": len(rows),
        "n_datasets": len({str(r.get("dataset")) for r in rows}),
        "pearson_pert_delta": bootstrap_equal_dataset(
            rows,
            "blend_delta_vs_anchor_pearson_pert",
            n_boot=n_boot,
            seed=seed_base,
        ),
        "mmd_clamped_delta": bootstrap_equal_dataset(
            rows,
            "blend_delta_vs_anchor_test_mmd_clamped",
            n_boot=n_boot,
            seed=seed_base + 1,
        ),
        "dataset_summary": dataset_summary(rows),
        "worst_pp_rows": worst_rows(rows, "blend_delta_vs_anchor_pearson_pert", reverse=False, n=10),
        "worst_mmd_rows": worst_rows(rows, "blend_delta_vs_anchor_test_mmd_clamped", reverse=True, n=10),
    }


def validate_boundaries(payloads: dict[str, dict[str, Any]]) -> list[str]:
    reasons = []
    support = payloads["support"]
    query = payloads["query"]
    if support.get("scope") != "support_trainselect":
        reasons.append(f"support_scope_unexpected:{support.get('scope')}")
    if (support.get("safety") or {}).get("heldout_query_read") is not False:
        reasons.append("support_payload_marks_query_read")
    if query.get("scope") != "heldout_query_once":
        reasons.append(f"query_scope_unexpected:{query.get('scope')}")
    qs = query.get("safety") or {}
    if qs.get("heldout_query_read") is not True:
        reasons.append("query_payload_does_not_mark_query_read")
    if qs.get("query_result_may_select_or_tune") is not False:
        reasons.append("query_payload_allows_selection_or_tuning")
    if qs.get("canonical_multi_selection") is not False:
        reasons.append("query_payload_marks_canonical_multi_selection")
    for key in ("canonical_single", "canonical_family"):
        payload = payloads[key]
        if payload.get("scope") != "canonical_noharm":
            reasons.append(f"{key}_scope_unexpected:{payload.get('scope')}")
        safety = payload.get("safety") or {}
        if safety.get("canonical_multi_selection") is not False:
            reasons.append(f"{key}_marks_canonical_multi_selection")
        if safety.get("heldout_query_read") is not False:
            reasons.append(f"{key}_marks_query_read")
    return reasons


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    payloads = {
        "support": load_json(args.support_json),
        "canonical_single": load_json(args.canonical_single_json),
        "canonical_family": load_json(args.canonical_family_json),
        "query": load_json(args.query_json),
    }
    groups = {}
    for idx, (label, payload_key, group_name) in enumerate(GROUP_ORDER):
        if payload_key == "support_trainselect":
            source_key = "support"
        elif payload_key == "heldout_query_once":
            source_key = "query"
        elif payload_key == "canonical_noharm":
            source_key = "canonical_single" if group_name == "test_single" else "canonical_family"
        else:
            raise ValueError(f"unknown payload key: {payload_key}")
        groups[label] = summarize_group(
            label,
            rows_for(payloads[source_key], group_name),
            n_boot=int(args.n_boot),
            seed_base=7000 + idx * 10,
        )
    return {
        "status": "trackc_anchor_gated_blend_reporting_ci_ready",
        "boundary": "read_only_frozen_artifacts_no_query_tuning",
        "inputs": {
            "support_json": str(args.support_json),
            "canonical_single_json": str(args.canonical_single_json),
            "canonical_family_json": str(args.canonical_family_json),
            "query_json": str(args.query_json),
        },
        "boundary_reasons": validate_boundaries(payloads),
        "groups": groups,
        "interpretation": {
            "overall": "supported_with_failure_cases",
            "unseen2": "weak_positive_pp_no_mmd_harm",
            "worst_case": "NormanWeissman2019_filtered/CNN1+MAPK1 remains a major failure row",
            "forbidden": "do_not_tune_alpha_gate_checkpoint_threshold_or_branch_on_query",
        },
    }


def table_row(label: str, group: dict[str, Any]) -> str:
    pp = group["pearson_pert_delta"]
    mmd = group["mmd_clamped_delta"]
    pp_ci = f"[{fmt(pp.get('ci_low'))}, {fmt(pp.get('ci_high'))}]"
    mmd_ci = f"[{fmt(mmd.get('ci_low'))}, {fmt(mmd.get('ci_high'))}]"
    return (
        f"| {label} | {group['n_rows']} | {group['n_datasets']} | "
        f"{fmt(pp.get('observed'))} | {pp_ci} | {fmt(pp.get('p_positive'))} | "
        f"{fmt(pp.get('p_harm_pp'))} | {fmt(mmd.get('observed'))} | {mmd_ci} | "
        f"{fmt(mmd.get('p_harm_mmd'))} |"
    )


def render(payload: dict[str, Any]) -> str:
    groups = payload["groups"]
    lines = [
        "# Track C Anchor-Gated Blend Reporting CI",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "Read-only summary over frozen artifacts.  Held-out query results must not be used for route, checkpoint, alpha, threshold, or branch selection.",
        "",
        "## Metric Summary",
        "",
        "| group | rows | datasets | pp delta | pp 95% CI | pp p_positive | pp p_harm<-0.02 | MMD delta | MMD 95% CI | MMD p_harm>0.005 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for label, _, _ in GROUP_ORDER:
        lines.append(table_row(label, groups[label]))
    lines += [
        "",
        "## Conservative Interpretation",
        "",
        "* Aggregate held-out query support is positive, with MMD improvement and no aggregate MMD hard-harm.",
        "* The unseen2 stratum is only weakly positive in pearson_pert and should be worded conservatively.",
        "* The route has condition-level failures; it should be reported as supported with failure cases, not uniformly solved formal multi capability.",
        "",
        "## Worst Held-Out Query Rows",
        "",
        "### Lowest pp deltas",
        "",
    ]
    for row in groups["query_all"]["worst_pp_rows"][:10]:
        lines.append(
            f"* `{row['dataset']}` / `{row['condition']}`: pp_delta `{fmt(row['blend_delta_vs_anchor_pearson_pert'])}`, "
            f"MMD_delta `{fmt(row['blend_delta_vs_anchor_test_mmd_clamped'])}`"
        )
    lines += ["", "### Highest MMD deltas", ""]
    for row in groups["query_all"]["worst_mmd_rows"][:10]:
        lines.append(
            f"* `{row['dataset']}` / `{row['condition']}`: pp_delta `{fmt(row['blend_delta_vs_anchor_pearson_pert'])}`, "
            f"MMD_delta `{fmt(row['blend_delta_vs_anchor_test_mmd_clamped'])}`"
        )
    lines += [
        "",
        "## Dataset-Level Query Summary",
        "",
        "| query group | dataset | rows | pp delta | pp negative frac | MMD delta | MMD harm frac |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("query_all", "query_seen", "query_unseen1", "query_unseen2"):
        for row in groups[label]["dataset_summary"]:
            lines.append(
                f"| {label} | {row['dataset']} | {row['n_rows']} | {fmt(row['pp_delta_mean'])} | "
                f"{fmt(row['pp_negative_fraction'])} | {fmt(row['mmd_delta_mean'])} | {fmt(row['mmd_harm_fraction'])} |"
            )
    lines += ["", "## Boundary Checks", ""]
    if payload["boundary_reasons"]:
        lines.extend(f"* `{reason}`" for reason in payload["boundary_reasons"])
    else:
        lines.append("* none")
    lines += ["", "## Inputs", ""]
    for key, value in payload["inputs"].items():
        lines.append(f"* {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-json", type=Path, default=DEFAULT_SUPPORT_JSON)
    parser.add_argument("--canonical-single-json", type=Path, default=DEFAULT_CANONICAL_SINGLE_JSON)
    parser.add_argument("--canonical-family-json", type=Path, default=DEFAULT_CANONICAL_FAMILY_JSON)
    parser.add_argument("--query-json", type=Path, default=DEFAULT_QUERY_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args()

    payload = summarize(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
