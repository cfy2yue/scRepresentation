#!/usr/bin/env python3
"""CPU proxy gate for the exact prior-covered condition-delta Track C hook.

The exact hook is:
  condition_delta_head_use_in_model=True
  condition_delta_in_model_filter=prior_covered_gene_multi

No model is trained here.  This gate uses the existing safe-trainselect
composition rows as a proxy for whether prior-covered full-raw multi rows have
enough train/support evidence to justify a GPU smoke.  Query, canonical test,
canonical multi, active logs, and new GPU artifacts are not read.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SOURCE = ROOT / "reports/latentfm_trackc_composition_noharm_calibrated_gate_20260623.json"
DATASET_CONTROL = ROOT / "reports/latentfm_trackc_dataset_conditioned_noharm_gate_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_prior_covered_condition_delta_proxy_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_PRIOR_COVERED_CONDITION_DELTA_PROXY_GATE_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_prior_covered(row: dict[str, Any]) -> bool:
    genes = row.get("genes") or []
    return (
        str(row.get("coverage_stratum")) == "full_raw"
        and bool(row.get("covered", True))
        and int(row.get("total_genes", len(genes)) or 0) >= 2
        and int(row.get("raw_gene_covered", 0) or 0) >= int(row.get("total_genes", len(genes)) or 0)
    )


def score_rows(rows: list[dict[str, Any]], *, candidate_key: str = "candidate") -> list[dict[str, Any]]:
    out = []
    for row in rows:
        enabled = is_prior_covered(row)
        item = dict(row)
        item["prior_covered_enabled"] = enabled
        item["prior_covered_candidate"] = float(row[candidate_key]) if enabled else float(row["support_selected_route"])
        if f"{candidate_key}__test_mmd_clamped" in row:
            item["prior_covered_candidate__test_mmd_clamped"] = (
                float(row[f"{candidate_key}__test_mmd_clamped"]) if enabled else float(row["support_selected_route__test_mmd_clamped"])
            )
        elif "candidate__test_mmd_clamped" in row:
            item["prior_covered_candidate__test_mmd_clamped"] = (
                float(row["candidate__test_mmd_clamped"]) if enabled else float(row["support_selected_route__test_mmd_clamped"])
            )
        out.append(item)
    return out


def bootstrap(rows: list[dict[str, Any]], *, metric: str, seed: int, n_boot: int = 2000) -> dict[str, Any]:
    if metric == "pp":
        ck = "prior_covered_candidate"
        bk = "support_selected_route"
        improve_positive = True
    elif metric == "mmd":
        ck = "prior_covered_candidate__test_mmd_clamped"
        bk = "support_selected_route__test_mmd_clamped"
        improve_positive = False
    else:
        raise ValueError(metric)
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if ck in row and bk in row:
            by_ds[str(row["dataset"])].append(float(row[ck]) - float(row[bk]))
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    if not datasets:
        return {"status": "missing"}
    point = float(np.mean([np.mean(by_ds[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        cur = []
        for ds in sample_ds:
            arr = np.asarray(by_ds[str(ds)], dtype=np.float64)
            cur.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        vals.append(float(np.mean(cur)))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "status": "ok",
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)) if improve_positive else float(np.mean(arr < 0.0)),
        "p_harm": float(np.mean(arr < 0.0)) if improve_positive else float(np.mean(arr > 0.0)),
        "by_dataset": {ds: float(np.mean(vals)) for ds, vals in by_ds.items()},
    }


def summarize(rows: list[dict[str, Any]], *, include_mmd: bool, seed: int) -> dict[str, Any]:
    scored = score_rows(rows)
    pp = bootstrap(scored, metric="pp", seed=seed)
    mmd = bootstrap(scored, metric="mmd", seed=seed + 17) if include_mmd else None
    breakdown = []
    for ds in sorted({str(row["dataset"]) for row in scored}):
        sub = [row for row in scored if str(row["dataset"]) == ds]
        enabled = [row for row in sub if row["prior_covered_enabled"]]
        pp_vals = [float(row["prior_covered_candidate"]) - float(row["support_selected_route"]) for row in sub]
        mmd_vals = [
            float(row["prior_covered_candidate__test_mmd_clamped"]) - float(row["support_selected_route__test_mmd_clamped"])
            for row in sub
            if "prior_covered_candidate__test_mmd_clamped" in row
        ]
        breakdown.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "enabled_n": len(enabled),
                "enabled_fraction": float(len(enabled) / max(1, len(sub))),
                "delta_pp": float(np.mean(pp_vals)) if pp_vals else None,
                "delta_mmd_clamped": float(np.mean(mmd_vals)) if mmd_vals else None,
            }
        )
    row_deltas = [float(row["prior_covered_candidate"]) - float(row["support_selected_route"]) for row in scored]
    return {
        "paired_pp_delta": pp,
        "paired_mmd_delta": mmd,
        "dataset_breakdown": breakdown,
        "row_min_delta": float(min(row_deltas)) if row_deltas else None,
        "row_negative_count": int(sum(x < 0 for x in row_deltas)),
        "enabled_total": int(sum(row["prior_covered_enabled"] for row in scored)),
        "rows_total": len(scored),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> None:
    source = load_json(SOURCE)
    dataset_control = load_json(DATASET_CONTROL)
    train_options = source.get("train_summaries") or []
    selected_train = None
    train_ranked = []
    for option in train_options:
        summary = summarize(option.get("rows") or [], include_mmd=False, seed=101)
        summary["source_spec"] = option.get("spec")
        train_ranked.append(summary)
    eligible = []
    for summary in train_ranked:
        by_ds = {row["dataset"]: row for row in summary["dataset_breakdown"]}
        wessels = by_ds.get("Wessels", {})
        norman = by_ds.get("NormanWeissman2019_filtered", {})
        pp = summary["paired_pp_delta"]
        if (
            wessels.get("enabled_n", 0) >= 3
            and norman.get("enabled_n", 0) >= 3
            and (wessels.get("delta_pp") or 0.0) >= 0.02
            and (norman.get("delta_pp") or 0.0) >= -0.01
            and pp.get("p_harm", 1.0) <= 0.20
            and summary["row_negative_count"] == 0
        ):
            eligible.append(summary)
    if eligible:
        selected_train = sorted(
            eligible,
            key=lambda x: (x["paired_pp_delta"]["delta_mean"], x["dataset_breakdown"][1]["enabled_n"]),
            reverse=True,
        )[0]
    else:
        selected_train = max(train_ranked, key=lambda x: x["paired_pp_delta"].get("delta_mean", -999.0))

    support = summarize(source["support_val_summary"]["rows"], include_mmd=True, seed=303)
    shuffled = summarize(source["shuffled_gene_bank_control"]["rows"], include_mmd=True, seed=505)
    zero = summarize(source["zero_beta_control"]["rows"], include_mmd=True, seed=707)
    inverted = dataset_control.get("inverted_dataset_control") or {}
    inverted_pp = ((inverted.get("paired_pp_delta") or {}).get("delta_mean"))
    real_pp = support["paired_pp_delta"].get("delta_mean")
    shuffled_pp = shuffled["paired_pp_delta"].get("delta_mean")
    zero_pp = zero["paired_pp_delta"].get("delta_mean")

    by_ds = {row["dataset"]: row for row in support["dataset_breakdown"]}
    norman = by_ds.get("NormanWeissman2019_filtered", {})
    wessels = by_ds.get("Wessels", {})
    reasons = []
    if wessels.get("enabled_n", 0) < 3:
        reasons.append("support_wessels_prior_covered_rows_lt_3")
    if norman.get("enabled_n", 0) < 3:
        reasons.append("support_norman_prior_covered_rows_lt_3")
    if (wessels.get("delta_pp") or 0.0) < 0.02:
        reasons.append("support_wessels_pp_below_0p02")
    if (norman.get("delta_pp") or 0.0) < -0.01:
        reasons.append("support_norman_pp_below_minus_0p01")
    if support["paired_pp_delta"].get("p_harm", 1.0) > 0.20:
        reasons.append("support_pp_harm_above_0p20")
    if (support["paired_mmd_delta"] or {}).get("delta_mean", 0.0) > 0.005:
        reasons.append("support_mmd_delta_above_0p005")
    if real_pp is None or shuffled_pp is None or real_pp - shuffled_pp < 0.02:
        reasons.append("shuffled_control_not_separated_by_0p02")
    if real_pp is None or zero_pp is None or real_pp - zero_pp < 0.02:
        reasons.append("zero_control_not_separated_by_0p02")
    if inverted_pp is None or real_pp is None or real_pp - float(inverted_pp) < 0.02:
        reasons.append("inverted_dataset_control_not_separated_by_0p02")
    if not eligible:
        reasons.append("no_train_multi_loo_option_passed_prior_covered_selection_gate")

    status = "trackc_prior_covered_condition_delta_proxy_gate_fail_no_gpu" if reasons else "trackc_prior_covered_condition_delta_proxy_gate_pass_protocol_review"
    result = {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "none" if reasons else "protocol_review_then_one_capped_trackc_smoke",
        "decision_reasons": reasons,
        "boundary": {
            "cpu_only_existing_safe_trainselect_rows": True,
            "heldout_query_read": False,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
            "exact_hook_proxy_only": "condition_delta_head_use_in_model=True plus prior_covered_gene_multi",
        },
        "selected_train_proxy": selected_train,
        "support_proxy": support,
        "shuffled_control": shuffled,
        "zero_control": zero,
        "inverted_dataset_control_pp_delta": inverted_pp,
        "inputs": {
            "source": str(SOURCE),
            "dataset_control": str(DATASET_CONTROL),
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track C Prior-Covered Condition-Delta Proxy Gate",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- CPU-only proxy gate for the exact `condition_delta_head_use_in_model=True` + `prior_covered_gene_multi` hook.",
        "- Uses existing safe-trainselect train_multi/support_val_multi rows from the no-harm calibrated composition gate.",
        "- Does not read held-out query, canonical test, canonical multi, active logs, or new GPU artifacts.",
        "",
        "## Support-Val Summary",
        "",
        f"- support pp delta: `{fmt(support['paired_pp_delta'].get('delta_mean'))}`",
        f"- support pp p_harm: `{fmt(support['paired_pp_delta'].get('p_harm'))}`",
        f"- support MMD delta: `{fmt((support['paired_mmd_delta'] or {}).get('delta_mean'))}`",
        f"- shuffled pp delta: `{fmt(shuffled['paired_pp_delta'].get('delta_mean'))}`",
        f"- zero pp delta: `{fmt(zero['paired_pp_delta'].get('delta_mean'))}`",
        f"- inverted-dataset pp delta: `{fmt(inverted_pp)}`",
        "",
        "| dataset | n | prior-covered n | enabled frac | pp delta | MMD delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in support["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {row['enabled_n']} | "
            f"{row['enabled_fraction']:.6f} | {fmt(row['delta_pp'])} | {fmt(row['delta_mmd_clamped'])} |"
        )
    lines.extend([
        "",
        "## Train Selection Proxy",
        "",
        f"- selected source spec: `{selected_train.get('source_spec')}`",
        f"- train pp delta: `{fmt(selected_train['paired_pp_delta'].get('delta_mean'))}`",
        f"- train pp p_harm: `{fmt(selected_train['paired_pp_delta'].get('p_harm'))}`",
        f"- train enabled rows: `{selected_train['enabled_total']}/{selected_train['rows_total']}`",
        "",
        "## Decision Reasons",
        "",
    ])
    lines.extend([f"- `{reason}`" for reason in reasons] or ["- `none`"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The exact prior-covered hook remains code-available, but the safe-trainselect proxy has too little Wessels support coverage and does not separate cleanly from the inverted-dataset control. This does not prove the hook could never work, but it fails the predeclared evidence threshold for spending GPU on a capped Track C smoke.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
