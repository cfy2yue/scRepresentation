#!/usr/bin/env python3
"""CPU-only preflight for Track C prior-covered condition-delta candidate 4.

This script answers a narrow gate question before any GPU work:
do safe trainselect train/support rows show a prior-covered condition-delta
signal that separates from shuffled and inverted controls without Norman harm?

It does not train, launch GPU jobs, read held-out Track C query, read canonical
test metrics, read canonical multi, or poll active logs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
COMPOSITION = REPORTS / "latentfm_trackc_composition_noharm_calibrated_gate_20260623.json"
DATASET_CONTROL = REPORTS / "latentfm_trackc_dataset_conditioned_noharm_gate_20260624.json"
META_GATE = REPORTS / "latentfm_trackc_condition_delta_prior_covered_meta_gate_20260624.json"
PROXY_GATE = REPORTS / "latentfm_trackc_prior_covered_condition_delta_proxy_gate_20260624.json"
PAIRTYPE_CONTEXT = REPORTS / "latentfm_trackc_support_only_pairtype_strata_summary_20260624.json"
OT_CONTEXT = REPORTS / "latentfm_ot_condition_overlap_reliability_gate_20260624.json"
MATCHED_BREADTH_CONTEXT = REPORTS / "latentfm_matched_dataset_breadth_gate_20260624.json"

OUT_JSON = REPORTS / "latentfm_trackc_prior_covered_condition_delta_preflight_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_PRIOR_COVERED_CONDITION_DELTA_PREFLIGHT_20260624.md"

EXPECTED_SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    value = fnum(value)
    if value is None:
        return "NA"
    return f"{value:+.6f}"


def is_prior_covered(row: dict[str, Any]) -> bool:
    genes = row.get("genes") or []
    total = int(row.get("total_genes", len(genes)) or 0)
    raw_covered = int(row.get("raw_gene_covered", 0) or 0)
    return (
        str(row.get("coverage_stratum")) == "full_raw"
        and bool(row.get("covered", True))
        and total >= 2
        and raw_covered >= total
    )


def candidate_value(row: dict[str, Any], candidate_key: str) -> float:
    value = row.get(candidate_key)
    if value is None and candidate_key != "candidate":
        value = row.get("candidate")
    if value is None:
        raise KeyError(f"missing candidate key {candidate_key!r}")
    return float(value)


def score_rows(
    rows: list[dict[str, Any]],
    *,
    candidate_key: str = "candidate",
) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        enabled = is_prior_covered(row)
        baseline = float(row["support_selected_route"])
        item = {
            "dataset": str(row["dataset"]),
            "condition": str(row["condition"]),
            "genes": list(row.get("genes") or []),
            "prior_covered_enabled": enabled,
            "coverage_stratum": row.get("coverage_stratum"),
            "total_genes": int(row.get("total_genes", 0) or 0),
            "raw_gene_covered": int(row.get("raw_gene_covered", 0) or 0),
            "support_selected_route": baseline,
            "candidate": candidate_value(row, candidate_key) if enabled else baseline,
        }
        if "support_selected_route__test_mmd_clamped" in row:
            mmd_key = f"{candidate_key}__test_mmd_clamped"
            if mmd_key not in row and candidate_key != "candidate":
                mmd_key = "candidate__test_mmd_clamped"
            base_mmd = row.get("support_selected_route__test_mmd_clamped")
            cand_mmd = row.get(mmd_key)
            if base_mmd is not None and cand_mmd is not None:
                item["support_selected_route__test_mmd_clamped"] = float(base_mmd)
                item["candidate__test_mmd_clamped"] = float(cand_mmd) if enabled else float(base_mmd)
        item["delta_pp"] = float(item["candidate"] - item["support_selected_route"])
        if "candidate__test_mmd_clamped" in item:
            item["delta_mmd_clamped"] = float(
                item["candidate__test_mmd_clamped"]
                - item["support_selected_route__test_mmd_clamped"]
            )
        scored.append(item)
    return scored


def bootstrap_dataset_mean(
    rows: list[dict[str, Any]],
    *,
    delta_key: str,
    positive_improves: bool,
    seed: int,
    n_boot: int = 2000,
) -> dict[str, Any]:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if delta_key in row:
            by_dataset[row["dataset"]].append(float(row[delta_key]))
    datasets = sorted(ds for ds, vals in by_dataset.items() if vals)
    if not datasets:
        return {"status": "missing"}
    point = float(np.mean([np.mean(by_dataset[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        cur = []
        for ds in sample_ds:
            arr = np.asarray(by_dataset[str(ds)], dtype=np.float64)
            cur.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        boot.append(float(np.mean(cur)))
    arr = np.asarray(boot, dtype=np.float64)
    return {
        "status": "ok",
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)) if positive_improves else float(np.mean(arr < 0.0)),
        "p_harm": float(np.mean(arr < 0.0)) if positive_improves else float(np.mean(arr > 0.0)),
        "by_dataset": {ds: float(np.mean(vals)) for ds, vals in by_dataset.items()},
    }


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    candidate_key: str = "candidate",
    include_mmd: bool,
    seed: int,
) -> dict[str, Any]:
    scored = score_rows(rows, candidate_key=candidate_key)
    dataset_rows = []
    for dataset in sorted({row["dataset"] for row in scored}):
        sub = [row for row in scored if row["dataset"] == dataset]
        enabled = [row for row in sub if row["prior_covered_enabled"]]
        pp = [row["delta_pp"] for row in sub]
        mmd = [row["delta_mmd_clamped"] for row in sub if "delta_mmd_clamped" in row]
        dataset_rows.append(
            {
                "dataset": dataset,
                "n_rows": len(sub),
                "enabled_n": len(enabled),
                "enabled_fraction": float(len(enabled) / max(1, len(sub))),
                "delta_pp": float(np.mean(pp)) if pp else None,
                "delta_mmd_clamped": float(np.mean(mmd)) if mmd else None,
                "min_row_delta_pp": float(min(pp)) if pp else None,
                "negative_row_count": int(sum(x < 0.0 for x in pp)),
            }
        )
    pp_summary = bootstrap_dataset_mean(
        scored,
        delta_key="delta_pp",
        positive_improves=True,
        seed=seed,
    )
    mmd_summary = None
    if include_mmd:
        mmd_summary = bootstrap_dataset_mean(
            scored,
            delta_key="delta_mmd_clamped",
            positive_improves=False,
            seed=seed + 17,
        )
    deltas = [row["delta_pp"] for row in scored]
    return {
        "rows_total": len(scored),
        "enabled_total": int(sum(row["prior_covered_enabled"] for row in scored)),
        "paired_pp_delta": pp_summary,
        "paired_mmd_delta": mmd_summary,
        "dataset_breakdown": dataset_rows,
        "row_min_delta_pp": float(min(deltas)) if deltas else None,
        "row_negative_count": int(sum(x < 0.0 for x in deltas)),
    }


def dataset_metric(summary: dict[str, Any], dataset: str, key: str, default: Any = None) -> Any:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row.get(key, default)
    return default


def select_train_summary(train_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for idx, item in enumerate(train_summaries):
        summary = summarize_rows(item.get("rows") or [], include_mmd=False, seed=1101 + idx)
        summary["source_spec"] = item.get("spec")
        scored.append(summary)

    eligible = []
    for summary in scored:
        pp = summary["paired_pp_delta"]
        norman_pp = dataset_metric(summary, "NormanWeissman2019_filtered", "delta_pp", 0.0)
        wessels_pp = dataset_metric(summary, "Wessels", "delta_pp", 0.0)
        if (
            pp.get("delta_mean", -999.0) >= 0.02
            and pp.get("p_harm", 1.0) <= 0.20
            and norman_pp >= -0.01
            and wessels_pp >= 0.02
            and summary["enabled_total"] >= 6
        ):
            eligible.append(summary)

    if eligible:
        selected = dict(max(eligible, key=lambda item: item["paired_pp_delta"]["delta_mean"]))
        selected["selection_status"] = "train_selection_pass"
    else:
        selected = dict(max(scored, key=lambda item: item["paired_pp_delta"].get("delta_mean", -999.0)))
        selected["selection_status"] = "train_selection_fail_best_available"
    selected["all_train_options"] = scored
    return selected


def split_guard_ok(payload: dict[str, Any]) -> bool:
    guard = payload.get("split_guard") or {}
    split_file = Path(str(guard.get("split_file", "")))
    leakage = str(guard.get("leakage_status", "")).lower()
    return (
        split_file == EXPECTED_SAFE_SPLIT
        and guard.get("sha256") == guard.get("expected_sha256")
        and "no_heldout_query" in leakage
        and "no_canonical_outputs" in leakage
    )


def context_status(payload: dict[str, Any]) -> str:
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    return str(payload.get("status") or decision.get("status") or "missing")


def main() -> None:
    composition = load_json(COMPOSITION)
    dataset_control = load_json(DATASET_CONTROL)
    context = {
        "meta_gate": load_json(META_GATE),
        "proxy_gate": load_json(PROXY_GATE),
        "pairtype_support_only": load_json(PAIRTYPE_CONTEXT),
        "ot_condition_overlap": load_json(OT_CONTEXT),
        "matched_dataset_breadth": load_json(MATCHED_BREADTH_CONTEXT),
    }

    support_actual = summarize_rows(
        composition["support_val_summary"]["rows"],
        include_mmd=True,
        seed=2101,
    )
    support_shuffled = summarize_rows(
        composition["shuffled_gene_bank_control"]["rows"],
        include_mmd=True,
        seed=2201,
    )
    support_zero = summarize_rows(
        composition["zero_beta_control"]["rows"],
        include_mmd=True,
        seed=2301,
    )
    support_inverted = summarize_rows(
        dataset_control["inverted_dataset_control"]["rows"],
        candidate_key="dataset_conditioned_candidate",
        include_mmd=True,
        seed=2401,
    )
    selected_train = select_train_summary(composition.get("train_summaries") or [])

    actual_pp = support_actual["paired_pp_delta"].get("delta_mean")
    actual_mmd = (support_actual["paired_mmd_delta"] or {}).get("delta_mean")
    shuffled_pp = support_shuffled["paired_pp_delta"].get("delta_mean")
    zero_pp = support_zero["paired_pp_delta"].get("delta_mean")
    inverted_pp = support_inverted["paired_pp_delta"].get("delta_mean")
    norman_pp = dataset_metric(support_actual, "NormanWeissman2019_filtered", "delta_pp")
    norman_mmd = dataset_metric(support_actual, "NormanWeissman2019_filtered", "delta_mmd_clamped")
    wessels_pp = dataset_metric(support_actual, "Wessels", "delta_pp")
    wessels_enabled = dataset_metric(support_actual, "Wessels", "enabled_n", 0)
    norman_enabled = dataset_metric(support_actual, "NormanWeissman2019_filtered", "enabled_n", 0)

    reasons: list[str] = []
    if not split_guard_ok(composition) or not split_guard_ok(dataset_control):
        reasons.append("safe_trainselect_split_guard_failed")
    if selected_train.get("selection_status") != "train_selection_pass":
        reasons.append("train_selection_prior_covered_gate_failed")
    if actual_pp is None or actual_pp < 0.02:
        reasons.append("support_actual_pp_below_0p02")
    if support_actual["paired_pp_delta"].get("p_harm", 1.0) > 0.20:
        reasons.append("support_actual_pp_harm_above_0p20")
    if actual_mmd is None or actual_mmd > 0.005:
        reasons.append("support_actual_mmd_above_0p005")
    if norman_pp is None or norman_pp < -0.01:
        reasons.append("support_norman_pp_below_minus_0p01")
    if norman_mmd is None or norman_mmd > 0.005:
        reasons.append("support_norman_mmd_above_0p005")
    if wessels_pp is None or wessels_pp < 0.02:
        reasons.append("support_wessels_pp_below_0p02")
    if int(wessels_enabled or 0) < 3:
        reasons.append("support_wessels_prior_covered_rows_lt_3")
    if int(norman_enabled or 0) < 3:
        reasons.append("support_norman_prior_covered_rows_lt_3")
    if actual_pp is None or shuffled_pp is None or actual_pp - shuffled_pp < 0.02:
        reasons.append("shuffled_control_not_separated_by_0p02")
    if actual_pp is None or zero_pp is None or actual_pp - zero_pp < 0.02:
        reasons.append("zero_control_not_separated_by_0p02")
    if actual_pp is None or inverted_pp is None or actual_pp - inverted_pp < 0.02:
        reasons.append("inverted_control_not_separated_by_0p02")

    passed = not reasons
    status = (
        "trackc_prior_covered_condition_delta_preflight_pass_one_capped_smoke_protocol_required"
        if passed
        else "trackc_prior_covered_condition_delta_preflight_fail_no_gpu"
    )
    gpu_authorization = "one_capped_2k_smoke_after_launcher_review" if passed else "none"

    launcher_requirements = {
        "authorized_only_if_status_passes": True,
        "max_gpu_smokes": 1,
        "max_total_steps": 2000,
        "split_file": str(EXPECTED_SAFE_SPLIT),
        "forbidden_reads": [
            "held_out_trackc_query",
            "canonical_multi",
            "canonical_metrics_for_selection",
        ],
        "required_config": {
            "condition_delta_head_use_in_model": True,
            "condition_delta_in_model_filter": "prior_covered_gene_multi",
            "trackc_support_context_pair_type_filter": "none",
            "train_eval_enabled": False,
        },
        "required_controls_after_smoke": [
            "query_free_safe_trainselect_support_actual",
            "query_free_safe_trainselect_support_shuffled_condition_delta",
            "query_free_safe_trainselect_support_inverted_or_wrong_dataset_delta",
            "Norman no-harm stratum",
        ],
        "promotion_gate_after_smoke": (
            "support pp delta >= +0.02, MMD delta <= +0.005, Norman pp >= -0.01, "
            "actual minus shuffled >= +0.02, actual minus inverted >= +0.02; "
            "then external review before any canonical no-harm"
        ),
        "fail_close_rule_after_smoke": (
            "close exact candidate if support/control separation or Norman no-harm fails; "
            "do not retune from canonical or held-out Track C query"
        ),
    }

    result = {
        "status": status,
        "gpu_authorization": gpu_authorization,
        "hypothesis": (
            "A condition-delta head gated to prior-covered multi-gene conditions should "
            "add support signal only for genuinely covered trainselect rows; shuffled or "
            "inverted controls should collapse, and Norman should not be harmed."
        ),
        "boundary": {
            "cpu_only": True,
            "safe_trainselect_train_support_rows_only_for_gate": True,
            "heldout_trackc_query_read": False,
            "canonical_metrics_read": False,
            "canonical_multi_read": False,
            "active_log_polling": False,
            "gpu_launched": False,
        },
        "inputs": {
            "composition_rows": str(COMPOSITION),
            "dataset_control_rows": str(DATASET_CONTROL),
            "meta_gate_context": str(META_GATE),
            "proxy_gate_context": str(PROXY_GATE),
            "pairtype_context": str(PAIRTYPE_CONTEXT),
            "ot_context": str(OT_CONTEXT),
            "matched_breadth_context": str(MATCHED_BREADTH_CONTEXT),
        },
        "context_statuses": {name: context_status(payload) for name, payload in context.items()},
        "criteria": {
            "train_selection": "train-only selected option pp >= +0.02, p_harm <= 0.20, Norman >= -0.01, Wessels >= +0.02, enabled rows >= 6",
            "support_actual": "support pp >= +0.02, pp p_harm <= 0.20, MMD <= +0.005",
            "norman_no_harm": "support Norman pp >= -0.01 and Norman MMD <= +0.005",
            "coverage_floor": "support Norman and Wessels each have at least 3 prior-covered rows",
            "control_separation": "actual pp exceeds shuffled, zero, and inverted controls by at least +0.02",
        },
        "decision_reasons": reasons,
        "selected_train": selected_train,
        "support_actual": support_actual,
        "support_shuffled_control": support_shuffled,
        "support_zero_control": support_zero,
        "support_inverted_control": support_inverted,
        "control_margins": {
            "actual_minus_shuffled_pp": None if actual_pp is None or shuffled_pp is None else float(actual_pp - shuffled_pp),
            "actual_minus_zero_pp": None if actual_pp is None or zero_pp is None else float(actual_pp - zero_pp),
            "actual_minus_inverted_pp": None if actual_pp is None or inverted_pp is None else float(actual_pp - inverted_pp),
        },
        "non_duplication": {
            "fixed_support_only": "Different hook: this tests prior-covered condition-delta injection, not a generic support-only/support-present adapter.",
            "pair_type_support_only": "Different routing axis: pair-type masking gates support context by single/multi labels; this gates condition deltas by full prior gene coverage.",
            "ot": "Does not use OT transport features, OT pair sampling, or OT condition-overlap signals.",
            "matched_breadth": "Does not change training-data breadth or dataset-count composition.",
        },
        "launcher_requirements_if_pass": launcher_requirements,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track C Prior-Covered Condition-Delta Preflight",
        "",
        f"Status: `{status}`",
        f"GPU authorization: `{gpu_authorization}`",
        "",
        "## Hypothesis",
        "",
        result["hypothesis"],
        "",
        "## Boundary",
        "",
        "- CPU-only preflight; no GPU launch.",
        "- Gate metrics use safe trainselect train/support rows only.",
        "- Does not read held-out Track C query, canonical metrics, canonical multi, or active logs.",
        "",
        "## Inputs / Provenance",
        "",
    ]
    for label, path in result["inputs"].items():
        lines.append(f"- `{label}`: `{path}`")
    lines.extend([
        "",
        "Context statuses:",
    ])
    for label, stat in result["context_statuses"].items():
        lines.append(f"- `{label}`: `{stat}`")

    lines.extend([
        "",
        "## Criteria",
        "",
    ])
    for label, text in result["criteria"].items():
        lines.append(f"- `{label}`: {text}")

    lines.extend([
        "",
        "## Key Metrics",
        "",
        f"- selected train spec: `{selected_train.get('source_spec')}`",
        f"- train selection status: `{selected_train.get('selection_status')}`",
        f"- train pp delta: `{fmt(selected_train['paired_pp_delta'].get('delta_mean'))}`",
        f"- train pp p_harm: `{fmt(selected_train['paired_pp_delta'].get('p_harm'))}`",
        f"- support actual pp delta: `{fmt(actual_pp)}`",
        f"- support actual pp p_harm: `{fmt(support_actual['paired_pp_delta'].get('p_harm'))}`",
        f"- support actual MMD delta: `{fmt(actual_mmd)}`",
        f"- shuffled control pp delta: `{fmt(shuffled_pp)}`",
        f"- zero control pp delta: `{fmt(zero_pp)}`",
        f"- inverted control pp delta: `{fmt(inverted_pp)}`",
        f"- actual minus shuffled pp: `{fmt(result['control_margins']['actual_minus_shuffled_pp'])}`",
        f"- actual minus zero pp: `{fmt(result['control_margins']['actual_minus_zero_pp'])}`",
        f"- actual minus inverted pp: `{fmt(result['control_margins']['actual_minus_inverted_pp'])}`",
        f"- support Norman pp / MMD: `{fmt(norman_pp)}` / `{fmt(norman_mmd)}`",
        f"- support Wessels pp / enabled rows: `{fmt(wessels_pp)}` / `{wessels_enabled}`",
        "",
        "| dataset | support rows | prior-covered rows | pp delta | MMD delta | min row pp | negative rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in support_actual["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_rows']} | {row['enabled_n']} | "
            f"{fmt(row['delta_pp'])} | {fmt(row['delta_mmd_clamped'])} | "
            f"{fmt(row['min_row_delta_pp'])} | {row['negative_row_count']} |"
        )

    lines.extend([
        "",
        "## Decision",
        "",
        "Pass/fail: `" + ("pass" if passed else "fail") + "`",
        "",
        "Reasons:",
    ])
    lines.extend([f"- `{reason}`" for reason in reasons] or ["- `none`"])

    lines.extend([
        "",
        "## Non-Duplication",
        "",
    ])
    for label, text in result["non_duplication"].items():
        lines.append(f"- `{label}`: {text}")

    lines.extend([
        "",
        "## Next Launcher / Config Requirements If Pass",
        "",
        f"- maximum GPU smokes: `{launcher_requirements['max_gpu_smokes']}`",
        f"- maximum total steps: `{launcher_requirements['max_total_steps']}`",
        f"- split file: `{launcher_requirements['split_file']}`",
        "- required config:",
    ])
    for key, value in launcher_requirements["required_config"].items():
        lines.append(f"  - `{key}={value}`")
    lines.extend([
        "- forbidden reads: `" + "`, `".join(launcher_requirements["forbidden_reads"]) + "`",
        "- required post-smoke controls: `" + "`, `".join(launcher_requirements["required_controls_after_smoke"]) + "`",
        f"- promotion gate after smoke: {launcher_requirements['promotion_gate_after_smoke']}",
        f"- fail-close rule after smoke: {launcher_requirements['fail_close_rule_after_smoke']}",
        "",
        "Because this preflight failed, these requirements are recorded for future use only; they do not authorize a launch now.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
