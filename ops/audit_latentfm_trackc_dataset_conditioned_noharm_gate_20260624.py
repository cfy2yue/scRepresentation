#!/usr/bin/env python3
"""Track C dataset-conditioned conservative no-harm CPU gate.

This gate uses only the existing safe-trainselect no-harm calibrated
composition rows.  It tests whether a deliberately conservative row-level
selector can keep the Wessels composition signal while avoiding the Norman
tail: enable composition only by frozen dataset/coverage-stratum rules selected
on train_multi LOO rows; score support_val once.

Held-out query, canonical test, canonical multi, active logs, and GPU artifacts
are not read.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
IN_JSON = ROOT / "reports/latentfm_trackc_composition_noharm_calibrated_gate_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_dataset_conditioned_noharm_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_DATASET_CONDITIONED_NOHARM_GATE_20260624.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"


@dataclass(frozen=True)
class Spec:
    name: str
    norman_full: bool
    norman_partial: bool
    wessels_full: bool
    wessels_partial: bool


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def specs() -> list[Spec]:
    out = [
        Spec("always_off_route", False, False, False, False),
        Spec("always_on_composition", True, True, True, True),
        Spec("full_raw_all_datasets_only", True, False, True, False),
        Spec("partial_raw_all_datasets_only", False, True, False, True),
        Spec("wessels_full_raw_only", False, False, True, False),
        Spec("wessels_full_and_partial_only", False, False, True, True),
        Spec("wessels_partial_only", False, False, False, True),
        Spec("norman_full_raw_wessels_full_raw", True, False, True, False),
        Spec("norman_off_wessels_all", False, False, True, True),
        Spec("norman_partial_off_wessels_all", True, False, True, True),
        Spec("norman_partial_on_wessels_full", False, True, True, False),
    ]
    # Exhaustive small grid; named specs above make reports easier to read.
    seen = {x.name for x in out}
    for nf in (False, True):
        for npart in (False, True):
            for wf in (False, True):
                for wpart in (False, True):
                    name = f"grid_nf{int(nf)}_np{int(npart)}_wf{int(wf)}_wp{int(wpart)}"
                    if name not in seen:
                        out.append(Spec(name, nf, npart, wf, wpart))
    return out


def is_norman(row: dict[str, Any]) -> bool:
    return str(row.get("dataset")) == "NormanWeissman2019_filtered"


def is_wessels(row: dict[str, Any]) -> bool:
    return str(row.get("dataset")) == "Wessels"


def enabled(row: dict[str, Any], spec: Spec) -> bool:
    stratum = str(row.get("coverage_stratum"))
    if is_norman(row):
        return (stratum == "full_raw" and spec.norman_full) or (stratum == "partial_raw" and spec.norman_partial)
    if is_wessels(row):
        return (stratum == "full_raw" and spec.wessels_full) or (stratum == "partial_raw" and spec.wessels_partial)
    return False


def invert_dataset_spec(spec: Spec) -> Spec:
    return Spec(
        f"inverted_dataset_control_of_{spec.name}",
        norman_full=spec.wessels_full,
        norman_partial=spec.wessels_partial,
        wessels_full=spec.norman_full,
        wessels_partial=spec.norman_partial,
    )


def score_rows(rows: list[dict[str, Any]], spec: Spec) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        use_comp = enabled(row, spec)
        item = dict(row)
        item["dataset_conditioned_enabled"] = bool(use_comp)
        item["dataset_conditioned_spec"] = spec.name
        item["dataset_conditioned_candidate"] = float(row["candidate"]) if use_comp else float(row["support_selected_route"])
        if "candidate__test_mmd_clamped" in row:
            item["dataset_conditioned_candidate__test_mmd_clamped"] = (
                float(row["candidate__test_mmd_clamped"]) if use_comp else float(row["support_selected_route__test_mmd_clamped"])
            )
        out.append(item)
    return out


def dataset_delta(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, float]:
    out = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        vals = [float(row[candidate]) - float(row[baseline]) for row in rows if str(row["dataset"]) == ds]
        if vals:
            out[ds] = float(np.mean(vals))
    return out


def paired_bootstrap(rows: list[dict[str, Any]], candidate: str, baseline: str, *, metric: str, n_boot: int, seed: int) -> dict[str, Any]:
    if metric == "pp":
        ck, bk = candidate, baseline
        improve_positive = True
    elif metric == "mmd_clamped":
        ck, bk = f"{candidate}__test_mmd_clamped", f"{baseline}__test_mmd_clamped"
        improve_positive = False
    else:
        raise ValueError(metric)
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(ck) is not None and row.get(bk) is not None:
            by_ds[str(row["dataset"])].append(float(row[ck]) - float(row[bk]))
    datasets = sorted(ds for ds, vals in by_ds.items() if vals)
    if not datasets:
        return {"status": "missing", "candidate": candidate, "baseline": baseline, "metric": metric}
    point = float(np.mean([np.mean(by_ds[ds]) for ds in datasets]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sample_ds = rng.choice(datasets, size=len(datasets), replace=True)
        vals = []
        for ds in sample_ds:
            arr = np.asarray(by_ds[str(ds)], dtype=np.float64)
            vals.append(float(np.mean(rng.choice(arr, size=len(arr), replace=True))))
        boot.append(float(np.mean(vals)))
    arr = np.asarray(boot, dtype=np.float64)
    return {
        "status": "ok",
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0.0)) if improve_positive else float(np.mean(arr < 0.0)),
        "p_harm": float(np.mean(arr < 0.0)) if improve_positive else float(np.mean(arr > 0.0)),
        "by_dataset": {ds: float(np.mean(vals)) for ds, vals in by_ds.items()},
    }


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def read_wessels_gap(payload: dict[str, Any]) -> float:
    for row in (payload.get("support_val_summary") or {}).get("dataset_breakdown") or []:
        if row.get("dataset") == "Wessels" and row.get("route_gap_pp") is not None:
            return float(row["route_gap_pp"])
    return 0.1536066201897797


def summarize(rows: list[dict[str, Any]], spec: Spec, *, n_boot: int, seed: int, wessels_gap: float, include_mmd: bool) -> dict[str, Any]:
    pp = paired_bootstrap(rows, "dataset_conditioned_candidate", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd = (
        paired_bootstrap(rows, "dataset_conditioned_candidate", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 100)
        if include_mmd and rows and "dataset_conditioned_candidate__test_mmd_clamped" in rows[0]
        else None
    )
    ds_pp = dataset_delta(rows, "dataset_conditioned_candidate", "support_selected_route")
    ds_mmd = dataset_delta(rows, "dataset_conditioned_candidate__test_mmd_clamped", "support_selected_route__test_mmd_clamped") if mmd else {}
    breakdown = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        delta = ds_pp.get(ds)
        gap = wessels_gap if ds == "Wessels" else None
        breakdown.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "enabled_fraction": float(np.mean([bool(row["dataset_conditioned_enabled"]) for row in sub])),
                "delta_pp": delta,
                "delta_mmd_clamped": ds_mmd.get(ds),
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or abs(gap) <= 1e-12 or delta is None else float(delta / gap),
            }
        )
    row_deltas = [float(row["dataset_conditioned_candidate"]) - float(row["support_selected_route"]) for row in rows]
    return {
        "spec": spec.name,
        "spec_params": spec.__dict__,
        "paired_pp_delta": pp,
        "paired_mmd_delta": mmd,
        "dataset_breakdown": breakdown,
        "enabled_fraction": float(np.mean([bool(row["dataset_conditioned_enabled"]) for row in rows])) if rows else 0.0,
        "row_negative_count": int(sum(v < 0.0 for v in row_deltas)),
        "row_min_delta": float(min(row_deltas)) if row_deltas else None,
        "rows": rows,
    }


def select_spec(train_summaries: list[dict[str, Any]]) -> str:
    eligible = []
    for row in train_summaries:
        w = find_dataset(row, "Wessels")
        n = find_dataset(row, "NormanWeissman2019_filtered")
        pp = row["paired_pp_delta"]
        if (
            float(w.get("delta_pp") if w.get("delta_pp") is not None else -999.0) >= 0.02
            and float(n.get("delta_pp") if n.get("delta_pp") is not None else -999.0) >= -0.01
            and float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) <= 0.20
            and int(row.get("row_negative_count") or 0) == 0
            and float(row.get("enabled_fraction") or 0.0) > 0.0
        ):
            eligible.append(row)
    pool = eligible or train_summaries
    # Conservative ordering: no negative rows first, then Wessels closure, then
    # overall pp.  This deliberately differs from the previous always-on
    # composition selection that optimized mean signal before row-level safety.
    return str(
        sorted(
            pool,
            key=lambda row: (
                -int(row.get("row_negative_count") or 0),
                float(find_dataset(row, "Wessels").get("delta_pp") if find_dataset(row, "Wessels").get("delta_pp") is not None else -999.0),
                float(find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") if find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") is not None else -999.0),
                float(row["paired_pp_delta"].get("delta_mean") if row["paired_pp_delta"].get("delta_mean") is not None else -999.0),
            ),
            reverse=True,
        )[0]["spec"]
    )


def decide(payload: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    split = payload.get("split_guard") or {}
    support = payload["selected_support_summary"]
    inverted = payload["inverted_dataset_control"]
    pp = support["paired_pp_delta"]
    mmd = support["paired_mmd_delta"] or {}
    w = find_dataset(support, "Wessels")
    n = find_dataset(support, "NormanWeissman2019_filtered")
    if split.get("sha256") != EXPECTED_TRAINSELECT_SHA256:
        reasons.append("trainselect_split_hash_mismatch")
    if float(w.get("delta_pp") if w.get("delta_pp") is not None else -999.0) < 0.02:
        reasons.append("support_wessels_delta_below_0p02")
    if float(w.get("route_gap_closed_fraction") if w.get("route_gap_closed_fraction") is not None else -999.0) < 0.05:
        reasons.append("wessels_route_gap_closure_below_0p05")
    if float(n.get("delta_pp") if n.get("delta_pp") is not None else -999.0) < -0.01:
        reasons.append("support_norman_delta_below_minus_0p01")
    if float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("bootstrap_pp_harm_above_0p20")
    if int(support.get("row_negative_count") or 0) > 0:
        reasons.append("support_row_negative_count_above_0")
    if float(mmd.get("delta_mean") if mmd.get("delta_mean") is not None else 999.0) > 0.005:
        reasons.append("mmd_delta_hard_harm_above_0p005")
    real_delta = float(pp.get("delta_mean") if pp.get("delta_mean") is not None else 0.0)
    inverted_delta = float(inverted["paired_pp_delta"].get("delta_mean") if inverted["paired_pp_delta"].get("delta_mean") is not None else 0.0)
    if inverted_delta > real_delta - 0.02:
        reasons.append("inverted_dataset_control_not_separated_by_0p02")
    status = (
        "trackc_dataset_conditioned_noharm_gate_pass_code_gate_next_no_gpu"
        if not reasons
        else "trackc_dataset_conditioned_noharm_gate_fail_no_gpu"
    )
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "code_gate_only_if_pass_else_none",
        "reasons": reasons,
    }


def render(payload: dict[str, Any]) -> str:
    support = payload["selected_support_summary"]
    inverted = payload["inverted_dataset_control"]
    lines = [
        "# Track C Dataset-Conditioned No-Harm Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"GPU authorization: `{payload['decision']['gpu_authorization']}`",
        "",
        "## Boundary",
        "",
        "- Uses only existing safe-trainselect no-harm calibrated composition rows.",
        "- Rule selection uses train_multi LOO rows only; support_val_multi is final scoring.",
        "- Rule features are dataset name and coverage stratum only.",
        "- Does not read held-out query, canonical test, canonical multi, active logs, or GPU artifacts.",
        "",
        "## Selection Rule",
        "",
        f"- selected spec: `{payload['selected_spec']}`",
        "- selection priority: eligible rules must have Wessels >= +0.02, Norman >= -0.01, pp p_harm <= 0.20, and zero negative train rows; then maximize Wessels and overall pp.",
        "",
        "## Support-Val Gate Criteria",
        "",
        f"- Wessels pp delta: `{fmt(find_dataset(support, 'Wessels').get('delta_pp'))}`",
        f"- Wessels route-gap closure: `{fmt(find_dataset(support, 'Wessels').get('route_gap_closed_fraction'))}`",
        f"- Norman pp delta: `{fmt(find_dataset(support, 'NormanWeissman2019_filtered').get('delta_pp'))}`",
        f"- bootstrap pp p_harm: `{fmt(support['paired_pp_delta'].get('p_harm'))}`",
        f"- support row negative count: `{support.get('row_negative_count')}`",
        f"- MMD delta: `{fmt((support['paired_mmd_delta'] or {}).get('delta_mean'))}`",
        f"- inverted dataset control pp delta: `{fmt(inverted['paired_pp_delta'].get('delta_mean'))}`",
        "",
        "## Support-Val Breakdown",
        "",
        "| dataset | n | enabled | pp delta | MMD delta | closure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in support["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('enabled_fraction'))} | "
            f"{fmt(row.get('delta_pp'))} | {fmt(row.get('delta_mmd_clamped'))} | {fmt(row.get('route_gap_closed_fraction'))} |"
        )
    lines.extend(["", "## Train Selection Top Rows", "", "| spec | enabled | neg rows | min row | pp delta | Norman | Wessels | p_harm |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in payload["train_summaries"][:20]:
        marker = " (selected)" if row["spec"] == payload["selected_spec"] else ""
        lines.append(
            f"| `{row['spec']}`{marker} | {fmt(row.get('enabled_fraction'))} | {row.get('row_negative_count')} | {fmt(row.get('row_min_delta'))} | "
            f"{fmt(row['paired_pp_delta'].get('delta_mean'))} | {fmt(find_dataset(row, 'NormanWeissman2019_filtered').get('delta_pp'))} | "
            f"{fmt(find_dataset(row, 'Wessels').get('delta_pp'))} | {fmt(row['paired_pp_delta'].get('p_harm'))} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in payload["decision"].get("reasons", [])] or ["- none"])
    lines.extend(["", "## Usage Rule", "", "- Passing authorizes only a code/protocol gate, not GPU, held-out query evaluation, or a formal multi-success claim.", ""])
    return "\n".join(lines)


def main() -> int:
    payload_in = load_json(IN_JSON)
    train_rows = payload_in["selected_train_summary"]["rows"]
    support_rows = payload_in["support_val_summary"]["rows"]
    wessels_gap = read_wessels_gap(payload_in)
    train_summaries = []
    scored_train_by_spec = {}
    all_specs = specs()
    for spec in all_specs:
        scored = score_rows(train_rows, spec)
        scored_train_by_spec[spec.name] = scored
        train_summaries.append(summarize(scored, spec, n_boot=2000, seed=20260624, wessels_gap=wessels_gap, include_mmd=False))
    selected = select_spec(train_summaries)
    selected_spec = next(spec for spec in all_specs if spec.name == selected)
    support_scored = score_rows(support_rows, selected_spec)
    support_summary = summarize(support_scored, selected_spec, n_boot=2000, seed=20260625, wessels_gap=wessels_gap, include_mmd=True)
    inverted_spec = invert_dataset_spec(selected_spec)
    inverted_scored = score_rows(support_rows, inverted_spec)
    inverted_summary = summarize(inverted_scored, inverted_spec, n_boot=2000, seed=20260626, wessels_gap=wessels_gap, include_mmd=True)
    ordered_train = sorted(
        train_summaries,
        key=lambda row: (
            -int(row.get("row_negative_count") or 0),
            float(find_dataset(row, "Wessels").get("delta_pp") if find_dataset(row, "Wessels").get("delta_pp") is not None else -999.0),
            float(find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") if find_dataset(row, "NormanWeissman2019_filtered").get("delta_pp") is not None else -999.0),
            float(row["paired_pp_delta"].get("delta_mean") if row["paired_pp_delta"].get("delta_mean") is not None else -999.0),
        ),
        reverse=True,
    )
    payload = {
        "status": "pending",
        "timestamp": "2026-06-24 00:45 CST",
        "input_json": str(IN_JSON),
        "boundary": {
            "safe_trainselect_only": True,
            "train_multi_loo_selection_only": True,
            "support_val_final_scoring_only": True,
            "heldout_query_read": False,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
        },
        "split_guard": payload_in.get("split_guard"),
        "selected_spec": selected,
        "train_summaries": ordered_train,
        "selected_train_rows": scored_train_by_spec[selected],
        "selected_support_summary": support_summary,
        "inverted_dataset_control": inverted_summary,
    }
    payload["decision"] = decide(payload)
    payload["status"] = payload["decision"]["status"]
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
