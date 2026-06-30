#!/usr/bin/env python3
"""Query-free Track C composition gene-risk CPU gate.

This gate is a policy over the existing no-harm calibrated composition rows.
It uses train_multi leave-one-condition-out row outcomes to estimate per-gene
harm history, then blocks partial-coverage corrections when a raw covered gene
has a risky train-only history. support_val_multi is final scoring only.

No held-out query, canonical test, canonical multi, active logs, GPU artifacts,
or new model outputs are read.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
SOURCE_JSON = ROOT / "reports/latentfm_trackc_composition_noharm_calibrated_gate_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_composition_gene_risk_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_COMPOSITION_GENE_RISK_GATE_20260623.md"


@dataclass(frozen=True)
class GeneRiskSpec:
    name: str
    min_gene_delta_threshold: float
    min_gene_observations: int


def specs() -> list[GeneRiskSpec]:
    out = []
    for obs in (1, 2):
        for thr in (-0.20, -0.15, -0.10, -0.05, 0.0):
            out.append(GeneRiskSpec(f"gene_min_ge_{thr:g}_n{obs}", thr, obs))
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def delta(row: dict[str, Any]) -> float:
    return float(row["candidate"]) - float(row["support_selected_route"])


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def build_gene_history(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    by_gene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        d = delta(row)
        for gene in row.get("genes") or []:
            by_gene[str(gene)].append(d)
    return by_gene


def should_block(row: dict[str, Any], history: dict[str, list[float]], spec: GeneRiskSpec) -> tuple[bool, list[dict[str, Any]]]:
    if row.get("coverage_stratum") != "partial_raw":
        return False, []
    risky = []
    # Only covered/raw genes are used for risk because missing genes do not have
    # a gene_raw correction in this policy.
    raw_genes = int(row.get("raw_gene_covered") or 0)
    for gene in (row.get("genes") or [])[:raw_genes]:
        vals = history.get(str(gene), [])
        if len(vals) >= spec.min_gene_observations and float(np.min(vals)) < spec.min_gene_delta_threshold:
            risky.append({"gene": str(gene), "n": len(vals), "min_delta": float(np.min(vals)), "mean_delta": float(np.mean(vals))})
    return bool(risky), risky


def apply_policy(rows: list[dict[str, Any]], histories: list[dict[str, list[float]]] | dict[str, list[float]], spec: GeneRiskSpec) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows):
        history = histories[idx] if isinstance(histories, list) else histories
        blocked, risky = should_block(row, history, spec)
        item = dict(row)
        item["base_candidate"] = row["candidate"]
        item["base_candidate__test_mmd_clamped"] = row.get("candidate__test_mmd_clamped")
        item["gene_risk_blocked"] = blocked
        item["gene_risk_reasons"] = risky
        if blocked:
            item["candidate"] = row["support_selected_route"]
            if "support_selected_route__test_mmd_clamped" in row:
                item["candidate__test_mmd_clamped"] = row["support_selected_route__test_mmd_clamped"]
        out.append(item)
    return out


def train_policy_rows(train_rows: list[dict[str, Any]], spec: GeneRiskSpec) -> list[dict[str, Any]]:
    histories = []
    for row in train_rows:
        fit = [other for other in train_rows if condition_key(other) != condition_key(row)]
        histories.append(build_gene_history(fit))
    return apply_policy(train_rows, histories, spec)


def dataset_delta(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        vals = [delta(row) for row in rows if str(row["dataset"]) == ds]
        out[ds] = float(np.mean(vals)) if vals else 0.0
    return out


def stratum_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for key in sorted({str(row.get("coverage_stratum")) for row in rows}):
        sub = [row for row in rows if str(row.get("coverage_stratum")) == key]
        vals = [delta(row) for row in sub]
        out[key] = {
            "n": len(sub),
            "mean_pp_delta": float(np.mean(vals)) if vals else None,
            "min_pp_delta": float(np.min(vals)) if vals else None,
            "n_negative": int(sum(v < 0 for v in vals)),
            "blocked_fraction": float(np.mean([bool(row.get("gene_risk_blocked")) for row in sub])) if sub else 0.0,
        }
    return out


def mmd_delta(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if not rows or "candidate__test_mmd_clamped" not in rows[0]:
        return None
    vals = [float(row["candidate__test_mmd_clamped"]) - float(row["support_selected_route__test_mmd_clamped"]) for row in rows]
    by_ds = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        dvals = [
            float(row["candidate__test_mmd_clamped"]) - float(row["support_selected_route__test_mmd_clamped"])
            for row in rows
            if str(row["dataset"]) == ds
        ]
        by_ds[ds] = float(np.mean(dvals))
    return {"delta_mean": float(np.mean(list(by_ds.values()))), "dataset_deltas": by_ds}


def bootstrap(rows: list[dict[str, Any]], *, n_boot: int, seed: int, metric: str = "pp") -> dict[str, Any]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if metric == "pp":
            value = delta(row)
        else:
            value = float(row["candidate__test_mmd_clamped"]) - float(row["support_selected_route__test_mmd_clamped"])
        by_ds[str(row["dataset"])].append(value)
    keys = sorted(by_ds)
    point_by_ds = {ds: float(np.mean(by_ds[ds])) for ds in keys}
    point = float(np.mean(list(point_by_ds.values()))) if point_by_ds else 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        vals.append(float(np.mean([np.mean(rng.choice(by_ds[ds], size=len(by_ds[ds]), replace=True)) for ds in sampled])))
    arr = np.asarray(vals)
    return {
        "delta_mean": point,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_improve": float(np.mean(arr > 0)),
        "p_harm": float(np.mean(arr < 0)),
        "dataset_deltas": point_by_ds,
    }


def summarize(rows: list[dict[str, Any]], spec: GeneRiskSpec, *, n_boot: int, seed: int, route_gap: float = 0.15360754709757318) -> dict[str, Any]:
    pp = bootstrap(rows, n_boot=n_boot, seed=seed)
    mmd = bootstrap(rows, n_boot=n_boot, seed=seed + 100, metric="mmd") if rows and "candidate__test_mmd_clamped" in rows[0] else None
    ds_pp = dataset_delta(rows)
    md = mmd_delta(rows)
    breakdown = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        d = ds_pp.get(ds)
        breakdown.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "blocked_fraction": float(np.mean([bool(row.get("gene_risk_blocked")) for row in sub])) if sub else 0.0,
                "delta_pp": d,
                "delta_mmd_clamped": None if md is None else md["dataset_deltas"].get(ds),
                "route_gap_pp": route_gap if ds == "Wessels" else None,
                "route_gap_closed_fraction": None if ds != "Wessels" or d is None else float(d / route_gap),
            }
        )
    return {
        "spec": spec.name,
        "min_gene_delta_threshold": float(spec.min_gene_delta_threshold),
        "min_gene_observations": int(spec.min_gene_observations),
        "paired_pp_delta": pp,
        "paired_mmd_delta": mmd,
        "dataset_breakdown": breakdown,
        "stratum_summary": stratum_summary(rows),
        "blocked_rows": [
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "genes": row.get("genes"),
                "base_delta": float(row["base_candidate"]) - float(row["support_selected_route"]),
                "risk": row.get("gene_risk_reasons"),
            }
            for row in rows
            if row.get("gene_risk_blocked")
        ],
        "rows": rows,
    }


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def train_reasons(summary: dict[str, Any]) -> list[str]:
    reasons = []
    pp = summary["paired_pp_delta"]
    w = find_dataset(summary, "Wessels")
    n = find_dataset(summary, "NormanWeissman2019_filtered")
    partial = (summary.get("stratum_summary") or {}).get("partial_raw") or {}
    if pp["p_harm"] > 0.10:
        reasons.append("train_pp_harm_above_0p10")
    if float(n.get("delta_pp") or -999.0) < 0.0:
        reasons.append("train_norman_delta_below_0")
    if float(w.get("route_gap_closed_fraction") or -999.0) < 0.30:
        reasons.append("train_wessels_closure_below_0p30")
    if partial and float(partial.get("mean_pp_delta") if partial.get("mean_pp_delta") is not None else -999.0) < -0.005:
        reasons.append("train_partial_raw_mean_below_minus_0p005")
    return reasons


def support_reasons(summary: dict[str, Any]) -> list[str]:
    reasons = []
    pp = summary["paired_pp_delta"]
    mmd = summary.get("paired_mmd_delta") or {}
    w = find_dataset(summary, "Wessels")
    n = find_dataset(summary, "NormanWeissman2019_filtered")
    if float(w.get("route_gap_closed_fraction") or -999.0) < 0.30 and float(w.get("delta_pp") or -999.0) < 0.05:
        reasons.append("support_wessels_signal_below_gate")
    if float(n.get("delta_pp") or -999.0) < -0.01:
        reasons.append("support_norman_delta_below_minus_0p01")
    if pp["p_harm"] > 0.20:
        reasons.append("support_pp_harm_above_0p20")
    if mmd and mmd["delta_mean"] > 0.005:
        reasons.append("support_mmd_delta_above_0p005")
    if mmd and mmd["p_harm"] > 0.80:
        reasons.append("support_mmd_harm_above_0p80")
    return reasons


def select_train(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [s for s in summaries if not s["train_reasons"]]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda s: (
            float(find_dataset(s, "Wessels").get("route_gap_closed_fraction") or -999.0),
            -float(s["paired_pp_delta"]["p_harm"]),
            float(s["paired_pp_delta"]["delta_mean"]),
        ),
        reverse=True,
    )[0]


def build_payload() -> dict[str, Any]:
    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    train_base = source["selected_train_summary"]["rows"]
    support_base = source["support_val_summary"]["rows"]
    train_summaries = []
    for i, spec in enumerate(specs()):
        rows = train_policy_rows(train_base, spec)
        summary = summarize(rows, spec, n_boot=2000, seed=20260623 + i)
        summary["train_reasons"] = train_reasons(summary)
        train_summaries.append(summary)
    selected = select_train(train_summaries)
    support_summary = zero = shuffled = None
    reasons: list[str]
    if selected is None:
        reasons = ["no_spec_passed_train_gene_risk_gate"]
    else:
        spec = next(s for s in specs() if s.name == selected["spec"])
        history = build_gene_history(train_base)
        support_summary = summarize(apply_policy(support_base, history, spec), spec, n_boot=2000, seed=20261623)
        support_summary["support_reasons"] = support_reasons(support_summary)
        zero_spec = GeneRiskSpec(f"{spec.name}_zero_control", -999.0, 999)
        zero_rows = []
        for row in support_base:
            item = dict(row)
            item["base_candidate"] = item["candidate"]
            item["candidate"] = item["support_selected_route"]
            if "support_selected_route__test_mmd_clamped" in item:
                item["candidate__test_mmd_clamped"] = item["support_selected_route__test_mmd_clamped"]
            item["gene_risk_blocked"] = True
            item["gene_risk_reasons"] = []
            zero_rows.append(item)
        zero = summarize(zero_rows, zero_spec, n_boot=2000, seed=20261624)
        zero["support_reasons"] = support_reasons(zero)
        # The calibrated source already contains a shuffled control with the same
        # base policy; applying gene-risk to it would use real gene names on a
        # shuffled bank. Keep it as the negative control boundary.
        shuffled = source["shuffled_gene_bank_control"]
        reasons = list(support_summary["support_reasons"])
        if zero and not zero["support_reasons"]:
            reasons.append("zero_control_passed_unexpectedly")
        if shuffled and not shuffled.get("support_reasons"):
            reasons.append("shuffled_control_passed_unexpectedly")
    status = "trackc_composition_gene_risk_gate_pass_posthoc_mmd_gate_next_no_gpu" if not reasons else "trackc_composition_gene_risk_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "query_free_posthoc_mmd_gate_only" if not reasons else "none",
        "reasons": reasons,
        "boundary": {
            "source_json": str(SOURCE_JSON),
            "safe_trainselect_existing_rows_only": True,
            "train_multi_loo_gene_risk_selection_only": True,
            "support_val_final_scoring_only": True,
            "heldout_query_read": False,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
        },
        "source_status": source["status"],
        "train_summaries": train_summaries,
        "selected_train_summary": selected,
        "support_val_summary": support_summary,
        "zero_control": zero,
        "shuffled_gene_bank_control": shuffled,
    }


def table(summary: dict[str, Any] | None, reason_key: str) -> list[str]:
    if not summary:
        return ["- not evaluated", ""]
    pp = summary["paired_pp_delta"]
    mmd = summary.get("paired_mmd_delta")
    lines = [
        f"- spec: `{summary['spec']}`",
        f"- paired pp delta: `{fmt(pp['delta_mean'])}`",
        f"- paired pp p_harm: `{fmt(pp['p_harm'])}`",
        f"- paired MMD delta: `{fmt(mmd['delta_mean']) if mmd else 'NA'}`",
        f"- paired MMD p_harm: `{fmt(mmd['p_harm']) if mmd else 'NA'}`",
        f"- reasons: `{', '.join(summary.get(reason_key) or []) or 'none'}`",
        "",
        "| dataset | n | blocked | delta pp | closure | delta MMD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row['blocked_fraction'])} | "
            f"{fmt(row['delta_pp'])} | {fmt(row.get('route_gap_closed_fraction'))} | {fmt(row.get('delta_mmd_clamped'))} |"
        )
    lines.extend(["", "| stratum | n | mean pp delta | min | n negative | blocked |", "|---|---:|---:|---:|---:|---:|"])
    for key, row in summary["stratum_summary"].items():
        lines.append(f"| `{key}` | {row['n']} | {fmt(row['mean_pp_delta'])} | {fmt(row['min_pp_delta'])} | {row['n_negative']} | {fmt(row['blocked_fraction'])} |")
    if summary.get("blocked_rows"):
        lines.extend(["", "Blocked rows:"])
        for row in summary["blocked_rows"][:12]:
            lines.append(f"- `{row['dataset']}::{row['condition']}` base_delta={fmt(row['base_delta'])} risk={row['risk']}")
    lines.append("")
    return lines


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Composition Gene-Risk Gate",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        f"Next authorization: `{payload['next_authorization']}`",
        "",
        "## Boundary",
        "",
        "- Uses existing no-harm calibrated safe-trainselect rows only.",
        "- Train_multi leave-one-condition-out gene-risk selection; support_val final scoring only.",
        "- No held-out query, canonical test, canonical multi, active logs, GPU artifacts, or new model outputs are read.",
        f"- source JSON: `{payload['boundary']['source_json']}`",
        "",
        "## Selected Train Spec",
        "",
    ]
    lines.extend(table(payload.get("selected_train_summary"), "train_reasons") if payload.get("selected_train_summary") else ["- none", ""])
    for title, key, reason_key in (
        ("Support-Val Summary", "support_val_summary", "support_reasons"),
        ("Zero Control", "zero_control", "support_reasons"),
    ):
        lines.extend([f"## {title}", ""])
        lines.extend(table(payload.get(key), reason_key))
    lines.extend(["## Decision Reasons", ""])
    lines.extend([f"- `{reason}`" for reason in payload.get("reasons") or []] or ["- none"])
    lines.extend(["", "## Interpretation", "", "This gate tests whether train-only gene-level harm history can identify partial-coverage tail risk without using support_val outcomes for selection.", ""])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
