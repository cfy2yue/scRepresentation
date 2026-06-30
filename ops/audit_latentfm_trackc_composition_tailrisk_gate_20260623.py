#!/usr/bin/env python3
"""Query-free Track C composition tail-risk CPU gate.

This gate is a materially different follow-up to the route-share additive
near-miss: it does not merely tune beta.  It uses train_multi
leave-one-condition-out scoring to select a partial-coverage risk filter based
on predeclared correction geometry (correction/route norm ratio and
additive-route cosine).  support_val_multi is final scoring only.

Held-out query, canonical test, canonical multi, active logs, and GPU artifacts
are not read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RESIDUAL_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_residual_operator_cpu_gate_20260623.py"
OUT_JSON = ROOT / "reports/latentfm_trackc_composition_tailrisk_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_COMPOSITION_TAILRISK_GATE_20260623.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"


@dataclass(frozen=True)
class TailRiskSpec:
    name: str
    beta_full: float
    beta_partial: float
    max_partial_norm_ratio: float
    min_partial_cosine: float


def load_residual_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_residual_operator_gate", RESIDUAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {RESIDUAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def specs() -> list[TailRiskSpec]:
    out = []
    for beta_full in (0.50, 0.75):
        for beta_partial in (0.25, 0.50, 0.75):
            for ratio in (0.50, 1.00, 1.50, 2.00, 3.00, math.inf):
                for cosine in (-1.00, -0.50, 0.00, 0.25, 0.50):
                    rlabel = "inf" if math.isinf(ratio) else f"{ratio:g}"
                    out.append(TailRiskSpec(f"tailrisk_full{beta_full:g}_partial{beta_partial:g}_r{rlabel}_c{cosine:g}", beta_full, beta_partial, ratio, cosine))
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def genes(row: dict[str, Any]) -> list[str]:
    return [str(g).strip().upper() for g in (row.get("genes") or []) if str(g).strip()]


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 3:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if denom <= 1e-12 else float(np.dot(x, y) / denom)


def route_vector(support: Any, row: dict[str, Any], single: dict[str, Any], multi: dict[str, Any]) -> np.ndarray:
    return np.asarray(support.predict_baselines(row, single, multi)["support_selected_route"], dtype=np.float32)


def shuffled_bank(bank: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    keys = sorted(bank)
    vals = [bank[key] for key in keys]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    return {key: vals[int(order[i])] for i, key in enumerate(keys)}


def build_additive(
    row: dict[str, Any],
    route: np.ndarray,
    single: dict[str, Any],
    *,
    bank_override: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, int, int, int, str]:
    bank = bank_override or single.get("gene_raw_mean") or {}
    gs = genes(row)
    parts = []
    raw = 0
    fallback = 0
    for gene in gs:
        value = bank.get(gene)
        if value is None:
            parts.append(np.asarray(route, dtype=np.float32) / max(len(gs), 1))
            fallback += 1
        else:
            parts.append(np.asarray(value, dtype=np.float32))
            raw += 1
    if not parts:
        return np.asarray(route, dtype=np.float32), 0, 0, 0, "empty"
    if raw == len(gs):
        stratum = "full_raw"
    elif raw == 0:
        stratum = "zero_raw"
    else:
        stratum = "partial_raw"
    return np.sum(np.stack(parts, axis=0), axis=0).astype(np.float32), raw, fallback, len(gs), stratum


def geometry(route: np.ndarray, add: np.ndarray) -> dict[str, float]:
    correction = np.asarray(add, dtype=np.float32) - np.asarray(route, dtype=np.float32)
    return {
        "correction_norm_ratio": float(np.linalg.norm(correction) / max(np.linalg.norm(route), 1e-8)),
        "additive_route_cosine": pearson(add, route),
    }


def beta_for(spec: TailRiskSpec, stratum: str, geom: dict[str, float]) -> tuple[float, bool]:
    if stratum == "full_raw":
        return float(spec.beta_full), False
    if stratum == "partial_raw":
        blocked = geom["correction_norm_ratio"] > spec.max_partial_norm_ratio or geom["additive_route_cosine"] < spec.min_partial_cosine
        return (0.0 if blocked else float(spec.beta_partial)), blocked
    return 0.0, False


def score_row(
    support: Any,
    row: dict[str, Any],
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    spec: TailRiskSpec,
    *,
    bank_override: dict[str, np.ndarray] | None = None,
    compute_mmd: bool,
) -> dict[str, Any]:
    route = route_vector(support, row, single, multi)
    add, raw, fallback, total, stratum = build_additive(row, route, single, bank_override=bank_override)
    geom = geometry(route, add)
    beta, risk_blocked = beta_for(spec, stratum, geom)
    pred = route + beta * (add - route)
    out = {
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "genes": genes(row),
        "candidate": support.pp_score(row, pred, pert_means),
        "support_selected_route": support.pp_score(row, route, pert_means),
        "raw_gene_covered": int(raw),
        "fallback_genes": int(fallback),
        "total_genes": int(total),
        "coverage_stratum": stratum,
        "covered": bool(total > 0 and raw + fallback == total),
        "beta_used": float(beta),
        "risk_blocked": bool(risk_blocked),
        **geom,
    }
    if compute_mmd:
        for metric, value in support.mmd_scores(row, pred).items():
            out[f"candidate__{metric}"] = value
        for metric, value in support.mmd_scores(row, route).items():
            out[f"support_selected_route__{metric}"] = value
    return out


def score_rows(
    support: Any,
    rows: list[dict[str, Any]],
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    spec: TailRiskSpec,
    *,
    bank_override: dict[str, np.ndarray] | None = None,
    compute_mmd: bool,
) -> list[dict[str, Any]]:
    return [
        score_row(support, row, single, multi, pert_means, spec, bank_override=bank_override, compute_mmd=compute_mmd)
        for row in rows
    ]


def train_loo_rows(
    support: Any,
    train_multi: list[dict[str, Any]],
    single: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    spec: TailRiskSpec,
) -> list[dict[str, Any]]:
    out = []
    for row in train_multi:
        fit_rows = [other for other in train_multi if condition_key(other) != condition_key(row)]
        multi = support.train_multi_components(fit_rows)
        out.append(score_row(support, row, single, multi, pert_means, spec, compute_mmd=False))
    return out


def dataset_delta(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        vals = [float(row["candidate"]) - float(row["support_selected_route"]) for row in rows if str(row["dataset"]) == ds]
        if vals:
            out[ds] = float(np.mean(vals))
    return out


def dataset_mmd_delta(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows or "candidate__test_mmd_clamped" not in rows[0]:
        return {}
    out = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        vals = [
            float(row["candidate__test_mmd_clamped"]) - float(row["support_selected_route__test_mmd_clamped"])
            for row in rows
            if str(row["dataset"]) == ds
        ]
        if vals:
            out[ds] = float(np.mean(vals))
    return out


def stratum_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for key in sorted({str(row.get("coverage_stratum")) for row in rows}):
        sub = [row for row in rows if str(row.get("coverage_stratum")) == key]
        vals = [float(row["candidate"]) - float(row["support_selected_route"]) for row in sub]
        out[key] = {
            "n": len(sub),
            "mean_pp_delta": float(np.mean(vals)) if vals else None,
            "min_pp_delta": float(np.min(vals)) if vals else None,
            "n_negative": int(sum(v < 0 for v in vals)),
            "blocked_fraction": float(np.mean([bool(row.get("risk_blocked")) for row in sub])) if sub else 0.0,
        }
    return out


def summarize(
    mod: Any,
    rows: list[dict[str, Any]],
    spec: TailRiskSpec,
    *,
    n_boot: int,
    seed: int,
    wessels_route_gap: float | None,
    include_mmd: bool,
) -> dict[str, Any]:
    pp = mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd = (
        mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 100)
        if include_mmd
        else None
    )
    ds_pp = dataset_delta(rows)
    ds_mmd = dataset_mmd_delta(rows)
    breakdown = []
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        delta = ds_pp.get(ds)
        gap = wessels_route_gap if ds == "Wessels" else None
        breakdown.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "coverage_fraction": float(np.mean([bool(row.get("covered")) for row in sub])) if sub else 0.0,
                "blocked_fraction": float(np.mean([bool(row.get("risk_blocked")) for row in sub])) if sub else 0.0,
                "delta_pp": delta,
                "delta_mmd_clamped": ds_mmd.get(ds),
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or delta is None or abs(gap) <= 1e-12 else float(delta / gap),
            }
        )
    return {
        "spec": spec.name,
        "beta_full": float(spec.beta_full),
        "beta_partial": float(spec.beta_partial),
        "max_partial_norm_ratio": None if math.isinf(spec.max_partial_norm_ratio) else float(spec.max_partial_norm_ratio),
        "min_partial_cosine": float(spec.min_partial_cosine),
        "paired_pp_delta": pp,
        "paired_mmd_delta": mmd,
        "dataset_breakdown": breakdown,
        "stratum_summary": stratum_summary(rows),
        "rows": rows,
    }


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def train_reasons(summary: dict[str, Any]) -> list[str]:
    reasons = []
    w = find_dataset(summary, "Wessels")
    n = find_dataset(summary, "NormanWeissman2019_filtered")
    pp = summary.get("paired_pp_delta") or {}
    partial = (summary.get("stratum_summary") or {}).get("partial_raw") or {}
    if float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) > 0.10:
        reasons.append("train_pp_harm_above_0p10")
    if float(w.get("delta_pp") if w.get("delta_pp") is not None else -999.0) < 0.02:
        reasons.append("train_wessels_delta_below_0p02")
    if float(w.get("route_gap_closed_fraction") if w.get("route_gap_closed_fraction") is not None else -999.0) < 0.05:
        reasons.append("train_wessels_closure_below_0p05")
    if float(n.get("delta_pp") if n.get("delta_pp") is not None else -999.0) < 0.0:
        reasons.append("train_norman_delta_below_0")
    if partial and float(partial.get("mean_pp_delta") if partial.get("mean_pp_delta") is not None else -999.0) < -0.01:
        reasons.append("train_partial_raw_stratum_delta_below_minus_0p01")
    if partial and float(partial.get("min_pp_delta") if partial.get("min_pp_delta") is not None else 0.0) < -0.20:
        reasons.append("train_partial_raw_tail_below_minus_0p20")
    return reasons


def support_reasons(summary: dict[str, Any]) -> list[str]:
    reasons = []
    w = find_dataset(summary, "Wessels")
    n = find_dataset(summary, "NormanWeissman2019_filtered")
    pp = summary.get("paired_pp_delta") or {}
    mmd = summary.get("paired_mmd_delta") or {}
    partial = (summary.get("stratum_summary") or {}).get("partial_raw") or {}
    all_cov = sum(row.get("coverage_fraction", 0.0) * row.get("n_conditions", 0) for row in summary.get("dataset_breakdown") or [])
    all_n = sum(row.get("n_conditions", 0) for row in summary.get("dataset_breakdown") or [])
    if all_cov / max(all_n, 1) < 0.999:
        reasons.append("support_coverage_below_1")
    if float(w.get("delta_pp") if w.get("delta_pp") is not None else -999.0) < 0.05 and float(
        w.get("route_gap_closed_fraction") if w.get("route_gap_closed_fraction") is not None else -999.0
    ) < 0.30:
        reasons.append("support_wessels_signal_below_gate")
    if float(n.get("delta_pp") if n.get("delta_pp") is not None else -999.0) < -0.01:
        reasons.append("support_norman_delta_below_minus_0p01")
    if float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("support_pp_harm_above_0p20")
    if float(mmd.get("delta_mean") if mmd and mmd.get("delta_mean") is not None else 999.0) > 0.005:
        reasons.append("support_mmd_delta_above_0p005")
    if float(mmd.get("p_harm") if mmd and mmd.get("p_harm") is not None else 1.0) > 0.80:
        reasons.append("support_mmd_harm_above_0p80")
    if partial and float(partial.get("mean_pp_delta") if partial.get("mean_pp_delta") is not None else -999.0) < -0.02:
        reasons.append("support_partial_raw_stratum_delta_below_minus_0p02")
    return reasons


def select_train_spec(train_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in train_summaries if not item.get("train_reasons")]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda item: (
            float((find_dataset(item, "Wessels") or {}).get("route_gap_closed_fraction") or -999.0),
            float((find_dataset(item, "Wessels") or {}).get("delta_pp") or -999.0),
            -float((item.get("paired_pp_delta") or {}).get("p_harm") or 1.0),
            float((item.get("paired_pp_delta") or {}).get("delta_mean") or -999.0),
        ),
        reverse=True,
    )[0]


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    mod = load_residual_module()
    support = mod.load_support_module()
    split = support.load_json(args.split_file)
    guard = mod.split_guard(args.split_file, split)
    if guard["sha256"] != EXPECTED_TRAINSELECT_SHA256:
        raise RuntimeError(f"unexpected trainselect split hash: {guard['sha256']}")
    manifest = support.load_json(args.data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {key: value.astype(np.float32) for key, value in np.load(args.pert_means).items()}
    train_multi = support.collect_role_rows(args.data_dir, split, metadata, "train_multi", max_cells=args.max_cells)
    support_val = support.collect_role_rows(args.data_dir, split, metadata, "support_val_multi", max_cells=args.max_cells)
    single = support.train_single_components(args.data_dir, split, metadata, max_cells=args.max_cells)
    multi = support.train_multi_components(train_multi)
    route_gap = mod.readout_wessels_route_gap(mod.DEFAULT_READOUT_JSON)

    train_summaries = []
    for idx, spec in enumerate(specs()):
        rows = train_loo_rows(support, train_multi, single, pert_means, spec)
        summary = summarize(mod, rows, spec, n_boot=args.n_boot, seed=args.seed + idx, wessels_route_gap=route_gap, include_mmd=False)
        summary["train_reasons"] = train_reasons(summary)
        train_summaries.append(summary)
    selected = select_train_spec(train_summaries)
    support_summary = zero = shuffled = None
    if selected is not None:
        spec = next(item for item in specs() if item.name == selected["spec"])
        support_summary = summarize(
            mod,
            score_rows(support, support_val, single, multi, pert_means, spec, compute_mmd=True),
            spec,
            n_boot=args.n_boot,
            seed=args.seed + 1000,
            wessels_route_gap=route_gap,
            include_mmd=True,
        )
        support_summary["support_reasons"] = support_reasons(support_summary)
        zero_spec = TailRiskSpec(f"{spec.name}_zero_beta_control", 0.0, 0.0, 0.0, 1.0)
        zero = summarize(
            mod,
            score_rows(support, support_val, single, multi, pert_means, zero_spec, compute_mmd=True),
            zero_spec,
            n_boot=args.n_boot,
            seed=args.seed + 1001,
            wessels_route_gap=route_gap,
            include_mmd=True,
        )
        zero["support_reasons"] = support_reasons(zero)
        shuf_bank = shuffled_bank(single.get("gene_raw_mean") or {}, args.seed + 1002)
        shuffled = summarize(
            mod,
            score_rows(support, support_val, single, multi, pert_means, spec, bank_override=shuf_bank, compute_mmd=True),
            spec,
            n_boot=args.n_boot,
            seed=args.seed + 1003,
            wessels_route_gap=route_gap,
            include_mmd=True,
        )
        shuffled["support_reasons"] = support_reasons(shuffled)

    if selected is None:
        reasons = ["no_spec_passed_train_tailrisk_gate"]
    else:
        reasons = list(support_summary.get("support_reasons") or [])
        if zero and not zero.get("support_reasons"):
            reasons.append("zero_beta_control_passed_unexpectedly")
        if shuffled and not shuffled.get("support_reasons"):
            reasons.append("shuffled_gene_bank_control_passed_unexpectedly")
    status = "trackc_composition_tailrisk_gate_pass_posthoc_mmd_gate_next_no_gpu" if not reasons else "trackc_composition_tailrisk_gate_fail_no_gpu"
    global_single_mean = single.get("global_single_mean")
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "query_free_posthoc_mmd_gate_only" if not reasons else "none",
        "reasons": reasons,
        "boundary": {
            "safe_trainselect_only": True,
            "train_multi_loo_selection_only": True,
            "support_val_final_scoring_only": True,
            "heldout_query_read": False,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
            "python": sys.executable,
        },
        "inputs": {
            "data_dir": str(args.data_dir),
            "split_file": str(args.split_file),
            "pert_means": str(args.pert_means),
            "residual_module": str(RESIDUAL_MODULE_PATH),
        },
        "split_guard": guard,
        "n_rows": {"train_multi": len(train_multi), "support_val_multi": len(support_val)},
        "single_bank_summary": {
            "gene_raw_mean_genes": len(single.get("gene_raw_mean") or {}),
            "dataset_single_mean_entries": len(single.get("dataset_single_mean") or {}),
            "global_single_mean_dim": int(len(global_single_mean)) if global_single_mean is not None else 0,
        },
        "train_summaries": train_summaries,
        "selected_train_summary": selected,
        "support_val_summary": support_summary,
        "zero_beta_control": zero,
        "shuffled_gene_bank_control": shuffled,
    }


def table(summary: dict[str, Any] | None, reason_key: str) -> list[str]:
    if not summary:
        return ["- not evaluated", ""]
    pp = summary.get("paired_pp_delta") or {}
    mmd = summary.get("paired_mmd_delta") or {}
    lines = [
        f"- spec: `{summary['spec']}`",
        f"- beta full/partial: `{fmt(summary.get('beta_full'))}` / `{fmt(summary.get('beta_partial'))}`",
        f"- max partial norm ratio: `{summary.get('max_partial_norm_ratio')}`",
        f"- min partial cosine: `{fmt(summary.get('min_partial_cosine'))}`",
        f"- paired pp delta: `{fmt(pp.get('delta_mean'))}`",
        f"- paired pp p_harm: `{fmt(pp.get('p_harm'))}`",
        f"- paired MMD delta: `{fmt(mmd.get('delta_mean')) if mmd else 'NA'}`",
        f"- paired MMD p_harm: `{fmt(mmd.get('p_harm')) if mmd else 'NA'}`",
        f"- reasons: `{', '.join(summary.get(reason_key) or []) or 'none'}`",
        "",
        "| dataset | n | coverage | blocked | delta pp | route gap | closure | delta MMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("dataset_breakdown") or []:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('coverage_fraction'))} | "
            f"{fmt(row.get('blocked_fraction'))} | {fmt(row.get('delta_pp'))} | {fmt(row.get('route_gap_pp'))} | "
            f"{fmt(row.get('route_gap_closed_fraction'))} | {fmt(row.get('delta_mmd_clamped'))} |"
        )
    lines.extend(["", "| stratum | n | mean pp delta | min pp delta | n negative | blocked |", "|---|---:|---:|---:|---:|---:|"])
    for key, row in (summary.get("stratum_summary") or {}).items():
        lines.append(
            f"| `{key}` | {row['n']} | {fmt(row.get('mean_pp_delta'))} | {fmt(row.get('min_pp_delta'))} | "
            f"{row['n_negative']} | {fmt(row.get('blocked_fraction'))} |"
        )
    lines.append("")
    return lines


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Composition Tail-Risk Gate",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        f"Next authorization: `{payload['next_authorization']}`",
        "",
        "## Scope",
        "",
        "This query-free CPU gate selects a partial-coverage correction risk filter using train_multi leave-one-condition-out scoring. support_val_multi is final scoring only.",
        "",
        "## Boundary",
        "",
        "- safe trainselect split only",
        "- train_multi leave-one-condition-out selection only",
        "- held-out query, canonical test, canonical multi, active logs, and GPU artifacts are not read",
        "- passing would still authorize only a later query-free MMD/no-harm posthoc gate, not GPU training or query",
        f"- python: `{payload['boundary']['python']}`",
        "",
        "## Inputs",
        "",
        f"- train_multi rows: `{payload['n_rows']['train_multi']}`",
        f"- support_val_multi rows: `{payload['n_rows']['support_val_multi']}`",
        f"- gene_raw_mean genes: `{payload['single_bank_summary']['gene_raw_mean_genes']}`",
        f"- split SHA256: `{payload['split_guard']['sha256']}`",
        "",
        "## Selected Train-Multi LOO Spec",
        "",
    ]
    lines.extend(table(payload.get("selected_train_summary"), "train_reasons") if payload.get("selected_train_summary") else ["- none", ""])
    for title, key in (
        ("Support-Val Summary", "support_val_summary"),
        ("Zero-Beta Control", "zero_beta_control"),
        ("Shuffled-Gene-Bank Control", "shuffled_gene_bank_control"),
    ):
        lines.extend([f"## {title}", ""])
        lines.extend(table(payload.get(key), "support_reasons"))
    lines.extend(["## Decision Reasons", ""])
    lines.extend([f"- `{reason}`" for reason in payload.get("reasons") or []] or ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This gate tests whether predeclared train-only correction geometry can control the partial-coverage harm tail. It must not use support_val to tune thresholds.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    mod = load_residual_module()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=mod.DEFAULT_DATA_DIR)
    parser.add_argument("--split-file", type=Path, default=mod.DEFAULT_SPLIT)
    parser.add_argument("--pert-means", type=Path, default=mod.DEFAULT_PERT_MEANS)
    parser.add_argument("--max-cells", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
