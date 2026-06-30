#!/usr/bin/env python3
"""Query-free Track C composition hybrid-prior CPU gate.

This follow-up asks whether the narrow Wessels-local signal from the
train-single gene_raw additive prior survives when missing component genes are
filled by train-only fallback priors. Beta and fallback mode are selected on
train_multi only; support_val_multi is final scoring only.

Held-out query, canonical test, canonical multi, active logs, and GPU artifacts
are not read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RESIDUAL_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_residual_operator_cpu_gate_20260623.py"
OUT_JSON = ROOT / "reports/latentfm_trackc_composition_hybrid_prior_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_COMPOSITION_HYBRID_PRIOR_GATE_20260623.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"


@dataclass(frozen=True)
class HybridSpec:
    name: str
    beta: float
    fallback: str


def load_residual_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_residual_operator_gate", RESIDUAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {RESIDUAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def specs() -> list[HybridSpec]:
    out = []
    for fallback in ("dataset_single", "global_single", "route_share", "zero"):
        for beta in (0.25, 0.50, 0.75, 1.00):
            out.append(HybridSpec(f"hybrid_{fallback}_beta{beta:g}", beta, fallback))
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


def route_vector(support: Any, row: dict[str, Any], single: dict[str, Any], multi: dict[str, Any]) -> np.ndarray:
    return np.asarray(support.predict_baselines(row, single, multi)["support_selected_route"], dtype=np.float32)


def shuffled_bank(bank: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    keys = sorted(bank)
    vals = [bank[key] for key in keys]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    return {key: vals[int(order[i])] for i, key in enumerate(keys)}


def fallback_vector(row: dict[str, Any], route: np.ndarray, single: dict[str, Any], spec: HybridSpec, total: int) -> np.ndarray:
    ds = str(row["dataset"])
    if spec.fallback == "dataset_single":
        return np.asarray(single["dataset_single_mean"].get(ds, single["global_single_mean"]), dtype=np.float32)
    if spec.fallback == "global_single":
        return np.asarray(single["global_single_mean"], dtype=np.float32)
    if spec.fallback == "route_share":
        return np.asarray(route, dtype=np.float32) / max(total, 1)
    if spec.fallback == "zero":
        return np.zeros_like(route, dtype=np.float32)
    raise ValueError(spec.fallback)


def hybrid_vector(
    row: dict[str, Any],
    route: np.ndarray,
    single: dict[str, Any],
    spec: HybridSpec,
    *,
    bank_override: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray | None, int, int, int]:
    bank = bank_override or single.get("gene_raw_mean") or {}
    gs = genes(row)
    if not gs:
        return None, 0, 0, 0
    found = 0
    fallback_used = 0
    parts = []
    for gene in gs:
        value = bank.get(gene)
        if value is None:
            parts.append(fallback_vector(row, route, single, spec, len(gs)))
            fallback_used += 1
        else:
            parts.append(np.asarray(value, dtype=np.float32))
            found += 1
    return np.sum(np.stack(parts, axis=0), axis=0).astype(np.float32), found, fallback_used, len(gs)


def score_rows(
    mod: Any,
    support: Any,
    rows: list[dict[str, Any]],
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    spec: HybridSpec,
    *,
    bank_override: dict[str, np.ndarray] | None = None,
    compute_mmd: bool = True,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        route = route_vector(support, row, single, multi)
        add, found, fallback_used, total = hybrid_vector(row, route, single, spec, bank_override=bank_override)
        pred = route if add is None else route + float(spec.beta) * (add - route)
        item = {
            "dataset": str(row["dataset"]),
            "condition": str(row["condition"]),
            "genes": genes(row),
            "covered_genes": int(found + fallback_used),
            "raw_gene_covered": int(found),
            "fallback_genes": int(fallback_used),
            "total_genes": int(total),
            "covered": add is not None and int(found + fallback_used) == int(total),
            "raw_coverage_fraction": float(found / max(total, 1)),
            "candidate": support.pp_score(row, pred, pert_means),
            "support_selected_route": support.pp_score(row, route, pert_means),
        }
        if compute_mmd:
            for metric, value in support.mmd_scores(row, pred).items():
                item[f"candidate__{metric}"] = value
            for metric, value in support.mmd_scores(row, route).items():
                item[f"support_selected_route__{metric}"] = value
        out.append(item)
    return out


def dataset_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        out[ds] = {
            "n_conditions": len(sub),
            "covered_conditions": int(sum(bool(row.get("covered")) for row in sub)),
            "coverage_fraction": float(np.mean([bool(row.get("covered")) for row in sub])) if sub else 0.0,
            "raw_gene_coverage_fraction": float(np.mean([float(row.get("raw_coverage_fraction") or 0.0) for row in sub])) if sub else 0.0,
            "fallback_gene_fraction": float(
                sum(int(row.get("fallback_genes") or 0) for row in sub)
                / max(sum(int(row.get("total_genes") or 0) for row in sub), 1)
            ),
        }
    return out


def dataset_delta(rows: list[dict[str, Any]], candidate: str, baseline: str) -> dict[str, float]:
    out = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        vals = [float(row[candidate]) - float(row[baseline]) for row in rows if str(row["dataset"]) == ds]
        if vals:
            out[ds] = float(np.mean(vals))
    return out


def summarize(
    mod: Any,
    rows: list[dict[str, Any]],
    spec: HybridSpec,
    *,
    n_boot: int,
    seed: int,
    wessels_route_gap: float | None,
) -> dict[str, Any]:
    pp = mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd = mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 100)
    cov = dataset_coverage(rows)
    ds_pp = dataset_delta(rows, "candidate", "support_selected_route")
    ds_mmd = dataset_delta(rows, "candidate__test_mmd_clamped", "support_selected_route__test_mmd_clamped")
    breakdown = []
    for ds in sorted(cov):
        delta = ds_pp.get(ds)
        gap = wessels_route_gap if ds == "Wessels" else None
        breakdown.append(
            {
                "dataset": ds,
                **cov[ds],
                "delta_pp": delta,
                "delta_mmd_clamped": ds_mmd.get(ds),
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or delta is None or abs(gap) <= 1e-12 else float(delta / gap),
            }
        )
    return {
        "spec": spec.name,
        "beta": float(spec.beta),
        "fallback": spec.fallback,
        "paired_pp_delta": pp,
        "paired_mmd_delta": mmd,
        "dataset_breakdown": breakdown,
        "rows": rows,
    }


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_breakdown") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def gate_reasons(summary: dict[str, Any], *, require_coverage: bool = True) -> list[str]:
    reasons = []
    w = find_dataset(summary, "Wessels")
    n = find_dataset(summary, "NormanWeissman2019_filtered")
    pp = summary.get("paired_pp_delta") or {}
    mmd = summary.get("paired_mmd_delta") or {}
    if float(w.get("delta_pp") if w.get("delta_pp") is not None else -999.0) < 0.02:
        reasons.append("wessels_delta_below_0p02")
    if float(w.get("route_gap_closed_fraction") if w.get("route_gap_closed_fraction") is not None else -999.0) < 0.05:
        reasons.append("wessels_route_gap_closure_below_0p05")
    if float(n.get("delta_pp") if n.get("delta_pp") is not None else -999.0) < -0.02:
        reasons.append("norman_material_pp_loss")
    if float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("bootstrap_pp_harm_above_0p20")
    if float(mmd.get("delta_mean") if mmd.get("delta_mean") is not None else 999.0) > 0.005:
        reasons.append("mmd_delta_hard_harm_above_0p005")
    if float(mmd.get("p_harm") if mmd.get("p_harm") is not None else 1.0) > 0.80:
        reasons.append("mmd_harm_probability_above_0p80")
    if require_coverage:
        all_cov = sum(row.get("covered_conditions", 0) for row in summary.get("dataset_breakdown") or [])
        all_n = sum(row.get("n_conditions", 0) for row in summary.get("dataset_breakdown") or [])
        all_frac = all_cov / max(all_n, 1)
        if all_frac < 0.80:
            reasons.append("overall_support_coverage_below_0p80")
        if float(w.get("coverage_fraction") or 0.0) < 0.80:
            reasons.append("wessels_support_coverage_below_0p80")
        if float(n.get("coverage_fraction") or 0.0) < 0.50:
            reasons.append("norman_support_coverage_below_0p50")
    return reasons


def select_train_spec(train_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in train_summaries if not gate_reasons(item, require_coverage=False)]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda item: (
            float((find_dataset(item, "Wessels") or {}).get("route_gap_closed_fraction") or -999.0),
            float((find_dataset(item, "Wessels") or {}).get("delta_pp") or -999.0),
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
        rows = score_rows(mod, support, train_multi, single, multi, pert_means, spec, compute_mmd=True)
        train_summaries.append(summarize(mod, rows, spec, n_boot=args.n_boot, seed=args.seed + idx, wessels_route_gap=route_gap))
    selected = select_train_spec(train_summaries)
    support_summary = zero = shuffled = None
    if selected is not None:
        spec = next(item for item in specs() if item.name == selected["spec"])
        support_summary = summarize(
            mod,
            score_rows(mod, support, support_val, single, multi, pert_means, spec, compute_mmd=True),
            spec,
            n_boot=args.n_boot,
            seed=args.seed + 1000,
            wessels_route_gap=route_gap,
        )
        zero_spec = HybridSpec(f"{spec.name}_zero_beta_control", 0.0, spec.fallback)
        zero = summarize(
            mod,
            score_rows(mod, support, support_val, single, multi, pert_means, zero_spec, compute_mmd=True),
            zero_spec,
            n_boot=args.n_boot,
            seed=args.seed + 1001,
            wessels_route_gap=route_gap,
        )
        shuf_bank = shuffled_bank(single.get("gene_raw_mean") or {}, args.seed + 1002)
        shuffled = summarize(
            mod,
            score_rows(mod, support, support_val, single, multi, pert_means, spec, bank_override=shuf_bank, compute_mmd=True),
            spec,
            n_boot=args.n_boot,
            seed=args.seed + 1003,
            wessels_route_gap=route_gap,
        )
    if support_summary is None:
        reasons = ["no_spec_passed_train_multi_gate"]
    else:
        reasons = gate_reasons(support_summary, require_coverage=True)
        if zero and not gate_reasons(zero, require_coverage=False):
            reasons.append("zero_beta_control_passed_unexpectedly")
        if shuffled and not gate_reasons(shuffled, require_coverage=False):
            reasons.append("shuffled_gene_bank_control_passed_unexpectedly")
    status = (
        "trackc_composition_hybrid_prior_gate_pass_posthoc_mmd_gate_next_no_gpu"
        if not reasons
        else "trackc_composition_hybrid_prior_gate_fail_no_gpu"
    )
    global_single_mean = single.get("global_single_mean")
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "query_free_posthoc_mmd_gate_only" if not reasons else "none",
        "reasons": reasons,
        "boundary": {
            "safe_trainselect_only": True,
            "train_multi_selection_only": True,
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


def table(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["- not evaluated", ""]
    lines = [
        f"- spec: `{summary['spec']}`",
        f"- paired pp delta: `{fmt((summary.get('paired_pp_delta') or {}).get('delta_mean'))}`",
        f"- paired pp p_harm: `{fmt((summary.get('paired_pp_delta') or {}).get('p_harm'))}`",
        f"- paired MMD delta: `{fmt((summary.get('paired_mmd_delta') or {}).get('delta_mean'))}`",
        f"- paired MMD p_harm: `{fmt((summary.get('paired_mmd_delta') or {}).get('p_harm'))}`",
        "",
        "| dataset | n | coverage | raw gene cov | fallback frac | delta pp | route gap | closure | delta MMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("dataset_breakdown") or []:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('coverage_fraction'))} | "
            f"{fmt(row.get('raw_gene_coverage_fraction'))} | {fmt(row.get('fallback_gene_fraction'))} | "
            f"{fmt(row.get('delta_pp'))} | {fmt(row.get('route_gap_pp'))} | "
            f"{fmt(row.get('route_gap_closed_fraction'))} | {fmt(row.get('delta_mmd_clamped'))} |"
        )
    lines.append("")
    return lines


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Composition Hybrid-Prior Gate",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        f"Next authorization: `{payload['next_authorization']}`",
        "",
        "## Scope",
        "",
        "This query-free CPU gate tests whether train-only fallback priors can broaden the narrow gene_raw additive composition signal. Beta and fallback mode are selected on train_multi only; support_val_multi is final scoring only.",
        "",
        "## Boundary",
        "",
        "- safe trainselect split only",
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
        "## Selected Train-Multi Spec",
        "",
    ]
    selected = payload.get("selected_train_summary")
    if selected:
        lines.extend(table(selected))
    else:
        lines.extend(["- none", ""])
    for title, key in (
        ("Support-Val Summary", "support_val_summary"),
        ("Zero-Beta Control", "zero_beta_control"),
        ("Shuffled-Gene-Bank Control", "shuffled_gene_bank_control"),
    ):
        lines.extend([f"## {title}", ""])
        lines.extend(table(payload.get(key)))
    lines.extend(["## Decision Reasons", ""])
    lines.extend([f"- `{reason}`" for reason in payload.get("reasons") or []] or ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This gate is a coverage-expansion test, not a GPU authorization. A pass requires broad support coverage, Wessels closure, Norman no-harm, MMD no-harm, and controls that fail appropriately.",
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
