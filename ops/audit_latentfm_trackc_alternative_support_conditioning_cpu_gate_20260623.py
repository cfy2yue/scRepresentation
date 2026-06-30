#!/usr/bin/env python3
"""CPU gate for an alternative Track C support-conditioning operator.

This is intentionally CPU-only and leakage-safe.  It tests a small
predeclared family of FiLM-like interaction operators:

    route + a * context_delta + b * (context_delta * route)

The active residual retry already tested a scalar residual pathway.  This gate
therefore requires a nontrivial interaction coefficient and keeps the
zero/shuffled-support negative controls from the residual CPU audit.
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
OUT_JSON = ROOT / "reports/latentfm_trackc_alternative_support_conditioning_cpu_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ALTERNATIVE_SUPPORT_CONDITIONING_CPU_GATE_20260623.md"


@dataclass(frozen=True)
class AltSpec:
    name: str
    ridge: float
    use_abs_route: bool = False


def load_residual_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_residual_operator_gate", RESIDUAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {RESIDUAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def alt_specs() -> list[AltSpec]:
    specs = []
    for ridge in (0.01, 0.1, 1.0, 10.0):
        specs.append(AltSpec(f"film_context_route_ridge{ridge:g}", ridge=ridge, use_abs_route=False))
        specs.append(AltSpec(f"film_context_absroute_ridge{ridge:g}", ridge=ridge, use_abs_route=True))
    return specs


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def feature_matrix(samples: list[dict[str, Any]], spec: AltSpec) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    targets = []
    for sample in samples:
        context = np.asarray(sample["context_delta"], dtype=np.float64)
        route = np.asarray(sample["route"], dtype=np.float64)
        interaction_base = np.abs(route) if spec.use_abs_route else route
        feats.append(np.stack([context, context * interaction_base], axis=1))
        targets.append(np.asarray(sample["target_delta"], dtype=np.float64))
    x = np.concatenate(feats, axis=0)
    y = np.concatenate(targets, axis=0)
    return x, y


def fit_alt_operator(samples: list[dict[str, Any]], spec: AltSpec) -> dict[str, Any]:
    if not samples:
        raise ValueError("no samples to fit")
    x, y = feature_matrix(samples, spec)
    lhs = x.T @ x + float(spec.ridge) * np.eye(x.shape[1], dtype=np.float64)
    rhs = x.T @ y
    coef = np.linalg.solve(lhs, rhs)
    coef = np.clip(coef, -2.0, 2.0)
    return {
        "kind": "film_interaction",
        "spec": spec.name,
        "ridge": float(spec.ridge),
        "use_abs_route": bool(spec.use_abs_route),
        "coef_context": float(coef[0]),
        "coef_interaction": float(coef[1]),
    }


def apply_alt_operator(route: np.ndarray, context_delta: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    route_f = np.asarray(route, dtype=np.float32)
    context = np.asarray(context_delta, dtype=np.float32)
    interaction_base = np.abs(route_f) if bool(fitted["use_abs_route"]) else route_f
    correction = float(fitted["coef_context"]) * context + float(fitted["coef_interaction"]) * context * interaction_base
    return (route_f + correction).astype(np.float32)


def train_cv_rows(
    mod: Any,
    train_rows: list[dict[str, Any]],
    mem_spec: Any,
    alt_spec: AltSpec,
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    support: Any,
) -> list[dict[str, Any]]:
    rows = []
    for heldout in train_rows:
        fit_rows = [r for r in train_rows if mod.condition_key(r) != mod.condition_key(heldout)]
        fit_samples = mod.make_samples(fit_rows, fit_rows, mem_spec, single, multi, support)
        heldout_samples = mod.make_samples([heldout], fit_rows, mem_spec, single, multi, support)
        if len(fit_samples) < 3 or not heldout_samples:
            row = mod.score_noop_row(heldout, single, multi, pert_means, support, compute_mmd=False)
        else:
            fitted = fit_alt_operator(fit_samples, alt_spec)
            sample = heldout_samples[0]
            pred = apply_alt_operator(sample["route"], sample["context_delta"], fitted)
            row = mod.score_prediction(sample, pred, pert_means, support, compute_mmd=False)
            row["coef_interaction"] = fitted["coef_interaction"]
        row.update({"memory_spec": mem_spec.name, "operator_spec": alt_spec.name})
        rows.append(row)
    return rows


def select_spec(cv_summary: list[dict[str, Any]]) -> str:
    eligible = [
        row
        for row in cv_summary
        if (mod_float(row.get("wessels_pp_delta")) or -999) > 0.0
        and (mod_float(row.get("norman_pp_delta")) or 0.0) >= -0.02
        and abs(float(row.get("mean_abs_interaction_coef") or 0.0)) > 1e-5
    ]
    pool = eligible or cv_summary
    return str(pool[0]["spec"])


def mod_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_cv(mod: Any, rows_by_spec: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out = []
    for name, rows in sorted(rows_by_spec.items()):
        by_ds = mod.dataset_delta(rows, "candidate", "support_selected_route")
        coefs = [abs(float(r.get("coef_interaction") or 0.0)) for r in rows]
        cand = mod.equal_dataset_mean(rows, "candidate")
        route = mod.equal_dataset_mean(rows, "support_selected_route")
        out.append(
            {
                "spec": name,
                "n_rows": len(rows),
                "equal_dataset_pp_delta": None if cand is None or route is None else float(cand - route),
                "norman_pp_delta": by_ds.get("NormanWeissman2019_filtered"),
                "wessels_pp_delta": by_ds.get("Wessels"),
                "mean_abs_interaction_coef": float(np.mean(coefs)) if coefs else 0.0,
            }
        )
    return sorted(
        out,
        key=lambda row: (
            mod_float(row.get("wessels_pp_delta")) if mod_float(row.get("wessels_pp_delta")) is not None else -999,
            mod_float(row.get("equal_dataset_pp_delta")) if mod_float(row.get("equal_dataset_pp_delta")) is not None else -999,
        ),
        reverse=True,
    )


def score_eval_rows(
    mod: Any,
    support: Any,
    support_val: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
    fitted: dict[str, Any],
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    *,
    control: str = "real",
    seed: int = 0,
) -> list[dict[str, Any]]:
    eval_by_key = {mod.condition_key(sample["row"]): sample for sample in eval_samples}
    rng = np.random.default_rng(seed)
    contexts = [np.asarray(sample["context_delta"], dtype=np.float32) for sample in eval_samples]
    if contexts:
        shuffled = [contexts[int(i)] for i in rng.permutation(len(contexts))]
    else:
        shuffled = []
    shuffled_by_key = {mod.condition_key(sample["row"]): shuffled[i] for i, sample in enumerate(eval_samples)}

    rows = []
    for target in support_val:
        sample = eval_by_key.get(mod.condition_key(target))
        if sample is None:
            row = mod.score_noop_row(target, single, multi, pert_means, support, compute_mmd=True)
            row["no_context_noop"] = True
        else:
            if control == "real":
                context = sample["context_delta"]
            elif control == "zero_context":
                context = np.zeros_like(sample["context_delta"], dtype=np.float32)
            elif control == "shuffled_context":
                context = shuffled_by_key[mod.condition_key(target)]
            else:
                raise ValueError(control)
            pred = apply_alt_operator(sample["route"], context, fitted)
            row = mod.score_prediction(sample, pred, pert_means, support, compute_mmd=True)
        row["control"] = control
        rows.append(row)
    return rows


def dataset_breakdown(mod: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ds in sorted({str(r["dataset"]) for r in rows}):
        sub = [r for r in rows if str(r["dataset"]) == ds]
        out.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "candidate": float(np.mean([r["candidate"] for r in sub if r.get("candidate") is not None])),
                "support_selected_route": float(np.mean([r["support_selected_route"] for r in sub if r.get("support_selected_route") is not None])),
                "delta_pp": mod.dataset_delta(sub, "candidate", "support_selected_route").get(ds),
                "candidate_mmd_clamped": float(np.mean([r["candidate__test_mmd_clamped"] for r in sub])),
                "route_mmd_clamped": float(np.mean([r["support_selected_route__test_mmd_clamped"] for r in sub])),
            }
        )
    return out


def summarize_gate(
    mod: Any,
    rows: list[dict[str, Any]],
    *,
    closed_delta: float,
    route_gap: float | None,
    split_guard: dict[str, Any],
    n_boot: int,
    seed: int,
    wiring_delta_l2: float,
) -> dict[str, Any]:
    pp_delta = mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd_delta = mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 100)
    decision = mod.decide(
        rows,
        pp_delta,
        mmd_delta,
        closed_wessels_delta=closed_delta,
        wessels_route_gap=route_gap,
        wiring_delta_l2=wiring_delta_l2,
        split=split_guard,
    )
    return {
        "decision": decision,
        "paired_pp_delta": pp_delta,
        "paired_mmd_delta": mmd_delta,
        "dataset_breakdown": dataset_breakdown(mod, rows),
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["real"]["decision"]
    lines = [
        "# Track C Alternative Support-Conditioning CPU Gate",
        "",
        f"Status: `{payload['status']}`",
        f"GPU authorization: `{payload['gpu_authorization']}`",
        "",
        "## Provenance",
        "",
        f"- split_file: `{payload['split_guard']['split_file']}`",
        f"- split SHA256: `{payload['split_guard']['sha256']}`",
        f"- selected spec: `{payload['selected_spec']}`",
        f"- fitted coefficients: context `{fmt(payload['fitted_operator']['coef_context'])}`, interaction `{fmt(payload['fitted_operator']['coef_interaction'])}`",
        f"- leakage status: `{payload['split_guard']['leakage_status']}`",
        "",
        "## Gate Criteria",
        "",
        f"- Wessels delta vs route: `{fmt(decision['wessels_delta_vs_route'])}`",
        f"- Wessels delta vs best closed family: `{fmt(decision['wessels_delta_vs_best_closed_family'])}` (gate `>= +0.020000`)",
        f"- Wessels route-gap closure: `{fmt(decision['wessels_route_gap_closure'])}` (gate `>= +0.050000`)",
        f"- Norman delta vs route: `{fmt(decision['norman_delta_vs_route'])}` (gate `>= -0.020000`)",
        f"- pp p_harm: `{fmt(payload['real']['paired_pp_delta'].get('p_harm'))}` (gate `<= 0.200000`)",
        f"- MMD p_harm: `{fmt(payload['real']['paired_mmd_delta'].get('p_harm'))}` (hard gate `<= 0.800000`)",
        f"- interaction coefficient abs: `{fmt(abs(payload['fitted_operator']['coef_interaction']))}` (must be nonzero)",
        "",
        "## Negative Controls",
        "",
        "| control | status | Wessels delta | closure | Norman delta | pp p_harm | MMD p_harm |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("zero_context", "shuffled_context"):
        ctrl = payload[name]
        d = ctrl["decision"]
        lines.append(
            f"| `{name}` | `{d['status']}` | {fmt(d.get('wessels_delta_vs_route'))} | "
            f"{fmt(d.get('wessels_route_gap_closure'))} | {fmt(d.get('norman_delta_vs_route'))} | "
            f"{fmt(ctrl['paired_pp_delta'].get('p_harm'))} | {fmt(ctrl['paired_mmd_delta'].get('p_harm'))} |"
        )
    lines += [
        "",
        "## Support-Val Dataset Breakdown",
        "",
        "| dataset | n | candidate pp | route pp | delta | candidate MMD | route MMD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["real"]["dataset_breakdown"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('candidate'))} | "
            f"{fmt(row.get('support_selected_route'))} | {fmt(row.get('delta_pp'))} | "
            f"{fmt(row.get('candidate_mmd_clamped'))} | {fmt(row.get('route_mmd_clamped'))} |"
        )
    lines += [
        "",
        "## Train-CV Selection Summary",
        "",
        "| spec | n | equal-dataset delta | Norman delta | Wessels delta | mean abs interaction coef |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["cv_summary"][:12]:
        marker = " (selected)" if row["spec"] == payload["selected_spec"] else ""
        lines.append(
            f"| `{row['spec']}`{marker} | {row['n_rows']} | {fmt(row.get('equal_dataset_pp_delta'))} | "
            f"{fmt(row.get('norman_pp_delta'))} | {fmt(row.get('wessels_pp_delta'))} | "
            f"{fmt(row.get('mean_abs_interaction_coef'))} |"
        )
    lines += ["", "## Decision Reasons", ""]
    reasons = payload.get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    mod = load_residual_module()
    support = mod.load_support_module()
    data_dir = mod.DEFAULT_DATA_DIR.resolve()
    split_file = mod.DEFAULT_SPLIT
    split = support.load_json(split_file)
    manifest = support.load_json(data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {k: v.astype(np.float32) for k, v in np.load(mod.DEFAULT_PERT_MEANS).items()}
    guard = mod.split_guard(split_file, split)
    if guard["sha256"] != mod.EXPECTED_TRAINSELECT_SHA256:
        raise RuntimeError(f"unexpected trainselect split hash: {guard['sha256']}")

    train_rows = support.collect_role_rows(data_dir, split, metadata, "train_multi", max_cells=256)
    support_val = support.collect_role_rows(data_dir, split, metadata, "support_val_multi", max_cells=256)
    single = support.train_single_components(data_dir, split, metadata, max_cells=256)
    multi = support.train_multi_components(train_rows)

    rows_by_spec: dict[str, list[dict[str, Any]]] = {}
    lookup: dict[str, tuple[Any, AltSpec]] = {}
    for mem_spec in mod.memory_specs():
        for alt_spec in alt_specs():
            name = f"{mem_spec.name}__{alt_spec.name}"
            lookup[name] = (mem_spec, alt_spec)
            rows_by_spec[name] = train_cv_rows(mod, train_rows, mem_spec, alt_spec, single, multi, pert_means, support)
    cv_summary = summarize_cv(mod, rows_by_spec)
    selected_name = select_spec(cv_summary)
    selected_mem, selected_alt = lookup[selected_name]
    fit_samples = mod.make_samples(train_rows, train_rows, selected_mem, single, multi, support)
    fitted = fit_alt_operator(fit_samples, selected_alt)
    eval_samples = mod.make_samples(support_val, train_rows, selected_mem, single, multi, support)
    real_rows = score_eval_rows(mod, support, support_val, eval_samples, fitted, single, multi, pert_means, control="real", seed=args.seed)
    zero_rows = score_eval_rows(mod, support, support_val, eval_samples, fitted, single, multi, pert_means, control="zero_context", seed=args.seed)
    shuffled_rows = score_eval_rows(mod, support, support_val, eval_samples, fitted, single, multi, pert_means, control="shuffled_context", seed=args.seed + 1)

    closed_delta = mod.load_closed_wessels_delta(mod.DEFAULT_BOTTLENECK_SUMMARY)
    route_gap = mod.readout_wessels_route_gap(mod.DEFAULT_READOUT_JSON)
    wiring_delta_l2 = 0.0
    if eval_samples:
        sample = eval_samples[0]
        pred_real = apply_alt_operator(sample["route"], sample["context_delta"], fitted)
        pred_zero = apply_alt_operator(sample["route"], np.zeros_like(sample["context_delta"]), fitted)
        wiring_delta_l2 = float(np.linalg.norm(pred_real - pred_zero))

    real = summarize_gate(mod, real_rows, closed_delta=closed_delta, route_gap=route_gap, split_guard=guard, n_boot=args.n_boot, seed=args.seed, wiring_delta_l2=wiring_delta_l2)
    zero = summarize_gate(mod, zero_rows, closed_delta=closed_delta, route_gap=route_gap, split_guard=guard, n_boot=args.n_boot, seed=args.seed + 10, wiring_delta_l2=0.0)
    shuffled = summarize_gate(mod, shuffled_rows, closed_delta=closed_delta, route_gap=route_gap, split_guard=guard, n_boot=args.n_boot, seed=args.seed + 20, wiring_delta_l2=wiring_delta_l2)

    reasons = list(real["decision"].get("reasons") or [])
    if abs(float(fitted["coef_interaction"])) <= 1e-5:
        reasons.append("interaction_coefficient_too_small")
    for name, control in (("zero_context", zero), ("shuffled_context", shuffled)):
        if str(control["decision"]["status"]).endswith("pass_authorize_one_capped_gpu_smoke"):
            reasons.append(f"{name}_unexpectedly_passed")
    status = "trackc_alternative_support_conditioning_cpu_gate_pass_authorize_one_capped_gpu_smoke" if not reasons else "trackc_alternative_support_conditioning_cpu_gate_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorization": "one_capped_trackc_support_only_smoke" if not reasons else "none",
        "reasons": reasons,
        "split_guard": guard,
        "selected_spec": selected_name,
        "fitted_operator": fitted,
        "cv_summary": cv_summary,
        "real": real,
        "zero_context": zero,
        "shuffled_context": shuffled,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorization": payload["gpu_authorization"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
