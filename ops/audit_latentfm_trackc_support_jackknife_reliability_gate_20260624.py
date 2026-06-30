#!/usr/bin/env python3
"""Query-free Track C support-jackknife reliability CPU gate.

The gate asks whether support-context residual transfer can be made safer by
using only support predictions whose direction is stable under jackknife
resampling of the support memory. Selection uses safe-trainselect train_multi
leave-one-condition episodes only; support_val_multi is final scoring only.

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
OUT_JSON = ROOT / "reports/latentfm_trackc_support_jackknife_reliability_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_JACKKNIFE_RELIABILITY_GATE_20260624.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"
FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")


@dataclass(frozen=True)
class ReliabilitySpec:
    name: str
    memory_mode: str
    k: int
    same_dataset: bool
    min_score: float
    min_jackknife_cos: float
    max_norm_cv: float
    min_context_rows: int
    ridge: float = 0.0


def load_residual_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_residual_operator_gate", RESIDUAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {RESIDUAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def specs() -> list[ReliabilitySpec]:
    out: list[ReliabilitySpec] = []
    for mode, min_score in (("overlap", 1.0), ("jaccard", 0.25)):
        for same_dataset in (True, False):
            for k in (3, 5):
                for min_cos in (0.50, 0.75, 0.90):
                    for max_cv in (0.50, 1.00, 2.00):
                        out.append(
                            ReliabilitySpec(
                                name=(
                                    f"{mode}_k{k}_{'same_ds' if same_dataset else 'all_ds'}"
                                    f"_cos{min_cos:g}_cv{max_cv:g}"
                                ),
                                memory_mode=mode,
                                k=k,
                                same_dataset=same_dataset,
                                min_score=min_score,
                                min_jackknife_cos=min_cos,
                                max_norm_cv=max_cv,
                                min_context_rows=2,
                            )
                        )
    return out


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def genes(row: dict[str, Any]) -> set[str]:
    return {str(g).strip().upper() for g in row.get("genes") or [] if str(g).strip()}


def memory_score(target: dict[str, Any], row: dict[str, Any], mode: str) -> float:
    a = genes(target)
    b = genes(row)
    if mode == "overlap":
        return float(len(a & b))
    if mode == "jaccard":
        return float(len(a & b) / max(len(a | b), 1))
    raise ValueError(mode)


def overlap_class(target: dict[str, Any], memory_rows: list[dict[str, Any]]) -> str:
    best = 0
    g = genes(target)
    for row in memory_rows:
        best = max(best, len(g & genes(row)))
    if best <= 0:
        return "zero_overlap"
    if best == 1:
        return "one_gene_overlap"
    return "seen_pair_or_two_gene_overlap"


def route_vector(mod: Any, support_mod: Any, row: dict[str, Any], single: dict[str, Any], multi: dict[str, Any]) -> np.ndarray:
    return mod.route_residual(row, single, multi, support_mod)


def select_memory(target: dict[str, Any], memory_rows: list[dict[str, Any]], spec: ReliabilitySpec) -> list[tuple[float, dict[str, Any]]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in memory_rows:
        if condition_key(row) == condition_key(target):
            continue
        if spec.same_dataset and str(row["dataset"]) != str(target["dataset"]):
            continue
        score = memory_score(target, row, spec.memory_mode)
        if score >= spec.min_score:
            candidates.append((score, row))
    candidates.sort(key=lambda item: (item[0], str(item[1]["dataset"]), str(item[1]["condition"])), reverse=True)
    return candidates[: max(1, int(spec.k))]


def weighted_mean(selected: list[tuple[float, dict[str, Any]]]) -> np.ndarray | None:
    if not selected:
        return None
    weights = np.asarray([max(score, 1e-6) for score, _row in selected], dtype=np.float64)
    weights = weights / weights.sum()
    residuals = np.vstack([np.asarray(row["residual"], dtype=np.float32) for _score, row in selected])
    return (weights[:, None] * residuals).sum(axis=0).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def context_and_reliability(
    target: dict[str, Any],
    memory_rows: list[dict[str, Any]],
    spec: ReliabilitySpec,
    route: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    selected = select_memory(target, memory_rows, spec)
    context = weighted_mean(selected)
    if context is None:
        return None, {
            "n_context_rows": 0,
            "jackknife_cos_mean": None,
            "jackknife_norm_cv": None,
            "reliable": False,
        }
    full_delta = context - route
    jack = []
    if len(selected) > 1:
        for i in range(len(selected)):
            sub = selected[:i] + selected[i + 1 :]
            sub_context = weighted_mean(sub)
            if sub_context is not None:
                jack.append(sub_context - route)
    cos_vals = [cosine(full_delta, item) for item in jack]
    cos_vals = [v for v in cos_vals if v is not None]
    norms = np.asarray([np.linalg.norm(item) for item in jack], dtype=np.float64)
    norm_mean = float(norms.mean()) if norms.size else None
    norm_cv = None
    if norms.size and norm_mean is not None and norm_mean > 1e-12:
        norm_cv = float(norms.std() / norm_mean)
    cos_mean = float(np.mean(cos_vals)) if cos_vals else None
    reliable = (
        len(selected) >= spec.min_context_rows
        and cos_mean is not None
        and cos_mean >= spec.min_jackknife_cos
        and norm_cv is not None
        and norm_cv <= spec.max_norm_cv
    )
    return context, {
        "n_context_rows": len(selected),
        "jackknife_cos_mean": cos_mean,
        "jackknife_norm_cv": norm_cv,
        "reliable": bool(reliable),
    }


def fit_alpha(samples: list[tuple[np.ndarray, np.ndarray]], ridge: float) -> float:
    if not samples:
        return 0.0
    x = np.vstack([item[0] for item in samples]).astype(np.float64)
    y = np.vstack([item[1] for item in samples]).astype(np.float64)
    denom = float(np.sum(x * x) + float(ridge))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.sum(x * y) / denom, -1.0, 1.5))


def fit_samples(
    rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    spec: ReliabilitySpec,
    mod: Any,
    support_mod: Any,
    single: dict[str, Any],
    multi: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows:
        route = route_vector(mod, support_mod, row, single, multi)
        context, _rel = context_and_reliability(row, memory_rows, spec, route)
        if context is None:
            continue
        out.append((context - route, np.asarray(row["residual"], dtype=np.float32) - route))
    return out


def score_row(
    row: dict[str, Any],
    memory_rows: list[dict[str, Any]],
    spec: ReliabilitySpec,
    alpha: float,
    mod: Any,
    support_mod: Any,
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    *,
    compute_mmd: bool,
) -> dict[str, Any]:
    route = route_vector(mod, support_mod, row, single, multi)
    context, rel = context_and_reliability(row, memory_rows, spec, route)
    if context is None or not rel["reliable"]:
        pred = route
        abstained = True
    else:
        pred = (route + alpha * (context - route)).astype(np.float32)
        abstained = False
    item = {
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "genes": sorted(genes(row)),
        "overlap_class": overlap_class(row, memory_rows),
        "candidate": support_mod.pp_score(row, pred, pert_means),
        "support_selected_route": support_mod.pp_score(row, route, pert_means),
        "abstained": bool(abstained),
        "alpha": float(alpha),
        **rel,
    }
    if compute_mmd:
        for metric, value in support_mod.mmd_scores(row, pred).items():
            item[f"candidate__{metric}"] = value
        for metric, value in support_mod.mmd_scores(row, route).items():
            item[f"support_selected_route__{metric}"] = value
    return item


def cv_rows(
    train_rows: list[dict[str, Any]],
    spec: ReliabilitySpec,
    mod: Any,
    support_mod: Any,
    single: dict[str, Any],
    pert_means: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    out = []
    for heldout in train_rows:
        memory = [row for row in train_rows if condition_key(row) != condition_key(heldout)]
        multi = support_mod.train_multi_components(memory)
        alpha = fit_alpha(fit_samples(memory, memory, spec, mod, support_mod, single, multi), spec.ridge)
        item = score_row(
            heldout,
            memory,
            spec,
            alpha,
            mod,
            support_mod,
            single,
            multi,
            pert_means,
            compute_mmd=False,
        )
        item["spec"] = spec.name
        out.append(item)
    return out


def shuffled_support(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    residuals = [np.asarray(row["residual"], dtype=np.float32) for row in rows]
    order = rng.permutation(len(rows))
    out = []
    for i, row in enumerate(rows):
        copy = dict(row)
        copy["residual"] = residuals[int(order[i])]
        out.append(copy)
    return out


def mean_delta(rows: list[dict[str, Any]], candidate: str, baseline: str) -> float | None:
    vals = []
    for row in rows:
        if row.get(candidate) is not None and row.get(baseline) is not None:
            vals.append(float(row[candidate]) - float(row[baseline]))
    return None if not vals else float(np.mean(vals))


def summarize(mod: Any, rows: list[dict[str, Any]], *, n_boot: int, seed: int, include_mmd: bool) -> dict[str, Any]:
    pp = mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd = (
        mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 100)
        if include_mmd
        else None
    )
    by_ds = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        by_ds[ds] = {
            "n": len(sub),
            "pp_delta": mean_delta(sub, "candidate", "support_selected_route"),
            "mmd_delta": mean_delta(sub, "candidate__test_mmd_clamped", "support_selected_route__test_mmd_clamped")
            if include_mmd
            else None,
            "enabled": int(sum(not bool(row.get("abstained")) for row in sub)),
        }
    by_overlap = {}
    for cls in sorted({str(row.get("overlap_class")) for row in rows}):
        sub = [row for row in rows if str(row.get("overlap_class")) == cls]
        by_overlap[cls] = {
            "n": len(sub),
            "pp_delta": mean_delta(sub, "candidate", "support_selected_route"),
            "mmd_delta": mean_delta(sub, "candidate__test_mmd_clamped", "support_selected_route__test_mmd_clamped")
            if include_mmd
            else None,
            "enabled": int(sum(not bool(row.get("abstained")) for row in sub)),
        }
    enabled_rows = [row for row in rows if not bool(row.get("abstained"))]
    enabled_deltas = [
        float(row["candidate"]) - float(row["support_selected_route"])
        for row in enabled_rows
        if row.get("candidate") is not None and row.get("support_selected_route") is not None
    ]
    return {
        "paired_pp_delta": pp,
        "paired_mmd_delta": mmd,
        "by_dataset": by_ds,
        "by_overlap_class": by_overlap,
        "enabled_rows": len(enabled_rows),
        "enabled_negative_rows": int(sum(v < 0 for v in enabled_deltas)),
        "enabled_min_pp_delta": float(np.min(enabled_deltas)) if enabled_deltas else None,
    }


def num_or(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def select_spec(summaries: list[dict[str, Any]]) -> str:
    eligible = []
    for item in summaries:
        summ = item["summary"]
        norman = (summ["by_dataset"].get("NormanWeissman2019_filtered") or {}).get("pp_delta")
        wessels = (summ["by_dataset"].get("Wessels") or {}).get("pp_delta")
        if (
            num_or(summ["paired_pp_delta"].get("delta_mean"), -999.0) > 0.0
            and num_or(summ["paired_pp_delta"].get("p_harm"), 1.0) <= 0.35
            and num_or(norman, -999.0) >= -0.02
            and num_or(wessels, -999.0) > 0.0
            and summ["enabled_rows"] >= 8
        ):
            eligible.append(item)
    pool = eligible or summaries
    pool = sorted(
        pool,
        key=lambda item: (
            num_or(item["summary"]["paired_pp_delta"].get("delta_mean"), -999.0),
            num_or((item["summary"]["by_dataset"].get("Wessels") or {}).get("pp_delta"), -999.0),
            item["summary"].get("enabled_rows", 0),
        ),
        reverse=True,
    )
    return str(pool[0]["spec"])


def decide(summary: dict[str, Any], shuffled: dict[str, Any], *, route_gap: float | None) -> dict[str, Any]:
    reasons: list[str] = []
    pp = summary["paired_pp_delta"]
    mmd = summary["paired_mmd_delta"] or {}
    norman = summary["by_dataset"].get("NormanWeissman2019_filtered", {})
    wessels = summary["by_dataset"].get("Wessels", {})
    wessels_delta = num_or(wessels.get("pp_delta"), -999.0)
    closure = None if route_gap is None or abs(route_gap) < 1e-12 else wessels_delta / route_gap
    sep = num_or(pp.get("delta_mean"), 0.0) - num_or(shuffled["paired_pp_delta"].get("delta_mean"), 0.0)
    if num_or(pp.get("delta_mean"), -999.0) < 0.03:
        reasons.append("support_val_pp_delta_below_0p03")
    if num_or(pp.get("p_harm"), 1.0) > 0.20:
        reasons.append("support_val_pp_harm_above_0p20")
    if num_or(norman.get("pp_delta"), -999.0) < -0.01:
        reasons.append("norman_delta_below_minus_0p01")
    if closure is None or closure < 0.30:
        reasons.append("wessels_route_gap_closure_below_0p30")
    if mmd.get("status") != "ok" or num_or(mmd.get("p_harm"), 1.0) > 0.80:
        reasons.append("mmd_hard_harm_or_missing")
    if num_or(mmd.get("delta_mean"), 999.0) > 0.0:
        reasons.append("mmd_delta_not_nonharm")
    if summary["enabled_rows"] < 8:
        reasons.append("enabled_support_rows_below_8")
    if summary["enabled_negative_rows"] > 2:
        reasons.append("too_many_enabled_negative_rows")
    if sep < 0.02:
        reasons.append("shuffled_support_control_not_separated_by_0p02")
    status = "trackc_support_jackknife_reliability_gate_pass_code_gate_next_no_gpu" if not reasons else "trackc_support_jackknife_reliability_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "code_launcher_gate_only" if not reasons else "none",
        "reasons": reasons,
        "support_val_pp_delta": pp.get("delta_mean"),
        "support_val_mmd_delta": mmd.get("delta_mean"),
        "norman_pp_delta": norman.get("pp_delta"),
        "wessels_pp_delta": wessels.get("pp_delta"),
        "wessels_route_gap": route_gap,
        "wessels_route_gap_closure": closure,
        "enabled_rows": summary["enabled_rows"],
        "enabled_negative_rows": summary["enabled_negative_rows"],
        "enabled_min_pp_delta": summary["enabled_min_pp_delta"],
        "shuffled_pp_delta": shuffled["paired_pp_delta"].get("delta_mean"),
        "candidate_minus_shuffled_pp_delta": sep,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Track C Support-Jackknife Reliability Gate",
        "",
        f"Status: `{decision['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- CPU-only query-free gate.",
        "- Selection uses safe-trainselect train_multi leave-one-condition episodes only.",
        "- support_val_multi is final scoring only.",
        "- Held-out query, canonical test, canonical multi, active logs, and GPU artifacts are not read.",
        "",
        "## Selected Spec",
        "",
        f"- `{payload['selected_spec']}`",
        f"- selected alpha: `{fmt(payload['selected_alpha'])}`",
        f"- train_multi rows: `{payload['n_train_multi_rows']}`",
        f"- support_val_multi rows: `{payload['n_support_val_rows']}`",
        "",
        "## Support-Val Gate Criteria",
        "",
        f"- support pp delta: `{fmt(decision['support_val_pp_delta'])}` (gate `>= +0.030000`)",
        f"- support MMD delta: `{fmt(decision['support_val_mmd_delta'])}` (gate `<= 0`)",
        f"- Norman pp delta: `{fmt(decision['norman_pp_delta'])}` (gate `>= -0.010000`)",
        f"- Wessels route-gap closure: `{fmt(decision['wessels_route_gap_closure'])}` (gate `>= +0.300000`)",
        f"- enabled rows: `{decision['enabled_rows']}`",
        f"- enabled negative rows: `{decision['enabled_negative_rows']}`",
        f"- candidate minus shuffled pp: `{fmt(decision['candidate_minus_shuffled_pp_delta'])}` (gate `>= +0.020000`)",
        "",
        "## Support-Val Breakdown",
        "",
        "### By Dataset",
        "",
        "| dataset | n | enabled | pp delta | MMD delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, row in payload["support_summary"]["by_dataset"].items():
        lines.append(f"| `{ds}` | {row['n']} | {row['enabled']} | {fmt(row.get('pp_delta'))} | {fmt(row.get('mmd_delta'))} |")
    lines.extend(["", "### By Overlap Class", "", "| class | n | enabled | pp delta | MMD delta |", "|---|---:|---:|---:|---:|"])
    for cls, row in payload["support_summary"]["by_overlap_class"].items():
        lines.append(f"| `{cls}` | {row['n']} | {row['enabled']} | {fmt(row.get('pp_delta'))} | {fmt(row.get('mmd_delta'))} |")
    lines.extend(
        [
            "",
            "## Train-CV Selection Summary",
            "",
            "| spec | pp delta | Norman | Wessels | enabled | p_harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["cv_summaries"][:12]:
        summ = item["summary"]
        marker = " (selected)" if item["spec"] == payload["selected_spec"] else ""
        norman = summ["by_dataset"].get("NormanWeissman2019_filtered", {})
        wessels = summ["by_dataset"].get("Wessels", {})
        lines.append(
            f"| `{item['spec']}`{marker} | {fmt(summ['paired_pp_delta'].get('delta_mean'))} | "
            f"{fmt(norman.get('pp_delta'))} | {fmt(wessels.get('pp_delta'))} | "
            f"{summ['enabled_rows']} | {fmt(summ['paired_pp_delta'].get('p_harm'))} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    reasons = decision.get("reasons") or []
    lines.extend([f"- `{r}`" for r in reasons] if reasons else ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Passing would authorize only a code/launcher gate, not GPU or held-out query. "
            "Failure closes support-jackknife reliability as the next Track C expansion route "
            "unless a materially new query-free reliability signal is documented.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/latentfm_full/xverse")
    parser.add_argument("--split-file", type=Path, default=ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json")
    parser.add_argument(
        "--pert-means-file",
        type=Path,
        default=ROOT / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
    )
    parser.add_argument("--readout-json", type=Path, default=ROOT / "reports/latentfm_trackc_trainonly_memory_readout_gate_20260622.json")
    parser.add_argument("--max-cells-per-condition", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    mod = load_residual_module()
    support_mod = mod.load_support_module()
    split = support_mod.load_json(args.split_file)
    guard = mod.split_guard(args.split_file, split)
    if guard["sha256"] != EXPECTED_TRAINSELECT_SHA256:
        raise RuntimeError(f"unexpected trainselect split hash: {guard['sha256']}")
    for ds in FOCUS_DATASETS:
        obj = split.get(ds) or {}
        if set(obj.get("support_val_multi") or []) & set(obj.get("heldout_query_multi_final_only") or []):
            raise RuntimeError(f"{ds}: support_val_multi overlaps heldout query")

    manifest = support_mod.load_json(args.data_dir / "manifest.json")
    metadata = support_mod.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {key: value.astype(np.float32) for key, value in np.load(args.pert_means_file).items()}
    train_rows = support_mod.collect_role_rows(args.data_dir, split, metadata, "train_multi", max_cells=args.max_cells_per_condition)
    support_val = support_mod.collect_role_rows(args.data_dir, split, metadata, "support_val_multi", max_cells=args.max_cells_per_condition)
    single = support_mod.train_single_components(args.data_dir, split, metadata, max_cells=args.max_cells_per_condition)

    cv_summaries: list[dict[str, Any]] = []
    for spec in specs():
        rows = cv_rows(train_rows, spec, mod, support_mod, single, pert_means)
        cv_summaries.append({"spec": spec.name, "summary": summarize(mod, rows, n_boot=args.n_boot, seed=args.seed, include_mmd=False)})
    cv_summaries = sorted(
        cv_summaries,
        key=lambda item: (
            num_or(item["summary"]["paired_pp_delta"].get("delta_mean"), -999.0),
            num_or((item["summary"]["by_dataset"].get("Wessels") or {}).get("pp_delta"), -999.0),
            item["summary"]["enabled_rows"],
        ),
        reverse=True,
    )
    selected_name = select_spec(cv_summaries)
    selected_spec = next(spec for spec in specs() if spec.name == selected_name)

    multi = support_mod.train_multi_components(train_rows)
    alpha = fit_alpha(fit_samples(train_rows, train_rows, selected_spec, mod, support_mod, single, multi), selected_spec.ridge)
    support_rows = [
        score_row(row, train_rows, selected_spec, alpha, mod, support_mod, single, multi, pert_means, compute_mmd=True)
        for row in support_val
    ]
    support_summary = summarize(mod, support_rows, n_boot=args.n_boot, seed=args.seed + 17, include_mmd=True)

    shuffled_train = shuffled_support(train_rows, args.seed + 301)
    shuffled_multi = support_mod.train_multi_components(shuffled_train)
    shuffled_alpha = fit_alpha(
        fit_samples(shuffled_train, shuffled_train, selected_spec, mod, support_mod, single, shuffled_multi),
        selected_spec.ridge,
    )
    shuffled_rows = [
        score_row(row, shuffled_train, selected_spec, shuffled_alpha, mod, support_mod, single, shuffled_multi, pert_means, compute_mmd=True)
        for row in support_val
    ]
    shuffled_summary = summarize(mod, shuffled_rows, n_boot=args.n_boot, seed=args.seed + 401, include_mmd=True)
    route_gap = mod.readout_wessels_route_gap(args.readout_json)
    decision = decide(support_summary, shuffled_summary, route_gap=route_gap)

    payload = {
        "status": decision["status"],
        "boundary": {
            "reads_raw_heldout_query": False,
            "reads_canonical_test": False,
            "reads_canonical_multi_for_selection": False,
            "reads_active_logs": False,
            "launches_gpu": False,
            "selection": "train_multi_leave_one_condition_episodes_only",
            "final_scoring": "support_val_multi_only",
        },
        "split_guard": guard,
        "data_dir": str(args.data_dir),
        "pert_means_file": str(args.pert_means_file),
        "readout_json": str(args.readout_json),
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_train_multi_rows": len(train_rows),
        "n_support_val_rows": len(support_val),
        "selected_spec": selected_name,
        "selected_spec_config": selected_spec.__dict__,
        "selected_alpha": alpha,
        "shuffled_alpha": shuffled_alpha,
        "cv_summaries": cv_summaries,
        "support_summary": support_summary,
        "shuffled_summary": shuffled_summary,
        "support_rows": support_rows,
        "shuffled_rows": shuffled_rows,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(render(payload))
    print(json.dumps({"status": decision["status"], "gpu_authorization": decision["gpu_authorization"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
