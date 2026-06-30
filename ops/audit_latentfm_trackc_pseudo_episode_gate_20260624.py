#!/usr/bin/env python3
"""Query-free Track C pseudo-episode CPU gate.

This gate targets the main post-v2 uncertainty without touching held-out query:
can support-conditioned residual transfer generalize to pseudo-query multi
conditions with no shared perturbed genes?

Selection uses only train_multi pseudo episodes. support_val_multi is final
scoring only. Held-out query, canonical test, canonical multi, active logs, and
new GPU artifacts are not read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RESIDUAL_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_residual_operator_cpu_gate_20260623.py"
OUT_JSON = ROOT / "reports/latentfm_trackc_pseudo_episode_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_PSEUDO_EPISODE_GATE_20260624.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"
FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")


@dataclass(frozen=True)
class EpisodeSpec:
    name: str
    memory_mode: str
    k: int
    same_dataset: bool
    min_score: float
    operator: str
    ridge: float = 0.0
    no_overlap_fallback: str = "none"


def load_residual_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_residual_operator_gate", RESIDUAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {RESIDUAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def specs() -> list[EpisodeSpec]:
    out: list[EpisodeSpec] = []
    for mode, min_score in (("overlap", 1.0), ("jaccard", 0.25)):
        for same_dataset in (True, False):
            for k in (1, 3, 5):
                for fallback in ("none", "dataset_mean", "global_mean"):
                    suffix = "" if fallback == "none" else f"_zero_{fallback}"
                    out.append(
                        EpisodeSpec(
                            name=f"{mode}_k{k}_{'same_ds' if same_dataset else 'all_ds'}_scalar{suffix}",
                            memory_mode=mode,
                            k=k,
                            same_dataset=same_dataset,
                            min_score=min_score,
                            operator="scalar",
                            no_overlap_fallback=fallback,
                        )
                    )
    return out


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def genes(row: dict[str, Any]) -> set[str]:
    return {str(g).strip().upper() for g in row.get("genes") or [] if str(g).strip()}


def overlap_class(target: dict[str, Any], support_rows: list[dict[str, Any]]) -> str:
    target_genes = genes(target)
    best = 0
    for row in support_rows:
        best = max(best, len(target_genes & genes(row)))
    if best <= 0:
        return "zero_overlap"
    if best == 1:
        return "one_gene_overlap"
    return "seen_pair_or_two_gene_overlap"


def memory_score(target: dict[str, Any], row: dict[str, Any], mode: str) -> float:
    a = genes(target)
    b = genes(row)
    if mode == "overlap":
        return float(len(a & b))
    if mode == "jaccard":
        return float(len(a & b) / max(len(a | b), 1))
    raise ValueError(mode)


def route_vector(mod: Any, support_mod: Any, row: dict[str, Any], single: dict[str, Any], multi: dict[str, Any]) -> np.ndarray:
    return mod.route_residual(row, single, multi, support_mod)


def weighted_context(
    target: dict[str, Any],
    support_rows: list[dict[str, Any]],
    spec: EpisodeSpec,
) -> np.ndarray | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in support_rows:
        if condition_key(row) == condition_key(target):
            continue
        if spec.same_dataset and str(row["dataset"]) != str(target["dataset"]):
            continue
        score = memory_score(target, row, spec.memory_mode)
        if score >= spec.min_score:
            candidates.append((score, row))
    if not candidates:
        if spec.no_overlap_fallback == "dataset_mean":
            vals = [
                np.asarray(row["residual"], dtype=np.float32)
                for row in support_rows
                if str(row["dataset"]) == str(target["dataset"])
            ]
            if vals:
                return np.mean(np.vstack(vals), axis=0).astype(np.float32)
        elif spec.no_overlap_fallback == "global_mean":
            vals = [np.asarray(row["residual"], dtype=np.float32) for row in support_rows]
            if vals:
                return np.mean(np.vstack(vals), axis=0).astype(np.float32)
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1]["dataset"]), str(item[1]["condition"])), reverse=True)
    selected = candidates[: max(1, int(spec.k))]
    weights = np.asarray([max(score, 1e-6) for score, _ in selected], dtype=np.float64)
    weights = weights / weights.sum()
    residuals = np.vstack([np.asarray(row["residual"], dtype=np.float32) for _, row in selected])
    return (weights[:, None] * residuals).sum(axis=0).astype(np.float32)


def fit_scalar(samples: list[tuple[np.ndarray, np.ndarray]], ridge: float) -> float:
    if not samples:
        return 0.0
    x = np.vstack([s[0] for s in samples]).astype(np.float64)
    y = np.vstack([s[1] for s in samples]).astype(np.float64)
    denom = float(np.sum(x * x) + float(ridge))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.sum(x * y) / denom, -1.0, 1.5))


def make_fit_samples(
    fit_rows: list[dict[str, Any]],
    memory_rows: list[dict[str, Any]],
    spec: EpisodeSpec,
    mod: Any,
    support_mod: Any,
    single: dict[str, Any],
    multi: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for row in fit_rows:
        context = weighted_context(row, memory_rows, spec)
        if context is None:
            continue
        route = route_vector(mod, support_mod, row, single, multi)
        out.append((context - route, np.asarray(row["residual"], dtype=np.float32) - route))
    return out


def score_row(
    row: dict[str, Any],
    memory_rows: list[dict[str, Any]],
    spec: EpisodeSpec,
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
    context = weighted_context(row, memory_rows, spec)
    if context is None:
        pred = route
        no_context = True
    else:
        pred = (route + alpha * (context - route)).astype(np.float32)
        no_context = False
    item = {
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "genes": sorted(genes(row)),
        "overlap_class": overlap_class(row, memory_rows),
        "candidate": support_mod.pp_score(row, pred, pert_means),
        "support_selected_route": support_mod.pp_score(row, route, pert_means),
        "no_context_noop": bool(no_context),
        "alpha": float(alpha),
    }
    if compute_mmd:
        for metric, value in support_mod.mmd_scores(row, pred).items():
            item[f"candidate__{metric}"] = value
        for metric, value in support_mod.mmd_scores(row, route).items():
            item[f"support_selected_route__{metric}"] = value
    return item


def pseudo_cv_rows(
    rows: list[dict[str, Any]],
    spec: EpisodeSpec,
    mod: Any,
    support_mod: Any,
    single: dict[str, Any],
    pert_means: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for heldout in rows:
        memory_rows = [r for r in rows if condition_key(r) != condition_key(heldout)]
        multi = support_mod.train_multi_components(memory_rows)
        fit_samples = make_fit_samples(memory_rows, memory_rows, spec, mod, support_mod, single, multi)
        alpha = fit_scalar(fit_samples, spec.ridge)
        item = score_row(
            heldout,
            memory_rows,
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
    residuals = [np.asarray(r["residual"], dtype=np.float32) for r in rows]
    order = rng.permutation(len(rows))
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        copy = dict(row)
        copy["residual"] = residuals[int(order[i])]
        out.append(copy)
    return out


def summarize_rows(mod: Any, rows: list[dict[str, Any]], *, n_boot: int, seed: int, include_mmd: bool) -> dict[str, Any]:
    pp = mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="pp", n_boot=n_boot, seed=seed)
    mmd = (
        mod.paired_bootstrap(rows, "candidate", "support_selected_route", metric="mmd_clamped", n_boot=n_boot, seed=seed + 101)
        if include_mmd
        else None
    )
    by_class: dict[str, dict[str, Any]] = {}
    for cls in sorted({str(r.get("overlap_class")) for r in rows}):
        sub = [r for r in rows if str(r.get("overlap_class")) == cls]
        by_class[cls] = {
            "n": len(sub),
            "pp_delta": mean_delta(sub, "candidate", "support_selected_route"),
            "mmd_delta": mean_delta(sub, "candidate__test_mmd_clamped", "support_selected_route__test_mmd_clamped")
            if include_mmd
            else None,
            "no_context_rows": int(sum(bool(r.get("no_context_noop")) for r in sub)),
        }
    by_ds = {}
    for ds in sorted({str(r["dataset"]) for r in rows}):
        sub = [r for r in rows if str(r["dataset"]) == ds]
        by_ds[ds] = {
            "n": len(sub),
            "pp_delta": mean_delta(sub, "candidate", "support_selected_route"),
            "mmd_delta": mean_delta(sub, "candidate__test_mmd_clamped", "support_selected_route__test_mmd_clamped")
            if include_mmd
            else None,
        }
    return {"paired_pp_delta": pp, "paired_mmd_delta": mmd, "by_overlap_class": by_class, "by_dataset": by_ds}


def mean_delta(rows: list[dict[str, Any]], candidate: str, baseline: str) -> float | None:
    vals = []
    for row in rows:
        if row.get(candidate) is not None and row.get(baseline) is not None:
            vals.append(float(row[candidate]) - float(row[baseline]))
    return None if not vals else float(np.mean(vals))


def select_spec(cv_summaries: list[dict[str, Any]]) -> str:
    eligible = []
    for item in cv_summaries:
        summ = item["summary"]
        zero = summ["by_overlap_class"].get("zero_overlap", {})
        pp = summ["paired_pp_delta"]
        norman = (summ["by_dataset"].get("NormanWeissman2019_filtered") or {}).get("pp_delta")
        if (
            float(zero.get("pp_delta") if zero.get("pp_delta") is not None else -999.0) >= 0.0
            and float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) <= 0.35
            and float(norman if norman is not None else -999.0) >= -0.02
        ):
            eligible.append(item)
    pool = eligible or cv_summaries
    pool = sorted(
        pool,
        key=lambda item: (
            num_or((item["summary"]["by_overlap_class"].get("zero_overlap") or {}).get("pp_delta"), -999.0),
            num_or(item["summary"]["paired_pp_delta"].get("delta_mean"), -999.0),
        ),
        reverse=True,
    )
    return str(pool[0]["spec"])


def num_or(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def decide(
    support_summary: dict[str, Any],
    shuffled_summary: dict[str, Any],
    *,
    n_support_rows: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    pp = support_summary["paired_pp_delta"]
    mmd = support_summary["paired_mmd_delta"] or {}
    zero = support_summary["by_overlap_class"].get("zero_overlap", {})
    norman = support_summary["by_dataset"].get("NormanWeissman2019_filtered", {})
    wessels = support_summary["by_dataset"].get("Wessels", {})
    shuf_pp = shuffled_summary["paired_pp_delta"]
    if n_support_rows != 24:
        reasons.append("support_val_coverage_not_complete")
    if float(zero.get("pp_delta") if zero.get("pp_delta") is not None else -999.0) < 0.02:
        reasons.append("zero_overlap_pseudo_query_delta_below_0p02")
    if float(pp.get("delta_mean") if pp.get("delta_mean") is not None else -999.0) <= 0.0:
        reasons.append("aggregate_support_pp_delta_not_positive")
    if float(pp.get("p_harm") if pp.get("p_harm") is not None else 1.0) > 0.20:
        reasons.append("support_pp_harm_above_0p20")
    if float(mmd.get("delta_mean") if mmd.get("delta_mean") is not None else 999.0) > 0.0:
        reasons.append("support_mmd_not_nonharm")
    if float(mmd.get("p_harm") if mmd.get("p_harm") is not None else 1.0) > 0.80:
        reasons.append("support_mmd_hard_harm")
    if float(norman.get("pp_delta") if norman.get("pp_delta") is not None else -999.0) < -0.02:
        reasons.append("norman_dataset_min_below_minus_0p02")
    if float(wessels.get("pp_delta") if wessels.get("pp_delta") is not None else -999.0) < 0.02:
        reasons.append("wessels_delta_below_0p02")
    sep = float(pp.get("delta_mean") or 0.0) - float(shuf_pp.get("delta_mean") or 0.0)
    if sep < 0.02:
        reasons.append("shuffled_support_control_not_separated_by_0p02")
    status = "trackc_pseudo_episode_gate_pass_code_gate_next_no_gpu" if not reasons else "trackc_pseudo_episode_gate_fail_no_gpu"
    return {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "code_launcher_gate_only" if not reasons else "none",
        "reasons": reasons,
        "zero_overlap_pp_delta": zero.get("pp_delta"),
        "aggregate_pp_delta": pp.get("delta_mean"),
        "aggregate_mmd_delta": mmd.get("delta_mean"),
        "norman_pp_delta": norman.get("pp_delta"),
        "wessels_pp_delta": wessels.get("pp_delta"),
        "shuffled_pp_delta": shuf_pp.get("delta_mean"),
        "candidate_minus_shuffled_pp_delta": sep,
    }


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Track C Pseudo-Episode Gate",
        "",
        f"Status: `{decision['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- CPU-only query-free gate.",
        "- Selection uses safe-trainselect train_multi pseudo episodes only.",
        "- support_val_multi is final scoring only.",
        "- Held-out query, canonical test, canonical multi, active logs, and GPU artifacts are not read.",
        "",
        "## Selected Spec",
        "",
        f"- `{payload['selected_spec']}`",
        f"- train_multi rows: `{payload['n_train_multi_rows']}`",
        f"- support_val_multi rows: `{payload['n_support_val_rows']}`",
        "",
        "## Support-Val Gate Criteria",
        "",
        f"- zero-overlap pp delta: `{fmt(decision['zero_overlap_pp_delta'])}` (gate `>= +0.020000`)",
        f"- aggregate pp delta: `{fmt(decision['aggregate_pp_delta'])}` (gate `> 0`)",
        f"- aggregate MMD delta: `{fmt(decision['aggregate_mmd_delta'])}` (gate `<= 0`)",
        f"- Norman pp delta: `{fmt(decision['norman_pp_delta'])}` (gate `>= -0.020000`)",
        f"- Wessels pp delta: `{fmt(decision['wessels_pp_delta'])}` (gate `>= +0.020000`)",
        f"- candidate minus shuffled pp delta: `{fmt(decision['candidate_minus_shuffled_pp_delta'])}` (gate `>= +0.020000`)",
        "",
        "## Support-Val Breakdown",
        "",
        "### By Overlap Class",
        "",
        "| class | n | pp delta | MMD delta | no-context rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls, row in payload["support_summary"]["by_overlap_class"].items():
        lines.append(
            f"| `{cls}` | {row['n']} | {fmt(row.get('pp_delta'))} | {fmt(row.get('mmd_delta'))} | {row['no_context_rows']} |"
        )
    lines.extend(["", "### By Dataset", "", "| dataset | n | pp delta | MMD delta |", "|---|---:|---:|---:|"])
    for ds, row in payload["support_summary"]["by_dataset"].items():
        lines.append(f"| `{ds}` | {row['n']} | {fmt(row.get('pp_delta'))} | {fmt(row.get('mmd_delta'))} |")
    lines.extend(
        [
            "",
            "## Train Pseudo-Episode Selection",
            "",
            "| spec | aggregate pp | zero-overlap pp | Norman pp | Wessels pp | pp p_harm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["cv_summaries"][:12]:
        summ = item["summary"]
        zero = summ["by_overlap_class"].get("zero_overlap", {})
        norman = summ["by_dataset"].get("NormanWeissman2019_filtered", {})
        wessels = summ["by_dataset"].get("Wessels", {})
        marker = " (selected)" if item["spec"] == payload["selected_spec"] else ""
        lines.append(
            f"| `{item['spec']}`{marker} | {fmt(summ['paired_pp_delta'].get('delta_mean'))} | "
            f"{fmt(zero.get('pp_delta'))} | {fmt(norman.get('pp_delta'))} | "
            f"{fmt(wessels.get('pp_delta'))} | {fmt(summ['paired_pp_delta'].get('p_harm'))} |"
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
            "Failure means the current support-context family remains a frozen diagnostic route; "
            "do not expand it toward unseen2-like generalization without a materially new query-free signal.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/latentfm_full/xverse")
    parser.add_argument(
        "--split-file",
        type=Path,
        default=ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json",
    )
    parser.add_argument(
        "--pert-means-file",
        type=Path,
        default=ROOT
        / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
    )
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
    pert_means = {k: v.astype(np.float32) for k, v in np.load(args.pert_means_file).items()}
    train_rows = support_mod.collect_role_rows(args.data_dir, split, metadata, "train_multi", max_cells=args.max_cells_per_condition)
    support_val = support_mod.collect_role_rows(args.data_dir, split, metadata, "support_val_multi", max_cells=args.max_cells_per_condition)
    single = support_mod.train_single_components(args.data_dir, split, metadata, max_cells=args.max_cells_per_condition)

    cv_summaries: list[dict[str, Any]] = []
    cv_rows_by_spec: dict[str, list[dict[str, Any]]] = {}
    for spec in specs():
        rows = pseudo_cv_rows(train_rows, spec, mod, support_mod, single, pert_means)
        cv_rows_by_spec[spec.name] = rows
        cv_summaries.append({"spec": spec.name, "summary": summarize_rows(mod, rows, n_boot=args.n_boot, seed=args.seed, include_mmd=False)})
    cv_summaries = sorted(
        cv_summaries,
        key=lambda item: (
            num_or((item["summary"]["by_overlap_class"].get("zero_overlap") or {}).get("pp_delta"), -999.0),
            num_or(item["summary"]["paired_pp_delta"].get("delta_mean"), -999.0),
        ),
        reverse=True,
    )
    selected_name = select_spec(cv_summaries)
    selected_spec = next(s for s in specs() if s.name == selected_name)

    train_multi_components = support_mod.train_multi_components(train_rows)
    fit_samples = make_fit_samples(train_rows, train_rows, selected_spec, mod, support_mod, single, train_multi_components)
    alpha = fit_scalar(fit_samples, selected_spec.ridge)
    support_rows = [
        score_row(
            row,
            train_rows,
            selected_spec,
            alpha,
            mod,
            support_mod,
            single,
            train_multi_components,
            pert_means,
            compute_mmd=True,
        )
        for row in support_val
    ]
    support_summary = summarize_rows(mod, support_rows, n_boot=args.n_boot, seed=args.seed + 11, include_mmd=True)

    shuffled_train = shuffled_support(train_rows, args.seed + 222)
    shuffled_multi = support_mod.train_multi_components(shuffled_train)
    shuffled_fit = make_fit_samples(shuffled_train, shuffled_train, selected_spec, mod, support_mod, single, shuffled_multi)
    shuffled_alpha = fit_scalar(shuffled_fit, selected_spec.ridge)
    shuffled_rows = [
        score_row(
            row,
            shuffled_train,
            selected_spec,
            shuffled_alpha,
            mod,
            support_mod,
            single,
            shuffled_multi,
            pert_means,
            compute_mmd=True,
        )
        for row in support_val
    ]
    shuffled_summary = summarize_rows(mod, shuffled_rows, n_boot=args.n_boot, seed=args.seed + 333, include_mmd=True)

    decision = decide(support_summary, shuffled_summary, n_support_rows=len(support_rows))
    payload = {
        "status": decision["status"],
        "boundary": {
            "reads_raw_heldout_query": False,
            "reads_canonical_test": False,
            "reads_canonical_multi_for_selection": False,
            "reads_active_logs": False,
            "launches_gpu": False,
            "selection": "train_multi_pseudo_episode_only",
            "final_scoring": "support_val_multi_only",
        },
        "split_guard": guard,
        "data_dir": str(args.data_dir),
        "pert_means_file": str(args.pert_means_file),
        "max_cells_per_condition": args.max_cells_per_condition,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_train_multi_rows": len(train_rows),
        "n_support_val_rows": len(support_val),
        "selected_spec": selected_name,
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
