#!/usr/bin/env python3
"""Query-free Track C response-derived nonadditivity CPU gate.

This gate tests whether response-derived interaction residuals

    multi_delta - sum(train_single_gene_delta)

can safely improve the support-selected route. The candidate is

    support_selected_route + alpha * transferred_interaction_residual

where alpha and the transfer spec are selected using safe-trainselect
train_multi leave-one-condition episodes only. support_val_multi is final
scoring only.

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
OUT_JSON = ROOT / "reports/latentfm_trackc_response_nonadditivity_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_RESPONSE_NONADDITIVITY_GATE_20260624.md"
EXPECTED_TRAINSELECT_SHA256 = "5f29dd5b582a40da3736770ca29950c12c54e46b590c3f7705c1d37da89f4f20"
FOCUS_DATASETS = ("NormanWeissman2019_filtered", "Wessels")


@dataclass(frozen=True)
class NonaddSpec:
    name: str
    memory_mode: str
    k: int
    same_dataset: bool
    min_score: float
    require_source_full_additive: bool = True
    ridge: float = 0.0


def load_residual_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_residual_operator_gate", RESIDUAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {RESIDUAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def specs() -> list[NonaddSpec]:
    out: list[NonaddSpec] = []
    for mode, min_score in (("overlap", 1.0), ("jaccard", 0.25)):
        for same_dataset in (True, False):
            for k in (1, 3, 5):
                out.append(
                    NonaddSpec(
                        name=f"{mode}_k{k}_{'same_ds' if same_dataset else 'all_ds'}_fullsrc",
                        memory_mode=mode,
                        k=k,
                        same_dataset=same_dataset,
                        min_score=min_score,
                        require_source_full_additive=True,
                    )
                )
    return out


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def genes(row: dict[str, Any]) -> list[str]:
    return [str(g).strip().upper() for g in row.get("genes") or [] if str(g).strip()]


def gene_set(row: dict[str, Any]) -> set[str]:
    return set(genes(row))


def memory_score(target: dict[str, Any], row: dict[str, Any], mode: str) -> float:
    a = gene_set(target)
    b = gene_set(row)
    if mode == "overlap":
        return float(len(a & b))
    if mode == "jaccard":
        return float(len(a & b) / max(len(a | b), 1))
    raise ValueError(mode)


def route_vector(mod: Any, support_mod: Any, row: dict[str, Any], single: dict[str, Any], multi: dict[str, Any]) -> np.ndarray:
    return mod.route_residual(row, single, multi, support_mod)


def additive_sum(row: dict[str, Any], single: dict[str, Any]) -> tuple[np.ndarray | None, int, int]:
    bank = single.get("gene_raw_mean") or {}
    parts = []
    for gene in genes(row):
        value = bank.get(gene)
        if value is not None:
            parts.append(np.asarray(value, dtype=np.float32))
    total = len(genes(row))
    if total == 0 or len(parts) != total:
        return None, len(parts), total
    return np.sum(np.vstack(parts), axis=0).astype(np.float32), len(parts), total


def attach_interactions(rows: list[dict[str, Any]], single: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        copy = dict(row)
        add, found, total = additive_sum(row, single)
        copy["additive_found_genes"] = int(found)
        copy["additive_total_genes"] = int(total)
        copy["additive_full_coverage"] = bool(total > 0 and found == total)
        if add is None:
            copy["interaction_residual"] = None
        else:
            copy["interaction_residual"] = (np.asarray(row["residual"], dtype=np.float32) - add).astype(np.float32)
        out.append(copy)
    return out


def select_memory(target: dict[str, Any], memory_rows: list[dict[str, Any]], spec: NonaddSpec) -> list[tuple[float, dict[str, Any]]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in memory_rows:
        if condition_key(row) == condition_key(target):
            continue
        if spec.same_dataset and str(row["dataset"]) != str(target["dataset"]):
            continue
        if row.get("interaction_residual") is None:
            continue
        if spec.require_source_full_additive and not bool(row.get("additive_full_coverage")):
            continue
        score = memory_score(target, row, spec.memory_mode)
        if score >= spec.min_score:
            candidates.append((score, row))
    candidates.sort(key=lambda item: (item[0], str(item[1]["dataset"]), str(item[1]["condition"])), reverse=True)
    return candidates[: max(1, int(spec.k))]


def weighted_interaction(selected: list[tuple[float, dict[str, Any]]]) -> np.ndarray | None:
    if not selected:
        return None
    weights = np.asarray([max(score, 1e-6) for score, _row in selected], dtype=np.float64)
    weights = weights / weights.sum()
    vals = np.vstack([np.asarray(row["interaction_residual"], dtype=np.float32) for _score, row in selected])
    return (weights[:, None] * vals).sum(axis=0).astype(np.float32)


def context_interaction(target: dict[str, Any], memory_rows: list[dict[str, Any]], spec: NonaddSpec) -> tuple[np.ndarray | None, int]:
    selected = select_memory(target, memory_rows, spec)
    return weighted_interaction(selected), len(selected)


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
    spec: NonaddSpec,
    mod: Any,
    support_mod: Any,
    single: dict[str, Any],
    multi: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for row in rows:
        x, _n = context_interaction(row, memory_rows, spec)
        if x is None:
            continue
        route = route_vector(mod, support_mod, row, single, multi)
        y = np.asarray(row["residual"], dtype=np.float32) - route
        out.append((x, y))
    return out


def score_row(
    row: dict[str, Any],
    memory_rows: list[dict[str, Any]],
    spec: NonaddSpec,
    alpha: float,
    mod: Any,
    support_mod: Any,
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    *,
    compute_mmd: bool,
    sign: float = 1.0,
) -> dict[str, Any]:
    route = route_vector(mod, support_mod, row, single, multi)
    x, n_context = context_interaction(row, memory_rows, spec)
    if x is None:
        pred = route
        abstained = True
    else:
        pred = (route + float(sign) * alpha * x).astype(np.float32)
        abstained = False
    item = {
        "dataset": str(row["dataset"]),
        "condition": str(row["condition"]),
        "genes": genes(row),
        "candidate": support_mod.pp_score(row, pred, pert_means),
        "support_selected_route": support_mod.pp_score(row, route, pert_means),
        "abstained": bool(abstained),
        "n_context_rows": int(n_context),
        "alpha": float(alpha),
        "sign": float(sign),
        "target_additive_full_coverage": bool(row.get("additive_full_coverage")),
    }
    if compute_mmd:
        for metric, value in support_mod.mmd_scores(row, pred).items():
            item[f"candidate__{metric}"] = value
        for metric, value in support_mod.mmd_scores(row, route).items():
            item[f"support_selected_route__{metric}"] = value
    return item


def cv_rows(
    train_rows: list[dict[str, Any]],
    spec: NonaddSpec,
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


def shuffled_interactions(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    interactions = [row.get("interaction_residual") for row in rows]
    valid = [x for x in interactions if x is not None]
    if not valid:
        return [dict(row) for row in rows]
    order = rng.permutation(len(valid))
    shuffled_valid = [valid[int(i)] for i in order]
    idx = 0
    out = []
    for row in rows:
        copy = dict(row)
        if copy.get("interaction_residual") is not None:
            copy["interaction_residual"] = shuffled_valid[idx]
            idx += 1
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
            "enabled": int(sum(not bool(row.get("abstained")) for row in sub)),
            "pp_delta": mean_delta(sub, "candidate", "support_selected_route"),
            "mmd_delta": mean_delta(sub, "candidate__test_mmd_clamped", "support_selected_route__test_mmd_clamped")
            if include_mmd
            else None,
        }
    enabled = [row for row in rows if not bool(row.get("abstained"))]
    enabled_deltas = [
        float(row["candidate"]) - float(row["support_selected_route"])
        for row in enabled
        if row.get("candidate") is not None and row.get("support_selected_route") is not None
    ]
    return {
        "paired_pp_delta": pp,
        "paired_mmd_delta": mmd,
        "by_dataset": by_ds,
        "enabled_rows": len(enabled),
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
            num_or(summ["paired_pp_delta"].get("delta_mean"), -999.0) >= 0.02
            and num_or(summ["paired_pp_delta"].get("p_harm"), 1.0) <= 0.20
            and num_or(norman, -999.0) >= -0.01
            and num_or(wessels, -999.0) >= 0.02
            and summ["enabled_rows"] >= 8
            and summ["enabled_negative_rows"] <= 2
        ):
            eligible.append(item)
    pool = eligible or summaries
    pool = sorted(
        pool,
        key=lambda item: (
            -item["summary"]["enabled_negative_rows"],
            num_or(item["summary"]["paired_pp_delta"].get("delta_mean"), -999.0),
            num_or((item["summary"]["by_dataset"].get("Wessels") or {}).get("pp_delta"), -999.0),
        ),
        reverse=True,
    )
    return str(pool[0]["spec"])


def decide(summary: dict[str, Any], shuffled: dict[str, Any], inverted: dict[str, Any], *, route_gap: float | None) -> dict[str, Any]:
    reasons = []
    pp = summary["paired_pp_delta"]
    mmd = summary["paired_mmd_delta"] or {}
    norman = summary["by_dataset"].get("NormanWeissman2019_filtered", {})
    wessels = summary["by_dataset"].get("Wessels", {})
    wessels_delta = num_or(wessels.get("pp_delta"), -999.0)
    closure = None if route_gap is None or abs(route_gap) < 1e-12 else wessels_delta / route_gap
    shuffled_sep = num_or(pp.get("delta_mean"), 0.0) - num_or(shuffled["paired_pp_delta"].get("delta_mean"), 0.0)
    inverted_sep = num_or(pp.get("delta_mean"), 0.0) - num_or(inverted["paired_pp_delta"].get("delta_mean"), 0.0)
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
    if shuffled_sep < 0.02:
        reasons.append("shuffled_interaction_control_not_separated_by_0p02")
    if inverted_sep < 0.02:
        reasons.append("sign_inverted_control_not_separated_by_0p02")
    status = "trackc_response_nonadditivity_gate_pass_code_gate_next_no_gpu" if not reasons else "trackc_response_nonadditivity_gate_fail_no_gpu"
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
        "inverted_pp_delta": inverted["paired_pp_delta"].get("delta_mean"),
        "candidate_minus_shuffled_pp_delta": shuffled_sep,
        "candidate_minus_inverted_pp_delta": inverted_sep,
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
        "# Track C Response-Derived Nonadditivity Gate",
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
        f"- train interaction coverage: `{payload['train_interaction_coverage']}`",
        f"- support interaction coverage: `{payload['support_interaction_coverage']}`",
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
        f"- candidate minus sign-inverted pp: `{fmt(decision['candidate_minus_inverted_pp_delta'])}` (gate `>= +0.020000`)",
        "",
        "## Support-Val Dataset Breakdown",
        "",
        "| dataset | n | enabled | pp delta | MMD delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for ds, row in payload["support_summary"]["by_dataset"].items():
        lines.append(f"| `{ds}` | {row['n']} | {row['enabled']} | {fmt(row.get('pp_delta'))} | {fmt(row.get('mmd_delta'))} |")
    lines.extend(
        [
            "",
            "## Train-CV Selection Summary",
            "",
            "| spec | pp delta | Norman | Wessels | enabled | negative | p_harm |",
            "|---|---:|---:|---:|---:|---:|---:|",
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
            f"{summ['enabled_rows']} | {summ['enabled_negative_rows']} | "
            f"{fmt(summ['paired_pp_delta'].get('p_harm'))} |"
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
            "Failure means response-derived nonadditivity does not currently provide a safe "
            "query-free Track C expansion route.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    full = int(sum(bool(row.get("additive_full_coverage")) for row in rows))
    by_ds = {}
    for ds in sorted({str(row["dataset"]) for row in rows}):
        sub = [row for row in rows if str(row["dataset"]) == ds]
        by_ds[ds] = {
            "n": len(sub),
            "full": int(sum(bool(row.get("additive_full_coverage")) for row in sub)),
        }
    return {"n": total, "full": full, "fraction": full / max(total, 1), "by_dataset": by_ds}


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
    raw_train_rows = support_mod.collect_role_rows(args.data_dir, split, metadata, "train_multi", max_cells=args.max_cells_per_condition)
    raw_support_val = support_mod.collect_role_rows(args.data_dir, split, metadata, "support_val_multi", max_cells=args.max_cells_per_condition)
    single = support_mod.train_single_components(args.data_dir, split, metadata, max_cells=args.max_cells_per_condition)
    train_rows = attach_interactions(raw_train_rows, single)
    support_val = attach_interactions(raw_support_val, single)

    cv_summaries = []
    for spec in specs():
        rows = cv_rows(train_rows, spec, mod, support_mod, single, pert_means)
        cv_summaries.append({"spec": spec.name, "summary": summarize(mod, rows, n_boot=args.n_boot, seed=args.seed, include_mmd=False)})
    cv_summaries = sorted(
        cv_summaries,
        key=lambda item: (
            -item["summary"]["enabled_negative_rows"],
            num_or(item["summary"]["paired_pp_delta"].get("delta_mean"), -999.0),
            num_or((item["summary"]["by_dataset"].get("Wessels") or {}).get("pp_delta"), -999.0),
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
    support_summary = summarize(mod, support_rows, n_boot=args.n_boot, seed=args.seed + 19, include_mmd=True)

    shuffled_train = shuffled_interactions(train_rows, args.seed + 503)
    shuffled_multi = support_mod.train_multi_components(shuffled_train)
    shuffled_alpha = fit_alpha(
        fit_samples(shuffled_train, shuffled_train, selected_spec, mod, support_mod, single, shuffled_multi),
        selected_spec.ridge,
    )
    shuffled_rows = [
        score_row(row, shuffled_train, selected_spec, shuffled_alpha, mod, support_mod, single, shuffled_multi, pert_means, compute_mmd=True)
        for row in support_val
    ]
    shuffled_summary = summarize(mod, shuffled_rows, n_boot=args.n_boot, seed=args.seed + 607, include_mmd=True)

    inverted_rows = [
        score_row(row, train_rows, selected_spec, alpha, mod, support_mod, single, multi, pert_means, compute_mmd=True, sign=-1.0)
        for row in support_val
    ]
    inverted_summary = summarize(mod, inverted_rows, n_boot=args.n_boot, seed=args.seed + 709, include_mmd=True)
    route_gap = mod.readout_wessels_route_gap(args.readout_json)
    decision = decide(support_summary, shuffled_summary, inverted_summary, route_gap=route_gap)

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
        "train_interaction_coverage": coverage(train_rows),
        "support_interaction_coverage": coverage(support_val),
        "selected_spec": selected_name,
        "selected_spec_config": selected_spec.__dict__,
        "selected_alpha": alpha,
        "shuffled_alpha": shuffled_alpha,
        "cv_summaries": cv_summaries,
        "support_summary": support_summary,
        "shuffled_summary": shuffled_summary,
        "inverted_summary": inverted_summary,
        "support_rows": support_rows,
        "shuffled_rows": shuffled_rows,
        "inverted_rows": inverted_rows,
        "decision": decision,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(render(payload))
    print(json.dumps({"status": decision["status"], "gpu_authorization": decision["gpu_authorization"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
