#!/usr/bin/env python3
"""CPU gate for Track C support-neighbor residual transfer.

This query-free gate tests a condition-level support mechanism distinct from
the failed dataset-mean support summary. For each target multi condition, it
finds nearby train_multi support conditions by perturbation-gene overlap and
transfers their candidate-minus-anchor residual vector to the anchor
prediction.

The gate uses only safe-trainselect train_multi for specification selection and
support_val_multi for final scoring. It does not read held-out query,
canonical test, canonical multi, active logs, or GPU artifacts.
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
SUMMARY_MODULE_PATH = ROOT / "ops/summarize_latentfm_trackc_support_set_task_summary_gate_20260623.py"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_neighbor_residual_transfer_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_NEIGHBOR_RESIDUAL_TRANSFER_GATE_20260623.md"


@dataclass(frozen=True)
class NeighborSpec:
    name: str
    alpha: float
    k: int
    similarity: str
    same_dataset: bool


def load_summary_module() -> Any:
    spec = importlib.util.spec_from_file_location("support_set_summary_gate", SUMMARY_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SUMMARY_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def specs() -> list[NeighborSpec]:
    out: list[NeighborSpec] = []
    for alpha in (0.25, 0.50, 0.75, 1.00):
        for k in (1, 2, 3, 5, 10):
            for similarity in ("overlap", "jaccard"):
                for same_dataset in (True, False):
                    tag = "same_ds" if same_dataset else "all_ds"
                    out.append(NeighborSpec(f"neighbor_{similarity}_k{k}_{tag}_alpha{alpha:g}", alpha, k, similarity, same_dataset))
    return out


def condition_genes(condition: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(condition).split("+") if part.strip())


def similarity(a: str, b: str, kind: str) -> float:
    left = set(condition_genes(a))
    right = set(condition_genes(b))
    if not left or not right:
        return 0.0
    inter = len(left & right)
    if kind == "overlap":
        return float(inter)
    if kind == "jaccard":
        return float(inter) / float(len(left | right))
    raise ValueError(kind)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def transfer_prediction(
    mod: Any,
    target: dict[str, Any],
    pool: list[dict[str, Any]],
    spec: NeighborSpec,
    *,
    residual_override: dict[tuple[str, str], np.ndarray] | None = None,
) -> float | None:
    candidates = [row for row in pool if (not spec.same_dataset or row["dataset"] == target["dataset"])]
    scored = [
        (similarity(str(target["condition"]), str(row["condition"]), spec.similarity), row)
        for row in candidates
    ]
    scored = [(score, row) for score, row in scored if score > 0.0]
    if not scored:
        return target.get("anchor_pp")
    scored = sorted(scored, key=lambda item: item[0], reverse=True)[: int(spec.k)]
    weights = np.asarray([score for score, _ in scored], dtype=np.float64)
    weights = weights / weights.sum()
    residuals = []
    for _, row in scored:
        key = (str(row["dataset"]), str(row["condition"]))
        if residual_override and key in residual_override:
            residuals.append(np.asarray(residual_override[key], dtype=np.float32))
        else:
            residuals.append(np.asarray(row["residual"], dtype=np.float32))
    correction = (np.stack(residuals, axis=0) * weights[:, None]).sum(axis=0)
    pred = target["pred_anchor"] + float(spec.alpha) * correction
    return mod.pearson_np(pred - target["pert_mean"], target["gt_mean"] - target["pert_mean"])


def score_rows(
    mod: Any,
    rows: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    spec: NeighborSpec,
    *,
    loo: bool = False,
    residual_override: dict[tuple[str, str], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        use_pool = [
            r for r in pool
            if not (loo and r["dataset"] == row["dataset"] and r["condition"] == row["condition"])
        ]
        pp = transfer_prediction(mod, row, use_pool, spec, residual_override=residual_override)
        out.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "anchor_pp": row["anchor_pp"],
                "task_pp": pp,
                "delta_vs_anchor": None if pp is None or row["anchor_pp"] is None else pp - row["anchor_pp"],
            }
        )
    return out


def summary(mod: Any, rows: list[dict[str, Any]], route_gaps: dict[str, float], spec: NeighborSpec, seed: int) -> dict[str, Any]:
    payload = mod.summary_for(rows, route_gaps, spec.alpha, seed=seed)
    payload.update(
        {
            "spec": spec.name,
            "alpha": float(spec.alpha),
            "k": int(spec.k),
            "similarity": spec.similarity,
            "same_dataset": bool(spec.same_dataset),
        }
    )
    return payload


def train_passes(mod: Any, item: dict[str, Any]) -> bool:
    return bool(mod.support_gate_passes(item))


def select_spec(mod: Any, train_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [item for item in train_summaries if train_passes(mod, item)]
    if not passing:
        return None
    return sorted(
        passing,
        key=lambda item: (
            float((mod.find_dataset(item, "Wessels") or {}).get("route_gap_closed_fraction") or -999.0),
            float((mod.find_dataset(item, "Wessels") or {}).get("mean_delta_pp") or -999.0),
            float((item.get("paired") or {}).get("delta_mean") or -999.0),
        ),
        reverse=True,
    )[0]


def shuffled_residuals(train_rows: list[dict[str, Any]], seed: int) -> dict[tuple[str, str], np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = [(str(row["dataset"]), str(row["condition"])) for row in train_rows]
    residuals = [np.asarray(row["residual"], dtype=np.float32) for row in train_rows]
    order = rng.permutation(len(keys))
    return {key: residuals[int(order[i])] for i, key in enumerate(keys)}


def support_gate_decision(mod: Any, support: dict[str, Any], zero: dict[str, Any] | None, shuffled: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    if not train_passes(mod, support):
        reasons.append("support_val_gate_failed")
    if zero is not None and train_passes(mod, zero):
        reasons.append("zero_support_control_passed_unexpectedly")
    if shuffled is not None:
        if train_passes(mod, shuffled):
            reasons.append("shuffled_residual_control_passed_unexpectedly")
        support_w = float((mod.find_dataset(support, "Wessels") or {}).get("mean_delta_pp") or 0.0)
        shuffled_w = float((mod.find_dataset(shuffled, "Wessels") or {}).get("mean_delta_pp") or 0.0)
        if shuffled_w >= 0.02 or (support_w > 0.0 and shuffled_w >= 0.5 * support_w):
            reasons.append("shuffled_residual_control_did_not_lose_wessels_signal")
    return {
        "status": (
            "trackc_support_neighbor_residual_transfer_gate_pass_posthoc_mmd_gate_next_no_gpu"
            if not reasons
            else "trackc_support_neighbor_residual_transfer_gate_fail_no_gpu"
        ),
        "gpu_authorization": "none",
        "next_authorization": "query_free_posthoc_mmd_gate_only" if not reasons else "none",
        "reasons": reasons,
    }


def build_payload(run_root: Path, route_gap_json: Path, seed: int) -> dict[str, Any]:
    mod = load_summary_module()
    cm_dir = run_root / "condition_means"
    anchor_path = cm_dir / "trainselect_anchor_train_support_multi_condition_means_ode20.json"
    candidate_path = cm_dir / "trainselect_candidate_train_support_multi_condition_means_ode20.json"
    anchor = mod.load_json(anchor_path)
    candidate = mod.load_json(candidate_path)
    train_rows = mod.paired_rows(anchor, candidate, "train_multi")
    support_rows = mod.paired_rows(anchor, candidate, "support_val_multi")
    route_gaps = mod.route_gap_by_dataset(route_gap_json)

    train_summaries = []
    lookup: dict[str, NeighborSpec] = {}
    for idx, spec in enumerate(specs()):
        lookup[spec.name] = spec
        rows = score_rows(mod, train_rows, train_rows, spec, loo=True)
        train_summaries.append(summary(mod, rows, route_gaps, spec, seed=seed + idx))
    selected = select_spec(mod, train_summaries)
    support_summary = None
    zero_summary = None
    shuffled_summary = None
    if selected is not None:
        spec = lookup[str(selected["spec"])]
        support_summary = summary(
            mod,
            score_rows(mod, support_rows, train_rows, spec),
            route_gaps,
            spec,
            seed=seed + 2000,
        )
        zero_spec = NeighborSpec(f"{spec.name}_zero_alpha_control", 0.0, spec.k, spec.similarity, spec.same_dataset)
        zero_summary = summary(
            mod,
            score_rows(mod, support_rows, train_rows, zero_spec),
            route_gaps,
            zero_spec,
            seed=seed + 2001,
        )
        shuffled_summary = summary(
            mod,
            score_rows(
                mod,
                support_rows,
                train_rows,
                spec,
                residual_override=shuffled_residuals(train_rows, seed + 2002),
            ),
            route_gaps,
            spec,
            seed=seed + 2003,
        )
    decision = (
        {
            "status": "trackc_support_neighbor_residual_transfer_gate_fail_no_gpu",
            "gpu_authorization": "none",
            "next_authorization": "none",
            "reasons": ["no_spec_passed_train_multi_loo_gate"],
        }
        if support_summary is None
        else support_gate_decision(mod, support_summary, zero_summary, shuffled_summary)
    )
    return {
        "run_root": str(run_root),
        "inputs": {
            "anchor_condition_means": str(anchor_path),
            "candidate_condition_means": str(candidate_path),
            "route_gap_json": str(route_gap_json),
            "summary_gate_module": str(SUMMARY_MODULE_PATH),
        },
        "boundary": {
            "safe_trainselect_only": True,
            "heldout_query_read": False,
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
            "selection_or_tuning_inputs": "train_multi_loo_only",
            "support_val_role": "final_query_free_gate_scoring_only",
        },
        "n_rows": {"train_multi": len(train_rows), "support_val_multi": len(support_rows)},
        "spec_grid_size": len(specs()),
        "train_loo_summaries": train_summaries,
        "selected_train_loo_summary": selected,
        "support_val_summary": support_summary,
        "zero_support_control": zero_summary,
        "shuffled_residual_control": shuffled_summary,
        "decision": decision,
    }


def render_dataset_table(mod: Any, summary_payload: dict[str, Any] | None) -> list[str]:
    if not summary_payload:
        return ["- not evaluated", ""]
    lines = [
        f"- spec: `{summary_payload['spec']}`",
        f"- paired pp delta: `{fmt((summary_payload.get('paired') or {}).get('delta_mean'))}`",
        f"- paired pp p_harm: `{fmt((summary_payload.get('paired') or {}).get('p_harm'))}`",
        "",
        "| dataset | n | mean delta pp | route gap | closure |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_payload.get("dataset_summary") or []:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {fmt(row.get('mean_delta_pp'))} | "
            f"{fmt(row.get('route_gap_pp'))} | {fmt(row.get('route_gap_closed_fraction'))} |"
        )
    lines.append("")
    return lines


def render(payload: dict[str, Any]) -> str:
    mod = load_summary_module()
    decision = payload["decision"]
    selected = payload.get("selected_train_loo_summary")
    lines = [
        "# Track C Support-Neighbor Residual Transfer Gate",
        "",
        f"Status: `{decision['status']}`",
        f"GPU authorization: `{decision['gpu_authorization']}`",
        f"Next authorization: `{decision['next_authorization']}`",
        "",
        "## Scope",
        "",
        "This query-free CPU gate tests condition-level support-neighbor residual transfer on safe trainselect multi rows. It is distinct from the failed dataset-mean support summary because each target condition receives residuals from nearest train_multi support conditions by perturbation-gene overlap/Jaccard.",
        "",
        "## Boundary",
        "",
        "- selection uses train_multi leave-one-condition only",
        "- support_val_multi is used once for final query-free gate scoring",
        "- held-out query, canonical test, canonical multi, active logs, and GPU artifacts are not read",
        "- passing would still authorize only a later query-free MMD/no-harm posthoc gate, not GPU training or query",
        "",
        "## Row Counts",
        "",
        f"- train_multi: `{payload['n_rows']['train_multi']}`",
        f"- support_val_multi: `{payload['n_rows']['support_val_multi']}`",
        f"- spec grid size: `{payload['spec_grid_size']}`",
        "",
        "## Selected Train-Multi LOO Spec",
        "",
    ]
    if selected:
        wessels = mod.find_dataset(selected, "Wessels")
        norman = mod.find_dataset(selected, "NormanWeissman2019_filtered")
        lines.extend(
            [
                f"- spec: `{selected['spec']}`",
                f"- paired pp delta: `{fmt((selected.get('paired') or {}).get('delta_mean'))}`",
                f"- paired pp p_harm: `{fmt((selected.get('paired') or {}).get('p_harm'))}`",
                f"- Wessels delta/closure: `{fmt(wessels.get('mean_delta_pp'))}` / `{fmt(wessels.get('route_gap_closed_fraction'))}`",
                f"- Norman delta: `{fmt(norman.get('mean_delta_pp'))}`",
                "",
            ]
        )
    else:
        lines.extend(["- none", ""])
    for title, key in (
        ("Support-Val Summary", "support_val_summary"),
        ("Zero-Support Control", "zero_support_control"),
        ("Shuffled-Residual Control", "shuffled_residual_control"),
    ):
        lines.extend([f"## {title}", ""])
        lines.extend(render_dataset_table(mod, payload.get(key)))
    lines.extend(["## Decision Reasons", ""])
    reasons = decision.get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A train-only neighbor residual rule can look strong in train_multi LOO, but promotion requires support_val Wessels gain and route-gap closure. Failure here is negative evidence against this condition-neighbor residual-transfer mechanism, not against all future support-set task adapters.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=load_summary_module().DEFAULT_RUN_ROOT)
    parser.add_argument("--route-gap-json", type=Path, default=load_summary_module().CPU_ROUTE_GAP_JSON)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload(args.run_root, args.route_gap_json, args.seed)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
