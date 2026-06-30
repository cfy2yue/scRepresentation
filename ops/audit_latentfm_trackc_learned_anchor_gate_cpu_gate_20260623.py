#!/usr/bin/env python3
"""Query-free CPU gate for a deployable Track C anchor-gate reliability rule.

The frozen diagnostic route used a scope oracle:
support/query groups used gate=1 and canonical single/family used gate=0.
This audit tests whether a simple train/support-derived, deployable gate can
replace that oracle before any new GPU work:

    g(condition, dataset, train-support metadata) -> {0, 1}

No held-out Track C query artifacts are read. Canonical multi is not used for
selection.  This is a short CPU-only gate over existing condition-mean
artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
HELPER = ROOT / "ops/summarize_latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.py"
DEFAULT_RUN_ROOT = (
    ROOT
    / "runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/"
    "xverse_support_film_retry1_condition_means_artifacts"
)
TRAINSELECT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
CPU_ROUTE_JSON = ROOT / "reports/latentfm_trackc_alternative_support_conditioning_cpu_gate_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_learned_anchor_gate_cpu_gate_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_LEARNED_ANCHOR_GATE_CPU_GATE_20260623.md"
ALPHA = 0.75


@dataclass(frozen=True)
class GateSpec:
    name: str
    description: str


GATE_SPECS = (
    GateSpec("multi_condition", "gate=1 for multi-gene conditions, gate=0 for single-gene conditions"),
    GateSpec(
        "multi_all_components_seen_in_train_single",
        "gate=1 for multi-gene conditions whose components are all seen in train_single for the same dataset",
    ),
    GateSpec(
        "multi_any_component_seen_in_train_single",
        "gate=1 for multi-gene conditions with at least one component seen in train_single for the same dataset",
    ),
)


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("anchor_gate_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {HELPER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_parts(condition: str) -> list[str]:
    return [part.strip().upper() for part in str(condition).replace("_", "+").split("+") if part.strip()]


def is_multi(condition: str) -> bool:
    return len(condition_parts(condition)) >= 2


def split_gene_sets(split: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for ds, groups in split.items():
        train_single = set()
        train_multi = set()
        for cond in groups.get("train_single") or []:
            train_single.update(condition_parts(cond))
        for cond in groups.get("train_multi") or []:
            train_multi.add("+".join(condition_parts(cond)))
        out[str(ds)] = {"train_single_genes": train_single, "train_multi_conditions": train_multi}
    return out


def gate_value(spec: GateSpec, row: dict[str, Any], genes: dict[str, dict[str, set[str]]]) -> float:
    cond = str(row["condition"])
    if not is_multi(cond):
        return 0.0
    if spec.name == "multi_condition":
        return 1.0
    parts = condition_parts(cond)
    seen = genes.get(str(row["dataset"]), {}).get("train_single_genes", set())
    if spec.name == "multi_all_components_seen_in_train_single":
        return 1.0 if parts and all(part in seen for part in parts) else 0.0
    if spec.name == "multi_any_component_seen_in_train_single":
        return 1.0 if any(part in seen for part in parts) else 0.0
    raise ValueError(spec.name)


def score_gate(mod: Any, rows: list[dict[str, Any]], spec: GateSpec, genes: dict[str, dict[str, set[str]]], *, control: str = "real") -> list[dict[str, Any]]:
    residuals = [row["pred_teacher"] - row["pred_anchor"] for row in rows]
    if control == "shuffled_residual":
        rng = np.random.default_rng(42)
        residuals = [residuals[int(i)] for i in rng.permutation(len(residuals))]
    scored = []
    for row, residual in zip(rows, residuals):
        gate = gate_value(spec, row, genes)
        if control == "zero_support":
            gate = 0.0
        elif control == "inverted_gate":
            gate = 1.0 - gate
        pred = row["pred_anchor"] + gate * ALPHA * residual
        pp = mod.pearson_np(pred - row["pert_mean"], row["gt_mean"] - row["pert_mean"])
        scored.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "is_multi": is_multi(row["condition"]),
                "gate": float(gate),
                "anchor_pp": row["anchor_pp"],
                "blend_pp": pp,
                "delta_vs_anchor": None if pp is None or row["anchor_pp"] is None else float(pp - row["anchor_pp"]),
            }
        )
    return scored


def dataset_means(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(float(value)):
            by_ds[str(row["dataset"])].append(float(value))
    return {ds: mean(vals) for ds, vals in by_ds.items() if vals}


def support_summary(mod: Any, rows: list[dict[str, Any]], route_gaps: dict[str, float], spec: GateSpec, genes: dict[str, dict[str, set[str]]], *, control: str = "real") -> dict[str, Any]:
    scored = score_gate(mod, rows, spec, genes, control=control)
    ds_delta = dataset_means(scored, "delta_vs_anchor")
    by_ds = []
    for ds, delta in sorted(ds_delta.items()):
        gap = route_gaps.get(ds)
        by_ds.append(
            {
                "dataset": ds,
                "n_conditions": sum(1 for row in scored if row["dataset"] == ds),
                "mean_gate": float(mean(row["gate"] for row in scored if row["dataset"] == ds)),
                "mean_delta_pp": float(delta),
                "route_gap_pp": gap,
                "route_gap_closed_fraction": None if gap is None or abs(gap) < 1e-12 else float(delta / gap),
            }
        )
    return {
        "control": control,
        "gate_spec": spec.name,
        "scored_rows": scored,
        "paired": mod.bootstrap_delta(scored, "blend_pp", "anchor_pp", n_boot=2000, seed=900 + len(control) + len(spec.name)),
        "dataset_summary": by_ds,
    }


def canonical_summary(mod: Any, rows: list[dict[str, Any]], spec: GateSpec, genes: dict[str, dict[str, set[str]]], group: str, *, control: str = "real") -> dict[str, Any]:
    scored = score_gate(mod, rows, spec, genes, control=control)
    paired = mod.bootstrap_delta(scored, "blend_pp", "anchor_pp", n_boot=2000, seed=1200 + len(group) + len(control) + len(spec.name))
    return {
        "group": group,
        "control": control,
        "gate_spec": spec.name,
        "mean_gate": float(mean(row["gate"] for row in scored)) if scored else 0.0,
        "max_abs_delta_pp": max(abs(float(row["delta_vs_anchor"] or 0.0)) for row in scored) if scored else None,
        "paired": paired,
    }


def find_dataset(summary: dict[str, Any], dataset: str) -> dict[str, Any]:
    for row in summary.get("dataset_summary") or []:
        if row.get("dataset") == dataset:
            return row
    return {}


def support_pass(summary: dict[str, Any]) -> bool:
    w = find_dataset(summary, "Wessels")
    n = find_dataset(summary, "NormanWeissman2019_filtered")
    paired = summary.get("paired") or {}
    return (
        float(w.get("mean_delta_pp") if w.get("mean_delta_pp") is not None else -999.0) >= 0.02
        and float(w.get("route_gap_closed_fraction") if w.get("route_gap_closed_fraction") is not None else -999.0) >= 0.05
        and float(n.get("mean_delta_pp") if n.get("mean_delta_pp") is not None else -999.0) >= -0.02
        and float(paired.get("p_harm") if paired.get("p_harm") is not None else 1.0) <= 0.20
    )


def canonical_pass(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        paired = row.get("paired") or {}
        if float(paired.get("p_harm") if paired.get("p_harm") is not None else 1.0) > 0.35:
            return False
        if float(row.get("max_abs_delta_pp") if row.get("max_abs_delta_pp") is not None else 999.0) > 1e-8:
            return False
    return True


def control_passes_fail(support_control: dict[str, Any], canonical_controls: list[dict[str, Any]], control: str) -> bool:
    if control == "zero_support":
        return not support_pass(support_control)
    if control == "shuffled_residual":
        return not support_pass(support_control)
    if control == "inverted_gate":
        return (not support_pass(support_control)) and (not canonical_pass(canonical_controls))
    raise ValueError(control)


def decide(real_support: dict[str, Any], real_canonical: list[dict[str, Any]], controls: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if not support_pass(real_support):
        reasons.append("support_gate_failed")
    if not canonical_pass(real_canonical):
        reasons.append("canonical_noharm_failed")
    for name, bundle in controls.items():
        if not control_passes_fail(bundle["support"], bundle["canonical"], name):
            reasons.append(f"{name}_control_did_not_fail")
    return {
        "status": "trackc_learned_anchor_gate_cpu_gate_pass_code_gate_next" if not reasons else "trackc_learned_anchor_gate_cpu_gate_fail_no_gpu",
        "gpu_authorization": "none",
        "next_authorization": "code_provenance_gate_only" if not reasons else "none",
        "reasons": reasons,
    }


def overlap_analysis(rows_by_group: dict[str, list[dict[str, Any]]], split: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    support_keys = set()
    for ds, groups in split.items():
        for cond in groups.get("support_val_multi") or []:
            support_keys.add((str(ds), "+".join(condition_parts(cond))))
    for group, rows in rows_by_group.items():
        multi_rows = [row for row in rows if is_multi(row["condition"])]
        by_ds: dict[str, dict[str, int]] = {}
        for row in multi_rows:
            ds = str(row["dataset"])
            cond_key = "+".join(condition_parts(row["condition"]))
            groups = split.get(ds, {})
            train_single = set()
            for cond in groups.get("train_single") or []:
                train_single.update(condition_parts(cond))
            train_multi = {"+".join(condition_parts(cond)) for cond in groups.get("train_multi") or []}
            entry = by_ds.setdefault(
                ds,
                {
                    "multi_rows": 0,
                    "all_components_train_single_seen": 0,
                    "any_component_train_single_seen": 0,
                    "train_multi_exact": 0,
                    "support_val_exact": 0,
                },
            )
            parts = condition_parts(row["condition"])
            entry["multi_rows"] += 1
            entry["all_components_train_single_seen"] += int(parts and all(part in train_single for part in parts))
            entry["any_component_train_single_seen"] += int(any(part in train_single for part in parts))
            entry["train_multi_exact"] += int(cond_key in train_multi)
            entry["support_val_exact"] += int((ds, cond_key) in support_keys)
        out[group] = {
            "n_rows": len(rows),
            "n_multi_rows": len(multi_rows),
            "by_dataset": by_ds,
        }
    return out


def render(payload: dict[str, Any]) -> str:
    d = payload["decision"]
    lines = [
        "# Track C Learned Anchor-Gate CPU Gate",
        "",
        f"Status: `{d['status']}`",
        f"GPU authorization: `{d['gpu_authorization']}`",
        f"Next authorization: `{d['next_authorization']}`",
        "",
        "## Scope",
        "",
        "Short CPU-only gate over existing condition-mean artifacts. No held-out Track C query artifacts are read. Canonical multi is not used for selection.",
        "",
        f"Selected gate: `{payload['selected_gate']['name']}` - {payload['selected_gate']['description']}",
        "",
        "## Real Support Gate",
        "",
        "| dataset | n | mean gate | delta pp | route gap | closure |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["real_support"]["dataset_summary"]:
        lines.append(
            f"| {row['dataset']} | {row['n_conditions']} | {row['mean_gate']:+.6f} | "
            f"{row['mean_delta_pp']:+.6f} | {row['route_gap_pp']:+.6f} | {row['route_gap_closed_fraction']:+.6f} |"
        )
    paired = payload["real_support"]["paired"]
    lines.extend(
        [
            "",
            f"Support paired pp delta: `{paired['delta_mean']:+.6f}`, CI `[{paired['ci95'][0]:+.6f},{paired['ci95'][1]:+.6f}]`, p_harm `{paired['p_harm']:+.6f}`.",
            "",
            "## Canonical No-Harm With Gate Active",
            "",
            "| group | mean gate | max abs pp delta | pp delta | p harm |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["real_canonical"]:
        p = row["paired"]
        lines.append(
            f"| {row['group']} | {row['mean_gate']:+.6f} | {row['max_abs_delta_pp']:+.6f} | "
            f"{p['delta_mean']:+.6f} | {p['p_harm']:+.6f} |"
        )
    lines.extend(["", "## All Predeclared Gate Summaries", "", "| gate | support pass | canonical pass | Wessels delta | Wessels closure | family mean gate | family max abs delta |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in payload["all_gate_summaries"]:
        wessels = find_dataset({"dataset_summary": row["support_dataset_summary"]}, "Wessels")
        family = next((item for item in row["canonical"] if item["group"] == "family_gene"), {})
        lines.append(
            f"| `{row['name']}` | `{row['support_pass']}` | `{row['canonical_pass']}` | "
            f"{float(wessels.get('mean_delta_pp') or 0.0):+.6f} | "
            f"{float(wessels.get('route_gap_closed_fraction') or 0.0):+.6f} | "
            f"{float(family.get('mean_gate') or 0.0):+.6f} | "
            f"{float(family.get('max_abs_delta_pp') or 0.0):+.6f} |"
        )
    lines.extend(["", "## Overlap / Identifiability Analysis", ""])
    for group, info in payload["overlap_analysis"].items():
        lines.append(f"- `{group}`: rows `{info['n_rows']}`, multi rows `{info['n_multi_rows']}`")
        for ds, vals in sorted(info["by_dataset"].items()):
            lines.append(
                f"  - {ds}: multi `{vals['multi_rows']}`, support-val exact `{vals['support_val_exact']}`, "
                f"train-multi exact `{vals['train_multi_exact']}`, all components train-single seen `{vals['all_components_train_single_seen']}`, "
                f"any component train-single seen `{vals['any_component_train_single_seen']}`"
            )
    lines.extend(["", "## Negative Controls", "", "| control | support pass? | canonical pass? | expected |", "|---|---:|---:|---|"])
    for name, bundle in payload["controls"].items():
        lines.append(
            f"| `{name}` | `{support_pass(bundle['support'])}` | `{canonical_pass(bundle['canonical'])}` | fail |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in d["reasons"]] if d["reasons"] else ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Passing this CPU gate only authorizes a code/provenance gate for an implemented deployable gate. It does not authorize held-out query reuse, alpha tuning, or a new GPU run by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--split-file", type=Path, default=TRAINSELECT_SPLIT)
    parser.add_argument("--cpu-route-json", type=Path, default=CPU_ROUTE_JSON)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mod = load_helper()
    split = load_json(args.split_file)
    genes = split_gene_sets(split)
    route_gaps = mod.cpu_route_gap_by_dataset(args.cpu_route_json)
    mean_dir = args.run_root / "condition_means"

    support_rows = mod.paired_rows(
        load_json(mean_dir / "support_anchor_split_condition_means_ode20.json"),
        load_json(mean_dir / "support_candidate_split_condition_means_ode20.json"),
        "test_multi",
    )
    canonical_single = mod.paired_rows(
        load_json(mean_dir / "canonical_anchor_split_test_single_condition_means_ode20.json"),
        load_json(mean_dir / "canonical_candidate_split_test_single_condition_means_ode20.json"),
        "test_single",
    )
    canonical_family = mod.paired_rows(
        load_json(mean_dir / "canonical_anchor_family_gene_condition_means_ode20.json"),
        load_json(mean_dir / "canonical_candidate_family_gene_condition_means_ode20.json"),
        "family_gene",
    )

    gate_results = []
    for spec in GATE_SPECS:
        real_support = support_summary(mod, support_rows, route_gaps, spec, genes)
        real_canonical = [
            canonical_summary(mod, canonical_single, spec, genes, "test_single"),
            canonical_summary(mod, canonical_family, spec, genes, "family_gene"),
        ]
        gate_results.append({"spec": spec, "support": real_support, "canonical": real_canonical})

    passing = [row for row in gate_results if support_pass(row["support"]) and canonical_pass(row["canonical"])]
    selected = passing[0] if passing else gate_results[0]
    selected_spec = selected["spec"]

    controls = {}
    for control in ("zero_support", "shuffled_residual", "inverted_gate"):
        controls[control] = {
            "support": support_summary(mod, support_rows, route_gaps, selected_spec, genes, control=control),
            "canonical": [
                canonical_summary(mod, canonical_single, selected_spec, genes, "test_single", control=control),
                canonical_summary(mod, canonical_family, selected_spec, genes, "family_gene", control=control),
            ],
        }

    decision = decide(selected["support"], selected["canonical"], controls)
    payload = {
        "heldout_query_used": False,
        "canonical_multi_selection_used": False,
        "alpha": ALPHA,
        "split_file": str(args.split_file),
        "run_root": str(args.run_root),
        "selected_gate": {"name": selected_spec.name, "description": selected_spec.description},
        "all_gate_summaries": [
            {
                "name": row["spec"].name,
                "description": row["spec"].description,
                "support_pass": support_pass(row["support"]),
                "canonical_pass": canonical_pass(row["canonical"]),
                "support_dataset_summary": row["support"]["dataset_summary"],
                "canonical": row["canonical"],
            }
            for row in gate_results
        ],
        "real_support": selected["support"],
        "real_canonical": selected["canonical"],
        "controls": controls,
        "overlap_analysis": overlap_analysis(
            {
                "support_val_multi": support_rows,
                "canonical_test_single": canonical_single,
                "canonical_family_gene": canonical_family,
            },
            split,
        ),
        "decision": decision,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
