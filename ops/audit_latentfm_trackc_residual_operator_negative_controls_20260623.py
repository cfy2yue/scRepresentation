#!/usr/bin/env python3
"""Negative-control audit for the Track C residual-operator CPU gate.

This short CPU audit reuses the selected residual-operator spec from the
existing CPU gate, then scores two support-context controls on support-val:

* zero support context: correction is forced to zero, so the candidate is route;
* shuffled support context: target rows receive another support context delta.

The controls must fail the same support-val promotion gates.  This protects the
next Track C support-conditioning branch from promoting a mechanism that only
works because of route baselines or support-val scoring noise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RESIDUAL_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_residual_operator_cpu_gate_20260623.py"
DEFAULT_CPU_GATE_JSON = ROOT / "reports/latentfm_trackc_residual_operator_cpu_gate_20260623.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_residual_operator_negative_controls_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_RESIDUAL_OPERATOR_NEGATIVE_CONTROLS_20260623.md"


def load_residual_module() -> Any:
    spec = importlib.util.spec_from_file_location("trackc_residual_operator_gate", RESIDUAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {RESIDUAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def spec_from_payload(mod: Any, payload: dict[str, Any]) -> tuple[Any, Any]:
    mem_payload = payload["selected_memory_spec"]
    op_payload = payload["selected_operator_spec"]
    mem = mod.MemorySpec(
        name=str(mem_payload["name"]),
        mode=str(mem_payload["mode"]),
        k=int(mem_payload["k"]),
        same_dataset=bool(mem_payload["same_dataset"]),
        min_score=float(mem_payload["min_score"]),
    )
    op = mod.OperatorSpec(
        name=str(op_payload["name"]),
        kind=str(op_payload["kind"]),
        rank=int(op_payload.get("rank") or 0),
        ridge=float(op_payload.get("ridge") or 0.0),
    )
    return mem, op


def dataset_breakdown(mod: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ds in sorted({str(r["dataset"]) for r in rows}):
        sub = [r for r in rows if str(r["dataset"]) == ds]
        out.append(
            {
                "dataset": ds,
                "n_conditions": len(sub),
                "candidate": float(np.mean([r["candidate"] for r in sub if r.get("candidate") is not None])),
                "support_selected_route": float(
                    np.mean([r["support_selected_route"] for r in sub if r.get("support_selected_route") is not None])
                ),
                "delta_pp": mod.dataset_delta(sub, "candidate", "support_selected_route").get(ds),
                "candidate_mmd_clamped": float(np.mean([r["candidate__test_mmd_clamped"] for r in sub])),
                "route_mmd_clamped": float(np.mean([r["support_selected_route__test_mmd_clamped"] for r in sub])),
            }
        )
    return out


def score_control(
    *,
    mod: Any,
    support: Any,
    support_val: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
    fitted: dict[str, Any],
    single: dict[str, Any],
    multi: dict[str, Any],
    pert_means: dict[str, np.ndarray],
    mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    eval_by_key = {mod.condition_key(sample["row"]): sample for sample in eval_samples}
    rng = np.random.default_rng(seed)
    shuffled = list(eval_samples)
    if shuffled:
        perm = rng.permutation(len(shuffled))
        shuffled_contexts = [np.asarray(shuffled[int(i)]["context_delta"], dtype=np.float32) for i in perm]
    else:
        shuffled_contexts = []
    context_by_key = {
        mod.condition_key(sample["row"]): shuffled_contexts[i]
        for i, sample in enumerate(eval_samples)
    }

    rows = []
    for target in support_val:
        sample = eval_by_key.get(mod.condition_key(target))
        if sample is None:
            row = mod.score_noop_row(target, single, multi, pert_means, support, compute_mmd=True)
            row["no_context_noop"] = True
        else:
            if mode == "zero_context":
                context_delta = np.zeros_like(sample["context_delta"], dtype=np.float32)
            elif mode == "shuffled_context":
                context_delta = context_by_key[mod.condition_key(target)]
            else:
                raise ValueError(mode)
            pred = mod.apply_operator(sample["route"], context_delta, fitted)
            row = mod.score_prediction(sample, pred, pert_means, support, compute_mmd=True)
        row["control_mode"] = mode
        rows.append(row)
    return rows


def summarize_control(
    *,
    mod: Any,
    rows: list[dict[str, Any]],
    closed_delta: float,
    route_gap: float | None,
    split_guard: dict[str, Any],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    pp_delta = mod.paired_bootstrap(
        rows,
        "candidate",
        "support_selected_route",
        metric="pp",
        n_boot=n_boot,
        seed=seed,
    )
    mmd_delta = mod.paired_bootstrap(
        rows,
        "candidate",
        "support_selected_route",
        metric="mmd_clamped",
        n_boot=n_boot,
        seed=seed + 100,
    )
    decision = mod.decide(
        rows,
        pp_delta,
        mmd_delta,
        closed_wessels_delta=closed_delta,
        wessels_route_gap=route_gap,
        wiring_delta_l2=1.0,
        split=split_guard,
    )
    return {
        "decision": decision,
        "paired_pp_delta": pp_delta,
        "paired_mmd_delta": mmd_delta,
        "dataset_breakdown": dataset_breakdown(mod, rows),
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track C Residual-Operator Negative Controls",
        "",
        f"Status: `{payload['status']}`",
        "GPU authorization: `none`",
        "",
        "## Provenance",
        "",
        f"- source CPU gate JSON: `{payload['cpu_gate_json']}`",
        f"- selected spec: `{payload['selected_spec']}`",
        f"- split_file: `{payload['split_guard']['split_file']}`",
        f"- split SHA256: `{payload['split_guard']['sha256']}`",
        f"- controls: `{', '.join(payload['controls'].keys())}`",
        "",
        "## Control Decisions",
        "",
        "| control | decision | Wessels delta | Wessels closure | Norman delta | pp p_harm | MMD p_harm | expected |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, control in payload["controls"].items():
        decision = control["decision"]
        lines.append(
            f"| `{name}` | `{decision['status']}` | {fmt(decision.get('wessels_delta_vs_route'))} | "
            f"{fmt(decision.get('wessels_route_gap_closure'))} | {fmt(decision.get('norman_delta_vs_route'))} | "
            f"{fmt(control['paired_pp_delta'].get('p_harm'))} | {fmt(control['paired_mmd_delta'].get('p_harm'))} | fail |"
        )
    lines += ["", "## Decision Reasons", ""]
    reasons = payload.get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines += [
        "",
        "## Usage Rule",
        "",
        "- Passing this audit does not authorize any new GPU by itself.",
        "- A future Track C alternative support-conditioning CPU gate should keep these controls or stronger equivalents.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-gate-json", type=Path, default=DEFAULT_CPU_GATE_JSON)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    mod = load_residual_module()
    support = mod.load_support_module()
    source = load_json(args.cpu_gate_json)
    selected_mem, selected_op = spec_from_payload(mod, source)
    split_file = Path(source["split_guard"]["split_file"])
    data_dir = Path(source["data_dir"])
    split = support.load_json(split_file)
    manifest = support.load_json(data_dir / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    pert_means = {k: v.astype(np.float32) for k, v in np.load(source["pert_means_file"]).items()}

    guard = mod.split_guard(split_file, split)
    if guard["sha256"] != mod.EXPECTED_TRAINSELECT_SHA256:
        raise RuntimeError(f"unexpected trainselect split hash: {guard['sha256']}")

    train_rows = support.collect_role_rows(
        data_dir,
        split,
        metadata,
        "train_multi",
        max_cells=int(source["max_cells_per_condition"]),
    )
    support_val = support.collect_role_rows(
        data_dir,
        split,
        metadata,
        "support_val_multi",
        max_cells=int(source["max_cells_per_condition"]),
    )
    single = support.train_single_components(data_dir, split, metadata, max_cells=int(source["max_cells_per_condition"]))
    multi = support.train_multi_components(train_rows)

    fit_samples = mod.make_samples(train_rows, train_rows, selected_mem, single, multi, support)
    fitted = mod.fit_operator(fit_samples, selected_op)
    eval_samples = mod.make_samples(support_val, train_rows, selected_mem, single, multi, support)
    closed_delta = mod.load_closed_wessels_delta(Path(source["bottleneck_summary"]))
    route_gap = mod.readout_wessels_route_gap(Path(source["readout_json"]))

    controls = {}
    for offset, mode in enumerate(("zero_context", "shuffled_context")):
        rows = score_control(
            mod=mod,
            support=support,
            support_val=support_val,
            eval_samples=eval_samples,
            fitted=fitted,
            single=single,
            multi=multi,
            pert_means=pert_means,
            mode=mode,
            seed=args.seed + offset,
        )
        controls[mode] = summarize_control(
            mod=mod,
            rows=rows,
            closed_delta=closed_delta,
            route_gap=route_gap,
            split_guard=guard,
            n_boot=args.n_boot,
            seed=args.seed + 10 * offset,
        )

    reasons = []
    for name, control in controls.items():
        status = str(control["decision"]["status"])
        if status.endswith("pass_authorize_one_capped_gpu_smoke"):
            reasons.append(f"{name}_unexpectedly_passed_support_gate")
    payload = {
        "status": "trackc_residual_operator_negative_controls_pass" if not reasons else "trackc_residual_operator_negative_controls_fail",
        "gpu_authorization": "none",
        "reasons": reasons,
        "cpu_gate_json": str(args.cpu_gate_json),
        "selected_spec": source["selected_spec"],
        "split_guard": guard,
        "controls": controls,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
