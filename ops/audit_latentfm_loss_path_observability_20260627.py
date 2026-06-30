#!/usr/bin/env python3
"""CPU-only loss/path observability audit for LatentFM training logs.

The audit is deliberately log-only. It does not run training, inference, or
gradient computation. Its purpose is to decide whether a more expensive
one-step gradient dry-run is justified before any new loss-schedule smoke.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "CoupledFM/output/latentfm_runs"
OUT_DIR = ROOT / "reports/latentfm_loss_path_observability_20260627"
JSON_PATH = ROOT / "reports/latentfm_loss_path_observability_20260627.json"
MD_PATH = ROOT / "reports/LATENTFM_LOSS_PATH_OBSERVABILITY_20260627.md"
CSV_PATH = OUT_DIR / "run_loss_path_inventory.csv"

STEP_RE = re.compile(r"step=(?P<step>\d+).*?avg_loss=(?P<avg_loss>[-+0-9.eE]+)")
FIELD_RE = re.compile(r"(?P<key>avg_[a-zA-Z0-9_]+|λ[a-zA-Z0-9_]+|γ)=(?P<val>[-+0-9.eE]+)")
CONFIG_RE = re.compile(r"Config:\s+Config\((?P<body>.*)\)")

LOSS_FIELDS = [
    "avg_mse",
    "avg_mmd",
    "avg_delta",
    "avg_comp",
    "avg_prior_delta",
    "avg_prior_add_delta",
    "avg_cond_delta",
    "avg_add_cond_delta",
    "avg_pert_resid",
    "avg_pert_ctr",
    "avg_pert_rel",
]

ROUTE_FIELDS = [
    "avg_comp",
    "avg_prior_delta",
    "avg_prior_add_delta",
    "avg_cond_delta",
    "avg_add_cond_delta",
    "avg_pert_resid",
    "avg_pert_ctr",
    "avg_pert_rel",
]

CONFIG_WEIGHTS = [
    "composition_delta_loss_weight",
    "condition_delta_head_loss_weight",
    "additive_condition_delta_loss_weight",
    "condition_prior_delta_loss_weight",
    "condition_prior_additive_delta_loss_weight",
    "pert_residual_direction_loss_weight",
    "pert_residual_contrastive_loss_weight",
    "pert_residual_relational_loss_weight",
    "endpoint_delta_loss_weight",
    "ds_loss_alpha",
]


@dataclass
class RunLossAudit:
    run: str
    train_log: str
    n_steps_logged: int
    max_route_ratio: float
    max_route_field: str
    mean_route_ratio: float
    active_route_fields: str
    nonzero_config_weights: str
    status: str
    reason: str


def _float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _parse_config_weights(text: str) -> dict[str, float]:
    weights = {}
    m = CONFIG_RE.search(text)
    if not m:
        return weights
    body = m.group("body")
    for key in CONFIG_WEIGHTS:
        km = re.search(rf"{re.escape(key)}=(?P<val>[-+0-9.eE]+)", body)
        if km:
            weights[key] = _float(km.group("val"))
    return weights


def _parse_steps(text: str) -> list[dict[str, float]]:
    rows = []
    for line in text.splitlines():
        if "avg_loss=" not in line or "step=" not in line:
            continue
        sm = STEP_RE.search(line)
        if not sm:
            continue
        row = {"step": float(sm.group("step")), "avg_loss": _float(sm.group("avg_loss"))}
        for fm in FIELD_RE.finditer(line):
            row[fm.group("key")] = _float(fm.group("val"))
        rows.append(row)
    return rows


def _audit_one(path: Path) -> RunLossAudit:
    text = path.read_text(errors="replace")
    rel = str(path.parent.relative_to(RUN_ROOT))
    weights = _parse_config_weights(text)
    rows = _parse_steps(text)

    ratios_by_field: dict[str, list[float]] = defaultdict(list)
    active = []
    for row in rows:
        denom = abs(row.get("avg_mse", row.get("avg_loss", 0.0))) + 1e-12
        for field in ROUTE_FIELDS:
            val = abs(row.get(field, 0.0))
            if math.isfinite(val) and val > 1e-10:
                ratios_by_field[field].append(val / denom)
    for field in ROUTE_FIELDS:
        vals = ratios_by_field.get(field, [])
        if vals and max(vals) >= 1e-4:
            active.append(field)

    all_ratios = [v for vals in ratios_by_field.values() for v in vals]
    max_field = "none"
    max_ratio = 0.0
    for field, vals in ratios_by_field.items():
        if vals and max(vals) > max_ratio:
            max_ratio = max(vals)
            max_field = field
    mean_ratio = float(sum(all_ratios) / len(all_ratios)) if all_ratios else 0.0
    nonzero_weights = [f"{k}={v:g}" for k, v in sorted(weights.items()) if abs(v) > 1e-12]

    if not rows:
        status = "not_informative"
        reason = "no_step_loss_lines"
    elif not nonzero_weights and not active:
        status = "inactive_default_like"
        reason = "no_nonzero_route_weights_or_logged_route_terms"
    elif max_ratio < 0.01:
        status = "near_inert_no_gpu"
        reason = f"max_route_ratio_{max_ratio:.6g}_lt_0p01"
    elif max_ratio < 0.05:
        status = "weak_route_signal_needs_gradient_dryrun_no_gpu"
        reason = f"max_route_ratio_{max_ratio:.6g}_lt_0p05"
    else:
        status = "route_observable_needs_gradient_dryrun_no_gpu"
        reason = f"max_route_ratio_{max_ratio:.6g}_requires_gradient_and_metric_gate"

    return RunLossAudit(
        run=rel,
        train_log=str(path),
        n_steps_logged=len(rows),
        max_route_ratio=max_ratio,
        max_route_field=max_field,
        mean_route_ratio=mean_ratio,
        active_route_fields=";".join(active),
        nonzero_config_weights=";".join(nonzero_weights),
        status=status,
        reason=reason,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audits = [_audit_one(p) for p in sorted(RUN_ROOT.glob("**/train.log"))]
    informative = [a for a in audits if a.status != "not_informative"]
    observable = [a for a in informative if a.status == "route_observable_needs_gradient_dryrun_no_gpu"]
    weak = [a for a in informative if a.status == "weak_route_signal_needs_gradient_dryrun_no_gpu"]
    near = [a for a in informative if a.status == "near_inert_no_gpu"]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RunLossAudit.__annotations__.keys()))
        writer.writeheader()
        for a in sorted(informative, key=lambda x: (-x.max_route_ratio, x.run)):
            writer.writerow(asdict(a))

    top = sorted(informative, key=lambda x: (-x.max_route_ratio, x.run))[:20]
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "boundary": {
            "task": "CPU-only log loss/path observability audit",
            "no_gpu": True,
            "no_training": True,
            "no_inference": True,
            "no_canonical_multi_or_trackc_query": True,
        },
        "counts": {
            "train_logs_total": len(audits),
            "informative_logs": len(informative),
            "route_observable_needs_gradient_dryrun_no_gpu": len(observable),
            "weak_route_signal_needs_gradient_dryrun_no_gpu": len(weak),
            "near_inert_no_gpu": len(near),
        },
        "decision": {
            "gpu_authorized": False,
            "status": "loss_path_log_audit_no_gpu",
            "next_gate": "one-step gradient/path dry-run only for the top observable routes; no capped GPU smoke from log magnitudes alone",
        },
        "top_runs": [asdict(a) for a in top],
        "output_csv": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = []
    for a in top[:12]:
        lines.append(
            f"| `{a.run}` | {a.n_steps_logged} | `{a.max_route_field}` | "
            f"{a.max_route_ratio:.6f} | `{a.active_route_fields or 'none'}` | "
            f"`{a.status}` | `{a.reason}` |"
        )
    md = f"""# LatentFM Loss/Path Observability Audit

## Status

`loss_path_log_audit_no_gpu`

## Boundary

CPU-only log/config audit. No training, no inference, no model-weight loading,
no GPU, no canonical multi selection, and no Track C query use.

## Summary

* Train logs scanned: `{summary['counts']['train_logs_total']}`
* Informative logs: `{summary['counts']['informative_logs']}`
* Observable route logs needing gradient dry-run: `{summary['counts']['route_observable_needs_gradient_dryrun_no_gpu']}`
* Weak route logs: `{summary['counts']['weak_route_signal_needs_gradient_dryrun_no_gpu']}`
* Near-inert logs: `{summary['counts']['near_inert_no_gpu']}`

## Gate

This audit cannot authorize GPU by itself. It only decides whether a one-step
gradient/path dry-run is worth writing. A future GPU smoke would still require
material condition-specific gradients, route movement, strict train-only
signal/no-harm controls, a bounded launcher, and fail-close criteria.

## Top Route-Loss Magnitudes

| Run | Step logs | Max route field | Max route/MSE ratio | Active route fields | Status | Reason |
|---|---:|---|---:|---|---|---|
{chr(10).join(lines) if lines else '| NA | 0 | NA | 0 | NA | NA | no informative logs |'}

## Decision

Use this report to select at most one narrowly scoped gradient dry-run. Do not
launch a loss-schedule GPU smoke from log magnitudes alone.

## Outputs

* JSON: `{JSON_PATH}`
* CSV: `{CSV_PATH}`
"""
    MD_PATH.write_text(md)
    print(json.dumps({"status": "loss_path_log_audit_no_gpu", "json": str(JSON_PATH), "md": str(MD_PATH), "csv": str(CSV_PATH)}, indent=2))


if __name__ == "__main__":
    main()
