#!/usr/bin/env python3
"""CPU gate for deployable risk overlays on the response-covariate router."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import audit_latentfm_tracka_xverse_response_covariate_router_gate_20260624 as base


OUT_JSON = ROOT / "reports/latentfm_tracka_xverse_deployable_risk_overlay_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_XVERSE_DEPLOYABLE_RISK_OVERLAY_GATE_20260624.md"

VARIANTS: dict[str, tuple[str, ...]] = {
    "full_forensics_diagnostic": (
        "gene_target_cosine",
        "dataset_target_cosine",
        "target_residual_norm",
        "gene_dataset_cosine",
        "gene_minus_dataset_score",
        "gene_pred_norm",
        "dataset_pred_norm",
        "global_pred_norm",
        "gene_train_count",
    ),
    "deployable_norms_count_geometry": (
        "gene_dataset_cosine",
        "gene_pred_norm",
        "dataset_pred_norm",
        "global_pred_norm",
        "gene_train_count",
    ),
    "deployable_norms_count": (
        "gene_pred_norm",
        "dataset_pred_norm",
        "global_pred_norm",
        "gene_train_count",
    ),
}
DEPLOYABLE = {
    "deployable_norms_count_geometry",
    "deployable_norms_count",
}
OVERLAY_FEATURES = (
    "gene_train_count",
    "gene_pred_norm",
    "dataset_pred_norm",
    "global_pred_norm",
    "gene_dataset_cosine",
)
THRESHOLDS = (-0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


def dataset_metrics(rows: list[dict[str, Any]], overlay: tuple[str, str, float] | None) -> dict[str, Any]:
    deltas: list[float] = []
    uses: list[bool] = []
    by_ds: dict[str, list[float]] = {}
    for row in rows:
        use = bool(row["router_use_anchor"])
        if overlay is not None:
            feat, op, cut = overlay
            val = float(row.get(feat, 0.0) or 0.0)
            risky = val <= cut if op == "<=" else val >= cut
            use = use and not risky
        uses.append(use)
        pred = float(row["anchor_pearson_pert"] if use else row["gene_raw_mean"])
        delta = pred - float(row["gene_raw_mean"])
        deltas.append(delta)
        by_ds.setdefault(str(row["dataset"]), []).append(delta)
    ds_vals = [float(np.mean(vals)) for vals in by_ds.values()]
    return {
        "delta_vs_gene": float(np.mean(ds_vals)) if ds_vals else float("nan"),
        "dataset_min_vs_gene": float(np.min(ds_vals)) if ds_vals else float("nan"),
        "dataset_harm_fraction": float(np.mean([v < 0.0 for v in ds_vals])) if ds_vals else float("nan"),
        "use_anchor_fraction": float(np.mean(uses)) if uses else 0.0,
    }


def candidate_cuts(rows: list[dict[str, Any]], feat: str) -> list[float]:
    vals = np.asarray([float(row.get(feat, 0.0) or 0.0) for row in rows], dtype=float)
    if len(set(vals.tolist())) > 30:
        return sorted(set(float(x) for x in np.quantile(vals, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])))
    return sorted(set(float(x) for x in vals.tolist()))


def pass_metrics(metrics: dict[str, Any]) -> bool:
    return (
        metrics["use_anchor_fraction"] >= 0.05
        and metrics["delta_vs_gene"] >= 0.02
        and metrics["dataset_min_vs_gene"] >= -0.02
        and metrics["dataset_harm_fraction"] <= 0.20
    )


def best_overlay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for feat in OVERLAY_FEATURES:
        if feat not in rows[0]:
            continue
        for cut in candidate_cuts(rows, feat):
            for op in ("<=", ">="):
                overlay = (feat, op, cut)
                metrics = dataset_metrics(rows, overlay)
                key = (
                    pass_metrics(metrics),
                    metrics["use_anchor_fraction"] >= 0.05,
                    metrics["dataset_min_vs_gene"] >= -0.02,
                    metrics["dataset_harm_fraction"] <= 0.20,
                    metrics["delta_vs_gene"],
                    metrics["use_anchor_fraction"],
                )
                item = {"overlay": {"feature": feat, "op": op, "cut": cut}, "metrics": metrics, "passes_gate": pass_metrics(metrics), "key": key}
                if best is None or item["key"] > best["key"]:
                    best = item
    assert best is not None
    best.pop("key", None)
    return best


def evaluate_variant(all_rows: list[dict[str, Any]], name: str, features: tuple[str, ...]) -> dict[str, Any]:
    base.FEATURES = features
    base.THRESHOLDS = THRESHOLDS
    groups = []
    for group in base.GROUPS:
        result = base.evaluate_group(all_rows, group)
        raw = dataset_metrics(result["scored_rows"], None)
        overlay = best_overlay(result["scored_rows"])
        groups.append({"group": group, "raw_metrics": raw, "best_overlay": overlay})
    deployable = name in DEPLOYABLE
    passes = deployable and all(g["best_overlay"]["passes_gate"] for g in groups)
    return {"name": name, "deployable": deployable, "features": list(features), "groups": groups, "passes_gate": passes}


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Track A xverse Deployable Risk Overlay Gate",
        "",
        f"Status: `{payload['decision']['status']}`",
        "GPU authorization: `none`",
        "",
        "## Boundary",
        "",
        "- Uses only train-only/internal response-forensics rows.",
        "- Does not read canonical outcomes, canonical multi, held-out query, active logs, or GPU artifacts.",
        "- Tests whether a simple deployable risk overlay can fix response-router tail harm.",
        "- Full-forensics rows are diagnostic only because they include target-derived covariates.",
        "",
        "## Gate Rule",
        "",
        "A deployable variant must pass both internal groups with use-anchor `>=0.05`, delta vs gene `>=+0.02`, dataset min `>=-0.02`, and dataset harm fraction `<=0.20`.",
        "",
        "## Rows",
        "",
        "| variant | deployable | group | overlay | use anchor | delta vs gene | dataset min | harm frac | pass |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in payload["variants"]:
        for group in variant["groups"]:
            over = group["best_overlay"]
            spec = over["overlay"]
            m = over["metrics"]
            lines.append(
                f"| `{variant['name']}` | {str(variant['deployable']).lower()} | {group['group']} | "
                f"{spec['feature']} {spec['op']} {float(spec['cut']):.6g} | "
                f"{m['use_anchor_fraction']:.3f} | {m['delta_vs_gene']:+.6f} | "
                f"{m['dataset_min_vs_gene']:+.6f} | {m['dataset_harm_fraction']:.3f} | "
                f"{str(over['passes_gate']).lower()} |"
            )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend([f"- `{r}`" for r in payload["decision"]["reasons"]] or ["- none"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    source = base.load_json(base.XVERSE_ROWS)
    rows = source["condition_rows"]
    variants = [evaluate_variant(rows, name, features) for name, features in VARIANTS.items()]
    deployable_passes = [v["name"] for v in variants if v["passes_gate"]]
    diagnostic_passes = [
        v["name"]
        for v in variants
        if (not v["deployable"]) and all(g["best_overlay"]["passes_gate"] for g in v["groups"])
    ]
    reasons = []
    if not deployable_passes:
        reasons.append("no_deployable_overlay_passed_gain_and_tail_gate")
    if diagnostic_passes:
        reasons.append("only_non_deployable_full_forensics_overlay_passed")
    decision = {
        "status": "tracka_xverse_deployable_risk_overlay_gate_pass_code_gate_next_no_gpu"
        if deployable_passes
        else "tracka_xverse_deployable_risk_overlay_gate_fail_no_gpu",
        "gpu_authorization": "none",
        "action": "design_code_gate_only_if_pass_else_pivot",
        "deployable_passes": deployable_passes,
        "diagnostic_passes": diagnostic_passes,
        "reasons": reasons,
    }
    payload = {
        "status": decision["status"],
        "boundary": {
            "canonical_test_read": False,
            "canonical_multi_read": False,
            "heldout_query_read": False,
            "active_log_read": False,
            "gpu_artifact_read": False,
        },
        "variants": variants,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "gpu_authorization": "none", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
