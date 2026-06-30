#!/usr/bin/env python3
"""Internal-only gate for selective cap120 anchor replay.

This script reads already completed train-only/internal posthoc artifacts. It
does not train, read canonical split outputs, or inspect Track C query files.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CAP120_INTERNAL = (
    ROOT
    / "runs/latentfm_xverse_scaling_count_smokes_20260624"
    / "xverse_scaling_cap120_all_3k_seed42"
    / "posthoc_eval_internal"
)
OUT_JSON = ROOT / "reports/latentfm_xverse_selective_replay_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_SELECTIVE_REPLAY_GATE_20260624.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _group(eval_dir: Path, stem: str, who: str, key: str) -> dict[str, Any]:
    return _load(eval_dir / f"{stem}_{who}_internal_ode20.json")["groups"][key]


def _mean(vals: list[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(statistics.fmean(vals)) if vals else float("nan")


def _condition_rows(anchor: dict[str, Any], cand: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = {
        (str(r["dataset"]), str(r["condition"])): r
        for r in anchor.get("condition_metrics", [])
    }
    rows: list[dict[str, Any]] = []
    for cr in cand.get("condition_metrics", []):
        key = (str(cr["dataset"]), str(cr["condition"]))
        ar = by_key.get(key)
        if ar is None:
            continue
        rows.append(
            {
                "dataset": key[0],
                "condition": key[1],
                "pp_delta": float(cr["pearson_pert"]) - float(ar["pearson_pert"]),
                "mmd_delta": float(cr["test_mmd_clamped"]) - float(ar["test_mmd_clamped"]),
            }
        )
    return rows


def _by_dataset(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[str(row["dataset"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for ds, ds_rows in sorted(by_ds.items()):
        out[ds] = {
            "n": len(ds_rows),
            "pp_delta_mean": _mean([r["pp_delta"] for r in ds_rows]),
            "mmd_delta_mean": _mean([r["mmd_delta"] for r in ds_rows]),
            "mmd_harm_frac": _mean([1.0 if r["mmd_delta"] > 0.001 else 0.0 for r in ds_rows]),
        }
    return out


def _summarize_subset(rows: list[dict[str, Any]], allow: set[str]) -> dict[str, Any]:
    selected = [r for r in rows if str(r["dataset"]) in allow]
    other = [r for r in rows if str(r["dataset"]) not in allow]
    total = max(len(rows), 1)
    return {
        "n_total": len(rows),
        "n_selected": len(selected),
        "selected_frac": len(selected) / total,
        "selected_pp_delta_mean": _mean([r["pp_delta"] for r in selected]),
        "selected_mmd_delta_mean": _mean([r["mmd_delta"] for r in selected]),
        "other_pp_delta_mean": _mean([r["pp_delta"] for r in other]),
        "other_mmd_delta_mean": _mean([r["mmd_delta"] for r in other]),
    }


def main() -> int:
    family_anchor = _group(CAP120_INTERNAL, "condition_family_eval", "anchor", "family_gene")
    family_cand = _group(CAP120_INTERNAL, "condition_family_eval", "candidate", "family_gene")
    cross_anchor = _group(
        CAP120_INTERNAL,
        "split_group_eval",
        "anchor",
        "internal_val_cross_background_seen_gene_proxy",
    )
    cross_cand = _group(
        CAP120_INTERNAL,
        "split_group_eval",
        "candidate",
        "internal_val_cross_background_seen_gene_proxy",
    )
    family_rows = _condition_rows(family_anchor, family_cand)
    cross_rows = _condition_rows(cross_anchor, cross_cand)
    family_ds = _by_dataset(family_rows)
    cross_ds = _by_dataset(cross_rows)

    risk_datasets = {
        ds
        for ds, row in family_ds.items()
        if float(row["mmd_delta_mean"]) > 0.001 and float(row["pp_delta_mean"]) < 0.0
    }
    family_subset = _summarize_subset(family_rows, risk_datasets)
    cross_subset = _summarize_subset(cross_rows, risk_datasets)

    reasons: list[str] = []
    if not risk_datasets:
        reasons.append("no_internal_family_risk_datasets")
    if len(risk_datasets) > 8:
        reasons.append("too_many_risk_datasets_for_selective_replay")
    if family_subset["selected_frac"] > 0.50:
        reasons.append("risk_family_conditions_not_minority")
    if cross_subset["selected_frac"] > 0.50:
        reasons.append("risk_cross_conditions_not_minority")
    if cross_subset["selected_pp_delta_mean"] > 0.0:
        reasons.append("risk_datasets_are_positive_cross_signal_source")
    if cross_subset["other_pp_delta_mean"] < 0.010:
        reasons.append("nonrisk_cross_signal_too_weak")

    status = "selective_replay_gate_pass_no_gpu" if not reasons else "selective_replay_gate_fail_no_gpu"
    decision = {
        "status": status,
        "reasons": reasons,
        "action": (
            "launch_selective_dataset_replay_smoke"
            if status.endswith("pass_no_gpu")
            else "do_not_launch_selective_dataset_replay"
        ),
        "risk_datasets": sorted(risk_datasets),
        "risk_dataset_csv": ",".join(sorted(risk_datasets)),
    }
    payload = {
        "decision": decision,
        "boundary": {
            "source": "cap120 train-only/internal posthoc only",
            "canonical_or_query_used": False,
            "cap120_internal_dir": str(CAP120_INTERNAL),
        },
        "family_subset": family_subset,
        "cross_subset": cross_subset,
        "family_by_dataset": family_ds,
        "cross_by_dataset": cross_ds,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    top = sorted(
        family_ds.items(),
        key=lambda kv: float(kv[1]["mmd_delta_mean"]),
        reverse=True,
    )[:10]
    lines = [
        "# LatentFM xverse Selective Replay Gate",
        "",
        "## Boundary",
        "",
        "- Reads cap120 train-only/internal posthoc artifacts only.",
        "- Does not read canonical split, Track C query, or active logs.",
        "- Selects risk datasets only if internal family MMD worsens and pp does not improve.",
        "",
        "## Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        f"Risk dataset CSV: `{decision['risk_dataset_csv']}`",
        "",
        "## Gate Metrics",
        "",
        f"- family selected frac: `{family_subset['selected_frac']:.4f}`",
        f"- family selected pp delta mean: `{family_subset['selected_pp_delta_mean']:+.6f}`",
        f"- family selected MMD delta mean: `{family_subset['selected_mmd_delta_mean']:+.6f}`",
        f"- cross selected frac: `{cross_subset['selected_frac']:.4f}`",
        f"- cross selected pp delta mean: `{cross_subset['selected_pp_delta_mean']:+.6f}`",
        f"- cross non-risk pp delta mean: `{cross_subset['other_pp_delta_mean']:+.6f}`",
        "",
        "## Top Internal Family MMD Risk Datasets",
        "",
        "| dataset | n | pp delta | MMD delta | MMD harm frac | selected |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for ds, row in top:
        lines.append(
            f"| {ds} | {int(row['n'])} | {float(row['pp_delta_mean']):+.6f} | "
            f"{float(row['mmd_delta_mean']):+.6f} | {float(row['mmd_harm_frac']):.3f} | "
            f"{'yes' if ds in risk_datasets else 'no'} |"
        )
    if reasons:
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend(f"- `{r}`" for r in reasons)
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": decision["status"], "risk_dataset_csv": decision["risk_dataset_csv"], "out_md": str(OUT_MD)}, indent=2))
    return 0 if status.endswith("pass_no_gpu") else 1


if __name__ == "__main__":
    raise SystemExit(main())
