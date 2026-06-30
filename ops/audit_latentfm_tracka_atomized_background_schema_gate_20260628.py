#!/usr/bin/env python3
"""Atomized background schema gate for Track A tail analysis.

This CPU-only gate checks whether composite cell-background strings explain
Track A tails strongly enough to justify a background-set representation smoke.
It does not create a paired candidate model; therefore it can only authorize a
future implementation if the schema signal is strong, non-dataset-only, and
control-safe. Current output is expected to be a conservative go/no-go report.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
IN_DIR = ROOT / "reports/latentfm_xverse_8k_seed_ensemble_internal_means_20260627"
SEED_FILES = {
    "seed42": IN_DIR / "seed42_internal_split_group_means_evalseed42.json",
    "seed43": IN_DIR / "seed43_internal_split_group_means_evalseed42.json",
}
METAINFOS = (
    ROOT / "dataset/raw/genepert_DE5000/metainfo.json",
    ROOT / "dataset/raw/chemicalpert_DE5000/metainfo.json",
)
EXPLICIT_ROWS = ROOT / "reports/tracka_explicit_group_proxy_benchmark_20260628/condition_rows.csv"
OUT_JSON = ROOT / "reports/latentfm_tracka_atomized_background_schema_gate_20260628.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_ATOMIZED_BACKGROUND_SCHEMA_GATE_20260628.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)


def load_metainfo() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in METAINFOS:
        if not path.is_file():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        for rec in obj:
            ds = str(rec.get("dataset", "")).strip()
            if not ds:
                continue
            out[ds] = {
                "cell_line": str(rec.get("cell_line", "") or "").strip(),
                "perturbation_type": str(rec.get("perturbation_type", "") or "").strip(),
                "source": str(path),
            }
    return out


def atomize(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"[/;,|]+", text) if p.strip()]
    return parts or [text]


def is_composite(text: str) -> bool:
    return len(atomize(text)) > 1


def bootstrap_diff(a: list[float], b: list[float], *, seed: int = 20260628) -> dict[str, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 or bb.size == 0:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_lt0": 0.0}
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, aa.size, size=(5000, aa.size))
    ib = rng.integers(0, bb.size, size=(5000, bb.size))
    diffs = aa[ia].mean(axis=1) - bb[ib].mean(axis=1)
    return {
        "diff": float(aa.mean() - bb.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_lt0": float(np.mean(diffs < 0.0)),
    }


def summarize_partition(rows: list[dict[str, Any]], value_key: str, flag_key: str = "is_composite") -> dict[str, Any]:
    comp = [float(r[value_key]) for r in rows if bool(r[flag_key])]
    single = [float(r[value_key]) for r in rows if not bool(r[flag_key])]
    return {
        "n_composite": len(comp),
        "n_singleton": len(single),
        "composite_mean": float(np.mean(comp)) if comp else 0.0,
        "singleton_mean": float(np.mean(single)) if single else 0.0,
        "composite_minus_singleton": bootstrap_diff(comp, single),
    }


def load_internal(seed: str, path: Path, meta: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        for row in obj.get("groups", {}).get(group, {}).get("condition_metrics", []):
            ds = str(row.get("dataset", ""))
            cell = meta.get(ds, {}).get("cell_line", "")
            atoms = atomize(cell)
            rows.append(
                {
                    "seed": seed,
                    "group": group,
                    "dataset": ds,
                    "condition": str(row.get("condition", "")),
                    "cell_background": cell,
                    "atoms": atoms,
                    "is_composite": len(atoms) > 1,
                    "atom_count": len(atoms),
                    "perturbation_type": meta.get(ds, {}).get("perturbation_type", ""),
                    "pearson_pert": float(row.get("pearson_pert", 0.0)),
                    "test_mmd_clamped": float(row.get("test_mmd_clamped", row.get("test_mmd", 0.0))),
                }
            )
    return rows


def internal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in GROUPS:
        part = [r for r in rows if str(r["group"]) == group]
        comp_datasets = sorted({r["dataset"] for r in part if r["is_composite"]})
        singleton_datasets = sorted({r["dataset"] for r in part if not r["is_composite"]})
        out[group] = {
            "n": len(part),
            "composite_datasets": comp_datasets,
            "singleton_dataset_count": len(singleton_datasets),
            "pp_partition": summarize_partition(part, "pearson_pert"),
            "mmd_partition": summarize_partition(part, "test_mmd_clamped"),
        }
    return out


def explicit_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cell = str(row.get("cell_background", "") or "")
            atoms = atomize(cell)
            rows.append(
                {
                    "group": str(row.get("explicit_group", "")),
                    "dataset": str(row.get("dataset", "")),
                    "condition": str(row.get("condition", "")),
                    "cell_background": cell,
                    "atoms": atoms,
                    "is_composite": len(atoms) > 1,
                    "pearson_pert": float(row.get("pearson_pert", 0.0)),
                    "pearson_ctrl": float(row.get("pearson_ctrl", 0.0)),
                    "test_mmd_clamped": float(row.get("test_mmd_clamped", 0.0)),
                }
            )
    out: dict[str, Any] = {"available": True, "path": str(path), "groups": {}}
    for group in sorted({r["group"] for r in rows}):
        part = [r for r in rows if r["group"] == group]
        comp_datasets = sorted({r["dataset"] for r in part if r["is_composite"]})
        out["groups"][group] = {
            "n": len(part),
            "composite_datasets": comp_datasets,
            "pp_partition": summarize_partition(part, "pearson_pert"),
            "ctrl_pp_partition": summarize_partition(part, "pearson_ctrl"),
            "mmd_partition": summarize_partition(part, "test_mmd_clamped"),
        }
    return out


def decide(seed_summaries: dict[str, Any], explicit: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for seed, groups in seed_summaries.items():
        for group, s in groups.items():
            pp = s["pp_partition"]["composite_minus_singleton"]
            mmd = s["mmd_partition"]["composite_minus_singleton"]
            if pp["diff"] > -0.03:
                reasons.append(f"{seed}_{group}_composite_pp_not_at_least_0p03_worse")
            if pp["ci_high"] >= 0.0:
                reasons.append(f"{seed}_{group}_composite_pp_ci_overlaps_nonharm")
            if mmd["diff"] < 0.001:
                reasons.append(f"{seed}_{group}_composite_mmd_not_higher_by_0p001")
            if len(s["composite_datasets"]) <= 2:
                reasons.append(f"{seed}_{group}_composite_signal_too_dataset_specific")
    reasons.append("no_paired_candidate_vs_anchor_delta")
    reasons.append("no_atom_shuffle_or_raw_string_candidate_can_improve_without_model_change")
    if explicit.get("available"):
        for group, s in explicit.get("groups", {}).items():
            if s["pp_partition"]["n_composite"] and s["pp_partition"]["composite_minus_singleton"]["diff"] > -0.03:
                reasons.append(f"explicit_{group}_composite_pp_not_strongly_worse")
    status = "tracka_atomized_background_schema_gate_fail_no_gpu"
    return status, sorted(set(reasons))


def main() -> None:
    meta = load_metainfo()
    seed_rows = {seed: load_internal(seed, path, meta) for seed, path in SEED_FILES.items() if path.is_file()}
    seed_summaries = {seed: internal_summary(rows) for seed, rows in seed_rows.items()}
    explicit = explicit_summary(EXPLICIT_ROWS)
    status, reasons = decide(seed_summaries, explicit)
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "metadata_only_schema_audit": True,
            "paired_candidate_vs_anchor_delta_available": False,
            "explicit_tracka_proxy_rows_selection_weight": 0,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
        },
        "metainfo_sources": [str(p) for p in METAINFOS if p.is_file()],
        "seed_summaries": seed_summaries,
        "explicit_locked_context": explicit,
        "decision_reasons": reasons,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Track A Atomized Background Schema Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "CPU-only metadata/schema audit. It joins dataset-level metainfo cell-line strings to frozen internal rows and reports explicit Track A proxy rows as locked context. It does not produce paired candidate-vs-anchor predictions, does not mutate splits, and does not use canonical multi or Track C query.",
        "",
        "## Internal Composite Background Signal",
        "",
        "| seed | group | composite datasets | pp composite-singleton | pp 95% CI | MMD composite-singleton | MMD 95% CI |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for seed, groups in sorted(seed_summaries.items()):
        for group, s in groups.items():
            pp = s["pp_partition"]["composite_minus_singleton"]
            mmd = s["mmd_partition"]["composite_minus_singleton"]
            lines.append(
                f"| `{seed}` | `{group}` | {len(s['composite_datasets'])} | {pp['diff']:+.6f} | "
                f"[{pp['ci_low']:+.6f},{pp['ci_high']:+.6f}] | {mmd['diff']:+.6f} | "
                f"[{mmd['ci_low']:+.6f},{mmd['ci_high']:+.6f}] |"
            )
    lines.extend(["", "## Explicit Track A Locked Context", ""])
    if explicit.get("available"):
        lines.extend(["| group | composite rows | pp composite-singleton | ctrl pp composite-singleton | MMD composite-singleton | composite datasets |", "|---|---:|---:|---:|---:|---:|"])
        for group, s in explicit["groups"].items():
            pp = s["pp_partition"]["composite_minus_singleton"]
            cp = s["ctrl_pp_partition"]["composite_minus_singleton"]
            mmd = s["mmd_partition"]["composite_minus_singleton"]
            lines.append(
                f"| `{group}` | {s['pp_partition']['n_composite']} | {pp['diff']:+.6f} | "
                f"{cp['diff']:+.6f} | {mmd['diff']:+.6f} | {len(s['composite_datasets'])} |"
            )
    else:
        lines.append("Explicit proxy rows were not found.")
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in reasons)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Do not launch a background-token GPU smoke from this evidence. Composite-background rows are a useful failure-analysis axis, but the current evidence is metadata-only, dataset-specific, and lacks paired candidate-vs-anchor deltas or atom-shuffle controls.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "md": str(OUT_MD), "json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
