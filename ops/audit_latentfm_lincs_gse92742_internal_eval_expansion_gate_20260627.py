#!/usr/bin/env python3
"""Feasibility gate for expanding GSE92742 with existing internal eval deltas.

This CPU-only audit scans already completed `posthoc_eval_internal` anchor /
candidate JSON pairs and asks whether they can expand the strict train/gene
LINCS GSE92742 signal-control scope beyond the current 19 conditions. It does
not train, infer, select checkpoints, use canonical multi, read Track C query,
or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
OVERLAP = REPORTS / "lincs_l1000_gse92742_condition_join_gate_20260627/gse92742_s0_overlap_rows.csv"

OUT_JSON = REPORTS / "latentfm_lincs_gse92742_internal_eval_expansion_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_LINCS_GSE92742_INTERNAL_EVAL_EXPANSION_GATE_20260627.md"
OUT_ROWS = REPORTS / "lincs_gse92742_internal_eval_expansion_rows_20260627.csv"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)


def norm_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def fnum(value: object) -> float | None:
    text = norm_text(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def read_overlap() -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with OVERLAP.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("membership") != "train" or row.get("modality") != "gene":
                continue
            key = (norm_text(row.get("dataset")), norm_text(row.get("condition")))
            rec = out.setdefault(
                key,
                {
                    "dataset": key[0],
                    "condition": key[1],
                    "n_lincs_overlap_rows": 0,
                    "lincs_cells": set(),
                    "lincs_types": Counter(),
                    "has_exact_bg": False,
                },
            )
            rec["n_lincs_overlap_rows"] += 1
            if norm_text(row.get("lincs_cell_id")):
                rec["lincs_cells"].add(norm_text(row.get("lincs_cell_id")))
            rec["lincs_types"][norm_text(row.get("lincs_pert_type"))] += 1
            if norm_text(row.get("s0_cell_background")).lower() == norm_text(row.get("lincs_cell_id")).lower():
                rec["has_exact_bg"] = True
    return out


def metrics_by_key(group_obj: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = group_obj.get("condition_metrics")
    if not isinstance(rows, list):
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (norm_text(row.get("dataset")), norm_text(row.get("condition")))
        if key[0] and key[1]:
            out[key] = row
    return out


def scan_pairs(overlap: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paired_dirs = 0
    group_pairs = 0
    all_internal_keys: set[tuple[str, str]] = set()
    parse_errors: list[str] = []
    for cand_path in RUNS.glob("**/posthoc_eval_internal/split_group_eval_candidate_internal_ode20.json"):
        anchor_path = cand_path.with_name("split_group_eval_anchor_internal_ode20.json")
        if not anchor_path.is_file():
            continue
        paired_dirs += 1
        try:
            cand = json.loads(cand_path.read_text(encoding="utf-8"))
            anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            parse_errors.append(f"{cand_path}: {exc}")
            continue
        run_name = cand_path.parents[1].name
        for group in GROUPS:
            cgroup = cand.get("groups", {}).get(group, {})
            agroup = anchor.get("groups", {}).get(group, {})
            cmetrics = metrics_by_key(cgroup if isinstance(cgroup, dict) else {})
            ametrics = metrics_by_key(agroup if isinstance(agroup, dict) else {})
            if not cmetrics or not ametrics:
                continue
            group_pairs += 1
            for key in sorted(set(cmetrics) & set(ametrics)):
                all_internal_keys.add(key)
                if key not in overlap:
                    continue
                c = cmetrics[key]
                a = ametrics[key]
                cpp = fnum(c.get("pearson_pert"))
                app = fnum(a.get("pearson_pert"))
                cmmd = fnum(c.get("test_mmd_clamped"))
                ammd = fnum(a.get("test_mmd_clamped"))
                if cpp is None or app is None or cmmd is None or ammd is None:
                    continue
                rows.append(
                    {
                        "run_name": run_name,
                        "group": group,
                        "candidate_json": str(cand_path),
                        "anchor_json": str(anchor_path),
                        "dataset": key[0],
                        "condition": key[1],
                        "candidate_pp": cpp,
                        "anchor_pp": app,
                        "pp_delta": cpp - app,
                        "candidate_mmd": cmmd,
                        "anchor_mmd": ammd,
                        "mmd_delta": cmmd - ammd,
                        "n_lincs_overlap_rows": overlap[key]["n_lincs_overlap_rows"],
                        "n_lincs_cells": len(overlap[key]["lincs_cells"]),
                        "has_exact_bg": bool(overlap[key]["has_exact_bg"]),
                    }
                )
    summary = {
        "paired_internal_dirs": paired_dirs,
        "internal_group_pairs": group_pairs,
        "unique_internal_eval_conditions": len(all_internal_keys),
        "unique_strict_lincs_train_gene_conditions": len(overlap),
        "unique_joined_conditions": len({(r["dataset"], r["condition"]) for r in rows}),
        "parse_errors": parse_errors[:20],
    }
    return rows, summary


def write_rows(rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_name",
        "group",
        "dataset",
        "condition",
        "candidate_pp",
        "anchor_pp",
        "pp_delta",
        "candidate_mmd",
        "anchor_mmd",
        "mmd_delta",
        "n_lincs_overlap_rows",
        "n_lincs_cells",
        "has_exact_bg",
        "candidate_json",
        "anchor_json",
    ]
    with OUT_ROWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def aggregate_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    out: list[dict[str, Any]] = []
    for (dataset, condition), part in sorted(grouped.items()):
        out.append(
            {
                "dataset": dataset,
                "condition": condition,
                "n_eval_rows": len(part),
                "n_runs": len({r["run_name"] for r in part}),
                "mean_pp_delta": mean(float(r["pp_delta"]) for r in part),
                "mean_mmd_delta": mean(float(r["mmd_delta"]) for r in part),
                "has_exact_bg": any(bool(r["has_exact_bg"]) for r in part),
            }
        )
    return out


def main() -> int:
    boundary = {
        "gpu_used": False,
        "training_or_inference_used": False,
        "canonical_multi_selection_used": False,
        "trackc_heldout_query_used": False,
        "uses_existing_internal_posthoc_only": True,
    }
    missing = [str(OVERLAP)] if not OVERLAP.is_file() else []
    if missing:
        payload = {
            "status": "lincs_gse92742_internal_eval_expansion_missing_inputs_no_gpu",
            "gpu_authorized": False,
            "boundary": boundary,
            "missing": missing,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text("# LINCS GSE92742 Internal Eval Expansion Gate\n\nMissing inputs; no GPU authorized.\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "gpu_authorized": False}, indent=2))
        return 0

    overlap = read_overlap()
    rows, summary = scan_pairs(overlap)
    write_rows(rows)
    cond_rows = aggregate_conditions(rows)
    exact_bg_conditions = sum(1 for row in cond_rows if row["has_exact_bg"])
    dataset_counts = Counter(row["dataset"] for row in cond_rows)
    pp_values = [float(row["mean_pp_delta"]) for row in cond_rows]
    mmd_values = [float(row["mean_mmd_delta"]) for row in cond_rows]

    reasons: list[str] = []
    if len(cond_rows) < 50:
        reasons.append("expanded_internal_overlap_condition_count_below_50")
    if len(dataset_counts) < 3:
        reasons.append("expanded_internal_overlap_dataset_count_below_3")
    if exact_bg_conditions < 3:
        reasons.append("expanded_internal_exact_background_condition_count_below_3")
    reasons.append("heterogeneous_closed_candidate_deltas_are_feasibility_only")
    reasons.append("no_new_train_internal_outcome_rows_beyond_existing_19_strict_conditions")

    status = "lincs_gse92742_internal_eval_expansion_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": boundary,
        "summary": {
            **summary,
            "joined_condition_count": len(cond_rows),
            "joined_dataset_count": len(dataset_counts),
            "joined_exact_background_condition_count": exact_bg_conditions,
            "joined_dataset_counts_top20": dataset_counts.most_common(20),
            "mean_pp_delta_across_joined_conditions": mean(pp_values) if pp_values else None,
            "mean_mmd_delta_across_joined_conditions": mean(mmd_values) if mmd_values else None,
        },
        "reasons": reasons,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "rows": str(OUT_ROWS),
        },
        "next_action": (
            "Do not launch GPU. Existing internal posthoc artifacts cannot "
            "expand the strict GSE92742 train/gene outcome scope beyond 19 "
            "conditions. A real unlock requires new leakage-safe train/internal "
            "condition-level outcomes or a reviewed no-harm launcher gate."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LINCS GSE92742 Internal Eval Expansion Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Scans only existing `posthoc_eval_internal` anchor/candidate JSON pairs.",
        "- Uses only `internal_val_cross_background_seen_gene_proxy` and `internal_val_family_gene_proxy` groups.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C held-out query, or GPU.",
        "",
        "## Evidence",
        "",
        f"- paired internal posthoc dirs: `{summary['paired_internal_dirs']}`",
        f"- internal group pairs: `{summary['internal_group_pairs']}`",
        f"- unique internal eval conditions: `{summary['unique_internal_eval_conditions']}`",
        f"- strict GSE92742 train/gene overlap conditions: `{summary['unique_strict_lincs_train_gene_conditions']}`",
        f"- joined strict conditions with existing internal deltas: `{len(cond_rows)}`",
        f"- joined datasets: `{len(dataset_counts)}`",
        f"- joined exact-background conditions: `{exact_bg_conditions}`",
        f"- mean pp delta over joined conditions: `{payload['summary']['mean_pp_delta_across_joined_conditions']}`",
        f"- mean MMD delta over joined conditions: `{payload['summary']['mean_mmd_delta_across_joined_conditions']}`",
        "",
        "## Decision",
        "",
        "Existing internal posthoc artifacts do not unlock a larger GSE92742 strict signal/control scope: the intersection remains the same 19 train/gene conditions. These heterogeneous closed-candidate deltas are useful only as feasibility evidence, not as launch-selection proof.",
        "",
        "## Reasons",
        "",
        *[f"- `{reason}`" for reason in reasons],
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_ROWS}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
