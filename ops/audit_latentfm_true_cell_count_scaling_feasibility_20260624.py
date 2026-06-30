#!/usr/bin/env python3
"""CPU-only feasibility gate for true cell-count scaling.

This gate uses the frozen scaling-law condition table and asks whether a
cell-count scaling experiment can hold condition identities fixed while varying
per-condition cell budgets. It does not read model outputs or launch training.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path("/data/cyx/1030/scLatent")
TABLE = ROOT / "reports/latentfm_scaling_law_condition_table_20260624.tsv"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_scaling_feasibility_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_SCALING_FEASIBILITY_GATE_20260624.md"

THRESHOLDS = [32, 64, 128, 256, 512, 1024]
MIN_DATASETS = 8
MIN_CONDITIONS = 100
MIN_SOURCE_VERIFIED_DATASETS = 5
MIN_MODALITIES = 2
MIN_CONDITIONS_PER_REQUIRED_MODALITY = 50


def parse_int(value: object) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(str(value)))
    except Exception:
        return 0


def read_rows() -> list[dict[str, str]]:
    with TABLE.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    datasets = {r["dataset"] for r in rows}
    modalities = Counter(r.get("modality") or "unknown" for r in rows)
    ptypes = Counter(r.get("perturbation_type") or "unknown" for r in rows)
    backgrounds = Counter(r.get("backgrounds") or "unknown" for r in rows)
    source_verified = {r["dataset"] for r in rows if r.get("source_quality") == "source_verified"}
    cell_counts = [parse_int(r.get("n_cells")) for r in rows]
    cell_counts = [x for x in cell_counts if x > 0]
    q = {}
    if cell_counts:
        sorted_counts = sorted(cell_counts)
        for name, frac in [("q10", 0.10), ("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q90", 0.90)]:
            idx = min(len(sorted_counts) - 1, max(0, int(round(frac * (len(sorted_counts) - 1)))))
            q[name] = sorted_counts[idx]
    return {
        "n_conditions": len(rows),
        "n_datasets": len(datasets),
        "n_source_verified_datasets": len(source_verified),
        "modalities": dict(modalities),
        "perturbation_types": dict(ptypes),
        "backgrounds_top10": dict(backgrounds.most_common(10)),
        "cell_count_min": min(cell_counts) if cell_counts else 0,
        "cell_count_median": median(cell_counts) if cell_counts else 0,
        "cell_count_max": max(cell_counts) if cell_counts else 0,
        "cell_count_quantiles": q,
    }


def threshold_rows(rows: list[dict[str, str]], threshold: int) -> list[dict[str, str]]:
    return [r for r in rows if parse_int(r.get("n_cells")) >= threshold]


def dataset_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_ds: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    out = []
    for dataset, ds_rows in sorted(by_ds.items()):
        counts = [parse_int(r.get("n_cells")) for r in ds_rows]
        out.append(
            {
                "dataset": dataset,
                "n_conditions": len(ds_rows),
                "n_cells_min": min(counts) if counts else 0,
                "n_cells_median": median(counts) if counts else 0,
                "n_cells_max": max(counts) if counts else 0,
                "modalities": dict(Counter(r.get("modality") or "unknown" for r in ds_rows)),
                "perturbation_types": dict(Counter(r.get("perturbation_type") or "unknown" for r in ds_rows)),
                "source_quality": ds_rows[0].get("source_quality") or "",
                "cell_background_source": ds_rows[0].get("cell_background_source") or "",
            }
        )
    return out


def decide(threshold_summaries: list[dict[str, object]]) -> tuple[str, list[str], int | None]:
    feasible = []
    for row in threshold_summaries:
        modalities = row.get("modalities") or {}
        n_modalities = sum(1 for v in modalities.values() if int(v) > 0)
        reasons = []
        if int(row["n_conditions"]) < MIN_CONDITIONS:
            reasons.append("condition_count_below_min")
        if int(row["n_datasets"]) < MIN_DATASETS:
            reasons.append("dataset_count_below_min")
        if int(row["n_source_verified_datasets"]) < MIN_SOURCE_VERIFIED_DATASETS:
            reasons.append("source_verified_dataset_count_below_min")
        if n_modalities < MIN_MODALITIES:
            reasons.append("both_gene_and_chemical_not_retained")
        for modality in ("gene", "chemical"):
            if int(modalities.get(modality, 0)) < MIN_CONDITIONS_PER_REQUIRED_MODALITY:
                reasons.append(f"{modality}_conditions_below_min")
        row["feasibility_reasons"] = reasons
        row["feasible_fixed_condition_protocol"] = not reasons
        if not reasons:
            feasible.append(int(row["threshold"]))

    if not feasible:
        return "true_cell_count_scaling_feasibility_fail_no_gpu", [
            "no_cell_threshold_retains_enough_fixed_conditions_datasets_modalities"
        ], None

    best = max(feasible)
    return "true_cell_count_scaling_feasibility_pass_protocol_only_no_gpu", [
        "fixed_condition_cell_count_protocol_feasible",
        "performance_signal_still_missing_so_gpu_not_authorized",
    ], best


def render_md(payload: dict[str, object]) -> str:
    lines = [
        "# LatentFM True Cell-Count Scaling Feasibility Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only feasibility audit from the frozen scaling-law condition table.",
        "- Holds condition identity fixed conceptually and varies only required per-condition cell-count thresholds.",
        "- Does not read model outputs, canonical metrics, canonical multi, held-out Track C query, train, infer, or use GPU.",
        "",
        "## Inputs",
        "",
        f"- condition table: `{TABLE}`",
        "",
        "## Overall Inventory",
        "",
        "```json",
        json.dumps(payload["overall"], indent=2, sort_keys=True),
        "```",
        "",
        "## Threshold Summary",
        "",
        "| min cells/condition | conditions | datasets | source-verified datasets | modalities | feasible | reasons |",
        "|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["threshold_summaries"]:
        lines.append(
            "| {threshold} | {n_conditions} | {n_datasets} | {n_source_verified_datasets} | {modalities} | `{feasible_fixed_condition_protocol}` | {reasons} |".format(
                threshold=row["threshold"],
                n_conditions=row["n_conditions"],
                n_datasets=row["n_datasets"],
                n_source_verified_datasets=row["n_source_verified_datasets"],
                modalities=json.dumps(row["modalities"], sort_keys=True),
                feasible_fixed_condition_protocol=row["feasible_fixed_condition_protocol"],
                reasons=", ".join(row["feasibility_reasons"]) or "none",
            )
        )
    lines.extend(
        [
            "",
        "## Decision",
        "",
        f"- best fixed-condition threshold: `{payload['best_threshold']}`",
        f"- suggested all-modality budgets: `{payload['suggested_all_modality_budgets']}`",
        f"- suggested gene-only deep budgets: `{payload['suggested_gene_only_deep_budgets']}`",
        f"- GPU authorized: `{payload['gpu_authorized']}`",
        f"- reasons: `{payload['reasons']}`",
            "",
            "## Recommended Next Action",
            "",
            "If this protocol-only gate passes, the next step is not immediate GPU training. First materialize a fixed-condition cell-count protocol with per-condition subsampling seeds and train-only pert-mean artifacts for at least three cell budgets, plus count-only and dataset-identity controls. GPU can be authorized only after that protocol gate confirms no split/provenance leakage and defines a bounded smoke with tail/no-harm stop rules.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = read_rows()
    threshold_summaries = []
    for threshold in THRESHOLDS:
        subset = threshold_rows(rows, threshold)
        s = summarize(subset)
        s["threshold"] = threshold
        threshold_summaries.append(s)
    status, reasons, best = decide(threshold_summaries)
    suggested_all_modality_budgets = [16, 32, 64] if best and best >= 64 else []
    suggested_gene_only_deep_budgets = [64, 128, 256] if any(
        int(row["threshold"]) >= 256 and int((row.get("modalities") or {}).get("gene", 0)) >= MIN_CONDITIONS
        for row in threshold_summaries
    ) else []
    payload = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "reads_model_outputs": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
        },
        "inputs": {"condition_table": str(TABLE)},
        "overall": summarize(rows),
        "threshold_summaries": threshold_summaries,
        "dataset_rows": dataset_rows(rows),
        "best_threshold": best,
        "suggested_all_modality_budgets": suggested_all_modality_budgets,
        "suggested_gene_only_deep_budgets": suggested_gene_only_deep_budgets,
        "gpu_authorized": False,
        "reasons": reasons,
        "next_action": "materialize_fixed_condition_cell_count_protocol_before_any_gpu" if best else "do_not_reopen_cell_count_scaling_without_more_cell_depth",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_md(payload))
    print(json.dumps({"status": status, "best_threshold": best, "gpu_authorized": False, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
