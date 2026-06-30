#!/usr/bin/env python3
"""Schema-specific gate for Jiang author-DE background artifacts.

CPU/report-only. This checks whether the background-resolved Jiang author-DE
source can be legally associated with frozen LatentFM outcomes. It does not
train, infer, select checkpoints, read canonical multi for selection, read
Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BG_CSV = ROOT / "reports/jiang_author_de_artifacts_20260627/jiang_author_de_background_artifacts.csv"
COND_GATE_JSON = ROOT / "reports/latentfm_jiang_author_de_artifact_gate_20260627.json"
EXPOSURE_JSON = ROOT / "reports/latentfm_xverse_jiang_background_exposure_gate_20260624.json"
OUT_DIR = ROOT / "reports/jiang_background_schema_specific_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_jiang_background_schema_specific_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_JIANG_BACKGROUND_SCHEMA_SPECIFIC_GATE_20260627.md"

SEED42_EVAL = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/posthoc_eval_uncapped_20260621/"
    / "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)
SEED43_EVAL = (
    ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
    / "xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/"
    / "posthoc_eval_uncapped_20260621/condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
)

if not SEED43_EVAL.exists():
    SEED43_EVAL = (
        ROOT
        / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
        / "xverse_comp006_endpoint5_8k_seed43_fulleval/posthoc_eval_uncapped_20260621/"
        / "condition_family_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
    )

JIANG_GROUPS = ("test_single", "family_gene", "test_all")
BACKGROUND_KEYS = ("cell_background", "background", "cell_type", "cell_line")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def jiang_eval_rows(path: Path, seed: str) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows: list[dict[str, Any]] = []
    for group in JIANG_GROUPS:
        for row in ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []:
            dataset = str(row.get("dataset", ""))
            if dataset.startswith("Jiang_"):
                rows.append({"seed": seed, "group": group, **row})
    return rows


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Jiang Background Schema-Specific Gate 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only schema-specific audit for Jiang author-DE background artifacts.",
        "- Reads materialized Jiang author-DE background CSV and frozen seed42/seed43 eval JSON schemas only.",
        "- Does not train, infer, select checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Source Coverage",
        "",
        f"- background artifact rows: `{payload['background_artifact_rows']}`",
        f"- datasets: `{payload['source_datasets']}`",
        f"- conditions: `{payload['source_conditions']}`",
        f"- backgrounds: `{payload['source_backgrounds']}`",
        f"- metrics: `{payload['source_metrics']}`",
        f"- Jiang background exposure gate: `{payload['jiang_background_exposure_status']}`",
        "",
        "## Outcome Schema Check",
        "",
        f"- frozen eval rows inspected: `{payload['eval_rows_inspected']}`",
        f"- eval row keys: `{', '.join(payload['eval_row_keys'])}`",
        f"- background key present in eval: `{payload['eval_has_background_key']}`",
        f"- pseudo-replication factor if joined directly: `{payload['pseudo_replication_factor']}`",
        f"- prior condition-aggregate gate: `{payload['condition_aggregate_gate_status']}`",
        "",
        "## Decision",
        "",
    ]
    if payload["reasons"]:
        lines.append("Fail/close reasons:")
        lines.extend(f"- `{reason}`" for reason in payload["reasons"])
        lines.append("")
    lines += [
        "The Jiang author-DE archive remains a useful source for failure analysis,",
        "but current frozen LatentFM outcomes are condition-level, not",
        "background-resolved. A direct background artifact association would",
        "replicate the same outcome across six cell backgrounds and is not a",
        "legal GPU admission gate.",
        "",
        "## Next Valid Gate",
        "",
        "Reopen only after producing a background-resolved evaluator or posthoc",
        "artifact with one row per `(dataset, condition, cell_background)` for the",
        "frozen anchor. The reopened gate must require within-background variation,",
        "stimulus/background shuffle collapse, leave-one-dataset and",
        "leave-one-background robustness, dataset-tail no-harm, and MMD no-harm.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{payload['outputs']['json']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bg_rows = read_csv(BG_CSV)
    eval_rows = jiang_eval_rows(SEED42_EVAL, "seed42") + jiang_eval_rows(SEED43_EVAL, "seed43")
    eval_keys = sorted({key for row in eval_rows for key in row.keys()})
    eval_has_background = any(key in eval_keys for key in BACKGROUND_KEYS)

    by_source_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    metrics = set()
    datasets = set()
    conditions = set()
    backgrounds = set()
    split_counts = Counter()
    for row in bg_rows:
        dataset = row.get("dataset", "")
        condition = row.get("condition", "")
        background = row.get("cell_background", "")
        datasets.add(dataset)
        conditions.add((dataset, condition))
        backgrounds.add(background)
        by_source_key[(dataset, condition)].add(background)
        split_counts[row.get("split", "")] += 1
        for key in row:
            if key.startswith("mean_") or key in {"sig_frac_p05_abs005", "valid_gene_count"}:
                metrics.add(key)

    matched_eval_keys = {
        (str(row.get("dataset", "")), str(row.get("condition", "")))
        for row in eval_rows
        if (str(row.get("dataset", "")), str(row.get("condition", ""))) in by_source_key
    }
    n_bg_per_matched = [len(by_source_key[key]) for key in matched_eval_keys]
    pseudo_factor = round(sum(n_bg_per_matched) / max(1, len(n_bg_per_matched)), 3)

    cond_gate = load_json(COND_GATE_JSON)
    exposure = load_json(EXPOSURE_JSON) if EXPOSURE_JSON.exists() else {}
    exposure_status = ((exposure.get("decision") or {}).get("status")) or "missing"

    reasons = []
    if not eval_has_background:
        reasons.append("frozen_eval_condition_metrics_lack_background_key")
    if pseudo_factor > 1.5:
        reasons.append("direct_background_join_would_pseudoreplicate_condition_outcomes")
    if cond_gate.get("status") != "jiang_author_de_signal_gate_pass_needs_external_audit_no_gpu":
        reasons.append("condition_aggregate_gate_already_failed_shuffle_lodo")
    if exposure_status != "jiang_background_exposure_gate_pass":
        reasons.append("jiang_background_exposure_not_passed")

    status = (
        "jiang_background_schema_specific_gate_pass_needs_background_resolved_eval"
        if not reasons
        else "jiang_background_schema_specific_gate_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "background_artifact_csv": str(BG_CSV),
        "background_artifact_rows": len(bg_rows),
        "source_datasets": len(datasets),
        "source_conditions": len(conditions),
        "source_backgrounds": len(backgrounds),
        "source_metrics": len(metrics),
        "source_split_counts": dict(sorted(split_counts.items())),
        "matched_eval_condition_keys": len(matched_eval_keys),
        "eval_rows_inspected": len(eval_rows),
        "eval_row_keys": eval_keys,
        "eval_has_background_key": eval_has_background,
        "pseudo_replication_factor": pseudo_factor,
        "condition_aggregate_gate_status": cond_gate.get("status"),
        "jiang_background_exposure_status": exposure_status,
        "reasons": reasons,
        "next_valid_gate": {
            "required_outcome_granularity": "dataset_condition_cell_background",
            "required_controls": [
                "within_background_variation",
                "stimulus_background_shuffle_collapse",
                "leave_one_dataset_and_leave_one_background_robustness",
                "dataset_tail_no_harm",
                "mmd_no_harm",
            ],
        },
        "outputs": {"json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "reasons": reasons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
