#!/usr/bin/env python3
"""Audit all-modality true-cell protocol label compatibility.

This is a CPU-only gate. It checks whether the dose-level SciPlex protocol rows
can be consumed by the current xverse split and latent H5 artifacts, and whether
a drug-level rollup would still preserve a valid scaling estimand.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py


ROOT = Path("/data/cyx/1030/scLatent")
PROTOCOL_TSV = ROOT / "reports/latentfm_true_cell_count_scaling_protocol_20260624/all_modality_fixed64_budget16_32_64.tsv"
BASE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
BASE_DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_label_compatibility_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_LABEL_COMPATIBILITY_GATE_20260625.md"

SCIPLEX_DATASETS = ("sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7")
EXCLUDED_SPLIT_KEYS = {"canonical_test_reference"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def decode(x: object) -> str:
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def read_h5_conditions(dataset: str) -> set[str]:
    with h5py.File(BASE_DATA_DIR / f"{dataset}.h5", "r") as h5:
        return {decode(x) for x in h5["conditions"][:]}


def background_from_dataset(dataset: str) -> str:
    return dataset.replace("sciplex3_", "", 1)


def drug_from_protocol_row(row: dict[str, str]) -> str:
    dataset = row.get("dataset") or ""
    bg = background_from_dataset(dataset)
    perturbation = row.get("perturbation") or ""
    if perturbation.startswith(bg + "_"):
        return perturbation[len(bg) + 1 :]
    condition = row.get("condition") or ""
    if condition.startswith(bg + "_"):
        tail = condition[len(bg) + 1 :]
        parts = tail.rsplit("_", 1)
        return parts[0] if len(parts) == 2 else tail
    return perturbation or condition


def protocol_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with PROTOCOL_TSV.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("dataset") in SCIPLEX_DATASETS:
                rows.append(row)
    return rows


def split_roles(groups: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    train = {str(x) for x in groups.get("train") or []}
    canonical = {str(x) for x in groups.get("canonical_test_reference") or []}
    eval_set: set[str] = set()
    for key, values in groups.items():
        if key == "train" or key in EXCLUDED_SPLIT_KEYS or not isinstance(values, list):
            continue
        eval_set.update(str(x) for x in values)
    return train, eval_set, canonical


def summarize_dataset(dataset: str, rows: list[dict[str, str]], split: dict[str, Any]) -> dict[str, Any]:
    ds_rows = [r for r in rows if r.get("dataset") == dataset]
    dose_conditions = {r.get("condition") or "" for r in ds_rows}
    drug_by_dose = {r.get("condition") or "": drug_from_protocol_row(r) for r in ds_rows}
    protocol_drugs = set(drug_by_dose.values())
    dose_conditions_by_drug: dict[str, set[str]] = defaultdict(set)
    for r in ds_rows:
        dose_conditions_by_drug[drug_from_protocol_row(r)].add(str(r.get("condition") or ""))

    h5_conditions = read_h5_conditions(dataset)
    train, eval_set, canonical = split_roles(split.get(dataset) or {})

    direct_train = dose_conditions & train
    direct_eval = dose_conditions & eval_set
    direct_h5 = dose_conditions & h5_conditions
    direct_canonical = dose_conditions & canonical

    drug_train = protocol_drugs & train
    drug_eval = protocol_drugs & eval_set
    drug_h5 = protocol_drugs & h5_conditions
    drug_canonical = protocol_drugs & canonical

    return {
        "dataset": dataset,
        "protocol_dose_conditions": len(dose_conditions),
        "protocol_drugs": len(protocol_drugs),
        "protocol_drugs_with_multiple_dose_conditions": sum(
            1 for vals in dose_conditions_by_drug.values() if len(vals) > 1
        ),
        "h5_conditions": len(h5_conditions),
        "split_train_conditions": len(train),
        "split_eval_conditions_excluding_canonical": len(eval_set),
        "split_canonical_reference_conditions": len(canonical),
        "direct_dose_key": {
            "h5_overlap": len(direct_h5),
            "train_overlap": len(direct_train),
            "eval_overlap_excluding_canonical": len(direct_eval),
            "canonical_overlap": len(direct_canonical),
            "examples_missing_from_h5": sorted(dose_conditions - h5_conditions)[:8],
        },
        "drug_rollup_key": {
            "h5_overlap": len(drug_h5),
            "train_overlap": len(drug_train),
            "eval_overlap_excluding_canonical": len(drug_eval),
            "canonical_overlap": len(drug_canonical),
            "examples_train": sorted(drug_train)[:8],
            "examples_canonical": sorted(drug_canonical)[:8],
        },
    }


def decide(rows: list[dict[str, Any]]) -> tuple[str, list[str], dict[str, Any]]:
    direct_train = sum(r["direct_dose_key"]["train_overlap"] for r in rows)
    direct_eval = sum(r["direct_dose_key"]["eval_overlap_excluding_canonical"] for r in rows)
    direct_h5 = sum(r["direct_dose_key"]["h5_overlap"] for r in rows)
    rollup_train = sum(r["drug_rollup_key"]["train_overlap"] for r in rows)
    rollup_eval = sum(r["drug_rollup_key"]["eval_overlap_excluding_canonical"] for r in rows)
    rollup_canonical = sum(r["drug_rollup_key"]["canonical_overlap"] for r in rows)
    multi_dose_drugs = sum(r["protocol_drugs_with_multiple_dose_conditions"] for r in rows)

    reasons: list[str] = []
    if direct_h5 == 0:
        reasons.append("dose_level_protocol_conditions_absent_from_current_xverse_h5")
    if direct_train == 0:
        reasons.append("dose_level_protocol_conditions_absent_from_current_train_split")
    if direct_eval == 0:
        reasons.append("dose_level_protocol_conditions_absent_from_query_blind_eval_split")
    if rollup_train > 0:
        reasons.append("drug_rollup_recovers_train_overlap_but_collapses_dose_estimand")
    if rollup_eval == 0:
        reasons.append("drug_rollup_has_no_query_blind_eval_conditions")
    if rollup_canonical > 0:
        reasons.append("only_drug_level_eval_overlap_is_canonical_reference_forbidden_for_selection")
    if multi_dose_drugs > 0:
        reasons.append("many_protocol_drugs_have_multiple_dose_rows_requiring_dose_aware_artifacts")

    status = "allmodality_label_compatibility_fail_no_gpu" if reasons else "allmodality_label_compatibility_pass_cpu_next"
    summary = {
        "direct_dose_h5_overlap": direct_h5,
        "direct_dose_train_overlap": direct_train,
        "direct_dose_eval_overlap_excluding_canonical": direct_eval,
        "drug_rollup_train_overlap": rollup_train,
        "drug_rollup_eval_overlap_excluding_canonical": rollup_eval,
        "drug_rollup_canonical_overlap": rollup_canonical,
        "protocol_drugs_with_multiple_dose_conditions": multi_dose_drugs,
    }
    return status, reasons, summary


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Label Compatibility Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only label/split/artifact compatibility audit.",
        "- Reads the all-modality true-cell protocol manifest, xverse train-only split, and current latent H5 condition keys.",
        "- Does not train, infer, read canonical metrics, read canonical multi, read held-out Track C query, or use GPU.",
        "",
        "## Summary",
        "",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Per-Dataset Compatibility",
            "",
            "| dataset | dose protocol rows | protocol drugs | direct H5 | direct train | direct eval excl canonical | drug train | drug eval excl canonical | drug canonical | multi-dose-condition drugs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["dataset_rows"]:
        lines.append(
            "| {dataset} | {protocol_dose_conditions} | {protocol_drugs} | {dh5} | {dtrain} | {deval} | {rtrain} | {reval} | {rcanon} | {multi} |".format(
                dataset=row["dataset"],
                protocol_dose_conditions=row["protocol_dose_conditions"],
                protocol_drugs=row["protocol_drugs"],
                dh5=row["direct_dose_key"]["h5_overlap"],
                dtrain=row["direct_dose_key"]["train_overlap"],
                deval=row["direct_dose_key"]["eval_overlap_excluding_canonical"],
                rtrain=row["drug_rollup_key"]["train_overlap"],
                reval=row["drug_rollup_key"]["eval_overlap_excluding_canonical"],
                rcanon=row["drug_rollup_key"]["canonical_overlap"],
                multi=row["protocol_drugs_with_multiple_dose_conditions"],
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- Use current partial all-modality artifacts: `{payload['use_current_partial_artifacts']}`",
            "- Reasons:",
        ]
    )
    for reason in payload["reasons"]:
        lines.append(f"  - `{reason}`")
    lines.extend(
        [
            "",
            "## Next Action",
            "",
            "Reopen all-modality scaling only with a dose-aware artifact path: either regenerate dose-level latent H5 conditions from raw/source cell-level data with a query-blind train/internal chemical split, or explicitly define a drug-level all-modality branch that does not claim dose scaling. The current drug-level rollup is not enough because it collapses the dose estimand and has no query-blind chemical eval conditions.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = protocol_rows()
    split = load_json(BASE_SPLIT)
    dataset_rows = [summarize_dataset(ds, rows, split) for ds in SCIPLEX_DATASETS]
    status, reasons, summary = decide(dataset_rows)
    payload = {
        "status": status,
        "boundary": {
            "cpu_only": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "uses_gpu": False,
            "excluded_split_keys": sorted(EXCLUDED_SPLIT_KEYS),
        },
        "inputs": {
            "protocol_tsv": str(PROTOCOL_TSV),
            "base_split": str(BASE_SPLIT),
            "base_data_dir": str(BASE_DATA_DIR),
        },
        "summary": summary,
        "dataset_rows": dataset_rows,
        "reasons": reasons,
        "gpu_authorized": False,
        "use_current_partial_artifacts": False,
        "next_action": "build_dose_aware_artifact_path_or_define_drug_level_branch_with_new_query_blind_eval",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
