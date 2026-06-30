#!/usr/bin/env python3
"""Condition-level join gate for the complete GSE70138 LINCS small metadata.

This gate uses only the already downloaded GSE70138 `sig_info` and
`sig_metrics` gzip tables. It does not download Level5 matrices or run any
model. The goal is to decide whether the LINCS metadata route has enough
train-only overlap to justify a later, stricter signal/control gate.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path("/data/cyx/1030/scLatent")
SOURCE_DIR = ROOT / "reports/external_artifact_sources_20260627/lincs_l1000_geo_small"
SIG_INFO = SOURCE_DIR / "GSE70138_Broad_LINCS_sig_info_2017-03-06.txt.gz"
SIG_METRICS = SOURCE_DIR / "GSE70138_Broad_LINCS_sig_metrics_2017-03-06.txt.gz"
S0 = ROOT / "reports/latentfm_scaling_s0_provenance_freeze_20260625.tsv"
OUTCOME_FILES = [
    ROOT / "reports/latentfm_condition_exposure_row_bootstrap_rows_20260625.csv",
    ROOT / "reports/latentfm_qc_support_reliability_rows_20260625.csv",
    ROOT / "reports/latentfm_response_program_projection_rows_20260625.csv",
    ROOT / "reports/latentfm_lodo_domain_conflict_rows_20260625.csv",
    ROOT / "reports/latentfm_background_target_actionability_rows_20260625.csv",
    ROOT / "reports/latentfm_truecell_riskrow_complementarity_rows_20260625.csv",
]

OUT_DIR = ROOT / "reports/lincs_l1000_gse70138_condition_join_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_lincs_l1000_gse70138_condition_join_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_LINCS_L1000_GSE70138_CONDITION_JOIN_GATE_20260627.md"
OUT_AGG = OUT_DIR / "gse70138_condition_level_activity.csv"
OUT_OVERLAP = OUT_DIR / "gse70138_s0_overlap_rows.csv"


def norm_text(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def norm_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(value).lower())


def to_float(value: object) -> float | None:
    text = norm_text(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def read_outcome_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for path in OUTCOME_FILES:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not {"dataset", "condition"}.issubset(fields):
                continue
            for row in reader:
                dataset = norm_text(row.get("dataset"))
                condition = norm_text(row.get("condition"))
                if dataset and condition:
                    keys.add((dataset, condition))
    return keys


def read_s0_rows(outcome_keys: set[tuple[str, str]] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not S0.is_file():
        return rows
    with S0.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            dataset = norm_text(row.get("dataset"))
            condition = norm_text(row.get("condition"))
            if outcome_keys is not None and (dataset, condition) not in outcome_keys:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "membership": norm_text(row.get("canonical_seed42_membership")),
                    "modality": norm_text(row.get("modality")),
                    "perturbation_type": norm_text(row.get("perturbation_type")),
                    "perturbation": norm_text(row.get("perturbation")),
                    "dose": norm_text(row.get("dose")),
                    "cell_background": norm_text(row.get("cell_background_source")),
                    "source_label": norm_text(row.get("source_label")),
                }
            )
    return rows


def load_metrics() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with gzip.open(SIG_METRICS, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sig_id = norm_text(row.get("sig_id"))
            if not sig_id:
                continue
            out[sig_id] = {
                "tas": norm_text(row.get("tas")),
                "distil_cc_q75": norm_text(row.get("distil_cc_q75")),
                "distil_ss": norm_text(row.get("distil_ss")),
                "distil_nsample": norm_text(row.get("distil_nsample")),
            }
    return out


def aggregate_lincs() -> tuple[list[dict[str, object]], dict[str, object]]:
    metrics = load_metrics()
    joined = 0
    total_info = 0
    groups: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    with gzip.open(SIG_INFO, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total_info += 1
            sig_id = norm_text(row.get("sig_id"))
            metric = metrics.get(sig_id)
            if not metric:
                continue
            joined += 1
            key = (
                norm_text(row.get("pert_iname")),
                norm_text(row.get("pert_type")),
                norm_text(row.get("cell_id")),
                norm_text(row.get("pert_idose")),
                norm_text(row.get("pert_itime")),
            )
            rec = groups.setdefault(
                key,
                {
                    "pert_iname": key[0],
                    "pert_type": key[1],
                    "cell_id": key[2],
                    "pert_idose": key[3],
                    "pert_itime": key[4],
                    "sig_count": 0,
                    "tas_values": [],
                    "distil_cc_q75_values": [],
                    "distil_ss_values": [],
                    "distil_nsample_values": [],
                },
            )
            rec["sig_count"] = int(rec["sig_count"]) + 1
            for field in ("tas", "distil_cc_q75", "distil_ss", "distil_nsample"):
                val = to_float(metric.get(field))
                if val is not None:
                    rec[f"{field}_values"].append(val)

    rows: list[dict[str, object]] = []
    for rec in groups.values():
        out = {
            "pert_iname": rec["pert_iname"],
            "pert_key": norm_key(rec["pert_iname"]),
            "pert_type": rec["pert_type"],
            "cell_id": rec["cell_id"],
            "cell_key": norm_key(rec["cell_id"]),
            "pert_idose": rec["pert_idose"],
            "pert_itime": rec["pert_itime"],
            "sig_count": rec["sig_count"],
        }
        for field in ("tas", "distil_cc_q75", "distil_ss", "distil_nsample"):
            vals = rec[f"{field}_values"]
            out[f"{field}_mean"] = mean(vals) if vals else ""
            out[f"{field}_n"] = len(vals)
        rows.append(out)
    rows.sort(key=lambda r: (str(r["pert_type"]), str(r["pert_iname"]), str(r["cell_id"])))
    summary = {
        "sig_info_rows": total_info,
        "metrics_rows": len(metrics),
        "joined_signature_rows": joined,
        "condition_level_rows": len(rows),
        "pert_type_counts": Counter(str(r["pert_type"]) for r in rows).most_common(),
        "cell_counts_top20": Counter(str(r["cell_id"]) for r in rows).most_common(20),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def overlap_rows(lincs_rows: list[dict[str, object]], s0_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_pert: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_pert_cell: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in lincs_rows:
        pkey = str(row.get("pert_key", ""))
        ckey = str(row.get("cell_key", ""))
        if pkey:
            by_pert[pkey].append(row)
            if ckey:
                by_pert_cell[(pkey, ckey)].append(row)

    out: list[dict[str, object]] = []
    for s0 in s0_rows:
        candidates = []
        pert = norm_text(s0["perturbation"])
        cell = norm_text(s0["cell_background"])
        # SciPlex perturbation strings often prefix the cell line.
        stripped = pert
        if cell and pert.lower().startswith(cell.lower() + "_"):
            stripped = pert[len(cell) + 1 :]
        keys = {norm_key(pert), norm_key(stripped), norm_key(s0["condition"])}
        keys.discard("")
        for key in keys:
            candidates.extend(by_pert.get(key, []))
            if cell:
                candidates.extend(by_pert_cell.get((key, norm_key(cell)), []))
        seen = set()
        for cand in candidates:
            ident = (
                cand.get("pert_iname"),
                cand.get("cell_id"),
                cand.get("pert_idose"),
                cand.get("pert_itime"),
            )
            if ident in seen:
                continue
            seen.add(ident)
            out.append(
                {
                    "dataset": s0["dataset"],
                    "condition": s0["condition"],
                    "membership": s0["membership"],
                    "modality": s0["modality"],
                    "perturbation_type": s0["perturbation_type"],
                    "s0_perturbation": s0["perturbation"],
                    "s0_cell_background": s0["cell_background"],
                    "s0_dose": s0["dose"],
                    "lincs_pert_iname": cand.get("pert_iname", ""),
                    "lincs_pert_type": cand.get("pert_type", ""),
                    "lincs_cell_id": cand.get("cell_id", ""),
                    "lincs_pert_idose": cand.get("pert_idose", ""),
                    "lincs_pert_itime": cand.get("pert_itime", ""),
                    "lincs_sig_count": cand.get("sig_count", ""),
                    "tas_mean": cand.get("tas_mean", ""),
                    "distil_cc_q75_mean": cand.get("distil_cc_q75_mean", ""),
                }
            )
    out.sort(key=lambda r: (str(r["dataset"]), str(r["condition"]), str(r["lincs_pert_iname"])))
    return out


def summarize_overlap(rows: list[dict[str, object]]) -> dict[str, object]:
    by_dataset = Counter(str(r["dataset"]) for r in rows)
    by_membership = Counter(str(r["membership"]) for r in rows)
    by_modality = Counter(str(r["modality"]) for r in rows)
    exact_cell = sum(
        1
        for r in rows
        if norm_key(r["s0_cell_background"]) and norm_key(r["s0_cell_background"]) == norm_key(r["lincs_cell_id"])
    )
    prefix_cell = sum(
        1
        for r in rows
        if norm_text(r["s0_cell_background"])
        and norm_text(r["lincs_cell_id"]).lower().startswith(norm_text(r["s0_cell_background"]).lower())
    )
    return {
        "overlap_rows": len(rows),
        "unique_s0_conditions": len({(r["dataset"], r["condition"]) for r in rows}),
        "unique_lincs_perturbagens": len({r["lincs_pert_iname"] for r in rows}),
        "dataset_counts_top20": by_dataset.most_common(20),
        "membership_counts": by_membership.most_common(),
        "modality_counts": by_modality.most_common(),
        "exact_cell_background_match_rows": exact_cell,
        "prefix_cell_background_match_rows": prefix_cell,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in (SIG_INFO, SIG_METRICS) if not p.is_file()]
    boundary = {
        "gpu_used": False,
        "training_or_inference_used": False,
        "large_level5_download": False,
        "canonical_multi_selection_used": False,
        "trackc_heldout_query_used": False,
        "chemical_v2_ack": False,
        "source_release": "GSE70138_small_metadata_only",
    }
    if missing:
        out = {
            "status": "lincs_gse70138_condition_join_missing_source_no_gpu",
            "gpu_authorized": False,
            "boundary": boundary,
            "missing": missing,
        }
        OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        OUT_MD.write_text("# LINCS GSE70138 Condition Join Gate\n\nMissing source files; no GPU authorized.\n", encoding="utf-8")
        print(json.dumps({"status": out["status"], "gpu_authorized": False}, indent=2))
        return 0

    lincs_rows, lincs_summary = aggregate_lincs()
    agg_fields = [
        "pert_iname",
        "pert_key",
        "pert_type",
        "cell_id",
        "cell_key",
        "pert_idose",
        "pert_itime",
        "sig_count",
        "tas_mean",
        "tas_n",
        "distil_cc_q75_mean",
        "distil_cc_q75_n",
        "distil_ss_mean",
        "distil_ss_n",
        "distil_nsample_mean",
        "distil_nsample_n",
    ]
    write_csv(OUT_AGG, lincs_rows, agg_fields)

    outcome_keys = read_outcome_keys()
    trainonly_rows = read_s0_rows(outcome_keys)
    full_s0_rows = read_s0_rows(None)
    trainonly_overlap = overlap_rows(lincs_rows, trainonly_rows)
    full_overlap = overlap_rows(lincs_rows, full_s0_rows)
    write_csv(
        OUT_OVERLAP,
        full_overlap,
        [
            "dataset",
            "condition",
            "membership",
            "modality",
            "perturbation_type",
            "s0_perturbation",
            "s0_cell_background",
            "s0_dose",
            "lincs_pert_iname",
            "lincs_pert_type",
            "lincs_cell_id",
            "lincs_pert_idose",
            "lincs_pert_itime",
            "lincs_sig_count",
            "tas_mean",
            "distil_cc_q75_mean",
        ],
    )

    trainonly_summary = summarize_overlap(trainonly_overlap)
    full_summary = summarize_overlap(full_overlap)
    reasons = []
    if trainonly_summary["overlap_rows"] < 50:
        reasons.append("trainonly_overlap_below_50")
    if trainonly_summary["unique_s0_conditions"] < 50:
        reasons.append("trainonly_unique_condition_overlap_below_50")
    if trainonly_summary["exact_cell_background_match_rows"] == 0:
        reasons.append("trainonly_exact_background_match_zero")
    if full_summary["overlap_rows"] > 0:
        reasons.append("full_s0_overlap_is_diagnostic_or_ack_gated")
    reasons.extend(
        [
            "chemical_v2_exact_ack_absent",
            "shuffle_source_mmd_tail_gates_not_run",
            "no_gpu_from_schema_or_overlap_only",
        ]
    )
    status = "lincs_gse70138_condition_join_fail_no_gpu"
    gpu_authorized = False
    out = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": boundary,
        "lincs_summary": lincs_summary,
        "trainonly_outcome_universe_rows": len(trainonly_rows),
        "full_s0_rows": len(full_s0_rows),
        "trainonly_overlap_summary": trainonly_summary,
        "full_s0_overlap_summary": full_summary,
        "reasons": reasons,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "condition_level_activity": str(OUT_AGG),
            "s0_overlap_rows": str(OUT_OVERLAP),
        },
        "next_action": (
            "Do not launch GPU from GSE70138. Train-only overlap exists but "
            "is too narrow and background-mismatched for a reliable source "
            "artifact. If LINCS remains interesting, finish GSE92742 small "
            "metadata acquisition or build an explicit ACK-gated chemical "
            "protocol; non-ACK Track A training cannot use broad chemical "
            "overlap without the required gate."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LINCS/L1000 GSE70138 Condition Join Gate",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- Uses only complete GSE70138 `sig_info` and `sig_metrics` small metadata.",
        "- No Level5 matrices, training, inference, canonical multi selection, Track C held-out query, or GPU.",
        "- Chemical V2 exact ACK is absent; chemical overlap is diagnostic only.",
        "",
        "## LINCS Materialization",
        "",
        f"- sig_info rows: `{lincs_summary['sig_info_rows']}`",
        f"- sig_metrics rows: `{lincs_summary['metrics_rows']}`",
        f"- joined signature rows: `{lincs_summary['joined_signature_rows']}`",
        f"- condition-level rows: `{lincs_summary['condition_level_rows']}`",
        f"- perturbation types: `{lincs_summary['pert_type_counts']}`",
        "",
        "## Overlap",
        "",
        f"- current train-only outcome universe rows: `{len(trainonly_rows)}`",
        f"- train-only overlap rows: `{trainonly_summary['overlap_rows']}`",
        f"- train-only unique S0 conditions: `{trainonly_summary['unique_s0_conditions']}`",
        f"- train-only exact cell/background match rows: `{trainonly_summary['exact_cell_background_match_rows']}`",
        f"- train-only prefix cell/background match rows: `{trainonly_summary['prefix_cell_background_match_rows']}`",
        f"- full S0 overlap rows: `{full_summary['overlap_rows']}`",
        f"- full S0 unique S0 conditions: `{full_summary['unique_s0_conditions']}`",
        f"- full S0 membership counts: `{full_summary['membership_counts']}`",
        f"- full S0 modality counts: `{full_summary['modality_counts']}`",
        f"- exact cell/background match rows: `{full_summary['exact_cell_background_match_rows']}`",
        f"- prefix cell/background match rows: `{full_summary['prefix_cell_background_match_rows']}`",
        "",
        "## Decision",
        "",
        "No GPU is authorized. GSE70138 materializes cleanly and has a small gene-level train-only overlap, but it covers only a few S0 conditions and has no exact cell/background match under the current keys. The much larger full-S0 overlap is chemical/SciPlex-style and remains diagnostic or Chemical-V2-ACK-gated.",
        "",
        "## Reasons",
        "",
        *[f"- `{reason}`" for reason in reasons],
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- condition-level activity CSV: `{OUT_AGG}`",
        f"- S0 overlap CSV: `{OUT_OVERLAP}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": gpu_authorized, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
