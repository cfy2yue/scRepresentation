#!/usr/bin/env python3
"""CPU-only Replogle bulk source-materializer readiness gate.

This audit answers whether the already acquired/inspected Replogle author bulk
sources are sufficient to materialize a condition-level external artifact, and
whether that artifact is a legal non-ACK GPU-smoke entry.

It is intentionally report-only: no training, inference, checkpoint selection,
canonical multi selection, Track C query, GPU use, or large h5ad reads.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")

ACQ_MD = ROOT / "reports/LATENTFM_REPLOGLE_BULK_SOURCE_ACQUISITION_20260627.md"
INSPECT_MD = ROOT / "reports/LATENTFM_REPLOGLE_BULK_SOURCE_INSPECTION_20260627.md"
INSPECT_JSON = ROOT / "reports/latentfm_replogle_bulk_source_inspection_20260627.json"
ARTICLE_JSON = ROOT / "reports/external_artifact_sources_20260627/replogle_figshare_bulk/article_20029387.json"
ARTIFACT_JSON = ROOT / "reports/latentfm_replogle_bulk_artifacts_20260627.json"
ARTIFACT_CSV = ROOT / "reports/replogle_bulk_artifacts_20260627/replogle_bulk_condition_artifacts.csv"
STRICT_JSON = ROOT / "reports/latentfm_replogle_bulk_artifact_strict_v2_20260627.json"
STRICT_SUMMARY = ROOT / "reports/replogle_bulk_artifact_strict_v2_20260627/replogle_bulk_strict_v2_summary.csv"
TRAINONLY_JSON = ROOT / "reports/latentfm_replogle_trainonly_internal_difficulty_gate_20260627.json"
TRAINONLY_SUMMARY = (
    ROOT
    / "reports/replogle_trainonly_internal_difficulty_gate_20260627/"
    / "replogle_trainonly_internal_difficulty_summary.csv"
)

OUT_DIR = ROOT / "reports/replogle_bulk_source_materializer_gate_20260627"
OUT_MANIFEST = OUT_DIR / "replogle_bulk_source_materializer_candidate_manifest.csv"
OUT_JSON = ROOT / "reports/latentfm_replogle_bulk_source_materializer_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_REPLOGLE_BULK_SOURCE_MATERIALIZER_GATE_20260627.md"

KEY_CANDIDATES = {"std_leverage_score", "cnv_score_z", "TE_ratio"}
QC_CONTROLS = {"UMI_count_unfiltered", "num_cells_filtered", "mitopercent", "z_gemgroup_UMI"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_artifact_csv(path: Path) -> dict[str, Any]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    datasets: set[str] = set()
    backgrounds: set[str] = set()
    sources: set[str] = set()
    splits: defaultdict[str, int] = defaultdict(int)
    conditions_by_dataset: defaultdict[str, set[str]] = defaultdict(set)
    rows = 0

    if not path.is_file():
        return {"exists": False}

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            dataset = row.get("dataset", "")
            condition = row.get("condition", "")
            source = row.get("source_label", "")
            raw = row.get("raw_column", "")
            value = fnum(row.get("artifact_value"))
            datasets.add(dataset)
            backgrounds.add(row.get("cell_background", ""))
            sources.add(source)
            splits[row.get("split", "")] += 1
            if dataset and condition:
                conditions_by_dataset[dataset].add(condition)
            key = (source, raw)
            item = stats.setdefault(
                key,
                {
                    "source_label": source,
                    "raw_column": raw,
                    "n": 0,
                    "finite_n": 0,
                    "min": None,
                    "max": None,
                    "unique_values": set(),
                    "role": row.get("artifact_role", ""),
                },
            )
            item["n"] += 1
            if value is not None:
                item["finite_n"] += 1
                item["min"] = value if item["min"] is None else min(item["min"], value)
                item["max"] = value if item["max"] is None else max(item["max"], value)
                if len(item["unique_values"]) <= 10000:
                    item["unique_values"].add(round(value, 12))

    variation_rows = []
    for item in stats.values():
        uniq = item.pop("unique_values")
        item["unique_value_count_capped"] = len(uniq)
        item["nonconstant"] = item["unique_value_count_capped"] > 1 and item["min"] != item["max"]
        variation_rows.append(item)

    return {
        "exists": True,
        "rows": rows,
        "datasets": sorted(x for x in datasets if x),
        "backgrounds": sorted(x for x in backgrounds if x),
        "source_labels": sorted(x for x in sources if x),
        "split_counts": dict(sorted(splits.items())),
        "unique_conditions_by_dataset": {k: len(v) for k, v in sorted(conditions_by_dataset.items())},
        "variation_by_source_raw_column": sorted(
            variation_rows, key=lambda x: (str(x["source_label"]), str(x["raw_column"]))
        ),
    }


def summarize_source_files(inspect: dict[str, Any], article: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    article_files = {f.get("name"): f for f in article.get("files", []) if isinstance(f, dict)}
    for label, info in sorted((inspect.get("sources") or {}).items()):
        path = Path(info.get("path", ""))
        file_info = article_files.get(path.name, {})
        selected[label] = {
            "path": str(path),
            "exists": path.is_file(),
            "size": path.stat().st_size if path.is_file() else None,
            "expected_size": info.get("size") or file_info.get("size"),
            "expected_md5": info.get("expected_md5") or file_info.get("computed_md5"),
            "status": info.get("status"),
            "shape": info.get("shape"),
            "local_datasets": info.get("local_datasets", []),
            "overlap_by_condition_col": info.get("overlap_by_condition_col", {}),
        }
    return selected


def source_overlap_summary(inspect: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for label, info in sorted((inspect.get("sources") or {}).items()):
        overlaps = info.get("overlap_by_condition_col") or {}
        best = 0
        best_col = None
        for col, oval in overlaps.items():
            count = int(oval.get("local_overlap_count") or 0)
            if count > best:
                best = count
                best_col = col
        out[label] = {
            "best_overlap_col": best_col,
            "best_local_overlap_count": best,
            "local_datasets": info.get("local_datasets", []),
            "shape": info.get("shape"),
        }
    return out


def write_candidate_manifest(strict_rows: list[dict[str, str]], train_rows: list[dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strict_by_raw = {row.get("raw_column", ""): row for row in strict_rows}
    train_by_raw = {row.get("raw_column", ""): row for row in train_rows}
    raw_columns = sorted((set(strict_by_raw) | set(train_by_raw) | KEY_CANDIDATES | QC_CONTROLS) - {""})
    fields = [
        "raw_column",
        "role",
        "strict_v2_status",
        "strict_v2_reasons",
        "strict_v2_min_signed_rho",
        "strict_v2_min_dataset_signed_rho",
        "strict_v2_max_abs_rho_mmd",
        "trainonly_status",
        "trainonly_reasons",
        "trainonly_discovery_signed_rho",
        "trainonly_confirm_signed_rho",
        "trainonly_confirm_abs_rho_mmd",
        "gpu_entry_decision",
    ]
    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw in raw_columns:
            srow = strict_by_raw.get(raw, {})
            trow = train_by_raw.get(raw, {})
            role = "response_candidate" if raw in KEY_CANDIDATES else "qc_control"
            writer.writerow(
                {
                    "raw_column": raw,
                    "role": role,
                    "strict_v2_status": srow.get("status", ""),
                    "strict_v2_reasons": srow.get("reasons", ""),
                    "strict_v2_min_signed_rho": srow.get("min_signed_rho", ""),
                    "strict_v2_min_dataset_signed_rho": srow.get("min_dataset_signed_rho", ""),
                    "strict_v2_max_abs_rho_mmd": srow.get("max_abs_rho_mmd", ""),
                    "trainonly_status": trow.get("status", ""),
                    "trainonly_reasons": trow.get("reasons", ""),
                    "trainonly_discovery_signed_rho": trow.get("discovery_signed_rho", ""),
                    "trainonly_confirm_signed_rho": trow.get("confirm_signed_rho", ""),
                    "trainonly_confirm_abs_rho_mmd": trow.get("confirm_abs_rho_mmd", ""),
                    "gpu_entry_decision": "blocked_no_gpu",
                }
            )


def main() -> int:
    inspect = load_json(INSPECT_JSON)
    article = load_json(ARTICLE_JSON)
    artifact = load_json(ARTIFACT_JSON)
    strict = load_json(STRICT_JSON)
    trainonly = load_json(TRAINONLY_JSON)
    strict_rows = read_csv_rows(STRICT_SUMMARY)
    train_rows = read_csv_rows(TRAINONLY_SUMMARY)

    csv_summary = summarize_artifact_csv(ARTIFACT_CSV)
    write_candidate_manifest(strict_rows, train_rows)

    source_files = summarize_source_files(inspect, article)
    overlaps = source_overlap_summary(inspect)
    readable_sources = [k for k, v in source_files.items() if v.get("exists") and v.get("status") == "readable"]
    enough_overlap = all(v.get("best_local_overlap_count", 0) >= 50 for v in overlaps.values())
    materializer_ready = (
        len(readable_sources) == 3
        and enough_overlap
        and ARTIFACT_CSV.is_file()
        and int(artifact.get("rows") or 0) > 0
    )

    local_datasets = set(csv_summary.get("datasets", []))
    backgrounds = set(csv_summary.get("backgrounds", []))
    source_labels = set(csv_summary.get("source_labels", []))
    variation_items = csv_summary.get("variation_by_source_raw_column", [])
    nonconstant_items = [item for item in variation_items if item.get("nonconstant")]
    response_candidates = [
        row
        for row in strict_rows
        if row.get("raw_column") in KEY_CANDIDATES and row.get("status") != "pass_gpu_candidate"
    ]

    blockers = [
        "strict_v2_has_no_pass_candidates_and_status_is_fail_no_gpu",
        "trainonly_internal_gate_has_no_signals_and_status_is_fail_no_gpu",
        "candidate_signal_is_mmd_or_qc_confounded_for_key_columns",
        "source_lodo_is_thin_two_backgrounds_only_k562_and_rpe1",
        "source_specific_artifacts_are_diagnostic_only_not_formal_pass",
    ]

    payload = {
        "status": "materializer_ready_but_gpu_entry_blocked_no_gpu"
        if materializer_ready
        else "materializer_not_ready_no_gpu",
        "gpu_authorized": False,
        "condition_level_artifact_materializable": bool(materializer_ready),
        "chemical_v2_ack_required": False,
        "chemical_v2_ack_present": None,
        "boundary": {
            "cpu_report_only": True,
            "no_training": True,
            "no_inference": True,
            "no_gpu": True,
            "canonical_multi_selection_used": False,
            "trackc_heldout_query_used": False,
        },
        "provenance_source_legality": {
            "article_title": article.get("title"),
            "doi": article.get("doi"),
            "license": article.get("license", {}),
            "figshare_url": article.get("figshare_url") or article.get("url_public_html"),
            "selected_source_files": source_files,
            "acquisition_report": str(ACQ_MD),
            "inspection_report": str(INSPECT_MD),
        },
        "overlap_rows": overlaps,
        "materialized_artifact": {
            "json_status": artifact.get("status"),
            "rows": artifact.get("rows"),
            "role_counts": artifact.get("role_counts"),
            "split_counts": artifact.get("split_counts"),
            "dataset_unique_conditions": artifact.get("dataset_unique_conditions"),
            "csv_summary": csv_summary,
        },
        "background_source_diversity": {
            "local_datasets": sorted(local_datasets),
            "backgrounds": sorted(backgrounds),
            "source_labels": sorted(source_labels),
            "source_lodo_possible": len(local_datasets) >= 2,
            "source_lodo_quality": "possible_but_thin_two_backgrounds_not_sufficient_for_gpu_promotion",
        },
        "within_source_variation": {
            "nonconstant_source_raw_columns": len(nonconstant_items),
            "total_source_raw_columns": len(variation_items),
            "all_observed_source_raw_columns_nonconstant": len(nonconstant_items) == len(variation_items)
            if variation_items
            else False,
            "details_in": "materialized_artifact.csv_summary.variation_by_source_raw_column",
        },
        "controls_and_vetoes": {
            "target_background_shuffle_controls_definable": True,
            "qc_controls_available": sorted(QC_CONTROLS),
            "source_background_lodo_possible": len(local_datasets) >= 2,
            "mmd_tail_veto_files_needed": [
                str(STRICT_JSON),
                str(STRICT_SUMMARY),
                str(TRAINONLY_JSON),
                str(TRAINONLY_SUMMARY),
                str(ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json"),
                str(
                    ROOT
                    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
                    / "xverse_comp006_endpoint5_8k_seed42_fulleval/"
                    / "posthoc_eval_uncapped_20260621/"
                    / "split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
                ),
                str(
                    ROOT
                    / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
                    / "xverse_comp006_endpoint5_8k_seed43_fulleval/"
                    / "posthoc_eval_uncapped_20260621/"
                    / "split_group_eval_best_ode20_condition_uncapped_mse2048_mmd2048.json"
                ),
            ],
        },
        "strict_v2_status": strict.get("status"),
        "strict_v2_pass_candidates": strict.get("pass_candidates", []),
        "strict_v2_key_response_candidates": response_candidates,
        "trainonly_status": trainonly.get("status"),
        "trainonly_signals": trainonly.get("signals", []),
        "blockers": blockers,
        "minimum_next_step": (
            "Do not launch Replogle GPU. Treat the existing materializer output as diagnostic "
            "failure-mechanism evidence. Reopen only with a new pre-registered train-only, "
            "deduped, residualized source/background LODO gate that beats QC controls and MMD "
            "vetoes; otherwise move to a genuinely new external condition-level source."
        ),
        "outputs": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "candidate_manifest_csv": str(OUT_MANIFEST),
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Replogle Bulk Source Materializer Gate 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only audit of existing Replogle source acquisition, inspection, materialization, and strict gates.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C held-out query, or GPU.",
        "- Chemical V2 exact ACK is not present and is not used here; this Replogle source is a separate Figshare CC BY 4.0 dataset branch.",
        "",
        "## Materializer Readiness",
        "",
        f"- Source files readable: `{len(readable_sources)}/3`.",
        f"- Source overlaps: `{overlaps}`.",
        f"- Existing materialized rows: `{artifact.get('rows')}`.",
        f"- Unique materialized conditions: `{artifact.get('dataset_unique_conditions')}`.",
        f"- Role counts: `{artifact.get('role_counts')}`.",
        "",
        "Decision: the acquisition/inspection is sufficient to materialize a condition-level external artifact; the existing materializer has already done so.",
        "",
        "## Gate Criteria Check",
        "",
        f"- provenance/source legality: Figshare dataset DOI `{article.get('doi')}`, license `{(article.get('license') or {}).get('name')}`, selected author normalized bulk h5ad files present.",
        f"- overlap rows: K562 essential `{overlaps.get('K562_essential', {}).get('best_local_overlap_count')}`, K562 GWPS `{overlaps.get('K562_gwps', {}).get('best_local_overlap_count')}`, RPE1 `{overlaps.get('RPE1', {}).get('best_local_overlap_count')}`.",
        f"- background/source diversity: `{len(local_datasets)}` local datasets, `{len(backgrounds)}` backgrounds, `{len(source_labels)}` source labels.",
        f"- within-source variation: `{len(nonconstant_items)}/{len(variation_items)}` observed source/raw-column groups are nonconstant.",
        "- target/background shuffle controls: definable from existing strict-v2 and train-only gate rows; QC controls are available.",
        "- source LODO: possible only as a thin K562-vs-RPE1/background holdout, not strong enough by itself for GPU promotion.",
        "- MMD/tail veto: already available from strict-v2 held-out summaries plus train-only internal summaries listed below.",
        "",
        "## GPU Entry Gate",
        "",
        f"- strict-v2 status: `{strict.get('status')}`; pass candidates: `{strict.get('pass_candidates', [])}`.",
        f"- train-only internal status: `{trainonly.get('status')}`; signals: `{trainonly.get('signals', [])}`.",
        "- Target/background shuffle controls and QC controls are definable from existing files, but the current candidates fail the required no-harm checks.",
        "- Source/background LODO is only thinly possible across K562 and RPE1; source-specific artifacts remain diagnostic-only.",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- `{b}`" for b in blockers])
    lines.extend(
        [
            "",
            "## Minimum Next Step",
            "",
            payload["minimum_next_step"],
            "",
            "## Files",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- candidate manifest: `{OUT_MANIFEST}`",
            f"- artifact CSV: `{ARTIFACT_CSV}`",
            f"- strict-v2 summary: `{STRICT_SUMMARY}`",
            f"- train-only summary: `{TRAINONLY_SUMMARY}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorized": False, "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
