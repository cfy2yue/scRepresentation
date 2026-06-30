#!/usr/bin/env python3
"""Stream-filter Simplicity pharmacogenomic summaries for SciPlex overlap.

This is a CPU/network source-materialization preflight. It downloads only
processed summary TSVs from Simplicity/OSF, filters rows for A549/K562/MCF7,
and joins by normalized drug name to local SciPlex conditions. It does not
train, run inference, select checkpoints, read canonical multi for selection,
read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_DIR = ROOT / "reports" / "sciplex_pharmacogenomic_source_preflight_20260630"
FILTER_DIR = OUT_DIR / "filtered_sources"
FILTER_DIR.mkdir(parents=True, exist_ok=True)

COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
CAP120_SPLIT = (
    ROOT
    / "dataset/biFlow_data/xverse_scaling_splits_v2_20260624"
    / "split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
)

SOURCES = {
    "CTRPv2": {
        "url": "https://osf.io/wm5jq/download",
        "expected_size_bytes": 100473603,
    },
    "GDSC1": {
        "url": "https://osf.io/aub4p/download",
        "expected_size_bytes": 88333234,
    },
    "GDSC2": {
        "url": "https://osf.io/tzkd2/download",
        "expected_size_bytes": 45186577,
    },
    "PRISM_Repurposing": {
        "url": "https://osf.io/awydb/download",
        "expected_size_bytes": 195329867,
    },
}

TARGET_DATASETS = {
    "sciplex3_A549": ["A549", "A-549"],
    "sciplex3_K562": ["K562", "K-562"],
    "sciplex3_MCF7": ["MCF7", "MCF-7"],
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def split_key(key: str) -> tuple[str, str]:
    key = str(key).strip().strip('"')
    if ":|:" not in key:
        return key, ""
    cell, drug = key.split(":|:", 1)
    cell = re.sub(r"\s*\\(RRID:[^)]+\\)", "", cell).strip()
    return cell, drug.strip()


def dataset_from_cell(cell: str) -> str | None:
    n = normalize(cell)
    if "a549" in n:
        return "sciplex3_A549"
    if "k562" in n:
        return "sciplex3_K562"
    if "mcf7" in n:
        return "sciplex3_MCF7"
    return None


def line_may_match(line: str) -> bool:
    low = line.lower()
    return any(tok.lower() in low for toks in TARGET_DATASETS.values() for tok in toks)


def load_local_targets() -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    meta = json.loads(COND_META.read_text(encoding="utf-8"))
    split = json.loads(CAP120_SPLIT.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for dataset in TARGET_DATASETS:
        all_conditions = sorted(meta.get(dataset, {}))
        group_by_condition: dict[str, str] = {}
        for group, vals in split.get(dataset, {}).items():
            if isinstance(vals, list):
                for val in vals:
                    group_by_condition[str(val)] = str(group)
        for condition in all_conditions:
            row = {
                "dataset": dataset,
                "cell_line": dataset.replace("sciplex3_", ""),
                "condition": condition,
                "condition_norm": normalize(condition),
                "split_group": group_by_condition.get(condition, "not_in_cap120_parent"),
            }
            rows.append(row)
            lookup[(dataset, row["condition_norm"])] = row
    return rows, lookup


def stream_filter_source(source_name: str, url: str, target_lookup: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    filtered_path = FILTER_DIR / f"{source_name}_simplicity_sciplex_filtered.tsv"
    matched_rows: list[dict[str, Any]] = []
    bytes_read = 0
    lines_seen = 0
    filtered_lines = 0
    exact_matches = 0
    started = time.time()
    print(f"[{now()}] streaming {source_name} {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "LatentFM-source-preflight/20260630"})
    with urllib.request.urlopen(req, timeout=120) as response, filtered_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_handle:
        header: list[str] | None = None
        writer: csv.DictWriter | None = None
        for raw in response:
            bytes_read += len(raw)
            line = raw.decode("utf-8", errors="replace")
            lines_seen += 1
            if header is None:
                header = next(csv.reader([line], delimiter="\t"))
                out_fields = ["source"] + header
                writer = csv.DictWriter(out_handle, fieldnames=out_fields, delimiter="\t")
                writer.writeheader()
                continue
            if not line_may_match(line):
                continue
            if header is None or writer is None:
                continue
            try:
                values = next(csv.reader([line], delimiter="\t"))
            except csv.Error:
                continue
            if len(values) != len(header):
                continue
            row = dict(zip(header, values))
            key = row.get("Key") or row.get('"Key"') or ""
            cell, drug = split_key(key)
            dataset = dataset_from_cell(cell)
            if dataset is None:
                continue
            filtered_lines += 1
            writer.writerow({"source": source_name, **row})
            local = target_lookup.get((dataset, normalize(drug)))
            if local is None:
                continue
            exact_matches += 1
            auc_cols = [col for col in header if col.startswith("AUC_")]
            percentile_cols = [col for col in header if col.startswith("Percentile_AUC")]
            matched_rows.append(
                {
                    "source": source_name,
                    "dataset": dataset,
                    "cell_line": local["cell_line"],
                    "split_group": local["split_group"],
                    "sciplex_condition": local["condition"],
                    "source_cell_line": cell,
                    "source_drug": drug,
                    "source_key": key,
                    "drug_norm": normalize(drug),
                    "AUC_all": row.get(next((c for c in auc_cols if "all_ccl" in c.lower()), ""), ""),
                    "AUC_mode": row.get(next((c for c in auc_cols if "mode_ccl" in c.lower()), ""), ""),
                    "Percentile_AUC_All_CCL": row.get(
                        next((c for c in percentile_cols if "All_CCL" in c), ""), ""
                    ),
                    "Percentile_AUC_Mode_CCL": row.get(
                        next((c for c in percentile_cols if "Mode_CCL" in c), ""), ""
                    ),
                    "IC50": row.get("IC50", ""),
                    "Percentile_IC50": row.get("Percentile_IC50", ""),
                    "RSE": row.get("RSE", ""),
                }
            )
    elapsed = time.time() - started
    print(
        f"[{now()}] done {source_name}: bytes={bytes_read} lines={lines_seen} "
        f"filtered={filtered_lines} exact_matches={exact_matches} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "source": source_name,
        "url": url,
        "filtered_path": str(filtered_path),
        "bytes_read": bytes_read,
        "lines_seen": lines_seen,
        "filtered_lines": filtered_lines,
        "exact_matches": exact_matches,
        "elapsed_seconds": elapsed,
        "matched_rows": matched_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(local_rows: list[dict[str, str]], match_rows: list[dict[str, Any]], source_stats: list[dict[str, Any]]) -> dict[str, Any]:
    local_train = {
        (row["dataset"], row["condition_norm"])
        for row in local_rows
        if row["split_group"] == "train"
    }
    all_local = {(row["dataset"], row["condition_norm"]) for row in local_rows}
    by_source = defaultdict(list)
    for row in match_rows:
        by_source[row["source"]].append(row)
    exact_keys = {(row["dataset"], row["drug_norm"]) for row in match_rows}
    train_keys = exact_keys & local_train
    source_blocks_per_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in match_rows:
        source_blocks_per_key[(row["dataset"], row["drug_norm"])].add(row["source"])
    multi_source_train = {
        key for key in train_keys if len(source_blocks_per_key.get(key, set())) >= 2
    }
    per_source_rows = []
    for source, rows in sorted(by_source.items()):
        keys = {(row["dataset"], row["drug_norm"]) for row in rows}
        train_source_keys = keys & local_train
        per_source_rows.append(
            {
                "source": source,
                "exact_rows": len(rows),
                "unique_drug_cell_rows": len(keys),
                "train_unique_drug_cell_rows": len(train_source_keys),
                "datasets": len({row["dataset"] for row in rows}),
                "unique_drugs": len({row["drug_norm"] for row in rows}),
            }
        )
    dataset_counts = Counter(row["dataset"] for row in match_rows)
    split_counts = Counter(row["split_group"] for row in match_rows)
    pass_reasons: list[str] = []
    if len(train_keys) < 50:
        pass_reasons.append("train_overlap_rows_lt_50")
    if len(multi_source_train) < 50:
        pass_reasons.append("train_multisource_overlap_rows_lt_50")
    if len({ds for ds, _ in train_keys}) < 3:
        pass_reasons.append("train_backgrounds_lt_3")
    if sum(1 for row in per_source_rows if row["train_unique_drug_cell_rows"] > 0) < 2:
        pass_reasons.append("source_blocks_with_train_overlap_lt_2")

    return {
        "local_sciplex_conditions": len(all_local),
        "local_cap120_train_rows": len(local_train),
        "all_exact_overlap_unique_drug_cell_rows": len(exact_keys),
        "train_exact_overlap_unique_drug_cell_rows": len(train_keys),
        "train_multisource_overlap_unique_drug_cell_rows": len(multi_source_train),
        "overlap_backgrounds": sorted({ds for ds, _ in exact_keys}),
        "train_overlap_backgrounds": sorted({ds for ds, _ in train_keys}),
        "dataset_counts": dict(dataset_counts),
        "split_group_counts": dict(split_counts),
        "per_source": per_source_rows,
        "source_stats": [
            {k: v for k, v in stat.items() if k != "matched_rows"} for stat in source_stats
        ],
        "ready_for_cpu_admission": len(pass_reasons) == 0,
        "reasons": pass_reasons,
    }


def write_report(summary: dict[str, Any], outputs: dict[str, str]) -> None:
    status = (
        "sciplex_pharmacogenomic_source_preflight_ready_cpu_admission_no_gpu"
        if summary["ready_for_cpu_admission"]
        else "sciplex_pharmacogenomic_source_preflight_fail_no_gpu"
    )
    md_path = Path(outputs["markdown"])
    lines = [
        "# SciPlex Pharmacogenomic Source Preflight",
        "",
        f"Created: `{now()}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/network source-materialization preflight over processed Simplicity summary TSVs.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "- This only decides whether an external source table is worth a strict CPU admission gate.",
        "",
        "## Local SciPlex Universe",
        "",
        f"- local drug-cell-line rows: `{summary['local_sciplex_conditions']}`",
        f"- cap120 train rows: `{summary['local_cap120_train_rows']}`",
        "",
        "## Overlap Summary",
        "",
        f"- all exact overlap rows: `{summary['all_exact_overlap_unique_drug_cell_rows']}`",
        f"- train exact overlap rows: `{summary['train_exact_overlap_unique_drug_cell_rows']}`",
        f"- train multi-source overlap rows: `{summary['train_multisource_overlap_unique_drug_cell_rows']}`",
        f"- train overlap backgrounds: `{', '.join(summary['train_overlap_backgrounds'])}`",
        f"- ready for strict CPU admission: `{summary['ready_for_cpu_admission']}`",
        f"- reasons: `{', '.join(summary['reasons']) if summary['reasons'] else 'none'}`",
        "",
        "## Per-Source Rows",
        "",
        "| source | exact rows | unique drug-cell rows | train rows | backgrounds | unique drugs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_source"]:
        lines.append(
            f"| `{row['source']}` | {row['exact_rows']} | {row['unique_drug_cell_rows']} | "
            f"{row['train_unique_drug_cell_rows']} | {row['datasets']} | {row['unique_drugs']} |"
        )
    lines += [
        "",
        "## Download/Filter Stats",
        "",
        "| source | bytes read | lines seen | filtered lines | exact matches | elapsed sec | filtered path |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["source_stats"]:
        lines.append(
            f"| `{row['source']}` | {row['bytes_read']} | {row['lines_seen']} | "
            f"{row['filtered_lines']} | {row['exact_matches']} | {row['elapsed_seconds']:.1f} | `{row['filtered_path']}` |"
        )
    lines += [
        "",
        "## Decision",
        "",
    ]
    if summary["ready_for_cpu_admission"]:
        lines.append(
            "Proceed to a strict CPU admission gate: source-block/LODO association, shuffle collapse, MMD/no-harm, and dual-baseline dominance. This preflight does not authorize GPU."
        )
    else:
        lines.append(
            "Do not proceed to GPU. If overlap is insufficient, close this external-source route or add a documented synonym/identifier mapping preflight before retrying."
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- JSON: `{outputs['json']}`",
        f"- overlap rows: `{outputs['overlap_rows']}`",
        f"- local target rows: `{outputs['local_rows']}`",
        f"- report: `{outputs['markdown']}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    local_rows, target_lookup = load_local_targets()
    source_stats: list[dict[str, Any]] = []
    all_matches: list[dict[str, Any]] = []
    errors: list[str] = []
    for source, meta in SOURCES.items():
        try:
            stat = stream_filter_source(source, meta["url"], target_lookup)
            source_stats.append(stat)
            all_matches.extend(stat["matched_rows"])
        except Exception as exc:  # keep partial source evidence
            errors.append(f"{source}: {type(exc).__name__}: {exc}")
            print(f"[{now()}] ERROR {errors[-1]}", file=sys.stderr, flush=True)

    local_csv = OUT_DIR / "sciplex_local_target_rows_20260630.csv"
    overlap_csv = OUT_DIR / "sciplex_pharmacogenomic_source_overlap_rows_20260630.csv"
    summary_json = OUT_DIR / "sciplex_pharmacogenomic_source_preflight_20260630.json"
    report_md = OUT_DIR / "LATENTFM_SCIPLEX_PHARMACOGENOMIC_SOURCE_PREFLIGHT_20260630.md"
    write_csv(local_csv, local_rows)
    write_csv(overlap_csv, all_matches)
    summary = summarize(local_rows, all_matches, source_stats)
    if errors:
        summary["errors"] = errors
        summary["ready_for_cpu_admission"] = False
        summary.setdefault("reasons", []).append("source_download_or_parse_errors")
    status = (
        "sciplex_pharmacogenomic_source_preflight_ready_cpu_admission_no_gpu"
        if summary["ready_for_cpu_admission"]
        else "sciplex_pharmacogenomic_source_preflight_fail_no_gpu"
    )
    outputs = {
        "json": str(summary_json),
        "overlap_rows": str(overlap_csv),
        "local_rows": str(local_csv),
        "markdown": str(report_md),
    }
    payload = {
        "timestamp": now(),
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_network_preflight": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
            "uses_gpu": False,
        },
        "sources": SOURCES,
        "summary": summary,
        "outputs": outputs,
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(summary, outputs)
    print(json.dumps({"status": status, **outputs}, indent=2), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
