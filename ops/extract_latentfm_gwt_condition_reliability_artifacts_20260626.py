#!/usr/bin/env python3
"""Extract GWT CD4 T-cell Perturb-seq reliability artifacts.

CPU/source-only. Downloads small CSV supplementary tables from the public GWT
Perturb-seq analysis repository and maps gene-level reliability fields onto
local single-gene train conditions. It does not download h5ad/expression data,
read checkpoints, read canonical multi, read Track C query outputs, train,
infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
import math
import re
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
COND_META = ROOT / "dataset/latentfm_full/xverse/condition_metadata.json"
OUT_DIR = ROOT / "reports/gwt_condition_reliability_artifacts_20260626"
SOURCE_DIR = OUT_DIR / "source_tables"
OUT_MD = ROOT / "reports/LATENTFM_GWT_CONDITION_RELIABILITY_ARTIFACTS_20260626.md"
OUT_JSON = ROOT / "reports/latentfm_gwt_condition_reliability_artifacts_20260626.json"
OUT_CONFIG = ROOT / "configs/latentfm_gwt_condition_reliability_artifact_manifest_20260626.json"

URLS = {
    "guide_kd_efficiency": "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/guide_kd_efficiency.suppl_table.csv",
    "de_stats": "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/DE_stats.suppl_table.csv",
    "k562_comparison": "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/K562_comparison.suppl_table.csv",
    "sample_metadata": "https://raw.githubusercontent.com/emdann/GWT_perturbseq_analysis_2025/master/metadata/suppl_tables/sample_metadata.suppl_table.csv",
}

EVIDENCE_URL = "https://github.com/emdann/GWT_perturbseq_analysis_2025"


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def to_float(value: Any) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def safe_gene(value: str) -> str:
    return norm(value).upper()


def guide_to_gene(guide_id: str) -> str:
    text = norm(guide_id)
    # GWT guide ids are typically SYMBOL-number, e.g. A1BG-2.
    match = re.match(r"^(.+)-\d+$", text)
    return safe_gene(match.group(1) if match else text)


def download_sources() -> dict[str, Path]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for label, url in URLS.items():
        out = SOURCE_DIR / f"{label}.csv"
        if not out.exists() or out.stat().st_size == 0:
            req = urllib.request.Request(url, headers={"User-Agent": "latentfm-gwt-artifact/20260626"})
            with urllib.request.urlopen(req, timeout=120) as response, out.open("wb") as handle:
                handle.write(response.read())
        paths[label] = out
    return paths


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_local_single_gene_conditions() -> list[dict[str, str]]:
    meta = json.loads(COND_META.read_text(encoding="utf-8"))
    rows = []
    for dataset, conds in meta.items():
        for condition, info in conds.items():
            genes = info.get("genes") or []
            if len(genes) != 1:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "target_gene": safe_gene(genes[0]),
                    "perturbation_type": norm(info.get("perturbation_type_raw")),
                }
            )
    return rows


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def build_guide_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_gene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        gene = guide_to_gene(row.get(""))
        if not gene:
            continue
        by_gene[gene].append(row)
    out = {}
    for gene, grows in by_gene.items():
        guide_n = len(grows)
        sig = [norm(r.get("signif_knockdown")).lower() == "true" for r in grows]
        no_effect = [norm(r.get("high_confidence_no_effect_guides")).lower() == "true" for r in grows]
        ranks = [v for v in (to_float(r.get("rank")) for r in grows) if v is not None]
        adj = [v for v in (to_float(r.get("adj_p_value")) for r in grows) if v is not None and v > 0]
        out[gene] = {
            "guide_count": guide_n,
            "signif_knockdown_fraction": sum(sig) / guide_n if guide_n else None,
            "no_effect_guide_fraction": sum(no_effect) / guide_n if guide_n else None,
            "mean_rank": mean(ranks),
            "mean_neglog10_adj_p": mean([-math.log10(v) for v in adj]),
        }
    return out


def build_de_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_gene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        gene = safe_gene(row.get("target_contrast_gene_name"))
        if gene:
            by_gene[gene].append(row)
    out = {}
    for gene, grows in by_gene.items():
        crossguide = [v for v in (to_float(r.get("crossguide_correlation")) for r in grows) if v is not None]
        crossdonor = [v for v in (to_float(r.get("crossdonor_correlation_mean")) for r in grows) if v is not None]
        n_cells = [v for v in (to_float(r.get("n_cells_target")) for r in grows) if v is not None]
        de_counts = [v for v in (to_float(r.get("n_total_de_genes")) for r in grows) if v is not None]
        ontarget = [norm(r.get("ontarget_significant")).lower() == "true" for r in grows]
        cultures = sorted({norm(r.get("culture_condition")) for r in grows if norm(r.get("culture_condition"))})
        out[gene] = {
            "de_row_count": len(grows),
            "culture_conditions": ";".join(cultures),
            "culture_condition_count": len(cultures),
            "crossguide_correlation_mean": mean(crossguide),
            "crossdonor_correlation_mean": mean(crossdonor),
            "n_cells_target_mean": mean(n_cells),
            "n_total_de_genes_mean": mean(de_counts),
            "ontarget_significant_fraction": sum(ontarget) / len(grows) if grows else None,
        }
    return out


def build_k562_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    by_gene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        gene = safe_gene(row.get("target_contrast_gene_name"))
        if gene:
            by_gene[gene].append(row)
    out = {}
    for gene, grows in by_gene.items():
        donor = [v for v in (to_float(r.get("donor_correlation_mean")) for r in grows) if v is not None]
        logfc = [v for v in (to_float(r.get("logfc_pearson_r")) for r in grows) if v is not None]
        out[gene] = {
            "k562_row_count": len(grows),
            "k562_donor_correlation_mean": mean(donor),
            "k562_logfc_pearson_mean": mean(logfc),
        }
    return out


def write_artifact(path: Path, rows: list[dict[str, Any]], extra_fields: list[str]) -> None:
    fields = [
        "dataset",
        "condition",
        "artifact_value",
        "target_gene",
        "source",
        "evidence_url",
    ] + extra_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def artifact_rows(
    local_rows: list[dict[str, str]],
    metric_by_gene: dict[str, dict[str, Any]],
    metric_key: str,
    source_name: str,
    extra_keys: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for local in local_rows:
        metrics = metric_by_gene.get(local["target_gene"])
        if not metrics:
            continue
        value = metrics.get(metric_key)
        if value is None:
            continue
        rows.append(
            {
                **local,
                "artifact_value": value,
                "source": source_name,
                "evidence_url": EVIDENCE_URL,
                **{key: metrics.get(key, "") for key in extra_keys},
            }
        )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    source_paths = download_sources()
    local_rows = load_local_single_gene_conditions()

    guide_metrics = build_guide_metrics(read_csv(source_paths["guide_kd_efficiency"]))
    de_metrics = build_de_metrics(read_csv(source_paths["de_stats"]))
    k562_metrics = build_k562_metrics(read_csv(source_paths["k562_comparison"]))

    artifacts = [
        {
            "artifact": "external_gwt_signif_knockdown_fraction",
            "description": "Fraction of GWT guides for this target with significant knockdown. External gene-reliability prior; not dataset-matched condition evidence.",
            "metric_by_gene": guide_metrics,
            "metric_key": "signif_knockdown_fraction",
            "source_name": "GWT guide_kd_efficiency.suppl_table.csv",
            "extra_keys": ["guide_count", "no_effect_guide_fraction", "mean_rank", "mean_neglog10_adj_p"],
            "priority": 1,
        },
        {
            "artifact": "external_gwt_crossguide_correlation_mean",
            "description": "Mean cross-guide correlation from GWT DE_stats across culture conditions. External gene-reliability prior.",
            "metric_by_gene": de_metrics,
            "metric_key": "crossguide_correlation_mean",
            "source_name": "GWT DE_stats.suppl_table.csv",
            "extra_keys": ["de_row_count", "culture_condition_count", "culture_conditions", "crossdonor_correlation_mean", "n_cells_target_mean", "n_total_de_genes_mean", "ontarget_significant_fraction"],
            "priority": 2,
        },
        {
            "artifact": "external_gwt_crossdonor_correlation_mean",
            "description": "Mean cross-donor correlation from GWT DE_stats across culture conditions. External gene-reliability prior.",
            "metric_by_gene": de_metrics,
            "metric_key": "crossdonor_correlation_mean",
            "source_name": "GWT DE_stats.suppl_table.csv",
            "extra_keys": ["de_row_count", "culture_condition_count", "culture_conditions", "crossguide_correlation_mean", "n_cells_target_mean", "n_total_de_genes_mean", "ontarget_significant_fraction"],
            "priority": 3,
        },
        {
            "artifact": "external_gwt_k562_logfc_pearson_mean",
            "description": "GWT comparison of CD4+ T-cell target response to K562 logFC response. External context/transfer prior, not dataset-matched evidence.",
            "metric_by_gene": k562_metrics,
            "metric_key": "k562_logfc_pearson_mean",
            "source_name": "GWT K562_comparison.suppl_table.csv",
            "extra_keys": ["k562_row_count", "k562_donor_correlation_mean"],
            "priority": 4,
        },
    ]

    manifest_artifacts = []
    artifact_summaries = []
    for spec in artifacts:
        rows = artifact_rows(
            local_rows,
            spec["metric_by_gene"],
            spec["metric_key"],
            spec["source_name"],
            spec["extra_keys"],
        )
        out_csv = OUT_DIR / f"{spec['artifact']}.csv"
        write_artifact(out_csv, rows, ["perturbation_type"] + spec["extra_keys"])
        datasets = sorted({r["dataset"] for r in rows})
        genes = sorted({r["target_gene"] for r in rows})
        artifact_summaries.append(
            {
                "artifact": spec["artifact"],
                "rows": len(rows),
                "datasets": len(datasets),
                "genes": len(genes),
                "output": str(out_csv),
            }
        )
        manifest_artifacts.append(
            {
                "artifact": spec["artifact"],
                "description": spec["description"],
                "priority": spec["priority"],
                "source_files": [str(out_csv.relative_to(ROOT))],
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["target_gene", "perturbation_type", "source", "evidence_url"] + spec["extra_keys"],
                "minimum_datasets": 3,
                "minimum_overlap_rows": 50,
                "minimum_varying_datasets": 3,
                "promotion_note": "External GWT gene-reliability prior; requires strict preflight plus shuffle/LODO/source/static-prior controls before any GPU.",
            }
        )

    manifest = {
        "version": "20260626_gwt_condition_reliability",
        "boundary": {
            "source": "GWT CD4 T-cell Perturb-seq supplementary small tables mapped to local single-gene conditions",
            "uses_train_only_internal_rows": True,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "uses_training": False,
            "uses_gpu": False,
            "downloads_large_data": False,
        },
        "source_urls": URLS,
        "artifacts": manifest_artifacts,
    }
    OUT_CONFIG.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "timestamp": timestamp,
        "status": "gwt_condition_reliability_artifacts_ready_cpu_preflight_next",
        "gpu_authorized": False,
        "local_single_gene_condition_rows": len(local_rows),
        "guide_metric_genes": len(guide_metrics),
        "de_metric_genes": len(de_metrics),
        "k562_metric_genes": len(k562_metrics),
        "artifact_summaries": artifact_summaries,
        "source_tables": {k: str(v) for k, v in source_paths.items()},
        "manifest": str(OUT_CONFIG),
        "decision": "run strict external-artifact preflight; no GPU is authorized from materialization alone",
        "boundary": manifest["boundary"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM GWT Condition Reliability Artifacts",
        "",
        f"Timestamp: `{timestamp}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Downloads public GWT supplementary CSV small tables only.",
        "- Maps gene-level reliability/context fields onto local single-gene conditions.",
        "- Does not download h5ad/expression data, train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "- These artifacts are external priors; strict CPU preflight must reject static/source-confounded signals.",
        "",
        "## Source Tables",
        "",
        "| table | local path | URL |",
        "|---|---|---|",
    ]
    for label, path in source_paths.items():
        lines.append(f"| `{label}` | `{path}` | {URLS[label]} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| artifact | rows | datasets | genes | output |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in artifact_summaries:
        lines.append(
            f"| `{row['artifact']}` | {row['rows']} | {row['datasets']} | {row['genes']} | `{row['output']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Source materialization is complete, but no GPU is authorized.",
            "- Next required step is strict CPU preflight using the emitted manifest.",
            "- A pass would still require artifact-specific shuffle/LODO/static-prior/source controls and external review before GPU.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
            f"- manifest: `{OUT_CONFIG}`",
            f"- artifact directory: `{OUT_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD), "manifest": str(OUT_CONFIG)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
