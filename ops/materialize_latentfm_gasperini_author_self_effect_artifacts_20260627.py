#!/usr/bin/env python3
"""Materialize Gasperini author self-effect/knockdown artifacts.

Reads GSE120861 processed DEG results and extracts selfTSS rows where the
perturbed gRNA group matches the measured target gene. CPU/source-only: no
training, inference, canonical multi selection, Track C query, or GPU.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SRC = ROOT / "reports/external_artifact_sources_20260627/adamson_gasperini_crispri_scout/gasperini_gse120861/GSE120861_all_deg_results.at_scale.txt.gz"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
OUT_DIR = ROOT / "reports/gasperini_author_self_effect_artifacts_20260627"
OUT_CSV = OUT_DIR / "gasperini_author_self_effect_artifacts.csv"
OUT_JSON = ROOT / "reports/latentfm_gasperini_author_self_effect_artifacts_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_GASPERINI_AUTHOR_SELF_EFFECT_ARTIFACTS_20260627.md"
MANIFEST = ROOT / "configs/latentfm_gasperini_author_self_effect_artifact_manifest_20260627.json"
DATASET = "GasperiniShendure2019_lowMOI"
SOURCE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120861/suppl/GSE120861_all_deg_results.at_scale.txt.gz"


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def neglog10(value: float | None) -> float | None:
    if value is None:
        return None
    if value <= 0:
        return 300.0
    return -math.log10(value)


def load_split_conditions() -> dict[str, str]:
    payload = json.loads(SPLIT.read_text(encoding="utf-8"))
    out = {}
    for split, items in payload[DATASET].items():
        if isinstance(items, list):
            for condition in items:
                out[str(condition)] = split
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    split_by_condition = load_split_conditions()
    rows = []
    seen_self = set()
    with gzip.open(SRC, "rt", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            grna = row.get("gRNA_group", "")
            base = grna[:-4] if grna.endswith("_TSS") else grna
            if base not in split_by_condition:
                continue
            if row.get("gene_short_name") != base or row.get("site_type") != "selfTSS":
                continue
            beta = fnum(row.get("beta"))
            fc = fnum(row.get("fold_change.transcript_remaining"))
            p_raw = fnum(row.get("pvalue.raw"))
            p_emp = fnum(row.get("pvalue.empirical"))
            p_adj = fnum(row.get("pvalue.empirical.adjusted"))
            artifacts = {
                "gasperini_self_beta": beta,
                "gasperini_self_abs_beta": abs(beta) if beta is not None else None,
                "gasperini_self_fold_change_remaining": fc,
                "gasperini_self_knockdown_strength": (1.0 - fc) if fc is not None else None,
                "gasperini_self_neglog10_p_raw": neglog10(p_raw),
                "gasperini_self_neglog10_p_empirical": neglog10(p_emp),
                "gasperini_self_neglog10_p_adjusted": neglog10(p_adj),
            }
            for artifact, value in artifacts.items():
                if value is None:
                    continue
                rows.append(
                    {
                        "dataset": DATASET,
                        "condition": base,
                        "split": split_by_condition[base],
                        "artifact": artifact,
                        "artifact_value": value,
                        "artifact_role": "response_candidate",
                        "raw_column": artifact.replace("gasperini_self_", ""),
                        "source_file": str(SRC),
                        "source_url": SOURCE_URL,
                        "gRNA_group": grna,
                        "pairs4merge": row.get("pairs4merge", ""),
                        "quality_rank_grna": row.get("quality_rank_grna", ""),
                        "site_type": row.get("site_type", ""),
                        "outlier_gene": row.get("outlier_gene", ""),
                    }
                )
            seen_self.add(base)

    fields = [
        "dataset",
        "condition",
        "split",
        "artifact",
        "artifact_value",
        "artifact_role",
        "raw_column",
        "source_file",
        "source_url",
        "gRNA_group",
        "pairs4merge",
        "quality_rank_grna",
        "site_type",
        "outlier_gene",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    artifacts = sorted({r["artifact"] for r in rows})
    payload = {
        "status": "gasperini_author_self_effect_artifacts_ready_no_gpu",
        "gpu_authorized": False,
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "local_conditions": len(split_by_condition),
        "matched_self_conditions": len(seen_self),
        "missing_self_conditions": sorted(set(split_by_condition) - seen_self),
        "rows": len(rows),
        "artifacts": artifacts,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON), "markdown": str(OUT_MD), "manifest": str(MANIFEST)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    MANIFEST.write_text(json.dumps({"status": payload["status"], "artifacts": artifacts, "csv": str(OUT_CSV), "source_url": SOURCE_URL}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Gasperini Author Self-Effect Artifacts 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/source-only materialization from GSE120861 author processed DEG table.",
        "- Extracts only `selfTSS` rows where `gRNA_group` base gene equals `gene_short_name` and local condition.",
        "- No training, inference, canonical multi selection, Track C query, or GPU.",
        "",
        "## Summary",
        "",
        f"- local conditions: `{payload['local_conditions']}`",
        f"- matched self conditions: `{payload['matched_self_conditions']}`",
        f"- artifact rows: `{payload['rows']}`",
        f"- artifacts: `{artifacts}`",
        f"- missing self conditions: `{payload['missing_self_conditions']}`",
        "",
        "## Outputs",
        "",
        f"- csv: `{OUT_CSV}`",
        f"- manifest: `{MANIFEST}`",
        f"- json: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "rows": len(rows), "report": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
