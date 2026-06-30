#!/usr/bin/env python3
"""Build a manifest for combined Norman/Frangieh/Dixit reagent read-support artifacts.

Short CPU task. It only checks which extracted artifact CSV files are present and
writes a manifest for the existing external-artifact preflight. It does not read
expression matrices, checkpoints, canonical multi, Track C query, train, infer,
or use GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
OUT_CONFIG = ROOT / "configs/latentfm_reagent_read_support_combined_manifest_20260626.json"
OUT_JSON = ROOT / "reports/latentfm_reagent_read_support_combined_manifest_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_REAGENT_READ_SUPPORT_COMBINED_MANIFEST_20260626.md"

ARTIFACT_GROUPS = {
    "external_reagent_mean_umi_count": [
        ROOT / "reports/norman_geo_reagent_artifacts_20260626/norman_geo_mean_umi_count.csv",
        ROOT / "reports/frangieh_figshare_reagent_artifacts_20260626/frangieh_figshare_mean_umi_count.csv",
    ],
    "external_reagent_read_or_guide_support": [
        ROOT / "reports/norman_geo_reagent_artifacts_20260626/norman_geo_mean_read_count.csv",
        ROOT / "reports/frangieh_figshare_reagent_artifacts_20260626/frangieh_figshare_mean_moi.csv",
        ROOT / "reports/frangieh_figshare_reagent_artifacts_20260626/frangieh_figshare_assigned_sgrna_rate.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_strict_unique_guide_count.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_strict_assigned_row_count.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_lenient_unique_guide_count.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_lenient_assigned_row_count.csv",
        ROOT / "reports/dixit_figshare_reagent_artifacts_20260626/dixit_figshare_unique_guide_count.csv",
        ROOT / "reports/dixit_figshare_reagent_artifacts_20260626/dixit_figshare_assigned_cell_count.csv",
    ],
    "external_condition_source_cell_support": [
        ROOT / "reports/frangieh_figshare_reagent_artifacts_20260626/frangieh_figshare_condition_cell_count.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_strict_assigned_row_fraction.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_lenient_assigned_row_fraction.csv",
        ROOT / "reports/dixit_figshare_reagent_artifacts_20260626/dixit_figshare_assigned_cell_fraction.csv",
    ],
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    artifacts = []
    summaries = []
    for artifact, paths in ARTIFACT_GROUPS.items():
        existing = [path for path in paths if path.is_file()]
        missing = [path for path in paths if not path.is_file()]
        artifacts.append(
            {
                "artifact": artifact,
                "description": "Combined condition-level reagent/read-support source artifact from extracted external metadata.",
                "priority": len(artifacts) + 1,
                "source_files": [rel(path) for path in existing],
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["source", "source_file", "source_column", "n_cells"],
                "minimum_datasets": 2,
                "minimum_overlap_rows": 20,
                "minimum_varying_datasets": 2,
                "promotion_note": (
                    "If this passes, run value-signal, shuffle, source/count, tail and MMD controls plus external audit before any GPU."
                ),
            }
        )
        summaries.append(
            {
                "artifact": artifact,
                "existing": [str(path) for path in existing],
                "missing": [str(path) for path in missing],
            }
        )
    config = {
        "version": "20260626_reagent_read_support_combined",
        "boundary": {
            "uses_train_only_internal_rows": True,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "uses_training": False,
            "uses_gpu": False,
            "source": "combined extracted condition-level external reagent/read-support metadata",
        },
        "artifacts": artifacts,
    }
    OUT_CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "status": "reagent_read_support_combined_manifest_ready",
        "gpu_authorized": False,
        "config": str(OUT_CONFIG),
        "summaries": summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM Reagent Read-Support Combined Manifest",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        "Status: `reagent_read_support_combined_manifest_ready`",
        "",
        "GPU authorized: `False`",
        "",
        "## Summary",
        "",
        "| artifact | existing files | missing files |",
        "|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| `{row['artifact']}` | {len(row['existing'])} | {len(row['missing'])} |")
    lines += [
        "",
        "## Decision",
        "",
        "- Manifest generation does not authorize GPU.",
        "- Run the external-artifact preflight only after at least Norman plus Frangieh artifact files exist.",
        "",
        "## Outputs",
        "",
        f"- config: `{OUT_CONFIG}`",
        f"- JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "config": str(OUT_CONFIG), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
