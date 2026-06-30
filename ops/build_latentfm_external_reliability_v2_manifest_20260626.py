#!/usr/bin/env python3
"""Build an external reliability-v2 artifact manifest.

This manifest separates candidate reliability artifacts from the already closed
read/UMI support branch. CPU-only; no checkpoint/canonical multi/Track C query,
training, inference, or GPU use.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
OUT_CONFIG = ROOT / "configs/latentfm_external_reliability_v2_manifest_20260626.json"
OUT_JSON = ROOT / "reports/latentfm_external_reliability_v2_manifest_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_EXTERNAL_RELIABILITY_V2_MANIFEST_20260626.md"


ARTIFACT_GROUPS = {
    "external_reliability_assignment_fraction": [
        ROOT / "reports/norman_geo_reagent_artifacts_20260626/norman_geo_good_coverage_rate.csv",
        ROOT / "reports/frangieh_figshare_reagent_artifacts_20260626/frangieh_figshare_assigned_sgrna_rate.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_strict_assigned_row_fraction.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_lenient_assigned_row_fraction.csv",
        ROOT / "reports/dixit_figshare_reagent_artifacts_20260626/dixit_figshare_assigned_cell_fraction.csv",
    ],
    "external_reliability_source_cell_support": [
        ROOT / "reports/frangieh_figshare_reagent_artifacts_20260626/frangieh_figshare_condition_cell_count.csv",
        ROOT / "reports/dixit_figshare_reagent_artifacts_20260626/dixit_figshare_assigned_cell_count.csv",
    ],
    "external_reliability_guide_multiplicity_consistency": [
        ROOT / "reports/norman_geo_reagent_artifacts_20260626/norman_geo_mean_guide_coverage.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_strict_unique_guide_count.csv",
        ROOT / "reports/dixit_geo_reagent_artifacts_20260626/dixit_geo_highmoi_lenient_unique_guide_count.csv",
        ROOT / "reports/dixit_figshare_reagent_artifacts_20260626/dixit_figshare_unique_guide_count.csv",
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
                "description": "External reliability-v2 condition-level artifact distinct from raw read/UMI depth.",
                "priority": len(artifacts) + 1,
                "source_files": [rel(path) for path in existing],
                "required_columns": ["dataset", "condition", "artifact_value"],
                "optional_columns": ["source", "source_file", "source_column", "n_cells"],
                "minimum_datasets": 2,
                "minimum_overlap_rows": 20,
                "minimum_varying_datasets": 2,
                "promotion_note": "Must pass MMD-safe residual and source-block shuffle/LODO gates before any GPU proposal.",
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
        "version": "20260626_external_reliability_v2",
        "boundary": {
            "uses_train_only_internal_rows": True,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "uses_training": False,
            "uses_gpu": False,
            "source": "condition-level external metadata reliability artifacts excluding raw read/UMI depth",
        },
        "artifacts": artifacts,
    }
    OUT_CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = {
        "status": "external_reliability_v2_manifest_ready",
        "gpu_authorized": False,
        "config": str(OUT_CONFIG),
        "summaries": summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM External Reliability V2 Manifest",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        "Status: `external_reliability_v2_manifest_ready`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only manifest for source-block reliability artifacts.",
        "- Excludes raw read/UMI depth as a candidate training rule.",
        "- Does not train, infer, read checkpoints, read canonical multi, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        "| artifact | existing files | missing files |",
        "|---|---:|---:|",
    ]
    for row in summaries:
        lines.append(f"| `{row['artifact']}` | {len(row['existing'])} | {len(row['missing'])} |")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- config: `{OUT_CONFIG}`",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "config": str(OUT_CONFIG), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
