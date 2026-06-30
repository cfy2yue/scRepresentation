#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def row_ids(df: pd.DataFrame) -> list[str]:
    return sorted(df["row_id"].astype(str).tolist())


def subset_record(name: str, purpose: str, df: pd.DataFrame, criteria: dict[str, Any]) -> dict[str, Any]:
    ratios = pd.to_numeric(df["effect_ratio_vs_max_null_p95"], errors="coerce")
    jsd = pd.to_numeric(df["exploratory_subtype_jsd"], errors="coerce")
    return {
        "name": name,
        "purpose": purpose,
        "criteria": criteria,
        "row_ids": row_ids(df),
        "n_rows": int(len(df)),
        "lineages": sorted(df["lineage"].astype(str).unique().tolist()),
        "targets": sorted(df["target"].astype(str).unique().tolist()),
        "timepoints": sorted(pd.to_numeric(df["timepoint"], errors="coerce").dropna().unique().tolist()),
        "mean_exploratory_ratio": float(ratios.mean()) if ratios.notna().any() else None,
        "max_exploratory_ratio": float(ratios.max()) if ratios.notna().any() else None,
        "mean_exploratory_subtype_jsd": float(jsd.mean()) if jsd.notna().any() else None,
        "max_exploratory_subtype_jsd": float(jsd.max()) if jsd.notna().any() else None,
        "formal_use": "candidate subset for later strict/fixed-cell robustness only; not biological proof",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.row_metrics)
    ratio = pd.to_numeric(df["effect_ratio_vs_max_null_p95"], errors="coerce")
    jsd = pd.to_numeric(df["exploratory_subtype_jsd"], errors="coerce")
    primary = df[df["audit_role"] == "primary_mechanism_test"].copy()
    primary_ratio = pd.to_numeric(primary["effect_ratio_vs_max_null_p95"], errors="coerce")
    primary_jsd = pd.to_numeric(primary["exploratory_subtype_jsd"], errors="coerce")

    specs: list[dict[str, Any]] = []
    specs.append(
        subset_record(
            "primary_high_effect_muscle",
            "High-effect mesoderm/muscle response axis; expected to be strong but subtype-confounded.",
            primary[(primary["lineage"] == "mature fast muscle") & (primary_ratio >= 1.50)],
            {"audit_role": "primary_mechanism_test", "lineage": "mature fast muscle", "ratio_min": 1.50},
        )
    )
    specs.append(
        subset_record(
            "primary_clean_periderm",
            "Cleaner epithelial/developmental response axis; expected to be lower effect but subtype-stable.",
            primary[(primary["lineage"] == "periderm") & (primary_jsd <= 0.10)],
            {"audit_role": "primary_mechanism_test", "lineage": "periderm", "subtype_jsd_max": 0.10},
        )
    )
    specs.append(
        subset_record(
            "primary_clean_mixed",
            "Primary rows with exploratory subtype JSD at or below 0.10, regardless of lineage.",
            primary[primary_jsd <= 0.10],
            {"audit_role": "primary_mechanism_test", "subtype_jsd_max": 0.10},
        )
    )
    specs.append(
        subset_record(
            "secondary_response_control_basal",
            "Basal response-control subset; should not be treated as trajectory-anchor evidence.",
            df[df["audit_role"] == "secondary_response_control"],
            {"audit_role": "secondary_response_control"},
        )
    )
    specs.append(
        subset_record(
            "secondary_retinal_or_demoted_low_signal",
            "Low-signal/selectivity-control retinal rows, including demoted weak-control row.",
            df[df["lineage"].eq("retinal neuron")],
            {"lineage": "retinal neuron"},
        )
    )
    specs.append(
        subset_record(
            "secondary_mixed_lineage_fb",
            "Mixed-lineage stress-test subset; useful as heterogeneity/control evidence.",
            df[df["audit_role"] == "secondary_mixed_lineage_stress_test"],
            {"audit_role": "secondary_mixed_lineage_stress_test"},
        )
    )

    spec = {
        "timestamp_utc": utc_now(),
        "status": "zscape_bioinformation_subset_specs_frozen_no_gpu",
        "gpu_authorized": False,
        "row_metrics": str(args.row_metrics),
        "selection_boundary": "exploratory OT row metrics only; no strict gate result used; no model metrics used",
        "minimum_formal_use": "strict-controls pass or interpretable lineage-specific partial support required before formal robustness run",
        "subsets": specs,
    }
    json_path = args.out_dir / "zscape_bioinformation_subset_specs_20260628.json"
    json_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for item in specs:
        for rid in item["row_ids"]:
            rows.append(
                {
                    "subset": item["name"],
                    "row_id": rid,
                    "purpose": item["purpose"],
                    "n_subset_rows": item["n_rows"],
                }
            )
    csv_path = args.out_dir / "zscape_bioinformation_subset_spec_rows.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    md_path = args.out_dir / "LATENTFM_ZSCAPE_BIOINFORMATION_SUBSET_SPECS_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Bioinformation Subset Specs",
        "",
        f"Timestamp: `{utc_now()}`",
        "",
        "Status: `zscape_bioinformation_subset_specs_frozen_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Freezes candidate subset definitions before strict-controls results are",
        "  interpreted.",
        "- Uses only exploratory bioinformation row metrics and biological roles.",
        "- Does not train, infer, run scFM embeddings, read canonical multi, read Track C query, or authorize GPU.",
        "",
        "## Frozen Subsets",
        "",
        "| subset | rows | lineages | targets | mean ratio | max JSD | purpose |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for item in specs:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["name"],
                    str(item["n_rows"]),
                    ", ".join(item["lineages"]),
                    ", ".join(item["targets"]),
                    f"{item['mean_exploratory_ratio']:.3f}" if item["mean_exploratory_ratio"] is not None else "NA",
                    f"{item['max_exploratory_subtype_jsd']:.3f}" if item["max_exploratory_subtype_jsd"] is not None else "NA",
                    item["purpose"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Formal-Use Rule",
            "",
            "Run fixed-cell high-I/low-I robustness only after the strict-controls gate",
            "passes or gives interpretable lineage-specific partial support. A subset",
            "spec alone is not biological evidence and does not authorize GPU.",
            "",
            "## Output Files",
            "",
            f"- JSON: `{json_path}`",
            f"- rows: `{csv_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
