#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ZPERTURB_CELL = "GSE202639_zperturb_full_cell_metadata.csv.gz"
REFERENCE_CELL = "GSE202639_reference_cell_metadata.csv.gz"


def now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def is_control(target: str) -> bool:
    return target.lower().startswith("ctrl")


def safe(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def scan_cell_metadata(path: Path, dataset_name: str) -> dict:
    totals = Counter()
    by_target = Counter()
    by_time = Counter()
    by_broad = Counter()
    by_sub = Counter()
    by_tissue = Counter()
    by_germ = Counter()
    by_condition_class = Counter()
    combo = Counter()
    broad_stats = defaultdict(lambda: {
        "cells": 0,
        "control_cells": 0,
        "perturb_cells": 0,
        "timepoints": set(),
        "control_timepoints": set(),
        "perturb_timepoints": set(),
        "targets": set(),
        "perturb_targets": set(),
        "embryos": set(),
        "samples": set(),
        "tissues": set(),
        "subtypes": set(),
    })
    target_stats = defaultdict(lambda: {
        "cells": 0,
        "timepoints": set(),
        "broad_cell_types": set(),
        "embryos": set(),
        "samples": set(),
    })
    target_time_broad = defaultdict(lambda: {
        "cells": 0,
        "embryos": set(),
        "samples": set(),
        "subtypes": set(),
        "tissues": set(),
    })

    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            target = safe(row.get("gene_target"))
            timepoint = safe(row.get("timepoint"))
            broad = safe(row.get("cell_type_broad"))
            subtype = safe(row.get("cell_type_sub"))
            tissue = safe(row.get("tissue"))
            germ = safe(row.get("germ_layer"))
            embryo = safe(row.get("embryo"))
            sample = safe(row.get("sample"))
            klass = "control" if is_control(target) else "perturbation"

            totals["cells"] += 1
            by_target[target] += 1
            by_time[timepoint] += 1
            by_broad[broad] += 1
            by_sub[subtype] += 1
            by_tissue[tissue] += 1
            by_germ[germ] += 1
            by_condition_class[klass] += 1
            combo[(target, timepoint, broad)] += 1

            bs = broad_stats[broad]
            bs["cells"] += 1
            bs["timepoints"].add(timepoint)
            bs["targets"].add(target)
            bs["embryos"].add(embryo)
            bs["samples"].add(sample)
            bs["tissues"].add(tissue)
            bs["subtypes"].add(subtype)
            if klass == "control":
                bs["control_cells"] += 1
                bs["control_timepoints"].add(timepoint)
            else:
                bs["perturb_cells"] += 1
                bs["perturb_timepoints"].add(timepoint)
                bs["perturb_targets"].add(target)

            ts = target_stats[target]
            ts["cells"] += 1
            ts["timepoints"].add(timepoint)
            ts["broad_cell_types"].add(broad)
            ts["embryos"].add(embryo)
            ts["samples"].add(sample)

            ttb = target_time_broad[(target, timepoint, broad)]
            ttb["cells"] += 1
            ttb["embryos"].add(embryo)
            ttb["samples"].add(sample)
            ttb["subtypes"].add(subtype)
            ttb["tissues"].add(tissue)

    def stats_to_json(dct: dict) -> dict:
        out = {}
        for key, value in dct.items():
            if isinstance(value, set):
                out[f"n_{key}"] = len([v for v in value if v != ""])
            else:
                out[key] = value
        return out

    candidate_broad = []
    for broad, stats in broad_stats.items():
        row = stats_to_json(stats)
        row["cell_type_broad"] = broad
        if dataset_name == "reference":
            pass_gate = (
                row["cells"] >= 5000
                and row["n_timepoints"] >= 3
                and row["n_embryos"] >= 5
                and row["n_subtypes"] >= 1
            )
        else:
            pass_gate = (
                row["cells"] >= 5000
                and row["control_cells"] >= 500
                and row["perturb_cells"] >= 500
                and row["n_control_timepoints"] >= 2
                and row["n_perturb_timepoints"] >= 2
                and row["n_perturb_targets"] >= 5
                and row["n_embryos"] >= 5
            )
        row["coverage_gate_candidate"] = pass_gate
        candidate_broad.append(row)
    candidate_broad.sort(key=lambda r: (r["coverage_gate_candidate"], r["perturb_cells"], r["cells"]), reverse=True)

    target_rows = []
    for target, stats in target_stats.items():
        row = stats_to_json(stats)
        row["gene_target"] = target
        row["condition_class"] = "control" if is_control(target) else "perturbation"
        target_rows.append(row)
    target_rows.sort(key=lambda r: r["cells"], reverse=True)

    combo_rows = []
    for (target, timepoint, broad), stats in target_time_broad.items():
        row = stats_to_json(stats)
        row["gene_target"] = target
        row["timepoint"] = timepoint
        row["cell_type_broad"] = broad
        row["condition_class"] = "control" if is_control(target) else "perturbation"
        combo_rows.append(row)
    combo_rows.sort(key=lambda r: r["cells"], reverse=True)

    return {
        "dataset_name": dataset_name,
        "path": str(path),
        "fieldnames": fieldnames,
        "totals": dict(totals),
        "condition_class_counts": dict(by_condition_class),
        "n_targets": len([k for k in by_target if k]),
        "n_perturbation_targets": len([k for k in by_target if k and not is_control(k)]),
        "n_control_targets": len([k for k in by_target if k and is_control(k)]),
        "n_timepoints": len([k for k in by_time if k]),
        "n_broad_cell_types": len([k for k in by_broad if k]),
        "n_sub_cell_types": len([k for k in by_sub if k]),
        "n_tissues": len([k for k in by_tissue if k]),
        "n_germ_layers": len([k for k in by_germ if k]),
        "top_targets": by_target.most_common(30),
        "top_timepoints": by_time.most_common(30),
        "top_broad_cell_types": by_broad.most_common(40),
        "top_tissues": by_tissue.most_common(30),
        "top_germ_layers": by_germ.most_common(30),
        "candidate_broad_cell_types": candidate_broad,
        "target_rows": target_rows,
        "combo_rows": combo_rows,
    }


def write_csv(path: Path, rows: list[dict], preferred: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(preferred)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compact_for_json(scan: dict) -> dict:
    out = dict(scan)
    out["candidate_broad_cell_types"] = scan["candidate_broad_cell_types"][:100]
    out["target_rows"] = scan["target_rows"][:200]
    out["combo_rows"] = scan["combo_rows"][:500]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zperturb_path = data_dir / ZPERTURB_CELL
    reference_path = data_dir / REFERENCE_CELL
    missing = [str(p) for p in [zperturb_path, reference_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing metadata inputs: {missing}")

    zperturb = scan_cell_metadata(zperturb_path, "zperturb_full")
    reference = scan_cell_metadata(reference_path, "reference")

    z_candidate = [r for r in zperturb["candidate_broad_cell_types"] if r["coverage_gate_candidate"]]
    ref_candidate = [r for r in reference["candidate_broad_cell_types"] if r["coverage_gate_candidate"]]
    shared_candidate = sorted(
        {
            row["cell_type_broad"]
            for row in z_candidate
        }
        & {
            row["cell_type_broad"]
            for row in ref_candidate
        }
    )

    status = "zscape_metadata_coverage_gate_pass" if len(z_candidate) >= 2 and len(shared_candidate) >= 2 else "zscape_metadata_coverage_gate_fail"
    if status.endswith("pass") and zperturb["n_perturbation_targets"] < 10:
        status = "zscape_metadata_coverage_gate_fail"

    z_broad_csv = out_dir / "zperturb_broad_cell_type_coverage.csv"
    z_target_csv = out_dir / "zperturb_target_coverage.csv"
    z_combo_csv = out_dir / "zperturb_target_time_broad_coverage.csv"
    ref_broad_csv = out_dir / "reference_broad_cell_type_coverage.csv"
    write_csv(
        z_broad_csv,
        zperturb["candidate_broad_cell_types"],
        [
            "cell_type_broad",
            "coverage_gate_candidate",
            "cells",
            "control_cells",
            "perturb_cells",
            "n_timepoints",
            "n_control_timepoints",
            "n_perturb_timepoints",
            "n_targets",
            "n_perturb_targets",
            "n_embryos",
            "n_samples",
            "n_tissues",
            "n_subtypes",
        ],
    )
    write_csv(
        z_target_csv,
        zperturb["target_rows"],
        ["gene_target", "condition_class", "cells", "n_timepoints", "n_broad_cell_types", "n_embryos", "n_samples"],
    )
    write_csv(
        z_combo_csv,
        zperturb["combo_rows"],
        [
            "gene_target",
            "condition_class",
            "timepoint",
            "cell_type_broad",
            "cells",
            "n_embryos",
            "n_samples",
            "n_subtypes",
            "n_tissues",
        ],
    )
    write_csv(
        ref_broad_csv,
        reference["candidate_broad_cell_types"],
        [
            "cell_type_broad",
            "coverage_gate_candidate",
            "cells",
            "control_cells",
            "perturb_cells",
            "n_timepoints",
            "n_control_timepoints",
            "n_perturb_timepoints",
            "n_targets",
            "n_perturb_targets",
            "n_embryos",
            "n_samples",
            "n_tissues",
            "n_subtypes",
        ],
    )

    payload = {
        "timestamp_utc": now_utc(),
        "run_name": args.run_name,
        "status": status,
        "gpu_authorized": False,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "zperturb_summary": compact_for_json(zperturb),
        "reference_summary": compact_for_json(reference),
        "n_zperturb_candidate_broad_cell_types": len(z_candidate),
        "n_reference_candidate_broad_cell_types": len(ref_candidate),
        "shared_candidate_broad_cell_types": shared_candidate,
        "outputs": {
            "zperturb_broad_cell_type_coverage": str(z_broad_csv),
            "zperturb_target_coverage": str(z_target_csv),
            "zperturb_target_time_broad_coverage": str(z_combo_csv),
            "reference_broad_cell_type_coverage": str(ref_broad_csv),
        },
    }

    json_path = out_dir / "zscape_metadata_coverage_audit.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = out_dir / "LATENTFM_ZSCAPE_METADATA_COVERAGE_AUDIT_20260628.md"
    lines = [
        "# LatentFM ZSCAPE Metadata Coverage Audit",
        "",
        f"Timestamp: `{payload['timestamp_utc']}`",
        "",
        f"Run: `{args.run_name}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only metadata coverage audit over downloaded GEO metadata CSV files.",
        "- No expression matrix/CDS/raw-count loading.",
        "- No training, inference, embedding, canonical multi, or Track C query use.",
        "",
        "## ZPERTURB Summary",
        "",
        f"- cells: `{zperturb['totals'].get('cells', 0)}`",
        f"- targets: `{zperturb['n_targets']}`",
        f"- perturbation targets: `{zperturb['n_perturbation_targets']}`",
        f"- control targets: `{zperturb['n_control_targets']}`",
        f"- timepoints: `{zperturb['n_timepoints']}`",
        f"- broad cell types: `{zperturb['n_broad_cell_types']}`",
        f"- sub cell types: `{zperturb['n_sub_cell_types']}`",
        f"- candidate broad cell types passing coverage gate: `{len(z_candidate)}`",
        "",
        "## Reference Summary",
        "",
        f"- cells: `{reference['totals'].get('cells', 0)}`",
        f"- timepoints: `{reference['n_timepoints']}`",
        f"- broad cell types: `{reference['n_broad_cell_types']}`",
        f"- sub cell types: `{reference['n_sub_cell_types']}`",
        f"- candidate broad cell types passing coverage gate: `{len(ref_candidate)}`",
        "",
        "## Shared Candidate Broad Cell Types",
        "",
        f"`{shared_candidate[:50]}`",
        "",
        "## Top ZPERTURB Candidate Cell Types",
        "",
        "| cell_type_broad | candidate | cells | control_cells | perturb_cells | perturb_targets | timepoints | embryos | subtypes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in zperturb["candidate_broad_cell_types"][:25]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["cell_type_broad"],
                    str(row["coverage_gate_candidate"]),
                    str(row["cells"]),
                    str(row["control_cells"]),
                    str(row["perturb_cells"]),
                    str(row["n_perturb_targets"]),
                    str(row["n_timepoints"]),
                    str(row["n_embryos"]),
                    str(row["n_subtypes"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if status == "zscape_metadata_coverage_gate_pass":
        lines.extend(
            [
                "Proceed to a CPU-only ZSCAPE continuity/OT planning gate using a small,",
                "predeclared subset of shared candidate lineages. This metadata result still",
                "does not authorize GPU training or claims about LatentFM improvement.",
            ]
        )
    else:
        lines.extend(
            [
                "Do not proceed to ZSCAPE trajectory or model work from this metadata audit.",
                "Find a better annotated atlas or relax the coverage criteria only with a",
                "written biological rationale.",
            ]
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- JSON: `{json_path}`",
            f"- zperturb broad coverage: `{z_broad_csv}`",
            f"- zperturb target coverage: `{z_target_csv}`",
            f"- zperturb target-time-broad coverage: `{z_combo_csv}`",
            f"- reference broad coverage: `{ref_broad_csv}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(report_path)
    print(json_path)
    print(status)
    return 0 if status == "zscape_metadata_coverage_gate_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
