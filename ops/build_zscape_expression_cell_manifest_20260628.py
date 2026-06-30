#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PRIMARY_LINEAGES = {"mature fast muscle", "periderm"}
SECONDARY_MIXED = {"connective tissue-meninges-dermal FB"}
SECONDARY_RESPONSE = {"basal cell"}
DEMOTED_ROWS = {("retinal neuron", "tbx16-msgn1", "24.0")}


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def norm_time(value: str) -> str:
    return f"{float(value):.1f}"


def row_id(cell_type: str, target: str, timepoint: str) -> str:
    clean = "__".join([cell_type, target, norm_time(timepoint) + "h"])
    return (
        clean.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "p")
        .replace("(", "")
        .replace(")", "")
    )


def classify_row(cell_type: str, target: str, timepoint: str) -> dict[str, Any]:
    key = (cell_type, target, norm_time(timepoint))
    if key in DEMOTED_ROWS:
        return {
            "audit_role": "demoted_weak_control_pool",
            "trajectory_anchor": False,
            "mechanism_priority": 4,
            "reason": "weakest retinal row and control pool barely above threshold",
        }
    if cell_type in PRIMARY_LINEAGES:
        return {
            "audit_role": "primary_mechanism_test",
            "trajectory_anchor": True,
            "mechanism_priority": 1,
            "reason": "subagent-supported primary lineage for dynamic perturbation mechanism",
        }
    if cell_type in SECONDARY_MIXED:
        return {
            "audit_role": "secondary_mixed_lineage_stress_test",
            "trajectory_anchor": True,
            "mechanism_priority": 2,
            "reason": "heterogeneous broad lineage, useful stress test with subtype audit",
        }
    if cell_type in SECONDARY_RESPONSE:
        return {
            "audit_role": "secondary_response_control",
            "trajectory_anchor": False,
            "mechanism_priority": 3,
            "reason": "basal reference continuity failed; use for perturbation response only",
        }
    return {
        "audit_role": "secondary_lineage_test",
        "trajectory_anchor": True,
        "mechanism_priority": 2,
        "reason": "kept as secondary lineage pending expression controls",
    }


def is_control_target(target: str) -> bool:
    return target.startswith("ctrl")


def stable_key(seed: int, row: dict[str, str], rid: str, role: str) -> str:
    text = f"{seed}|{rid}|{role}|{row.get('cell', '')}|{row.get('embryo', '')}|{row.get('sample', '')}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def balanced_select(rows: list[dict[str, str]], cap: int, seed: int, rid: str, role: str) -> list[dict[str, str]]:
    if len(rows) <= cap:
        return sorted(rows, key=lambda r: stable_key(seed, r, rid, role))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("embryo", "")].append(row)
    for embryo_rows in groups.values():
        embryo_rows.sort(key=lambda r: stable_key(seed, r, rid, role))

    embryos = sorted(groups, key=lambda e: (len(groups[e]), e))
    selected: list[dict[str, str]] = []
    per_group = max(1, cap // max(1, len(embryos)))
    for embryo in embryos:
        selected.extend(groups[embryo][:per_group])
    if len(selected) < cap:
        used = {row["cell"] for row in selected}
        leftovers = [row for row in rows if row["cell"] not in used]
        leftovers.sort(key=lambda r: stable_key(seed, r, rid, role))
        selected.extend(leftovers[: cap - len(selected)])
    return selected[:cap]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["timepoint"] = norm_time(row["timepoint"])
        row["row_id"] = row_id(row["cell_type_broad"], row["gene_target"], row["timepoint"])
        row.update(classify_row(row["cell_type_broad"], row["gene_target"], row["timepoint"]))
    return rows


def make_lookup(rows: list[dict[str, str]]) -> tuple[dict[tuple[str, str, str], dict[str, str]], set[tuple[str, str]]]:
    perturb_keys = {}
    control_keys = set()
    for row in rows:
        cell_type = row["cell_type_broad"]
        timepoint = row["timepoint"]
        target = row["gene_target"]
        perturb_keys[(cell_type, timepoint, target)] = row
        control_keys.add((cell_type, timepoint))
    return perturb_keys, control_keys


def scan_metadata(
    metadata_path: Path,
    manifest_rows: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    perturb_keys, control_keys = make_lookup(manifest_rows)
    perturb: dict[str, list[dict[str, str]]] = defaultdict(list)
    controls: dict[str, list[dict[str, str]]] = defaultdict(list)
    row_by_control_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        row_by_control_key[(row["cell_type_broad"], row["timepoint"])].append(row)

    with gzip.open(metadata_path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for meta in reader:
            cell_type = meta["cell_type_broad"]
            timepoint = norm_time(meta["timepoint"])
            target = meta["gene_target"]
            pkey = (cell_type, timepoint, target)
            if pkey in perturb_keys:
                perturb[perturb_keys[pkey]["row_id"]].append(meta)
            ckey = (cell_type, timepoint)
            if ckey in control_keys and is_control_target(target):
                for manifest_row in row_by_control_key[ckey]:
                    controls[manifest_row["row_id"]].append(meta)
    return perturb, controls


def write_outputs(
    out_dir: Path,
    manifest_rows: list[dict[str, str]],
    perturb: dict[str, list[dict[str, str]]],
    controls: dict[str, list[dict[str, str]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / "zscape_expression_selected_cell_ids.csv"
    row_summary_path = out_dir / "zscape_expression_cell_manifest_row_summary.csv"
    json_path = out_dir / "zscape_expression_cell_manifest_20260628.json"
    md_path = out_dir / "LATENTFM_ZSCAPE_EXPRESSION_CELL_MANIFEST_20260628.md"

    selected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    primary_failures: list[str] = []

    for manifest_row in sorted(manifest_rows, key=lambda r: (int(r["mechanism_priority"]), r["cell_type_broad"], r["gene_target"], r["timepoint"])):
        rid = manifest_row["row_id"]
        p_all = perturb.get(rid, [])
        c_all = controls.get(rid, [])
        p_sel = balanced_select(p_all, args.max_perturb_cells, args.seed, rid, "perturb")
        control_cap = max(args.min_control_cells, min(args.max_control_cells, args.control_multiplier * max(1, len(p_sel))))
        c_sel = balanced_select(c_all, control_cap, args.seed, rid, "control")
        p_embryos = len({row.get("embryo", "") for row in p_all if row.get("embryo", "")})
        c_embryos = len({row.get("embryo", "") for row in c_all if row.get("embryo", "")})
        row_gate = (
            len(p_all) >= args.min_perturb_cells
            and p_embryos >= args.min_perturb_embryos
            and len(c_all) >= args.min_control_cells
            and c_embryos >= args.min_control_embryos
        )
        if manifest_row["audit_role"] == "primary_mechanism_test" and not row_gate:
            primary_failures.append(rid)

        summary = {
            "row_id": rid,
            "cell_type_broad": manifest_row["cell_type_broad"],
            "gene_target": manifest_row["gene_target"],
            "timepoint": manifest_row["timepoint"],
            "audit_role": manifest_row["audit_role"],
            "trajectory_anchor": manifest_row["trajectory_anchor"],
            "eligible_perturb_cells": len(p_all),
            "eligible_perturb_embryos": p_embryos,
            "selected_perturb_cells": len(p_sel),
            "eligible_control_cells": len(c_all),
            "eligible_control_embryos": c_embryos,
            "selected_control_cells": len(c_sel),
            "row_gate": row_gate,
            "audit_reason": manifest_row["reason"],
        }
        summary_rows.append(summary)

        for role, chosen in [("perturb", p_sel), ("control", c_sel)]:
            for rank, meta in enumerate(chosen):
                selected_rows.append(
                    {
                        "row_id": rid,
                        "selection_role": role,
                        "selection_rank": rank,
                        "audit_role": manifest_row["audit_role"],
                        "trajectory_anchor": manifest_row["trajectory_anchor"],
                        "manifest_cell_type_broad": manifest_row["cell_type_broad"],
                        "manifest_gene_target": manifest_row["gene_target"],
                        "manifest_timepoint": manifest_row["timepoint"],
                        "cell": meta.get("cell", ""),
                        "gene_target": meta.get("gene_target", ""),
                        "timepoint": norm_time(meta.get("timepoint", "nan")),
                        "cell_type_broad": meta.get("cell_type_broad", ""),
                        "cell_type_sub": meta.get("cell_type_sub", ""),
                        "tissue": meta.get("tissue", ""),
                        "germ_layer": meta.get("germ_layer", ""),
                        "embryo": meta.get("embryo", ""),
                        "sample": meta.get("sample", ""),
                        "expt": meta.get("expt", ""),
                        "n_umi": meta.get("n.umi", ""),
                        "num_genes_expressed": meta.get("num_genes_expressed", ""),
                        "umap3d_1": meta.get("umap3d_1", ""),
                        "umap3d_2": meta.get("umap3d_2", ""),
                        "umap3d_3": meta.get("umap3d_3", ""),
                    }
                )

    with row_summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with selected_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    n_rows_pass = sum(1 for row in summary_rows if row["row_gate"])
    n_primary = sum(1 for row in summary_rows if row["audit_role"] == "primary_mechanism_test")
    n_primary_pass = sum(1 for row in summary_rows if row["audit_role"] == "primary_mechanism_test" and row["row_gate"])
    status = (
        "zscape_expression_cell_manifest_ready_waiting_raw_counts_no_gpu"
        if not primary_failures and n_rows_pass >= args.min_total_rows_pass
        else "zscape_expression_cell_manifest_gate_fail_no_gpu"
    )
    payload = {
        "timestamp_utc": utc_now(),
        "status": status,
        "gpu_authorized": False,
        "expression_join_authorized_after_raw_counts": status.endswith("waiting_raw_counts_no_gpu"),
        "manifest_csv": str(args.manifest),
        "metadata_csv_gz": str(args.metadata),
        "selected_cell_ids_csv": str(selected_path),
        "row_summary_csv": str(row_summary_path),
        "filters": {
            "min_perturb_cells": args.min_perturb_cells,
            "min_perturb_embryos": args.min_perturb_embryos,
            "min_control_cells": args.min_control_cells,
            "min_control_embryos": args.min_control_embryos,
            "max_perturb_cells": args.max_perturb_cells,
            "max_control_cells": args.max_control_cells,
            "control_multiplier": args.control_multiplier,
            "seed": args.seed,
            "min_total_rows_pass": args.min_total_rows_pass,
        },
        "summary": {
            "manifest_rows": len(manifest_rows),
            "rows_passing_cell_gate": n_rows_pass,
            "primary_rows": n_primary,
            "primary_rows_passing_cell_gate": n_primary_pass,
            "selected_cells": len(selected_rows),
            "primary_failures": primary_failures,
        },
        "row_summary": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Expression Cell-ID Manifest",
        "",
        f"Timestamp: `{payload['timestamp_utc']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only metadata cell-ID manifest.",
        "- Does not read expression counts, CDS, reference matrices, model outputs, canonical multi, or Track C query.",
        "- Converts the external audit into primary/secondary/demoted rows and deterministic expression-subset cell IDs.",
        "",
        "## External-Audit Interpretation",
        "",
        "- Primary biological mechanism tests: `mature fast muscle` and `periderm`.",
        "- Secondary stress tests: `connective tissue-meninges-dermal FB` and most `retinal neuron` rows.",
        "- Secondary response controls: `basal cell` rows, because basal reference continuity was not a trajectory anchor.",
        "- Demoted weak row: `retinal neuron / tbx16-msgn1 / 24h`.",
        "",
        "## Gate Summary",
        "",
        f"- manifest rows: `{len(manifest_rows)}`",
        f"- rows passing metadata cell gate: `{n_rows_pass}`",
        f"- primary rows passing metadata cell gate: `{n_primary_pass}/{n_primary}`",
        f"- selected cells for expression join: `{len(selected_rows)}`",
        f"- primary failures: `{primary_failures}`",
        "",
        "## Row Summary",
        "",
        "| row_id | role | perturb cells/embryos | selected perturb | control cells/embryos | selected control | gate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["row_id"],
                    row["audit_role"],
                    f"{row['eligible_perturb_cells']}/{row['eligible_perturb_embryos']}",
                    str(row["selected_perturb_cells"]),
                    f"{row['eligible_control_cells']}/{row['eligible_control_embryos']}",
                    str(row["selected_control_cells"]),
                    str(row["row_gate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "Proceed to expression join after the raw-count download completes."
                if payload["expression_join_authorized_after_raw_counts"]
                else "Do not proceed to expression join without revising the manifest."
            ),
            "This still does not authorize GPU training.",
            "",
            "## Output Files",
            "",
            f"- selected cell IDs: `{selected_path}`",
            f"- row summary: `{row_summary_path}`",
            f"- JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-perturb-cells", type=int, default=100)
    parser.add_argument("--min-perturb-embryos", type=int, default=5)
    parser.add_argument("--min-control-cells", type=int, default=500)
    parser.add_argument("--min-control-embryos", type=int, default=30)
    parser.add_argument("--max-perturb-cells", type=int, default=512)
    parser.add_argument("--max-control-cells", type=int, default=1024)
    parser.add_argument("--control-multiplier", type=int, default=2)
    parser.add_argument("--min-total-rows-pass", type=int, default=20)
    args = parser.parse_args()

    manifest_rows = read_manifest(args.manifest)
    perturb, controls = scan_metadata(args.metadata, manifest_rows)
    payload = write_outputs(args.out_dir, manifest_rows, perturb, controls, args)
    print(args.out_dir / "LATENTFM_ZSCAPE_EXPRESSION_CELL_MANIFEST_20260628.md")
    print(args.out_dir / "zscape_expression_cell_manifest_20260628.json")
    print(payload["status"])
    return 0 if payload["status"].endswith("waiting_raw_counts_no_gpu") else 2


if __name__ == "__main__":
    raise SystemExit(main())
