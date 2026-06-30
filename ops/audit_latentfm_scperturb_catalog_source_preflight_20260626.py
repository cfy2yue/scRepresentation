#!/usr/bin/env python3
"""Audit scPerturb catalogue as an external artifact source index.

Short CPU task. Reads the downloaded scPerturb data_table.csv and local
condition inventory. It does not download large h5ad files, train, infer, read
canonical multi, read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CATALOG = ROOT / "reports/external_artifact_sources_20260626/scperturb_data_table_20260626.csv"
CONDITION_INV = ROOT / "reports/latentfm_condition_level_inventory_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_scperturb_catalog_source_preflight_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_SCPERTURB_CATALOG_SOURCE_PREFLIGHT_20260626.md"
OUT_CSV = ROOT / "reports/latentfm_scperturb_catalog_source_preflight_rows_20260626.csv"

TOKEN_MAP = {
    "Adamson": ["adamson"],
    "DixitRegev2016_K562_TFs_High_MOI": ["dixit"],
    "Frangieh": ["frangieh"],
    "GasperiniShendure2019_lowMOI": ["gasperini"],
    "NormanWeissman2019_filtered": ["norman"],
    "Papalexi": ["papalexi"],
    "ReplogleWeissman2022_K562_gwps": ["replogle"],
    "Replogle_RPE1essential": ["replogle"],
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def range_has_variation(text: str) -> bool:
    t = norm(text).lower()
    if not t or t in {"-", "nan"}:
        return False
    if "-" in t:
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
        return len(nums) >= 2 and max(nums) > min(nums)
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    return len(set(nums)) >= 2


def load_catalog() -> list[dict[str, str]]:
    with CATALOG.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def local_datasets() -> list[str]:
    payload = read_json(CONDITION_INV)
    return sorted({row["dataset"] for row in payload.get("rows", [])})


def row_text(row: dict[str, str]) -> str:
    return " ".join(norm(row.get(k)) for k in ("Shorthand", "Title", "Treatment", "Cell source", "Data location"))


def main() -> int:
    catalog = load_catalog()
    local = local_datasets()
    rows = []
    for ds in local:
        tokens = TOKEN_MAP.get(ds, [])
        if not tokens:
            continue
        for row in catalog:
            text = row_text(row).lower()
            if not any(tok in text for tok in tokens):
                continue
            timepoints = norm(row.get("# timepoints"))
            doses = norm(row.get("# doses"))
            h5ad = norm(row.get(".h5ad availability"))
            data_location = norm(row.get("Data location"))
            treatment = norm(row.get("Treatment"))
            technique = norm(row.get("Technique"))
            matched = {
                "local_dataset": ds,
                "scperturb_shorthand": norm(row.get("Shorthand")),
                "title": norm(row.get("Title")),
                "treatment": treatment,
                "technique": technique,
                "data_location": data_location,
                "h5ad_available": bool(h5ad),
                "h5ad_field": h5ad,
                "doses": doses,
                "timepoints": timepoints,
                "time_or_dose_variation": range_has_variation(timepoints) or range_has_variation(doses),
                "potential_reagent_route": "CRISPR" in treatment.upper() or "sgRNA" in norm(row.get("# perturbations")),
                "potential_maturity_route": range_has_variation(timepoints),
                "potential_viability_route": False,
                "notes": "",
            }
            notes = []
            if matched["potential_maturity_route"]:
                notes.append("timepoint variation in catalogue")
            if range_has_variation(doses):
                notes.append("dose variation in catalogue; chemical/dose may be ACK-gated or protocol-confounded")
            if matched["potential_reagent_route"]:
                notes.append("CRISPR/sgRNA route may require original guide-level supplement")
            if not h5ad and not data_location:
                notes.append("no direct h5ad/data location in catalogue")
            matched["notes"] = "; ".join(notes)
            rows.append(matched)

    # Deduplicate exact local/scperturb shorthand rows.
    unique = {}
    for row in rows:
        unique[(row["local_dataset"], row["scperturb_shorthand"])] = row
    rows = list(unique.values())
    rows.sort(key=lambda r: (r["local_dataset"], r["scperturb_shorthand"]))

    local_with_match = sorted({r["local_dataset"] for r in rows})
    maturity_candidates = [r for r in rows if r["potential_maturity_route"]]
    reagent_candidates = [r for r in rows if r["potential_reagent_route"]]
    h5ad_candidates = [r for r in rows if r["h5ad_available"]]
    status = "scperturb_catalog_source_preflight_partial_no_gpu"
    action = (
        "catalogue identifies source leads, but no condition-level external artifact file exists yet; "
        "use catalogue to target original supplements/GEO/figshare metadata"
    )

    payload = {
        "status": status,
        "gpu_authorized": False,
        "catalog": str(CATALOG),
        "local_dataset_count": len(local),
        "matched_local_dataset_count": len(local_with_match),
        "matched_local_datasets": local_with_match,
        "row_count": len(rows),
        "maturity_candidate_rows": len(maturity_candidates),
        "reagent_candidate_rows": len(reagent_candidates),
        "h5ad_available_rows": len(h5ad_candidates),
        "action": action,
        "rows": rows,
        "sources": {
            "scperturb_repo": "https://github.com/theislab/sc-pert",
            "scperturb_catalog": "https://raw.githubusercontent.com/theislab/sc-pert/main/data_table.csv",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = [
        "local_dataset",
        "scperturb_shorthand",
        "title",
        "treatment",
        "technique",
        "data_location",
        "h5ad_available",
        "doses",
        "timepoints",
        "potential_reagent_route",
        "potential_maturity_route",
        "notes",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# LatentFM scPerturb Catalog Source Preflight",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Uses the small scPerturb catalogue CSV plus local condition inventory.",
        "- Does not download large h5ad files, train, infer, read canonical multi, read Track C query, or use GPU.",
        "- This is a source-location preflight, not a condition-level artifact gate.",
        "",
        "## Summary",
        "",
        f"- local datasets with catalogue matches: `{len(local_with_match)}` / `{len(local)}`",
        f"- matched catalogue rows: `{len(rows)}`",
        f"- potential reagent-route rows: `{len(reagent_candidates)}`",
        f"- potential maturity/time-route rows: `{len(maturity_candidates)}`",
        f"- rows with scPerturb h5ad links: `{len(h5ad_candidates)}`",
        "",
        "## Matched Rows",
        "",
        "| local dataset | scPerturb row | treatment | doses | timepoints | h5ad | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['local_dataset']}` | {row['scperturb_shorthand']} | "
            f"{row['treatment']} | `{row['doses']}` | `{row['timepoints']}` | "
            f"`{row['h5ad_available']}` | {row['notes']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- action: {action}",
        "- No GPU is authorized.",
        "- The next concrete acquisition target is original guide-level supplements for matched CRISPR datasets, especially rows with figshare/GEO pointers.",
        "- Maturity/time appears sparse for local gene-perturbation matches; any timing route must prove within-dataset condition-level variation.",
        "",
        "## Sources",
        "",
        "- scPerturb repo: https://github.com/theislab/sc-pert",
        "- scPerturb catalogue CSV: https://raw.githubusercontent.com/theislab/sc-pert/main/data_table.csv",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- rows: `{OUT_CSV}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
