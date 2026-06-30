#!/usr/bin/env python3
"""Audit downloaded scPerturb notebooks for metadata fields useful to scaling.

Short CPU task. This only parses local notebook text/JSON and the small
scPerturb catalogue. It does not download h5ad files, read expression matrices,
read checkpoints, use canonical multi or Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SRC_DIR = ROOT / "reports/external_artifact_sources_20260626/scperturb_notebooks"
CATALOG = ROOT / "reports/external_artifact_sources_20260626/scperturb_data_table_20260626.csv"
CATALOG_PREFLIGHT = ROOT / "reports/latentfm_scperturb_catalog_source_preflight_20260626.json"
OUT_JSON = ROOT / "reports/latentfm_scperturb_notebook_metadata_preflight_20260626.json"
OUT_MD = ROOT / "reports/LATENTFM_SCPERTURB_NOTEBOOK_METADATA_PREFLIGHT_20260626.md"
OUT_CSV = ROOT / "reports/latentfm_scperturb_notebook_metadata_preflight_rows_20260626.csv"

KEYWORDS = {
    "guide_reagent": [
        "guide",
        "sgrna",
        "grna",
        "protospacer",
        "guide_id",
        "guide assignment",
        "target gene",
        "gene_target",
    ],
    "perturbation_type": [
        "perturbation",
        "perturbation_type",
        "crispr",
        "crispri",
        "crisprko",
        "crop-seq",
        "perturb-seq",
    ],
    "background_cell": [
        "cell type",
        "cell_type",
        "cell line",
        "cell_line",
        "k562",
        "jurkat",
        "bmdc",
        "melanoma",
        "a375",
    ],
    "time_maturity": [
        "timepoint",
        "time point",
        "time_point",
        "hour",
        "hours",
        "day",
        "days",
        "7d",
        "10d",
    ],
    "dose": [
        "dose",
        "dosage",
        "concentration",
        "um",
        "micromolar",
    ],
    "viability_fitness": [
        "viability",
        "fitness",
        "growth",
        "proliferation",
        "essential",
        "depleted",
        "cell cycle",
        "quality control",
    ],
    "batch_qc": [
        "batch",
        "replicate",
        "library",
        "donor",
        "umi",
        "n_counts",
        "pct_counts",
        "mito",
        "doublet",
    ],
}

NOTEBOOK_TO_LOCAL_DATASETS = {
    "Dixit_2016": ["DixitRegev2016_K562_TFs_High_MOI"],
    "Frangieh_2021": ["Frangieh"],
    "Norman_2019": ["NormanWeissman2019_filtered"],
    "Norman_2019_curation": ["NormanWeissman2019_filtered"],
    "Srivatsan_2019_sciplex3": [],
}

FIELD_PATTERNS = [
    re.compile(r"\.obs\[['\"]([^'\"]+)['\"]\]"),
    re.compile(r"\.obs\.([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"obs\[['\"]([^'\"]+)['\"]\]"),
    re.compile(r"obs\.([A-Za-z_][A-Za-z0-9_]*)"),
    re.compile(r"['\"]([A-Za-z][A-Za-z0-9_ ./+-]{2,40})['\"]"),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_notebook_sources(path: Path) -> tuple[str, bool, str]:
    text = read_text(path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return text, False, f"json_decode_error:{exc.msg}"
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.extend(str(x) for x in source)
        else:
            chunks.append(str(source))
    return "\n".join(chunks), True, ""


def catalogue_rows() -> list[dict[str, str]]:
    if not CATALOG.exists():
        return []
    with CATALOG.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def matched_catalogue_for(name: str, catalog: list[dict[str, str]]) -> list[dict[str, str]]:
    token = name.split("_")[0].lower()
    if name.startswith("Srivatsan"):
        token = "srivatsan"
    return [
        row
        for row in catalog
        if token in " ".join(str(row.get(k, "")) for k in row).lower()
    ]


def keyword_counts(text: str) -> dict[str, int]:
    low = text.lower()
    return {
        family: sum(low.count(keyword.lower()) for keyword in keywords)
        for family, keywords in KEYWORDS.items()
    }


def likely_fields(text: str) -> list[str]:
    counts: Counter[str] = Counter()
    for pattern in FIELD_PATTERNS:
        for match in pattern.findall(text):
            field = str(match).strip()
            if len(field) < 3 or len(field) > 48:
                continue
            low = field.lower()
            if low in {"read", "write", "copy", "true", "false", "none"}:
                continue
            if "/" in field or "\\" in field:
                continue
            if re.fullmatch(r"\d+(\.\d+)?", field):
                continue
            counts[field] += 1
    return [field for field, _ in counts.most_common(40)]


def evidence_snippets(text: str, family: str, max_snippets: int = 5) -> list[str]:
    low = text.lower()
    snippets: list[str] = []
    for keyword in KEYWORDS[family]:
        start = low.find(keyword.lower())
        if start < 0:
            continue
        left = max(0, start - 80)
        right = min(len(text), start + len(keyword) + 120)
        snippet = re.sub(r"\s+", " ", text[left:right]).strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= max_snippets:
            break
    return snippets


def route_decision(counts: dict[str, int], fields: list[str]) -> dict[str, Any]:
    field_text = " ".join(fields).lower()
    reagent = counts["guide_reagent"] >= 3 or any(x in field_text for x in ["guide", "sgrna", "grna"])
    maturity = counts["time_maturity"] >= 2 or any(x in field_text for x in ["time", "hour", "day"])
    viability = counts["viability_fitness"] >= 2 or any(x in field_text for x in ["viability", "fitness", "growth"])
    background = counts["background_cell"] >= 2
    return {
        "supports_reagent_route": reagent,
        "supports_maturity_route": maturity,
        "supports_viability_route": viability,
        "supports_background_route": background,
        "usable_now": False,
        "required_next_source": (
            "condition-level CSV/TSV with dataset,condition,artifact_value extracted from original supplement/GEO/figshare"
        ),
    }


def main() -> int:
    catalog = catalogue_rows()
    catalog_preflight = json.loads(CATALOG_PREFLIGHT.read_text(encoding="utf-8")) if CATALOG_PREFLIGHT.exists() else {}
    rows: list[dict[str, Any]] = []
    family_totals: defaultdict[str, int] = defaultdict(int)

    for path in sorted(SRC_DIR.glob("*.ipynb")):
        name = path.stem
        text, valid_json, parse_error = load_notebook_sources(path)
        counts = keyword_counts(text)
        fields = likely_fields(text)
        decision = route_decision(counts, fields)
        for family, count in counts.items():
            family_totals[family] += count
        cat_rows = matched_catalogue_for(name, catalog)
        rows.append(
            {
                "notebook": name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "valid_json": valid_json,
                "parse_error": parse_error,
                "local_datasets": NOTEBOOK_TO_LOCAL_DATASETS.get(name, []),
                "catalogue_match_count": len(cat_rows),
                "catalogue_timepoints": "; ".join(sorted({str(r.get("# timepoints", "")) for r in cat_rows if r.get("# timepoints")})),
                "catalogue_doses": "; ".join(sorted({str(r.get("# doses", "")) for r in cat_rows if r.get("# doses")})),
                "keyword_counts": counts,
                "likely_fields": fields,
                "snippets": {family: evidence_snippets(text, family, max_snippets=3) for family in KEYWORDS},
                **decision,
            }
        )

    route_support = {
        "reagent_route_notebooks": [r["notebook"] for r in rows if r["supports_reagent_route"]],
        "maturity_route_notebooks": [r["notebook"] for r in rows if r["supports_maturity_route"]],
        "viability_route_notebooks": [r["notebook"] for r in rows if r["supports_viability_route"]],
        "background_route_notebooks": [r["notebook"] for r in rows if r["supports_background_route"]],
    }
    status = "scperturb_notebook_metadata_preflight_partial_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": "local notebook/catalogue metadata scan only; no h5ad/expression/checkpoint/canonical multi/Track C query/training/inference/GPU",
        "source_dir": str(SRC_DIR),
        "notebook_count": len(rows),
        "catalogue_preflight_status": catalog_preflight.get("status"),
        "route_support": route_support,
        "keyword_family_totals": dict(sorted(family_totals.items())),
        "rows": rows,
        "decision": (
            "Notebooks/catalogue identify source metadata routes, especially guide/reagent assignment and sparse timepoint "
            "metadata, but no condition-level artifact table exists yet. Scaling completion needs extraction of source-specific "
            "CSV artifacts before any GPU or training-set weighting route."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "notebook",
            "size_bytes",
            "valid_json",
            "local_datasets",
            "catalogue_match_count",
            "catalogue_timepoints",
            "catalogue_doses",
            "supports_reagent_route",
            "supports_maturity_route",
            "supports_viability_route",
            "supports_background_route",
            "likely_fields",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "notebook": row["notebook"],
                    "size_bytes": row["size_bytes"],
                    "valid_json": row["valid_json"],
                    "local_datasets": ";".join(row["local_datasets"]),
                    "catalogue_match_count": row["catalogue_match_count"],
                    "catalogue_timepoints": row["catalogue_timepoints"],
                    "catalogue_doses": row["catalogue_doses"],
                    "supports_reagent_route": row["supports_reagent_route"],
                    "supports_maturity_route": row["supports_maturity_route"],
                    "supports_viability_route": row["supports_viability_route"],
                    "supports_background_route": row["supports_background_route"],
                    "likely_fields": ";".join(row["likely_fields"][:20]),
                }
            )

    lines = [
        "# LatentFM scPerturb Notebook Metadata Preflight",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- Parses only downloaded local scPerturb notebooks and the small scPerturb catalogue.",
        "- Does not download h5ad files, read expression matrices, read checkpoints, use canonical multi, use Track C query, train, infer, or use GPU.",
        "- This is a metadata/source preflight, not a condition-level artifact gate.",
        "",
        "## Summary",
        "",
        f"- notebooks scanned: `{len(rows)}`",
        f"- reagent/guide route notebooks: `{len(route_support['reagent_route_notebooks'])}` `{route_support['reagent_route_notebooks']}`",
        f"- maturity/time route notebooks: `{len(route_support['maturity_route_notebooks'])}` `{route_support['maturity_route_notebooks']}`",
        f"- viability/fitness route notebooks: `{len(route_support['viability_route_notebooks'])}` `{route_support['viability_route_notebooks']}`",
        f"- background/cell route notebooks: `{len(route_support['background_route_notebooks'])}` `{route_support['background_route_notebooks']}`",
        "",
        "## Notebook Rows",
        "",
        "| notebook | valid JSON | local datasets | catalogue timepoints | catalogue doses | routes | top likely fields |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        routes = []
        if row["supports_reagent_route"]:
            routes.append("reagent")
        if row["supports_maturity_route"]:
            routes.append("maturity")
        if row["supports_viability_route"]:
            routes.append("viability")
        if row["supports_background_route"]:
            routes.append("background")
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |".format(
                row["notebook"],
                row["valid_json"],
                ";".join(row["local_datasets"]),
                row["catalogue_timepoints"],
                row["catalogue_doses"],
                ",".join(routes) if routes else "none",
                ", ".join(f"`{x}`" for x in row["likely_fields"][:8]),
            )
        )
    lines += [
        "",
        "## Decision",
        "",
        "- This scan strengthens the source-acquisition path for systematic scaling: guide/reagent assignment and background metadata are real source leads; time/maturity is sparse and mostly Dixit/SciPlex-like; viability/fitness is not yet a condition-level artifact.",
        "- No GPU is authorized until one of these sources is converted into a condition-level `dataset,condition,artifact_value` table and passes the existing external-artifact preflight with shuffle/LODO/tail checks.",
        "- Mainline implication: do not launch generic weighting/balancing from catalogue metadata alone; first test whether source-level reagent quality or maturity explains current scaling tails.",
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
