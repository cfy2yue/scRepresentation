#!/usr/bin/env python3
"""Extract Norman curation gene-program growth artifacts.

CPU/source-only. Parses the local scPerturb Norman curation notebook for
published/curated gene-program condition labels and materializes condition-level
numeric artifacts. It does not read expression matrices, checkpoints, canonical
multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import ast
import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
NOTEBOOK = ROOT / "reports/external_artifact_sources_20260626/scperturb_notebooks/Norman_2019_curation.ipynb"
OUT_DIR = ROOT / "reports/norman_program_growth_artifacts_20260626"
OUT_MD = ROOT / "reports/LATENTFM_NORMAN_PROGRAM_GROWTH_ARTIFACTS_20260626.md"
OUT_JSON = ROOT / "reports/latentfm_norman_program_growth_artifacts_20260626.json"

PROGRAM_NAMES = {
    "G1_CYCLE": "G1 cell cycle arrest",
    "ERYTHROID": "Erythroid",
    "PIONEER_FACTORS": "Pioneer factors",
    "GRANULOCYTE_APOPTOSIS": "Granulocyte/apoptosis",
    "PRO_GROWTH": "Pro-growth",
    "MEGAKARYOCYTE": "Megakaryocyte",
}


def parse_program_lists() -> dict[str, list[str]]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    values: dict[str, list[str]] = {}
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in PROGRAM_NAMES:
                continue
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(val, list):
                values[target.id] = [str(x) for x in val]
    return values


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["dataset", "condition", "artifact_value", "program", "source", "evidence_url"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    programs = parse_program_lists()
    condition_to_program: dict[str, str] = {}
    for key, values in programs.items():
        for condition in values:
            condition_to_program[condition] = PROGRAM_NAMES[key]

    source = "scPerturb Norman_2019_curation notebook gene_program mapping"
    evidence_url = "https://github.com/theislab/sc-pert/blob/main/datasets/Norman_2019_curation.ipynb"
    pro_rows = []
    contrast_rows = []
    for condition, program in sorted(condition_to_program.items()):
        pro_rows.append(
            {
                "dataset": "NormanWeissman2019_filtered",
                "condition": condition,
                "artifact_value": 1.0 if program == "Pro-growth" else 0.0,
                "program": program,
                "source": source,
                "evidence_url": evidence_url,
            }
        )
        if program == "Pro-growth":
            contrast = 1.0
        elif program in {"G1 cell cycle arrest", "Granulocyte/apoptosis"}:
            contrast = -1.0
        else:
            contrast = 0.0
        contrast_rows.append(
            {
                "dataset": "NormanWeissman2019_filtered",
                "condition": condition,
                "artifact_value": contrast,
                "program": program,
                "source": source,
                "evidence_url": evidence_url,
            }
        )

    pro_csv = OUT_DIR / "norman_program_pro_growth_indicator.csv"
    contrast_csv = OUT_DIR / "norman_program_growth_arrest_contrast.csv"
    program_csv = OUT_DIR / "norman_program_mapping.csv"
    write_artifact(pro_csv, pro_rows)
    write_artifact(contrast_csv, contrast_rows)
    write_artifact(program_csv, [{**row, "artifact_value": ""} for row in pro_rows])

    counts: dict[str, int] = {}
    for program in condition_to_program.values():
        counts[program] = counts.get(program, 0) + 1

    payload = {
        "timestamp": timestamp,
        "status": "norman_program_growth_artifacts_ready_cpu_preflight_next",
        "gpu_authorized": False,
        "boundary": {
            "reads_expression": False,
            "reads_checkpoints": False,
            "uses_canonical_multi": False,
            "uses_trackc_query": False,
            "uses_training": False,
            "uses_gpu": False,
        },
        "notebook": str(NOTEBOOK),
        "n_conditions": len(condition_to_program),
        "program_counts": counts,
        "outputs": {
            "pro_growth_indicator": str(pro_csv),
            "growth_arrest_contrast": str(contrast_csv),
            "program_mapping": str(program_csv),
        },
        "decision": "source artifact only; run strict external-artifact preflight before considering any training route",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md = f"""# LatentFM Norman Program Growth Artifacts

Timestamp: `{timestamp}`

Status: `norman_program_growth_artifacts_ready_cpu_preflight_next`

GPU authorized: `False`

## Boundary

- Parses only the local scPerturb `Norman_2019_curation.ipynb` source notebook.
- Materializes condition-level program/growth annotations; no expression matrix, checkpoint, canonical multi, Track C query, training, inference, or GPU.
- This is a source artifact, not a training authorization.

## Program Counts

{md_table(["program", "conditions"], [[k, v] for k, v in sorted(counts.items())])}

## Artifacts

{md_table(["artifact", "rows", "output"], [
    ["external_norman_program_pro_growth_indicator", len(pro_rows), pro_csv],
    ["external_norman_program_growth_arrest_contrast", len(contrast_rows), contrast_csv],
])}

## Decision

- This route is potentially non-duplicate because it targets curated growth/program burden rather than read/UMI/QC/source labels.
- It is still high-risk and likely underpowered because it is Norman-only; strict preflight must reject it unless overlap, dataset count, variation, tail, and MMD controls pass.

## Outputs

- JSON: `{OUT_JSON}`
- pro-growth indicator: `{pro_csv}`
- growth/arrest contrast: `{contrast_csv}`
- program mapping: `{program_csv}`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
