#!/usr/bin/env python3
"""Build a local scaffold for DepMap 24Q4 dependency artifact materialization.

CPU/report-only. Reads canonical split metadata only. It does not train, infer,
read checkpoints, read canonical multi for selection, read Track C query, or use
GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
OUT_DIR = ROOT / "reports/depmap_24q4_artifact_scaffold_20260627"
OUT_CSV = OUT_DIR / "depmap_24q4_gene_condition_scaffold.csv"
OUT_JSON = ROOT / "reports/latentfm_depmap_24q4_artifact_scaffold_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_DEPMAP_24Q4_ARTIFACT_SCAFFOLD_20260627.md"


DATASET_TO_MODEL = {
    "DixitRegev2016_K562_TFs_High_MOI": ("K562", "ACH-000551"),
    "ReplogleWeissman2022_K562_gwps": ("K562", "ACH-000551"),
    "Replogle_RPE1essential": ("RPE1", "ACH-002464"),
    "Nadig_hepg2": ("HepG2", "ACH-000739"),
    "Nadig_jurket": ("Jurkat", "ACH-000995"),
    "Frangieh": ("A375", "ACH-000219"),
}

EXCLUDED_AMBIGUOUS = {
    "sciplex3_A549": "drug conditions, not target-gene conditions",
    "sciplex3_MCF7": "drug conditions, not target-gene conditions",
    "sciplex3_K562": "drug conditions, not target-gene conditions",
}


def load_split() -> dict[str, Any]:
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def main() -> int:
    split = load_split()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    dataset_summary: dict[str, Any] = {}
    for ds, (cell_line, model_id) in DATASET_TO_MODEL.items():
        if ds not in split:
            dataset_summary[ds] = {"status": "missing_from_split", "cell_line": cell_line, "model_id": model_id}
            continue
        parts = split[ds]
        seen = set()
        for split_name in ("train", "test", "test_single"):
            for condition in parts.get(split_name, []):
                key = (ds, split_name, condition)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "dataset": ds,
                        "condition": condition,
                        "split": split_name,
                        "cell_background": cell_line,
                        "depmap_model_id": model_id,
                        "target_gene": condition,
                    }
                )
        dataset_summary[ds] = {
            "status": "included",
            "cell_line": cell_line,
            "model_id": model_id,
            "train_conditions": len(set(parts.get("train", []))),
            "test_conditions": len(set(parts.get("test", []))),
            "test_single_conditions": len(set(parts.get("test_single", []))),
        }

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["dataset", "condition", "split", "cell_background", "depmap_model_id", "target_gene"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "depmap_24q4_artifact_scaffold_ready_no_gpu",
        "gpu_authorized": False,
        "canonical_split": str(SPLIT),
        "rows": len(rows),
        "included_datasets": sorted(DATASET_TO_MODEL),
        "excluded_ambiguous_datasets": EXCLUDED_AMBIGUOUS,
        "dataset_summary": dataset_summary,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
        "decision": (
            "Use this scaffold only after DepMap sources download and ModelID/gene columns are verified. "
            "SciPlex A549/K562/MCF7 drug conditions are deliberately excluded from gene-dependency materialization."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM DepMap 24Q4 Artifact Scaffold 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only scaffold for DepMap source materialization.",
        "- Reads canonical split metadata only.",
        "- Does not train, infer, read checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- scaffold rows: `{payload['rows']}`",
        f"- CSV: `{OUT_CSV}`",
        "",
        "| dataset | cell line | ModelID | train | test | test_single |",
        "|---|---|---|---:|---:|---:|",
    ]
    for ds, row in dataset_summary.items():
        if row["status"] != "included":
            lines.append(f"| `{ds}` | `{row['cell_line']}` | `{row['model_id']}` | NA | NA | NA |")
        else:
            lines.append(
                f"| `{ds}` | `{row['cell_line']}` | `{row['model_id']}` | "
                f"{row['train_conditions']} | {row['test_conditions']} | {row['test_single_conditions']} |"
            )
    lines += [
        "",
        "## Excluded Ambiguous Datasets",
        "",
    ]
    for ds, reason in EXCLUDED_AMBIGUOUS.items():
        lines.append(f"- `{ds}`: {reason}")
    lines += ["", "## Decision", "", payload["decision"], "", f"- JSON: `{OUT_JSON}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": payload["status"], "rows": len(rows), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
