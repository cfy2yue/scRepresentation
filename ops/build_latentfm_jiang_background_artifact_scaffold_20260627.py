#!/usr/bin/env python3
"""Build a local Jiang condition/background scaffold for external artifacts.

CPU/report-only. Reads canonical split metadata and Jiang h5ad obs metadata
only. It does not train, infer, read checkpoints, read canonical multi for
selection, read Track C query, or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
H5AD_DIR = ROOT / "dataset/biFlow_data/gt_stack"
OUT_DIR = ROOT / "reports/jiang_background_artifact_scaffold_20260627"
OUT_CSV = OUT_DIR / "jiang_condition_background_scaffold.csv"
OUT_JSON = ROOT / "reports/latentfm_jiang_background_artifact_scaffold_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_JIANG_BACKGROUND_ARTIFACT_SCAFFOLD_20260627.md"


def load_split() -> dict[str, Any]:
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def jiang_datasets(split: dict[str, Any]) -> list[str]:
    return sorted(ds for ds in split if ds.startswith("Jiang_"))


def read_backgrounds(dataset: str) -> tuple[list[str], dict[str, int]]:
    path = H5AD_DIR / f"{dataset}.h5ad"
    a = ad.read_h5ad(path, backed="r")
    try:
        counts = a.obs["cell_type"].astype(str).value_counts().to_dict()
    finally:
        a.file.close()
    return sorted(counts), {str(k): int(v) for k, v in counts.items()}


def main() -> int:
    split = load_split()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    dataset_summary: dict[str, Any] = {}
    for ds in jiang_datasets(split):
        cytokine = ds.removeprefix("Jiang_")
        backgrounds, bg_counts = read_backgrounds(ds)
        parts = split[ds]
        for split_name in ("train", "test", "test_single"):
            for condition in sorted(set(parts.get(split_name, []))):
                for background in backgrounds:
                    rows.append(
                        {
                            "dataset": ds,
                            "cytokine": cytokine,
                            "condition": condition,
                            "split": split_name,
                            "cell_background": background,
                            "join_key": f"{cytokine}|{condition}|{background}",
                        }
                    )
        dataset_summary[ds] = {
            "cytokine": cytokine,
            "backgrounds": backgrounds,
            "background_cell_counts": bg_counts,
            "train_conditions": len(set(parts.get("train", []))),
            "test_conditions": len(set(parts.get("test", []))),
            "test_single_conditions": len(set(parts.get("test_single", []))),
        }

    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["dataset", "cytokine", "condition", "split", "cell_background", "join_key"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "jiang_background_artifact_scaffold_ready_no_gpu",
        "gpu_authorized": False,
        "canonical_split": str(SPLIT),
        "h5ad_dir": str(H5AD_DIR),
        "rows": len(rows),
        "datasets": len(dataset_summary),
        "dataset_summary": dataset_summary,
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON), "markdown": str(OUT_MD)},
        "decision": (
            "Use this scaffold to materialize author-DE rows by "
            "cytokine/pathway + perturbation/condition + cell_background. It is "
            "not itself an artifact signal and does not authorize GPU."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Jiang Background Artifact Scaffold 2026-06-27",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only scaffold for external Jiang artifact materialization.",
        "- Reads canonical split and Jiang h5ad `.obs['cell_type']` metadata only.",
        "- Does not train, infer, read checkpoints, read canonical multi for selection, read Track C query, or use GPU.",
        "",
        "## Summary",
        "",
        f"- datasets: `{payload['datasets']}`",
        f"- scaffold rows: `{payload['rows']}`",
        f"- CSV: `{OUT_CSV}`",
        "",
        "| dataset | cytokine | backgrounds | train | test | test_single |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ds, row in dataset_summary.items():
        lines.append(
            f"| `{ds}` | `{row['cytokine']}` | {len(row['backgrounds'])} | "
            f"{row['train_conditions']} | {row['test_conditions']} | {row['test_single_conditions']} |"
        )
    lines += ["", "## Decision", "", payload["decision"], "", f"- JSON: `{OUT_JSON}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": payload["status"], "rows": len(rows), "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
