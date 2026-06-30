#!/usr/bin/env python3
"""Audit xverse h5ad obs schemas for cell-background balancing feasibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
DEFAULT_MANIFEST = ROOT / "dataset/latentfm_full/xverse/manifest.json"
DEFAULT_BIFLOW = ROOT / "dataset/biFlow_data"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_xverse_obs_schema_background_gate_20260624.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_XVERSE_OBS_SCHEMA_BACKGROUND_GATE_20260624.md"

BACKGROUND_TOKENS = ("cell", "line", "type", "cov", "batch", "donor", "tissue")
CONDITION_TOKENS = ("pert", "condition", "drug", "dose", "pathway", "cov")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_preview(values: Any, limit: int) -> dict[str, Any]:
    vals = [str(x) for x in values.dropna().unique().tolist()]
    vals = sorted(vals)
    return {"n_unique": len(vals), "values": vals[:limit]}


def inspect_h5ad(path: Path, *, preview_limit: int) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    a = ad.read_h5ad(path, backed="r")
    try:
        obs = a.obs
        columns = [str(c) for c in obs.columns]
        bg_cols = [
            c for c in columns
            if any(tok in c.lower() for tok in BACKGROUND_TOKENS)
        ]
        condition_cols = [
            c for c in columns
            if any(tok in c.lower() for tok in CONDITION_TOKENS)
        ]
        previews = {}
        for col in sorted(set(bg_cols + condition_cols)):
            try:
                previews[col] = unique_preview(obs[col], preview_limit)
            except Exception as exc:  # pragma: no cover - defensive audit path
                previews[col] = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "path": str(path),
            "status": "ok",
            "n_obs": int(a.n_obs),
            "n_vars": int(a.n_vars),
            "obs_columns": columns,
            "background_candidate_columns": bg_cols,
            "condition_candidate_columns": condition_cols,
            "previews": previews,
        }
    finally:
        a.file.close()


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM xverse Obs Schema / Background Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only h5ad obs-schema audit.",
        "- Reads `.obs` columns and small unique-value previews only.",
        "- Does not read expression matrices, model outputs, canonical outcomes, or Track C query.",
        "",
        "## Gate Summary",
        "",
        f"- datasets inspected: `{payload['datasets_inspected']}`",
        f"- mixed-background datasets: `{', '.join(payload['mixed_background_datasets'])}`",
        f"- mixed datasets with candidate background columns: `{payload['mixed_with_candidate_background_columns']}`",
        "",
        "| dataset | gt status | gt background cols | gt condition cols | control background cols |",
        "|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        gt = row["gt_stack"]
        ctrl = row["control_stack"]
        lines.append(
            f"| `{row['dataset']}` | `{gt.get('status')}` | "
            f"`{', '.join(gt.get('background_candidate_columns') or [])}` | "
            f"`{', '.join(gt.get('condition_candidate_columns') or [])}` | "
            f"`{', '.join(ctrl.get('background_candidate_columns') or [])}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if payload["status"] == "background_obs_schema_gate_pass":
        lines.append(
            "- At least one mixed-background dataset exposes candidate obs columns, so a condition-level "
            "background-balancing split can be designed with a stricter follow-up audit."
        )
    else:
        lines.append(
            "- Mixed-background datasets do not expose enough candidate obs columns for condition-level "
            "background balancing; use dataset-level background labels only unless raw source mapping is added."
        )
    lines += [
        "",
        "## JSON",
        "",
        f"`{payload['out_json']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--biflow-dir", type=Path, default=DEFAULT_BIFLOW)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    ap.add_argument("--preview-limit", type=int, default=12)
    args = ap.parse_args()

    manifest = read_json(args.manifest)
    datasets = sorted((manifest.get("datasets") or {}).keys())
    mixed = []
    inventory = ROOT / "reports/latentfm_dataset_scaling_inventory_20260624.json"
    if inventory.is_file():
        inv = read_json(inventory)
        mixed = list((inv.get("summary") or {}).get("datasets_with_multiple_cell_backgrounds") or [])

    rows = []
    mixed_with_bg = 0
    for ds in datasets:
        gt = inspect_h5ad(args.biflow_dir / "gt_stack" / f"{ds}.h5ad", preview_limit=args.preview_limit)
        ctrl = inspect_h5ad(args.biflow_dir / "control_stack" / f"{ds}.h5ad", preview_limit=args.preview_limit)
        if ds in set(mixed) and gt.get("background_candidate_columns"):
            mixed_with_bg += 1
        rows.append({"dataset": ds, "gt_stack": gt, "control_stack": ctrl})

    status = (
        "background_obs_schema_gate_pass"
        if mixed and mixed_with_bg > 0
        else "background_obs_schema_gate_diagnostic_only"
    )
    payload = {
        "status": status,
        "manifest": str(args.manifest),
        "biflow_dir": str(args.biflow_dir),
        "datasets_inspected": len(rows),
        "mixed_background_datasets": mixed,
        "mixed_with_candidate_background_columns": mixed_with_bg,
        "rows": rows,
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
