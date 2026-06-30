#!/usr/bin/env python3
"""Build loader-compatible train/test splits for all-modality dose-aware runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
MATERIALIZER_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_materializer_gate_20260625.json"
POST_GATE_JSONS = {
    "schema": ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_schema_gate_20260625.json",
    "dryload": ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_dryload_gate_20260625.json",
    "chemical_conditioning": ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_chemical_conditioning_gate_20260625.json",
    "design": ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_design_controls_20260625.json",
}
OUT_DIR = ROOT / "dataset/biFlow_data/xverse_true_cell_count_allmodality_doseaware_loader_splits_20260625"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_loader_splits_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_ALLMODALITY_DOSEAWARE_LOADER_SPLITS_20260625.md"


def load_json(path: Path) -> Any:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def convert_split(raw: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for ds, groups in sorted(raw.items()):
        train = [str(c) for c in groups.get("train") or []]
        test = [str(c) for c in groups.get("internal_val_allmodality_doseaware") or []]
        if train or test:
            out[str(ds)] = {"train": train, "test": test}
    return out


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Dose-Aware Loader Splits",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only split adapter.",
        "- Maps `internal_val_allmodality_doseaware` to `test` for existing train/eval loaders.",
        "- Does not change condition membership, train/eval boundary, metrics, checkpoints, canonical multi, or Track C query.",
        "",
        "## Rows",
        "",
        "| run id | status | train | test | output | reasons |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| `{row['run_id']}` | `{row['status']}` | {row.get('train_conditions', 0)} | {row.get('test_conditions', 0)} | `{row.get('loader_split_file', '')}` | {', '.join(row.get('reasons') or []) or 'none'} |"
        )
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    materializer = load_json(MATERIALIZER_JSON)
    reasons: list[str] = []
    if materializer.get("status") != "allmodality_doseaware_materialized_no_gpu":
        reasons.append(f"materializer_not_complete:{materializer.get('status')}")
    expected = {
        "schema": "allmodality_doseaware_schema_pass_no_gpu",
        "dryload": "allmodality_doseaware_dryload_pass_no_gpu",
        "chemical_conditioning": "allmodality_doseaware_chemical_conditioning_pass_no_gpu",
    }
    for name, status in expected.items():
        payload = load_json(POST_GATE_JSONS[name])
        if payload.get("status") != status:
            reasons.append(f"{name}_not_pass:{payload.get('status')}")
    design = load_json(POST_GATE_JSONS["design"])
    if not design.get("smoke_ready_after_schema_dryload"):
        reasons.append(f"design_not_smoke_ready:{design.get('status')}")
    rows = []
    if not reasons:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for row in materializer.get("materialized_rows") or []:
            row_reasons: list[str] = []
            raw_split = load_json(Path(row["split_file"]))
            split = convert_split(raw_split)
            train_n = sum(len(v.get("train") or []) for v in split.values())
            test_n = sum(len(v.get("test") or []) for v in split.values())
            if train_n <= 0 or test_n <= 0:
                row_reasons.append("empty_train_or_test")
            out_file = OUT_DIR / f"loader_split_{row['run_id']}.json"
            if not row_reasons:
                out_file.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            rows.append(
                {
                    "run_id": row["run_id"],
                    "status": "ok" if not row_reasons else "fail",
                    "reasons": row_reasons,
                    "source_split_file": row["split_file"],
                    "loader_split_file": str(out_file),
                    "train_conditions": train_n,
                    "test_conditions": test_n,
                }
            )
    if reasons:
        status = "allmodality_doseaware_loader_splits_not_ready_no_gpu"
        next_action = "run after post-materialization gates pass"
    elif all(row.get("status") == "ok" for row in rows):
        status = "allmodality_doseaware_loader_splits_ready_no_gpu"
        next_action = "resource_audit_then_bounded_gpu_smoke"
    else:
        status = "allmodality_doseaware_loader_splits_fail_no_gpu"
        next_action = "fix split conversion"
    payload = {
        "status": status,
        "reasons": reasons,
        "rows": rows,
        "gpu_authorized": False,
        "next_action": next_action,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
