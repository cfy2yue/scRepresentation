#!/usr/bin/env python3
"""Inventory frozen canonical no-harm decisions for nontrivial positives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_global_noharm_positive_class_inventory_20260624.json"
OUT_MD = REPORTS / "LATENTFM_GLOBAL_NOHARM_POSITIVE_CLASS_INVENTORY_20260624.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_decision_files() -> list[Path]:
    paths = set(REPORTS.glob("*noharm*decision*.json")) | set(REPORTS.glob("*canonical*decision*.json"))
    return sorted(path for path in paths if path.is_file())


def metric_value(metrics: dict[str, Any], key: str, field: str) -> float | None:
    item = metrics.get(key)
    if not isinstance(item, dict):
        return None
    value = item.get(field)
    return None if value is None else float(value)


def rows_from_object(path: Path, obj: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj.get("rows"), list):
        for row in obj["rows"]:
            if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
                rows.append(
                    {
                        "path": str(path),
                        "run": row.get("run") or obj.get("run_name") or path.stem,
                        "status": row.get("gate_status") or row.get("status") or obj.get("status"),
                        "top_status": obj.get("status") or (obj.get("decision") or {}).get("status"),
                        "metrics": row.get("metrics") or {},
                    }
                )
    if isinstance(obj.get("metrics"), dict):
        rows.append(
            {
                "path": str(path),
                "run": obj.get("run_name") or path.stem,
                "status": obj.get("gate_status") or obj.get("status"),
                "top_status": obj.get("status") or (obj.get("decision") or {}).get("status"),
                "metrics": obj.get("metrics") or {},
            }
        )
    if isinstance(obj.get("tables"), dict):
        merged: dict[str, Any] = {}
        for table in obj["tables"].values():
            if isinstance(table, dict):
                merged.update(table)
        rows.append(
            {
                "path": str(path),
                "run": path.stem,
                "status": (obj.get("decision") or {}).get("status"),
                "top_status": (obj.get("decision") or {}).get("status"),
                "metrics": merged,
            }
        )
    return rows


def classify(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    test_pp = metric_value(metrics, "test_single:pearson_pert", "delta_mean")
    test_harm = metric_value(metrics, "test_single:pearson_pert", "p_harm")
    if test_pp is None:
        test_pp = metric_value(metrics, "all_test_single:pearson_pert", "delta_mean")
    if test_harm is None:
        test_harm = metric_value(metrics, "all_test_single:pearson_pert", "p_harm")
    family_pp = metric_value(metrics, "family_gene:pearson_pert", "delta_mean")
    family_harm = metric_value(metrics, "family_gene:pearson_pert", "p_harm")
    test_mmd_harm = metric_value(metrics, "test_single:test_mmd_clamped", "p_harm")
    if test_mmd_harm is None:
        test_mmd_harm = metric_value(metrics, "all_test_single:test_mmd_clamped", "p_harm")
    family_mmd_harm = metric_value(metrics, "family_gene:test_mmd_clamped", "p_harm")

    noharm = (
        test_pp is not None
        and family_pp is not None
        and test_harm is not None
        and family_harm is not None
        and test_harm <= 0.35
        and family_harm <= 0.35
        and (test_mmd_harm is None or test_mmd_harm <= 0.80)
        and (family_mmd_harm is None or family_mmd_harm <= 0.80)
    )
    material_positive = noharm and (test_pp >= 0.005 or family_pp >= 0.005)
    trivial_noop = noharm and abs(test_pp or 0.0) < 1e-9 and abs(family_pp or 0.0) < 1e-9
    status = "nontrivial_positive" if material_positive else ("trivial_or_noop_noharm" if noharm else "not_noharm_positive")
    reasons: list[str] = []
    if test_harm is None or family_harm is None:
        reasons.append("missing_primary_pp_harm")
    else:
        if test_harm > 0.35:
            reasons.append("test_single_pp_harm_gt_0p35")
        if family_harm > 0.35:
            reasons.append("family_gene_pp_harm_gt_0p35")
    if noharm and not material_positive:
        reasons.append("no_material_primary_pp_gain")
    return {
        **{k: v for k, v in row.items() if k != "metrics"},
        "test_single_pp_delta": test_pp,
        "test_single_pp_p_harm": test_harm,
        "family_gene_pp_delta": family_pp,
        "family_gene_pp_p_harm": family_harm,
        "test_single_mmd_p_harm": test_mmd_harm,
        "family_gene_mmd_p_harm": family_mmd_harm,
        "class": status,
        "reasons": reasons,
        "trivial_noop": trivial_noop,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for path in iter_decision_files():
        try:
            rows.extend(classify(row) for row in rows_from_object(path, load(path)))
        except Exception as exc:
            rows.append({"path": str(path), "run": path.stem, "class": "parse_failed", "reasons": [str(exc)]})

    nontrivial = [row for row in rows if row.get("class") == "nontrivial_positive"]
    trivial = [row for row in rows if row.get("class") == "trivial_or_noop_noharm"]
    status = (
        "global_noharm_positive_class_inventory_has_nontrivial_positive_no_gpu"
        if nontrivial
        else "global_noharm_positive_class_inventory_no_nontrivial_positive_no_gpu"
    )
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            row.get("class") == "nontrivial_positive",
            row.get("class") == "trivial_or_noop_noharm",
            float(row.get("test_single_pp_delta") or -999),
            float(row.get("family_gene_pp_delta") or -999),
        ),
        reverse=True,
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "historical_frozen_canonical_noharm_audit": True,
            "uses_for_current_checkpoint_selection": False,
            "reads_canonical_multi": False,
            "reads_heldout_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "summary": {
            "n_files": len(iter_decision_files()),
            "n_rows": len(rows),
            "n_nontrivial_positive": len(nontrivial),
            "n_trivial_or_noop_noharm": len(trivial),
        },
        "decision": {
            "gpu_next_action": "none",
            "surrogate_promotion_route": "closed_without_nontrivial_positive_class" if not nontrivial else "requires_new_train_only_surrogate_gate",
        },
        "rows": rows_sorted,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Global No-Harm Positive-Class Inventory",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only historical audit of frozen canonical no-harm decision JSONs.",
        "- Canonical metrics are used only to inventory whether a positive class exists; this does not select or promote any current checkpoint.",
        "- Does not read canonical multi, held-out Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- decision files scanned: `{len(iter_decision_files())}`",
        f"- decision rows parsed: `{len(rows)}`",
        f"- nontrivial positive no-harm rows: `{len(nontrivial)}`",
        f"- trivial/no-op no-harm rows: `{len(trivial)}`",
        "",
        "## Top Rows",
        "",
        "| class | run | test pp | test p_harm | family pp | family p_harm | reasons |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows_sorted[:20]:
        lines.append(
            f"| `{row.get('class')}` | `{row.get('run')}` | {fmt(row.get('test_single_pp_delta'))} | {fmt(row.get('test_single_pp_p_harm'))} | {fmt(row.get('family_gene_pp_delta'))} | {fmt(row.get('family_gene_pp_p_harm'))} | `{','.join(row.get('reasons') or [])}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- GPU authorized: `False`",
            "- A nontrivial positive class is required before another global no-harm surrogate can authorize GPU.",
            "- If no nontrivial positive exists, surrogate-to-promotion remains closed; future GPU requires a genuinely new mechanism and its own train-only gate.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
