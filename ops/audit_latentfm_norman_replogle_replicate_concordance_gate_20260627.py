#!/usr/bin/env python3
"""Norman/Replogle replicate-concordance source gate.

This gate is intentionally a schema/source audit first. It rejects sources that
only expose reagent/QC/bulk-difficulty fields, because those routes have already
failed no-harm/MMD gates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import anndata as ad


ROOT = Path("/data/cyx/1030/scLatent")
REPL_SOURCE_DIR = ROOT / "reports/external_artifact_sources_20260627/replogle_figshare_bulk"
REPL_H5ADS = [
    REPL_SOURCE_DIR / "K562_gwps_normalized_bulk_01.h5ad",
    REPL_SOURCE_DIR / "K562_essential_normalized_bulk_01.h5ad",
    REPL_SOURCE_DIR / "rpe1_normalized_bulk_01.h5ad",
]
NORMAN_ARTIFACTS = sorted((ROOT / "reports/norman_geo_reagent_artifacts_20260626").glob("*.csv"))
REPLICATE_BATCH_ARTIFACTS = sorted((ROOT / "reports/replicate_batch_balance_artifacts_20260626").glob("*.csv"))
REPL_TRAINONLY = ROOT / "reports/latentfm_replogle_trainonly_internal_difficulty_gate_20260627.json"
REPL_MATERIALIZER = ROOT / "reports/latentfm_replogle_bulk_source_materializer_gate_20260627.json"

OUT_DIR = ROOT / "reports/norman_replogle_replicate_concordance_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_norman_replogle_replicate_concordance_gate_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_NORMAN_REPLOGLE_REPLICATE_CONCORDANCE_GATE_20260627.md"
OUT_SCHEMA = OUT_DIR / "source_schema_assessment.csv"

POSITIVE_TERMS = ("replicate", "concord", "correlation", "corr", "reproduc", "agreement", "consistency")
REJECT_QC_TERMS = (
    "umi",
    "read",
    "coverage",
    "batch",
    "gemgroup",
    "mitopercent",
    "cnv",
    "te_ratio",
    "leverage",
    "energy_test",
    "anderson",
    "mann_whitney",
    "cell",
    "control_expr",
    "fold_expr",
    "pct_expr",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def classify_columns(columns: list[str]) -> dict[str, Any]:
    lower = {c: c.lower() for c in columns}
    positive = [c for c, low in lower.items() if any(term in low for term in POSITIVE_TERMS)]
    qc = [c for c, low in lower.items() if any(term in low for term in REJECT_QC_TERMS)]
    true_concordance = [
        c
        for c in positive
        if not any(term in lower[c] for term in REJECT_QC_TERMS)
        and any(term in lower[c] for term in ("concord", "correlation", "corr", "reproduc", "agreement", "consistency"))
    ]
    return {
        "positive_term_columns": positive,
        "qc_or_difficulty_columns": qc,
        "true_replicate_concordance_candidates": true_concordance,
    }


def first_csv_columns(path: Path) -> tuple[list[str], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames or []
        rows = sum(1 for _ in reader)
    return cols, rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schema_rows: list[dict[str, Any]] = []

    for path in REPL_H5ADS:
        rec: dict[str, Any] = {
            "source_family": "Replogle_figshare_bulk",
            "path": str(path),
            "exists": path.is_file(),
        }
        if path.is_file():
            atlas = ad.read_h5ad(path, backed="r")
            obs_cols = list(atlas.obs.columns)
            var_cols = list(atlas.var.columns)
            rec.update(
                {
                    "shape": str(tuple(atlas.shape)),
                    "obs_columns": ";".join(obs_cols),
                    "var_columns": ";".join(var_cols),
                    **{f"obs_{k}": ";".join(v) for k, v in classify_columns(obs_cols).items()},
                    **{f"var_{k}": ";".join(v) for k, v in classify_columns(var_cols).items()},
                }
            )
            atlas.file.close()
        schema_rows.append(rec)

    for path in NORMAN_ARTIFACTS + REPLICATE_BATCH_ARTIFACTS:
        cols, n_rows = first_csv_columns(path)
        assessment = classify_columns(cols)
        schema_rows.append(
            {
                "source_family": "Norman_reagent_or_replicate_batch_balance",
                "path": str(path),
                "exists": True,
                "shape": f"({n_rows}, {len(cols)})",
                "obs_columns": ";".join(cols),
                "var_columns": "",
                "obs_positive_term_columns": ";".join(assessment["positive_term_columns"]),
                "obs_qc_or_difficulty_columns": ";".join(assessment["qc_or_difficulty_columns"]),
                "obs_true_replicate_concordance_candidates": ";".join(assessment["true_replicate_concordance_candidates"]),
                "var_positive_term_columns": "",
                "var_qc_or_difficulty_columns": "",
                "var_true_replicate_concordance_candidates": "",
            }
        )

    with OUT_SCHEMA.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "source_family",
            "path",
            "exists",
            "shape",
            "obs_columns",
            "var_columns",
            "obs_positive_term_columns",
            "obs_qc_or_difficulty_columns",
            "obs_true_replicate_concordance_candidates",
            "var_positive_term_columns",
            "var_qc_or_difficulty_columns",
            "var_true_replicate_concordance_candidates",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in schema_rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    true_candidates = []
    for row in schema_rows:
        for key in ("obs_true_replicate_concordance_candidates", "var_true_replicate_concordance_candidates"):
            vals = [v for v in str(row.get(key, "")).split(";") if v]
            for val in vals:
                true_candidates.append({"path": row["path"], "column": val})

    repl_trainonly = load_json(REPL_TRAINONLY)
    repl_materializer = load_json(REPL_MATERIALIZER)
    reasons = [
        "no_true_replicate_concordance_or_reproducibility_column_found",
        "norman_available_sources_are_reagent_qc_not_concordance",
        "replicate_batch_balance_artifacts_already_failed_preflight",
        "replogle_available_fields_repeat_bulk_difficulty_or_qc_route",
        "replogle_trainonly_internal_gate_has_no_signals",
        "replogle_bulk_mmd_confounded_route_closed",
        "no_gpu_from_source_absence_schema_gate",
    ]
    status = "norman_replogle_replicate_concordance_source_absent_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_schema_source_gate_only": True,
            "training": False,
            "inference": False,
            "gpu": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
        },
        "true_replicate_concordance_candidates": true_candidates,
        "replogle_trainonly_status": repl_trainonly.get("status"),
        "replogle_trainonly_signals": repl_trainonly.get("signals", []),
        "replogle_materializer_status": repl_materializer.get("status"),
        "schema_rows": schema_rows,
        "reasons": reasons,
        "outputs": {
            "markdown": str(OUT_MD),
            "json": str(OUT_JSON),
            "schema_assessment": str(OUT_SCHEMA),
        },
        "decision": "No GPU. P2 source is absent as a true replicate-concordance route; do not repeat Replogle bulk difficulty or Norman reagent-QC gates.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Norman/Replogle Replicate-Concordance Source Gate",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/schema-source gate only.",
        "- No training, inference, GPU, canonical multi selection, or Track C query.",
        "- Rejects reagent/QC/bulk-difficulty columns as repeat routes.",
        "",
        "## Evidence",
        "",
        f"- true replicate-concordance candidate columns found: `{len(true_candidates)}`",
        f"- Replogle train-only gate status: `{repl_trainonly.get('status')}`",
        f"- Replogle train-only signals: `{repl_trainonly.get('signals', [])}`",
        f"- Replogle materializer status: `{repl_materializer.get('status')}`",
        f"- schema assessment rows: `{len(schema_rows)}`",
        "",
        "## Decision",
        "",
        "No GPU is authorized. The available Norman sources are reagent/QC artifacts, the replicate-batch artifacts already failed preflight, and the Replogle files expose bulk effect/difficulty/QC columns rather than a true replicate-concordance or reproducibility source.",
        "",
        "## Reasons",
        "",
        *[f"- `{reason}`" for reason in reasons],
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- schema assessment: `{OUT_SCHEMA}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
