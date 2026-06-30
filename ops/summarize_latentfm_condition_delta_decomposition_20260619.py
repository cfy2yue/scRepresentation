#!/usr/bin/env python3
"""Summarize LatentFM condition-delta decomposition diagnostics."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
RUN_DIR = (
    ROOT
    / "CoupledFM/output/latentfm_runs/condition_prior_teacher_injection_20260619/"
    "scf_prior010_inject_e2_4k"
)
DECOMP_JSON = RUN_DIR / "posthoc_eval/condition_delta_decomposition_multi_best.json"
DECOMP_CSV = RUN_DIR / "posthoc_eval/condition_delta_decomposition_multi_best.csv"
OUT_MD = ROOT / "reports/LATENTFM_CONDITION_DELTA_DECOMPOSITION_20260619.md"
OUT_JSON = ROOT / "reports/latentfm_condition_delta_decomposition_20260619.json"


def fmt(value: Any) -> str:
    if isinstance(value, (int, float, np.floating)):
        return f"{float(value):.4f}"
    return "NA"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out: dict[str, Any] = {}
            for k, v in row.items():
                if v is None:
                    out[k] = v
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            rows.append(out)
    return rows


def _top(rows: list[dict[str, Any]], key: str, *, reverse: bool, n: int = 10) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: float(r.get(key, 0.0)), reverse=reverse)[:n]


def main() -> int:
    payload = json.loads(DECOMP_JSON.read_text(encoding="utf-8"))
    rows = _read_rows(DECOMP_CSV)
    summary = payload.get("summary", [])
    norman_wessels = [
        row for row in summary
        if row.get("dataset") in {"NormanWeissman2019_filtered", "Wessels"}
    ]
    combo_additive = [
        float(row.get("combo_additive_cosine", 0.0))
        for row in rows
        if str(row.get("dataset")) in {"NormanWeissman2019_filtered", "Wessels"}
    ]
    interaction_ratio = [
        float(row.get("interaction_norm_ratio", 0.0))
        for row in rows
        if str(row.get("dataset")) in {"NormanWeissman2019_filtered", "Wessels"}
    ]
    result = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "checkpoint": payload.get("checkpoint"),
        "rows": payload.get("selected_conditions"),
        "used_ema": payload.get("used_ema"),
        "device": payload.get("device"),
        "mean_combo_additive_cosine_norman_wessels": float(np.mean(combo_additive)) if combo_additive else None,
        "mean_interaction_norm_ratio_norman_wessels": float(np.mean(interaction_ratio)) if interaction_ratio else None,
        "summary": summary,
        "top_interaction_endpoint_positive": _top(rows, "interaction_endpoint_cosine", reverse=True, n=10),
        "top_interaction_endpoint_negative": _top(rows, "interaction_endpoint_cosine", reverse=False, n=10),
        "top_combo_additive_low_cosine": _top(rows, "combo_additive_cosine", reverse=False, n=10),
        "csv": str(DECOMP_CSV),
        "json": str(DECOMP_JSON),
        "report": str(OUT_MD),
    }

    lines = [
        "# LatentFM Condition-Delta Decomposition 2026-06-19",
        "",
        f"Generated: {result['generated']}",
        "",
        "## Status",
        "",
        "`complete`",
        "",
        "This is a head-only diagnostic for `scf_prior010_inject_e2_4k`. It loads the best checkpoint with EMA, evaluates condition-delta head predictions, and does not run ODE integration or training.",
        "",
        "## Inputs",
        "",
        f"- Checkpoint: `{payload.get('checkpoint')}`",
        f"- Rows: `{payload.get('selected_conditions')}`",
        f"- Device: `{payload.get('device')}`",
        f"- Used EMA: `{payload.get('used_ema')}`",
        f"- CSV: `{DECOMP_CSV}`",
        f"- JSON: `{DECOMP_JSON}`",
        "",
        "## Group Summary",
        "",
        "| Dataset | Group | n | combo endpoint cos | additive endpoint cos | interaction endpoint cos | combo-additive cos | additive norm / combo | interaction norm / combo |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in norman_wessels:
        lines.append(
            f"| `{row.get('dataset')}` | `{row.get('group')}` | {row.get('n')} | "
            f"{fmt(row.get('mean_combo_endpoint_cosine'))} | "
            f"{fmt(row.get('mean_additive_endpoint_cosine'))} | "
            f"{fmt(row.get('mean_interaction_endpoint_cosine'))} | "
            f"{fmt(row.get('mean_combo_additive_cosine'))} | "
            f"{fmt(row.get('mean_additive_norm_ratio'))} | "
            f"{fmt(row.get('mean_interaction_norm_ratio'))} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Mean combo-additive cosine over Norman/Wessels multi rows is `{fmt(result['mean_combo_additive_cosine_norman_wessels'])}`.",
        f"- Mean interaction-norm/combo-norm ratio over Norman/Wessels multi rows is `{fmt(result['mean_interaction_norm_ratio_norman_wessels'])}`.",
        "- The combo and additive head predictions are almost colinear, while the additive prediction has about twice the combo norm. Therefore `combo - additive` is close to `-combo`, with nearly the same norm as the combo prediction. This means the current injected head is not yet a clean additive atom model.",
        "- Endpoint alignment remains small in absolute value. Norman groups have negative combo/additive endpoint cosine but positive residual endpoint cosine; Wessels shows the opposite sign pattern. This mirrors the previous dataset-specific tradeoff rather than solving Wessels interaction biology.",
        "- Therefore the new decomposition surface is useful for diagnosis, but the current checkpoint should not be promoted as an additive-plus-interaction architecture.",
        "",
        "## Top Positive Interaction Endpoint Cosine",
        "",
        "| Dataset | Condition | Group | interaction endpoint cos | combo endpoint cos | additive endpoint cos | combo-additive cos |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in result["top_interaction_endpoint_positive"]:
        lines.append(
            f"| `{row.get('dataset')}` | `{row.get('condition')}` | `{row.get('groups')}` | "
            f"{fmt(row.get('interaction_endpoint_cosine'))} | {fmt(row.get('combo_endpoint_cosine'))} | "
            f"{fmt(row.get('additive_endpoint_cosine'))} | {fmt(row.get('combo_additive_cosine'))} |"
        )
    lines.extend([
        "",
        "## Top Negative Interaction Endpoint Cosine",
        "",
        "| Dataset | Condition | Group | interaction endpoint cos | combo endpoint cos | additive endpoint cos | combo-additive cos |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in result["top_interaction_endpoint_negative"]:
        lines.append(
            f"| `{row.get('dataset')}` | `{row.get('condition')}` | `{row.get('groups')}` | "
            f"{fmt(row.get('interaction_endpoint_cosine'))} | {fmt(row.get('combo_endpoint_cosine'))} | "
            f"{fmt(row.get('additive_endpoint_cosine'))} | {fmt(row.get('combo_additive_cosine'))} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "`diagnostic_only_no_gpu_followup_from_this_checkpoint`",
        "",
        "Next code direction should train or regularize the additive atom surface explicitly from train-single priors before using the residual as an interaction hypothesis. Under the current zero-shot split, interaction residuals must remain diagnostic unless a future non-zero-shot split with train multi supervision is declared.",
        "",
    ])

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
