#!/usr/bin/env python3
"""Summarize general exposure-cap v2 internal failure cases.

Read-only posthoc diagnostic. It uses only internal train-selection posthoc
artifacts and does not read canonical or Track C query outputs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_DIR = (
    ROOT
    / "runs/latentfm_xverse_scaling_count_smokes_20260624"
    / "xverse_scaling_general_exposure_cap_v2_3k_seed42"
)
EVAL_DIR = RUN_DIR / "posthoc_eval_internal"
OUT_JSON = ROOT / "reports/latentfm_xverse_general_exposure_failure_cases_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_XVERSE_GENERAL_EXPOSURE_FAILURE_CASES_20260624.md"


def _f(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def load_group(name: str, group: str) -> dict[str, Any]:
    path = EVAL_DIR / name
    obj = json.loads(path.read_text(encoding="utf-8"))
    return (obj.get("groups") or {})[group]


def condition_deltas(anchor: dict[str, Any], cand: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = {
        (str(row["dataset"]), str(row["condition"])): row
        for row in anchor.get("condition_metrics", [])
    }
    rows = []
    for c in cand.get("condition_metrics", []):
        key = (str(c["dataset"]), str(c["condition"]))
        a = by_key.get(key)
        if not a:
            continue
        row = {
            "dataset": key[0],
            "condition": key[1],
            "anchor_mmd": _f(a.get("test_mmd")),
            "candidate_mmd": _f(c.get("test_mmd")),
            "anchor_pp": _f(a.get("pearson_pert")),
            "candidate_pp": _f(c.get("pearson_pert")),
            "n_gt_eval": c.get("n_gt_eval"),
        }
        row["delta_mmd"] = None if row["anchor_mmd"] is None or row["candidate_mmd"] is None else row["candidate_mmd"] - row["anchor_mmd"]
        row["delta_pp"] = None if row["anchor_pp"] is None or row["candidate_pp"] is None else row["candidate_pp"] - row["anchor_pp"]
        rows.append(row)
    return rows


def per_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    out = []
    for ds, ds_rows in sorted(by_ds.items()):
        valid_mmd = [r["delta_mmd"] for r in ds_rows if r["delta_mmd"] is not None]
        valid_pp = [r["delta_pp"] for r in ds_rows if r["delta_pp"] is not None]
        out.append(
            {
                "dataset": ds,
                "n": len(ds_rows),
                "delta_mmd_mean": sum(valid_mmd) / len(valid_mmd) if valid_mmd else None,
                "delta_pp_mean": sum(valid_pp) / len(valid_pp) if valid_pp else None,
                "n_mmd_harm": sum(1 for v in valid_mmd if v > 0),
                "n_pp_harm": sum(1 for v in valid_pp if v < 0),
            }
        )
    return sorted(out, key=lambda r: (r["delta_mmd_mean"] is None, -(r["delta_mmd_mean"] or float("-inf"))))


def _fmt(x: Any, digits: int = 6) -> str:
    val = _f(x)
    if val is None:
        return "NA"
    return f"{val:+.{digits}f}"


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# LatentFM xverse General Exposure-Cap v2 Failure Cases",
        "",
        "## Boundary",
        "",
        "- Read-only diagnostic over completed internal posthoc artifacts.",
        "- Does not read canonical split, Track C query, active logs, or held-out query artifacts.",
        "- Purpose: explain why the branch failed despite improving cross-background/family Pearson.",
        "",
        "## Summary",
        "",
        f"- decision status: `{payload['decision_status']}`",
        f"- family_gene MMD delta vs anchor: `{_fmt(payload['family_gene_delta_mmd'])}`",
        f"- family_gene Pearson perturbation delta vs anchor: `{_fmt(payload['family_gene_delta_pp'])}`",
        f"- cross-background Pearson perturbation delta vs anchor: `{_fmt(payload['cross_background_delta_pp'])}`",
        "",
        "## Worst Dataset MMD Deltas",
        "",
        "| dataset | n | mean delta MMD | mean delta pp | MMD-harm rows | pp-harm rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_dataset"][:12]:
        lines.append(
            f"| {row['dataset']} | {row['n']} | {_fmt(row['delta_mmd_mean'])} | "
            f"{_fmt(row['delta_pp_mean'])} | {row['n_mmd_harm']} | {row['n_pp_harm']} |"
        )
    lines.extend([
        "",
        "## Worst Condition MMD Deltas",
        "",
        "| dataset | condition | delta MMD | delta pp | candidate MMD | anchor MMD |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in payload["worst_conditions"][:20]:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {_fmt(row['delta_mmd'])} | "
            f"{_fmt(row['delta_pp'])} | {_fmt(row['candidate_mmd'])} | {_fmt(row['anchor_mmd'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- General exposure capping recovers Pearson signal relative to Jiang but creates a family-gene distributional harm.",
        "- This branch should not proceed to canonical no-harm unless a new mechanism explicitly addresses the MMD failure.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    decision = json.loads((ROOT / "reports/latentfm_xverse_scaling_count_smokes_decision_20260624.json").read_text(encoding="utf-8"))
    anchor_family = load_group("condition_family_eval_anchor_internal_ode20.json", "family_gene")
    cand_family = load_group("condition_family_eval_candidate_internal_ode20.json", "family_gene")
    anchor_split = load_group("split_group_eval_anchor_internal_ode20.json", "internal_val_cross_background_seen_gene_proxy")
    cand_split = load_group("split_group_eval_candidate_internal_ode20.json", "internal_val_cross_background_seen_gene_proxy")
    rows = condition_deltas(anchor_family, cand_family)
    payload = {
        "boundary": {
            "run_dir": str(RUN_DIR),
            "no_canonical_or_query": True,
            "read_only": True,
        },
        "decision_status": (decision.get("general_exposure_extension_decision") or {}).get("status"),
        "decision_reasons": (decision.get("general_exposure_extension_decision") or {}).get("reasons", []),
        "family_gene_delta_mmd": _f(cand_family.get("test_mmd")) - _f(anchor_family.get("test_mmd")),
        "family_gene_delta_pp": _f(cand_family.get("pearson_pert")) - _f(anchor_family.get("pearson_pert")),
        "cross_background_delta_pp": _f(cand_split.get("pearson_pert")) - _f(anchor_split.get("pearson_pert")),
        "per_dataset": per_dataset(rows),
        "worst_conditions": sorted(
            rows,
            key=lambda r: (r["delta_mmd"] is None, -(r["delta_mmd"] or float("-inf"))),
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_md(payload)
    print(json.dumps({"out_md": str(OUT_MD), "out_json": str(OUT_JSON), "status": payload["decision_status"]}, indent=2))


if __name__ == "__main__":
    main()
