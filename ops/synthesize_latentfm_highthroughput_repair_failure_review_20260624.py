#!/usr/bin/env python3
"""Failure review for latest high-throughput LatentFM repair branches."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_JSON = ROOT / "reports/latentfm_highthroughput_repair_failure_review_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_HIGHTHROUGHPUT_REPAIR_FAILURE_REVIEW_20260624.md"

RESPONSE_CANON_ROOT = ROOT / "runs/latentfm_scaling_cap60_response_canonical_noharm_20260624"
GENERAL_ROOT = ROOT / "runs/latentfm_general_exposure_mmdguard_repair_20260624"
GENERAL_RUN = "xverse_general_exposure_mmdguard_replay05_mmd05_3k_seed42"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(payload: dict[str, Any], group: str) -> dict[tuple[str, str], dict[str, Any]]:
    group_obj = (payload.get("groups") or {}).get(group) or {}
    out = {}
    for row in group_obj.get("condition_metrics") or []:
        if isinstance(row, dict) and row.get("dataset") and row.get("condition"):
            out[(str(row["dataset"]), str(row["condition"]))] = row
    return out


def fnum(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def paired_condition_deltas(anchor: dict[str, Any], cand: dict[str, Any], group: str) -> list[dict[str, Any]]:
    arows = rows(anchor, group)
    crows = rows(cand, group)
    out = []
    for key in sorted(set(arows) & set(crows)):
        ds, cond = key
        av, cv = arows[key], crows[key]
        item = {"dataset": ds, "condition": cond}
        for metric in ("pearson_pert", "pearson_ctrl", "test_mmd", "test_mmd_clamped"):
            a = fnum(av.get(metric))
            c = fnum(cv.get(metric))
            item[f"{metric}_anchor"] = a
            item[f"{metric}_candidate"] = c
            item[f"{metric}_delta"] = None if a is None or c is None else c - a
        out.append(item)
    return out


def by_dataset_mean(items: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    tmp: dict[str, list[float]] = defaultdict(list)
    for item in items:
        value = item.get(metric)
        if value is not None:
            tmp[item["dataset"]].append(float(value))
    rows_out = []
    for ds, vals in sorted(tmp.items()):
        rows_out.append({"dataset": ds, "n": len(vals), "mean": sum(vals) / len(vals)})
    return rows_out


def top_items(items: list[dict[str, Any]], key: str, *, reverse: bool, n: int = 8) -> list[dict[str, Any]]:
    valid = [item for item in items if item.get(key) is not None]
    valid.sort(key=lambda item: float(item[key]), reverse=reverse)
    return valid[:n]


def response_reviews() -> list[dict[str, Any]]:
    reviews = []
    for gate_path in sorted(RESPONSE_CANON_ROOT.glob("*/posthoc_eval_canonical/single_background_candidate_gate.json")):
        run = gate_path.parts[-3]
        gate = load_json(gate_path)
        anchor_family = load_json(Path(gate["anchor_family_json"]))
        cand_family = load_json(Path(gate["candidate_family_json"]))
        anchor_split = load_json(Path(gate["anchor_split_json"]))
        cand_split = load_json(Path(gate["candidate_split_json"]))
        family = paired_condition_deltas(anchor_family, cand_family, "family_gene")
        single = paired_condition_deltas(anchor_split, cand_split, "test_single")
        reviews.append(
            {
                "run": run,
                "gate": gate.get("gate"),
                "family_gene_dataset_pp": by_dataset_mean(family, "pearson_pert_delta"),
                "family_gene_dataset_mmd": by_dataset_mean(family, "test_mmd_clamped_delta"),
                "test_single_dataset_pp": by_dataset_mean(single, "pearson_pert_delta"),
                "top_family_pp_harm": top_items(family, "pearson_pert_delta", reverse=False),
                "top_family_mmd_harm": top_items(family, "test_mmd_clamped_delta", reverse=True),
                "top_single_pp_harm": top_items(single, "pearson_pert_delta", reverse=False),
            }
        )
    return reviews


def general_review() -> dict[str, Any]:
    eval_dir = GENERAL_ROOT / GENERAL_RUN / "posthoc_eval_internal"
    anchor_family = load_json(eval_dir / "condition_family_eval_anchor_internal_ode20.json")
    cand_family = load_json(eval_dir / "condition_family_eval_candidate_internal_ode20.json")
    family = paired_condition_deltas(anchor_family, cand_family, "family_gene")
    return {
        "run": GENERAL_RUN,
        "family_gene_dataset_pp": by_dataset_mean(family, "pearson_pert_delta"),
        "family_gene_dataset_mmd": by_dataset_mean(family, "test_mmd_clamped_delta"),
        "top_family_pp_harm": top_items(family, "pearson_pert_delta", reverse=False),
        "top_family_mmd_harm": top_items(family, "test_mmd_clamped_delta", reverse=True),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def short_condition_table(items: list[dict[str, Any]], metric: str) -> list[str]:
    lines = []
    for item in items[:6]:
        lines.append(f"- `{item['dataset']} / {item['condition']}`: {metric} {fmt(item.get(metric))}")
    return lines


def main() -> int:
    response = response_reviews()
    general = general_review()
    payload = {
        "status": "failure_review_ready",
        "boundary": {
            "canonical_multi_read": False,
            "trackc_query_read": False,
            "new_gpu_launched": False,
        },
        "response_canonical": response,
        "general_exposure_internal": general,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LatentFM High-Throughput Repair Failure Review",
        "",
        "Status: `failure_review_ready`",
        "",
        "## Boundary",
        "",
        "- Reads only completed response canonical single/family posthoc and completed general-exposure internal posthoc.",
        "- Does not read canonical multi or Track C held-out query.",
        "- Does not launch GPU work.",
        "",
        "## Response Canonical No-Harm Failure",
        "",
        "Both response-normalized cap60 arms had positive train-only internal signal but failed frozen canonical no-harm. The failure is not a borderline bootstrap artifact: cross-background canonical pp is negative and all/family pp harm probability is high.",
    ]
    for review in response:
        gate = review.get("gate") or {}
        lines.extend(["", f"### `{review['run']}`", "", f"- gate: `{gate.get('status')}`", f"- reasons: `{gate.get('reasons')}`", "", "Largest family-gene pp harms:"])
        lines.extend(short_condition_table(review["top_family_pp_harm"], "pearson_pert_delta"))
        lines.extend(["", "Largest family-gene MMD harms:"])
        lines.extend(short_condition_table(review["top_family_mmd_harm"], "test_mmd_clamped_delta"))
        lines.extend(["", "Largest test-single pp harms:"])
        lines.extend(short_condition_table(review["top_single_pp_harm"], "pearson_pert_delta"))

    lines.extend(
        [
            "",
            "## General Exposure MMD-Guard Failure",
            "",
            "The general-exposure repair retained Pearson signal but failed because MMD harm remained concentrated and large.",
            "",
            "Largest family-gene pp harms:",
        ]
    )
    lines.extend(short_condition_table(general["top_family_pp_harm"], "pearson_pert_delta"))
    lines.extend(["", "Largest family-gene MMD harms:"])
    lines.extend(short_condition_table(general["top_family_mmd_harm"], "test_mmd_clamped_delta"))
    lines.extend(
        [
            "",
            "## Decision Use",
            "",
            "- Close response-normalized cap60 seed42 arms for promotion unless a later independent seed/internal result motivates a new predeclared repair.",
            "- Close the tested general-exposure MMD-guard primary arm; do not mutate it without new evidence that targets the observed worst-dataset MMD tails.",
            "- Treat future branches as needing explicit tail-risk controls, not only aggregate Pearson gains.",
            "",
            "## Output",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "failure_review_ready", "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
