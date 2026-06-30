#!/usr/bin/env python3
"""Build final NM-style scaling claim/failure package.

Short CPU/report task. Reads completed report CSV/JSON artifacts only. Does not
read checkpoints, canonical multi, Track C query, train, infer, or use GPU.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "scaling_nm_claim_failure_package_20260625"
OUT_MD = REPORTS / "LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md"
OUT_JSON = REPORTS / "latentfm_scaling_nm_claim_failure_package_20260625.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def top_condition_failures() -> list[dict[str, Any]]:
    rows = read_csv(REPORTS / "latentfm_condition_exposure_row_bootstrap_rows_20260625.csv")
    out = []
    for row in rows:
        if row.get("comparison") == "cap120_minus_cap30" and row.get("group") == "cross":
            diff = to_float(row.get("cross_pp_diff"))
            if diff < -0.02:
                out.append(
                    {
                        "source": "condition_exposure",
                        "axis": "condition_count_exposure",
                        "dataset": row.get("dataset"),
                        "condition": row.get("condition"),
                        "metric": "cap120_minus_cap30_cross_pp_diff",
                        "value": f"{diff:+.6f}",
                        "mmd": f"{to_float(row.get('cross_mmd_diff')):+.6f}",
                        "interpretation": "moderate exposure harms this condition; blocks monotonic exposure claim",
                    }
                )
    return sorted(out, key=lambda r: to_float(r["value"]))[:12]


def source_tail_failures() -> list[dict[str, Any]]:
    rows = read_csv(REPORTS / "scaling_completion_readiness_20260625/source_resolved_dataset_tails.csv")
    out = []
    for row in rows:
        diff = to_float(row.get("pp_delta_mean"))
        if row.get("tail_flag") == "True" or diff < -0.02:
            out.append(
                {
                    "source": "source_resolved_dataset_tails",
                    "axis": "background_type_source",
                    "dataset": row.get("dataset"),
                    "condition": f"{row.get('background')}/{row.get('perturbation_type')}",
                    "metric": "source_resolved_pp_delta_mean",
                    "value": f"{diff:+.6f}",
                    "mmd": f"{to_float(row.get('mmd_delta_mean')):+.6f}",
                    "interpretation": "source/background/type tail; supports failure-map rather than scaling law",
                }
            )
    return sorted(out, key=lambda r: to_float(r["value"]))[:12]


def metadata_failures() -> list[dict[str, Any]]:
    out = []
    qc_rows = read_csv(REPORTS / "latentfm_qc_support_reliability_rows_20260625.csv")
    for row in qc_rows:
        diff = to_float(row.get("cross_pp_diff"))
        if diff < -0.02:
            out.append(
                {
                    "source": "qc_support",
                    "axis": "training_set_metadata_qc",
                    "dataset": row.get("dataset"),
                    "condition": row.get("condition"),
                    "metric": "cross_pp_diff",
                    "value": f"{diff:+.6f}",
                    "mmd": f"{to_float(row.get('cross_mmd_diff')):+.6f}",
                    "interpretation": "QC/support reliability does not remove tail harm",
                }
            )
    jiang_rows = read_csv(REPORTS / "latentfm_jiang_guide_cytokine_context_rows_20260625.csv")
    for row in jiang_rows:
        diff = to_float(row.get("cross_pp_diff"))
        if diff < -0.02:
            out.append(
                {
                    "source": "jiang_context",
                    "axis": "guide_cytokine_mixscale",
                    "dataset": row.get("dataset"),
                    "condition": row.get("condition"),
                    "metric": "cross_pp_diff",
                    "value": f"{diff:+.6f}",
                    "mmd": f"{to_float(row.get('cross_mmd_diff')):+.6f}",
                    "interpretation": "Jiang context signal is underpowered and tail-unsafe",
                }
            )
    return sorted(out, key=lambda r: to_float(r["value"]))[:12]


def axis_claim_rows() -> list[dict[str, Any]]:
    rows = read_csv(REPORTS / "latentfm_scaling_axis_claim_matrix_20260625.csv")
    extra = [
        {
            "axis": "training_set_metadata_qc",
            "claim_level": "negative gate",
            "support": "broad obs-QC metadata covers 22 datasets/221 overlaps but fails CI/shuffle/tail",
            "boundary": "no generic QC filtering/weighted loss/hard balancing from current evidence",
            "next_gate": "new outcome-overlap artifact or preregistered external metadata",
            "manuscript_use": "supplement_or_failure_map",
            "promotion_allowed": "false",
        },
        {
            "axis": "jiang_guide_cytokine_mixscale",
            "claim_level": "supplemental mechanism hint",
            "support": "mixscale Spearman +0.829515 and high-low pp +0.547802 over only 8 overlaps",
            "boundary": "shuffle p 0.4859 and dataset tail -0.317634 block GPU/training claim",
            "next_gate": "new overlap artifact or independent Jiang validation; supplement only",
            "manuscript_use": "supplement_or_failure_map",
            "promotion_allowed": "false",
        },
    ]
    return rows + extra


def literature_rows() -> list[dict[str, Any]]:
    return [
        {
            "source": "X-Cell model card / X-Atlas-Pisces",
            "url": "https://huggingface.co/Xaira-Therapeutics/X-Cell",
            "relevant_claim": "Scaling causal perturbation prediction; LLM-class scaling laws with train-loss power law; large CRISPRi compendium across contexts",
            "impact_on_our_claim": "Do not claim absolute first perturbation-prediction scaling law; position our work as leakage-safe multi-axis audit with no-harm and failure-map controls",
        },
        {
            "source": "Ahlmann-Eltze, Huber, Anders, Nature Methods 2025",
            "url": "https://www.nature.com/articles/s41592-025-02772-6",
            "relevant_claim": "Deep-learning perturbation predictors did not outperform simple baselines in their benchmark",
            "impact_on_our_claim": "Justifies conservative benchmark framing, baseline/no-harm vetoes, and explicit negative evidence before model-promotion claims",
        },
        {
            "source": "This LatentFM scaling audit",
            "url": "/data/cyx/1030/scLatent/reports/LATENTFM_SCALING_NM_CLAIM_FAILURE_PACKAGE_20260625.md",
            "relevant_claim": "Axis-specific scaling behavior with true-cell support as strongest mechanism and multiple confounded/tail-unsafe axes",
            "impact_on_our_claim": "Claim leakage-safe cross-dataset scaling-axis audit, not deployable monotonic law",
        },
    ]


def guidance_rows() -> list[dict[str, Any]]:
    return [
        {
            "decision": "keep_default_model",
            "action": "Use `xverse_8k_anchor` as default",
            "reason": "No scaling-derived route passes canonical no-harm/model-promotion gates",
        },
        {
            "decision": "use_true_cell_guidance",
            "action": "Prefer moderate per-condition true-cell support when designing future train sets",
            "reason": "6k budget128 has strongest internal mechanism signal but still needs no-harm repair",
        },
        {
            "decision": "avoid_naive_broadening",
            "action": "Do not assume more datasets/backgrounds/types or full exposure helps",
            "reason": "source/background/type and full-vs-cap tails are unsafe/confounded",
        },
        {
            "decision": "avoid_generic_weights",
            "action": "Do not launch QC filtering, generic weighted loss, hard balancing, or Jiang-specialized training",
            "reason": "QC and Jiang metadata gates fail bootstrap/shuffle/tail safeguards",
        },
        {
            "decision": "ot_default_off",
            "action": "Keep OT minibatch-pair variants out of scaling claims unless a new pairing-quality gate passes",
            "reason": "Prior OT/no-OT/random/Hungarian evidence does not justify OT as scaling mechanism",
        },
        {
            "decision": "chemical_ack_only",
            "action": "Run chemical V2 fixed-step real Morgan512 seed43/44 only after exact ACK",
            "reason": "Current chemical technical/semantic metadata is protocol-ready but ACK-gated",
        },
    ]


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Scaling NM Claim/Failure Package",
        "",
        "Timestamp: `2026-06-25 23:26 CST`",
        "",
        "Status: `scaling_nm_claim_failure_package_ready_no_gpu`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report package from completed report artifacts.",
        "- Does not read checkpoints, canonical multi, Track C query, train, infer, or use GPU.",
        "- Canonical single/family evidence is used only as frozen no-harm veto context.",
        "",
        "## Headline Claim",
        "",
        "Scaling is a scientifically valuable axis-specific mechanism/failure-map, not a deployable monotonic law from current evidence. The default model remains `xverse_8k_anchor`.",
        "",
        "## Manuscript Claim Boundary",
        "",
        "| axis | claim level | manuscript use | promotion allowed | boundary |",
        "|---|---|---|---|---|",
    ]
    for row in payload["axis_claim_rows"]:
        lines.append(
            f"| `{row['axis']}` | {row['claim_level']} | `{row['manuscript_use']}` | "
            f"`{row['promotion_allowed']}` | {row['boundary']} |"
        )
    lines += [
        "",
        "## Top Failure Cases",
        "",
        "| source | axis | dataset | condition | metric | value | interpretation |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in payload["failure_cases"][:20]:
        lines.append(
            f"| `{row['source']}` | `{row['axis']}` | `{row['dataset']}` | `{row['condition']}` | "
            f"`{row['metric']}` | {row['value']} | {row['interpretation']} |"
        )
    lines += [
        "",
        "## Literature/Novelty Boundary",
        "",
        "| source | relevance | impact on our claim |",
        "|---|---|---|",
    ]
    for row in payload["literature_rows"]:
        lines.append(f"| [{row['source']}]({row['url']}) | {row['relevant_claim']} | {row['impact_on_our_claim']} |")
    lines += [
        "",
        "## Mainline Training Guidance",
        "",
        "| decision | action | reason |",
        "|---|---|---|",
    ]
    for row in payload["guidance_rows"]:
        lines.append(f"| `{row['decision']}` | {row['action']} | {row['reason']} |")
    lines += [
        "",
        "## Outputs",
        "",
        f"- axis claim boundary: `{payload['outputs']['axis_claim_boundary']}`",
        f"- top failure cases: `{payload['outputs']['failure_cases']}`",
        f"- literature boundary: `{payload['outputs']['literature_boundary']}`",
        f"- mainline guidance: `{payload['outputs']['mainline_guidance']}`",
        f"- JSON: `{OUT_JSON}`",
        "",
        "## Decision",
        "",
        "No GPU launch is authorized by this package. Continue with manuscript-grade figure/table completion, or launch chemical V2 only after exact ACK and fresh resource audit.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failure_cases = top_condition_failures() + source_tail_failures() + metadata_failures()
    failure_cases = sorted(failure_cases, key=lambda r: to_float(r["value"]))[:30]
    axes = axis_claim_rows()
    lit = literature_rows()
    guidance = guidance_rows()
    outputs = {
        "axis_claim_boundary": str(OUT_DIR / "axis_claim_boundary.csv"),
        "failure_cases": str(OUT_DIR / "top_failure_cases.csv"),
        "literature_boundary": str(OUT_DIR / "literature_claim_boundary.csv"),
        "mainline_guidance": str(OUT_DIR / "mainline_training_guidance.csv"),
    }
    write_csv(Path(outputs["axis_claim_boundary"]), axes, ["axis", "claim_level", "support", "boundary", "next_gate", "manuscript_use", "promotion_allowed"])
    write_csv(Path(outputs["failure_cases"]), failure_cases, ["source", "axis", "dataset", "condition", "metric", "value", "mmd", "interpretation"])
    write_csv(Path(outputs["literature_boundary"]), lit, ["source", "url", "relevant_claim", "impact_on_our_claim"])
    write_csv(Path(outputs["mainline_guidance"]), guidance, ["decision", "action", "reason"])
    payload = {
        "status": "scaling_nm_claim_failure_package_ready_no_gpu",
        "gpu_authorized": False,
        "outputs": outputs,
        "axis_claim_rows": axes,
        "failure_cases": failure_cases,
        "literature_rows": lit,
        "guidance_rows": guidance,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gpu_authorized": False, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
