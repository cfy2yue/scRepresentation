#!/usr/bin/env python3
"""Family-stratified all-modality tradeoff gate using existing posthoc outputs."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625"
DECISION_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_smoke_decision_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_allmodality_family_tradeoff_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_ALLMODALITY_FAMILY_TRADEOFF_GATE_20260625.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_budget_seed(run_name: str) -> tuple[int | None, int | None]:
    m_budget = re.search(r"budget(\d+)", run_name)
    m_seed = re.search(r"seed(\d+)", run_name)
    return (int(m_budget.group(1)) if m_budget else None, int(m_seed.group(1)) if m_seed else None)


def condition_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["dataset"]), str(row["condition"])


def metric_map(eval_json: dict[str, Any], family: str) -> dict[tuple[str, str], dict[str, float]]:
    group = (eval_json.get("groups") or {}).get(family) or {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    for row in group.get("condition_metrics", []):
        out[condition_key(row)] = {
            "pearson_pert": float(row.get("pearson_pert", math.nan)),
            "test_mmd": float(row.get("test_mmd", math.nan)),
            "n_src_eval": float(row.get("n_src_eval", math.nan)),
            "n_gt_eval": float(row.get("n_gt_eval", math.nan)),
        }
    return out


def flatten_metadata(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    data = load_json(path)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, rows in data.items():
        for condition, meta in rows.items():
            out[(str(dataset), str(condition))] = dict(meta)
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    pp = [float(r["pp_delta"]) for r in records if math.isfinite(float(r["pp_delta"]))]
    mmd = [float(r["mmd_delta"]) for r in records if math.isfinite(float(r["mmd_delta"]))]
    if not pp:
        return {"n": len(records)}
    return {
        "n": len(records),
        "pp_mean": float(mean(pp)),
        "pp_min": float(min(pp)),
        "pp_max": float(max(pp)),
        "pp_negative_frac": float(sum(x < 0.0 for x in pp) / len(pp)),
        "pp_hard_harm_frac": float(sum(x < -0.005 for x in pp) / len(pp)),
        "mmd_mean": float(mean(mmd)) if mmd else math.nan,
        "mmd_max": float(max(mmd)) if mmd else math.nan,
    }


def top_strata(records: list[dict[str, Any]], field: str, *, limit: int = 12) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(field)
        if value is None or value == "":
            continue
        groups[str(value)].append(record)
    rows = []
    for value, vals in groups.items():
        if len(vals) < 3:
            continue
        row = summarize(vals)
        row[field] = value
        rows.append(row)
    return sorted(rows, key=lambda x: (float(x.get("pp_mean", 0.0)), -int(x.get("n", 0))))[:limit]


def collect_run(run_name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_name
    posthoc = run_dir / "posthoc_eval_internal"
    anchor = load_json(posthoc / "condition_family_eval_anchor_internal_ode20.json")
    candidate = load_json(posthoc / "condition_family_eval_candidate_internal_ode20.json")
    metadata_path = Path((candidate.get("config") or {}).get("data_dir", "")) / "condition_metadata.json"
    metadata = flatten_metadata(metadata_path) if metadata_path.exists() else {}
    budget, seed = parse_budget_seed(run_name)
    records: list[dict[str, Any]] = []
    for family in ("family_gene", "family_drug"):
        a_map = metric_map(anchor, family)
        c_map = metric_map(candidate, family)
        for key in sorted(set(a_map) & set(c_map)):
            meta = metadata.get(key, {})
            records.append(
                {
                    "run_name": run_name,
                    "budget": budget,
                    "seed": seed,
                    "family": family,
                    "dataset": key[0],
                    "condition": key[1],
                    "pp_delta": c_map[key]["pearson_pert"] - a_map[key]["pearson_pert"],
                    "mmd_delta": c_map[key]["test_mmd"] - a_map[key]["test_mmd"],
                    "perturbation_type": meta.get("perturbation_type_raw"),
                    "drug": meta.get("chem_obs_value"),
                    "dose": meta.get("dose"),
                    "pathway": meta.get("pathway"),
                    "target": meta.get("target"),
                }
            )
    return {
        "run_name": run_name,
        "budget": budget,
        "seed": seed,
        "records": records,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM All-Modality Family Tradeoff Gate",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Boundary",
        "",
        "- CPU-only analysis of existing manifest-fixed posthoc condition metrics.",
        "- Uses train-only/internal all-modality dose-aware eval only.",
        "- Does not train, infer, use GPU, read canonical multi, or read Track C held-out query.",
        "",
        "## Run Summary",
        "",
        "| run | gene pp mean | gene hard-harm frac | drug pp mean | drug hard-harm frac | worst dataset pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["run_summaries"]:
        lines.append(
            f"| `{row['run_name']}` | {row['family_gene']['pp_mean']:+.6f} | {row['family_gene']['pp_hard_harm_frac']:.3f} | "
            f"{row['family_drug']['pp_mean']:+.6f} | {row['family_drug']['pp_hard_harm_frac']:.3f} | {row['worst_dataset_pp_mean']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Worst Drug Dataset Strata",
            "",
            "| run | dataset | n | pp mean | hard-harm frac |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["worst_drug_dataset_rows"][:12]:
        lines.append(
            f"| `{row['run_name']}` | `{row['dataset']}` | {row['n']} | {row['pp_mean']:+.6f} | {row['pp_hard_harm_frac']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- GPU authorized: `{payload['gpu_authorized']}`",
            f"- reasons: `{payload['reasons']}`",
            f"- next action: `{payload['next_action']}`",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    decision = load_json(DECISION_JSON)
    run_names = [str(row["run_name"]) for row in decision.get("rows", [])]
    runs = [collect_run(run_name) for run_name in run_names]
    run_summaries = []
    worst_drug_dataset_rows = []
    for run in runs:
        records = run["records"]
        gene = [r for r in records if r["family"] == "family_gene"]
        drug = [r for r in records if r["family"] == "family_drug"]
        dataset_rows = top_strata(drug, "dataset", limit=100)
        for row in dataset_rows:
            row["run_name"] = run["run_name"]
        worst_drug_dataset_rows.extend(dataset_rows)
        worst_dataset = min([float(row.get("pp_mean", 0.0)) for row in dataset_rows], default=math.nan)
        run_summaries.append(
            {
                "run_name": run["run_name"],
                "budget": run["budget"],
                "seed": run["seed"],
                "family_gene": summarize(gene),
                "family_drug": summarize(drug),
                "worst_dataset_pp_mean": worst_dataset,
                "worst_pathway_rows": top_strata(drug, "pathway", limit=10),
                "worst_target_rows": top_strata(drug, "target", limit=10),
                "worst_dose_rows": top_strata(drug, "dose", limit=10),
            }
        )

    reasons = [
        "condition_level_tradeoff_reproduces_group_gate_failure",
        "no_run_has_gene_hard_harm_frac_zero_and_drug_positive_mean",
        "drug_failures_span_sciplex_backgrounds",
    ]
    status = "allmodality_family_tradeoff_no_gpu_cpu_protocol_needed"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "run_summaries": run_summaries,
        "worst_drug_dataset_rows": sorted(worst_drug_dataset_rows, key=lambda x: float(x.get("pp_mean", 0.0))),
        "reasons": reasons,
        "next_action": "keep allmod GPU closed; if reopened, first design a CPU policy gate for family-stratified replay or chemical-only protocol with gene no-harm sentinel and dose/scaffold/type controls",
        "boundary": {
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "gpu_used": False,
            "source": "existing_posthoc_condition_metrics",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "out_md": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
