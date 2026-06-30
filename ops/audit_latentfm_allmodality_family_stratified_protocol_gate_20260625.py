#!/usr/bin/env python3
"""CPU gate for a family-stratified all-modality chemical protocol.

This gate tests whether the closed all-modality dose-aware smokes contain a
predeclared, control-robust chemical-only/family-stratified signal that could
justify a new GPU protocol.  It uses existing train-only/internal posthoc
condition metrics only.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "runs/latentfm_true_cell_count_allmodality_doseaware_smokes_20260625"
DECISION_JSON = ROOT / "reports/latentfm_true_cell_count_allmodality_doseaware_smoke_decision_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_allmodality_family_stratified_protocol_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_ALLMODALITY_FAMILY_STRATIFIED_PROTOCOL_GATE_20260625.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_budget_seed(run_name: str) -> tuple[int | None, int | None]:
    m_budget = re.search(r"budget(\d+)", run_name)
    m_seed = re.search(r"seed(\d+)", run_name)
    return (int(m_budget.group(1)) if m_budget else None, int(m_seed.group(1)) if m_seed else None)


def flatten_metadata(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    data = load_json(path)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, rows in data.items():
        for condition, meta in rows.items():
            out[(str(dataset), str(condition))] = dict(meta)
    return out


def group_rows(payload: dict[str, Any], family: str) -> dict[tuple[str, str], dict[str, Any]]:
    group = (payload.get("groups") or {}).get(family) or {}
    return {(str(r["dataset"]), str(r["condition"])): r for r in group.get("condition_metrics", [])}


def collect_run(run_name: str) -> dict[str, Any]:
    posthoc = RUN_ROOT / run_name / "posthoc_eval_internal"
    anchor = load_json(posthoc / "condition_family_eval_anchor_internal_ode20.json")
    candidate = load_json(posthoc / "condition_family_eval_candidate_internal_ode20.json")
    metadata_path = Path((candidate.get("config") or {}).get("data_dir", "")) / "condition_metadata.json"
    metadata = flatten_metadata(metadata_path) if metadata_path.exists() else {}
    budget, seed = parse_budget_seed(run_name)
    drug_records = []
    gene_records = []
    for family, target in (("family_drug", drug_records), ("family_gene", gene_records)):
        a_rows = group_rows(anchor, family)
        c_rows = group_rows(candidate, family)
        for key in sorted(set(a_rows) & set(c_rows)):
            a = a_rows[key]
            c = c_rows[key]
            meta = metadata.get(key, {})
            pp_delta = float(c["pearson_pert"]) - float(a["pearson_pert"])
            mmd_delta = float(c["test_mmd"]) - float(a["test_mmd"])
            target.append(
                {
                    "dataset": key[0],
                    "condition": key[1],
                    "pp_delta": pp_delta,
                    "mmd_delta": mmd_delta,
                    "dose": str(meta.get("dose") or ""),
                    "pathway": str(meta.get("pathway") or ""),
                    "target": str(meta.get("target") or ""),
                }
            )
    return {
        "run_name": run_name,
        "budget": budget,
        "seed": seed,
        "drug_records": drug_records,
        "gene_records": gene_records,
    }


def mean(xs: list[float]) -> float | None:
    return None if not xs else float(statistics.mean(xs))


def summarize_selected(selected: list[dict[str, Any]]) -> dict[str, Any]:
    pp = [float(r["pp_delta"]) for r in selected if math.isfinite(float(r["pp_delta"]))]
    mmd = [float(r["mmd_delta"]) for r in selected if math.isfinite(float(r["mmd_delta"]))]
    by_ds: dict[str, list[float]] = defaultdict(list)
    for r in selected:
        by_ds[str(r["dataset"])].append(float(r["pp_delta"]))
    ds_rows = [{"dataset": ds, "n": len(vals), "pp_mean": float(statistics.mean(vals))} for ds, vals in sorted(by_ds.items())]
    return {
        "n": len(selected),
        "n_datasets": len(by_ds),
        "pp_mean": mean(pp),
        "pp_median": None if not pp else float(statistics.median(pp)),
        "pp_min": None if not pp else float(min(pp)),
        "pp_hard_harm_frac_lt_minus_0p005": None if not pp else float(sum(x < -0.005 for x in pp) / len(pp)),
        "mmd_mean": mean(mmd),
        "mmd_max": None if not mmd else float(max(mmd)),
        "dataset_rows": ds_rows,
        "dataset_min_pp_mean": None if not ds_rows else float(min(r["pp_mean"] for r in ds_rows)),
    }


def policy_rows(records: list[dict[str, Any]], policy: dict[str, str]) -> list[dict[str, Any]]:
    kind = policy["kind"]
    if kind == "all_drug":
        return list(records)
    if kind == "background":
        return [r for r in records if r["dataset"] == policy["value"]]
    if kind == "dose":
        return [r for r in records if r["dose"] == policy["value"]]
    if kind == "pathway":
        return [r for r in records if r["pathway"] == policy["value"]]
    raise ValueError(f"unknown policy kind: {kind}")


def policy_label(policy: dict[str, str]) -> str:
    return f"{policy['kind']}={policy['value']}"


def shuffle_control(records: list[dict[str, Any]], n_select: int, actual_pp: float, *, n: int = 1000) -> dict[str, Any]:
    if n_select <= 0 or not records:
        return {"n": 0}
    rng = random.Random(20260625)
    vals = []
    hard = []
    for _ in range(n):
        selected = rng.sample(records, min(n_select, len(records)))
        summary = summarize_selected(selected)
        vals.append(float(summary["pp_mean"] or 0.0))
        hard.append(float(summary["pp_hard_harm_frac_lt_minus_0p005"] or 0.0))
    vals.sort()
    hard.sort()
    return {
        "n": n,
        "pp_mean_mean": float(statistics.mean(vals)),
        "pp_mean_p95": vals[int(0.95 * (len(vals) - 1))],
        "p_ge_actual": float(sum(v >= actual_pp for v in vals) / len(vals)),
        "hard_harm_frac_p05": hard[int(0.05 * (len(hard) - 1))],
        "hard_harm_frac_mean": float(statistics.mean(hard)),
    }


def candidate_policies(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    policies = [{"kind": "all_drug", "value": "all"}]
    for ds in sorted({r["dataset"] for r in records}):
        policies.append({"kind": "background", "value": ds})
    for dose in ("1.0", "0.1", "0.01"):
        policies.append({"kind": "dose", "value": dose})
    # Pathway policies are metadata-defined but many; keep only adequately sized
    # pathways and require a stricter control margin downstream.
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        if r["pathway"]:
            counts[r["pathway"]] += 1
    for pathway, count in sorted(counts.items()):
        if count >= 20:
            policies.append({"kind": "pathway", "value": pathway})
    return policies


def evaluate_policy(run: dict[str, Any], policy: dict[str, str]) -> dict[str, Any]:
    selected = policy_rows(run["drug_records"], policy)
    summary = summarize_selected(selected)
    control = shuffle_control(run["drug_records"], int(summary["n"]), float(summary["pp_mean"] or 0.0))
    gene_summary = summarize_selected(run["gene_records"])
    reasons = []
    if summary["n"] < 30:
        reasons.append("selected_n_lt_30")
    if policy["kind"] in {"all_drug", "dose", "pathway"} and summary["n_datasets"] < 2:
        reasons.append("selected_backgrounds_lt_2")
    if summary["pp_mean"] is None or summary["pp_mean"] < 0.020:
        reasons.append("drug_pp_mean_lt_0p020")
    if summary["dataset_min_pp_mean"] is None or summary["dataset_min_pp_mean"] < 0.005:
        reasons.append("dataset_min_pp_lt_0p005")
    if summary["pp_hard_harm_frac_lt_minus_0p005"] is None or summary["pp_hard_harm_frac_lt_minus_0p005"] > 0.25:
        reasons.append("drug_hard_harm_frac_gt_0p25")
    if summary["mmd_mean"] is None or summary["mmd_mean"] > 0.0005:
        reasons.append("mmd_mean_gt_0p0005")
    if summary["mmd_max"] is None or summary["mmd_max"] > 0.002:
        reasons.append("mmd_max_gt_0p002")
    if control.get("p_ge_actual") is None or float(control.get("p_ge_actual", 1.0)) > 0.05:
        reasons.append("count_matched_shuffle_not_collapsed")
    if summary["pp_mean"] is None or control.get("pp_mean_p95") is None or summary["pp_mean"] <= float(control["pp_mean_p95"]):
        reasons.append("drug_pp_not_above_shuffle_p95")
    if policy["kind"] == "pathway" and (control.get("p_ge_actual") is None or float(control["p_ge_actual"]) > 0.01):
        reasons.append("pathway_policy_requires_stricter_shuffle_p_le_0p01")
    if gene_summary["pp_mean"] is None or gene_summary["pp_mean"] < -0.002:
        reasons.append("gene_sentinel_pp_lt_minus_0p002")
    return {
        "run_name": run["run_name"],
        "budget": run["budget"],
        "seed": run["seed"],
        "policy": policy,
        "policy_label": policy_label(policy),
        "summary": summary,
        "gene_sentinel": gene_summary,
        "shuffle_control": control,
        "passes": not reasons,
        "reasons": reasons,
    }


def main() -> int:
    decision = load_json(DECISION_JSON)
    run_names = [str(r["run_name"]) for r in decision.get("rows", [])]
    runs = [collect_run(name) for name in run_names]
    rows = []
    for run in runs:
        for policy in candidate_policies(run["drug_records"]):
            rows.append(evaluate_policy(run, policy))
    rows.sort(
        key=lambda r: (
            bool(r["passes"]),
            float(r["summary"]["pp_mean"] or -999.0),
            -len(r["reasons"]),
        ),
        reverse=True,
    )
    passing = [r for r in rows if r["passes"]]
    status = (
        "allmodality_family_stratified_protocol_pass_gpu_candidate"
        if passing
        else "allmodality_family_stratified_protocol_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": bool(passing),
        "passing_count": len(passing),
        "top_rows": rows[:20],
        "all_rows_count": len(rows),
        "boundary": {
            "source": "existing allmodality dose-aware train-only/internal posthoc condition metrics",
            "canonical_multi_used": False,
            "trackc_query_used": False,
            "gpu_used": False,
            "canonical_or_v2_route_selection": False,
        },
        "decision": {
            "next_action": (
                "prepare one bounded family-stratified GPU smoke after external review"
                if passing
                else "keep allmod direct/family-stratified GPU closed; use as failure-map evidence"
            ),
            "stop_rule": "close if best policy remains near count-matched shuffle, has hard-harm >25%, dataset min <+0.005, or gene sentinel harm",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM All-Modality Family-Stratified Protocol Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{bool(passing)}`",
        "",
        "## Boundary",
        "",
        "- CPU-only gate using existing manifest-fixed all-modality dose-aware train-only/internal posthoc condition metrics.",
        "- No training, inference, GPU, canonical multi, Track C query, or V2 outcome is used.",
        "- Policies are predeclared from family/background/dose/pathway metadata; pathway policies require stricter shuffle support.",
        "",
        "## Top Policies",
        "",
        "| run | policy | n | backgrounds | pp mean | dataset min | hard-harm frac | MMD mean | shuffle p95 | p(shuffle>=actual) | pass | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows[:20]:
        s = row["summary"]
        c = row["shuffle_control"]
        lines.append(
            "| `{run}` | `{policy}` | {n} | {nds} | {pp:+.6f} | {mn:+.6f} | {hh:.3f} | {mmd:+.6f} | {p95:+.6f} | {pge:.3f} | `{passes}` | {reasons} |".format(
                run=row["run_name"],
                policy=row["policy_label"],
                n=s.get("n", 0),
                nds=s.get("n_datasets", 0),
                pp=float(s.get("pp_mean") or 0.0),
                mn=float(s.get("dataset_min_pp_mean") or 0.0),
                hh=float(s.get("pp_hard_harm_frac_lt_minus_0p005") or 0.0),
                mmd=float(s.get("mmd_mean") or 0.0),
                p95=float(c.get("pp_mean_p95") or 0.0),
                pge=float(c.get("p_ge_actual") if c.get("p_ge_actual") is not None else 1.0),
                passes=row["passes"],
                reasons=", ".join(row["reasons"]) or "none",
            )
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- passing policies: `{len(passing)}`",
        f"- next action: `{payload['decision']['next_action']}`",
        "- A positive mean alone is insufficient; the gate requires low hard-harm, dataset-tail safety, MMD safety, and count-matched shuffle collapse.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
