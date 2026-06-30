#!/usr/bin/env python3
"""Synthesize train-set strategy / no-harm admission queue.

This is a CPU-only report over completed artifacts. It asks which training-data
or scaling-law ideas are ready for GPU smokes now, which require a CPU gate,
and which should stay as biology/descriptive evidence. It does not train,
infer, select checkpoints, use canonical multi for Track A selection, or read
held-out Track C query data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_DIR = REPORTS / "trainset_strategy_queue_20260630"

INPUTS = {
    "run_outcome_rows": REPORTS
    / "latentfm_run_outcome_mechanism_panel_20260630"
    / "latentfm_run_outcome_mechanism_rows_20260630.csv",
    "run_outcome_json": REPORTS
    / "latentfm_run_outcome_mechanism_panel_20260630"
    / "latentfm_run_outcome_mechanism_panel_20260630.json",
    "condition_count_rows": REPORTS
    / "condition_count_scaling_rescue_gate_20260630"
    / "condition_count_scaling_rescue_rows_20260630.csv",
    "condition_count_json": REPORTS
    / "condition_count_scaling_rescue_gate_20260630"
    / "condition_count_scaling_rescue_gate_20260630.json",
    "observable_budget_json": REPORTS
    / "observable_gene_budget_scaling_law_gate_20260630"
    / "latentfm_observable_gene_budget_scaling_law_gate_20260630.json",
    "response_compressibility_json": REPORTS
    / "response_compressibility_pairability_gate_20260630"
    / "response_compressibility_pairability_gate_20260630.json",
    "zscape_neighborhood_json": REPORTS
    / "zscape_condition_response_neighborhood_gate_20260630"
    / "latentfm_zscape_condition_response_neighborhood_gate_20260630.json",
    "zscape_structural_json": REPORTS
    / "zscape_structural_dynamic_scaling_x_20260630"
    / "zscape_structural_dynamic_scaling_x_20260630.json",
    "zscape_atlas_json": REPORTS
    / "zscape_dynamic_pairability_atlas_20260630"
    / "zscape_dynamic_pairability_atlas_20260630.json",
    "zscape_expansion_gate_json": REPORTS
    / "zscape_pairability_strict_control_expansion_gate_20260630"
    / "zscape_pairability_strict_control_expansion_gate_20260630.json",
    "ot_pairing_json": REPORTS / "latentfm_ot_pairing_gate_20260630" / "latentfm_ot_pairing_gate_20260630.json",
}


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(obj: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def build_outcome_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "rows": 0,
            "candidate_nonduplicates": 0,
            "weak_nonclosed": 0,
            "nonclosed_positive_no_mmd": 0,
            "best_nonclosed_cross_pp": None,
            "best_nonclosed_family_pp": None,
            "best_nonclosed_record": "",
        }
    rows = rows.copy()
    for col in ["candidate_nonduplicate", "closed_family_prior", "mmd_harm", "positive_pp", "weak_positive_pp"]:
        if col in rows.columns:
            rows[col] = rows[col].map(truthy)
    rows["cross_pp_delta"] = pd.to_numeric(rows.get("cross_pp_delta"), errors="coerce")
    rows["family_pp_delta"] = pd.to_numeric(rows.get("family_pp_delta"), errors="coerce")
    nonclosed = rows[~rows.get("closed_family_prior", False)]
    positive_no_mmd = nonclosed[
        (nonclosed.get("positive_pp", False)) & (~nonclosed.get("mmd_harm", False))
    ]
    weak_nonclosed = nonclosed[nonclosed.get("weak_positive_pp", False)]
    rank = nonclosed.sort_values(["cross_pp_delta", "family_pp_delta"], ascending=False)
    best = rank.iloc[0] if not rank.empty else None
    return {
        "rows": int(len(rows)),
        "candidate_nonduplicates": int(rows.get("candidate_nonduplicate", pd.Series(dtype=bool)).sum()),
        "weak_nonclosed": int(len(weak_nonclosed)),
        "nonclosed_positive_no_mmd": int(len(positive_no_mmd)),
        "best_nonclosed_cross_pp": finite_float(best.get("cross_pp_delta")) if best is not None else None,
        "best_nonclosed_family_pp": finite_float(best.get("family_pp_delta")) if best is not None else None,
        "best_nonclosed_record": str(best.get("record_name")) if best is not None else "",
    }


def build_count_summary(rows: pd.DataFrame, decision: dict[str, Any]) -> dict[str, Any]:
    if rows.empty:
        return {"rows": 0}
    rows = rows.copy()
    rows["cross_pp_delta"] = pd.to_numeric(rows.get("cross_pp_delta"), errors="coerce")
    rows["family_pp_delta"] = pd.to_numeric(rows.get("family_pp_delta"), errors="coerce")
    rows["family_mmd_delta"] = pd.to_numeric(rows.get("family_mmd_delta"), errors="coerce")
    keep = rows[rows.get("keep_or_close", "").astype(str).str.contains("keep", na=False)]
    canonical_fail = rows[rows.get("keep_or_close", "").astype(str).str.contains("canonical_noharm_fail", na=False)]
    mmd_harm = rows[rows["family_mmd_delta"] > 0.005]
    best = rows.sort_values(["cross_pp_delta", "family_pp_delta"], ascending=False).iloc[0]
    return {
        "rows": int(len(rows)),
        "kept_train_only_rows": int(len(keep)),
        "canonical_noharm_fail_rows": int(len(canonical_fail)),
        "family_mmd_harm_rows": int(len(mmd_harm)),
        "best_cross_pp": finite_float(best.get("cross_pp_delta")),
        "best_family_pp": finite_float(best.get("family_pp_delta")),
        "best_family_mmd": finite_float(best.get("family_mmd_delta")),
        "best_route": str(best.get("route")),
        "decision_status": nested_get(decision, ["decision", "status"], decision.get("status")),
    }


def candidate_rows(summaries: dict[str, Any]) -> pd.DataFrame:
    outcome = summaries["outcome"]
    count = summaries["count"]
    observable = summaries["observable"]
    response = summaries["response"]
    z_neigh = summaries["zscape_neighborhood"]
    z_struct = summaries["zscape_structural"]
    z_atlas = summaries["zscape_atlas"]
    z_expansion = summaries["zscape_expansion_gate"]
    ot = summaries["ot_pairing"]

    rows = [
        {
            "candidate": "count_scaling_noharm_predictor",
            "hypothesis": "weak count-scaling gains become useful only if a train-only no-harm predictor can veto MMD/canonical failures",
            "current_evidence": (
                f"best route {count.get('best_route')} cross {fmt(count.get('best_cross_pp'))}, "
                f"family {fmt(count.get('best_family_pp'))}, MMD {fmt(count.get('best_family_mmd'))}; "
                f"{count.get('canonical_noharm_fail_rows')} canonical no-harm fail rows"
            ),
            "split_boundary": "train-only/internal Track A artifacts only; canonical single groups only as existing no-harm veto context; no canonical multi selection",
            "gpu_authorized_now": False,
            "cpu_gate_authorized_now": True,
            "resource_plan": "short CPU synthesis <=4 cores; any future smoke <=2 GPUs, <=2 train jobs/GPU, <=24 LatentFM cores",
            "launcher_or_config": "new CPU no-harm predictor or split guard; no GPU launcher yet",
            "promotion_gate": "prospective candidate must predict no-harm risk and leave at least one nonclosed route with cross/family pp >=0.02 and no MMD hard harm",
            "fail_close_rule": "if no nonclosed positive/no-harm rows or predictor cannot separate MMD/canonical failures, do not relaunch count-scaling GPU",
            "status": "cpu_gate_next",
        },
        {
            "candidate": "observable_or_hvg_condition_weighting",
            "hypothesis": "observable response concentration may define information-rich conditions after abundance/detection confounding is controlled",
            "current_evidence": (
                f"observable status {observable.get('status')}; response-compressibility status {response.get('status')}"
            ),
            "split_boundary": "completed raw-expression and Track A joins only; no training until mean-matched and response-energy confounding gates pass",
            "gpu_authorized_now": False,
            "cpu_gate_authorized_now": False,
            "resource_plan": "blocked; only descriptive use",
            "launcher_or_config": "none",
            "promotion_gate": "must show stable expected-direction association after mean-matched/abundance/detection controls",
            "fail_close_rule": "current static proxy already failed stable Track A signal, so no GPU mutation without a new nonstatic axis",
            "status": "descriptor_only_blocked",
        },
        {
            "candidate": "zscape_dynamic_pairability_translation",
            "hypothesis": "state-preserved dynamic pairability, not response magnitude, can become the scaling x for perturbation response information",
            "current_evidence": (
                f"structural status {z_struct.get('status')}; atlas status {z_atlas.get('status')}; "
                f"strict-expansion candidates {nested_get(z_expansion, ['decision', 'candidate_rows'], 'NA')}"
            ),
            "split_boundary": "ZSCAPE CPU OT pseudo-pairs only until strict controls pass; no human LatentFM loss/sampling before translation/no-harm gate",
            "gpu_authorized_now": False,
            "cpu_gate_authorized_now": False,
            "resource_plan": "await running CPU strict-control expansion; do not poll before cadence",
            "launcher_or_config": "running strict-control expansion; future translation table if pass",
            "promotion_gate": ">=4 rows and >=2 lineages pass strict matched-null controls, followed by train-set translation/no-harm",
            "fail_close_rule": "if strict expansion fails, keep as biology atlas only and do not use as model positives",
            "status": "await_running_cpu_gate",
        },
        {
            "candidate": "condition_response_neighborhood_concordance",
            "hypothesis": "cross-dataset support plus response-vector direction concordance can identify high-information train conditions",
            "current_evidence": f"status {z_neigh.get('status')}; GPU authorized {z_neigh.get('gpu_authorized', z_neigh.get('gpu_authorized_next'))}",
            "split_boundary": "parent train condition-neighborhood rows only; direction-shuffle null required; no canonical multi/Track C query",
            "gpu_authorized_now": False,
            "cpu_gate_authorized_now": False,
            "resource_plan": "blocked by balance/null; no GPU",
            "launcher_or_config": "none",
            "promotion_gate": ">=300 matched pairs, >=2 claim-ready perturbation types, covariate SMD<=0.15/AUC<=0.60, null p95 below real gap",
            "fail_close_rule": "current design failed strict balance and null specificity; do not mutate directly into launcher",
            "status": "null_or_balance_blocked",
        },
        {
            "candidate": "ot_pairing_ablation",
            "hypothesis": "more exact OT assignment pairs may improve minibatch supervision",
            "current_evidence": (
                f"default-vs-random reduction {fmt(ot.get('default_sinkhorn_reduction_vs_random'))}; "
                f"assignment extra {fmt(ot.get('assignment_extra_reduction'))}"
            ),
            "split_boundary": "train-only pairing profile only; no model checkpoint selection",
            "gpu_authorized_now": False,
            "cpu_gate_authorized_now": False,
            "resource_plan": "blocked; assignment is ~277x slower for ~0.014 extra cost reduction",
            "launcher_or_config": "none",
            "promotion_gate": "need a cheap approximation or evidence that extra assignment reduction changes validation metrics",
            "fail_close_rule": "keep default Sinkhorn multinomial unless a new low-cost pairing mode appears",
            "status": "engineering_blocked",
        },
    ]

    if outcome.get("candidate_nonduplicates", 0) == 0:
        for row in rows:
            if row["candidate"] == "count_scaling_noharm_predictor":
                row["current_evidence"] += (
                    f"; retrospective panel has {outcome.get('candidate_nonduplicates')} nonduplicate GPU candidates "
                    f"and {outcome.get('weak_nonclosed')} weak nonclosed rows"
                )
    return pd.DataFrame(rows)


def fmt(value: Any, digits: int = 4) -> str:
    x = finite_float(value)
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def decide(queue: pd.DataFrame, summaries: dict[str, Any]) -> dict[str, Any]:
    gpu_now = queue[queue["gpu_authorized_now"] == True]  # noqa: E712
    cpu_now = queue[queue["cpu_gate_authorized_now"] == True]  # noqa: E712
    if not gpu_now.empty:
        status = "trainset_strategy_queue_has_gpu_candidate"
        next_action = "run required 3-sample GPU audit and launch the top bounded GPU candidate with RUN_STATUS"
    elif not cpu_now.empty:
        status = "trainset_strategy_queue_cpu_gate_only_no_gpu"
        next_action = "run or prepare the listed CPU gate while awaiting ZSCAPE strict-control expansion cadence"
    else:
        status = "trainset_strategy_queue_all_gpu_blocked"
        next_action = "await running strict-control expansion or subagent slate; record blocker before GPU launch"
    return {
        "status": status,
        "gpu_authorized_now": bool(not gpu_now.empty),
        "gpu_candidate_count": int(len(gpu_now)),
        "cpu_gate_candidate_count": int(len(cpu_now)),
        "top_gpu_candidates": gpu_now["candidate"].astype(str).tolist(),
        "top_cpu_candidates": cpu_now["candidate"].astype(str).tolist(),
        "retrospective_nonduplicate_candidates": int(summaries["outcome"].get("candidate_nonduplicates", 0)),
        "next_action": next_action,
        "resource_caps_used": {
            "physical_gpus": 2,
            "latentfm_train_jobs_per_gpu": 2,
            "latentfm_cpu_cores": 24,
        },
    }


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "_None._"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = [str(row.get(col, "")).replace("\n", " ") for col in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(queue: pd.DataFrame, summaries: dict[str, Any], decision: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = OUT_DIR / "trainset_strategy_queue_rows_20260630.csv"
    json_path = OUT_DIR / "trainset_strategy_queue_20260630.json"
    md_path = OUT_DIR / "LATENTFM_TRAINSET_STRATEGY_QUEUE_20260630.md"
    queue.to_csv(rows_path, index=False)
    payload = {
        "boundary": {
            "reads_completed_reports_only": True,
            "training_or_inference": False,
            "checkpoint_selection": False,
            "canonical_multi_selection": False,
            "trackc_query_access": False,
        },
        "inputs": {k: str(v) for k, v in INPUTS.items()},
        "summaries": summaries,
        "decision": decision,
        "outputs": {"rows": str(rows_path), "markdown_report": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    text = f"""# LatentFM Train-Set Strategy Queue 20260630

## Boundary

- CPU/report-only synthesis of completed LatentFM/scaling/ZSCAPE artifacts.
- No training, inference, active-log polling, checkpoint selection, canonical multi selection, or Track C query access.
- Resource caps assumed for any future GPU candidate: `2` physical GPUs, `2` LatentFM train jobs per GPU, `24` LatentFM CPU cores.

## Decision

- status: `{decision['status']}`
- GPU authorized now: `{decision['gpu_authorized_now']}`
- GPU candidate count: `{decision['gpu_candidate_count']}`
- CPU-gate candidate count: `{decision['cpu_gate_candidate_count']}`
- next action: `{decision['next_action']}`

## Queue

{markdown_table(queue, ['candidate', 'status', 'gpu_authorized_now', 'cpu_gate_authorized_now', 'current_evidence', 'promotion_gate', 'fail_close_rule'])}

## Interpretation

- No completed nonduplicate axis currently authorizes a GPU smoke without another gate.
- The most actionable train-set route is not another blind count-scaling sweep; it is a no-harm/failure-localization CPU gate that can explain why weak internal count-scaling gains repeatedly fail canonical no-harm.
- ZSCAPE dynamic pairability remains the best biology/scaling idea, but it is waiting on the detached strict-control expansion and must not be used as a model positive yet.
- OT minibatch pairing is already doing useful work; the exact assignment variant is too slow for the small extra cost reduction observed so far.

## Artifacts

- JSON: `{json_path}`
- rows: `{rows_path}`
"""
    md_path.write_text(text, encoding="utf-8")


def main() -> None:
    outcome_rows = pd.read_csv(INPUTS["run_outcome_rows"]) if INPUTS["run_outcome_rows"].exists() else pd.DataFrame()
    count_rows = pd.read_csv(INPUTS["condition_count_rows"]) if INPUTS["condition_count_rows"].exists() else pd.DataFrame()
    run_outcome_json = load_json(INPUTS["run_outcome_json"])
    count_json = load_json(INPUTS["condition_count_json"])
    observable = load_json(INPUTS["observable_budget_json"])
    response = load_json(INPUTS["response_compressibility_json"])
    z_neigh = load_json(INPUTS["zscape_neighborhood_json"])
    z_struct = load_json(INPUTS["zscape_structural_json"])
    z_atlas = load_json(INPUTS["zscape_atlas_json"])
    z_expansion = load_json(INPUTS["zscape_expansion_gate_json"])
    ot = load_json(INPUTS["ot_pairing_json"])

    summaries = {
        "outcome": build_outcome_summary(outcome_rows),
        "outcome_decision": nested_get(run_outcome_json, ["decision"], run_outcome_json.get("decision", {})),
        "count": build_count_summary(count_rows, count_json),
        "observable": nested_get(observable, ["decision"], observable.get("decision", observable)),
        "response": nested_get(response, ["decision"], response.get("decision", response)),
        "zscape_neighborhood": nested_get(z_neigh, ["decision"], z_neigh.get("decision", z_neigh)),
        "zscape_structural": nested_get(z_struct, ["decision"], z_struct.get("decision", z_struct)),
        "zscape_atlas": nested_get(z_atlas, ["decision"], z_atlas.get("decision", z_atlas)),
        "zscape_expansion_gate": z_expansion,
        "ot_pairing": nested_get(ot, ["decision"], ot.get("decision", ot)),
    }
    queue = candidate_rows(summaries)
    decision = decide(queue, summaries)
    write_outputs(queue, summaries, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
