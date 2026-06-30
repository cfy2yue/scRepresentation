#!/usr/bin/env python3
"""Summarize frozen canonical no-harm for budget128 6k true-cell-count route."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_RUN_ROOT", ROOT / "runs/latentfm_true_cell_count_budget128_6k_canonical_noharm_20260625"))
OUT_JSON = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_OUT_JSON", ROOT / "reports/latentfm_true_cell_count_budget128_6k_canonical_noharm_decision_20260625.json"))
OUT_MD = Path(os.environ.get("LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_OUT_MD", ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_CANONICAL_NOHARM_DECISION_20260625.md"))
SEEDS = [42, 43, 44]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def find_delta(gate: dict[str, Any], stratum: str, metric: str) -> dict[str, Any]:
    for row in gate.get("paired_deltas", []) or []:
        if row.get("stratum") == stratum and row.get("metric") == metric:
            return row
    return {}


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):+.6f}"
    except Exception:
        return str(x)


def summarize_seed(seed: int) -> dict[str, Any]:
    run_name = f"xverse_truecell_nested_budget128_tailstable_seed{seed}_6000"
    run_dir = RUN_ROOT / run_name
    gate_json = run_dir / "posthoc_eval_canonical" / "single_background_candidate_gate.json"
    gate = load_json(gate_json)
    row = {
        "seed": seed,
        "run": run_name,
        "run_dir": str(run_dir),
        "posthoc_exit_code": read_text(run_dir / "POSTHOC_EXIT_CODE"),
        "posthoc_finished": read_text(run_dir / "POSTHOC_FINISHED"),
        "gate_json": str(gate_json),
        "status": "pending",
        "gate_status": None,
        "gate_reasons": [],
        "metrics": {},
    }
    if row["posthoc_exit_code"] not in (None, "0"):
        row["status"] = "posthoc_failed"
        return row
    if not gate:
        return row
    row["status"] = "done"
    row["gate_status"] = (gate.get("gate") or {}).get("status")
    row["gate_reasons"] = (gate.get("gate") or {}).get("reasons") or []
    for stratum, metric in [
        ("cross_background_seen_gene", "pearson_pert"),
        ("all_test_single", "pearson_pert"),
        ("all_test_single", "test_mmd_clamped"),
        ("family_gene", "pearson_pert"),
        ("family_gene", "test_mmd_clamped"),
    ]:
        item = find_delta(gate, stratum, metric)
        row["metrics"][f"{stratum}:{metric}"] = {
            "delta_mean": item.get("delta_mean"),
            "p_harm": item.get("p_harm"),
            "p_improve": item.get("p_improve"),
            "ci95": item.get("ci95"),
            "status": item.get("status"),
        }
    return row


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(r["status"] == "pending" for r in rows):
        return {"status": "canonical_noharm_pending", "action": "wait_1800s_or_work_on_parallel_branches"}
    failed_posthoc = [r["run"] for r in rows if r["status"] == "posthoc_failed"]
    if failed_posthoc:
        return {"status": "canonical_noharm_posthoc_failed", "action": "inspect_failed_logs_once", "failed_runs": failed_posthoc}
    failed = [r for r in rows if r.get("gate_status") != "candidate_gate_pass"]
    if failed:
        return {
            "status": "canonical_noharm_fail_close_promotion",
            "action": "close_deployable_promotion_keep_mechanism_claim",
            "failed_runs": [r["run"] for r in failed],
        }
    return {
        "status": "canonical_noharm_all3_pass_review_next",
        "action": "review_before_any_promotion_or_broader_eval",
        "passed_runs": [r["run"] for r in rows],
    }


def render_md(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# LatentFM True Cell-Count Budget128 6k Canonical No-Harm Decision",
        "",
        f"Status: `{decision['status']}`",
        f"Action: `{decision['action']}`",
        "",
        "## Boundary",
        "",
        "- Frozen no-harm veto for the route frozen in `LATENTFM_TRUE_CELL_COUNT_BUDGET128_6K_ROUTE_FREEZE_20260625.md`.",
        "- Evaluates all seeds 42/43/44; canonical results are not used to choose a seed.",
        "- Canonical multi is not evaluated or selected.",
        "- Held-out Track C query is not read.",
        "",
        "## Rows",
        "",
        "| seed | run | status | gate | cross-bg pp delta | all-single p_harm | family-gene p_harm | family MMD p_harm | reasons |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        metrics = row.get("metrics") or {}
        cross = metrics.get("cross_background_seen_gene:pearson_pert", {})
        all_single = metrics.get("all_test_single:pearson_pert", {})
        fam = metrics.get("family_gene:pearson_pert", {})
        fam_mmd = metrics.get("family_gene:test_mmd_clamped", {})
        lines.append(
            f"| {row['seed']} | `{row['run']}` | `{row['status']}` | `{row.get('gate_status')}` | "
            f"{fmt(cross.get('delta_mean'))} | {fmt(all_single.get('p_harm'))} | "
            f"{fmt(fam.get('p_harm'))} | {fmt(fam_mmd.get('p_harm'))} | "
            f"{', '.join(row.get('gate_reasons') or []) or 'none'} |"
        )
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`"])
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = [summarize_seed(seed) for seed in SEEDS]
    payload = {
        "decision": decide(rows),
        "boundary": {
            "canonical_metrics_read": True,
            "canonical_multi_eval": False,
            "canonical_multi_selection": False,
            "trackc_query_read": False,
            "checkpoint_selection": False,
            "all_seeds_required": SEEDS,
        },
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["decision"]["status"], "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
