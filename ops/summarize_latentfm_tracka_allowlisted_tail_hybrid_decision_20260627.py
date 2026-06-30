#!/usr/bin/env python3
"""Summarize allowlisted-tail hybrid seed42/43 exact-tail gates.

CPU/report-only. It reads RUN_STATUS marker files and predeclared exact-tail gate
JSONs if present. It does not run training, inference, or model evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_BLOCK = ROOT / "runs/latentfm_tracka_xverse_allowlisted_tail_hybrid_20260627"
GATE_DIR = ROOT / "reports/tracka_exact_tail_candidate_gate_20260627"
OUT_JSON = ROOT / "reports/latentfm_tracka_allowlisted_tail_hybrid_decision_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKA_ALLOWLISTED_TAIL_HYBRID_DECISION_20260627.md"

RUNS = [
    {
        "seed": 42,
        "run_name": "xverse_allowtail_hybrid_pertresid_prior_w003_p002_replay1_2k_seed42",
        "tmux": "lfm_xverse_allowtail_20260627",
    },
    {
        "seed": 43,
        "run_name": "xverse_allowtail_hybrid_pertresid_prior_w003_p002_replay1_2k_seed43",
        "tmux": "lfm_xverse_allowtail_seed43_20260627",
    },
]


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def metric(payload: dict[str, Any] | None, group: str, name: str) -> dict[str, Any] | None:
    if not payload:
        return None
    for row in payload.get("summaries", []):
        if row.get("group") == group and row.get("metric") == name:
            return row
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def summarize_run(item: dict[str, Any]) -> dict[str, Any]:
    run_dir = RUN_BLOCK / item["run_name"]
    gate_json = GATE_DIR / f"{item['run_name']}.json"
    gate_md = GATE_DIR / f"{item['run_name']}.md"
    payload = read_json(gate_json)
    exit_code = read_text(run_dir / "EXIT_CODE")
    posthoc_exit = read_text(run_dir / "POSTHOC_EXIT_CODE")
    posthoc_rerun_exit = read_text(run_dir / "POSTHOC_RERUN_EXIT_CODE")
    posthoc_rerun_session = read_text(run_dir / "POSTHOC_RERUN_SESSION_NAME")
    train_finished = (run_dir / "FINISHED").exists()
    posthoc_finished = (run_dir / "POSTHOC_FINISHED").exists()
    posthoc_rerun_finished = (run_dir / "POSTHOC_RERUN_FINISHED").exists()
    rerun_pending = posthoc_rerun_session is not None and posthoc_rerun_exit is None
    effective_posthoc_exit = posthoc_rerun_exit if posthoc_rerun_exit is not None else posthoc_exit
    if payload:
        status = payload.get("status", "gate_unknown")
    elif exit_code is None:
        status = "running_train"
    elif exit_code != "0":
        status = "train_failed"
    elif rerun_pending:
        status = "running_or_pending_posthoc_rerun"
    elif effective_posthoc_exit is None:
        status = "running_or_pending_posthoc"
    elif effective_posthoc_exit != "0":
        status = "posthoc_failed"
    else:
        status = "gate_missing_after_posthoc"
    return {
        **item,
        "status": status,
        "run_dir": str(run_dir),
        "run_status": str(run_dir / "RUN_STATUS.md"),
        "exit_code": exit_code,
        "posthoc_exit": posthoc_exit,
        "posthoc_rerun_exit": posthoc_rerun_exit,
        "posthoc_rerun_session": posthoc_rerun_session,
        "effective_posthoc_exit": effective_posthoc_exit,
        "rerun_pending": rerun_pending,
        "train_finished": train_finished,
        "posthoc_finished": posthoc_finished,
        "posthoc_rerun_finished": posthoc_rerun_finished,
        "gate_json": str(gate_json),
        "gate_md": str(gate_md),
        "gate_present": payload is not None,
        "gate_reasons": [] if not payload else payload.get("reasons", []),
        "exact_simple_pp": None if not payload else (metric(payload, "exact_simple_single_unseen", "pearson_pert") or {}).get("delta_mean"),
        "exact_simple_mmd": None if not payload else (metric(payload, "exact_simple_single_unseen", "test_mmd_clamped") or {}).get("delta_mean"),
        "exact_cross_pp": None if not payload else (metric(payload, "exact_cross_background_seen_gene", "pearson_pert") or {}).get("delta_mean"),
        "exact_cross_mmd": None if not payload else (metric(payload, "exact_cross_background_seen_gene", "test_mmd_clamped") or {}).get("delta_mean"),
        "recurrent_cross_pp": None if not payload else (metric(payload, "recurrent_cross_background_hard_tail", "pearson_pert") or {}).get("delta_mean"),
        "recurrent_cross_mmd": None if not payload else (metric(payload, "recurrent_cross_background_hard_tail", "test_mmd_clamped") or {}).get("delta_mean"),
    }


def main() -> int:
    rows = [summarize_run(item) for item in RUNS]
    complete = all(row["gate_present"] for row in rows)
    pass_like = complete and all(row["status"] == "candidate_exact_tail_gate_pass_gpu_candidate" for row in rows)
    any_failed = any(
        row["status"] not in {
            "running_train",
            "running_or_pending_posthoc",
            "running_or_pending_posthoc_rerun",
            "candidate_exact_tail_gate_pass_gpu_candidate",
        }
        for row in rows
    )
    if pass_like:
        status = "allowlisted_tail_hybrid_two_seed_gate_pass_needs_noharm_review"
    elif complete or any_failed:
        status = "allowlisted_tail_hybrid_gate_fail_or_incomplete_close_review"
    else:
        status = "allowlisted_tail_hybrid_pending"

    payload = {
        "status": status,
        "gpu_authorized_for_new_followup": bool(pass_like),
        "boundary": {
            "cpu_report_only": True,
            "canonical_multi_selection_weight": 0,
            "trackc_query_used": False,
        },
        "runs": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# LatentFM Track A Allowlisted-Tail Hybrid Decision 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized for new follow-up: `{bool(pass_like)}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only decision over predeclared exact-tail gate JSONs.",
        "- Does not train, infer, select checkpoints, read canonical multi for selection, or read Track C query.",
        "",
        "## Run Summary",
        "",
        "| seed | status | exit | posthoc | exact simple pp | exact simple MMD | exact cross pp | exact cross MMD | recurrent cross pp | reasons |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | `{row['status']}` | {row['exit_code'] or 'NA'} | "
            f"{row['effective_posthoc_exit'] or 'NA'} | {fmt(row['exact_simple_pp'])} | "
            f"{fmt(row['exact_simple_mmd'])} | {fmt(row['exact_cross_pp'])} | "
            f"{fmt(row['exact_cross_mmd'])} | {fmt(row['recurrent_cross_pp'])} | "
            f"`{', '.join(row['gate_reasons']) if row['gate_reasons'] else 'NA'}` |"
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "Both seeds must pass exact simple, exact cross-background, recurrent hard-tail, and canonical no-harm gates before any follow-up GPU is authorized. Any hard no-harm failure closes this branch.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
