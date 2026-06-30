#!/usr/bin/env python3
"""Audit the frozen support-context v2 final diagnostic package.

This script is read-only. It validates provenance and claim-boundary artifacts
after the one-shot query has already been consumed. It must not be used for
model selection or tuning.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_NAME = "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42"

PATHS = {
    "smoke_json": ROOT / f"reports/latentfm_trackc_routed_distill_smoke_decision_{RUN_NAME}.json",
    "smoke_md": ROOT / f"reports/LATENTFM_TRACKC_ROUTED_DISTILL_SMOKE_DECISION_{RUN_NAME}.md",
    "uncapped_json": ROOT / f"reports/latentfm_trackc_support_context_v2_uncapped_noharm_{RUN_NAME}_20260623_decision.json",
    "uncapped_md": ROOT / f"reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_UNCAPPED_NOHARM_{RUN_NAME}_DECISION_20260623.md",
    "freeze_json": ROOT / f"reports/latentfm_trackc_support_context_v2_query_freeze_{RUN_NAME}_20260623.json",
    "freeze_md": ROOT / f"reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_FREEZE_{RUN_NAME}_20260623.md",
    "query_json": ROOT / f"reports/latentfm_trackc_support_context_v2_query_once_decision_{RUN_NAME}_20260623.json",
    "query_md": ROOT / f"reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_ONCE_DECISION_{RUN_NAME}_20260623.md",
    "failure_json": ROOT / "reports/latentfm_trackc_support_context_v2_query_failure_cases_20260623.json",
    "failure_md": ROOT / "reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_FAILURE_CASES_20260623.md",
    "synthesis_md": ROOT / "reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FINAL_DIAGNOSTIC_SYNTHESIS_20260623.md",
    "residual_uncapped_json": ROOT / "reports/latentfm_trackc_support_context_v2_uncapped_noharm_xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42_20260623_decision.json",
    "residual_uncapped_md": ROOT / "reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_UNCAPPED_NOHARM_xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42_DECISION_20260623.md",
    "query_run_status": ROOT / f"runs/latentfm_trackc_support_context_v2_query_once_{RUN_NAME}_20260623/RUN_STATUS.md",
    "residual_uncapped_run_status": ROOT / "runs/latentfm_trackc_support_context_v2_uncapped_noharm_xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42_20260623/RUN_STATUS.md",
}

SPLITS = {
    "safe_trainselect_split": ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json",
    "full_v2_split": ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json",
    "canonical_split": ROOT / "dataset/biFlow_data/split_seed42.json",
}

OUT_JSON = ROOT / "reports/latentfm_trackc_support_context_v2_final_package_audit_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_FINAL_PACKAGE_AUDIT_20260623.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def nested_status(payload: dict[str, Any]) -> str | None:
    decision = payload.get("decision")
    if isinstance(decision, dict):
        return decision.get("status")
    return payload.get("status")


def table(payload: dict[str, Any], section: str, key: str) -> dict[str, Any]:
    obj = (payload.get("tables") or {}).get(section) or {}
    return obj.get(key) or {}


def text_contains(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def num(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def try_git_commit(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def tmux_sessions() -> list[str]:
    try:
        proc = subprocess.run(["tmux", "ls"], check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line.split(":", 1)[0] for line in proc.stdout.splitlines() if line.strip()]


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {"exists": True, "size_bytes": st.st_size, "mtime": st.st_mtime}


def main() -> None:
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            failures.append(name)

    for name, path in PATHS.items():
        check(f"exists:{name}", path.exists(), str(path))

    smoke = load_json(PATHS["smoke_json"])
    uncapped = load_json(PATHS["uncapped_json"])
    freeze = load_json(PATHS["freeze_json"])
    query = load_json(PATHS["query_json"])
    failure = load_json(PATHS["failure_json"])
    residual_uncapped = load_json(PATHS["residual_uncapped_json"])

    statuses = {
        "smoke": nested_status(smoke),
        "uncapped": nested_status(uncapped),
        "freeze": nested_status(freeze),
        "query": nested_status(query),
        "failure_cases": nested_status(failure),
        "residual_uncapped": nested_status(residual_uncapped),
    }
    expected_statuses = {
        "smoke": "trackc_smoke_support_pass_needs_uncapped_noharm_before_query",
        "uncapped": "trackc_uncapped_canonical_noharm_pass_query_allowed_once",
        "freeze": "trackc_support_context_v2_query_freeze_pass_query_allowed_once",
        "query": "trackc_query_diagnostic_candidate_supported",
        "failure_cases": "trackc_support_context_v2_query_failure_cases_ready",
        "residual_uncapped": "trackc_uncapped_canonical_noharm_pass_query_allowed_once",
    }
    for key, expected in expected_statuses.items():
        check(f"status:{key}", statuses.get(key) == expected, {"observed": statuses.get(key), "expected": expected})

    freeze_hashes = freeze.get("hashes") or {}
    split_hashes = {name: sha256(path) for name, path in SPLITS.items()}
    for key, observed in split_hashes.items():
        check(f"split_hash:{key}", observed == freeze_hashes.get(key), {"observed": observed, "freeze": freeze_hashes.get(key)})

    check("freeze_failed_checks_empty", freeze.get("failed_checks") == [], freeze.get("failed_checks"))
    check("freeze_query_authorization_one_shot", freeze.get("query_authorization") == "one_shot_query_allowed", freeze.get("query_authorization"))

    q_pp = table(query, "split", "query_multi:pearson_pert")
    q_mmd = table(query, "split", "query_multi:test_mmd_clamped")
    q_unseen2_pp = table(query, "split", "query_multi_unseen2:pearson_pert")
    q_unseen2_mmd = table(query, "split", "query_multi_unseen2:test_mmd_clamped")
    metrics = {
        "query_multi_pp_delta": q_pp.get("delta_mean"),
        "query_multi_pp_ci": [q_pp.get("ci95_low"), q_pp.get("ci95_high")],
        "query_multi_pp_p_harm": q_pp.get("p_harm"),
        "query_multi_mmd_delta": q_mmd.get("delta_mean"),
        "query_multi_mmd_p_harm": q_mmd.get("p_harm"),
        "query_unseen2_pp_delta": q_unseen2_pp.get("delta_mean"),
        "query_unseen2_mmd_delta": q_unseen2_mmd.get("delta_mean"),
    }
    check("query_primary_pp_supported", num(q_pp.get("delta_mean"), 0.0) > 0 and num(q_pp.get("p_improvement"), 0.0) >= 0.8, q_pp)
    check("query_primary_mmd_no_hard_harm", num(q_mmd.get("p_harm"), 1.0) <= 0.8, q_mmd)
    check("query_unseen2_mmd_no_hard_harm", num(q_unseen2_mmd.get("p_harm"), 1.0) <= 0.8, q_unseen2_mmd)

    uncapped_single_pp = table(uncapped, "split", "test_single:pearson_pert")
    uncapped_family_pp = table(uncapped, "family", "family_gene:pearson_pert")
    residual_single_pp = table(residual_uncapped, "split", "test_single:pearson_pert")
    residual_family_pp = table(residual_uncapped, "family", "family_gene:pearson_pert")
    check("resfilm_uncapped_single_exact_noharm", uncapped_single_pp.get("delta_mean") == 0, uncapped_single_pp)
    check("resfilm_uncapped_family_exact_noharm", uncapped_family_pp.get("delta_mean") == 0, uncapped_family_pp)
    check("residual_uncapped_single_exact_noharm", residual_single_pp.get("delta_mean") == 0, residual_single_pp)
    check("residual_uncapped_family_exact_noharm", residual_family_pp.get("delta_mean") == 0, residual_family_pp)

    boundary_terms = [
        "must not be used to tune",
        "Do not claim a blanket formal multi-perturbation solution",
        "does not authorize a second",
    ]
    synthesis_text = PATHS["synthesis_md"].read_text(encoding="utf-8")
    synthesis_flat = " ".join(synthesis_text.split())
    for term in boundary_terms:
        check(f"synthesis_boundary:{term}", term in synthesis_flat, term)
    check("query_decision_no_reuse_action", (query.get("decision") or {}).get("action") == "do_not_reuse_query_for_selection", (query.get("decision") or {}).get("action"))
    check("query_run_status_finished", text_contains(PATHS["query_run_status"], "Finished. Exit code `0`"), str(PATHS["query_run_status"]))
    check("residual_run_status_finished", text_contains(PATHS["residual_uncapped_run_status"], "Finished. Exit code `0`"), str(PATHS["residual_uncapped_run_status"]))

    sessions = tmux_sessions()
    check("no_active_tmux_sessions", not sessions, sessions)

    artifact_info = {name: file_info(path) for name, path in PATHS.items()}
    checkpoint_info = {
        "anchor_checkpoint": file_info(Path(freeze.get("anchor_checkpoint", ""))),
        "candidate_checkpoint": file_info(Path(freeze.get("candidate_checkpoint", ""))),
        "hashes_from_freeze_gate": {
            "anchor_checkpoint": freeze_hashes.get("anchor_checkpoint"),
            "candidate_checkpoint": freeze_hashes.get("candidate_checkpoint"),
        },
        "note": "Checkpoint files are large; this audit verifies existence/size and reuses the query-free freeze gate checkpoint hashes.",
    }

    payload = {
        "status": "trackc_support_context_v2_final_package_audit_pass" if not failures else "trackc_support_context_v2_final_package_audit_fail",
        "run_name": RUN_NAME,
        "checks": checks,
        "failed_checks": failures,
        "statuses": statuses,
        "metrics": metrics,
        "split_hashes": split_hashes,
        "freeze_hashes": freeze_hashes,
        "artifact_info": artifact_info,
        "checkpoint_info": checkpoint_info,
        "git": {
            "project_commit": try_git_commit(ROOT),
            "coupledfm_commit": try_git_commit(ROOT / "CoupledFM"),
        },
        "rules": [
            "read-only final-package audit",
            "query artifacts are already consumed and may not drive future selection",
            "residual uncapped pass is robustness/no-harm evidence only",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Track C Support-Context V2 Final Package Audit",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Summary",
        "",
        f"- run: `{RUN_NAME}`",
        f"- query status: `{statuses['query']}`",
        f"- query Pearson delta: `{metrics['query_multi_pp_delta']:+.6f}`",
        f"- query MMD delta: `{metrics['query_multi_mmd_delta']:+.6f}`",
        f"- active tmux sessions: `{len(sessions)}`",
        "",
        "## Checks",
        "",
        "| check | passed | evidence |",
        "|---|---:|---|",
    ]
    for row in checks:
        evidence = row.get("evidence")
        ev = json.dumps(evidence, sort_keys=True) if not isinstance(evidence, str) else evidence
        if len(ev) > 240:
            ev = ev[:237] + "..."
        lines.append(f"| `{row['name']}` | `{row['passed']}` | {ev} |")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- Supported as a frozen Track C support-context v2 diagnostic route.",
        "- Not a blanket formal multi-perturbation claim.",
        "- Do not reuse the held-out query result for route/checkpoint/threshold/feature tuning.",
        "- Do not trigger residual query from this package.",
        "",
        "## Key Artifacts",
        "",
    ]
    for name, path in PATHS.items():
        lines.append(f"- `{name}`: `{path}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failed_checks": failures, "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
