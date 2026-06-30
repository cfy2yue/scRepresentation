#!/usr/bin/env python3
"""Freeze a v2 Track C checkpoint before one-shot held-out query.

This gate is query-blind.  It reads only capped smoke, uncapped canonical
no-harm, split, checkpoint, and launcher provenance.  Passing this gate does
not run query; it only creates the explicit freeze artifact required by the
v2 query wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
ANCHOR = (
    COUPLED
    / "output/latentfm_runs/xverse_8k_full_eval_20260620"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt"
)
SAFE_TRAINSELECT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
FULL_V2 = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
CANONICAL = ROOT / "dataset/biFlow_data/split_seed42.json"
QUERY_WRAPPER = ROOT / "ops/launch_latentfm_trackc_support_context_v2_query_if_pass_20260623.sh"


def safe_id(run_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in run_name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def decision_status(payload: dict[str, Any]) -> str:
    return str(payload.get("status") or (payload.get("decision") or {}).get("status") or "")


def bash_n(path: Path) -> tuple[bool, str]:
    proc = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
    return proc.returncode == 0, (proc.stderr or proc.stdout).strip()


def infer_paths(run_name: str) -> dict[str, Path]:
    sid = safe_id(run_name)
    if run_name == "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42":
        out_root = COUPLED / "output/latentfm_runs/xverse_trackc_support_context_v2_20260623"
    elif run_name in {
        "xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42",
        "xverse_trackc_support_context_v2_contextc_ep050_replay2_2k_seed42",
    }:
        out_root = COUPLED / "output/latentfm_runs/xverse_trackc_support_context_v2_parallel_20260623"
    else:
        raise SystemExit(f"unsupported v2 run name: {run_name}")
    label = f"latentfm_trackc_support_context_v2_uncapped_noharm_{sid}_20260623"
    query_label = f"latentfm_trackc_support_context_v2_query_once_{sid}_20260623"
    return {
        "candidate_checkpoint": out_root / run_name / "best.pt",
        "smoke_decision": ROOT / f"reports/latentfm_trackc_routed_distill_smoke_decision_{run_name}.json",
        "uncapped_decision": ROOT / f"reports/{label}_decision.json",
        "uncapped_md": ROOT / f"reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_UNCAPPED_NOHARM_{sid}_DECISION_20260623.md",
        "freeze_json": ROOT / f"reports/latentfm_trackc_support_context_v2_query_freeze_{sid}_20260623.json",
        "freeze_md": ROOT / f"reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_FREEZE_{sid}_20260623.md",
        "query_run": ROOT / "runs" / query_label,
        "query_decision_json": ROOT / f"reports/latentfm_trackc_support_context_v2_query_once_decision_{sid}_20260623.json",
        "query_decision_md": ROOT / f"reports/LATENTFM_TRACKC_SUPPORT_CONTEXT_V2_QUERY_ONCE_DECISION_{sid}_20260623.md",
        "query_boot_dir": ROOT / f"reports/latentfm_trackc_support_context_v2_query_once_bootstrap_{sid}_20260623",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    run_name = args.run_name
    paths = infer_paths(run_name)

    checks: list[dict[str, Any]] = []
    required = {
        "anchor_checkpoint": ANCHOR,
        "candidate_checkpoint": paths["candidate_checkpoint"],
        "smoke_decision": paths["smoke_decision"],
        "uncapped_decision": paths["uncapped_decision"],
        "uncapped_md": paths["uncapped_md"],
        "safe_trainselect_split": SAFE_TRAINSELECT,
        "full_v2_split": FULL_V2,
        "canonical_split": CANONICAL,
        "query_wrapper": QUERY_WRAPPER,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    checks.append({"name": "required_artifacts_exist", "passed": not missing, "evidence": missing})

    smoke_status = ""
    uncapped_status = ""
    if paths["smoke_decision"].exists():
        smoke_status = decision_status(load(paths["smoke_decision"]))
    if paths["uncapped_decision"].exists():
        uncapped_status = decision_status(load(paths["uncapped_decision"]))
    checks.append(
        {
            "name": "capped_smoke_passed",
            "passed": smoke_status == "trackc_smoke_support_pass_needs_uncapped_noharm_before_query",
            "evidence": smoke_status or "missing",
        }
    )
    checks.append(
        {
            "name": "uncapped_noharm_passed",
            "passed": uncapped_status == "trackc_uncapped_canonical_noharm_pass_query_allowed_once",
            "evidence": uncapped_status or "missing",
        }
    )

    syntax_ok, syntax_err = bash_n(QUERY_WRAPPER) if QUERY_WRAPPER.exists() else (False, "missing")
    checks.append({"name": "query_wrapper_bash_n", "passed": syntax_ok, "evidence": syntax_err})

    preexisting_query = [
        str(path)
        for path in [
            paths["query_run"],
            paths["query_decision_json"],
            paths["query_decision_md"],
            paths["query_boot_dir"],
        ]
        if path.exists()
    ]
    checks.append({"name": "query_artifacts_absent", "passed": not preexisting_query, "evidence": preexisting_query})

    hashes: dict[str, str] = {}
    for name, path in {
        "safe_trainselect_split": SAFE_TRAINSELECT,
        "full_v2_split": FULL_V2,
        "canonical_split": CANONICAL,
        "anchor_checkpoint": ANCHOR,
        "candidate_checkpoint": paths["candidate_checkpoint"],
    }.items():
        if path.exists():
            hashes[name] = sha256(path)

    failed = [check for check in checks if not check["passed"]]
    status = "trackc_support_context_v2_query_freeze_pass_query_allowed_once" if not failed else "trackc_support_context_v2_query_freeze_fail_no_query"
    payload = {
        "status": status,
        "run_name": run_name,
        "query_authorization": "one_shot_query_allowed" if not failed else "none",
        "anchor_checkpoint": str(ANCHOR),
        "candidate_checkpoint": str(paths["candidate_checkpoint"]),
        "smoke_decision": str(paths["smoke_decision"]),
        "uncapped_decision": str(paths["uncapped_decision"]),
        "splits": {
            "safe_trainselect": str(SAFE_TRAINSELECT),
            "full_v2_query": str(FULL_V2),
            "canonical_noharm": str(CANONICAL),
        },
        "hashes": hashes,
        "query_wrapper": str(QUERY_WRAPPER),
        "checks": checks,
        "failed_checks": [check["name"] for check in failed],
        "rules": [
            "no held-out query outputs are read by this freeze gate",
            "capped smoke must pass support/canonical gates",
            "uncapped canonical no-harm must pass",
            "query artifacts must not already exist",
            "query result is one-shot diagnostic only and must not tune route, checkpoint, alpha, threshold, or features",
        ],
    }
    paths["freeze_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Track C Support-Context V2 Query Freeze Gate",
        "",
        f"Status: `{status}`",
        f"Run: `{run_name}`",
        f"Query authorization: `{payload['query_authorization']}`",
        "",
        "## Checks",
        "",
        "| check | passed | evidence |",
        "|---|---:|---|",
    ]
    for check in checks:
        evidence = check["evidence"]
        if isinstance(evidence, (dict, list)):
            evidence_s = json.dumps(evidence, sort_keys=True)
        else:
            evidence_s = str(evidence)
        lines.append(f"| `{check['name']}` | `{check['passed']}` | {evidence_s} |")
    lines += ["", "## Hashes", ""]
    for name, digest in hashes.items():
        lines.append(f"- `{name}`: `{digest}`")
    lines += ["", "## Failed Checks", ""]
    lines.extend([f"- `{name}`" for name in payload["failed_checks"]] or ["- none"])
    paths["freeze_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_json": str(paths["freeze_json"]), "out_md": str(paths["freeze_md"])}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
