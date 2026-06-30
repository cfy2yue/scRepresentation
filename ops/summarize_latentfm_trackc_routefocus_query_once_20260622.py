#!/usr/bin/env python3
"""Summarize the frozen one-shot Track C route-focused query evaluation.

This script intentionally reads held-out Track C query outputs, but only after
the support smoke and uncapped canonical no-harm decisions have already passed.
It produces a diagnostic report and explicitly forbids feeding the query result
back into route or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
BOOTSTRAP = ROOT / "ops/bootstrap_latentfm_paired_posthoc_20260621.py"
RUN_NAME = "xverse_trackc_routefocus_condprior_w05_replay1_2k_seed42"
DEFAULT_EVAL_DIR = ROOT / "reports/latentfm_trackc_routefocus_query_once_20260622/eval"
DEFAULT_SMOKE_DECISION = ROOT / f"reports/latentfm_trackc_routed_distill_smoke_decision_{RUN_NAME}.json"
DEFAULT_UNCAPPED_DECISION = ROOT / "reports/latentfm_trackc_routefocus_uncapped_noharm_decision_20260622.json"
DEFAULT_OUT_JSON = ROOT / "reports/latentfm_trackc_routefocus_query_once_decision_20260622.json"
DEFAULT_OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ROUTEFOCUS_QUERY_ONCE_DECISION_20260622.md"
DEFAULT_BOOT_DIR = ROOT / "reports/latentfm_trackc_routefocus_query_once_bootstrap_20260622"

SPLIT_GROUPS = ["query_multi", "query_multi_seen", "query_multi_unseen1", "query_multi_unseen2", "test_multi"]
FAMILY_GROUPS = ["test_multi"]
METRICS = ["pearson_pert", "pearson_ctrl", "direct_pearson", "test_mmd_clamped"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def require_decision(path: Path, expected: str, label: str) -> dict[str, Any]:
    payload = load_json(path)
    status = (payload.get("decision") or {}).get("status")
    if status != expected:
        raise RuntimeError(f"{label} status is {status!r}, expected {expected!r}")
    return payload


def bootstrap_pair(
    *,
    baseline: Path,
    candidate: Path,
    groups: list[str],
    out_json: Path,
    out_md: Path,
    title: str,
    n_boot: int,
    seed: int,
    python: str,
) -> dict[str, Any]:
    if not baseline.is_file():
        raise FileNotFoundError(f"missing anchor JSON: {baseline}")
    if not candidate.is_file():
        raise FileNotFoundError(f"missing candidate JSON: {candidate}")
    run(
        [
            python,
            str(BOOTSTRAP),
            "--baseline-json",
            str(baseline),
            "--candidate-json",
            str(candidate),
            "--groups",
            *groups,
            "--metrics",
            *METRICS,
            "--n-boot",
            str(int(n_boot)),
            "--seed",
            str(int(seed)),
            "--title",
            title,
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    return load_json(out_json)


def index_rows(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        out[(str(row.get("group")), str(row.get("metric")))] = row
    return out


def metric_row(
    tables: dict[str, dict[tuple[str, str], dict[str, Any]]],
    table: str,
    group: str,
    metric: str,
) -> dict[str, Any] | None:
    return tables.get(table, {}).get((group, metric))


def row_float(row: dict[str, Any] | None, key: str, default: float) -> float:
    if not row:
        return default
    value = row.get(key)
    if value is None:
        return default
    return float(value)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def ci_fmt(row: dict[str, Any] | None) -> str:
    if not row:
        return "[NA, NA]"
    return f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]"


def evaluate_diagnostic(tables: dict[str, dict[tuple[str, str], dict[str, Any]]]) -> dict[str, Any]:
    reasons: list[str] = []
    primary_pp = metric_row(tables, "split", "query_multi", "pearson_pert")
    primary_mmd = metric_row(tables, "split", "query_multi", "test_mmd_clamped")
    unseen2_pp = metric_row(tables, "split", "query_multi_unseen2", "pearson_pert")
    unseen2_mmd = metric_row(tables, "split", "query_multi_unseen2", "test_mmd_clamped")

    required = {
        "query_multi_pp": primary_pp,
        "query_multi_mmd": primary_mmd,
        "query_multi_unseen2_pp": unseen2_pp,
        "query_multi_unseen2_mmd": unseen2_mmd,
    }
    for name, row in required.items():
        if not row or row.get("status") != "ok":
            reasons.append(f"missing_or_bad_{name}")

    if not reasons:
        if row_float(primary_pp, "delta_mean", 0.0) <= 0.0:
            reasons.append("query_multi_pp_delta_not_positive")
        if row_float(primary_pp, "p_improvement", 0.0) < 0.80:
            reasons.append("query_multi_pp_improvement_probability_lt_0.80")
        if row_float(primary_mmd, "p_harm", 1.0) > 0.80:
            reasons.append("query_multi_mmd_hard_harm")
        if row_float(unseen2_mmd, "p_harm", 1.0) > 0.80:
            reasons.append("query_unseen2_mmd_hard_harm")

    if reasons:
        status = "trackc_query_diagnostic_not_supported"
    else:
        status = "trackc_query_diagnostic_candidate_supported"
    return {
        "status": status,
        "action": "do_not_reuse_query_for_selection",
        "reasons": reasons,
        "rules": [
            "frozen route/checkpoint only: smoke support gate and uncapped canonical no-harm gate must already pass",
            "primary query_multi pearson_pert delta must be positive with p_improvement >= 0.80",
            "primary query_multi MMD hard-harm probability must be <= 0.80",
            "query_multi_unseen2 MMD hard-harm probability must be <= 0.80",
            "this held-out query result is final diagnostic evidence only and cannot select or tune another checkpoint",
        ],
    }


def row_line(role: str, row: dict[str, Any] | None) -> str:
    if not row:
        return f"| {role} | NA | NA | 0 | 0 | NA | [NA, NA] | NA | NA | missing |"
    return (
        f"| {role} | {row.get('group')} | {row.get('metric')} | "
        f"{row.get('n_matched_conditions', 0)} | {row.get('n_matched_datasets', 0)} | "
        f"{fmt(row.get('delta_mean'))} | {ci_fmt(row)} | "
        f"{fmt(row.get('p_improvement'))} | {fmt(row.get('p_harm'))} | {row.get('status', 'NA')} |"
    )


def render_report(payload: dict[str, Any]) -> str:
    tables = payload["tables"]
    rows = {
        "query pp": metric_row(tables, "split", "query_multi", "pearson_pert"),
        "query mmd": metric_row(tables, "split", "query_multi", "test_mmd_clamped"),
        "query seen pp": metric_row(tables, "split", "query_multi_seen", "pearson_pert"),
        "query unseen1 pp": metric_row(tables, "split", "query_multi_unseen1", "pearson_pert"),
        "query unseen2 pp": metric_row(tables, "split", "query_multi_unseen2", "pearson_pert"),
        "query unseen2 mmd": metric_row(tables, "split", "query_multi_unseen2", "test_mmd_clamped"),
        "test_multi alias pp": metric_row(tables, "split", "test_multi", "pearson_pert"),
    }
    lines = [
        f"# {payload.get('report_title') or 'LatentFM Track C Route-Focused One-Shot Query Decision'}",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Action: `{payload['decision']['action']}`",
        "",
        "## Provenance",
        "",
    ]
    for key, value in payload["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "- leakage boundary: this report reads held-out Track C `query_multi*` only after route/checkpoint freeze and canonical no-harm pass.",
        "- reuse boundary: query results must not be used for future selection, tuning, or branch promotion.",
        "",
        "## Diagnostic Rows",
        "",
        "| role | group | metric | n cond | n ds | delta | 95% CI | p improve | p harm | status |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for role, row in rows.items():
        lines.append(row_line(role, row))
    lines += ["", "## Decision Reasons", ""]
    reasons = payload["decision"].get("reasons") or []
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines += ["", "## Rules", ""]
    lines.extend(f"- {rule}" for rule in payload["decision"].get("rules") or [])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--smoke-decision-json", type=Path, default=DEFAULT_SMOKE_DECISION)
    parser.add_argument("--uncapped-decision-json", type=Path, default=DEFAULT_UNCAPPED_DECISION)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--boot-dir", type=Path, default=DEFAULT_BOOT_DIR)
    parser.add_argument(
        "--report-title",
        default="LatentFM Track C Route-Focused One-Shot Query Decision",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    boot_dir = args.boot_dir
    existing = [p for p in (args.out_json, args.out_md, boot_dir) if p.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite one-shot query decision artifacts: "
            + ", ".join(str(p) for p in existing)
        )
    if not BOOTSTRAP.is_file():
        raise FileNotFoundError(BOOTSTRAP)
    require_decision(
        args.smoke_decision_json,
        "trackc_smoke_support_pass_needs_uncapped_noharm_before_query",
        "smoke decision",
    )
    require_decision(
        args.uncapped_decision_json,
        "trackc_uncapped_canonical_noharm_pass_query_allowed_once",
        "uncapped no-harm decision",
    )

    paths = {
        "anchor_split": args.eval_dir / "query_anchor_split_ode20.json",
        "candidate_split": args.eval_dir / "query_candidate_split_ode20.json",
        "anchor_family": args.eval_dir / "query_anchor_family_ode20.json",
        "candidate_family": args.eval_dir / "query_candidate_family_ode20.json",
        "smoke_decision": args.smoke_decision_json,
        "uncapped_decision": args.uncapped_decision_json,
    }
    boot_dir.mkdir(parents=True, exist_ok=True)
    boot_payloads = {
        "split": bootstrap_pair(
            baseline=paths["anchor_split"],
            candidate=paths["candidate_split"],
            groups=SPLIT_GROUPS,
            out_json=boot_dir / "query_split_anchor_vs_candidate.json",
            out_md=boot_dir / "query_split_anchor_vs_candidate.md",
            title="Track C Route-Focused One-Shot Query Split Diagnostic",
            n_boot=args.n_boot,
            seed=args.seed,
            python=args.python,
        ),
        "family": bootstrap_pair(
            baseline=paths["anchor_family"],
            candidate=paths["candidate_family"],
            groups=FAMILY_GROUPS,
            out_json=boot_dir / "query_family_anchor_vs_candidate.json",
            out_md=boot_dir / "query_family_anchor_vs_candidate.md",
            title="Track C Route-Focused One-Shot Query Family Diagnostic",
            n_boot=args.n_boot,
            seed=args.seed + 1,
            python=args.python,
        ),
    }
    tables = {name: index_rows(payload) for name, payload in boot_payloads.items()}
    decision = evaluate_diagnostic(tables)
    payload = {
        "inputs": {k: str(v) for k, v in paths.items()},
        "bootstrap_dir": str(boot_dir),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "report_title": str(args.report_title),
        "decision": decision,
        "tables": tables,
    }
    json_payload = dict(payload)
    json_payload["tables"] = {
        name: {f"{group}:{metric}": row for (group, metric), row in table.items()}
        for name, table in tables.items()
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"status": decision["status"], "action": decision["action"], "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
