#!/usr/bin/env python3
"""CPU gate for a DepMap dependency-conditioned Track C/tail router entry.

This audit only reads existing DepMap 24Q4 dependency artifacts and follow-up
gate outputs. It does not train, infer, select checkpoints, read canonical multi
for Track A selection, read Track C held-out query, require Chemical V2 ACK, or
use GPU.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"

ARTIFACT_JSON = REPORTS / "latentfm_depmap_24q4_dependency_artifacts_20260627.json"
ASSOCIATION_JSON = REPORTS / "latentfm_depmap_24q4_dependency_gate_20260627.json"
RESIDUAL_JSON = REPORTS / "latentfm_depmap_dependency_residual_mmd_gate_20260627.json"
MMD_MATCHED_JSON = REPORTS / "latentfm_depmap_mmd_matched_dependency_noharm_gate_20260627.json"
JOINED_ROWS = REPORTS / "depmap_24q4_dependency_gate_20260627/depmap_24q4_dependency_gate_joined_rows.csv"

OUT_JSON = REPORTS / "latentfm_depmap_dependency_conditioned_router_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_DEPMAP_DEPENDENCY_CONDITIONED_ROUTER_GATE_20260627.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def read_joined_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}

    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    group_counts = Counter(row.get("group", "") for row in rows)
    split_counts = Counter(row.get("split", "") for row in rows)
    seed_counts = Counter(row.get("seed", "") for row in rows)
    dataset_counts = Counter(row.get("dataset", "") for row in rows)
    test_single = [row for row in rows if row.get("group") == "test_single"]

    unique_test_keys = {(row.get("seed", ""), row.get("dataset", ""), row.get("condition", "")) for row in test_single}
    per_seed_dataset_counts: dict[str, dict[str, int]] = defaultdict(dict)
    high_dep_rows: dict[str, dict[str, Any]] = {}

    for seed in sorted({row.get("seed", "") for row in test_single}):
        srows = [row for row in test_single if row.get("seed") == seed]
        per_seed_dataset_counts[seed] = dict(Counter(row.get("dataset", "") for row in srows))
        values = [fnum(row.get("artifact_value")) for row in srows]
        values = [v for v in values if v is not None]
        if values:
            sorted_values = sorted(values)
            threshold = sorted_values[int(2 * (len(sorted_values) - 1) / 3)]
            high_rows = [row for row in srows if (fnum(row.get("artifact_value")) or -math.inf) > threshold]
            high_pp = [fnum(row.get("pearson_pert")) for row in high_rows]
            all_pp = [fnum(row.get("pearson_pert")) for row in srows]
            high_mmd = [fnum(row.get("test_mmd_clamped")) for row in high_rows]
            all_mmd = [fnum(row.get("test_mmd_clamped")) for row in srows]
            high_dep_rows[seed] = {
                "threshold_dependency_score": threshold,
                "high_dependency_rows": len(high_rows),
                "all_rows": len(srows),
                "high_dependency_pp_mean": mean([v for v in high_pp if v is not None]) if high_pp else None,
                "all_pp_mean": mean([v for v in all_pp if v is not None]) if all_pp else None,
                "high_dependency_mmd_mean": mean([v for v in high_mmd if v is not None]) if high_mmd else None,
                "all_mmd_mean": mean([v for v in all_mmd if v is not None]) if all_mmd else None,
            }

    return {
        "exists": True,
        "path": str(path),
        "rows": len(rows),
        "group_counts": dict(group_counts),
        "split_counts": dict(split_counts),
        "seed_counts": dict(seed_counts),
        "dataset_counts": dict(dataset_counts),
        "unique_test_single_seed_dataset_condition_rows": len(unique_test_keys),
        "test_single_rows": len(test_single),
        "per_seed_dataset_counts": dict(per_seed_dataset_counts),
        "high_dependency_summary": high_dep_rows,
    }


def association_signal_summary(association: dict[str, Any]) -> dict[str, Any]:
    summaries = [row for row in association.get("summaries", []) if row.get("group") == "test_single"]
    return {
        "status": association.get("status"),
        "reasons": association.get("reasons", []),
        "gpu_authorized": association.get("gpu_authorized"),
        "test_single": [
            {
                "seed": row.get("seed"),
                "n": row.get("n"),
                "datasets": row.get("datasets"),
                "spearman_artifact_vs_pp": row.get("spearman_artifact_vs_pp"),
                "within_dataset_shuffle_p_abs": row.get("within_dataset_shuffle_p_abs"),
                "lodo_min_signed_rho": row.get("lodo_min_signed_rho"),
                "failure_minus_ok_artifact_mean": row.get("failure_minus_ok_artifact_mean"),
            }
            for row in summaries
        ],
    }


def gate_metric_table(residual: dict[str, Any], matched: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    by_seed_residual = {row.get("seed"): row for row in residual.get("summaries", [])}
    by_seed_matched = {row.get("seed"): row for row in matched.get("summaries", [])}
    for seed in sorted(set(by_seed_residual) | set(by_seed_matched)):
        r = by_seed_residual.get(seed, {})
        m = by_seed_matched.get(seed, {})
        out.append(
            {
                "seed": seed,
                "residual_signed_rho": r.get("signed_rho_dependency_z_vs_pp_residual_mmd"),
                "residual_lodo_min": r.get("lodo_min_signed_residual_rho"),
                "residual_dep_mmd_rho": r.get("rho_dependency_z_vs_mmd"),
                "residual_high_minus_lowmid_mmd": r.get("high_minus_lowmid_mmd"),
                "matched_n": m.get("n"),
                "matched_datasets": m.get("datasets"),
                "matched_signed_rho": m.get("signed_rho_dependency_pp"),
                "matched_ci_low": m.get("signed_rho_ci_low"),
                "matched_shuffle_p": m.get("within_mmdbin_shuffle_p"),
                "matched_high_minus_low_mmd": m.get("high_minus_low_mmd"),
            }
        )
    return out


def decide(
    artifacts: dict[str, Any],
    association: dict[str, Any],
    residual: dict[str, Any],
    matched: dict[str, Any],
    joined: dict[str, Any],
) -> tuple[str, list[str], dict[str, Any]]:
    checks = {
        "source_artifact_materialized": artifacts.get("status") == "depmap_24q4_dependency_artifacts_materialized_no_gpu"
        and int(artifacts.get("materialized_rows", 0) or 0) >= 1000,
        "non_ack_source": True,
        "chemical_v2_ack_required": False,
        "chemical_v2_ack_present": False,
        "joined_rows_present": joined.get("exists") is True and int(joined.get("unique_test_single_seed_dataset_condition_rows", 0)) >= 200,
        "association_gate_passed": association.get("status") == "depmap_24q4_dependency_signal_gate_pass_needs_external_audit_no_gpu"
        and not association.get("reasons"),
        "residual_mmd_gate_passed": str(residual.get("status", "")).startswith("depmap_dependency_residual_mmd_gate_pass"),
        "mmd_matched_noharm_gate_passed": str(matched.get("status", "")).startswith(
            "depmap_mmd_matched_dependency_noharm_gate_pass"
        ),
        "candidate_level_router_delta_available": False,
        "canonical_multi_used_for_tracka_selection": False,
        "trackc_heldout_query_used": False,
        "gpu_used_or_authorized_by_inputs": any(
            bool(src.get("gpu_authorized")) for src in [artifacts, association, residual, matched]
        ),
    }

    reasons: list[str] = []
    if not checks["source_artifact_materialized"]:
        reasons.append("depmap_source_artifact_not_materialized_or_too_small")
    if not checks["joined_rows_present"]:
        reasons.append("depmap_joined_test_single_rows_missing_or_too_small")
    if not checks["association_gate_passed"]:
        reasons.append("depmap_association_gate_not_passing")
    if not checks["residual_mmd_gate_passed"]:
        reasons.append("depmap_dependency_signal_mmd_confounded")
    if not checks["mmd_matched_noharm_gate_passed"]:
        reasons.append("depmap_mmd_matched_noharm_gate_failed")
    if not checks["candidate_level_router_delta_available"]:
        reasons.append("no_candidate_level_dependency_router_delta_artifact")
    if checks["gpu_used_or_authorized_by_inputs"]:
        reasons.append("unexpected_prior_gpu_authorization_seen")

    status = (
        "depmap_dependency_conditioned_router_gate_pass_launcher_design_next_no_gpu"
        if not reasons
        else "depmap_dependency_conditioned_router_gate_fail_no_gpu"
    )
    return status, reasons, checks


def main() -> int:
    artifacts = load_json(ARTIFACT_JSON)
    association = load_json(ASSOCIATION_JSON)
    residual = load_json(RESIDUAL_JSON)
    matched = load_json(MMD_MATCHED_JSON)
    joined = read_joined_summary(JOINED_ROWS)

    status, reasons, checks = decide(artifacts, association, residual, matched, joined)
    metric_rows = gate_metric_table(residual, matched)
    association_summary = association_signal_summary(association)

    split_eval_boundary = None
    launcher_config_requirements = None
    promotion_gate = None
    fail_close_rule = None
    closing_reason = (
        "Do not form a DepMap dependency-conditioned Track C/tail router GPU smoke from the current artifact. "
        "The source is non-ACK and leakage-safe as an external failure-mechanism artifact, and the simple association "
        "gate passed, but the residual/MMD gate failed as MMD-confounded, the stricter MMD-bin matched no-harm gate "
        "failed, and there is no candidate-level router delta artifact showing pp improvement without MMD harm."
    )

    if status.startswith("depmap_dependency_conditioned_router_gate_pass"):
        split_eval_boundary = {
            "training_or_router_fit": "seed42 train rows with DepMap dependency joined by dataset/background/target gene only",
            "selection": "predeclared CPU validation only; no canonical multi Track A selection and no Track C held-out query",
            "evaluation": "frozen seed42/seed43 test_single dependency-covered rows plus exact tail/no-harm sentinels",
        }
        launcher_config_requirements = [
            "default-off dependency-conditioned router flag",
            "read-only DepMap 24Q4 dependency sidecar with explicit missing-value fallback to anchor",
            "bounded one-seed smoke, no canonical multi selection, no Track C held-out query",
            "posthoc report must stratify dependency tertiles and MMD bins before any promotion",
        ]
        promotion_gate = [
            "candidate beats anchor on high-dependency rows in pp",
            "global and high-dependency MMD no-harm",
            "seed42/seed43 agreement or independent validation",
            "LODO signed rho and within-bin shuffle controls remain positive",
        ]
        fail_close_rule = [
            "any MMD harm above predeclared threshold closes the branch",
            "bootstrap CI crossing zero closes the branch",
            "using canonical multi for selection or Track C held-out query invalidates the run",
        ]

    payload = {
        "status": status,
        "gpu_authorized": False,
        "router_gpu_smoke_authorized": status.startswith("depmap_dependency_conditioned_router_gate_pass"),
        "reasons": reasons,
        "checks": checks,
        "closing_reason": None if status.startswith("depmap_dependency_conditioned_router_gate_pass") else closing_reason,
        "split_eval_boundary": split_eval_boundary,
        "launcher_config_requirements": launcher_config_requirements,
        "promotion_gate": promotion_gate,
        "fail_close_rule": fail_close_rule,
        "association_summary": association_summary,
        "residual_status": residual.get("status"),
        "residual_reasons": residual.get("reasons", []),
        "mmd_matched_status": matched.get("status"),
        "mmd_matched_reasons": matched.get("reasons", []),
        "joined_summary": joined,
        "metric_rows": metric_rows,
        "input_files": {
            "artifact_json": str(ARTIFACT_JSON),
            "association_json": str(ASSOCIATION_JSON),
            "residual_json": str(RESIDUAL_JSON),
            "mmd_matched_json": str(MMD_MATCHED_JSON),
            "joined_rows": str(JOINED_ROWS),
        },
        "outputs": {"json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM DepMap Dependency-Conditioned Router Gate 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis over existing DepMap 24Q4 dependency artifacts and gate outputs.",
        "- Does not train, infer, select checkpoints, use GPU, use Chemical V2, require Chemical V2 ACK, read canonical multi for Track A selection, or read Track C held-out query.",
        "- DepMap value is `-CRISPRGeneEffect`; higher means stronger dependency.",
        "",
        "## Evidence",
        "",
        f"- materialized rows: `{artifacts.get('materialized_rows', 'NA')}` from scaffold `{artifacts.get('scaffold_rows', 'NA')}`; missing model rows `{artifacts.get('missing_model_rows', 'NA')}`, missing gene rows `{artifacts.get('missing_gene_rows', 'NA')}`.",
        f"- joined unique `test_single` seed/dataset/condition rows: `{joined.get('unique_test_single_seed_dataset_condition_rows', 'NA')}`.",
        f"- association status: `{association_summary.get('status')}`; reasons `{', '.join(association_summary.get('reasons') or []) or 'none'}`.",
        f"- residual/MMD status: `{residual.get('status')}`; reasons `{', '.join(residual.get('reasons') or []) or 'none'}`.",
        f"- MMD-matched no-harm status: `{matched.get('status')}`; reasons `{', '.join(matched.get('reasons') or []) or 'none'}`.",
        "",
        "| seed | residual signed rho | residual dep~MMD rho | residual high-low MMD | matched n | matched signed rho | matched CI low | matched shuffle p | matched high-low MMD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| `{row['seed']}` | {fmt(row['residual_signed_rho'])} | {fmt(row['residual_dep_mmd_rho'])} | "
            f"{fmt(row['residual_high_minus_lowmid_mmd'])} | {row.get('matched_n', 'NA')} | "
            f"{fmt(row['matched_signed_rho'])} | {fmt(row['matched_ci_low'])} | "
            f"{fmt(row['matched_shuffle_p'])} | {fmt(row['matched_high_minus_low_mmd'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
    ]
    if status.startswith("depmap_dependency_conditioned_router_gate_pass"):
        lines += [
            "A bounded dependency-conditioned router smoke can be designed, but the smoke itself is not launched by this gate.",
            "",
            "Split/eval boundary:",
            "",
            f"- `{split_eval_boundary}`",
            "",
            "Launcher/config requirements:",
            "",
            *[f"- {item}" for item in launcher_config_requirements or []],
            "",
            "Promotion gate:",
            "",
            *[f"- {item}" for item in promotion_gate or []],
            "",
            "Fail-close rule:",
            "",
            *[f"- {item}" for item in fail_close_rule or []],
        ]
    else:
        lines += [
            closing_reason,
            "",
            "Close this DepMap dependency-conditioned router entry for GPU purposes under the current evidence.",
            "Keep DepMap as failure-mechanism and stratification evidence only.",
        ]
    lines += [
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- Markdown: `{OUT_MD}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": False, "reasons": reasons, "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
