#!/usr/bin/env python3
"""Track A exogenous-covariate discovery closure gate.

This CPU-only synthesis checks whether any current Track A exogenous covariate
family remains eligible to reopen GPU search. It does not launch training or
read canonical multi / Track C query artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_tracka_exogenous_covariate_discovery_closure_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKA_EXOGENOUS_COVARIATE_DISCOVERY_CLOSURE_20260624.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"parse_error": str(exc)}


def status_of(path: Path, md_path: Path | None = None) -> str:
    obj = load_json(path)
    if obj.get("missing"):
        if md_path and md_path.exists():
            text = md_path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines()[:20]:
                if line.lower().startswith("status:"):
                    return line.split(":", 1)[1].strip().strip("`")
            return "md_present_json_missing"
        return "missing"
    if obj.get("parse_error"):
        return f"parse_error:{obj['parse_error']}"
    decision = obj.get("decision") if isinstance(obj.get("decision"), dict) else {}
    return str(obj.get("status") or decision.get("status") or "unknown")


def nested_get(obj: dict[str, Any], path: list[Any]) -> Any:
    cur: Any = obj
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
            cur = cur[key]
        else:
            return None
    return cur


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def main() -> int:
    candidates: list[dict[str, Any]] = []

    def add(
        family: str,
        json_name: str,
        md_name: str,
        novelty: str,
        gate_summary: str,
        duplicate_of_closed: str,
        key_paths: list[list[Any]] | None = None,
    ) -> None:
        jpath = REPORTS / json_name
        mpath = REPORTS / md_name
        obj = load_json(jpath)
        status = status_of(jpath, mpath)
        key_metrics = {}
        for i, p in enumerate(key_paths or []):
            key_metrics["metric_" + str(i + 1)] = nested_get(obj, p)
        gpu_ready = ("pass" in status and "fail" not in status and "no_gpu" not in status and "stop" not in status)
        candidates.append(
            {
                "family": family,
                "status": status,
                "gpu_ready": gpu_ready,
                "json": str(jpath),
                "md": str(mpath),
                "novelty": novelty,
                "gate_summary": gate_summary,
                "duplicate_or_blocker": duplicate_of_closed,
                "key_metrics": key_metrics,
            }
        )

    add(
        "forbidden-oracle headroom / stop model search",
        "latentfm_tracka_identifiability_ceiling_20260624.json",
        "LATENTFM_TRACKA_IDENTIFIABILITY_CEILING_20260624.md",
        "Quantifies headroom but not deployable signal.",
        "Forbidden oracle cross/family gains exist, but recovered deployable oracle fraction is 0.",
        "Not launchable; oracle/outcome-like information is forbidden.",
        [["decision", "status"]],
    )
    add(
        "response covariate deployability",
        "latentfm_tracka_xverse_response_covariate_deployability_gate_20260624.json",
        "LATENTFM_TRACKA_XVERSE_RESPONSE_COVARIATE_DEPLOYABILITY_GATE_20260624.md",
        "Deployable subset of response-forensics covariates.",
        "No deployable variant passed gain/tail/control criteria.",
        "Closed response-router/proxy family.",
        [["decision", "reasons"]],
    )
    add(
        "forensic risk distillation",
        "latentfm_tracka_xverse_forensic_distillation_gate_20260624.json",
        "LATENTFM_TRACKA_XVERSE_FORENSIC_DISTILLATION_GATE_20260624.md",
        "Distills nondeployable forensic decisions into deployable covariates.",
        "Main deployable distillation had too-small gains and controls not separated.",
        "Closed distillation of target-derived forensic oracle.",
        [["decision", "reasons"]],
    )
    add(
        "deployable risk overlay",
        "latentfm_tracka_xverse_deployable_risk_overlay_gate_20260624.json",
        "LATENTFM_TRACKA_XVERSE_DEPLOYABLE_RISK_OVERLAY_GATE_20260624.md",
        "Risk overlay on response router using deployable features.",
        "Only nondeployable full-forensics overlay passed; deployable overlays missed gain.",
        "Closed response-router risk overlay family.",
        [["decision", "reasons"]],
    )
    add(
        "external/source prior portfolio",
        "latentfm_tracka_external_response_prior_portfolio_20260624.json",
        "LATENTFM_TRACKA_EXTERNAL_RESPONSE_PRIOR_PORTFOLIO_20260624.md",
        "Curated external priors: GOA, Reactome, OmniPath, cytokine, CORUM, local reliability.",
        "Portfolio synthesis closed all current external/source priors.",
        "No new independent local pathway/ontology/gene-set/complex/source prior remains in current artifacts.",
        [],
    )
    add(
        "CORUM complex reliability",
        "latentfm_tracka_corum_complex_reliability_gate_20260624.json",
        "LATENTFM_TRACKA_CORUM_COMPLEX_RELIABILITY_GATE_20260624.md",
        "Complex-membership exogenous prior.",
        "Degenerated to always-gene/all-flagged behavior; no delta vs gene.",
        "Closed complex-prior family.",
        [["decision", "reasons"]],
    )
    add(
        "perturbation-equivariant prototype",
        "latentfm_perturbation_equivariant_prototype_gate_20260624.json",
        "LATENTFM_PERTURBATION_EQUIVARIANT_PROTOTYPE_GATE_20260624.md",
        "Same-gene equivariant prototype from train split deltas.",
        "Cross/family pp negative and dataset tails severe.",
        "Closed prototype reopen path.",
        [["decision", "cross_mean_pp_delta"], ["decision", "cross_dataset_min"]],
    )
    add(
        "factorized gene x context",
        "latentfm_factorized_gene_context_gate_20260624.json",
        "LATENTFM_FACTORIZED_GENE_CONTEXT_GATE_20260624.md",
        "Factorized gene/context surrogate with nested LODO.",
        "Mean signal positive but dataset tails severe and shuffle controls did not collapse.",
        "Closed factorized surrogate path.",
        [["decision", "cross_mean_pp_delta"], ["decision", "cross_dataset_min"]],
    )
    add(
        "risk-constrained update proxy",
        "latentfm_xverse_risk_constrained_update_cpu_gate_20260624.json",
        "LATENTFM_XVERSE_RISK_CONSTRAINED_UPDATE_CPU_GATE_20260624.md",
        "Anchor-relative update constraint proxy.",
        "Deployable LODO was no-op; only nondeployable oracle policy passed; inverted control did not collapse.",
        "Closed current risk-constrained update route.",
        [["deployable_lodo", "mean_pp_delta_vs_anchor"], ["all_policy_lodo", "mean_pp_delta_vs_anchor"], ["decision_reasons"]],
    )
    add(
        "risk-conditioned general exposure hook",
        "latentfm_risk_conditioned_general_exposure_feasibility_20260624.json",
        "LATENTFM_RISK_CONDITIONED_GENERAL_EXPOSURE_FEASIBILITY_20260624.md",
        "Dataset-risk-conditioned MMD/replay hook idea.",
        "Feasible only after hook support; subsequent risk-conditioned evidence failed broad risk stratification.",
        "Not a current GPU candidate; hook/code gate would be needed before any smoke.",
        [["decision", "status"], ["current_hook_support", "launcher_dataset_risk_filter_present"]],
    )
    add(
        "risk row response preservation",
        "latentfm_risk_row_response_preservation_gate_20260624.json",
        "LATENTFM_RISK_ROW_RESPONSE_PRESERVATION_GATE_20260624.md",
        "Tests whether exact risk-row branch preserves nonrisk response.",
        "Nonrisk response not preserved and risk rows retain joint pp/MMD harm.",
        "Closed risk-row continuation for this evidence.",
        [["decision", "reasons"]],
    )

    gpu_ready = [c for c in candidates if c["gpu_ready"]]
    reasons = []
    if gpu_ready:
        reasons.append("unexpected_gpu_ready_candidate_requires_manual_review")
    else:
        reasons.append("no_tracka_exogenous_covariate_family_gpu_ready")
    if any("pass" in c["status"] and "fail" not in c["status"] for c in candidates):
        reasons.append("one_or_more_nonfinal_pass_like_statuses_require_external_review")
    reasons.append("all_current_positive_signals_are_oracle_like_non_deployable_tail_unsafe_or_control_unsafe")
    status = "tracka_exogenous_covariate_discovery_closure_no_gpu"

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "reads_completed_reports_only": True,
            "uses_for_current_checkpoint_selection": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "summary": {
            "n_families": len(candidates),
            "n_gpu_ready": len(gpu_ready),
            "gpu_ready_families": [c["family"] for c in gpu_ready],
        },
        "decision": {
            "gpu_next_action": "none",
            "next_cpu_action": "only a genuinely new information source or code hook not listed here can reopen Track A",
        },
        "reasons": reasons,
        "families": candidates,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Track A Exogenous-Covariate Discovery Closure",
        "",
        f"Status: `{status}`",
        "",
        "## Boundary",
        "",
        "- CPU-only synthesis of completed Track A covariate/prior/hook gates.",
        "- Does not select a checkpoint, read canonical multi, read Track C query, train, infer, or use GPU.",
        "",
        "## Summary",
        "",
        f"- families audited: `{len(candidates)}`",
        f"- GPU-ready families: `{len(gpu_ready)}`",
        "",
        "| family | status | GPU-ready | blocker |",
        "|---|---|---:|---|",
    ]
    for row in candidates:
        lines.append(
            f"| `{row['family']}` | `{row['status']}` | `{str(row['gpu_ready']).lower()}` | {row['duplicate_or_blocker']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reasons: `{reasons}`",
            "- GPU authorized: `False`",
            "- Current default remains `xverse_8k_anchor`.",
            "- Track A can reopen only with a genuinely new exogenous information source or a code hook not already covered here, followed by a strict CPU gate.",
            "",
            "## JSON",
            "",
            f"`{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "gpu_authorized": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
