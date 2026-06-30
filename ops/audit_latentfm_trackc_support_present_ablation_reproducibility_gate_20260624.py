#!/usr/bin/env python3
"""CPU-only artifact gate for Track C support-present ablation reproducibility.

This gate intentionally does not evaluate held-out query or launch GPU work. It
checks whether the frozen v2 support-context gain is supported by the required
query-free ablation artifacts before any new Track C modeling/GPU branch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_trackc_support_present_ablation_reproducibility_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_TRACKC_SUPPORT_PRESENT_ABLATION_REPRODUCIBILITY_GATE_20260624.md"

PRIMARY_RUN = "xverse_trackc_support_context_v2_resfilm_ep050_replay2_2k_seed42"
ROBUSTNESS_RUN = "xverse_trackc_support_context_v2_residual_ep050_replay2_2k_seed42"

INPUTS = {
    "post_v2_portfolio": REPORTS / "latentfm_trackc_post_v2_portfolio_20260624.json",
    "v2_family_closure": REPORTS / "latentfm_trackc_v2_family_closure_synthesis_20260624.json",
    "primary_decision": REPORTS / f"latentfm_trackc_routed_distill_smoke_decision_{PRIMARY_RUN}.json",
    "robustness_decision": REPORTS / f"latentfm_trackc_routed_distill_smoke_decision_{ROBUSTNESS_RUN}.json",
}

ABLATION_PATTERNS = {
    "zero_support_control": [
        "*support*zero*ode20*.json",
        "*zero*support*bootstrap*.json",
    ],
    "shuffled_support_control": [
        "*support*shuffle*ode20*.json",
        "*shuffled*support*bootstrap*.json",
    ],
    "support_absent_supportval_control": [
        "*support_absent*support*ode20*.json",
        "*forced_absent*support*ode20*.json",
        "*support*absent*bootstrap*.json",
    ],
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_metric(path: Path, group: str = "test_multi") -> dict[str, Any]:
    obj = load(path)
    groups = obj.get("groups") or {}
    row = groups.get(group) or groups.get("test") or {}
    return {
        "path": str(path),
        "group": group if group in groups else "test",
        "pearson_pert": row.get("pearson_pert"),
        "test_mmd_clamped": row.get("test_mmd_clamped"),
        "n_conds": row.get("n_conds"),
        "support_context_forced_absent": obj.get("support_context_forced_absent"),
        "support_context_control": obj.get("support_context_control", "actual"),
    }


def delta_against_anchor(candidate_path: Path, anchor_path: Path) -> dict[str, Any]:
    cand = group_metric(candidate_path)
    anchor = group_metric(anchor_path)

    def diff(key: str) -> float | None:
        if cand.get(key) is None or anchor.get(key) is None:
            return None
        return float(cand[key]) - float(anchor[key])

    return {
        "candidate": cand,
        "anchor": anchor,
        "pp_delta": diff("pearson_pert"),
        "mmd_delta": diff("test_mmd_clamped"),
    }


def metric(decision: dict[str, Any], table: str, key: str) -> dict[str, Any]:
    return dict(((decision.get("tables") or {}).get(table) or {}).get(key) or {})


def support_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    obj = load(path)
    pp = metric(obj, "support_split", "test_multi:pearson_pert") or metric(
        obj, "support_split", "test:pearson_pert"
    )
    mmd = metric(obj, "support_split", "test_multi:test_mmd_clamped") or metric(
        obj, "support_split", "test:test_mmd_clamped"
    )
    return {
        "exists": True,
        "path": str(path),
        "status": (obj.get("decision") or {}).get("status"),
        "support_pp_delta": pp.get("delta_mean"),
        "support_pp_p_harm": pp.get("p_harm"),
        "support_pp_p_improvement": pp.get("p_improvement"),
        "support_pp_rows": pp.get("n_matched_conditions"),
        "support_mmd_delta": mmd.get("delta_mean"),
        "support_mmd_p_harm": mmd.get("p_harm"),
    }


def artifact_search(root_dirs: list[Path]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for label, patterns in ABLATION_PATTERNS.items():
        hits: list[str] = []
        for root in root_dirs:
            if root.exists():
                for pat in patterns:
                    hits.extend(str(p) for p in root.rglob(pat))
        found[label] = sorted(set(hits))
    return found


def control_metric_summary(root_dirs: list[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for root in root_dirs:
        anchor = root / "posthoc_eval" / "support_anchor_split_ode20.json"
        if not anchor.exists():
            continue
        eval_dir = root / "posthoc_eval"
        candidates = {
            "zero_support_control": eval_dir / "support_zero_candidate_split_ode20.json",
            "shuffled_support_control": eval_dir / "support_shuffle_condition_candidate_split_ode20.json",
            "support_absent_supportval_control": eval_dir / "support_absent_support_candidate_split_ode20.json",
        }
        root_summary: dict[str, Any] = {"root": str(root), "anchor": str(anchor), "controls": {}}
        for label, path in candidates.items():
            if path.exists():
                root_summary["controls"][label] = delta_against_anchor(path, anchor)
        if root_summary["controls"]:
            out[str(root)] = root_summary
    return out


def main() -> None:
    missing_inputs = [str(path) for path in INPUTS.values() if not path.exists()]
    primary = support_summary(INPUTS["primary_decision"])
    robustness = support_summary(INPUTS["robustness_decision"])
    root_dirs = []
    for item in (primary, robustness):
        if item.get("exists"):
            obj = load(Path(item["path"]))
            run_root = obj.get("run_root")
            if run_root:
                root_dirs.append(Path(run_root))
    controls = artifact_search(root_dirs)
    control_metrics = control_metric_summary(root_dirs)

    reasons: list[str] = []
    if missing_inputs:
        reasons.append("required_input_missing")
    if not primary.get("exists"):
        reasons.append("primary_v2_support_decision_missing")
    if (primary.get("support_pp_delta") or 0.0) < 0.04:
        reasons.append("primary_support_pp_delta_below_0p04")
    if (primary.get("support_pp_p_harm") or 1.0) > 0.20:
        reasons.append("primary_support_pp_harm_above_0p20")
    if (primary.get("support_mmd_delta") or 999.0) > 0.0:
        reasons.append("primary_support_mmd_harm")
    for label, hits in controls.items():
        if not hits:
            reasons.append(f"missing_{label}_artifact")
    actual_pp = float(primary.get("support_pp_delta") or 0.0)
    collapse_threshold = 0.04
    for root, summary in control_metrics.items():
        for label, row in (summary.get("controls") or {}).items():
            pp_delta = row.get("pp_delta")
            if pp_delta is None:
                reasons.append(f"{label}_metric_missing")
                continue
            if float(pp_delta) > collapse_threshold and (actual_pp - float(pp_delta)) < collapse_threshold:
                reasons.append(f"{label}_did_not_collapse")

    status = (
        "trackc_support_present_ablation_reproducibility_gate_pass_gpu_protocol_next"
        if not reasons
        else "trackc_support_present_ablation_reproducibility_gate_needs_ablation_artifacts_no_gpu"
    )
    gpu_authorized = not reasons
    payload = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "heldout_query_read": False,
        "canonical_multi_selection": False,
        "missing_inputs": missing_inputs,
        "decision": {
            "reasons": reasons,
            "next_action": (
                "write fail-closed ablation artifact launcher/gate before any GPU modeling smoke"
                if reasons
                else "external review then prepare one bounded support-only GPU protocol"
            ),
        },
        "primary": primary,
        "robustness": robustness,
        "control_artifacts": controls,
        "control_metrics": control_metrics,
        "control_gate": {
            "actual_support_pp_delta": actual_pp,
            "control_pp_delta_max_allowed": collapse_threshold,
            "actual_minus_control_delta_required": collapse_threshold,
        },
        "boundary": {
            "reads_raw_heldout_query": False,
            "reads_canonical_multi_for_selection": False,
            "launches_gpu": False,
            "selection_split": "safe trainselect support summaries only",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def fmt(x: Any) -> str:
        return "NA" if x is None else (f"{x:+.6f}" if isinstance(x, float) else str(x))

    control_lines = []
    for label, hits in controls.items():
        control_lines.append(f"- `{label}`: {len(hits)} artifact(s)")
        for hit in hits[:5]:
            control_lines.append(f"  - `{hit}`")
    metric_lines = []
    for root, summary in control_metrics.items():
        metric_lines.append(f"- root: `{root}`")
        for label, row in (summary.get("controls") or {}).items():
            metric_lines.append(
                f"  - `{label}` pp delta `{fmt(row.get('pp_delta'))}`, "
                f"MMD delta `{fmt(row.get('mmd_delta'))}`"
            )

    OUT_MD.write_text(
        "\n".join(
            [
                "# Track C Support-Present Ablation Reproducibility Gate",
                "",
                f"Status: `{status}`",
                f"GPU authorization: `{str(gpu_authorized)}`",
                "",
                "## Boundary",
                "",
                "- CPU-only artifact/reproducibility gate.",
                "- Reads frozen v2 support-context support summaries and local artifact inventory only.",
                "- Does not read raw held-out query, canonical multi for selection, active logs, or launch GPU work.",
                "",
                "## Primary Support-Present Evidence",
                "",
                f"- run: `{PRIMARY_RUN}`",
                f"- support pp delta: `{fmt(primary.get('support_pp_delta'))}`",
                f"- support pp p_harm: `{fmt(primary.get('support_pp_p_harm'))}`",
                f"- support MMD delta: `{fmt(primary.get('support_mmd_delta'))}`",
                "",
                "## Required Ablation Artifacts",
                "",
                *control_lines,
                "",
                "## Control Metrics",
                "",
                *(metric_lines if metric_lines else ["- pending: no control metrics found yet"]),
                "",
                "## Decision",
                "",
                f"- reasons: `{reasons}`",
                f"- next action: `{payload['decision']['next_action']}`",
                "",
                "Passing this gate would still authorize only an external review and a new",
                "support-only protocol/launcher gate. It would not authorize held-out query",
                "reuse or canonical-multi selection.",
                "",
                "## JSON",
                "",
                f"`{OUT_JSON}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "gpu_authorized": gpu_authorized, "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
