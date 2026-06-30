#!/usr/bin/env python3
"""Synthesize manuscript/report-ready LatentFM claims, captions, and provenance."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    failure = load_json(REPORTS / "latentfm_failure_map_provenance_20260624.json")
    tables = load_json(REPORTS / "latentfm_figure_table_candidates_20260624.json")
    fig_manifest = load_json(REPORTS / "figures" / "latentfm_consolidation_20260624" / "manifest.json")
    post_locke = load_json(REPORTS / "latentfm_post_locke_portfolio_decision_20260624.json")

    trackc_best = post_locke["decision"]["current_trackc_best"]
    claim_package = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "manuscript_claim_package_ready_no_gpu",
        "boundary": {
            "reads_consolidation_outputs_only": True,
            "active_logs": False,
            "raw_canonical_or_query": False,
            "canonical_multi_selection": False,
            "training_or_inference": False,
            "gpu": False,
        },
        "final_model_statement": {
            "track_a": "Current deployable/default Track A model remains xverse_8k_anchor.",
            "track_c": (
                "Frozen support-context v2 resfilm is a diagnostic/reporting route only, "
                "not a formal multi solution."
            ),
            "track_c_route": trackc_best["route"],
            "track_c_query_multi_pearson_delta": trackc_best["query_multi_pearson_delta"],
            "track_c_query_multi_mmd_delta": trackc_best["query_multi_mmd_delta"],
            "track_c_unseen2_pearson_delta": trackc_best["unseen2_pearson_delta"],
        },
        "allowed_claims": [
            "Track A deployable/default is xverse_8k_anchor under the current strict gates.",
            "Forbidden-oracle headroom exists, but no deployable train-only gate recovered it safely.",
            "Several train-only proxies show average internal gains, but fail worst-dataset or negative-control safety.",
            "Track C support-context v2 has diagnostic aggregate support/query signal, but does not establish zero-overlap or formal multi generalization.",
            "OT minibatch pairing is wired and changes coupling; marginal-preserving assignment fixes drift but did not improve model gates.",
            "Scaling evidence supports only a narrow train-only condition-count midpoint signal and negative breadth/canonical no-harm evidence.",
        ],
        "forbidden_claims": [
            "Do not claim a new Track A model is promoted over xverse_8k_anchor.",
            "Do not claim formal multi perturbation capability is solved.",
            "Do not claim strong unseen2 multi Pearson improvement.",
            "Do not claim OT improves downstream generalization.",
            "Do not claim broad cross-dataset scaling superiority from current evidence.",
            "Do not use canonical multi or held-out Track C query as a selection signal.",
        ],
        "figure_captions": {
            "oracle_headroom_ladder": (
                "Oracle-headroom ladder for Track A internal validation. Forbidden outcome "
                "oracles show substantial recoverable signal, but strict train-only gates "
                "recover no safe deployable oracle fraction once worst-dataset and no-harm "
                "constraints are enforced."
            ),
            "gain_vs_tail_risk": (
                "Average-gain versus tail-risk map across candidate train-only mechanisms. "
                "Several mechanisms achieve positive mean perturbation-Pearson deltas, but "
                "all remain below the worst-dataset safety threshold, revealing the dominant "
                "failure mode behind non-promotion."
            ),
            "trackc_overlap_failure": (
                "Track C support-context behavior after route freeze. One-gene-overlap and "
                "positive support-proximal signals are observed, whereas zero-overlap/"
                "unseen2-like signals are weak or absent; the route is therefore diagnostic "
                "rather than a formal multi-solution claim."
            ),
            "ot_wired_no_gain": (
                "OT minibatch-pairing audit. OT materially changes cell pairing and "
                "marginal-preserving assignment addressed the observed batch-marginal drift "
                "in the audited variant, but model gates and reliability correlations do not "
                "support OT as a downstream improvement."
            ),
        },
        "figure_paths": fig_manifest["figures"],
        "provenance_checklist": [
            {
                "item": "Canonical split integrity",
                "status": "satisfied_by_policy",
                "evidence": "canonical split_seed42.json was not re-cut; canonical multi remained diagnostic/selection weight 0",
            },
            {
                "item": "Track C query isolation",
                "status": "satisfied_for_current_claim_scope",
                "evidence": "held-out query used only once for frozen diagnostic route context; no new query/GPU authorized",
            },
            {
                "item": "No active job dependency",
                "status": "satisfied",
                "evidence": "current packages read completed reports/CSVs/manifests only",
            },
            {
                "item": "Bootstrap / CI / no-harm evidence",
                "status": "represented",
                "evidence": "oracle ladder, gain-vs-tail-risk, and failure map include p_harm/CI/dataset-min where available",
            },
            {
                "item": "Negative controls",
                "status": "represented",
                "evidence": "failure map records shuffled, inverted, equal/random, and control-collapse failures by branch",
            },
            {
                "item": "Artifact provenance",
                "status": "represented",
                "evidence": "failure map, figure-table CSVs, rendered figure manifest, and source scripts are recorded",
            },
        ],
        "source_artifacts": {
            "failure_map": str(REPORTS / "LATENTFM_FAILURE_MAP_PROVENANCE_20260624.md"),
            "figure_tables": str(REPORTS / "LATENTFM_FIGURE_TABLE_CANDIDATES_20260624.md"),
            "figure_manifest": str(REPORTS / "figures" / "latentfm_consolidation_20260624" / "manifest.json"),
            "post_locke_decision": str(REPORTS / "LATENTFM_POST_LOCKE_PORTFOLIO_DECISION_20260624.md"),
        },
        "next_actions": [
            "Write a concise Results subsection around the four rendered figures.",
            "Optionally run an external read-only review of claim wording before manuscript use.",
            "Only reopen experiments if a genuinely exogenous train-only/query-blind hypothesis appears.",
        ],
    }

    json_path = REPORTS / "latentfm_manuscript_claim_package_20260624.json"
    md_path = REPORTS / "LATENTFM_MANUSCRIPT_CLAIM_PACKAGE_20260624.md"
    json_path.write_text(json.dumps(claim_package, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Manuscript Claim Package",
        "",
        "Status: `manuscript_claim_package_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Reads consolidation outputs only: failure map, figure/table candidates, figure manifest, and post-Locke portfolio decision.",
        "- Does not read active logs, raw canonical/query artifacts, canonical multi for selection, train, infer, or use GPU.",
        "",
        "## Final Model Statement",
        "",
        f"- Track A: {claim_package['final_model_statement']['track_a']}",
        f"- Track C: {claim_package['final_model_statement']['track_c']}",
        f"- Track C diagnostic route: `{trackc_best['route']}`.",
        f"- Frozen diagnostic query_multi pp/MMD deltas: `{trackc_best['query_multi_pearson_delta']:+.6f}` / `{trackc_best['query_multi_mmd_delta']:+.6f}`.",
        f"- Unseen2 Pearson delta remains weak: `{trackc_best['unseen2_pearson_delta']:+.6f}`.",
        "",
        "## Allowed Claims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in claim_package["allowed_claims"])
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend(f"- {claim}" for claim in claim_package["forbidden_claims"])
    lines.extend(["", "## Figure Captions", ""])
    for name, caption in claim_package["figure_captions"].items():
        paths = claim_package["figure_paths"][name]
        lines.extend(
            [
                f"### `{name}`",
                "",
                caption,
                "",
                f"PNG: `{paths['png']}`",
                "",
                f"SVG: `{paths['svg']}`",
                "",
            ]
        )
    lines.extend(["## Provenance Checklist", "", "| Item | Status | Evidence |", "|---|---|---|"])
    for row in claim_package["provenance_checklist"]:
        lines.append(f"| {row['item']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in claim_package["next_actions"])
    lines.extend(["", "## JSON", "", f"`{json_path}`", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
