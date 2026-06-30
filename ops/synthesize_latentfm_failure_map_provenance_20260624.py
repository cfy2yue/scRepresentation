#!/usr/bin/env python3
"""Build the LatentFM failure-map and provenance consolidation package."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:+.6f}"
    if isinstance(value, list):
        return "; ".join(str(x) for x in value)
    return str(value)


def main() -> None:
    portfolio_path = REPORTS / "latentfm_post_locke_portfolio_decision_20260624.json"
    portfolio = load_json(portfolio_path)

    rows = [
        {
            "branch": "Track A default",
            "hypothesis": "Current deployable model remains the safest Track A checkpoint.",
            "boundary": "Frozen completed evidence; no new selection.",
            "decisive_metric": "No recent candidate passed both train-only gate and frozen canonical no-harm.",
            "control_or_provenance_failure": "None; this is the retained default.",
            "closure_or_scope": "Claim deployable/default only: xverse_8k_anchor.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_TRACKA_STOP_MODEL_SEARCH_SYNTHESIS_20260624.md",
        },
        {
            "branch": "Track A oracle headroom",
            "hypothesis": "Deployable train-only signals recover enough forbidden-oracle headroom.",
            "boundary": "Completed internal artifacts only; no canonical/query/GPU.",
            "decisive_metric": "Forbidden oracle pp +0.066540/+0.068584; strict safe gates 0.",
            "control_or_provenance_failure": "Recovered oracle fraction 0.000 under strict safety.",
            "closure_or_scope": "Use as paper framing; no GPU authorization.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_TRACKA_IDENTIFIABILITY_CEILING_20260624.md",
        },
        {
            "branch": "Scaling / training-data strategy",
            "hypothesis": "More or differently balanced train conditions improve cross-background/family generalization.",
            "boundary": "Train-only protocol matrix; canonical only frozen no-harm after internal pass.",
            "decisive_metric": "cap60 internal +0.010495/+0.012273; frozen canonical cross pp -0.006441.",
            "control_or_provenance_failure": "Matched-budget breadth arms negative; canonical no-harm failed.",
            "closure_or_scope": "No promotion; scaling evidence is internal/negative for breadth.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_SCALING_PROTOCOL_CANONICAL_NOHARM_DECISION_20260624.md",
        },
        {
            "branch": "OT minibatch pairing",
            "hypothesis": "Better same-condition cell pairing improves flow supervision.",
            "boundary": "Wiring audit, train-only pair audits, default-off smokes; no query selection.",
            "decisive_metric": "Hungarian fixed marginals but cross-bg pp -0.022693; random pp -0.016667.",
            "control_or_provenance_failure": "Pairing-quality reliability gate contradictory; no model gain.",
            "closure_or_scope": "OT is wired-but-no-gain; no more OT sweeps without new condition-level gate.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_OT_MINIBATCH_PAIRING_SYNTHESIS_20260624.md",
        },
        {
            "branch": "Reliability-weighted robust loss",
            "hypothesis": "Condition-level measurement reliability can safely weight updates.",
            "boundary": "Train split H5 embeddings plus internal posthoc; nested LODO; no canonical/query/GPU.",
            "decisive_metric": "cap60 cross/family pp +0.008403/+0.008787.",
            "control_or_provenance_failure": "Below +0.010 gate, dataset-min -0.036668, inverted control failed.",
            "closure_or_scope": "No reliability-weighted GPU smoke.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_TRAINONLY_RELIABILITY_CONDITION_GATE_20260624.md",
        },
        {
            "branch": "Support/control/signed safe subset",
            "hypothesis": "Train-only support geometry and signed neighborhood features identify safe subsets.",
            "boundary": "Nested LODO over internal condition metrics; negative controls.",
            "decisive_metric": "Composite cross/family pp +0.028255/+0.029219.",
            "control_or_provenance_failure": "Dataset-min -0.068145/-0.066206; inverted/shuffle controls failed.",
            "closure_or_scope": "Diagnostic average signal only; no GPU.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_COMPOSITE_SAFE_SUBSET_GATE_20260624.md",
        },
        {
            "branch": "Track C support-context v2",
            "hypothesis": "Support-context transfer expands into formal multi capability.",
            "boundary": "Safe trainselect gates; one frozen query diagnostic only after route freeze.",
            "decisive_metric": "query_multi pp/MMD +0.066480/-0.006551, unseen2 pp +0.005451.",
            "control_or_provenance_failure": "Pseudo zero-overlap +0.000000; jackknife 5/17 negative; nonadditivity support pp -0.010556.",
            "closure_or_scope": "Diagnostic/reporting route only; no formal multi claim or new query/GPU.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_TRACKC_V2_FAMILY_CLOSURE_SYNTHESIS_20260624.md",
        },
        {
            "branch": "Distributional MMD-harm safety",
            "hypothesis": "Train-only distributional features route away MMD harm while retaining pp.",
            "boundary": "General-exposure internal posthoc and H5 features; nested LODO.",
            "decisive_metric": "Cross/family pp -0.009746/-0.002202; family MMD +0.001533.",
            "control_or_provenance_failure": "Shuffled/inverted/count controls did not collapse.",
            "closure_or_scope": "Close this MMD-risk routing variant.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_DISTRIBUTIONAL_MMD_HARM_GATE_20260624.md",
        },
        {
            "branch": "Deployable forensic-risk distillation",
            "hypothesis": "Full-forensics oracle decisions can be distilled into deployable covariates.",
            "boundary": "Nested LODO; held-out uses only deployable covariates.",
            "decisive_metric": "Cross/family delta vs gene +0.005048/+0.014416; cross p_harm 0.233.",
            "control_or_provenance_failure": "Below +0.025 gate; shuffled-label cross not separated.",
            "closure_or_scope": "Forensic oracle remains nondeployable diagnostic.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_TRACKA_XVERSE_FORENSIC_DISTILLATION_GATE_20260624.md",
        },
        {
            "branch": "Jiang cell-type response program",
            "hypothesis": "Cell-type-stratified Jiang response programs identify safe anchor/gene routing.",
            "boundary": "Jiang h5ad obs/obsm only plus residual-forensics proxy rows; nested leave-one-Jiang-dataset-out.",
            "decisive_metric": "Cross/family delta vs gene -0.031093/-0.031093; dataset-min -0.342784.",
            "control_or_provenance_failure": "Harm fraction 0.400; shuffled control better.",
            "closure_or_scope": "Close current Jiang cell-type program prior.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_JIANG_CELLTYPE_PROGRAM_GATE_20260624.md",
        },
        {
            "branch": "Perturbation-equivariant prototype",
            "hypothesis": "Same-gene cross-dataset prototype deltas reveal a new representation objective.",
            "boundary": "Train split H5 deltas/metadata and internal means; nested LODO.",
            "decisive_metric": "Cross/family pp -0.014761/-0.012321.",
            "control_or_provenance_failure": "Dataset-min -0.711791; same-gene support sparse.",
            "closure_or_scope": "No representation/prototype GPU reopen.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_PERTURBATION_EQUIVARIANT_PROTOTYPE_GATE_20260624.md",
        },
        {
            "branch": "Factorized gene x context",
            "hypothesis": "Factorized gene plus context surrogate identifies a deployable interaction objective.",
            "boundary": "Train H5 deltas/control means and metadata; nested LODO and shuffle controls.",
            "decisive_metric": "Average cross/family pp +0.075737/+0.077607.",
            "control_or_provenance_failure": "Dataset-min -0.608195; gene/context/delta shuffle controls did not collapse.",
            "closure_or_scope": "Average signal is unsafe/nonmechanistic; no GPU.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_FACTORIZED_GENE_CONTEXT_GATE_20260624.md",
        },
    ]

    claim_scope = [
        {
            "claim": "Track A deployable/default model",
            "allowed": "xverse_8k_anchor remains current default",
            "not_allowed": "Do not claim newer scaling/reliability/routing branches are promoted.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_TRACKA_STOP_MODEL_SEARCH_SYNTHESIS_20260624.md",
        },
        {
            "claim": "Track C frozen support-context v2 diagnostic",
            "allowed": "Report frozen route aggregate query_multi pp/MMD +0.066480/-0.006551 as diagnostic context.",
            "not_allowed": "Do not claim formal multi solved, strong unseen2 Pearson, or query-tuned route.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_TRACKC_V2_FAMILY_CLOSURE_SYNTHESIS_20260624.md",
        },
        {
            "claim": "OT minibatch pairing",
            "allowed": "Report OT is wired and changes coupling; marginal-preserving Hungarian fixed drift.",
            "not_allowed": "Do not claim OT improves model generalization.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_OT_MINIBATCH_PAIRING_SYNTHESIS_20260624.md",
        },
        {
            "claim": "Scaling effect",
            "allowed": "Report a narrow train-only condition-count midpoint signal and negative matched-budget breadth/canonical no-harm evidence.",
            "not_allowed": "Do not claim broad cross-dataset scaling superiority from current results.",
            "evidence": "/data/cyx/1030/scLatent/reports/LATENTFM_SCALING_PROTOCOL_CANONICAL_NOHARM_DECISION_20260624.md",
        },
    ]

    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "boundary": {
            "source": str(portfolio_path),
            "read_completed_reports_only": True,
            "active_logs": False,
            "raw_query_or_canonical_selection": False,
            "gpu": False,
        },
        "status": "failure_map_provenance_ready_no_gpu",
        "rows": rows,
        "claim_scope": claim_scope,
        "next_artifacts": [
            "oracle-headroom ladder figure/table",
            "average-gain-vs-tail-risk table",
            "Track C one-gene-overlap vs zero-overlap panel",
            "OT wired-but-no-gain panel",
        ],
    }

    json_path = REPORTS / "latentfm_failure_map_provenance_20260624.json"
    csv_path = REPORTS / "latentfm_failure_map_provenance_20260624.csv"
    md_path = REPORTS / "LATENTFM_FAILURE_MAP_PROVENANCE_20260624.md"

    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "branch",
        "hypothesis",
        "boundary",
        "decisive_metric",
        "control_or_provenance_failure",
        "closure_or_scope",
        "evidence",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# LatentFM Failure Map And Provenance",
        "",
        "Status: `failure_map_provenance_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Consolidation only; reads completed reports and the post-Locke portfolio synthesis.",
        "- No active logs, raw canonical/query artifacts, canonical multi selection, GPU, training, or inference.",
        "",
        "## Failure Map",
        "",
        "| Branch | Hypothesis | Decisive metric | Failure / scope | Evidence |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["branch"],
                    row["hypothesis"],
                    row["decisive_metric"],
                    f"{row['control_or_provenance_failure']} {row['closure_or_scope']}",
                    f"`{row['evidence']}`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Claim Scope", "", "| Claim | Allowed | Not allowed | Evidence |", "|---|---|---|---|"])
    for row in claim_scope:
        lines.append(
            f"| {row['claim']} | {row['allowed']} | {row['not_allowed']} | `{row['evidence']}` |"
        )

    lines.extend(
        [
            "",
            "## Next Figure/Table Candidates",
            "",
            "- Oracle-headroom ladder.",
            "- Average gain versus tail-risk table.",
            "- Track C one-gene-overlap versus zero-overlap panel.",
            "- OT wired-but-no-gain panel.",
            "",
            "## Machine-Readable Outputs",
            "",
            f"- `{json_path}`",
            f"- `{csv_path}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
