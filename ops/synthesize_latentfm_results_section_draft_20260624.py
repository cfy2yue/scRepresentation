#!/usr/bin/env python3
"""Create a concise Results-section draft from the LatentFM claim package."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    claim = load_json(REPORTS / "latentfm_manuscript_claim_package_20260624.json")
    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "results_section_draft_ready_no_gpu",
        "boundary": {
            "reads_claim_package_only": True,
            "active_logs": False,
            "raw_canonical_or_query": False,
            "canonical_multi_selection": False,
            "training_or_inference": False,
            "gpu": False,
        },
    }
    json_path = REPORTS / "latentfm_results_section_draft_20260624.json"
    md_path = REPORTS / "LATENTFM_RESULTS_SECTION_DRAFT_20260624.md"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fs = claim["final_model_statement"]
    captions = claim["figure_captions"]
    figures = claim["figure_paths"]
    lines = [
        "# LatentFM Results Section Draft",
        "",
        "Status: `results_section_draft_ready_no_gpu`",
        "",
        "## Boundary",
        "",
        "- Draft text from the manuscript claim package only.",
        "- No active logs, raw canonical/query artifacts, canonical multi selection, training, inference, or GPU.",
        "",
        "## Draft Results Text",
        "",
        "### Strict gates expose headroom that is not yet deployable",
        "",
        "Across the current LatentFM Track A portfolio, the retained deployable/default model remains `xverse_8k_anchor`. "
        "The oracle-headroom analysis shows that forbidden outcome-aware selectors could recover substantial internal-validation signal, "
        "but none of the train-only gates recovered that signal while satisfying worst-dataset, no-harm, and negative-control constraints. "
        "This separates internal-validation headroom from a deployable model-improvement claim.",
        "",
        f"Figure: `{figures['oracle_headroom_ladder']['png']}`",
        "",
        "### Average internal gains repeatedly fail tail-risk safety",
        "",
        "Several train-only mechanisms produced positive average perturbation-Pearson deltas, including control-state support, composite safe subsets, "
        "and factorized gene-context surrogates. These gains did not pass promotion criteria because they retained unacceptable worst-dataset losses "
        "or failed shuffled, inverted, or feature-control checks. The dominant failure mode is therefore not absence of signal, but lack of robust, "
        "deployable safety across backgrounds and families.",
        "",
        f"Figure: `{figures['gain_vs_tail_risk']['png']}`",
        "",
        "### Track C support-context transfer is diagnostic, not a formal multi solution",
        "",
        f"The frozen Track C route `{fs['track_c_route']}` is reportable only as a diagnostic result: aggregate query_multi pp/MMD deltas were "
        f"`{fs['track_c_query_multi_pearson_delta']:+.6f}` / `{fs['track_c_query_multi_mmd_delta']:+.6f}` after the route was frozen. "
        f"However, unseen2 Pearson remained weak (`{fs['track_c_unseen2_pearson_delta']:+.6f}`), and query-free expansion gates showed that "
        "support-proximal or one-gene-overlap gains do not extend to zero-overlap/general multi conditions. This supports a diagnostic transfer signal, "
        "not a formal multi-perturbation capability claim.",
        "",
        f"Figure: `{figures['trackc_overlap_failure']['png']}`",
        "",
        "### OT pairing is implemented but not beneficial under current gates",
        "",
        "OT minibatch pairing is implemented in the training path and materially changes same-condition source/target coupling. The marginal-preserving assignment "
        "variant addressed the observed replacement-sampling drift in the audited variant, but random/no-OT and Hungarian OT smokes did not pass model gates, and pairing-quality "
        "features did not robustly predict response reliability. Current evidence therefore supports reporting OT as wired-but-no-gain rather than as a "
        "promoted optimization.",
        "",
        f"Figure: `{figures['ot_wired_no_gain']['png']}`",
        "",
        "### Current claim scope",
        "",
        "- Track A: `xverse_8k_anchor` is the current deployable/default model.",
        "- Track C: frozen support-context v2 resfilm is diagnostic/reporting only.",
        "- Scaling: current evidence supports only a narrow train-only condition-count midpoint signal and negative breadth/canonical no-harm evidence.",
        "- Future experiments require a genuinely exogenous train-only/query-blind gate with explicit negative controls and fail-close criteria.",
        "",
        "## Source Claim Package",
        "",
        "`/data/cyx/1030/scLatent/reports/LATENTFM_MANUSCRIPT_CLAIM_PACKAGE_20260624.md`",
        "",
        "## JSON",
        "",
        f"`{json_path}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
