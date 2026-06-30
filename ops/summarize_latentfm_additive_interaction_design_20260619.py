#!/usr/bin/env python3
"""Summarize the next LatentFM additive-plus-interaction design gate."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT_MD = ROOT / "reports/LATENTFM_ADDITIVE_INTERACTION_MODULE_DESIGN_20260619.md"
OUT_JSON = ROOT / "reports/latentfm_additive_interaction_module_design_20260619.json"

SPLIT_AUDITS = {
    "stack": ROOT / "reports/latentfm_composition_split_audit_stack_20260618.json",
    "scldm": ROOT / "reports/latentfm_composition_split_audit_scldm_20260618.json",
    "scfoundation": ROOT / "reports/latentfm_composition_split_audit_scfoundation_20260618.json",
}
INJECTION_JSON = ROOT / "reports/latentfm_condition_prior_injection_comparison_20260619.json"
CONDITION_JSON = ROOT / "reports/latentfm_condition_prior_injection_condition_level_20260619.json"
MLP_PATH = ROOT / "CoupledFM/model/latent/models/mlp.py"
TRAIN_PATH = ROOT / "CoupledFM/model/latent/train.py"
DECOMP_TEST_PATH = ROOT / "CoupledFM/model/tests/test_latent_condition_delta_decomposition.py"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "NA"


def main() -> int:
    split_rows = []
    for encoder, path in SPLIT_AUDITS.items():
        payload = load_json(path)
        totals = payload.get("totals", {})
        split_rows.append({
            "encoder": encoder,
            "train_single": int(totals.get("train_single", 0)),
            "train_multi": int(totals.get("train_multi", 0)),
            "test_multi": int(totals.get("test_multi", 0)),
            "multi_seen": int(totals.get("multi_seen", 0)),
            "multi_unseen1": int(totals.get("multi_unseen1", 0)),
            "multi_unseen2": int(totals.get("multi_unseen2", 0)),
            "leak": int(totals.get("test_multi_with_exact_train_leak", 0)),
        })

    injection = load_json(INJECTION_JSON)
    condition = load_json(CONDITION_JSON)
    group_summary = condition.get("group_summary", [])
    persistent_failures = condition.get("persistent_failures", [])[:10]

    no_multi_train = all(row["train_multi"] == 0 for row in split_rows)
    no_exact_leak = all(row["leak"] == 0 for row in split_rows)
    decision = (
        "design_default_off_additive_plus_interaction_gate"
        if no_multi_train and no_exact_leak
        else "pause_and_reaudit_split_leakage"
    )

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "complete",
        "decision": decision,
        "split_rows": split_rows,
        "injection_state": {
            "status": injection.get("status"),
            "best": injection.get("best"),
            "repeat_candidates": sum(
                1 for row in injection.get("rows", []) if row.get("decision") == "repeat_candidate"
            ),
        },
        "n_common_conditions": condition.get("n_common_conditions"),
        "group_summary": group_summary,
        "persistent_failures": persistent_failures,
        "implementation": {
            "interaction_method_present": (
                MLP_PATH.is_file()
                and "def predict_interaction_condition_delta" in MLP_PATH.read_text(encoding="utf-8")
            ),
            "prior_additive_head_loss_present": (
                TRAIN_PATH.is_file()
                and "condition_prior_additive_delta_loss_weight" in TRAIN_PATH.read_text(encoding="utf-8")
            ),
            "decomposition_test_present": DECOMP_TEST_PATH.is_file(),
            "model_file": str(MLP_PATH),
            "train_file": str(TRAIN_PATH),
            "test_file": str(DECOMP_TEST_PATH),
        },
        "report": str(OUT_MD),
    }

    lines = [
        "# LatentFM Additive-Plus-Interaction Module Design 2026-06-19",
        "",
        f"Generated: {payload['generated']}",
        "",
        "## Decision",
        "",
        f"`{decision}`",
        "",
        "Do not launch another global scalar teacher-weight or simple head-injection sweep. The next model step should be a default-off, split-aware design that keeps the additive train-single prior identifiable while treating interaction residuals as a separately gated hypothesis.",
        "",
        "## Evidence Used",
        "",
        "- Condition-prior dose report: `reports/LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md`",
        "- Injection comparison: `reports/LATENTFM_CONDITION_PRIOR_INJECTION_COMPARISON_20260619.md`",
        "- Injection condition-level table: `reports/LATENTFM_CONDITION_PRIOR_INJECTION_CONDITION_LEVEL_20260619.md`",
        "- Top3 split audits: `reports/latentfm_composition_split_audit_{stack,scldm,scfoundation}_20260618.json`",
        "",
        "## Split Identifiability Check",
        "",
        "| Encoder | train single | train multi | test multi | seen | unseen1 | unseen2 | exact leak |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(
            f"| `{row['encoder']}` | {row['train_single']} | {row['train_multi']} | "
            f"{row['test_multi']} | {row['multi_seen']} | {row['multi_unseen1']} | "
            f"{row['multi_unseen2']} | {row['leak']} |"
        )
    lines.extend([
        "",
        "Interpretation: the formal top3 splits have no exact multi-condition training supervision and no exact test multi leak. Therefore a supervised interaction residual branch cannot honestly learn Wessels-style multi-gene interactions from multi-condition labels under the current split. Any next architecture must keep interaction claims separate from additive no-leakage evidence.",
        "",
        "## Injection Readout",
        "",
        f"- Injection comparison status: `{injection.get('status', 'NA')}`",
        f"- Best branch: `{injection.get('best', 'NA')}`",
        f"- Repeat candidates: `{payload['injection_state']['repeat_candidates']}`",
        f"- Common condition rows in condition-level analysis: `{condition.get('n_common_conditions', 'NA')}`",
        "",
        "| Dataset | Group | n | primary | no injection | injection | injection - no injection | prior best pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in group_summary:
        lines.append(
            f"| `{row.get('dataset')}` | `{row.get('group')}` | {row.get('n')} | "
            f"{fmt(row.get('mean_primary_pearson'))} | {fmt(row.get('mean_no_inject_pearson'))} | "
            f"{fmt(row.get('mean_inject_pearson'))} | {fmt(row.get('mean_inject_delta_vs_no_inject'))} | "
            f"{fmt(row.get('mean_prior_best_pp'))} |"
        )
    lines.extend([
        "",
        "## Minimal Default-Off Module Contract",
        "",
        "1. Additive atom path: reuse the existing `condition_delta_head` atom prediction and `predict_additive_condition_delta` surface. It should remain trainable only from train-single records or synthetic train-single combinations.",
        "2. Combo path: keep the existing combo perturbation projection and optional head injection, but do not promote it unless it beats the injection diagnostic on aggregate and split-specific gates.",
        "3. Interaction residual path: if implemented, initialize to zero and report it separately as `combo_delta - additive_atom_delta`. Under the current split it is a hypothesis/diagnostic branch, not a supervised multi-label learner.",
        "4. Inference baseline: formalize the train-single KNN/additive prior as a no-leakage baseline alongside LatentFM predictions, because it is the strongest current evidence that Norman additive composition is recoverable.",
        "5. Leakage guard: before any interaction residual claim, rerun `model.latent.audit_composition_split` and require `train_multi=0` and `exact leak=0` to be reported explicitly, or state that the run is no longer zero-shot.",
        "",
        "## Implementation Checkpoint",
        "",
        f"- `predict_interaction_condition_delta` present: `{payload['implementation']['interaction_method_present']}`",
        f"- `condition_prior_additive_delta_loss_weight` present: `{payload['implementation']['prior_additive_head_loss_present']}`",
        f"- Decomposition CPU test present: `{payload['implementation']['decomposition_test_present']}`",
        "- The interaction surface is diagnostic only. The prior-additive head loss is default-off and must be enabled explicitly for capped smokes.",
        "",
        "## Promotion Gate For The Next Short Smoke",
        "",
        "- Must improve over `scf_prior010_inject_e2_4k` on aggregate pp and unseen2 pp while keeping MMD within 15 percent of primary scFoundation.",
        "- Must not reduce Norman seen/unseen1/unseen2 condition-level means versus injection.",
        "- Must improve Wessels unseen2 versus injection and report Wessels seen/unseen1 tradeoffs separately.",
        "- Must preserve or improve family gene pp versus primary scFoundation and avoid worsening drug pp beyond the current diagnostic range.",
        "- Must include persistent-failure table for Mediator/chromatin/transcription-regulator combinations.",
        "",
        "## Persistent Negative-Control Cases",
        "",
        "| Dataset | Condition | Group | primary | injection | injection - no injection | prior best pp |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in persistent_failures:
        lines.append(
            f"| `{row.get('dataset')}` | `{row.get('condition')}` | `{row.get('group')}` | "
            f"{fmt(row.get('primary_pearson'))} | {fmt(row.get('inject_pearson'))} | "
            f"{fmt(row.get('inject_delta_vs_no_inject'))} | {fmt(row.get('prior_best_pp'))} |"
        )
    lines.extend([
        "",
        "## Next Concrete Action",
        "",
        "Before launching GPU work, implement or validate only the smallest default-off surfaces needed to report additive atom predictions, combo predictions, and their residual separately. If code changes are made, add CPU unit tests proving defaults are off, old checkpoints still load, synthetic combo `nperts` is correct, and the residual branch is zero-initialized or disabled unless explicitly requested.",
        "",
    ])

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
