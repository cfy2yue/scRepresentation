#!/usr/bin/env python3
"""Generate a lightweight status dashboard for the /data/cyx/1030/scLatent workspace."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
OUT = ROOT / "reports/WORKSPACE_STATUS.md"
PERTRESID_RUN_ROOT = ROOT / "runs/latentfm_condition_delta_pertresid_smoke_20260617"
PERTRESID_LOG = (
    ROOT
    / "logs/latentfm_condition_delta_pertresid_smoke/"
    "20260617_scfoundation_conddelta005_pertresidtarget_comp006_endpoint5_3k_smoke.log"
)
SCHEDULED_ONE_SHOT_ROOT = PERTRESID_RUN_ROOT
FULLCAP_RUN_ROOT = ROOT / "runs/latentfm_fullcap_posthoc_20260618"
FULLCAP_LOG = ROOT / "logs/latentfm_fullcap_posthoc_20260618/run.log"
FULLCAP_REPORT = ROOT / "reports/LATENTFM_FULLCAP_POSTHOC_REPORT_20260618.md"
STACK_GUARD_RUN_ROOT = ROOT / "runs/latentfm_stack_composite_selection_20260618"
STACK_GUARD_LOG_ROOT = ROOT / "logs/latentfm_stack_composite_selection_20260618"
REL_RUN_ROOT = ROOT / "runs/latentfm_scfoundation_relational_residual_20260619"
REL_LOG_ROOT = ROOT / "logs/latentfm_scfoundation_relational_residual_20260619"
REL_REPORT = ROOT / "reports/LATENTFM_SCFOUNDATION_RELATIONAL_RESIDUAL_REPORT_20260619.md"
STRATEGY_RUN_ROOT = ROOT / "runs/latentfm_strategy_probe_20260619"
STRATEGY_POSTHOC_RUN_ROOT = ROOT / "runs/latentfm_strategy_probe_posthoc_20260619"
STRATEGY_EXPANDED_RUN_ROOT = ROOT / "runs/latentfm_strategy_probe_expanded_20260619"
STRATEGY_ALL_DECISION_RUN_ROOT = ROOT / "runs/latentfm_strategy_all_decision_20260619"
STRATEGY_ALL_DECISION_JSON = ROOT / "reports/latentfm_strategy_all_decision_20260619.json"
CONDITION_PRIOR_PROBE_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_teacher_probe_20260619"
CONDITION_PRIOR_PRIOR002_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_teacher_prior002_20260619"
CONDITION_PRIOR_PRIOR010_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_teacher_prior010_20260619"
CONDITION_PRIOR_POSTHOC_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_teacher_posthoc_20260619"
CONDITION_PRIOR_SISTER_POSTHOC_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_teacher_sister_posthoc_20260619"
CONDITION_PRIOR_DOSE_SUMMARY_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_teacher_dose_summary_20260619"
CONDITION_PRIOR_ONE_SHOT_RUN_ROOT = ROOT / "runs/condition_prior_dose_one_shot_1318_20260619"
CONDITION_PRIOR_ONE_SHOT_1350_RUN_ROOT = ROOT / "runs/condition_prior_dose_one_shot_1350_20260619"
CONDITION_PRIOR_INJECTION_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_teacher_injection_20260619"
CONDITION_PRIOR_INJECTION_POSTHOC_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_teacher_injection_posthoc_20260619"
CONDITION_PRIOR_INJECTION_SUMMARY_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_teacher_injection_summary_20260619"
CONDITION_PRIOR_ADDITIVE_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_additive_head_20260619"
CONDITION_PRIOR_ADDITIVE_POSTHOC_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_additive_head_posthoc_20260619"
CONDITION_PRIOR_ADDITIVE_SUMMARY_RUN_ROOT = ROOT / "runs/latentfm_condition_prior_additive_head_summary_20260619"
CONDITION_PRIOR_ADDITIVE_ONE_SHOT_RUN_ROOT = (
    ROOT / "runs/latentfm_condition_prior_additive_head_one_shot_1656_20260619"
)
CONDITION_PRIOR_ADDITIVE_ONE_SHOT_1730_RUN_ROOT = (
    ROOT / "runs/latentfm_condition_prior_additive_head_one_shot_1730_20260619"
)
CONDITION_PRIOR_ADDITIVE_ONE_SHOT_1810_RUN_ROOT = (
    ROOT / "runs/latentfm_condition_prior_additive_head_one_shot_1810_20260619"
)
CONDITION_PRIOR_DOSE_JSON = ROOT / "reports/latentfm_condition_prior_teacher_dose_20260619.json"
CONDITION_PRIOR_DOSE_FIGURE_META = ROOT / "reports/latentfm_condition_prior_teacher_dose_20260619.figure_meta.json"
CONDITION_PRIOR_INJECTION_JSON = ROOT / "reports/latentfm_condition_prior_injection_comparison_20260619.json"
CONDITION_PRIOR_INJECTION_CONDITION_LEVEL_JSON = (
    ROOT / "reports/latentfm_condition_prior_injection_condition_level_20260619.json"
)
CONDITION_PRIOR_ADDITIVE_JSON = (
    ROOT / "reports/latentfm_condition_prior_additive_head_comparison_20260619.json"
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def git_status(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--short", "--branch"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        return f"error: {exc.output.strip()}"
    return out or "clean"


def latest_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "log", "-1", "--oneline"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        return f"error: {exc.output.strip()}"


def read_text(path: Path) -> str:
    if not path.is_file():
        return "NA"
    return path.read_text(encoding="utf-8", errors="replace").strip() or "NA"


def status_summary(path: Path) -> str:
    if not path.is_file():
        return "NA"
    text = path.read_text(encoding="utf-8", errors="replace")
    fields: dict[str, str] = {}
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().lower() in {"## current status", "## current status:"}:
            for follow in lines[idx + 1 :]:
                follow = follow.strip()
                if follow and not follow.startswith("#"):
                    fields.setdefault("current_status", follow.removesuffix("."))
                    break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    parts = []
    for key in ("status", "current_status", "gpu", "pert_residual_relational_loss_weight", "exit_code", "finished_at"):
        if key in fields:
            parts.append(f"{key}={fields[key]}")
    return ", ".join(parts) if parts else text.splitlines()[0]


def run_marker_summary(run_root: Path) -> str:
    exit_code = read_text(run_root / "EXIT_CODE")
    finished = read_text(run_root / "FINISHED")
    if exit_code != "NA" or finished != "NA":
        return f"exit_code={exit_code}; finished={finished}"
    return status_summary(run_root / "RUN_STATUS.md")


def four_run_launch_summary(launch_status: Path, posthoc_status: Path) -> str:
    launch = status_summary(launch_status)
    posthoc = status_summary(posthoc_status)
    if "status=finished" in posthoc:
        return f"superseded_by_posthoc_finished (launch file says {launch})"
    return launch


def strategy_decision_state(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    missing = payload.get("missing_inputs") or []
    rows = payload.get("rows") or []
    if missing:
        return {
            "status": "pending",
            "rows": len(rows),
            "repeat_candidates": 0,
            "best": "NA",
            "message": f"pending; missing inputs: {', '.join(map(str, missing))}",
        }
    if not rows:
        return {
            "status": "pending",
            "rows": 0,
            "repeat_candidates": 0,
            "best": "NA",
            "message": "pending; no candidate rows",
        }
    repeat = [r for r in rows if str(r.get("decision", "")) == "repeat_candidate"]
    def score(row: dict[str, Any]) -> float:
        try:
            return float(row.get("score"))
        except (TypeError, ValueError):
            return float("-inf")

    best = max(rows, key=score)
    best_name = str(best.get("run", "NA"))
    status = "complete_with_repeat_candidate" if repeat else "complete_no_repeat_candidate"
    return {
        "status": status,
        "rows": len(rows),
        "repeat_candidates": len(repeat),
        "best": best_name,
        "message": f"{status}; rows={len(rows)}; best={best_name}",
    }


def condition_prior_dose_state(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "pending; summary JSON missing"
    rows = payload.get("rows") or []
    status = str(payload.get("status", "pending"))
    best_raw = payload.get("best") or "NA"
    if isinstance(best_raw, dict):
        best = str(best_raw.get("run", "NA"))
    else:
        best = str(best_raw)
    complete = sum(1 for row in rows if row.get("complete"))
    repeat = sum(1 for row in rows if str(row.get("decision")) == "repeat_candidate")
    return f"{status}; complete={complete}/{len(rows)}; repeat_candidates={repeat}; best={best}"


def injection_state(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "pending; comparison JSON missing"
    rows = payload.get("rows") or []
    status = str(payload.get("status", "pending"))
    best = str(payload.get("best") or "NA")
    repeat = sum(1 for row in rows if str(row.get("decision")) == "repeat_candidate")
    decisions = ",".join(sorted({str(row.get("decision", "NA")) for row in rows})) or "NA"
    return f"{status}; rows={len(rows)}; repeat_candidates={repeat}; best={best}; decisions={decisions}"


def file_state(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_dir():
        return "present-dir"
    return f"present-file ({path.stat().st_size} bytes)"


def _scheduled_stem(path: Path) -> str:
    stem = path.name.removesuffix("_EXIT_CODE").removesuffix("_STATUS.md")
    if stem.startswith("SCHEDULED_LAUNCH"):
        stem = stem.replace("SCHEDULED_LAUNCH", "SCHEDULED_STACK_LAUNCH", 1)
    return stem


def _scheduled_status_file(run_root: Path, stem: str) -> Path:
    direct = run_root / f"{stem}_STATUS.md"
    if direct.is_file():
        return direct
    if stem.startswith("SCHEDULED_STACK_LAUNCH"):
        alt = run_root / f"{stem.replace('SCHEDULED_STACK_LAUNCH', 'SCHEDULED_LAUNCH', 1)}_STATUS.md"
        if alt.is_file():
            return alt
    return direct


def scheduled_marker_rows(run_root: Path) -> list[str]:
    stems = {
        _scheduled_stem(path)
        for path in run_root.glob("SCHEDULED_*_EXIT_CODE")
    }
    stems.update(
        _scheduled_stem(path)
        for path in run_root.glob("SCHEDULED_*_STATUS.md")
    )
    markers = sorted(stems)
    rows: list[str] = []
    for stem in markers:
        marker = run_root / f"{stem}_EXIT_CODE"
        finished = run_root / f"{stem}_FINISHED"
        status_file = _scheduled_status_file(run_root, stem)
        rows.append(
            f"| `{stem}` | `{read_text(marker)}` | `{read_text(finished)}` | `{file_state(status_file)}` |"
        )
    return rows


def chempert_only_models(figure_manifest: dict[str, Any]) -> list[str]:
    model_coverage = figure_manifest.get("model_coverage") or {}
    explicit = model_coverage.get("chempert_only_models") or figure_manifest.get("chempert_only_models")
    if explicit:
        return sorted(str(x) for x in explicit)
    by_model = model_coverage.get("by_model") or {}
    out = []
    for model, info in by_model.items():
        cats = set(info.get("categories") or [])
        if cats == {"chempert"}:
            out.append(str(model))
    return sorted(out)


def main() -> int:
    decision = load_json(ROOT / "reports/latentfm_followup_decision_status_20260617.json") or {}
    fullcap_decision = load_json(ROOT / "reports/latentfm_fullcap_decision_status_20260618.json") or {}
    relational_decision = (
        load_json(ROOT / "reports/latentfm_scfoundation_relational_residual_decision_20260619.json")
        or {}
    )
    strategy_decision = strategy_decision_state(load_json(STRATEGY_ALL_DECISION_JSON))
    condition_prior_payload = load_json(CONDITION_PRIOR_DOSE_JSON)
    condition_prior_dose = condition_prior_dose_state(condition_prior_payload)
    condition_prior_dose_status = str((condition_prior_payload or {}).get("status", "pending"))
    condition_prior_repeat_count = sum(
        1
        for row in (condition_prior_payload or {}).get("rows", [])
        if str(row.get("decision")) == "repeat_candidate"
    )
    condition_prior_dose_figure = load_json(CONDITION_PRIOR_DOSE_FIGURE_META) or {}
    injection_payload = load_json(CONDITION_PRIOR_INJECTION_JSON)
    condition_prior_injection = injection_state(injection_payload)
    injection_status = str((injection_payload or {}).get("status", "pending"))
    injection_repeat_count = sum(
        1
        for row in (injection_payload or {}).get("rows", [])
        if str(row.get("decision")) == "repeat_candidate"
    )
    injection_condition_level = load_json(CONDITION_PRIOR_INJECTION_CONDITION_LEVEL_JSON) or {}
    additive_payload = load_json(CONDITION_PRIOR_ADDITIVE_JSON)
    condition_prior_additive = injection_state(additive_payload)
    additive_status = str((additive_payload or {}).get("status", "pending"))
    additive_repeat_count = sum(
        1
        for row in (additive_payload or {}).get("rows", [])
        if str(row.get("decision")) == "repeat_candidate"
    )
    figure_manifest = load_json(ROOT / "scFM_output/figures/manifest.json") or {}
    manuscript_figure_manifest = load_json(ROOT / "scFM_output/figures_manuscript/manifest.json") or {}
    dataset_manifest = ROOT / "reports/dataset_training_package_manifest.tsv"
    pertresid_out = (
        ROOT
        / "CoupledFM/output/latentfm_runs/full_scfoundation_alignment_smoke/"
        "20260617_scfoundation_conddelta005_pertresidtarget_comp006_endpoint5_3k_smoke"
    )
    posthoc_eval = pertresid_out / "posthoc_eval"

    fullcap_status = str(fullcap_decision.get("status", "NA"))
    relational_status = str(relational_decision.get("status", "NA"))
    strategy_status_text = "\n".join(
        read_text(path / "RUN_STATUS.md").lower()
        for path in (STRATEGY_RUN_ROOT, STRATEGY_POSTHOC_RUN_ROOT, STRATEGY_EXPANDED_RUN_ROOT)
    )
    strategy_active = "running" in strategy_status_text or "waiting_for_training" in strategy_status_text
    if str(strategy_decision["status"]).startswith("complete"):
        if condition_prior_dose_status == "complete" and condition_prior_repeat_count == 0:
            condition_prior_lines = [
                f"- Condition-prior dose summary state: `{condition_prior_dose}`.",
                "- Completed dose table has no strict repeat candidate; all completed branches are diagnostic candidates.",
                f"- Condition-prior injection summary state: `{condition_prior_injection}`.",
            ]
            if injection_status == "complete" and injection_repeat_count == 0:
                condition_prior_lines.extend([
                    "- Explicit head injection is also diagnostic only: it improves MMD/overall pp/unseen2 slightly, but still has no strict repeat candidate.",
                    "- Best provisional branch is useful for mechanism analysis only. Do not launch another broad scalar teacher-weight or head-injection sweep.",
                ])
                if additive_status == "complete" and additive_repeat_count > 0:
                    condition_prior_lines.append(
                        "- Additive-head supervision produced at least one repeat candidate; plan exactly one repeat seed before any manuscript claim."
                    )
                elif additive_status == "complete":
                    condition_prior_lines.append(
                        "- Additive-head supervision completed without a strict repeat candidate; pivot to split-aware additive-plus-interaction architecture design."
                    )
                else:
                    condition_prior_lines.extend([
                        f"- Condition-prior additive-head smoke state: `{condition_prior_additive}`.",
                        "- Current active LatentFM task: wait for the single additive-head smoke/posthoc/summary chain; do not launch another broad sweep.",
                    ])
            elif injection_status == "complete" and injection_repeat_count > 0:
                condition_prior_lines.append(
                    "- At least one injection branch is a repeat candidate. Plan exactly one repeat seed before any manuscript claim."
                )
            else:
                condition_prior_lines.extend([
                    "- The targeted prior-injection follow-up is still pending or incomplete; wait for its posthoc/summary artifacts before a new GPU branch.",
                    "- Best provisional dose branch is useful for mechanism analysis only. Do not launch another broad scalar teacher-weight sweep.",
                ])
        elif condition_prior_dose_status == "complete" and condition_prior_repeat_count > 0:
            condition_prior_lines = [
                f"- Condition-prior dose summary state: `{condition_prior_dose}`.",
                "- At least one dose is a repeat candidate. Plan exactly one repeat seed for the best candidate before any manuscript claim.",
            ]
        else:
            condition_prior_lines = [
                "- Condition-prior teacher smoke passed. Three capped scFoundation dose probes are now active or queued for posthoc: weights 0.02, 0.05, and 0.10.",
                f"- Condition-prior dose summary state: `{condition_prior_dose}`.",
                "- Next after the dose posthoc finishes: inspect the dose-response report before any repeat seed or full formal run.",
            ]
        active_next_action = [
            "- Active 12-run LatentFM strategy decision is complete.",
            f"- Strategy decision state: `{strategy_decision['message']}`.",
            "- No branch should be scaled directly unless it is explicitly marked `repeat_candidate`.",
            "- Prior-correction diagnostics are complete: train-single KNN/additive residual priors strongly improve multi-condition pp without collapsing pc, especially for scFoundation on Norman.",
            "- Current result supports a representation/objective tradeoff: Stack is less bad before correction on some multi-unseen residuals, while scFoundation keeps stronger MMD/gene-family signal and benefits more from explicit composition priors.",
            *condition_prior_lines,
            "- Do not launch another broad scalar-weight sweep as a formal run.",
            "- Before any new low-util strategy training, use `ops/select_available_gpus.py` for the 3-sample shared-GPU plus CPU/RAM check.",
        ]
    elif strategy_active:
        four_posthoc = status_summary(STRATEGY_POSTHOC_RUN_ROOT / "RUN_STATUS.md")
        expanded_posthoc = status_summary(STRATEGY_EXPANDED_RUN_ROOT / "RUN_STATUS.md")
        if "status=finished" in four_posthoc and "status=finished" not in expanded_posthoc:
            strategy_wait = "- Four-run posthoc is complete with only diagnostic candidates so far; wait for expanded posthoc and the full 12-run combined decision before launching more training."
        elif "status=finished" in four_posthoc and "status=finished" in expanded_posthoc:
            strategy_wait = "- Strategy posthoc jobs are finished; inspect the combined decision report before launching more training."
        else:
            strategy_wait = "- Wait for the strategy posthoc watchers to finish, then select whether any branch merits repeat/deepening."
        active_next_action = [
            "- Active LatentFM direction is the endpoint/composition/condition-delta strategy probe block.",
            "- The relational-residual branch has already been rejected as mainline; do not continue tuning `rel_w` directly.",
            strategy_wait,
            "- Respect AGENTS.md: long GPU jobs stay detached and progress checks should be at >=30-minute windows unless status files report failure.",
            "- Before launching further low-util strategy jobs, use `ops/select_available_gpus.py` for the 3-sample shared-GPU check.",
        ]
    elif relational_status == "promote_candidate":
        best = relational_decision.get("best") or "NA"
        active_next_action = [
            "- Active relational-residual decision is `promote_candidate`.",
            f"- Candidate `{best}` must be repeated/deepened before any manuscript claim.",
            "- Next: repeat seed, full split/family posthoc, condition-level top improved/failed tables, and comparison to primary scFoundation, Stack composite, and strong-composition references.",
            "- Long training jobs should still be detached and checked only at scheduled or >=30-minute windows.",
        ]
    elif relational_status == "reject_as_mainline":
        active_next_action = [
            "- Active relational-residual decision is `reject_as_mainline`.",
            "- Both relational branches improved MMD but failed aggregate perturbation PP, gene-family signal, and multi seen/unseen split gates.",
            "- Next: stop tuning `rel_w` directly and run a checkpoint-selection / residual-ranking audit before launching another long branch.",
            "- New low-util exploration policy applies for future training: up to three strategy jobs per GPU only when GPU memory, CPU, RAM, I/O, and dataloader pressure remain safe.",
        ]
    elif relational_status == "pending":
        active_next_action = [
            "- Active relational-residual decision is still `pending`.",
            "- Wait for scheduled posthoc/summary/decision windows and use the read-only one-shot checker rather than manual polling.",
        ]
    elif fullcap_status == "pivot_from_scfoundation_head_smokes":
        active_next_action = [
            "- Full-cap scFoundation posthoc, Stack composite selection, and scFoundation strong-composition probes are complete.",
            "- Decision: Stack/strong-composition improve parts of multi-unseen behavior but are not promotable as the main branch because they lose MMD, aggregate pp, gene-family, or drug-family signal.",
            "- Current active direction: condition-level residual relational supervision for scFoundation, targeting perturbation residual ranking without hard negative labels.",
            "- Long training jobs should only be checked at scheduled or >=30-minute windows after launch.",
        ]
    elif fullcap_status == "pending":
        active_next_action = [
            "- Current active action is full-cap posthoc evaluation for scFoundation primary/smoke branches, overriding smoke checkpoint eval caps.",
            "- Do not manually poll the full-cap posthoc log before the next 30-minute window unless there is evidence of failure.",
        ]
    else:
        active_next_action = [
            f"- Full-cap gate status is `{fullcap_status}`.",
            "- Inspect the gate recommendations and current run status before launching additional LatentFM work.",
        ]

    lines: list[str] = [
        "# Workspace Status",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "This is a lightweight dashboard. It does not tail training logs, launch jobs, or read large data contents.",
        "",
        "## Active Next Action",
        "",
        *active_next_action,
        "",
        "## Active Strategy Probes",
        "",
        f"- Four-run status: `{STRATEGY_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Four-run status summary: `{four_run_launch_summary(STRATEGY_RUN_ROOT / 'RUN_STATUS.md', STRATEGY_POSTHOC_RUN_ROOT / 'RUN_STATUS.md')}`",
        f"- Four-run posthoc status: `{STRATEGY_POSTHOC_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Four-run posthoc summary: `{status_summary(STRATEGY_POSTHOC_RUN_ROOT / 'RUN_STATUS.md')}`",
        f"- Expanded status: `{STRATEGY_EXPANDED_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Expanded status summary: `{status_summary(STRATEGY_EXPANDED_RUN_ROOT / 'RUN_STATUS.md')}`",
        f"- Combined decision watcher status: `{STRATEGY_ALL_DECISION_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Combined decision watcher summary: `{status_summary(STRATEGY_ALL_DECISION_RUN_ROOT / 'RUN_STATUS.md')}`",
        f"- Combined decision JSON summary: `{strategy_decision['message']}`",
        "- Strategy reports:",
        "  - `reports/LATENTFM_STRATEGY_PROBE_20260619.md`",
        "  - `reports/LATENTFM_STRATEGY_PROBE_EXPANDED_20260619.md`",
        "  - `reports/LATENTFM_STRATEGY_ALL_DECISION_20260619.md`",
        "  - `reports/LATENTFM_CONDITION_RESIDUAL_AUDIT_20260619.md`",
        "  - `reports/LATENTFM_KNN_ADDITIVE_RESIDUAL_DIAGNOSTIC_20260619.md`",
        "  - `reports/LATENTFM_PRIOR_CORRECTION_DECISION_20260619.md`",
        "  - `reports/LATENTFM_CONDITION_PRIOR_TEACHER_SMOKE_20260619.md`",
        "- Active strategy labels:",
        "  - `scf_e2_comp012_pr0`, `scf_e2_comp020_pr0`, `stack_e2_comp006_pr0`, `stack_e2_comp012_pr0`",
        "  - `scf_e1_comp012`, `scf_e3_comp012`, `scf_head_pert005`, `scf_add005`",
        "  - `stack_e1_comp012`, `stack_e3_comp012`, `stack_head_pert005`, `stack_e2_comp020`",
        "",
        "## Active Condition-Prior Teacher Dose Probes",
        "",
        f"- Dose summary JSON summary: `{condition_prior_dose}`",
        f"- `scf_prior005_e2_4k` training status: `{CONDITION_PRIOR_PROBE_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- `scf_prior005_e2_4k` status summary: `{run_marker_summary(CONDITION_PRIOR_PROBE_RUN_ROOT)}`",
        f"- `scf_prior002_e2_4k` training status: `{CONDITION_PRIOR_PRIOR002_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- `scf_prior002_e2_4k` status summary: `{run_marker_summary(CONDITION_PRIOR_PRIOR002_RUN_ROOT)}`",
        f"- `scf_prior010_e2_4k` training status: `{CONDITION_PRIOR_PRIOR010_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- `scf_prior010_e2_4k` status summary: `{run_marker_summary(CONDITION_PRIOR_PRIOR010_RUN_ROOT)}`",
        f"- Primary posthoc watcher: `{CONDITION_PRIOR_POSTHOC_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Primary posthoc watcher summary: `{run_marker_summary(CONDITION_PRIOR_POSTHOC_RUN_ROOT)}`",
        f"- Sister posthoc watcher: `{CONDITION_PRIOR_SISTER_POSTHOC_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Sister posthoc watcher summary: `{run_marker_summary(CONDITION_PRIOR_SISTER_POSTHOC_RUN_ROOT)}`",
        f"- Dose summary watcher: `{CONDITION_PRIOR_DOSE_SUMMARY_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Dose summary watcher summary: `{run_marker_summary(CONDITION_PRIOR_DOSE_SUMMARY_RUN_ROOT)}`",
        f"- Scheduled one-shot checker: `{CONDITION_PRIOR_ONE_SHOT_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Scheduled one-shot checker summary: `{run_marker_summary(CONDITION_PRIOR_ONE_SHOT_RUN_ROOT)}`",
        f"- Scheduled one-shot checker sleep plan: `{read_text(CONDITION_PRIOR_ONE_SHOT_RUN_ROOT / 'SCHEDULE')}`",
        f"- Follow-up scheduled one-shot checker: `{CONDITION_PRIOR_ONE_SHOT_1350_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Follow-up scheduled one-shot summary: `{run_marker_summary(CONDITION_PRIOR_ONE_SHOT_1350_RUN_ROOT)}`",
        f"- Follow-up scheduled one-shot sleep plan: `{read_text(CONDITION_PRIOR_ONE_SHOT_1350_RUN_ROOT / 'SCHEDULE')}`",
        "- Read-only dose checker: `ops/check_condition_prior_dose_once.sh`",
        "- Dose checker validation: `ops/validate_condition_prior_dose_one_shot.py`",
        "- Dose pipeline validation: `ops/validate_latentfm_condition_prior_teacher_dose_pipeline.py`",
        "- Dose readout summarizer: `ops/summarize_condition_prior_one_shot_readout.py`",
        f"- Dose figure status: `{condition_prior_dose_figure.get('status', 'NA')}`",
        "- Dose report:",
        "  - `reports/LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md`",
        "- Dose figure output stem:",
        "  - `reports/latentfm_condition_prior_teacher_dose_20260619` with `.pdf`, `.svg`, and `.png` suffixes",
        "",
        "## Active Condition-Prior Injection Follow-Up",
        "",
        f"- Run status: `{CONDITION_PRIOR_INJECTION_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Status summary: `{run_marker_summary(CONDITION_PRIOR_INJECTION_RUN_ROOT)}`",
        f"- Session: `{read_text(CONDITION_PRIOR_INJECTION_RUN_ROOT / 'SESSION_NAME')}`",
        f"- Started: `{read_text(CONDITION_PRIOR_INJECTION_RUN_ROOT / 'STARTED')}`",
        "- Branch: `scf_prior010_inject_e2_4k`",
        "- Purpose: single targeted test of `condition_delta_head_use_in_model=True` with `condition_prior_delta_loss_weight=0.10`; not a scalar sweep.",
        f"- Posthoc watcher status: `{CONDITION_PRIOR_INJECTION_POSTHOC_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Posthoc watcher summary: `{run_marker_summary(CONDITION_PRIOR_INJECTION_POSTHOC_RUN_ROOT)}`",
        f"- Summary watcher status: `{CONDITION_PRIOR_INJECTION_SUMMARY_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Summary watcher summary: `{run_marker_summary(CONDITION_PRIOR_INJECTION_SUMMARY_RUN_ROOT)}`",
        f"- Injection comparison JSON summary: `{condition_prior_injection}`",
        f"- Common condition-level rows: `{injection_condition_level.get('n_common_conditions', 'NA')}`",
        "- Injection comparison report: `reports/LATENTFM_CONDITION_PRIOR_INJECTION_COMPARISON_20260619.md`",
        "- Injection condition-level report: `reports/LATENTFM_CONDITION_PRIOR_INJECTION_CONDITION_LEVEL_20260619.md`",
        "- Current recommendation: diagnostic only; no more global scalar/head-injection sweep. Next useful work is a split-aware additive-plus-interaction architecture design.",
        "",
        "## Active Condition-Prior Additive-Head Smoke",
        "",
        f"- Run status: `{CONDITION_PRIOR_ADDITIVE_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Status summary: `{run_marker_summary(CONDITION_PRIOR_ADDITIVE_RUN_ROOT)}`",
        f"- Session: `{read_text(CONDITION_PRIOR_ADDITIVE_RUN_ROOT / 'SESSION_NAME')}`",
        f"- Started: `{read_text(CONDITION_PRIOR_ADDITIVE_RUN_ROOT / 'STARTED')}`",
        "- Branch: `scf_prioradd005_prior010_inject_e2_4k`",
        "- Purpose: single targeted test of `condition_prior_additive_delta_loss_weight=0.05` on top of the prior010 injected-head diagnostic branch.",
        f"- Posthoc watcher status: `{CONDITION_PRIOR_ADDITIVE_POSTHOC_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Posthoc watcher summary: `{run_marker_summary(CONDITION_PRIOR_ADDITIVE_POSTHOC_RUN_ROOT)}`",
        f"- Summary watcher status: `{CONDITION_PRIOR_ADDITIVE_SUMMARY_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Summary watcher summary: `{run_marker_summary(CONDITION_PRIOR_ADDITIVE_SUMMARY_RUN_ROOT)}`",
        f"- Additive comparison JSON summary: `{condition_prior_additive}`",
        f"- Scheduled one-shot checker: `{CONDITION_PRIOR_ADDITIVE_ONE_SHOT_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Scheduled one-shot summary: `{run_marker_summary(CONDITION_PRIOR_ADDITIVE_ONE_SHOT_RUN_ROOT)}`",
        f"- Scheduled one-shot sleep plan: `{read_text(CONDITION_PRIOR_ADDITIVE_ONE_SHOT_RUN_ROOT / 'SCHEDULE')}`",
        f"- Backup one-shot checker: `{CONDITION_PRIOR_ADDITIVE_ONE_SHOT_1730_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Backup one-shot summary: `{run_marker_summary(CONDITION_PRIOR_ADDITIVE_ONE_SHOT_1730_RUN_ROOT)}`",
        f"- Backup one-shot sleep plan: `{read_text(CONDITION_PRIOR_ADDITIVE_ONE_SHOT_1730_RUN_ROOT / 'SCHEDULE')}`",
        f"- Late one-shot checker: `{CONDITION_PRIOR_ADDITIVE_ONE_SHOT_1810_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Late one-shot summary: `{run_marker_summary(CONDITION_PRIOR_ADDITIVE_ONE_SHOT_1810_RUN_ROOT)}`",
        f"- Late one-shot sleep plan: `{read_text(CONDITION_PRIOR_ADDITIVE_ONE_SHOT_1810_RUN_ROOT / 'SCHEDULE')}`",
        "- Launch report: `reports/目标推进阶段报告_20260619_1628.md`",
        "- Expected comparison report: `reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_COMPARISON_20260619.md`",
        "- Expected one-shot report: `reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_ONE_SHOT_STATUS_20260619.md`",
        "- Expected readout: `reports/CONDITION_PRIOR_ADDITIVE_HEAD_READOUT_SUMMARY_20260619.md`",
        "- Outcome playbook: `reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_NEXT_ACTIONS_20260619.md`",
        "- Polling policy: training/posthoc/summary watchers check marker files at 1800-second intervals.",
        "",
        "## Historical Relational Residual Probe (Rejected As Mainline)",
        "",
        f"- Run status: `{REL_RUN_ROOT / 'RUN_STATUS.md'}`",
        "- Training sessions:",
        "  - `latentfm_20260619_scfoundation_rel002_comp006_endpoint5_8k`",
        "  - `latentfm_20260619_scfoundation_rel005_comp006_endpoint5_8k`",
        "- Scheduled posthoc readiness one-shots: `07:45 CST`, backup `08:45 CST`.",
        "- Scheduled summary one-shots: `08:35 CST`, backup `09:35 CST`.",
        "- Scheduled decision gate one-shot: `09:45 CST`.",
        "- Scheduled finalize/dashboard refresh one-shot: `09:50 CST`.",
        f"- Decision status: `{relational_status}`",
        f"- Launch log state: `{file_state(REL_LOG_ROOT / 'launch.log')}`",
        f"- Automation validation: `{file_state(REL_RUN_ROOT / 'AUTOMATION_CHAIN_VALIDATION.json')}`",
        f"- Posthoc launcher status: `{file_state(REL_RUN_ROOT / 'POSTHOC_LAUNCH_STATUS.md')}`",
        f"- Report: `{file_state(REL_REPORT)}`",
        f"- rel002 status: `{status_summary(REL_RUN_ROOT / '20260619_scfoundation_rel002_comp006_endpoint5_8k.status')}`",
        f"- rel005 status: `{status_summary(REL_RUN_ROOT / '20260619_scfoundation_rel005_comp006_endpoint5_8k.status')}`",
        "",
        "## Full-Cap Posthoc",
        "",
        f"- Run status: `{FULLCAP_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Session: `{read_text(FULLCAP_RUN_ROOT / 'SESSION_NAME')}`",
        f"- Started: `{read_text(FULLCAP_RUN_ROOT / 'STARTED')}`",
        f"- Exit code marker: `{read_text(FULLCAP_RUN_ROOT / 'EXIT_CODE')}`",
        f"- Log state: `{file_state(FULLCAP_LOG)}`",
        f"- Report: `{FULLCAP_REPORT}`",
        f"- Scheduled one-shot check session: `{read_text(FULLCAP_RUN_ROOT / 'SCHEDULED_ONE_SHOT_SESSION_NAME')}`.",
        f"- Scheduled one-shot target: `{read_text(FULLCAP_RUN_ROOT / 'SCHEDULED_ONE_SHOT_TARGET')}`.",
        f"- Scheduled one-shot exit code: `{read_text(FULLCAP_RUN_ROOT / 'SCHEDULED_ONE_SHOT_EXIT_CODE')}`.",
        "- Scheduled one-shot check session 2: `scheduled_20260619_latentfm_fullcap_one_shot_0040`.",
        "- Scheduled one-shot target 2: `2026-06-19 00:40:00 CST`.",
        f"- Scheduled one-shot exit code 2: `{read_text(FULLCAP_RUN_ROOT / 'SCHEDULED_ONE_SHOT_0040_EXIT_CODE')}`.",
        "- Scheduled one-shot check session 3: `scheduled_20260619_latentfm_fullcap_one_shot_0115`.",
        "- Scheduled one-shot target 3: `2026-06-19 01:15:00 CST`.",
        f"- Scheduled one-shot exit code 3: `{read_text(FULLCAP_RUN_ROOT / 'SCHEDULED_ONE_SHOT_0115_EXIT_CODE')}`.",
        "- Next lightweight check should be at least 30 minutes after `STARTED` if still running.",
        "",
        "## LatentFM Full-Cap Gate",
        "",
        f"- Decision status: `{fullcap_decision.get('status', 'NA')}`",
        f"- Generated: `{fullcap_decision.get('generated', 'NA')}`",
        "- Gate validation: `/data/cyx/1030/scLatent/ops/validate_fullcap_gate.py`.",
        "- Recommendations:",
    ]
    recommendations = fullcap_decision.get("recommendations") or []
    if recommendations:
        lines.extend([f"  - {rec}" for rec in recommendations])
    else:
        lines.append("  - NA")
    lines.extend([
        "",
        "| Label | Complete split | Complete family | test n | test MMD | test pp | unseen1 pp | unseen2 pp | family gene pp |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in fullcap_decision.get("rows", []):
        lines.append(
            f"| `{row.get('label')}` | {row.get('complete_split')} | {row.get('complete_family')} | "
            f"{_fmt_int(row.get('test_n'))} | {_fmt(row.get('test_mmd'))} | {_fmt(row.get('test_pp'))} | "
            f"{_fmt(row.get('multi_unseen1_pp'))} | {_fmt(row.get('multi_unseen2_pp'))} | "
            f"{_fmt(row.get('family_gene_pp'))} |"
        )

    lines.extend([
        "",
        "## Prepared Stack Composite Guard",
        "",
        f"- Run status: `{STACK_GUARD_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Scheduled launch status: `{STACK_GUARD_RUN_ROOT / 'SCHEDULED_LAUNCH_STATUS.md'}`",
        "- Guard condition: launch only when full-cap gate status is `pivot_from_scfoundation_head_smokes`.",
        "- GPU chooser validation: `/data/cyx/1030/scLatent/ops/validate_stack_guard_gpu_chooser.py`.",
        "- General shared-GPU sampler: `/data/cyx/1030/scLatent/ops/select_available_gpus.py`; validation: `/data/cyx/1030/scLatent/ops/validate_gpu_availability_helper.py`.",
        "- Posthoc launcher validation: `/data/cyx/1030/scLatent/ops/validate_stack_posthoc_launcher.py`.",
        "- Summary validation: `/data/cyx/1030/scLatent/ops/validate_stack_summary.py`.",
        "- Planned sessions if launched: `latentfm_20260618_stack_comp003_selppmmd05_8k`, `latentfm_20260618_stack_comp006_selppmmd05_8k`.",
        "- Scheduled one-shot launch session: `scheduled_20260619_stack_composite_launch`.",
        "- Scheduled one-shot launch target: `2026-06-19 00:12:00 CST`.",
        f"- Scheduled one-shot launch exit code: `{read_text(STACK_GUARD_RUN_ROOT / 'SCHEDULED_STACK_LAUNCH_EXIT_CODE')}`.",
        "- Scheduled one-shot launch session 2: `scheduled_20260619_stack_composite_launch_0045`.",
        "- Scheduled one-shot launch target 2: `2026-06-19 00:45:00 CST`.",
        f"- Scheduled one-shot launch exit code 2: `{read_text(STACK_GUARD_RUN_ROOT / 'SCHEDULED_STACK_LAUNCH_0045_EXIT_CODE')}`.",
        "- Scheduled one-shot launch session 3: `scheduled_20260619_stack_composite_launch_0120`.",
        "- Scheduled one-shot launch target 3: `2026-06-19 01:20:00 CST`.",
        f"- Scheduled one-shot launch exit code 3: `{read_text(STACK_GUARD_RUN_ROOT / 'SCHEDULED_STACK_LAUNCH_0120_EXIT_CODE')}`.",
        f"- Guard log state: `{file_state(STACK_GUARD_LOG_ROOT / 'launch_if_fullcap_pivot.log')}`.",
        f"- Current run status file: `{file_state(STACK_GUARD_RUN_ROOT / 'RUN_STATUS.md')}`.",
        f"- Posthoc launcher status: `{file_state(STACK_GUARD_RUN_ROOT / 'POSTHOC_LAUNCH_STATUS.md')}`.",
        f"- Stack composite report: `{file_state(ROOT / 'reports/LATENTFM_STACK_COMPOSITE_SELECTION_REPORT_20260619.md')}`.",
        "",
        "## Scheduled Marker Summary",
        "",
        "| Run root | Marker | Exit code | Finished | Status file |",
        "|---|---|---:|---|---|",
    ])
    fullcap_marker_rows = scheduled_marker_rows(FULLCAP_RUN_ROOT)
    stack_marker_rows = scheduled_marker_rows(STACK_GUARD_RUN_ROOT)
    rel_marker_rows = scheduled_marker_rows(REL_RUN_ROOT)
    if fullcap_marker_rows:
        lines.extend(row.replace("| `", "| `fullcap` | `", 1) for row in fullcap_marker_rows)
    if stack_marker_rows:
        lines.extend(row.replace("| `", "| `stack_guard` | `", 1) for row in stack_marker_rows)
    if rel_marker_rows:
        lines.extend(row.replace("| `", "| `relational` | `", 1) for row in rel_marker_rows)
    if not fullcap_marker_rows and not stack_marker_rows and not rel_marker_rows:
        lines.append("| NA | NA | NA | NA | NA |")
    lines.extend([
        "",
        "## Recent Pert-Residual Smoke",
        "",
        f"- Run status: `{PERTRESID_RUN_ROOT / 'RUN_STATUS.md'}`",
        f"- Training session: `{read_text(PERTRESID_RUN_ROOT / 'SESSION_NAME')}`",
        f"- Started: `{read_text(PERTRESID_RUN_ROOT / 'STARTED')}`",
        f"- Exit code marker: `{read_text(PERTRESID_RUN_ROOT / 'EXIT_CODE')}`",
        f"- Training log state: `{file_state(PERTRESID_LOG)}`",
        f"- Output dir state: `{file_state(pertresid_out)}`",
        f"- Posthoc dir state: `{file_state(posthoc_eval)}`",
        "- Key config: `CONDITION_DELTA_HEAD_TARGET=pert_residual`.",
        "- Posthoc watcher: `watcher_20260618_scfoundation_conddelta_pertresid_posthoc`.",
        f"- Scheduled one-shot check session: `{read_text(SCHEDULED_ONE_SHOT_ROOT / 'SCHEDULED_ONE_SHOT_SESSION_NAME')}`.",
        f"- Scheduled one-shot target: `{read_text(SCHEDULED_ONE_SHOT_ROOT / 'SCHEDULED_ONE_SHOT_TARGET')}`.",
        f"- Scheduled one-shot exit code: `{read_text(SCHEDULED_ONE_SHOT_ROOT / 'SCHEDULED_ONE_SHOT_EXIT_CODE')}`.",
        "- Launch report: `reports/LATENTFM_PERTRESID_TARGET_SMOKE_LAUNCH_20260618.md`.",
        "",
        "## Historical LatentFM Follow-Up Gate",
        "",
        f"- Decision recommendation: {decision.get('recommendation', 'NA')}",
        "",
        "| Run | Status | Reason | multi_unseen1 pp | multi_unseen2 pp | family_gene pp |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in decision.get("rows", []):
        lines.append(
            f"| `{row.get('label')}` | {row.get('status')} | {row.get('reason')} | "
            f"{_fmt(row.get('multi_unseen1_pp'))} | {_fmt(row.get('multi_unseen2_pp'))} | "
            f"{_fmt(row.get('family_gene_pp'))} |"
        )

    lines.extend([
        "",
        "## scFMBench Artifact Layer",
        "",
        f"- Figure manifest: `{ROOT / 'scFM_output/figures/manifest.json'}`",
        f"- Figures: `{figure_manifest.get('n_figures', 'NA')}`",
        f"- Failed figures: `{figure_manifest.get('n_failed_figures', 'NA')}`",
        f"- Skipped figures: `{figure_manifest.get('n_skipped_figures', 'NA')}`",
        f"- Aggregate rows: `{figure_manifest.get('n_rows_summary_all', 'NA')}`",
        f"- Chempert-only models: `{', '.join(chempert_only_models(figure_manifest))}`",
        f"- Manuscript figure manifest: `{ROOT / 'scFM_output/figures_manuscript/manifest.json'}`",
        f"- Manuscript figures: `{manuscript_figure_manifest.get('n_figures', 'NA')}`",
        f"- Manuscript failed figures: `{manuscript_figure_manifest.get('n_failed_figures', 'NA')}`",
        f"- Manuscript skipped figures: `{manuscript_figure_manifest.get('n_skipped_figures', 'NA')}`",
        f"- Manuscript chempert-only models: `{', '.join(chempert_only_models(manuscript_figure_manifest))}`",
        "",
        "## Dataset Package",
        "",
        "- Training-ready package paths are documented in `reports/DATASET_CLOUD_BACKUP_MANIFEST_20260617.md`.",
        f"- Lightweight manifest: `{dataset_manifest}`",
        f"- Manifest exists: `{dataset_manifest.is_file()}`",
        "",
        "## Git Repositories",
        "",
        "| Repo | Status | Latest commit |",
        "|---|---|---|",
        f"| `CoupledFM` | `{git_status(ROOT / 'CoupledFM')}` | `{latest_commit(ROOT / 'CoupledFM')}` |",
        f"| `scFMBench` | `{git_status(ROOT / 'scFMBench')}` | `{latest_commit(ROOT / 'scFMBench')}` |",
        "",
        "## Key Reports",
        "",
        "- `reports/GOAL_REQUIREMENT_STATUS_20260619.md`",
        "- `reports/GOAL_COMPLETION_AUDIT_20260619_1252.md`",
        "- `reports/目标推进阶段报告_20260619_1309.md`",
        "- `reports/目标推进阶段报告_20260619_1628.md`",
        "- `reports/GOAL_REQUIREMENT_STATUS_20260618.md`",
        "- `reports/NEXT_ACTIONS_20260618_2359.md`",
        "- `reports/LATENTFM_PROMOTE_CANDIDATE_PLAYBOOK_20260619.md`",
        "- `reports/LATENTFM_RELATIONAL_RESIDUAL_DECISION_GATE_20260619.md`",
        "- `reports/LATENTFM_SCFOUNDATION_RELATIONAL_RESIDUAL_DECISION_20260619.md`",
        "- `reports/LATENTFM_RELATIONAL_RESIDUAL_AUTOMATION_CHAIN_20260619.md`",
        "- `reports/LATENTFM_POST_RELATIONAL_NEXT_ACTIONS_20260619.md`",
        "- `reports/MANUSCRIPT_FIGURE_ARCHITECTURE_20260619.md`",
        "- `reports/LATENTFM_RELATIONAL_ONE_SHOT_STATUS_20260619.md`",
        "- `reports/OPERATIONS_HANDOFF_20260619.md`",
        "- `reports/HANDOFF_DOCS_VALIDATION_20260619.md`",
        "- `reports/NATURE_METHODS_READINESS_CHECKLIST_20260619.md`",
        "- `reports/NEXT_ACTIONS_20260618_0027.md`",
        "- `reports/LATENTFM_PERTRESID_TARGET_SMOKE_LAUNCH_20260618.md`",
        "- `reports/LATENTFM_FOLLOWUP_DECISION_STATUS_20260617.md`",
        "- `reports/LATENTFM_ALIGNMENT_SMOKE_REPORT_20260617.md`",
        "- `reports/SCFMBENCH_FIGURE_ARTIFACT_AUDIT_20260617_2147.md`",
        "- `reports/SCFMBENCH_CONTINUATION_CHECK_20260619.md`",
        "- `reports/SCFMBENCH_MANUSCRIPT_FIGURE_QC_20260619.md`",
        "- `reports/LATENTFM_STRATEGY_ALL_DECISION_20260619.md`",
        "- `reports/LATENTFM_STRATEGY_FOUR_RUN_PARTIAL_INTERPRETATION_20260619.md`",
        "- `reports/LATENTFM_CONDITION_RESIDUAL_AUDIT_20260619.md`",
        "- `reports/LATENTFM_KNN_ADDITIVE_RESIDUAL_DIAGNOSTIC_20260619.md`",
        "- `reports/LATENTFM_NEXT_MECHANISM_DECISION_20260619.md`",
        "- `reports/LATENTFM_PRIOR_CORRECTION_DECISION_20260619.md`",
        "- `reports/LATENTFM_PRIOR_CORRECTION_EVAL_20260619.md`",
        "- `reports/LATENTFM_PRIOR_CORRECTION_STACK_E2_COMP006_EVAL_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_TEACHER_SMOKE_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_TEACHER_DOSE_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_DOSE_NEXT_ACTIONS_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_DOSE_ONE_SHOT_STATUS_20260619.md`",
        "- `reports/CONDITION_PRIOR_DOSE_READOUT_SUMMARY_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_DIAGNOSTIC_INTERPRETATION_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_CONDITION_LEVEL_COMPARISON_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_BIOLOGICAL_INSIGHT_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_INJECTION_COMPARISON_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_INJECTION_CONDITION_LEVEL_20260619.md`",
        "- `reports/LATENTFM_ADDITIVE_INTERACTION_MODULE_DESIGN_20260619.md`",
        "- `reports/LATENTFM_CONDITION_DELTA_DECOMPOSITION_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_COMPARISON_20260619.md`",
        "- `reports/CONDITION_PRIOR_ADDITIVE_HEAD_READOUT_SUMMARY_20260619.md`",
        "- `reports/LATENTFM_CONDITION_PRIOR_ADDITIVE_HEAD_NEXT_ACTIONS_20260619.md`",
        "- `reports/DATASET_SIZE_RECHECK_20260619.md`",
        "- `reports/DATASET_CLOUD_BACKUP_MANIFEST_20260617.md`",
        "- `docs/PROJECT_OVERVIEW.md`",
        "- `docs/EXPERIMENT_INDEX.md`",
        "- `docs/DATA_PIPELINE.md`",
        "- `docs/RESULTS_SUMMARY.md`",
        "- `docs/MODEL_NOTES.md`",
        "- `docs/DECISIONS.md`",
        "- `docs/BUGS_AND_FIXES.md`",
        "- `ops/validate_handoff_docs.py`",
        "- `ops/validate_workspace_status.py`",
        "- `ops/validate_latentfm_strategy_all_summary.py`",
        "- `ops/validate_latentfm_strategy_all_plotter.py`",
        "- `ops/check_condition_prior_dose_once.sh`",
        "- `ops/validate_condition_prior_dose_one_shot.py`",
        "- `ops/validate_latentfm_condition_prior_teacher_dose_pipeline.py`",
        "- `ops/summarize_condition_prior_one_shot_readout.py`",
        "- `ops/summarize_condition_prior_condition_level_20260619.py`",
        "- `ops/summarize_latentfm_condition_prior_injection_20260619.py`",
        "- `ops/summarize_condition_prior_injection_condition_level_20260619.py`",
        "- `ops/summarize_latentfm_additive_interaction_design_20260619.py`",
        "- `ops/summarize_latentfm_condition_delta_decomposition_20260619.py`",
        "- `ops/launch_latentfm_condition_prior_additive_head_20260619.sh`",
        "- `ops/run_latentfm_condition_prior_additive_head_posthoc_20260619.sh`",
        "- `ops/run_latentfm_condition_prior_additive_head_summary_20260619.sh`",
        "- `ops/summarize_latentfm_condition_prior_additive_head_20260619.py`",
        "- `ops/summarize_condition_prior_additive_head_readout.py`",
        "- `ops/validate_condition_prior_additive_head_pipeline.py`",
        "- `ops/validate_condition_prior_additive_head_readout.py`",
        "- `ops/validate_condition_prior_additive_head_next_actions.py`",
        "- `ops/validate_additive_head_doc_sync.py`",
        "- `ops/validate_condition_prior_readout_summary.py`",
        "- `ops/diagnose_latentfm_knn_additive_residual_20260619.py`",
    ])
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    return 0


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "NA"


def _fmt_int(value: Any) -> str:
    if isinstance(value, bool):
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.1f}"
    return "NA"


if __name__ == "__main__":
    raise SystemExit(main())
