#!/usr/bin/env python3
"""CPU-only train-log checkpoint/selection audit for LatentFM.

This script intentionally does not load model weights or run inference. It
inspects existing train logs and checkpoint file availability to decide whether
a train-only checkpoint selector could plausibly unlock a bounded posthoc eval.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
RUN_ROOT = ROOT / "CoupledFM/output/latentfm_runs"
OUT_DIR = ROOT / "reports/latentfm_trainonly_checkpoint_selection_audit_20260627"
JSON_PATH = ROOT / "reports/latentfm_trainonly_checkpoint_selection_audit_20260627.json"
MD_PATH = ROOT / "reports/LATENTFM_TRAINONLY_CHECKPOINT_SELECTION_AUDIT_20260627.md"
CSV_PATH = OUT_DIR / "run_selection_inventory.csv"


RE_IID = re.compile(
    r"\[IID eval\]\s+epoch=(?P<epoch>-?\d+)\s+"
    r"test_mse=(?P<mse>[-+0-9.eE]+)\s+"
    r"test_mae=(?P<mae>[-+0-9.eE]+)\s+"
    r"test_mmd=(?P<mmd>[-+0-9.eE]+)\s+"
    r"dp=(?P<dp>[-+0-9.eE]+)\s+"
    r"pc=(?P<pc>[-+0-9.eE]+)\s+"
    r"pp=(?P<pp>[-+0-9.eE]+)\s+"
    r"n_conds=(?P<n>\d+)"
)
RE_TEST = re.compile(
    r"\[TEST\]\s+GLOBAL\s+.*?"
    r"(?:mmd|test_mmd)=(?P<mmd>[-+0-9.eE]+).*?"
    r"(?:pp|corr_pert_mean)=(?P<pp>[-+0-9.eE]+)"
)
RE_SELECTION = re.compile(r"Selection metric:\s+(?P<metric>\S+)")
RE_TRAIN_FINISH = re.compile(r"Training finished at step\s+(?P<step>\d+)")
RE_CHECKPOINT = re.compile(r"saved latest\.pt at step\s+(?P<step>\d+)")


@dataclass
class EvalRecord:
    source: str
    epoch: int | None
    step: int | None
    pp: float
    mmd: float
    n_conds: int | None


@dataclass
class RunAudit:
    run: str
    train_log: str
    selection_metric: str
    has_best: bool
    has_latest: bool
    extra_pt_count: int
    eval_count: int
    checkpoint_steps_seen: int
    finished_step: int | None
    best_by_selection_step: int | None
    best_by_selection_pp: float | None
    best_by_selection_mmd: float | None
    best_by_pp_step: int | None
    best_by_pp_pp: float | None
    best_by_pp_mmd: float | None
    pp_gain_vs_selection: float | None
    mmd_delta_vs_selection: float | None
    selector_cpu_gate: str
    actionable_reason: str


def _f(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _selection_score(rec: EvalRecord, metric: str) -> float:
    metric = (metric or "").lower()
    if metric in {"test_mmd", "mmd", "test_mmd_clamped"}:
        return -rec.mmd
    if metric in {"corr_pert_mean", "pearson_pert", "pp"}:
        return rec.pp
    if metric == "corr_minus_mmd":
        return rec.pp - rec.mmd
    return -rec.mmd


def _parse_log(path: Path) -> tuple[str, list[EvalRecord], int | None, int]:
    metric = ""
    records: list[EvalRecord] = []
    finished_step: int | None = None
    checkpoint_steps = 0
    last_step: int | None = None

    for line in path.read_text(errors="replace").splitlines():
        m = RE_SELECTION.search(line)
        if m:
            metric = m.group("metric")
        m = RE_CHECKPOINT.search(line)
        if m:
            last_step = int(m.group("step"))
            checkpoint_steps += 1
        m = RE_TRAIN_FINISH.search(line)
        if m:
            finished_step = int(m.group("step"))
            last_step = finished_step

        m = RE_IID.search(line)
        if m:
            records.append(
                EvalRecord(
                    source="iid_eval",
                    epoch=int(m.group("epoch")),
                    step=last_step,
                    pp=_f(m.group("pp")),
                    mmd=_f(m.group("mmd")),
                    n_conds=int(m.group("n")),
                )
            )
            continue

        m = RE_TEST.search(line)
        if m:
            records.append(
                EvalRecord(
                    source="test_global",
                    epoch=None,
                    step=last_step,
                    pp=_f(m.group("pp")),
                    mmd=_f(m.group("mmd")),
                    n_conds=None,
                )
            )

    return metric, records, finished_step, checkpoint_steps


def _audit_one(log_path: Path) -> RunAudit:
    run_dir = log_path.parent
    rel_run = str(run_dir.relative_to(RUN_ROOT))
    metric, records, finished_step, checkpoint_steps = _parse_log(log_path)
    pt_files = sorted(run_dir.glob("*.pt"))
    names = {p.name for p in pt_files}
    extra_pt_count = len([p for p in pt_files if p.name not in {"best.pt", "latest.pt", "last.pt"}])

    valid = [r for r in records if math.isfinite(r.pp) and math.isfinite(r.mmd)]
    best_sel = max(valid, key=lambda r: _selection_score(r, metric)) if valid else None
    best_pp = max(valid, key=lambda r: r.pp) if valid else None

    pp_gain = None
    mmd_delta = None
    gate = "fail_no_gpu"
    reason = "no_evaluable_train_records"
    if best_sel and best_pp:
        pp_gain = best_pp.pp - best_sel.pp
        mmd_delta = best_pp.mmd - best_sel.mmd
        available_steps = {finished_step}
        if "best.pt" in names and best_sel.step is not None:
            available_steps.add(best_sel.step)
        if ("latest.pt" in names or "last.pt" in names) and finished_step is not None:
            available_steps.add(finished_step)
        step_available = best_pp.step in available_steps or extra_pt_count > 0
        if len(valid) < 2:
            reason = "only_one_train_eval_record_no_selector_degrees_of_freedom"
        elif not step_available:
            reason = "pp_best_step_not_available_as_checkpoint"
        elif pp_gain >= 0.03 and mmd_delta <= 0.001:
            gate = "pass_needs_locked_posthoc_eval"
            reason = "train_log_selector_signal_and_checkpoint_availability_pass"
        else:
            reason = (
                f"gate_fail_pp_gain_{pp_gain:+.6f}_mmd_delta_{mmd_delta:+.6f}"
            )

    return RunAudit(
        run=rel_run,
        train_log=str(log_path),
        selection_metric=metric or "unknown",
        has_best="best.pt" in names,
        has_latest=("latest.pt" in names or "last.pt" in names),
        extra_pt_count=extra_pt_count,
        eval_count=len(valid),
        checkpoint_steps_seen=checkpoint_steps,
        finished_step=finished_step,
        best_by_selection_step=best_sel.step if best_sel else None,
        best_by_selection_pp=best_sel.pp if best_sel else None,
        best_by_selection_mmd=best_sel.mmd if best_sel else None,
        best_by_pp_step=best_pp.step if best_pp else None,
        best_by_pp_pp=best_pp.pp if best_pp else None,
        best_by_pp_mmd=best_pp.mmd if best_pp else None,
        pp_gain_vs_selection=pp_gain,
        mmd_delta_vs_selection=mmd_delta,
        selector_cpu_gate=gate,
        actionable_reason=reason,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audits = [_audit_one(p) for p in sorted(RUN_ROOT.glob("**/train.log"))]
    candidate_audits = [
        a for a in audits
        if a.eval_count > 0 and ("xverse" in a.run or "scf" in a.run or "stack" in a.run)
    ]
    pass_audits = [a for a in candidate_audits if a.selector_cpu_gate.startswith("pass")]
    multi_eval = [a for a in candidate_audits if a.eval_count >= 2]
    single_eval = [a for a in candidate_audits if a.eval_count == 1]

    with CSV_PATH.open("w", newline="") as f:
        fieldnames = list(asdict(candidate_audits[0]).keys()) if candidate_audits else list(RunAudit.__annotations__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in sorted(
            candidate_audits,
            key=lambda x: (
                x.selector_cpu_gate != "pass_needs_locked_posthoc_eval",
                -(x.pp_gain_vs_selection or -999),
                x.run,
            ),
        ):
            writer.writerow(asdict(a))

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "boundary": {
            "task": "CPU-only train-log checkpoint/selection audit",
            "read_roots": [str(RUN_ROOT)],
            "no_gpu": True,
            "no_inference": True,
            "no_canonical_multi_or_trackc_query": True,
        },
        "criteria": {
            "cpu_gate_pass": "alternative train-only pp-best checkpoint improves pp by >= +0.03 over logged selection-best, MMD delta <= +0.001, and checkpoint is available",
            "promotion_if_pass": "bounded locked posthoc eval only; no deployable claim before exact-tail/canonical no-harm",
        },
        "counts": {
            "train_logs_total": len(audits),
            "candidate_logs_with_eval": len(candidate_audits),
            "multi_eval_logs": len(multi_eval),
            "single_eval_logs": len(single_eval),
            "pass_needs_locked_posthoc_eval": len(pass_audits),
        },
        "top_pp_gain_runs": [
            asdict(a)
            for a in sorted(
                [x for x in candidate_audits if x.pp_gain_vs_selection is not None],
                key=lambda x: x.pp_gain_vs_selection or -999,
                reverse=True,
            )[:20]
        ],
        "passed_runs": [asdict(a) for a in pass_audits],
        "output_csv": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    top_lines = []
    for a in summary["top_pp_gain_runs"][:10]:
        top_lines.append(
            "| `{run}` | {eval_count} | `{selection_metric}` | {pp_gain_vs_selection} | {mmd_delta_vs_selection} | `{selector_cpu_gate}` | `{actionable_reason}` |".format(
                run=a["run"],
                eval_count=a["eval_count"],
                selection_metric=a["selection_metric"],
                pp_gain_vs_selection=(
                    "NA" if a["pp_gain_vs_selection"] is None else f"{a['pp_gain_vs_selection']:+.6f}"
                ),
                mmd_delta_vs_selection=(
                    "NA" if a["mmd_delta_vs_selection"] is None else f"{a['mmd_delta_vs_selection']:+.6f}"
                ),
                selector_cpu_gate=a["selector_cpu_gate"],
                actionable_reason=a["actionable_reason"],
            )
        )

    status = "pass_needs_locked_posthoc_eval" if pass_audits else "fail_no_gpu"
    md = f"""# LatentFM Train-Only Checkpoint Selection Audit

## Status

`{status}`

## Boundary

CPU-only train-log/checkpoint inventory. No model weights were loaded, no
inference/training was run, no GPU was used, and no canonical multi or Track C
query rows were used for selection.

## Gate

Pass requires a logged train-only alternative checkpoint/selector to improve
pp by `>= +0.03` over the logged selection-best checkpoint, while keeping MMD
delta `<= +0.001`, and the alternative checkpoint must still be available.

## Summary

* Train logs scanned: `{summary['counts']['train_logs_total']}`
* Candidate logs with eval records: `{summary['counts']['candidate_logs_with_eval']}`
* Logs with at least two eval records: `{summary['counts']['multi_eval_logs']}`
* Logs with only one eval record: `{summary['counts']['single_eval_logs']}`
* Passed selector gate: `{summary['counts']['pass_needs_locked_posthoc_eval']}`

## Top Logged PP-Gain Opportunities

| Run | Eval records | Selection metric | pp gain vs selection | MMD delta vs selection | Gate | Reason |
|---|---:|---|---:|---:|---|---|
{chr(10).join(top_lines) if top_lines else '| NA | 0 | NA | NA | NA | NA | no eval records |'}

## Decision

No immediate GPU is authorized unless `Passed selector gate` is nonzero. If this
report is fail/no-gpu, the checkpoint-selector route is closed for current
artifacts because existing logs/checkpoints do not expose a locked train-only
alternative with sufficient pp gain and no MMD harm.

## Outputs

* JSON: `{JSON_PATH}`
* CSV: `{CSV_PATH}`
"""
    MD_PATH.write_text(md)
    print(json.dumps({"status": status, "json": str(JSON_PATH), "md": str(MD_PATH), "csv": str(CSV_PATH)}, indent=2))


if __name__ == "__main__":
    main()
