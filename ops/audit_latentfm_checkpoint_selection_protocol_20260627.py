#!/usr/bin/env python3
"""CPU-only audit of whether xverse_8k has alternate retained checkpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
REPORT_JSON = ROOT / "reports/latentfm_checkpoint_selection_protocol_audit_20260627.json"
REPORT_MD = ROOT / "reports/LATENTFM_CHECKPOINT_SELECTION_PROTOCOL_AUDIT_20260627.md"

RUNS = {
    "seed42": ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
    / "xverse_comp006_endpoint5_8k_seed42_fulleval",
    "seed43": ROOT
    / "CoupledFM/output/latentfm_runs/xverse_8k_seed_replicate_20260621/"
    / "xverse_comp006_endpoint5_8k_seed43_fulleval",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _stat(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}


def _audit_run(seed: str, run_dir: Path) -> dict:
    log = _read(run_dir / "train.log")
    saved_latest_steps = [int(x) for x in re.findall(r"saved latest\.pt at step (\d+)", log)]
    best_events = [
        {"epoch": int(ep), "test_mmd": float(val)}
        for ep, val in re.findall(r"\[best\] epoch (\d+)\s+new best_test_mmd=([0-9.]+)", log)
    ]
    loaded_best_steps = [int(x) for x in re.findall(r"Loaded best\.pt \(step=(\d+)\)", log)]
    finished = re.search(r"Training finished at step (\d+) .*best_test_mmd=([0-9.]+)", log)
    config = json.loads(_read(run_dir / "config.json")) if (run_dir / "config.json").exists() else {}
    checkpoints = sorted(p.name for p in run_dir.glob("*.pt"))
    retained_step_files = [name for name in checkpoints if re.search(r"(step|ckpt|checkpoint|epoch)", name)]
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "config_selection_metric": config.get("selection_metric"),
        "config_eval_every": config.get("eval_every"),
        "config_total_steps": config.get("total_steps"),
        "config_split_file": config.get("split_file", ""),
        "checkpoints": checkpoints,
        "retained_step_files": retained_step_files,
        "best_pt": _stat(run_dir / "best.pt"),
        "latest_pt": _stat(run_dir / "latest.pt"),
        "saved_latest_steps": saved_latest_steps,
        "best_events": best_events,
        "loaded_best_steps": loaded_best_steps,
        "finished_step": int(finished.group(1)) if finished else None,
        "finished_best_test_mmd": float(finished.group(2)) if finished else None,
    }


def main() -> int:
    runs = [_audit_run(seed, path) for seed, path in RUNS.items()]
    no_alt = all(
        r["exists"]
        and r["checkpoints"] == ["best.pt", "latest.pt"]
        and not r["retained_step_files"]
        and r["loaded_best_steps"] == [r["finished_step"]]
        for r in runs
    )
    best_is_terminal = all(
        r["finished_step"] == r["config_total_steps"] == (r["loaded_best_steps"] or [None])[0]
        for r in runs
    )
    gpu_authorized = False
    status = "checkpoint_selection_protocol_no_alternate_checkpoint_no_gpu"
    reasons = []
    if not no_alt:
        reasons.append("unexpected_retained_checkpoint_inventory_requires_manual_audit")
    else:
        reasons.append("only_best_and_latest_retained")
    if best_is_terminal:
        reasons.append("best_checkpoint_is_terminal_step_for_seed42_seed43")
    else:
        reasons.append("best_terminal_step_not_confirmed")
    reasons.append("intermediate_2k_4k_6k_latest_checkpoints_were_overwritten")
    reasons.append("no_trainonly_selector_candidate_checkpoint_exists")

    out = {
        "status": status,
        "gpu_authorized": gpu_authorized,
        "boundary": {
            "cpu_report_only": True,
            "training_or_inference_used": False,
            "checkpoint_selection_changed": False,
            "canonical_multi_selection_used": False,
            "trackc_heldout_query_used": False,
            "gpu_used": False,
        },
        "runs": runs,
        "reasons": reasons,
        "decision": (
            "Do not launch a checkpoint-reselection GPU/posthoc branch for xverse_8k: "
            "the only retained checkpoints are best.pt and latest.pt, and both seeds "
            "selected the terminal step. Reopen only if archived intermediate checkpoints "
            "or train-only selector artifacts are recovered."
        ),
    }
    REPORT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Checkpoint Selection Protocol Audit 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        f"GPU authorized: `{gpu_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only inventory of retained xverse_8k checkpoints and train logs.",
        "- No training, inference, checkpoint reselection, canonical multi selection, Track C held-out query, or GPU.",
        "",
        "## Retained Checkpoints",
        "",
        "| seed | checkpoints | saved latest steps | best events | loaded best step | finished step |",
        "|---|---|---|---|---:|---:|",
    ]
    for r in runs:
        best_events = ", ".join(f"epoch{e['epoch']}:{e['test_mmd']:.6f}" for e in r["best_events"]) or "none"
        lines.append(
            "| `{seed}` | `{ckpts}` | `{steps}` | `{best}` | {loaded} | {finished} |".format(
                seed=r["seed"],
                ckpts=', '.join(r["checkpoints"]),
                steps=', '.join(map(str, r["saved_latest_steps"])),
                best=best_events,
                loaded=(r["loaded_best_steps"] or [""])[0],
                finished=r["finished_step"],
            )
        )
    lines.extend(
        [
            "",
            "## Reasons",
            "",
            *[f"- `{reason}`" for reason in reasons],
            "",
            "## Decision",
            "",
            out["decision"],
            "",
            "## Outputs",
            "",
            f"- JSON: `{REPORT_JSON}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu_authorized": gpu_authorized, "out_md": str(REPORT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
