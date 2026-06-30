#!/usr/bin/env python3
"""Launcher preflight for Track C support-set abstention router.

CPU/report-only. Confirms the router hypothesis is frozen, no held-out Track C
query/canonical multi boundary is crossed, and the launcher plumbing exposes a
minimum support-count router that preserves exact no-op behavior.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"
if str(COUPLED) not in sys.path:
    sys.path.insert(0, str(COUPLED))

from model.latent.config import Config  # noqa: E402
from model.latent.train import build_trackc_support_set_task_bank, make_trackc_support_set_task_batch  # noqa: E402


GATE_JSON = ROOT / "reports/latentfm_trackc_support_set_abstention_router_gate_20260627.json"
WRAPPER = ROOT / "ops/launch_latentfm_trackc_support_set_abstention_router_smoke_20260627.sh"
BASE_LAUNCHER = ROOT / "ops/launch_latentfm_trackc_support_set_smoke_20260627.sh"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_abstention_launcher_preflight_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_ABSTENTION_LAUNCHER_PREFLIGHT_20260627.md"
SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
FULL_V2 = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"


FROZEN_SPEC = {
    "policy": "shared_gene_component",
    "beta": 1.0,
    "min_support": 2,
    "min_confidence": -1.0,
}


def run_min_support_smoke() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        safe_split = root / "split_seed42_multi_support_v2_trainselect.json"
        safe_split.write_text("{}", encoding="utf-8")
        anchor_path = root / "anchor.json"
        candidate_path = root / "candidate.json"

        def row(condition: str, pred: list[float]) -> dict[str, object]:
            return {"dataset": "D", "condition": condition, "pred_mean": pred}

        anchor = {
            "split_file": str(safe_split),
            "groups": {
                "train_multi": {
                    "condition_metrics": [
                        row("A+B", [0.0, 0.0, 0.0]),
                        row("A+C", [0.0, 0.0, 0.0]),
                        row("B+D", [0.0, 0.0, 0.0]),
                    ]
                }
            },
        }
        candidate = {
            "split_file": str(safe_split),
            "groups": {
                "train_multi": {
                    "condition_metrics": [
                        row("A+B", [2.0, 0.0, 0.0]),
                        row("A+C", [1.0, 0.0, 0.0]),
                        row("B+D", [0.0, 1.0, 0.0]),
                    ]
                }
            },
        }
        anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        cfg = Config(
            emb_dim=3,
            trackc_support_set_task_use_in_model=True,
            trackc_support_set_task_dim=3,
            trackc_support_set_task_source="shared_gene_condition_means",
            trackc_support_set_task_safe_split_file=str(safe_split),
            trackc_support_set_task_anchor_condition_means=str(anchor_path),
            trackc_support_set_task_candidate_condition_means=str(candidate_path),
            trackc_support_set_task_min_support_count=2,
        )
        bank = build_trackc_support_set_task_bank(cfg)
        task_two, present_two = make_trackc_support_set_task_batch(bank, "D", "A+B", 2, cfg, torch.device("cpu"))
        task_one, present_one = make_trackc_support_set_task_batch(bank, "D", "A+C", 2, cfg, torch.device("cpu"))
        two_ok = (
            task_two is not None
            and present_two is not None
            and torch.allclose(present_two, torch.ones(2, 1))
            and torch.allclose(task_two[0], torch.tensor([0.5, 0.5, 0.0]))
        )
        one_ok = (
            task_one is not None
            and present_one is not None
            and torch.allclose(present_one, torch.zeros(2, 1))
            and torch.allclose(task_one, torch.zeros(2, 3))
        )
        return {"two_support_present": bool(two_ok), "one_support_exact_noop": bool(one_ok)}


def main() -> int:
    reasons = []
    gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))
    selected = gate.get("selected") or {}
    if gate.get("status") != "trackc_support_set_abstention_router_gate_pass_launcher_design_next_no_gpu":
        reasons.append(f"abstention_gate_status_not_pass: {gate.get('status')}")
    for key, value in FROZEN_SPEC.items():
        if selected.get(key) != value:
            reasons.append(f"frozen_spec_mismatch_{key}: {selected.get(key)!r}")
    boundary = gate.get("boundary") or {}
    if not boundary.get("safe_trainselect_only"):
        reasons.append("abstention_gate_not_safe_trainselect_only")
    if boundary.get("full_trackc_query_used") or boundary.get("canonical_multi_selection_used"):
        reasons.append("unsafe_query_or_canonical_multi_boundary")
    if str(FULL_V2) in json.dumps(gate):
        reasons.append("full_v2_query_path_appears_in_gate_payload")
    if not SAFE_SPLIT.is_file():
        reasons.append("safe_trainselect_split_missing")

    wrapper_text = WRAPPER.read_text(encoding="utf-8") if WRAPPER.is_file() else ""
    base_text = BASE_LAUNCHER.read_text(encoding="utf-8") if BASE_LAUNCHER.is_file() else ""
    if "LATENTFM_TRACKC_SUPPORT_SET_MIN_SUPPORT_COUNT=2" not in wrapper_text:
        reasons.append("wrapper_does_not_freeze_min_support_2")
    if "TRACKC_SUPPORT_SET_TASK_MIN_SUPPORT_COUNT" not in base_text:
        reasons.append("base_launcher_does_not_forward_min_support")
    smoke = run_min_support_smoke()
    if not smoke.get("two_support_present"):
        reasons.append("two_support_token_not_present")
    if not smoke.get("one_support_exact_noop"):
        reasons.append("one_support_not_exact_noop")

    status = "trackc_support_set_abstention_launcher_preflight_pass_no_gpu" if not reasons else "trackc_support_set_abstention_launcher_preflight_fail_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "reasons": reasons,
        "frozen_router_spec": dict(FROZEN_SPEC),
        "inputs": {"gate_json": str(GATE_JSON), "safe_split": str(SAFE_SPLIT), "wrapper": str(WRAPPER), "base_launcher": str(BASE_LAUNCHER)},
        "min_support_smoke": smoke,
        "decision": "Pass permits code/protocol audit for a bounded GPU smoke; it does not directly authorize model promotion or held-out query evaluation.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Track C Support-Set Abstention Launcher Preflight 2026-06-27",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only launcher preflight.",
        "- Safe trainselect support-val design only; held-out Track C query remains forbidden.",
        "- No training, inference, canonical multi selection, or GPU.",
        "",
        "## Frozen Router Spec",
        "",
        f"- `{FROZEN_SPEC}`",
        "",
        "## Checks",
        "",
        f"- min-support smoke: `{smoke}`",
        f"- reasons: `{', '.join(reasons) if reasons else 'none'}`",
        "",
        "## Decision",
        "",
        "A pass allows external/code audit and resource audit for a bounded smoke. It is not a promotion claim.",
        "",
        f"- JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "reasons": reasons, "out_md": str(OUT_MD)}, indent=2))
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
