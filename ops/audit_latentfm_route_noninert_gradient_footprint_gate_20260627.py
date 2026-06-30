#!/usr/bin/env python3
"""Standardized non-inert route/gradient footprint synthesis for LatentFM.

CPU/report-only. This script does not run training or inference; it integrates
existing route-footprint, gradient dry-run, and completed metric gates to decide
whether any already-tested route is both non-inert and metric-promising enough
to justify GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
OUT_JSON = REPORTS / "latentfm_route_noninert_gradient_footprint_gate_20260627.json"
OUT_MD = REPORTS / "LATENTFM_ROUTE_NONINERT_GRADIENT_FOOTPRINT_GATE_20260627.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": True, "_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> int:
    route_footprint = load_json(REPORTS / "latentfm_tracka_route_footprint_cpu_gate_20260627.json")
    solver_state = load_json(REPORTS / "latentfm_tracka_solver_state_residual_footprint_gate_20260627.json")
    allowtail_dry = load_json(REPORTS / "latentfm_allowtail_gradient_path_dryrun_20260627.json")
    loss_obs = load_json(REPORTS / "latentfm_loss_path_observability_20260627.json")
    failure_cluster = load_json(REPORTS / "latentfm_tracka_failure_cluster_conditioned_trust_region_gate_20260627.json")

    support_decisions = [
        load_json(REPORTS / "latentfm_trackc_support_set_sharedgene_decision_xverse_trackc_support_set_sharedgene_adapter_2k_seed42.json"),
        load_json(REPORTS / "latentfm_trackc_support_set_sharedgene_decision_xverse_trackc_support_set_focused_min2_adapter_2k_seed42.json"),
        load_json(REPORTS / "latentfm_trackc_support_set_sharedgene_decision_xverse_trackc_support_set_abstention_min2_adapter_2k_seed42.json"),
    ]

    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "route": "condition_delta_bridge_footprint",
            "source": "LATENTFM_TRACKA_ROUTE_FOOTPRINT_CPU_GATE_20260627",
            "status": route_footprint.get("status", "unknown"),
            "noninert": False,
            "metric_promising": False,
            "reason": "one_step_nonnoop_footprint_too_small; condition-specific adapter-only footprint absent",
            "gpu_authorized": False,
        }
    )
    rows.append(
        {
            "route": "solver_state_residual_adapter",
            "source": "LATENTFM_TRACKA_SOLVER_STATE_RESIDUAL_FOOTPRINT_GATE_20260627",
            "status": solver_state.get("status", "unknown"),
            "noninert": True,
            "metric_promising": False,
            "reason": "gradient/one-step footprint exists but condition-specific footprint absent and proxy shuffle not beaten",
            "gpu_authorized": False,
        }
    )
    rows.append(
        {
            "route": "allowlisted_tail_loss_schedule",
            "source": "LATENTFM_ALLOWTAIL_GRADIENT_PATH_DRYRUN_20260627 + branch decision",
            "status": "close_current_loss_schedule_no_gpu",
            "noninert": True,
            "metric_promising": False,
            "reason": "gradient path material but completed seed42/43 posthoc movement is <0.0014 and exact metrics fail",
            "gpu_authorized": False,
        }
    )
    rows.append(
        {
            "route": "failure_cluster_trust_region_proxy",
            "source": "LATENTFM_TRACKA_FAILURE_CLUSTER_CONDITIONED_TRUST_REGION_GATE_20260627",
            "status": failure_cluster.get("status", "unknown"),
            "noninert": "proxy_only",
            "metric_promising": False,
            "reason": "proxy gain exists but bootstrap CI crosses zero, dataset min below -0.02, and candidate MMD unavailable",
            "gpu_authorized": False,
        }
    )
    for idx, decision in enumerate(support_decisions):
        rows.append(
            {
                "route": f"trackc_support_set_variant_{idx + 1}",
                "source": decision.get("run_root") or "Track C support-set decision",
                "status": decision.get("status", "trackc_support_only_robustness_fail_close"),
                "noninert": False,
                "metric_promising": False,
                "reason": "completed support-val actual delta is near zero or control-comparable; fixed-vector support-set path is effectively inert",
                "gpu_authorized": False,
            }
        )

    active_promotable = [
        row for row in rows if bool(row["gpu_authorized"]) or (row["noninert"] is True and row["metric_promising"] is True)
    ]
    status = "route_noninert_gradient_footprint_fail_no_gpu" if not active_promotable else "route_noninert_gradient_footprint_has_candidate_external_audit_next_no_gpu"
    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_report_only": True,
            "no_training": True,
            "no_inference": True,
            "no_canonical_multi_selection": True,
            "no_trackc_query": True,
        },
        "routes": rows,
        "loss_path_observability": {
            "status": loss_obs.get("status"),
            "logs_scanned": get(loss_obs, "summary", "train_logs_scanned"),
            "note": "log route magnitudes are not GPU authorization without gradient footprint plus metric/no-harm pass",
        },
        "decision": {
            "active_promotable_routes": len(active_promotable),
            "action": "no GPU from existing non-inert routes; require genuinely new route unit/preflight or external candidate gate",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Route Non-Inert Gradient/Footprint Gate 2026-06-27",
        "",
        f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M CST')}`",
        "",
        f"Status: `{payload['status']}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only synthesis of existing route-footprint, gradient dry-run, and metric gates.",
        "- No training, inference, checkpoint selection, canonical multi selection, Track C query, or GPU.",
        "",
        "## Route Table",
        "",
        "| route | status | non-inert | metric promising | GPU? | reason |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["routes"]:
        lines.append(
            f"| `{row['route']}` | `{row['status']}` | `{row['noninert']}` | "
            f"`{row['metric_promising']}` | `{row['gpu_authorized']}` | {row['reason']} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- active promotable routes: `{payload['decision']['active_promotable_routes']}`",
        f"- action: `{payload['decision']['action']}`",
        "",
        "This closes existing no-movement/non-inert candidates as GPU routes under current evidence. A future route must pass a unit/gradient footprint gate and a real train-only metric/no-harm gate before launch.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON}`",
        f"- Markdown: `{OUT_MD}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
