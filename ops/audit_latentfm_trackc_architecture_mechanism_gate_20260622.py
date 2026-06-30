#!/usr/bin/env python3
"""Static CPU-only gate for Track C architecture-level support conditioning.

The post-memory gates ruled out more endpoint/replay/memory-dose and subset
sweeps.  This audit checks whether the current CoupledFM Track C implementation
already contains a genuinely different support-conditioned architecture path
that could justify GPU use, or whether a new design/code gate is required first.

It reads source code and already frozen CPU/decision JSONs only.  It does not
read held-out query or canonical evaluation outputs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
CONFIG = ROOT / "CoupledFM/model/latent/config.py"
TRAIN = ROOT / "CoupledFM/model/latent/train.py"
MLP = ROOT / "CoupledFM/model/latent/models/mlp.py"
OUT_JSON = ROOT / "reports/latentfm_trackc_architecture_mechanism_gate_20260622.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_ARCHITECTURE_MECHANISM_GATE_20260622.md"
EVIDENCE_JSONS = {
    "memory_transfer_bottleneck": ROOT / "reports/latentfm_trackc_memory_transfer_bottleneck_gate_20260622.json",
    "wessels_absorbable_subset": ROOT / "reports/latentfm_trackc_wessels_absorbable_subset_gate_20260622.json",
    "crosslatent_deployable_source": ROOT / "reports/latentfm_xverse_crosslatent_deployable_source_gate_20260622.json",
}


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def find_patterns(path: Path, patterns: list[str]) -> dict[str, list[int]]:
    lines = read_lines(path)
    out: dict[str, list[int]] = {}
    for pattern in patterns:
        rx = re.compile(pattern)
        out[pattern] = [i + 1 for i, line in enumerate(lines) if rx.search(line)]
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def status_of(payload: dict[str, Any]) -> str | None:
    decision = payload.get("decision")
    if isinstance(decision, dict) and decision.get("status"):
        return str(decision["status"])
    if payload.get("status"):
        return str(payload["status"])
    return None


def first_line(hits: dict[str, list[int]], pattern: str) -> int | None:
    vals = hits.get(pattern) or []
    return vals[0] if vals else None


def collect() -> dict[str, Any]:
    distill_loss_pattern = r"trackc_routed_distill_t = F\.mse_loss"
    endpoint_loss_pattern = r"trackc_routed_endpoint_t = F\.mse_loss"
    target_fn_pattern = r"def get_trackc_routed_distill_target"
    condition_head_pattern = r"condition_delta_head_use_in_model"
    pairwise_scope_pattern = r"pairwise_condition_adapter"
    config_patterns = [
        r"trackc_routed_distill_loss_weight",
        r"trackc_routed_endpoint_loss_weight",
        r"trackc_routed_distill_memory_mode",
        r"finetune_trainable_scope",
        r"condition_delta_head_use_in_model",
        r"use_pert_in_fusion",
        r"support_context|support_token|support_attention|support_condition",
    ]
    train_patterns = [
        r"def build_trackc_routed_distill_bank",
        r"def get_trackc_routed_distill_target",
        r"trackc_routed_distill_t = F\.mse_loss",
        r"trackc_routed_endpoint_t = F\.mse_loss",
        r"def apply_finetune_freeze",
        r"pairwise_condition_adapter",
        r"condition_prior_adapter",
        r"support_context|support_token|support_attention|support_condition",
    ]
    model_patterns = [
        r"condition_delta_head_use_in_model",
        r"use_pert_in_fusion",
        r"def forward",
        r"support_context|support_token|support_attention|support_condition",
        r"cross_attention|cross-attention",
    ]
    evidence = {}
    for name, path in EVIDENCE_JSONS.items():
        payload = load_json(path)
        evidence[name] = {
            "path": str(path),
            "status": status_of(payload),
            "reasons": (payload.get("decision") or {}).get("reasons", []),
        }
        if name == "wessels_absorbable_subset":
            decision = payload.get("decision") or {}
            evidence[name]["best_oracle_top3_mean_delta"] = decision.get("best_oracle_top3_mean_delta")
            evidence[name]["best_rule"] = (decision.get("best_rule") or {}).get("rule")
            evidence[name]["best_rule_route_gap_closure"] = (
                decision.get("best_rule") or {}
            ).get("weighted_route_gap_closure")
        if name == "memory_transfer_bottleneck":
            decision = payload.get("decision") or {}
            evidence[name]["best_wessels_pairwise_delta"] = decision.get("best_wessels_pairwise_delta")
            evidence[name]["best_wessels_pairwise_route_gap_closure"] = decision.get(
                "best_wessels_pairwise_route_gap_closure"
            )

    hits = {
        "config": find_patterns(CONFIG, config_patterns),
        "train": find_patterns(TRAIN, train_patterns),
        "mlp": find_patterns(MLP, model_patterns),
    }
    has_support_context = any(
        hits[file_key].get(pattern)
        for file_key in hits
        for pattern in hits[file_key]
        if "support_context" in pattern
    )
    existing_trackc_paths = [
        {
            "path": "routed_condition_delta_head_loss",
            "evidence": f"{TRAIN}:{first_line(hits['train'], distill_loss_pattern)}",
            "mechanism_class": "auxiliary_loss_target",
        },
        {
            "path": "routed_endpoint_loss",
            "evidence": f"{TRAIN}:{first_line(hits['train'], endpoint_loss_pattern)}",
            "mechanism_class": "endpoint_loss_target",
        },
        {
            "path": "train_multi_memory_teacher",
            "evidence": f"{TRAIN}:{first_line(hits['train'], target_fn_pattern)}",
            "mechanism_class": "teacher_target_selection",
        },
        {
            "path": "condition_delta_head_in_model",
            "evidence": f"{MLP}:{first_line(hits['mlp'], condition_head_pattern)}",
            "mechanism_class": "condition_vector_injection_not_support_context",
        },
        {
            "path": "pairwise_condition_adapter",
            "evidence": f"{TRAIN}:{first_line(hits['train'], pairwise_scope_pattern)}",
            "mechanism_class": "small_trainable_scope_already_tested",
        },
    ]

    reasons = []
    if not has_support_context:
        reasons.append("no_explicit_support_context_or_support_token_architecture_found")
    failed_statuses = {
        name: item["status"]
        for name, item in evidence.items()
        if item.get("status") and str(item["status"]).endswith("fail")
    }
    if "memory_transfer_bottleneck" in failed_statuses:
        reasons.append("completed_memory_transfer_family_has_wessels_absorption_failure")
    if "wessels_absorbable_subset" in failed_statuses:
        reasons.append("focused_wessels_subset_gate_failed_even_oracle_top3_below_material_threshold")
    if "crosslatent_deployable_source" in failed_statuses:
        reasons.append("immediate_tracka_new_information_source_failed_noharm")

    status = (
        "trackc_architecture_mechanism_gate_fail_requires_new_design"
        if reasons
        else "trackc_architecture_mechanism_gate_pass_existing_gpu_candidate"
    )
    return {
        "status": status,
        "heldout_query_used": False,
        "canonical_outputs_used": False,
        "source_files": {
            "config": str(CONFIG),
            "train": str(TRAIN),
            "mlp": str(MLP),
        },
        "code_hits": hits,
        "has_explicit_support_context_architecture": bool(has_support_context),
        "existing_trackc_paths": existing_trackc_paths,
        "evidence": evidence,
        "decision": {
            "status": status,
            "reasons": reasons,
            "recommended_action": (
                "design_and_cpu_validate_new_support_conditioning_mechanism_before_gpu"
                if reasons
                else "prepare_one_capped_gpu_smoke_with_run_status_and_noharm_gate"
            ),
        },
        "future_gate_requirements": [
            "default-off support-conditioned architecture path, not only endpoint/teacher loss weight changes",
            "support context/token/source built only from safe trainselect train/support artifacts; no full-v2 query during selection",
            "unit tests proving split guards, default-off behavior, and query exclusion",
            "CPU synthetic or tiny real-data smoke showing the new support context is wired into the forward path",
            "support-val material gate before GPU: Wessels route-gap closure >= +0.05 and support pp delta >= +0.02",
            "canonical single/background no-harm before any frozen one-shot held-out query evaluation",
        ],
    }


def write_md(payload: dict[str, Any]) -> None:
    decision = payload["decision"]
    evidence = payload["evidence"]
    lines = [
        "# Track C Architecture Mechanism CPU Gate",
        "",
        "Static CPU-only audit of whether current CoupledFM contains an existing",
        "architecture-level support-conditioned Track C mechanism that can justify",
        "another GPU launch.",
        "",
        f"Status: `{decision['status']}`",
        f"Recommended action: `{decision['recommended_action']}`",
        "",
        "## Boundary",
        "",
        "- Reads source code plus frozen CPU/decision JSONs only.",
        "- Does not read held-out Track C query.",
        "- Does not read canonical evaluation outputs for selection.",
        "- This is not a training run and cannot promote a model.",
        "",
        "## Code Audit",
        "",
        "| item | finding | evidence |",
        "|---|---|---|",
    ]
    for item in payload["existing_trackc_paths"]:
        lines.append(
            f"| `{item['path']}` | `{item['mechanism_class']}` | `{item['evidence']}` |"
        )
    lines.extend(
        [
            f"| `explicit_support_context_architecture` | `{payload['has_explicit_support_context_architecture']}` | searched `support_context/support_token/support_attention/support_condition` in config/train/mlp |",
            "",
            "## Decision Evidence",
            "",
            "| artifact | status | key evidence |",
            "|---|---|---|",
        ]
    )
    memory = evidence["memory_transfer_bottleneck"]
    lines.append(
        "| `memory_transfer_bottleneck` | `{}` | Wessels pairwise delta `{:+.6f}`, route-gap closure `{:+.6f}` |".format(
            memory["status"],
            float(memory.get("best_wessels_pairwise_delta") or 0.0),
            float(memory.get("best_wessels_pairwise_route_gap_closure") or 0.0),
        )
    )
    subset = evidence["wessels_absorbable_subset"]
    lines.append(
        "| `wessels_absorbable_subset` | `{}` | best oracle top3 mean delta `{:+.6f}`, best route closure `{:+.6f}` |".format(
            subset["status"],
            float(subset.get("best_oracle_top3_mean_delta") or 0.0),
            float(subset.get("best_rule_route_gap_closure") or 0.0),
        )
    )
    cross = evidence["crosslatent_deployable_source"]
    lines.append(
        f"| `crosslatent_deployable_source` | `{cross['status']}` | reasons `{';'.join(cross.get('reasons') or [])}` |"
    )
    lines.extend(["", "## Gate Reasons", ""])
    if decision["reasons"]:
        lines.extend([f"- `{reason}`" for reason in decision["reasons"]])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Future Gate Requirements",
            "",
            *[f"- {item}" for item in payload["future_gate_requirements"]],
            "",
            "## Interpretation",
            "",
            "The current code has routed support teachers and trainable adapter scopes,",
            "but those completed families already failed support/canonical gates. The",
            "audit did not find an existing explicit support-context/token architecture",
            "that would constitute a new mechanism. A future GPU launch therefore needs",
            "new default-off code plus CPU/unit gates before any detached training.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    payload = collect()
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_md(payload)
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
