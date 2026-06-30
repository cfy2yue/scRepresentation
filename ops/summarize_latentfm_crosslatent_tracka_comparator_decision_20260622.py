#!/usr/bin/env python3
"""Aggregate cross-latent Track A anchor comparator reports.

This script is intended to run after the GPU posthoc comparator has produced
latent-specific anchor internal-val summaries for stack/scfoundation/scldm.
It performs no model inference and does not read canonical test, canonical
multi, or Track C query artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
LATENTS = ("stack", "scfoundation", "scldm")
OUT_JSON = ROOT / "reports/latentfm_crosslatent_tracka_anchor_comparator_decision_20260622.json"
OUT_MD = ROOT / "reports/LATENTFM_CROSSLATENT_TRACKA_ANCHOR_COMPARATOR_DECISION_20260622.md"


def report_json(latent: str) -> Path:
    return ROOT / f"reports/latentfm_crosslatent_{latent}_tracka_anchor_internal_val_20260622.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):+.6f}"
    except Exception:
        return str(value)


def extract_delta(payload: dict[str, Any], group: str, baseline: str) -> dict[str, Any] | None:
    for row in payload.get("paired_deltas") or []:
        if row.get("group") == group and row.get("baseline") == baseline:
            return row
    return None


def audit_latent(latent: str, payload: dict[str, Any]) -> dict[str, Any]:
    means = payload.get("means_files") or {}
    reasons = []
    if not means.get("pert_means_override"):
        reasons.append("missing_explicit_trainonly_pert_means_override")
    if payload.get("decision", {}).get("status") != "crosslatent_anchor_internal_val_candidate":
        reasons.extend(payload.get("decision", {}).get("reasons") or ["latent_anchor_gate_failed"])
    row = {
        "latent": latent,
        "status": "candidate" if not reasons else "not_promotable",
        "reasons": reasons,
        "checkpoint": payload.get("checkpoint"),
        "means_files": means,
        "n_rows": payload.get("n_rows"),
        "decision": payload.get("decision"),
        "key_deltas": {},
    }
    for group in (
        "internal_val_cross_background_seen_gene_proxy",
        "internal_val_family_gene_proxy",
    ):
        row["key_deltas"][group] = {}
        for baseline in ("gene_raw_mean", "dataset_mean"):
            delta = extract_delta(payload, group, baseline)
            row["key_deltas"][group][baseline] = None if delta is None else {
                "delta_mean": delta.get("delta_mean"),
                "ci95": delta.get("ci95"),
                "p_harm": delta.get("p_harm"),
                "n_conditions": delta.get("n_conditions"),
                "n_datasets": delta.get("n_datasets"),
            }
    return row


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Cross-Latent Track A Anchor Comparator Decision",
        "",
        f"Status: `{payload['status']}`",
        f"Recommended action: `{payload['recommended_action']}`",
        "",
        "## Scope",
        "",
        "- Aggregates latent-specific train-only internal-val anchor comparator reports.",
        "- Requires explicit train-only pert means provenance.",
        "- Does not read canonical test, canonical multi, or Track C query artifacts.",
        "",
        "## Summary",
        "",
        "| latent | status | rows | crossbg anchor-gene | crossbg anchor-dataset | family anchor-gene | family anchor-dataset | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["latents"]:
        kd = row.get("key_deltas") or {}
        cb = kd.get("internal_val_cross_background_seen_gene_proxy") or {}
        fam = kd.get("internal_val_family_gene_proxy") or {}
        lines.append(
            f"| `{row['latent']}` | `{row['status']}` | {row.get('n_rows')} | "
            f"{fmt((cb.get('gene_raw_mean') or {}).get('delta_mean'))} | "
            f"{fmt((cb.get('dataset_mean') or {}).get('delta_mean'))} | "
            f"{fmt((fam.get('gene_raw_mean') or {}).get('delta_mean'))} | "
            f"{fmt((fam.get('dataset_mean') or {}).get('delta_mean'))} | "
            f"{', '.join(row.get('reasons') or ['none'])} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- `{payload['status']}`: {payload['reason']}",
        f"- next action: `{payload['recommended_action']}`",
        "",
        "## Guardrail",
        "",
        "A candidate result here does not authorize training. It only authorizes a",
        "separate mechanism review with hypothesis, resource plan, launcher,",
        "RUN_STATUS, and frozen canonical no-harm gate.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = []
    missing = []
    for latent in LATENTS:
        path = report_json(latent)
        if not path.is_file():
            missing.append(str(path))
            continue
        rows.append(audit_latent(latent, load_json(path)))
    if missing:
        status = "comparator_decision_waiting_for_reports"
        action = "do_not_launch_training_or_claim_crosslatent_result"
        reason = "missing latent comparator reports"
    else:
        candidates = [row for row in rows if row["status"] == "candidate"]
        if candidates:
            status = "crosslatent_anchor_candidate_exists_needs_review"
            action = "request_or_perform_mechanism_review_before_any_training"
            reason = "at least one latent passed strict internal-val anchor gate"
        else:
            status = "crosslatent_anchor_comparator_all_failed"
            action = "close_crosslatent_anchor_branch_or_seek_new_information_source"
            reason = "no latent passed strict anchor-vs-own-baseline gate"
    payload = {
        "status": status,
        "recommended_action": action,
        "reason": reason,
        "missing_reports": missing,
        "latents": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    print(status)
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
