#!/usr/bin/env python3
"""Non-noop canonical guard for uncertainty-gated anchor fallback.

The earlier train-only/internal gate found a strong tail-protection signal, but
the frozen canonical adjudication showed zero enabled canonical rows.  This
script makes the non-noop requirement explicit so the branch cannot be reused
as a GPU/pass candidate without a real deployable footprint.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
IN_JSON = ROOT / "reports/latentfm_uncertainty_gated_anchor_fallback_canonical_noharm_20260625.json"
OUT_JSON = ROOT / "reports/latentfm_uncertainty_gated_anchor_fallback_nonnoop_gate_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_UNCERTAINTY_GATED_ANCHOR_FALLBACK_NONNOOP_GATE_20260625.md"


def main() -> int:
    payload = json.loads(IN_JSON.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    requirements = []
    pass_rows = []
    for group in ("test_single", "family_gene"):
        n_conditions = sum(int(r[group]["n_conditions"]) for r in rows)
        n_enabled = sum(int(r[group]["n_enabled"]) for r in rows)
        enabled_fraction = n_enabled / n_conditions if n_conditions else 0.0
        group_pass = n_enabled >= 25 or enabled_fraction >= 0.05
        requirements.append(
            {
                "group": group,
                "n_conditions": n_conditions,
                "n_enabled": n_enabled,
                "enabled_fraction": enabled_fraction,
                "required": "n_enabled >= 25 OR enabled_fraction >= 0.05",
                "passes": group_pass,
            }
        )
        pass_rows.append(group_pass)

    all_pass = all(pass_rows)
    status = (
        "uncertainty_gated_anchor_fallback_nonnoop_pass_protocol_next"
        if all_pass
        else "uncertainty_gated_anchor_fallback_nonnoop_fail_no_gpu"
    )
    out = {
        "status": status,
        "gpu_authorized": False,
        "route_freeze_authorized": False,
        "boundary": {
            "source": str(IN_JSON),
            "canonical_use": "non-noop footprint guard only; no selection",
            "canonical_multi_used": False,
            "trackc_query_used": False,
        },
        "requirements": requirements,
        "decision": {
            "summary": (
                "Frozen route has a nontrivial canonical footprint"
                if all_pass
                else "Frozen route is exact/no-op on canonical single/family and cannot authorize GPU"
            ),
            "next_action": (
                "external review before any route-freeze implementation"
                if all_pass
                else "close as deployable route; keep as train-only scaling mechanism insight"
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Uncertainty-Gated Anchor Fallback Non-Noop Gate",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        f"- Source: `{IN_JSON}`.",
        "- Canonical rows are used only to require a nontrivial frozen-route footprint.",
        "- No canonical multi, Track C query, training, inference, or GPU is used.",
        "",
        "## Requirements",
        "",
        "| group | enabled / conditions | enabled fraction | required | pass |",
        "|---|---:|---:|---|---|",
    ]
    for row in requirements:
        lines.append(
            "| `{group}` | {en}/{n} | {frac:.3f} | {req} | `{passes}` |".format(
                group=row["group"],
                en=row["n_enabled"],
                n=row["n_conditions"],
                frac=row["enabled_fraction"],
                req=row["required"],
                passes=row["passes"],
            )
        )
    lines += [
        "",
        "## Decision",
        "",
        "- The internal uncertainty-gated fallback remains useful mechanism evidence: it shows where true-cell budget gains are stable.",
        "- It is not a deployable route because the frozen rule enables no canonical `test_single` or `family_gene` rows.",
        "- Do not launch GPU from this branch unless a new train-only rule maps to a nontrivial canonical footprint and then passes no-harm.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
