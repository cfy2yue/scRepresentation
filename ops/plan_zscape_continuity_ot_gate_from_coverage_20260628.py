#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def by_name(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("cell_type_broad", "")): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    coverage_path = Path(args.coverage_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(coverage_path.read_text(encoding="utf-8"))
    z_rows = payload.get("zperturb_summary", {}).get("candidate_broad_cell_types", [])
    r_rows = payload.get("reference_summary", {}).get("candidate_broad_cell_types", [])
    shared = set(payload.get("shared_candidate_broad_cell_types", []))
    z_by = by_name(z_rows)
    r_by = by_name(r_rows)

    selected = []
    for name in shared:
        z = z_by.get(name, {})
        r = r_by.get(name, {})
        selected.append(
            {
                "cell_type_broad": name,
                "zperturb_cells": int(z.get("cells", 0)),
                "zperturb_control_cells": int(z.get("control_cells", 0)),
                "zperturb_perturb_cells": int(z.get("perturb_cells", 0)),
                "zperturb_perturb_targets": int(z.get("n_perturb_targets", 0)),
                "zperturb_timepoints": int(z.get("n_timepoints", 0)),
                "zperturb_embryos": int(z.get("n_embryos", 0)),
                "zperturb_subtypes": int(z.get("n_subtypes", 0)),
                "reference_cells": int(r.get("cells", 0)),
                "reference_timepoints": int(r.get("n_timepoints", 0)),
                "reference_embryos": int(r.get("n_embryos", 0)),
                "reference_subtypes": int(r.get("n_subtypes", 0)),
            }
        )
    selected.sort(
        key=lambda row: (
            row["zperturb_perturb_targets"],
            row["zperturb_perturb_cells"],
            row["reference_timepoints"],
            row["reference_cells"],
        ),
        reverse=True,
    )
    selected = selected[: args.top_k]

    coverage_status = payload.get("status", "")
    status = "zscape_continuity_ot_gate_plan_ready_no_gpu"
    if coverage_status != "zscape_metadata_coverage_gate_pass":
        status = "zscape_continuity_ot_gate_plan_blocked_by_coverage"
    elif len(selected) < 2:
        status = "zscape_continuity_ot_gate_plan_fail_too_few_lineages"

    out_json = out_dir / "zscape_continuity_ot_gate_plan.json"
    out_md = out_dir / "LATENTFM_ZSCAPE_CONTINUITY_OT_GATE_PLAN_20260628.md"
    plan = {
        "timestamp_utc": now_utc(),
        "status": status,
        "gpu_authorized": False,
        "coverage_json": str(coverage_path),
        "coverage_status": coverage_status,
        "selected_cell_types": selected,
        "next_required_inputs": [
            "ZSCAPE cell metadata coverage CSVs",
            "small predeclared expression subset or processed CDS/raw-count access for selected cell types only",
            "gene metadata and HVG/PCA vocabulary decision",
            "optional scFM latent embedding source if available without using test/canonical selection",
        ],
        "continuity_metrics": [
            "adjacent-time centroid distance rank vs non-adjacent controls",
            "time-order kNN or nearest-centroid accuracy",
            "velocity smoothness and magnitude jump across adjacent timepoints",
            "embryo bootstrap stability",
            "timepoint shuffle negative control",
        ],
        "ot_metrics": [
            "within-cell-type adjacent-stage OT cost vs random pairing",
            "stage-matched control-control OT stability",
            "time-shuffle and target-shuffle collapse",
            "cost-space agreement between HVG/PCA and optional scFM latent",
            "entropic regularization sensitivity",
        ],
        "promotion_gate": [
            "at least two selected lineages have stable control time-order continuity significantly above shuffle",
            "within-cell-type adjacent-stage OT is embryo-bootstrap stable and better than random/time-shuffle",
            "batch/QC covariates do not explain the continuity/OT signal",
            "no canonical Track A multi or Track C held-out query is used for selecting any trajectory parameter",
        ],
        "fail_close": [
            "coverage pass is absent",
            "fewer than two lineages have adequate shared reference/ZPERTURB coverage",
            "time-order continuity collapses under embryo bootstrap",
            "OT pairing is highly sensitive to seed, cost space, or entropic regularization",
            "signals are explained by batch/QC rather than cell-type/time biology",
        ],
    }
    out_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM ZSCAPE Continuity And OT Gate Plan",
        "",
        f"Timestamp: `{plan['timestamp_utc']}`",
        "",
        f"Status: `{status}`",
        "",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU/report-only next-gate planner from ZSCAPE metadata coverage outputs.",
        "- Does not download expression matrices, train, infer, embed, use canonical multi, or use Track C query.",
        "- It selects biological lineages for a later continuity/OT stability gate.",
        "",
        "## Selected Candidate Cell Types",
        "",
        "| cell_type_broad | zperturb perturb targets | zperturb perturb cells | zperturb timepoints | zperturb embryos | reference cells | reference timepoints |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["cell_type_broad"],
                    str(row["zperturb_perturb_targets"]),
                    str(row["zperturb_perturb_cells"]),
                    str(row["zperturb_timepoints"]),
                    str(row["zperturb_embryos"]),
                    str(row["reference_cells"]),
                    str(row["reference_timepoints"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Continuity Metrics",
            "",
        ]
    )
    for item in plan["continuity_metrics"]:
        lines.append(f"- {item}.")
    lines.extend(["", "## OT Metrics", ""])
    for item in plan["ot_metrics"]:
        lines.append(f"- {item}.")
    lines.extend(["", "## Promotion Gate", ""])
    for item in plan["promotion_gate"]:
        lines.append(f"- {item}.")
    lines.extend(["", "## Fail-Close", ""])
    for item in plan["fail_close"]:
        lines.append(f"- {item}.")
    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if status == "zscape_continuity_ot_gate_plan_ready_no_gpu":
        lines.append(
            "Proceed to a CPU-only continuity/OT stability launcher design for the selected lineages."
        )
    else:
        lines.append(
            "Do not proceed to expression subset download or trajectory analysis until the blocking coverage issue is resolved."
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- JSON: `{out_json}`",
            f"- report: `{out_md}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)
    print(out_json)
    print(status)
    return 0 if status != "zscape_continuity_ot_gate_plan_blocked_by_coverage" else 2


if __name__ == "__main__":
    raise SystemExit(main())
