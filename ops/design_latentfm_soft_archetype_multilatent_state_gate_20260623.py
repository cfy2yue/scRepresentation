#!/usr/bin/env python3
"""Protocol for reopening archetype/state prior as a continuous multi-latent CPU gate."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data/cyx/1030/scLatent")
OUT_MD = ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_MULTILATENT_STATE_GATE_PROTOCOL_20260623.md"
OUT_JSON = ROOT / "reports/latentfm_soft_archetype_multilatent_state_gate_protocol_20260623.json"

INPUT_REPORTS = {
    "predictive_gate": ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_PREDICTIVE_GATE_20260623.md",
    "conditional_router": ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_CONDITIONAL_ROUTER_CPU_GATE_20260623.md",
    "orthogonal_router": ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_ORTHOGONAL_ROUTER_CPU_GATE_20260623.md",
    "dataset_effects": ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_DATASET_EFFECTS_20260623.md",
    "reopen_protocol": ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_REOPEN_PROTOCOL_20260623.md",
    "portfolio": ROOT / "reports/LATENTFM_HIGH_THROUGHPUT_EXPLORATION_PORTFOLIO_20260623.md",
}


def main() -> int:
    missing = [str(path) for path in INPUT_REPORTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required archetype reports: " + ", ".join(missing))

    status = "soft_archetype_multilatent_state_protocol_ready_cpu_implementation_next_no_gpu"
    payload = {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "cpu_implementation_only",
        "inputs": {k: str(v) for k, v in INPUT_REPORTS.items()},
        "negative_evidence_to_respect": [
            "hard residualized/consensus labels were unstable",
            "soft archetypes had structure but failed dataset_mean no-harm",
            "conditional/orthogonal routers did not pass shuffled-control and coverage gates",
            "positive pockets such as Wessels/Jiang_IFNB are not sufficient without a train-only safety rule",
        ],
        "candidate": {
            "name": "continuous_multilatent_state_agreement_gate",
            "allowed_features": [
                "control/source residualized continuous PCs",
                "prototype-distance or soft-membership vectors without hard labels",
                "agreement/stability features across xverse/scFoundation/SCLDM only when each latent has train-only control/source panels",
                "train-only gene/context metadata already allowed for internal proxy baselines",
            ],
            "forbidden_features": [
                "canonical test outputs for selection",
                "canonical test_multi as positive signal",
                "held-out Track C query rows",
                "validation target residuals when fitting state features",
                "posthoc prediction errors from active GPU runs",
                "renamed hard/soft KMeans threshold rules from failed gates",
            ],
        },
        "cpu_gate": {
            "split": str(ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"),
            "trackc_optional_split": str(ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"),
            "baselines": [
                "dataset_mean",
                "gene_raw_mean",
                "gene_only_ridge",
                "old_soft_archetype_gene_interact_ridge",
                "dataset_only_router",
            ],
            "negative_controls": [
                "within-dataset shuffled state features",
                "between-condition permuted state agreement",
                "latent-source dropout/leave-one-latent",
                "dataset-label-only proxy",
            ],
            "pass_criteria": [
                "state feature stability >= 0.70 across seeds/subsamples",
                "dataset NMI/purity no worse than soft K16 gate and dataset-only proxy cannot explain the gain",
                "Track A internal cross_background and family delta >= +0.02 vs dataset_mean and gene_only_ridge, or CI lower > 0",
                "bootstrap p_harm <= 0.20 and leave-one-dataset min delta >= -0.02",
                "shuffled/permuted state controls lose at least +0.02 pp of the claimed gain",
                "predeclared abstain rule must keep Norman/Jiang_TGFB/Jiang_TNFA/Jiang_IFNG harm pockets off without reading validation targets",
            ],
            "trackc_promotion_alternative": [
                "If used for Track C instead of Track A, require Wessels support-val pp delta >= +0.02",
                "Wessels route-gap closure >= +0.05",
                "Norman delta >= -0.02",
                "canonical support-absent no-op preserved by protocol",
            ],
        },
        "gpu_consequence_if_passed": (
            "At most one capped smoke, either Track A state-gated adapter or Track C support-state adapter; "
            "no query, no seed sweep, no broad archetype GPU sweep."
        ),
        "fail_close": (
            "If the continuous/multi-latent state gate fails stability, dataset-proxy, no-harm, or shuffled-control "
            "criteria, archetype remains diagnostic-only."
        ),
    }

    lines = [
        "# LatentFM Soft-Archetype Multi-Latent State Gate Protocol",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        "Next authorization: `cpu_implementation_only`",
        "",
        "## Why Reopen",
        "",
        "Old archetype work found real state signal but no safe deployable rule.  K16 soft archetypes were stable and not pure dataset labels, with strong positive pockets such as Wessels and Jiang_IFNB, but aggregate dataset-mean no-harm and shuffled-control gates failed.  This protocol reopens the idea only as continuous state agreement, not as another hard/soft KMeans threshold sweep.",
        "",
        "## Required Negative Evidence",
        "",
    ]
    for item in payload["negative_evidence_to_respect"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Allowed Inputs",
            "",
        ]
    )
    for item in payload["candidate"]["allowed_features"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Forbidden Inputs", ""])
    for item in payload["candidate"]["forbidden_features"]:
        lines.append(f"- {item}")
    lines.extend(["", "## CPU Gate", ""])
    lines.append(f"- Track A split: `{payload['cpu_gate']['split']}`")
    lines.append(f"- Optional Track C split: `{payload['cpu_gate']['trackc_optional_split']}`")
    lines.append("- Baselines: `" + "`, `".join(payload["cpu_gate"]["baselines"]) + "`")
    lines.extend(["", "### Negative Controls", ""])
    for item in payload["cpu_gate"]["negative_controls"]:
        lines.append(f"- {item}")
    lines.extend(["", "### Pass Criteria", ""])
    for item in payload["cpu_gate"]["pass_criteria"]:
        lines.append(f"- {item}")
    lines.extend(["", "### Track C Alternative", ""])
    for item in payload["cpu_gate"]["trackc_promotion_alternative"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## GPU Consequence If Passed",
            "",
            payload["gpu_consequence_if_passed"],
            "",
            "## Fail-Close Rule",
            "",
            payload["fail_close"],
            "",
            "## Inputs Read",
            "",
        ]
    )
    for name, path in payload["inputs"].items():
        lines.append(f"- `{name}`: `{path}`")

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
