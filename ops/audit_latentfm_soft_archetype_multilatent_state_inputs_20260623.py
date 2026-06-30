#!/usr/bin/env python3
"""Input/provenance audit for the soft-archetype multi-latent state CPU gate."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path("/data/cyx/1030/scLatent")
OUT_MD = ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_MULTILATENT_STATE_INPUT_AUDIT_20260623.md"
OUT_JSON = ROOT / "reports/latentfm_soft_archetype_multilatent_state_input_audit_20260623.json"
TRAINONLY_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"

LATENT_JSONS = {
    "xverse": ROOT / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json",
    "stack": ROOT / "reports/latentfm_crosslatent_stack_tracka_anchor_internal_val_20260622.json",
    "scfoundation": ROOT / "reports/latentfm_crosslatent_scfoundation_tracka_anchor_internal_val_20260622.json",
    "scldm": ROOT / "reports/latentfm_crosslatent_scldm_tracka_anchor_internal_val_20260622.json",
}
TRAINONLY_PERT_MEANS = {
    "xverse": ROOT / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
    "stack": ROOT / "runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622/artifacts/stack_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
    "scfoundation": ROOT / "runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622/artifacts/scfoundation_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
    "scldm": ROOT / "runs/latentfm_crosslatent_tracka_trainonly_baselines_20260622/artifacts/scldm_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
}
OTHER_INPUTS = {
    "protocol": ROOT / "reports/latentfm_soft_archetype_multilatent_state_gate_protocol_20260623.json",
    "soft_archetype_predictive": ROOT / "reports/latentfm_soft_archetype_predictive_gate_20260623.json",
    "soft_archetype_dataset_effects": ROOT / "reports/LATENTFM_SOFT_ARCHETYPE_DATASET_EFFECTS_20260623.md",
    "crosslatent_deployable_source": ROOT / "reports/latentfm_xverse_crosslatent_deployable_source_gate_20260622.json",
    "trainonly_baselines": ROOT / "reports/latentfm_crosslatent_tracka_trainonly_baselines_20260622.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("group", "")),
        str(row.get("dataset", "")),
        str(row.get("condition", "")),
        str(row.get("gene", "")),
    )


def row_feature_values(row: dict) -> list[float]:
    out = []
    for key in (
        "anchor_pearson_pert",
        "anchor_pearson_ctrl",
        "anchor_mmd_clamped",
        "anchor_minus_gene_raw_mean",
        "anchor_minus_dataset_mean",
        "gene_train_count",
    ):
        value = row.get(key)
        if value is None:
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            pass
    return out


def main() -> int:
    checks: list[dict] = []
    required = {"trainonly_split": TRAINONLY_SPLIT, **LATENT_JSONS, **TRAINONLY_PERT_MEANS, **OTHER_INPUTS}
    missing = [name for name, path in required.items() if not path.exists()]
    checks.append({"name": "required_inputs_exist", "passed": not missing, "evidence": missing})

    latent_payloads = {name: load_json(path) for name, path in LATENT_JSONS.items() if path.exists()}
    protocol = load_json(OTHER_INPUTS["protocol"]) if OTHER_INPUTS["protocol"].exists() else {}
    checks.append(
        {
            "name": "protocol_ready_no_gpu",
            "passed": protocol.get("status") == "soft_archetype_multilatent_state_protocol_ready_cpu_implementation_next_no_gpu"
            and protocol.get("gpu_authorization") == "none",
            "evidence": {"status": protocol.get("status"), "gpu": protocol.get("gpu_authorization")},
        }
    )

    split_ok = True
    split_evidence = {}
    for name, payload in latent_payloads.items():
        split = str(payload.get("split_file", ""))
        split_evidence[name] = split
        if Path(split) != TRAINONLY_SPLIT:
            split_ok = False
    checks.append({"name": "all_latents_use_trainonly_crossbg_split", "passed": split_ok, "evidence": split_evidence})

    row_counts = {name: int(payload.get("n_rows") or len(payload.get("condition_rows") or [])) for name, payload in latent_payloads.items()}
    checks.append(
        {
            "name": "all_latents_have_expected_internal_val_rows",
            "passed": bool(row_counts) and all(v == 350 for v in row_counts.values()),
            "evidence": row_counts,
        }
    )

    indexed = {
        name: {row_key(row): row for row in payload.get("condition_rows") or []}
        for name, payload in latent_payloads.items()
    }
    common_keys = set.intersection(*(set(v) for v in indexed.values())) if indexed else set()
    union_keys = set.union(*(set(v) for v in indexed.values())) if indexed else set()
    checks.append(
        {
            "name": "condition_rows_align_across_latents",
            "passed": len(common_keys) == 350 and len(union_keys) == 350,
            "evidence": {"common": len(common_keys), "union": len(union_keys)},
        }
    )

    feature_missing = {}
    for name, rows in indexed.items():
        missing_features = 0
        for key in common_keys:
            if len(row_feature_values(rows[key])) < 4:
                missing_features += 1
        feature_missing[name] = missing_features
    checks.append(
        {
            "name": "continuous_features_available",
            "passed": bool(feature_missing) and all(v == 0 for v in feature_missing.values()),
            "evidence": feature_missing,
        }
    )

    leakage_strings = []
    for name, payload in latent_payloads.items():
        text = json.dumps(payload)[:20000].lower()
        if "heldout" in text or "query" in text or "split_seed42_multi_support_v2.json" in text:
            leakage_strings.append(name)
    checks.append(
        {
            "name": "no_obvious_query_or_full_multi_strings",
            "passed": not leakage_strings,
            "evidence": leakage_strings,
        }
    )

    agreement_summary = {}
    if len(common_keys) == 350:
        for target in ("anchor_minus_gene_raw_mean", "anchor_minus_dataset_mean"):
            per_key_std = []
            for key in common_keys:
                vals = []
                for rows in indexed.values():
                    value = rows[key].get(target)
                    if value is not None:
                        vals.append(float(value))
                if len(vals) >= 3:
                    m = mean(vals)
                    per_key_std.append((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5)
            agreement_summary[target] = {
                "n": len(per_key_std),
                "mean_crosslatent_std": mean(per_key_std) if per_key_std else None,
            }

    failed = [c for c in checks if not c["passed"]]
    status = (
        "soft_archetype_multilatent_state_inputs_ready_cpu_gate_next_no_gpu"
        if not failed
        else "soft_archetype_multilatent_state_inputs_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorization": "none",
        "next_authorization": "cpu_gate_implementation_only" if not failed else "none",
        "latents": sorted(LATENT_JSONS),
        "trainonly_split": str(TRAINONLY_SPLIT),
        "input_paths": {name: str(path) for name, path in required.items()},
        "row_counts": row_counts,
        "common_condition_rows": len(common_keys),
        "union_condition_rows": len(union_keys),
        "agreement_summary": agreement_summary,
        "checks": checks,
        "failed_checks": [c["name"] for c in failed],
        "implementation_note": (
            "Inputs are sufficient to implement a CPU-only continuous/multi-latent state gate. "
            "This audit does not compute a promotion metric and does not authorize GPU."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# LatentFM Soft-Archetype Multi-Latent State Input Audit",
        "",
        f"Status: `{status}`",
        "GPU authorization: `none`",
        f"Next authorization: `{payload['next_authorization']}`",
        "",
        "## Summary",
        "",
        f"- latents: `{', '.join(payload['latents'])}`",
        f"- common condition rows: `{len(common_keys)}`",
        f"- union condition rows: `{len(union_keys)}`",
        f"- train-only split: `{TRAINONLY_SPLIT}`",
        "",
        "## Checks",
        "",
        "| check | passed | evidence |",
        "|---|---:|---|",
    ]
    for check in checks:
        evidence = check["evidence"]
        evidence_s = json.dumps(evidence, sort_keys=True) if isinstance(evidence, (dict, list)) else str(evidence)
        lines.append(f"| `{check['name']}` | `{check['passed']}` | {evidence_s} |")
    lines.extend(["", "## Agreement Diagnostics", ""])
    for key, diag in agreement_summary.items():
        lines.append(f"- `{key}`: n `{diag['n']}`, mean cross-latent std `{diag['mean_crosslatent_std']:.6f}`")
    lines.extend(["", "## Decision Boundary", ""])
    lines.append(payload["implementation_note"])
    lines.append("A future CPU gate must still pass the protocol's no-harm, stability, and shuffled/permuted controls.")
    lines.extend(["", "## Failed Checks", ""])
    lines.extend([f"- `{name}`" for name in payload["failed_checks"]] or ["- none"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
