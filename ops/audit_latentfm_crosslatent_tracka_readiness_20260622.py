#!/usr/bin/env python3
"""Readiness audit for a Track A cross-latent internal-val comparator.

This is a lightweight provenance/protocol audit. It does not read GT cell
matrices, does not run model inference, and does not use canonical test or
held-out multi query outcomes for selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py


ROOT = Path("/data/cyx/1030/scLatent")
SPLIT = ROOT / "dataset/biFlow_data/split_seed42_xverse_trainonly_crossbg_val_v2.json"
OUT_JSON = ROOT / "reports/latentfm_crosslatent_tracka_readiness_20260622.json"
OUT_MD = ROOT / "reports/LATENTFM_CROSSLATENT_TRACKA_READINESS_20260622.md"

GROUPS = (
    "internal_val_cross_background_seen_gene_proxy",
    "internal_val_family_gene_proxy",
)

LATENTS = {
    "xverse": {
        "data_dir": ROOT / "dataset/latentfm_full/xverse",
        "checkpoint": ROOT
        / "CoupledFM/output/latentfm_runs/xverse_8k_full_eval_20260620/"
        "xverse_comp006_endpoint5_8k_seed42_fulleval/best.pt",
        "trainonly_pert_means": ROOT
        / "runs/latentfm_xverse_trainonly_crossbg_val_20260622/artifacts/"
        "xverse_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
        "baseline_json": ROOT / "reports/latentfm_xverse_gene_reliability_router_gate_20260622.json",
        "anchor_internal_val_json": ROOT
        / "reports/latentfm_xverse_tracka_anchor_internal_val_error_map_20260622.json",
    },
    "stack": {
        "data_dir": ROOT / "dataset/latentfm_full/stack",
        "checkpoint": ROOT
        / "CoupledFM/output/latentfm_runs/full_stack/20260617_stack_comp006_delta_w5_12k/best.pt",
        "trainonly_pert_means": ROOT
        / "runs/latentfm_crosslatent_tracka_20260622/artifacts/"
        "stack_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
        "baseline_json": ROOT
        / "reports/latentfm_crosslatent_stack_gene_reliability_router_gate_20260622.json",
        "anchor_internal_val_json": ROOT
        / "reports/latentfm_crosslatent_stack_tracka_anchor_internal_val_20260622.json",
    },
    "scfoundation": {
        "data_dir": ROOT / "dataset/latentfm_full/scfoundation",
        "checkpoint": ROOT
        / "CoupledFM/output/latentfm_runs/full_scfoundation/"
        "20260617_scfoundation_comp006_delta_w5_12k/best.pt",
        "trainonly_pert_means": ROOT
        / "runs/latentfm_crosslatent_tracka_20260622/artifacts/"
        "scfoundation_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
        "baseline_json": ROOT
        / "reports/latentfm_crosslatent_scfoundation_gene_reliability_router_gate_20260622.json",
        "anchor_internal_val_json": ROOT
        / "reports/latentfm_crosslatent_scfoundation_tracka_anchor_internal_val_20260622.json",
    },
    "scldm": {
        "data_dir": ROOT / "dataset/latentfm_full/scldm",
        "checkpoint": ROOT
        / "CoupledFM/output/latentfm_runs/full_scldm/20260617_scldm_comp006_delta_w5_12k/best.pt",
        "trainonly_pert_means": ROOT
        / "runs/latentfm_crosslatent_tracka_20260622/artifacts/"
        "scldm_trainonly_pert_means_split_seed42_crossbgval_v2.npz",
        "baseline_json": ROOT
        / "reports/latentfm_crosslatent_scldm_gene_reliability_router_gate_20260622.json",
        "anchor_internal_val_json": ROOT
        / "reports/latentfm_crosslatent_scldm_tracka_anchor_internal_val_20260622.json",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_conditions(path: Path) -> set[str]:
    with h5py.File(path, "r") as handle:
        raw = handle["conditions"][:]
    return {v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw}


def split_condition_counts(split: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    total = {group: 0 for group in GROUPS}
    missing_h5 = []
    missing_conditions = {group: [] for group in GROUPS}
    train_conditions = 0
    train_missing = []
    for ds, obj in sorted(split.items()):
        h5_path = data_dir / f"{ds}.h5"
        if not h5_path.is_file():
            missing_h5.append(str(ds))
            continue
        conds = read_conditions(h5_path)
        train = [str(c) for c in obj.get("train") or []]
        train_conditions += sum(1 for c in train if c in conds)
        train_missing.extend([f"{ds}:{c}" for c in train if c not in conds][:5])
        for group in GROUPS:
            for cond in obj.get(group) or []:
                cond = str(cond)
                if cond in conds:
                    total[group] += 1
                else:
                    missing_conditions[group].append(f"{ds}:{cond}")
    return {
        "train_conditions_present": train_conditions,
        "group_conditions_present": total,
        "missing_h5": missing_h5,
        "missing_conditions": {k: v[:20] for k, v in missing_conditions.items()},
        "n_missing_conditions": {k: len(v) for k, v in missing_conditions.items()},
        "train_missing_examples": train_missing[:20],
    }


def inspect_latent(name: str, cfg: dict[str, Path], split: dict[str, Any]) -> dict[str, Any]:
    data_dir = cfg["data_dir"]
    manifest = data_dir / "manifest.json"
    metadata = data_dir / "condition_metadata.json"
    row: dict[str, Any] = {
        "latent": name,
        "data_dir": str(data_dir),
        "checkpoint": str(cfg["checkpoint"]),
        "trainonly_pert_means": str(cfg["trainonly_pert_means"]),
        "baseline_json": str(cfg["baseline_json"]),
        "anchor_internal_val_json": str(cfg["anchor_internal_val_json"]),
        "exists": {
            "data_dir": data_dir.is_dir(),
            "manifest": manifest.is_file(),
            "condition_metadata": metadata.is_file(),
            "checkpoint": cfg["checkpoint"].is_file(),
            "trainonly_pert_means": cfg["trainonly_pert_means"].is_file(),
            "baseline_json": cfg["baseline_json"].is_file(),
            "anchor_internal_val_json": cfg["anchor_internal_val_json"].is_file(),
        },
    }
    if manifest.is_file():
        info = load_json(manifest)
        row["manifest_datasets"] = len(info.get("datasets", {}))
        row["manifest_emb_dim"] = info.get("emb_dim") or info.get("latent_dim")
    if data_dir.is_dir():
        row["split_coverage"] = split_condition_counts(split, data_dir)
    blockers = []
    for key in ("data_dir", "manifest", "condition_metadata", "checkpoint"):
        if not row["exists"][key]:
            blockers.append(f"missing_{key}")
    coverage = row.get("split_coverage") or {}
    if coverage.get("missing_h5"):
        blockers.append("split_dataset_h5_missing")
    if any(int(v) > 0 for v in (coverage.get("n_missing_conditions") or {}).values()):
        blockers.append("internal_val_condition_missing")
    if not row["exists"]["trainonly_pert_means"]:
        blockers.append("needs_trainonly_pert_means")
    if not row["exists"]["baseline_json"]:
        blockers.append("needs_latent_specific_gene_baseline_gate")
    if not row["exists"]["anchor_internal_val_json"]:
        blockers.append("needs_anchor_internal_val_gpu_audit")
    row["readiness_status"] = "ready_existing_artifacts" if not blockers else "needs_artifacts"
    row["blockers"] = blockers
    return row


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Cross-Latent Track A Readiness Audit",
        "",
        f"Status: `{payload['decision']['status']}`",
        f"Recommended action: `{payload['decision']['recommended_action']}`",
        "",
        "## Scope",
        "",
        "- This is a lightweight readiness/protocol audit for a cross-latent Track A internal-val comparator.",
        "- It checks bundle, split, checkpoint, baseline, and train-only pert-mean artifact availability.",
        "- It does not read GT cell matrices, run model inference, or use canonical test/multi/query outcomes for selection.",
        "",
        "## Candidate Latents",
        "",
        "| latent | emb dim | checkpoint | train-only pert means | baseline JSON | anchor internal-val JSON | group coverage | blockers |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for row in payload["latents"]:
        cov = row.get("split_coverage") or {}
        group_cov = cov.get("group_conditions_present") or {}
        coverage = ", ".join(f"{k}={v}" for k, v in group_cov.items())
        blockers = ", ".join(row["blockers"]) if row["blockers"] else "none"
        lines.append(
            f"| `{row['latent']}` | {row.get('manifest_emb_dim', 'NA')} | "
            f"`{row['exists']['checkpoint']}` | `{row['exists']['trainonly_pert_means']}` | "
            f"`{row['exists']['baseline_json']}` | `{row['exists']['anchor_internal_val_json']}` | "
            f"{coverage} | {blockers} |"
        )
    lines += [
        "",
        "## Protocol Gate",
        "",
        "Before any cross-latent GPU comparator:",
        "",
        "1. Build latent-specific train-only pert means from `split_seed42_xverse_trainonly_crossbg_val_v2.json` train rows.",
        "2. Run the gene-reliability baseline gate separately for each latent using its own data dir and train-only pert means.",
        "3. Only then run anchor internal-val GPU posthoc for comparator checkpoints, detached with RUN_STATUS and the AGENTS.md GPU audit.",
        "4. Interpret within-latent deltas versus each latent's own `gene_raw_mean` and `dataset_mean`; do not compare raw pp values across latent spaces as a promotion claim.",
        "",
        "## Decision",
        "",
        f"- `{payload['decision']['status']}`: {payload['decision']['reason']}",
        f"- next action: `{payload['decision']['recommended_action']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    split = load_json(SPLIT)
    latents = [inspect_latent(name, cfg, split) for name, cfg in LATENTS.items()]
    missing_artifacts = [
        row for row in latents if row["latent"] != "xverse" and row["readiness_status"] != "ready_existing_artifacts"
    ]
    if missing_artifacts:
        decision = {
            "status": "crosslatent_not_ready_build_trainonly_baselines_first",
            "recommended_action": "build_latent_specific_trainonly_pert_means_and_baseline_gates_before_gpu",
            "reason": "comparator latents have bundle/checkpoint coverage, but lack train-only pert means, latent-specific baseline JSONs, and anchor internal-val artifacts",
        }
    else:
        decision = {
            "status": "crosslatent_ready_for_detached_gpu_comparator",
            "recommended_action": "launch_detached_anchor_internal_val_comparator_after_gpu_audit",
            "reason": "all comparator artifacts needed before GPU are present",
        }
    payload = {
        "split_file": str(SPLIT),
        "groups": list(GROUPS),
        "latents": latents,
        "decision": decision,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    print(decision["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
