#!/usr/bin/env python3
"""Validate default-off Track C train-only memory teacher wiring.

This is a short CPU preflight. It checks that the training-code teacher bank can
express the frozen train-only memory readout rule without reading held-out query
or using support-val as memory.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
SUPPORT_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_support_route_readiness_20260622.py"
MEMORY_MODULE_PATH = ROOT / "ops/audit_latentfm_trackc_support_memory_readout_gate_20260622.py"
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"
FULL_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
TRAINSELECT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
ROUTE_FILE = ROOT / "reports/latentfm_trackc_trainonly_memory_route_teacher_20260622.json"
GENE_CACHE = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
OUT_JSON = ROOT / "reports/latentfm_trackc_trainonly_memory_teacher_validation_20260622.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_TRAINONLY_MEMORY_TEACHER_VALIDATION_20260622.md"


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Track C Train-Only Memory Teacher Validation",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Provenance",
        "",
        f"- route_file: `{payload['route_file']}`",
        f"- full_split: `{payload['full_split']}`",
        f"- trainselect_split: `{payload['trainselect_split']}`",
        f"- data_dir: `{payload['data_dir']}`",
        f"- leakage_status: `{payload['leakage_status']}`",
        "",
        "## Rule",
        "",
        f"- route names: `{payload['routes']}`",
        f"- memory_mode: `{payload['memory_mode']}`",
        f"- memory_k: `{payload['memory_k']}`",
        f"- memory_min_score: `{payload['memory_min_score']}`",
        f"- memory_scope: `{payload['memory_scope']}`",
        "",
        "## Coverage",
        "",
        f"- CPU readout available support conditions: `{payload['n_cpu_available']}`",
        f"- training teacher available support conditions: `{payload['n_code_available']}`",
        f"- mismatched availability: `{payload['availability_mismatch_count']}`",
        f"- non-finite targets: `{payload['nonfinite_target_count']}`",
        "",
        "## Gate Reasons",
        "",
    ]
    reasons = payload.get("reasons") or []
    lines.extend([f"- `{reason}`" for reason in reasons] if reasons else ["- none"])
    lines.extend(["", "## Consequence", ""])
    if payload["status"] == "trainonly_memory_teacher_validation_pass":
        lines.append(
            "- The code path can express the frozen train-only memory teacher; "
            "this still does not authorize GPU launch before the latest-checkpoint gate is resolved."
        )
    else:
        lines.append("- Do not launch a memory-transfer GPU smoke until these validation failures are fixed.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    sys.path.insert(0, str(COUPLEDFM))
    support = import_module("trackc_support_route_readiness", SUPPORT_MODULE_PATH)
    memory = import_module("trackc_support_memory_readout", MEMORY_MODULE_PATH)

    from model.latent.config import Config
    from model.latent.dataset import CrossDatasetFMDataset
    import model.latent.train as train_module

    full_split = support.load_json(FULL_SPLIT)
    trainselect = support.load_json(TRAINSELECT_SPLIT)
    manifest = support.load_json(DATA_DIR / "manifest.json")
    metadata = support.load_json(Path(manifest["condition_metadata_file"]))
    route_payload = support.load_json(ROUTE_FILE)

    val_rows = support.collect_role_rows(
        DATA_DIR,
        full_split,
        metadata,
        "support_val_multi",
        max_cells=256,
    )
    train_rows = support.collect_role_rows(
        DATA_DIR,
        full_split,
        metadata,
        "train_multi",
        max_cells=256,
    )
    selected_spec = {
        "mode": "jaccard",
        "k": 3,
        "same_dataset": True,
        "min_score": 0.25,
    }
    cpu_available = set()
    for target in val_rows:
        pred = memory.weighted_memory_prediction(target, train_rows, **selected_spec)
        if pred is not None:
            cpu_available.add(memory.condition_key(target))

    cfg = Config()
    cfg.data_dir = str(DATA_DIR)
    cfg.biflow_dir = str(ROOT / "dataset/biFlow_data")
    cfg.pert_gene_emb_cache_dir = str(GENE_CACHE)
    cfg.use_pert_condition = True
    cfg.trackc_routed_endpoint_loss_weight = 1.0
    cfg.trackc_routed_distill_route_file = str(ROUTE_FILE)
    cfg.trackc_routed_distill_bank_split_file = str(TRAINSELECT_SPLIT)
    cfg.trackc_routed_distill_memory_mode = "jaccard"
    cfg.trackc_routed_distill_memory_k = 3
    cfg.trackc_routed_distill_memory_min_score = 0.25
    cfg.trackc_routed_distill_memory_scope = "same_dataset"
    cfg.condition_prior_bank_max_cells = 256

    train_ds = CrossDatasetFMDataset(
        str(DATA_DIR),
        trainselect,
        batch_size=8,
        seed=42,
        mode="train",
        min_cells=16,
        use_pert_condition=True,
        gene_embedding_cache_dir=str(GENE_CACHE),
        biflow_dir=str(ROOT / "dataset/biFlow_data"),
        silent=True,
    )
    try:
        bank = train_module.build_trackc_routed_distill_bank(train_ds, cfg, log=None)
        code_available = set()
        nonfinite_targets = []
        for row in val_rows:
            ds = str(row["dataset"])
            cond = str(row["condition"])
            meta = train_ds.metadata_for_condition(ds, cond)
            target = train_module.get_trackc_routed_distill_target(bank, ds, meta)
            if target is not None:
                key = (ds, cond)
                code_available.add(key)
                if not torch.isfinite(target).all():
                    nonfinite_targets.append(key)
    finally:
        train_ds.close()

    reasons: list[str] = []
    expected_routes = {"NormanWeissman2019_filtered": "train_multi_memory", "Wessels": "train_multi_memory"}
    if route_payload.get("route") != expected_routes:
        reasons.append("route_file_not_frozen_train_multi_memory")
    if set(bank.get("routes") or {}) != set(expected_routes):
        reasons.append("bank_routes_missing_expected_datasets")
    if cpu_available != code_available:
        reasons.append("code_cpu_memory_availability_mismatch")
    if nonfinite_targets:
        reasons.append("nonfinite_teacher_targets")
    summary = dict(train_module.LAST_TRACKC_ROUTED_DISTILL_SUMMARY)
    if summary.get("bank_split_file") != str(TRAINSELECT_SPLIT):
        reasons.append("teacher_bank_not_built_from_trainselect_split")
    if summary.get("memory_mode") != "jaccard":
        reasons.append("memory_mode_not_frozen_jaccard")
    if int(summary.get("memory_k") or 0) != 3:
        reasons.append("memory_k_not_frozen_3")
    if abs(float(summary.get("memory_min_score") or 0.0) - 0.25) > 1e-12:
        reasons.append("memory_min_score_not_frozen_0p25")
    if summary.get("memory_scope") != "same_dataset":
        reasons.append("memory_scope_not_frozen_same_dataset")

    payload = {
        "status": "trainonly_memory_teacher_validation_pass" if not reasons else "trainonly_memory_teacher_validation_fail",
        "reasons": reasons,
        "route_file": str(ROUTE_FILE),
        "full_split": str(FULL_SPLIT),
        "trainselect_split": str(TRAINSELECT_SPLIT),
        "data_dir": str(DATA_DIR),
        "leakage_status": "training_teacher_memory_from_trainselect_train_only_support_val_for_validation_no_query",
        "routes": bank.get("routes"),
        "memory_mode": summary.get("memory_mode"),
        "memory_k": summary.get("memory_k"),
        "memory_min_score": summary.get("memory_min_score"),
        "memory_scope": summary.get("memory_scope"),
        "bank_summary": summary,
        "n_cpu_available": len(cpu_available),
        "n_code_available": len(code_available),
        "availability_mismatch_count": len(cpu_available ^ code_available),
        "availability_mismatch": sorted(cpu_available ^ code_available),
        "nonfinite_target_count": len(nonfinite_targets),
        "nonfinite_targets": sorted(nonfinite_targets),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_md": str(OUT_MD)}))
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
