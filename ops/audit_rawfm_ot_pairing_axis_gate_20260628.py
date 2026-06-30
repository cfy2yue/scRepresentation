#!/usr/bin/env python3
"""CPU gate for RawFM OT pairing-cap/sample-mode ablations."""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
if str(COUPLEDFM) not in sys.path:
    sys.path.insert(0, str(COUPLEDFM))

from model.data.dataset import CoupledFMDataset  # noqa: E402
from model.data.vocab import GeneVocab  # noqa: E402
from model.paths import gene_name_path, nichenet_node2idx_path  # noqa: E402
from model.utils.data.split import load_split_json  # noqa: E402


OUT_DIR = ROOT / "reports/rawfm_ot_pairing_axis_gate_20260628"
SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
BIFLOW = ROOT / "dataset/biFlow_data"
MANIFEST = ROOT / "reports/rawfm_lowdose_residual_manifest_20260628/residual64_abundance96_random96_k256_seed42.json"
DATASET = "Wessels"


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def make_dataset(split: dict, vocab: GeneVocab, label: str, cap: int | None, mode: str) -> CoupledFMDataset:
    return CoupledFMDataset(
        str(BIFLOW),
        vocab,
        split,
        mode="train",
        coupling_mode="ot",
        batch_size=4,
        min_cells=16,
        ds_alpha=1.0,
        ot_method="torch_sinkhorn",
        ot_feature="raw",
        ot_emb_cap_src=cap,
        ot_emb_cap_gt=cap,
        ot_sample_mode=mode,
        gene_mask_prob=0.0,
        gene_mask_all_prob=0.0,
        gene_budget_manifest_path=str(MANIFEST),
        gene_budget_label=label,
        dataset_names=[DATASET],
        latent_backbone="stack",
        seed=42,
    )


def pool_summary(ds: CoupledFMDataset) -> dict[str, Any]:
    h = ds.handles[DATASET]
    rows = []
    for cond in ds.ds_conds[DATASET]:
        rows.append(
            {
                "cond": cond,
                "src_pool": int(len(h.pert_cond2idx[cond])),
                "gt_pool": int(len(h.gt_cond2idx[cond])),
            }
        )
    src = np.asarray([r["src_pool"] for r in rows], dtype=float)
    gt = np.asarray([r["gt_pool"] for r in rows], dtype=float)
    return {
        "n_train_conditions": int(len(rows)),
        "src_pool_min": int(np.min(src)),
        "src_pool_median": float(np.median(src)),
        "gt_pool_min": int(np.min(gt)),
        "gt_pool_median": float(np.median(gt)),
    }


def audit_one(split: dict, vocab: GeneVocab, label: str, cap: int | None, mode: str) -> dict[str, Any]:
    t0 = time.time()
    ds = make_dataset(split, vocab, label, cap, mode)
    try:
        pools = pool_summary(ds)
        batch = next(iter(ds))
        x_t, x_ctrl, _t, _gene_ids, dx_t, gene_mask = batch[:6]
        mask = ds.budget_mask_for_eval(DATASET)
        if mask is None:
            raise RuntimeError("budget mask missing")
        masked = mask > 0.5
        kept = ~masked
        row = {
            "label": label,
            "requested_cap": "default_batch" if cap is None else int(cap),
            "effective_cap_src": int(ds._cap_src_eff),
            "effective_cap_gt": int(ds._cap_gt_eff),
            "ot_sample_mode": str(ds.ot_sample_mode),
            "elapsed_sec": time.time() - t0,
            "batch_shape": "x".join(map(str, x_t.shape)),
            "kept_genes": int(kept.sum()),
            "masked_genes": int(masked.sum()),
            "x_t_masked_zero": bool(np.allclose(x_t.numpy()[:, masked], 0.0)),
            "x_ctrl_masked_zero": bool(np.allclose(x_ctrl.numpy()[:, masked], 0.0)),
            "dx_t_masked_zero": bool(np.allclose(dx_t.numpy()[:, masked], 0.0)),
            "gene_mask_masked_one": bool(np.all(gene_mask.numpy()[:, masked] > 0.5)),
            "gene_mask_kept_zero": bool(np.all(gene_mask.numpy()[:, kept] < 0.5)),
            **pools,
        }
        row["cap_is_meaningful"] = bool(
            int(row["effective_cap_src"]) > 4
            and int(row["effective_cap_gt"]) > 4
            and int(row["src_pool_min"]) >= int(row["effective_cap_src"])
            and int(row["gt_pool_min"]) >= int(row["effective_cap_gt"])
        )
        row["pass"] = bool(
            row["kept_genes"] == 256
            and row["x_t_masked_zero"]
            and row["x_ctrl_masked_zero"]
            and row["dx_t_masked_zero"]
            and row["gene_mask_masked_one"]
            and row["gene_mask_kept_zero"]
        )
        return row
    finally:
        ds.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split = load_split_json(SPLIT)
    vocab = GeneVocab(str(gene_name_path()), str(nichenet_node2idx_path()))
    specs = [
        ("default_assignment", None, "assignment"),
        ("cap32_assignment", 32, "assignment"),
        ("cap32_multinomial", 32, "multinomial"),
        ("cap64_assignment", 64, "assignment"),
        ("cap64_multinomial", 64, "multinomial"),
    ]
    rows = [audit_one(split, vocab, label, cap, mode) for label, cap, mode in specs]
    df = pd.DataFrame(rows)
    loader_gate = bool(df["pass"].all())
    cap_gate = bool(df.loc[df["label"].str.startswith("cap32"), "cap_is_meaningful"].all())
    gpu_packet_authorized = bool(loader_gate and cap_gate)
    status = "rawfm_ot_pairing_axis_gate_gpu_packet_ready" if gpu_packet_authorized else "rawfm_ot_pairing_axis_gate_fail_no_gpu"

    csv_path = OUT_DIR / "rawfm_ot_pairing_axis_gate_rows.csv"
    df.to_csv(csv_path, index=False)
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_packet_authorized": gpu_packet_authorized,
        "loader_gate": loader_gate,
        "cap_gate": cap_gate,
        "manifest": str(MANIFEST),
        "rows_csv": str(csv_path),
    }
    json_path = OUT_DIR / "rawfm_ot_pairing_axis_gate_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RawFM OT Pairing Axis Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU packet authorized: `{gpu_packet_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU/loader-only audit over Wessels low-dose residual64 manifest.",
        "- No training, no inference, no GPU, no checkpoint selection.",
        "- Tests whether OT candidate-pool cap and sample mode are launchable experimental axes.",
        "",
        "## Rows",
        "",
        "| label | requested cap | effective cap | mode | kept | src min/median | gt min/median | meaningful | pass | sec |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['label']} | {row['requested_cap']} | {int(row['effective_cap_src'])}/{int(row['effective_cap_gt'])} | "
            f"{row['ot_sample_mode']} | {int(row['kept_genes'])} | "
            f"{int(row['src_pool_min'])}/{fmt(row['src_pool_median'])} | "
            f"{int(row['gt_pool_min'])}/{fmt(row['gt_pool_median'])} | "
            f"{bool(row['cap_is_meaningful'])} | {bool(row['pass'])} | {fmt(row['elapsed_sec'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- loader gate: `{loader_gate}`",
            f"- cap gate: `{cap_gate}`",
        ]
    )
    if gpu_packet_authorized:
        lines.append(
            "- GPU packet is allowed after the current short-warm/low-dose gates: compare default minibatch OT with cap32 assignment/multinomial using the same split and final-only protocol. Cap64 remains diagnostic because at least one Wessels training condition has only 53 GT cells."
        )
    else:
        lines.append("- No GPU launch from OT axis yet; fix loader/cap gate first.")
    lines.extend(["", "## Outputs", "", f"- rows: `{csv_path}`", f"- JSON: `{json_path}`", ""])
    (OUT_DIR / "LATENTFM_RAWFM_OT_PAIRING_AXIS_GATE_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
