#!/usr/bin/env python3
"""CPU gate for structural RawFM gene-budget manifests.

The gate evaluates whether a newly proposed gene budget is more than another
abundance proxy before any GPU smoke is launched.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
if str(COUPLEDFM) not in sys.path:
    sys.path.insert(0, str(COUPLEDFM))

from model.data.dataset import CoupledFMDataset, _DatasetHandle  # noqa: E402
from model.data.vocab import GeneVocab  # noqa: E402
from model.paths import gene_name_path, nichenet_node2idx_path  # noqa: E402
from model.utils.data.biflow_paths import resolve_biflow_control_gt_h5ad  # noqa: E402
from model.utils.data.split import load_split_json  # noqa: E402


BUILDER_PATH = COUPLEDFM / "model/tools/build_rawfm_gene_budget_manifest.py"
DEFAULT_MANIFEST_DIR = ROOT / "reports/rawfm_structural_gene_budget_manifest_20260628"
DEFAULT_OUT = ROOT / "reports/rawfm_structural_gene_budget_gate_20260628"
DEFAULT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42.json"
DEFAULT_BIFLOW = ROOT / "dataset/biFlow_data"
DEFAULT_DATASET = "Wessels"


def load_builder_module():
    spec = importlib.util.spec_from_file_location("rawfm_gene_budget_builder_20260628", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any, digits: int = 4) -> str:
    val = safe_float(value)
    if not math.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def smd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    pooled = math.sqrt((float(np.var(a)) + float(np.var(b))) / 2.0 + 1e-12)
    return float((np.mean(a) - np.mean(b)) / pooled)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(len(set(map(int, a)) & set(map(int, b))) / len(set(map(int, a))))


def read_keep(path: Path, dataset: str) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(obj["datasets"][dataset]["keep_indices"], dtype=int)


def load_scores(dataset: str, split_file: Path, biflow_dir: Path):
    builder = load_builder_module()
    split = load_split_json(split_file)
    train_conds = list((split.get(dataset) or {}).get("train", []))
    vocab = GeneVocab(str(gene_name_path()), str(nichenet_node2idx_path()))
    pair = resolve_biflow_control_gt_h5ad(str(biflow_dir), dataset, latent_backbone="stack")
    if pair is None:
        raise RuntimeError(f"missing biflow pair for {dataset}")
    cc_p, gt_p = pair
    handle = _DatasetHandle(dataset, str(cc_p), str(gt_p), vocab)
    try:
        response_matrix = builder._condition_response_matrix(handle, train_conds)
        if response_matrix.shape[0] == 0:
            response = np.zeros(len(handle.gene_ids_valid), dtype=np.float32)
            diversity = np.zeros(len(handle.gene_ids_valid), dtype=np.float32)
        else:
            response = response_matrix.mean(axis=0).astype(np.float32)
            diversity = response_matrix.std(axis=0).astype(np.float32)
        abundance, variance, detection = builder._control_gene_moments(handle)
        residual = builder._residualize_score(response, [abundance, variance, detection])
    finally:
        handle.close()
    return {
        "response": response,
        "diversity": diversity,
        "residual": residual,
        "abundance": abundance,
        "variance": variance,
        "detection": detection,
        "n_train_conditions": len(train_conds),
    }


def loader_dryrun(manifest: Path, label: str, dataset: str, split_file: Path, biflow_dir: Path) -> dict[str, Any]:
    split = load_split_json(split_file)
    vocab = GeneVocab(str(gene_name_path()), str(nichenet_node2idx_path()))
    ds = CoupledFMDataset(
        str(biflow_dir),
        vocab,
        split,
        mode="train",
        coupling_mode="ot",
        batch_size=4,
        min_cells=16,
        ds_alpha=1.0,
        ot_method="torch_sinkhorn",
        ot_feature="raw",
        ot_emb_cap_src=8,
        ot_emb_cap_gt=8,
        gene_mask_prob=0.0,
        gene_mask_all_prob=0.0,
        gene_budget_manifest_path=str(manifest),
        gene_budget_label=label,
        dataset_names=[dataset],
        latent_backbone="stack",
        seed=42,
    )
    test_ds = None
    try:
        batch = next(iter(ds))
        x_t, x_ctrl, _t, _gene_ids, dx_t, gene_mask = batch[:6]
        mask = ds.budget_mask_for_eval(dataset)
        if mask is None:
            raise RuntimeError("budget mask missing")
        masked = mask > 0.5
        kept = ~masked
        test_ds = CoupledFMDataset(
            str(biflow_dir),
            vocab,
            split,
            mode="test",
            coupling_mode="ot",
            batch_size=4,
            min_cells=16,
            ds_alpha=1.0,
            ot_method="torch_sinkhorn",
            ot_feature="raw",
            ot_emb_cap_src=8,
            ot_emb_cap_gt=8,
            gene_mask_prob=0.0,
            gene_mask_all_prob=0.0,
            gene_budget_manifest_path=str(manifest),
            gene_budget_label=label,
            dataset_names=[dataset],
            shared_handles=ds.handles,
            latent_backbone="stack",
            seed=1042,
        )
        eval_mask = test_ds.budget_mask_for_eval(dataset)
        checks = {
            "kept_genes": int(kept.sum()),
            "masked_genes": int(masked.sum()),
            "x_t_masked_zero": bool(np.allclose(x_t.numpy()[:, masked], 0.0)),
            "x_ctrl_masked_zero": bool(np.allclose(x_ctrl.numpy()[:, masked], 0.0)),
            "dx_t_masked_zero": bool(np.allclose(dx_t.numpy()[:, masked], 0.0)),
            "gene_mask_masked_one": bool(np.all(gene_mask.numpy()[:, masked] > 0.5)),
            "gene_mask_kept_zero": bool(np.all(gene_mask.numpy()[:, kept] < 0.5)),
            "eval_mask_matches_train": bool(np.array_equal(mask, eval_mask)),
        }
    finally:
        if test_ds is not None:
            test_ds.close()
        ds.close()
    checks["pass"] = all(v for k, v in checks.items() if k not in {"kept_genes", "masked_genes"})
    return checks


def main() -> None:
    out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = DEFAULT_DATASET
    scores = load_scores(dataset, DEFAULT_SPLIT, DEFAULT_BIFLOW)

    labels = {
        "response_topk": DEFAULT_MANIFEST_DIR / "response_topk_k256_seed42.json",
        "response_abundance_residual_topk": DEFAULT_MANIFEST_DIR
        / "response_abundance_residual_topk_k256_seed42.json",
        "condition_diversity_topk": DEFAULT_MANIFEST_DIR / "condition_diversity_topk_k256_seed42.json",
        "abundance_topk": DEFAULT_MANIFEST_DIR / "abundance_topk_k256_seed42.json",
        "abundance_matched_random": DEFAULT_MANIFEST_DIR / "abundance_matched_random_k256_seed42.json",
        "residual_abundance_matched_random": DEFAULT_MANIFEST_DIR
        / "residual_abundance_matched_random_k256_seed42.json",
        "residual_confound_matched_random": DEFAULT_MANIFEST_DIR
        / "residual_confound_matched_random_k256_seed42.json",
        "random_gene_set": DEFAULT_MANIFEST_DIR / "random_gene_set_k256_seed42.json",
    }
    keeps = {label: read_keep(path, dataset) for label, path in labels.items()}
    rows: list[dict[str, Any]] = []
    for label, keep in keeps.items():
        rows.append(
            {
                "label": label,
                "n_genes": int(len(keep)),
                "response_mean": float(np.mean(scores["response"][keep])),
                "residual_mean": float(np.mean(scores["residual"][keep])),
                "diversity_mean": float(np.mean(scores["diversity"][keep])),
                "abundance_mean": float(np.mean(scores["abundance"][keep])),
                "variance_mean": float(np.mean(scores["variance"][keep])),
                "detection_mean": float(np.mean(scores["detection"][keep])),
                "overlap_response_topk": overlap(keep, keeps["response_topk"]),
                "overlap_abundance_topk": overlap(keep, keeps["abundance_topk"]),
                "overlap_residual_topk": overlap(keep, keeps["response_abundance_residual_topk"]),
            }
        )
    metric_df = pd.DataFrame(rows)

    cand = keeps["response_abundance_residual_topk"]
    conf = keeps["residual_confound_matched_random"]
    abundance_match = keeps["residual_abundance_matched_random"]
    diversity = keeps["condition_diversity_topk"]
    residual_delta_confound = float(np.mean(scores["residual"][cand]) - np.mean(scores["residual"][conf]))
    residual_delta_abundance_match = float(
        np.mean(scores["residual"][cand]) - np.mean(scores["residual"][abundance_match])
    )
    diversity_delta_random = float(np.mean(scores["diversity"][diversity]) - np.mean(scores["diversity"][keeps["random_gene_set"]]))
    confound_rows = [
        {
            "comparison": "residual_topk_vs_confound_matched_random",
            "abundance_smd": smd(scores["abundance"][cand], scores["abundance"][conf]),
            "variance_smd": smd(scores["variance"][cand], scores["variance"][conf]),
            "detection_smd": smd(scores["detection"][cand], scores["detection"][conf]),
            "residual_delta": residual_delta_confound,
        },
        {
            "comparison": "residual_topk_vs_abundance_matched_random",
            "abundance_smd": smd(scores["abundance"][cand], scores["abundance"][abundance_match]),
            "variance_smd": smd(scores["variance"][cand], scores["variance"][abundance_match]),
            "detection_smd": smd(scores["detection"][cand], scores["detection"][abundance_match]),
            "residual_delta": residual_delta_abundance_match,
        },
        {
            "comparison": "condition_diversity_topk_vs_random",
            "abundance_smd": smd(scores["abundance"][diversity], scores["abundance"][keeps["random_gene_set"]]),
            "variance_smd": smd(scores["variance"][diversity], scores["variance"][keeps["random_gene_set"]]),
            "detection_smd": smd(scores["detection"][diversity], scores["detection"][keeps["random_gene_set"]]),
            "diversity_delta": diversity_delta_random,
        },
    ]
    confound_df = pd.DataFrame(confound_rows)

    loader_rows = []
    for label in [
        "response_abundance_residual_topk",
        "residual_confound_matched_random",
        "condition_diversity_topk",
    ]:
        checks = loader_dryrun(labels[label], label, dataset, DEFAULT_SPLIT, DEFAULT_BIFLOW)
        loader_rows.append({"label": label, **checks})
    loader_df = pd.DataFrame(loader_rows)

    residual_gate = bool(
        residual_delta_confound > 0.25
        and residual_delta_abundance_match > 0.25
        and abs(confound_rows[0]["abundance_smd"]) <= 0.75
        and abs(confound_rows[0]["variance_smd"]) <= 0.75
        and abs(confound_rows[0]["detection_smd"]) <= 0.75
        and metric_df.loc[
            metric_df["label"] == "response_abundance_residual_topk",
            "overlap_abundance_topk",
        ].iloc[0]
        <= 0.70
    )
    diversity_gate = bool(
        diversity_delta_random > 0.0
        and metric_df.loc[metric_df["label"] == "condition_diversity_topk", "overlap_abundance_topk"].iloc[0]
        <= 0.80
    )
    loader_gate = bool(loader_df["pass"].all())
    gpu_packet_authorized = bool(loader_gate and (residual_gate or diversity_gate))
    status = (
        "rawfm_structural_gene_budget_gate_gpu_packet_ready"
        if gpu_packet_authorized
        else "rawfm_structural_gene_budget_gate_fail_no_gpu"
    )

    metric_csv = out_dir / "rawfm_structural_gene_budget_metric_rows.csv"
    confound_csv = out_dir / "rawfm_structural_gene_budget_confound_rows.csv"
    loader_csv = out_dir / "rawfm_structural_gene_budget_loader_rows.csv"
    metric_df.to_csv(metric_csv, index=False)
    confound_df.to_csv(confound_csv, index=False)
    loader_df.to_csv(loader_csv, index=False)
    payload = {
        "timestamp": now_cst(),
        "status": status,
        "gpu_packet_authorized": gpu_packet_authorized,
        "dataset": dataset,
        "split_file": str(DEFAULT_SPLIT),
        "manifest_dir": str(DEFAULT_MANIFEST_DIR),
        "n_train_conditions": int(scores["n_train_conditions"]),
        "residual_gate": residual_gate,
        "diversity_gate": diversity_gate,
        "loader_gate": loader_gate,
        "residual_delta_confound": residual_delta_confound,
        "residual_delta_abundance_match": residual_delta_abundance_match,
        "diversity_delta_random": diversity_delta_random,
        "outputs": {
            "metric_csv": str(metric_csv),
            "confound_csv": str(confound_csv),
            "loader_csv": str(loader_csv),
        },
    }
    json_path = out_dir / "rawfm_structural_gene_budget_gate_20260628.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# RawFM Structural Gene-Budget Gate",
        "",
        f"Timestamp: `{payload['timestamp']}`",
        "",
        f"Status: `{status}`",
        "",
        f"GPU packet authorized: `{gpu_packet_authorized}`",
        "",
        "## Boundary",
        "",
        "- CPU-only manifest/loader audit for Wessels k=256.",
        "- No training, no inference, no GPU, no canonical multi, no Track C query.",
        "- Candidate budgets are train-only and evaluated against abundance/random confound controls before launch.",
        "",
        "## Metric Rows",
        "",
        "| label | residual | response | diversity | abundance | overlap abundance | overlap residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metric_df.iterrows():
        lines.append(
            f"| {row['label']} | {fmt(row['residual_mean'])} | {fmt(row['response_mean'])} | "
            f"{fmt(row['diversity_mean'])} | {fmt(row['abundance_mean'])} | "
            f"{fmt(row['overlap_abundance_topk'])} | {fmt(row['overlap_residual_topk'])} |"
        )
    lines.extend(
        [
            "",
            "## Confound Checks",
            "",
            "| comparison | residual/diversity delta | abundance SMD | variance SMD | detection SMD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in confound_df.iterrows():
        delta = row.get("residual_delta", row.get("diversity_delta", float("nan")))
        lines.append(
            f"| {row['comparison']} | {fmt(delta)} | {fmt(row['abundance_smd'])} | "
            f"{fmt(row['variance_smd'])} | {fmt(row['detection_smd'])} |"
        )
    lines.extend(
        [
            "",
            "## Loader Checks",
            "",
            "| label | kept | pass |",
            "|---|---:|---:|",
        ]
    )
    for _, row in loader_df.iterrows():
        lines.append(f"| {row['label']} | {int(row['kept_genes'])} | {bool(row['pass'])} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- residual gate: `{residual_gate}`",
            f"- diversity gate: `{diversity_gate}`",
            f"- loader gate: `{loader_gate}`",
        ]
    )
    if gpu_packet_authorized:
        lines.append(
            "- GPU packet: launch fixed-step/no-selection RawFM smokes for the passing candidate(s) plus matched controls; promotion still requires beating controls on final-only Wessels metrics."
        )
    else:
        lines.append("- No GPU launch: structural budgets are still too abundance/confound-linked or loader checks failed.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- metrics: `{metric_csv}`",
            f"- confounds: `{confound_csv}`",
            f"- loader: `{loader_csv}`",
            f"- JSON: `{json_path}`",
            "",
        ]
    )
    (out_dir / "LATENTFM_RAWFM_STRUCTURAL_GENE_BUDGET_GATE_20260628.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
