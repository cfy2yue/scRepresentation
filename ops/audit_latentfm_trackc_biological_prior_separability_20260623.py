#!/usr/bin/env python3
"""CPU-only separability audit for biological-prior Track C gates.

The current frozen Track C blend is diagnostic because the route is a scope
oracle. This audit asks whether a materially new condition/dataset biological
prior (GO overlap, scGPT gene-embedding geometry, CellNavi gene-embedding
geometry) could distinguish support-val multi rows from canonical family_gene
rows before any GPU work.

No held-out query artifacts are read, and no model is trained. Canonical
family_gene is used only as a no-harm collision audit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
HELPER = ROOT / "ops/summarize_latentfm_trackc_anchor_gated_support_teacher_cpu_gate_20260623.py"
DEFAULT_RUN_ROOT = (
    ROOT
    / "runs/latentfm_trackc_anchor_gated_support_teacher_artifacts_20260623/"
    "xverse_support_film_retry1_condition_means_artifacts"
)
TRAINSELECT_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
GO_PICKLE = ROOT / "scFM_third_party/scFoundation/GEARS/data/gene2go.pkl"
SCGPT_CACHE = ROOT / "pretrainckpt/genepert_cache/scgpt_embed_gene"
CELLNAVI_CACHE = ROOT / "pretrainckpt/genepert_cache/cellnavi_embed_gene"
UCE_ESM2 = ROOT / "scFM_pretrained/uce/model_files/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
OUT_JSON = ROOT / "reports/latentfm_trackc_biological_prior_separability_20260623.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_BIOLOGICAL_PRIOR_SEPARABILITY_20260623.md"
ALPHA = 0.75


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("anchor_gate_helper", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {HELPER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_parts(condition: str) -> list[str]:
    return [part.strip().upper() for part in str(condition).replace("_", "+").split("+") if part.strip()]


def condition_key(dataset: str, condition: str) -> tuple[str, str]:
    return str(dataset), "+".join(condition_parts(condition))


def load_gene_cache(cache_dir: Path) -> dict[str, Any]:
    index_path = cache_dir / "gene_index.tsv"
    emb_path = cache_dir / "gene_embeddings.npy"
    gene_to_idx: dict[str, int] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            gene_to_idx[parts[0].strip().upper()] = int(parts[-1])
        except ValueError:
            continue
    emb = np.load(emb_path)
    norms = np.linalg.norm(emb, axis=1)
    norms[norms == 0] = 1.0
    emb = emb / norms[:, None]
    return {"path": str(cache_dir), "n_genes": len(gene_to_idx), "dim": int(emb.shape[1]), "gene_to_idx": gene_to_idx, "emb": emb}


def pair_cos(cache: dict[str, Any], genes: list[str]) -> dict[str, Any]:
    vals = []
    missing = []
    for gene in genes:
        if gene not in cache["gene_to_idx"]:
            missing.append(gene)
    for a, b in combinations(genes, 2):
        ia = cache["gene_to_idx"].get(a)
        ib = cache["gene_to_idx"].get(b)
        if ia is None or ib is None:
            continue
        vals.append(float(np.dot(cache["emb"][ia], cache["emb"][ib])))
    return {
        "coverage": float((len(genes) - len(missing)) / len(genes)) if genes else 0.0,
        "missing": missing,
        "pair_cos_mean": None if not vals else float(mean(vals)),
        "pair_cos_min": None if not vals else float(min(vals)),
        "pair_cos_max": None if not vals else float(max(vals)),
    }


def go_features(gene2go: dict[str, set[str]], genes: list[str]) -> dict[str, Any]:
    go_sets = [set(gene2go.get(gene, set())) for gene in genes]
    covered = sum(1 for s in go_sets if s)
    vals = []
    for a, b in combinations(go_sets, 2):
        if not a or not b:
            continue
        union = a | b
        vals.append(float(len(a & b) / len(union)) if union else 0.0)
    return {
        "coverage": float(covered / len(genes)) if genes else 0.0,
        "pair_jaccard_mean": None if not vals else float(mean(vals)),
        "pair_jaccard_min": None if not vals else float(min(vals)),
        "pair_jaccard_max": None if not vals else float(max(vals)),
    }


def row_delta(mod: Any, row: dict[str, Any], gate: float = 1.0) -> float | None:
    pred = row["pred_anchor"] + gate * ALPHA * (row["pred_teacher"] - row["pred_anchor"])
    pp = mod.pearson_np(pred - row["pert_mean"], row["gt_mean"] - row["pert_mean"])
    if pp is None or row.get("anchor_pp") is None:
        return None
    return float(pp - row["anchor_pp"])


def feature_row(
    row: dict[str, Any],
    gene2go: dict[str, set[str]],
    scgpt: dict[str, Any],
    cellnavi: dict[str, Any],
) -> dict[str, Any]:
    genes = condition_parts(row["condition"])
    go = go_features(gene2go, genes)
    sg = pair_cos(scgpt, genes)
    cn = pair_cos(cellnavi, genes)
    return {
        "dataset": str(row["dataset"]),
        "condition": "+".join(genes),
        "genes": genes,
        "n_genes": len(genes),
        "go_coverage": go["coverage"],
        "go_pair_jaccard_mean": go["pair_jaccard_mean"],
        "scgpt_coverage": sg["coverage"],
        "scgpt_pair_cos_mean": sg["pair_cos_mean"],
        "cellnavi_coverage": cn["coverage"],
        "cellnavi_pair_cos_mean": cn["pair_cos_mean"],
    }


def rounded_signature(feat: dict[str, Any]) -> tuple[Any, ...]:
    def r(v: Any) -> Any:
        return None if v is None else round(float(v), 8)

    return (
        feat["dataset"],
        feat["condition"],
        tuple(feat["genes"]),
        r(feat["go_pair_jaccard_mean"]),
        r(feat["scgpt_pair_cos_mean"]),
        r(feat["cellnavi_pair_cos_mean"]),
    )


def summarize_values(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(vals), "mean": float(mean(vals)), "min": float(min(vals)), "max": float(max(vals))}


def render(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Track C Biological-Prior Separability Audit",
        "",
        f"Status: `{decision['status']}`",
        f"GPU authorization: `{decision['gpu_authorization']}`",
        "",
        "## Scope",
        "",
        "Short CPU-only audit. Inputs are GO annotations, scGPT/CellNavi gene embeddings, safe trainselect split, and frozen condition-mean artifacts. Held-out query is not read.",
        "",
        "## Biological Feature Sources",
        "",
    ]
    for name, info in payload["feature_sources"].items():
        lines.append(f"- `{name}`: `{info['status']}`; {info.get('summary', '')}")
    lines.extend(
        [
            "",
            "## Exact Collision Result",
            "",
            f"- support rows: `{payload['collision']['support_rows']}`",
            f"- support rows with exact canonical family_gene match: `{payload['collision']['exact_family_matches']}`",
            f"- feature signatures identical across support/canonical exact matches: `{payload['collision']['identical_feature_signatures']}`",
            f"- maximum support rows usable by a condition/dataset-only gate while preserving exact canonical family no-op: `{payload['collision']['max_support_rows_under_exact_family_noop']}`",
            "",
            "## Feature Distributions On Support Rows",
            "",
            "| feature | n | mean | min | max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, stats in payload["support_feature_stats"].items():
        lines.append(
            f"| `{key}` | {stats['n']} | {fmt(stats['mean'])} | {fmt(stats['min'])} | {fmt(stats['max'])} |"
        )
    lines.extend(
        [
            "",
            "## Canonical Family Collision Harm If Opened",
            "",
            "| dataset | condition | support pp delta if opened | canonical family pp delta if opened | GO Jaccard | scGPT cos | CellNavi cos |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["collision_examples"][:24]:
        lines.append(
            f"| {row['dataset']} | `{row['condition']}` | {fmt(row['support_delta_pp_if_opened'])} | "
            f"{fmt(row['canonical_family_delta_pp_if_opened'])} | {fmt(row['go_pair_jaccard_mean'])} | "
            f"{fmt(row['scgpt_pair_cos_mean'])} | {fmt(row['cellnavi_pair_cos_mean'])} |"
        )
    lines.extend(["", "## Decision Reasons", ""])
    lines.extend(f"- `{reason}`" for reason in decision["reasons"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "GO and pretrained gene-embedding features are materially new information sources, but they are still condition/dataset-only. Because every support-val row has an exact canonical family_gene counterpart with identical biological features, such gates cannot open on support while staying exact no-op on canonical family. This closes biological-prior threshold gates for the current frozen support-teacher residual unless the feature includes a legitimate support-context/protocol signal or a different training objective.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--split-file", type=Path, default=TRAINSELECT_SPLIT)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mod = load_helper()
    mean_dir = args.run_root / "condition_means"
    split = load_json(args.split_file)
    gene2go = pickle.load(GO_PICKLE.open("rb"))
    scgpt = load_gene_cache(SCGPT_CACHE)
    cellnavi = load_gene_cache(CELLNAVI_CACHE)

    support_rows = mod.paired_rows(
        load_json(mean_dir / "support_anchor_split_condition_means_ode20.json"),
        load_json(mean_dir / "support_candidate_split_condition_means_ode20.json"),
        "test_multi",
    )
    canonical_family = mod.paired_rows(
        load_json(mean_dir / "canonical_anchor_family_gene_condition_means_ode20.json"),
        load_json(mean_dir / "canonical_candidate_family_gene_condition_means_ode20.json"),
        "family_gene",
    )
    family_by_key = {condition_key(r["dataset"], r["condition"]): r for r in canonical_family}

    support_features = []
    collision_examples = []
    exact_matches = 0
    identical_signatures = 0
    for row in support_rows:
        feat = feature_row(row, gene2go, scgpt, cellnavi)
        feat["support_delta_pp_if_opened"] = row_delta(mod, row, gate=1.0)
        support_features.append(feat)
        key = condition_key(row["dataset"], row["condition"])
        fam = family_by_key.get(key)
        if fam is None:
            continue
        exact_matches += 1
        fam_feat = feature_row(fam, gene2go, scgpt, cellnavi)
        if rounded_signature(feat) == rounded_signature(fam_feat):
            identical_signatures += 1
        collision_examples.append(
            {
                **feat,
                "canonical_family_delta_pp_if_opened": row_delta(mod, fam, gate=1.0),
            }
        )

    max_support_rows_under_noop = len(support_rows) - exact_matches
    reasons = []
    if exact_matches == len(support_rows):
        reasons.append("all_support_rows_have_exact_canonical_family_collision")
    if identical_signatures == exact_matches:
        reasons.append("biological_condition_features_identical_for_all_exact_collisions")
    if max_support_rows_under_noop == 0:
        reasons.append("condition_dataset_biological_gate_cannot_improve_support_under_exact_family_noop")

    status = "trackc_biological_prior_separability_fail_no_gpu" if reasons else "trackc_biological_prior_separability_pass_protocol_next"
    payload = {
        "heldout_query_used": False,
        "canonical_multi_selection_used": False,
        "split_file": str(args.split_file),
        "run_root": str(args.run_root),
        "alpha": ALPHA,
        "feature_sources": {
            "gene2go": {"status": "loaded", "summary": f"{len(gene2go)} genes from {GO_PICKLE}"},
            "scgpt_gene_embedding": {"status": "loaded", "summary": f"{scgpt['n_genes']} genes x {scgpt['dim']} dims from {SCGPT_CACHE}"},
            "cellnavi_gene_embedding": {"status": "loaded", "summary": f"{cellnavi['n_genes']} genes x {cellnavi['dim']} dims from {CELLNAVI_CACHE}"},
            "uce_esm2_protein_embedding": {"status": "unavailable_no_torch_in_current_env", "summary": str(UCE_ESM2)},
        },
        "collision": {
            "support_rows": len(support_rows),
            "exact_family_matches": exact_matches,
            "identical_feature_signatures": identical_signatures,
            "max_support_rows_under_exact_family_noop": max_support_rows_under_noop,
        },
        "support_feature_stats": {
            key: summarize_values(support_features, key)
            for key in ("go_pair_jaccard_mean", "scgpt_pair_cos_mean", "cellnavi_pair_cos_mean")
        },
        "collision_examples": collision_examples,
        "decision": {
            "status": status,
            "gpu_authorization": "none",
            "next_authorization": "none" if reasons else "protocol_gate_only",
            "reasons": reasons,
        },
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(args.out_md), "out_json": str(args.out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
