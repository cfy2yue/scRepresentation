#!/usr/bin/env python3
"""CPU-only MMD-preservation gate for a modality/pathway sampling redesign.

This gate asks whether a materially new pathway-quota split, constrained by
train-only MMD-risk evidence, is eligible for exactly one bounded GPU smoke.
It reads only train-only/internal artifacts and local metadata. It does not
read canonical metrics, canonical multi, Track C query, active logs, or launch
GPU work.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/cyx/1030/scLatent")
REPORTS = ROOT / "reports"
BIOFLOW = ROOT / "dataset/biFlow_data"
DATA_DIR = ROOT / "dataset/latentfm_full/xverse"

INVENTORY = REPORTS / "latentfm_condition_level_inventory_20260624.json"
BASE_SPLIT = BIOFLOW / "split_seed42_xverse_trainonly_crossbg_val_v2.json"
CAP120_SPLIT = BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_cap120_all_v2.json"
TYPE_BALANCED_SPLIT = BIOFLOW / "xverse_scaling_splits_v2_20260624/split_seed42_xverse_trainonly_scaling_type_balanced_cap120_v2.json"
QUOTA12_SPLIT = BIOFLOW / "xverse_modality_pathway_sampling_splits_20260624/split_seed42_xverse_modality_pathway_quota12_cap120_parent.json"
RANDOMCOUNT_SPLIT = BIOFLOW / "xverse_modality_pathway_sampling_splits_20260624/split_seed42_xverse_modality_pathway_randomcount_cap120_parent.json"

QUOTA_DECISION_MD = REPORTS / "LATENTFM_MODALITY_PATHWAY_SAMPLING_SMOKE_DECISION_20260624.md"
RANDOMCOUNT_DECISION_MD = REPORTS / "LATENTFM_MODALITY_PATHWAY_RANDOMCOUNT_CONTROL_SMOKE_DECISION_20260624.md"
RANDOMCOUNT_MMD_GATE_MD = REPORTS / "LATENTFM_RANDOMCOUNT_MMD_PRESERVATION_GATE_20260624.md"
MATCHED_BREADTH_MD = REPORTS / "LATENTFM_MATCHED_DATASET_BREADTH_GATE_20260624.md"
OT_OVERLAP_MD = REPORTS / "LATENTFM_OT_CONDITION_OVERLAP_RELIABILITY_GATE_20260624.md"

QUOTA_POSTHOC = (
    ROOT
    / "runs/latentfm_modality_pathway_sampling_smoke_20260624"
    / "xverse_scaling_pathway_quota12_3k_seed42/posthoc_eval_internal"
)
RANDOM_POSTHOC = (
    ROOT
    / "runs/latentfm_modality_pathway_randomcount_control_smoke_20260624"
    / "xverse_scaling_pathway_randomcount_3k_seed42/posthoc_eval_internal"
)
CAP120_MEANS = ROOT / "runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/xverse_trainonly_scaling_cap120_all_v2_pert_means.npz"
QUOTA12_MEANS = ROOT / "runs/latentfm_modality_pathway_sampling_artifacts_20260624/artifacts/pathway_quota12_cap120_parent_trainonly_pert_means.npz"
RANDOMCOUNT_MEANS = ROOT / "runs/latentfm_modality_pathway_sampling_artifacts_20260624/artifacts/pathway_randomcount_cap120_parent_trainonly_pert_means.npz"
TYPE_BALANCED_MEANS = ROOT / "runs/latentfm_xverse_scaling_splits_v2_20260624/artifacts/xverse_trainonly_scaling_type_balanced_cap120_v2_pert_means.npz"

SPLIT_HELPER = ROOT / "ops/build_latentfm_xverse_scaling_splits_20260624.py"
OLD_GATE_SCRIPT = ROOT / "ops/audit_latentfm_modality_pathway_sampling_gate_20260624.py"
OLD_MATERIALIZER = ROOT / "ops/materialize_latentfm_modality_pathway_sampling_artifacts_20260624.py"
RANDOM_MATERIALIZER = ROOT / "ops/materialize_latentfm_modality_pathway_randomcount_control_20260624.py"
RANDOM_MMD_SCRIPT = ROOT / "ops/audit_latentfm_randomcount_mmd_preservation_gate_20260624.py"

OUT_JSON = REPORTS / "latentfm_modality_pathway_mmd_preservation_gate_20260624.json"
OUT_MD = REPORTS / "LATENTFM_MODALITY_PATHWAY_MMD_PRESERVATION_GATE_20260624.md"
OUT_SPLIT = REPORTS / "latentfm_modality_pathway_mmd_preservation_candidate_split_20260624.json"

CHEMICAL_DATASETS = {"sciplex3_A549", "sciplex3_K562", "sciplex3_MCF7"}
SEED = 20260624
TARGET_CHEMICAL_PER_BACKGROUND = 112
PATHWAY_SOFT_CAP_PER_BACKGROUND = 15


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_score(*parts: object) -> str:
    return hashlib.sha256("\t".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def norm_drug(row: dict[str, Any]) -> str:
    backgrounds = row.get("backgrounds") or []
    background = str(backgrounds[0]) if backgrounds else ""
    perturbation = str(row.get("perturbation") or "")
    if background and perturbation.startswith(f"{background}_"):
        return perturbation[len(background) + 1 :]
    condition = str(row.get("condition") or "")
    if background and condition.startswith(f"{background}_"):
        return condition[len(background) + 1 :].rsplit("_", 1)[0]
    return perturbation


def build_chemical_meta(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        ds = str(row.get("dataset") or "")
        if ds not in CHEMICAL_DATASETS:
            continue
        drug = norm_drug(row)
        if not drug:
            continue
        item = meta.setdefault((ds, drug), {"pathways": set(), "doses": set(), "backgrounds": set()})
        pathway = str(row.get("pathway") or "")
        dose = str(row.get("dose") or "")
        if pathway:
            item["pathways"].add(pathway)
        if dose:
            item["doses"].add(dose)
        for background in row.get("backgrounds") or []:
            item["backgrounds"].add(str(background))
    return meta


def pathway_for(meta: dict[tuple[str, str], dict[str, Any]], ds: str, cond: str) -> str:
    pathways = sorted((meta.get((ds, cond)) or {}).get("pathways") or [])
    return pathways[0] if len(pathways) == 1 else "__missing_or_multi_pathway__"


def copy_groups(groups: dict[str, Any], train: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in groups.items():
        out[key] = [str(x) for x in val] if isinstance(val, list) else val
    out["train"] = sorted(str(x) for x in train)
    return out


def train_set(split: dict[str, Any], *, chemical_only: bool = False) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for ds, groups in split.items():
        ds_s = str(ds)
        if chemical_only and ds_s not in CHEMICAL_DATASETS:
            continue
        for cond in groups.get("train") or []:
            out.add((ds_s, str(cond)))
    return out


def jaccard(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
    return float(len(a & b) / max(1, len(a | b)))


def l1_distribution(a: Counter[str], b: Counter[str]) -> float:
    keys = set(a) | set(b)
    na = sum(a.values())
    nb = sum(b.values())
    return sum(abs((a[k] / max(1, na)) - (b[k] / max(1, nb))) for k in keys)


def build_candidate(
    *,
    cap120: dict[str, Any],
    quota12: dict[str, Any],
    meta: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Keep the MMD-safe quota12 core and add back a small pathway-balanced buffer."""
    out: dict[str, Any] = {}
    for ds, groups in sorted(cap120.items()):
        ds_s = str(ds)
        cap_train = [str(x) for x in groups.get("train") or []]
        if ds_s not in CHEMICAL_DATASETS:
            out[ds_s] = copy_groups(groups, cap_train)
            continue
        selected = {str(x) for x in (quota12.get(ds_s) or {}).get("train") or []}
        pathway_counts = Counter(pathway_for(meta, ds_s, cond) for cond in selected)
        parent_counts = Counter(pathway_for(meta, ds_s, cond) for cond in cap_train)
        pool = [cond for cond in cap_train if cond not in selected]

        def addback_key(cond: str) -> tuple[float, int, str]:
            pathway = pathway_for(meta, ds_s, cond)
            parent_frac = parent_counts[pathway] / max(1, sum(parent_counts.values()))
            selected_frac = pathway_counts[pathway] / max(1, sum(pathway_counts.values()))
            return (selected_frac - parent_frac, pathway_counts[pathway], stable_score(SEED, ds_s, pathway, cond))

        for cond in sorted(pool, key=addback_key):
            if len(selected) >= TARGET_CHEMICAL_PER_BACKGROUND:
                break
            pathway = pathway_for(meta, ds_s, cond)
            if pathway_counts[pathway] >= PATHWAY_SOFT_CAP_PER_BACKGROUND:
                continue
            selected.add(cond)
            pathway_counts[pathway] += 1

        # Fallback should rarely be needed; it preserves subset/validation safety
        # while making the target count deterministic if a background is sparse.
        for cond in sorted(pool, key=lambda c: stable_score(SEED + 1, ds_s, c)):
            if len(selected) >= TARGET_CHEMICAL_PER_BACKGROUND:
                break
            selected.add(cond)
        out[ds_s] = copy_groups(groups, sorted(selected))
    return out


def summarize_split(split: dict[str, Any], meta: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    pathway_counts: Counter[str] = Counter()
    dose_counts: Counter[str] = Counter()
    dataset_counts: dict[str, int] = {}
    missing_meta = []
    for ds in sorted(CHEMICAL_DATASETS):
        train = [str(x) for x in (split.get(ds) or {}).get("train") or []]
        dataset_counts[ds] = len(train)
        for cond in train:
            item = meta.get((ds, cond))
            if not item:
                missing_meta.append(f"{ds}:{cond}")
                continue
            for pathway in item["pathways"]:
                pathway_counts[str(pathway)] += 1
            for dose in item["doses"]:
                dose_counts[str(dose)] += 1
    n_chem = sum(dataset_counts.values())
    return {
        "chemical_train_conditions": n_chem,
        "chemical_dataset_counts": dataset_counts,
        "unique_pathways": len(pathway_counts),
        "pathway_counts": dict(sorted(pathway_counts.items())),
        "pathway_top_share": float(max(pathway_counts.values(), default=0) / max(1, n_chem)),
        "unique_doses": len(dose_counts),
        "dose_count_range": [min(dose_counts.values(), default=0), max(dose_counts.values(), default=0)],
        "missing_metadata_examples": missing_meta[:10],
    }


def split_safety(candidate: dict[str, Any], base: dict[str, Any], cap120: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for ds, groups in sorted(candidate.items()):
        ds_s = str(ds)
        train = {str(x) for x in groups.get("train") or []}
        base_groups = base.get(ds_s) or {}
        cap_groups = cap120.get(ds_s) or {}
        base_train = {str(x) for x in base_groups.get("train") or []}
        cap_train = {str(x) for x in cap_groups.get("train") or []}
        eval_set: set[str] = set()
        for key, val in base_groups.items():
            if key != "train" and isinstance(val, list):
                eval_set.update(str(x) for x in val)
        if not train.issubset(base_train):
            reasons.append(f"{ds_s}:train_not_subset_base_train")
        if not train.issubset(cap_train):
            reasons.append(f"{ds_s}:train_not_subset_cap120_parent")
        if train & eval_set:
            reasons.append(f"{ds_s}:train_eval_overlap")
        for key, val in base_groups.items():
            if key == "train" or not isinstance(val, list):
                continue
            if [str(x) for x in groups.get(key, [])] != [str(x) for x in val]:
                reasons.append(f"{ds_s}:{key}_validation_changed")
                break
    return reasons


def nonchemical_unchanged(candidate: dict[str, Any], cap120: dict[str, Any]) -> bool:
    for ds, groups in cap120.items():
        if str(ds) in CHEMICAL_DATASETS:
            continue
        if sorted(str(x) for x in groups.get("train") or []) != sorted(
            str(x) for x in (candidate.get(str(ds)) or {}).get("train") or []
        ):
            return False
    return True


def load_helper():
    spec = importlib.util.spec_from_file_location("xverse_scaling_split_helper", SPLIT_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {SPLIT_HELPER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mean_drift(means: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    rows = []
    for ds in sorted(set(means) & set(reference)):
        rows.append({"dataset": ds, "l2": float(np.linalg.norm(means[ds] - reference[ds]))})
    return {
        "mean_l2": float(np.mean([r["l2"] for r in rows])) if rows else None,
        "max_l2": float(max([r["l2"] for r in rows])) if rows else None,
        "top_l2": sorted(rows, key=lambda r: r["l2"], reverse=True)[:8],
    }


def npz_to_dict(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {key: z[key] for key in z.files}


def group(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return dict(((payload.get("groups") or {}).get(name) or {}))


def metric_delta(a_path: Path, c_path: Path, group_name: str, metric: str) -> float | None:
    a = load_json(a_path)
    c = load_json(c_path)
    av = fnum(group(a, group_name).get(metric))
    cv = fnum(group(c, group_name).get(metric))
    if av is None or cv is None:
        return None
    return cv - av


def prior_metric_summary() -> dict[str, Any]:
    return {
        "pathway_quota12": {
            "cross_pp_delta": metric_delta(
                QUOTA_POSTHOC / "split_group_eval_anchor_internal_ode20.json",
                QUOTA_POSTHOC / "split_group_eval_candidate_internal_ode20.json",
                "internal_val_cross_background_seen_gene_proxy",
                "pearson_pert",
            ),
            "family_pp_delta": metric_delta(
                QUOTA_POSTHOC / "condition_family_eval_anchor_internal_ode20.json",
                QUOTA_POSTHOC / "condition_family_eval_candidate_internal_ode20.json",
                "family_gene",
                "pearson_pert",
            ),
            "family_mmd_delta": metric_delta(
                QUOTA_POSTHOC / "condition_family_eval_anchor_internal_ode20.json",
                QUOTA_POSTHOC / "condition_family_eval_candidate_internal_ode20.json",
                "family_gene",
                "test_mmd",
            ),
        },
        "randomcount": {
            "cross_pp_delta": metric_delta(
                RANDOM_POSTHOC / "split_group_eval_anchor_internal_ode20.json",
                RANDOM_POSTHOC / "split_group_eval_candidate_internal_ode20.json",
                "internal_val_cross_background_seen_gene_proxy",
                "pearson_pert",
            ),
            "family_pp_delta": metric_delta(
                RANDOM_POSTHOC / "condition_family_eval_anchor_internal_ode20.json",
                RANDOM_POSTHOC / "condition_family_eval_candidate_internal_ode20.json",
                "family_gene",
                "pearson_pert",
            ),
            "family_mmd_delta": metric_delta(
                RANDOM_POSTHOC / "condition_family_eval_anchor_internal_ode20.json",
                RANDOM_POSTHOC / "condition_family_eval_candidate_internal_ode20.json",
                "family_gene",
                "test_mmd",
            ),
        },
    }


def decide(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    metrics = payload["prior_metrics"]
    q = metrics["pathway_quota12"]
    r = metrics["randomcount"]
    structural = payload["structural_checks"]
    candidate = payload["summaries"]["candidate_mmdguard_pathway_buffer"]

    if structural["safety_reasons"]:
        reasons.extend(structural["safety_reasons"])
    if not structural["nonchemical_unchanged_from_cap120"]:
        reasons.append("gene_or_nonchemical_train_changed")
    if candidate["chemical_train_conditions"] != TARGET_CHEMICAL_PER_BACKGROUND * len(CHEMICAL_DATASETS):
        reasons.append("candidate_chemical_count_not_target")
    if candidate["unique_pathways"] < 20:
        reasons.append("candidate_lost_pathway_coverage")
    if candidate["pathway_top_share"] > 0.160:
        reasons.append("candidate_top_pathway_share_gt_0p160")
    if candidate["dose_count_range"][0] != candidate["dose_count_range"][1]:
        reasons.append("candidate_dose_rows_not_balanced")
    if structural["jaccard_vs_pathway_quota12_chemical"] > 0.94:
        reasons.append("candidate_too_close_to_closed_pathway_quota12")
    if structural["jaccard_vs_randomcount_chemical"] > 0.82:
        reasons.append("candidate_too_close_to_mmd_unsafe_randomcount")
    if structural["jaccard_vs_cap120_chemical"] > 0.96:
        reasons.append("candidate_degenerates_to_cap120_all")
    if structural["jaccard_vs_type_balanced_chemical"] > 0.85:
        reasons.append("candidate_too_close_to_closed_type_balanced")
    if not math.isfinite(float(q.get("family_mmd_delta") or 999.0)) or float(q.get("family_mmd_delta") or 999.0) > 0.001:
        reasons.append("safe_core_quota12_not_mmd_safe")
    if not math.isfinite(float(r.get("family_mmd_delta") or -999.0)) or float(r.get("family_mmd_delta") or -999.0) < 0.010:
        reasons.append("randomcount_mmd_risk_not_established")
    if float(q.get("cross_pp_delta") or -999.0) >= 0.010:
        reasons.append("quota12_not_closed_for_weak_cross_signal")

    drift = payload["train_pert_mean_drift_vs_cap120"]
    cand_drift = drift["candidate_mmdguard_pathway_buffer"]["mean_l2"]
    quota_drift = drift["pathway_quota12"]["mean_l2"]
    if cand_drift is None or quota_drift is None or cand_drift > quota_drift:
        reasons.append("candidate_train_pert_mean_drift_exceeds_quota12")
    if structural["candidate_pathway_l1_vs_cap120"] > structural["quota12_pathway_l1_vs_cap120"]:
        reasons.append("candidate_pathway_distribution_further_from_cap120_than_quota12")
    return reasons


def render_md(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    s = payload["structural_checks"]
    pm = payload["prior_metrics"]
    drift = payload["train_pert_mean_drift_vs_cap120"]

    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:+.6f}"
        return str(value)

    lines = [
        "# LatentFM Modality/Pathway MMD-Preservation Gate",
        "",
        f"Status: `{decision['status']}`",
        "",
        "## Hypothesis",
        "",
        payload["hypothesis"],
        "",
        "## Boundary",
        "",
        "- CPU-only gate.",
        "- Reads only train-only/internal reports, split JSONs, local condition metadata, and train H5 GT embeddings for pert-mean drift.",
        "- Does not read canonical metrics, canonical multi, Track C query, active logs, or launch GPU.",
        "",
        "## Inputs And Provenance",
        "",
    ]
    for key, value in payload["inputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Prior Evidence",
            "",
            "| branch | cross pp delta | family pp delta | family MMD delta | interpretation |",
            "|---|---:|---:|---:|---|",
            "| `pathway_quota12` | {cross} | {fam} | {mmd} | MMD-safe but weak cross signal |".format(
                cross=fmt(pm["pathway_quota12"]["cross_pp_delta"]),
                fam=fmt(pm["pathway_quota12"]["family_pp_delta"]),
                mmd=fmt(pm["pathway_quota12"]["family_mmd_delta"]),
            ),
            "| `randomcount` | {cross} | {fam} | {mmd} | stronger Pearson but MMD-unsafe |".format(
                cross=fmt(pm["randomcount"]["cross_pp_delta"]),
                fam=fmt(pm["randomcount"]["family_pp_delta"]),
                mmd=fmt(pm["randomcount"]["family_mmd_delta"]),
            ),
            "",
            "## Candidate Design",
            "",
            f"- proposed split: `{payload['candidate_design']['proposed_split_json']}`",
            f"- design: {payload['candidate_design']['design']}",
            f"- target chemical count: `{payload['candidate_design']['target_chemical_conditions']}`",
            f"- pathway soft cap per background: `{payload['candidate_design']['pathway_soft_cap_per_background']}`",
            "",
            "## Non-Duplication",
            "",
            f"- chemical Jaccard vs pathway_quota12: `{s['jaccard_vs_pathway_quota12_chemical']:.3f}`",
            f"- chemical Jaccard vs randomcount: `{s['jaccard_vs_randomcount_chemical']:.3f}`",
            f"- chemical Jaccard vs type-balanced cap120: `{s['jaccard_vs_type_balanced_chemical']:.3f}`",
            f"- chemical Jaccard vs cap120 all: `{s['jaccard_vs_cap120_chemical']:.3f}`",
            f"- matched breadth duplication: `{payload['non_duplication']['matched_breadth']}`",
            f"- OT duplication: `{payload['non_duplication']['ot']}`",
            "",
            "## MMD-Risk Constraints",
            "",
            f"- safe quota12 core retained: `{s['quota12_core_retention']:.3f}`",
            f"- candidate pathway L1 vs cap120: `{s['candidate_pathway_l1_vs_cap120']:.3f}`",
            f"- quota12 pathway L1 vs cap120: `{s['quota12_pathway_l1_vs_cap120']:.3f}`",
            f"- candidate train pert-mean drift mean/max vs cap120: `{fmt(drift['candidate_mmdguard_pathway_buffer']['mean_l2'])}` / `{fmt(drift['candidate_mmdguard_pathway_buffer']['max_l2'])}`",
            f"- quota12 train pert-mean drift mean/max vs cap120: `{fmt(drift['pathway_quota12']['mean_l2'])}` / `{fmt(drift['pathway_quota12']['max_l2'])}`",
            f"- randomcount train pert-mean drift mean/max vs cap120: `{fmt(drift['randomcount']['mean_l2'])}` / `{fmt(drift['randomcount']['max_l2'])}`",
            "",
            "## Criteria",
            "",
            "- no split safety violation and nonchemical coverage unchanged",
            "- keep all quota12 safe-core chemical drugs",
            "- 336 chemical drugs total, 112 per sci-Plex background",
            "- 20 pathways retained, top pathway share <= 0.160, dose rows balanced",
            "- not too close to closed quota12, randomcount, type-balanced, or cap120-all splits",
            "- quota12 must be confirmed MMD-safe but cross-weak; randomcount must be confirmed MMD-unsafe",
            "- candidate train pert-mean drift and pathway L1 must not exceed quota12",
            "",
            "## Decision",
            "",
            f"- GPU authorized by this gate: `{decision['gpu_authorized']}`",
            f"- pass/fail: `{decision['pass_fail']}`",
            f"- reasons: `{decision['reasons']}`",
            f"- next action: {decision['next_action']}",
        ]
    )
    if decision["gpu_authorized"]:
        lines.extend(
            [
                "",
                "## Proposed Smoke Requirements",
                "",
                f"- split artifact to accept/materialize: `{payload['candidate_design']['proposed_split_json']}`",
                "- materializer requirement: copy the split under `dataset/biFlow_data/xverse_modality_pathway_sampling_splits_20260624/`, compute matching train-only pert means from train H5 GT embeddings, and write an artifact audit.",
                "- launcher requirement: one fresh detached 2k-3k train-only internal smoke with RUN_STATUS, no canonical multi, no Track C query, no active log polling.",
                "- resource plan: one physical GPU after fresh 3-sample audit, 3-4 CPU threads, no colocation unless coordinator resource policy allows it.",
                "- promotion gate: internal cross pp delta >= +0.010, family pp no regression, family MMD delta <= +0.001, dataset-min pp >= -0.020, and no randomcount-like MMD tail harm.",
                "- stop rule: any missing artifact, split-boundary violation, MMD hard harm, weak cross signal, or randomcount-like tail harm closes this exact branch.",
            ]
        )
    lines.extend(["", "## JSON", "", f"`{OUT_JSON}`", ""])
    return "\n".join(lines)


def main() -> int:
    required = [
        INVENTORY,
        BASE_SPLIT,
        CAP120_SPLIT,
        TYPE_BALANCED_SPLIT,
        QUOTA12_SPLIT,
        RANDOMCOUNT_SPLIT,
        QUOTA_DECISION_MD,
        RANDOMCOUNT_DECISION_MD,
        RANDOMCOUNT_MMD_GATE_MD,
        MATCHED_BREADTH_MD,
        OT_OVERLAP_MD,
        CAP120_MEANS,
        QUOTA12_MEANS,
        RANDOMCOUNT_MEANS,
        TYPE_BALANCED_MEANS,
        SPLIT_HELPER,
        OLD_GATE_SCRIPT,
        OLD_MATERIALIZER,
        RANDOM_MATERIALIZER,
        RANDOM_MMD_SCRIPT,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)

    inventory = load_json(INVENTORY)
    base = load_json(BASE_SPLIT)
    cap120 = load_json(CAP120_SPLIT)
    type_balanced = load_json(TYPE_BALANCED_SPLIT)
    quota12 = load_json(QUOTA12_SPLIT)
    randomcount = load_json(RANDOMCOUNT_SPLIT)
    meta = build_chemical_meta(inventory["rows"])
    candidate = build_candidate(cap120=cap120, quota12=quota12, meta=meta)
    OUT_SPLIT.write_text(json.dumps(candidate, indent=2, sort_keys=True), encoding="utf-8")

    helper = load_helper()
    candidate_means, candidate_audit = helper.compute_train_pert_means(DATA_DIR, candidate)
    cap_means = npz_to_dict(CAP120_MEANS)
    quota_means = npz_to_dict(QUOTA12_MEANS)
    random_means = npz_to_dict(RANDOMCOUNT_MEANS)
    type_means = npz_to_dict(TYPE_BALANCED_MEANS)

    candidate_summary = summarize_split(candidate, meta)
    cap_summary = summarize_split(cap120, meta)
    quota_summary = summarize_split(quota12, meta)
    random_summary = summarize_split(randomcount, meta)
    type_summary = summarize_split(type_balanced, meta)

    cap_pathways = Counter(cap_summary["pathway_counts"])
    quota_pathways = Counter(quota_summary["pathway_counts"])
    candidate_pathways = Counter(candidate_summary["pathway_counts"])
    random_pathways = Counter(random_summary["pathway_counts"])

    candidate_chem = train_set(candidate, chemical_only=True)
    quota_chem = train_set(quota12, chemical_only=True)
    random_chem = train_set(randomcount, chemical_only=True)
    cap_chem = train_set(cap120, chemical_only=True)
    type_chem = train_set(type_balanced, chemical_only=True)

    structural = {
        "safety_reasons": split_safety(candidate, base, cap120),
        "nonchemical_unchanged_from_cap120": nonchemical_unchanged(candidate, cap120),
        "jaccard_vs_pathway_quota12_chemical": jaccard(candidate_chem, quota_chem),
        "jaccard_vs_randomcount_chemical": jaccard(candidate_chem, random_chem),
        "jaccard_vs_cap120_chemical": jaccard(candidate_chem, cap_chem),
        "jaccard_vs_type_balanced_chemical": jaccard(candidate_chem, type_chem),
        "quota12_core_retention": float(len(candidate_chem & quota_chem) / max(1, len(quota_chem))),
        "randomcount_intersection_fraction": float(len(candidate_chem & random_chem) / max(1, len(candidate_chem))),
        "candidate_pathway_l1_vs_cap120": l1_distribution(candidate_pathways, cap_pathways),
        "quota12_pathway_l1_vs_cap120": l1_distribution(quota_pathways, cap_pathways),
        "randomcount_pathway_l1_vs_cap120": l1_distribution(random_pathways, cap_pathways),
    }
    prior = prior_metric_summary()

    payload: dict[str, Any] = {
        "hypothesis": (
            "A quota12-safe-core plus pathway-balanced add-back split can test whether the randomcount Pearson signal "
            "was a useful exposure signal while avoiding the randomcount MMD tail harm. It is eligible only if the "
            "candidate remains train-only, preserves the MMD-safe quota core, stays structurally distinct from closed "
            "quota/randomcount/type-balanced/matched-breadth/OT branches, and passes predeclared MMD-risk constraints."
        ),
        "inputs": {
            "inventory": str(INVENTORY),
            "base_split": str(BASE_SPLIT),
            "cap120_split": str(CAP120_SPLIT),
            "quota12_split": str(QUOTA12_SPLIT),
            "randomcount_split": str(RANDOMCOUNT_SPLIT),
            "type_balanced_split": str(TYPE_BALANCED_SPLIT),
            "quota_decision": str(QUOTA_DECISION_MD),
            "randomcount_decision": str(RANDOMCOUNT_DECISION_MD),
            "randomcount_mmd_gate": str(RANDOMCOUNT_MMD_GATE_MD),
            "matched_breadth_gate": str(MATCHED_BREADTH_MD),
            "ot_overlap_gate": str(OT_OVERLAP_MD),
            "split_helper": str(SPLIT_HELPER),
            "old_materializer": str(OLD_MATERIALIZER),
            "random_materializer": str(RANDOM_MATERIALIZER),
        },
        "boundary": {
            "reads_train_only_internal_artifacts": True,
            "reads_train_h5_gt_embeddings_for_candidate_pert_mean_drift": True,
            "reads_canonical_metrics": False,
            "reads_canonical_multi": False,
            "reads_trackc_query": False,
            "reads_active_logs": False,
            "launches_gpu": False,
        },
        "hashes": {
            "quota12_split_sha256": sha256_file(QUOTA12_SPLIT),
            "randomcount_split_sha256": sha256_file(RANDOMCOUNT_SPLIT),
            "candidate_split_sha256": sha256_file(OUT_SPLIT),
            "old_gate_script_sha256": sha256_file(OLD_GATE_SCRIPT),
            "old_materializer_sha256": sha256_file(OLD_MATERIALIZER),
            "random_mmd_script_sha256": sha256_file(RANDOM_MMD_SCRIPT),
        },
        "candidate_design": {
            "proposed_split_json": str(OUT_SPLIT),
            "design": "retain all pathway_quota12 chemical drugs, then add a small cap120-parent buffer prioritized toward underrepresented pathways with a per-background pathway soft cap",
            "target_chemical_conditions": TARGET_CHEMICAL_PER_BACKGROUND * len(CHEMICAL_DATASETS),
            "target_per_background": TARGET_CHEMICAL_PER_BACKGROUND,
            "pathway_soft_cap_per_background": PATHWAY_SOFT_CAP_PER_BACKGROUND,
            "seed": SEED,
        },
        "summaries": {
            "cap120_all": cap_summary,
            "pathway_quota12": quota_summary,
            "randomcount": random_summary,
            "type_balanced_cap120": type_summary,
            "candidate_mmdguard_pathway_buffer": candidate_summary,
        },
        "structural_checks": structural,
        "prior_metrics": prior,
        "train_pert_mean_drift_vs_cap120": {
            "candidate_mmdguard_pathway_buffer": mean_drift(candidate_means, cap_means),
            "pathway_quota12": mean_drift(quota_means, cap_means),
            "randomcount": mean_drift(random_means, cap_means),
            "type_balanced_cap120": mean_drift(type_means, cap_means),
        },
        "candidate_pert_mean_audit_bad_rows": [
            row for row in candidate_audit if row.get("status") not in {"ok", "empty_train_dataset"}
        ],
        "non_duplication": {
            "pathway_quota12": "keeps the safe core but changes 36 chemical drugs and raises chemical count from 300 to 336",
            "randomcount": "does not random-sample; randomcount Jaccard is bounded and the randomcount MMD gate remains a negative control",
            "matched_breadth": "does not alter gene/nonchemical dataset breadth; only sciplex chemical composition changes within cap120 parent",
            "ot": "does not change minibatch pairing, OT mode, or OT cost; OT condition-overlap gate remains closed",
        },
    }
    reasons = decide(payload)
    status = (
        "modality_pathway_mmd_preservation_gate_pass_one_bounded_smoke_authorized"
        if not reasons
        else "modality_pathway_mmd_preservation_gate_fail_close_gpu"
    )
    payload["decision"] = {
        "status": status,
        "pass_fail": "pass" if not reasons else "fail",
        "gpu_authorized": not reasons,
        "authorized_gpu_smokes": 1 if not reasons else 0,
        "reasons": reasons,
        "next_action": (
            "Coordinator may materialize this exact split and launch exactly one bounded train-only internal GPU smoke after fresh resource audit."
            if not reasons
            else "Close this GPU branch; do not launch a modality/pathway MMD-preservation smoke from this candidate."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "gpu_authorized": not reasons,
                "reasons": reasons,
                "out_md": str(OUT_MD),
                "out_json": str(OUT_JSON),
                "candidate_split": str(OUT_SPLIT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
