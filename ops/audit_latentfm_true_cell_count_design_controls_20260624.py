#!/usr/bin/env python3
"""Design-control gate for true cell-count scaling artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
MATERIALIZER_JSON = ROOT / "reports/latentfm_true_cell_count_capped_h5_materializer_gate_20260624.json"
SCHEMA_JSON = ROOT / "reports/latentfm_true_cell_count_capped_h5_schema_gate_20260624.json"
OUT_JSON = ROOT / "reports/latentfm_true_cell_count_design_controls_gate_20260624.json"
OUT_MD = ROOT / "reports/LATENTFM_TRUE_CELL_COUNT_DESIGN_CONTROLS_GATE_20260624.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_signature(split_file: Path) -> dict[str, Any]:
    split = load_json(split_file)
    train = {}
    eval_groups = {}
    for ds, groups in sorted(split.items()):
        train[ds] = sorted(str(c) for c in (groups.get("train") or []))
        eval_union = set()
        for key, values in groups.items():
            if key == "train" or key == "canonical_test_reference" or not isinstance(values, list):
                continue
            eval_union.update(str(c) for c in values)
        eval_groups[ds] = sorted(eval_union)
    return {"train": train, "eval": eval_groups}


def row_budget_seed(row: dict[str, Any]) -> tuple[int, int]:
    manifest_path = Path(row["data_dir"]) / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        return int(manifest["budget"]), int(manifest["seed"])
    run_id = str(row.get("run_id", ""))
    budget = int(run_id.split("_budget")[-1].split("_seed")[0])
    seed = int(run_id.split("_seed")[-1].split("_")[0])
    return budget, seed


def load_sample_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[(str(row["dataset"]), str(row["condition"]))] = row
    return out


def load_sample_arrays(data_dir: Path, manifest: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], dict[str, set[int]]]:
    out: dict[tuple[str, str], dict[str, set[int]]] = {}
    with np_load(data_dir / "sampled_indices.npz") as npz:
        for key, row in manifest.items():
            out[key] = {
                "gt": set(int(x) for x in npz[str(row["gt_key"])]),
                "ctrl": set(int(x) for x in npz[str(row["ctrl_key"])]),
            }
    return out


class np_load:
    def __init__(self, path: Path):
        import numpy as np

        self._np = np
        self.path = path
        self.obj = None

    def __enter__(self):
        self.obj = self._np.load(self.path)
        return self.obj

    def __exit__(self, exc_type, exc, tb):
        if self.obj is not None:
            self.obj.close()
        return False


def audit_nested_sampling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    by_seed: dict[int, dict[int, dict[tuple[str, str], dict[str, Any]]]] = defaultdict(dict)
    arrays_by_seed: dict[int, dict[int, dict[tuple[str, str], dict[str, set[int]]]]] = defaultdict(dict)
    for row in rows:
        budget, seed = row_budget_seed(row)
        manifest_path = Path(row["data_dir"]) / "sampled_index_manifest.jsonl"
        npz_path = Path(row["data_dir"]) / "sampled_indices.npz"
        if not manifest_path.exists() or not npz_path.exists():
            reasons.append(f"missing_sample_manifest:{row['run_id']}")
            continue
        manifest = load_sample_manifest(manifest_path)
        by_seed[seed][budget] = manifest
        arrays_by_seed[seed][budget] = load_sample_arrays(Path(row["data_dir"]), manifest)
    subset_checks = []
    for seed, budget_map in sorted(by_seed.items()):
        budgets = sorted(budget_map)
        for low, high in zip(budgets, budgets[1:]):
            low_map = budget_map[low]
            high_map = budget_map[high]
            low_arrays = arrays_by_seed[seed][low]
            high_arrays = arrays_by_seed[seed][high]
            not_nested = 0
            for key in low_map:
                if key not in high_map:
                    reasons.append(f"nested_missing_condition:seed{seed}:budget{low}_in_{high}:{key}")
                    continue
                if not low_arrays[key]["gt"].issubset(high_arrays[key]["gt"]) or not low_arrays[key]["ctrl"].issubset(high_arrays[key]["ctrl"]):
                    not_nested += 1
            status = "ok" if not_nested == 0 else "not_nested"
            subset_checks.append({"seed": seed, "low_budget": low, "high_budget": high, "status": status, "not_nested_conditions": not_nested})
    return {
        "status": "ok" if subset_checks and not reasons and all(r["status"] == "ok" for r in subset_checks) else ("not_ready" if not subset_checks else "fail"),
        "reasons": reasons[:20],
        "subset_checks": subset_checks[:20],
    }


def main() -> int:
    materializer = load_json(MATERIALIZER_JSON)
    rows = materializer.get("materialized_rows") or []
    schema = load_json(SCHEMA_JSON) if SCHEMA_JSON.exists() else {}
    reasons = []
    if not materializer.get("materialized"):
        reasons.append("materializer_not_materialized")
    if schema.get("status") != "capped_h5_schema_gate_pass_no_gpu":
        reasons.append(f"schema_gate_not_pass:{schema.get('status')}")
    if not rows:
        reasons.append("no_materialized_rows")
    all_modality = [r["run_id"] for r in rows if str(r.get("run_id", "")).startswith("all_modality")]
    if all_modality:
        reasons.append(f"all_modality_rows_present:{len(all_modality)}")

    signatures = {}
    for row in rows:
        signatures[row["run_id"]] = condition_signature(Path(row["split_file"]))
    train_sigs = {json.dumps(sig["train"], sort_keys=True) for sig in signatures.values()}
    eval_sigs = {json.dumps(sig["eval"], sort_keys=True) for sig in signatures.values()}
    if len(train_sigs) > 1:
        reasons.append(f"train_condition_identity_varies:{len(train_sigs)}")
    if len(eval_sigs) > 1:
        reasons.append(f"eval_condition_identity_varies:{len(eval_sigs)}")

    parsed_budget_seed = [row_budget_seed(row) for row in rows]
    budgets = sorted({budget for budget, _seed in parsed_budget_seed})
    seeds = sorted({seed for _budget, seed in parsed_budget_seed})
    expected_budgets = [64, 128, 256]
    expected_seeds = [42, 43, 44]
    if budgets != expected_budgets:
        reasons.append(f"budget_set_unexpected:{budgets}")
    if seeds != expected_seeds:
        reasons.append(f"seed_set_unexpected:{seeds}")
    expected_n = len(expected_budgets) * len(expected_seeds)
    if len(rows) != expected_n:
        reasons.append(f"row_count_unexpected:{len(rows)}_expected_{expected_n}")

    nested = audit_nested_sampling(rows)
    warnings = []
    if nested["status"] != "ok":
        warnings.append("sample_nestedness_not_proven_current_artifacts_preliminary_only")

    if reasons:
        status = "true_cell_count_design_controls_fail_no_gpu"
        next_action = "fix materialization/schema/design-control failures"
    else:
        status = "true_cell_count_design_controls_pass_preliminary_only_no_gpu"
        next_action = "eligible for one bounded exploratory smoke after resource audit; final scaling-law claim needs nested-v2 or explicit non-nested limitation"

    payload = {
        "status": status,
        "materializer_json": str(MATERIALIZER_JSON),
        "schema_json": str(SCHEMA_JSON),
        "budgets": budgets,
        "seeds": seeds,
        "n_rows": len(rows),
        "reasons": reasons,
        "warnings": warnings,
        "nested_sampling": nested,
        "gpu_authorized": False,
        "next_action": next_action,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# LatentFM True Cell-Count Design Controls Gate",
        "",
        f"Status: `{status}`",
        "",
        "## Checks",
        "",
        f"- budgets: `{budgets}`",
        f"- seeds: `{seeds}`",
        f"- materialized rows: `{len(rows)}`",
        f"- reasons: `{reasons or 'none'}`",
        f"- warnings: `{warnings or 'none'}`",
        f"- nested sampling status: `{nested['status']}`",
        "",
        "## Decision",
        "",
        f"- GPU authorized: `{payload['gpu_authorized']}`",
        f"- next action: `{next_action}`",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
