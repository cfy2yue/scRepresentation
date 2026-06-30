#!/usr/bin/env python3
"""CPU source/control gate for Track C shared-gene support-set task plumbing.

This audit exercises the train/eval helper functions that will feed the
support-set task adapter. It is CPU-only and reads only the safe trainselect
condition-mean artifacts. It does not train, run inference, use canonical multi
for selection, or read held-out Track C query.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/data/cyx/1030/scLatent")
COUPLEDFM = ROOT / "CoupledFM"
if str(COUPLEDFM) not in sys.path:
    sys.path.insert(0, str(COUPLEDFM))

from model.latent.config import Config  # noqa: E402
from model.latent import train as latent_train  # noqa: E402


SAFE_SPLIT = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2_trainselect.json"
FULL_V2 = ROOT / "dataset/biFlow_data/split_seed42_multi_support_v2.json"
ART_DIR = (
    ROOT
    / "runs/latentfm_trackc_support_set_task_input_artifacts_20260623/"
    "xverse_support_film_retry1_trainmulti_condition_means/condition_means"
)
ANCHOR = ART_DIR / "trainselect_anchor_train_support_multi_condition_means_ode20.json"
CANDIDATE = ART_DIR / "trainselect_candidate_train_support_multi_condition_means_ode20.json"
OUT_JSON = ROOT / "reports/latentfm_trackc_support_set_source_plumbing_20260627.json"
OUT_MD = ROOT / "reports/LATENTFM_TRACKC_SUPPORT_SET_SOURCE_PLUMBING_20260627.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_rows(payload: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return ((payload.get("groups") or {}).get(group) or {}).get("condition_metrics") or []


def cfg(control: str = "actual", split: Path = SAFE_SPLIT) -> Config:
    c = Config()
    c.model_type = "control_mlp"
    c.emb_dim = 384
    c.trackc_support_set_task_use_in_model = True
    c.trackc_support_set_task_dim = 384
    c.trackc_support_set_task_source = "shared_gene_condition_means"
    c.trackc_support_set_task_safe_split_file = str(split)
    c.trackc_support_set_task_anchor_condition_means = str(ANCHOR)
    c.trackc_support_set_task_candidate_condition_means = str(CANDIDATE)
    c.trackc_support_set_task_scale = 1.0
    c.trackc_support_set_task_eval_control = control
    return c


def tensor_norm(x: torch.Tensor | None) -> float:
    if x is None:
        return 0.0
    return float(torch.linalg.vector_norm(x.float()).item())


def allclose_zero(x: torch.Tensor | None) -> bool:
    return bool(x is not None and torch.allclose(x, torch.zeros_like(x), atol=0.0, rtol=0.0))


def mask_value(mask: torch.Tensor | None) -> float | None:
    if mask is None:
        return None
    vals = torch.unique(mask.detach().cpu())
    if int(vals.numel()) != 1:
        return None
    return float(vals.item())


def validation_rejects_full_v2() -> tuple[bool, str]:
    try:
        latent_train.validate_support_set_task_config(cfg(split=FULL_V2))
    except Exception as exc:  # noqa: BLE001 - audit records exact fail-closed reason
        return True, str(exc)
    return False, "validation unexpectedly accepted full v2 split"


def main() -> None:
    torch.set_num_threads(1)
    base_cfg = cfg()
    latent_train.validate_support_set_task_config(base_cfg)
    full_v2_rejected, full_v2_reason = validation_rejects_full_v2()

    bank = latent_train.build_trackc_support_set_task_bank(base_cfg)
    summary = dict(latent_train.LAST_TRACKC_SUPPORT_SET_TASK_SUMMARY)
    anchor = load_json(ANCHOR)
    support_val = [
        (str(row.get("dataset")), str(row.get("condition")))
        for row in group_rows(anchor, "support_val_multi")
    ]
    train_multi = [
        (str(row.get("dataset")), str(row.get("condition")))
        for row in group_rows(anchor, "train_multi")
    ]

    supported: list[tuple[str, str]] = []
    unsupported: list[tuple[str, str]] = []
    for ds_name, cond in support_val:
        token, present = latent_train._support_set_task_token_for(bank, ds_name, cond, base_cfg)
        if present and token is not None:
            supported.append((ds_name, cond))
        else:
            unsupported.append((ds_name, cond))

    preferred = ("NormanWeissman2019_filtered", "CBL+PTPN9")
    query = preferred if preferred in supported else supported[0]
    ds_name, cond = query
    batch_size = 5
    device = torch.device("cpu")

    actual_task, actual_present = latent_train.make_trackc_support_set_task_batch(
        bank, ds_name, cond, batch_size, base_cfg, device
    )
    actual_norm = tensor_norm(actual_task)
    actual_mask = mask_value(actual_present)

    zero_task, zero_present = latent_train._apply_trackc_support_set_task_eval_control(
        actual_task,
        actual_present,
        cfg=cfg("zero"),
        ds_name=ds_name,
        cond=cond,
        batch_size=batch_size,
        device=device,
    )
    absent_task, absent_present = latent_train._apply_trackc_support_set_task_eval_control(
        actual_task,
        actual_present,
        cfg=cfg("absent"),
        ds_name=ds_name,
        cond=cond,
        batch_size=batch_size,
        device=device,
    )
    shuffle_targets = latent_train._build_trackc_support_set_task_shuffle_targets(
        bank, support_val, base_cfg
    )
    shuffle_task, shuffle_present = latent_train._apply_trackc_support_set_task_eval_control(
        actual_task,
        actual_present,
        cfg=cfg("shuffle_condition"),
        ds_name=ds_name,
        cond=cond,
        batch_size=batch_size,
        device=device,
        shuffle_targets=shuffle_targets,
    )
    shuffle_diff = float(torch.linalg.vector_norm((shuffle_task - actual_task).float()).item())

    if unsupported:
        uds, ucond = unsupported[0]
    else:
        uds, ucond = ds_name, "UNSUPPORTED_A+UNSUPPORTED_B"
    unsupported_task, unsupported_present = latent_train.make_trackc_support_set_task_batch(
        bank, uds, ucond, batch_size, base_cfg, device
    )

    self_exclusion_ok = False
    self_exclusion_example = None
    for tds, tcond in train_multi:
        qgenes = set(latent_train._pair_genes_from_condition(tcond) or ())
        if not qgenes:
            continue
        rows = bank.get(tds, [])
        included = [
            row
            for row in rows
            if str(row.get("condition")) != tcond
            and bool(qgenes & {str(g).upper() for g in row.get("genes", ())})
        ]
        same_cond_rows = [row for row in rows if str(row.get("condition")) == tcond]
        token, present = latent_train._support_set_task_token_for(bank, tds, tcond, base_cfg)
        if present and token is not None and included and same_cond_rows:
            manual = torch.stack([row["residual"].float() for row in included], dim=0).mean(dim=0)
            self_exclusion_ok = bool(torch.allclose(token.float(), manual.float(), atol=1e-7, rtol=1e-5))
            self_exclusion_example = {"dataset": tds, "condition": tcond, "support_rows": len(included)}
            break

    checks = {
        "safe_split_validation_passed": True,
        "full_v2_rejected": full_v2_rejected,
        "bank_records_ge_40": int(summary.get("records", 0) or 0) >= 40,
        "bank_has_two_datasets": len(summary.get("datasets", {}) or {}) >= 2,
        "support_val_has_supported_rows": len(supported) > 0,
        "support_val_has_unsupported_rows": len(unsupported) > 0,
        "actual_nonzero_present": actual_norm > 1e-8 and actual_mask == 1.0,
        "zero_control_zero_present_one": allclose_zero(zero_task) and mask_value(zero_present) == 1.0,
        "absent_control_zero_present_zero": allclose_zero(absent_task) and mask_value(absent_present) == 0.0,
        "shuffle_condition_different_present": shuffle_diff > 1e-8 and mask_value(shuffle_present) == 1.0,
        "unsupported_exact_absent_noop": allclose_zero(unsupported_task) and mask_value(unsupported_present) == 0.0,
        "train_multi_self_exclusion_ok": self_exclusion_ok,
    }
    status = (
        "support_set_source_plumbing_pass_launcher_gate_next_no_gpu"
        if all(checks.values())
        else "support_set_source_plumbing_fail_no_gpu"
    )
    failed = [key for key, passed in checks.items() if not passed]
    reasons = failed or ["source_and_eval_controls_pass_but_gpu_still_forbidden"]

    payload = {
        "status": status,
        "gpu_authorized": False,
        "boundary": {
            "cpu_only": True,
            "training": False,
            "inference": False,
            "canonical_multi_selection_used": False,
            "trackc_query_used": False,
            "safe_trainselect_split": str(SAFE_SPLIT),
        },
        "summary": summary,
        "query": {"dataset": ds_name, "condition": cond},
        "unsupported_query": {"dataset": uds, "condition": ucond},
        "counts": {
            "support_val_rows": len(support_val),
            "supported_support_val_rows": len(supported),
            "unsupported_support_val_rows": len(unsupported),
            "shuffle_targets": len(shuffle_targets),
        },
        "metrics": {
            "actual_task_norm": actual_norm,
            "actual_present_mask": actual_mask,
            "zero_task_norm": tensor_norm(zero_task),
            "zero_present_mask": mask_value(zero_present),
            "absent_task_norm": tensor_norm(absent_task),
            "absent_present_mask": mask_value(absent_present),
            "shuffle_task_norm": tensor_norm(shuffle_task),
            "shuffle_present_mask": mask_value(shuffle_present),
            "shuffle_vs_actual_diff": shuffle_diff,
            "unsupported_task_norm": tensor_norm(unsupported_task),
            "unsupported_present_mask": mask_value(unsupported_present),
        },
        "self_exclusion_example": self_exclusion_example,
        "full_v2_rejection_reason": full_v2_reason,
        "checks": checks,
        "decision_reasons": reasons,
        "next_action": (
            "Prepare a leakage-safe launcher/control gate for an adapter-only frozen-backbone "
            "Track C support-val smoke; do not launch GPU from this audit alone."
            if status.endswith("next_no_gpu")
            else "Fix failed source/control checks before preparing any launcher."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Track C Support-Set Source Plumbing Gate",
        "",
        f"Status: `{status}`",
        f"GPU authorized: `{payload['gpu_authorized']}`",
        "",
        "## Boundary",
        "",
        "CPU-only helper audit over safe-trainselect condition-mean artifacts. No training, inference, canonical multi selection, held-out Track C query, or GPU.",
        "",
        "## Source Summary",
        "",
        f"* records: `{summary.get('records')}`",
        f"* datasets: `{summary.get('datasets')}`",
        f"* skipped non-pair rows: `{summary.get('skipped_non_pair')}`",
        f"* safe split: `{summary.get('safe_split_file')}`",
        "",
        "## Query And Controls",
        "",
        f"* selected supported query: `{ds_name} / {cond}`",
        f"* supported support-val rows: `{len(supported)}/{len(support_val)}`",
        f"* unsupported example: `{uds} / {ucond}`",
        f"* actual norm / mask: `{actual_norm:.6e}` / `{actual_mask}`",
        f"* zero norm / mask: `{tensor_norm(zero_task):.6e}` / `{mask_value(zero_present)}`",
        f"* absent norm / mask: `{tensor_norm(absent_task):.6e}` / `{mask_value(absent_present)}`",
        f"* shuffle diff / mask: `{shuffle_diff:.6e}` / `{mask_value(shuffle_present)}`",
        f"* unsupported norm / mask: `{tensor_norm(unsupported_task):.6e}` / `{mask_value(unsupported_present)}`",
        "",
        "## Checks",
        "",
    ]
    for key, passed in checks.items():
        lines.append(f"* `{key}`: `{passed}`")
    lines.extend(
        [
            "",
            "## Decision Reasons",
            "",
        ]
    )
    for reason in reasons:
        lines.append(f"- `{reason}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            payload["next_action"],
            "",
            "## Outputs",
            "",
            f"* JSON: `{OUT_JSON}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "failed": failed, "report": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
