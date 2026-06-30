#!/usr/bin/env python3
"""CPU-only protocol audit for chemical unseen-scaffold V2 fixed-step launcher."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
CPU_UNLOCK_JSON = ROOT / "reports/latentfm_chemical_unseen_scaffold_v2_cpu_unlock_20260625.json"
LORENTZ_AUDIT = ROOT / "reports/LATENTFM_CHEMICAL_UNSEEN_SCAFFOLD_V2_EXTERNAL_AUDIT_LORENTZ_20260625.md"
HEGEL_AUDIT = ROOT / "reports/LATENTFM_SCALING_NM_COMPLETION_EXTERNAL_AUDIT_HEGEL_20260625.md"
LAUNCHER = ROOT / "ops/launch_latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625.sh"
SUMMARIZER = ROOT / "ops/summarize_latentfm_chemical_unseen_scaffold_v2_fixedstep_controls_20260625.py"
OUT_JSON = ROOT / "reports/latentfm_chemical_v2_fixedstep_launcher_protocol_audit_20260625.json"
OUT_MD = ROOT / "reports/LATENTFM_CHEMICAL_V2_FIXEDSTEP_LAUNCHER_PROTOCOL_AUDIT_20260625.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    unlock = json.loads(CPU_UNLOCK_JSON.read_text(encoding="utf-8"))
    launcher = read(LAUNCHER)
    summarizer = read(SUMMARIZER)
    lorentz = read(LORENTZ_AUDIT)
    hegel = read(HEGEL_AUDIT)

    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    add(
        "v2_cpu_unlock_ready",
        unlock.get("status") == "chemical_unseen_scaffold_v2_cpu_unlock_ready_protocol_next_no_gpu",
        f"status={unlock.get('status')!r}, gpu_authorized={unlock.get('gpu_authorized')!r}",
    )
    rows = unlock.get("rows") or []
    add(
        "seed43_44_overlap_zero",
        len(rows) == 2
        and all(int(r.get("drug_overlap", -1)) == 0 and int(r.get("scaffold_overlap", -1)) == 0 for r in rows),
        "drug/scaffold overlap from V2 CPU unlock rows",
    )
    add(
        "control_caches_exist",
        all(Path(c["cache_dir"], "drug_embeddings.npy").is_file() for c in unlock.get("control_caches", []))
        and (ROOT / "dataset/drug_cache/sciplex_smiles_morgan512_projected_20260625/drug_embeddings.npy").is_file(),
        "real, shuffled, and random Morgan512 cache embeddings present",
    )
    add(
        "external_audits_present",
        "completed_protocol_unlock_no_immediate_gpu" in lorentz
        and "completed_no_immediate_gpu" in hegel
        and "Immediate GPU candidate: `none`" in hegel,
        "Lorentz/Hegel audits are present and explicitly no-immediate-GPU",
    )
    add(
        "ack_guard_present",
        "LATENTFM_CHEM_V2_FIXEDSTEP_ACK" in launcher
        and "launch_v2_fixedstep_controls_after_protocol_review" in launcher,
        "launcher has explicit protocol ACK guard",
    )
    add(
        "fixed_latest_candidate",
        re.search(r"candidate_ckpt=.*\$\{out_dir\}/latest\.pt", launcher) is not None
        and '--checkpoint "\\${candidate_ckpt}"' in launcher
        and "FIXED_CANDIDATE_CHECKPOINT" in launcher,
        "candidate posthoc uses latest.pt and records FIXED_CANDIDATE_CHECKPOINT",
    )
    add(
        "train_eval_disabled",
        "export TRAIN_EVAL_ENABLED=0" in launcher,
        "train-time IID eval / best-checkpoint selection is disabled for V2 arms",
    )
    add(
        "candidate_best_not_used",
        "${out_dir}/best.pt" not in launcher and "best.pt --groups" not in launcher,
        "launcher does not evaluate candidate best.pt",
    )
    add(
        "anchor_best_only_as_baseline",
        "ANCHOR_CKPT=" in launcher and "xverse_8k" in launcher,
        "anchor checkpoint is fixed xverse_8k baseline",
    )
    add(
        "no_trackc_query_or_canonical_multi",
        "Track C query" in launcher
        and "canonical multi" in launcher
        and not re.search(r"test_multi|multi_support_v2.*query|held.?out.*query", launcher, flags=re.I),
        "launcher notes exclusions and contains no canonical multi/query artifact use",
    )
    add(
        "summarizer_controls_gate",
        all(s in summarizer for s in ["real_morgan512", "shuffled_morgan512", "random_morgan512", "real_margin_over_best_control_ge_0p005"]),
        "summarizer requires real vs shuffled/random control margin",
    )
    add(
        "summarizer_fixed_latest_guard",
        "fixed_latest_checkpoint_not_recorded" in summarizer and "latest.pt" in summarizer,
        "summarizer checks fixed latest checkpoint record",
    )

    protocol_safe = all(c["pass"] for c in checks)
    # This audit deliberately does not convert protocol safety into launch
    # authorization because the independent audits explicitly say no immediate
    # GPU without protocol ACK.
    status = (
        "chemical_v2_fixedstep_launcher_protocol_safe_ack_still_required"
        if protocol_safe
        else "chemical_v2_fixedstep_launcher_protocol_fail_no_gpu"
    )
    payload = {
        "status": status,
        "gpu_authorized": False,
        "protocol_safe": protocol_safe,
        "checks": checks,
        "decision": {
            "launch_authorization": "not granted by this CPU audit",
            "reason": "Lorentz/Hegel audits explicitly no-immediate-GPU; this audit only verifies launcher safety.",
            "minimal_next": "independent launch/no-launch audit or explicit protocol ACK before starting V2 arms",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# LatentFM Chemical V2 Fixed-Step Launcher Protocol Audit",
        "",
        f"Status: `{status}`",
        "GPU authorized: `False`",
        "",
        "## Boundary",
        "",
        "- CPU-only static/artifact audit of the prepared V2 fixed-step launcher and summarizer.",
        "- Does not train, infer, read canonical multi, read Track C query, or use GPU.",
        "- This report can show the launcher is technically safe; it does not override external no-immediate-GPU audits.",
        "",
        "## Checks",
        "",
        "| check | pass | detail |",
        "|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| `{c['name']}` | `{c['pass']}` | {c['detail']} |")
    lines += [
        "",
        "## Decision",
        "",
        "- Protocol safety is necessary but not sufficient for launch.",
        "- The prepared V2 matrix still needs an independent launch/no-launch audit or explicit protocol ACK before GPU training.",
        "",
        "## JSON",
        "",
        f"`{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
