#!/usr/bin/env python3
"""Collect LatentFM artifact provenance for the active 2026-06-21 stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/data/cyx/1030/scLatent")
COUPLED = ROOT / "CoupledFM"


DEFAULT_PATHS = [
    ROOT / "goal.md",
    ROOT / "AGENTS.md",
    ROOT / "docs/PROJECT_REVIEW.md",
    ROOT / "reports/LATENTFM_INSIGHTS_AND_NEXT_DECISIONS_20260621.md",
    ROOT / "reports/LATENTFM_ACTIVE_STATUS_20260621_0251.md",
    ROOT / "reports/LATENTFM_FEWSHOT_MULTI_CALIBRATION_ONE_SHOT_STATUS_20260621_022114.txt",
    ROOT / "reports/LATENTFM_FEWSHOT_MULTI_CALIBRATION_ONE_SHOT_STATUS_20260621_025135.txt",
    ROOT / "reports/LATENTFM_TRAIN_RESIDUAL_GEOMETRY_AUDIT_20260621.md",
    ROOT / "reports/LATENTFM_CONTROL_ARCHETYPE_CPU_AUDIT_20260621.md",
    ROOT / "reports/LATENTFM_RESPONSE_GEOMETRY_IMPLEMENTATION_REVIEW_20260621.md",
    ROOT / "reports/LATENTFM_PAIRWISE_CONDITION_ENCODER_PREAUDIT_20260621.md",
    ROOT / "runs/latentfm_fewshot_multi_calibration_20260621/RUN_STATUS.md",
    ROOT / "runs/latentfm_response_normalization_20260621/RUN_STATUS.md",
    ROOT / "runs/latentfm_active_posthoc_bootstrap_20260621/RUN_STATUS.md",
    ROOT / "runs/latentfm_active_decision_20260621/RUN_STATUS.md",
    ROOT / "runs/latentfm_pairwise_auto_trigger_20260621/RUN_STATUS.md",
    ROOT / "runs/latentfm_response_normalization_20260621/artifacts/scfoundation_trainonly_dataset_scale_pca32.npz",
    ROOT / "ops/launch_latentfm_fewshot_multi_calibration_20260621.sh",
    ROOT / "ops/run_latentfm_fewshot_multi_calibration_posthoc_20260621.sh",
    ROOT / "ops/launch_latentfm_response_geometry_smoke_20260621.sh",
    ROOT / "ops/run_latentfm_response_geometry_posthoc_20260621.sh",
    ROOT / "ops/launch_latentfm_pairwise_condition_smoke_20260621.sh",
    ROOT / "ops/run_latentfm_pairwise_condition_posthoc_20260621.sh",
    ROOT / "ops/bootstrap_latentfm_paired_posthoc_20260621.py",
    ROOT / "ops/run_latentfm_posthoc_bootstrap_from_manifest_20260621.py",
    ROOT / "ops/synthesize_latentfm_active_decision_20260621.py",
    ROOT / "ops/run_latentfm_uncapped_posthoc_from_manifest_20260621.py",
    COUPLED / "model/latent/response_normalizer.py",
    COUPLED / "model/latent/fit_response_normalizer.py",
    COUPLED / "model/latent/config.py",
    COUPLED / "model/latent/train.py",
    COUPLED / "model/condition_emb/genepert/perturbation_encoder.py",
    COUPLED / "model/tests/test_latent_response_normalizer.py",
    COUPLED / "model/tests/test_multi_pool_aggregation.py",
    COUPLED / "model/tests/test_latent_condition_embedding_sources.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=str(cwd), text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def file_record(path: Path, *, hash_max_mb: float) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return rec
    st = path.stat()
    rec["size_bytes"] = st.st_size
    rec["mtime"] = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    if path.is_file() and st.st_size <= int(hash_max_mb * 1024 * 1024):
        rec["sha256"] = sha256(path)
    elif path.is_file():
        rec["sha256"] = None
        rec["hash_skipped_reason"] = f"file larger than {hash_max_mb} MiB"
    return rec


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# LatentFM Artifact Provenance",
        "",
        f"Generated: `{payload['generated']}`",
        "",
        "## Git",
        "",
        f"- CoupledFM HEAD: `{payload['git']['coupled_head']}`",
        f"- CoupledFM status: `{payload['git']['coupled_status_short'] or 'clean'}`",
        "",
        "## Files",
        "",
        "| exists | size | sha256 | path |",
        "|---|---:|---|---|",
    ]
    for rec in payload["files"]:
        sha = rec.get("sha256")
        if sha:
            sha = sha[:16] + "..."
        elif rec.get("hash_skipped_reason"):
            sha = rec["hash_skipped_reason"]
        else:
            sha = "NA"
        lines.append(
            f"| {rec.get('exists')} | {rec.get('size_bytes', 'NA')} | `{sha}` | `{rec['path']}` |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- This is provenance metadata only; it does not imply promotion.",
        "- Large binary artifacts may have hashes skipped according to `hash_max_mb`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=ROOT / "reports/latentfm_artifact_provenance_20260621.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "reports/LATENTFM_ARTIFACT_PROVENANCE_20260621.md")
    parser.add_argument("--hash-max-mb", type=float, default=64.0)
    args = parser.parse_args()

    files = [file_record(p, hash_max_mb=float(args.hash_max_mb)) for p in DEFAULT_PATHS]
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "hash_max_mb": float(args.hash_max_mb),
        "git": {
            "coupled_head": git(["rev-parse", "HEAD"], COUPLED),
            "coupled_status_short": git(["status", "--short"], COUPLED),
            "coupled_diff_stat": git(["diff", "--stat"], COUPLED),
        },
        "files": files,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"out_md": str(args.out_md), "n_files": len(files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
