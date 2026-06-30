# Workspace Organization

Updated: 2026-06-30

## Purpose

`/data/cyx/1030` is a multi-project workspace. The goal of this organization
pass is to make the root directory readable while keeping active projects and
shared data usable.

## Intended Top-Level Layout

```text
/data/cyx/1030/
  scLatent/       # current LatentFM/scFMBench/scaling/zebrafish project
  CellClip/       # independent CellClip project
  stock/          # independent stock/quant project
  dataset/        # shared data root for scLatent and CellClip
  software/       # shared installed runtimes/tools
  codex/          # Codex state
  vscode-server/  # VS Code remote runtime
  AGENTS.md
  agent.md
  README.md
  init-codex.sh
```

Hidden/small helper files such as `.git-askpass-1030.sh` may also exist.

## Project Boundaries

| Path | Meaning | Rule |
|---|---|---|
| `/data/cyx/1030/scLatent` | Current main project: LatentFM, scFMBench, scaling-law work, zebrafish/dynamic-flow ideas, CoupledFM-related work, benchmark/evaluation infrastructure. | Use as the primary folder for this session. |
| `/data/cyx/1030/CellClip` | Independent CellClip project. | May share `dataset/`; do not edit from scLatent unless explicitly requested. |
| `/data/cyx/1030/stock` | Independent stock/quant project. | Do not edit from scLatent unless explicitly requested. |
| `/data/cyx/1030/dataset` | Shared data root for scLatent and CellClip. | Keep at root unless both project sessions are coordinated. |
| `/data/cyx/1030/software` | Shared runtime/tool installs. | Keep at root. Project-specific virtualenvs belong inside the project. |

## scLatent Physical Assets

These are now under `/data/cyx/1030/scLatent`:

```text
CoupledFM/
scFMBench/
ops/
runs/
reports/
logs/
docs/
configs/
external_review/
pretrainckpt/
scFM_cache/
scFM_output/
scFM_pretrained/
scFM_third_party/
scripts/
.venvs/
goal.md
init-scdfm.sh
post_sync_validate_and_smoke.sh
sync_from_lilab.sh
prompts/
```

Convenience links:

```text
scLatent/AGENTS.md -> ../AGENTS.md
scLatent/agent.md -> ../agent.md
scLatent/dataset -> ../dataset
scLatent/scFM_data -> dataset/scFM_data
scLatent/software -> ../software
```

## Compatibility Note

Historical reports, old run statuses, and old `goal.md` entries may mention
paths like `/data/cyx/1030/reports/...` or `/data/cyx/1030/runs/...`. For
current work, project-owned paths should be interpreted as:

```text
/data/cyx/1030/scLatent/reports/...
/data/cyx/1030/scLatent/runs/...
/data/cyx/1030/scLatent/ops/...
/data/cyx/1030/scLatent/logs/...
```

Do not bulk rewrite old historical reports just to update paths. New scripts
and new docs should use the current layout.

## Keep / Archive / Delete Policy

Keep:

- `scLatent/goal.md`, root `AGENTS.md`, root `agent.md`;
- `scLatent/docs/`, `runs/`, `reports/`, `logs/`, `ops/`, `configs/`;
- shared `dataset/`;
- scLatent pretrained resources, third-party mirrors, outputs, caches, and
  project-local virtual environments;
- all `RUN_STATUS.md`, final reports, decision reports, and provenance files.

Do not touch from this project:

- `CellClip/` unless explicitly assigned;
- `stock/` unless explicitly assigned.

Safe deletion candidates:

- reproducible caches such as `.pytest_cache`;
- temporary files with no provenance value after confirming they are not named
  in a run status, report, or script;
- duplicate scratch files only after a short inventory notes what was removed.

Avoid deleting:

- historical logs, even if old;
- failed run outputs, because they are negative evidence;
- raw datasets, converted datasets, embeddings, checkpoints, or manifests;
- anything referenced by `goal.md`, `docs/PROJECT_REVIEW.md`,
  `docs/EXPERIMENT_INDEX.md`, or `runs/*/RUN_STATUS.md`.

## Current Cleanup Performed

The first cleanup pass removed only reproducible pytest caches outside
CellClip/stock:

```text
.pytest_cache
CoupledFM/model/.pytest_cache
scFMBench/.pytest_cache
```

The second pass physically moved scLatent-owned directories into
`/data/cyx/1030/scLatent` and mechanically updated scLatent executable/source
paths to use the new project root. Shared `dataset/` stayed at root because
CellClip and scLatent can both use it.

No experiment outputs, logs, reports, datasets, CellClip files, or stock files
were deleted.
