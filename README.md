# scLatent Project Entrypoint

This is the current Codex-managed project inside `/data/cyx/1030`.

Most project-owned assets now live physically under this folder. The shared
dataset root remains `/data/cyx/1030/dataset` because `scLatent` and `CellClip`
can both use it.

GitHub target:

```text
https://github.com/cfy2yue/scRepresentation
```

Server entry:

```bash
ssh cyx-server-cfy
cd /data/cyx/1030/scLatent
```

## What Belongs Here

```text
scLatent/
  CoupledFM/
  scFMBench/
  ops/
  runs/
  reports/
  logs/
  docs/
  configs/
  pretrainckpt/
  scFM_cache/
  scFM_output/
  scFM_pretrained/
  scFM_third_party/
  .venvs/
  goal.md
  init-scdfm.sh
  prompts/
```

`dataset/` is a symlink to the shared root data directory:

```text
scLatent/dataset -> ../dataset
scLatent/scFM_data -> dataset/scFM_data
```

`software/` is also linked for convenience:

```text
scLatent/software -> ../software
```

## Not This Project

- `../CellClip`: independent CellClip project, sharing `dataset/` when useful.
- `../stock`: independent stock/quant project.
- `../software`, `../codex`, `../vscode-server`: shared runtime/system state.

## Read First

```text
AGENTS.md
local_goal.md
local_audit.md
local_suggestion.md
goal.md
docs/START_HERE.md
docs/WORKSPACE_ORGANIZATION.md
docs/GIT_AND_COLLABORATION.md
docs/GITHUB_FILE_MAP.md
docs/PROJECT_OVERVIEW.md
docs/PROJECT_REVIEW.md
docs/EXPERIMENT_INDEX.md
docs/RESULTS_SUMMARY.md
docs/DECISIONS.md
docs/BUGS_AND_FIXES.md
```

The current CC/Codex workflow is manual and audit-first. `local_goal.md`,
`local_audit.md`, and `local_suggestion.md` form the local-authored remote
execution packet. Remote Codex reads them to execute a user-started goal, but
does not edit them; local CC/Codex updates and pushes them between remote runs.
`local_goal.md` becomes executable only after local audit fills its
`Exact Next Task` section for one bounded remote task. The remote-side trigger
words are documented in `docs/START_HERE.md`: `本地审计指令` and `本地审计结束`.

Legacy auto-coordination docs and prompts are archived under
`docs/archive/legacy_auto_coordination_20260701/` and
`prompts/archive/legacy_auto_coordination_20260701/`. They are historical
evidence, not active workflow instructions.

## Runtime

```bash
source /data/cyx/1030/scLatent/init-scdfm.sh
```

The script sets:

```text
SCDFM_WORKSPACE=/data/cyx/1030/scLatent
SCDFM_DATASET_ROOT=/data/cyx/1030/dataset
SCFM_OUTPUT_ROOT=/data/cyx/1030/scLatent/scFM_output
SCFM_PRETRAINED_ROOT=/data/cyx/1030/scLatent/scFM_pretrained
SCFM_THIRD_PARTY_ROOT=/data/cyx/1030/scLatent/scFM_third_party
SCFM_ENVS_ROOT=/data/cyx/1030/scLatent/.venvs
```

## Operating Mode

For active research, follow `AGENTS.md` exactly. Long jobs need detached
execution and `RUN_STATUS.md`; GPU work needs resource audits; Track A/Track C
split rules remain hard constraints.

For documentation/organization work, do not launch experiments unless the user
asks to resume active exploration. Preserve provenance-bearing outputs and
record any future physical migrations in `docs/DECISIONS.md` and `goal.md`.

For CC/Windows review, clone the GitHub repo locally for reading, planning, and
Markdown/code review. Server-only work such as GPU jobs, large data reads, and
run integration belongs on `cyx-server-cfy`, normally through Codex after the
local-authored remote execution packet names the exact task and limits.

## Prompts

Reusable prompts for other sessions live in:

```text
prompts/
```
