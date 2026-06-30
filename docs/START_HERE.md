# Start Here

Updated: 2026-07-01

scLatent is the umbrella repo for LatentFM, scFMBench, CoupledFM-derived code,
benchmark/evaluation infrastructure, scaling-axis audits, and project-owned
scripts/docs.

## Entry

- Server directory: `/data/cyx/1030/scLatent`
- GitHub target: `https://github.com/cfy2yue/scRepresentation`
- SSH entry: `ssh cyx-server-proxy-cfy`, then `cd /data/cyx/1030/scLatent`
- Shared data root: `/data/cyx/1030/dataset`
- Shared runtime root: `/data/cyx/1030/software`
- Runtime entry: `source /data/cyx/1030/scLatent/init-scdfm.sh`

## Read First

1. `goal.md`
2. `AGENTS.md`
3. `docs/GIT_AND_COLLABORATION.md`
4. `docs/GITHUB_FILE_MAP.md`
5. `docs/WORKSPACE_ORGANIZATION.md`
6. `docs/CODEX_CC_COLLABORATION.md`
7. `docs/PROJECT_OVERVIEW.md`
8. `docs/PROJECT_REVIEW.md`
9. `docs/EXPERIMENT_INDEX.md`

## CC And Codex Boundary

CC on Windows can clone the GitHub repo for reading, direction audit, Markdown
cleanup, goal/plan drafting, and code review. CC should not run GPU jobs, large
data reads, checkpoint evaluation, or server-only cache workflows locally.

Codex on the server owns resource audits, experiment execution, long jobs,
result integration, and canonical state updates unless the user pauses Codex for
a scoped file, branch, or task.

When both are active, record file/task ownership in `goal.md`,
`docs/PROJECT_REVIEW.md`, or a dated handoff note. Avoid simultaneous edits to
the same code file. Markdown can be parallelized when dated append sections are
used.

## Do Not Touch Without Approval

- `/data/cyx/1030/dataset` destructive edits.
- `runs/`, `reports/`, `logs/`, checkpoints, pretrained weights, and caches.
- `docs/local_archive/`, which stores full pre-slim Markdown provenance.
- Secrets, tokens, credentials, and local environment files.
- Other project roots (`../CellClip`, `../stock`) unless explicitly requested.

## GitHub Publication Rule

Git should contain source code, small configs, README/AGENTS, prompts, and
current high-signal docs. It should not contain datasets, checkpoints, generated
runs/reports/logs, venvs, large references, tokens, or server-local archives.
