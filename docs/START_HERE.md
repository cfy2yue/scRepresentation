# Start Here

Updated: 2026-07-01

scLatent is the umbrella repo for LatentFM, scFMBench, CoupledFM-derived code,
benchmark/evaluation infrastructure, scaling-axis audits, and project-owned
scripts/docs.

## Entry

- Server directory: `/data/cyx/1030/scLatent`
- GitHub target: `https://github.com/cfy2yue/scRepresentation`
- SSH entry: `ssh cyx-server-cfy`, then `cd /data/cyx/1030/scLatent`
- Shared data root: `/data/cyx/1030/dataset`
- Shared runtime root: `/data/cyx/1030/software`
- Runtime entry: `source /data/cyx/1030/scLatent/init-scdfm.sh`

## Read First

Current manual local-audit workflow:

1. `local_goal.md`
2. `local_audit.md`
3. `local_suggestion.md`
4. `goal.md`
5. `AGENTS.md`
6. `docs/GIT_AND_COLLABORATION.md`
7. `docs/GITHUB_FILE_MAP.md`
8. `docs/WORKSPACE_ORGANIZATION.md`
9. `docs/PROJECT_OVERVIEW.md`
10. `docs/PROJECT_REVIEW.md`
11. `docs/EXPERIMENT_INDEX.md`

Files under `docs/archive/legacy_auto_coordination_20260701/` and
`prompts/archive/legacy_auto_coordination_20260701/` are historical evidence
only. Do not treat them as active instructions.

## Manual Local/Remote Boundary

Local CC/Codex reviews the GitHub clone, audits goals, finds bugs and
bottlenecks, optionally runs small checks, and updates `local_goal.md`,
`local_audit.md`, and `local_suggestion.md`.

Remote Codex executes only after the user manually pulls GitHub and starts a
goal. Remote Codex should read the three `local_*.md` files plus `goal.md`,
record decisions and results, and output a structured local-audit request when
blocked.

## Remote Trigger Protocol

When the user types `本地审计指令`, remote Codex must pause new large work,
avoid git commit/push/reset/delete operations, and output a structured
`LOCAL_AUDIT_REQUEST` containing project path, branch, HEAD, dirty state, files
read, final target, current route, recent commands, changed files, metrics,
best/negative/anomalous results, suspected bottlenecks, at least three
directions for local audit, and suggested updates to `local_goal.md`,
`local_audit.md`, and `local_suggestion.md`.

When the user types `本地审计结束`, remote Codex must run `git fetch origin`
and `git pull --ff-only`, read `goal.md`, `local_goal.md`, `local_audit.md`,
`local_suggestion.md`, and this file, then summarize the next task, resource
limits, stop rules, and any document conflicts. It must then wait for the user
to start or continue goal mode.

Manual goal prompt:

```text
目标与路线：local_goal.md
本地审计：local_audit.md（不存在则跳过）
本地建议：local_suggestion.md（不存在则跳过）
资源限制：<fill in this round's limits>
```

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
