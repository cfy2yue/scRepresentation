# Start Here

Updated: 2026-07-02

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
4. `remote_decision.md`
5. `goal.md`
6. `AGENTS.md`
7. `docs/GIT_AND_COLLABORATION.md`
8. `docs/GITHUB_FILE_MAP.md`
9. `docs/WORKSPACE_ORGANIZATION.md`
10. `docs/PROJECT_OVERVIEW.md`
11. `docs/PROJECT_REVIEW.md`
12. `docs/EXPERIMENT_INDEX.md`

`local_goal.md`, `local_audit.md`, and `local_suggestion.md` are the
local-authored remote execution packet. Remote Codex reads them to execute a
user-started goal, but must not edit them. Local CC/Codex updates these files
between remote runs and pushes them to GitHub.

`remote_decision.md` is the remote-side Chinese decision log. Remote Codex may
create or append it during goal execution. Use it to record AUTONOMOUS_DECISION,
ROUTE_PIVOT, negative or underpowered results, subagent advice, and why remote
continued instead of stopping.

Remote execution starts only after local audit fills `local_goal.md` ->
`Exact Next Task`. If it is not filled, the packet is in waiting state.

Files under `docs/archive/legacy_auto_coordination_20260701/` and
`prompts/archive/legacy_auto_coordination_20260701/` are historical evidence
only. Do not treat them as active instructions.

## Manual Local/Remote Boundary

Local CC/Codex reviews the GitHub clone, audits goals, finds bugs and
bottlenecks, optionally runs small checks, and authors/updates `local_goal.md`,
`local_audit.md`, and `local_suggestion.md`.

Remote Codex executes only after the user manually pulls GitHub and starts a
goal. Remote Codex should read the three `local_*.md` files, `remote_decision.md`,
and `goal.md`; execute from the filled `Exact Next Task`; record decisions and
results in RUN_STATUS/reports and `remote_decision.md`; and continue toward the
final goal unless it reaches ACHIEVED, a hard BLOCKED boundary, or user
interruption.

`LOCAL_AUDIT_REQUEST` is a soft audit marker, not a long-goal stop reason. If a
scaling, zebrafish, architecture, materialization, or regression route misses
its gate or becomes locally data-blocked, remote Codex should record evidence
in `remote_decision.md`, design a new safe route or bounded diagnostic inside
the resource/split/safety limits, optionally use subagents, and continue. Mark
hard BLOCKED only if all reasonable next routes require changing the final
target, resource boundary, data source, held-out/query permission, or
destructive operation.

## Remote Trigger Protocol

When the user types `本地审计指令`, remote Codex must pause new large work,
avoid remote git commit/push/reset/delete operations in that status-export
turn, and output a structured `LOCAL_AUDIT_REQUEST` containing project path,
branch, HEAD, dirty state, files read, final target, current route, recent
commands, changed files, metrics, best/negative/anomalous results, suspected
bottlenecks, at least three directions for local audit, and suggested updates
to `local_goal.md`, `local_audit.md`, and `local_suggestion.md`.

This trigger is user-requested manual audit. It does not mean ordinary negative
results during goal mode should stop the goal.

When the user types `本地审计结束`, remote Codex must run `git fetch origin` and
`git pull --ff-only`, read `goal.md`, `local_goal.md`, `local_audit.md`,
`local_suggestion.md`, `remote_decision.md`, and this file, then summarize the
next task, resource limits, hard stop rules, output paths, remote decision
history, and any document conflicts. It must not edit the three `local_*.md`
files. It must then wait for the user to start or continue goal mode. If
`local_goal.md` still says `Exact Next Task` is `NOT ACTIVE`, remote Codex must
not start a remote goal.

Manual goal prompt:

```text
目标与路线：local_goal.md
本地审计：local_audit.md
本地建议：local_suggestion.md
远端决策日志：remote_decision.md
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
current high-signal docs. It should not contain datasets, checkpoints,
generated runs/reports/logs, venvs, large references, tokens, or server-local
archives.
