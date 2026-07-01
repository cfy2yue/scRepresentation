# CC Default Operating Model

Updated: 2026-07-01

Canonical tracked copy of the CC/Cursor default operating model. The local
working copy is `E:\cc_workspace\CC_DEFAULT_OPERATING_MODEL.md`.

This applies to the current projects under `E:\cc_workspace` and should scale to
more projects and more remote servers without changing the core responsibilities.

## Default Division Of Labor

CC is the local coordinator and thinking layer. Remote Codex is the remote
execution layer.

CC owns:

- Interpreting the user's request and deciding which projects are in scope.
- Checking local, GitHub, and remote sync before work starts.
- Auditing docs, goals, source code, evidence, and strategy.
- Doing small local experiments or static checks when they are cheap and safe.
- Pulling one independent CC subagent per project for parallel audits.
- Synthesizing subagent findings into one coherent priority order.
- Updating `goal.md`, handoff docs, decision docs, protocol docs, and prompts.
- Committing and pushing documentation or small safe local fixes when sync is
  required.
- Starting or requesting one remote Codex goal session per remote task.
- Polling remote sessions at a low frequency, checking reasonableness, and
  correcting drift.

Remote Codex owns:

- Long-running implementation and execution on the remote server.
- Server-side experiments, GPU/data/cache/API/backtest work, and generated
  evidence.
- Detailed code changes inside the task ownership boundary assigned by CC.
- Writing progress to `runs/<run>/RUN_STATUS.md` or the task's equivalent status
  path.
- Producing final concise summaries, exact changed files, and blockers.

## Scalable Project Registry

Treat the project list as a registry, not a hard-coded three-project assumption.
Each active project entry should specify:

- Local path.
- GitHub remote.
- Remote server alias.
- Remote path.
- Current owner lane: CC audit, remote Codex execution, or waiting.
- Current goal doc and handoff doc.
- Current remote session name, if any.
- Status path such as `runs/<run>/RUN_STATUS.md`.

Current registry:

| project | local path | GitHub | remote |
|---|---|---|---|
| scLatent / scRepresentation | `E:\cc_workspace\scLatent` | `cfy2yue/scRepresentation` | `cyx-server-proxy-cfy:/data/cyx/1030/scLatent` |
| CellClip | `E:\cc_workspace\CellClip` | `cfy2yue/CellCLIP` | `cyx-server-proxy-cfy:/data/cyx/1030/CellClip` |
| StockHome | `E:\cc_workspace\stock` | `cfy2yue/StockHome` | `cyx-server-proxy-cfy:/data/cyx/1030/stock` |

When new projects or servers are added, update the registry first, then reuse
the same workflow below.

## Default Multi-Project Workflow

1. Main CC performs the intake and sync gate.
2. Main CC checks `ccusage`; if the current session/day cost is above the user's
   stop threshold, do not start new expensive work.
3. Main CC launches one CC subagent per in-scope project.
4. Each subagent audits exactly one project and returns a structured finding
   set; subagents do not push, launch remote jobs, or edit outside their project
   unless explicitly delegated.
5. Main CC compares the subagent outputs, resolves contradictions, and decides
   the project-level goals and priorities.
6. Main CC updates version-controlled goal/handoff docs in each project.
7. Main CC commits and pushes the docs/fixes needed for remote synchronization.
8. Main CC syncs the relevant remote repo(s) through GitHub.
9. Main CC starts one remote Codex goal session per approved remote task.
10. Main CC polls remote sessions at the agreed interval and corrects drift with
    new docs/prompts rather than ad hoc terminal nudges whenever possible.

## CC Subagent Contract

Each project subagent should receive a narrow prompt with:

- Project name and exact local path.
- Exact files to read first: `goal.md`, startup docs, handoff docs, protocol
  docs, decision docs, and the most relevant source/test paths.
- Forbidden actions: no secrets, no large jobs, no broad rewrites, no Git push,
  no remote job launch unless explicitly delegated.
- Expected output:
  - Goal reasonableness.
  - Direction reasonableness.
  - Biggest risks and bottlenecks.
  - Suggested goal doc edits.
  - Suggested remote Codex task.
  - Success criteria, stop rules, and files not to touch.
  - Cheap local checks or experiments already run.

Main CC is responsible for synthesis. Subagent outputs are evidence, not final
decisions.

## Remote Codex Goal Session Contract

Open one remote Codex session per task, not one overloaded session for many
projects. Prefer one `tmux` session per goal:

```bash
tmux new -d -s codex_<project>_<goal>_$(date +%Y%m%d) 'bash /tmp/launch_<project>_<goal>.sh'
```

The launch prompt should be a thin pointer to version-controlled goal docs:

```text
Read and execute the goal in docs/CC_AUDIT_AND_HANDOFF_<date>.md and goal.md.
Honor ownership, success criteria, stop rules, files to read first, and files
not to touch. Start by writing a brief plan to runs/<run>/RUN_STATUS.md. Keep
progress there and produce a concise final summary.
```

Use remote Codex goal mode when available for complex work:

```bash
codex features enable goals
```

For noninteractive execution on the current remote CLI, keep the known-safe
global flag order:

```bash
codex -a never exec -C /remote/project/path -m gpt-5.5 -s workspace-write - < /tmp/handoff.txt
```

Use cheaper models for status/doc smoke checks and stronger models for hard
implementation or research planning.

## Monitoring And Correction

Default polling interval for long-running work is 3600 seconds unless the user
asks otherwise. Polling should check:

- `ccusage` cost gate before starting new expensive actions.
- Remote `tmux ls`.
- Codex process/log health.
- Remote `git status -sb`.
- `runs/<run>/RUN_STATUS.md` latest dated entry.
- Final message path such as `/tmp/codex_last_<project>_<goal>.md`.
- Whether scope, cost, model, and stop rules still match the goal doc.

If a session stalls, expands scope, ignores a stop rule, or produces confusing
evidence, CC should start a new audit/correction round: update the goal/handoff
doc, commit/push it, and then resume or restart the remote Codex session with a
thin pointer to the revised doc.

## Conflict Rules

- Do not let CC and remote Codex edit the same code file at the same time.
- Main CC owns commits and pushes by default.
- Remote Codex may leave working-tree changes and generated artifacts, but should
  not push unless the user explicitly requests that mode.
- Generated reports, raw data, caches, logs, secrets, and credentials stay out of
  Git unless a curated artifact is deliberately selected.
- Prefer docs as the coordination surface. If a terminal instruction matters, put
  it into a version-controlled handoff/status doc.

## Stop Gates

Stop starting new high-cost work and report to the user when:

- `ccusage` crosses the user's configured threshold, currently USD 80.
- Local, GitHub, and remote diverge in a way that cannot be safely fast-forwarded.
- A remote goal would require secrets to be printed, copied, committed, or pasted.
- The task would delete or move data, reports, runs, checkpoints, caches, archives,
  or credentials without explicit approval.
- The remote task exceeds the cost/scope/model limits written in the goal doc.

